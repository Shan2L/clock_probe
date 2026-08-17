"""Stream-align PyTorch Chrome Trace timestamps to the reference clock."""

from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

BASE_TIME_PATTERN = re.compile(
    r'(?P<prefix>"baseTimeNanoseconds"\s*:\s*)(?P<value>-?\d+)'
)
TIMESTAMP_PATTERN = re.compile(
    r'(?P<prefix>"ts"\s*:\s*)'
    r'(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
)


def _microseconds_to_ns(value: str) -> int:
    """Parse the fixed-point microsecond values emitted by PyTorch."""
    if "e" in value.lower():
        return round(Decimal(value) * 1_000)

    negative = value.startswith("-")
    unsigned = value[1:] if negative else value
    whole, separator, fraction = unsigned.partition(".")
    whole_ns = int(whole or "0") * 1_000
    fraction_digits = (fraction + "000")[:3] if separator else "000"
    result = whole_ns + int(fraction_digits)
    if separator and len(fraction) > 3 and fraction[3] >= "5":
        result += 1
    return -result if negative else result


def _ns_to_microseconds(value_ns: int) -> str:
    sign = "-" if value_ns < 0 else ""
    whole, fraction = divmod(abs(value_ns), 1_000)
    return f"{sign}{whole}.{fraction:03d}"


def _model_identifiers(model: dict[str, Any]) -> set[str]:
    source = model.get("source", {})
    return {
        str(value)
        for key in (
            "hostname",
            "ray_node_name",
            "ray_node_address",
            "ray_node_id",
        )
        if (value := source.get(key)) is not None
    }


def select_clock_model(
    session: dict[str, Any],
    source_node: str,
) -> dict[str, Any]:
    """Select one node model by hostname, Ray name, address, or node ID."""
    matches = [
        model
        for model in session.get("models", [])
        if source_node in _model_identifiers(model)
    ]
    if not matches:
        available = sorted(
            identifier
            for model in session.get("models", [])
            for identifier in _model_identifiers(model)
        )
        raise ValueError(
            f"No clock model matches {source_node!r}; available: {available}"
        )
    if len(matches) > 1:
        raise ValueError(f"Clock model identifier {source_node!r} is ambiguous")
    model = matches[0]
    if model.get("status") != "PASS":
        raise ValueError(
            f"Clock model for {source_node!r} has status "
            f"{model.get('status')!r}"
        )
    return model


class CompiledClockModel:  # pylint: disable=too-few-public-methods
    """Fast timestamp lookup for millions of Trace events."""

    def __init__(self, model: dict[str, Any]):
        self.identity = model.get("model_type") == "identity"
        self.segments = sorted(
            model.get("segments", []),
            key=lambda segment: int(segment["valid_from_monotonic_ns"]),
        )
        self.starts = [
            int(segment["valid_from_monotonic_ns"]) for segment in self.segments
        ]
        self.ends = [
            int(segment["valid_to_monotonic_ns"]) for segment in self.segments
        ]
        config = model.get("config", {})
        health = model.get("health", {})
        if self.segments and config and health:
            origin_ns = int(health["origin_monotonic_ns"])
            window_ns = round(float(config["window_seconds"]) * 1_000_000_000)
            segment_ns = round(
                float(config["segment_seconds"]) * 1_000_000_000
            )
            collection_end_ns = model.get("collection", {}).get(
                "ended_monotonic_ns"
            )
            self.starts = [
                origin_ns + int(segment["segment_index"]) * segment_ns
                for segment in self.segments
            ]
            self.ends = [
                min(
                    start_ns + segment_ns - 1,
                    int(segment["valid_to_monotonic_ns"]) + window_ns - 1,
                    (
                        int(collection_end_ns)
                        if collection_end_ns is not None
                        else start_ns + segment_ns - 1
                    ),
                )
                for start_ns, segment in zip(self.starts, self.segments)
            ]
        if not self.identity and not self.segments:
            raise ValueError("Piecewise model has no usable segments")

    def align_realtime_ns(
        self,
        local_realtime_ns: int,
        local_monotonic_ns: int,
    ) -> int:
        """Convert local realtime to reference realtime."""
        if self.identity:
            return local_realtime_ns

        index = bisect.bisect_right(self.starts, local_monotonic_ns) - 1
        if index < 0:
            raise ValueError(
                f"Trace timestamp {local_monotonic_ns} precedes model coverage"
            )
        segment = self.segments[index]
        if local_monotonic_ns > self.ends[index]:
            raise ValueError(
                f"No model segment covers Trace timestamp "
                f"{local_monotonic_ns}"
            )
        if segment.get("status") != "PASS":
            raise ValueError(
                f"Trace uses failed segment {segment['segment_index']}"
            )

        elapsed_ns = (
            local_monotonic_ns - float(segment["base_monotonic_ns"])
        )
        offset_ns = float(segment["offset_at_base_ns"]) + float(
            segment["drift_ns_per_ns"]
        ) * elapsed_ns
        return round(local_realtime_ns + offset_ns)


@dataclass
class AlignmentStats:  # pylint: disable=too-many-instance-attributes
    """Summary returned after a streaming Trace rewrite."""

    source_node: str
    input_path: str
    output_path: str
    source_base_time_ns: int
    target_base_time_ns: int
    timestamp_count: int
    first_source_monotonic_ns: int | None
    last_source_monotonic_ns: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return self.__dict__.copy()


def align_trace_file(  # pylint: disable=too-many-locals,too-many-arguments
    *,
    trace_path: Path,
    session_path: Path,
    source_node: str,
    output_path: Path,
    target_base_time_ns: int = 0,
    progress: Callable[[int], None] | None = None,
) -> AlignmentStats:
    """Rewrite a PyTorch Trace without loading it into memory."""
    trace_path = trace_path.resolve()
    output_path = output_path.resolve()
    if trace_path == output_path:
        raise ValueError("Input and output Trace paths must differ")

    session = json.loads(session_path.read_text(encoding="utf-8"))
    model = CompiledClockModel(select_clock_model(session, source_node))
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_base_time_ns: int | None = None
    timestamp_count = 0
    first_monotonic_ns: int | None = None
    last_monotonic_ns: int | None = None
    inside_trace_events = False

    def replace_timestamp(match: re.Match[str]) -> str:
        nonlocal timestamp_count, first_monotonic_ns, last_monotonic_ns
        if source_base_time_ns is None:
            raise ValueError("Trace timestamp appeared before baseTimeNanoseconds")
        monotonic_ns = _microseconds_to_ns(match.group("value"))
        local_realtime_ns = source_base_time_ns + monotonic_ns
        aligned_realtime_ns = model.align_realtime_ns(
            local_realtime_ns,
            monotonic_ns,
        )
        aligned_trace_ns = aligned_realtime_ns - target_base_time_ns
        timestamp_count += 1
        if first_monotonic_ns is None:
            first_monotonic_ns = monotonic_ns
        last_monotonic_ns = monotonic_ns
        if progress is not None and timestamp_count % 1_000_000 == 0:
            progress(timestamp_count)
        return match.group("prefix") + _ns_to_microseconds(aligned_trace_ns)

    try:
        with (
            trace_path.open("r", encoding="utf-8", buffering=1024 * 1024) as source,
            temporary_path.open(
                "w",
                encoding="utf-8",
                buffering=1024 * 1024,
            ) as destination,
        ):
            for line in source:
                if source_base_time_ns is None:
                    base_match = BASE_TIME_PATTERN.search(line)
                    if base_match is not None:
                        source_base_time_ns = int(base_match.group("value"))
                if '"traceEvents"' in line:
                    inside_trace_events = True

                if source_base_time_ns is not None and BASE_TIME_PATTERN.search(line):
                    line = BASE_TIME_PATTERN.sub(
                        lambda match: (
                            match.group("prefix") + str(target_base_time_ns)
                        ),
                        line,
                        count=1,
                    )
                # PyTorch formats event fields at exactly four-space indentation;
                # nested args are deeper and must not have unrelated "ts" keys
                # rewritten.
                indentation = len(line) - len(line.lstrip(" "))
                if (
                    inside_trace_events
                    and indentation == 4
                    and '"ts"' in line
                ):
                    line = TIMESTAMP_PATTERN.sub(replace_timestamp, line)
                destination.write(line)

        if source_base_time_ns is None:
            raise ValueError("Trace has no baseTimeNanoseconds field")
        if timestamp_count == 0:
            raise ValueError(
                "Trace has no supported four-space-indented event timestamps"
            )
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return AlignmentStats(
        source_node=source_node,
        input_path=str(trace_path),
        output_path=str(output_path),
        source_base_time_ns=source_base_time_ns,
        target_base_time_ns=target_base_time_ns,
        timestamp_count=timestamp_count,
        first_source_monotonic_ns=first_monotonic_ns,
        last_source_monotonic_ns=last_monotonic_ns,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the streaming Trace alignment CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Align one PyTorch Chrome Trace to the Ray Head clock domain."
        )
    )
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--clock-session", required=True, type=Path)
    parser.add_argument(
        "--source-node",
        required=True,
        help="Source hostname, Ray node name, IP, or Ray node ID.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--target-base-time-ns",
        type=int,
        default=0,
        help="Common output base; zero produces absolute epoch microseconds.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    """Run a streaming alignment and print its summary."""
    args = build_parser().parse_args()
    progress = None
    if not args.quiet:
        def report_progress(count: int) -> None:
            print(f"aligned_timestamps={count}", file=sys.stderr)

        progress = report_progress
    try:
        stats = align_trace_file(
            trace_path=args.trace,
            session_path=args.clock_session,
            source_node=args.source_node,
            output_path=args.output,
            target_base_time_ns=args.target_base_time_ns,
            progress=progress,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Trace alignment failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(stats.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
