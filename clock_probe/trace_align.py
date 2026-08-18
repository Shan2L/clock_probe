"""Stream-align PyTorch Chrome Trace timestamps to the reference clock."""

from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .chrome_trace import (
    BASE_TIME_PATTERN,
    TIMESTAMP_PATTERN,
    microseconds_to_ns as _microseconds_to_ns,
    ns_to_microseconds as _ns_to_microseconds,
)
from .clock_bridge import CompiledClockBridge


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
    ) -> tuple[int, float]:
        """Convert local realtime and return model uncertainty in microseconds."""
        if self.identity:
            return local_realtime_ns, 0.0

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
        return (
            round(local_realtime_ns + offset_ns),
            float(segment.get("uncertainty_us", 0.0)),
        )


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
    bridge_boot_id: str | None
    max_bridge_uncertainty_us: float
    max_total_uncertainty_us: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return self.__dict__.copy()


# pylint: disable=too-many-locals,too-many-arguments
# pylint: disable=too-many-statements,too-many-branches
def align_trace_file(
    *,
    trace_path: Path,
    session_path: Path,
    source_node: str,
    output_path: Path,
    target_base_time_ns: int | None = None,
    source_boot_id: str | None = None,
    progress: Callable[[int], None] | None = None,
) -> AlignmentStats:
    """Rewrite a PyTorch Trace without loading it into memory."""
    trace_path = trace_path.resolve()
    output_path = output_path.resolve()
    if trace_path == output_path:
        raise ValueError("Input and output Trace paths must differ")

    session = json.loads(session_path.read_text(encoding="utf-8"))
    if target_base_time_ns is None:
        session_target_base = session.get("target_base_time_ns")
        if session_target_base is None:
            raise ValueError(
                "Clock session has no common target_base_time_ns; "
                "provide --target-base-time-ns explicitly"
            )
        target_base_time_ns = int(session_target_base)
    selected_model = select_clock_model(session, source_node)
    model = CompiledClockModel(selected_model)
    bridge: CompiledClockBridge | None = None
    if not model.identity:
        bridge_payload = selected_model.get("realtime_monotonic_bridge")
        if not isinstance(bridge_payload, dict):
            raise ValueError(
                "Clock model has no REALTIME-to-MONOTONIC bridge; "
                "legacy sessions cannot safely align Kineto Trace timestamps"
            )
        bridge = CompiledClockBridge(bridge_payload)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_base_time_ns: int | None = None
    timestamp_count = 0
    first_monotonic_ns: int | None = None
    last_monotonic_ns: int | None = None
    max_bridge_uncertainty_us = 0.0
    max_total_uncertainty_us = 0.0
    inside_trace_events = False

    def replace_timestamp(match: re.Match[str]) -> str:
        nonlocal timestamp_count, first_monotonic_ns, last_monotonic_ns
        nonlocal max_bridge_uncertainty_us, max_total_uncertainty_us
        if source_base_time_ns is None:
            raise ValueError("Trace timestamp appeared before baseTimeNanoseconds")
        trace_relative_ns = _microseconds_to_ns(match.group("value"))
        local_realtime_ns = source_base_time_ns + trace_relative_ns
        bridge_uncertainty_us = 0.0
        if bridge is None:
            local_monotonic_ns = 0
        else:
            local_monotonic_ns, bridge_uncertainty_us = (
                bridge.realtime_to_monotonic_ns(
                    local_realtime_ns,
                    expected_boot_id=source_boot_id,
                )
            )
        aligned_realtime_ns, model_uncertainty_us = model.align_realtime_ns(
            local_realtime_ns,
            local_monotonic_ns,
        )
        aligned_trace_ns = aligned_realtime_ns - target_base_time_ns
        timestamp_count += 1
        if bridge is not None:
            if first_monotonic_ns is None:
                first_monotonic_ns = local_monotonic_ns
            last_monotonic_ns = local_monotonic_ns
        max_bridge_uncertainty_us = max(
            max_bridge_uncertainty_us,
            bridge_uncertainty_us,
        )
        max_total_uncertainty_us = max(
            max_total_uncertainty_us,
            bridge_uncertainty_us + model_uncertainty_us,
        )
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
        bridge_boot_id=bridge.boot_id if bridge is not None else None,
        max_bridge_uncertainty_us=max_bridge_uncertainty_us,
        max_total_uncertainty_us=max_total_uncertainty_us,
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
        default=None,
        help=(
            "Common output base; defaults to the precision-safe base stored "
            "in the clock session."
        ),
    )
    parser.add_argument(
        "--source-boot-id",
        help=(
            "Optional source-node boot ID captured with the Trace; a mismatch "
            "rejects alignment."
        ),
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
            source_boot_id=args.source_boot_id,
            progress=progress,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Trace alignment failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(stats.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
