"""Session report that keeps raw, aligned, and optional CLC Traces distinct."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .chrome_trace import emit_json, parse_rank_trace_arg


@dataclass
class TraceBundle:
    """One rank's files. Raw is never used as an output path."""

    rank: int
    raw: str
    aligned: str
    clc: str | None = None


@dataclass
class ClockReport:
    """Layer-11 contract: paths, NCCL status, and whether aligned is primary."""

    schema: int
    session_status: str
    primary_timeline: str
    traces: list[TraceBundle]
    nccl_check: dict[str, Any]
    clc: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable session contract."""
        payload = asdict(self)
        payload["traces"] = [asdict(bundle) for bundle in self.traces]
        return payload


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def build_clock_report(  # pylint: disable=too-many-locals,too-many-branches
    *,
    raw: dict[int, Path],
    aligned: dict[int, Path],
    clc: dict[int, Path] | None,
    nccl_check_path: Path,
    clc_report_path: Path | None,
) -> ClockReport:
    """Assemble the session contract without copying Trace bytes."""
    if set(raw) != set(aligned):
        raise ValueError("Raw and aligned --trace ranks must match")
    nccl_check = json.loads(nccl_check_path.read_text(encoding="utf-8"))
    clc_payload = None
    if clc_report_path is not None:
        clc_payload = json.loads(clc_report_path.read_text(encoding="utf-8"))
    notes: list[str] = []
    bundles: list[TraceBundle] = []
    for rank in sorted(raw):
        raw_path = _resolved(raw[rank])
        aligned_path = _resolved(aligned[rank])
        clc_path = _resolved(clc[rank]) if clc and rank in clc else None
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        if not aligned_path.is_file():
            raise FileNotFoundError(aligned_path)
        if raw_path == aligned_path:
            raise ValueError(f"Rank {rank} raw and aligned paths are the same file")
        if clc_path is not None:
            if clc_path in {raw_path, aligned_path}:
                raise ValueError(f"Rank {rank} CLC path overwrites raw or aligned")
            if not clc_path.is_file():
                raise FileNotFoundError(clc_path)
        bundles.append(
            TraceBundle(
                rank=rank,
                raw=str(raw_path),
                aligned=str(aligned_path),
                clc=None if clc_path is None else str(clc_path),
            )
        )

    nccl_status = str(nccl_check.get("status", "FAIL"))
    if nccl_status == "FAIL":
        session_status = "FAIL"
        primary = "none"
        notes.append("Aligned Trace is candidate-only; do not use it as the primary timeline")
    elif nccl_status == "WARNING":
        session_status = "WARNING"
        clc_status = str((clc_payload or {}).get("status", "SKIPPED"))
        if clc_status == "APPLIED":
            primary = "clc"
            notes.append("Use CLC Traces for causal views; keep affine aligned for comparison")
        else:
            primary = "aligned"
            notes.append("NCCL inversions are within uncertainty; CLC was not applied")
    else:
        session_status = "PASS"
        primary = "aligned"
        notes.append("Affine aligned Trace is the primary timeline; raw is retained")

    return ClockReport(
        schema=1,
        session_status=session_status,
        primary_timeline=primary,
        traces=bundles,
        nccl_check={
            "path": str(_resolved(nccl_check_path)),
            "status": nccl_status,
            "inversion_count": nccl_check.get("inversion_count"),
            "max_gap_us": nccl_check.get("max_gap_us"),
            "uncertainty_us": nccl_check.get("uncertainty_us"),
        },
        clc=None
        if clc_payload is None
        else {
            "path": str(_resolved(clc_report_path)) if clc_report_path else None,
            "status": clc_payload.get("status"),
            "shifted_event_count": clc_payload.get("shifted_event_count"),
            "max_delta_us": clc_payload.get("max_delta_us"),
        },
        notes=notes,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the raw/aligned/CLC session-report CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Write clock-report.json that records raw, aligned, and optional CLC "
            "paths. Never overwrites a Trace."
        )
    )
    parser.add_argument(
        "--raw",
        action="append",
        required=True,
        dest="raw",
        help="Original Profiler Trace as rank:path, repeatable.",
    )
    parser.add_argument(
        "--aligned",
        action="append",
        required=True,
        dest="aligned",
        help="Affine aligned Trace as rank:path, repeatable.",
    )
    parser.add_argument(
        "--clc",
        action="append",
        dest="clc",
        default=None,
        help="Optional CLC Trace as rank:path, repeatable.",
    )
    parser.add_argument("--nccl-check", required=True, type=Path)
    parser.add_argument("--clc-report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    """Write the session report and exit 2 when NCCL check failed."""
    args = build_parser().parse_args()
    raw = dict(parse_rank_trace_arg(item) for item in args.raw)
    aligned = dict(parse_rank_trace_arg(item) for item in args.aligned)
    clc = (
        dict(parse_rank_trace_arg(item) for item in args.clc)
        if args.clc
        else None
    )
    try:
        report = build_clock_report(
            raw=raw,
            aligned=aligned,
            clc=clc,
            nccl_check_path=args.nccl_check,
            clc_report_path=args.clc_report,
        )
    except (OSError, ValueError, json.JSONDecodeError, FileNotFoundError) as error:
        print(f"Clock report failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    emit_json(args.output, report.to_dict())
    if report.session_status == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
