"""One-call offline Trace alignment and causal validation pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .clc import apply_clc
from .nccl import check_nccl_traces
from .report import build_clock_report
from ..session import SessionInput, load_session, session_uncertainty_us
from .align import AlignmentStats, align_trace_file


@dataclass(frozen=True)
class TraceInput:
    """One raw rank Trace and the node clock that produced it."""

    path: Path
    source_node: str
    boot_id: str | None = None


@dataclass
class ProcessManifest:
    """Stable result returned to embedding applications."""

    status: str
    primary_timeline: str
    raw: dict[int, str]
    aligned: dict[int, str]
    clc: dict[int, str]
    alignment: dict[int, dict[str, Any]]
    nccl_check: dict[str, Any]
    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def process_traces(
    traces: Mapping[int, TraceInput],
    session: SessionInput,
    output_dir: str | Path,
    *,
    apply_clc_on_warning: bool = False,
) -> ProcessManifest:
    """Align all ranks, validate NCCL, optionally apply CLC, and write a manifest."""
    if len(traces) < 2:
        raise ValueError("Trace processing needs at least two ranks")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    session_payload = load_session(session)
    uncertainty_us = session_uncertainty_us(session_payload)

    raw = {rank: spec.path.expanduser().resolve() for rank, spec in traces.items()}
    aligned = {
        rank: root / f"rank-{rank}.aligned.json"
        for rank in traces
    }
    alignment: dict[int, AlignmentStats] = {}
    for rank, spec in sorted(traces.items()):
        alignment[rank] = align_trace_file(
            trace_path=raw[rank],
            session_path=session_payload,
            source_node=spec.source_node,
            source_boot_id=spec.boot_id,
            output_path=aligned[rank],
            progress=None,
        )

    check = check_nccl_traces(aligned, uncertainty_us=uncertainty_us)
    check_payload = check.to_dict()
    check_path = root / "nccl-check.json"
    _write_json(check_path, check_payload)

    clc_paths: dict[int, Path] = {}
    clc_payload: dict[str, Any] | None = None
    clc_report_path: Path | None = None
    if apply_clc_on_warning and check.status == "WARNING":
        clc_paths = {
            rank: root / f"rank-{rank}.clc.json"
            for rank in aligned
        }
        clc_report = apply_clc(
            aligned,
            clc_paths,
            uncertainty_us=uncertainty_us,
            check_status=check.status,
        )
        clc_payload = clc_report.to_dict()
        clc_report_path = root / "clc-report.json"
        _write_json(clc_report_path, clc_payload)
        if clc_report.status != "APPLIED":
            clc_paths = {}

    report = build_clock_report(
        raw=raw,
        aligned=aligned,
        clc=clc_paths or None,
        nccl_check_path=check_path,
        clc_report_path=clc_report_path,
    )
    report_payload = report.to_dict()
    _write_json(root / "clock-report.json", report_payload)

    manifest = ProcessManifest(
        status=report.session_status,
        primary_timeline=report.primary_timeline,
        raw={rank: str(path) for rank, path in raw.items()},
        aligned={rank: str(path) for rank, path in aligned.items()},
        clc={rank: str(path) for rank, path in clc_paths.items()},
        alignment={
            rank: stats.to_dict()
            for rank, stats in alignment.items()
        },
        nccl_check=check_payload,
        report=report_payload,
    )
    _write_json(root / "manifest.json", manifest.to_dict())
    return manifest
