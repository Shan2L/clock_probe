"""Restricted Controlled Logical Clock rewrite of aligned NCCL Traces."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .chrome import (
    microseconds_to_ns,
    replace_timestamp,
)
from .nccl import (
    IMPLICIT_SYNC,
    NCCLKernel,
    check_nccl_traces,
    extract_nccl_kernels,
    match_collectives,
)


@dataclass(frozen=True)
class KernelShift:
    """One collective kernel that must be delayed to restore overlap."""

    rank: int
    pid: int | None
    tid: int | None
    ts_ns: int
    delta_ns: int
    collective_id: str
    collective: str


@dataclass
class CLCReport:
    """Sidecar describing CLC edits. Affine aligned files are not overwritten."""

    status: str
    uncertainty_us: float
    shifted_event_count: int
    max_delta_us: float
    shifts: list[KernelShift] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable CLC sidecar."""
        payload = asdict(self)
        payload["shifts"] = [asdict(shift) for shift in self.shifts]
        return payload


def plan_kernel_shifts(
    traces: dict[int, Path],
    *,
    uncertainty_us: float,
) -> list[KernelShift]:
    """Advance early-finishing ranks just enough to overlap the last start."""
    kernels: list[NCCLKernel] = []
    for rank, path in sorted(traces.items()):
        kernels.extend(extract_nccl_kernels(path, rank))
    match_result = match_collectives(kernels)
    if match_result.matching_failed:
        raise ValueError(
            match_result.failure_reason or "Collective matching failed before CLC"
        )
    matched = match_result.matched
    uncertainty_ns = round(uncertainty_us * 1_000.0)
    shifts: list[KernelShift] = []
    for collective_id, members in matched.items():
        if not any(member.collective in IMPLICIT_SYNC for member in members):
            continue
        last_start_ns = max(member.ts_ns for member in members)
        for member in members:
            delta_ns = last_start_ns - member.end_ns
            if delta_ns <= 0:
                continue
            if delta_ns > uncertainty_ns:
                raise ValueError(
                    f"CLC shift {delta_ns / 1_000.0:.3f} μs for "
                    f"{collective_id} rank {member.rank} exceeds "
                    f"uncertainty_us={uncertainty_us}"
                )
            shifts.append(
                KernelShift(
                    rank=member.rank,
                    pid=member.pid,
                    tid=member.tid,
                    ts_ns=member.ts_ns,
                    delta_ns=delta_ns,
                    collective_id=collective_id,
                    collective=member.collective,
                )
            )
    return shifts


def _stream_delta_ns(
    shifts: list[KernelShift],
    pid: int | None,
    tid: int | None,
    timestamp_ns: int,
) -> int:
    applied = 0
    for shift in shifts:
        if shift.pid != pid or shift.tid != tid:
            continue
        if shift.ts_ns <= timestamp_ns:
            applied = max(applied, shift.delta_ns)
    return applied


def rewrite_trace_with_clc(  # pylint: disable=too-many-locals,too-many-statements
    *,
    input_path: Path,
    output_path: Path,
    shifts: list[KernelShift],
) -> int:
    """Copy one Trace, delaying events on streams that needed a CLC shift."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("CLC output must differ from the aligned input")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    rewritten = 0
    try:
        with (
            input_path.open("r", encoding="utf-8", buffering=1024 * 1024) as source,
            temporary_path.open(
                "w", encoding="utf-8", buffering=1024 * 1024
            ) as destination,
        ):
            in_array = False
            pending: list[str] = []
            depth = 0
            for line in source:
                if not in_array:
                    destination.write(line)
                    if '"traceEvents"' in line:
                        in_array = True
                    continue
                if depth == 0 and "]" in line and "{" not in line:
                    destination.write(line)
                    destination.writelines(pending)
                    pending = []
                    in_array = False
                    continue
                if depth == 0 and "{" not in line:
                    destination.write(line)
                    continue
                depth += line.count("{") - line.count("}")
                pending.append(line)
                if depth != 0:
                    continue
                block = "".join(pending)
                pending = []
                stripped = block.strip()[:-1] if block.strip().endswith(",") else block.strip()
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    destination.write(block)
                    continue
                if "ts" not in event:
                    destination.write(block)
                    continue
                delta_ns = _stream_delta_ns(
                    shifts,
                    event.get("pid"),
                    event.get("tid"),
                    microseconds_to_ns(event["ts"]),
                )
                if delta_ns <= 0:
                    destination.write(block)
                    continue
                new_ts_ns = microseconds_to_ns(event["ts"]) + delta_ns
                destination.write(replace_timestamp(block, new_ts_ns))
                rewritten += 1
            if pending:
                destination.writelines(pending)
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return rewritten


def apply_clc(
    traces: dict[int, Path],
    outputs: dict[int, Path],
    *,
    uncertainty_us: float,
    check_status: str,
) -> CLCReport:
    """Rewrite WARNING Traces. FAIL is rejected; PASS is a no-op."""
    if check_status not in {"PASS", "WARNING", "FAIL"}:
        raise ValueError(f"Unknown NCCL check status {check_status!r}")
    if check_status == "FAIL":
        raise ValueError("Refuse CLC on a FAIL NCCL check; aligned Trace is candidate-only")
    if set(outputs) - set(traces):
        raise ValueError("CLC --output ranks must be a subset of --trace ranks")
    notes: list[str] = []
    if check_status == "PASS":
        notes.append("NCCL check is PASS; no CLC rewrite")
        return CLCReport(
            status="SKIPPED",
            uncertainty_us=uncertainty_us,
            shifted_event_count=0,
            max_delta_us=0.0,
            notes=notes,
        )
    report = check_nccl_traces(traces, uncertainty_us=uncertainty_us)
    if report.status == "FAIL":
        raise ValueError(
            "NCCL inversions exceed uncertainty_us; CLC would hide a bad affine model"
        )
    shifts = plan_kernel_shifts(traces, uncertainty_us=uncertainty_us)
    if not shifts:
        notes.append("No implicit-sync kernel required a CLC delay")
        return CLCReport(
            status="SKIPPED",
            uncertainty_us=uncertainty_us,
            shifted_event_count=0,
            max_delta_us=0.0,
            notes=notes,
        )
    by_rank: dict[int, list[KernelShift]] = defaultdict(list)
    for shift in shifts:
        by_rank[shift.rank].append(shift)
    written: dict[str, str] = {}
    shifted_events = 0
    for rank, input_path in traces.items():
        output_path = outputs.get(rank)
        if output_path is None:
            continue
        count = rewrite_trace_with_clc(
            input_path=input_path,
            output_path=output_path,
            shifts=by_rank.get(rank, []),
        )
        shifted_events += count
        written[str(rank)] = str(output_path)
    notes.append("CLC delayed early-finishing ranks only; stragglers were not pulled in")
    return CLCReport(
        status="APPLIED",
        uncertainty_us=uncertainty_us,
        shifted_event_count=shifted_events,
        max_delta_us=max(shift.delta_ns for shift in shifts) / 1_000.0,
        shifts=shifts,
        outputs=written,
        notes=notes,
    )
