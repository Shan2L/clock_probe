"""Stream-align PyTorch Chrome Trace timestamps to the reference clock."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .chrome import (
    BASE_TIME_PATTERN,
    TIMESTAMP_PATTERN,
    microseconds_to_ns as _microseconds_to_ns,
    ns_to_microseconds as _ns_to_microseconds,
)
from ..calibration.clock_bridge import CompiledClockBridge
from ..calibration.phc_bridge import CompiledPhcBridge
from ..calibration.software import CompiledClockModel
from ..session import SessionInput, load_session, select_model


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
    first_source_phc_ns: int | None
    last_source_phc_ns: int | None
    bridge_boot_id: str | None
    clock_source: str
    max_bridge_uncertainty_us: float
    max_ptp_uncertainty_us: float
    max_total_uncertainty_us: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return self.__dict__.copy()


# pylint: disable=too-many-locals,too-many-arguments
# pylint: disable=too-many-statements,too-many-branches
def align_trace_file(
    *,
    trace_path: Path,
    session_path: SessionInput,
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

    session = load_session(session_path)
    if target_base_time_ns is None:
        session_target_base = session.get("target_base_time_ns")
        if session_target_base is None:
            raise ValueError(
                "Clock session has no common target_base_time_ns; "
                "provide --target-base-time-ns explicitly"
            )
        target_base_time_ns = int(session_target_base)
    selected_model = select_model(session, source_node)
    clock_source = str(session.get("clock_source") or "udp_software")
    model: CompiledClockModel | None = None
    bridge: CompiledClockBridge | None = None
    phc_bridge: CompiledPhcBridge | None = None
    ptp_uncertainty_us = 0.0
    if clock_source == "ptp_hardware":
        if session.get("timestamp_domain") != "PHC":
            raise ValueError(
                "ptp_hardware sessions must use timestamp_domain=PHC"
            )
        if selected_model.get("model_type") != "phc_bridge":
            raise ValueError(
                "ptp_hardware alignment requires a phc_bridge node model"
            )
        phc_payload = selected_model.get("realtime_phc_bridge")
        if not isinstance(phc_payload, dict):
            raise ValueError("Clock model has no REALTIME-to-PHC bridge")
        phc_bridge = CompiledPhcBridge(phc_payload)
        ptp_uncertainty_us = float(
            selected_model.get("ptp_uncertainty_us")
            or session.get("ptp", {}).get("uncertainty_us")
            or 0.0
        )
    else:
        model = CompiledClockModel(selected_model)
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
    first_phc_ns: int | None = None
    last_phc_ns: int | None = None
    max_bridge_uncertainty_us = 0.0
    max_total_uncertainty_us = 0.0
    inside_trace_events = False

    def replace_timestamp(match: re.Match[str]) -> str:
        nonlocal timestamp_count, first_monotonic_ns, last_monotonic_ns
        nonlocal first_phc_ns, last_phc_ns
        nonlocal max_bridge_uncertainty_us, max_total_uncertainty_us
        if source_base_time_ns is None:
            raise ValueError("Trace timestamp appeared before baseTimeNanoseconds")
        trace_relative_ns = _microseconds_to_ns(match.group("value"))
        local_realtime_ns = source_base_time_ns + trace_relative_ns
        bridge_uncertainty_us = 0.0
        model_uncertainty_us = 0.0
        if phc_bridge is not None:
            aligned_ns, bridge_uncertainty_us = phc_bridge.realtime_to_phc_ns(
                local_realtime_ns,
                expected_boot_id=source_boot_id,
            )
            model_uncertainty_us = ptp_uncertainty_us
            if first_phc_ns is None:
                first_phc_ns = aligned_ns
            last_phc_ns = aligned_ns
        else:
            if bridge is None:
                local_monotonic_ns = 0
            else:
                local_monotonic_ns, bridge_uncertainty_us = (
                    bridge.realtime_to_monotonic_ns(
                        local_realtime_ns,
                        expected_boot_id=source_boot_id,
                    )
                )
            if model is None:
                raise ValueError("Software alignment is missing a clock model")
            aligned_ns, model_uncertainty_us = model.align_realtime_ns(
                local_realtime_ns,
                local_monotonic_ns,
            )
            if bridge is not None:
                if first_monotonic_ns is None:
                    first_monotonic_ns = local_monotonic_ns
                last_monotonic_ns = local_monotonic_ns
        aligned_trace_ns = aligned_ns - target_base_time_ns
        timestamp_count += 1
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
        first_source_phc_ns=first_phc_ns,
        last_source_phc_ns=last_phc_ns,
        bridge_boot_id=(
            phc_bridge.boot_id
            if phc_bridge is not None
            else (bridge.boot_id if bridge is not None else None)
        ),
        clock_source=clock_source,
        max_bridge_uncertainty_us=max_bridge_uncertainty_us,
        max_ptp_uncertainty_us=ptp_uncertainty_us,
        max_total_uncertainty_us=max_total_uncertainty_us,
    )
