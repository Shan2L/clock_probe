"""Parse ptp4l logs and fail closed unless the NIC PHC stays locked."""

from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .core import percentile as _percentile

SUMMARY_RE = re.compile(
    r"ptp4l\[(?P<mono>[0-9.]+)\]:\s+"
    r"rms\s+(?P<rms>-?\d+)\s+max\s+(?P<max>-?\d+)\s+"
    r"freq\s+(?P<freq>-?\d+)\s+\+/-\s+(?P<freq_dev>\d+)"
    r"(?:\s+delay\s+(?P<delay>\d+)\s+\+/-\s+(?P<delay_dev>\d+))?"
)
STATE_RE = re.compile(
    r"ptp4l\[(?P<mono>[0-9.]+)\]:\s+port \d+ \((?P<iface>[^/)][^)]*)\): "
    r"\S+ to (?P<state>MASTER|SLAVE|LISTENING|UNCALIBRATED|FAULTY|DISABLED)"
)
GM_FOREIGN_RE = re.compile(
    r"selected best master clock (?P<clock_id>[0-9a-fA-F.]+)"
)
GM_LOCAL_RE = re.compile(
    r"selected local clock (?P<clock_id>[0-9a-fA-F.]+) as best master"
)
CLOCKCHECK_RE = re.compile(
    r"ptp4l\[(?P<mono>[0-9.]+)\]:\s+clockcheck: clock frequency changed unexpectedly"
)
ASSUMING_GM_RE = re.compile(r"assuming the grand master role")

ALLOWED_STATES = {"MASTER", "SLAVE"}


@dataclass
class PtpSummarySample:
    """One ptp4l one-second summary line."""

    monotonic_s: float
    rms_ns: int
    max_ns: int
    freq_ppb: int
    delay_ns: int | None


@dataclass
class PtpHealth:  # pylint: disable=too-many-instance-attributes
    """Lock quality derived from a ptp4l log. Not an independent validation stream."""

    role: str
    port_state: str | None
    grandmaster_clock_id: str | None
    assuming_grandmaster: bool
    summary_count: int
    clockcheck_count: int
    late_clockcheck_count: int
    offset_rms_p50_ns: float | None
    offset_rms_p95_ns: float | None
    offset_max_ns: int | None
    path_delay_ns: float | None
    freq_ppb: float | None
    lock_ok: bool
    status: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable health record."""
        return asdict(self)


def parse_ptp4l_log(text: str) -> dict[str, Any]:
    """Extract port state, GM identity, and offset summaries from a ptp4l log."""
    states = [
        {"monotonic_s": float(match.group("mono")), "state": match.group("state")}
        for match in STATE_RE.finditer(text)
    ]
    summaries: list[PtpSummarySample] = []
    for match in SUMMARY_RE.finditer(text):
        delay = match.group("delay")
        summaries.append(
            PtpSummarySample(
                monotonic_s=float(match.group("mono")),
                rms_ns=int(match.group("rms")),
                max_ns=int(match.group("max")),
                freq_ppb=int(match.group("freq")),
                delay_ns=int(delay) if delay is not None else None,
            )
        )
    clockchecks = [float(match.group("mono")) for match in CLOCKCHECK_RE.finditer(text)]
    gm_ids = [match.group("clock_id") for match in GM_FOREIGN_RE.finditer(text)]
    local_ids = [match.group("clock_id") for match in GM_LOCAL_RE.finditer(text)]
    return {
        "states": states,
        "summaries": summaries,
        "clockcheck_monotonic_s": clockchecks,
        "foreign_master_ids": gm_ids,
        "local_clock_ids": local_ids,
        "assuming_grandmaster": bool(ASSUMING_GM_RE.search(text)),
    }


def evaluate_ptp_health(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    text: str,
    *,
    role: str,
    max_offset_p95_ns: float = 1_000.0,
    settle_summaries: int = 10,
    late_clockcheck_window_s: float = 10.0,
) -> PtpHealth:
    """Fail closed unless the log ends in a locked MASTER or SLAVE state."""
    if role not in {"master", "slave"}:
        raise ValueError("PTP role must be 'master' or 'slave'")
    parsed = parse_ptp4l_log(text)
    reasons: list[str] = []
    port_state = (
        parsed["states"][-1]["state"] if parsed["states"] else None
    )
    summaries: list[PtpSummarySample] = parsed["summaries"]
    settled = summaries[settle_summaries:] if len(summaries) > settle_summaries else summaries
    last_mono = None
    if summaries:
        last_mono = summaries[-1].monotonic_s
    elif parsed["states"]:
        last_mono = parsed["states"][-1]["monotonic_s"]
    late_clockchecks = 0
    if last_mono is not None:
        late_clockchecks = sum(
            1
            for stamp in parsed["clockcheck_monotonic_s"]
            if last_mono - stamp <= late_clockcheck_window_s
        )

    grandmaster_clock_id: str | None = None
    if role == "slave":
        if parsed["foreign_master_ids"]:
            grandmaster_clock_id = parsed["foreign_master_ids"][-1]
    elif parsed["local_clock_ids"]:
        grandmaster_clock_id = parsed["local_clock_ids"][-1]
    elif parsed["foreign_master_ids"]:
        grandmaster_clock_id = parsed["foreign_master_ids"][-1]

    expected_state = "MASTER" if role == "master" else "SLAVE"
    if port_state != expected_state:
        reasons.append(
            f"port_state is {port_state!r}, expected {expected_state} for role {role}"
        )
    if role == "master" and not parsed["assuming_grandmaster"] and port_state != "MASTER":
        reasons.append("master log never assumed the grand master role")
    if role == "slave" and grandmaster_clock_id is None:
        reasons.append("slave log has no selected best master clock")
    if role == "slave" and not settled:
        reasons.append("slave log has no ptp4l rms summaries after lock")
    if late_clockchecks:
        reasons.append(
            f"{late_clockchecks} clockcheck warning(s) in the last "
            f"{late_clockcheck_window_s:.0f}s"
        )

    offset_rms_p50_ns = None
    offset_rms_p95_ns = None
    offset_max_ns = None
    path_delay_ns = None
    freq_ppb = None
    if settled:
        rms_values = [float(sample.rms_ns) for sample in settled]
        offset_rms_p50_ns = statistics.median(rms_values)
        offset_rms_p95_ns = _percentile(rms_values, 0.95)
        offset_max_ns = max(sample.max_ns for sample in settled)
        delays = [
            float(sample.delay_ns)
            for sample in settled
            if sample.delay_ns is not None
        ]
        if delays:
            path_delay_ns = statistics.median(delays)
        freq_ppb = statistics.median(float(sample.freq_ppb) for sample in settled)
        if offset_rms_p95_ns > max_offset_p95_ns:
            reasons.append(
                f"ptp4l rms p95 {offset_rms_p95_ns:.1f} ns exceeds "
                f"{max_offset_p95_ns:.1f} ns"
            )

    lock_ok = not reasons
    return PtpHealth(
        role=role,
        port_state=port_state,
        grandmaster_clock_id=grandmaster_clock_id,
        assuming_grandmaster=bool(parsed["assuming_grandmaster"]),
        summary_count=len(summaries),
        clockcheck_count=len(parsed["clockcheck_monotonic_s"]),
        late_clockcheck_count=late_clockchecks,
        offset_rms_p50_ns=offset_rms_p50_ns,
        offset_rms_p95_ns=offset_rms_p95_ns,
        offset_max_ns=offset_max_ns,
        path_delay_ns=path_delay_ns,
        freq_ppb=freq_ppb,
        lock_ok=lock_ok,
        status="PASS" if lock_ok else "FAIL",
        reasons=reasons,
    )


def load_ptp_health(path: Path, *, role: str, **kwargs: Any) -> PtpHealth:
    """Read one ptp4l log and evaluate lock health."""
    return evaluate_ptp_health(path.read_text(encoding="utf-8"), role=role, **kwargs)


def ptp_uncertainty_us(
    health: PtpHealth,
    *,
    path_delay_asymmetry: float = 0.1,
) -> float:
    """Conservative PHC-PHC bound from ptp4l offset plus delay asymmetry."""
    if path_delay_asymmetry < 0 or path_delay_asymmetry > 0.5:
        raise ValueError("path_delay_asymmetry must be between 0 and 0.5")
    offset_ns = float(health.offset_rms_p95_ns or 0.0)
    delay_ns = float(health.path_delay_ns or 0.0)
    return (offset_ns + delay_ns * path_delay_asymmetry) / 1_000.0
