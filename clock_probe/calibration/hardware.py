"""Build per-node PHC models and a shared hardware clock session."""

from __future__ import annotations

import json
import socket
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .clock_bridge import ClockBridgeConfig, read_boot_id
from ..sampling.phc import (
    PhcClock,
    capture_phc_sample,
)
from .phc_bridge import build_phc_bridge
from .ptp_health import PtpHealth, ptp_uncertainty_us

HARDWARE_SCHEMA_VERSION = 3
CLOCK_SOURCE = "ptp_hardware"
HOUR_NS = 3_600_000_000_000
HARDWARE_MAX_TOTAL_UNCERTAINTY_US = 2.0


@dataclass(frozen=True)
class HardwareModelConfig:  # pylint: disable=too-many-instance-attributes
    """Quality gates for one PHC-backed node model."""

    segment_seconds: float = 30.0
    validation_fraction: float = 0.2
    max_bridge_p95_us: float = 1.0
    min_segment_samples: int = 10
    step_threshold_us: float = 1_000.0
    max_read_span_us: float = 100.0
    capture_attempts: int = 10
    max_ptp_offset_p95_ns: float = 1_000.0
    path_delay_asymmetry: float = 0.1
    max_total_uncertainty_us: float = 2.0
    phc_bridge_method: str = "auto"
    candidate_segment_seconds: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0)
    candidate_sample_strides: tuple[int, ...] = (1, 2, 4)
    tuning_fraction: float = 0.6

    def bridge_config(self) -> ClockBridgeConfig:
        """Return the REALTIME-PHC bridge config."""
        return ClockBridgeConfig(
            segment_seconds=self.segment_seconds,
            validation_fraction=self.validation_fraction,
            max_validation_p95_us=self.max_bridge_p95_us,
            min_segment_samples=self.min_segment_samples,
            step_threshold_us=self.step_threshold_us,
            max_read_span_us=self.max_read_span_us,
            capture_attempts=self.capture_attempts,
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect_phc_samples(
    *,
    phc: PhcClock,
    duration_seconds: float,
    interval_ms: float,
    attempts: int = 5,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Sample PHC/REALTIME/MONOTONIC triplets for a fixed duration."""
    if duration_seconds <= 0 or interval_ms <= 0:
        raise ValueError("PHC collection duration and interval must be positive")
    samples: list[dict[str, Any]] = []
    output_file = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_path.open("w", encoding="utf-8", buffering=1)
    deadline = time.monotonic() + duration_seconds
    interval_s = interval_ms / 1_000.0
    next_at = time.monotonic()
    try:
        while time.monotonic() < deadline:
            sample = capture_phc_sample(phc, attempts=attempts)
            samples.append(sample)
            if output_file is not None:
                output_file.write(json.dumps(sample) + "\n")
            next_at += interval_s
            wait_s = next_at - time.monotonic()
            if wait_s > 0:
                time.sleep(wait_s)
    finally:
        if output_file is not None:
            output_file.close()
    return samples


def load_phc_samples(path: Path) -> list[dict[str, Any]]:
    """Load JSONL PHC bridge samples."""
    samples: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        samples.append(json.loads(line))
    return samples


def build_hardware_model(  # pylint: disable=too-many-arguments
    samples: Sequence[dict[str, Any]],
    *,
    role: str,
    ptp_health: PtpHealth,
    source: dict[str, Any] | None = None,
    config: HardwareModelConfig | None = None,
    collection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit one node's REALTIME-PHC bridge and attach ptp4l health."""
    selected = config or HardwareModelConfig()
    if (
        selected.max_total_uncertainty_us <= 0
        or selected.max_total_uncertainty_us
        > HARDWARE_MAX_TOTAL_UNCERTAINTY_US
    ):
        raise ValueError(
            "Hardware total uncertainty gate must be positive and no greater "
            f"than {HARDWARE_MAX_TOTAL_UNCERTAINTY_US:.1f} us"
        )
    identity = {
        "hostname": socket.gethostname(),
        "boot_id": read_boot_id(),
        **(source or {}),
    }
    ptp_us = ptp_uncertainty_us(
        ptp_health,
        path_delay_asymmetry=selected.path_delay_asymmetry,
    )
    remaining_budget_us = selected.max_total_uncertainty_us - ptp_us
    bridge = build_phc_bridge(
        samples,
        boot_id=str(identity["boot_id"]),
        config=selected.bridge_config(),
        method=selected.phc_bridge_method,
        candidate_segment_seconds=selected.candidate_segment_seconds,
        candidate_sample_strides=selected.candidate_sample_strides,
        tuning_fraction=selected.tuning_fraction,
        max_uncertainty_us=remaining_budget_us,
    )
    bridge_us = float(bridge.get("uncertainty_us", 0.0))
    if not bridge_us and bridge.get("segments"):
        bridge_us = max(
            float(segment["uncertainty_us"])
            for segment in bridge["segments"]
            if segment.get("status") == "PASS"
        )
    status = "PASS"
    reasons: list[str] = []
    if ptp_health.role != role:
        status = "FAIL"
        reasons.append(
            f"ptp health role {ptp_health.role!r} does not match {role!r}"
        )
    if ptp_health.status != "PASS":
        status = "FAIL"
        reasons.extend(ptp_health.reasons)
    if bridge.get("status") != "PASS":
        status = "FAIL"
        reasons.append("REALTIME-PHC bridge failed validation")
    total_uncertainty_us = bridge_us + ptp_us
    if total_uncertainty_us > selected.max_total_uncertainty_us:
        status = "FAIL"
        reasons.append(
            f"total uncertainty {total_uncertainty_us:.3f} us exceeds "
            f"{selected.max_total_uncertainty_us:.3f} us"
        )
    return {
        "schema_version": HARDWARE_SCHEMA_VERSION,
        "model_type": "phc_bridge",
        "clock_source": CLOCK_SOURCE,
        "timestamp_domain": "PHC",
        "offset_direction": "phc_minus_realtime",
        "status": status,
        "source": identity,
        "ptp": ptp_health.to_dict(),
        "realtime_phc_bridge": bridge,
        "uncertainty_us": total_uncertainty_us,
        "ptp_uncertainty_us": ptp_us,
        "bridge_uncertainty_us": bridge_us,
        "fail_reasons": reasons,
        "collection": collection or {},
        "config": asdict(selected),
    }


def _model_phc_start_ns(model: dict[str, Any]) -> int:
    bridge = model.get("realtime_phc_bridge", {})
    if bridge.get("valid_from_phc_ns") is not None:
        return int(bridge["valid_from_phc_ns"])
    segments = bridge.get("segments", [])
    if not segments:
        raise ValueError("Hardware model has no PHC bridge segments")
    return int(segments[0].get("valid_from_phc_ns", segments[0]["valid_from_monotonic_ns"]))


def _trace_base_from_phc(models: Sequence[dict[str, Any]]) -> int:
    first_phc_ns = min(_model_phc_start_ns(model) for model in models)
    return first_phc_ns // HOUR_NS * HOUR_NS


def build_hardware_session(
    models: Sequence[dict[str, Any]],
    *,
    session_id: str | None = None,
    path_delay_asymmetry: float = 0.1,
) -> dict[str, Any]:
    """Merge per-node PHC models. All nodes must share one grandmaster."""
    if len(models) < 2:
        raise ValueError("A hardware session needs at least two node models")
    gm_ids = {
        model.get("ptp", {}).get("grandmaster_clock_id")
        for model in models
    }
    gm_ids.discard(None)
    if len(gm_ids) != 1:
        raise ValueError(
            f"Hardware models do not share one grandmaster clock id: {sorted(gm_ids)}"
        )
    roles = {model.get("ptp", {}).get("role") for model in models}
    if "master" not in roles or "slave" not in roles:
        raise ValueError("Hardware session needs one master model and at least one slave")
    failed = [
        model
        for model in models
        if model.get("status") != "PASS"
        or float(model.get("uncertainty_us", float("inf")))
        > HARDWARE_MAX_TOTAL_UNCERTAINTY_US
    ]
    node_uncertainties = [
        float(model["uncertainty_us"])
        for model in models
    ]
    ptp_uncertainties = [
        float(model.get("ptp_uncertainty_us", 0.0))
        for model in models
        if model.get("ptp", {}).get("role") == "slave"
    ]
    return {
        "schema_version": HARDWARE_SCHEMA_VERSION,
        "clock_source": CLOCK_SOURCE,
        "timestamp_domain": "PHC",
        "system_clock_policy": "phc_only_never_phc2sys",
        "session_id": session_id or f"ptp-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
        "started_at": min(
            (model.get("collection", {}).get("started_at") or _utc_now())
            for model in models
        ),
        "completed_at": _utc_now(),
        "target_base_time_ns": _trace_base_from_phc(models),
        "status": "PASS" if not failed else "FAIL",
        "ptp": {
            "grandmaster_clock_id": next(iter(gm_ids)),
            "path_delay_asymmetry": path_delay_asymmetry,
            "uncertainty_us": max(ptp_uncertainties, default=0.0),
            "notes": [
                "ptp4l rms is PHC-PHC residual, not Kineto end-to-end error",
                "CPU and GPU events stay on CLOCK_REALTIME and share one PHC bridge",
            ],
        },
        "node_count": len(models),
        "models": list(models),
        "failures": [
            {
                "hostname": model.get("source", {}).get("hostname"),
                "reasons": (
                    model.get("fail_reasons", [])
                    or [
                        "total uncertainty exceeds the hard "
                        f"{HARDWARE_MAX_TOTAL_UNCERTAINTY_US:.1f} us limit"
                    ]
                ),
            }
            for model in failed
        ],
        "max_uncertainty_us": max(node_uncertainties, default=0.0),
    }
