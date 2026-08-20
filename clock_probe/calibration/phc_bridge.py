"""Map local CLOCK_REALTIME timestamps onto a NIC PHC."""

from __future__ import annotations

import bisect
from dataclasses import asdict, replace
from typing import Any, Sequence

from .core import Candidate, percentile as _percentile, select_candidate
from .clock_bridge import (
    ClockBridgeConfig,
    CompiledClockBridge,
    build_clock_bridge,
)

def _as_axis_samples(
    samples: Sequence[dict[str, Any]],
    config: ClockBridgeConfig,
) -> tuple[list[dict[str, int]], int]:
    mapped: list[dict[str, Any]] = []
    rejected = 0
    max_span_ns = round(config.max_read_span_us * 1_000.0)
    for sample in samples:
        required = (
            "bridge_phc_ns",
            "bridge_realtime_ns",
            "bridge_read_span_ns",
        )
        if any(key not in sample for key in required):
            continue
        read_span_ns = int(sample["bridge_read_span_ns"])
        if read_span_ns < 0 or read_span_ns > max_span_ns:
            rejected += 1
            continue
        mapped.append(
            {
                "phc_ns": int(sample["bridge_phc_ns"]),
                "realtime_ns": int(sample["bridge_realtime_ns"]),
                "read_span_ns": read_span_ns,
            }
        )
    mapped.sort(key=lambda sample: sample["realtime_ns"])
    return mapped, rejected


def _interpolate_ns(
    left: dict[str, int],
    right: dict[str, int],
    realtime_ns: int,
) -> int:
    realtime_span = right["realtime_ns"] - left["realtime_ns"]
    if realtime_span <= 0:
        raise ValueError("PHC bridge REALTIME knots must be strictly increasing")
    phc_span = right["phc_ns"] - left["phc_ns"]
    if phc_span <= 0:
        raise ValueError("PHC bridge PHC knots must be strictly increasing")
    elapsed = realtime_ns - left["realtime_ns"]
    return left["phc_ns"] + round(phc_span * elapsed / realtime_span)


def _build_interpolated_phc_bridge(
    samples: Sequence[dict[str, Any]],
    *,
    boot_id: str | None = None,
    config: ClockBridgeConfig | None = None,
    sample_stride: int = 1,
) -> dict[str, Any]:
    """Build a held-out-validated local interpolation from REALTIME to PHC."""
    selected = config or ClockBridgeConfig()
    selected.validate()
    mapped, rejected = _as_axis_samples(samples, selected)
    if sample_stride <= 0:
        raise ValueError("PHC interpolation sample_stride must be positive")
    mapped = mapped[::sample_stride]
    if len(mapped) < selected.min_segment_samples:
        raise ValueError("No PHC bridge samples contained PHC and REALTIME")

    validation_errors_us: list[float] = []
    for left, sample, right in zip(mapped, mapped[1:], mapped[2:]):
        predicted_phc_ns = _interpolate_ns(
            left,
            right,
            sample["realtime_ns"],
        )
        validation_errors_us.append(
            abs(predicted_phc_ns - sample["phc_ns"]) / 1_000.0
        )
    if not validation_errors_us:
        raise ValueError("PHC bridge has no interior held-out validation samples")

    validation_p95_us = _percentile(validation_errors_us, 0.95)
    validation_max_us = max(validation_errors_us)
    max_read_span_us = max(
        sample["read_span_ns"] for sample in mapped
    ) / 1_000.0
    uncertainty_us = validation_max_us + max_read_span_us / 2.0
    passed = validation_p95_us <= selected.max_validation_p95_us
    return {
        "schema_version": 2,
        "model_type": "interpolated_realtime_phc",
        "source_domain": "CLOCK_REALTIME",
        "target_domain": "PHC",
        "boot_id": boot_id or "",
        "status": "PASS" if passed else "FAIL",
        "config": asdict(selected),
        "sample_stride": sample_stride,
        "valid_from_realtime_ns": mapped[0]["realtime_ns"],
        "valid_to_realtime_ns": mapped[-1]["realtime_ns"],
        "valid_from_phc_ns": mapped[0]["phc_ns"],
        "valid_to_phc_ns": mapped[-1]["phc_ns"],
        "knots": [
            {
                "realtime_ns": sample["realtime_ns"],
                "phc_ns": sample["phc_ns"],
            }
            for sample in mapped
        ],
        "validation_sample_count": len(validation_errors_us),
        "validation_p95_error_us": validation_p95_us,
        "validation_max_error_us": validation_max_us,
        "max_read_span_us": max_read_span_us,
        "uncertainty_us": uncertainty_us,
        "health": {
            "raw_sample_count": len(samples),
            "accepted_sample_count": len(mapped),
            "rejected_read_span_count": rejected,
        },
    }


def _build_affine_phc_bridge(
    samples: Sequence[dict[str, Any]],
    *,
    boot_id: str | None,
    config: ClockBridgeConfig,
    segment_seconds: float,
) -> dict[str, Any]:
    axis_samples = [
        {
            "bridge_monotonic_ns": int(sample["bridge_phc_ns"]),
            "bridge_realtime_ns": int(sample["bridge_realtime_ns"]),
            "bridge_read_span_ns": int(sample["bridge_read_span_ns"]),
        }
        for sample in samples
        if all(
            key in sample
            for key in (
                "bridge_phc_ns",
                "bridge_realtime_ns",
                "bridge_read_span_ns",
            )
        )
    ]
    selected = replace(config, segment_seconds=segment_seconds)
    bridge = build_clock_bridge(
        axis_samples,
        boot_id=boot_id,
        config=selected,
    )
    bridge["model_type"] = "piecewise_affine_realtime_phc"
    bridge["source_domain"] = "CLOCK_REALTIME"
    bridge["target_domain"] = "PHC"
    values: list[float] = []
    for segment in bridge.get("segments", []):
        segment["valid_from_phc_ns"] = segment["valid_from_monotonic_ns"]
        segment["valid_to_phc_ns"] = segment["valid_to_monotonic_ns"]
        segment["base_phc_ns"] = segment["base_monotonic_ns"]
        if segment.get("status") == "PASS":
            values.append(float(segment["uncertainty_us"]))
    if not values:
        raise ValueError("Affine PHC bridge has no PASS segments")
    bridge["uncertainty_us"] = max(values)
    bridge["valid_from_phc_ns"] = min(
        int(segment["valid_from_phc_ns"]) for segment in bridge["segments"]
    )
    bridge["valid_to_phc_ns"] = max(
        int(segment["valid_to_phc_ns"]) for segment in bridge["segments"]
    )
    bridge["affine_segment_seconds"] = segment_seconds
    return bridge


def _bridge_score(bridge: dict[str, Any]) -> dict[str, float]:
    if int(bridge.get("health", {}).get("skipped_group_count", 0)):
        raise ValueError("PHC bridge has skipped affine groups")
    validation_p95 = bridge.get("validation_p95_error_us")
    validation_max = bridge.get("validation_max_error_us")
    if validation_p95 is None:
        segments = bridge.get("segments", [])
        validation_p95 = max(
            float(segment["validation_p95_error_us"]) for segment in segments
        )
        validation_max = max(
            float(segment["validation_max_error_us"]) for segment in segments
        )
    return {
        "uncertainty_us": float(bridge["uncertainty_us"]),
        "validation_p95_error_us": float(validation_p95),
        "validation_max_error_us": float(validation_max),
    }


def build_phc_bridge(  # pylint: disable=too-many-arguments,too-many-locals
    samples: Sequence[dict[str, Any]],
    *,
    boot_id: str | None = None,
    config: ClockBridgeConfig | None = None,
    method: str = "auto",
    candidate_segment_seconds: Sequence[float] = (0.5, 1.0, 2.0, 5.0, 10.0),
    candidate_sample_strides: Sequence[int] = (1, 2, 4),
    tuning_fraction: float = 0.6,
    max_uncertainty_us: float | None = None,
) -> dict[str, Any]:
    """Build or automatically select a REALTIME-to-PHC bridge."""
    selected = config or ClockBridgeConfig()
    selected.validate()
    if method == "interpolation":
        return _build_interpolated_phc_bridge(
            samples,
            boot_id=boot_id,
            config=selected,
        )
    if method == "piecewise_affine":
        return _build_affine_phc_bridge(
            samples,
            boot_id=boot_id,
            config=selected,
            segment_seconds=selected.segment_seconds,
        )
    if method != "auto":
        raise ValueError(f"Unsupported PHC bridge method {method!r}")
    candidates = [
        Candidate(
            value=("interpolation", float(stride)),
            description={"method": "interpolation", "parameter": float(stride)},
            complexity=1,
        )
        for stride in candidate_sample_strides
    ]
    candidates.extend(
        Candidate(
            value=("piecewise_affine", float(seconds)),
            description={
                "method": "piecewise_affine",
                "parameter": float(seconds),
            },
            complexity=0,
        )
        for seconds in candidate_segment_seconds
    )

    def build(
        candidate: tuple[str, float],
        selected_samples: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        candidate_method, parameter = candidate
        if candidate_method == "interpolation":
            return _build_interpolated_phc_bridge(
                selected_samples,
                boot_id=boot_id,
                config=selected,
                sample_stride=round(parameter),
            )
        return _build_affine_phc_bridge(
            selected_samples,
            boot_id=boot_id,
            config=selected,
            segment_seconds=parameter,
        )

    final_bridge, selection = select_candidate(
        samples,
        candidates,
        tuning_fraction=tuning_fraction,
        time_key=lambda sample: int(sample["bridge_realtime_ns"]),
        build=build,
        score=_bridge_score,
        objective_key="uncertainty_us",
        budget=max_uncertainty_us,
        mark_failed=lambda bridge: bridge.__setitem__("status", "FAIL"),
    )
    selected_description = selection.pop("selected")
    selection["selected_method"] = selected_description["method"]
    selection["selected_parameter"] = selected_description["parameter"]
    final_bridge["model_selection"] = selection
    return final_bridge


class CompiledPhcBridge:  # pylint: disable=too-few-public-methods
    """Fast inverse lookup from CLOCK_REALTIME to the local PHC."""

    def __init__(self, bridge: dict[str, Any]):
        if bridge.get("status") != "PASS":
            raise ValueError("PHC bridge has not passed validation")
        if bridge.get("target_domain") not in {None, "PHC"}:
            raise ValueError(
                f"Expected a PHC bridge, got target_domain="
                f"{bridge.get('target_domain')!r}"
            )
        self.boot_id = str(bridge.get("boot_id", ""))
        self._knots = list(bridge.get("knots", []))
        self._inner: CompiledClockBridge | None = None
        if self._knots:
            if len(self._knots) < 2:
                raise ValueError("PHC bridge has fewer than two interpolation knots")
            self._realtime = [int(knot["realtime_ns"]) for knot in self._knots]
        else:
            self._realtime = []
            self._inner = CompiledClockBridge(bridge)
        self._uncertainty_us = float(bridge["uncertainty_us"])

    def realtime_to_phc_ns(
        self,
        realtime_ns: int,
        *,
        expected_boot_id: str | None = None,
    ) -> tuple[int, float]:
        """Map one REALTIME timestamp onto the local PHC."""
        if expected_boot_id is not None and expected_boot_id != self.boot_id:
            raise ValueError(
                "PHC bridge boot ID does not match the Trace source boot ID"
            )
        if self._inner is not None:
            return self._inner.realtime_to_monotonic_ns(
                realtime_ns,
                expected_boot_id=expected_boot_id,
            )
        index = bisect.bisect_left(self._realtime, realtime_ns)
        if index < len(self._realtime) and self._realtime[index] == realtime_ns:
            return int(self._knots[index]["phc_ns"]), self._uncertainty_us
        if index == 0 or index == len(self._knots):
            raise ValueError(
                f"No PHC interpolation interval covers realtime timestamp "
                f"{realtime_ns}"
            )
        phc_ns = _interpolate_ns(
            self._knots[index - 1],
            self._knots[index],
            realtime_ns,
        )
        return phc_ns, self._uncertainty_us
