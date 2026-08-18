"""Map local CLOCK_REALTIME timestamps to CLOCK_MONOTONIC."""

from __future__ import annotations

# The percentile implementation intentionally mirrors the clock-model validator.
# pylint: disable=duplicate-code

import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")


@dataclass(frozen=True)
class ClockBridgeConfig:
    """Quality and segmentation parameters for one local clock bridge."""

    segment_seconds: float = 30.0
    validation_fraction: float = 0.2
    max_validation_p95_us: float = 20.0
    min_segment_samples: int = 10
    step_threshold_us: float = 1_000.0
    max_read_span_us: float = 100.0
    capture_attempts: int = 5

    def validate(self) -> None:
        """Reject internally inconsistent bridge settings."""
        if self.segment_seconds <= 0:
            raise ValueError("Bridge segment duration must be positive")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("Bridge validation_fraction must be between 0 and 0.5")
        if self.min_segment_samples < 3:
            raise ValueError("Bridge min_segment_samples must be at least 3")
        if self.step_threshold_us <= 0 or self.max_read_span_us <= 0:
            raise ValueError("Bridge thresholds must be positive")
        if self.capture_attempts <= 0:
            raise ValueError("Bridge capture_attempts must be positive")


def read_boot_id() -> str:
    """Return the Linux boot identifier for the current node."""
    return BOOT_ID_PATH.read_text(encoding="utf-8").strip()


def capture_clock_pair(attempts: int = 5) -> dict[str, int]:
    """Capture the tightest bracketed REALTIME/MONOTONIC clock pair."""
    if attempts <= 0:
        raise ValueError("Clock-pair attempts must be positive")
    candidates: list[dict[str, int]] = []
    for _ in range(attempts):
        before_ns = time.monotonic_ns()
        realtime_ns = time.time_ns()
        after_ns = time.monotonic_ns()
        read_span_ns = after_ns - before_ns
        monotonic_ns = before_ns + read_span_ns // 2
        candidates.append(
            {
                "bridge_monotonic_ns": monotonic_ns,
                "bridge_realtime_ns": realtime_ns,
                "bridge_offset_ns": realtime_ns - monotonic_ns,
                "bridge_read_span_ns": read_span_ns,
            }
        )
    return min(candidates, key=lambda pair: pair["bridge_read_span_ns"])


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _split_train_validation(
    samples: Sequence[dict[str, float]],
    validation_fraction: float,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    stride = max(2, round(1.0 / validation_fraction))
    validation = [
        sample
        for index, sample in enumerate(samples)
        if index % stride == stride - 1
    ]
    training = [
        sample
        for index, sample in enumerate(samples)
        if index % stride != stride - 1
    ]
    if not validation and len(training) > 2:
        validation = [training.pop()]
    return training, validation


def _fit_bridge(samples: Sequence[dict[str, float]]) -> dict[str, float]:
    """Fit realtime-minus-monotonic as a function of monotonic time."""
    if len(samples) < 2:
        raise ValueError("At least two bridge samples are required")
    base_ns = float(samples[0]["monotonic_ns"])
    x_values = [sample["monotonic_ns"] - base_ns for sample in samples]
    y_values = [sample["offset_ns"] for sample in samples]
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0:
        raise ValueError("Bridge samples do not span time")
    drift = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    ) / denominator
    offset_at_base_ns = y_mean - drift * x_mean
    return {
        "base_monotonic_ns": base_ns,
        "offset_at_base_ns": offset_at_base_ns,
        "drift_ns_per_ns": drift,
        "drift_ppm": drift * 1_000_000.0,
    }


def _predict_realtime(monotonic_ns: float, fit: dict[str, float]) -> float:
    elapsed_ns = monotonic_ns - fit["base_monotonic_ns"]
    offset_ns = fit["offset_at_base_ns"] + fit["drift_ns_per_ns"] * elapsed_ns
    return monotonic_ns + offset_ns


def _normalize_samples(
    samples: Sequence[dict[str, Any]],
    config: ClockBridgeConfig,
) -> tuple[list[dict[str, float]], int]:
    normalized: list[dict[str, float]] = []
    rejected = 0
    max_span_ns = config.max_read_span_us * 1_000.0
    for sample in samples:
        required = (
            "bridge_monotonic_ns",
            "bridge_realtime_ns",
            "bridge_read_span_ns",
        )
        if any(key not in sample for key in required):
            continue
        span_ns = float(sample["bridge_read_span_ns"])
        if span_ns < 0 or span_ns > max_span_ns:
            rejected += 1
            continue
        monotonic_ns = float(sample["bridge_monotonic_ns"])
        realtime_ns = float(sample["bridge_realtime_ns"])
        normalized.append(
            {
                "monotonic_ns": monotonic_ns,
                "realtime_ns": realtime_ns,
                "offset_ns": realtime_ns - monotonic_ns,
                "read_span_ns": span_ns,
            }
        )
    return sorted(normalized, key=lambda item: item["monotonic_ns"]), rejected


def _group_segments(
    samples: Sequence[dict[str, float]],
    config: ClockBridgeConfig,
) -> list[tuple[list[dict[str, float]], bool]]:
    """Split fixed-duration groups and start a new epoch after a wall-clock step."""
    segment_ns = config.segment_seconds * 1_000_000_000.0
    step_ns = config.step_threshold_us * 1_000.0
    groups: list[tuple[list[dict[str, float]], bool]] = []
    current: list[dict[str, float]] = []
    step_before_current = False
    group_origin_ns = 0.0
    previous_offset_ns: float | None = None
    for sample in samples:
        offset_ns = sample["offset_ns"]
        stepped = (
            previous_offset_ns is not None
            and abs(offset_ns - previous_offset_ns) > step_ns
        )
        expired = current and sample["monotonic_ns"] - group_origin_ns >= segment_ns
        if current and (stepped or expired):
            groups.append((current, step_before_current))
            current = []
        if not current:
            group_origin_ns = sample["monotonic_ns"]
            step_before_current = stepped
        current.append(sample)
        previous_offset_ns = offset_ns
    if current:
        groups.append((current, step_before_current))
    return groups


def _build_segment(
    index: int,
    samples: Sequence[dict[str, float]],
    config: ClockBridgeConfig,
    *,
    step_before: bool,
) -> dict[str, Any]:
    training, validation = _split_train_validation(
        samples,
        config.validation_fraction,
    )
    fit = _fit_bridge(training)
    validation_errors_us = [
        abs(
            sample["realtime_ns"]
            - _predict_realtime(sample["monotonic_ns"], fit)
        )
        / 1_000.0
        for sample in validation
    ]
    training_errors_us = [
        abs(
            sample["realtime_ns"]
            - _predict_realtime(sample["monotonic_ns"], fit)
        )
        / 1_000.0
        for sample in training
    ]
    validation_p95_us = _percentile(validation_errors_us, 0.95)
    valid_from_monotonic_ns = int(samples[0]["monotonic_ns"])
    valid_to_monotonic_ns = int(samples[-1]["monotonic_ns"])
    realtime_bounds = sorted(
        (
            round(_predict_realtime(valid_from_monotonic_ns, fit)),
            round(_predict_realtime(valid_to_monotonic_ns, fit)),
        )
    )
    median_read_span_us = statistics.median(
        sample["read_span_ns"] for sample in samples
    ) / 1_000.0
    passed = validation_p95_us <= config.max_validation_p95_us
    return {
        "segment_index": index,
        "clock_step_before": step_before,
        "status": "PASS" if passed else "FAIL",
        "valid_from_monotonic_ns": valid_from_monotonic_ns,
        "valid_to_monotonic_ns": valid_to_monotonic_ns,
        "valid_from_realtime_ns": realtime_bounds[0],
        "valid_to_realtime_ns": realtime_bounds[1],
        **fit,
        "sample_count": len(samples),
        "training_sample_count": len(training),
        "validation_sample_count": len(validation),
        "training_p95_error_us": _percentile(training_errors_us, 0.95),
        "validation_p95_error_us": validation_p95_us,
        "validation_max_error_us": max(validation_errors_us),
        "median_read_span_us": median_read_span_us,
        "uncertainty_us": validation_p95_us + median_read_span_us / 2.0,
    }


def _set_segment_coverage(
    segment: dict[str, Any],
    valid_from_monotonic_ns: int,
    valid_to_monotonic_ns: int,
) -> None:
    """Update monotonic and corresponding realtime coverage bounds."""
    segment["valid_from_monotonic_ns"] = valid_from_monotonic_ns
    segment["valid_to_monotonic_ns"] = valid_to_monotonic_ns
    realtime_bounds = sorted(
        (
            round(_predict_realtime(valid_from_monotonic_ns, segment)),
            round(_predict_realtime(valid_to_monotonic_ns, segment)),
        )
    )
    segment["valid_from_realtime_ns"] = realtime_bounds[0]
    segment["valid_to_realtime_ns"] = realtime_bounds[1]


def build_clock_bridge(
    samples: Sequence[dict[str, Any]],
    *,
    boot_id: str | None = None,
    config: ClockBridgeConfig | None = None,
) -> dict[str, Any]:
    """Build a validated piecewise REALTIME-to-MONOTONIC bridge."""
    selected_config = config or ClockBridgeConfig()
    selected_config.validate()
    normalized, rejected = _normalize_samples(samples, selected_config)
    if len(normalized) < selected_config.min_segment_samples:
        raise ValueError(
            "Too few healthy clock-pair samples to build a bridge: "
            f"{len(normalized)} < {selected_config.min_segment_samples}"
        )
    groups = _group_segments(normalized, selected_config)
    segments = [
        _build_segment(
            index,
            group,
            selected_config,
            step_before=step_before,
        )
        for index, (group, step_before) in enumerate(groups)
        if len(group) >= selected_config.min_segment_samples
    ]
    skipped_group_count = sum(
        len(group) < selected_config.min_segment_samples
        for group, _ in groups
    )
    if not segments:
        raise ValueError("No clock-bridge segment contained enough samples")
    for previous, following in zip(segments, segments[1:]):
        if following["clock_step_before"]:
            continue
        midpoint_ns = (
            int(previous["valid_to_monotonic_ns"])
            + int(following["valid_from_monotonic_ns"])
        ) // 2
        _set_segment_coverage(
            previous,
            int(previous["valid_from_monotonic_ns"]),
            midpoint_ns,
        )
        _set_segment_coverage(
            following,
            midpoint_ns + 1,
            int(following["valid_to_monotonic_ns"]),
        )
    failed_segment_count = sum(
        segment["status"] != "PASS" for segment in segments
    )
    return {
        "schema_version": 1,
        "model_type": "piecewise_realtime_monotonic",
        "source_domain": "CLOCK_REALTIME",
        "target_domain": "CLOCK_MONOTONIC",
        "boot_id": boot_id or read_boot_id(),
        "status": "PASS" if failed_segment_count == 0 else "FAIL",
        "config": asdict(selected_config),
        "health": {
            "raw_sample_count": len(samples),
            "accepted_sample_count": len(normalized),
            "rejected_read_span_count": rejected,
            "segment_count": len(segments),
            "failed_segment_count": failed_segment_count,
            "skipped_group_count": skipped_group_count,
        },
        "segments": segments,
    }


class CompiledClockBridge:  # pylint: disable=too-few-public-methods
    """Fast inverse lookup from CLOCK_REALTIME to CLOCK_MONOTONIC."""

    def __init__(self, bridge: dict[str, Any]):
        if bridge.get("status") != "PASS":
            raise ValueError("Clock bridge has not passed validation")
        self.boot_id = str(bridge.get("boot_id", ""))
        self.segments = list(bridge.get("segments", []))
        if not self.segments:
            raise ValueError("Clock bridge has no usable segments")

    def realtime_to_monotonic_ns(
        self,
        realtime_ns: int,
        *,
        expected_boot_id: str | None = None,
    ) -> tuple[int, float]:
        """Map one REALTIME timestamp and return its uncertainty in microseconds."""
        if expected_boot_id is not None and expected_boot_id != self.boot_id:
            raise ValueError(
                "Clock bridge boot ID does not match the Trace source boot ID"
            )
        candidates = [
            segment
            for segment in self.segments
            if int(segment["valid_from_realtime_ns"])
            <= realtime_ns
            <= int(segment["valid_to_realtime_ns"])
        ]
        if not candidates:
            raise ValueError(
                f"No clock-bridge segment covers realtime timestamp {realtime_ns}"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Realtime timestamp {realtime_ns} is ambiguous across "
                "clock-bridge segments"
            )
        segment = candidates[0]
        if segment.get("status") != "PASS":
            raise ValueError(
                f"Trace uses failed clock-bridge segment "
                f"{segment['segment_index']}"
            )
        base_monotonic_ns = float(segment["base_monotonic_ns"])
        offset_at_base_ns = float(segment["offset_at_base_ns"])
        drift = float(segment["drift_ns_per_ns"])
        scale = 1.0 + drift
        if scale <= 0:
            raise ValueError("Clock bridge has a non-positive scale")
        base_realtime_ns = base_monotonic_ns + offset_at_base_ns
        monotonic_ns = base_monotonic_ns + (
            realtime_ns - base_realtime_ns
        ) / scale
        rounded_ns = round(monotonic_ns)
        if not (
            int(segment["valid_from_monotonic_ns"])
            <= rounded_ns
            <= int(segment["valid_to_monotonic_ns"])
        ):
            raise ValueError(
                f"Mapped monotonic timestamp {rounded_ns} is outside "
                "clock-bridge coverage"
            )
        return rounded_ns, float(segment["uncertainty_us"])
