"""Piecewise affine clock-model construction and application."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class ModelConfig:  # pylint: disable=too-many-instance-attributes
    """Quality and segmentation parameters for one worker model."""

    window_seconds: float = 1.0
    samples_per_window: int = 3
    rtt_slack_us: float = 20.0
    max_window_rtt_excess_us: float = 50.0
    segment_seconds: float = 30.0
    validation_fraction: float = 0.2
    max_validation_p95_us: float = 20.0
    min_segment_samples: int = 10
    min_segment_coverage: float = 0.7

    def validate(self) -> None:
        """Reject internally inconsistent model settings."""
        if self.window_seconds <= 0 or self.segment_seconds <= 0:
            raise ValueError("Window and segment durations must be positive")
        if self.samples_per_window <= 0:
            raise ValueError("samples_per_window must be positive")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")
        if self.min_segment_samples < 3:
            raise ValueError("min_segment_samples must be at least 3")
        if not 0 < self.min_segment_coverage <= 1:
            raise ValueError("min_segment_coverage must be in (0, 1]")


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


def _fit_affine(samples: Sequence[dict[str, float]]) -> dict[str, float]:
    """Fit offset = offset_at_base + slope * (monotonic - base)."""
    if len(samples) < 2:
        raise ValueError("At least two samples are required for affine fitting")

    base_ns = float(samples[0]["monotonic_ns"])
    x_values = [float(sample["monotonic_ns"]) - base_ns for sample in samples]
    y_values = [float(sample["offset_ns"]) for sample in samples]
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0:
        raise ValueError("Samples do not span time")
    slope = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    ) / denominator
    offset_at_base_ns = y_mean - slope * x_mean
    return {
        "base_monotonic_ns": base_ns,
        "offset_at_base_ns": offset_at_base_ns,
        "drift_ns_per_ns": slope,
        "drift_ppm": slope * 1_000_000.0,
    }


def _predict_offset(monotonic_ns: float, fit: dict[str, float]) -> float:
    return fit["offset_at_base_ns"] + fit["drift_ns_per_ns"] * (
        monotonic_ns - fit["base_monotonic_ns"]
    )


def _residuals_us(
    samples: Iterable[dict[str, float]],
    fit: dict[str, float],
) -> list[float]:
    return [
        abs(
            sample["offset_ns"]
            - _predict_offset(sample["monotonic_ns"], fit)
        )
        / 1_000.0
        for sample in samples
    ]


def _window_samples(
    samples: Sequence[dict[str, Any]],
    config: ModelConfig,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Filter unhealthy windows and create one representative per window."""
    # pylint: disable=too-many-locals
    if not samples:
        raise ValueError("No clock samples were provided")

    ordered = sorted(samples, key=lambda sample: int(sample["monotonic_ns"]))
    origin_ns = int(ordered[0]["monotonic_ns"])
    window_ns = int(config.window_seconds * 1_000_000_000)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for sample in ordered:
        rtt_ns = float(sample["rtt_ns"])
        if rtt_ns < 0:
            continue
        index = (int(sample["monotonic_ns"]) - origin_ns) // window_ns
        grouped.setdefault(index, []).append(sample)

    if not grouped:
        raise ValueError("No samples had a non-negative RTT")

    window_min_rtt_ns = {
        index: min(float(sample["rtt_ns"]) for sample in group)
        for index, group in grouped.items()
    }
    baseline_rtt_ns = statistics.median(window_min_rtt_ns.values())
    maximum_healthy_rtt_ns = (
        baseline_rtt_ns + config.max_window_rtt_excess_us * 1_000.0
    )
    slack_ns = config.rtt_slack_us * 1_000.0

    representatives: list[dict[str, float]] = []
    rejected_window_indexes: list[int] = []
    for index, group in sorted(grouped.items()):
        minimum_rtt_ns = window_min_rtt_ns[index]
        if minimum_rtt_ns > maximum_healthy_rtt_ns:
            rejected_window_indexes.append(index)
            continue

        candidates = sorted(
            (
                sample
                for sample in group
                if float(sample["rtt_ns"]) <= minimum_rtt_ns + slack_ns
            ),
            key=lambda sample: float(sample["rtt_ns"]),
        )[: config.samples_per_window]
        if not candidates:
            rejected_window_indexes.append(index)
            continue

        representatives.append(
            {
                "window_index": float(index),
                "monotonic_ns": statistics.fmean(
                    float(sample["monotonic_ns"]) for sample in candidates
                ),
                "offset_ns": statistics.fmean(
                    float(sample["offset_ns"]) for sample in candidates
                ),
                "rtt_ns": statistics.median(
                    float(sample["rtt_ns"]) for sample in candidates
                ),
                "candidate_count": float(len(candidates)),
            }
        )

    return representatives, {
        "origin_monotonic_ns": origin_ns,
        "raw_sample_count": len(samples),
        "window_count": len(grouped),
        "accepted_window_count": len(representatives),
        "rejected_window_count": len(rejected_window_indexes),
        "rejected_window_indexes": rejected_window_indexes,
        "baseline_window_min_rtt_us": baseline_rtt_ns / 1_000.0,
        "maximum_healthy_window_rtt_us": maximum_healthy_rtt_ns / 1_000.0,
    }


def _split_train_validation(
    samples: Sequence[dict[str, float]],
    validation_fraction: float,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    validation_stride = max(2, round(1.0 / validation_fraction))
    validation = [
        sample
        for index, sample in enumerate(samples)
        if index % validation_stride == validation_stride - 1
    ]
    training = [
        sample
        for index, sample in enumerate(samples)
        if index % validation_stride != validation_stride - 1
    ]
    if not validation and len(training) > 2:
        validation = [training.pop()]
    return training, validation


def _build_segment(
    segment_index: int,
    samples: Sequence[dict[str, float]],
    config: ModelConfig,
    expected_windows: int,
) -> dict[str, Any]:
    training, validation = _split_train_validation(
        samples,
        config.validation_fraction,
    )
    fit = _fit_affine(training)
    training_residuals = _residuals_us(training, fit)
    validation_residuals = _residuals_us(validation, fit)
    coverage = min(1.0, len(samples) / max(1, expected_windows))
    validation_p95_us = _percentile(validation_residuals, 0.95)
    passed = (
        validation_p95_us <= config.max_validation_p95_us
        and coverage >= config.min_segment_coverage
    )
    median_rtt_us = statistics.median(
        sample["rtt_ns"] for sample in samples
    ) / 1_000.0

    return {
        "segment_index": segment_index,
        "status": "PASS" if passed else "FAIL",
        "valid_from_monotonic_ns": int(samples[0]["monotonic_ns"]),
        "valid_to_monotonic_ns": int(samples[-1]["monotonic_ns"]),
        **fit,
        "sample_count": len(samples),
        "training_sample_count": len(training),
        "validation_sample_count": len(validation),
        "expected_window_count": expected_windows,
        "coverage": coverage,
        "median_rtt_us": median_rtt_us,
        "training_p95_error_us": _percentile(training_residuals, 0.95),
        "training_max_error_us": max(training_residuals),
        "validation_p95_error_us": validation_p95_us,
        "validation_max_error_us": max(validation_residuals),
        # Includes fit error and a conservative path-asymmetry allowance.
        "uncertainty_us": validation_p95_us + median_rtt_us / 2.0,
    }


def build_piecewise_model(
    samples: Sequence[dict[str, Any]],
    *,
    source: dict[str, Any],
    reference: dict[str, Any],
    config: ModelConfig | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable worker-to-reference clock model."""
    # pylint: disable=too-many-locals
    selected_config = config or ModelConfig()
    selected_config.validate()
    representatives, window_health = _window_samples(samples, selected_config)
    if len(representatives) < selected_config.min_segment_samples:
        raise ValueError(
            "Too few healthy windows to build a model: "
            f"{len(representatives)} < {selected_config.min_segment_samples}"
        )

    origin_ns = int(window_health["origin_monotonic_ns"])
    window_ns = int(selected_config.window_seconds * 1_000_000_000)
    segment_ns = int(selected_config.segment_seconds * 1_000_000_000)
    grouped: dict[int, list[dict[str, float]]] = {}
    for sample in representatives:
        segment_index = (
            int(sample["monotonic_ns"]) - origin_ns
        ) // segment_ns
        grouped.setdefault(segment_index, []).append(sample)

    last_window_index = max(
        (int(sample["monotonic_ns"]) - origin_ns) // window_ns
        for sample in samples
    )
    expected_windows_by_segment: dict[int, int] = {}
    for window_index in range(last_window_index + 1):
        segment_index = (window_index * window_ns) // segment_ns
        expected_windows_by_segment[segment_index] = (
            expected_windows_by_segment.get(segment_index, 0) + 1
        )

    segments: list[dict[str, Any]] = []
    skipped_segments: list[dict[str, Any]] = []
    for segment_index, expected_windows in sorted(
        expected_windows_by_segment.items()
    ):
        segment_samples = grouped.get(segment_index, [])
        if len(segment_samples) < selected_config.min_segment_samples:
            skipped_segments.append(
                {
                    "segment_index": segment_index,
                    "healthy_window_count": len(segment_samples),
                    "expected_window_count": expected_windows,
                    "reason": "too_few_healthy_windows",
                }
            )
            continue
        segments.append(
            _build_segment(
                segment_index,
                segment_samples,
                selected_config,
                expected_windows,
            )
        )

    if not segments:
        raise ValueError("No segment contained enough healthy windows")

    failed_segment_count = sum(
        segment["status"] != "PASS" for segment in segments
    )
    return {
        "schema_version": 1,
        "model_type": "piecewise_affine",
        "offset_direction": "reference_minus_source",
        "timestamp_domain": "CLOCK_MONOTONIC",
        "source": source,
        "reference": reference,
        "status": "PASS" if failed_segment_count == 0 else "FAIL",
        "config": asdict(selected_config),
        "health": {
            **window_health,
            "segment_count": len(segments),
            "failed_segment_count": failed_segment_count,
            "skipped_segments": skipped_segments,
        },
        "segments": segments,
    }


def apply_clock_model(
    local_timestamp_ns: int,
    model: dict[str, Any],
    *,
    local_monotonic_ns: int | None = None,
    allow_failed_segment: bool = False,
) -> int:
    """Convert a source timestamp to the reference clock domain.

    ``local_monotonic_ns`` selects the segment and evaluates drift. It may be
    omitted only when ``local_timestamp_ns`` itself is CLOCK_MONOTONIC.
    """
    if model.get("model_type") == "identity":
        return local_timestamp_ns

    point_ns = (
        local_timestamp_ns
        if local_monotonic_ns is None
        else local_monotonic_ns
    )
    candidates = [
        segment
        for segment in model.get("segments", [])
        if int(segment["valid_from_monotonic_ns"])
        <= point_ns
        <= int(segment["valid_to_monotonic_ns"])
    ]
    if not candidates:
        raise ValueError(
            f"No clock-model segment covers monotonic timestamp {point_ns}"
        )

    segment = candidates[0]
    if segment.get("status") != "PASS" and not allow_failed_segment:
        raise ValueError(
            f"Clock-model segment {segment['segment_index']} failed validation"
        )
    offset_ns = _predict_offset(float(point_ns), segment)
    return round(local_timestamp_ns + offset_ns)
