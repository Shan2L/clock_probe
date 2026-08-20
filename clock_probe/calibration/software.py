"""Piecewise affine clock-model construction and application."""

from __future__ import annotations

import bisect
import statistics
from dataclasses import asdict, dataclass, replace
from itertools import product
from typing import Any, Iterable, Sequence

from .core import (
    Candidate,
    fit_affine as _fit_affine_values,
    percentile as _percentile,
    select_candidate,
    split_train_validation as _split_train_validation,
)


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
    model_method: str = "auto"
    candidate_window_seconds: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 30.0)
    candidate_samples_per_window: tuple[int, ...] = (1, 2, 3)
    candidate_rtt_slack_us: tuple[float, ...] = (10.0, 20.0)
    candidate_segment_seconds: tuple[float, ...] = (15.0, 30.0, 60.0)
    tuning_fraction: float = 0.6

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
        if self.model_method not in {"auto", "interpolation", "piecewise_affine"}:
            raise ValueError(
                "model_method must be 'auto', 'interpolation', or 'piecewise_affine'"
            )
        if not 0.5 <= self.tuning_fraction < 0.9:
            raise ValueError("tuning_fraction must be in [0.5, 0.9)")
        if (
            not self.candidate_window_seconds
            or not self.candidate_samples_per_window
            or not self.candidate_rtt_slack_us
            or not self.candidate_segment_seconds
        ):
            raise ValueError("Auto-selection candidate lists must not be empty")


def _fit_affine(samples: Sequence[dict[str, float]]) -> dict[str, float]:
    """Fit offset = offset_at_base + slope * (monotonic - base)."""
    base_ns, offset_at_base_ns, slope = _fit_affine_values(
        [sample["monotonic_ns"] for sample in samples],
        [sample["offset_ns"] for sample in samples],
    )
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


def _interpolate_offset(
    left: dict[str, float],
    right: dict[str, float],
    monotonic_ns: float,
) -> float:
    span_ns = right["monotonic_ns"] - left["monotonic_ns"]
    if span_ns <= 0:
        raise ValueError("Software clock anchors must be strictly increasing")
    fraction = (monotonic_ns - left["monotonic_ns"]) / span_ns
    return left["offset_ns"] + fraction * (
        right["offset_ns"] - left["offset_ns"]
    )


def build_interpolated_model(
    samples: Sequence[dict[str, Any]],
    *,
    source: dict[str, Any],
    reference: dict[str, Any],
    config: ModelConfig | None = None,
) -> dict[str, Any]:
    """Build a held-out-validated interpolation between low-RTT anchors."""
    selected = config or ModelConfig()
    selected.validate()
    anchors, window_health = _window_samples(samples, selected)
    if len(anchors) < 3:
        raise ValueError("Too few healthy windows to build an interpolated model")

    validation_errors_us: list[float] = []
    for left, anchor, right in zip(anchors, anchors[1:], anchors[2:]):
        if int(right["window_index"] - left["window_index"]) != 2:
            continue
        predicted_offset = _interpolate_offset(
            left,
            right,
            anchor["monotonic_ns"],
        )
        validation_errors_us.append(
            abs(anchor["offset_ns"] - predicted_offset) / 1_000.0
        )
    if not validation_errors_us:
        raise ValueError("Software interpolation has no held-out validation samples")

    validation_p95_us = _percentile(validation_errors_us, 0.95)
    validation_max_us = max(validation_errors_us)
    passed = validation_p95_us <= selected.max_validation_p95_us
    segments: list[dict[str, Any]] = []
    for left, right in zip(anchors, anchors[1:]):
        if int(right["window_index"] - left["window_index"]) != 1:
            continue
        span_ns = right["monotonic_ns"] - left["monotonic_ns"]
        drift = (right["offset_ns"] - left["offset_ns"]) / span_ns
        path_uncertainty_us = max(left["rtt_ns"], right["rtt_ns"]) / 2_000.0
        segments.append(
            {
                "segment_index": len(segments),
                "status": "PASS" if passed else "FAIL",
                "valid_from_monotonic_ns": round(left["monotonic_ns"]),
                "valid_to_monotonic_ns": round(right["monotonic_ns"]),
                "base_monotonic_ns": left["monotonic_ns"],
                "offset_at_base_ns": left["offset_ns"],
                "drift_ns_per_ns": drift,
                "drift_ppm": drift * 1_000_000.0,
                "left_window_index": int(left["window_index"]),
                "right_window_index": int(right["window_index"]),
                "left_rtt_us": left["rtt_ns"] / 1_000.0,
                "right_rtt_us": right["rtt_ns"] / 1_000.0,
                "path_uncertainty_us": path_uncertainty_us,
                "validation_p95_error_us": validation_p95_us,
                "validation_max_error_us": validation_max_us,
                "uncertainty_us": path_uncertainty_us + validation_max_us,
            }
        )
    if not segments:
        raise ValueError("No consecutive software clock anchors were usable")
    return {
        "schema_version": 1,
        "model_type": "interpolated_offset",
        "offset_direction": "reference_minus_source",
        "timestamp_domain": "CLOCK_MONOTONIC",
        "source": source,
        "reference": reference,
        "status": "PASS" if passed else "FAIL",
        "config": asdict(selected),
        "health": {
            **window_health,
            "anchor_count": len(anchors),
            "segment_count": len(segments),
            "failed_segment_count": 0 if passed else len(segments),
            "validation_sample_count": len(validation_errors_us),
            "validation_p95_error_us": validation_p95_us,
            "validation_max_error_us": validation_max_us,
        },
        "segments": segments,
    }


def _build_clock_model_method(
    samples: Sequence[dict[str, Any]],
    *,
    source: dict[str, Any],
    reference: dict[str, Any],
    config: ModelConfig,
) -> dict[str, Any]:
    if config.model_method == "piecewise_affine":
        return build_piecewise_model(
            samples,
            source=source,
            reference=reference,
            config=config,
        )
    return build_interpolated_model(
        samples,
        source=source,
        reference=reference,
        config=config,
    )


def _model_score(
    model: dict[str, Any],
    *,
    local_bridge_uncertainty_us: float,
) -> dict[str, float]:
    segments = [
        segment
        for segment in model.get("segments", [])
        if segment.get("status") == "PASS"
    ]
    if not segments:
        raise ValueError("Candidate has no PASS segments")
    health = model.get("health", {})
    if int(health.get("rejected_window_count", 0)):
        raise ValueError("Candidate has rejected timing windows and coverage gaps")
    if health.get("skipped_segments"):
        raise ValueError("Candidate has skipped affine segments")
    if (
        model.get("model_type") == "interpolated_offset"
        and int(health.get("segment_count", 0))
        != int(health.get("anchor_count", 0)) - 1
    ):
        raise ValueError("Candidate interpolation coverage is not contiguous")
    validation_p95 = health.get("validation_p95_error_us")
    validation_max = health.get("validation_max_error_us")
    if validation_p95 is None:
        validation_p95 = max(
            float(segment["validation_p95_error_us"]) for segment in segments
        )
    if validation_max is None:
        validation_max = max(
            float(segment["validation_max_error_us"]) for segment in segments
        )
    return {
        "max_total_uncertainty_us": max(
            float(segment["uncertainty_us"]) for segment in segments
        )
        + local_bridge_uncertainty_us,
        "validation_p95_error_us": float(validation_p95),
        "validation_max_error_us": float(validation_max),
        "segment_count": float(len(segments)),
    }


def _software_candidate_configs(selected: ModelConfig) -> list[ModelConfig]:
    candidates: list[ModelConfig] = []
    for window, sample_count, slack in product(
        selected.candidate_window_seconds,
        selected.candidate_samples_per_window,
        selected.candidate_rtt_slack_us,
    ):
        candidates.append(
            replace(
                selected,
                model_method="interpolation",
                window_seconds=float(window),
                samples_per_window=int(sample_count),
                rtt_slack_us=float(slack),
                min_segment_samples=3,
            )
        )
    affine_windows = [
        window
        for window in selected.candidate_window_seconds
        if window <= 5.0
    ]
    for window, sample_count, slack, segment_seconds in product(
        affine_windows,
        selected.candidate_samples_per_window,
        selected.candidate_rtt_slack_us,
        selected.candidate_segment_seconds,
    ):
        candidates.append(
            replace(
                selected,
                model_method="piecewise_affine",
                window_seconds=float(window),
                samples_per_window=int(sample_count),
                rtt_slack_us=float(slack),
                segment_seconds=float(segment_seconds),
                min_segment_samples=3,
            )
        )
    return candidates


def build_auto_clock_model(
    samples: Sequence[dict[str, Any]],
    *,
    source: dict[str, Any],
    reference: dict[str, Any],
    config: ModelConfig,
    local_bridge_uncertainty_us: float = 0.0,
) -> dict[str, Any]:
    """Select on the tuning period and verify the winner on unseen time."""
    candidates = [
        Candidate(
            value=candidate,
            description={
                "method": candidate.model_method,
                "window_seconds": candidate.window_seconds,
                "samples_per_window": candidate.samples_per_window,
                "rtt_slack_us": candidate.rtt_slack_us,
                "segment_seconds": candidate.segment_seconds,
            },
            complexity=0 if candidate.model_method == "piecewise_affine" else 1,
        )
        for candidate in _software_candidate_configs(config)
    ]

    def build(
        candidate: ModelConfig,
        selected_samples: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        return _build_clock_model_method(
            selected_samples,
            source=source,
            reference=reference,
            config=candidate,
        )

    def score(model: dict[str, Any]) -> dict[str, float]:
        return _model_score(
            model,
            local_bridge_uncertainty_us=local_bridge_uncertainty_us,
        )

    def mark_failed(model: dict[str, Any]) -> None:
        model["status"] = "FAIL"
        for segment in model.get("segments", []):
            segment["status"] = "FAIL"

    final_model, selection = select_candidate(
        samples,
        candidates,
        tuning_fraction=config.tuning_fraction,
        time_key=lambda sample: int(sample["monotonic_ns"]),
        build=build,
        score=score,
        objective_key="max_total_uncertainty_us",
        mark_failed=mark_failed,
    )
    selection["selected_config"] = final_model["config"]
    selection.pop("selected", None)
    final_model["model_selection"] = selection
    return final_model


def build_clock_model(
    samples: Sequence[dict[str, Any]],
    *,
    source: dict[str, Any],
    reference: dict[str, Any],
    config: ModelConfig | None = None,
    local_bridge_uncertainty_us: float = 0.0,
) -> dict[str, Any]:
    """Build or automatically select a software clock model."""
    selected = config or ModelConfig()
    selected.validate()
    if selected.model_method == "auto":
        return build_auto_clock_model(
            samples,
            source=source,
            reference=reference,
            config=selected,
            local_bridge_uncertainty_us=local_bridge_uncertainty_us,
        )
    return _build_clock_model_method(
        samples,
        source=source,
        reference=reference,
        config=selected,
    )


class CompiledClockModel:  # pylint: disable=too-few-public-methods
    """Compiled segment lookup shared by direct application and Trace alignment."""

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
        if (
            model.get("model_type") == "piecewise_affine"
            and self.segments
            and config
            and health
        ):
            origin_ns = int(health["origin_monotonic_ns"])
            window_ns = round(float(config["window_seconds"]) * 1_000_000_000)
            segment_ns = round(float(config["segment_seconds"]) * 1_000_000_000)
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
            raise ValueError("Clock model has no usable segments")

    def align_realtime_ns(
        self,
        local_realtime_ns: int,
        local_monotonic_ns: int,
        *,
        allow_failed_segment: bool = False,
    ) -> tuple[int, float]:
        if self.identity:
            return local_realtime_ns, 0.0
        index = bisect.bisect_right(self.starts, local_monotonic_ns) - 1
        if index < 0 or local_monotonic_ns > self.ends[index]:
            raise ValueError(
                f"No model segment covers Trace timestamp "
                f"{local_monotonic_ns}"
            )
        segment = self.segments[index]
        if segment.get("status") != "PASS" and not allow_failed_segment:
            raise ValueError(
                f"Clock-model segment {segment['segment_index']} failed validation"
            )
        offset_ns = _predict_offset(float(local_monotonic_ns), segment)
        return (
            round(local_realtime_ns + offset_ns),
            float(segment.get("uncertainty_us", 0.0)),
        )


def apply_clock_model(
    local_timestamp_ns: int,
    model: dict[str, Any],
    *,
    local_monotonic_ns: int,
    allow_failed_segment: bool = False,
) -> int:
    """Convert a source timestamp to the reference clock domain.

    ``local_monotonic_ns`` explicitly selects the segment and evaluates drift;
    callers must never substitute a timestamp from another clock domain.
    """
    aligned_ns, _uncertainty_us = CompiledClockModel(model).align_realtime_ns(
        local_timestamp_ns,
        local_monotonic_ns,
        allow_failed_segment=allow_failed_segment,
    )
    return aligned_ns
