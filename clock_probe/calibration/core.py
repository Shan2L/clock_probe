"""Shared calibration math and time-isolated candidate selection."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Generic, Sequence, TypeVar

Sample = dict[str, Any]
CandidateT = TypeVar("CandidateT")


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def split_train_validation(
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


def fit_affine(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> tuple[float, float, float]:
    """Return ``base_x, intercept_at_base, slope``."""
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("Affine fit requires matching x/y samples")
    base_x = float(x_values[0])
    centered = [float(value) - base_x for value in x_values]
    x_mean = statistics.fmean(centered)
    y_mean = statistics.fmean(float(value) for value in y_values)
    denominator = sum((value - x_mean) ** 2 for value in centered)
    if denominator == 0:
        raise ValueError("Affine samples do not span time")
    slope = sum(
        (x_value - x_mean) * (float(y_value) - y_mean)
        for x_value, y_value in zip(centered, y_values)
    ) / denominator
    return base_x, y_mean - slope * x_mean, slope


def split_time(
    samples: Sequence[Sample],
    *,
    tuning_fraction: float,
    time_key: Callable[[Sample], int],
) -> tuple[list[Sample], list[Sample]]:
    if not 0.5 <= tuning_fraction < 0.9:
        raise ValueError("tuning_fraction must be in [0.5, 0.9)")
    ordered = sorted(samples, key=time_key)
    if len(ordered) < 2:
        raise ValueError("Candidate selection needs at least two samples")
    split = round(len(ordered) * tuning_fraction)
    split = max(1, min(len(ordered) - 1, split))
    return ordered[:split], ordered[split:]


@dataclass(frozen=True)
class Candidate(Generic[CandidateT]):
    value: CandidateT
    description: dict[str, Any]
    complexity: int = 0


def select_candidate(  # pylint: disable=too-many-arguments,too-many-locals
    samples: Sequence[Sample],
    candidates: Sequence[Candidate[CandidateT]],
    *,
    tuning_fraction: float,
    time_key: Callable[[Sample], int],
    build: Callable[[CandidateT, Sequence[Sample]], dict[str, Any]],
    score: Callable[[dict[str, Any]], dict[str, float]],
    objective_key: str,
    budget: float | None = None,
    mark_failed: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select on tuning time, verify on unseen time, then rebuild on all data."""
    tuning, validation = split_time(
        samples,
        tuning_fraction=tuning_fraction,
        time_key=time_key,
    )
    leaderboard: list[dict[str, Any]] = []
    viable: list[tuple[tuple[Any, ...], Candidate[CandidateT]]] = []
    for candidate in candidates:
        entry = dict(candidate.description)
        try:
            payload = build(candidate.value, tuning)
            metrics = score(payload)
            payload_passed = payload.get("status") == "PASS"
            over_budget = (
                budget is not None and metrics[objective_key] > budget
            )
            status = (
                "PASS"
                if payload_passed and not over_budget
                else ("OVER_BUDGET" if payload_passed else "FAIL")
            )
            entry.update(status=status, score=metrics, error=None)
            if payload_passed:
                viable.append(
                    (
                        (
                            int(over_budget),
                            metrics[objective_key],
                            metrics.get("validation_max_error_us", float("inf")),
                            metrics.get("validation_p95_error_us", float("inf")),
                            candidate.complexity,
                        ),
                        candidate,
                    )
                )
        except (ValueError, ZeroDivisionError) as error:
            entry.update(status="REJECTED", score=None, error=str(error))
        leaderboard.append(entry)
    if not viable:
        raise ValueError("No calibration candidate passed validation")
    viable.sort(key=lambda item: item[0])
    winner = viable[0][1]

    validation_payload = build(winner.value, validation)
    validation_score = score(validation_payload)
    validation_passed = validation_payload.get("status") == "PASS"
    if budget is not None and validation_score[objective_key] > budget:
        validation_passed = False

    final_payload = build(winner.value, samples)
    final_score = score(final_payload)
    if not validation_passed:
        if mark_failed is not None:
            mark_failed(final_payload)
        else:
            final_payload["status"] = "FAIL"
    leaderboard.sort(
        key=lambda entry: (
            entry["status"] != "PASS",
            float((entry.get("score") or {}).get(objective_key, float("inf"))),
        )
    )
    metadata = {
        "mode": "auto",
        "objective": objective_key,
        "tuning_fraction": tuning_fraction,
        "tuning_sample_count": len(tuning),
        "validation_sample_count": len(validation),
        "remaining_budget_us": budget,
        "selected": winner.description,
        "validation_status": "PASS" if validation_passed else "FAIL",
        "validation_score": validation_score,
        "final_score": final_score,
        "candidates": leaderboard,
    }
    return final_payload, metadata
