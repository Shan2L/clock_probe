import argparse
import json
import math
import statistics


def load_sample(path):
    samples = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            sample = json.loads(line)

            if sample["rtt_ns"] >= 0:
                samples.append(sample)
    return samples


def ns_to_us(value):
    return value / 1_000


def select_lowest_rtt_per_window(
    samples,
    window_seconds,
):
    if window_seconds <= 0:
        raise ValueError("Window seconds must be positive")

    window_ns = int(window_seconds * 1_000_000_000)

    start_ns = min(
        sample["monotonic_ns"]
        for sample in samples
    )

    best_by_window = {}

    for sample in samples:
        elapsed_ns = sample["monotonic_ns"] - start_ns
        window_index = elapsed_ns // window_ns

        current_best = best_by_window.get(window_index)
        if (
            current_best is None
            or sample["rtt_ns"] < current_best["rtt_ns"]
        ):
            best_by_window[window_index] = sample

    return [
        best_by_window[index]
        for index in sorted(best_by_window)
    ]


def fit_drift(samples):
    if len(samples) < 2:
        raise ValueError(
            "At least two samples are required to fit a drift"
        )

    base_monotonic_ns = samples[0]["monotonic_ns"]
    x_values = [
        sample["monotonic_ns"] - base_monotonic_ns
        for sample in samples
    ]

    y_values = [
        sample["offset_ns"]
        for sample in samples
    ]

    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)

    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(x_values, y_values)
    )

    denominator = sum(
        (x - x_mean) ** 2
        for x in x_values
    )

    if denominator == 0:
        raise ValueError("Samples have no time span")

    slope = numerator / denominator

    intercept_ns = y_mean - slope * x_mean
    drift_ppm = slope * 1_000_000

    residuals_ns = [
        y - (intercept_ns + slope * x)
        for x, y in zip(x_values, y_values)
    ]

    return intercept_ns, drift_ppm, residuals_ns


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--fraction", type=float, default=0.3)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    
    args = parser.parse_args()

    samples = load_sample(args.input)

    if not samples:
        raise RuntimeError("No valid samples")

    window_samples = select_lowest_rtt_per_window(
        samples,
        args.window_seconds,
    )

    validation_samples = []
    training_samples = []

    for index, sample in enumerate(window_samples):
        if (index + 1) % 5 == 0:
            validation_samples.append(sample)
        else:
            training_samples.append(sample)

    window_offset_median = statistics.median(
        sample["offset_ns"]
        for sample in window_samples
    )

    window_rtt_median = statistics.median(
        sample["rtt_ns"]
        for sample in window_samples
    )

    start_offset_ns, drift_ppm, residuals_ns = fit_drift(
        training_samples
    )

    base_monotonic_ns = training_samples[0]["monotonic_ns"]
    slope = drift_ppm / 1_000_000

    validation_residuals_ns = []

    for sample in validation_samples:
        elapsed_ns = (
            sample["monotonic_ns"] - base_monotonic_ns
        )

        predicted_offset_ns = start_offset_ns + slope * elapsed_ns
        residual_ns = sample["offset_ns"] - predicted_offset_ns
        validation_residuals_ns.append(residual_ns)


    absolute_validation_residuals_ns = [
        abs(residual)
        for residual in validation_residuals_ns
    ]

    validation_residual_p50_ns = statistics.median(
        absolute_validation_residuals_ns
    )

    validation_residual_p95_ns = statistics.quantiles(
        absolute_validation_residuals_ns,
        n=100,
        method="inclusive",
    )[94]

    validation_residual_max_ns = max(absolute_validation_residuals_ns)

    absolute_residuals_ns = [
        abs(residual)
        for residual in residuals_ns
    ]

    residual_p50_ns = statistics.median(
        absolute_residuals_ns
    )

    residual_p95_ns = statistics.quantiles(
        absolute_residuals_ns,
        n=100,
        method="inclusive",
    )[94]

    residual_max_ns = max(absolute_residuals_ns)

    low_count = max(
        1,
        math.ceil(len(samples) * args.fraction),
    )

    low_rtt_samples = sorted(
        samples,
        key=lambda sample: sample["rtt_ns"],
    )[:low_count]

    all_offset_median = statistics.median(
        sample["offset_ns"] for sample in samples
    )

    all_rtt_median = statistics.median(
        sample["rtt_ns"] for sample in samples
    )

    low_offset_median = statistics.median(
        sample["offset_ns"] for sample in low_rtt_samples
    )

    low_rtt_median = statistics.median(
        sample["rtt_ns"] for sample in low_rtt_samples
    )

    print(f"Sample count: {len(samples)}")
    print(
        f"All offset median: "
        f"{ns_to_us(all_offset_median):.3f} us"
    )
    print(
        f"All RTT median: "
        f"{ns_to_us(all_rtt_median)} us"
    )
    

    print(f"Lowest-RTT offset count: {low_count}")
    print(
        f"Lowest-Rtt offset median: "
        f"{ns_to_us(low_offset_median):.3f} us"
    )
    print(
        f"lowest-RTT RTT medina: "
        f"{ns_to_us(low_rtt_median):.3f} us"
    )

    print(f"Window count: {len(window_samples)}")
    print(
        f"Windowed offset median:"
        f"{ns_to_us(window_offset_median):.3f} us"
    )
    print(
        f"Windowed RTT median: "
        f"{ns_to_us(window_rtt_median)} us"
    )

    print(
        f"Fitted start offset: "
        f"{ns_to_us(start_offset_ns):.3f} us"
    )
    print(
        f"Fitted drift:"
        f"{drift_ppm:.3f} ppm"
    )

    print(
        f"Absolute residual p50: "
        f"{ns_to_us(residual_p50_ns):.3f} us"
    )
    print(
        f"Absolute residual p95: "
        f"{ns_to_us(residual_p95_ns):.3f} us"
    )
    print(
        f"Absolute residual max: "
        f"{ns_to_us(residual_max_ns):.3f} us"
    )

    print(f"Training count: {len(training_samples)}")
    print(f"Validation count: {len(validation_samples)}")

    print(
        f"Validation residual count: "
        f"{len(validation_residuals_ns)}"
    )

    print(
        f"Validation residual p50:  "
        f"{ns_to_us(validation_residual_p50_ns):.3f} us"
    )
    print(
        f"Validation residual p95: "
        f"{ns_to_us(validation_residual_p95_ns):.3f} us"
    )
    print(
        f"Validation residual max: "
        f"{ns_to_us(validation_residual_max_ns):.3f} us"
    )
