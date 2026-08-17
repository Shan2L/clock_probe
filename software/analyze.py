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
    samples_per_window,
    rtt_slack_us,
):
    if window_seconds <= 0:
        raise ValueError("Window seconds must be positive")

    if samples_per_window <= 0:
        raise ValueError("Samples per windows must be positive")


    window_ns = int(window_seconds * 1_000_000_000)

    start_ns = min(
        sample["monotonic_ns"]
        for sample in samples
    )

    samples_by_window = {}

    for sample in samples:
        elapsed_ns = sample["monotonic_ns"] - start_ns
        window_index = elapsed_ns // window_ns

        samples_by_window.setdefault(
            window_index,
            [],
        ).append(sample)

    window_minimum_rtts_ns = [
        min(
            sample['rtt_ns']
            for sample in candidates
        )
        for candidates in samples_by_window.values()
    ]

    baseline_window_min_rtt_ns = statistics.median(
        window_minimum_rtts_ns
    )

    maximum_healthy_window_min_rtt_ns = (
        baseline_window_min_rtt_ns + 100_000
    )

    window_samples = []

    for window_index in sorted(samples_by_window):
        candidates = samples_by_window[window_index]

        ordered_candidates = sorted(
            candidates,
            key=lambda sample: sample["rtt_ns"],
        )

        minimum_rtt_ns = ordered_candidates[0]["rtt_ns"]
        if (
            minimum_rtt_ns > maximum_healthy_window_min_rtt_ns
        ):
            continue

        maximum_allowed_rtt_ns = (
            minimum_rtt_ns + int(rtt_slack_us * 1_000)
        )

        selected = [
            sample
            for sample in ordered_candidates
            if sample["rtt_ns"] <= maximum_allowed_rtt_ns
        ][:samples_per_window]

        representative = {
            "window_index": window_index,
            "selected_count": len(selected),
            "monotonic_ns": int(
                statistics.median(
                    sample["monotonic_ns"]
                    for sample in selected
                )
            ),
            "offset_ns": statistics.median(
                sample["offset_ns"]
                for sample in selected
            ),
            "rtt_ns": statistics.median(
                sample["rtt_ns"]
                for sample in selected
            ),
        }

        window_samples.append(representative)


    return window_samples


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
    parser.add_argument("--samples-per-window", type=int, default=3)
    parser.add_argument("--rtt-slack-us", type=float, default=20.0)
    parser.add_argument("--segment-seconds", type=float, default=60.0)
    
    args = parser.parse_args()

    samples = load_sample(args.input)

    if not samples:
        raise RuntimeError("No valid samples")

    window_samples = select_lowest_rtt_per_window(
        samples,
        args.window_seconds,
        args.samples_per_window,
        args.rtt_slack_us,
    )

    segment_seconds = args.segment_seconds
    segment_ns = int(segment_seconds * 1_000_000_000)
    segment_start_ns = window_samples[0]["monotonic_ns"]
    samples_by_segment = {}
    for sample in window_samples:
        elapsed_ns = (sample["monotonic_ns"] - segment_start_ns)
        segment_index = elapsed_ns // segment_ns
        samples_by_segment.setdefault(
            segment_index,
            [],
        ).append(sample)

    segment_reports = []

    for segment_index in sorted(samples_by_segment):
        segment_samples = samples_by_segment[segment_index]

        if len(segment_samples) < 10:
            continue
        
        segment_training_samples = []
        segment_validation_samples = []

        for index, sample in enumerate(segment_samples):
            if (index + 1) % 5 == 0:
                segment_validation_samples.append(sample)
            else:
                segment_training_samples.append(sample)

        if (
            len(segment_training_samples) < 2
            or len(segment_validation_samples) < 2
        ):
            continue

        (
            segment_offset_ns,
            segment_drift_ppm,
            segment_residuals_ns,
        ) = fit_drift(segment_training_samples)

        absolute_segment_residuals_ns = [
            abs(value)
            for value in segment_residuals_ns
        ]

        segment_base_monotonic_ns = segment_training_samples[0]["monotonic_ns"]
        segment_slope = segment_drift_ppm / 1_000_000
        segment_validation_residual_ns = []
        for sample in segment_validation_samples:
            elapsed_ns = sample["monotonic_ns"] - segment_base_monotonic_ns
            predicted_offset_ns = segment_offset_ns + segment_slope * elapsed_ns
            residual_ns = sample["offset_ns"] - predicted_offset_ns
            segment_validation_residual_ns.append(abs(residual_ns))

        segment_validation_p95_ns = statistics.quantiles(
            segment_validation_residual_ns,
            n=100,
            method='inclusive'
        )[94]

        segment_validation_max_ns = max(
            segment_validation_residual_ns)


        segment_p95_ns = statistics.quantiles(
            absolute_segment_residuals_ns,
            n=100,
            method="inclusive"
        )[94]
        segment_reports.append({
            "segment_index": segment_index,
            "sample_count": len(segment_samples),
            "training_count": len(segment_training_samples),
            "validation_count": len(segment_validation_samples),
            "start_offset_ns": segment_offset_ns,
            "drift_ppm": segment_drift_ppm,
            "residual_p95_ns": segment_p95_ns,
            "residual_max_ns": max(
                absolute_segment_residuals_ns
            ),
            "validation_p95_ns": segment_validation_p95_ns,
            "validation_max_ns": segment_validation_max_ns,

        })


    offset_jumps = []
    start_monotonic_ns = window_samples[0]["monotonic_ns"]

    for previous, current in zip(
        window_samples,
        window_samples[1:],
    ):
        elapsed_seconds = (current["monotonic_ns"] - start_monotonic_ns) / 1_000_000_000
        interval_seconds = (current["monotonic_ns"] - previous["monotonic_ns"]) / 1_000_000_000
        offset_jump_us = (current["offset_ns"] - previous["offset_ns"]) / 1_000
        offset_jumps.append({
            "elapsed_seconds": elapsed_seconds,
            "interval_seconds": interval_seconds,
            "offset_jump_us": offset_jump_us,
            "previous_rtt_us": previous["rtt_ns"] / 1_000,
            "current_rtt_us": current["rtt_ns"] / 1_000,
            "previous_selected_count": previous["selected_count"],
            "current_selected_count": current["selected_count"],
        })

    largest_jumps = sorted(
        offset_jumps,
        key=lambda item: abs(item["offset_jump_us"]),
        reverse=True,
    )[:5]


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

    print("Largest consecutive offset jumps: ")
    for jump in largest_jumps:
        print(
            f" at={jump['elapsed_seconds']:.1f}s "
            f"interval={jump['interval_seconds']:.1f}s "
            f"jump={jump['offset_jump_us']:+.3f}us "
            f"rtt={jump['previous_rtt_us']:.3f}"
            f"->{jump['current_rtt_us']:.3f} us"
            f"selected={jump['previous_selected_count']}"
            f"->{jump['current_selected_count']}"
        )

    print(f"Piecewise {segment_seconds}-second segment:")
    for report in segment_reports:
        print(
            f"segment={report['segment_index']: 02d} "
            f"count={report['sample_count']: 03d} "
            f"train={report['training_count']: 03d} "
            f"valid={report['validation_count']: 03d} "
            f"offset={ns_to_us(report['start_offset_ns']):+.3f}us "
            f"drift={report['drift_ppm']:.3f}ppm "
            f"train_p95={ns_to_us(report['residual_p95_ns']):+.3f}us "
            f"train_max={ns_to_us(report['residual_max_ns']):+.3f}us"
            f"valid_p95={ns_to_us(report['validation_p95_ns']):+.3f}us "
            f"valid_max={ns_to_us(report['validation_max_ns']):+.3f}us"
        )
