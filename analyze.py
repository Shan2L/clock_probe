import argparse
import json
import math
import statistics

from pydantic_core.core_schema import none_schema

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

    window_offset_median = statistics.median(
        sample["offset_ns"]
        for sample in window_samples
    )

    window_rtt_median = statistics.median(
        sample["rtt_ns"]
        for sample in window_samples
    )

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
