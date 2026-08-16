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


if __name__== "__main__":
    parser = argparse.ArgummentParser()
    parser.add_argument("input")
    parser.add_argument("--fraction", type=float, default=0.3)
    
    args = parser.parse.args()

    samples = load_sample(args.input)

    if not samples:
        raise RuntimeError("No valid samples")

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
    print(f"All offset median: {ns_to_us(all_offset_median)} us")
    print(f"All RTT median: {ns_to_us(all_rtt_median)} us")
    

    print(f"Lowest-RTT offset count: {low_count}")
    print(
        f"Lowest-Rtt offset median: "
        f"{ns_to_us(low_offset_median):.3f} us"
    )
    print(
        f"lowest-RTT RTT medina: "
        f"{ns_to_us(low_rtt_median):.3f} us"
    )
