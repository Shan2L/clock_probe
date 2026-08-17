import unittest

from clock_probe.model import (
    ModelConfig,
    apply_clock_model,
    build_piecewise_model,
)


def synthetic_samples(
    *,
    duration_seconds: int = 90,
    interval_ms: int = 100,
    drift_ppm: float = 2.0,
) -> list[dict[str, float | int]]:
    origin_ns = 1_000_000_000_000
    samples = []
    for sequence in range(duration_seconds * 1_000 // interval_ms):
        elapsed_ns = sequence * interval_ms * 1_000_000
        offset_ns = 125_000.0 + drift_ppm * elapsed_ns / 1_000_000.0
        # Deterministic sub-microsecond measurement noise.
        offset_ns += ((sequence % 7) - 3) * 100.0
        rtt_ns = 100_000.0 + (sequence % 5) * 1_000.0
        samples.append(
            {
                "sequence": sequence,
                "monotonic_ns": origin_ns + elapsed_ns,
                "offset_ns": offset_ns,
                "rtt_ns": rtt_ns,
            }
        )
    return samples


class PiecewiseModelTest(unittest.TestCase):
    def test_builds_and_applies_model(self) -> None:
        model = build_piecewise_model(
            synthetic_samples(),
            source={"hostname": "worker-a"},
            reference={"hostname": "head"},
            config=ModelConfig(),
        )

        self.assertEqual(model["status"], "PASS")
        self.assertEqual(len(model["segments"]), 3)
        self.assertAlmostEqual(
            model["segments"][0]["drift_ppm"],
            2.0,
            delta=0.05,
        )

        point_ns = 1_010_000_000_000
        aligned_ns = apply_clock_model(point_ns, model)
        self.assertAlmostEqual(
            aligned_ns - point_ns,
            145_000,
            delta=1_000,
        )

    def test_rejects_congested_windows(self) -> None:
        samples = synthetic_samples()
        origin_ns = int(samples[0]["monotonic_ns"])
        for sample in samples:
            elapsed_seconds = (
                int(sample["monotonic_ns"]) - origin_ns
            ) / 1_000_000_000
            if 10 <= elapsed_seconds < 11:
                sample["rtt_ns"] = 300_000.0
                sample["offset_ns"] = 225_000.0

        model = build_piecewise_model(
            samples,
            source={"hostname": "worker-a"},
            reference={"hostname": "head"},
        )
        self.assertGreaterEqual(model["health"]["rejected_window_count"], 1)
        self.assertEqual(model["status"], "PASS")

    def test_refuses_timestamp_outside_coverage(self) -> None:
        model = build_piecewise_model(
            synthetic_samples(),
            source={"hostname": "worker-a"},
            reference={"hostname": "head"},
        )
        with self.assertRaises(ValueError):
            apply_clock_model(999, model)


if __name__ == "__main__":
    unittest.main()
