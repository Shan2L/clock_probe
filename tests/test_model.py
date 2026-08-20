import unittest

from clock_probe.calibration.software import (
    ModelConfig,
    apply_clock_model,
    build_clock_model,
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
        aligned_ns = apply_clock_model(
            point_ns,
            model,
            local_monotonic_ns=point_ns,
        )
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

    def test_partial_trailing_segment_uses_observed_duration(self) -> None:
        model = build_piecewise_model(
            synthetic_samples(duration_seconds=138),
            source={"hostname": "worker-a"},
            reference={"hostname": "head"},
        )

        trailing_segment = model["segments"][-1]
        self.assertEqual(model["status"], "PASS")
        self.assertEqual(len(model["segments"]), 5)
        self.assertEqual(trailing_segment["segment_index"], 4)
        self.assertEqual(trailing_segment["expected_window_count"], 18)
        self.assertEqual(trailing_segment["sample_count"], 18)
        self.assertEqual(trailing_segment["coverage"], 1.0)
        self.assertEqual(trailing_segment["status"], "PASS")

    def test_refuses_timestamp_outside_coverage(self) -> None:
        model = build_piecewise_model(
            synthetic_samples(),
            source={"hostname": "worker-a"},
            reference={"hostname": "head"},
        )
        with self.assertRaises(ValueError):
            apply_clock_model(999, model, local_monotonic_ns=999)

    def test_builds_held_out_interpolated_model(self) -> None:
        model = build_clock_model(
            synthetic_samples(duration_seconds=180),
            source={"hostname": "worker-a"},
            reference={"hostname": "head"},
            config=ModelConfig(
                window_seconds=20,
                samples_per_window=2,
                model_method="interpolation",
            ),
        )
        point_ns = 1_030_000_000_000
        aligned_ns = apply_clock_model(
            point_ns,
            model,
            local_monotonic_ns=point_ns,
        )
        self.assertEqual(model["model_type"], "interpolated_offset")
        self.assertEqual(model["status"], "PASS")
        self.assertGreater(model["health"]["validation_sample_count"], 0)
        self.assertAlmostEqual(
            aligned_ns - point_ns,
            185_000,
            delta=1_000,
        )

    def test_auto_selects_with_unseen_time_validation(self) -> None:
        model = build_clock_model(
            synthetic_samples(duration_seconds=300),
            source={"hostname": "worker-a"},
            reference={"hostname": "head"},
            config=ModelConfig(
                model_method="auto",
                candidate_window_seconds=(5.0, 10.0, 20.0),
                candidate_samples_per_window=(1, 2),
                candidate_rtt_slack_us=(10.0,),
                candidate_segment_seconds=(15.0, 30.0),
            ),
        )
        selection = model["model_selection"]
        self.assertEqual(model["status"], "PASS")
        self.assertEqual(selection["mode"], "auto")
        self.assertEqual(selection["validation_status"], "PASS")
        self.assertGreater(selection["validation_sample_count"], 0)
        self.assertTrue(
            any(candidate["status"] == "PASS" for candidate in selection["candidates"])
        )


if __name__ == "__main__":
    unittest.main()
