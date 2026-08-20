"""Tests for the local REALTIME-to-MONOTONIC bridge."""

import unittest

from clock_probe.calibration.clock_bridge import (
    ClockBridgeConfig,
    CompiledClockBridge,
    build_clock_bridge,
)


def synthetic_bridge_samples(
    *,
    duration_seconds: int = 65,
    interval_ms: int = 100,
    drift_ppm: float = 3.0,
    step_at_seconds: int | None = None,
    step_ns: int = 0,
) -> list[dict[str, int]]:
    """Create deterministic paired-clock samples."""
    origin_monotonic_ns = 1_000_000_000_000
    realtime_offset_ns = 1_700_000_000_000_000_000
    samples = []
    for sequence in range(duration_seconds * 1_000 // interval_ms):
        elapsed_ns = sequence * interval_ms * 1_000_000
        monotonic_ns = origin_monotonic_ns + elapsed_ns
        drift_ns = round(drift_ppm * elapsed_ns / 1_000_000.0)
        applied_step_ns = (
            step_ns
            if step_at_seconds is not None
            and elapsed_ns >= step_at_seconds * 1_000_000_000
            else 0
        )
        realtime_ns = (
            monotonic_ns + realtime_offset_ns + drift_ns + applied_step_ns
        )
        samples.append(
            {
                "bridge_monotonic_ns": monotonic_ns,
                "bridge_realtime_ns": realtime_ns,
                "bridge_offset_ns": realtime_ns - monotonic_ns,
                "bridge_read_span_ns": 400 + sequence % 5,
            }
        )
    return samples


class ClockBridgeTest(unittest.TestCase):
    """Validate bridge fitting, inversion, steps, and safety checks."""

    def test_builds_and_inverts_piecewise_bridge(self) -> None:
        samples = synthetic_bridge_samples()
        bridge = build_clock_bridge(samples, boot_id="boot-a")
        compiled = CompiledClockBridge(bridge)
        event = samples[405]

        monotonic_ns, uncertainty_us = compiled.realtime_to_monotonic_ns(
            event["bridge_realtime_ns"],
            expected_boot_id="boot-a",
        )

        self.assertEqual(bridge["status"], "PASS")
        self.assertEqual(len(bridge["segments"]), 3)
        self.assertAlmostEqual(
            bridge["segments"][0]["drift_ppm"],
            3.0,
            delta=0.01,
        )
        self.assertAlmostEqual(
            monotonic_ns,
            event["bridge_monotonic_ns"],
            delta=1_000,
        )
        self.assertLess(uncertainty_us, 1.0)

        boundary_monotonic_ns = 1_000_000_000_000 + 29_950_000_000
        boundary_realtime_ns = (
            boundary_monotonic_ns
            + 1_700_000_000_000_000_000
            + round(3.0 * 29_950_000_000 / 1_000_000.0)
        )
        mapped_boundary_ns, _ = compiled.realtime_to_monotonic_ns(
            boundary_realtime_ns
        )
        self.assertAlmostEqual(
            mapped_boundary_ns,
            boundary_monotonic_ns,
            delta=1_000,
        )

    def test_starts_new_segment_after_realtime_step(self) -> None:
        samples = synthetic_bridge_samples(
            duration_seconds=60,
            step_at_seconds=25,
            step_ns=5_000_000,
        )
        bridge = build_clock_bridge(samples, boot_id="boot-a")
        compiled = CompiledClockBridge(bridge)
        before = samples[200]
        after = samples[400]

        before_monotonic, _ = compiled.realtime_to_monotonic_ns(
            before["bridge_realtime_ns"]
        )
        after_monotonic, _ = compiled.realtime_to_monotonic_ns(
            after["bridge_realtime_ns"]
        )

        self.assertGreaterEqual(len(bridge["segments"]), 2)
        self.assertAlmostEqual(
            before_monotonic,
            before["bridge_monotonic_ns"],
            delta=1_000,
        )
        self.assertAlmostEqual(
            after_monotonic,
            after["bridge_monotonic_ns"],
            delta=1_000,
        )
        with self.assertRaisesRegex(ValueError, "No clock-bridge segment"):
            compiled.realtime_to_monotonic_ns(
                samples[249]["bridge_realtime_ns"] + 2_000_000
            )

    def test_fills_interior_realtime_stitch_gap(self) -> None:
        def segment(
            index: int,
            realtime_from: int,
            realtime_to: int,
            monotonic_from: int,
            monotonic_to: int,
        ) -> dict:
            return {
                "segment_index": index,
                "status": "PASS",
                "valid_from_realtime_ns": realtime_from,
                "valid_to_realtime_ns": realtime_to,
                "valid_from_monotonic_ns": monotonic_from,
                "valid_to_monotonic_ns": monotonic_to,
                "base_monotonic_ns": monotonic_from,
                "offset_at_base_ns": realtime_from - monotonic_from,
                "drift_ns_per_ns": 0.0,
                "uncertainty_us": 1.0,
            }

        compiled = CompiledClockBridge(
            {
                "status": "PASS",
                "boot_id": "boot-a",
                "segments": [
                    segment(0, 1_000, 2_000, 0, 1_000),
                    segment(1, 5_000, 6_000, 1_001, 2_000),
                ],
            }
        )
        mapped, uncertainty_us = compiled.realtime_to_monotonic_ns(2_500)
        self.assertEqual(mapped, 1_000)
        self.assertEqual(uncertainty_us, 1.0)
        with self.assertRaisesRegex(ValueError, "No clock-bridge segment"):
            compiled.realtime_to_monotonic_ns(1_000_000)

    def test_rejects_boot_id_mismatch(self) -> None:
        bridge = build_clock_bridge(
            synthetic_bridge_samples(),
            boot_id="boot-a",
        )
        with self.assertRaisesRegex(ValueError, "boot ID"):
            CompiledClockBridge(bridge).realtime_to_monotonic_ns(
                synthetic_bridge_samples()[100]["bridge_realtime_ns"],
                expected_boot_id="boot-b",
            )

    def test_rejects_unhealthy_read_spans(self) -> None:
        samples = synthetic_bridge_samples(duration_seconds=2)
        for sample in samples:
            sample["bridge_read_span_ns"] = 1_000_000
        with self.assertRaisesRegex(ValueError, "Too few healthy"):
            build_clock_bridge(
                samples,
                boot_id="boot-a",
                config=ClockBridgeConfig(max_read_span_us=100.0),
            )


if __name__ == "__main__":
    unittest.main()
