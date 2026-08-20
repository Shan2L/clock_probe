"""Tests for hardware node models and multi-node PHC sessions."""

import unittest
from unittest.mock import patch

from clock_probe.calibration.hardware import (
    HardwareModelConfig,
    build_hardware_model,
    build_hardware_session,
)
from clock_probe.session import session_uncertainty_us
from clock_probe.calibration.ptp_health import evaluate_ptp_health
from test_phc import synthetic_phc_samples
from test_ptp_health import MASTER_LOG, SLAVE_LOG


def _model(role: str, hostname: str, gm_id: str | None = None) -> dict:
    health = evaluate_ptp_health(
        MASTER_LOG if role == "master" else SLAVE_LOG,
        role=role,
    )
    if gm_id is not None:
        health.grandmaster_clock_id = gm_id
    with patch("clock_probe.calibration.hardware.read_boot_id", return_value="boot-a"):
        return build_hardware_model(
            synthetic_phc_samples(duration_seconds=30),
            role=role,
            ptp_health=health,
            source={"hostname": hostname, "boot_id": "boot-a"},
        )


class HardwareSessionTest(unittest.TestCase):
    """Merge PHC models only when they share a locked grandmaster."""

    def test_builds_session_from_gm_and_slave(self) -> None:
        master = _model("master", "cse-ai-6")
        slave = _model("slave", "cse-ai-9")
        session = build_hardware_session([master, slave], session_id="ptp-test")

        self.assertEqual(master["status"], "PASS")
        self.assertEqual(slave["status"], "PASS")
        self.assertEqual(session["status"], "PASS")
        self.assertEqual(session["clock_source"], "ptp_hardware")
        self.assertEqual(session["timestamp_domain"], "PHC")
        self.assertEqual(
            session["ptp"]["grandmaster_clock_id"],
            "cc40f3.fffe.f58385",
        )
        self.assertEqual(session["system_clock_policy"], "phc_only_never_phc2sys")
        self.assertGreater(session["max_uncertainty_us"], 0.2)
        self.assertLess(session["max_uncertainty_us"], 1.0)
        self.assertAlmostEqual(
            session_uncertainty_us(session),
            session["max_uncertainty_us"],
            delta=0.05,
        )

    def test_rejects_mismatched_grandmaster(self) -> None:
        master = _model("master", "cse-ai-6")
        slave = _model("slave", "cse-ai-9", gm_id="deadbeef.fffe.000000")
        with self.assertRaisesRegex(ValueError, "do not share one grandmaster"):
            build_hardware_session([master, slave])

    def test_rejects_missing_slave(self) -> None:
        master_a = _model("master", "cse-ai-6")
        master_b = _model("master", "cse-ai-9")
        with self.assertRaisesRegex(ValueError, "one master model"):
            build_hardware_session([master_a, master_b])

    def test_fails_hard_total_uncertainty_gate(self) -> None:
        health = evaluate_ptp_health(SLAVE_LOG, role="slave")
        with patch(
            "clock_probe.calibration.hardware.read_boot_id",
            return_value="boot-a",
        ):
            model = build_hardware_model(
                synthetic_phc_samples(duration_seconds=30),
                role="slave",
                ptp_health=health,
                source={"hostname": "cse-ai-9", "boot_id": "boot-a"},
                config=HardwareModelConfig(max_total_uncertainty_us=0.1),
            )
        self.assertEqual(model["status"], "FAIL")
        self.assertTrue(
            any("total uncertainty" in reason for reason in model["fail_reasons"])
        )

    def test_rejects_relaxing_hardware_limit_above_two_microseconds(self) -> None:
        health = evaluate_ptp_health(SLAVE_LOG, role="slave")
        with self.assertRaisesRegex(ValueError, "no greater than 2.0 us"):
            build_hardware_model(
                synthetic_phc_samples(duration_seconds=30),
                role="slave",
                ptp_health=health,
                config=HardwareModelConfig(max_total_uncertainty_us=2.1),
            )


if __name__ == "__main__":
    unittest.main()
