"""Tests for ptp4l log parsing and lock health gates."""

import unittest

from clock_probe.calibration.ptp_health import (
    evaluate_ptp_health,
    ptp_uncertainty_us,
)

MASTER_LOG = """\
ptp4l[596894.507]: selected /dev/ptp3 as PTP clock
ptp4l[596894.509]: port 1 (enp196s0f1np1): INITIALIZING to LISTENING on INIT_COMPLETE
ptp4l[596894.509]: port 0 (/var/run/ptp4l): INITIALIZING to LISTENING on INIT_COMPLETE
ptp4l[596898.287]: port 1 (enp196s0f1np1): LISTENING to MASTER on ANNOUNCE_RECEIPT_TIMEOUT_EXPIRES
ptp4l[596898.287]: selected local clock cc40f3.fffe.f58385 as best master
ptp4l[596898.287]: port 1 (enp196s0f1np1): assuming the grand master role
"""

SLAVE_LOG = """\
ptp4l[3610740.021]: selected /dev/ptp3 as PTP clock
ptp4l[3610740.022]: port 1 (enp196s0f1np1): INITIALIZING to LISTENING on INIT_COMPLETE
ptp4l[3610740.185]: port 1 (enp196s0f1np1): new foreign master cc40f3.fffe.f58385-1
ptp4l[3610742.185]: selected best master clock cc40f3.fffe.f58385
ptp4l[3610742.185]: port 1 (enp196s0f1np1): LISTENING to UNCALIBRATED on RS_SLAVE
ptp4l[3610743.937]: port 1 (enp196s0f1np1): UNCALIBRATED to SLAVE on MASTER_CLOCK_SELECTED
ptp4l[3610744.187]: clockcheck: clock frequency changed unexpectedly!
ptp4l[3610744.687]: rms  471 max  761 freq   -855 +/- 527 delay  2860 +/-  19
ptp4l[3610745.688]: rms  732 max  890 freq  -2016 +/- 215 delay  2841 +/-   0
ptp4l[3610842.743]: rms   31 max   44 freq  -3507 +/-  43 delay  2864 +/-   0
ptp4l[3610843.743]: rms   27 max   50 freq  -3507 +/-  37 delay  2864 +/-   3
ptp4l[3610844.744]: rms   37 max   58 freq  -3523 +/-  49 delay  2850 +/-   0
ptp4l[3610845.745]: rms   26 max   38 freq  -3519 +/-  36 delay  2862 +/-   0
ptp4l[3610846.745]: rms   38 max   72 freq  -3503 +/-  52 delay  2850 +/-   0
ptp4l[3610847.746]: rms   38 max   65 freq  -3523 +/-  51
ptp4l[3610848.746]: rms   25 max   59 freq  -3518 +/-  34 delay  2851 +/-   0
ptp4l[3610849.747]: rms   23 max   55 freq  -3497 +/-  29
ptp4l[3610850.748]: rms   40 max   63 freq  -3520 +/-  54 delay  2863 +/-   0
ptp4l[3610851.748]: rms   24 max   53 freq  -3522 +/-  32 delay  2863 +/-   0
ptp4l[3610852.749]: rms   26 max   42 freq  -3531 +/-  34 delay  2864 +/-   0
ptp4l[3610853.749]: rms   31 max   46 freq  -3501 +/-  39 delay  2871 +/-   0
ptp4l[3610854.750]: rms   23 max   37 freq  -3509 +/-  32 delay  2871 +/-   0
ptp4l[3610855.751]: rms   25 max   51 freq  -3541 +/-  26 delay  2871 +/-   0
"""


class PtpHealthTest(unittest.TestCase):
    """Fail closed unless ptp4l ends locked on the physical NIC."""

    def test_accepts_hardware_gm_without_rms(self) -> None:
        health = evaluate_ptp_health(MASTER_LOG, role="master")
        self.assertEqual(health.status, "PASS")
        self.assertEqual(health.port_state, "MASTER")
        self.assertEqual(health.grandmaster_clock_id, "cc40f3.fffe.f58385")
        self.assertTrue(health.assuming_grandmaster)

    def test_accepts_settled_slave_and_ignores_startup_clockcheck(self) -> None:
        health = evaluate_ptp_health(SLAVE_LOG, role="slave")
        self.assertEqual(health.status, "PASS")
        self.assertEqual(health.port_state, "SLAVE")
        self.assertEqual(health.grandmaster_clock_id, "cc40f3.fffe.f58385")
        self.assertGreater(health.clockcheck_count, 0)
        self.assertEqual(health.late_clockcheck_count, 0)
        self.assertLess(health.offset_rms_p95_ns, 50.0)
        uncertainty_us = ptp_uncertainty_us(health, path_delay_asymmetry=0.1)
        self.assertGreater(uncertainty_us, 0.2)
        self.assertLess(uncertainty_us, 0.5)

    def test_ignores_uds_port_state_transitions(self) -> None:
        health = evaluate_ptp_health(MASTER_LOG, role="master")
        self.assertNotEqual(health.port_state, "LISTENING")

    def test_rejects_unlocked_slave(self) -> None:
        health = evaluate_ptp_health(MASTER_LOG, role="slave")
        self.assertEqual(health.status, "FAIL")
        self.assertFalse(health.lock_ok)

    def test_rejects_late_clockcheck(self) -> None:
        text = SLAVE_LOG + (
            "ptp4l[3610855.900]: clockcheck: clock frequency changed unexpectedly!\n"
        )
        health = evaluate_ptp_health(text, role="slave")
        self.assertEqual(health.status, "FAIL")
        self.assertTrue(any("clockcheck" in reason for reason in health.reasons))


if __name__ == "__main__":
    unittest.main()
