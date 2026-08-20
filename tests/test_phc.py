"""Tests for PHC clock ids, ethtool parsing, and REALTIME-to-PHC mapping."""

import ctypes
import time
import unittest

from clock_probe.sampling.phc import (
    PTP_CLOCK_GETCAPS,
    PTP_SYS_OFFSET_EXTENDED,
    PTP_SYS_OFFSET_PRECISE,
    PtpClockCaps,
    PtpClockTime,
    PtpSysOffsetExtended,
    PtpSysOffsetPrecise,
    PtpSysOffsetTriplet,
    clock_gettime_ns,
    fd_to_clockid,
    parse_ethtool_hardware,
    ptp_time_to_ns,
    sample_from_extended_triplet,
    tightest_extended_sample,
)
from clock_probe.calibration.phc_bridge import CompiledPhcBridge, build_phc_bridge


ETHTOOL_CX6 = """\
Time stamping parameters for enp196s0f1np1:
Capabilities:
	hardware-transmit
	software-transmit
	hardware-receive
	software-receive
	hardware-raw-clock
	software-system-clock
PTP Hardware Clock: 3
Hardware Transmit Timestamp Modes:
	off
	on
Hardware Receive Filter Modes:
	none
	all
"""


def synthetic_phc_samples(
    *,
    duration_seconds: int = 65,
    interval_ms: int = 100,
    drift_ppm: float = 2.0,
    realtime_minus_phc_ns: int = 56_000_000_000,
) -> list[dict[str, int]]:
    """Create deterministic PHC/REALTIME pairs."""
    origin_phc_ns = 1_787_101_000_000_000_000
    samples = []
    for sequence in range(duration_seconds * 1_000 // interval_ms):
        elapsed_ns = sequence * interval_ms * 1_000_000
        phc_ns = origin_phc_ns + elapsed_ns
        drift_ns = round(drift_ppm * elapsed_ns / 1_000_000.0)
        realtime_ns = phc_ns + realtime_minus_phc_ns + drift_ns
        samples.append(
            {
                "bridge_phc_ns": phc_ns,
                "bridge_realtime_ns": realtime_ns,
                "bridge_monotonic_ns": 600_000_000_000 + elapsed_ns,
                "bridge_offset_ns": realtime_ns - phc_ns,
                "bridge_read_span_ns": 800 + sequence % 7,
            }
        )
    return samples


class PhcHelperTest(unittest.TestCase):
    """Validate clock-id conversion and NIC capability parsing."""

    def test_fd_to_clockid_matches_linux_macro(self) -> None:
        self.assertEqual(fd_to_clockid(3), -29)
        self.assertEqual(fd_to_clockid(0), -5)

    def test_clock_gettime_realtime_agrees_with_time_ns(self) -> None:
        posix_ns = clock_gettime_ns(0)
        python_ns = time.time_ns()
        self.assertLess(abs(python_ns - posix_ns), 50_000_000)

    def test_parses_connectx_hardware_timestamping(self) -> None:
        hardware = parse_ethtool_hardware(ETHTOOL_CX6, "enp196s0f1np1")
        self.assertTrue(hardware.usable)
        self.assertEqual(hardware.ptp_clock_index, 3)
        self.assertTrue(hardware.hardware_transmit)
        self.assertTrue(hardware.hardware_receive)

    def test_rejects_software_only_nic(self) -> None:
        hardware = parse_ethtool_hardware(
            "Capabilities:\n\tsoftware-transmit\n\tsoftware-receive\n",
            "eth0",
        )
        self.assertFalse(hardware.usable)

    def test_ptp_ioctl_numbers_match_kernel_headers(self) -> None:
        self.assertEqual(ctypes.sizeof(PtpClockTime), 16)
        self.assertEqual(ctypes.sizeof(PtpClockCaps), 80)
        self.assertEqual(ctypes.sizeof(PtpSysOffsetPrecise), 64)
        self.assertEqual(ctypes.sizeof(PtpSysOffsetExtended), 1216)
        self.assertEqual(PTP_CLOCK_GETCAPS, 2152742145)
        self.assertEqual(PTP_SYS_OFFSET_PRECISE, 3225435400)
        self.assertEqual(PTP_SYS_OFFSET_EXTENDED, 3300932873)

    def test_picks_tightest_kernel_extended_sandwich(self) -> None:
        request = PtpSysOffsetExtended()
        request.n_samples = 3

        def fill(triplet: PtpSysOffsetTriplet, before: int, phc: int, after: int) -> None:
            triplet.sys_before.sec = before // 1_000_000_000
            triplet.sys_before.nsec = before % 1_000_000_000
            triplet.phc.sec = phc // 1_000_000_000
            triplet.phc.nsec = phc % 1_000_000_000
            triplet.sys_after.sec = after // 1_000_000_000
            triplet.sys_after.nsec = after % 1_000_000_000

        fill(request.ts[0], 1_000, 500, 1_900)
        fill(request.ts[1], 2_000, 1_500, 2_080)
        fill(request.ts[2], 3_000, 2_500, 3_400)
        sample = tightest_extended_sample(request)
        self.assertEqual(sample["capture_method"], "extended")
        self.assertEqual(sample["bridge_read_span_ns"], 80)
        self.assertEqual(sample["bridge_phc_ns"], 1_500)
        self.assertEqual(sample["bridge_realtime_ns"], 2_040)

    def test_extended_triplet_rejects_inverted_window(self) -> None:
        triplet = PtpSysOffsetTriplet()
        triplet.sys_before.nsec = 200
        triplet.sys_after.nsec = 50
        with self.assertRaisesRegex(ValueError, "inverted"):
            sample_from_extended_triplet(triplet)

    def test_ptp_time_to_ns(self) -> None:
        stamp = PtpClockTime(sec=3, nsec=250, reserved=0)
        self.assertEqual(ptp_time_to_ns(stamp), 3_000_000_250)


class PhcBridgeTest(unittest.TestCase):
    """Validate REALTIME to PHC inversion."""

    def test_inverts_piecewise_phc_bridge(self) -> None:
        samples = synthetic_phc_samples()
        bridge = build_phc_bridge(samples, boot_id="boot-a")
        compiled = CompiledPhcBridge(bridge)
        event = samples[405]

        phc_ns, uncertainty_us = compiled.realtime_to_phc_ns(
            event["bridge_realtime_ns"],
            expected_boot_id="boot-a",
        )

        self.assertEqual(bridge["status"], "PASS")
        self.assertEqual(bridge["model_type"], "interpolated_realtime_phc")
        self.assertEqual(bridge["target_domain"], "PHC")
        self.assertEqual(bridge["model_selection"]["mode"], "auto")
        self.assertEqual(bridge["model_selection"]["validation_status"], "PASS")
        self.assertAlmostEqual(phc_ns, event["bridge_phc_ns"], delta=1_000)
        self.assertLess(uncertainty_us, 1.0)

    def test_rejects_failed_bridge(self) -> None:
        with self.assertRaisesRegex(ValueError, "has not passed"):
            CompiledPhcBridge({"status": "FAIL", "target_domain": "PHC", "segments": []})


if __name__ == "__main__":
    unittest.main()
