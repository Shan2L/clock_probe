"""Tests for stoppable kernel-timestamp probe collection."""

import socket
import tempfile
import time
import unittest
from pathlib import Path

from clock_probe.sampling.probe import ContinuousProbeCollector, TimestampProbeServer


def unused_udp_port() -> int:
    """Reserve and release a local UDP port for a short test."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ContinuousProbeCollectorTest(unittest.TestCase):
    """Exercise collector lifecycle against a real timestamp server."""

    def test_collects_in_background_and_stops(self) -> None:
        port = unused_udp_port()
        server = TimestampProbeServer("127.0.0.1", port)
        server.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                output_path = Path(directory) / "samples.jsonl"
                collector = ContinuousProbeCollector(
                    reference_host="127.0.0.1",
                    reference_port=port,
                    source_host="127.0.0.1",
                    interval_ms=20,
                    output_path=output_path,
                )
                collector.start()
                time.sleep(0.15)

                running_status = collector.status()
                samples, final_status = collector.stop()

                self.assertTrue(running_status["running"])
                self.assertFalse(final_status["running"])
                self.assertGreaterEqual(len(samples), 3)
                self.assertEqual(
                    final_status["successful_sample_count"],
                    len(samples),
                )
                self.assertEqual(final_status["failed_sample_count"], 0)
                self.assertIsNone(final_status["fatal_error"])
                self.assertTrue(
                    all(
                        {
                            "bridge_monotonic_ns",
                            "bridge_realtime_ns",
                            "bridge_offset_ns",
                            "bridge_read_span_ns",
                        }.issubset(sample)
                        for sample in samples
                    )
                )
                self.assertEqual(
                    len(output_path.read_text(encoding="utf-8").splitlines()),
                    len(samples),
                )
        finally:
            server.stop()

    def test_stop_before_start_is_rejected(self) -> None:
        collector = ContinuousProbeCollector(
            reference_host="127.0.0.1",
            reference_port=1,
            source_host="127.0.0.1",
            interval_ms=100,
            output_path=Path("/tmp/not-created-clock-probe.jsonl"),
        )
        with self.assertRaises(RuntimeError):
            collector.stop()


if __name__ == "__main__":
    unittest.main()
