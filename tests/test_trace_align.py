"""Tests for streaming PyTorch Trace alignment."""

import json
import tempfile
import unittest
from pathlib import Path

from clock_probe.calibration.clock_bridge import build_clock_bridge
from clock_probe.calibration.software import build_piecewise_model
from clock_probe.calibration.phc_bridge import build_phc_bridge
from clock_probe.postprocess.align import align_trace_file
from test_clock_bridge import synthetic_bridge_samples
from test_model import synthetic_samples
from test_phc import synthetic_phc_samples


def write_trace(path: Path, base_time_ns: int, timestamps_us: list[str]) -> None:
    """Write a small Trace with the same indentation as PyTorch export."""
    events = ",\n".join(
        (
            "  {\n"
            f'    "ph": "X", "name": "event-{index}",\n'
            f'    "ts": {timestamp}, "dur": 1.000,\n'
            '    "args": {\n'
            '      "ts": 123\n'
            "    }\n"
            "  }"
        )
        for index, timestamp in enumerate(timestamps_us)
    )
    path.write_text(
        "{\n"
        f'  "baseTimeNanoseconds": {base_time_ns},\n'
        '  "traceEvents": [\n'
        f"{events}\n"
        "  ]\n"
        "}\n",
        encoding="utf-8",
    )


class TraceAlignmentTest(unittest.TestCase):
    """Validate timestamp conversion and strict coverage behavior."""

    def test_aligns_worker_trace_to_absolute_epoch(self) -> None:
        samples = synthetic_samples(duration_seconds=30)
        model = build_piecewise_model(
            samples,
            source={"hostname": "worker-a"},
            reference={"hostname": "head"},
        )
        bridge_samples = synthetic_bridge_samples(duration_seconds=30)
        model["realtime_monotonic_bridge"] = build_clock_bridge(
            bridge_samples,
            boot_id="boot-a",
        )
        target_base_ns = 1_700_000_000_000_000_000
        session = {
            "target_base_time_ns": target_base_ns,
            "models": [model],
        }
        event_bridge_sample = bridge_samples[100]
        trace_relative_ns = 5_000_000_000
        source_base_ns = (
            event_bridge_sample["bridge_realtime_ns"] - trace_relative_ns
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.json"
            session_path = root / "session.json"
            output_path = root / "aligned.json"
            write_trace(trace_path, source_base_ns, ["5000000.000"])
            session_path.write_text(json.dumps(session), encoding="utf-8")

            stats = align_trace_file(
                trace_path=trace_path,
                session_path=session_path,
                source_node="worker-a",
                output_path=output_path,
                source_boot_id="boot-a",
            )
            aligned = json.loads(output_path.read_text(encoding="utf-8"))

        event = aligned["traceEvents"][0]
        expected_offset_ns = 145_000
        self.assertEqual(aligned["baseTimeNanoseconds"], target_base_ns)
        self.assertAlmostEqual(
            event["ts"],
            (
                event_bridge_sample["bridge_realtime_ns"]
                + expected_offset_ns
                - target_base_ns
            )
            / 1_000,
            delta=1.0,
        )
        self.assertEqual(event["args"]["ts"], 123)
        self.assertEqual(stats.timestamp_count, 1)
        self.assertAlmostEqual(
            stats.first_source_monotonic_ns,
            event_bridge_sample["bridge_monotonic_ns"],
            delta=1_000,
        )
        self.assertEqual(stats.bridge_boot_id, "boot-a")
        self.assertGreater(stats.max_total_uncertainty_us, 50.0)
        self.assertLess(stats.max_total_uncertainty_us, 51.0)

    def test_identity_model_only_changes_base(self) -> None:
        session = {
            "target_base_time_ns": 0,
            "models": [
                {
                    "model_type": "identity",
                    "status": "PASS",
                    "source": {"hostname": "head"},
                    "segments": [],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.json"
            session_path = root / "session.json"
            output_path = root / "aligned.json"
            write_trace(trace_path, 10_000, ["2.123"])
            session_path.write_text(json.dumps(session), encoding="utf-8")

            align_trace_file(
                trace_path=trace_path,
                session_path=session_path,
                source_node="head",
                output_path=output_path,
            )
            aligned = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(aligned["baseTimeNanoseconds"], 0)
        self.assertEqual(aligned["traceEvents"][0]["ts"], 12.123)

    def test_rejects_trace_outside_model_coverage_atomically(self) -> None:
        model = build_piecewise_model(
            synthetic_samples(duration_seconds=30),
            source={"hostname": "worker-a"},
            reference={"hostname": "head"},
        )
        model["realtime_monotonic_bridge"] = build_clock_bridge(
            synthetic_bridge_samples(duration_seconds=30),
            boot_id="boot-a",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.json"
            session_path = root / "session.json"
            output_path = root / "aligned.json"
            write_trace(trace_path, 10_000, ["1.000"])
            session_path.write_text(
                json.dumps(
                    {"target_base_time_ns": 0, "models": [model]}
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "No clock-bridge segment"):
                align_trace_file(
                    trace_path=trace_path,
                    session_path=session_path,
                    source_node="worker-a",
                    output_path=output_path,
                )
            self.assertFalse(output_path.exists())

    def test_rejects_trace_outside_probe_model_coverage(self) -> None:
        model = build_piecewise_model(
            synthetic_samples(duration_seconds=30),
            source={"hostname": "worker-a"},
            reference={"hostname": "head"},
        )
        bridge_samples = synthetic_bridge_samples(duration_seconds=60)
        model["realtime_monotonic_bridge"] = build_clock_bridge(
            bridge_samples,
            boot_id="boot-a",
        )
        event = bridge_samples[400]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.json"
            session_path = root / "session.json"
            output_path = root / "aligned.json"
            write_trace(
                trace_path,
                event["bridge_realtime_ns"] - 2_000_000_000,
                ["2000000.000"],
            )
            session_path.write_text(
                json.dumps(
                    {"target_base_time_ns": 0, "models": [model]}
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "No model segment covers"):
                align_trace_file(
                    trace_path=trace_path,
                    session_path=session_path,
                    source_node="worker-a",
                    output_path=output_path,
                )
            self.assertFalse(output_path.exists())

    def test_rejects_legacy_model_without_clock_bridge(self) -> None:
        model = build_piecewise_model(
            synthetic_samples(duration_seconds=30),
            source={"hostname": "worker-a"},
            reference={"hostname": "head"},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.json"
            session_path = root / "session.json"
            output_path = root / "aligned.json"
            write_trace(trace_path, 10_000, ["1.000"])
            session_path.write_text(
                json.dumps(
                    {"target_base_time_ns": 0, "models": [model]}
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "no REALTIME-to-MONOTONIC"):
                align_trace_file(
                    trace_path=trace_path,
                    session_path=session_path,
                    source_node="worker-a",
                    output_path=output_path,
                )

    def test_rejects_source_boot_id_mismatch(self) -> None:
        model = build_piecewise_model(
            synthetic_samples(duration_seconds=30),
            source={"hostname": "worker-a"},
            reference={"hostname": "head"},
        )
        bridge_samples = synthetic_bridge_samples(duration_seconds=30)
        model["realtime_monotonic_bridge"] = build_clock_bridge(
            bridge_samples,
            boot_id="boot-a",
        )
        event = bridge_samples[100]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.json"
            session_path = root / "session.json"
            output_path = root / "aligned.json"
            write_trace(
                trace_path,
                event["bridge_realtime_ns"] - 1_000_000_000,
                ["1000000.000"],
            )
            session_path.write_text(
                json.dumps(
                    {"target_base_time_ns": 0, "models": [model]}
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "boot ID"):
                align_trace_file(
                    trace_path=trace_path,
                    session_path=session_path,
                    source_node="worker-a",
                    output_path=output_path,
                    source_boot_id="boot-b",
                )
            self.assertFalse(output_path.exists())

    def test_aligns_hardware_trace_to_phc_domain(self) -> None:
        samples = synthetic_phc_samples(duration_seconds=30)
        bridge = build_phc_bridge(samples, boot_id="boot-a")
        event = samples[100]
        target_base_ns = 1_700_000_000_000_000_000
        session = {
            "clock_source": "ptp_hardware",
            "timestamp_domain": "PHC",
            "target_base_time_ns": target_base_ns,
            "ptp": {"uncertainty_us": 0.32},
            "models": [
                {
                    "model_type": "phc_bridge",
                    "status": "PASS",
                    "source": {"hostname": "cse-ai-6"},
                    "ptp_uncertainty_us": 0.32,
                    "realtime_phc_bridge": bridge,
                }
            ],
        }
        trace_relative_ns = 5_000_000_000
        source_base_ns = event["bridge_realtime_ns"] - trace_relative_ns
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.json"
            session_path = root / "session.json"
            output_path = root / "aligned.json"
            write_trace(trace_path, source_base_ns, ["5000000.000"])
            session_path.write_text(json.dumps(session), encoding="utf-8")

            stats = align_trace_file(
                trace_path=trace_path,
                session_path=session_path,
                source_node="cse-ai-6",
                output_path=output_path,
                source_boot_id="boot-a",
            )
            aligned = json.loads(output_path.read_text(encoding="utf-8"))

        mapped_phc_ns = stats.first_source_phc_ns
        self.assertIsNotNone(mapped_phc_ns)
        self.assertEqual(stats.clock_source, "ptp_hardware")
        self.assertAlmostEqual(mapped_phc_ns, event["bridge_phc_ns"], delta=1_000)
        self.assertEqual(aligned["baseTimeNanoseconds"], target_base_ns)
        self.assertAlmostEqual(
            aligned["traceEvents"][0]["ts"],
            (mapped_phc_ns - target_base_ns) / 1_000,
            delta=1.0,
        )
        self.assertGreaterEqual(stats.max_total_uncertainty_us, 0.32)
        self.assertLess(stats.max_total_uncertainty_us, 1.5)

    def test_rejects_hardware_session_without_phc_domain(self) -> None:
        session = {
            "clock_source": "ptp_hardware",
            "timestamp_domain": "CLOCK_REALTIME",
            "target_base_time_ns": 0,
            "models": [
                {
                    "model_type": "phc_bridge",
                    "status": "PASS",
                    "source": {"hostname": "cse-ai-6"},
                    "realtime_phc_bridge": build_phc_bridge(
                        synthetic_phc_samples(duration_seconds=30),
                        boot_id="boot-a",
                    ),
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.json"
            session_path = root / "session.json"
            output_path = root / "aligned.json"
            write_trace(trace_path, 10_000, ["1.000"])
            session_path.write_text(json.dumps(session), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "timestamp_domain=PHC"):
                align_trace_file(
                    trace_path=trace_path,
                    session_path=session_path,
                    source_node="cse-ai-6",
                    output_path=output_path,
                )
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
