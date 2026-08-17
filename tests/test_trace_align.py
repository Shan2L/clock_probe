"""Tests for streaming PyTorch Trace alignment."""

import json
import tempfile
import unittest
from pathlib import Path

from clock_probe.model import build_piecewise_model
from clock_probe.trace_align import align_trace_file
from test_model import synthetic_samples


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
        session = {"models": [model]}
        source_base_ns = 1_700_000_000_000_000_000

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.json"
            session_path = root / "session.json"
            output_path = root / "aligned.json"
            write_trace(trace_path, source_base_ns, ["1000000000.000"])
            session_path.write_text(json.dumps(session), encoding="utf-8")

            stats = align_trace_file(
                trace_path=trace_path,
                session_path=session_path,
                source_node="worker-a",
                output_path=output_path,
            )
            aligned = json.loads(output_path.read_text(encoding="utf-8"))

        event = aligned["traceEvents"][0]
        expected_offset_ns = 125_000
        self.assertEqual(aligned["baseTimeNanoseconds"], 0)
        self.assertAlmostEqual(
            event["ts"],
            (
                source_base_ns
                + 1_000_000_000_000
                + expected_offset_ns
            )
            / 1_000,
            delta=1.0,
        )
        self.assertEqual(event["args"]["ts"], 123)
        self.assertEqual(stats.timestamp_count, 1)

    def test_identity_model_only_changes_base(self) -> None:
        session = {
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
            write_trace(trace_path, 10_000, ["2.500"])
            session_path.write_text(json.dumps(session), encoding="utf-8")

            align_trace_file(
                trace_path=trace_path,
                session_path=session_path,
                source_node="head",
                output_path=output_path,
            )
            aligned = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(aligned["baseTimeNanoseconds"], 0)
        self.assertEqual(aligned["traceEvents"][0]["ts"], 12.5)

    def test_rejects_trace_outside_model_coverage_atomically(self) -> None:
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
                json.dumps({"models": [model]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "precedes model coverage"):
                align_trace_file(
                    trace_path=trace_path,
                    session_path=session_path,
                    source_node="worker-a",
                    output_path=output_path,
                )
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
