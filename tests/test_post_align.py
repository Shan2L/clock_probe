"""Tests for NCCL check, restricted CLC, and the raw/aligned report."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from clock_probe.clc import apply_clc, plan_kernel_shifts
from clock_probe.nccl_check import check_nccl_traces, extract_nccl_kernels, match_collectives
from clock_probe.report import build_clock_report


def write_nccl_trace(
    path: Path,
    *,
    pid: int,
    kernels: list[dict[str, object]],
) -> None:
    """Write a pretty-printed Trace with GPU NCCL kernels."""
    events = []
    for kernel in kernels:
        extra = ""
        if "follower_ts" in kernel:
            extra = (
                ",\n"
                "  {\n"
                '    "ph": "X", "cat": "kernel", "name": "vector_add", '
                f'"pid": {pid}, "tid": 0,\n'
                f'    "ts": {kernel["follower_ts"]}, "dur": 1.000,\n'
                '    "args": {"stream": 0}\n'
                "  }"
            )
        seq_field = ""
        if "seq" in kernel:
            seq_field = f', "Seq": {kernel["seq"]}'
        in_nelems = kernel.get("in_nelems", 1024)
        out_nelems = kernel.get("out_nelems", 1024)
        events.append(
            "  {\n"
            '    "ph": "X", "cat": "kernel", '
            '"name": "ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)", '
            f'"pid": {pid}, "tid": 0,\n'
            f'    "ts": {kernel["ts"]}, "dur": {kernel["dur"]},\n'
            "    \"args\": {\n"
            f'      "stream": 0, "correlation": {kernel["correlation"]}, '
            f'"Collective name": "{kernel["collective"]}", '
            f'"Process Group Name": "{kernel["pg"]}", "Group size": 2, '
            f'"In msg nelems": {in_nelems}, "Out msg nelems": {out_nelems}, '
            f'"dtype": "float16"{seq_field}\n'
            "    }\n"
            "  }"
            f"{extra}"
        )
    path.write_text(
        "{\n"
        '  "baseTimeNanoseconds": 1000,\n'
        '  "traceEvents": [\n'
        + ",\n".join(events)
        + "\n  ]\n}\n",
        encoding="utf-8",
    )


class NCCLCheckTest(unittest.TestCase):
    """Held-out overlap checks on implicit-sync collectives."""

    def test_pass_when_collectives_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "rank0.json"
            right = root / "rank1.json"
            write_nccl_trace(
                left,
                pid=2,
                kernels=[
                    {
                        "ts": 100.000,
                        "dur": 20.000,
                        "correlation": 1,
                        "collective": "all_gather",
                        "pg": "0",
                    }
                ],
            )
            write_nccl_trace(
                right,
                pid=3,
                kernels=[
                    {
                        "ts": 105.000,
                        "dur": 20.000,
                        "correlation": 1,
                        "collective": "all_gather",
                        "pg": "0",
                    }
                ],
            )
            report = check_nccl_traces(
                {0: left, 1: right},
                uncertainty_us=5.0,
            )
        self.assertEqual(report.status, "PASS")
        self.assertEqual(report.inversion_count, 0)
        self.assertEqual(report.matched_collectives, 1)

    def test_warning_when_gap_within_uncertainty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "rank0.json"
            right = root / "rank1.json"
            write_nccl_trace(
                left,
                pid=2,
                kernels=[
                    {
                        "ts": 100.000,
                        "dur": 10.000,
                        "correlation": 1,
                        "collective": "all_gather",
                        "pg": "0",
                    }
                ],
            )
            write_nccl_trace(
                right,
                pid=3,
                kernels=[
                    {
                        "ts": 112.000,
                        "dur": 10.000,
                        "correlation": 1,
                        "collective": "all_gather",
                        "pg": "0",
                    }
                ],
            )
            report = check_nccl_traces(
                {0: left, 1: right},
                uncertainty_us=5.0,
            )
        self.assertEqual(report.status, "WARNING")
        self.assertEqual(report.inversion_count, 1)
        self.assertAlmostEqual(report.max_gap_us, 2.0, delta=0.001)

    def test_fail_when_gap_exceeds_uncertainty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "rank0.json"
            right = root / "rank1.json"
            write_nccl_trace(
                left,
                pid=2,
                kernels=[
                    {
                        "ts": 100.000,
                        "dur": 10.000,
                        "correlation": 1,
                        "collective": "all_gather",
                        "pg": "0",
                    }
                ],
            )
            write_nccl_trace(
                right,
                pid=3,
                kernels=[
                    {
                        "ts": 200.000,
                        "dur": 10.000,
                        "correlation": 1,
                        "collective": "all_gather",
                        "pg": "0",
                    }
                ],
            )
            report = check_nccl_traces(
                {0: left, 1: right},
                uncertainty_us=5.0,
            )
        self.assertEqual(report.status, "FAIL")
        self.assertGreater(report.max_gap_us, 5.0)

    def test_matches_by_seq_num_when_order_differs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "rank0.json"
            right = root / "rank1.json"
            write_nccl_trace(
                left,
                pid=2,
                kernels=[
                    {
                        "ts": 100.000,
                        "dur": 20.000,
                        "correlation": 1,
                        "collective": "all_reduce",
                        "pg": "0",
                        "seq": 1,
                    },
                    {
                        "ts": 130.000,
                        "dur": 20.000,
                        "correlation": 2,
                        "collective": "all_gather",
                        "pg": "0",
                        "seq": 2,
                    },
                ],
            )
            write_nccl_trace(
                right,
                pid=3,
                kernels=[
                    {
                        "ts": 128.000,
                        "dur": 20.000,
                        "correlation": 1,
                        "collective": "all_gather",
                        "pg": "0",
                        "seq": 2,
                    },
                    {
                        "ts": 105.000,
                        "dur": 20.000,
                        "correlation": 2,
                        "collective": "all_reduce",
                        "pg": "0",
                        "seq": 1,
                    },
                ],
            )
            kernels = extract_nccl_kernels(left, 0) + extract_nccl_kernels(right, 1)
            match = match_collectives(kernels)
            report = check_nccl_traces(
                {0: left, 1: right},
                uncertainty_us=5.0,
            )
        self.assertFalse(match.matching_failed)
        self.assertEqual(match.matching_mode, "seq_num")
        self.assertEqual(len(match.matched), 2)
        self.assertEqual(report.status, "PASS")
        self.assertEqual(report.matching_mode, "seq_num")

    def test_matches_by_fingerprint_without_seq_num(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "rank0.json"
            right = root / "rank1.json"
            write_nccl_trace(
                left,
                pid=2,
                kernels=[
                    {
                        "ts": 100.000,
                        "dur": 10.000,
                        "correlation": 1,
                        "collective": "all_gather",
                        "pg": "0",
                        "in_nelems": 1024,
                        "out_nelems": 1024,
                    },
                    {
                        "ts": 130.000,
                        "dur": 10.000,
                        "correlation": 2,
                        "collective": "all_gather",
                        "pg": "0",
                        "in_nelems": 2048,
                        "out_nelems": 2048,
                    },
                ],
            )
            write_nccl_trace(
                right,
                pid=3,
                kernels=[
                    {
                        "ts": 140.000,
                        "dur": 10.000,
                        "correlation": 1,
                        "collective": "all_gather",
                        "pg": "0",
                        "in_nelems": 2048,
                        "out_nelems": 2048,
                    },
                    {
                        "ts": 110.000,
                        "dur": 10.000,
                        "correlation": 2,
                        "collective": "all_gather",
                        "pg": "0",
                        "in_nelems": 1024,
                        "out_nelems": 1024,
                    },
                ],
            )
            match = match_collectives(
                extract_nccl_kernels(left, 0) + extract_nccl_kernels(right, 1)
            )
            report = check_nccl_traces(
                {0: left, 1: right},
                uncertainty_us=5.0,
            )
        self.assertFalse(match.matching_failed)
        self.assertEqual(match.matching_mode, "fingerprint")
        self.assertEqual(len(match.matched), 2)
        self.assertEqual(report.status, "PASS")
        self.assertEqual(report.matching_mode, "fingerprint")

    def test_fails_on_duplicate_seq_num(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "rank0.json"
            right = root / "rank1.json"
            write_nccl_trace(
                left,
                pid=2,
                kernels=[
                    {
                        "ts": 100.000,
                        "dur": 10.000,
                        "correlation": 1,
                        "collective": "all_gather",
                        "pg": "0",
                        "seq": 1,
                    },
                    {
                        "ts": 120.000,
                        "dur": 10.000,
                        "correlation": 2,
                        "collective": "all_gather",
                        "pg": "0",
                        "seq": 1,
                    },
                ],
            )
            write_nccl_trace(
                right,
                pid=3,
                kernels=[
                    {
                        "ts": 105.000,
                        "dur": 10.000,
                        "correlation": 1,
                        "collective": "all_gather",
                        "pg": "0",
                        "seq": 1,
                    }
                ],
            )
            report = check_nccl_traces(
                {0: left, 1: right},
                uncertainty_us=5.0,
            )
        self.assertEqual(report.status, "FAIL")
        self.assertGreater(report.ambiguous_groups, 0)
        self.assertIn("Duplicate seq_num", report.notes[1])


class CLCTest(unittest.TestCase):
    """Restricted CLC delays early finishers only."""

    def test_delays_early_rank_and_later_stream_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "rank0.aligned.json"
            right = root / "rank1.aligned.json"
            out0 = root / "rank0.clc.json"
            out1 = root / "rank1.clc.json"
            write_nccl_trace(
                left,
                pid=2,
                kernels=[
                    {
                        "ts": 100.000,
                        "dur": 10.000,
                        "correlation": 1,
                        "collective": "all_gather",
                        "pg": "0",
                        "follower_ts": 111.000,
                    }
                ],
            )
            write_nccl_trace(
                right,
                pid=3,
                kernels=[
                    {
                        "ts": 112.000,
                        "dur": 10.000,
                        "correlation": 1,
                        "collective": "all_gather",
                        "pg": "0",
                    }
                ],
            )
            shifts = plan_kernel_shifts(
                {0: left, 1: right},
                uncertainty_us=5.0,
            )
            self.assertEqual(len(shifts), 1)
            self.assertEqual(shifts[0].rank, 0)
            self.assertEqual(shifts[0].delta_ns, 2_000)

            report = apply_clc(
                {0: left, 1: right},
                {0: out0, 1: out1},
                uncertainty_us=5.0,
                check_status="WARNING",
            )
            self.assertEqual(report.status, "APPLIED")
            aligned = json.loads(out0.read_text(encoding="utf-8"))
            events = aligned["traceEvents"]
            self.assertAlmostEqual(events[0]["ts"], 102.000, delta=0.001)
            self.assertAlmostEqual(events[1]["ts"], 113.000, delta=0.001)
            late = json.loads(out1.read_text(encoding="utf-8"))
            self.assertAlmostEqual(late["traceEvents"][0]["ts"], 112.000, delta=0.001)
            self.assertTrue(left.exists())

    def test_refuses_fail_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "rank0.aligned.json"
            right = root / "rank1.aligned.json"
            write_nccl_trace(
                left,
                pid=2,
                kernels=[
                    {
                        "ts": 100.000,
                        "dur": 10.000,
                        "correlation": 1,
                        "collective": "all_gather",
                        "pg": "0",
                    }
                ],
            )
            write_nccl_trace(
                right,
                pid=3,
                kernels=[
                    {
                        "ts": 200.000,
                        "dur": 10.000,
                        "correlation": 1,
                        "collective": "all_gather",
                        "pg": "0",
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "Refuse CLC"):
                apply_clc(
                    {0: left, 1: right},
                    {0: root / "rank0.clc.json"},
                    uncertainty_us=5.0,
                    check_status="FAIL",
                )
            self.assertFalse((root / "rank0.clc.json").exists())

    def test_skips_when_check_is_pass(self) -> None:
        report = apply_clc(
            {},
            {},
            uncertainty_us=5.0,
            check_status="PASS",
        )
        self.assertEqual(report.status, "SKIPPED")


class ClockReportTest(unittest.TestCase):
    """Raw / aligned / CLC paths stay distinct."""

    def test_marks_aligned_candidate_on_nccl_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "rank0.json"
            aligned = root / "rank0.aligned.json"
            raw.write_text("{}", encoding="utf-8")
            aligned.write_text("{}", encoding="utf-8")
            check_path = root / "nccl-check.json"
            check_path.write_text(
                json.dumps(
                    {
                        "status": "FAIL",
                        "inversion_count": 3,
                        "max_gap_us": 90.0,
                        "uncertainty_us": 5.0,
                    }
                ),
                encoding="utf-8",
            )
            report = build_clock_report(
                raw={0: raw},
                aligned={0: aligned},
                clc=None,
                nccl_check_path=check_path,
                clc_report_path=None,
            )
        self.assertEqual(report.session_status, "FAIL")
        self.assertEqual(report.primary_timeline, "none")

    def test_rejects_identical_raw_and_aligned_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "rank0.json"
            raw.write_text("{}", encoding="utf-8")
            check_path = root / "nccl-check.json"
            check_path.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "same file"):
                build_clock_report(
                    raw={0: raw},
                    aligned={0: raw},
                    clc=None,
                    nccl_check_path=check_path,
                    clc_report_path=None,
                )


if __name__ == "__main__":
    unittest.main()
