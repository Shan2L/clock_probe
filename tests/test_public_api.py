"""Contract tests for the small embedding API."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import clock_probe
from clock_probe.cli import build_parser
from clock_probe.postprocess.pipeline import TraceInput, process_traces
from clock_probe.session import load_session, select_model, session_uncertainty_us
from clock_probe.postprocess.align import AlignmentStats


def software_session() -> dict:
    identity = {
        "model_type": "identity",
        "status": "PASS",
        "source": {"hostname": "head"},
        "segments": [],
    }
    worker = {
        "model_type": "interpolated_offset",
        "status": "PASS",
        "source": {"hostname": "worker"},
        "segments": [
            {
                "status": "PASS",
                "uncertainty_us": 5.0,
            }
        ],
        "realtime_monotonic_bridge": {
            "status": "PASS",
            "segments": [
                {
                    "status": "PASS",
                    "uncertainty_us": 1.0,
                }
            ],
        },
    }
    return {
        "status": "PASS",
        "target_base_time_ns": 1_000_000_000,
        "models": [identity, worker],
    }


class PublicApiTest(unittest.TestCase):
    def test_import_surface_is_small_and_ray_optional(self) -> None:
        self.assertTrue(callable(clock_probe.process_traces))
        self.assertTrue(hasattr(clock_probe.probe, "start"))
        self.assertTrue(hasattr(clock_probe.hardware, "fit"))

    def test_loads_legacy_software_session(self) -> None:
        session = load_session(software_session())
        self.assertEqual(session["clock_source"], "udp_software")
        self.assertEqual(select_model(session, "worker")["model_type"], "interpolated_offset")
        self.assertEqual(session_uncertainty_us(session), 6.0)

    def test_reads_legacy_piecewise_phc_session(self) -> None:
        session = load_session(
            {
                "clock_source": "ptp_hardware",
                "timestamp_domain": "PHC",
                "status": "PASS",
                "ptp": {"uncertainty_us": 0.2},
                "models": [
                    {
                        "model_type": "phc_bridge",
                        "status": "PASS",
                        "source": {"hostname": "node-a"},
                        "uncertainty_us": 1.0,
                        "realtime_phc_bridge": {
                            "model_type": "piecewise_realtime_phc",
                            "status": "PASS",
                            "segments": [
                                {"status": "PASS", "uncertainty_us": 0.8}
                            ],
                        },
                    }
                ],
            }
        )
        self.assertEqual(session["clock_source"], "ptp_hardware")
        self.assertEqual(session["timestamp_domain"], "PHC")
        self.assertEqual(session_uncertainty_us(session), 1.0)

    def test_single_cli_surface(self) -> None:
        args = build_parser().parse_args(["status"])
        self.assertEqual(args.command, "status")

    def test_pipeline_returns_fail_closed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            traces = {}
            for rank, node in ((0, "head"), (1, "worker")):
                path = root / f"rank-{rank}.json"
                path.write_text("{}", encoding="utf-8")
                traces[rank] = TraceInput(path=path, source_node=node)

            def align(**kwargs):
                kwargs["output_path"].write_text("{}", encoding="utf-8")
                return AlignmentStats(
                    source_node=kwargs["source_node"],
                    input_path=str(kwargs["trace_path"]),
                    output_path=str(kwargs["output_path"]),
                    source_base_time_ns=0,
                    target_base_time_ns=0,
                    timestamp_count=1,
                    first_source_monotonic_ns=None,
                    last_source_monotonic_ns=None,
                    first_source_phc_ns=None,
                    last_source_phc_ns=None,
                    bridge_boot_id=None,
                    clock_source="udp_software",
                    max_bridge_uncertainty_us=0.0,
                    max_ptp_uncertainty_us=0.0,
                    max_total_uncertainty_us=0.0,
                )

            check = SimpleNamespace(
                status="FAIL",
                to_dict=lambda: {
                    "status": "FAIL",
                    "uncertainty_us": 6.0,
                    "inversion_count": 1,
                    "max_gap_us": 20.0,
                },
            )
            with (
                patch(
                    "clock_probe.postprocess.pipeline.align_trace_file",
                    side_effect=align,
                ),
                patch(
                    "clock_probe.postprocess.pipeline.check_nccl_traces",
                    return_value=check,
                ),
            ):
                manifest = process_traces(
                    traces,
                    software_session(),
                    root / "out",
                    apply_clc_on_warning=True,
                )
            self.assertEqual(manifest.status, "FAIL")
            self.assertEqual(manifest.primary_timeline, "none")
            self.assertFalse(manifest.clc)


if __name__ == "__main__":
    unittest.main()
