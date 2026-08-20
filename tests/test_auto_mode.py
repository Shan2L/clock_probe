"""Hardware-first mode selection contract."""

import unittest
from contextlib import redirect_stderr
from io import StringIO

from clock_probe import ProbeConfig
from clock_probe.cli import build_parser
from clock_probe.execution.ray import _select_probe_mode


class AutoModeTest(unittest.TestCase):
    def test_default_is_hardware_first_auto(self) -> None:
        self.assertEqual(ProbeConfig().mode, "auto")

    def test_auto_uses_hardware_only_when_every_node_passes(self) -> None:
        mode, failures = _select_probe_mode(
            "auto",
            [
                {"hostname": "a", "usable": True},
                {"hostname": "b", "usable": True},
            ],
        )
        self.assertEqual(mode, "hardware")
        self.assertFalse(failures)

    def test_auto_falls_back_before_sampling(self) -> None:
        failure = {"hostname": "b", "usable": False, "reason": "no PHC"}
        mode, failures = _select_probe_mode(
            "auto",
            [{"hostname": "a", "usable": True}, failure],
        )
        self.assertEqual(mode, "software")
        self.assertEqual(failures, [failure])

    def test_explicit_hardware_never_silently_falls_back(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Hardware preflight failed"):
            _select_probe_mode(
                "hardware",
                [{"hostname": "a", "usable": False, "reason": "not locked"}],
            )

    def test_cli_has_one_probe_lifecycle(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args(["start"]).command, "start")
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["software", "start"])
            with self.assertRaises(SystemExit):
                parser.parse_args(["hardware", "sample"])


if __name__ == "__main__":
    unittest.main()
