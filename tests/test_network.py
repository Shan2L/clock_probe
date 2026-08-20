"""Tests for timestamp-capable interface selection."""

import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from clock_probe.execution.network import (
    InterfaceAddress,
    NetworkInterface,
    parse_timestamping_capabilities,
    reference_candidates,
    route_to,
)


def interface(
    name: str,
    address: str,
    *,
    transmit: bool = True,
    receive: bool = True,
) -> NetworkInterface:
    """Create one UP test interface."""
    return NetworkInterface(
        name=name,
        index=1,
        is_up=True,
        is_loopback=False,
        software_transmit=transmit,
        software_receive=receive,
        ipv4_addresses=(
            InterfaceAddress(address, 24, "global"),
        ),
    )


class NetworkSelectionTest(unittest.TestCase):
    """Validate capability parsing, candidates, and worker routes."""

    def test_parses_required_ethtool_capabilities(self) -> None:
        output = """
Capabilities:
\tsoftware-transmit
\tsoftware-receive
\tsoftware-system-clock
"""
        self.assertEqual(
            parse_timestamping_capabilities(output),
            (True, True),
        )

    def test_prefers_ray_address_but_keeps_fallbacks(self) -> None:
        candidates = reference_candidates(
            [
                interface("roce0", "192.168.10.1"),
                interface("mgmt0", "10.67.91.123"),
                interface("partial0", "10.1.1.1", receive=False),
            ],
            preferred_address="10.67.91.123",
        )
        self.assertEqual(candidates[0]["interface"], "mgmt0")
        self.assertEqual(len(candidates), 2)

    def test_strict_selection_rejects_unsupported_address(self) -> None:
        with self.assertRaises(RuntimeError):
            reference_candidates(
                [interface("partial0", "10.1.1.1", receive=False)],
                preferred_address="10.1.1.1",
                strict=True,
            )

    @patch("clock_probe.execution.network.subprocess.run")
    def test_route_requires_capable_output_interface(self, run_mock) -> None:
        run_mock.return_value = SimpleNamespace(
            stdout=(
                '[{"dst":"10.67.91.123","dev":"mgmt0",'
                '"prefsrc":"10.67.93.244"}]'
            ),
            returncode=0,
        )
        route = route_to(
            "10.67.91.123",
            [interface("mgmt0", "10.67.93.244")],
        )
        self.assertTrue(route["usable"])
        self.assertEqual(route["interface"], "mgmt0")
        self.assertEqual(route["source_address"], "10.67.93.244")
        run_mock.assert_called_once_with(
            ["ip", "-j", "route", "get", "10.67.91.123"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )

    @patch(
        "clock_probe.execution.network.subprocess.run",
        side_effect=subprocess.CalledProcessError(2, "ip"),
    )
    def test_route_lookup_failure_is_reported(self, _run_mock) -> None:
        route = route_to(
            "10.67.91.123",
            [interface("mgmt0", "10.67.93.244")],
        )
        self.assertFalse(route["usable"])
        self.assertIn("route lookup failed", route["reason"])


if __name__ == "__main__":
    unittest.main()
