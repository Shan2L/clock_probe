"""Tests for manual lifecycle guards and coordinator cleanup."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from clock_probe.ray_agent import (
    _get_active_coordinator,
    start_manual_calibration,
    stop_manual_calibration,
)


class RemoteMethod:
    """Small stand-in for a Ray actor method."""

    def __init__(self, result):
        self.result = result

    def remote(self):
        return self.result


class FakeCoordinator:
    """Coordinator handle exposing methods used by lifecycle helpers."""

    def __init__(self, *, status=None, session=None):
        self.status = RemoteMethod(status)
        self.stop_and_build = RemoteMethod(session)


class FakeRay:
    """Minimal Ray surface for lifecycle unit tests."""

    def __init__(self, coordinator=None):
        self.coordinator = coordinator
        self.killed = False
        self.shutdown_called = False

    def get_actor(self, _name, namespace):
        del namespace
        if self.coordinator is None:
            raise ValueError("not found")
        return self.coordinator

    @staticmethod
    def get(value):
        return value

    def kill(self, actor, no_restart):
        del actor, no_restart
        self.killed = True

    def shutdown(self):
        self.shutdown_called = True


class ManualLifecycleTest(unittest.TestCase):
    """Validate duplicate/missing-session guards and stop cleanup."""

    def test_missing_session_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "No active"):
            _get_active_coordinator(FakeRay())

    def test_duplicate_start_is_rejected(self) -> None:
        fake_ray = FakeRay(
            FakeCoordinator(
                status={"session_id": "existing", "state": "RUNNING"}
            )
        )
        with (
            patch(
                "clock_probe.ray_agent._connect_ray",
                return_value=fake_ray,
            ),
            self.assertRaisesRegex(RuntimeError, "already active"),
        ):
            start_manual_calibration(
                SimpleNamespace(ray_address="auto")
            )
        self.assertTrue(fake_ray.shutdown_called)

    def test_stop_writes_session_and_kills_coordinator(self) -> None:
        session = {
            "session_id": "test-session",
            "status": "PASS",
            "node_count": 2,
            "worker_model_count": 1,
        }
        coordinator = FakeCoordinator(session=session)
        fake_ray = FakeRay(coordinator)
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "session.json"
            with patch(
                "clock_probe.ray_agent._connect_ray",
                return_value=fake_ray,
            ):
                result = stop_manual_calibration(
                    SimpleNamespace(
                        ray_address="auto",
                        output=str(output_path),
                    )
                )
            self.assertEqual(result, session)
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                session,
            )
        self.assertTrue(fake_ray.killed)
        self.assertTrue(fake_ray.shutdown_called)


if __name__ == "__main__":
    unittest.main()
