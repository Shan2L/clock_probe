"""One-command Ray cluster clock calibration orchestrator."""

# Ray is an optional runtime dependency available in the vLLM environment.
# pylint: disable=import-error,too-many-lines

from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..calibration.clock_bridge import build_clock_bridge, read_boot_id
from ..calibration.hardware import (
    HardwareModelConfig,
    build_hardware_model,
    build_hardware_session,
)
from ..calibration.ptp_health import evaluate_ptp_health, parse_ptp4l_log
from ..calibration.software import ModelConfig, build_clock_model
from .network import (
    list_network_interfaces,
    reference_candidates,
    route_to,
)
from ..sampling.probe import (
    ContinuousProbeCollector,
    TimestampProbeServer,
)
from ..sampling.phc import (
    PhcClock,
    assert_phc_matches_interface,
    capture_phc_sample,
)
from .topology import RayNode, discover_alive_nodes, get_head_node

COORDINATOR_NAME = "clock-probe-coordinator"
COORDINATOR_NAMESPACE = "clock-probe"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trace_base_time_ns() -> int:
    """Return one hour-aligned Unix base shared by aligned Trace files."""
    hour_ns = 3_600_000_000_000
    return time.time_ns() // hour_ns * hour_ns


def _max_bridge_uncertainty_us(bridge: dict[str, Any]) -> float:
    values = [
        float(segment["uncertainty_us"])
        for segment in bridge.get("segments", [])
        if segment.get("status") == "PASS"
    ]
    return max(values, default=0.0)


def _node_identity(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "ray_node_id": node["node_id"],
        "ray_node_name": node["name"],
        "ray_node_address": node["address"],
        "hostname": socket.gethostname(),
        "boot_id": read_boot_id(),
    }


def _select_probe_mode(
    requested_mode: str,
    preflight: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Choose hardware only when every node passes before sampling starts."""
    if requested_mode == "software":
        return "software", []
    failures = [result for result in preflight if not result.get("usable")]
    if not failures:
        return "hardware", []
    if requested_mode == "hardware":
        raise RuntimeError(f"Hardware preflight failed: {failures}")
    return "software", failures


class ClockNodeAgent:  # pylint: disable=too-many-instance-attributes
    """Actor implementation pinned once to each physical Ray node."""

    def __init__(
        self,
        *,
        node: dict[str, Any],
        reference: dict[str, Any],
        session_id: str,
        raw_output_root: str,
    ):
        self.node = node
        self.reference = reference
        self.identity = _node_identity(node)
        self.interfaces = list_network_interfaces()
        self.reference_identity = {
            "ray_node_id": reference["node_id"],
            "ray_node_name": reference["name"],
            "ray_node_address": reference["address"],
        }
        self.reference_host: str | None = None
        self.reference_port: int | None = None
        self.selected_route: dict[str, Any] | None = None
        self.session_id = session_id
        self.raw_output_root = Path(raw_output_root)
        self.server: TimestampProbeServer | None = None
        self.collector: ContinuousProbeCollector | None = None
        self.continuous_model_config: ModelConfig | None = None
        self.hardware_config: dict[str, Any] | None = None
        self.hardware_role: str | None = None
        self.hardware_samples: list[dict[str, Any]] = []
        self.hardware_thread: threading.Thread | None = None
        self.hardware_stop = threading.Event()
        self.hardware_started_at: str | None = None
        self.hardware_started_monotonic_ns: int | None = None
        self.hardware_error: str | None = None

    def status(self) -> dict[str, Any]:
        """Return actor identity and reference-server status."""
        return {
            "identity": self.identity,
            "is_reference": self.server is not None,
            "server": self.server.status() if self.server is not None else None,
            "collection": (
                self.collector.status()
                if self.collector is not None
                else None
            ),
            "network": self.selected_route,
            "hardware": {
                "running": (
                    self.hardware_thread is not None
                    and self.hardware_thread.is_alive()
                ),
                "role": self.hardware_role,
                "sample_count": len(self.hardware_samples),
                "error": self.hardware_error,
            }
            if self.hardware_config is not None
            else None,
        }

    def network_inventory(self) -> list[dict[str, Any]]:
        """Return local interfaces and timestamping capabilities."""
        return [interface.to_dict() for interface in self.interfaces]

    def reference_candidates(
        self,
        preferred_address: str | None,
        preferred_interface: str | None,
        strict: bool,
    ) -> list[dict[str, Any]]:
        """Return eligible reference addresses when this actor is the Head."""
        if self.node["node_id"] != self.reference["node_id"]:
            raise RuntimeError("Reference candidates can only be listed on Head")
        return reference_candidates(
            self.interfaces,
            preferred_address=preferred_address,
            preferred_interface=preferred_interface,
            strict=strict,
        )

    def inspect_route(self, destination: str) -> dict[str, Any]:
        """Check which local interface would carry traffic to destination."""
        return route_to(destination, self.interfaces)

    def configure_reference(
        self,
        reference_host: str,
        reference_port: int,
        selected_route: dict[str, Any],
    ) -> dict[str, Any]:
        """Configure a worker with its previously validated route."""
        if self.node["node_id"] == self.reference["node_id"]:
            raise RuntimeError("Use start_reference on the Head actor")
        if not selected_route.get("usable"):
            raise RuntimeError(
                f"Unusable route to reference: {selected_route.get('reason')}"
            )
        self.reference_host = reference_host
        self.reference_port = reference_port
        self.selected_route = selected_route
        self.reference_identity["ray_node_address"] = reference_host
        return selected_route

    def start_reference(
        self,
        candidate: dict[str, Any],
        reference_port: int,
    ) -> dict[str, Any]:
        """Bind and start the reference server on the selected Head NIC."""
        if self.node["node_id"] != self.reference["node_id"]:
            raise RuntimeError("Clock reference can only start on Ray Head")
        if self.server is not None:
            raise RuntimeError("Clock reference server is already running")
        self.reference_host = str(candidate["address"])
        self.reference_port = reference_port
        self.selected_route = {
            "usable": True,
            "interface": candidate["interface"],
            "source_address": candidate["address"],
            "destination": candidate["address"],
        }
        self.reference_identity["ray_node_address"] = self.reference_host
        self.server = TimestampProbeServer(
            self.reference_host,
            self.reference_port,
        )
        self.server.start()
        return self.status()

    def _node_output_path(self, suffix: str) -> Path:
        safe_node_id = self.node["node_id"].replace("/", "_")
        return self.raw_output_root / self.session_id / f"{safe_node_id}{suffix}"

    def _persist_model(self, model: dict[str, Any]) -> str:
        model_path = self._node_output_path(".clock-model.json")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(
            json.dumps(model, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return str(model_path)

    def identity_model(self) -> dict[str, Any]:
        """Return the Head's zero-offset model."""
        if self.server is None:
            raise RuntimeError("Only the reference agent has an identity model")
        model = {
            "schema_version": 2,
            "model_type": "identity",
            "offset_direction": "reference_minus_source",
            "timestamp_domain": "CLOCK_MONOTONIC",
            "source": self.identity,
            "reference": self.identity,
            "status": "PASS",
            "roles": ["ray_head", "ray_worker", "clock_reference"],
            "network": self.selected_route,
            "segments": [],
        }
        model["model_path_on_node"] = self._persist_model(model)
        return model

    def start_sampling(
        self,
        *,
        interval_ms: float,
        model_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Start continuous worker sampling and return immediately."""
        if self.server is not None:
            return self.status()
        if (
            self.reference_host is None
            or self.reference_port is None
            or self.selected_route is None
        ):
            raise RuntimeError("Clock reference route has not been configured")
        if self.collector is not None:
            raise RuntimeError("Clock sampling is already active")

        selected_config = ModelConfig(**model_config)
        selected_config.validate()
        self.continuous_model_config = selected_config
        self.collector = ContinuousProbeCollector(
            reference_host=self.reference_host,
            reference_port=self.reference_port,
            source_host=str(self.selected_route["source_address"]),
            interval_ms=interval_ms,
            output_path=self._node_output_path(".jsonl"),
        )
        self.collector.start()
        return self.status()

    def stop_sampling_and_build_model(self) -> dict[str, Any]:
        """Stop continuous sampling, fit, persist, and return one worker model."""
        if self.collector is None or self.continuous_model_config is None:
            raise RuntimeError("Clock sampling is not active")
        samples, collection = self.collector.stop()
        source_identity = {
            **self.identity,
            "clock_probe_interface": self.selected_route["interface"],
            "clock_probe_address": self.selected_route["source_address"],
        }
        bridge = build_clock_bridge(
            samples,
            boot_id=str(self.identity["boot_id"]),
        )
        model = build_clock_model(
            samples,
            source=source_identity,
            reference=self.reference_identity,
            config=self.continuous_model_config,
            local_bridge_uncertainty_us=_max_bridge_uncertainty_us(bridge),
        )
        model["schema_version"] = 2
        model["realtime_monotonic_bridge"] = bridge
        model["roles"] = ["ray_worker"]
        model["network"] = self.selected_route
        model["collection"] = collection
        model["model_path_on_node"] = self._persist_model(model)
        return model

    def hardware_preflight(self, config: dict[str, Any]) -> dict[str, Any]:
        """Check PHC access and current ptp4l lock before choosing hardware mode."""
        hostname = str(self.identity["hostname"])
        log_path = (config.get("ptp_logs") or {}).get(hostname) or config.get(
            "ptp_log"
        )
        try:
            if not log_path:
                raise ValueError(f"No ptp4l log configured for {hostname}")
            hardware = assert_phc_matches_interface(
                str(config["interface"]),
                str(config["phc_device"]),
            )
            text = Path(log_path).read_text(encoding="utf-8")
            parsed = parse_ptp4l_log(text)
            states = parsed["states"]
            if not states:
                raise ValueError("ptp4l log has no port state")
            state = str(states[-1]["state"])
            role = "master" if state == "MASTER" else ("slave" if state == "SLAVE" else "")
            if not role:
                raise ValueError(f"ptp4l port is not locked: {state}")
            model_config = HardwareModelConfig(**config.get("model_config", {}))
            health = evaluate_ptp_health(
                text,
                role=role,
                max_offset_p95_ns=model_config.max_ptp_offset_p95_ns,
            )
            if health.status != "PASS":
                raise ValueError("; ".join(health.reasons))
            self.hardware_config = {
                **config,
                "ptp_log": str(log_path),
                "hardware_timestamping": hardware.to_dict(),
            }
            self.hardware_role = role
            return {
                "usable": True,
                "hostname": hostname,
                "role": role,
                "phc_device": config["phc_device"],
                "interface": config["interface"],
            }
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            return {
                "usable": False,
                "hostname": hostname,
                "reason": str(error),
            }

    def start_hardware_sampling(self) -> dict[str, Any]:
        """Start local PHC sampling after successful preflight."""
        if self.hardware_config is None or self.hardware_role is None:
            raise RuntimeError("Hardware preflight has not passed")
        if self.hardware_thread is not None:
            raise RuntimeError("Hardware sampling is already active")
        self.hardware_samples = []
        self.hardware_stop.clear()
        self.hardware_error = None
        self.hardware_started_at = _utc_now()
        self.hardware_started_monotonic_ns = time.monotonic_ns()
        self.hardware_thread = threading.Thread(
            target=self._hardware_sampling_loop,
            name="clock-probe-phc",
            daemon=True,
        )
        self.hardware_thread.start()
        return self.status()

    def _hardware_sampling_loop(self) -> None:
        assert self.hardware_config is not None
        output_path = self._node_output_path(".phc.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        interval_s = float(self.hardware_config["interval_ms"]) / 1_000.0
        attempts = int(
            self.hardware_config.get("model_config", {}).get(
                "capture_attempts", 10
            )
        )
        next_at = time.monotonic()
        try:
            with (
                PhcClock(str(self.hardware_config["phc_device"])) as phc,
                output_path.open("w", encoding="utf-8", buffering=1) as output,
            ):
                while not self.hardware_stop.is_set():
                    sample = capture_phc_sample(phc, attempts=attempts)
                    self.hardware_samples.append(sample)
                    output.write(json.dumps(sample) + "\n")
                    next_at += interval_s
                    wait_s = next_at - time.monotonic()
                    if wait_s > 0:
                        self.hardware_stop.wait(wait_s)
        except (OSError, RuntimeError, ValueError) as error:
            self.hardware_error = repr(error)

    def stop_hardware_and_build_model(self) -> dict[str, Any]:
        """Stop local PHC sampling and build the node hardware model."""
        if (
            self.hardware_config is None
            or self.hardware_role is None
            or self.hardware_thread is None
        ):
            raise RuntimeError("Hardware sampling is not active")
        self.hardware_stop.set()
        self.hardware_thread.join(timeout=10)
        if self.hardware_thread.is_alive():
            raise RuntimeError("Hardware sampling thread did not stop")
        if self.hardware_error is not None:
            raise RuntimeError(self.hardware_error)
        text = Path(self.hardware_config["ptp_log"]).read_text(encoding="utf-8")
        selected = HardwareModelConfig(
            **self.hardware_config.get("model_config", {})
        )
        health = evaluate_ptp_health(
            text,
            role=self.hardware_role,
            max_offset_p95_ns=selected.max_ptp_offset_p95_ns,
        )
        completed_monotonic_ns = time.monotonic_ns()
        model = build_hardware_model(
            self.hardware_samples,
            role=self.hardware_role,
            ptp_health=health,
            source={
                **self.identity,
                "interface": self.hardware_config["interface"],
                "phc_device": self.hardware_config["phc_device"],
                "hardware_timestamping": self.hardware_config[
                    "hardware_timestamping"
                ],
            },
            config=selected,
            collection={
                "started_at": self.hardware_started_at,
                "completed_at": _utc_now(),
                "started_monotonic_ns": self.hardware_started_monotonic_ns,
                "ended_monotonic_ns": completed_monotonic_ns,
                "interval_ms": self.hardware_config["interval_ms"],
                "successful_sample_count": len(self.hardware_samples),
                "raw_samples_path": str(self._node_output_path(".phc.jsonl")),
                "ptp_log": self.hardware_config["ptp_log"],
            },
        )
        model["model_path_on_node"] = self._persist_model(model)
        return model

    def stop(self) -> None:
        """Release reference resources owned by this actor."""
        if self.collector is not None and self.collector.status()["running"]:
            self.collector.stop()
        if self.server is not None:
            self.server.stop()
        if self.hardware_thread is not None and self.hardware_thread.is_alive():
            self.hardware_stop.set()
            self.hardware_thread.join(timeout=2)


def _make_remote_actor(ray_module: Any, num_cpus: float) -> Any:
    return ray_module.remote(num_cpus=num_cpus)(ClockNodeAgent)


def _select_network_path(  # pylint: disable=too-many-arguments
    *,
    ray_module: Any,
    reference_actor: Any,
    actors: dict[str, Any],
    nodes: list[RayNode],
    head: RayNode,
    requested_host: str | None,
    requested_interface: str | None,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    """Select a Head candidate reachable through capable worker interfaces."""
    strict_selection = bool(requested_host or requested_interface)
    preferred_address = (
        requested_host
        if strict_selection
        else head.address
    )
    candidates = ray_module.get(
        reference_actor.reference_candidates.remote(
            preferred_address,
            requested_interface,
            strict_selection,
        )
    )
    if not candidates:
        raise RuntimeError(
            "Ray Head has no UP, non-loopback IPv4 interface with both "
            "software-transmit and software-receive timestamping"
        )

    worker_nodes = [node for node in nodes if not node.is_head]
    diagnostics: list[dict[str, Any]] = []
    for candidate in candidates:
        route_futures = {
            node.node_id: actors[node.node_id].inspect_route.remote(
                candidate["address"]
            )
            for node in worker_nodes
        }
        routes = {
            node_id: ray_module.get(future)
            for node_id, future in route_futures.items()
        }
        diagnostics.append(
            {"candidate": candidate, "worker_routes": routes}
        )
        if all(route["usable"] for route in routes.values()):
            return candidate, routes, diagnostics

    raise RuntimeError(
        "No Head interface provides a software-timestamp-capable route from "
        "every worker: "
        f"{json.dumps(diagnostics, sort_keys=True)}"
    )


class ClockCalibrationCoordinator:  # pylint: disable=too-many-instance-attributes
    """Detached actor coordinating a manually stopped calibration session."""

    def __init__(
        self,
        *,
        nodes: list[dict[str, Any]],
        head: dict[str, Any],
        session_id: str,
        raw_output_root: str,
        port: int,
        interval_ms: float,
        model_config: dict[str, Any],
        reference_host: str | None,
        reference_interface: str | None,
        agent_cpus: float,
        mode: str,
        hardware_config: dict[str, Any] | None,
    ):  # pylint: disable=too-many-arguments
        self.nodes = [RayNode(**node) for node in nodes]
        self.head = RayNode(**head)
        self.session_id = session_id
        self.raw_output_root = raw_output_root
        self.port = port
        self.interval_ms = interval_ms
        self.model_config = model_config
        self.reference_host = reference_host
        self.reference_interface = reference_interface
        self.agent_cpus = agent_cpus
        self.requested_mode = mode
        self.hardware_config = hardware_config
        self.selected_mode: str | None = None
        self.fallback_reasons: list[dict[str, Any]] = []
        self.actors: dict[str, Any] = {}
        self.selected_candidate: dict[str, Any] | None = None
        self.selected_routes: dict[str, dict[str, Any]] = {}
        self.candidate_diagnostics: list[dict[str, Any]] = []
        self.started_at: str | None = None
        self.started_monotonic_ns: int | None = None
        self.state = "CREATED"
        self.failure: str | None = None

    def start(self) -> dict[str, Any]:
        """Create per-node actors and begin continuous sampling."""
        # pylint: disable=import-outside-toplevel
        if self.state != "CREATED":
            raise RuntimeError(f"Cannot start coordinator in state {self.state}")
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        self.state = "STARTING"
        remote_actor = _make_remote_actor(ray, self.agent_cpus)
        try:
            for node in sorted(self.nodes, key=lambda item: not item.is_head):
                self.actors[node.node_id] = remote_actor.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node.node_id,
                        soft=False,
                    ),
                ).remote(
                    node=node.to_dict(),
                    reference=self.head.to_dict(),
                    session_id=self.session_id,
                    raw_output_root=self.raw_output_root,
                )

            if self.requested_mode != "software":
                if self.hardware_config is None:
                    preflight = [
                        {"usable": False, "reason": "hardware config is missing"}
                    ]
                else:
                    preflight = ray.get(
                        [
                            actor.hardware_preflight.remote(self.hardware_config)
                            for actor in self.actors.values()
                        ]
                    )
                self.selected_mode, self.fallback_reasons = _select_probe_mode(
                    self.requested_mode,
                    preflight,
                )
                if self.selected_mode == "hardware":
                    ray.get(
                        [
                            actor.start_hardware_sampling.remote()
                            for actor in self.actors.values()
                        ]
                    )
                    self.started_at = _utc_now()
                    self.started_monotonic_ns = time.monotonic_ns()
                    self.state = "RUNNING"
                    return self.status()

            self.selected_mode = "software"
            reference_actor = self.actors[self.head.node_id]
            (
                self.selected_candidate,
                self.selected_routes,
                self.candidate_diagnostics,
            ) = _select_network_path(
                ray_module=ray,
                reference_actor=reference_actor,
                actors=self.actors,
                nodes=self.nodes,
                head=self.head,
                requested_host=self.reference_host,
                requested_interface=self.reference_interface,
            )
            reference_status = ray.get(
                reference_actor.start_reference.remote(
                    self.selected_candidate,
                    self.port,
                )
            )
            if not reference_status["server"]["running"]:
                raise RuntimeError("Clock reference server failed to start")

            worker_nodes = [node for node in self.nodes if not node.is_head]
            configure_futures = [
                self.actors[node.node_id].configure_reference.remote(
                    self.selected_candidate["address"],
                    self.port,
                    self.selected_routes[node.node_id],
                )
                for node in worker_nodes
            ]
            if configure_futures:
                ray.get(configure_futures)
            sampling_futures = [
                self.actors[node.node_id].start_sampling.remote(
                    interval_ms=self.interval_ms,
                    model_config=self.model_config,
                )
                for node in worker_nodes
            ]
            if sampling_futures:
                ray.get(sampling_futures)

            self.started_at = _utc_now()
            self.started_monotonic_ns = time.monotonic_ns()
            self.state = "RUNNING"
            return self.status()
        except (
            OSError,
            RuntimeError,
            ValueError,
            ray.exceptions.RayError,
        ) as error:
            self.state = "FAILED"
            self.failure = repr(error)
            self._cleanup_actors(ray)
            raise

    def status(self) -> dict[str, Any]:
        """Return coordinator and per-node live collection state."""
        # pylint: disable=import-outside-toplevel
        import ray

        actor_statuses: dict[str, Any] = {}
        if self.actors:
            futures = {
                node_id: actor.status.remote()
                for node_id, actor in self.actors.items()
            }
            for node_id, future in futures.items():
                try:
                    actor_statuses[node_id] = ray.get(future)
                except ray.exceptions.RayError as error:
                    actor_statuses[node_id] = {"error": repr(error)}

        elapsed_seconds = 0.0
        if self.started_monotonic_ns is not None:
            elapsed_seconds = (
                time.monotonic_ns() - self.started_monotonic_ns
            ) / 1_000_000_000
        return {
            "session_id": self.session_id,
            "state": self.state,
            "started_at": self.started_at,
            "elapsed_seconds": elapsed_seconds,
            "mode": self.selected_mode,
            "requested_mode": self.requested_mode,
            "fallback_reasons": self.fallback_reasons,
            "reference": self.selected_candidate,
            "nodes": actor_statuses,
            "failure": self.failure,
        }

    def stop_and_build(self) -> dict[str, Any]:
        """Stop workers, build all models, and clean up node actors."""
        # pylint: disable=import-outside-toplevel
        if self.state != "RUNNING":
            raise RuntimeError(f"Cannot stop coordinator in state {self.state}")
        import ray

        self.state = "STOPPING"
        if self.selected_mode == "hardware":
            futures = [
                (
                    node,
                    self.actors[node.node_id].stop_hardware_and_build_model.remote(),
                )
                for node in self.nodes
            ]
            models: list[dict[str, Any]] = []
            failures: list[dict[str, Any]] = []
            for node, future in futures:
                try:
                    models.append(ray.get(future))
                except ray.exceptions.RayError as error:
                    failures.append(
                        {"node": node.to_dict(), "error": repr(error)}
                    )
            self.state = "STOPPED"
            if failures:
                session = {
                    "schema_version": 3,
                    "clock_source": "ptp_hardware",
                    "session_id": self.session_id,
                    "status": "FAIL",
                    "models": models,
                    "failures": failures,
                }
            else:
                session = build_hardware_session(
                    models,
                    session_id=self.session_id,
                )
                session["execution"] = {
                    "requested_mode": self.requested_mode,
                    "selected_mode": self.selected_mode,
                    "fallback_reasons": self.fallback_reasons,
                }
            self._cleanup_actors(ray)
            return session

        worker_nodes = [node for node in self.nodes if not node.is_head]
        futures = [
            (
                node,
                self.actors[node.node_id].stop_sampling_and_build_model.remote(),
            )
            for node in worker_nodes
        ]
        models: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for node, future in futures:
            try:
                models.append(ray.get(future))
            except ray.exceptions.RayError as error:
                failures.append(
                    {"node": node.to_dict(), "error": repr(error)}
                )

        reference_actor = self.actors[self.head.node_id]
        try:
            models.insert(0, ray.get(reference_actor.identity_model.remote()))
        except ray.exceptions.RayError as error:
            failures.append(
                {"node": self.head.to_dict(), "error": repr(error)}
            )

        self.state = "STOPPED"
        session = self._build_session(models, failures)
        stop_futures = [actor.stop.remote() for actor in self.actors.values()]
        if stop_futures:
            try:
                ray.get(stop_futures, timeout=10)
            except ray.exceptions.RayError:
                pass
        self._cleanup_actors(ray)
        return session

    def _build_session(
        self,
        models: list[dict[str, Any]],
        failures: list[dict[str, Any]],
    ) -> dict[str, Any]:
        failed_models = [
            model for model in models if model.get("status") != "PASS"
        ]
        return {
            "schema_version": 2,
            "clock_source": "udp_software",
            "session_id": self.session_id,
            "started_at": self.started_at,
            "completed_at": _utc_now(),
            "target_base_time_ns": _trace_base_time_ns(),
            "status": (
                "PASS" if not failures and not failed_models else "FAIL"
            ),
            "reference": {
                **self.head.to_dict(),
                "clock_reference_address": self.selected_candidate["address"],
                "clock_reference_interface": self.selected_candidate["interface"],
                "clock_reference_port": self.port,
                "roles": ["ray_head", "ray_worker", "clock_reference"],
            },
            "network_selection": {
                "selected_candidate": self.selected_candidate,
                "worker_routes": self.selected_routes,
                "candidates_checked": self.candidate_diagnostics,
            },
            "node_count": len(self.nodes),
            "worker_model_count": max(0, len(models) - 1),
            "nodes": [node.to_dict() for node in self.nodes],
            "models": models,
            "failures": failures,
            "execution": {
                "requested_mode": self.requested_mode,
                "selected_mode": self.selected_mode,
                "fallback_reasons": self.fallback_reasons,
            },
        }

    def _cleanup_actors(self, ray_module: Any) -> None:
        for actor in self.actors.values():
            ray_module.kill(actor, no_restart=True)
        self.actors.clear()


def _runtime_env(working_dir: str | None) -> dict[str, Any] | None:
    if not working_dir:
        return None
    return {
        "working_dir": str(Path(working_dir).resolve()),
        "excludes": [
            "/.git/",
            "/.venv/",
            "**/__pycache__/",
            "*.pyc",
            "*.jsonl",
            "*.log",
            "clock-session*.json",
        ],
    }


def _import_ray() -> Any:
    try:
        import ray  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise RuntimeError(
            "Ray is required. Install this project with the 'ray' extra or "
            "run inside vLLM's Ray environment."
        ) from error
    return ray


def _connect_ray(args: Any, *, upload_code: bool) -> Any:
    ray = _import_ray()
    ray.init(
        address=args.ray_address,
        ignore_reinit_error=True,
        namespace=COORDINATOR_NAMESPACE,
        runtime_env=(
            _runtime_env(args.working_dir)
            if upload_code
            else None
        ),
    )
    return ray


def _model_config_from_args(args: Any) -> ModelConfig:
    config = ModelConfig(
        window_seconds=args.window_seconds,
        samples_per_window=args.samples_per_window,
        rtt_slack_us=args.rtt_slack_us,
        max_window_rtt_excess_us=args.max_window_rtt_excess_us,
        segment_seconds=args.segment_seconds,
        max_validation_p95_us=args.max_validation_p95_us,
        model_method=args.model_method,
        candidate_window_seconds=tuple(args.candidate_window_seconds),
        candidate_samples_per_window=tuple(args.candidate_samples_per_window),
        candidate_rtt_slack_us=tuple(args.candidate_rtt_slack_us),
        candidate_segment_seconds=tuple(args.candidate_segment_seconds),
        tuning_fraction=args.tuning_fraction,
    )
    config.validate()
    return config


def _get_active_coordinator(ray_module: Any) -> Any:
    try:
        return ray_module.get_actor(
            COORDINATOR_NAME,
            namespace=COORDINATOR_NAMESPACE,
        )
    except ValueError as error:
        raise RuntimeError("No active Clock Probe session") from error


def start_manual_calibration(args: Any) -> dict[str, Any]:
    """Create one detached coordinator and begin continuous calibration."""
    # pylint: disable=import-outside-toplevel
    ray = _connect_ray(args, upload_code=True)
    try:
        try:
            existing = ray.get_actor(
                COORDINATOR_NAME,
                namespace=COORDINATOR_NAMESPACE,
            )
        except ValueError:
            existing = None
        if existing is not None:
            status = ray.get(existing.status.remote())
            raise RuntimeError(
                "A Clock Probe session is already active: "
                f"{status['session_id']} ({status['state']})"
            )

        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        nodes = discover_alive_nodes(ray.nodes())
        head = get_head_node(nodes)
        session_id = (
            args.session_id
            or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        )
        remote_coordinator = ray.remote(num_cpus=0)(
            ClockCalibrationCoordinator
        )
        coordinator = remote_coordinator.options(
            name=COORDINATOR_NAME,
            lifetime="detached",
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=head.node_id,
                soft=False,
            ),
        ).remote(
            nodes=[node.to_dict() for node in nodes],
            head=head.to_dict(),
            session_id=session_id,
            raw_output_root=args.raw_output_root,
            port=args.port,
            interval_ms=args.interval_ms,
            model_config=asdict(_model_config_from_args(args)),
            reference_host=args.reference_host,
            reference_interface=args.reference_interface,
            agent_cpus=args.agent_cpus,
            mode=args.mode,
            hardware_config={
                "interface": args.hardware_interface,
                "phc_device": args.hardware_phc_device,
                "ptp_logs": dict(args.hardware_ptp_logs),
                "interval_ms": args.hardware_interval_ms,
                "model_config": dict(args.hardware_model_config),
            },
        )
        try:
            return ray.get(coordinator.start.remote())
        except ray.exceptions.RayError:
            ray.kill(coordinator, no_restart=True)
            raise
    finally:
        ray.shutdown()
def get_manual_calibration_status(args: Any) -> dict[str, Any]:
    """Return the active detached calibration status."""
    ray = _connect_ray(args, upload_code=False)
    try:
        coordinator = _get_active_coordinator(ray)
        return ray.get(coordinator.status.remote())
    finally:
        ray.shutdown()

def stop_manual_calibration(args: Any) -> dict[str, Any]:
    """Stop, build, persist, and remove the detached calibration."""
    ray = _connect_ray(args, upload_code=False)
    try:
        coordinator = _get_active_coordinator(ray)
        session = ray.get(coordinator.stop_and_build.remote())
        try:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(session, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        finally:
            ray.kill(coordinator, no_restart=True)
        return session
    finally:
        ray.shutdown()


@dataclass(frozen=True)
class ProbeConfig:  # pylint: disable=too-many-instance-attributes
    """Hardware-first cluster probe configuration."""

    ray_address: str = "auto"
    mode: str = "auto"
    hardware_interface: str = "enp196s0f1np1"
    hardware_phc_device: str = "/dev/ptp3"
    hardware_ptp_logs: dict[str, str] = field(default_factory=dict)
    hardware_interval_ms: float = 50.0
    hardware_model_config: dict[str, Any] = field(default_factory=dict)
    reference_host: str | None = None
    reference_interface: str | None = None
    port: int = 31990
    interval_ms: float = 100.0
    window_seconds: float = 20.0
    samples_per_window: int = 2
    rtt_slack_us: float = 20.0
    max_window_rtt_excess_us: float = 50.0
    segment_seconds: float = 30.0
    max_validation_p95_us: float = 20.0
    model_method: str = "auto"
    candidate_window_seconds: tuple[float, ...] = (5, 10, 15, 20, 30)
    candidate_samples_per_window: tuple[int, ...] = (1, 2, 3)
    candidate_rtt_slack_us: tuple[float, ...] = (10, 20)
    candidate_segment_seconds: tuple[float, ...] = (15, 30, 60)
    tuning_fraction: float = 0.6
    agent_cpus: float = 0.0
    working_dir: str | None = None
    raw_output_root: str = "/tmp/clock-probe"
    session_id: str | None = None
    duration_seconds: float = 120.0
    output: str = "clock-session.json"


class ProbeRun:
    """Handle to one detached hardware-first calibration session."""

    def __init__(self, config: ProbeConfig):
        self.config = config

    def status(self) -> dict[str, Any]:
        return get_manual_calibration_status(self.config)

    def stop(self, output: str | Path | None = None) -> dict[str, Any]:
        target = str(output or self.config.output)
        options = _StopOptions(self.config.ray_address, target)
        return stop_manual_calibration(options)


@dataclass(frozen=True)
class _StopOptions:
    ray_address: str
    output: str


def start_session(config: ProbeConfig | None = None) -> ProbeRun:
    """Start hardware when preflight passes, otherwise start software."""
    selected = config or ProbeConfig()
    if selected.mode not in {"auto", "hardware", "software"}:
        raise ValueError("Probe mode must be auto, hardware, or software")
    start_manual_calibration(selected)
    return ProbeRun(selected)


def run_session(config: ProbeConfig | None = None) -> dict[str, Any]:
    """Run a fixed-duration calibration through the same detached lifecycle."""
    selected = config or ProbeConfig()
    run = start_session(selected)
    try:
        time.sleep(selected.duration_seconds)
    finally:
        session = run.stop(selected.output)
    return session


# Import compatibility for embedding code written before the unified probe API.
SoftwareConfig = ProbeConfig
SoftwareRun = ProbeRun
