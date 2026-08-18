"""One-command Ray cluster clock calibration orchestrator."""

# Ray is an optional runtime dependency available in the vLLM environment.
# pylint: disable=import-error,too-many-lines

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .clock_bridge import build_clock_bridge, read_boot_id
from .model import ModelConfig, build_piecewise_model
from .network import (
    list_network_interfaces,
    reference_candidates,
    route_to,
)
from .probe import (
    ContinuousProbeCollector,
    TimestampProbeClient,
    TimestampProbeServer,
)
from .ray_topology import RayNode, discover_alive_nodes, get_head_node

COORDINATOR_NAME = "clock-probe-coordinator"
COORDINATOR_NAMESPACE = "clock-probe"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trace_base_time_ns() -> int:
    """Return one hour-aligned Unix base shared by aligned Trace files."""
    hour_ns = 3_600_000_000_000
    return time.time_ns() // hour_ns * hour_ns


def _node_identity(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "ray_node_id": node["node_id"],
        "ray_node_name": node["name"],
        "ray_node_address": node["address"],
        "hostname": socket.gethostname(),
        "boot_id": read_boot_id(),
    }


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
            "realtime_monotonic_bridge": {
                "schema_version": 1,
                "model_type": "identity_not_required",
                "boot_id": self.identity["boot_id"],
                "status": "PASS",
                "segments": [],
            },
        }
        model["model_path_on_node"] = self._persist_model(model)
        return model

    def collect_model(
        self,
        *,
        duration_seconds: float,
        interval_ms: float,
        model_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Probe the Head and fit this worker's model locally."""
        if self.server is not None:
            return self.identity_model()
        if (
            self.reference_host is None
            or self.reference_port is None
            or self.selected_route is None
        ):
            raise RuntimeError("Clock reference route has not been configured")

        raw_path = self._node_output_path(".jsonl")
        started_at = _utc_now()
        started_monotonic_ns = time.monotonic_ns()
        with TimestampProbeClient(
            self.reference_host,
            self.reference_port,
            source_host=str(self.selected_route["source_address"]),
        ) as client:
            samples, errors = client.collect(
                duration_seconds=duration_seconds,
                interval_ms=interval_ms,
                output_path=raw_path,
            )
        ended_monotonic_ns = time.monotonic_ns()

        source_identity = {
            **self.identity,
            "clock_probe_interface": self.selected_route["interface"],
            "clock_probe_address": self.selected_route["source_address"],
        }
        model = build_piecewise_model(
            samples,
            source=source_identity,
            reference=self.reference_identity,
            config=ModelConfig(**model_config),
        )
        model["schema_version"] = 2
        model["realtime_monotonic_bridge"] = build_clock_bridge(
            samples,
            boot_id=str(self.identity["boot_id"]),
        )
        model["roles"] = ["ray_worker"]
        model["network"] = self.selected_route
        model["collection"] = {
            "started_at": started_at,
            "started_monotonic_ns": started_monotonic_ns,
            "ended_monotonic_ns": ended_monotonic_ns,
            "duration_seconds": duration_seconds,
            "interval_ms": interval_ms,
            "successful_sample_count": len(samples),
            "failed_sample_count": len(errors),
            "recent_errors": errors[-20:],
            "raw_samples_path_on_node": str(raw_path),
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
        model = build_piecewise_model(
            samples,
            source=source_identity,
            reference=self.reference_identity,
            config=self.continuous_model_config,
        )
        model["schema_version"] = 2
        model["realtime_monotonic_bridge"] = build_clock_bridge(
            samples,
            boot_id=str(self.identity["boot_id"]),
        )
        model["roles"] = ["ray_worker"]
        model["network"] = self.selected_route
        model["collection"] = collection
        model["model_path_on_node"] = self._persist_model(model)
        return model

    def stop(self) -> None:
        """Release reference resources owned by this actor."""
        if self.collector is not None and self.collector.status()["running"]:
            self.collector.stop()
        if self.server is not None:
            self.server.stop()


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
        }

    def _cleanup_actors(self, ray_module: Any) -> None:
        for actor in self.actors.values():
            ray_module.kill(actor, no_restart=True)
        self.actors.clear()


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    """Discover nodes, calibrate workers concurrently, and persist a session."""
    # pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
    try:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    except ImportError as error:
        raise RuntimeError(
            "Ray is required for cluster orchestration. "
            "Install this project with the 'ray' extra or run inside vLLM's "
            "Ray environment."
        ) from error

    runtime_env = None
    if args.working_dir:
        runtime_env = {
            "working_dir": str(Path(args.working_dir).resolve()),
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
    ray.init(
        address=args.ray_address,
        ignore_reinit_error=True,
        runtime_env=runtime_env,
    )

    nodes = discover_alive_nodes(ray.nodes())
    head = get_head_node(nodes)
    session_id = (
        args.session_id
        or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    )
    model_config = ModelConfig(
        window_seconds=args.window_seconds,
        samples_per_window=args.samples_per_window,
        rtt_slack_us=args.rtt_slack_us,
        max_window_rtt_excess_us=args.max_window_rtt_excess_us,
        segment_seconds=args.segment_seconds,
        max_validation_p95_us=args.max_validation_p95_us,
    )
    model_config.validate()

    remote_actor = _make_remote_actor(ray, args.agent_cpus)
    actors: dict[str, Any] = {}
    session_started_at = _utc_now()
    try:
        ordered_nodes = sorted(nodes, key=lambda node: not node.is_head)
        for node in ordered_nodes:
            actors[node.node_id] = remote_actor.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node.node_id,
                    soft=False,
                ),
            ).remote(
                node=node.to_dict(),
                reference=head.to_dict(),
                session_id=session_id,
                raw_output_root=args.raw_output_root,
            )

        reference_actor = actors[head.node_id]
        worker_nodes = [node for node in nodes if not node.is_head]
        (
            selected_candidate,
            selected_routes,
            candidate_diagnostics,
        ) = _select_network_path(
            ray_module=ray,
            reference_actor=reference_actor,
            actors=actors,
            nodes=nodes,
            head=head,
            requested_host=args.reference_host,
            requested_interface=args.reference_interface,
        )

        reference_status = ray.get(
            reference_actor.start_reference.remote(
                selected_candidate,
                args.port,
            )
        )
        if not reference_status["server"]["running"]:
            raise RuntimeError("Clock reference server failed to start")
        configure_futures = [
            actors[node.node_id].configure_reference.remote(
                selected_candidate["address"],
                args.port,
                selected_routes[node.node_id],
            )
            for node in worker_nodes
        ]
        if configure_futures:
            ray.get(configure_futures)

        worker_futures: list[tuple[RayNode, Any]] = []
        for node in worker_nodes:
            future = actors[node.node_id].collect_model.remote(
                duration_seconds=args.duration_seconds,
                interval_ms=args.interval_ms,
                model_config=asdict(model_config),
            )
            worker_futures.append((node, future))

        models = [ray.get(reference_actor.identity_model.remote())]
        failures: list[dict[str, Any]] = []
        # All remote calls above are already running concurrently.
        for node, future in worker_futures:
            try:
                models.append(ray.get(future))
            except ray.exceptions.RayError as error:
                failures.append(
                    {
                        "node": node.to_dict(),
                        "error": repr(error),
                    }
                )

        failed_models = [
            model
            for model in models
            if model.get("status") != "PASS"
        ]
        session = {
            "schema_version": 2,
            "session_id": session_id,
            "started_at": session_started_at,
            "completed_at": _utc_now(),
            "target_base_time_ns": _trace_base_time_ns(),
            "status": (
                "PASS"
                if not failures and not failed_models
                else "FAIL"
            ),
            "reference": {
                **head.to_dict(),
                "clock_reference_address": selected_candidate["address"],
                "clock_reference_interface": selected_candidate["interface"],
                "clock_reference_port": args.port,
                "roles": ["ray_head", "ray_worker", "clock_reference"],
            },
            "network_selection": {
                "selected_candidate": selected_candidate,
                "worker_routes": selected_routes,
                "candidates_checked": candidate_diagnostics,
            },
            "node_count": len(nodes),
            "worker_model_count": len(models) - 1,
            "nodes": [node.to_dict() for node in nodes],
            "models": models,
            "failures": failures,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(session, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return session
    finally:
        stop_futures = [
            actor.stop.remote()
            for actor in actors.values()
        ]
        if stop_futures:
            try:
                ray.get(stop_futures, timeout=10)
            except ray.exceptions.RayError:
                # Preserve the calibration exception if an actor had already
                # failed; healthy actors still receive their stop requests.
                pass
        ray.shutdown()


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


def _connect_ray(args: argparse.Namespace, *, upload_code: bool) -> Any:
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


def _model_config_from_args(args: argparse.Namespace) -> ModelConfig:
    config = ModelConfig(
        window_seconds=args.window_seconds,
        samples_per_window=args.samples_per_window,
        rtt_slack_us=args.rtt_slack_us,
        max_window_rtt_excess_us=args.max_window_rtt_excess_us,
        segment_seconds=args.segment_seconds,
        max_validation_p95_us=args.max_validation_p95_us,
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


def start_manual_calibration(args: argparse.Namespace) -> dict[str, Any]:
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
        )
        try:
            return ray.get(coordinator.start.remote())
        except ray.exceptions.RayError:
            ray.kill(coordinator, no_restart=True)
            raise
    finally:
        ray.shutdown()


def get_manual_calibration_status(args: argparse.Namespace) -> dict[str, Any]:
    """Return the active detached calibration status."""
    ray = _connect_ray(args, upload_code=False)
    try:
        coordinator = _get_active_coordinator(ray)
        return ray.get(coordinator.status.remote())
    finally:
        ray.shutdown()


def stop_manual_calibration(args: argparse.Namespace) -> dict[str, Any]:
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


def _add_ray_connection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ray-address", default="auto")


def _add_calibration_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reference-host",
        help=(
            "Require this Head-node timing address instead of auto-selection."
        ),
    )
    parser.add_argument(
        "--reference-interface",
        help="Require this Head interface instead of auto-selection.",
    )
    parser.add_argument("--port", type=int, default=31990)
    parser.add_argument("--interval-ms", type=float, default=100.0)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--samples-per-window", type=int, default=3)
    parser.add_argument("--rtt-slack-us", type=float, default=20.0)
    parser.add_argument(
        "--max-window-rtt-excess-us",
        type=float,
        default=50.0,
    )
    parser.add_argument("--segment-seconds", type=float, default=30.0)
    parser.add_argument("--max-validation-p95-us", type=float, default=20.0)
    parser.add_argument("--agent-cpus", type=float, default=0.0)
    parser.add_argument(
        "--working-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Directory Ray uploads so workers can import clock_probe.",
    )
    parser.add_argument(
        "--raw-output-root",
        default="/tmp/clock-probe",
        help="Node-local directory for raw JSONL files.",
    )
    parser.add_argument("--session-id")


def build_parser() -> argparse.ArgumentParser:
    """Build run and manual-lifecycle subcommands."""
    parser = argparse.ArgumentParser(
        description=(
            "Build cross-node clock models using an existing Ray cluster."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run one fixed-duration calibration.",
    )
    _add_ray_connection_arguments(run_parser)
    _add_calibration_arguments(run_parser)
    run_parser.add_argument("--duration-seconds", type=float, default=120.0)
    run_parser.add_argument(
        "--output",
        default="clock-session.json",
        help="Driver-local aggregate model JSON path.",
    )

    start_parser = subparsers.add_parser(
        "start",
        help="Start a detached continuous calibration.",
    )
    _add_ray_connection_arguments(start_parser)
    _add_calibration_arguments(start_parser)

    status_parser = subparsers.add_parser(
        "status",
        help="Show the active manual calibration.",
    )
    _add_ray_connection_arguments(status_parser)

    stop_parser = subparsers.add_parser(
        "stop",
        help="Stop the active calibration and build models.",
    )
    _add_ray_connection_arguments(stop_parser)
    stop_parser.add_argument(
        "--output",
        default="clock-session.json",
        help="Driver-local aggregate model JSON path.",
    )
    return parser


def main() -> None:
    """Dispatch fixed-duration and manual-lifecycle commands."""
    arguments = sys.argv[1:]
    commands = {"run", "start", "status", "stop"}
    if not arguments or arguments[0] not in commands:
        # Preserve the original no-subcommand CLI as fixed-duration run mode.
        arguments = ["run", *arguments]
    args = build_parser().parse_args(arguments)
    try:
        if args.command == "run":
            session = run_calibration(args)
        elif args.command == "start":
            status = start_manual_calibration(args)
            print(
                f"session={status['session_id']} state={status['state']} "
                f"reference={status['reference']['interface']}:"
                f"{status['reference']['address']}"
            )
            return
        elif args.command == "status":
            status = get_manual_calibration_status(args)
            print(json.dumps(status, indent=2, sort_keys=True))
            return
        else:
            session = stop_manual_calibration(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Clock calibration failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print(
        f"session={session['session_id']} status={session['status']} "
        f"nodes={session['node_count']} "
        f"worker_models={session['worker_model_count']} "
        f"output={getattr(args, 'output', 'clock-session.json')}"
    )
    if session["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
