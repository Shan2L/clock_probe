"""Minimal public Python facade for embedding Clock Probe."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .calibration.hardware import (
    HardwareModelConfig,
    build_hardware_model,
    build_hardware_session,
    collect_phc_samples,
    load_phc_samples,
)
from .calibration.ptp_health import load_ptp_health
from .calibration.software import ModelConfig, build_clock_model
from .postprocess.pipeline import ProcessManifest, TraceInput, process_traces
from .sampling.phc import PhcClock, assert_phc_matches_interface
from .execution.ray import ProbeConfig, ProbeRun, run_session, start_session


class ProbeAPI:
    """Hardware-first cluster calibration entry points."""

    @staticmethod
    def start(config: ProbeConfig | None = None) -> ProbeRun:
        return start_session(config)

    @staticmethod
    def run(config: ProbeConfig | None = None) -> dict[str, Any]:
        return run_session(config)

    @staticmethod
    def fit(
        samples: Sequence[dict[str, Any]],
        *,
        source: dict[str, Any],
        reference: dict[str, Any],
        config: ModelConfig | None = None,
        local_bridge_uncertainty_us: float = 0.0,
    ) -> dict[str, Any]:
        return build_clock_model(
            samples,
            source=source,
            reference=reference,
            config=config,
            local_bridge_uncertainty_us=local_bridge_uncertainty_us,
        )


class HardwareAPI:
    """Hardware PHC sampling, fitting, and session entry points."""

    @staticmethod
    def sample(
        *,
        interface: str,
        phc_device: str,
        duration_seconds: float,
        interval_ms: float = 50.0,
        output: str | Path | None = None,
        attempts: int = 5,
    ) -> list[dict[str, Any]]:
        assert_phc_matches_interface(interface, phc_device)
        with PhcClock(phc_device) as phc:
            return collect_phc_samples(
                phc=phc,
                duration_seconds=duration_seconds,
                interval_ms=interval_ms,
                attempts=attempts,
                output_path=None if output is None else Path(output),
            )

    @staticmethod
    def fit(
        samples: Sequence[dict[str, Any]] | str | Path,
        *,
        role: str,
        ptp_log: str | Path,
        source: dict[str, Any] | None = None,
        config: HardwareModelConfig | None = None,
        collection: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = (
            load_phc_samples(Path(samples))
            if isinstance(samples, (str, Path))
            else list(samples)
        )
        selected = config or HardwareModelConfig()
        health = load_ptp_health(
            Path(ptp_log),
            role=role,
            max_offset_p95_ns=selected.max_ptp_offset_p95_ns,
        )
        return build_hardware_model(
            payload,
            role=role,
            ptp_health=health,
            source=source,
            config=selected,
            collection=collection,
        )

    @staticmethod
    def session(
        models: Sequence[dict[str, Any]],
        *,
        session_id: str | None = None,
        path_delay_asymmetry: float = 0.1,
    ) -> dict[str, Any]:
        return build_hardware_session(
            models,
            session_id=session_id,
            path_delay_asymmetry=path_delay_asymmetry,
        )


probe = ProbeAPI()
# Compatibility alias; new integrations should use ``probe``.
software = probe
hardware = HardwareAPI()

__all__ = [
    "HardwareModelConfig",
    "ModelConfig",
    "ProbeConfig",
    "ProbeRun",
    "ProcessManifest",
    "TraceInput",
    "hardware",
    "probe",
    "process_traces",
]
