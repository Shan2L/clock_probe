"""Session loading, compatibility normalization, and shared lookups."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SessionInput = str | Path | Mapping[str, Any]


def load_session(source: SessionInput) -> dict[str, Any]:
    """Load a current or historical clock session into a canonical mapping."""
    if isinstance(source, Mapping):
        payload = copy.deepcopy(dict(source))
    else:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Clock session must be a JSON object")
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Clock session has no node models")
    payload.setdefault("clock_source", "udp_software")
    if payload["clock_source"] == "ptp_hardware":
        payload.setdefault("timestamp_domain", "PHC")
    else:
        payload.setdefault("timestamp_domain", "CLOCK_MONOTONIC")
    payload.setdefault("status", "FAIL")
    return payload


def model_identifiers(model: Mapping[str, Any]) -> set[str]:
    """Return all supported node identifiers attached to one model."""
    source = model.get("source", {})
    return {
        str(value)
        for key in (
            "hostname",
            "ray_node_name",
            "ray_node_address",
            "ray_node_id",
        )
        if (value := source.get(key)) is not None
    }


def select_model(
    session: Mapping[str, Any],
    source_node: str,
) -> dict[str, Any]:
    """Select exactly one model by hostname, Ray name, address, or node ID."""
    matches = [
        model
        for model in session.get("models", [])
        if source_node in model_identifiers(model)
    ]
    if not matches:
        available = sorted(
            identifier
            for model in session.get("models", [])
            for identifier in model_identifiers(model)
        )
        raise ValueError(
            f"No clock model matches source node {source_node!r}; "
            f"available={available}"
        )
    if len(matches) != 1:
        raise ValueError(
            f"Source node {source_node!r} matches {len(matches)} clock models"
        )
    model = matches[0]
    if model.get("status") != "PASS":
        raise ValueError(
            f"Clock model for {source_node!r} has status {model.get('status')!r}"
        )
    return model


def session_uncertainty_us(source: SessionInput) -> float:
    """Return the conservative end-to-end uncertainty for validation."""
    session = load_session(source)
    if session["clock_source"] == "ptp_hardware":
        values: list[float] = []
        ptp_us = float(session.get("ptp", {}).get("uncertainty_us") or 0.0)
        session_max = session.get("max_uncertainty_us")
        if session_max is not None:
            values.append(float(session_max))
        for model in session["models"]:
            if model.get("status") != "PASS":
                continue
            if model.get("uncertainty_us") is not None:
                values.append(float(model["uncertainty_us"]))
            bridge = model.get("realtime_phc_bridge", {})
            if bridge.get("uncertainty_us") is not None:
                values.append(float(bridge["uncertainty_us"]) + ptp_us)
            for segment in bridge.get("segments", []):
                if segment.get("status") == "PASS":
                    values.append(float(segment["uncertainty_us"]) + ptp_us)
        if not values:
            raise ValueError("Hardware session has no PASS uncertainty")
        return max(values)

    values = []
    for model in session["models"]:
        if model.get("model_type") == "identity" or model.get("status") != "PASS":
            continue
        bridge_values = [
            float(segment["uncertainty_us"])
            for segment in model.get("realtime_monotonic_bridge", {}).get(
                "segments", []
            )
            if segment.get("status") == "PASS"
        ]
        bridge_us = max(bridge_values, default=0.0)
        values.extend(
            float(segment["uncertainty_us"]) + bridge_us
            for segment in model.get("segments", [])
            if segment.get("status") == "PASS"
        )
    if not values:
        raise ValueError("Software session has no PASS uncertainty")
    return max(values)
