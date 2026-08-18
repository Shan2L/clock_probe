#!/usr/bin/env python3
"""Collect samples from a remote reference and fit one worker model."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

from clock_probe.clock_bridge import build_clock_bridge, read_boot_id
from clock_probe.model import ModelConfig, build_piecewise_model
from clock_probe.probe import TimestampProbeClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-host", required=True)
    parser.add_argument("--reference-port", type=int, default=31990)
    parser.add_argument("--source-host", required=True)
    parser.add_argument("--duration-seconds", type=float, default=90.0)
    parser.add_argument("--interval-ms", type=float, default=100.0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/pair-probe.jsonl"))
    parser.add_argument("--model-output", type=Path, default=Path("/tmp/pair-probe-model.json"))
    args = parser.parse_args()

    with TimestampProbeClient(
        args.reference_host,
        args.reference_port,
        source_host=args.source_host,
    ) as client:
        samples, errors = client.collect(
            duration_seconds=args.duration_seconds,
            interval_ms=args.interval_ms,
            output_path=args.output,
        )

    model = build_piecewise_model(
        samples,
        source={
            "hostname": socket.gethostname(),
            "boot_id": read_boot_id(),
        },
        reference={"hostname": args.reference_host},
        config=ModelConfig(),
    )
    model["schema_version"] = 2
    model["realtime_monotonic_bridge"] = build_clock_bridge(
        samples,
        boot_id=read_boot_id(),
    )
    args.model_output.write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    segment_uncertainties = [
        float(segment["uncertainty_us"])
        for segment in model.get("segments", [])
        if segment.get("status") == "PASS"
    ]
    summary = {
        "hostname": socket.gethostname(),
        "sample_count": len(samples),
        "error_count": len(errors),
        "model_status": model.get("status"),
        "segment_count": len(model.get("segments", [])),
        "max_validation_p95_us": max(
            (
                float(segment.get("validation_p95_error_us", 0.0))
                for segment in model.get("segments", [])
            ),
            default=0.0,
        ),
        "max_uncertainty_us": max(segment_uncertainties, default=0.0),
        "bridge_status": model.get("realtime_monotonic_bridge", {}).get("status"),
        "raw_output": str(args.output),
        "model_output": str(args.model_output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
