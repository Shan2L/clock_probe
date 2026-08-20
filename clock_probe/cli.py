"""Single thin command-line adapter over the public Python API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .api import (
    ProbeConfig,
    ProbeRun,
    TraceInput,
    probe,
    process_traces,
)
from .sampling.phc import PhcClock, assert_phc_matches_interface, capture_phc_sample


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _emit(payload: Any, output: str | Path | None = None) -> None:
    serializable = payload.to_dict() if hasattr(payload, "to_dict") else payload
    text = json.dumps(serializable, indent=2, sort_keys=True)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def _probe(args: argparse.Namespace) -> int:
    config = ProbeConfig(**_load_json(args.config))
    if args.command == "start":
        run = probe.start(config)
        _emit(run.status())
    elif args.command == "status":
        _emit(ProbeRun(config).status())
    elif args.command == "stop":
        _emit(ProbeRun(config).stop(args.output))
    else:
        _emit(probe.run(config), args.output)
    return 0


def _process(args: argparse.Namespace) -> int:
    spec = _load_json(args.spec)
    traces = {
        int(rank): TraceInput(
            path=Path(item["path"]),
            source_node=str(item["source_node"]),
            boot_id=item.get("boot_id"),
        )
        for rank, item in spec["traces"].items()
    }
    manifest = process_traces(
        traces,
        spec["session"],
        spec["output_dir"],
        apply_clc_on_warning=bool(spec.get("apply_clc", False)),
    )
    _emit(manifest)
    return 0 if manifest.status != "FAIL" else 2


def _inspect(args: argparse.Namespace) -> int:
    hardware_info = assert_phc_matches_interface(args.interface, args.phc)
    with PhcClock(args.phc) as phc:
        payload = {
            "interface": args.interface,
            "phc_device": args.phc,
            "hardware_timestamping": hardware_info.to_dict(),
            "capture_method": phc.capture_method(),
            "sample": capture_phc_sample(phc),
        }
    _emit(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clock-probe")
    commands = parser.add_subparsers(dest="command", required=True)

    for action in ("start", "status", "stop", "run"):
        action_parser = commands.add_parser(action)
        action_parser.add_argument("--config", help="ProbeConfig JSON")
        action_parser.add_argument("--output", default="clock-session.json")

    process_parser = commands.add_parser("process")
    process_parser.add_argument("--spec", required=True)

    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("--interface", required=True)
    inspect_parser.add_argument("--phc", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command in {"start", "status", "stop", "run"}:
            code = _probe(args)
        elif args.command == "process":
            code = _process(args)
        else:
            code = _inspect(args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"clock-probe failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    raise SystemExit(code)
