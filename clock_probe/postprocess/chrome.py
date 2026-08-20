"""Streaming helpers for pretty-printed PyTorch Chrome Traces."""

from __future__ import annotations

import json
import re
from ast import literal_eval
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator

BASE_TIME_PATTERN = re.compile(
    r'(?P<prefix>"baseTimeNanoseconds"\s*:\s*)(?P<value>-?\d+)'
)
TIMESTAMP_PATTERN = re.compile(
    r'(?P<prefix>"ts"\s*:\s*)'
    r'(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
)
def microseconds_to_ns(value: str | float | int) -> int:
    """Parse Chrome Trace microsecond values to nanoseconds."""
    if isinstance(value, int):
        return value * 1_000
    if isinstance(value, float):
        return round(value * 1_000)
    if "e" in value.lower():
        return round(Decimal(value) * 1_000)

    negative = value.startswith("-")
    unsigned = value[1:] if negative else value
    whole, separator, fraction = unsigned.partition(".")
    whole_ns = int(whole or "0") * 1_000
    fraction_digits = (fraction + "000")[:3] if separator else "000"
    result = whole_ns + int(fraction_digits)
    if separator and len(fraction) > 3 and fraction[3] >= "5":
        result += 1
    return -result if negative else result


def ns_to_microseconds(value_ns: int) -> str:
    """Format nanoseconds as the fixed-point microseconds PyTorch emits."""
    sign = "-" if value_ns < 0 else ""
    whole, fraction = divmod(abs(value_ns), 1_000)
    return f"{sign}{whole}.{fraction:03d}"


def parse_rank_list(value: Any) -> tuple[int, ...] | None:
    """Parse Process Group Ranks from a list, tuple, or string."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = literal_eval(stripped)
        if isinstance(parsed, (list, tuple)):
            return tuple(int(item) for item in parsed)
    raise ValueError(f"Cannot parse process-group ranks from {value!r}")


def iter_event_blocks(path: Path) -> Iterator[tuple[str, dict[str, Any] | None]]:
    """Yield pretty-printed Trace event blocks without loading the file."""
    in_array = False
    depth = 0
    block: list[str] = []
    with path.open("r", encoding="utf-8", buffering=1024 * 1024) as handle:
        for line in handle:
            if not in_array:
                if '"traceEvents"' not in line:
                    continue
                in_array = True
                remainder = line.split("[", 1)[1] if "[" in line else ""
                if remainder.strip():
                    line = remainder
                else:
                    continue
            if depth == 0 and "]" in line and "{" not in line:
                return
            if depth == 0 and "{" not in line:
                continue
            depth += line.count("{") - line.count("}")
            block.append(line)
            if depth == 0:
                text = "".join(block)
                block = []
                stripped = text.strip()
                if stripped.endswith(","):
                    stripped = stripped[:-1]
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    event = None
                if isinstance(event, dict) or event is None:
                    yield text, event if isinstance(event, dict) else None


def iter_parsed_events(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed event objects, skipping unreadable blocks."""
    for _text, event in iter_event_blocks(path):
        if event is not None:
            yield event


def replace_timestamp(event_text: str, timestamp_ns: int) -> str:
    """Replace the first event-level ``ts`` in a pretty-printed block."""
    replacement = ns_to_microseconds(timestamp_ns)
    updated, count = TIMESTAMP_PATTERN.subn(
        lambda match: match.group("prefix") + replacement,
        event_text,
        count=1,
    )
    if count != 1:
        raise ValueError("Event block has no timestamp to rewrite")
    return updated


def is_nccl_kernel(event: dict[str, Any]) -> bool:
    """Return True for GPU kernels whose name is NCCL or RCCL."""
    if event.get("ph") != "X" or event.get("cat") != "kernel":
        return False
    name = str(event.get("name", "")).lower()
    return "nccl" in name or "rccl" in name
