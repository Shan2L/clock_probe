#!/usr/bin/env python3
"""Run a clock reference UDP server for two-node pair tests."""

from __future__ import annotations

import argparse
import signal
import time

from clock_probe.probe import TimestampProbeServer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=31990)
    args = parser.parse_args()

    server = TimestampProbeServer(args.host, args.port)
    server.start()
    print(f"server_started host={args.host} port={args.port}", flush=True)

    def stop(_signum: int, _frame: object) -> None:
        server.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
