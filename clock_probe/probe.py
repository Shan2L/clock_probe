"""Reusable four-timestamp UDP server and client."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from .clock_bridge import capture_clock_pair
from .timestamping import (
    ANCILLARY_BUFFER_SIZE,
    create_timestamp_socket,
    extract_timestamp_ns,
    read_tx_timestamp_ns,
)

MAX_DATAGRAM_SIZE = 2048


def _encode(message: dict[str, Any]) -> bytes:
    return json.dumps(message, separators=(",", ":")).encode("utf-8")


def _decode(data: bytes) -> dict[str, Any]:
    message = json.loads(data.decode("utf-8"))
    if not isinstance(message, dict):
        raise ValueError("Probe message must be a JSON object")
    return message


class TimestampProbeServer:
    """Clock-reference UDP server using kernel RX/TX timestamps."""

    def __init__(self, host: str, port: int, socket_timeout_seconds: float = 0.2):
        self.host = host
        self.port = port
        self._sock = create_timestamp_socket(
            bind=(host, port),
            timeout_seconds=socket_timeout_seconds,
        )
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.requests_handled = 0
        self.last_error: str | None = None

    def start(self) -> None:
        """Start the reference service on a daemon thread."""
        if self._thread is not None:
            raise RuntimeError("Timestamp probe server is already running")
        self._thread = threading.Thread(
            target=self.serve_forever,
            name="clock-reference-udp",
            daemon=True,
        )
        self._thread.start()

    def serve_forever(self) -> None:
        """Serve requests until ``stop`` is called."""
        while not self._stop_event.is_set():
            try:
                data, ancillary_data, _, peer = self._sock.recvmsg(
                    MAX_DATAGRAM_SIZE,
                    ANCILLARY_BUFFER_SIZE,
                )
            except socket.timeout:
                continue
            except OSError as error:
                if self._stop_event.is_set():
                    break
                self.last_error = repr(error)
                continue

            try:
                self._handle_request(data, ancillary_data, peer)
                self.requests_handled += 1
            except (KeyError, ValueError, RuntimeError, TimeoutError) as error:
                self.last_error = repr(error)

    def _handle_request(
        self,
        data: bytes,
        ancillary_data: list[tuple[int, int, bytes]],
        peer: tuple[str, int],
    ) -> None:
        request = _decode(data)
        if request.get("type") != "request":
            raise ValueError("Unexpected probe request type")
        sequence = int(request["sequence"])
        t2_ns = extract_timestamp_ns(ancillary_data)

        self._sock.sendto(
            _encode({"type": "response", "sequence": sequence}),
            peer,
        )
        t3_ns = read_tx_timestamp_ns(self._sock)

        self._sock.sendto(
            _encode(
                {
                    "type": "follow_up",
                    "sequence": sequence,
                    "t2_ns": t2_ns,
                    "t3_ns": t3_ns,
                }
            ),
            peer,
        )
        # Drain the follow-up packet's TX timestamp so it cannot be mistaken
        # for the next response packet.
        read_tx_timestamp_ns(self._sock)

    def status(self) -> dict[str, Any]:
        """Return serializable server health information."""
        return {
            "host": self.host,
            "port": self.port,
            "running": self._thread is not None and self._thread.is_alive(),
            "requests_handled": self.requests_handled,
            "last_error": self.last_error,
        }

    def stop(self) -> None:
        """Stop the service and close its socket."""
        self._stop_event.set()
        self._sock.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class TimestampProbeClient:
    """Collect NTP-style four-timestamp samples from one reference."""

    def __init__(
        self,
        reference_host: str,
        reference_port: int,
        timeout_seconds: float = 2.0,
        source_host: str | None = None,
    ):
        self.reference = (reference_host, reference_port)
        self._sock = create_timestamp_socket(
            bind=(source_host, 0) if source_host is not None else None,
            timeout_seconds=timeout_seconds,
        )
        self.timeout_seconds = timeout_seconds

    def measure_once(self, sequence: int) -> dict[str, int | float]:
        """Perform one four-timestamp exchange."""
        clock_pair = capture_clock_pair()
        sample_monotonic_ns = clock_pair["bridge_monotonic_ns"]
        self._sock.sendto(
            _encode({"type": "request", "sequence": sequence}),
            self.reference,
        )
        t1_ns = read_tx_timestamp_ns(self._sock, self.timeout_seconds)

        t2_ns: int | None = None
        t3_ns: int | None = None
        t4_ns: int | None = None
        deadline = time.monotonic() + self.timeout_seconds

        while time.monotonic() < deadline and (
            t2_ns is None or t3_ns is None or t4_ns is None
        ):
            data, ancillary_data, _, _ = self._sock.recvmsg(
                MAX_DATAGRAM_SIZE,
                ANCILLARY_BUFFER_SIZE,
            )
            message = _decode(data)
            if int(message.get("sequence", -1)) != sequence:
                continue

            if message.get("type") == "response":
                t4_ns = extract_timestamp_ns(ancillary_data)
            elif message.get("type") == "follow_up":
                t2_ns = int(message["t2_ns"])
                t3_ns = int(message["t3_ns"])

        if t2_ns is None or t3_ns is None or t4_ns is None:
            raise TimeoutError(f"Incomplete probe exchange for sequence {sequence}")

        offset_ns = ((t2_ns - t1_ns) + (t3_ns - t4_ns)) / 2.0
        rtt_ns = (t4_ns - t1_ns) - (t3_ns - t2_ns)
        return {
            "sequence": sequence,
            "monotonic_ns": sample_monotonic_ns,
            **clock_pair,
            "t1_ns": t1_ns,
            "t2_ns": t2_ns,
            "t3_ns": t3_ns,
            "t4_ns": t4_ns,
            "offset_ns": offset_ns,
            "rtt_ns": rtt_ns,
        }

    def collect(
        self,
        *,
        duration_seconds: float,
        interval_ms: float,
        output_path: Path | None = None,
    ) -> tuple[list[dict[str, int | float]], list[str]]:
        """Collect until duration expires, retaining errors for health reports."""
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")

        samples: list[dict[str, int | float]] = []
        errors: list[str] = []
        output_file = None
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_file = output_path.open("w", encoding="utf-8", buffering=1)

        interval_seconds = interval_ms / 1_000.0
        started = time.monotonic()
        deadline = started + duration_seconds
        sequence = 1
        next_sample_at = started
        try:
            while time.monotonic() < deadline:
                try:
                    sample = self.measure_once(sequence)
                    samples.append(sample)
                    if output_file is not None:
                        output_file.write(json.dumps(sample) + "\n")
                except (OSError, ValueError, RuntimeError, TimeoutError) as error:
                    errors.append(f"sequence={sequence}: {error!r}")

                sequence += 1
                next_sample_at += interval_seconds
                sleep_seconds = next_sample_at - time.monotonic()
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
        finally:
            if output_file is not None:
                output_file.close()

        return samples, errors

    def close(self) -> None:
        """Close the UDP client socket."""
        self._sock.close()

    def __enter__(self) -> "TimestampProbeClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ContinuousProbeCollector:  # pylint: disable=too-many-instance-attributes
    """Run a probe client in a stoppable background thread."""

    def __init__(
        self,
        *,
        reference_host: str,
        reference_port: int,
        source_host: str,
        interval_ms: float,
        output_path: Path,
        timeout_seconds: float = 2.0,
    ):  # pylint: disable=too-many-arguments
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        self.reference_host = reference_host
        self.reference_port = reference_port
        self.source_host = source_host
        self.interval_ms = interval_ms
        self.output_path = output_path
        self.timeout_seconds = timeout_seconds
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, int | float]] = []
        self._recent_errors: deque[str] = deque(maxlen=20)
        self._failed_sample_count = 0
        self._started_monotonic_ns: int | None = None
        self._ended_monotonic_ns: int | None = None
        self._fatal_error: str | None = None

    def start(self) -> None:
        """Start collecting and return immediately."""
        if self._thread is not None:
            raise RuntimeError("Continuous probe collector is already started")
        self._started_monotonic_ns = time.monotonic_ns()
        self._thread = threading.Thread(
            target=self._run,
            name="clock-probe-collector",
            daemon=True,
        )
        self._thread.start()
        if not self._ready_event.wait(self.timeout_seconds + 1.0):
            raise TimeoutError("Timed out starting continuous probe collector")
        if self._fatal_error is not None:
            raise RuntimeError(
                f"Continuous probe collector failed to start: {self._fatal_error}"
            )

    def _run(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        interval_seconds = self.interval_ms / 1_000.0
        next_sample_at = time.monotonic()
        sequence = 1
        try:
            with (
                self.output_path.open(
                    "w",
                    encoding="utf-8",
                    buffering=1,
                ) as output_file,
                TimestampProbeClient(
                    self.reference_host,
                    self.reference_port,
                    timeout_seconds=self.timeout_seconds,
                    source_host=self.source_host,
                ) as client,
            ):
                self._ready_event.set()
                while not self._stop_event.is_set():
                    try:
                        sample = client.measure_once(sequence)
                        output_file.write(json.dumps(sample) + "\n")
                        with self._lock:
                            self._samples.append(sample)
                    except (
                        OSError,
                        ValueError,
                        RuntimeError,
                        TimeoutError,
                    ) as error:
                        message = f"sequence={sequence}: {error!r}"
                        with self._lock:
                            self._failed_sample_count += 1
                            self._recent_errors.append(message)

                    sequence += 1
                    next_sample_at += interval_seconds
                    wait_seconds = max(0.0, next_sample_at - time.monotonic())
                    self._stop_event.wait(wait_seconds)
        except (
            OSError,
            ValueError,
            RuntimeError,
            TimeoutError,
        ) as error:
            message = f"collector: {error!r}"
            with self._lock:
                self._fatal_error = message
                self._failed_sample_count += 1
                self._recent_errors.append(message)
            self._ready_event.set()
        finally:
            self._ended_monotonic_ns = time.monotonic_ns()
            self._ready_event.set()

    def status(self) -> dict[str, Any]:
        """Return a thread-safe serializable collection snapshot."""
        with self._lock:
            successful_sample_count = len(self._samples)
            failed_sample_count = self._failed_sample_count
            recent_errors = list(self._recent_errors)
            fatal_error = self._fatal_error
        started_ns = self._started_monotonic_ns
        ended_ns = self._ended_monotonic_ns
        current_ns = ended_ns or time.monotonic_ns()
        elapsed_seconds = (
            (current_ns - started_ns) / 1_000_000_000
            if started_ns is not None
            else 0.0
        )
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "started_monotonic_ns": started_ns,
            "ended_monotonic_ns": ended_ns,
            "elapsed_seconds": elapsed_seconds,
            "interval_ms": self.interval_ms,
            "successful_sample_count": successful_sample_count,
            "failed_sample_count": failed_sample_count,
            "recent_errors": recent_errors,
            "fatal_error": fatal_error,
            "raw_samples_path_on_node": str(self.output_path),
        }

    def stop(
        self,
        timeout_seconds: float = 5.0,
    ) -> tuple[list[dict[str, int | float]], dict[str, Any]]:
        """Stop collecting and return samples plus final statistics."""
        if self._thread is None:
            raise RuntimeError("Continuous probe collector was not started")
        self._stop_event.set()
        self._thread.join(timeout=timeout_seconds)
        if self._thread.is_alive():
            raise TimeoutError("Timed out stopping continuous probe collector")
        with self._lock:
            samples = list(self._samples)
        return samples, self.status()
