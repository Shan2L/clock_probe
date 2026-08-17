"""Linux kernel software timestamp helpers."""

from __future__ import annotations

import socket
import struct
import time
from typing import Iterable

SO_TIMESTAMPING = getattr(socket, "SO_TIMESTAMPING", 37)
SOF_TIMESTAMPING_TX_SOFTWARE = 1 << 1
SOF_TIMESTAMPING_RX_SOFTWARE = 1 << 3
SOF_TIMESTAMPING_SOFTWARE = 1 << 4
MSG_ERRQUEUE = getattr(socket, "MSG_ERRQUEUE", 0x2000)

# struct scm_timestamping contains three native struct timespec values.
TIMESTAMP_STRUCT = struct.Struct("@llllll")
ANCILLARY_BUFFER_SIZE = socket.CMSG_SPACE(TIMESTAMP_STRUCT.size)


def extract_timestamp_ns(
    ancillary_data: Iterable[tuple[int, int, bytes]],
) -> int:
    """Return the first non-zero software timestamp from ancillary data."""
    for level, message_type, data in ancillary_data:
        if level != socket.SOL_SOCKET or message_type != SO_TIMESTAMPING:
            continue
        if len(data) < TIMESTAMP_STRUCT.size:
            continue

        values = TIMESTAMP_STRUCT.unpack_from(data)
        seconds, nanoseconds = values[0], values[1]
        if seconds or nanoseconds:
            return seconds * 1_000_000_000 + nanoseconds

    raise RuntimeError(
        "Kernel software timestamp missing; check SO_TIMESTAMPING and NIC support"
    )


def enable_software_timestamping(sock: socket.socket) -> None:
    """Enable software TX and RX timestamps on a UDP socket."""
    flags = (
        SOF_TIMESTAMPING_TX_SOFTWARE
        | SOF_TIMESTAMPING_RX_SOFTWARE
        | SOF_TIMESTAMPING_SOFTWARE
    )
    sock.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPING, flags)


def create_timestamp_socket(
    *,
    bind: tuple[str, int] | None = None,
    timeout_seconds: float = 2.0,
) -> socket.socket:
    """Create a UDP socket configured for kernel software timestamps."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    enable_software_timestamping(sock)
    sock.settimeout(timeout_seconds)
    if bind is not None:
        sock.bind(bind)
    return sock


def read_tx_timestamp_ns(
    sock: socket.socket,
    timeout_seconds: float = 1.0,
) -> int:
    """Read one asynchronous TX timestamp from the socket error queue."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            _, ancillary_data, _, _ = sock.recvmsg(
                2048,
                ANCILLARY_BUFFER_SIZE,
                MSG_ERRQUEUE,
            )
            return extract_timestamp_ns(ancillary_data)
        except (BlockingIOError, TimeoutError, socket.timeout):
            time.sleep(0.001)

    raise TimeoutError("Timed out waiting for kernel TX timestamp")
