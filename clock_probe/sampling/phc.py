"""Read a NIC PTP Hardware Clock without steering CLOCK_REALTIME."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CLOCK_REALTIME = 0
CLOCK_MONOTONIC = 1
CLOCKFD = 3
PTP_CLK_MAGIC = ord("=")
PTP_MAX_SAMPLES = 25
PTP_DEVICE_RE = re.compile(r"^ptp(?P<index>\d+)$")
PTP_CLOCK_INDEX_RE = re.compile(r"PTP Hardware Clock:\s*(?P<index>\d+)")


def _ioc(direction: int, ioc_type: int, number: int, size: int) -> int:
    """Linux _IOC encoder for PTP ioctl numbers."""
    return (direction << 30) | (ioc_type << 8) | number | (size << 16)


class Timespec(ctypes.Structure):  # pylint: disable=too-few-public-methods
    """Linux 64-bit timespec."""

    _fields_ = (("tv_sec", ctypes.c_int64), ("tv_nsec", ctypes.c_int64))


class PtpClockTime(ctypes.Structure):  # pylint: disable=too-few-public-methods
    """linux/ptp_clock.h struct ptp_clock_time."""

    _fields_ = (
        ("sec", ctypes.c_int64),
        ("nsec", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    )


class PtpClockCaps(ctypes.Structure):  # pylint: disable=too-few-public-methods
    """linux/ptp_clock.h struct ptp_clock_caps."""

    _fields_ = (
        ("max_adj", ctypes.c_int),
        ("n_alarm", ctypes.c_int),
        ("n_ext_ts", ctypes.c_int),
        ("n_per_out", ctypes.c_int),
        ("pps", ctypes.c_int),
        ("n_pins", ctypes.c_int),
        ("cross_timestamping", ctypes.c_int),
        ("adjust_phase", ctypes.c_int),
        ("max_phase_adj", ctypes.c_int),
        ("rsv", ctypes.c_int * 11),
    )


class PtpSysOffsetTriplet(ctypes.Structure):  # pylint: disable=too-few-public-methods
    """One [sys_before, phc, sys_after] kernel crosstamp."""

    _fields_ = (
        ("sys_before", PtpClockTime),
        ("phc", PtpClockTime),
        ("sys_after", PtpClockTime),
    )


class PtpSysOffsetExtended(ctypes.Structure):  # pylint: disable=too-few-public-methods
    """linux/ptp_clock.h struct ptp_sys_offset_extended."""

    _fields_ = (
        ("n_samples", ctypes.c_uint32),
        ("rsv", ctypes.c_uint32 * 3),
        ("ts", PtpSysOffsetTriplet * PTP_MAX_SAMPLES),
    )


class PtpSysOffsetPrecise(ctypes.Structure):  # pylint: disable=too-few-public-methods
    """linux/ptp_clock.h struct ptp_sys_offset_precise."""

    _fields_ = (
        ("device", PtpClockTime),
        ("sys_realtime", PtpClockTime),
        ("sys_monoraw", PtpClockTime),
        ("rsv", ctypes.c_uint32 * 4),
    )


PTP_CLOCK_GETCAPS = _ioc(2, PTP_CLK_MAGIC, 1, ctypes.sizeof(PtpClockCaps))
PTP_SYS_OFFSET_PRECISE = _ioc(3, PTP_CLK_MAGIC, 8, ctypes.sizeof(PtpSysOffsetPrecise))
PTP_SYS_OFFSET_EXTENDED = _ioc(3, PTP_CLK_MAGIC, 9, ctypes.sizeof(PtpSysOffsetExtended))


def _libc() -> ctypes.CDLL:
    library = ctypes.util.find_library("c")
    if library is None:
        raise RuntimeError("Cannot locate libc for clock_gettime")
    libc = ctypes.CDLL(library, use_errno=True)
    libc.clock_gettime.argtypes = [ctypes.c_int, ctypes.POINTER(Timespec)]
    libc.clock_gettime.restype = ctypes.c_int
    libc.ioctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p]
    libc.ioctl.restype = ctypes.c_int
    return libc


_LIBC = _libc()


def fd_to_clockid(file_descriptor: int) -> int:
    """Convert a PHC file descriptor to a POSIX clock id (FD_TO_CLOCKID)."""
    if file_descriptor < 0:
        raise ValueError("PHC file descriptor must be non-negative")
    return int((~file_descriptor << 3) | CLOCKFD)


def clock_gettime_ns(clock_id: int) -> int:
    """Return one POSIX clock reading in nanoseconds."""
    value = Timespec()
    if _LIBC.clock_gettime(int(clock_id), ctypes.byref(value)) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return int(value.tv_sec) * 1_000_000_000 + int(value.tv_nsec)


def ptp_time_to_ns(value: PtpClockTime) -> int:
    """Convert a PTP ioctl timestamp to nanoseconds."""
    return int(value.sec) * 1_000_000_000 + int(value.nsec)


def sample_from_extended_triplet(
    triplet: PtpSysOffsetTriplet,
) -> dict[str, int | str]:
    """Keep the kernel [sys, phc, sys] sandwich as one bridge sample."""
    before_ns = ptp_time_to_ns(triplet.sys_before)
    after_ns = ptp_time_to_ns(triplet.sys_after)
    phc_ns = ptp_time_to_ns(triplet.phc)
    if after_ns < before_ns:
        raise ValueError("PTP_SYS_OFFSET_EXTENDED returned inverted timestamps")
    realtime_ns = before_ns + (after_ns - before_ns) // 2
    return {
        "bridge_phc_ns": phc_ns,
        "bridge_realtime_ns": realtime_ns,
        "bridge_monotonic_ns": 0,
        "bridge_offset_ns": realtime_ns - phc_ns,
        "bridge_read_span_ns": after_ns - before_ns,
        "capture_method": "extended",
    }


def tightest_extended_sample(
    request: PtpSysOffsetExtended,
) -> dict[str, int | str]:
    """Select the kernel sandwich with the smallest REALTIME span."""
    count = int(request.n_samples)
    if count <= 0 or count > PTP_MAX_SAMPLES:
        raise ValueError(f"Invalid PTP_SYS_OFFSET_EXTENDED n_samples={count}")
    return min(
        (sample_from_extended_triplet(request.ts[index]) for index in range(count)),
        key=lambda sample: int(sample["bridge_read_span_ns"]),
    )


@dataclass(frozen=True)
class HardwareTimestamping:
    """ethtool -T hardware timestamping summary for one interface."""

    interface: str
    hardware_transmit: bool
    hardware_receive: bool
    hardware_raw_clock: bool
    ptp_clock_index: int | None

    @property
    def usable(self) -> bool:
        """Whether the NIC can run hardware ptp4l."""
        return (
            self.hardware_transmit
            and self.hardware_receive
            and self.hardware_raw_clock
            and self.ptp_clock_index is not None
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self) | {"usable": self.usable}


def parse_ethtool_hardware(output: str, interface: str) -> HardwareTimestamping:
    """Parse hardware TX/RX/raw-clock flags and the PHC index."""
    capabilities = {
        line.strip()
        for line in output.splitlines()
        if line.startswith(("\t", " "))
    }
    match = PTP_CLOCK_INDEX_RE.search(output)
    ptp_clock_index = int(match.group("index")) if match else None
    return HardwareTimestamping(
        interface=interface,
        hardware_transmit="hardware-transmit" in capabilities,
        hardware_receive="hardware-receive" in capabilities,
        hardware_raw_clock="hardware-raw-clock" in capabilities,
        ptp_clock_index=ptp_clock_index,
    )


def inspect_interface_hardware(interface: str) -> HardwareTimestamping:
    """Run ethtool -T and parse hardware PTP capabilities."""
    try:
        result = subprocess.run(
            ["ethtool", "-T", interface],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(
            f"Cannot inspect hardware timestamping on {interface}: {error!r}"
        ) from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "ethtool failed"
        raise RuntimeError(
            f"ethtool -T {interface} failed: {detail}"
        )
    return parse_ethtool_hardware(result.stdout, interface)


def phc_device_for_index(index: int) -> Path:
    """Return the character device path for one PHC index."""
    if index < 0:
        raise ValueError("PHC index must be non-negative")
    return Path(f"/dev/ptp{index}")


class PhcClock:
    """Open a PHC for gettime and SYS_OFFSET. Never adjtime CLOCK_REALTIME."""

    def __init__(self, device: str | Path):
        self.device = Path(device)
        if not self.device.exists():
            raise FileNotFoundError(
                f"PHC device {self.device} is missing; pass --device=/dev/ptpN "
                "into the container"
            )
        if not os.access(self.device, os.R_OK):
            raise PermissionError(
                f"PHC device {self.device} is not readable in this container"
            )
        match = PTP_DEVICE_RE.match(self.device.name)
        self.index = int(match.group("index")) if match else None
        self._fd = self._open_phc()
        self.clock_id = fd_to_clockid(self._fd)
        self._method: str | None = None
        self.caps: dict[str, Any] | None = None

    def _open_phc(self) -> int:
        try:
            return os.open(self.device, os.O_RDWR)
        except OSError:
            return os.open(self.device, os.O_RDONLY)

    def _ioctl(self, request: int, payload: ctypes.Structure) -> None:
        if _LIBC.ioctl(self._fd, request, ctypes.byref(payload)) != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))

    def get_caps(self) -> dict[str, Any]:
        """Read PTP_CLOCK_GETCAPS once and cache it."""
        if self.caps is not None:
            return self.caps
        caps = PtpClockCaps()
        self._ioctl(PTP_CLOCK_GETCAPS, caps)
        self.caps = {
            "max_adj": int(caps.max_adj),
            "cross_timestamping": bool(caps.cross_timestamping),
            "pps": bool(caps.pps),
            "n_ext_ts": int(caps.n_ext_ts),
        }
        return self.caps

    def capture_method(self) -> str:
        """Pick the tightest available PHC/system crosstamp method."""
        if self._method is not None:
            return self._method
        try:
            caps = self.get_caps()
        except OSError:
            caps = {"cross_timestamping": False}
        if caps.get("cross_timestamping"):
            try:
                self.capture_precise()
                self._method = "precise"
                return self._method
            except OSError:
                pass
        try:
            self.capture_extended(n_samples=1)
            self._method = "extended"
            return self._method
        except OSError:
            self._method = "userspace"
            return self._method

    def capture_precise(self) -> dict[str, int | str]:
        """Hardware PHC/system crosstamp when the NIC supports it."""
        request = PtpSysOffsetPrecise()
        self._ioctl(PTP_SYS_OFFSET_PRECISE, request)
        phc_ns = ptp_time_to_ns(request.device)
        realtime_ns = ptp_time_to_ns(request.sys_realtime)
        monotonic_ns = ptp_time_to_ns(request.sys_monoraw)
        return {
            "bridge_phc_ns": phc_ns,
            "bridge_realtime_ns": realtime_ns,
            "bridge_monotonic_ns": monotonic_ns,
            "bridge_offset_ns": realtime_ns - phc_ns,
            "bridge_read_span_ns": 0,
            "capture_method": "precise",
        }

    def capture_extended(self, *, n_samples: int = 10) -> dict[str, int | str]:
        """Kernel-space REALTIME/PHC/REALTIME sandwiches; keep the tightest."""
        if n_samples <= 0 or n_samples > PTP_MAX_SAMPLES:
            raise ValueError(
                f"n_samples must be between 1 and {PTP_MAX_SAMPLES}"
            )
        request = PtpSysOffsetExtended()
        request.n_samples = n_samples  # pylint: disable=attribute-defined-outside-init
        self._ioctl(PTP_SYS_OFFSET_EXTENDED, request)
        sample = tightest_extended_sample(request)
        sample["bridge_monotonic_ns"] = clock_gettime_ns(CLOCK_MONOTONIC)
        return sample

    def capture_userspace(self, *, attempts: int) -> dict[str, int | str]:
        """Fallback sandwich using only CLOCK_REALTIME around the PHC read."""
        if attempts <= 0:
            raise ValueError("PHC capture attempts must be positive")
        candidates: list[dict[str, int | str]] = []
        for _ in range(attempts):
            before_ns = clock_gettime_ns(CLOCK_REALTIME)
            phc_ns = clock_gettime_ns(self.clock_id)
            after_ns = clock_gettime_ns(CLOCK_REALTIME)
            span_ns = after_ns - before_ns
            realtime_ns = before_ns + span_ns // 2
            candidates.append(
                {
                    "bridge_phc_ns": phc_ns,
                    "bridge_realtime_ns": realtime_ns,
                    "bridge_monotonic_ns": 0,
                    "bridge_offset_ns": realtime_ns - phc_ns,
                    "bridge_read_span_ns": span_ns,
                    "capture_method": "userspace",
                }
            )
        sample = min(candidates, key=lambda item: int(item["bridge_read_span_ns"]))
        sample["bridge_monotonic_ns"] = clock_gettime_ns(CLOCK_MONOTONIC)
        return sample

    def gettime_ns(self) -> int:
        """Read the NIC hardware clock."""
        return clock_gettime_ns(self.clock_id)

    def close(self) -> None:
        """Close the PHC file descriptor."""
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> "PhcClock":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def capture_phc_sample(
    phc: PhcClock,
    *,
    attempts: int = 10,
) -> dict[str, int | str]:
    """Capture the tightest REALTIME/PHC pair using kernel SYS_OFFSET if possible."""
    if attempts <= 0:
        raise ValueError("PHC capture attempts must be positive")
    method = phc.capture_method()
    if method == "precise":
        return phc.capture_precise()
    if method == "extended":
        return phc.capture_extended(n_samples=min(PTP_MAX_SAMPLES, max(attempts, 8)))
    return phc.capture_userspace(attempts=max(attempts, 11))


def assert_phc_matches_interface(
    interface: str,
    device: str | Path,
) -> HardwareTimestamping:
    """Fail closed when /dev/ptpN is not the NIC's advertised PHC."""
    hardware = inspect_interface_hardware(interface)
    if not hardware.usable:
        raise RuntimeError(
            f"{interface} does not advertise hardware TX/RX/raw PTP timestamping"
        )
    path = Path(device)
    expected = phc_device_for_index(int(hardware.ptp_clock_index))
    if path.resolve() != expected.resolve():
        raise RuntimeError(
            f"{interface} PHC is {expected}, not {path}"
        )
    return hardware
