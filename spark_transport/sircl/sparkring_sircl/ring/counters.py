"""RDMA and Ethernet error counters of one Spark, read before and after a run.

Two sources, both read-only:

- sysfs: every file under ``/sys/class/infiniband/<device>/ports/1/hw_counters``
  and ``.../counters`` of the Spark's RDMA devices (``out_of_buffer``,
  ``out_of_sequence``, ``packet_seq_err``, ``roce_adp_retrans``,
  ``local_ack_timeout_err`` and the others the driver exposes);
- the Ethernet statistics of each device's network interface through the
  ``SIOCETHTOOL`` ioctl (the ``ethtool -S`` view), which hold
  ``rx_out_of_buffer`` and, where the driver has one, a hairpin drop counter.
  When the ioctl is unavailable, ``ethtool -S`` is tried; a missing source is
  recorded as such, never guessed.

:func:`key_deltas` keeps the counters that indicate drops, retransmissions or
transport errors, plus every counter whose name mentions ``hairpin``.
"""

from __future__ import annotations

import ctypes
import socket
import struct
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

SYSFS = Path("/sys/class/infiniband")
KEY_COUNTERS = (
    "rx_out_of_buffer", "out_of_buffer", "out_of_sequence", "packet_seq_err", "roce_adp_retrans",
    "roce_adp_retrans_to", "local_ack_timeout_err", "implied_nak_seq_err", "rnr_nak_retry_err",
    "req_transport_retries_exceeded", "req_cqe_error", "resp_cqe_error", "duplicate_request",
    "rx_discards_phy", "rx_crc_errors_phy", "port_rcv_errors", "port_xmit_discards",
)
_SIOCETHTOOL = 0x8946
_ETHTOOL_GSSET_INFO = 0x37
_ETHTOOL_GSTRINGS = 0x1B
_ETHTOOL_GSTATS = 0x1D
_ETH_SS_STATS = 1
_ETH_GSTRING_LEN = 32


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def rdma_counters(device: str, root: Path = SYSFS) -> dict[str, int]:
    values: dict[str, int] = {}
    for directory in ("hw_counters", "counters"):
        folder = root / device / "ports" / "1" / directory
        if not folder.is_dir():
            continue
        for entry in sorted(folder.iterdir()):
            value = _read_int(entry)
            if value is not None:
                values[f"{directory}/{entry.name}"] = value
    return values


def netdev_of(device: str, root: Path = SYSFS) -> str | None:
    folder = root / device / "device" / "net"
    try:
        names = sorted(entry.name for entry in folder.iterdir())
    except OSError:
        return None
    return names[0] if names else None


def _ethtool_ioctl(netdev: str, payload: ctypes.Array) -> None:
    import fcntl  # POSIX only; the harness worker runs on Linux

    request = struct.pack("16sP", netdev.encode()[:15], ctypes.addressof(payload)).ljust(40, b"\0")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        fcntl.ioctl(sock.fileno(), _SIOCETHTOOL, bytearray(request))


def ethtool_stats_ioctl(netdev: str) -> dict[str, int]:
    """Driver statistics of ``netdev`` through ``SIOCETHTOOL`` (raises OSError when unavailable)."""
    info = (ctypes.c_uint8 * 24)()
    struct.pack_into("=IIQ", info, 0, _ETHTOOL_GSSET_INFO, 0, 1 << _ETH_SS_STATS)
    _ethtool_ioctl(netdev, info)
    _, _, mask = struct.unpack_from("=IIQ", info, 0)
    if not mask & (1 << _ETH_SS_STATS):
        return {}
    count = struct.unpack_from("=I", info, 16)[0]
    strings = (ctypes.c_uint8 * (12 + count * _ETH_GSTRING_LEN))()
    struct.pack_into("=III", strings, 0, _ETHTOOL_GSTRINGS, _ETH_SS_STATS, count)
    _ethtool_ioctl(netdev, strings)
    stats = (ctypes.c_uint8 * (8 + count * 8))()
    struct.pack_into("=II", stats, 0, _ETHTOOL_GSTATS, count)
    _ethtool_ioctl(netdev, stats)
    names = bytes(strings)[12:]
    values = struct.unpack_from(f"={count}Q", stats, 8)
    result = {}
    for index in range(count):
        raw = names[index * _ETH_GSTRING_LEN:(index + 1) * _ETH_GSTRING_LEN]
        result[raw.split(b"\0", 1)[0].decode(errors="replace")] = int(values[index])
    return result


def parse_ethtool_text(text: str) -> dict[str, int]:
    """``name: value`` lines of ``ethtool -S``."""
    result = {}
    for line in text.splitlines():
        name, separator, value = line.strip().rpartition(":")
        if separator and name and value.strip().isdigit():
            result[name.strip()] = int(value.strip())
    return result


def ethtool_stats(netdev: str) -> tuple[dict[str, int], str]:
    """``(statistics, source)``; source is ``ioctl``, ``ethtool`` or ``unavailable: <reason>``."""
    try:
        return ethtool_stats_ioctl(netdev), "ioctl"
    except (OSError, ImportError) as error:
        reason = str(error)
    try:
        process = subprocess.run(["ethtool", "-S", netdev], capture_output=True, text=True, timeout=10)
        if process.returncode == 0:
            return parse_ethtool_text(process.stdout), "ethtool"
        reason += f"; ethtool -S: {process.stderr.strip()}"
    except (OSError, subprocess.TimeoutExpired) as error:
        reason += f"; ethtool -S: {error}"
    return {}, f"unavailable: {reason}"


def snapshot(devices: Iterable[str], root: Path = SYSFS) -> dict[str, dict]:
    """Counters of every device: ``{device: {"rdma": {...}, "netdev": name, "ethtool": {...}, "source": ...}}``."""
    result: dict[str, dict] = {}
    for device in devices:
        netdev = netdev_of(device, root)
        ethtool, source = ethtool_stats(netdev) if netdev else ({}, "unavailable: no network device")
        result[device] = {"rdma": rdma_counters(device, root), "netdev": netdev, "ethtool": ethtool,
                          "source": source}
    return result


def delta(before: Mapping[str, dict], after: Mapping[str, dict]) -> dict[str, dict[str, int]]:
    """Changed counters per device (both sources), as after minus before."""
    changes: dict[str, dict[str, int]] = {}
    for device, now in after.items():
        then = before.get(device, {})
        moved = {}
        for source in ("rdma", "ethtool"):
            old = then.get(source, {})
            for name, value in now.get(source, {}).items():
                if name in old and value != old[name]:
                    moved[name] = value - old[name]
        if moved:
            changes[device] = moved
    return changes


def is_key(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return base in KEY_COUNTERS or "hairpin" in base


def key_deltas(changes: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    """``{"<device>:<counter>": increase}`` for drop, retransmission and error counters."""
    return {f"{device}:{name}": value for device, moved in changes.items()
            for name, value in moved.items() if is_key(name) and value}
