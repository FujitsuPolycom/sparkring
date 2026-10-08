"""NCCL baseline of the ring harness: NCCL's all-reduce and all-gather timed beside SIRCL's.

The binding calls the public C API of one NCCL library through ctypes: ``ncclGetVersion``,
``ncclGetUniqueId``, ``ncclCommInitRank``, ``ncclAllReduce``, ``ncclAllGather``, ``ncclCommDestroy`` and
``ncclGetErrorString``. NCCL's operations are capturable in a CUDA graph, so the harness times them eager
and in graph replay like SIRCL's.

NCCL runs on a group whose ranks reach each other without relays: a pair or the whole cycle. Every rank
of such a group uses one environment (:func:`environment`), the one NCCL needs on this fabric: the image's
NCCL preloaded (``LD_PRELOAD``), so torch and this binding share one library; the InfiniBand transport on
all four RDMA functions (``NCCL_NET=IB``, ``NCCL_IB_HCA``), extended IPv4 GIDs at the site's GID index,
subnet-aware routing, the ring algorithm on the switchless ring (``NCCL_SWITCHLESS_RING_ONLY``), four
channels and the site's wired-LAN interface for bootstrap. Each variable can be overridden. NCCL logs its
initialization and network setup (``NCCL_DEBUG=INFO``, subsystems ``INIT,NET``) to a file per rank, from
which :func:`transport` reads whether it uses ``NET/IB`` or ``NET/Socket``.
"""

from __future__ import annotations

import ctypes
import re
from collections.abc import Iterable, Mapping, Sequence

DEFAULT_LIBRARY = "/opt/sparkring/toolchain/nccl/lib/libnccl.so.2"
# The RDMA functions of a Spark in the order NCCL's device list names them.
DEVICES = ("rocep1s0f0", "rocep1s0f1", "roceP2p1s0f0", "roceP2p1s0f1")
# The environment of NCCL on the switchless ring (pairs and the whole cycle), apart from the site's
# interface, the site's GID index and the library, which environment() adds.
FABRIC_SETTINGS = {
    "NCCL_ALGO": "Ring", "NCCL_CROSS_NIC": "1", "NCCL_CUMEM_ENABLE": "0", "NCCL_IB_DISABLE": "0",
    "NCCL_IB_EXTENDED_IPV4_GIDS": "1", "NCCL_IB_HCA": "=" + ",".join(f"{device}:1" for device in DEVICES),
    "NCCL_IB_MERGE_NICS": "0", "NCCL_IB_PRESERVE_PCI_DOMAIN": "1", "NCCL_IB_ROUTE_DIAGNOSTICS": "1",
    "NCCL_IB_SUBNET_AWARE_ROUTING": "1", "NCCL_IGNORE_CPU_AFFINITY": "1", "NCCL_MAX_NCHANNELS": "4",
    "NCCL_MIN_NCHANNELS": "4", "NCCL_NET": "IB", "NCCL_NET_PLUGIN": "none", "NCCL_P2P_LEVEL": "SYS",
    "NCCL_PROTO": "LL,LL128,Simple", "NCCL_SWITCHLESS_RING_ONLY": "1",
}
# NCCL's log of its initialization and network setup, read for the transport.
DEBUG_SETTINGS = {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,NET"}
# ncclDataType_t and ncclRedOp_t values of the public API.
DTYPES = {"int8": 0, "uint8": 1, "int32": 2, "uint32": 3, "int64": 4, "uint64": 5, "float16": 6, "float32": 7,
          "float64": 8, "bfloat16": 9}
SUM = 0
# An 8 KiB eager all-reduce on a pair above this many microseconds marks NCCL's rows as degraded.
DEGRADED_PAIR_8K_US = 100.0


class NcclError(RuntimeError):
    pass


class UniqueId(ctypes.Structure):
    """``ncclUniqueId``: 128 opaque bytes that rank 0 creates and every rank passes to ``ncclCommInitRank``."""

    # Unsigned bytes: a c_char array would convert to bytes only up to its first zero byte.
    _fields_ = [("internal", ctypes.c_ubyte * 128)]


def baseline_kind(layout_kind: str, world: int) -> str | None:
    """``pair`` or ``cycle`` when NCCL may run on a group of this layout, else None."""
    if world == 2:
        return "pair"
    if layout_kind == "cycle" and world > 2:
        return "cycle"
    return None


def environment(lan_interface: str, gid_index: int | None, library: str = "",
                overrides: Iterable[tuple[str, str]] = ()) -> dict[str, str]:
    """The NCCL environment of every rank of a baseline group: ``FABRIC_SETTINGS``, the site's interface
    (``NCCL_SOCKET_IFNAME``), the site's GID index (``NCCL_IB_GID_INDEX``, 3 when the site names none) and
    the library (``LD_PRELOAD``), then ``overrides`` in order."""
    values = dict(FABRIC_SETTINGS)
    values["NCCL_IB_GID_INDEX"] = str(3 if gid_index is None else int(gid_index))
    values["NCCL_SOCKET_IFNAME"] = lan_interface
    values["LD_PRELOAD"] = library or DEFAULT_LIBRARY
    for name, value in overrides:
        values[str(name)] = str(value)
    return dict(sorted(values.items()))


def check_override(name: str, value: str) -> None:
    """Raise unless ``name`` is an NCCL variable or ``LD_PRELOAD`` and ``value`` is a plain value."""
    if not (name.startswith("NCCL_") or name == "LD_PRELOAD") or not name.replace("_", "").isalnum():
        raise NcclError(f"--nccl-env names an NCCL_* variable or LD_PRELOAD, got {name!r}")
    if not value or not all(ch.isalnum() or ch in "._,:/=+-%" for ch in value):
        raise NcclError(f"the value of {name} is empty or holds characters other than letters, digits and "
                        "._,:/=+-%")


def transport(log_text: str) -> str | None:
    """``NET/IB`` or ``NET/Socket`` from NCCL's log (``NCCL_DEBUG=INFO``, subsystems ``INIT,NET``), or None
    when the log names neither. ``NET/Socket`` wins when both appear: a socket connection is a
    degraded one."""
    found = set(re.findall(r"NET/(IB|Socket)", log_text))
    if "Socket" in found:
        return "NET/Socket"
    if "IB" in found:
        return "NET/IB"
    return None


def loaded_libraries(maps_text: str) -> list[str]:
    """Every NCCL library file mapped into a process (the text of ``/proc/self/maps``)."""
    found = []
    for line in maps_text.splitlines():
        path = line.split()[-1] if line.split() else ""
        if "libnccl" in path and path not in found:
            found.append(path)
    return found


class Library:
    """One loaded NCCL library."""

    def __init__(self, path: str = DEFAULT_LIBRARY) -> None:
        self.path = path
        try:
            self._lib = ctypes.CDLL(path)
        except OSError as error:
            raise NcclError(f"cannot load NCCL from {path}: {error}") from None
        lib = self._lib
        lib.ncclGetErrorString.restype = ctypes.c_char_p
        lib.ncclGetErrorString.argtypes = [ctypes.c_int]
        lib.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(UniqueId)]
        lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, UniqueId, ctypes.c_int]
        lib.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_void_p, ctypes.c_void_p]
        lib.ncclAllGather.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                                      ctypes.c_void_p, ctypes.c_void_p]
        lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
        for name in ("ncclGetVersion", "ncclGetUniqueId", "ncclCommInitRank", "ncclAllReduce", "ncclAllGather",
                     "ncclCommDestroy"):
            getattr(lib, name).restype = ctypes.c_int

    def check(self, result: int, call: str) -> None:
        if result != 0:
            raise NcclError(f"{call} failed: {self._lib.ncclGetErrorString(result).decode()} (code {result})")

    def version(self) -> str:
        """The library's version as ``major.minor.patch``."""
        code = ctypes.c_int()
        self.check(self._lib.ncclGetVersion(ctypes.byref(code)), "ncclGetVersion")
        value = code.value
        if value >= 10000:
            return f"{value // 10000}.{value // 100 % 100}.{value % 100}"
        return f"{value // 1000}.{value // 100 % 10}.{value % 100}"

    def unique_id(self) -> bytes:
        uid = UniqueId()
        self.check(self._lib.ncclGetUniqueId(ctypes.byref(uid)), "ncclGetUniqueId")
        return bytes(uid.internal)

    def communicator(self, world: int, uid: bytes, rank: int) -> "Communicator":
        """A communicator of ``world`` ranks; every rank calls this with rank 0's ``unique_id``."""
        if len(uid) != 128:
            raise NcclError(f"an NCCL unique id has 128 bytes, got {len(uid)}")
        handle = ctypes.c_void_p()
        token = UniqueId()
        ctypes.memmove(ctypes.byref(token), uid, 128)
        self.check(self._lib.ncclCommInitRank(ctypes.byref(handle), int(world), token, int(rank)),
                   "ncclCommInitRank")
        return Communicator(self, handle)


class Communicator:
    """An initialized NCCL communicator; its operations take device addresses and a CUDA stream handle."""

    def __init__(self, library: Library, handle: ctypes.c_void_p) -> None:
        self.library = library
        self._handle = handle

    def all_reduce(self, send: int, recv: int, count: int, dtype: str, stream: int) -> None:
        """Sum ``count`` elements of ``dtype`` over the ranks."""
        self.library.check(self.library._lib.ncclAllReduce(
            ctypes.c_void_p(send), ctypes.c_void_p(recv), ctypes.c_size_t(count), DTYPES[dtype], SUM,
            self._handle, ctypes.c_void_p(stream)), "ncclAllReduce")

    def all_gather(self, send: int, recv: int, count: int, dtype: str, stream: int) -> None:
        """Concatenate every rank's ``count`` elements in rank order."""
        self.library.check(self.library._lib.ncclAllGather(
            ctypes.c_void_p(send), ctypes.c_void_p(recv), ctypes.c_size_t(count), DTYPES[dtype], self._handle,
            ctypes.c_void_p(stream)), "ncclAllGather")

    def close(self) -> None:
        if self._handle:
            self.library.check(self.library._lib.ncclCommDestroy(self._handle), "ncclCommDestroy")
            self._handle = ctypes.c_void_p()


def degraded_pairs(cases: Sequence[Mapping]) -> dict[int, float]:
    """Groups of two ranks whose 8 KiB eager NCCL all-reduce took more than ``DEGRADED_PAIR_8K_US``
    (slowest-rank median), by group, with that time."""
    found = {}
    for case in cases:
        if (case.get("collective") == "nccl_all_reduce" and case.get("mode") == "eager"
                and case.get("bytes") == 8192 and case.get("world_ranks") == 2
                and (case.get("slowest_p50_us") or 0) > DEGRADED_PAIR_8K_US):
            found[int(case["group"])] = float(case["slowest_p50_us"])
    return found


__all__ = ["Communicator", "DEFAULT_LIBRARY", "DEGRADED_PAIR_8K_US", "DEVICES", "FABRIC_SETTINGS", "Library",
           "NcclError", "UniqueId", "baseline_kind", "check_override", "degraded_pairs", "environment",
           "loaded_libraries", "transport"]
