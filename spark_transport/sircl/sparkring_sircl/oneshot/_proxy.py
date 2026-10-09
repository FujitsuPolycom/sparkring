"""ctypes binding of SIRCL's native ring-session library (``_roce_proxy.c``).

:func:`load` builds the library when no cached build of the same source
exists (:mod:`sparkring_sircl.build`), declares the C signatures and checks
the native ABI (the wire contract peers compare) and the library's local
features (``roce_local_features``: what it offers this process's binding and
kernels), refusing a library that lacks :data:`REQUIRED_FEATURES`.
``SIRCL_NATIVE_LIBRARY`` names a prebuilt library to load instead (for
example one built against the CPU simulator's verbs stand-in); a prebuilt
library of the same ABI from an earlier source fails the feature check before
any context exists.
:class:`Layout` reports the arena offsets; :class:`Proxy` owns one rank's
native context: devices, registered arena, queue pairs and progress thread.

This module needs no torch, so the CPU tests exercise it with the simulator
build of the library.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import threading
from collections.abc import Sequence
from pathlib import Path

from .. import build

ABI_VERSION = 9
# Local features (roce_local_features): what a library offers this process apart from the wire contract of
# ABI_VERSION, which peers compare. The binding requires REQUIRED_FEATURES of every library it loads (the source
# build, an explicit path or SIRCL_NATIVE_LIBRARY); a library of the same wire ABI built from earlier sources lacks
# the identity or its bits and is refused before any context exists.
FEATURE_DESTROY_COUNT = 1      # roce_destroy returns the number of verbs calls that failed (an arena kept on failure)
FEATURE_OWN_FLAGS = 2          # link op word bit 24 (protocol.ring_op_word(own_flags=True)) is accepted
REQUIRED_FEATURES = FEATURE_DESTROY_COUNT
_LOCK = threading.Lock()
_LIBRARIES: dict[str, ctypes.CDLL] = {}


def _declare(lib: ctypes.CDLL) -> ctypes.CDLL:
    u64, i32, p = ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p
    signatures = {
        "roce_abi_version": (i32, []),
        "roce_layout": (i32, [i32, u64, ctypes.POINTER(u64)]),
        "roce_blob_bytes": (u64, []),
        "roce_create": (p, [i32, i32, ctypes.POINTER(ctypes.c_char_p), i32, ctypes.POINTER(i32), i32,
                            ctypes.POINTER(i32), i32, p, u64, u64, ctypes.c_char_p, u64]),
        "roce_local_blob": (i32, [p, p, u64]),
        "roce_connect": (i32, [p, p, u64]),
        "roce_lane_check": (i32, [p, i32]),
        "roce_start": (i32, [p]),
        "roce_stop": (None, [p]),
        "roce_failed": (i32, [p]),
        "roce_error": (ctypes.c_char_p, [p]),
        "roce_stat": (u64, [p, i32]),
        "roce_hca_stat": (u64, [p, i32, i32]),
        "roce_tracing": (i32, [p]),
        "roce_trace_read": (i32, [p, ctypes.POINTER(u64), u64]),
        "roce_destroy": (i32, [p]),
        "roce_store_release_u32": (None, [p, ctypes.c_uint32]),
        "roce_load_acquire_u32": (ctypes.c_uint32, [p]),
        "roce_set_forward": (i32, [p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]),
        "roce_chain_layout": (i32, [i32, i32, u64, ctypes.POINTER(u64)]),
        "roce_set_chain": (i32, [p, i32, i32, i32, u64, u64]),
        "roce_link_layout": (i32, [i32, i32, u64, ctypes.POINTER(u64)]),
        "roce_set_links": (i32, [p, i32, i32, i32, i32, i32, ctypes.c_uint32, i32, u64, u64]),
        "roce_set_trace": (i32, [p, ctypes.c_uint32]),
        "roce_trace_take": (ctypes.c_int64, [p, ctypes.POINTER(u64), u64, ctypes.POINTER(u64)]),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(lib, name)
        function.restype = restype
        function.argtypes = argtypes
    native = lib.roce_abi_version()
    if native != ABI_VERSION:
        raise RuntimeError(f"unexpected native ABI version {native}; this binding expects {ABI_VERSION}")
    features = local_features(lib)
    if features is None or features & REQUIRED_FEATURES != REQUIRED_FEATURES:
        found = "no local feature identity (roce_local_features)" if features is None else f"local features {features:#x}"
        raise RuntimeError(f"the native library has {found}; this binding requires {REQUIRED_FEATURES:#x} (a "
                           f"failed-verbs count from roce_destroy): it was built from an earlier source of ABI "
                           f"{ABI_VERSION}; rebuild it (sircl-prepare) or unset SIRCL_NATIVE_LIBRARY")
    return lib


def local_features(lib) -> int | None:
    """The library's local feature bits (``roce_local_features``), or None for a library without the identity."""
    function = getattr(lib, "roce_local_features", None)
    if function is None:
        return None
    function.restype = ctypes.c_uint
    function.argtypes = []
    return int(function())


def load(path: str | os.PathLike | None = None) -> ctypes.CDLL:
    """The bound native library: ``path``, else ``SIRCL_NATIVE_LIBRARY``, else the cached build."""
    chosen = path or os.environ.get("SIRCL_NATIVE_LIBRARY") or build.build()
    key = str(Path(chosen).resolve())
    with _LOCK:
        lib = _LIBRARIES.get(key)
        if lib is None:
            lib = _declare(ctypes.CDLL(key, use_errno=True))
            _LIBRARIES[key] = lib
        return lib


def stand_in_library(path: str | os.PathLike | None = None) -> bool:
    """Whether the native library ``path`` (default ``SIRCL_NATIVE_LIBRARY``) is built against the verbs
    stand-in of the CPU simulator and the GPU emulation (``testing/fake_verbs``, which exports
    ``fv_add_device``). Its devices are the stand-in's, which the host's ``/sys/class/infiniband`` does not
    list; a library without a path (the cached build) is the real one."""
    chosen = path if path is not None else os.environ.get("SIRCL_NATIVE_LIBRARY", "")
    if not chosen:
        return False
    try:
        return hasattr(ctypes.CDLL(str(Path(chosen).resolve())), "fv_add_device")
    except OSError:
        return False


def traffic_class() -> int:
    """``SIRCL_TRAFFIC_CLASS``, else ``NCCL_IB_TC``, else 0 (the IP DSCP/ECN byte)."""
    for name in ("SIRCL_TRAFFIC_CLASS", "NCCL_IB_TC"):
        raw = os.environ.get(name)
        if raw is not None and raw.strip():
            try:
                value = int(raw, 0)
            except ValueError:
                raise ValueError(f"{name}={raw} is not an integer from 0 to 255") from None
            if not 0 <= value <= 255:
                raise ValueError(f"{name}={raw} is not an integer from 0 to 255")
            return value
    return 0


class Layout:
    """Byte offsets of the arena shared by the kernels and the progress thread."""

    __slots__ = ("recv_off", "flag_off", "send_off", "ctrl_off", "total_bytes", "flag_stride", "slots")

    def __init__(self, world_size: int, slot_bytes: int, *, library: ctypes.CDLL | None = None) -> None:
        out = (ctypes.c_uint64 * 7)()
        lib = library or load()
        if lib.roce_layout(int(world_size), int(slot_bytes), out) != 0:
            raise ValueError(
                f"unsupported arena geometry: world size {world_size}, slot bytes {slot_bytes} "
                "(2 to 16 ranks; slots a positive multiple of 4096 bytes up to 2^40)"
            )
        (self.recv_off, self.flag_off, self.send_off, self.ctrl_off, self.total_bytes,
         self.flag_stride, self.slots) = (int(value) for value in out)


def chain_layout(lanes: int, slots: int, slot_bytes: int, *, library: ctypes.CDLL | None = None) -> tuple[int, ...]:
    """Offsets of the chain area as the native layer computes them (``protocol.ChainLayout.as_tuple``)."""
    out = (ctypes.c_uint64 * 9)()
    lib = library or load()
    if lib.roce_chain_layout(int(lanes), int(slots), int(slot_bytes), out) != 0:
        raise ValueError(f"unsupported chain geometry: {lanes} lanes, {slots} slots of {slot_bytes} bytes")
    return tuple(int(value) for value in out)


def link_layout(lanes: int, slots: int, slot_bytes: int, *, library: ctypes.CDLL | None = None) -> tuple[int, ...]:
    """Offsets of the link area as the native layer computes them (``protocol.LinkLayout.as_tuple``)."""
    out = (ctypes.c_uint64 * 9)()
    lib = library or load()
    if lib.roce_link_layout(int(lanes), int(slots), int(slot_bytes), out) != 0:
        raise ValueError(f"unsupported link geometry: {lanes} lanes, {slots} slots of {slot_bytes} bytes")
    return tuple(int(value) for value in out)


class Proxy:
    """One rank's native context of one session."""

    def __init__(
        self,
        *,
        world_size: int,
        rank: int,
        hca_names: Sequence[str],
        peer_lane_devices: Sequence[Sequence[int]],
        lane_count: int,
        gid_indices: Sequence[int],
        region_ptr: int,
        region_bytes: int,
        slot_bytes: int,
        library: ctypes.CDLL | None = None,
    ) -> None:
        self._lib = library or load()
        self._ctx = None
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.hca_names = tuple(hca_names)
        self.lane_count = int(lane_count)
        self.traffic_class = traffic_class()
        if len(peer_lane_devices) != self.world_size:
            raise ValueError("peer_lane_devices needs one entry per rank")
        if len(gid_indices) != len(self.hca_names):
            raise ValueError("gid_indices needs one GID index per device")
        flat = []
        for peer, devices in enumerate(peer_lane_devices):
            devices = tuple(devices)
            if peer == self.rank:
                if devices:
                    raise ValueError("the own rank has no lanes")
                flat.extend([-1] * self.lane_count)
            else:
                if len(devices) != self.lane_count:
                    raise ValueError(f"rank {peer} needs {self.lane_count} lane devices, got {len(devices)}")
                flat.extend(int(device) for device in devices)
        names = (ctypes.c_char_p * len(self.hca_names))(*[name.encode() for name in self.hca_names])
        lanes = (ctypes.c_int * len(flat))(*flat)
        gids = (ctypes.c_int * len(gid_indices))(*[int(index) for index in gid_indices])
        err = ctypes.create_string_buffer(512)
        ctx = self._lib.roce_create(
            self.world_size, self.rank, names, len(self.hca_names), lanes, self.lane_count, gids,
            self.traffic_class, ctypes.c_void_p(int(region_ptr)), int(region_bytes), int(slot_bytes),
            err, len(err),
        )
        if not ctx:
            raise RuntimeError(f"SIRCL native setup failed: {err.value.decode(errors='replace')}")
        self._ctx = ctx
        self.traces = bool(self._lib.roce_tracing(ctx))
        self.post_mode = "direct" if int(self._lib.roce_stat(ctx, 9)) else "verbs"

    def _handle(self):
        if not self._ctx:
            raise RuntimeError("SIRCL native context is closed")
        return self._ctx

    def local_blob(self) -> bytes:
        """This rank's connection record."""
        size = int(self._lib.roce_blob_bytes())
        buffer = ctypes.create_string_buffer(size)
        if self._lib.roce_local_blob(self._handle(), buffer, size) != 0:
            raise RuntimeError("SIRCL connection record could not be written")
        return buffer.raw

    def connect(self, blobs: Sequence[bytes]) -> None:
        """Validate every rank's record (in rank order), then connect every lane."""
        size = int(self._lib.roce_blob_bytes())
        if len(blobs) != self.world_size or any(len(blob) != size for blob in blobs):
            raise RuntimeError(f"connection records must be {self.world_size} records of {size} bytes")
        joined = b"".join(blobs)
        buffer = ctypes.create_string_buffer(joined, len(joined))
        if self._lib.roce_connect(self._handle(), buffer, len(joined)) != 0:
            raise RuntimeError(f"SIRCL queue-pair connection failed: {self.error()}")

    def lane_check(self, timeout_ms: int = 2000) -> None:
        """Prove every lane with one small write (run after every rank connected)."""
        if self._lib.roce_lane_check(self._handle(), int(timeout_ms)) != 0:
            raise RuntimeError(self.error())

    def set_forward(self, lane_windows: Sequence[Sequence[int]] | None, chunk_bytes: int = 32768) -> None:
        """Forward windows in bytes, ``lane_windows[peer][lane]`` (0: post the lane's stripe whole).

        A lane with a window posts its stripe as signaled chunks of
        ``chunk_bytes`` and keeps at most the window unacknowledged. Set before
        :meth:`start`; ``None`` clears every window.
        """
        if lane_windows is None:
            if self._lib.roce_set_forward(self._handle(), None, 0) != 0:
                raise RuntimeError(self.error())
            return
        if len(lane_windows) != self.world_size:
            raise ValueError("lane_windows needs one entry per rank")
        flat = []
        for peer, windows in enumerate(lane_windows):
            windows = [int(window) for window in windows]
            if len(windows) != self.lane_count:
                raise ValueError(f"rank {peer} needs {self.lane_count} lane windows, got {len(windows)}")
            flat.extend(windows)
        array = (ctypes.c_uint32 * len(flat))(*flat)
        if self._lib.roce_set_forward(self._handle(), array, int(chunk_bytes)) != 0:
            raise RuntimeError(self.error())

    def set_chain(self, prev: int, next_rank: int, slots: int, slot_bytes: int, chain_offset: int) -> None:
        """The chain schedule: this rank's neighbors in chain order (-1 at an end), the ring
        geometry and the chain area's offset in every rank's arena. Set before :meth:`start`;
        ``slots`` 0 removes it."""
        if self._lib.roce_set_chain(self._handle(), int(prev), int(next_rank), int(slots), int(slot_bytes),
                                    int(chain_offset)) != 0:
            raise RuntimeError(self.error())

    def set_links(self, prev: int, next_rank: int, index: int, slots: int, slot_bytes: int, link_offset: int,
                  *, ring_prev: int = -1, ring_next: int = -1, ring_window: int = 0) -> None:
        """The links: this rank's neighbors and index in chain order (-1 at an end), its
        neighbors on the ring that closes the chain (-1: no ring links) and the bytes each of
        its lanes toward ``ring_next`` keeps unacknowledged when they run through relays (0: a
        direct cable), the ring geometry and the link area's offset in every rank's arena.
        Set before :meth:`start`; ``slots`` 0 removes them."""
        if self._lib.roce_set_links(self._handle(), int(prev), int(next_rank), int(index), int(ring_prev),
                                    int(ring_next), int(ring_window), int(slots), int(slot_bytes),
                                    int(link_offset)) != 0:
            raise RuntimeError(self.error())

    def set_trace(self, capacity: int) -> None:
        """Keep the newest ``capacity`` event records of chain streams and links (0: none).
        Set before :meth:`start`."""
        if self._lib.roce_set_trace(self._handle(), int(capacity)) != 0:
            raise RuntimeError(self.error())

    def take_trace(self, max_records: int = 1 << 20) -> tuple[list[tuple[int, int, int, int]], int]:
        """Event records written since the last take, oldest first, as (CLOCK_REALTIME
        nanoseconds, event, stream, value) (``protocol.TraceEvent``), and the number of records
        overwritten before they were taken."""
        out = (ctypes.c_uint64 * (2 * int(max_records)))()
        lost = ctypes.c_uint64(0)
        count = int(self._lib.roce_trace_take(self._handle(), out, int(max_records), ctypes.byref(lost)))
        records = []
        for i in range(count):
            word = int(out[2 * i + 1])
            records.append((int(out[2 * i]), (word >> 32) & 0xFFFF, word >> 48, word & 0xFFFFFFFF))
        return records, int(lost.value)

    def start(self) -> None:
        if self._lib.roce_start(self._handle()) != 0:
            raise RuntimeError(f"SIRCL progress thread failed to start: {self.error()}")

    def stop(self) -> None:
        if self._ctx:
            self._lib.roce_stop(self._ctx)

    def failed(self) -> bool:
        return bool(self._ctx) and bool(self._lib.roce_failed(self._ctx))

    def error(self) -> str:
        if not self._ctx:
            return ""
        raw = self._lib.roce_error(self._ctx)
        return raw.decode(errors="replace") if raw else ""

    def stats(self) -> dict[str, object]:
        ctx = self._handle()
        stat = self._lib.roce_stat
        return {
            "ops_posted": int(stat(ctx, 0)),
            "writes_completed": int(stat(ctx, 1)),
            "last_seq": int(stat(ctx, 2)),
            "lane_count": int(stat(ctx, 3)),
            "later_phases_posted": int(stat(ctx, 5)),
            "multi_phase": bool(stat(ctx, 6)),
            "proxy_cpu": int(stat(ctx, 7)) - 1,
            "proxy_cpu_migrations": int(stat(ctx, 8)),
            "post_mode": self.post_mode,
            "forward_chunks_posted": int(stat(ctx, 10)),
            "forward_max_unacked_bytes": int(stat(ctx, 11)),
            "chain_ops": int(stat(ctx, 12)),
            "chain_chunks_posted": int(stat(ctx, 13)),
            "chain_credits_sent": int(stat(ctx, 14)),
            "chain_bytes_posted": int(stat(ctx, 15)),
            "link_ops": int(stat(ctx, 16)),
            "link_items_posted": int(stat(ctx, 17)),
            "link_credits_sent": int(stat(ctx, 18)),
            "link_bytes_posted": int(stat(ctx, 19)),
            "link_window_chunks_posted": int(stat(ctx, 24)),
            "forward_waits": int(stat(ctx, 25)),
            "forward_wait_ns": int(stat(ctx, 26)),
            "forward_wait_max_ns": int(stat(ctx, 27)),
            "forward_proven_bytes": int(stat(ctx, 28)),
            "forward_proof": bool(stat(ctx, 29)),
            "writes_completed_per_hca": [int(self._lib.roce_hca_stat(ctx, d, 0))
                                         for d in range(len(self.hca_names))],
            "bytes_posted_per_hca": [int(self._lib.roce_hca_stat(ctx, d, 1))
                                     for d in range(len(self.hca_names))],
        }

    def trace_records(self):
        """Phase tracing is unsupported by this build of the native library."""
        return None

    def close(self) -> int:
        """Stop the progress thread and release the verbs objects (idempotent): the number of verbs calls
        that failed, 0 when every object was released. Nonzero means a queue pair or memory registration may
        survive, so the caller keeps the registered arena allocated."""
        ctx, self._ctx = self._ctx, None
        if ctx:
            return int(self._lib.roce_destroy(ctx))
        return 0

    def __del__(self) -> None:  # pragma: no cover - defensive teardown
        with contextlib.suppress(Exception):
            self.close()


__all__ = ["ABI_VERSION", "FEATURE_DESTROY_COUNT", "FEATURE_OWN_FLAGS", "Layout", "Proxy", "REQUIRED_FEATURES",
           "chain_layout", "link_layout", "load", "local_features", "stand_in_library", "traffic_class"]
