"""ctypes binding of SIRCL's point-to-point native library (``p2p/_p2p_proxy.c``).

:func:`load` builds the library when no cached build of the same source
exists (:mod:`sparkring_sircl.p2p.build`), declares the C signatures and
checks the native ABI. ``SIRCL_P2P_NATIVE_LIBRARY`` names a prebuilt library
to load instead (for example one built against the CPU simulator's verbs
stand-in). :func:`layout` reports the arena offsets; :class:`Native` owns one
rank's native context: devices, registered arena, queue pairs and progress
thread.

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

from ..oneshot._proxy import traffic_class
from . import build
from .protocol import ABI_VERSION, LAYOUT_WORDS, P2PLayout

_LOCK = threading.Lock()
_LIBRARIES: dict[str, ctypes.CDLL] = {}


def _declare(lib: ctypes.CDLL) -> ctypes.CDLL:
    u64, u32, i32, p = ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p
    signatures = {
        "p2p_abi_version": (i32, []),
        "p2p_layout": (i32, [i32, i32, i32, u64, ctypes.POINTER(u64)]),
        "p2p_blob_bytes": (u64, []),
        "p2p_create": (p, [i32, i32, ctypes.POINTER(ctypes.c_char_p), i32, ctypes.POINTER(i32), i32,
                           ctypes.POINTER(i32), i32, ctypes.POINTER(i32), p, u64, i32, u64, ctypes.c_char_p, u64]),
        "p2p_local_blob": (i32, [p, p, u64]),
        "p2p_connect": (i32, [p, p, u64]),
        "p2p_lane_check": (i32, [p, i32]),
        "p2p_set_windows": (i32, [p, ctypes.POINTER(u32), u32]),
        "p2p_start": (i32, [p]),
        "p2p_stop": (None, [p]),
        "p2p_failed": (i32, [p]),
        "p2p_error": (ctypes.c_char_p, [p]),
        "p2p_stat": (u64, [p, i32]),
        "p2p_peer_stat": (u64, [p, i32, i32]),
        "p2p_hca_stat": (u64, [p, i32, i32]),
        "p2p_destroy": (i32, [p]),
        "p2p_store_release_u32": (None, [p, u32]),
        "p2p_load_acquire_u32": (u32, [p]),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(lib, name)
        function.restype = restype
        function.argtypes = argtypes
    for name, restype, argtypes in (("p2p_test_set_base", i32, [p, u32]), ("p2p_test_qp_num", u32, [p, i32, i32])):
        function = getattr(lib, name, None)
        if function is not None:
            function.restype = restype
            function.argtypes = argtypes
    native = lib.p2p_abi_version()
    if native != ABI_VERSION:
        raise RuntimeError(f"unexpected point-to-point native ABI version {native}; this binding expects {ABI_VERSION}")
    features = local_features(lib)
    if features is None or features & REQUIRED_FEATURES != REQUIRED_FEATURES:
        found = "no local feature identity (p2p_local_features)" if features is None else f"local features {features:#x}"
        raise RuntimeError(f"the point-to-point native library has {found}; this binding requires "
                           f"{REQUIRED_FEATURES:#x} (a failed-verbs count from p2p_destroy): it was built from an earlier "
                           f"source of ABI {ABI_VERSION}; rebuild it or unset SIRCL_P2P_NATIVE_LIBRARY")
    return lib


# Local features (p2p_local_features): what a library offers this process apart from the wire contract of
# ABI_VERSION; the binding requires REQUIRED_FEATURES of every library it loads.
FEATURE_DESTROY_COUNT = 1      # p2p_destroy returns the number of verbs calls that failed
REQUIRED_FEATURES = FEATURE_DESTROY_COUNT


def local_features(lib) -> int | None:
    """The library's local feature bits (``p2p_local_features``), or None for a library without the identity."""
    function = getattr(lib, "p2p_local_features", None)
    if function is None:
        return None
    function.restype = ctypes.c_uint
    function.argtypes = []
    return int(function())


def load(path: str | os.PathLike | None = None) -> ctypes.CDLL:
    """The bound native library: ``path``, else ``SIRCL_P2P_NATIVE_LIBRARY``, else the cached build."""
    chosen = path or os.environ.get("SIRCL_P2P_NATIVE_LIBRARY") or build.build()
    key = str(Path(chosen).resolve())
    with _LOCK:
        lib = _LIBRARIES.get(key)
        if lib is None:
            lib = _declare(ctypes.CDLL(key, use_errno=True))
            _LIBRARIES[key] = lib
        return lib


def layout(world: int, lanes: int, slots: int, slot_bytes: int, *, library: ctypes.CDLL | None = None) -> tuple[int, ...]:
    """The arena offsets as the native layer computes them (:meth:`P2PLayout.as_tuple`)."""
    out = (ctypes.c_uint64 * LAYOUT_WORDS)()
    lib = library or load()
    if lib.p2p_layout(int(world), int(lanes), int(slots), int(slot_bytes), out) != 0:
        raise ValueError(f"unsupported point-to-point geometry: {world} ranks, {lanes} lanes, {slots} slots of "
                         f"{slot_bytes} bytes")
    return tuple(int(value) for value in out)


class Native:
    """One rank's native point-to-point context."""

    def __init__(
        self,
        *,
        world_size: int,
        rank: int,
        hca_names: Sequence[str],
        peer_lane_devices: Sequence[Sequence[int]],
        lane_count: int,
        gid_indices: Sequence[int],
        channels: Sequence[bool],
        region_ptr: int,
        region_bytes: int,
        slots: int,
        slot_bytes: int,
        library: ctypes.CDLL | None = None,
    ) -> None:
        self._lib = library or load()
        self._ctx = None
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.hca_names = tuple(hca_names)
        self.lane_count = int(lane_count)
        self.layout = P2PLayout(self.world_size, self.lane_count, int(slots), int(slot_bytes))
        self.traffic_class = traffic_class()
        if len(peer_lane_devices) != self.world_size or len(channels) != self.world_size:
            raise ValueError("peer_lane_devices and channels need one entry per rank")
        if len(gid_indices) != len(self.hca_names):
            raise ValueError("gid_indices needs one GID index per device")
        flat = []
        for peer, devices in enumerate(peer_lane_devices):
            devices = tuple(devices)
            if peer == self.rank or not channels[peer]:
                if devices:
                    raise ValueError(f"rank {peer} has no channel with rank {self.rank} and names lane devices")
                flat.extend([-1] * self.lane_count)
            else:
                if len(devices) != self.lane_count:
                    raise ValueError(f"rank {peer} needs {self.lane_count} lane devices, got {len(devices)}")
                flat.extend(int(device) for device in devices)
        names = (ctypes.c_char_p * len(self.hca_names))(*[name.encode() for name in self.hca_names])
        lanes = (ctypes.c_int * len(flat))(*flat)
        gids = (ctypes.c_int * len(gid_indices))(*[int(index) for index in gid_indices])
        enabled = (ctypes.c_int * self.world_size)(*[1 if peer != self.rank and channels[peer] else 0
                                                     for peer in range(self.world_size)])
        err = ctypes.create_string_buffer(512)
        ctx = self._lib.p2p_create(
            self.world_size, self.rank, names, len(self.hca_names), lanes, self.lane_count, gids, self.traffic_class,
            enabled, ctypes.c_void_p(int(region_ptr)), int(region_bytes), int(slots), int(slot_bytes), err, len(err),
        )
        if not ctx:
            raise RuntimeError(f"SIRCL point-to-point native setup failed: {err.value.decode(errors='replace')}")
        self._ctx = ctx

    def _handle(self):
        if not self._ctx:
            raise RuntimeError("SIRCL point-to-point native context is closed")
        return self._ctx

    def local_blob(self) -> bytes:
        size = int(self._lib.p2p_blob_bytes())
        buffer = ctypes.create_string_buffer(size)
        if self._lib.p2p_local_blob(self._handle(), buffer, size) != 0:
            raise RuntimeError("SIRCL point-to-point connection record could not be written")
        return buffer.raw

    def connect(self, blobs: Sequence[bytes]) -> None:
        """Validate every rank's record (in rank order), then connect every lane of every channel."""
        size = int(self._lib.p2p_blob_bytes())
        if len(blobs) != self.world_size or any(len(blob) != size for blob in blobs):
            raise RuntimeError(f"connection records must be {self.world_size} records of {size} bytes")
        joined = b"".join(blobs)
        buffer = ctypes.create_string_buffer(joined, len(joined))
        if self._lib.p2p_connect(self._handle(), buffer, len(joined)) != 0:
            raise RuntimeError(f"SIRCL point-to-point queue-pair connection failed: {self.error()}")

    def lane_check(self, timeout_ms: int = 2000) -> None:
        if self._lib.p2p_lane_check(self._handle(), int(timeout_ms)) != 0:
            raise RuntimeError(self.error())

    def set_windows(self, lane_windows: Sequence[Sequence[int]] | None, chunk_bytes: int) -> None:
        """Forward windows in bytes, ``lane_windows[peer][lane]`` (0: a direct lane), set before :meth:`start`."""
        if lane_windows is None:
            if self._lib.p2p_set_windows(self._handle(), None, 0) != 0:
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
        if self._lib.p2p_set_windows(self._handle(), array, int(chunk_bytes)) != 0:
            raise RuntimeError(self.error())

    def set_base(self, base: int) -> None:
        """Start every channel's item counters at ``base`` (test builds only, before :meth:`start`)."""
        function = getattr(self._lib, "p2p_test_set_base", None)
        if function is None:
            raise RuntimeError("this point-to-point library has no test hooks")
        if function(self._handle(), int(base) & 0xFFFFFFFF) != 0:
            raise RuntimeError("item counters are set before the progress thread starts")

    def start(self) -> None:
        if self._lib.p2p_start(self._handle()) != 0:
            raise RuntimeError(f"SIRCL point-to-point progress thread failed to start: {self.error()}")

    def stop(self) -> None:
        if self._ctx:
            self._lib.p2p_stop(self._ctx)

    def failed(self) -> bool:
        return bool(self._ctx) and bool(self._lib.p2p_failed(self._ctx))

    def error(self) -> str:
        if not self._ctx:
            return ""
        raw = self._lib.p2p_error(self._ctx)
        return raw.decode(errors="replace") if raw else ""

    def stats(self) -> dict[str, object]:
        ctx = self._handle()
        stat = self._lib.p2p_stat
        peers = {}
        for peer in range(self.world_size):
            if peer == self.rank:
                continue
            values = [int(self._lib.p2p_peer_stat(ctx, peer, which)) for which in range(6)]
            if any(values):
                peers[str(peer)] = {"items_posted": values[0], "bytes_posted": values[1], "items_released": values[2],
                                    "credits_sent": values[3], "credit": values[4], "sent": values[5]}
        return {
            "items_posted": int(stat(ctx, 0)),
            "bytes_posted": int(stat(ctx, 1)),
            "items_released": int(stat(ctx, 2)),
            "credits_sent": int(stat(ctx, 3)),
            "writes_completed": int(stat(ctx, 4)),
            "window_waits": int(stat(ctx, 5)),
            "window_wait_ns": int(stat(ctx, 6)),
            "window_wait_max_ns": int(stat(ctx, 7)),
            "window_max_unacked_bytes": int(stat(ctx, 8)),
            "proven_bytes": int(stat(ctx, 9)),
            "proxy_cpu": int(stat(ctx, 10)) - 1,
            "proxy_cpu_migrations": int(stat(ctx, 11)),
            "abort_from_rank": int(stat(ctx, 12)) - 1,
            "per_peer": peers,
            "writes_completed_per_hca": [int(self._lib.p2p_hca_stat(ctx, d, 0)) for d in range(len(self.hca_names))],
            "bytes_posted_per_hca": [int(self._lib.p2p_hca_stat(ctx, d, 1)) for d in range(len(self.hca_names))],
        }

    def qp_num(self, device: int, peer: int) -> int:
        """Queue-pair number of ``device`` toward ``peer`` (test builds only)."""
        function = getattr(self._lib, "p2p_test_qp_num", None)
        if function is None:
            raise RuntimeError("this point-to-point library has no test hooks")
        return int(function(self._handle(), int(device), int(peer)))

    def close(self) -> int:
        """Stop the progress thread and release the verbs objects (idempotent): the number of verbs calls
        that failed, 0 when every object was released. Nonzero means a queue pair or memory registration may
        survive, so the caller keeps the registered arena allocated."""
        ctx, self._ctx = self._ctx, None
        if ctx:
            return int(self._lib.p2p_destroy(ctx))
        return 0

    def __del__(self) -> None:  # pragma: no cover - defensive teardown
        with contextlib.suppress(Exception):
            self.close()


def store_release_u32(address: int, value: int, *, library: ctypes.CDLL) -> None:
    library.p2p_store_release_u32(ctypes.c_void_p(int(address)), int(value) & 0xFFFFFFFF)


def load_acquire_u32(address: int, *, library: ctypes.CDLL) -> int:
    return int(library.p2p_load_acquire_u32(ctypes.c_void_p(int(address))))


__all__ = ["ABI_VERSION", "Native", "layout", "load", "load_acquire_u32", "store_release_u32"]
