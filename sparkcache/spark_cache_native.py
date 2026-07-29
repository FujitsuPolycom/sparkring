"""Strict ctypes binding for the native SparkCache placement ABI.

This module is intentionally dependency-free. The full-artifact gate may use
``arena_memoryview`` with ``os.preadv`` to fill cudaHostAllocMapped memory
without creating an intermediate Python ``bytes`` object.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any, Iterable


ABI_VERSION = 1
ARENA_COUNT = 2
MAX_RECORD_KINDS = 4

STATUS_OK = 0

ARENA_MAPPED_HOST = 1
ARENA_MANAGED = 2
ARENA_STAGED_DEVICE = 3

RECORD_TARGET_CKV = 0
RECORD_SPARSE_INDEXER = 1
RECORD_MTP_DRAFT_KV = 2
RECORD_BOUNDARY_HIDDEN = 3

CAP_MAPPED_HOST = 1 << 0
CAP_MANAGED = 1 << 1
CAP_STAGED_DEVICE = 1 << 2
CAP_DIRECT_ENCODED = 1 << 3
CAP_TRANSPOSED = 1 << 4
CAP_LOW_PRIORITY_STREAMS = 1 << 5


class PlacementConfig(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("arena_mode", ctypes.c_uint32),
        ("arena_bytes", ctypes.c_uint64),
        ("max_destinations", ctypes.c_uint32),
        ("max_slots", ctypes.c_uint32),
        ("max_chunks_per_slab", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


class DestinationDescriptor(ctypes.Structure):
    _fields_ = [
        ("destination_base", ctypes.c_uint64),
        ("destination_rows", ctypes.c_uint64),
        ("destination_row_stride_bytes", ctypes.c_uint32),
        ("bytes_per_token", ctypes.c_uint32),
        ("record_kind", ctypes.c_uint32),
        ("source_layer_ordinal", ctypes.c_uint32),
    ]


class ChunkDescriptor(ctypes.Structure):
    _fields_ = [
        ("arena_offset_bytes", ctypes.c_uint64),
        ("encoded_bytes", ctypes.c_uint32),
        ("payload_offset_bytes", ctypes.c_uint32),
        ("first_slot_index", ctypes.c_uint32),
        ("row_count", ctypes.c_uint32),
        ("record_mask", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("record_offset_bytes", ctypes.c_uint32 * MAX_RECORD_KINDS),
        ("record_length_bytes", ctypes.c_uint32 * MAX_RECORD_KINDS),
    ]


class TransposedSource(ctypes.Structure):
    _fields_ = [
        ("source_offset_bytes", ctypes.c_uint64),
        ("destination_index", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


class PlacementStats(ctypes.Structure):
    _fields_ = [
        ("source_bytes", ctypes.c_uint64),
        ("staged_h2d_bytes", ctypes.c_uint64),
        ("restored_rows", ctypes.c_uint64),
        ("slot_uploads", ctypes.c_uint32),
        ("destination_table_uploads", ctypes.c_uint32),
        ("slabs_submitted", ctypes.c_uint32),
        ("scatter_kernel_launches", ctypes.c_uint32),
        ("device_error", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


class AbiInfo(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("cudart_version", ctypes.c_uint32),
        ("arena_count", ctypes.c_uint32),
        ("max_record_kinds", ctypes.c_uint32),
        ("sizeof_config", ctypes.c_uint32),
        ("sizeof_destination", ctypes.c_uint32),
        ("sizeof_chunk", ctypes.c_uint32),
        ("sizeof_transposed_source", ctypes.c_uint32),
        ("sizeof_stats", ctypes.c_uint32),
        ("sizeof_arena_view", ctypes.c_uint32),
        ("capability_flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 5),
    ]


class ArenaView(ctypes.Structure):
    _fields_ = [
        ("host_address", ctypes.c_uint64),
        ("device_address", ctypes.c_uint64),
        ("capacity_bytes", ctypes.c_uint64),
        ("arena_index", ctypes.c_uint32),
        ("arena_mode", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


EXPECTED_SIZES = {
    "sizeof_config": ctypes.sizeof(PlacementConfig),
    "sizeof_destination": ctypes.sizeof(DestinationDescriptor),
    "sizeof_chunk": ctypes.sizeof(ChunkDescriptor),
    "sizeof_transposed_source": ctypes.sizeof(TransposedSource),
    "sizeof_stats": ctypes.sizeof(PlacementStats),
    "sizeof_arena_view": ctypes.sizeof(ArenaView),
}


class NativePlacementError(RuntimeError):
    pass


def _bind(lib: ctypes.CDLL) -> None:
    handle = ctypes.c_void_p
    lib.spark_cache_placement_query_abi.argtypes = [ctypes.POINTER(AbiInfo)]
    lib.spark_cache_placement_query_abi.restype = ctypes.c_int
    lib.spark_cache_placement_create.argtypes = [
        ctypes.POINTER(PlacementConfig),
        ctypes.POINTER(handle),
    ]
    lib.spark_cache_placement_create.restype = ctypes.c_int
    lib.spark_cache_placement_destroy.argtypes = [handle]
    lib.spark_cache_placement_destroy.restype = None
    lib.spark_cache_placement_configure_destinations.argtypes = [
        handle,
        ctypes.POINTER(DestinationDescriptor),
        ctypes.c_uint32,
    ]
    lib.spark_cache_placement_configure_destinations.restype = ctypes.c_int
    lib.spark_cache_placement_begin_restore.argtypes = [
        handle,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
    ]
    lib.spark_cache_placement_begin_restore.restype = ctypes.c_int
    lib.spark_cache_placement_acquire_arena_view.argtypes = [
        handle,
        ctypes.c_uint32,
        ctypes.POINTER(ArenaView),
    ]
    lib.spark_cache_placement_acquire_arena_view.restype = ctypes.c_int
    lib.spark_cache_parse_verified_v1_chunk.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ChunkDescriptor),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.spark_cache_parse_verified_v1_chunk.restype = ctypes.c_int
    lib.spark_cache_placement_submit_direct_slab.argtypes = [
        handle,
        ctypes.c_uint32,
        ctypes.c_uint64,
        ctypes.POINTER(ChunkDescriptor),
        ctypes.c_uint32,
    ]
    lib.spark_cache_placement_submit_direct_slab.restype = ctypes.c_int
    lib.spark_cache_placement_submit_transposed_slab.argtypes = [
        handle,
        ctypes.c_uint32,
        ctypes.c_uint64,
        ctypes.POINTER(TransposedSource),
        ctypes.c_uint32,
    ]
    lib.spark_cache_placement_submit_transposed_slab.restype = ctypes.c_int
    lib.spark_cache_placement_finish_restore.argtypes = [
        handle,
        ctypes.POINTER(PlacementStats),
    ]
    lib.spark_cache_placement_finish_restore.restype = ctypes.c_int
    lib.spark_cache_placement_abort_restore.argtypes = [handle]
    lib.spark_cache_placement_abort_restore.restype = ctypes.c_int
    lib.spark_cache_placement_get_stats.argtypes = [
        handle,
        ctypes.POINTER(PlacementStats),
    ]
    lib.spark_cache_placement_get_stats.restype = ctypes.c_int
    lib.spark_cache_placement_copy_last_error.argtypes = [
        handle,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.spark_cache_placement_copy_last_error.restype = ctypes.c_int
    lib.spark_cache_placement_runtime_last_error.argtypes = []
    lib.spark_cache_placement_runtime_last_error.restype = ctypes.c_char_p
    lib.spark_cache_placement_status_string.argtypes = [ctypes.c_int]
    lib.spark_cache_placement_status_string.restype = ctypes.c_char_p


def _decode(value: bytes | None) -> str:
    return "" if value is None else value.decode("utf-8", errors="replace")


def runtime_error(lib: ctypes.CDLL) -> str:
    return _decode(lib.spark_cache_placement_runtime_last_error())


def handle_error(lib: ctypes.CDLL, handle: ctypes.c_void_p) -> str:
    output = ctypes.create_string_buffer(512)
    result = lib.spark_cache_placement_copy_last_error(
        handle, output, ctypes.sizeof(output)
    )
    if result != STATUS_OK:
        return runtime_error(lib)
    return output.value.decode("utf-8", errors="replace")


def check(
    lib: ctypes.CDLL,
    result: int,
    operation: str,
    handle: ctypes.c_void_p | None = None,
) -> None:
    if result == STATUS_OK:
        return
    status = _decode(lib.spark_cache_placement_status_string(result))
    detail = (
        handle_error(lib, handle)
        if handle is not None and handle.value
        else runtime_error(lib)
    )
    raise NativePlacementError(
        f"{operation} failed: status={result} ({status}) detail={detail}"
    )


def load_library(path: str | os.PathLike[str]) -> tuple[ctypes.CDLL, AbiInfo]:
    resolved = Path(path).resolve(strict=True)
    mode = getattr(os, "RTLD_LOCAL", 0) | getattr(os, "RTLD_NOW", 0)
    lib = ctypes.CDLL(str(resolved), mode=mode)
    _bind(lib)
    info = AbiInfo()
    check(lib, lib.spark_cache_placement_query_abi(ctypes.byref(info)), "ABI")
    if info.abi_version != ABI_VERSION:
        raise NativePlacementError(
            f"ABI version mismatch: library={info.abi_version} binding={ABI_VERSION}"
        )
    if info.arena_count != ARENA_COUNT or (info.max_record_kinds != MAX_RECORD_KINDS):
        raise NativePlacementError(
            "library arena/record constants disagree with binding"
        )
    mismatches = {
        field: (getattr(info, field), expected)
        for field, expected in EXPECTED_SIZES.items()
        if getattr(info, field) != expected
    }
    if mismatches:
        raise NativePlacementError(f"ctypes ABI size mismatch: {mismatches}")
    required = CAP_MAPPED_HOST | CAP_DIRECT_ENCODED
    if info.capability_flags & required != required:
        raise NativePlacementError(
            "library lacks mapped-host direct-encoded capability"
        )
    return lib, info


def arena_memoryview(view: ArenaView, *, length: int | None = None) -> memoryview:
    size = view.capacity_bytes if length is None else length
    if not view.host_address or size < 0 or size > view.capacity_bytes:
        raise ValueError("invalid mapped arena view or requested length")
    array_type = ctypes.c_ubyte * size
    owner = array_type.from_address(view.host_address)
    # ctypes exposes a platform-qualified format (often "<B"). Casting makes
    # the view a standard writable byte buffer accepted by os.preadv/hashlib.
    return memoryview(owner).cast("B")


def pread_exact(fd: int, target: memoryview, offset: int = 0) -> None:
    if target.readonly or not target.contiguous or target.format != "B":
        raise ValueError("pread target must be contiguous writable bytes")
    if offset < 0:
        raise ValueError("pread offset must be non-negative")
    if not hasattr(os, "preadv"):
        raise OSError("os.preadv is required for zero-copy arena reads")
    consumed = 0
    while consumed < len(target):
        count = os.preadv(fd, [target[consumed:]], offset + consumed)
        if count <= 0:
            raise EOFError(f"short pread: received {consumed} of {len(target)} bytes")
        consumed += count


class PlacementHandle:
    """Small ownership wrapper; full-artifact orchestration remains explicit."""

    def __init__(self, lib: ctypes.CDLL, config: PlacementConfig):
        self.lib = lib
        self.value = ctypes.c_void_p()
        check(
            lib,
            lib.spark_cache_placement_create(
                ctypes.byref(config), ctypes.byref(self.value)
            ),
            "create",
        )

    def close(self) -> None:
        if self.value.value:
            self.lib.spark_cache_placement_destroy(self.value)
            self.value = ctypes.c_void_p()

    def __enter__(self) -> "PlacementHandle":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _require_open(self) -> ctypes.c_void_p:
        if not self.value.value:
            raise NativePlacementError("placement handle is closed")
        return self.value

    def configure_destinations(
        self, destinations: Iterable[DestinationDescriptor]
    ) -> None:
        values = tuple(destinations)
        if not values:
            raise ValueError("at least one destination is required")
        descriptors = (DestinationDescriptor * len(values))(*values)
        check(
            self.lib,
            self.lib.spark_cache_placement_configure_destinations(
                self._require_open(), descriptors, len(values)
            ),
            "configure destinations",
            self.value,
        )

    def begin_restore(self, slots: Iterable[int]) -> None:
        values = tuple(slots)
        if not values:
            raise ValueError("at least one slot is required")
        if any(value < 0 or value > 0xFFFFFFFF for value in values):
            raise ValueError("slot values must fit uint32")
        slot_array = (ctypes.c_uint32 * len(values))(*values)
        check(
            self.lib,
            self.lib.spark_cache_placement_begin_restore(
                self._require_open(), slot_array, len(values)
            ),
            "begin restore",
            self.value,
        )

    def acquire_arena(self, index: int) -> ArenaView:
        view = ArenaView()
        check(
            self.lib,
            self.lib.spark_cache_placement_acquire_arena_view(
                self._require_open(), index, ctypes.byref(view)
            ),
            "acquire arena",
            self.value,
        )
        return view

    def parse_verified_chunk(
        self,
        view: ArenaView,
        *,
        arena_used_bytes: int,
        arena_offset_bytes: int,
        encoded_bytes: int,
        expected_logical_start: int,
        dcp_degree: int,
        dcp_rank: int,
        first_slot_index: int,
        required_data_record_mask: int,
    ) -> ChunkDescriptor:
        self._require_open()
        if view.host_address == 0:
            raise ValueError("arena view has no host address")
        output = ChunkDescriptor()
        error = ctypes.create_string_buffer(512)
        result = self.lib.spark_cache_parse_verified_v1_chunk(
            ctypes.c_void_p(view.host_address),
            arena_used_bytes,
            arena_offset_bytes,
            encoded_bytes,
            expected_logical_start,
            dcp_degree,
            dcp_rank,
            first_slot_index,
            required_data_record_mask,
            ctypes.byref(output),
            error,
            ctypes.sizeof(error),
        )
        if result != STATUS_OK:
            status = _decode(self.lib.spark_cache_placement_status_string(result))
            detail = error.value.decode("utf-8", errors="replace")
            raise NativePlacementError(
                "parse verified chunk failed: "
                f"status={result} ({status}) detail={detail}"
            )
        return output

    def submit_direct_slab(
        self,
        arena_index: int,
        arena_used_bytes: int,
        chunks: Iterable[ChunkDescriptor],
    ) -> None:
        values = tuple(chunks)
        if not values:
            raise ValueError("at least one chunk is required")
        descriptors = (ChunkDescriptor * len(values))(*values)
        check(
            self.lib,
            self.lib.spark_cache_placement_submit_direct_slab(
                self._require_open(),
                arena_index,
                arena_used_bytes,
                descriptors,
                len(values),
            ),
            "submit direct slab",
            self.value,
        )

    def submit_transposed_slab(
        self,
        arena_index: int,
        arena_used_bytes: int,
        sources: Iterable[TransposedSource],
    ) -> None:
        values = tuple(sources)
        if not values:
            raise ValueError("at least one transposed source is required")
        descriptors = (TransposedSource * len(values))(*values)
        check(
            self.lib,
            self.lib.spark_cache_placement_submit_transposed_slab(
                self._require_open(),
                arena_index,
                arena_used_bytes,
                descriptors,
                len(values),
            ),
            "submit transposed slab",
            self.value,
        )

    def finish_restore(self) -> PlacementStats:
        stats = PlacementStats()
        check(
            self.lib,
            self.lib.spark_cache_placement_finish_restore(
                self._require_open(), ctypes.byref(stats)
            ),
            "finish restore",
            self.value,
        )
        return stats

    def abort_restore(self) -> None:
        check(
            self.lib,
            self.lib.spark_cache_placement_abort_restore(self._require_open()),
            "abort restore",
            self.value,
        )

    def stats(self) -> PlacementStats:
        stats = PlacementStats()
        check(
            self.lib,
            self.lib.spark_cache_placement_get_stats(
                self._require_open(), ctypes.byref(stats)
            ),
            "stats",
            self.value,
        )
        return stats


__all__ = [
    "ABI_VERSION",
    "ARENA_COUNT",
    "ARENA_MANAGED",
    "ARENA_MAPPED_HOST",
    "ARENA_STAGED_DEVICE",
    "AbiInfo",
    "ArenaView",
    "ChunkDescriptor",
    "DestinationDescriptor",
    "NativePlacementError",
    "PlacementConfig",
    "PlacementHandle",
    "PlacementStats",
    "TransposedSource",
    "arena_memoryview",
    "check",
    "handle_error",
    "load_library",
    "pread_exact",
    "runtime_error",
]
