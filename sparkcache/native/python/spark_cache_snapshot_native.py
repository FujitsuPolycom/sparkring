"""Checksum-attested ctypes boundary for the SparkCache snapshot ring.

The CUDA library is deliberately separate from restore placement. Snapshot
publication is opportunistic: WOULD_BLOCK, NOT_READY, and DROPPED are normal
outcomes and must never stop inference.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import stat
import sys
from pathlib import Path
from typing import Any, Callable, Iterable


ABI_VERSION = 1
MIN_SLOTS = 2
MAX_SLOTS = 3
MAX_RECORD_KINDS = 4

STATUS_OK = 0
STATUS_WOULD_BLOCK = 4
STATUS_NOT_READY = 5
STATUS_DROPPED = 6

CAP_MAPPED_HOST = 1 << 0
CAP_EXTERNAL_STREAM = 1 << 2
CAP_NONBLOCKING_ACQUIRE = 1 << 3
CAP_CONTEXT_ABANDON = 1 << 4
CAP_ORDERLY_SHUTDOWN = 1 << 5


class SnapshotConfig(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("arena_mode", ctypes.c_uint32),
        ("slot_bytes", ctypes.c_uint64),
        ("slot_count", ctypes.c_uint32),
        ("max_sources", ctypes.c_uint32),
        ("max_rows", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


class SnapshotSource(ctypes.Structure):
    _fields_ = [
        ("source_base", ctypes.c_uint64),
        ("source_rows", ctypes.c_uint64),
        ("source_row_stride_bytes", ctypes.c_uint32),
        ("bytes_per_token", ctypes.c_uint32),
        ("record_kind", ctypes.c_uint32),
        ("source_layer_ordinal", ctypes.c_uint32),
    ]


class SnapshotSubmission(ctypes.Structure):
    _fields_ = [
        ("context_sequence", ctypes.c_uint64),
        ("logical_start", ctypes.c_uint64),
        ("row_count", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 2),
    ]


class SnapshotTicket(ctypes.Structure):
    _fields_ = [
        ("generation", ctypes.c_uint64),
        ("slot_index", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class SnapshotReadyView(ctypes.Structure):
    _fields_ = [
        ("host_address", ctypes.c_uint64),
        ("device_address", ctypes.c_uint64),
        ("capacity_bytes", ctypes.c_uint64),
        ("used_bytes", ctypes.c_uint64),
        ("context_sequence", ctypes.c_uint64),
        ("logical_start", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("row_count", ctypes.c_uint32),
        ("slot_index", ctypes.c_uint32),
        ("record_mask", ctypes.c_uint32),
        ("state", ctypes.c_uint32),
        ("record_offset_bytes", ctypes.c_uint32 * MAX_RECORD_KINDS),
        ("record_length_bytes", ctypes.c_uint32 * MAX_RECORD_KINDS),
    ]


class SnapshotStats(ctypes.Structure):
    _fields_ = [
        ("submitted_bytes", ctypes.c_uint64),
        ("completed_bytes", ctypes.c_uint64),
        ("released_bytes", ctypes.c_uint64),
        ("submissions", ctypes.c_uint64),
        ("claims", ctypes.c_uint64),
        ("releases", ctypes.c_uint64),
        ("would_block", ctypes.c_uint64),
        ("abandoned", ctypes.c_uint64),
        ("stale_tickets", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64 * 3),
    ]


class SnapshotAbiInfo(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("cudart_version", ctypes.c_uint32),
        ("min_slots", ctypes.c_uint32),
        ("max_slots", ctypes.c_uint32),
        ("max_record_kinds", ctypes.c_uint32),
        ("sizeof_config", ctypes.c_uint32),
        ("sizeof_source", ctypes.c_uint32),
        ("sizeof_submission", ctypes.c_uint32),
        ("sizeof_ticket", ctypes.c_uint32),
        ("sizeof_ready_view", ctypes.c_uint32),
        ("sizeof_stats", ctypes.c_uint32),
        ("capability_flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


EXPECTED_SIZES = {
    "sizeof_config": ctypes.sizeof(SnapshotConfig),
    "sizeof_source": ctypes.sizeof(SnapshotSource),
    "sizeof_submission": ctypes.sizeof(SnapshotSubmission),
    "sizeof_ticket": ctypes.sizeof(SnapshotTicket),
    "sizeof_ready_view": ctypes.sizeof(SnapshotReadyView),
    "sizeof_stats": ctypes.sizeof(SnapshotStats),
}


class NativeSnapshotError(RuntimeError):
    pass


def _sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    duplicate = os.dup(fd)
    with os.fdopen(duplicate, "rb", closefd=True) as source:
        source.seek(0)
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )


def _secure_fd_load_path(fd: int) -> str:
    if not sys.platform.startswith("linux"):
        raise NativeSnapshotError(
            "secure snapshot-library loading requires Linux /proc/self/fd; "
            f"platform {sys.platform!r} is fail-closed"
        )
    path = f"/proc/self/fd/{fd}"
    try:
        linked = os.stat(path)
    except OSError as error:
        raise NativeSnapshotError(
            "Linux /proc/self/fd is unavailable; refusing pathname dlopen"
        ) from error
    if _identity(linked) != _identity(os.fstat(fd)):
        raise NativeSnapshotError(
            "/proc/self/fd identity disagrees with attested descriptor"
        )
    return path


def _normalize_digest(expected_sha256: str) -> str:
    normalized = expected_sha256.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise NativeSnapshotError("expected_sha256 must be 64 lowercase hex")
    return normalized


def _dlopen_attested(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    cdll_factory: Callable[..., Any] = ctypes.CDLL,
) -> Any:
    """Hash and dlopen one retained inode, never two pathname resolutions."""

    resolved = Path(path).resolve(strict=True)
    normalized = _normalize_digest(expected_sha256)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(resolved, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise NativeSnapshotError(
                "snapshot library must be a regular file"
            )
        actual = _sha256_fd(fd)
        if actual != normalized:
            raise NativeSnapshotError(
                "snapshot library SHA-256 mismatch: "
                f"expected={normalized} actual={actual}"
            )
        load_path = _secure_fd_load_path(fd)
        mode = getattr(os, "RTLD_LOCAL", 0) | getattr(os, "RTLD_NOW", 0)
        library = cdll_factory(load_path, mode=mode)
        after = os.fstat(fd)
        if _identity(after) != _identity(before):
            raise NativeSnapshotError(
                "snapshot library inode changed while loading"
            )
        # Keep the exact attested inode open for the CDLL object's lifetime.
        # This also makes the ownership boundary observable in tests.
        owner = os.fdopen(fd, "rb", closefd=True)
        fd = -1
        library._spark_cache_snapshot_attested_file = owner
        library._spark_cache_snapshot_attested_identity = _identity(after)
        return library
    finally:
        if fd >= 0:
            os.close(fd)


def _bind(lib: ctypes.CDLL) -> None:
    handle = ctypes.c_void_p
    lib.spark_cache_snapshot_query_abi.argtypes = [
        ctypes.POINTER(SnapshotAbiInfo)
    ]
    lib.spark_cache_snapshot_query_abi.restype = ctypes.c_int
    lib.spark_cache_snapshot_create.argtypes = [
        ctypes.POINTER(SnapshotConfig),
        ctypes.POINTER(handle),
    ]
    lib.spark_cache_snapshot_create.restype = ctypes.c_int
    lib.spark_cache_snapshot_destroy.argtypes = [handle]
    lib.spark_cache_snapshot_destroy.restype = None
    lib.spark_cache_snapshot_shutdown.argtypes = [handle]
    lib.spark_cache_snapshot_shutdown.restype = ctypes.c_int
    lib.spark_cache_snapshot_configure_sources.argtypes = [
        handle,
        ctypes.POINTER(SnapshotSource),
        ctypes.c_uint32,
    ]
    lib.spark_cache_snapshot_configure_sources.restype = ctypes.c_int
    lib.spark_cache_snapshot_try_submit.argtypes = [
        handle,
        ctypes.POINTER(SnapshotSubmission),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint64,
        ctypes.POINTER(SnapshotTicket),
    ]
    lib.spark_cache_snapshot_try_submit.restype = ctypes.c_int
    for name in ("poll", "claim"):
        function = getattr(lib, f"spark_cache_snapshot_{name}")
        function.argtypes = [
            handle,
            ctypes.POINTER(SnapshotTicket),
            ctypes.POINTER(SnapshotReadyView),
        ]
        function.restype = ctypes.c_int
    lib.spark_cache_snapshot_release.argtypes = [
        handle,
        ctypes.POINTER(SnapshotTicket),
    ]
    lib.spark_cache_snapshot_release.restype = ctypes.c_int
    lib.spark_cache_snapshot_abandon_context.argtypes = [
        handle,
        ctypes.c_uint64,
    ]
    lib.spark_cache_snapshot_abandon_context.restype = ctypes.c_int
    lib.spark_cache_snapshot_get_stats.argtypes = [
        handle,
        ctypes.POINTER(SnapshotStats),
    ]
    lib.spark_cache_snapshot_get_stats.restype = ctypes.c_int
    lib.spark_cache_snapshot_status_string.argtypes = [ctypes.c_int]
    lib.spark_cache_snapshot_status_string.restype = ctypes.c_char_p


def _check(lib: ctypes.CDLL, result: int, operation: str) -> None:
    if result == STATUS_OK:
        return
    text = lib.spark_cache_snapshot_status_string(result)
    status = "" if text is None else text.decode("utf-8", errors="replace")
    raise NativeSnapshotError(
        f"{operation} failed: status={result} ({status})"
    )


def load_library(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
) -> tuple[ctypes.CDLL, SnapshotAbiInfo]:
    """Load only the exact immutable inode whose SHA-256 was attested."""

    lib = _dlopen_attested(path, expected_sha256=expected_sha256)
    try:
        _bind(lib)
        info = SnapshotAbiInfo()
        _check(
            lib,
            lib.spark_cache_snapshot_query_abi(ctypes.byref(info)),
            "ABI",
        )
        if (
            info.abi_version != ABI_VERSION
            or info.min_slots != MIN_SLOTS
            or info.max_slots != MAX_SLOTS
            or info.max_record_kinds != MAX_RECORD_KINDS
        ):
            raise NativeSnapshotError(
                "snapshot ABI constants disagree with binding"
            )
        mismatches = {
            field: (getattr(info, field), expected)
            for field, expected in EXPECTED_SIZES.items()
            if getattr(info, field) != expected
        }
        if mismatches:
            raise NativeSnapshotError(
                f"snapshot ABI size mismatch: {mismatches}"
            )
        required = (
            CAP_MAPPED_HOST
            | CAP_EXTERNAL_STREAM
            | CAP_NONBLOCKING_ACQUIRE
            | CAP_CONTEXT_ABANDON
            | CAP_ORDERLY_SHUTDOWN
        )
        if info.capability_flags & required != required:
            raise NativeSnapshotError(
                "snapshot library lacks required capabilities"
            )
        return lib, info
    except Exception:
        owner = getattr(
            lib, "_spark_cache_snapshot_attested_file", None
        )
        if owner is not None:
            owner.close()
        raise


def ready_memoryview(view: SnapshotReadyView) -> memoryview:
    if (
        not view.host_address
        or view.used_bytes == 0
        or view.used_bytes > view.capacity_bytes
    ):
        raise ValueError("invalid snapshot ready view")
    owner = (ctypes.c_ubyte * view.used_bytes).from_address(view.host_address)
    return memoryview(owner).cast("B").toreadonly()


def slot_array(values: Iterable[int]) -> ctypes.Array[ctypes.c_uint32]:
    materialized = list(values)
    if not materialized or any(value < 0 or value > 0xFFFFFFFF for value in materialized):
        raise ValueError("physical slots must be nonempty uint32 values")
    return (ctypes.c_uint32 * len(materialized))(*materialized)
