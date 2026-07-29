"""Fail-closed orchestration for native SparkCache restore placement.

This is the production path between a validated ``LookupResult`` and the
attested placement adapter.  It never imports the model-down experiment gate.
Each content-addressed chunk is read directly into a mapped host arena, its
manifest length and one whole-file SHA-256 are checked, and only then is the
native parser allowed to describe bytes to the scatter kernel.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import os
import re
import stat
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from spark_context_cache_native_placement import (
    NativePlacementAdapter,
    NativePlacementContractError,
    ParkedRestore,
    RestoreState,
)

try:
    import spark_cache_native as native
except ModuleNotFoundError as error:
    if error.name != "spark_cache_native":
        raise
    from spark_transport.experiments.cache_placement.native.python import (
        spark_cache_native as native,
    )


_CHUNK_PREFIX = struct.Struct("<8sII")
_CHUNK_MAGIC = b"SPCKV001"
_FORMAT_ABI = 1
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_DESCRIPTOR_KEYS = frozenset({"sha256", "bytes", "logical_start", "logical_end"})
_U32_MAX = (1 << 32) - 1


class NativeRestoreError(RuntimeError):
    """A native restore cannot safely complete and must be recomputed."""


@dataclass(frozen=True)
class NativeChunkPlan:
    path: Path
    sha256: str
    encoded_bytes: int
    logical_start: int
    logical_end: int
    payload_offset_bytes: int
    arena_offset_bytes: int
    first_slot_index: int
    row_count: int


@dataclass(frozen=True)
class NativeSlabPlan:
    chunks: tuple[NativeChunkPlan, ...]
    arena_used_bytes: int


@dataclass(frozen=True)
class NativeRestoreResult:
    placement_stats: Any
    verified_chunks: int
    verified_encoded_bytes: int
    slabs: int
    read_and_hash_ms: float
    parse_and_submit_ms: float
    finish_ms: float


def _strict_positive_int(value: Any, label: str, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > maximum
    ):
        raise NativeRestoreError(f"{label} must be an integer in [1, {maximum}]")
    return value


def _strict_nonnegative_int(value: Any, label: str, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > maximum
    ):
        raise NativeRestoreError(f"{label} must be an integer in [0, {maximum}]")
    return value


def _open_regular_chunk(path: Path, expected_bytes: int) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise NativeRestoreError(f"cannot open chunk {path}: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise NativeRestoreError(f"chunk is not a regular file: {path}")
        if metadata.st_size != expected_bytes:
            raise NativeRestoreError(
                f"chunk length changed for {path}: expected={expected_bytes},"
                f" actual={metadata.st_size}"
            )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_prefix(path: Path, expected_bytes: int) -> bytes:
    descriptor = _open_regular_chunk(path, expected_bytes)
    try:
        if hasattr(os, "pread"):
            return os.pread(descriptor, _CHUNK_PREFIX.size, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return os.read(descriptor, _CHUNK_PREFIX.size)
    finally:
        os.close(descriptor)


def _payload_offset(path: Path, expected_bytes: int) -> int:
    prefix = _read_prefix(path, expected_bytes)
    if len(prefix) != _CHUNK_PREFIX.size:
        raise NativeRestoreError(f"truncated chunk prefix: {path}")
    magic, abi, header_bytes = _CHUNK_PREFIX.unpack(prefix)
    if magic != _CHUNK_MAGIC or abi != _FORMAT_ABI:
        raise NativeRestoreError(f"unsupported chunk magic or ABI: {path}")
    payload_offset = _CHUNK_PREFIX.size + int(header_bytes)
    if payload_offset > expected_bytes:
        raise NativeRestoreError(f"chunk header exceeds encoded file: {path}")
    return payload_offset


def _aligned_arena_offset(cursor: int, payload_offset: int, alignment: int) -> int:
    return cursor + (-(cursor + payload_offset) % alignment)


def plan_native_restore(
    lookup: Any,
    *,
    cache_root: str | Path,
    expected_span_tokens: int,
    dcp_degree: int,
    arena_bytes: int,
    payload_alignment: int = 256,
) -> tuple[NativeSlabPlan, ...]:
    """Validate one lookup and pack its complete chunks into bounded slabs."""

    expected_span_tokens = _strict_positive_int(
        expected_span_tokens, "expected_span_tokens", maximum=_U32_MAX
    )
    dcp_degree = _strict_positive_int(dcp_degree, "dcp_degree", maximum=_U32_MAX)
    arena_bytes = _strict_positive_int(
        arena_bytes, "arena_bytes", maximum=(1 << 63) - 1
    )
    payload_alignment = _strict_positive_int(
        payload_alignment, "payload_alignment", maximum=1 << 20
    )
    if not getattr(lookup, "is_hit", False):
        raise NativeRestoreError("native restore requires a cache hit")
    manifest = getattr(lookup, "_manifest", None)
    if not isinstance(manifest, Mapping):
        raise NativeRestoreError("cache hit has no validated manifest")
    committed = manifest.get("committed_tokens")
    if committed != expected_span_tokens:
        raise NativeRestoreError(
            "manifest committed token count disagrees with restore plan:"
            f" expected={expected_span_tokens}, actual={committed}"
        )
    raw_descriptors = manifest.get("chunks")
    if not isinstance(raw_descriptors, list) or not raw_descriptors:
        raise NativeRestoreError("manifest has no chunk descriptors")

    chunk_root = (Path(cache_root) / "chunks").resolve()
    slabs: list[NativeSlabPlan] = []
    current: list[NativeChunkPlan] = []
    cursor = 0
    expected_logical_start = 0
    first_slot_index = 0

    for index, raw in enumerate(raw_descriptors):
        if not isinstance(raw, Mapping) or set(raw) != _DESCRIPTOR_KEYS:
            raise NativeRestoreError(
                f"manifest chunk {index} has an invalid descriptor"
            )
        digest = raw["sha256"]
        if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
            raise NativeRestoreError(f"manifest chunk {index} has an invalid SHA-256")
        encoded_bytes = _strict_positive_int(
            raw["bytes"], f"chunk {index} bytes", maximum=_U32_MAX
        )
        logical_start = _strict_nonnegative_int(
            raw["logical_start"],
            f"chunk {index} logical_start",
            maximum=_U32_MAX,
        )
        logical_end = _strict_positive_int(
            raw["logical_end"],
            f"chunk {index} logical_end",
            maximum=_U32_MAX,
        )
        logical_tokens = logical_end - logical_start
        if logical_start != expected_logical_start or logical_tokens <= 0:
            raise NativeRestoreError("manifest chunk ranges are not contiguous")
        if logical_tokens % dcp_degree:
            raise NativeRestoreError(f"chunk {index} token count is not DCP-divisible")
        row_count = logical_tokens // dcp_degree
        path = chunk_root / f"{digest}.spcc"
        payload_offset = _payload_offset(path, encoded_bytes)
        arena_offset = _aligned_arena_offset(cursor, payload_offset, payload_alignment)
        if current and arena_offset + encoded_bytes > arena_bytes:
            slabs.append(NativeSlabPlan(tuple(current), cursor))
            current = []
            cursor = 0
            arena_offset = _aligned_arena_offset(
                cursor, payload_offset, payload_alignment
            )
        if arena_offset + encoded_bytes > arena_bytes:
            raise NativeRestoreError(
                f"chunk {index} does not fit the configured native arena"
            )
        current.append(
            NativeChunkPlan(
                path=path,
                sha256=digest,
                encoded_bytes=encoded_bytes,
                logical_start=logical_start,
                logical_end=logical_end,
                payload_offset_bytes=payload_offset,
                arena_offset_bytes=arena_offset,
                first_slot_index=first_slot_index,
                row_count=row_count,
            )
        )
        cursor = arena_offset + encoded_bytes
        first_slot_index += row_count
        expected_logical_start = logical_end

    if expected_logical_start != expected_span_tokens:
        raise NativeRestoreError(
            "manifest chunks do not cover the expected restore span"
        )
    if current:
        slabs.append(NativeSlabPlan(tuple(current), cursor))
    if not slabs:
        raise NativeRestoreError("native restore produced no slabs")
    return tuple(slabs)


def _pread_exact_into(path: Path, expected_bytes: int, target: memoryview) -> None:
    descriptor = _open_regular_chunk(path, expected_bytes)
    try:
        if hasattr(os, "preadv"):
            native.pread_exact(descriptor, target)
            return
        # Repository tests run on Windows; Linux production always takes the
        # preadv branch above. readinto still writes directly into the arena.
        with os.fdopen(descriptor, "rb", buffering=0, closefd=False) as stream:
            completed = 0
            while completed < len(target):
                count = stream.readinto(target[completed:])
                if not count:
                    raise EOFError(
                        f"short read: received {completed} of {len(target)} bytes"
                    )
                completed += count
    except (EOFError, OSError, ValueError) as error:
        raise NativeRestoreError(
            f"direct arena read failed for {path}: {error}"
        ) from error
    finally:
        os.close(descriptor)


def _read_and_authenticate(chunk: NativeChunkPlan, target: memoryview) -> None:
    _pread_exact_into(chunk.path, chunk.encoded_bytes, target)
    actual = hashlib.sha256(target).hexdigest()
    if actual != chunk.sha256:
        raise NativeRestoreError(
            f"chunk SHA-256 mismatch for {chunk.path}:"
            f" expected={chunk.sha256}, actual={actual}"
        )


def _check_stats(
    stats: Any,
    *,
    slots: int,
    slabs: int,
    source_bytes: int,
) -> None:
    expected = {
        "source_bytes": source_bytes,
        "restored_rows": slots,
        "slot_uploads": 1,
        "destination_table_uploads": 1,
        "slabs_submitted": slabs,
        "scatter_kernel_launches": slabs,
        "device_error": 0,
        "staged_h2d_bytes": 0,
    }
    mismatches = {
        name: {"expected": value, "actual": int(getattr(stats, name, -1))}
        for name, value in expected.items()
        if int(getattr(stats, name, -1)) != value
    }
    if mismatches:
        raise NativeRestoreError(
            f"native placement statistics violate restore contract: {mismatches}"
        )


def execute_native_restore(
    *,
    adapter: NativePlacementAdapter | Any,
    request_id: str,
    lookup: Any,
    cache_root: str | Path,
    slots: Sequence[int],
    expected_span_tokens: int,
    dcp_degree: int,
    dcp_rank: int,
    arena_bytes: int,
    required_data_record_mask: int,
    io_workers: int = 8,
    payload_alignment: int = 256,
) -> NativeRestoreResult:
    """Install one lookup through a single fail-closed native transaction."""

    dcp_degree = _strict_positive_int(dcp_degree, "dcp_degree", maximum=_U32_MAX)
    dcp_rank = _strict_nonnegative_int(dcp_rank, "dcp_rank", maximum=_U32_MAX)
    if dcp_rank >= dcp_degree:
        raise NativeRestoreError("dcp_rank must be below dcp_degree")
    required_data_record_mask = _strict_positive_int(
        required_data_record_mask,
        "required_data_record_mask",
        maximum=(1 << native.MAX_RECORD_KINDS) - 1,
    )
    io_workers = _strict_positive_int(io_workers, "io_workers", maximum=32)
    slot_tuple = tuple(slots)
    slabs = plan_native_restore(
        lookup,
        cache_root=cache_root,
        expected_span_tokens=expected_span_tokens,
        dcp_degree=dcp_degree,
        arena_bytes=arena_bytes,
        payload_alignment=payload_alignment,
    )
    row_count = sum(chunk.row_count for slab in slabs for chunk in slab.chunks)
    if row_count != len(slot_tuple):
        raise NativeRestoreError(
            "manifest DCP row count disagrees with allocated slot count:"
            f" rows={row_count}, slots={len(slot_tuple)}"
        )

    verified_chunks = 0
    verified_encoded_bytes = 0
    read_and_hash_ms = 0.0
    parse_and_submit_ms = 0.0
    finish_ms = 0.0
    transaction: ParkedRestore | Any
    try:
        transaction = adapter.begin_parked_restore(request_id, slot_tuple)
        with transaction:
            for slab_index, slab in enumerate(slabs):
                arena_index = slab_index % native.ARENA_COUNT
                arena = transaction.acquire_arena(arena_index)
                if (
                    arena.arena_mode != native.ARENA_MAPPED_HOST
                    or arena.capacity_bytes != arena_bytes
                    or slab.arena_used_bytes > arena.capacity_bytes
                ):
                    raise NativeRestoreError(
                        "native mapped arena disagrees with configured capacity"
                    )
                arena_buffer = native.arena_memoryview(
                    arena, length=slab.arena_used_bytes
                )
                chunk_views = tuple(
                    arena_buffer[
                        chunk.arena_offset_bytes : chunk.arena_offset_bytes
                        + chunk.encoded_bytes
                    ]
                    for chunk in slab.chunks
                )
                started = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(io_workers, len(slab.chunks))
                ) as pool:
                    tuple(
                        pool.map(
                            lambda item: _read_and_authenticate(*item),
                            zip(slab.chunks, chunk_views, strict=True),
                        )
                    )
                read_and_hash_ms += 1e3 * (time.perf_counter() - started)
                verified_chunks += len(slab.chunks)
                verified_encoded_bytes += sum(
                    chunk.encoded_bytes for chunk in slab.chunks
                )
                # The C parser and placement own the native arena address;
                # release every exported Python buffer before native records
                # or reuses that arena.
                del chunk_views
                del arena_buffer

                started = time.perf_counter()
                parsed = []
                for chunk in slab.chunks:
                    output = transaction.parse_verified_chunk(
                        arena=arena,
                        arena_used_bytes=slab.arena_used_bytes,
                        arena_offset_bytes=chunk.arena_offset_bytes,
                        encoded_bytes=chunk.encoded_bytes,
                        expected_logical_start=chunk.logical_start,
                        dcp_degree=dcp_degree,
                        dcp_rank=dcp_rank,
                        first_slot_index=chunk.first_slot_index,
                        required_data_record_mask=required_data_record_mask,
                    )
                    if (
                        int(output.payload_offset_bytes) != chunk.payload_offset_bytes
                        or int(output.first_slot_index) != chunk.first_slot_index
                        or int(output.row_count) != chunk.row_count
                    ):
                        raise NativeRestoreError(
                            "native parser disagrees with the authenticated"
                            " manifest/slab plan"
                        )
                    parsed.append(output)
                transaction.submit_direct_slab(
                    arena_index=arena_index,
                    arena_used_bytes=slab.arena_used_bytes,
                    chunks=parsed,
                )
                parse_and_submit_ms += 1e3 * (time.perf_counter() - started)

            started = time.perf_counter()
            stats = transaction.finish()
            finish_ms = 1e3 * (time.perf_counter() - started)
            _check_stats(
                stats,
                slots=len(slot_tuple),
                slabs=len(slabs),
                source_bytes=sum(slab.arena_used_bytes for slab in slabs),
            )
            if (
                transaction.state is not RestoreState.FINISHED
                or not transaction.can_resume
            ):
                raise NativeRestoreError(
                    "native finish did not release the parked request"
                )
    except NativeRestoreError:
        raise
    except (
        NativePlacementContractError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise NativeRestoreError(f"native restore failed closed: {error}") from error

    return NativeRestoreResult(
        placement_stats=stats,
        verified_chunks=verified_chunks,
        verified_encoded_bytes=verified_encoded_bytes,
        slabs=len(slabs),
        read_and_hash_ms=read_and_hash_ms,
        parse_and_submit_ms=parse_and_submit_ms,
        finish_ms=finish_ms,
    )


__all__ = [
    "NativeChunkPlan",
    "NativeRestoreError",
    "NativeRestoreResult",
    "NativeSlabPlan",
    "execute_native_restore",
    "plan_native_restore",
]
