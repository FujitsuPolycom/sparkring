"""SparkCache CUDA restore and placement for authenticated opaque page snapshots."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from sparkcache import spark_cache_cuda as cuda
from sparkcache.page_base_read_flights import (
    PageBaseReadEvidence,
    PageBaseReadError,
    PageBaseReadResult,
)
from sparkcache.persistent_context_cache.cache_manifest import (
    CacheFormatError,
    CacheIdentity,
    StateRecord,
    _authenticated_delta_header,
    _hot_page_snapshot,
    _hot_page_snapshot_fits,
    _is_page_delta_root,
    _read_object_extent,
    _read_object_extent_into,
    _remember_authenticated_delta_header,
    _remember_hot_page_snapshot,
    _is_page_snapshot_root,
    _validate_page_delta_root,
    _validate_page_snapshot_root,
)
from sparkcache.spark_context_cache_hybrid import PageDeltaPlan, PageSnapshotPlan
from sparkcache.spark_context_cache_hybrid import (
    HybridCodecError,
    PageLayout,
    encode_page_snapshot_header,
    page_delta_header_prefix,
    page_snapshot_encoded_size,
    payload_matches,
    plan_page_delta,
    plan_page_snapshot,
)
from sparkcache.persistent_context_cache.cache_manifest import (
    _PAGE_TAIL_SCHEMAS,
    _descriptor_integrity,
    _read_header_prefix,
)
from sparkcache.spark_context_cache_cuda_placement import (
    CudaPlacementContractError,
    RestoreState,
)
from sparkcache.spark_context_cache_cuda_restore import plan_cuda_restore
from sparkcache.spark_context_cache_native_restore import (
    _pread_exact_into,
    _read_and_authenticate,
)

_CHUNK_PREFIX = struct.Struct("<8sII")
_CHUNK_MAGIC = b"SPCKV001"
_TARGET_KIND = cuda.RECORD_TARGET_CKV
_PAGE_SNAPSHOT_MANIFEST_SCHEMA = "sparkcache-page-snapshot-manifest/v2"
_PAGE_DELTA_MANIFEST_SCHEMA = "sparkcache-page-delta-manifest/v2"
_PAGE_DELTA_MANIFEST_SCHEMA_V3 = "sparkcache-page-delta-manifest/v3"
_MAX_PAGE_OBJECT_READ_WORKERS = 4
# Tail extents and stage-header objects are authenticated independently, so
# their reads and SHA-256 passes scale across cores; the bound keeps thread
# count and in-flight private buffers finite on a 20-core host.
_MAX_PAGE_TAIL_READ_WORKERS = 16
_MAX_PAGE_OBJECT_PREFETCH_BYTES = 256 * 1024 * 1024
_MAX_PAGE_DELTA_DEPTH = 2
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class CudaHybridRestoreError(RuntimeError):
    """SparkCache CUDA page placement was rejected before safe completion."""


@dataclass(frozen=True)
class CudaHybridRestoreResult:
    placement_stats: Any
    source_bytes: int
    copy_and_submit_ms: float
    finish_ms: float
    slabs: int = 1
    read_and_hash_ms: float = 0.0
    arena_wait_ms: float = 0.0
    host_copy_ms: float = 0.0
    submit_call_ms: float = 0.0
    read_source_bytes: int = 0
    skipped_base_object_bytes: int = 0
    # Sequential SHA-256 of the composed result, overlapped with the reads of
    # the following batch; counted separately from read_and_hash_ms.
    final_hash_ms: float = 0.0
    # Manifest validation, header authentication and base read before the
    # first batch; included in read_and_hash_ms.
    planning_ms: float = 0.0
    # True when the result was placed from the process-local hot snapshot
    # cache without reading or hashing any object.
    hot_snapshot_hit: bool = False


@dataclass(frozen=True)
class CudaPageSlab:
    payload_start: int
    payload_end: int
    spans: tuple[cuda.PageCopySpan, ...]


@dataclass(frozen=True)
class CudaPageObject:
    path: Path
    sha256: str
    encoded_bytes: int
    encoded_start: int
    encoded_end: int
    # Payload check the descriptor declares: "sha256" (content address),
    # "fast" (CRC32 in ``crc32``) or "none" (length and geometry only).
    integrity: str = "sha256"
    crc32: int | None = None


@dataclass(frozen=True)
class CudaPageTailExtent:
    """One delta layer tail authenticated by its header-bound SHA-256.

    ``path`` is a process-local cache key, not a file. The bytes come from
    ``pieces`` ``(object path, object bytes, object offset, length)`` and are
    usable only after their digest equals ``sha256``, which an authenticated
    delta header supplied.
    """

    path: Path
    sha256: str | None
    encoded_bytes: int
    encoded_start: int
    encoded_end: int
    pieces: tuple[tuple[Path, int, int, int], ...]
    integrity: str = "sha256"
    crc32: int | None = None


@dataclass(frozen=True)
class CudaAuthenticatedPageObject:
    source: CudaPageObject | CudaPageTailExtent
    payload: bytes


@dataclass(frozen=True)
class CudaPageSourceSpan:
    """One final logical payload range backed by an authenticated object."""

    source: CudaPageObject | CudaPageTailExtent
    source_offset_bytes: int
    snapshot_offset_bytes: int
    destination_byte_offset: int
    byte_count: int
    destination_index: int


@dataclass(frozen=True)
class CudaPageDeltaRestorePlan:
    page_plan: PageSnapshotPlan
    result_snapshot_sha256: str
    source_spans: tuple[CudaPageSourceSpan, ...]
    prefetched_objects: tuple[CudaAuthenticatedPageObject, ...]
    referenced_object_bytes: int
    skipped_base_object_bytes: int
    planning_read_source_bytes: int = 0
    # Payload integrity of the newest stage: "sha256" verifies the composed
    # result against its digest, "fast" and "none" rely on per-tail checks.
    integrity: str = "sha256"
    # The complete result from the process-local hot snapshot cache. When set,
    # source_spans is empty and the executor places these bytes directly.
    hot_snapshot: bytes | bytearray | None = None


@dataclass(frozen=True)
class _CudaPageDeltaStage:
    plan: PageDeltaPlan
    objects: tuple[CudaPageObject, ...]


@dataclass(frozen=True)
class _LogicalSourceRange:
    destination_start: int
    destination_end: int
    source_start: int
    objects: tuple[CudaPageObject, ...]


def _page_objects(
    descriptors: Sequence[Any],
    *,
    cache_root: Any,
    total_bytes: int,
    arena_bytes: int,
    label: str,
) -> tuple[CudaPageObject, ...]:
    if (
        isinstance(total_bytes, bool)
        or not isinstance(total_bytes, int)
        or total_bytes <= 0
        or arena_bytes <= 0
    ):
        raise CudaHybridRestoreError(f"{label} object geometry is invalid")
    root = (Path(cache_root) / "chunks").resolve()
    expected_start = 0
    objects = []
    expected_keys = {"sha256", "bytes", "encoded_start", "encoded_end"}
    for index, raw in enumerate(descriptors):
        if not isinstance(raw, dict) or set(raw) - {"crc32"} != expected_keys:
            raise CudaHybridRestoreError(
                f"{label} object {index} has an invalid descriptor"
            )
        digest = raw["sha256"]
        size = raw["bytes"]
        start = raw["encoded_start"]
        end = raw["encoded_end"]
        crc32 = raw.get("crc32")
        integrity = _descriptor_integrity(raw)
        if (
            not isinstance(digest, str)
            or _DIGEST.fullmatch(digest) is None
            or (integrity == "fast" and (type(crc32) is not int or not 0 <= crc32 <= 0xFFFFFFFF))
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or size > arena_bytes
            or size > _MAX_PAGE_OBJECT_PREFETCH_BYTES
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start != expected_start
            or end != start + size
        ):
            raise CudaHybridRestoreError(
                f"{label} object {index} geometry is invalid"
            )
        objects.append(
            CudaPageObject(
                path=root / f"{digest}.spcc",
                sha256=digest,
                encoded_bytes=size,
                encoded_start=start,
                encoded_end=end,
                integrity=integrity,
                crc32=crc32,
            )
        )
        expected_start = end
    if not objects or expected_start != total_bytes:
        raise CudaHybridRestoreError(f"{label} objects do not cover the payload")
    return tuple(objects)


def _source_payload_matches(source, data) -> bool:
    """Apply the source's declared payload check to ``data``."""

    if len(data) != source.encoded_bytes:
        return False
    return payload_matches(
        source.integrity, data, sha256=source.sha256, crc32=source.crc32
    )


def _read_authenticated_page_object(
    source: CudaPageObject,
) -> CudaAuthenticatedPageObject:
    payload = bytearray(source.encoded_bytes)
    view = memoryview(payload)
    try:
        _pread_exact_into(source.path, source.encoded_bytes, view)
    finally:
        view.release()
    if not _source_payload_matches(source, payload):
        raise CudaHybridRestoreError(
            f"page object SHA-256 mismatch for {source.path}"
        )
    return CudaAuthenticatedPageObject(source, bytes(payload))


def _tail_extents(
    objects: Sequence[CudaPageObject],
    plan: PageDeltaPlan,
    *,
    max_bytes: int,
) -> tuple[CudaPageTailExtent, ...] | None:
    """Return per-layer tail sources, or None when a tail exceeds a bound.

    Tail-granular sources let a restore read each surviving tail once instead
    of re-reading complete macro objects for every destination layer.
    """

    extents = []
    for tail in plan.tails:
        size = tail.source_end - tail.source_start
        if size <= 0 or size > max_bytes:
            return None
        pieces = []
        cursor = tail.source_start
        for source in objects:
            start = max(cursor, source.encoded_start)
            end = min(tail.source_end, source.encoded_end)
            if start >= end:
                continue
            pieces.append(
                (source.path, source.encoded_bytes, start - source.encoded_start,
                 end - start)
            )
            cursor = end
            if cursor == tail.source_end:
                break
        if cursor != tail.source_end:
            raise CudaHybridRestoreError("page-delta tail is not covered by objects")
        extents.append(
            CudaPageTailExtent(
                path=Path(
                    f"{objects[0].path}#tail-{tail.source_start}-{tail.source_end}"
                ),
                sha256=tail.sha256,
                encoded_bytes=size,
                encoded_start=tail.source_start,
                encoded_end=tail.source_end,
                pieces=tuple(pieces),
                integrity=plan.integrity,
                crc32=tail.crc32,
            )
        )
    return tuple(extents)


def _read_authenticated_source(
    source: CudaPageObject | CudaPageTailExtent,
) -> CudaAuthenticatedPageObject:
    if not isinstance(source, CudaPageTailExtent):
        return _read_authenticated_page_object(source)
    try:
        payload = _read_object_extent(source.pieces)
    except CacheFormatError as error:
        raise CudaHybridRestoreError(
            f"page-delta tail read failed for {source.path}: {error}"
        ) from error
    if not _source_payload_matches(source, payload):
        raise CudaHybridRestoreError(
            f"page-delta tail SHA-256 mismatch for {source.path}"
        )
    return CudaAuthenticatedPageObject(source, bytes(payload))


def _read_authenticated_tail_into(
    source: CudaPageTailExtent,
    target: memoryview,
) -> int:
    """Read one tail extent into ``target`` and authenticate it in place.

    ``target`` is a slice of a mapped placement arena. Nothing is submitted
    from that arena until every tail in its batch has passed this check, and
    a failure aborts the transaction, so the arena never carries unverified
    bytes into a submitted slab. Returns the bytes read.
    """

    try:
        try:
            _read_object_extent_into(source.pieces, target)
        except CacheFormatError as error:
            raise CudaHybridRestoreError(
                f"page-delta tail read failed for {source.path}: {error}"
            ) from error
        if not _source_payload_matches(source, target):
            raise CudaHybridRestoreError(
                f"page-delta tail SHA-256 mismatch for {source.path}"
            )
    finally:
        target.release()
    return source.encoded_bytes


def _authenticated_header_from_object(source: CudaPageObject) -> bytes:
    """Hash one stage's first object and return its header prefix.

    The object is read into a private buffer that is discarded afterwards;
    the header is trusted only because the whole object hashed to the
    manifest-bound content address.
    """

    payload = bytearray(source.encoded_bytes)
    view = memoryview(payload)
    try:
        _pread_exact_into(source.path, source.encoded_bytes, view)
    finally:
        view.release()
    if not _source_payload_matches(source, payload):
        raise CudaHybridRestoreError(
            f"page object SHA-256 mismatch for {source.path}"
        )
    return page_delta_header_prefix(payload)


def _read_authenticated_page_base(
    objects: Sequence[CudaPageObject],
) -> PageBaseReadResult:
    """Read every flat-base object and publish only authenticated bytes."""

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(_MAX_PAGE_OBJECT_READ_WORKERS, len(objects))
    ) as read_pool:
        authenticated = tuple(
            read_pool.map(_read_authenticated_page_object, objects)
        )
    return PageBaseReadResult(
        value=authenticated,
        encoded_bytes=sum(item.source.encoded_bytes for item in authenticated),
    )


def _hot_page_delta_plan(
    layout: PageLayout,
    root: dict[str, Any],
    stages: Sequence[_CudaPageDeltaStage],
    *,
    expected_counts: tuple[int, ...],
    planning_read_source_bytes: int,
) -> CudaPageDeltaRestorePlan | None:
    """Return a plan that places the hot-cached result, or None.

    The stage headers are authenticated and chain from the base snapshot's
    manifest-bound digest to the newest result digest. A process-local
    snapshot that hashed to that result digest is the result; it needs no
    base or tail read. Chain continuity is checked here exactly as the full
    planner checks it, so a hit never bypasses a validation the disk path
    performs on the metadata.
    """

    if not _is_page_snapshot_root(root):
        return None
    base_counts = stages[0].plan.base_block_counts
    if (
        root.get("committed_tokens") != stages[0].plan.base_boundary_tokens
        or root.get("snapshot_encoded_bytes")
        != page_snapshot_encoded_size(layout, base_counts)
        or root.get("snapshot_sha256") != stages[0].plan.base_snapshot_sha256
    ):
        raise CudaHybridRestoreError("page-delta base snapshot differs")
    current_counts = base_counts
    current_sha256 = root["snapshot_sha256"]
    for stage in stages:
        if (
            stage.plan.base_block_counts != current_counts
            or stage.plan.base_snapshot_sha256 != current_sha256
        ):
            raise CudaHybridRestoreError("page-delta stage base identity differs")
        current_counts = stage.plan.result_block_counts
        current_sha256 = stage.plan.result_snapshot_sha256
    if current_counts != expected_counts:
        raise CudaHybridRestoreError("page-delta result block counts differ")
    result_total = page_snapshot_encoded_size(layout, current_counts)
    hot = _hot_page_snapshot(current_sha256, result_total)
    if hot is None:
        return None
    page_plan = plan_page_snapshot(layout, hot, current_counts, total_bytes=result_total)
    return CudaPageDeltaRestorePlan(
        page_plan=page_plan,
        result_snapshot_sha256=current_sha256,
        source_spans=(),
        prefetched_objects=(),
        referenced_object_bytes=0,
        skipped_base_object_bytes=0,
        planning_read_source_bytes=planning_read_source_bytes,
        hot_snapshot=hot,
    )


def _validated_shared_page_base(
    result: PageBaseReadResult | bytes | None,
    objects: Sequence[CudaPageObject],
) -> tuple[CudaAuthenticatedPageObject, ...]:
    # The only producer is _read_authenticated_page_base, which hashes every
    # immutable object before publication. Re-hashing here would turn every
    # follower into another full-base authentication pass and defeat the
    # SparkCache CUDA flight; descriptor and length checks reject cross-base reuse.
    if not isinstance(result, PageBaseReadResult) or not isinstance(
        result.value, tuple
    ):
        raise CudaHybridRestoreError(
            "SparkCache CUDA page-base reader returned an incompatible representation"
        )
    authenticated = result.value
    if (
        len(authenticated) != len(objects)
        or any(
            not isinstance(item, CudaAuthenticatedPageObject)
            or item.source != source
            or not isinstance(item.payload, bytes)
            or len(item.payload) != source.encoded_bytes
            for item, source in zip(authenticated, objects, strict=True)
        )
        or result.encoded_bytes != sum(item.encoded_bytes for item in objects)
    ):
        raise CudaHybridRestoreError(
            "SparkCache CUDA page-base reader returned incompatible authenticated objects"
        )
    return authenticated


def _split_logical_source_range(
    source_range: _LogicalSourceRange,
    *,
    snapshot_base: int,
    destination_index: int,
) -> tuple[CudaPageSourceSpan, ...]:
    source_start = source_range.source_start
    source_end = source_start + (
        source_range.destination_end - source_range.destination_start
    )
    cursor = source_start
    result = []
    for source in source_range.objects:
        start = max(cursor, source.encoded_start)
        end = min(source_end, source.encoded_end)
        if start >= end:
            continue
        logical_offset = start - source_start
        result.append(
            CudaPageSourceSpan(
                source=source,
                source_offset_bytes=start - source.encoded_start,
                snapshot_offset_bytes=(
                    snapshot_base + source_range.destination_start + logical_offset
                ),
                destination_byte_offset=(
                    source_range.destination_start + logical_offset
                ),
                byte_count=end - start,
                destination_index=destination_index,
            )
        )
        cursor = end
        if cursor == source_end:
            break
    if cursor != source_end:
        raise CudaHybridRestoreError("page source objects do not cover a logical span")
    return tuple(result)


def plan_cuda_page_delta_restore(
    lookup: Any,
    *,
    cache_root: Any,
    layout: PageLayout,
    group_slots: Sequence[Sequence[int]],
    expected_span_tokens: int,
    arena_bytes: int,
    base_reader: Callable[
        [PageBaseReadEvidence, Callable[[], PageBaseReadResult]],
        PageBaseReadResult | bytes | None,
    ]
    | None = None,
) -> CudaPageDeltaRestorePlan:
    """Authenticate a page-delta graph and plan final-order source fragments."""

    manifest = getattr(lookup, "_manifest", None)
    if (
        not getattr(lookup, "is_hit", False)
        or getattr(lookup, "root_kind", None) != "page_delta"
        or not isinstance(manifest, dict)
        or manifest.get("schema")
        not in (_PAGE_DELTA_MANIFEST_SCHEMA, _PAGE_DELTA_MANIFEST_SCHEMA_V3)
        or manifest.get("committed_tokens") != expected_span_tokens
    ):
        raise CudaHybridRestoreError(
            "page-delta direct restore requires a compatible cache hit"
        )
    try:
        identity_wire = dict(manifest["identity"])
        if "record_schema" in identity_wire:
            identity_wire["record_schema"] = tuple(identity_wire["record_schema"])
        identity = CacheIdentity(**identity_wire)
        context_digest = manifest["context_digest"]
        if (
            not isinstance(context_digest, str)
            or _DIGEST.fullmatch(context_digest) is None
        ):
            raise ValueError("manifest digest fields are invalid")
        manifest_path = (
            Path(cache_root)
            / "manifests"
            / identity.storage_key
            / f"{context_digest}.json"
        )
        encoded_manifest = manifest_path.read_bytes()
        persisted = json.loads(encoded_manifest)
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise CudaHybridRestoreError(
            f"page-delta root identity was rejected: {error}"
        ) from error
    if (
        identity.publication_schema not in _PAGE_TAIL_SCHEMAS
        or identity.required_records
        != frozenset((StateRecord.TARGET_CKV, StateRecord.LOGICAL_POSITIONS))
        or persisted != manifest
        or hashlib.sha256(encoded_manifest).hexdigest()
        != getattr(lookup, "manifest_digest", None)
    ):
        raise CudaHybridRestoreError("page-delta root identity is not authenticated")

    stages_newest_first: list[_CudaPageDeltaStage] = []
    prefetched: OrderedDict[Path, CudaAuthenticatedPageObject] = OrderedDict()
    prefetch_limit = min(arena_bytes, _MAX_PAGE_OBJECT_PREFETCH_BYTES)
    prefetched_bytes = 0
    planning_read_source_bytes = 0

    def read_header_object(source: CudaPageObject) -> CudaAuthenticatedPageObject:
        nonlocal prefetched_bytes, planning_read_source_bytes
        cached = prefetched.get(source.path)
        if cached is not None:
            return CudaAuthenticatedPageObject(source, cached.payload)
        # Full-object authentication is required before trusting a header.
        # Retain only a bounded working set as flat histories can grow without
        # the nested-manifest depth limit. Evict before allocating the read.
        while prefetched and prefetched_bytes + source.encoded_bytes > prefetch_limit:
            _, evicted = prefetched.popitem(last=False)
            prefetched_bytes -= evicted.source.encoded_bytes
            del evicted
        authenticated = _read_authenticated_page_object(source)
        planning_read_source_bytes += source.encoded_bytes
        prefetched[source.path] = authenticated
        prefetched_bytes += source.encoded_bytes
        return authenticated

    def delta_header(source: CudaPageObject, header_sha256: str | None = None) -> bytes:
        # A cached header was taken from bytes hashed to this object's
        # content address, or checked against the manifest-bound header
        # digest; tails are still checked against its digests or checksums.
        cached = _authenticated_delta_header(source.sha256, source.encoded_bytes)
        if cached is not None:
            return cached
        if header_sha256 is not None:
            try:
                header = _read_header_prefix(source.path, header_sha256)
            except CacheFormatError as error:
                raise CudaHybridRestoreError(
                    f"page-delta header rejected for {source.path}: {error}"
                ) from error
        else:
            header = page_delta_header_prefix(read_header_object(source).payload)
        _remember_authenticated_delta_header(
            source.sha256, source.encoded_bytes, header
        )
        return header

    root = manifest
    root_digest = context_digest
    try:
        if root.get("schema") == _PAGE_DELTA_MANIFEST_SCHEMA_V3:
            base_root, _descriptors = _validate_page_delta_root(
                root,
                identity=identity,
                context_digest=root_digest,
            )
            flat_stages: list[
                tuple[dict[str, Any], tuple[CudaPageObject, ...]]
            ] = []
            for stage in reversed(root["delta_stages"]):
                objects = _page_objects(
                    stage["delta_objects"],
                    cache_root=cache_root,
                    total_bytes=stage["delta_encoded_bytes"],
                    arena_bytes=arena_bytes,
                    label="flat page delta",
                )
                flat_stages.append((stage, objects))
            # A process that has not seen this history must hash one whole
            # first object per stage to trust its header. Those objects are
            # independent, so authenticate them concurrently; nothing is
            # retained beyond the header prefix.
            unseen = []
            seen_paths = set()
            for stage, objects in flat_stages:
                first = objects[0]
                if (
                    stage.get("header_sha256") is not None
                    or first.path in seen_paths
                    or first.path in prefetched
                    or _authenticated_delta_header(first.sha256, first.encoded_bytes)
                    is not None
                ):
                    continue
                seen_paths.add(first.path)
                unseen.append(first)
            if len(unseen) > 1:
                # Each worker holds one whole object while hashing it, so the
                # transient working set is bounded by the object-read lane.
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(_MAX_PAGE_OBJECT_READ_WORKERS, len(unseen))
                ) as header_pool:
                    headers = tuple(
                        header_pool.map(_authenticated_header_from_object, unseen)
                    )
                for source, header in zip(unseen, headers, strict=True):
                    _remember_authenticated_delta_header(
                        source.sha256, source.encoded_bytes, header
                    )
                    planning_read_source_bytes += source.encoded_bytes
                del headers
            for stage, objects in flat_stages:
                delta_plan = plan_page_delta(
                    layout,
                    delta_header(objects[0], stage.get("header_sha256")),
                    base_block_counts=stage["base_block_counts"],
                    result_block_counts=stage["result_block_counts"],
                    base_boundary_tokens=stage["base_committed_tokens"],
                    result_boundary_tokens=stage["committed_tokens"],
                    total_bytes=stage["delta_encoded_bytes"],
                )
                stages_newest_first.append(
                    _CudaPageDeltaStage(delta_plan, objects)
                )
            root_digest = root["base_context_digest"]
            root = base_root
        else:
            for depth in range(_MAX_PAGE_DELTA_DEPTH):
                if not _is_page_delta_root(root):
                    break
                if root.get("schema") != _PAGE_DELTA_MANIFEST_SCHEMA:
                    raise CudaHybridRestoreError(
                        "page-delta direct restore requires v2 macro objects"
                    )
                if root.get("layout_sha256") != layout.digest:
                    raise CudaHybridRestoreError("page-delta root layout differs")
                base_root, descriptors = _validate_page_delta_root(
                    root,
                    identity=identity,
                    context_digest=root_digest,
                )
                objects = _page_objects(
                    descriptors,
                    cache_root=cache_root,
                    total_bytes=root["delta_encoded_bytes"],
                    arena_bytes=arena_bytes,
                    label="page delta",
                )
                delta_plan = plan_page_delta(
                    layout,
                    delta_header(objects[0]),
                    base_block_counts=root["base_block_counts"],
                    result_block_counts=root["result_block_counts"],
                    base_boundary_tokens=root["base_committed_tokens"],
                    result_boundary_tokens=root["committed_tokens"],
                    total_bytes=root["delta_encoded_bytes"],
                )
                stages_newest_first.append(
                    _CudaPageDeltaStage(delta_plan, objects)
                )
                if base_root.get("committed_tokens") != root["base_committed_tokens"]:
                    raise CudaHybridRestoreError(
                        "page-delta base boundary differs"
                    )
                root_digest = root["base_context_digest"]
                root = base_root
        if _is_page_delta_root(root):
            raise CudaHybridRestoreError("page-delta graph exceeds the depth limit")
        if not _is_page_snapshot_root(root):
            raise CudaHybridRestoreError(
                "page-delta direct restore requires a v2 page-snapshot base"
            )
        base_descriptors = _validate_page_snapshot_root(
            root,
            identity=identity,
            context_digest=root_digest,
        )
        base_objects = _page_objects(
            base_descriptors,
            cache_root=cache_root,
            total_bytes=root["snapshot_encoded_bytes"],
            arena_bytes=arena_bytes,
            label="page snapshot",
        )
        stages = tuple(reversed(stages_newest_first))
        if not stages:
            raise CudaHybridRestoreError("page-delta graph contains no delta")
        expected_counts = tuple(len(group) for group in group_slots)
        hot_plan = _hot_page_delta_plan(
            layout,
            root,
            stages,
            expected_counts=expected_counts,
            planning_read_source_bytes=planning_read_source_bytes,
        )
        if hot_plan is not None:
            return hot_plan
        shared_base_used = base_reader is not None and (
            len(stages) == 1
            or manifest.get("schema") == _PAGE_DELTA_MANIFEST_SCHEMA_V3
        )
        if shared_base_used:
            evidence = PageBaseReadEvidence(
                identity_storage_key=identity.storage_key,
                base_context_digest=manifest["base_context_digest"],
                base_root_sha256=manifest["base_root_sha256"],
                base_root_kind="page_snapshot",
                layout_sha256=manifest["layout_sha256"],
                base_block_counts=tuple(manifest["base_block_counts"]),
                base_boundary_tokens=manifest["base_committed_tokens"],
                base_encoded_bytes=root["snapshot_encoded_bytes"],
            )
            try:
                shared_base = base_reader(
                    evidence,
                    lambda: _read_authenticated_page_base(base_objects),
                )
            except PageBaseReadError as error:
                raise CudaHybridRestoreError(
                    f"SparkCache CUDA shared page-base read was rejected: {error}"
                ) from error
            shared_base_used = shared_base is not None
            if shared_base_used:
                authenticated_base_objects = _validated_shared_page_base(
                    shared_base,
                    base_objects,
                )
                prefetched.update(
                    (item.source.path, item)
                    for item in authenticated_base_objects
                )
                planning_read_source_bytes += sum(
                    item.source.encoded_bytes for item in authenticated_base_objects
                )
                authenticated_base = authenticated_base_objects[0]
        if not shared_base_used:
            authenticated_base = read_header_object(base_objects[0])
        base_counts = stages[0].plan.base_block_counts
        base_page_plan = plan_page_snapshot(
            layout,
            authenticated_base.payload,
            base_counts,
            total_bytes=root["snapshot_encoded_bytes"],
        )
        if (
            root["committed_tokens"] != stages[0].plan.base_boundary_tokens
            or root["snapshot_encoded_bytes"]
            != page_snapshot_encoded_size(layout, base_counts)
            or root["snapshot_sha256"] != stages[0].plan.base_snapshot_sha256
        ):
            raise CudaHybridRestoreError("page-delta base snapshot differs")
    except (CacheFormatError, HybridCodecError, KeyError, TypeError, ValueError) as error:
        raise CudaHybridRestoreError(
            f"page-delta descriptor planning was rejected: {error}"
        ) from error

    ranges: list[list[_LogicalSourceRange]] = []
    for span in base_page_plan.spans:
        ranges.append(
            [
                _LogicalSourceRange(
                    destination_start=0,
                    destination_end=span.source_end - span.source_start,
                    source_start=span.source_start,
                    objects=base_objects,
                )
            ]
        )
    current_counts = base_counts
    current_sha256 = root["snapshot_sha256"]
    tail_bound = min(arena_bytes, _MAX_PAGE_OBJECT_PREFETCH_BYTES)
    for stage in stages:
        if (
            stage.plan.base_block_counts != current_counts
            or stage.plan.base_snapshot_sha256 != current_sha256
        ):
            raise CudaHybridRestoreError("page-delta stage base identity differs")
        extents = _tail_extents(stage.objects, stage.plan, max_bytes=tail_bound)
        for destination_index, tail in enumerate(stage.plan.tails):
            existing = ranges[destination_index]
            cutoff = tail.destination_byte_offset
            kept = []
            for item in existing:
                if item.destination_start >= cutoff:
                    break
                end = min(item.destination_end, cutoff)
                kept.append(
                    _LogicalSourceRange(
                        destination_start=item.destination_start,
                        destination_end=end,
                        source_start=item.source_start,
                        objects=item.objects,
                    )
                )
                if end == cutoff:
                    break
            expected_prefix = (
                kept[-1].destination_end if kept else 0
            )
            if expected_prefix != cutoff:
                raise CudaHybridRestoreError("page-delta prefix coverage differs")
            kept.append(
                _LogicalSourceRange(
                    destination_start=cutoff,
                    destination_end=cutoff + tail.source_end - tail.source_start,
                    source_start=tail.source_start,
                    objects=(
                        stage.objects
                        if extents is None
                        else (extents[destination_index],)
                    ),
                )
            )
            ranges[destination_index] = kept
        current_counts = stage.plan.result_block_counts
        current_sha256 = stage.plan.result_snapshot_sha256

    expected_counts = tuple(len(group) for group in group_slots)
    if current_counts != expected_counts:
        raise CudaHybridRestoreError("page-delta result block counts differ")
    result_header = encode_page_snapshot_header(layout, current_counts)
    result_total = page_snapshot_encoded_size(layout, current_counts)
    page_plan = plan_page_snapshot(
        layout,
        result_header,
        current_counts,
        total_bytes=result_total,
    )
    source_spans = []
    snapshot_base = 0
    for destination_index, (layer_plan, layer_ranges) in enumerate(
        zip(page_plan.spans, ranges, strict=True)
    ):
        expected_destination = 0
        for item in layer_ranges:
            if item.destination_start != expected_destination:
                raise CudaHybridRestoreError("page-delta result coverage has a gap")
            source_spans.extend(
                _split_logical_source_range(
                    item,
                    snapshot_base=snapshot_base,
                    destination_index=destination_index,
                )
            )
            expected_destination = item.destination_end
        expected_bytes = layer_plan.source_end - layer_plan.source_start
        if expected_destination != expected_bytes:
            raise CudaHybridRestoreError("page-delta result coverage differs")
        snapshot_base += expected_bytes
    if snapshot_base != result_total - page_plan.header_bytes:
        raise CudaHybridRestoreError("page-delta result byte count differs")
    referenced = {span.source.path: span.source for span in source_spans}
    base_paths = {source.path: source for source in base_objects}
    return CudaPageDeltaRestorePlan(
        page_plan=page_plan,
        result_snapshot_sha256=current_sha256,
        source_spans=tuple(source_spans),
        prefetched_objects=tuple(prefetched.values()),
        integrity=stages[-1].plan.integrity,
        referenced_object_bytes=sum(
            source.encoded_bytes for source in referenced.values()
        ),
        skipped_base_object_bytes=sum(
            source.encoded_bytes
            for path, source in base_paths.items()
            if (
                not shared_base_used
                and path not in referenced
                and path != base_objects[0].path
            )
        ),
        planning_read_source_bytes=planning_read_source_bytes,
    )


def build_page_copy_spans(plan: PageSnapshotPlan) -> tuple[cuda.PageCopySpan, ...]:
    """Map authenticated snapshot payload ranges onto page destinations."""

    return tuple(
        cuda.PageCopySpan(
            span.source_start - plan.header_bytes,
            span.source_start - plan.header_bytes,
            0,
            span.source_end - span.source_start,
            destination_index,
            0,
        )
        for destination_index, span in enumerate(plan.spans)
    )


def plan_page_slabs(
    plan: PageSnapshotPlan,
    *,
    arena_bytes: int,
) -> tuple[CudaPageSlab, ...]:
    if arena_bytes <= 0:
        raise CudaHybridRestoreError(
            "SparkCache CUDA placement arena bytes must be positive"
        )
    payload_bytes = plan.total_bytes - plan.header_bytes
    slabs = []
    for slab_start in range(0, payload_bytes, arena_bytes):
        slab_end = min(payload_bytes, slab_start + arena_bytes)
        spans = []
        for destination_index, layer in enumerate(plan.spans):
            layer_start = layer.source_start - plan.header_bytes
            layer_end = layer.source_end - plan.header_bytes
            start = max(slab_start, layer_start)
            end = min(slab_end, layer_end)
            if start >= end:
                continue
            spans.append(
                cuda.PageCopySpan(
                    start - slab_start,
                    start,
                    start - layer_start,
                    end - start,
                    destination_index,
                    0,
                )
            )
        if not spans:
            raise CudaHybridRestoreError("SparkCache CUDA page slab has no copy spans")
        slabs.append(CudaPageSlab(slab_start, slab_end, tuple(spans)))
    return tuple(slabs)


def build_page_object_spans(
    plan: PageSnapshotPlan,
    *,
    encoded_start: int,
    encoded_end: int,
) -> tuple[cuda.PageCopySpan, ...]:
    """Map one authenticated flat-object range onto page destinations."""

    spans = []
    for destination_index, layer in enumerate(plan.spans):
        start = max(encoded_start, layer.source_start)
        end = min(encoded_end, layer.source_end)
        if start >= end:
            continue
        spans.append(
            cuda.PageCopySpan(
                start - encoded_start,
                start - plan.header_bytes,
                start - layer.source_start,
                end - start,
                destination_index,
                0,
            )
        )
    return tuple(spans)


def execute_cuda_hybrid_placement(
    *,
    adapter: Any,
    request_id: str,
    encoded_pages: bytes,
    plan: PageSnapshotPlan,
    group_slots: Sequence[Sequence[int]],
) -> CudaHybridRestoreResult:
    """Copy one verified snapshot into a mapped arena and scatter it once."""

    if len(encoded_pages) != plan.total_bytes:
        raise CudaHybridRestoreError("hybrid snapshot length differs from page plan")
    payload_bytes = plan.total_bytes - plan.header_bytes
    # Slicing a bytes object allocates and copies the slice before the arena
    # copy; a memoryview slice is a window onto the same bytes.
    source = memoryview(encoded_pages).cast("B")
    try:
        transaction = adapter.begin_parked_page_restore(
            request_id,
            group_slots,
            snapshot_bytes=payload_bytes,
        )
        with transaction:
            started = time.perf_counter()
            first_arena = transaction.acquire_arena(0)
            if first_arena.arena_mode != cuda.ARENA_MAPPED_HOST:
                raise CudaHybridRestoreError(
                    "SparkCache CUDA restore requires a mapped-host arena"
                )
            slabs = plan_page_slabs(plan, arena_bytes=first_arena.capacity_bytes)
            for slab_index, slab in enumerate(slabs):
                arena_index = slab_index % cuda.ARENA_COUNT
                arena = (
                    first_arena
                    if slab_index == 0
                    else transaction.acquire_arena(arena_index)
                )
                used = slab.payload_end - slab.payload_start
                arena_buffer = cuda.arena_memoryview(arena, length=used)
                source_start = plan.header_bytes + slab.payload_start
                source_end = plan.header_bytes + slab.payload_end
                arena_buffer[:] = source[source_start:source_end]
                arena_buffer.release()
                transaction.submit_page_slab(
                    arena_index=arena_index,
                    arena_used_bytes=used,
                    spans=slab.spans,
                )
            copy_and_submit_ms = 1e3 * (time.perf_counter() - started)
            started = time.perf_counter()
            stats = transaction.finish()
            finish_ms = 1e3 * (time.perf_counter() - started)
            if (
                transaction.state is not RestoreState.FINISHED
                or not transaction.can_resume
            ):
                raise CudaHybridRestoreError(
                    "SparkCache CUDA placement did not release the parked request"
                )
    except CudaHybridRestoreError:
        raise
    except (CudaPlacementContractError, RuntimeError, TypeError, ValueError) as error:
        raise CudaHybridRestoreError(
            f"SparkCache CUDA page placement was rejected: {error}"
        ) from error
    finally:
        source.release()
    if (
        int(stats.slot_uploads) != 1
        or int(stats.destination_table_uploads) != 1
        or int(stats.slabs_submitted) != len(slabs)
        or int(stats.scatter_kernel_launches) != len(slabs)
        or int(stats.device_error) != 0
        or int(stats.staged_h2d_bytes) != 0
    ):
        raise CudaHybridRestoreError(
            "SparkCache CUDA placement statistics violate the mapped transaction"
        )
    return CudaHybridRestoreResult(
        placement_stats=stats,
        source_bytes=len(encoded_pages),
        copy_and_submit_ms=copy_and_submit_ms,
        finish_ms=finish_ms,
        slabs=len(slabs),
        read_and_hash_ms=0.0,
    )


def _target_record_plan(path: Any, encoded_bytes: int) -> tuple[int, int, int]:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        prefix = os.pread(fd, _CHUNK_PREFIX.size, 0)
        if len(prefix) != _CHUNK_PREFIX.size:
            raise CudaHybridRestoreError("hybrid chunk prefix is truncated")
        magic, abi, header_bytes = _CHUNK_PREFIX.unpack(prefix)
        if magic != _CHUNK_MAGIC or abi != 1 or header_bytes > 1 << 20:
            raise CudaHybridRestoreError("hybrid chunk prefix is unsupported")
        raw = os.pread(fd, header_bytes, _CHUNK_PREFIX.size)
        if len(raw) != header_bytes:
            raise CudaHybridRestoreError("hybrid chunk header is truncated")
        header = json.loads(raw)
        payload_offset = _CHUNK_PREFIX.size + header_bytes
        for record in header.get("records", ()):
            if record.get("kind") == "target_ckv":
                offset = int(record["offset"])
                length = int(record["length"])
                if (
                    offset < 0
                    or length <= 0
                    or payload_offset + offset + length > encoded_bytes
                ):
                    break
                return payload_offset, offset, length
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise CudaHybridRestoreError(
            f"hybrid chunk planning did not complete: {error}"
        ) from error
    finally:
        os.close(fd)
    raise CudaHybridRestoreError("hybrid chunk has no valid target_ckv record")


def _plan_page_objects(
    lookup: Any,
    *,
    cache_root: Any,
    expected_span_tokens: int,
    arena_bytes: int,
) -> tuple[tuple[CudaPageObject, ...], int]:
    manifest = getattr(lookup, "_manifest", None)
    if (
        not getattr(lookup, "is_hit", False)
        or getattr(lookup, "root_kind", None) != "page_snapshot"
        or not isinstance(manifest, dict)
        or manifest.get("schema") != _PAGE_SNAPSHOT_MANIFEST_SCHEMA
        or manifest.get("committed_tokens") != expected_span_tokens
    ):
        raise CudaHybridRestoreError(
            "flat page macro restore requires a compatible cache hit"
        )
    try:
        identity_wire = dict(manifest["identity"])
        if "record_schema" in identity_wire:
            identity_wire["record_schema"] = tuple(identity_wire["record_schema"])
        identity = CacheIdentity(**identity_wire)
        context_digest = manifest["context_digest"]
        if (
            not isinstance(context_digest, str)
            or _DIGEST.fullmatch(context_digest) is None
        ):
            raise ValueError("manifest digest fields are invalid")
        validated_objects = _validate_page_snapshot_root(
            manifest,
            identity=identity,
            context_digest=context_digest,
        )
        manifest_path = (
            Path(cache_root)
            / "manifests"
            / identity.storage_key
            / f"{context_digest}.json"
        )
        encoded_manifest = manifest_path.read_bytes()
        persisted = json.loads(encoded_manifest)
    except (
        CacheFormatError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise CudaHybridRestoreError(
            f"flat page root identity was rejected: {error}"
        ) from error
    if (
        identity.publication_schema not in ("", *_PAGE_TAIL_SCHEMAS)
        or identity.required_records
        != frozenset((StateRecord.TARGET_CKV, StateRecord.LOGICAL_POSITIONS))
        or persisted != manifest
        or hashlib.sha256(encoded_manifest).hexdigest()
        != getattr(lookup, "manifest_digest", None)
    ):
        raise CudaHybridRestoreError("flat page root identity is not authenticated")
    total = manifest.get("snapshot_encoded_bytes")
    raw_objects = validated_objects
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total <= 0
        or not raw_objects
    ):
        raise CudaHybridRestoreError("flat page macro geometry is invalid")
    root = (Path(cache_root) / "chunks").resolve()
    expected_start = 0
    objects = []
    expected_keys = {"sha256", "bytes", "encoded_start", "encoded_end"}
    for index, raw in enumerate(raw_objects):
        if not isinstance(raw, dict) or set(raw) - {"crc32"} != expected_keys:
            raise CudaHybridRestoreError(
                f"flat page object {index} has an invalid descriptor"
            )
        digest = raw["sha256"]
        size = raw["bytes"]
        start = raw["encoded_start"]
        end = raw["encoded_end"]
        crc32 = raw.get("crc32")
        integrity = _descriptor_integrity(raw)
        if (
            not isinstance(digest, str)
            or _DIGEST.fullmatch(digest) is None
            or (integrity == "fast" and (type(crc32) is not int or not 0 <= crc32 <= 0xFFFFFFFF))
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or size > arena_bytes
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start != expected_start
            or end != start + size
        ):
            raise CudaHybridRestoreError(
                f"flat page object {index} geometry is invalid"
            )
        objects.append(
            CudaPageObject(
                path=root / f"{digest}.spcc",
                sha256=digest,
                encoded_bytes=size,
                encoded_start=start,
                encoded_end=end,
                integrity=integrity,
                crc32=crc32,
            )
        )
        expected_start = end
    if expected_start != total:
        raise CudaHybridRestoreError("flat page objects do not cover the snapshot")
    return tuple(objects), total


def _execute_page_object_restore(
    *,
    adapter: Any,
    request_id: str,
    lookup: Any,
    cache_root: Any,
    layout: PageLayout,
    group_slots: Sequence[Sequence[int]],
    expected_span_tokens: int,
    arena_bytes: int,
    io_workers: int,
) -> CudaHybridRestoreResult:
    """Authenticate each flat extent before submitting its page-copy spans."""

    planning_started = time.perf_counter()
    objects, snapshot_bytes = _plan_page_objects(
        lookup,
        cache_root=cache_root,
        expected_span_tokens=expected_span_tokens,
        arena_bytes=arena_bytes,
    )
    hot = _hot_page_snapshot(lookup._manifest["snapshot_sha256"], snapshot_bytes)
    if hot is not None:
        # The manifest-bound snapshot digest named a snapshot this process
        # hashed to it (a publication or Python restore of this boundary).
        page_plan = plan_page_snapshot(
            layout,
            hot,
            tuple(len(group) for group in group_slots),
            total_bytes=snapshot_bytes,
        )
        planning_ms = 1e3 * (time.perf_counter() - planning_started)
        placed = execute_cuda_hybrid_placement(
            adapter=adapter,
            request_id=request_id,
            encoded_pages=hot,
            plan=page_plan,
            group_slots=group_slots,
        )
        return replace(
            placed,
            read_and_hash_ms=planning_ms,
            planning_ms=planning_ms,
            hot_snapshot_hit=True,
        )
    first = objects[0]
    first_payload = bytearray(first.encoded_bytes)
    started = time.perf_counter()
    _pread_exact_into(first.path, first.encoded_bytes, memoryview(first_payload))
    if not _source_payload_matches(first, first_payload):
        raise CudaHybridRestoreError(f"page object SHA-256 mismatch for {first.path}")
    read_ms = 1e3 * (time.perf_counter() - started)
    page_plan = plan_page_snapshot(
        layout,
        first_payload,
        tuple(len(group) for group in group_slots),
        total_bytes=snapshot_bytes,
    )
    transaction = adapter.begin_parked_page_restore(
        request_id,
        group_slots,
        snapshot_bytes=snapshot_bytes - page_plan.header_bytes,
    )
    arena_wait_ms = 0.0
    host_copy_ms = 0.0
    submit_call_ms = 0.0

    def submit_authenticated_object(
        object_index: int,
        page_object: CudaPageObject,
        payload: bytearray,
    ) -> None:
        nonlocal arena_wait_ms, host_copy_ms, submit_call_ms
        arena_index = object_index % cuda.ARENA_COUNT
        started = time.perf_counter()
        arena = transaction.acquire_arena(arena_index)
        arena_wait_ms += 1e3 * (time.perf_counter() - started)
        buffer = cuda.arena_memoryview(
            arena,
            length=page_object.encoded_bytes,
        )
        try:
            started = time.perf_counter()
            buffer[:] = payload
            host_copy_ms += 1e3 * (time.perf_counter() - started)
            spans = build_page_object_spans(
                page_plan,
                encoded_start=page_object.encoded_start,
                encoded_end=page_object.encoded_end,
            )
            if not spans:
                raise CudaHybridRestoreError(
                    "flat page object contains no restorable payload"
                )
            started = time.perf_counter()
            transaction.submit_page_slab(
                arena_index=arena_index,
                arena_used_bytes=page_object.encoded_bytes,
                spans=spans,
            )
            submit_call_ms += 1e3 * (time.perf_counter() - started)
        finally:
            buffer.release()
            payload.clear()

    with transaction:
        submit_authenticated_object(0, first, first_payload)

        read_workers = min(
            io_workers,
            _MAX_PAGE_OBJECT_READ_WORKERS,
        )

        def read_and_authenticate(
            item: tuple[CudaPageObject, bytearray],
        ) -> None:
            page_object, buffer = item
            target = memoryview(buffer)
            try:
                _pread_exact_into(
                    page_object.path,
                    page_object.encoded_bytes,
                    target,
                )
            finally:
                target.release()
            if not _source_payload_matches(page_object, buffer):
                raise CudaHybridRestoreError(
                    f"page object SHA-256 mismatch for {page_object.path}"
                )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=read_workers
        ) as read_pool:
            object_index = 1
            while object_index < len(objects):
                batch: list[tuple[int, CudaPageObject]] = []
                batch_bytes = 0
                while (
                    object_index < len(objects)
                    and len(batch) < read_workers
                ):
                    page_object = objects[object_index]
                    if (
                        batch
                        and batch_bytes + page_object.encoded_bytes
                        > _MAX_PAGE_OBJECT_PREFETCH_BYTES
                    ):
                        break
                    if page_object.encoded_bytes > _MAX_PAGE_OBJECT_PREFETCH_BYTES:
                        raise CudaHybridRestoreError(
                            "flat page object exceeds the host prefetch bound"
                        )
                    batch.append((object_index, page_object))
                    batch_bytes += page_object.encoded_bytes
                    object_index += 1
                staged = [
                    (index, page_object, bytearray(page_object.encoded_bytes))
                    for index, page_object in batch
                ]
                started = time.perf_counter()
                tuple(
                    read_pool.map(
                        read_and_authenticate,
                        (
                            (page_object, buffer)
                            for _index, page_object, buffer in staged
                        ),
                    )
                )
                read_ms += 1e3 * (time.perf_counter() - started)
                # Every object in the bounded batch is authenticated before
                # any of its bytes reach a mapped CUDA arena. Reading into
                # request-private host buffers lets storage work overlap the
                # preceding arena's in-flight placement without exposing an
                # unauthenticated partial batch to the GPU.
                for index, page_object, buffer in staged:
                    submit_authenticated_object(index, page_object, buffer)
        started = time.perf_counter()
        stats = transaction.finish()
        finish_ms = 1e3 * (time.perf_counter() - started)
        if transaction.state is not RestoreState.FINISHED or not transaction.can_resume:
            raise CudaHybridRestoreError(
                "flat page placement did not release the parked request"
            )
    expected_stats = {
        # The C++ ABI counts every authenticated arena byte submitted through
        # spark_cache_placement_submit_page_slab. Flat macro objects include
        # the encoded snapshot header in the first arena even though page-copy
        # spans cover only payload bytes after that header.
        "source_bytes": snapshot_bytes,
        "slabs_submitted": len(objects),
        "scatter_kernel_launches": len(objects),
        "slot_uploads": 1,
        "destination_table_uploads": 1,
        "device_error": 0,
        "staged_h2d_bytes": 0,
    }
    mismatches = {
        name: {"expected": expected, "actual": int(getattr(stats, name, -1))}
        for name, expected in expected_stats.items()
        if int(getattr(stats, name, -1)) != expected
    }
    if mismatches:
        raise CudaHybridRestoreError(
            f"flat page placement statistics violate the restore contract: {mismatches}"
        )
    return CudaHybridRestoreResult(
        placement_stats=stats,
        source_bytes=snapshot_bytes,
        copy_and_submit_ms=arena_wait_ms + host_copy_ms + submit_call_ms,
        finish_ms=finish_ms,
        slabs=len(objects),
        read_and_hash_ms=read_ms,
        arena_wait_ms=arena_wait_ms,
        host_copy_ms=host_copy_ms,
        submit_call_ms=submit_call_ms,
    )


def _delta_submission_batches(
    spans: Sequence[CudaPageSourceSpan],
    *,
    arena_bytes: int,
) -> tuple[tuple[CudaPageSourceSpan, ...], ...]:
    """Pack final-order fragments within arena, prefetch, and ABI bounds."""

    batches: list[tuple[CudaPageSourceSpan, ...]] = []
    batch: list[CudaPageSourceSpan] = []
    batch_bytes = 0
    batch_objects: dict[Path, CudaPageObject] = {}
    batch_object_bytes = 0
    expected_snapshot = 0
    destination_covered: dict[int, int] = {}
    for span in spans:
        if (
            span.byte_count <= 0
            or span.byte_count > arena_bytes
            or span.snapshot_offset_bytes != expected_snapshot
            or span.destination_byte_offset
            != destination_covered.get(span.destination_index, 0)
        ):
            raise CudaHybridRestoreError(
                "page-delta source spans violate final coverage"
            )
        new_object_bytes = (
            0
            if span.source.path in batch_objects
            else span.source.encoded_bytes
        )
        if batch and (
            batch_bytes + span.byte_count > arena_bytes
            or batch_object_bytes + new_object_bytes
            > _MAX_PAGE_OBJECT_PREFETCH_BYTES
            or len(batch) >= 4096
        ):
            batches.append(tuple(batch))
            batch = []
            batch_bytes = 0
            batch_objects = {}
            batch_object_bytes = 0
            new_object_bytes = span.source.encoded_bytes
        if new_object_bytes > _MAX_PAGE_OBJECT_PREFETCH_BYTES:
            raise CudaHybridRestoreError(
                "page-delta source object exceeds the host prefetch bound"
            )
        batch.append(span)
        batch_bytes += span.byte_count
        if span.source.path not in batch_objects:
            batch_objects[span.source.path] = span.source
            batch_object_bytes += span.source.encoded_bytes
        expected_snapshot += span.byte_count
        destination_covered[span.destination_index] = (
            span.destination_byte_offset + span.byte_count
        )
    if batch:
        batches.append(tuple(batch))
    if not batches:
        raise CudaHybridRestoreError("page-delta restore has no source spans")
    return tuple(batches)


def _execute_page_delta_restore(
    *,
    adapter: Any,
    request_id: str,
    lookup: Any,
    cache_root: Any,
    layout: PageLayout,
    group_slots: Sequence[Sequence[int]],
    expected_span_tokens: int,
    arena_bytes: int,
    io_workers: int,
    base_reader: Callable[
        [PageBaseReadEvidence, Callable[[], PageBaseReadResult]],
        PageBaseReadResult | bytes | None,
    ]
    | None,
) -> CudaHybridRestoreResult:
    """Place a verified base+delta graph without assembling its final bytes."""

    planning_started = time.perf_counter()
    plan = plan_cuda_page_delta_restore(
        lookup,
        cache_root=cache_root,
        layout=layout,
        group_slots=group_slots,
        expected_span_tokens=expected_span_tokens,
        arena_bytes=arena_bytes,
        base_reader=base_reader,
    )
    planning_ms = 1e3 * (time.perf_counter() - planning_started)
    read_ms = planning_ms
    if plan.hot_snapshot is not None:
        # The newest authenticated header's result digest named a snapshot
        # this process hashed to it. Place it as one flat snapshot.
        placed = execute_cuda_hybrid_placement(
            adapter=adapter,
            request_id=request_id,
            encoded_pages=plan.hot_snapshot,
            plan=plan.page_plan,
            group_slots=group_slots,
        )
        return replace(
            placed,
            read_and_hash_ms=read_ms,
            planning_ms=planning_ms,
            read_source_bytes=plan.planning_read_source_bytes,
            hot_snapshot_hit=True,
        )
    batches = _delta_submission_batches(plan.source_spans, arena_bytes=arena_bytes)
    header_bytes = plan.page_plan.header_bytes
    payload_bytes = plan.page_plan.total_bytes - header_bytes
    prefetched = {
        item.source.path: item for item in plan.prefetched_objects
    }
    referenced_paths = {span.source.path for span in plan.source_spans}
    for path in tuple(prefetched):
        if path not in referenced_paths:
            prefetched.pop(path)
    read_source_bytes = plan.planning_read_source_bytes or sum(
        item.source.encoded_bytes for item in plan.prefetched_objects
    )
    # The plan must not keep a second reference to every retained object after
    # its final submission; admitted bases can be much larger than one arena.
    plan = replace(plan, prefetched_objects=())
    last_use = {
        span.source.path: batch_index
        for batch_index, batch in enumerate(batches)
        for span in batch
    }
    span_uses: dict[Path, int] = {}
    for span in plan.source_spans:
        span_uses[span.source.path] = span_uses.get(span.source.path, 0) + 1
    source_cache: OrderedDict[Path, CudaAuthenticatedPageObject] = OrderedDict()
    cache_bytes = 0
    cache_limit = min(arena_bytes, _MAX_PAGE_OBJECT_PREFETCH_BYTES)
    transaction = adapter.begin_parked_page_restore(
        request_id,
        group_slots,
        snapshot_bytes=payload_bytes,
    )
    arena_wait_ms = 0.0
    host_copy_ms = 0.0
    submit_call_ms = 0.0
    final_hash_ms = 0.0
    result_header = encode_page_snapshot_header(layout, plan.page_plan.block_counts)
    final_sha256 = hashlib.sha256(result_header)
    # Under fast and none the result digest is a boundary token; every placed
    # byte was already checked by its own object or tail check.
    verify_result = plan.integrity == "sha256"
    # With a hot snapshot budget, keep a private copy of the composed result
    # so a later restore or publication of this boundary needs no object
    # read. It is retained only after the final digest check passes.
    retained: bytearray | None = None
    if _hot_page_snapshot_fits(plan.page_plan.total_bytes):
        retained = bytearray(plan.page_plan.total_bytes)
        retained[:header_bytes] = result_header
    # The sequential result digest of one batch runs while the next batch's
    # tails are read and authenticated by the pool. (arena view, arena index,
    # snapshot offset) of the batch not yet digested.
    pending_hash: tuple[memoryview, int, int] | None = None

    def hash_pending() -> None:
        nonlocal pending_hash, final_hash_ms
        if pending_hash is None:
            return
        view, _arena_index, offset = pending_hash
        pending_hash = None
        started = time.perf_counter()
        try:
            if verify_result:
                final_sha256.update(view)
            if retained is not None:
                start = header_bytes + offset
                retained[start : start + len(view)] = view
        finally:
            view.release()
        final_hash_ms += 1e3 * (time.perf_counter() - started)

    # Each submission launches CUDA work asynchronously. Reading and
    # authenticating the following batch therefore overlaps the preceding
    # placement. Tail extents are read straight into the mapped arena and
    # authenticated there by the pool; whole objects (the base, and tails
    # above the extent bound) are authenticated in private buffers first.
    read_workers = min(max(1, io_workers), _MAX_PAGE_TAIL_READ_WORKERS)
    object_workers = min(max(1, io_workers), _MAX_PAGE_OBJECT_READ_WORKERS)
    with transaction:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=read_workers
        ) as read_pool:
            for batch_index, batch in enumerate(batches):
                sources: dict[Path, CudaPageObject] = {}
                for span in batch:
                    sources.setdefault(span.source.path, span.source)
                loaded: dict[Path, CudaAuthenticatedPageObject] = {
                    path: prefetched[path]
                    for path in sources
                    if path in prefetched
                }
                loaded.update(
                    (path, source_cache[path])
                    for path in sources
                    if path in source_cache
                )
                # A tail extent that one span places whole is read into its
                # arena slot and authenticated in place.
                direct: set[Path] = set()
                for span in batch:
                    source = span.source
                    if (
                        isinstance(source, CudaPageTailExtent)
                        and source.path not in loaded
                        and span_uses.get(source.path) == 1
                        and span.source_offset_bytes == 0
                        and span.byte_count == source.encoded_bytes
                    ):
                        direct.add(source.path)
                pending = tuple(
                    source
                    for path, source in sources.items()
                    if path not in loaded and path not in direct
                )
                if pending:
                    started = time.perf_counter()
                    authenticated = tuple(
                        read_pool.map(
                            _read_authenticated_source,
                            pending,
                            chunksize=max(1, len(pending) // object_workers),
                        )
                    )
                    read_ms += 1e3 * (time.perf_counter() - started)
                    for item in authenticated:
                        loaded[item.source.path] = item
                        read_source_bytes += item.source.encoded_bytes
                    del authenticated, item

                arena_index = batch_index % cuda.ARENA_COUNT
                if pending_hash is not None and pending_hash[1] == arena_index:
                    # A single-arena transaction would overwrite the batch
                    # that has not been digested yet.
                    hash_pending()
                started = time.perf_counter()
                arena = transaction.acquire_arena(arena_index)
                arena_wait_ms += 1e3 * (time.perf_counter() - started)
                used_bytes = sum(span.byte_count for span in batch)
                arena_buffer = cuda.arena_memoryview(arena, length=used_bytes)
                batch_snapshot_offset = batch[0].snapshot_offset_bytes
                native_spans = []
                copies: list[tuple[CudaPageSourceSpan, int]] = []
                futures = []
                arena_offset = 0
                for span in batch:
                    if span.source.path in direct:
                        futures.append(
                            read_pool.submit(
                                _read_authenticated_tail_into,
                                span.source,
                                arena_buffer[
                                    arena_offset : arena_offset + span.byte_count
                                ],
                            )
                        )
                    else:
                        copies.append((span, arena_offset))
                    native_spans.append(
                        cuda.PageCopySpan(
                            arena_offset,
                            span.snapshot_offset_bytes,
                            span.destination_byte_offset,
                            span.byte_count,
                            span.destination_index,
                            0,
                        )
                    )
                    arena_offset += span.byte_count
                try:
                    started = time.perf_counter()
                    for span, offset in copies:
                        payload = loaded[span.source.path].payload
                        payload_view = memoryview(payload)
                        try:
                            source_end = span.source_offset_bytes + span.byte_count
                            arena_buffer[
                                offset : offset + span.byte_count
                            ] = payload_view[span.source_offset_bytes:source_end]
                        finally:
                            payload_view.release()
                        del payload
                    host_copy_ms += 1e3 * (time.perf_counter() - started)
                    # Digest the previous batch while this batch's tails are
                    # still being read and authenticated.
                    hash_pending()
                    started = time.perf_counter()
                    for future in futures:
                        read_source_bytes += future.result()
                    read_ms += 1e3 * (time.perf_counter() - started)
                except BaseException:
                    # Every worker releases its arena slice itself; wait so
                    # the arena view can be released before the abort.
                    concurrent.futures.wait(futures)
                    arena_buffer.release()
                    raise
                started = time.perf_counter()
                transaction.submit_page_slab(
                    arena_index=arena_index,
                    arena_used_bytes=used_bytes,
                    spans=native_spans,
                )
                submit_call_ms += 1e3 * (time.perf_counter() - started)
                pending_hash = (arena_buffer, arena_index, batch_snapshot_offset)
                del futures, copies
                # Source bytes are no longer needed by CUDA after the host
                # copy into the fenced arena. Release their final use promptly;
                # retain recurring objects within a separate bounded cache.
                for path, item in loaded.items():
                    if last_use[path] == batch_index:
                        prefetched.pop(path, None)
                        cached = source_cache.pop(path, None)
                        if cached is not None:
                            cache_bytes -= cached.source.encoded_bytes
                        del cached
                    elif path not in prefetched and path not in source_cache:
                        size = item.source.encoded_bytes
                        if size <= cache_limit:
                            while source_cache and cache_bytes + size > cache_limit:
                                _, evicted = source_cache.popitem(last=False)
                                cache_bytes -= evicted.source.encoded_bytes
                                del evicted
                            source_cache[path] = item
                            cache_bytes += size
                item = None  # drop the last payload reference with the batch
                loaded.clear()
            hash_pending()
        if verify_result and final_sha256.hexdigest() != plan.result_snapshot_sha256:
            raise CudaHybridRestoreError(
                "page-delta reconstructed snapshot checksum mismatch"
            )
        if retained is not None:
            # Exactly these bytes hashed to the header-bound result digest.
            _remember_hot_page_snapshot(plan.result_snapshot_sha256, retained)
            retained = None
        started = time.perf_counter()
        stats = transaction.finish()
        finish_ms = 1e3 * (time.perf_counter() - started)
        if transaction.state is not RestoreState.FINISHED or not transaction.can_resume:
            raise CudaHybridRestoreError(
                "page-delta placement did not release the parked request"
            )
    expected_stats = {
        "source_bytes": payload_bytes,
        "slabs_submitted": len(batches),
        "scatter_kernel_launches": len(batches),
        "slot_uploads": 1,
        "destination_table_uploads": 1,
        "device_error": 0,
        "staged_h2d_bytes": 0,
    }
    mismatches = {
        name: {"expected": expected, "actual": int(getattr(stats, name, -1))}
        for name, expected in expected_stats.items()
        if int(getattr(stats, name, -1)) != expected
    }
    if mismatches:
        raise CudaHybridRestoreError(
            "page-delta placement statistics violate the restore contract:"
            f" {mismatches}"
        )
    return CudaHybridRestoreResult(
        placement_stats=stats,
        source_bytes=plan.page_plan.total_bytes,
        copy_and_submit_ms=arena_wait_ms + host_copy_ms + submit_call_ms,
        finish_ms=finish_ms,
        slabs=len(batches),
        read_and_hash_ms=read_ms,
        arena_wait_ms=arena_wait_ms,
        host_copy_ms=host_copy_ms,
        submit_call_ms=submit_call_ms,
        read_source_bytes=read_source_bytes,
        skipped_base_object_bytes=plan.skipped_base_object_bytes,
        final_hash_ms=final_hash_ms,
        planning_ms=planning_ms,
    )


def execute_cuda_hybrid_restore(
    *,
    adapter: Any,
    request_id: str,
    lookup: Any,
    cache_root: Any,
    layout: PageLayout,
    group_slots: Sequence[Sequence[int]],
    expected_span_tokens: int,
    arena_bytes: int,
    io_workers: int = 8,
    dcp_degree: int = 1,
    dcp_rank: int = 0,
    base_reader: Callable[
        [PageBaseReadEvidence, Callable[[], PageBaseReadResult]],
        PageBaseReadResult | bytes | None,
    ]
    | None = None,
) -> CudaHybridRestoreResult:
    """Pipeline authenticated .spcc reads directly into mapped page scatter."""

    manifest = getattr(lookup, "_manifest", None)
    if isinstance(manifest, dict) and (
        manifest.get("schema")
        in (_PAGE_DELTA_MANIFEST_SCHEMA, _PAGE_DELTA_MANIFEST_SCHEMA_V3)
    ):
        return _execute_page_delta_restore(
            adapter=adapter,
            request_id=request_id,
            lookup=lookup,
            cache_root=cache_root,
            layout=layout,
            group_slots=group_slots,
            expected_span_tokens=expected_span_tokens,
            arena_bytes=arena_bytes,
            io_workers=io_workers,
            base_reader=base_reader,
        )
    if isinstance(manifest, dict) and (
        manifest.get("schema") == _PAGE_SNAPSHOT_MANIFEST_SCHEMA
    ):
        return _execute_page_object_restore(
            adapter=adapter,
            request_id=request_id,
            lookup=lookup,
            cache_root=cache_root,
            layout=layout,
            group_slots=group_slots,
            expected_span_tokens=expected_span_tokens,
            arena_bytes=arena_bytes,
            io_workers=io_workers,
        )

    if dcp_degree != 1 or dcp_rank != 0:
        raise CudaHybridRestoreError(
            "legacy chunk-based page restore supports only DCP1;"
            " publish a v2 page snapshot before using DCP"
        )

    slabs = plan_cuda_restore(
        lookup,
        cache_root=cache_root,
        expected_span_tokens=expected_span_tokens,
        dcp_degree=1,
        arena_bytes=arena_bytes,
    )
    target_plans = {}
    snapshot_bytes = 0
    for slab in slabs:
        for chunk in slab.chunks:
            target = _target_record_plan(chunk.path, chunk.encoded_bytes)
            target_plans[chunk.path] = target
            snapshot_bytes += target[2]
    first = slabs[0].chunks[0]
    payload_offset, target_offset, target_length = target_plans[first.path]
    fd = os.open(first.path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        header_prefix = os.pread(
            fd,
            min(target_length, 1 << 20),
            payload_offset + target_offset,
        )
    finally:
        os.close(fd)
    page_plan = plan_page_snapshot(
        layout,
        header_prefix,
        tuple(len(group) for group in group_slots),
        total_bytes=snapshot_bytes,
    )
    transaction = adapter.begin_parked_page_restore(
        request_id,
        group_slots,
        snapshot_bytes=snapshot_bytes - page_plan.header_bytes,
    )
    read_ms = submit_ms = 0.0
    stream_offset = 0
    with transaction:
        for slab_index, slab in enumerate(slabs):
            arena_index = slab_index % cuda.ARENA_COUNT
            arena = transaction.acquire_arena(arena_index)
            buffer = cuda.arena_memoryview(arena, length=slab.arena_used_bytes)
            views = tuple(
                buffer[c.arena_offset_bytes : c.arena_offset_bytes + c.encoded_bytes]
                for c in slab.chunks
            )
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(io_workers, len(slab.chunks))
            ) as pool:
                tuple(
                    pool.map(
                        lambda item: _read_and_authenticate(*item),
                        zip(slab.chunks, views, strict=True),
                    )
                )
            read_ms += 1e3 * (time.perf_counter() - started)
            del views
            del buffer
            started = time.perf_counter()
            spans = []
            for chunk in slab.chunks:
                parsed = transaction.parse_verified_chunk(
                    arena=arena,
                    arena_used_bytes=slab.arena_used_bytes,
                    arena_offset_bytes=chunk.arena_offset_bytes,
                    encoded_bytes=chunk.encoded_bytes,
                    expected_logical_start=chunk.logical_start,
                    dcp_degree=1,
                    dcp_rank=0,
                    first_slot_index=chunk.first_slot_index,
                    required_data_record_mask=1 << _TARGET_KIND,
                )
                payload, target_offset, target_length = target_plans[chunk.path]
                if (
                    int(parsed.payload_offset_bytes) != payload
                    or int(parsed.record_offset_bytes[_TARGET_KIND]) != target_offset
                    or int(parsed.record_length_bytes[_TARGET_KIND]) != target_length
                ):
                    raise CudaHybridRestoreError("authenticated target extent changed")
                extent_start = stream_offset
                extent_end = stream_offset + target_length
                for destination_index, layer in enumerate(page_plan.spans):
                    start = max(extent_start, layer.source_start)
                    end = min(extent_end, layer.source_end)
                    if start >= end:
                        continue
                    spans.append(
                        cuda.PageCopySpan(
                            chunk.arena_offset_bytes
                            + payload
                            + target_offset
                            + start
                            - extent_start,
                            start - page_plan.header_bytes,
                            start - layer.source_start,
                            end - start,
                            destination_index,
                            0,
                        )
                    )
                stream_offset = extent_end
            transaction.submit_page_slab(
                arena_index=arena_index,
                arena_used_bytes=slab.arena_used_bytes,
                spans=spans,
            )
            submit_ms += 1e3 * (time.perf_counter() - started)
        started = time.perf_counter()
        stats = transaction.finish()
        finish_ms = 1e3 * (time.perf_counter() - started)
    return CudaHybridRestoreResult(
        placement_stats=stats,
        source_bytes=snapshot_bytes,
        copy_and_submit_ms=submit_ms,
        finish_ms=finish_ms,
        slabs=len(slabs),
        read_and_hash_ms=read_ms,
    )


CudaPageRestoreError = CudaHybridRestoreError
CudaPageRestoreResult = CudaHybridRestoreResult
execute_cuda_direct_restore = execute_cuda_hybrid_restore
execute_cuda_page_placement = execute_cuda_hybrid_placement


__all__ = [
    "CudaAuthenticatedPageObject",
    "CudaHybridRestoreError",
    "CudaHybridRestoreResult",
    "CudaPageDeltaRestorePlan",
    "CudaPageObject",
    "CudaPageRestoreError",
    "CudaPageRestoreResult",
    "CudaPageSlab",
    "CudaPageSourceSpan",
    "CudaPageTailExtent",
    "build_page_copy_spans",
    "build_page_object_spans",
    "execute_cuda_direct_restore",
    "execute_cuda_hybrid_placement",
    "execute_cuda_hybrid_restore",
    "execute_cuda_page_placement",
    "plan_cuda_page_delta_restore",
    "plan_page_slabs",
]
