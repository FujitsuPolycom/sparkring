"""Model-free storage ABI for persistent context-cache state.

This module deliberately treats tensor records as opaque bytes.  It proves the
identity, completeness, atomic-publication, and corruption semantics before a
vLLM adapter is allowed to supply real CKV or MTP buffers.
"""

from __future__ import annotations

import hashlib
import json
import zlib
import os
import re
import stat
import struct
import threading
import time
import uuid
from collections import Counter, OrderedDict
from contextlib import AbstractContextManager
from contextvars import ContextVar
from dataclasses import dataclass, field as dataclass_field, replace
from enum import Enum
from functools import wraps
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

if TYPE_CHECKING:
    from sparkcache.spark_context_cache_hybrid import _VerifiedPageSnapshot

from sparkcache.page_base_read_flights import PageBaseReadEvidence
from sparkcache.publication_telemetry import (
    PublicationAttempt,
    PublicationByteReceipt,
    PublicationKind,
    PublicationTelemetry,
    PublicationTelemetrySnapshot,
)

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows GPU-free test fallback
    _fcntl = None


FORMAT_ABI = 1
_CHUNK_MAGIC = b"SPCKV001"
_CHUNK_PREFIX = struct.Struct("<8sII")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_COMMIT_CHUNK_BATCH_SIZE = 8
_PREFIX_SEGMENT_DESCRIPTORS = 16
_MAX_PREFIX_ALIASES = 64
_PREFIX_SEGMENT_SCHEMA = "sparkcache-prefix-descriptor-segment/v1"
_PREFIX_ALIAS_SCHEMA = "sparkcache-prefix-alias/v1"
_TAIL_MANIFEST_SCHEMA = "sparkcache-tail-manifest/v1"
_PAGE_DELTA_MANIFEST_SCHEMA = "sparkcache-page-delta-manifest/v1"
_PAGE_DELTA_MANIFEST_SCHEMA_V2 = "sparkcache-page-delta-manifest/v2"
_PAGE_DELTA_MANIFEST_SCHEMA_V3 = "sparkcache-page-delta-manifest/v3"
_PAGE_DELTA_MANIFEST_SCHEMAS = frozenset(
    (
        _PAGE_DELTA_MANIFEST_SCHEMA,
        _PAGE_DELTA_MANIFEST_SCHEMA_V2,
        _PAGE_DELTA_MANIFEST_SCHEMA_V3,
    )
)
_PAGE_SNAPSHOT_MANIFEST_SCHEMA = "sparkcache-page-snapshot-manifest/v2"
# The v2 physical geometry is independent of the 256-token digest and
# admission boundary. A 64-MiB extent reduces a 1.58-GB delta to 24 objects;
# bounded batches cap temporary payload bytes at 128 MiB while publishing and
# 256 MiB while reading, in addition to the assembled authenticated delta.
_PAGE_DELTA_OBJECT_BYTES = 64 * 1024 * 1024
_MAX_PAGE_DELTA_OBJECT_BYTES = 64 * 1024 * 1024
_PAGE_DELTA_WRITE_BATCH_SIZE = 2
_PAGE_DELTA_READ_BATCH_SIZE = 4
# Flat opaque page snapshots use the same bounded physical geometry as page
# deltas. Their logical cache identity and admission geometry remain the
# identity's 256-token chunks; only the content-addressed storage objects are
# coalesced. A distinct root schema lets legacy flat manifests remain readable
# while runtimes that do not understand macro objects cleanly reject v2 roots.
_PAGE_SNAPSHOT_OBJECT_BYTES = 64 * 1024 * 1024
_MAX_PAGE_SNAPSHOT_OBJECT_BYTES = 64 * 1024 * 1024
_PAGE_SNAPSHOT_WRITE_BATCH_SIZE = 2
_PAGE_SNAPSHOT_READ_BATCH_SIZE = 4
# Two delta roots cap reconstruction at two full-snapshot applications. A
# following extension is compacted by the connector into a fresh flat root.
_MAX_PAGE_DELTA_DEPTH = 2
_CLEAR_ONCE_SCHEMA = "sparkcache-clear-once/v1"
_CLEAR_ONCE_MARKER_DIRECTORY = ".sparkcache-clear-once"
_CLEAR_ONCE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@+-]{0,127}\Z")
_CACHE_DATA_DIRECTORIES = (
    "chunks",
    "manifests",
    "prefix-aliases",
    "prefix-index",
)
_ACTIVE_PUBLICATION: ContextVar[PublicationAttempt | None] = ContextVar(
    "sparkcache_active_publication",
    default=None,
)


class _ProcessRootLock:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers = 0
        self._writer = False

    def acquire(
        self,
        *,
        shared: bool,
        blocking: bool,
        timeout_seconds: float | None = None,
    ) -> bool:
        with self._condition:

            def available() -> bool:
                return (
                    not self._writer
                    if shared
                    else not self._writer and self._readers == 0
                )

            if not blocking and not available():
                return False
            if blocking:
                if not self._condition.wait_for(
                    available,
                    timeout=timeout_seconds,
                ):
                    return False
            if shared:
                self._readers += 1
            else:
                self._writer = True
            return True

    def release(self, *, shared: bool) -> None:
        with self._condition:
            if shared:
                if self._readers <= 0:
                    raise RuntimeError("shared root lock is not held")
                self._readers -= 1
            else:
                if not self._writer:
                    raise RuntimeError("exclusive root lock is not held")
                self._writer = False
            self._condition.notify_all()


_ROOT_LOCKS: dict[str, _ProcessRootLock] = {}
_ROOT_LOCKS_GUARD = threading.Lock()


class StateRecord(str, Enum):
    TARGET_CKV = "target_ckv"
    SPARSE_INDEXER = "sparse_indexer"
    MTP_DRAFT_KV = "mtp_draft_kv"
    BOUNDARY_HIDDEN = "boundary_hidden"
    LOGICAL_POSITIONS = "logical_positions"


_REQUIRED_RECORDS = frozenset(StateRecord)


class CacheFormatError(ValueError):
    """Internal format failure that public lookup converts to a clean miss."""


class _IncompatibleManifestError(CacheFormatError):
    """A structurally valid manifest belongs to another runtime contract."""


class CommitConflict(RuntimeError):
    """A different immutable object already owns the cache key."""


class IncompleteEntry(ValueError):
    """Required target, indexer, draft, or boundary state is absent."""


class PageDeltaDepthExceeded(ValueError):
    """Another page delta would exceed the bounded reconstruction depth."""


def _is_page_delta_root(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("schema") in _PAGE_DELTA_MANIFEST_SCHEMAS
    )


def _is_page_snapshot_root(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("schema") == _PAGE_SNAPSHOT_MANIFEST_SCHEMA
    )


@dataclass(frozen=True)
class CacheIdentity:
    target_checkpoint: str
    draft_checkpoint: str
    quantization_layout: str
    rope_layout: str
    tp_degree: int
    dcp_degree: int
    # Number of consecutive global token positions assigned to one DCP rank
    # before ownership advances to the next rank. Values above one change the
    # physical DCP ownership mapping and therefore belong to cache identity.
    cp_kv_cache_interleave_size: int = 1
    chunk_tokens: int = 256
    # DCP shard ownership: entries written by one rank must never restore
    # into another. -1 means "not sharded" (DCP1 whole-context entries).
    dcp_shard_rank: int = -1
    # Physical TP worker identity: unique across all TP ranks.  Under
    # TP4/DCP2, two physical workers (e.g. TP0 and TP2) can share the
    # same dcp_shard_rank (both DCP rank 0).  Without tp_shard_rank
    # they would share the same storage_key and could cross-restore
    # complementary TP shards. -1 selects an identity without a physical-TP
    # dimension and is valid only for DCP1 whole-context entries.
    #
    # Wire identities that omit this field hash differently from identities
    # with a concrete tp_shard_rank. Entries without physical-worker identity
    # therefore clean-miss and cannot be reinterpreted across TP workers.
    tp_shard_rank: int = -1
    # "persisted": every chunk carries a boundary_hidden record.
    # "live_forward": boundary hidden state is not
    # persisted; the first post-restore forward regenerates it.
    boundary_hidden_policy: str = "persisted"
    # "separate": chunks carry a distinct mtp_draft_kv record.
    # "colocated_target": the runtime registers drafter KV layers in the
    # same cache pool without a distinguishing name, so draft state is
    # persisted and restored inside target_ckv records and no separate
    # mtp_draft_kv record exists.
    draft_kv_policy: str = "separate"
    # Empty selects the policy-derived record set and its compatibility wire
    # identity. Non-empty schemas explicitly name the records required by a
    # storage mode whose opaque payloads do not map to the policy-derived set.
    record_schema: tuple[str, ...] = ()
    # Empty preserves the full-snapshot wire identity. Physical flat-manifest
    # encodings may evolve under strict schema validation; an older runtime
    # then cleanly misses them. Tail-only publication remains opt-in because
    # its reusable-base semantics require a distinct identity namespace.
    publication_schema: str = ""

    def __post_init__(self) -> None:
        for field in ("target_checkpoint", "draft_checkpoint"):
            value = getattr(self, field)
            if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
                raise ValueError(f"{field} must be a 64-character lowercase SHA-256")
        for field in ("quantization_layout", "rope_layout"):
            if not getattr(self, field):
                raise ValueError(f"{field} must be non-empty")
        for field in (
            "tp_degree",
            "dcp_degree",
            "cp_kv_cache_interleave_size",
            "chunk_tokens",
        ):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if self.boundary_hidden_policy not in ("persisted", "live_forward"):
            raise ValueError(
                "boundary_hidden_policy must be 'persisted' or 'live_forward'"
            )
        if self.draft_kv_policy not in ("separate", "colocated_target"):
            raise ValueError("draft_kv_policy must be 'separate' or 'colocated_target'")
        if not -1 <= self.dcp_shard_rank < self.dcp_degree:
            raise ValueError("dcp_shard_rank must be -1 or in [0, dcp_degree)")
        if not -1 <= self.tp_shard_rank < self.tp_degree:
            raise ValueError("tp_shard_rank must be -1 or in [0, tp_degree)")
        if self.record_schema:
            try:
                records = tuple(StateRecord(value) for value in self.record_schema)
            except ValueError as error:
                raise ValueError("record_schema contains an unknown record") from error
            if len(records) != len(set(records)):
                raise ValueError("record_schema records must be unique")
            if StateRecord.LOGICAL_POSITIONS not in records:
                raise ValueError("record_schema must include logical_positions")
        if self.publication_schema not in (
            "",
            "tail-cow-v1",
            "page-tail-cow-v1",
            "page-tail-cow-v2",
            "page-tail-cow-v3-fast",
            "page-tail-cow-v3-none",
        ):
            raise ValueError("publication_schema is unsupported")

    @property
    def required_records(self) -> frozenset["StateRecord"]:
        if self.record_schema:
            return frozenset(StateRecord(value) for value in self.record_schema)
        dropped: set[StateRecord] = set()
        if self.boundary_hidden_policy == "live_forward":
            dropped.add(StateRecord.BOUNDARY_HIDDEN)
        if self.draft_kv_policy == "colocated_target":
            dropped.add(StateRecord.MTP_DRAFT_KV)
        return frozenset(_REQUIRED_RECORDS - dropped)

    def to_wire(self) -> dict[str, Any]:
        wire = {
            "target_checkpoint": self.target_checkpoint,
            "draft_checkpoint": self.draft_checkpoint,
            "quantization_layout": self.quantization_layout,
            "rope_layout": self.rope_layout,
            "tp_degree": self.tp_degree,
            "dcp_degree": self.dcp_degree,
            "cp_kv_cache_interleave_size": self.cp_kv_cache_interleave_size,
            "chunk_tokens": self.chunk_tokens,
            "dcp_shard_rank": self.dcp_shard_rank,
            "tp_shard_rank": self.tp_shard_rank,
            "boundary_hidden_policy": self.boundary_hidden_policy,
            "draft_kv_policy": self.draft_kv_policy,
        }
        if self.record_schema:
            wire["record_schema"] = list(self.record_schema)
        if self.publication_schema:
            wire["publication_schema"] = self.publication_schema
        return wire

    @property
    def storage_key(self) -> str:
        return _sha256(_canonical_json(self.to_wire()))


@dataclass(frozen=True)
class ContextChunk:
    logical_start: int
    logical_end: int
    records: Mapping[StateRecord, bytes]

    def __post_init__(self) -> None:
        if self.logical_start < 0 or self.logical_end <= self.logical_start:
            raise ValueError("chunk logical range must be positive and ordered")
        normalized: dict[StateRecord, bytes] = {}
        for supplied_kind, supplied_payload in self.records.items():
            try:
                kind = StateRecord(supplied_kind)
            except ValueError as error:
                raise ValueError(
                    f"unsupported persistent record {supplied_kind!r}"
                ) from error
            if not isinstance(supplied_payload, bytes):
                raise TypeError(f"{kind.value} payload must be bytes")
            if not supplied_payload:
                raise IncompleteEntry(f"{kind.value} payload must not be empty")
            normalized[kind] = supplied_payload
        if not normalized:
            raise IncompleteEntry("cache chunk carries no records")
        object.__setattr__(self, "records", MappingProxyType(normalized))


def _require_complete_chunk(
    chunk: ContextChunk, required: frozenset[StateRecord]
) -> None:
    """Chunk completeness is identity-dependent (boundary_hidden_policy),
    so it is enforced wherever an identity is in scope, not in the chunk."""
    missing = required - chunk.records.keys()
    if missing:
        names = ", ".join(sorted(item.value for item in missing))
        raise IncompleteEntry(f"incomplete speculative cache chunk: missing {names}")


def _required_records_for_identity_wire(
    identity_wire: Mapping[str, Any],
) -> frozenset[StateRecord]:
    explicit = identity_wire.get("record_schema")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise CacheFormatError("identity record_schema is invalid")
        try:
            records = tuple(StateRecord(value) for value in explicit)
        except (TypeError, ValueError) as error:
            raise CacheFormatError("identity record_schema is invalid") from error
        if (
            len(records) != len(set(records))
            or StateRecord.LOGICAL_POSITIONS not in records
        ):
            raise CacheFormatError("identity record_schema is invalid")
        return frozenset(records)
    dropped: set[StateRecord] = set()
    if identity_wire.get("boundary_hidden_policy") == "live_forward":
        dropped.add(StateRecord.BOUNDARY_HIDDEN)
    if identity_wire.get("draft_kv_policy") == "colocated_target":
        dropped.add(StateRecord.MTP_DRAFT_KV)
    return frozenset(_REQUIRED_RECORDS - dropped)


@dataclass(frozen=True)
class CommitReceipt:
    manifest_digest: str
    committed_tokens: int
    encoded_bytes: int
    allocated_bytes_upper_bound: int
    publication: PublicationByteReceipt | None = dataclass_field(
        default=None,
        compare=False,
    )


@dataclass(frozen=True)
class ChunkReceipt:
    chunk_digest: str
    encoded_bytes: int
    logical_start: int
    logical_end: int


@dataclass(frozen=True)
class PrefixAliasReceipt:
    source_manifest_digest: str
    aliases_published: int
    segments_published: int
    alias_keys: tuple["EntryKey", ...]
    publication: PublicationByteReceipt | None = dataclass_field(
        default=None,
        compare=False,
    )


@dataclass(frozen=True)
class LookupResult:
    is_hit: bool
    reason: str
    manifest_digest: str = ""
    _manifest: Mapping[str, Any] | None = None
    # Identifies the root whose authenticated metadata produced this result.
    # It is process-local lookup state, not part of any persisted schema.
    root_kind: str = "manifest"


@dataclass(frozen=True, order=True)
class EntryKey:
    storage_key: str
    context_digest: str
    # The default preserves the two-argument exact-manifest API and equality
    # used by existing callers. Prefix aliases occupy a distinct root kind so
    # an alias eviction cannot be mistaken for eviction of an exact entry.
    root_kind: str = "manifest"


@dataclass(frozen=True)
class CapacityPolicy:
    max_bytes: int = 0
    low_watermark_bytes: int = 0
    ttl_seconds: int = 0
    maintenance_max_deletions: int = 0
    maintenance_interval_ms: int = 0

    def __post_init__(self) -> None:
        for field in ("max_bytes", "low_watermark_bytes", "ttl_seconds",
                      "maintenance_max_deletions", "maintenance_interval_ms"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if self.max_bytes == 0 and self.low_watermark_bytes != 0:
            raise ValueError("low_watermark_bytes requires max_bytes")
        if self.max_bytes > 0 and not (0 < self.low_watermark_bytes <= self.max_bytes):
            raise ValueError(
                "low_watermark_bytes must be in (0, max_bytes] when bounded"
            )

    @property
    def enabled(self) -> bool:
        return self.max_bytes > 0 or self.ttl_seconds > 0


@dataclass(frozen=True)
class MaintenanceReport:
    bytes_before: int = 0
    bytes_after: int = 0
    bytes_reclaimed: int = 0
    manifests_evicted: int = 0
    chunks_deleted: int = 0
    orphan_chunks_deleted: int = 0
    evicted_entries: tuple[EntryKey, ...] = ()
    capacity_satisfied: bool = True
    skipped_busy: bool = False
    skipped_cooldown: bool = False
    deletion_attempts: int = 0
    work_pending: bool = False
    aliases_evicted: int = 0
    segments_deleted: int = 0
    orphan_segments_deleted: int = 0
    # Metadata-qualified offers at the end of the exclusive inventory pass.
    # None means no authoritative inventory was completed, not an empty cache.
    surviving_entries: tuple[EntryKey, ...] | None = None


@dataclass(frozen=True)
class _CapacityEntry:
    key: EntryKey
    path: Path
    manifest_bytes: int
    mtime_ns: int
    chunks: tuple[tuple[str, int], ...]
    valid: bool
    segments: tuple[str, ...] = ()


def _process_root_lock(root: Path) -> _ProcessRootLock:
    key = str(root.resolve())
    with _ROOT_LOCKS_GUARD:
        return _ROOT_LOCKS.setdefault(key, _ProcessRootLock())


class _RootGuard(AbstractContextManager["_RootGuard"]):
    """Process-local plus POSIX advisory lock for one manifest root."""

    def __init__(
        self,
        root: Path,
        *,
        shared: bool,
        blocking: bool,
        timeout_seconds: float | None = None,
    ) -> None:
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("root-lock timeout must be non-negative")
        self._root = root
        self._shared = shared
        self._blocking = blocking
        self._timeout_seconds = timeout_seconds
        self._process_lock = _process_root_lock(root)
        self._stream: Any = None
        self._entered = False

    def __enter__(self) -> "_RootGuard":
        deadline = (
            time.monotonic() + self._timeout_seconds
            if self._blocking and self._timeout_seconds is not None
            else None
        )
        process_timeout = (
            max(0.0, deadline - time.monotonic()) if deadline is not None else None
        )
        if not self._process_lock.acquire(
            shared=self._shared,
            blocking=self._blocking,
            timeout_seconds=process_timeout,
        ):
            raise BlockingIOError("manifest root is busy")
        try:
            _ensure_durable_directory(self._root)
            if _fcntl is not None:
                self._stream = (self._root / ".maintenance.lock").open("a+b")
                operation = _fcntl.LOCK_SH if self._shared else _fcntl.LOCK_EX
                if not self._blocking or self._timeout_seconds is not None:
                    operation |= _fcntl.LOCK_NB
                if self._blocking and self._timeout_seconds is not None:
                    assert deadline is not None
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise BlockingIOError("manifest root is busy")
                        try:
                            _fcntl.flock(self._stream.fileno(), operation)
                            break
                        except BlockingIOError:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise BlockingIOError("manifest root is busy")
                            time.sleep(min(0.05, remaining))
                else:
                    _fcntl.flock(self._stream.fileno(), operation)
            self._entered = True
            return self
        except BaseException:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            self._process_lock.release(shared=self._shared)
            raise

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if not self._entered:
            return
        try:
            if _fcntl is not None and self._stream is not None:
                _fcntl.flock(self._stream.fileno(), _fcntl.LOCK_UN)
        finally:
            if self._stream is not None:
                self._stream.close()
            self._stream = None
            self._entered = False
            self._process_lock.release(shared=self._shared)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _sha256(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


# Flat page-history publication schemas. ``page-tail-cow-v2`` content-addresses
# every payload object by SHA-256; the ``v3`` schemas hash only identities,
# manifests and stage headers, carry a CRC32 per payload object and per layer
# tail (``fast``) or no payload check (``none``), and name objects by a digest
# of the boundary identity, kind, index and a nonce instead of their content.
_FLAT_PAGE_SCHEMAS = ("page-tail-cow-v2", "page-tail-cow-v3-fast", "page-tail-cow-v3-none")
_PAGE_TAIL_SCHEMAS = ("page-tail-cow-v1", *_FLAT_PAGE_SCHEMAS)


def _payload_integrity(publication_schema: str) -> str:
    if publication_schema == "page-tail-cow-v3-fast":
        return "fast"
    if publication_schema == "page-tail-cow-v3-none":
        return "none"
    return "sha256"


def _boundary_token(context_digest: str, committed_tokens: int) -> str:
    return hashlib.sha256(
        b"sparkcache-boundary-token/v1\x00"
        + f"{context_digest}:{int(committed_tokens)}".encode("ascii")
    ).hexdigest()


def _derived_object_name(
    identity: "CacheIdentity", context_digest: str, kind: str, index: int, nonce: str
) -> str:
    """Object name for payloads that are not content-addressed (v3 schemas)."""

    return hashlib.sha256(
        b"sparkcache-object-name/v1\x00"
        + f"{identity.storage_key}:{context_digest}:{kind}:{index}:{nonce}".encode("ascii")
    ).hexdigest()


def _object_descriptor(
    integrity: str, name: str, encoded: bytes, start: int, end: int
) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "sha256": name,
        "bytes": len(encoded),
        "encoded_start": start,
        "encoded_end": end,
    }
    if integrity == "fast":
        descriptor["crc32"] = zlib.crc32(encoded) & 0xFFFFFFFF
    elif integrity == "none":
        descriptor["crc32"] = None
    return descriptor


def _descriptor_integrity(descriptor: Mapping[str, Any]) -> str:
    """Integrity mode a payload object descriptor declares."""

    if "crc32" not in descriptor:
        return "sha256"
    return "none" if descriptor["crc32"] is None else "fast"


def _payload_matches_descriptor(descriptor: Mapping[str, Any], encoded) -> bool:
    """Length plus the descriptor's own payload check (SHA-256, CRC32, or none)."""

    if len(encoded) != int(descriptor["bytes"]):
        return False
    mode = _descriptor_integrity(descriptor)
    if mode == "sha256":
        return _sha256(encoded) == descriptor["sha256"]
    if mode == "fast":
        return (zlib.crc32(encoded) & 0xFFFFFFFF) == descriptor["crc32"]
    return True


def _validate_object_descriptor_keys(descriptor: Mapping[str, Any], label: str) -> None:
    keys = {"sha256", "bytes", "encoded_start", "encoded_end"}
    if "crc32" in descriptor:
        keys.add("crc32")
        crc = descriptor["crc32"]
        if crc is not None and (type(crc) is not int or not 0 <= crc <= 0xFFFFFFFF):
            raise CacheFormatError(f"{label} crc32 is invalid")
    _strict_keys(descriptor, keys, label)


# Flat page-history publication schemas. ``page-tail-cow-v2`` content-addresses
# every payload object by SHA-256; the ``v3`` schemas hash only identities,
# manifests and stage headers, carry a CRC32 per payload object and per layer
# tail (``fast``) or no payload check (``none``), and name objects by a digest
# of the boundary identity, kind, index and a nonce instead of their content.
_FLAT_PAGE_SCHEMAS = ("page-tail-cow-v2", "page-tail-cow-v3-fast", "page-tail-cow-v3-none")
_PAGE_TAIL_SCHEMAS = ("page-tail-cow-v1", *_FLAT_PAGE_SCHEMAS)


def _payload_integrity(publication_schema: str) -> str:
    if publication_schema == "page-tail-cow-v3-fast":
        return "fast"
    if publication_schema == "page-tail-cow-v3-none":
        return "none"
    return "sha256"


def _boundary_token(context_digest: str, committed_tokens: int) -> str:
    return hashlib.sha256(
        b"sparkcache-boundary-token/v1\x00"
        + f"{context_digest}:{int(committed_tokens)}".encode("ascii")
    ).hexdigest()


def _derived_object_name(
    identity: "CacheIdentity", context_digest: str, kind: str, index: int, nonce: str
) -> str:
    """Object name for payloads that are not content-addressed (v3 schemas)."""

    return hashlib.sha256(
        b"sparkcache-object-name/v1\x00"
        + f"{identity.storage_key}:{context_digest}:{kind}:{index}:{nonce}".encode("ascii")
    ).hexdigest()


def _object_descriptor(
    integrity: str, name: str, encoded: bytes, start: int, end: int
) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "sha256": name,
        "bytes": len(encoded),
        "encoded_start": start,
        "encoded_end": end,
    }
    if integrity == "fast":
        descriptor["crc32"] = zlib.crc32(encoded) & 0xFFFFFFFF
    elif integrity == "none":
        descriptor["crc32"] = None
    return descriptor


def _descriptor_integrity(descriptor: Mapping[str, Any]) -> str:
    """Integrity mode a payload object descriptor declares."""

    if "crc32" not in descriptor:
        return "sha256"
    return "none" if descriptor["crc32"] is None else "fast"


def _payload_matches_descriptor(descriptor: Mapping[str, Any], encoded) -> bool:
    """Length plus the descriptor's own payload check (SHA-256, CRC32, or none)."""

    if len(encoded) != int(descriptor["bytes"]):
        return False
    mode = _descriptor_integrity(descriptor)
    if mode == "sha256":
        return _sha256(encoded) == descriptor["sha256"]
    if mode == "fast":
        return (zlib.crc32(encoded) & 0xFFFFFFFF) == descriptor["crc32"]
    return True


def _validate_object_descriptor_keys(descriptor: Mapping[str, Any], label: str) -> None:
    keys = {"sha256", "bytes", "encoded_start", "encoded_end"}
    if "crc32" in descriptor:
        keys.add("crc32")
        crc = descriptor["crc32"]
        if crc is not None and (type(crc) is not int or not 0 <= crc <= 0xFFFFFFFF):
            raise CacheFormatError(f"{label} crc32 is invalid")
    _strict_keys(descriptor, keys, label)


def _allocated_bytes(metadata: os.stat_result) -> int:
    blocks = getattr(metadata, "st_blocks", 0)
    return int(blocks) * 512 if blocks else int(metadata.st_size)


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes on POSIX filesystems.

    Windows does not expose a portable directory ``fsync`` through Python.
    SparkCache's deployment target is Linux; skipping here keeps the
    model-free test suite portable without weakening the Linux contract.
    """

    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_durable_directory(path: Path) -> None:
    """Create missing path components and persist each parent entry."""

    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            raise OSError(f"cannot find an existing ancestor for {path}")
        cursor = parent
    if not cursor.is_dir():
        raise NotADirectoryError(cursor)
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            if not directory.is_dir():
                raise
        else:
            _fsync_directory(directory)
            _fsync_directory(directory.parent)


def validate_clear_once_request(
    root: Path | str,
    token: object,
) -> tuple[Path, str]:
    """Validate a bounded operator token and its rank-local cache root.

    The clear operation removes only named SparkCache data directories, never
    the configured root itself. Requiring an absolute, non-broad, non-symlinked
    root prevents a malformed launch option from turning those directory names
    into deletion targets under a filesystem root or user home.
    """

    if not isinstance(token, str) or _CLEAR_ONCE_TOKEN.fullmatch(token) is None:
        raise ValueError(
            "spark_cache_clear_once must be a 1-128 character string using"
            " letters, digits, '.', '_', ':', '@', '+', or '-'"
        )
    configured = Path(root)
    if not configured.is_absolute():
        raise ValueError("spark_cache_clear_once requires an absolute cache root")
    if ".." in configured.parts:
        raise ValueError("spark_cache_clear_once rejects parent path traversal")
    try:
        resolved = configured.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ValueError(
            "spark_cache_clear_once cache root cannot be resolved"
        ) from error
    anchor = Path(resolved.anchor)
    if resolved == anchor or len(resolved.parts) < 3:
        raise ValueError("spark_cache_clear_once rejects broad cache roots")
    try:
        user_home = Path.home().resolve(strict=False)
    except (OSError, RuntimeError):
        user_home = None
    if user_home is not None and resolved == user_home:
        raise ValueError("spark_cache_clear_once rejects the user home directory")

    # Reject an existing symlink in any configured path component. A mount may
    # be the intended rank-local NVMe root, but a symlink could redirect the
    # fixed SparkCache child names after operator validation.
    cursor = configured
    while cursor != cursor.parent:
        if cursor.is_symlink():
            raise ValueError("spark_cache_clear_once rejects symlinked cache roots")
        cursor = cursor.parent

    token_digest = _sha256(
        _CLEAR_ONCE_SCHEMA.encode("ascii") + b"\0" + token.encode("utf-8")
    )
    return resolved, token_digest


def _remove_tree_without_following_links(path: Path) -> None:
    """Remove one owned tree without traversing symlinks or nested mounts."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        path.unlink()
        _fsync_directory(path.parent)
        return
    if path.is_mount():
        raise OSError(f"cache-owned path is a separate mount: {path}")
    for child in tuple(path.iterdir()):
        _remove_tree_without_following_links(child)
    _fsync_directory(path)
    path.rmdir()
    _fsync_directory(path.parent)


def _publish_clear_once_completion(path: Path, token_digest: str) -> None:
    """Atomically and durably record one successfully completed clear."""

    payload = _canonical_json(
        {
            "schema": _CLEAR_ONCE_SCHEMA,
            "token_sha256": token_digest,
        }
    )
    temporary = path.with_name(f".{path.name}.writing-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    _fsync_directory(path.parent)


def _clear_once_completed(path: Path, token_digest: str) -> bool:
    """Return whether one marker proves completion for its token digest."""

    try:
        if path.is_symlink() or not path.is_file():
            return False
        marker = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(marker, dict)
        and marker.get("schema") == _CLEAR_ONCE_SCHEMA
        and marker.get("token_sha256") == token_digest
        and set(marker) == {"schema", "token_sha256"}
    )


def _durable_payload_matches(path: Path, payload: bytes) -> bool:
    """Authenticate and flush an existing object before adopting its bytes."""

    try:
        # Windows requires a writable handle for FlushFileBuffers; no bytes
        # are modified. POSIX permits fsync on the read-only handle.
        with path.open("r+b" if os.name == "nt" else "rb") as stream:
            if stream.read(len(payload) + 1) != payload:
                return False
            # A concurrent publisher can expose its hard link before its
            # directory barrier. Adoption also needs a data barrier for files
            # that were populated outside the immutable publisher.
            os.fsync(stream.fileno())
            return True
    except FileNotFoundError:
        return False
    except OSError as error:
        raise CommitConflict(
            f"cannot verify existing immutable object {path}"
        ) from error


def _publish_immutable(
    path: Path,
    payload: bytes,
) -> None:
    """Durably publish complete bytes once without an overwrite race."""

    _ensure_durable_directory(path.parent)
    temporary = path.with_name(f".{path.name}.writing-{uuid.uuid4().hex}")
    try:
        publication = _ACTIVE_PUBLICATION.get()
        if _durable_payload_matches(path, payload):
            if publication is not None:
                publication.record_deduplicated(len(payload))
            return
        with temporary.open("xb") as stream:
            if publication is not None:
                publication.record_staged(len(payload))
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
            if publication is not None:
                publication.record_unique(len(payload))
        except FileExistsError:
            if not _durable_payload_matches(path, payload):
                raise CommitConflict(
                    f"different immutable object already committed at {path}"
                )
            if publication is not None:
                publication.record_deduplicated(len(payload))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        # One directory fsync after link+temporary-unlink durably records both
        # changes. Manifest publication does not return success until this
        # barrier completes.
        _fsync_directory(path.parent)


def _publish_immutable_batch(
    objects: Sequence[tuple[Path, bytes]],
) -> None:
    """Durably publish one directory-local content-addressed macro-batch.

    Every object's data reaches stable storage before any descriptor can be
    appended to a transaction. File-data barriers run concurrently; all hard
    links, repairs, and temporary-name removals share one final directory
    barrier. A differently encoded object at the expected content-addressed
    path is corruption, so a publisher holding the matching payload repairs it
    by atomic replacement.
    """

    if not objects:
        return
    parent = objects[0][0].parent
    if any(path.parent != parent for path, _payload in objects):
        raise ValueError("immutable macro-batch must share one directory")
    _ensure_durable_directory(parent)
    staged: list[tuple[Path, bytes, Path]] = []
    publication = _ACTIVE_PUBLICATION.get()
    def stage(item: tuple[Path, bytes, Path]) -> None:
        _path, payload, temporary = item
        with temporary.open("xb") as stream:
            if publication is not None:
                publication.record_staged(len(payload))
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    try:
        seen: dict[Path, bytes] = {}
        for path, payload in objects:
            previous = seen.get(path)
            if previous is not None:
                if previous != payload:
                    raise CommitConflict(
                        f"different immutable payloads in one batch for {path}"
                    )
                if publication is not None:
                    publication.record_deduplicated(len(payload))
                continue
            seen[path] = payload
            if _durable_payload_matches(path, payload):
                if publication is not None:
                    publication.record_deduplicated(len(payload))
            else:
                staged.append(
                    (
                        path,
                        payload,
                        path.with_name(f".{path.name}.writing-{uuid.uuid4().hex}"),
                    )
                )
        if staged:
            worker_count = min(8, len(staged))
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                tuple(pool.map(stage, staged))
        for path, payload, temporary in staged:
            try:
                os.link(temporary, path)
                if publication is not None:
                    publication.record_unique(len(payload))
            except FileExistsError:
                if not _durable_payload_matches(path, payload):
                    expected_name = f"{_sha256(payload)}{path.suffix}"
                    if path.name != expected_name:
                        raise CommitConflict(
                            f"different immutable object already committed at {path}"
                        )
                    os.replace(temporary, path)
                    if publication is not None:
                        publication.record_unique(len(payload))
                elif publication is not None:
                    publication.record_deduplicated(len(payload))
    finally:
        for _path, _payload, temporary in staged:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        # Chunk contents were each fsynced above. This one metadata barrier
        # makes every successful hard link and temporary unlink durable.
        _fsync_directory(parent)


def _validate_digest(value: str, field: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _encode_chunk(chunk: ContextChunk) -> bytes:
    records: list[dict[str, Any]] = []
    ordered_records: list[bytes] = []
    payload_bytes = 0
    for kind in sorted(chunk.records, key=lambda item: item.value):
        value = chunk.records[kind]
        offset = payload_bytes
        payload_bytes += len(value)
        ordered_records.append(value)
        records.append(
            {
                "kind": kind.value,
                "offset": offset,
                "length": len(value),
                "sha256": _sha256(value),
            }
        )
    header = _canonical_json(
        {
            "format_abi": FORMAT_ABI,
            "logical_start": chunk.logical_start,
            "logical_end": chunk.logical_end,
            "records": records,
        }
    )
    # bytes.join calculates the final size once and copies every component
    # directly into that allocation. The v1 encoder first copied records into
    # a payload bytearray and then copied that payload during concatenation.
    # Offsets/header/checksums remain byte-identical.
    prefix = _CHUNK_PREFIX.pack(_CHUNK_MAGIC, FORMAT_ABI, len(header))
    return b"".join((prefix, header, *ordered_records))


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise CacheFormatError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _validate_manifest_metadata(
    manifest: Any,
    key: EntryKey,
    *,
    expected_identity: CacheIdentity | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Validate every manifest field without reading referenced chunk payloads."""

    if not isinstance(manifest, dict):
        raise CacheFormatError("manifest is not an object")
    _strict_keys(
        manifest,
        {
            "format_abi",
            "identity",
            "context_digest",
            "committed_tokens",
            "chunks",
        },
        "manifest",
    )
    identity_wire = manifest["identity"]
    if not isinstance(identity_wire, dict):
        raise CacheFormatError("manifest identity is not an object")
    try:
        identity = CacheIdentity(**identity_wire)
    except (TypeError, ValueError) as error:
        raise CacheFormatError("manifest identity is invalid") from error
    for field in (
        "tp_degree",
        "dcp_degree",
        "chunk_tokens",
        "dcp_shard_rank",
        "tp_shard_rank",
    ):
        if type(getattr(identity, field)) is not int:
            raise CacheFormatError(f"manifest identity {field} is not an integer")
    if identity.to_wire() != identity_wire:
        raise CacheFormatError("manifest identity fields differ")

    incompatible = (
        type(manifest["format_abi"]) is not int
        or manifest["format_abi"] != FORMAT_ABI
        or identity.storage_key != key.storage_key
        or manifest["context_digest"] != key.context_digest
    )
    if expected_identity is not None:
        expected_wire = expected_identity.to_wire()
        incompatible = incompatible or (
            identity_wire != expected_wire
            or _canonical_json(identity_wire) != _canonical_json(expected_wire)
        )
    if incompatible:
        error_type = (
            _IncompatibleManifestError
            if expected_identity is not None
            else CacheFormatError
        )
        raise error_type("manifest identity differs")

    committed_tokens = manifest["committed_tokens"]
    if type(committed_tokens) is not int or committed_tokens <= 0:
        raise CacheFormatError("committed_tokens must be a positive integer")
    chunks = manifest["chunks"]
    if not isinstance(chunks, list) or not chunks:
        raise CacheFormatError("manifest has no chunks")

    expected_start = 0
    for chunk_index, descriptor in enumerate(chunks):
        if not isinstance(descriptor, dict):
            raise CacheFormatError("chunk descriptor is not an object")
        _strict_keys(
            descriptor,
            {"sha256", "bytes", "logical_start", "logical_end"},
            "chunk descriptor",
        )
        _validate_digest(descriptor["sha256"], "chunk sha256")
        encoded_bytes = descriptor["bytes"]
        logical_start = descriptor["logical_start"]
        logical_end = descriptor["logical_end"]
        if type(encoded_bytes) is not int or encoded_bytes <= 0:
            raise CacheFormatError("chunk bytes must be a positive integer")
        if (
            type(logical_start) is not int
            or type(logical_end) is not int
            or logical_start != expected_start
            or logical_end <= expected_start
        ):
            raise CacheFormatError("non-contiguous logical chunk range")
        token_count = logical_end - logical_start
        if token_count > identity.chunk_tokens or (
            chunk_index != len(chunks) - 1 and token_count != identity.chunk_tokens
        ):
            raise CacheFormatError("chunk range disagrees with identity geometry")
        expected_start = logical_end
    if expected_start != committed_tokens:
        raise CacheFormatError("committed token count mismatch")
    return tuple(chunks)


def _validate_prefix_segment(
    segment: Any,
    *,
    storage_key: str,
    expected_first_chunk: int,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(segment, dict):
        raise CacheFormatError("prefix descriptor segment is not an object")
    _strict_keys(
        segment,
        {
            "schema",
            "storage_key",
            "parent_sha256",
            "first_chunk_index",
            "descriptors",
        },
        "prefix descriptor segment",
    )
    if (
        segment["schema"] != _PREFIX_SEGMENT_SCHEMA
        or segment["storage_key"] != storage_key
        or type(segment["first_chunk_index"]) is not int
        or segment["first_chunk_index"] != expected_first_chunk
    ):
        raise CacheFormatError("prefix descriptor segment identity differs")
    parent = segment["parent_sha256"]
    if parent is not None:
        try:
            _validate_digest(parent, "prefix segment parent_sha256")
        except ValueError as error:
            raise CacheFormatError(str(error)) from error
    descriptors = segment["descriptors"]
    if (
        not isinstance(descriptors, list)
        or not descriptors
        or len(descriptors) > _PREFIX_SEGMENT_DESCRIPTORS
    ):
        raise CacheFormatError("prefix descriptor segment size is invalid")
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise CacheFormatError("prefix chunk descriptor is not an object")
        _strict_keys(
            descriptor,
            {"sha256", "bytes", "logical_start", "logical_end"},
            "prefix chunk descriptor",
        )
    return tuple(descriptors)


def _validate_prefix_alias(
    alias: Any,
    *,
    identity: CacheIdentity,
    context_digest: str,
) -> tuple[int, int, str]:
    if not isinstance(alias, dict):
        raise CacheFormatError("prefix alias is not an object")
    _strict_keys(
        alias,
        {
            "schema",
            "format_abi",
            "storage_mode",
            "identity",
            "context_digest",
            "committed_tokens",
            "chunk_count",
            "tail_segment_sha256",
            "source_manifest_digest",
            "metadata_sha256",
        },
        "prefix alias",
    )
    if (
        alias["schema"] != _PREFIX_ALIAS_SCHEMA
        or type(alias["format_abi"]) is not int
        or alias["format_abi"] != FORMAT_ABI
        or alias["storage_mode"] != "per_token_rows"
        or alias["identity"] != identity.to_wire()
        or alias["context_digest"] != context_digest
    ):
        raise _IncompatibleManifestError("prefix alias identity differs")
    committed_tokens = alias["committed_tokens"]
    chunk_count = alias["chunk_count"]
    if (
        type(committed_tokens) is not int
        or committed_tokens <= 0
        or type(chunk_count) is not int
        or chunk_count <= 0
    ):
        raise CacheFormatError("prefix alias geometry is invalid")
    try:
        _validate_digest(alias["tail_segment_sha256"], "tail_segment_sha256")
        _validate_digest(alias["source_manifest_digest"], "source_manifest_digest")
        _validate_digest(alias["metadata_sha256"], "metadata_sha256")
    except ValueError as error:
        raise CacheFormatError(str(error)) from error
    authenticated = dict(alias)
    metadata_digest = authenticated.pop("metadata_sha256")
    if _sha256(_canonical_json(authenticated)) != metadata_digest:
        raise CacheFormatError("prefix alias metadata checksum mismatch")
    return committed_tokens, chunk_count, alias["tail_segment_sha256"]


def _validate_tail_manifest_root(
    manifest: Any,
    *,
    identity: CacheIdentity,
    context_digest: str,
) -> tuple[int, int, int, str | None, tuple[Mapping[str, Any], ...]]:
    """Validate the authenticated root of one copy-on-write context graph."""

    if not isinstance(manifest, dict):
        raise CacheFormatError("tail manifest is not an object")
    _strict_keys(
        manifest,
        {
            "schema",
            "format_abi",
            "identity",
            "context_digest",
            "committed_tokens",
            "base_context_digest",
            "base_manifest_sha256",
            "base_committed_tokens",
            "reused_tokens",
            "base_chunk_count",
            "base_tail_segment_sha256",
            "tail_chunks",
            "metadata_sha256",
        },
        "tail manifest",
    )
    try:
        _validate_digest(
            manifest["base_context_digest"],
            "tail manifest base_context_digest",
        )
        _validate_digest(
            manifest["base_manifest_sha256"],
            "tail manifest base_manifest_sha256",
        )
        _validate_digest(manifest["metadata_sha256"], "tail manifest metadata_sha256")
    except ValueError as error:
        raise CacheFormatError(str(error)) from error
    authenticated = dict(manifest)
    metadata_digest = authenticated.pop("metadata_sha256")
    if _sha256(_canonical_json(authenticated)) != metadata_digest:
        raise CacheFormatError("tail manifest metadata checksum mismatch")
    if (
        manifest["schema"] != _TAIL_MANIFEST_SCHEMA
        or type(manifest["format_abi"]) is not int
        or manifest["format_abi"] != FORMAT_ABI
        or identity.publication_schema != "tail-cow-v1"
        or manifest["identity"] != identity.to_wire()
        or manifest["context_digest"] != context_digest
    ):
        raise _IncompatibleManifestError("tail manifest identity differs")
    committed_tokens = manifest["committed_tokens"]
    base_committed_tokens = manifest["base_committed_tokens"]
    reused_tokens = manifest["reused_tokens"]
    chunk_count = manifest["base_chunk_count"]
    if any(
        type(value) is not int
        for value in (
            committed_tokens,
            base_committed_tokens,
            reused_tokens,
            chunk_count,
        )
    ):
        raise CacheFormatError("tail manifest token and chunk counts must be integers")
    if not (
        committed_tokens > base_committed_tokens > 0
        and 0 <= reused_tokens <= base_committed_tokens
        and reused_tokens % identity.chunk_tokens == 0
        and chunk_count == reused_tokens // identity.chunk_tokens
    ):
        raise CacheFormatError("tail manifest boundary geometry differs")
    segment_digest = manifest["base_tail_segment_sha256"]
    if chunk_count == 0:
        if segment_digest is not None:
            raise CacheFormatError("tail manifest has a segment without base chunks")
    elif not isinstance(segment_digest, str):
        raise CacheFormatError("tail manifest base descriptor chain is missing")
    else:
        try:
            _validate_digest(segment_digest, "tail manifest base_tail_segment_sha256")
        except ValueError as error:
            raise CacheFormatError(str(error)) from error
    tail_chunks = manifest["tail_chunks"]
    if not isinstance(tail_chunks, list) or not tail_chunks:
        raise CacheFormatError("tail manifest has no tail chunks")
    for descriptor in tail_chunks:
        if not isinstance(descriptor, dict):
            raise CacheFormatError("tail chunk descriptor is not an object")
        _strict_keys(
            descriptor,
            {"sha256", "bytes", "logical_start", "logical_end"},
            "tail chunk descriptor",
        )
    return (
        committed_tokens,
        reused_tokens,
        chunk_count,
        segment_digest,
        tuple(tail_chunks),
    )


def _validate_page_delta_root(
    manifest: Any,
    *,
    identity: CacheIdentity,
    context_digest: str,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    if not isinstance(manifest, dict):
        raise CacheFormatError("page delta manifest is not an object")
    schema = manifest.get("schema")
    if schema == _PAGE_DELTA_MANIFEST_SCHEMA_V3:
        _strict_keys(
            manifest,
            {
                "schema",
                "format_abi",
                "identity",
                "context_digest",
                "committed_tokens",
                "base_context_digest",
                "base_committed_tokens",
                "base_root",
                "base_root_sha256",
                "layout_sha256",
                "base_block_counts",
                "result_block_counts",
                "delta_stages",
                "logical_chunk_tokens",
                "metadata_sha256",
            },
            "flat page delta manifest",
        )
        authenticated = dict(manifest)
        metadata_digest = authenticated.pop("metadata_sha256")
        try:
            for field in (
                "metadata_sha256",
                "context_digest",
                "base_context_digest",
                "base_root_sha256",
                "layout_sha256",
            ):
                _validate_digest(
                    metadata_digest if field == "metadata_sha256" else manifest[field],
                    f"flat page delta {field}",
                )
        except ValueError as error:
            raise CacheFormatError(str(error)) from error
        if (
            _sha256(_canonical_json(authenticated)) != metadata_digest
            or manifest["format_abi"] != FORMAT_ABI
            or identity.publication_schema not in _FLAT_PAGE_SCHEMAS
            or manifest["identity"] != identity.to_wire()
            or manifest["context_digest"] != context_digest
            or manifest["logical_chunk_tokens"] != identity.chunk_tokens
            or not isinstance(manifest["base_root"], dict)
            or _sha256(_canonical_json(manifest["base_root"]))
            != manifest["base_root_sha256"]
        ):
            raise _IncompatibleManifestError(
                "flat page delta manifest identity differs"
            )
        base_counts = manifest["base_block_counts"]
        result_counts = manifest["result_block_counts"]
        stages = manifest["delta_stages"]
        if (
            type(manifest["base_committed_tokens"]) is not int
            or type(manifest["committed_tokens"]) is not int
            or manifest["committed_tokens"]
            <= manifest["base_committed_tokens"]
            <= 0
            or not isinstance(base_counts, list)
            or not base_counts
            or not isinstance(result_counts, list)
            or len(result_counts) != len(base_counts)
            or not isinstance(stages, list)
            or not stages
        ):
            raise CacheFormatError("flat page delta geometry differs")
        prior_digest = manifest["base_context_digest"]
        prior_tokens = manifest["base_committed_tokens"]
        prior_counts = base_counts
        descriptors: list[Mapping[str, Any]] = []
        for stage in stages:
            if not isinstance(stage, dict):
                raise CacheFormatError("flat page delta stage is not an object")
            stage_keys = {
                "context_digest",
                "committed_tokens",
                "base_context_digest",
                "base_committed_tokens",
                "base_block_counts",
                "result_block_counts",
                "delta_encoded_bytes",
                "delta_object_bytes",
                "delta_objects",
                "delta_sha256",
            }
            # The v3 schemas bind each stage header by digest here because
            # their first object is not content-addressed.
            if identity.publication_schema != "page-tail-cow-v2":
                stage_keys.add("header_sha256")
            _strict_keys(stage, stage_keys, "flat page delta stage")
            try:
                _validate_digest(stage["context_digest"], "stage context digest")
                _validate_digest(stage["delta_sha256"], "stage delta sha256")
                if "header_sha256" in stage:
                    _validate_digest(stage["header_sha256"], "stage header sha256")
            except ValueError as error:
                raise CacheFormatError(str(error)) from error
            objects = stage["delta_objects"]
            if (
                stage["base_context_digest"] != prior_digest
                or stage["base_committed_tokens"] != prior_tokens
                or stage["base_block_counts"] != prior_counts
                or type(stage["committed_tokens"]) is not int
                or stage["committed_tokens"] <= prior_tokens
                or not isinstance(stage["result_block_counts"], list)
                or len(stage["result_block_counts"]) != len(base_counts)
                or type(stage["delta_encoded_bytes"]) is not int
                or stage["delta_encoded_bytes"] <= 0
                or stage["delta_object_bytes"] != _PAGE_DELTA_OBJECT_BYTES
                or not isinstance(objects, list)
                or not objects
            ):
                raise CacheFormatError("flat page delta stage geometry differs")
            expected_start = 0
            for index, descriptor in enumerate(objects):
                if not isinstance(descriptor, dict):
                    raise CacheFormatError(
                        "flat page delta object descriptor is not an object"
                    )
                _validate_object_descriptor_keys(
                    descriptor, "flat page delta object descriptor"
                )
                try:
                    _validate_digest(
                        descriptor["sha256"], "flat page delta object sha256"
                    )
                except ValueError as error:
                    raise CacheFormatError(str(error)) from error
                size = descriptor["bytes"]
                start = descriptor["encoded_start"]
                end = descriptor["encoded_end"]
                if (
                    type(size) is not int
                    or type(start) is not int
                    or type(end) is not int
                    or size <= 0
                    or start != expected_start
                    or end != start + size
                    or (
                        index < len(objects) - 1
                        and size != _PAGE_DELTA_OBJECT_BYTES
                    )
                    or size > _PAGE_DELTA_OBJECT_BYTES
                ):
                    raise CacheFormatError(
                        "flat page delta object descriptor geometry differs"
                    )
                descriptors.append(descriptor)
                expected_start = end
            if expected_start != stage["delta_encoded_bytes"]:
                raise CacheFormatError("flat page delta object coverage differs")
            prior_digest = stage["context_digest"]
            prior_tokens = stage["committed_tokens"]
            prior_counts = stage["result_block_counts"]
        if (
            prior_digest != manifest["context_digest"]
            or prior_tokens != manifest["committed_tokens"]
            or prior_counts != result_counts
        ):
            raise CacheFormatError("flat page delta terminal stage differs")
        return manifest["base_root"], tuple(descriptors)
    if schema == _PAGE_DELTA_MANIFEST_SCHEMA:
        delta_keys = {"delta_chunks"}
    elif schema == _PAGE_DELTA_MANIFEST_SCHEMA_V2:
        delta_keys = {
            "delta_encoded_bytes",
            "delta_object_bytes",
            "delta_objects",
            "delta_sha256",
            "logical_chunk_tokens",
        }
    else:
        raise _IncompatibleManifestError("page delta manifest schema differs")
    _strict_keys(
        manifest,
        {
            "schema",
            "format_abi",
            "identity",
            "context_digest",
            "committed_tokens",
            "base_context_digest",
            "base_committed_tokens",
            "base_root",
            "base_root_sha256",
            "layout_sha256",
            "base_block_counts",
            "result_block_counts",
            "metadata_sha256",
            *delta_keys,
        },
        "page delta manifest",
    )
    authenticated = dict(manifest)
    metadata_digest = authenticated.pop("metadata_sha256")
    try:
        _validate_digest(metadata_digest, "page delta metadata_sha256")
        _validate_digest(manifest["base_context_digest"], "base_context_digest")
        _validate_digest(manifest["base_root_sha256"], "base_root_sha256")
        _validate_digest(manifest["layout_sha256"], "layout_sha256")
    except ValueError as error:
        raise CacheFormatError(str(error)) from error
    if _sha256(_canonical_json(authenticated)) != metadata_digest:
        raise CacheFormatError("page delta metadata checksum mismatch")
    if (
        manifest["format_abi"] != FORMAT_ABI
        or identity.publication_schema != "page-tail-cow-v1"
        or manifest["identity"] != identity.to_wire()
        or manifest["context_digest"] != context_digest
    ):
        raise _IncompatibleManifestError("page delta manifest identity differs")
    if (
        type(manifest["committed_tokens"]) is not int
        or type(manifest["base_committed_tokens"]) is not int
        or manifest["committed_tokens"] <= manifest["base_committed_tokens"] <= 0
        or not isinstance(manifest["base_root"], dict)
        or _sha256(_canonical_json(manifest["base_root"]))
        != manifest["base_root_sha256"]
        or not isinstance(manifest["base_block_counts"], list)
        or not isinstance(manifest["result_block_counts"], list)
        or not manifest["base_block_counts"]
        or len(manifest["base_block_counts"]) != len(manifest["result_block_counts"])
    ):
        raise CacheFormatError("page delta manifest geometry differs")
    if schema == _PAGE_DELTA_MANIFEST_SCHEMA:
        synthetic = {
            "format_abi": FORMAT_ABI,
            "identity": identity.to_wire(),
            "context_digest": context_digest,
            "committed_tokens": manifest["committed_tokens"],
            "chunks": manifest["delta_chunks"],
        }
        descriptors = _validate_manifest_metadata(
            synthetic,
            EntryKey(identity.storage_key, context_digest),
            expected_identity=identity,
        )
        return manifest["base_root"], descriptors

    try:
        _validate_digest(manifest["delta_sha256"], "page delta payload sha256")
    except ValueError as error:
        raise CacheFormatError(str(error)) from error
    encoded_bytes = manifest["delta_encoded_bytes"]
    object_bytes = manifest["delta_object_bytes"]
    if (
        type(encoded_bytes) is not int
        or encoded_bytes <= 0
        or type(object_bytes) is not int
        or not 0 < object_bytes <= _MAX_PAGE_DELTA_OBJECT_BYTES
        or manifest["logical_chunk_tokens"] != identity.chunk_tokens
    ):
        raise CacheFormatError("page delta object geometry differs")
    objects = manifest["delta_objects"]
    if not isinstance(objects, list) or not objects:
        raise CacheFormatError("page delta object descriptors are invalid")
    expected_start = 0
    descriptors: list[Mapping[str, Any]] = []
    for index, descriptor in enumerate(objects):
        if not isinstance(descriptor, dict):
            raise CacheFormatError("page delta object descriptor is not an object")
        _strict_keys(
            descriptor,
            {"sha256", "bytes", "encoded_start", "encoded_end"},
            "page delta object descriptor",
        )
        try:
            _validate_digest(descriptor["sha256"], "page delta object sha256")
        except ValueError as error:
            raise CacheFormatError(str(error)) from error
        size = descriptor["bytes"]
        start = descriptor["encoded_start"]
        end = descriptor["encoded_end"]
        if (
            type(size) is not int
            or type(start) is not int
            or type(end) is not int
            or size <= 0
            or start != expected_start
            or end != start + size
            or (index < len(objects) - 1 and size != object_bytes)
            or size > object_bytes
        ):
            raise CacheFormatError("page delta object descriptor geometry differs")
        descriptors.append(descriptor)
        expected_start = end
    if expected_start != encoded_bytes:
        raise CacheFormatError("page delta object coverage differs")
    return manifest["base_root"], tuple(descriptors)


def _validate_page_snapshot_root(
    manifest: Any,
    *,
    identity: CacheIdentity,
    context_digest: str,
) -> tuple[Mapping[str, Any], ...]:
    """Validate one flat opaque page snapshot without reading its objects."""

    if not isinstance(manifest, dict):
        raise CacheFormatError("page snapshot manifest is not an object")
    _strict_keys(
        manifest,
        {
            "schema",
            "format_abi",
            "storage_mode",
            "identity",
            "context_digest",
            "committed_tokens",
            "snapshot_encoded_bytes",
            "snapshot_object_bytes",
            "snapshot_objects",
            "snapshot_sha256",
            "logical_chunk_tokens",
            "logical_chunk_count",
            "metadata_sha256",
        },
        "page snapshot manifest",
    )
    authenticated = dict(manifest)
    metadata_digest = authenticated.pop("metadata_sha256")
    try:
        _validate_digest(metadata_digest, "page snapshot metadata_sha256")
        _validate_digest(manifest["snapshot_sha256"], "page snapshot sha256")
    except ValueError as error:
        raise CacheFormatError(str(error)) from error
    if _sha256(_canonical_json(authenticated)) != metadata_digest:
        raise CacheFormatError("page snapshot metadata checksum mismatch")
    if (
        manifest["schema"] != _PAGE_SNAPSHOT_MANIFEST_SCHEMA
        or type(manifest["format_abi"]) is not int
        or manifest["format_abi"] != FORMAT_ABI
        or manifest["storage_mode"] != "block_pages_v1"
        or identity.publication_schema not in ("", *_PAGE_TAIL_SCHEMAS)
        or identity.required_records
        != frozenset((StateRecord.TARGET_CKV, StateRecord.LOGICAL_POSITIONS))
        or manifest["identity"] != identity.to_wire()
        or manifest["context_digest"] != context_digest
    ):
        raise _IncompatibleManifestError("page snapshot manifest identity differs")

    committed_tokens = manifest["committed_tokens"]
    encoded_bytes = manifest["snapshot_encoded_bytes"]
    object_bytes = manifest["snapshot_object_bytes"]
    logical_chunk_count = manifest["logical_chunk_count"]
    if (
        type(committed_tokens) is not int
        or committed_tokens <= 0
        or committed_tokens % identity.chunk_tokens
        or type(encoded_bytes) is not int
        or encoded_bytes <= 0
        or type(object_bytes) is not int
        or not 0 < object_bytes <= _MAX_PAGE_SNAPSHOT_OBJECT_BYTES
        or manifest["logical_chunk_tokens"] != identity.chunk_tokens
        or type(logical_chunk_count) is not int
        or logical_chunk_count != committed_tokens // identity.chunk_tokens
    ):
        raise CacheFormatError("page snapshot object geometry differs")

    objects = manifest["snapshot_objects"]
    if not isinstance(objects, list) or not objects:
        raise CacheFormatError("page snapshot object descriptors are invalid")
    expected_start = 0
    descriptors: list[Mapping[str, Any]] = []
    for index, descriptor in enumerate(objects):
        if not isinstance(descriptor, dict):
            raise CacheFormatError("page snapshot object descriptor is not an object")
        _validate_object_descriptor_keys(descriptor, "page snapshot object descriptor")
        try:
            _validate_digest(descriptor["sha256"], "page snapshot object sha256")
        except ValueError as error:
            raise CacheFormatError(str(error)) from error
        size = descriptor["bytes"]
        start = descriptor["encoded_start"]
        end = descriptor["encoded_end"]
        if (
            type(size) is not int
            or type(start) is not int
            or type(end) is not int
            or size <= 0
            or start != expected_start
            or end != start + size
            or (index < len(objects) - 1 and size != object_bytes)
            or size > object_bytes
        ):
            raise CacheFormatError("page snapshot object descriptor geometry differs")
        descriptors.append(descriptor)
        expected_start = end
    if expected_start != encoded_bytes:
        raise CacheFormatError("page snapshot object coverage differs")
    return tuple(descriptors)


def _decode_chunk(
    encoded: bytes,
    *,
    verify_record_checksums: bool = True,
) -> ContextChunk:
    if len(encoded) < _CHUNK_PREFIX.size:
        raise CacheFormatError("truncated chunk prefix")
    magic, abi, header_length = _CHUNK_PREFIX.unpack_from(encoded)
    if magic != _CHUNK_MAGIC or abi != FORMAT_ABI:
        raise CacheFormatError("unsupported chunk magic or ABI")
    header_end = _CHUNK_PREFIX.size + header_length
    if header_end > len(encoded):
        raise CacheFormatError("truncated chunk header")
    try:
        header = json.loads(encoded[_CHUNK_PREFIX.size : header_end])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CacheFormatError("invalid chunk header") from error
    if not isinstance(header, dict):
        raise CacheFormatError("chunk header is not an object")
    _strict_keys(
        header,
        {"format_abi", "logical_start", "logical_end", "records"},
        "chunk header",
    )
    if header["format_abi"] != FORMAT_ABI or not isinstance(header["records"], list):
        raise CacheFormatError("unsupported chunk header")
    # Keep the encoded chunk as the backing store while descriptors are
    # validated. Each record is copied exactly once into its immutable bytes
    # snapshot instead of first copying the whole payload and then slicing it.
    raw_payload = memoryview(encoded)[header_end:]
    records: dict[StateRecord, bytes] = {}
    expected_offset = 0
    for item in header["records"]:
        if not isinstance(item, dict):
            raise CacheFormatError("record descriptor is not an object")
        _strict_keys(item, {"kind", "offset", "length", "sha256"}, "record")
        try:
            kind = StateRecord(item["kind"])
            offset = int(item["offset"])
            length = int(item["length"])
        except (ValueError, TypeError) as error:
            raise CacheFormatError("invalid record descriptor") from error
        if kind in records or offset < 0 or length < 0 or offset != expected_offset:
            raise CacheFormatError("duplicate or invalid record descriptor")
        value = raw_payload[offset : offset + length]
        if len(value) != length or (
            verify_record_checksums and _sha256(value) != item["sha256"]
        ):
            raise CacheFormatError("record payload checksum mismatch")
        records[kind] = value.tobytes()
        expected_offset += length
    if expected_offset != len(raw_payload):
        raise CacheFormatError("chunk payload contains unclaimed bytes")
    try:
        return ContextChunk(
            logical_start=int(header["logical_start"]),
            logical_end=int(header["logical_end"]),
            records=records,
        )
    except (TypeError, ValueError) as error:
        raise CacheFormatError(str(error)) from error


def _read_page_delta_object_batch(
    object_root: Path,
    descriptors: Sequence[Mapping[str, Any]],
) -> tuple[bytes, ...]:
    """Read one bounded batch of authenticated page-delta byte extents."""

    def read_one(descriptor: Mapping[str, Any]) -> bytes:
        encoded = (object_root / f"{descriptor['sha256']}.spcc").read_bytes()
        if not _payload_matches_descriptor(descriptor, encoded):
            raise CacheFormatError("page delta object checksum mismatch")
        return encoded

    if not descriptors:
        return ()
    with ThreadPoolExecutor(
        max_workers=min(len(descriptors), _PAGE_DELTA_READ_BATCH_SIZE)
    ) as pool:
        return tuple(pool.map(read_one, descriptors))


def _read_page_snapshot_object_batch(
    object_root: Path,
    descriptors: Sequence[Mapping[str, Any]],
) -> tuple[bytes, ...]:
    """Read one bounded batch of authenticated flat-snapshot extents."""

    def read_one(descriptor: Mapping[str, Any]) -> bytes:
        encoded = (object_root / f"{descriptor['sha256']}.spcc").read_bytes()
        if not _payload_matches_descriptor(descriptor, encoded):
            raise CacheFormatError("page snapshot object checksum mismatch")
        return encoded

    if not descriptors:
        return ()
    with ThreadPoolExecutor(
        max_workers=min(len(descriptors), _PAGE_SNAPSHOT_READ_BATCH_SIZE)
    ) as pool:
        return tuple(pool.map(read_one, descriptors))


def _object_extent_pieces(
    object_root: Path,
    descriptors: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
) -> tuple[tuple[Path, int, int, int], ...]:
    """Map one encoded-stream byte range onto its macro objects.

    Each piece is ``(path, object_bytes, object_offset, length)``. Descriptor
    geometry must already be validated by the manifest validator.
    """

    if start < 0 or end < start:
        raise CacheFormatError("page object extent is invalid")
    pieces = []
    cursor = start
    for descriptor in descriptors:
        object_start = int(descriptor["encoded_start"])
        object_end = int(descriptor["encoded_end"])
        piece_start = max(cursor, object_start)
        piece_end = min(end, object_end)
        if piece_start >= piece_end:
            continue
        pieces.append(
            (
                object_root / f"{descriptor['sha256']}.spcc",
                int(descriptor["bytes"]),
                piece_start - object_start,
                piece_end - piece_start,
            )
        )
        cursor = piece_end
        if cursor == end:
            break
    if cursor != end:
        raise CacheFormatError("page object extent is not covered")
    return tuple(pieces)


_PUBLICATION_READ_FILES = 64
_publication_read_files: ContextVar[Any] = ContextVar("publication_read_files", default=None)


class _PublicationReadFiles(AbstractContextManager):
    """Bound open-file reuse to one synchronous publication base restore.

    No payload or file descriptor survives the context. Each range is still
    checked against its authenticated tail digest, including on a handle hit.
    """

    def __init__(self) -> None:
        self.files: OrderedDict[Path, int] = OrderedDict()

    def __enter__(self):
        self.token = _publication_read_files.set(self)
        return self

    def open(self, path: Path, flags: int) -> int:
        descriptor = self.files.pop(path, None)
        if descriptor is None:
            if len(self.files) >= _PUBLICATION_READ_FILES:
                _, evicted = self.files.popitem(last=False)
                os.close(evicted)
            descriptor = os.open(path, flags)
        self.files[path] = descriptor
        return descriptor

    def __exit__(self, *_args):
        _publication_read_files.reset(self.token)
        for descriptor in self.files.values():
            os.close(descriptor)
        self.files.clear()


def _read_object_extent(
    pieces: Sequence[tuple[Path, int, int, int]],
) -> bytearray:
    """Read unauthenticated byte ranges from immutable macro objects.

    The caller must authenticate the returned bytes against a digest bound by
    already-authenticated metadata before using them. Each object must still
    be a regular file of its descriptor length.
    """

    total = sum(length for _path, _size, _offset, length in pieces)
    result = bytearray(total)
    view = memoryview(result)
    cursor = 0
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    files = _publication_read_files.get()
    try:
        for path, object_bytes, offset, length in pieces:
            try:
                descriptor = files.open(path, flags) if files is not None else os.open(path, flags)
            except OSError as error:
                raise CacheFormatError(f"page object is unavailable: {error}") from error
            try:
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_size != object_bytes
                ):
                    raise CacheFormatError("page object length differs")
                completed = 0
                while completed < length:
                    if hasattr(os, "pread"):
                        chunk = os.pread(
                            descriptor, length - completed, offset + completed
                        )
                    else:  # pragma: no cover - Windows test fallback
                        os.lseek(descriptor, offset + completed, os.SEEK_SET)
                        chunk = os.read(descriptor, length - completed)
                    if not chunk:
                        raise CacheFormatError("page object read was truncated")
                    view[cursor + completed : cursor + completed + len(chunk)] = chunk
                    completed += len(chunk)
            finally:
                if files is None:
                    os.close(descriptor)
            cursor += length
    finally:
        view.release()
    return result


def _read_object_extent_into(
    pieces: Sequence[tuple[Path, int, int, int]],
    target: memoryview,
) -> None:
    """Read unauthenticated byte ranges from macro objects into ``target``.

    ``target`` must be a writable contiguous byte view whose length equals the
    total of the piece lengths. On Linux each piece is read with ``preadv``
    straight into the view, so a caller can fill a mapped placement arena
    without an intermediate buffer. The caller must authenticate the bytes in
    place, against a digest bound by already-authenticated metadata, before
    submitting them anywhere.
    """

    total = sum(length for _path, _size, _offset, length in pieces)
    if target.readonly or not target.contiguous or target.format != "B":
        raise CacheFormatError("page object extent target must be writable bytes")
    if len(target) != total:
        raise CacheFormatError("page object extent target length differs")
    cursor = 0
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    use_preadv = hasattr(os, "preadv")
    for path, object_bytes, offset, length in pieces:
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise CacheFormatError(f"page object is unavailable: {error}") from error
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size != object_bytes
            ):
                raise CacheFormatError("page object length differs")
            completed = 0
            while completed < length:
                if use_preadv:
                    count = os.preadv(
                        descriptor,
                        [target[cursor + completed : cursor + length]],
                        offset + completed,
                    )
                    if count <= 0:
                        raise CacheFormatError("page object read was truncated")
                else:  # pragma: no cover - Windows test fallback
                    if hasattr(os, "pread"):
                        chunk = os.pread(
                            descriptor, length - completed, offset + completed
                        )
                    else:
                        os.lseek(descriptor, offset + completed, os.SEEK_SET)
                        chunk = os.read(descriptor, length - completed)
                    if not chunk:
                        raise CacheFormatError("page object read was truncated")
                    count = len(chunk)
                    target[cursor + completed : cursor + completed + count] = chunk
                completed += count
        except OSError as error:
            raise CacheFormatError(f"page object read failed: {error}") from error
        finally:
            os.close(descriptor)
        cursor += length


# Process-local cache of complete page snapshots this process authenticated.
# Keys are (snapshot SHA-256, encoded length). A value is inserted only after
# this process computed SHA-256 over exactly those bytes and found the digest
# equal to one bound by authenticated metadata (a snapshot manifest's
# ``snapshot_sha256``, or the ``result_snapshot_sha256`` of a stage header
# taken from an authenticated first object), or after it composed the bytes
# and bound their digest into a header it published. The trust argument is
# the one the header cache relies on: the content address fixes the value.
# Values are never mutated after insertion. The budget is process-wide and is
# zero (disabled) until a store sets it.
_hot_page_snapshot_limit_bytes = 0
_hot_page_snapshots: OrderedDict[tuple[str, int], bytes | bytearray] = OrderedDict()
_hot_page_snapshot_bytes = 0
_hot_page_snapshot_lock = threading.Lock()
_hot_page_snapshot_hits = 0
_hot_page_snapshot_misses = 0


def _set_hot_page_snapshot_limit(limit_bytes: int) -> None:
    global _hot_page_snapshot_limit_bytes
    if type(limit_bytes) is not int or limit_bytes < 0:
        raise ValueError("hot page snapshot limit must be a non-negative integer")
    with _hot_page_snapshot_lock:
        _hot_page_snapshot_limit_bytes = limit_bytes
        _evict_hot_page_snapshots_locked()


def _hot_page_snapshots_enabled() -> bool:
    return _hot_page_snapshot_limit_bytes > 0


def _hot_page_snapshot_fits(encoded_bytes: int) -> bool:
    return 0 < encoded_bytes <= _hot_page_snapshot_limit_bytes


def _evict_hot_page_snapshots_locked() -> None:
    global _hot_page_snapshot_bytes
    while _hot_page_snapshots and (
        _hot_page_snapshot_bytes > _hot_page_snapshot_limit_bytes
    ):
        _, evicted = _hot_page_snapshots.popitem(last=False)
        _hot_page_snapshot_bytes -= len(evicted)
        del evicted


def _hot_page_snapshot(snapshot_sha256: str, encoded_bytes: int) -> bytes | bytearray | None:
    """Return this process's authenticated copy of a snapshot, or None."""

    global _hot_page_snapshot_hits, _hot_page_snapshot_misses
    if not _hot_page_snapshots_enabled():
        return None
    key = (snapshot_sha256, int(encoded_bytes))
    with _hot_page_snapshot_lock:
        payload = _hot_page_snapshots.get(key)
        if payload is None:
            _hot_page_snapshot_misses += 1
            return None
        _hot_page_snapshots.move_to_end(key)
        _hot_page_snapshot_hits += 1
        return payload


def _remember_hot_page_snapshot(
    snapshot_sha256: str,
    payload: bytes | bytearray,
) -> bool:
    """Retain a snapshot this process hashed to ``snapshot_sha256``.

    The caller passes ownership: the buffer must not be modified afterwards.
    Returns whether the snapshot was retained.
    """

    global _hot_page_snapshot_bytes
    size = len(payload)
    if not _hot_page_snapshot_fits(size):
        return False
    key = (snapshot_sha256, size)
    with _hot_page_snapshot_lock:
        previous = _hot_page_snapshots.pop(key, None)
        if previous is not None:
            _hot_page_snapshot_bytes -= len(previous)
        _hot_page_snapshots[key] = payload
        _hot_page_snapshot_bytes += size
        _evict_hot_page_snapshots_locked()
    return True


def _clear_hot_page_snapshots() -> None:
    global _hot_page_snapshot_bytes, _hot_page_snapshot_hits, _hot_page_snapshot_misses
    with _hot_page_snapshot_lock:
        _hot_page_snapshots.clear()
        _hot_page_snapshot_bytes = 0
        _hot_page_snapshot_hits = 0
        _hot_page_snapshot_misses = 0


def hot_page_snapshot_stats() -> dict[str, int]:
    with _hot_page_snapshot_lock:
        return {
            "limit_bytes": _hot_page_snapshot_limit_bytes,
            "resident_bytes": _hot_page_snapshot_bytes,
            "entries": len(_hot_page_snapshots),
            "hits": _hot_page_snapshot_hits,
            "misses": _hot_page_snapshot_misses,
        }


# Process-local cache of page-delta headers taken from authenticated bytes.
# Keys are (object SHA-256, object bytes) of a delta's first macro object; the
# value is that object's SPHD1 header prefix. An entry is inserted only after
# this process hashed the object to its key, or wrote the object under that
# key, so the content address fixes the header. A hit therefore proves the
# same header a re-read would. Tail bytes are still read and checked against
# the header's per-layer digests, so later damage to an object on disk is
# detected whenever a damaged byte would be used.
_AUTHENTICATED_DELTA_HEADER_BYTES = 16 * 1024 * 1024
_authenticated_delta_headers: OrderedDict[tuple[str, int], bytes] = OrderedDict()
_authenticated_delta_header_bytes = 0
_authenticated_delta_header_lock = threading.Lock()


def _authenticated_delta_header(object_sha256: str, object_bytes: int) -> bytes | None:
    key = (object_sha256, int(object_bytes))
    with _authenticated_delta_header_lock:
        header = _authenticated_delta_headers.get(key)
        if header is not None:
            _authenticated_delta_headers.move_to_end(key)
        return header


def _remember_authenticated_delta_header(
    object_sha256: str,
    object_bytes: int,
    header: bytes,
) -> None:
    """Record a header prefix of an object authenticated by this process."""

    global _authenticated_delta_header_bytes
    if len(header) > min(int(object_bytes), _AUTHENTICATED_DELTA_HEADER_BYTES):
        return
    key = (object_sha256, int(object_bytes))
    with _authenticated_delta_header_lock:
        previous = _authenticated_delta_headers.pop(key, None)
        if previous is not None:
            _authenticated_delta_header_bytes -= len(previous)
        _authenticated_delta_headers[key] = bytes(header)
        _authenticated_delta_header_bytes += len(header)
        while _authenticated_delta_header_bytes > _AUTHENTICATED_DELTA_HEADER_BYTES:
            _, evicted = _authenticated_delta_headers.popitem(last=False)
            _authenticated_delta_header_bytes -= len(evicted)


def _clear_authenticated_delta_headers() -> None:
    global _authenticated_delta_header_bytes
    with _authenticated_delta_header_lock:
        _authenticated_delta_headers.clear()
        _authenticated_delta_header_bytes = 0


_MAX_DELTA_HEADER_PREFIX_BYTES = 16 * 1024 * 1024


def _read_header_prefix(path: Path, header_sha256: str) -> bytes:
    """Read a stage header prefix and check it against its manifest-bound digest."""

    from sparkcache.spark_context_cache_hybrid import page_delta_header_prefix

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CacheFormatError(f"page object is unavailable: {error}") from error
    try:
        prefix = os.read(descriptor, 10)
        if len(prefix) < 10:
            raise CacheFormatError("page delta header prefix is truncated")
        (header_length,) = struct.unpack_from("<I", prefix, 6)
        if header_length <= 0 or header_length > _MAX_DELTA_HEADER_PREFIX_BYTES:
            raise CacheFormatError("page delta header length is invalid")
        os.lseek(descriptor, 0, os.SEEK_SET)
        raw = os.read(descriptor, 10 + header_length)
    finally:
        os.close(descriptor)
    header = page_delta_header_prefix(raw)
    if _sha256(header) != header_sha256:
        raise CacheFormatError("page delta header digest mismatch")
    return header


def _authenticated_page_delta_header(
    object_root: Path,
    descriptor: Mapping[str, Any],
    header_sha256: str | None = None,
) -> bytes:
    """Return a delta's header from the cache or its authenticated first object.

    With ``header_sha256`` (the v3 schemas) only the header prefix is read and
    checked against that manifest-bound digest; otherwise the whole first
    object is read and must hash to its content address.
    """

    from sparkcache.spark_context_cache_hybrid import page_delta_header_prefix

    cached = _authenticated_delta_header(descriptor["sha256"], descriptor["bytes"])
    if cached is not None:
        return cached
    if header_sha256 is not None:
        header = _read_header_prefix(
            object_root / f"{descriptor['sha256']}.spcc", header_sha256
        )
    else:
        (header_object,) = _read_page_delta_object_batch(object_root, (descriptor,))
        header = page_delta_header_prefix(header_object)
    _remember_authenticated_delta_header(
        descriptor["sha256"], descriptor["bytes"], header
    )
    return header


class ManifestTransaction:
    """Incrementally publish chunks, then expose them with one final manifest.

    Appended chunks are durable, immutable content-addressed objects. The
    transaction retains only their small descriptors, never the chunk payloads.
    Until ``commit_manifest`` publishes the manifest, lookup cannot observe the
    transaction. Aborting (or crashing) may leave unreferenced chunks, which
    are harmless and can be reclaimed by a later orphan collector.
    """

    def __init__(
        self,
        *,
        store: "ManifestStore",
        identity: CacheIdentity,
        context_digest: str,
        span_tokens: int | None = None,
    ) -> None:
        _validate_digest(context_digest, "context_digest")
        if span_tokens is not None and span_tokens <= 0:
            raise ValueError("span_tokens must be positive")
        self._store = store
        self._identity = identity
        self._context_digest = context_digest
        self._span_tokens = span_tokens
        self._descriptors: list[dict[str, Any]] = []
        self._expected_start = 0
        self._state = "open"
        self._commit_failed = False
        self._receipt: CommitReceipt | None = None
        self._publication = self._store.publication_telemetry.begin(
            "complete_snapshot"
        )
        self._lock = threading.RLock()
        self._root_guard: _RootGuard | None = _RootGuard(
            self._store.root,
            shared=True,
            blocking=True,
        )
        self._root_guard.__enter__()

    def _release_root_guard(self) -> None:
        guard = self._root_guard
        if guard is None:
            return
        self._root_guard = None
        guard.__exit__(None, None, None)

    def __del__(self) -> None:
        try:
            self._release_root_guard()
        except Exception:
            pass

    def _require_open(self) -> None:
        if self._state != "open":
            raise RuntimeError(f"context transaction is {self._state}")

    def append_chunk(self, chunk: ContextChunk) -> ChunkReceipt:
        """Durably append one chunk without retaining its payload in memory."""

        return self.append_chunks((chunk,))[0]

    def append_chunks(
        self,
        chunks: Sequence[ContextChunk],
    ) -> tuple[ChunkReceipt, ...]:
        """Durably append one contiguous macro-batch with one metadata barrier."""

        with self._lock:
            if self._state == "aborted":
                self._require_open()
            if not chunks:
                raise ValueError("at least one context chunk is required")

            descriptors_by_range = {
                (descriptor["logical_start"], descriptor["logical_end"]): descriptor
                for descriptor in self._descriptors
            }
            pending_descriptors: list[dict[str, Any]] = []
            pending_objects: list[tuple[Path, bytes]] = []
            receipts: list[ChunkReceipt] = []
            expected_start = self._expected_start
            previous = self._descriptors[-1] if self._descriptors else None

            for chunk in chunks:
                token_count = chunk.logical_end - chunk.logical_start
                if token_count > self._identity.chunk_tokens:
                    raise ValueError("chunk exceeds identity chunk_tokens")
                if (
                    self._span_tokens is not None
                    and chunk.logical_end > self._span_tokens
                ):
                    raise ValueError("chunk exceeds the declared context span")
                _require_complete_chunk(chunk, self._identity.required_records)

                encoded = _encode_chunk(chunk)
                chunk_digest = _sha256(encoded)
                receipt = ChunkReceipt(
                    chunk_digest=chunk_digest,
                    encoded_bytes=len(encoded),
                    logical_start=chunk.logical_start,
                    logical_end=chunk.logical_end,
                )
                receipts.append(receipt)
                logical_range = (chunk.logical_start, chunk.logical_end)
                existing = descriptors_by_range.get(logical_range)
                if existing is not None:
                    if existing["sha256"] != chunk_digest or existing["bytes"] != len(
                        encoded
                    ):
                        raise CommitConflict(
                            "different immutable chunk already appended for "
                            f"logical range [{chunk.logical_start},"
                            f"{chunk.logical_end})"
                        )
                    continue

                self._require_open()
                if chunk.logical_start != expected_start:
                    raise ValueError(
                        "chunk logical ranges must be contiguous from zero"
                    )
                if previous is not None:
                    previous_tokens = (
                        previous["logical_end"] - previous["logical_start"]
                    )
                    if previous_tokens != self._identity.chunk_tokens:
                        raise ValueError("only the final context chunk may be partial")

                descriptor = {
                    "sha256": chunk_digest,
                    "bytes": len(encoded),
                    "logical_start": chunk.logical_start,
                    "logical_end": chunk.logical_end,
                }
                descriptors_by_range[logical_range] = descriptor
                pending_descriptors.append(descriptor)
                pending_objects.append(
                    (
                        self._store.root / "chunks" / f"{chunk_digest}.spcc",
                        encoded,
                    )
                )
                expected_start = chunk.logical_end
                previous = descriptor

            token = _ACTIVE_PUBLICATION.set(self._publication)
            try:
                _publish_immutable_batch(pending_objects)
            finally:
                _ACTIVE_PUBLICATION.reset(token)
            self._descriptors.extend(pending_descriptors)
            self._expected_start = expected_start
            return tuple(receipts)

    def commit_manifest(self) -> CommitReceipt:
        """Publish the visibility point after every referenced chunk is durable."""

        with self._lock:
            if self._state == "committed":
                assert self._receipt is not None
                return self._receipt
            self._require_open()
            if not self._descriptors:
                raise ValueError("at least one context chunk is required")
            if (
                self._span_tokens is not None
                and self._expected_start != self._span_tokens
            ):
                raise IncompleteEntry(
                    "transaction does not cover the declared context span"
                )
            manifest = {
                "format_abi": FORMAT_ABI,
                "identity": self._identity.to_wire(),
                "context_digest": self._context_digest,
                "committed_tokens": self._expected_start,
                "chunks": list(self._descriptors),
            }
            encoded_manifest = _canonical_json(manifest)
            logical_payload_bytes = sum(
                int(item["bytes"]) for item in self._descriptors
            )
            self._publication.describe_payload(logical_payload_bytes)
            try:
                token = _ACTIVE_PUBLICATION.set(self._publication)
                try:
                    _publish_immutable(
                        self._store._manifest_path(
                            self._identity,
                            self._context_digest,
                        ),
                        encoded_manifest,
                    )
                finally:
                    _ACTIVE_PUBLICATION.reset(token)
            except Exception:
                # Manifest publication is retryable because a durability
                # failure can occur after the complete manifest is linked.
                # Defer the terminal telemetry outcome until retry succeeds
                # or the caller abandons the transaction, so reachable chunk
                # bytes are never classified as uncommitted.
                self._commit_failed = True
                raise
            publication = self._publication.finish("committed")
            receipt = CommitReceipt(
                manifest_digest=_sha256(encoded_manifest),
                committed_tokens=self._expected_start,
                encoded_bytes=len(encoded_manifest) + logical_payload_bytes,
                allocated_bytes_upper_bound=sum(
                    (size + 4095) // 4096 * 4096
                    for size in (
                        len(encoded_manifest),
                        *(int(item["bytes"]) for item in self._descriptors),
                    )
                ),
                publication=publication,
            )
            self._receipt = receipt
            self._state = "committed"
            self._release_root_guard()
            return receipt

    def abort(self) -> None:
        """Make the transaction terminal without publishing a manifest."""

        with self._lock:
            if self._state == "committed":
                raise RuntimeError("context transaction is committed")
            if self._state == "aborted":
                return
            self._state = "aborted"
            if self._publication.has_activity:
                self._publication.finish(
                    "failed" if self._commit_failed else "aborted"
                )
            self._descriptors.clear()
            self._release_root_guard()


def _tracked_publication(kind: PublicationKind):
    """Attach observational byte accounting to one store publication method."""

    def decorate(method):
        @wraps(method)
        def wrapped(store: "ManifestStore", *args, **kwargs):
            attempt = store.publication_telemetry.begin(kind)
            token = _ACTIVE_PUBLICATION.set(attempt)
            try:
                result = method(store, *args, **kwargs)
            except Exception:
                if attempt.has_activity:
                    attempt.finish("failed")
                raise
            finally:
                _ACTIVE_PUBLICATION.reset(token)
            # A delta may delegate to a complete snapshot before doing any
            # writes. Its nested publication owns the sole terminal receipt.
            if (not attempt.has_activity
                    and isinstance(result, CommitReceipt)
                    and result.publication is not None):
                return result
            receipt = attempt.finish("committed")
            if isinstance(result, (CommitReceipt, PrefixAliasReceipt)):
                return replace(result, publication=receipt)
            return result

        return wrapped

    return decorate


class ManifestStore:
    """Atomic local-NVMe publisher and verified-or-miss reader."""

    def __init__(
        self,
        root: Path | str,
        *,
        publication_telemetry: PublicationTelemetry | None = None,
        page_delta_stage_limit_count: int = 0,
        hot_page_snapshot_bytes: int | None = None,
    ) -> None:
        if type(page_delta_stage_limit_count) is not int or page_delta_stage_limit_count < 0:
            raise ValueError("page_delta_stage_limit_count must be a non-negative integer")
        self.root = Path(root)
        self.publication_telemetry = publication_telemetry or PublicationTelemetry()
        self._page_delta_stage_limit_count = page_delta_stage_limit_count
        # The hot snapshot budget is process-wide (see _hot_page_snapshots);
        # a store that names one sets it for every store in the process.
        if hot_page_snapshot_bytes is not None:
            _set_hot_page_snapshot_limit(hot_page_snapshot_bytes)

    def publication_telemetry_snapshot(self) -> PublicationTelemetrySnapshot:
        """Return monotonic observational byte counters for this store."""

        return self.publication_telemetry.snapshot()

    def _active_publication(self) -> PublicationAttempt:
        attempt = _ACTIVE_PUBLICATION.get()
        if attempt is None:
            raise RuntimeError("publication telemetry scope is missing")
        return attempt

    def clear_once(
        self,
        token: str,
        *,
        lock_timeout_seconds: float = 30.0,
    ) -> bool:
        """Clear owned cache data once for one durable operator token.

        Returns ``True`` when this call removed the configured root's owned
        data and published a completion marker. Returns ``False`` when a valid
        marker already proves that the same token completed. Lock contention
        raises :class:`BlockingIOError` after the bounded timeout so a caller
        can continue serving without using the optional persistent cache.
        """

        root, token_digest = validate_clear_once_request(self.root, token)
        guard = _RootGuard(
            root,
            shared=False,
            blocking=True,
            timeout_seconds=lock_timeout_seconds,
        )
        guard.__enter__()
        try:
            # Revalidate after acquiring the root-level lock. This catches an
            # operator or external process replacing the configured root while
            # this process waited.
            locked_root, locked_digest = validate_clear_once_request(root, token)
            if locked_root != root or locked_digest != token_digest:
                raise OSError("cache root changed while waiting for clear")
            marker_directory = root / _CLEAR_ONCE_MARKER_DIRECTORY
            if marker_directory.is_symlink():
                raise OSError("clear-once marker directory is a symlink")
            if marker_directory.exists() and not marker_directory.is_dir():
                raise OSError("clear-once marker path is not a directory")
            if marker_directory.exists() and marker_directory.is_mount():
                raise OSError("clear-once marker directory is a separate mount")
            marker_path = marker_directory / f"{token_digest}.json"
            if marker_path.is_symlink():
                raise OSError("clear-once completion marker is a symlink")
            if _clear_once_completed(marker_path, token_digest):
                return False

            for name in _CACHE_DATA_DIRECTORIES:
                _remove_tree_without_following_links(root / name)

            _ensure_durable_directory(marker_directory)
            _publish_clear_once_completion(marker_path, token_digest)
            _fsync_directory(root)
            return True
        finally:
            guard.__exit__(None, None, None)

    def _capacity_entry(self, path: Path) -> _CapacityEntry:
        metadata = path.stat()
        key = EntryKey(path.parent.name, path.stem)
        try:
            _validate_digest(key.storage_key, "storage_key")
            _validate_digest(key.context_digest, "context_digest")
            manifest = json.loads(path.read_bytes())
            segments: tuple[str, ...] = ()
            schema_name = manifest.get("schema") if isinstance(manifest, dict) else None
            if (
                schema_name == _TAIL_MANIFEST_SCHEMA
                or schema_name in _PAGE_DELTA_MANIFEST_SCHEMAS
                or schema_name == _PAGE_SNAPSHOT_MANIFEST_SCHEMA
            ):
                identity_wire = dict(manifest.get("identity", {}))
                if "record_schema" in identity_wire:
                    schema = identity_wire["record_schema"]
                    if not isinstance(schema, list):
                        raise CacheFormatError("manifest record schema is invalid")
                    identity_wire["record_schema"] = tuple(schema)
                try:
                    identity = CacheIdentity(**identity_wire)
                except (TypeError, ValueError) as error:
                    raise CacheFormatError("manifest identity is invalid") from error
                if identity.storage_key != key.storage_key:
                    raise _IncompatibleManifestError("manifest storage key differs")
                if schema_name == _TAIL_MANIFEST_SCHEMA:
                    resolved, segments = self._resolve_tail_manifest(
                        manifest,
                        identity=identity,
                        context_digest=key.context_digest,
                    )
                    descriptors = _validate_manifest_metadata(
                        resolved,
                        key,
                        expected_identity=identity,
                    )
                elif schema_name in _PAGE_DELTA_MANIFEST_SCHEMAS:
                    descriptors = self._page_graph_descriptors(
                        manifest,
                        identity=identity,
                        context_digest=key.context_digest,
                    )
                else:
                    descriptors = _validate_page_snapshot_root(
                        manifest,
                        identity=identity,
                        context_digest=key.context_digest,
                    )
            else:
                descriptors = _validate_manifest_metadata(manifest, key)
            chunks: dict[str, int] = {}
            for descriptor in descriptors:
                digest = descriptor["sha256"]
                encoded_bytes = descriptor["bytes"]
                if digest in chunks and chunks[digest] != encoded_bytes:
                    raise CacheFormatError("capacity chunk descriptor differs")
                chunks[digest] = encoded_bytes
            return _CapacityEntry(
                key=key,
                path=path,
                manifest_bytes=_allocated_bytes(metadata),
                mtime_ns=metadata.st_mtime_ns,
                chunks=tuple(sorted(chunks.items())),
                valid=True,
                segments=segments,
            )
        except (CacheFormatError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return _CapacityEntry(
                key=key,
                path=path,
                manifest_bytes=_allocated_bytes(metadata),
                mtime_ns=metadata.st_mtime_ns,
                chunks=(),
                valid=False,
            )

    def _capacity_alias_entry(
        self,
        path: Path,
        segment_cache: dict[
            tuple[str, str, int],
            tuple[tuple[Mapping[str, Any], ...], str | None],
        ] | None = None,
    ) -> _CapacityEntry:
        """Describe one alias root only when its complete graph authenticates.

        Maintenance may remove malformed metadata, but it must never use an
        unverified alias or segment descriptor as authority to retain or delete
        shared content-addressed chunks.
        """

        metadata = path.stat()
        key = EntryKey(path.parent.name, path.stem, "prefix_alias")
        try:
            _validate_digest(key.storage_key, "storage_key")
            _validate_digest(key.context_digest, "context_digest")
            alias = json.loads(path.read_bytes())
            if not isinstance(alias, dict) or not isinstance(
                alias.get("identity"), dict
            ):
                raise CacheFormatError("prefix alias identity is invalid")
            identity_wire = dict(alias["identity"])
            if "record_schema" in identity_wire:
                schema = identity_wire["record_schema"]
                if not isinstance(schema, list):
                    raise CacheFormatError("prefix alias record schema is invalid")
                identity_wire["record_schema"] = tuple(schema)
            try:
                identity = CacheIdentity(**identity_wire)
            except (TypeError, ValueError) as error:
                raise CacheFormatError("prefix alias identity is invalid") from error
            if identity.storage_key != key.storage_key:
                raise _IncompatibleManifestError("prefix alias storage key differs")
            committed_tokens, chunk_count, segment_digest = _validate_prefix_alias(
                alias,
                identity=identity,
                context_digest=key.context_digest,
            )
            segment_count = (
                chunk_count + _PREFIX_SEGMENT_DESCRIPTORS - 1
            ) // _PREFIX_SEGMENT_DESCRIPTORS
            visited: set[str] = set()
            reversed_segments: list[tuple[Mapping[str, Any], ...]] = []
            for segment_index in range(segment_count - 1, -1, -1):
                if segment_digest in visited:
                    raise CacheFormatError("prefix descriptor chain contains a cycle")
                visited.add(segment_digest)
                segment_path = (
                    self.root
                    / "prefix-index"
                    / key.storage_key
                    / f"{segment_digest}.spix"
                )
                cache_key = (key.storage_key, segment_digest, segment_index)
                cached = (
                    segment_cache.get(cache_key) if segment_cache is not None else None
                )
                if cached is None:
                    encoded_segment = segment_path.read_bytes()
                    if _sha256(encoded_segment) != segment_digest:
                        raise CacheFormatError(
                            "prefix descriptor segment checksum mismatch"
                        )
                    segment = json.loads(encoded_segment)
                    descriptors = _validate_prefix_segment(
                        segment,
                        storage_key=key.storage_key,
                        expected_first_chunk=(
                            segment_index * _PREFIX_SEGMENT_DESCRIPTORS
                        ),
                    )
                    parent = segment["parent_sha256"]
                    if segment_cache is not None:
                        segment_cache[cache_key] = (descriptors, parent)
                else:
                    descriptors, parent = cached
                if segment_index < segment_count - 1 and (
                    len(descriptors) != _PREFIX_SEGMENT_DESCRIPTORS
                ):
                    raise CacheFormatError("non-tail prefix segment is incomplete")
                reversed_segments.append(descriptors)
                if segment_index == 0:
                    if parent is not None:
                        raise CacheFormatError(
                            "prefix descriptor chain has extra parent"
                        )
                elif parent is None:
                    raise CacheFormatError("prefix descriptor chain is truncated")
                else:
                    segment_digest = parent

            descriptors = [
                descriptor
                for segment in reversed(reversed_segments)
                for descriptor in segment
            ][:chunk_count]
            synthetic_manifest = {
                "format_abi": FORMAT_ABI,
                "identity": identity.to_wire(),
                "context_digest": key.context_digest,
                "committed_tokens": committed_tokens,
                "chunks": descriptors,
            }
            validated = _validate_manifest_metadata(
                synthetic_manifest,
                EntryKey(key.storage_key, key.context_digest),
                expected_identity=identity,
            )
            chunks: dict[str, int] = {}
            for descriptor in validated:
                digest = descriptor["sha256"]
                encoded_bytes = descriptor["bytes"]
                if digest in chunks and chunks[digest] != encoded_bytes:
                    raise CacheFormatError("capacity chunk descriptor differs")
                chunks[digest] = encoded_bytes
            return _CapacityEntry(
                key=key,
                path=path,
                manifest_bytes=_allocated_bytes(metadata),
                mtime_ns=metadata.st_mtime_ns,
                chunks=tuple(sorted(chunks.items())),
                valid=True,
                segments=tuple(sorted(visited)),
            )
        except (
            CacheFormatError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return _CapacityEntry(
                key=key,
                path=path,
                manifest_bytes=_allocated_bytes(metadata),
                mtime_ns=metadata.st_mtime_ns,
                chunks=(),
                valid=False,
            )

    def touch(
        self,
        identity: CacheIdentity,
        context_digest: str,
        *,
        now_ns: int | None = None,
        minimum_interval_seconds: int = 60,
    ) -> bool:
        """Best-effort, rate-limited manifest-mtime refresh after verified use."""

        if minimum_interval_seconds < 0:
            raise ValueError("minimum_interval_seconds must be non-negative")
        current_ns = time.time_ns() if now_ns is None else now_ns
        path = self._manifest_path(identity, context_digest)
        if not path.exists():
            path = self._prefix_alias_path(identity, context_digest)
        try:
            with _RootGuard(self.root, shared=True, blocking=False):
                metadata = path.stat()
                if current_ns - metadata.st_mtime_ns < minimum_interval_seconds * 10**9:
                    return False
                os.utime(path, ns=(metadata.st_atime_ns, current_ns))
                return True
        except (BlockingIOError, OSError):
            return False

    def expired(
        self,
        identity: CacheIdentity,
        context_digest: str,
        ttl_seconds: int,
        *,
        now_ns: int | None = None,
    ) -> bool:
        if ttl_seconds <= 0:
            return False
        current_ns = time.time_ns() if now_ns is None else now_ns
        try:
            path = self._manifest_path(identity, context_digest)
            if not path.exists():
                path = self._prefix_alias_path(identity, context_digest)
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            return True
        return current_ns - mtime_ns >= ttl_seconds * 10**9

    def maintain(
        self,
        policy: CapacityPolicy,
        *,
        now_ns: int | None = None,
        protected_entries: Sequence[EntryKey] = (),
    ) -> MaintenanceReport:
        """Apply metadata-only orphan, TTL, and LRU maintenance.

        The exclusive lock is nonblocking. A live transaction therefore makes
        maintenance skip instead of delaying a store or serving callback.
        Protected valid roots retain their complete object graph for an
        admitted publication. They remain counted against capacity; a budget
        that cannot be met reports unsatisfied instead of evicting those roots.
        """

        if not policy.enabled:
            return MaintenanceReport()
        current_ns = time.time_ns() if now_ns is None else now_ns
        protected = frozenset(protected_entries)
        try:
            guard = _RootGuard(self.root, shared=False, blocking=False)
            guard.__enter__()
        except BlockingIOError:
            return MaintenanceReport(capacity_satisfied=False, skipped_busy=True)
        if time.monotonic() < getattr(self, "_maintenance_not_before", 0.0):
            guard.__exit__(None, None, None)
            return MaintenanceReport(capacity_satisfied=False, skipped_cooldown=True)
        try:
            deletion_attempts = 0
            work_pending = False

            def admit_deletion() -> bool:
                nonlocal deletion_attempts, work_pending
                if (policy.maintenance_max_deletions > 0
                        and deletion_attempts >= policy.maintenance_max_deletions):
                    work_pending = True
                    return False
                deletion_attempts += 1
                return True

            manifests_root = self.root / "manifests"
            aliases_root = self.root / "prefix-aliases"
            manifest_paths = (
                tuple(sorted(manifests_root.glob("*/*.json"), key=os.fspath))
                if manifests_root.is_dir()
                else ()
            )
            alias_paths = (
                tuple(sorted(aliases_root.glob("*/*.json"), key=os.fspath))
                if aliases_root.is_dir()
                else ()
            )
            entries = [self._capacity_entry(path) for path in manifest_paths]
            # Shared descriptors are immutable while the exclusive root guard
            # is held. Retain authentication only for this maintenance pass,
            # keyed by namespace, digest, and position in the descriptor chain.
            segment_cache: dict[
                tuple[str, str, int],
                tuple[tuple[Mapping[str, Any], ...], str | None],
            ] = {}
            entries.extend(
                self._capacity_alias_entry(path, segment_cache) for path in alias_paths
            )

            def root_files(root: Path) -> tuple[Path, ...]:
                if not root.is_dir():
                    return ()
                return tuple(
                    sorted(
                        (path
                        for directory in root.iterdir()
                        if directory.is_dir()
                        for path in directory.iterdir()
                        if path.is_file()),
                        key=os.fspath,
                    )
                )

            root_path_set = set((*manifest_paths, *alias_paths))
            root_debris = tuple(
                path
                for path in (*root_files(manifests_root), *root_files(aliases_root))
                if path not in root_path_set
            )
            root_debris_sizes: dict[Path, int] = {}
            for path in root_debris:
                try:
                    root_debris_sizes[path] = _allocated_bytes(path.stat())
                except FileNotFoundError:
                    continue

            segment_root = self.root / "prefix-index"
            segment_paths = root_files(segment_root)
            segment_sizes: dict[Path, int] = {}
            canonical_segments: dict[tuple[str, str], Path] = {}
            for path in segment_paths:
                try:
                    segment_sizes[path] = _allocated_bytes(path.stat())
                except FileNotFoundError:
                    continue
                if (
                    path.suffix == ".spix"
                    and _DIGEST.fullmatch(path.stem)
                    and _DIGEST.fullmatch(path.parent.name)
                ):
                    canonical_segments[(path.parent.name, path.stem)] = path
            chunk_directory = self.root / "chunks"
            chunk_paths = (
                tuple(
                    sorted((path for path in chunk_directory.iterdir() if path.is_file()),
                           key=os.fspath)
                )
                if chunk_directory.is_dir()
                else ()
            )
            chunk_sizes: dict[Path, int] = {}
            chunk_logical_sizes: dict[str, int] = {}
            canonical_chunks: dict[str, Path] = {}
            for path in chunk_paths:
                try:
                    metadata = path.stat()
                    chunk_sizes[path] = _allocated_bytes(metadata)
                except FileNotFoundError:
                    continue
                if path.suffix == ".spcc" and _DIGEST.fullmatch(path.stem):
                    canonical_chunks[path.stem] = path
                    chunk_logical_sizes[path.stem] = metadata.st_size

            references: Counter[str] = Counter()
            segment_references: Counter[tuple[str, str]] = Counter()
            for entry in entries:
                references.update(digest for digest, _size in entry.chunks)
                segment_references.update(
                    (entry.key.storage_key, digest) for digest in entry.segments
                )
            bytes_before = (
                sum(entry.manifest_bytes for entry in entries)
                + sum(root_debris_sizes.values())
                + sum(segment_sizes.values())
                + sum(chunk_sizes.values())
            )
            pressure_target = (policy.max_bytes, policy.low_watermark_bytes)
            retained_target = getattr(self, "_maintenance_pressure_target", None)
            pressure_triggered = policy.max_bytes > 0 and (
                bytes_before > policy.max_bytes
                or (policy.maintenance_max_deletions > 0
                    and retained_target == pressure_target
                    and bytes_before > policy.low_watermark_bytes)
            )
            # Preserve high-to-low hysteresis across bounded passes. Falling
            # below the high watermark alone does not complete reclamation.
            self._maintenance_pressure_target = (
                pressure_target if pressure_triggered and policy.maintenance_max_deletions > 0 else None
            )
            projected_bytes = (
                bytes_before
                - sum(root_debris_sizes.values())
                - sum(
                    size
                    for path, size in segment_sizes.items()
                    if canonical_segments.get((path.parent.name, path.stem)) != path
                    or segment_references.get((path.parent.name, path.stem), 0) == 0
                )
                - sum(
                    size
                    for path, size in chunk_sizes.items()
                    if canonical_chunks.get(path.stem) != path
                    or references.get(path.stem, 0) == 0
                )
            )
            # With a deletion budget, reclaim already-unreferenced payloads
            # before choosing further live roots. Otherwise a small budget
            # could repeatedly remove roots while their orphan payloads wait.
            orphan_debt = bool(root_debris_sizes) or any(
                canonical_segments.get((path.parent.name, path.stem)) != path
                or segment_references.get((path.parent.name, path.stem), 0) == 0
                for path in segment_sizes
            ) or any(
                canonical_chunks.get(path.stem) != path
                or references.get(path.stem, 0) == 0
                for path in chunk_sizes
            )
            projected_references = references.copy()
            projected_segment_references = segment_references.copy()
            selected: list[_CapacityEntry] = []
            selected_paths: set[Path] = set()

            def select(entry: _CapacityEntry) -> None:
                nonlocal projected_bytes, work_pending
                if entry.path in selected_paths or (
                    entry.valid and entry.key in protected
                ):
                    return
                if policy.maintenance_max_deletions > 0 and orphan_debt:
                    work_pending = True
                    return
                if (policy.maintenance_max_deletions > 0
                        and len(selected) >= policy.maintenance_max_deletions):
                    work_pending = True
                    return
                selected.append(entry)
                selected_paths.add(entry.path)
                projected_bytes -= entry.manifest_bytes
                for digest, _declared_size in entry.chunks:
                    projected_references[digest] -= 1
                    if projected_references[digest] == 0:
                        path = canonical_chunks.get(digest)
                        if path is not None:
                            projected_bytes -= chunk_sizes.get(path, 0)
                for digest in entry.segments:
                    reference = (entry.key.storage_key, digest)
                    projected_segment_references[reference] -= 1
                    if projected_segment_references[reference] == 0:
                        path = canonical_segments.get(reference)
                        if path is not None:
                            projected_bytes -= segment_sizes.get(path, 0)

            # Cache namespaces and digests have fixed-width canonical names.
            # Compare each path's string once instead of normalizing Path
            # components throughout every comparison in the inventory sort.
            ordered = sorted(entries, key=lambda entry: (entry.mtime_ns, os.fspath(entry.path)))
            for entry in ordered:
                expired = policy.ttl_seconds > 0 and (
                    current_ns - entry.mtime_ns >= policy.ttl_seconds * 10**9
                )
                if not entry.valid or expired:
                    select(entry)
            if pressure_triggered and projected_bytes > policy.low_watermark_bytes:
                for entry in ordered:
                    select(entry)
                    if projected_bytes <= policy.low_watermark_bytes:
                        break

            removed: list[_CapacityEntry] = []
            affected_root_directories: set[Path] = set()
            root_debris_deleted = 0
            for path, size in root_debris_sizes.items():
                if not admit_deletion():
                    continue
                try:
                    path.unlink()
                except FileNotFoundError:
                    affected_root_directories.add(path.parent)
                    root_debris_deleted += size
                except OSError:
                    continue
                else:
                    affected_root_directories.add(path.parent)
                    root_debris_deleted += size
            for entry in selected:
                if not admit_deletion():
                    continue
                try:
                    entry.path.unlink()
                except FileNotFoundError:
                    removed.append(entry)
                    affected_root_directories.add(entry.path.parent)
                except OSError:
                    continue
                else:
                    removed.append(entry)
                    affected_root_directories.add(entry.path.parent)
            # Root removals are durable before any object they authorized can
            # be collected. A failed root-directory barrier stops maintenance
            # with every shared segment and chunk still present.
            for directory in sorted(affected_root_directories, key=os.fspath):
                _fsync_directory(directory)

            remaining_references: Counter[str] = Counter()
            remaining_segment_references: Counter[tuple[str, str]] = Counter()
            removed_paths = {entry.path for entry in removed}
            for entry in entries:
                if entry.path not in removed_paths:
                    remaining_references.update(
                        digest for digest, _size in entry.chunks
                    )
                    remaining_segment_references.update(
                        (entry.key.storage_key, digest) for digest in entry.segments
                    )

            segments_deleted = 0
            orphan_segments_deleted = 0
            segment_bytes_deleted = 0
            initial_orphan_segments = {
                path
                for path in segment_sizes
                if canonical_segments.get((path.parent.name, path.stem)) != path
                or segment_references.get((path.parent.name, path.stem), 0) == 0
            }
            affected_segment_directories: set[Path] = set()
            for path, size in segment_sizes.items():
                reference = (path.parent.name, path.stem)
                if (
                    canonical_segments.get(reference) == path
                    and remaining_segment_references.get(reference, 0) > 0
                ):
                    continue
                if not admit_deletion():
                    continue
                try:
                    path.unlink()
                except OSError:
                    continue
                segments_deleted += 1
                segment_bytes_deleted += size
                affected_segment_directories.add(path.parent)
                if path in initial_orphan_segments:
                    orphan_segments_deleted += 1
            for directory in sorted(affected_segment_directories, key=os.fspath):
                _fsync_directory(directory)

            chunks_deleted = 0
            orphan_chunks_deleted = 0
            chunk_bytes_deleted = 0
            initial_orphans = {
                path
                for path in chunk_sizes
                if canonical_chunks.get(path.stem) != path
                or references.get(path.stem, 0) == 0
            }
            for path, size in chunk_sizes.items():
                if (
                    canonical_chunks.get(path.stem) == path
                    and remaining_references.get(path.stem, 0) > 0
                ):
                    continue
                if not admit_deletion():
                    continue
                try:
                    path.unlink()
                except OSError:
                    continue
                chunks_deleted += 1
                chunk_bytes_deleted += size
                if path in initial_orphans:
                    orphan_chunks_deleted += 1
            if chunks_deleted:
                _fsync_directory(chunk_directory)

            manifest_bytes_deleted = sum(entry.manifest_bytes for entry in removed)
            bytes_after = max(
                0,
                bytes_before
                - root_debris_deleted
                - manifest_bytes_deleted
                - segment_bytes_deleted
                - chunk_bytes_deleted,
            )
            if (pressure_triggered and policy.maintenance_max_deletions > 0
                    and bytes_after > policy.low_watermark_bytes):
                work_pending = True
            else:
                self._maintenance_pressure_target = None
            exact_removed = sum(entry.key.root_kind == "manifest" for entry in removed)
            aliases_removed = sum(
                entry.key.root_kind == "prefix_alias" for entry in removed
            )
            remaining_entries = [entry for entry in entries if entry.path not in removed_paths]
            # Exact roots shadow aliases even when corrupt or when an unlink
            # failed. Match lookup's exact-first policy without re-reading roots.
            exact_keys = {
                (entry.key.storage_key, entry.key.context_digest)
                for entry in remaining_entries if entry.key.root_kind == "manifest"
            }
            surviving_entries = tuple(
                entry.key for entry in remaining_entries
                if entry.valid
                and (entry.key.root_kind == "manifest"
                     or (entry.key.storage_key, entry.key.context_digest) not in exact_keys)
                and all(chunk_logical_sizes.get(digest) == size
                        for digest, size in entry.chunks)
            )
            return MaintenanceReport(
                bytes_before=bytes_before,
                bytes_after=bytes_after,
                deletion_attempts=deletion_attempts,
                work_pending=work_pending,
                bytes_reclaimed=bytes_before - bytes_after,
                manifests_evicted=exact_removed,
                chunks_deleted=chunks_deleted,
                orphan_chunks_deleted=orphan_chunks_deleted,
                evicted_entries=tuple(entry.key for entry in removed),
                capacity_satisfied=(
                    policy.max_bytes == 0 or bytes_after <= policy.max_bytes
                ),
                aliases_evicted=aliases_removed,
                segments_deleted=segments_deleted,
                orphan_segments_deleted=orphan_segments_deleted,
                surviving_entries=surviving_entries,
            )
        finally:
            self._maintenance_not_before = (
                time.monotonic() + policy.maintenance_interval_ms / 1000.0
            )
            guard.__exit__(None, None, None)

    def _manifest_path(self, identity: CacheIdentity, context_digest: str) -> Path:
        return self.root / "manifests" / identity.storage_key / f"{context_digest}.json"

    def _prefix_segment_path(
        self, identity: CacheIdentity, segment_digest: str
    ) -> Path:
        return (
            self.root / "prefix-index" / identity.storage_key / f"{segment_digest}.spix"
        )

    def _prefix_alias_path(self, identity: CacheIdentity, context_digest: str) -> Path:
        return (
            self.root
            / "prefix-aliases"
            / identity.storage_key
            / f"{context_digest}.json"
        )

    def _read_descriptor_chain(
        self,
        identity: CacheIdentity,
        segment_digest: str | None,
        chunk_count: int,
    ) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
        if chunk_count == 0:
            if segment_digest is not None:
                raise CacheFormatError("descriptor chain has an unexpected root")
            return (), ()
        if segment_digest is None:
            raise CacheFormatError("descriptor chain root is missing")
        segment_count = (
            chunk_count + _PREFIX_SEGMENT_DESCRIPTORS - 1
        ) // _PREFIX_SEGMENT_DESCRIPTORS
        visited: set[str] = set()
        reversed_segments: list[tuple[Mapping[str, Any], ...]] = []
        for segment_index in range(segment_count - 1, -1, -1):
            if segment_digest in visited:
                raise CacheFormatError("descriptor chain contains a cycle")
            visited.add(segment_digest)
            encoded = self._prefix_segment_path(identity, segment_digest).read_bytes()
            if _sha256(encoded) != segment_digest:
                raise CacheFormatError("descriptor segment checksum mismatch")
            segment = json.loads(encoded)
            descriptors = _validate_prefix_segment(
                segment,
                storage_key=identity.storage_key,
                expected_first_chunk=(segment_index * _PREFIX_SEGMENT_DESCRIPTORS),
            )
            if segment_index < segment_count - 1 and (
                len(descriptors) != _PREFIX_SEGMENT_DESCRIPTORS
            ):
                raise CacheFormatError("non-tail descriptor segment is incomplete")
            reversed_segments.append(descriptors)
            parent = segment["parent_sha256"]
            if segment_index == 0:
                if parent is not None:
                    raise CacheFormatError("descriptor chain has an extra parent")
            elif parent is None:
                raise CacheFormatError("descriptor chain is truncated")
            else:
                segment_digest = parent
        descriptors = tuple(
            descriptor
            for segment in reversed(reversed_segments)
            for descriptor in segment
        )[:chunk_count]
        if len(descriptors) != chunk_count:
            raise CacheFormatError("descriptor chain length differs")
        return descriptors, tuple(sorted(visited))

    def _publish_descriptor_chain(
        self,
        identity: CacheIdentity,
        descriptors: Sequence[Mapping[str, Any]],
    ) -> tuple[str | None, int, int]:
        if not descriptors:
            return None, 0, 0
        objects: list[tuple[Path, bytes]] = []
        parent_digest: str | None = None
        for first in range(0, len(descriptors), _PREFIX_SEGMENT_DESCRIPTORS):
            segment = {
                "schema": _PREFIX_SEGMENT_SCHEMA,
                "storage_key": identity.storage_key,
                "parent_sha256": parent_digest,
                "first_chunk_index": first,
                "descriptors": list(
                    descriptors[first : first + _PREFIX_SEGMENT_DESCRIPTORS]
                ),
            }
            encoded = _canonical_json(segment)
            parent_digest = _sha256(encoded)
            objects.append(
                (self._prefix_segment_path(identity, parent_digest), encoded)
            )
        _publish_immutable_batch(objects)
        allocated_bytes = sum(
            (len(encoded) + 4095) // 4096 * 4096 for _path, encoded in objects
        )
        return parent_digest, len(objects), allocated_bytes

    def _resolve_tail_manifest(
        self,
        manifest: Mapping[str, Any],
        *,
        identity: CacheIdentity,
        context_digest: str,
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        (
            committed_tokens,
            _reused_tokens,
            chunk_count,
            segment_digest,
            tail_chunks,
        ) = _validate_tail_manifest_root(
            manifest,
            identity=identity,
            context_digest=context_digest,
        )
        base_chunks, segments = self._read_descriptor_chain(
            identity,
            segment_digest,
            chunk_count,
        )
        synthetic = {
            "format_abi": FORMAT_ABI,
            "identity": identity.to_wire(),
            "context_digest": context_digest,
            "committed_tokens": committed_tokens,
            "chunks": [*base_chunks, *tail_chunks],
        }
        _validate_manifest_metadata(
            synthetic,
            EntryKey(identity.storage_key, context_digest),
            expected_identity=identity,
        )
        return synthetic, segments

    def _page_graph_descriptors(
        self,
        manifest: Mapping[str, Any],
        *,
        identity: CacheIdentity,
        context_digest: str,
        depth: int = 0,
    ) -> tuple[Mapping[str, Any], ...]:
        if depth >= _MAX_PAGE_DELTA_DEPTH:
            raise CacheFormatError("page delta graph exceeds the depth limit")
        base_root, delta_chunks = _validate_page_delta_root(
            manifest,
            identity=identity,
            context_digest=context_digest,
        )
        base_digest = manifest["base_context_digest"]
        if _is_page_delta_root(base_root):
            base_chunks = self._page_graph_descriptors(
                base_root,
                identity=identity,
                context_digest=base_digest,
                depth=depth + 1,
            )
        elif _is_page_snapshot_root(base_root):
            base_chunks = _validate_page_snapshot_root(
                base_root,
                identity=identity,
                context_digest=base_digest,
            )
        else:
            base_chunks = _validate_manifest_metadata(
                base_root,
                EntryKey(identity.storage_key, base_digest),
                expected_identity=identity,
            )
        if base_root.get("committed_tokens") != manifest["base_committed_tokens"]:
            raise CacheFormatError("page delta base boundary differs")
        return (*base_chunks, *delta_chunks)

    @staticmethod
    def _page_delta_root_count(manifest: Mapping[str, Any]) -> int:
        count = 0
        root: Any = manifest
        while _is_page_delta_root(root):
            count += 1
            root = root.get("base_root")
        return count

    def _read_context_chunks(
        self,
        descriptors: Sequence[Mapping[str, Any]],
        required: frozenset[StateRecord],
    ) -> tuple[ContextChunk, ...]:
        def _read_one(descriptor: Mapping[str, Any]) -> ContextChunk:
            encoded = (
                self.root / "chunks" / f"{descriptor['sha256']}.spcc"
            ).read_bytes()
            if (
                len(encoded) != descriptor["bytes"]
                or _sha256(encoded) != descriptor["sha256"]
            ):
                raise CacheFormatError("chunk checksum mismatch")
            chunk = _decode_chunk(encoded, verify_record_checksums=False)
            _require_complete_chunk(chunk, required)
            if (
                chunk.logical_start != descriptor["logical_start"]
                or chunk.logical_end != descriptor["logical_end"]
            ):
                raise CacheFormatError("chunk range disagrees with descriptor")
            return chunk

        if not descriptors:
            return ()
        with ThreadPoolExecutor(max_workers=min(8, len(descriptors))) as pool:
            return tuple(pool.map(_read_one, descriptors))

    def _read_page_delta_objects(
        self,
        descriptors: Sequence[Mapping[str, Any]],
        *,
        encoded_bytes: int,
        encoded_sha256: str,
    ) -> bytearray:
        """Read ordered macro objects with bounded transient payload memory."""

        if encoded_bytes <= 0 or not descriptors:
            raise CacheFormatError("page delta object coverage is empty")
        result = bytearray(encoded_bytes)
        digest = hashlib.sha256()
        expected_start = 0
        object_root = self.root / "chunks"
        for first in range(0, len(descriptors), _PAGE_DELTA_READ_BATCH_SIZE):
            batch = tuple(descriptors[first : first + _PAGE_DELTA_READ_BATCH_SIZE])
            payloads = _read_page_delta_object_batch(object_root, batch)
            for descriptor, payload in zip(batch, payloads, strict=True):
                start = int(descriptor["encoded_start"])
                end = int(descriptor["encoded_end"])
                if start != expected_start or end != start + len(payload):
                    raise CacheFormatError("page delta object coverage differs")
                # A memoryview source is copied in place; a bytes source
                # would be re-allocated before the copy.
                with memoryview(payload) as payload_view:
                    result[start:end] = payload_view
                if "crc32" not in descriptor:
                    digest.update(payload)
                expected_start = end
        if expected_start != encoded_bytes or (
            _descriptor_integrity(descriptors[0]) == "sha256"
            and digest.hexdigest() != encoded_sha256
        ):
            raise CacheFormatError("page delta payload checksum mismatch")
        return result

    def _read_page_snapshot_objects(
        self,
        descriptors: Sequence[Mapping[str, Any]],
        *,
        encoded_bytes: int,
        encoded_sha256: str,
    ) -> bytearray:
        """Reassemble flat page extents with bounded concurrent read memory."""

        if encoded_bytes <= 0 or not descriptors:
            raise CacheFormatError("page snapshot object coverage is empty")
        result = bytearray(encoded_bytes)
        digest = hashlib.sha256()
        expected_start = 0
        object_root = self.root / "chunks"
        for first in range(0, len(descriptors), _PAGE_SNAPSHOT_READ_BATCH_SIZE):
            batch = tuple(descriptors[first : first + _PAGE_SNAPSHOT_READ_BATCH_SIZE])
            payloads = _read_page_snapshot_object_batch(object_root, batch)
            for descriptor, payload in zip(batch, payloads, strict=True):
                start = int(descriptor["encoded_start"])
                end = int(descriptor["encoded_end"])
                if start != expected_start or end != start + len(payload):
                    raise CacheFormatError("page snapshot object coverage differs")
                # A memoryview source is copied in place; a bytes source
                # would be re-allocated before the copy.
                with memoryview(payload) as payload_view:
                    result[start:end] = payload_view
                if "crc32" not in descriptor:
                    digest.update(payload)
                expected_start = end
        if expected_start != encoded_bytes or (
            _descriptor_integrity(descriptors[0]) == "sha256"
            and digest.hexdigest() != encoded_sha256
        ):
            raise CacheFormatError("page snapshot payload checksum mismatch")
        return result

    @_tracked_publication("prefix_alias")
    def publish_prefix_aliases(
        self,
        *,
        identity: CacheIdentity,
        source_context_digest: str,
        token_ids: Sequence[int],
        identity_salt: str,
        prefix_tokens: Sequence[int] | None = None,
        storage_mode: str,
    ) -> PrefixAliasReceipt:
        """Publish aliases while preventing concurrent capacity maintenance."""

        with _RootGuard(self.root, shared=True, blocking=True):
            return self._publish_prefix_aliases(
                identity=identity,
                source_context_digest=source_context_digest,
                token_ids=token_ids,
                identity_salt=identity_salt,
                prefix_tokens=prefix_tokens,
                storage_mode=storage_mode,
            )

    def _publish_prefix_aliases(
        self,
        *,
        identity: CacheIdentity,
        source_context_digest: str,
        token_ids: Sequence[int],
        identity_salt: str,
        prefix_tokens: Sequence[int] | None,
        storage_mode: str,
    ) -> PrefixAliasReceipt:
        """Publish sparse, authenticated aliases over one exact row manifest.

        Descriptor segments contain at most sixteen chunk descriptors and form
        one content-addressed parent chain. Alias files select a token boundary
        and a tail segment. They never rewrite chunk payloads or exact manifests.

        Prefix aliases are intentionally restricted to ``per_token_rows``.
        Opaque block-page snapshots can contain recurrent state whose validity
        is tied to the exact snapshot boundary and therefore cannot be safely
        shortened by metadata alone.
        """

        if storage_mode != "per_token_rows":
            raise ValueError("prefix aliases require per_token_rows storage")
        _validate_digest(source_context_digest, "source_context_digest")
        source_path = self._manifest_path(identity, source_context_digest)
        source_encoded = source_path.read_bytes()
        try:
            source_manifest = json.loads(source_encoded)
            if (
                isinstance(source_manifest, dict)
                and source_manifest.get("schema") == _TAIL_MANIFEST_SCHEMA
            ):
                source_manifest, _segments = self._resolve_tail_manifest(
                    source_manifest,
                    identity=identity,
                    context_digest=source_context_digest,
                )
            descriptors = _validate_manifest_metadata(
                source_manifest,
                EntryKey(identity.storage_key, source_context_digest),
                expected_identity=identity,
            )
        except (json.JSONDecodeError, CacheFormatError) as error:
            raise CacheFormatError("source manifest is not publishable") from error
        committed_tokens = int(source_manifest["committed_tokens"])
        if committed_tokens % identity.chunk_tokens:
            raise ValueError("prefix alias source must end on chunk geometry")
        if prefix_tokens is None:
            stride = _PREFIX_SEGMENT_DESCRIPTORS * identity.chunk_tokens
            boundaries = tuple(range(stride, committed_tokens + 1, stride))
            if not boundaries or boundaries[-1] != committed_tokens:
                boundaries = (*boundaries, committed_tokens)
            boundaries = boundaries[-_MAX_PREFIX_ALIASES:]
        else:
            boundaries = tuple(prefix_tokens)
            if not boundaries:
                raise ValueError("prefix_tokens must not be empty")
            if len(boundaries) > _MAX_PREFIX_ALIASES:
                raise ValueError(
                    f"at most {_MAX_PREFIX_ALIASES} prefix aliases may be published"
                )
            if tuple(sorted(set(boundaries))) != boundaries:
                raise ValueError("prefix_tokens must be strictly increasing and unique")

        # Digest every requested boundary and the exact source boundary in one
        # incremental pass. This prevents a caller from pairing descriptors
        # with unrelated tokens or supplying a forged (span, digest) tuple.
        digest_boundaries = tuple(sorted(set((*boundaries, committed_tokens))))
        from sparkcache.spark_context_cache_codec import chunk_prefix_digests

        digests = dict(
            chunk_prefix_digests(
                token_ids,
                identity_salt,
                boundaries=digest_boundaries,
            )
        )
        if digests[committed_tokens] != source_context_digest:
            raise CommitConflict("source digest disagrees with supplied token sequence")
        descriptor_ends = {
            int(descriptor["logical_end"]): index + 1
            for index, descriptor in enumerate(descriptors)
        }
        for boundary in boundaries:
            if boundary not in descriptor_ends:
                raise ValueError("prefix token boundary is not a source chunk boundary")

        maximum_chunks = max(descriptor_ends[boundary] for boundary in boundaries)
        publication = self._active_publication()
        publication.describe_payload(
            0,
            sum(int(item["bytes"]) for item in descriptors[:maximum_chunks]),
        )
        segment_digests: list[str] = []
        segment_objects: list[tuple[Path, bytes]] = []
        parent_digest: str | None = None
        for first in range(0, maximum_chunks, _PREFIX_SEGMENT_DESCRIPTORS):
            segment = {
                "schema": _PREFIX_SEGMENT_SCHEMA,
                "storage_key": identity.storage_key,
                "parent_sha256": parent_digest,
                "first_chunk_index": first,
                "descriptors": list(
                    descriptors[first : first + _PREFIX_SEGMENT_DESCRIPTORS]
                ),
            }
            encoded_segment = _canonical_json(segment)
            segment_digest = _sha256(encoded_segment)
            segment_objects.append(
                (
                    self._prefix_segment_path(identity, segment_digest),
                    encoded_segment,
                )
            )
            segment_digests.append(segment_digest)
            parent_digest = segment_digest
        _publish_immutable_batch(segment_objects)

        source_manifest_digest = _sha256(source_encoded)
        alias_keys: list[EntryKey] = []
        for boundary in boundaries:
            chunk_count = descriptor_ends[boundary]
            tail_segment = segment_digests[
                (chunk_count - 1) // _PREFIX_SEGMENT_DESCRIPTORS
            ]
            context_digest = digests[boundary]
            alias = {
                "schema": _PREFIX_ALIAS_SCHEMA,
                "format_abi": FORMAT_ABI,
                "storage_mode": storage_mode,
                "identity": identity.to_wire(),
                "context_digest": context_digest,
                "committed_tokens": boundary,
                "chunk_count": chunk_count,
                "tail_segment_sha256": tail_segment,
                "source_manifest_digest": source_manifest_digest,
            }
            alias["metadata_sha256"] = _sha256(_canonical_json(alias))
            _publish_immutable(
                self._prefix_alias_path(identity, context_digest),
                _canonical_json(alias),
            )
            alias_keys.append(
                EntryKey(identity.storage_key, context_digest, "prefix_alias")
            )
        return PrefixAliasReceipt(
            source_manifest_digest=source_manifest_digest,
            aliases_published=len(alias_keys),
            segments_published=len(segment_digests),
            alias_keys=tuple(alias_keys),
        )

    @_tracked_publication("row_tail")
    def commit_extension(
        self,
        *,
        identity: CacheIdentity,
        base_context_digest: str,
        token_ids: Sequence[int],
        identity_salt: str,
        tail_chunks: Sequence[ContextChunk],
    ) -> CommitReceipt:
        """Publish a context extension without rewriting its reusable payload.

        The supplied token sequence is the authority for both the base and
        result commitments. The prior root is fully verified before its chunk
        descriptors are copied into an immutable descriptor chain. Tail
        chunks begin at the last complete reusable chunk; this replaces a
        prior partial terminal object without mutating it.
        """

        if identity.publication_schema != "tail-cow-v1":
            raise ValueError("tail publication requires publication_schema tail-cow-v1")
        if not tail_chunks:
            raise ValueError("tail publication requires at least one tail chunk")
        with _RootGuard(self.root, shared=True, blocking=True):
            base = self.lookup(
                identity,
                base_context_digest,
                verify_chunks=True,
                storage_mode="per_token_rows",
            )
            if not base.is_hit or base._manifest is None:
                raise CacheFormatError(f"base context is not reusable: {base.reason}")
            base_committed_tokens = int(base._manifest["committed_tokens"])
            result_tokens = tail_chunks[-1].logical_end
            if result_tokens > len(token_ids):
                raise ValueError("tail token span exceeds supplied token sequence")
            from sparkcache.spark_context_cache_codec import context_prefix_digest

            base_commitment = context_prefix_digest(
                token_ids,
                identity_salt,
                token_count=base_committed_tokens,
            )
            if base_commitment != base_context_digest:
                raise CommitConflict(
                    "base digest disagrees with supplied token sequence"
                )
            result_context_digest = context_prefix_digest(
                token_ids,
                identity_salt,
                token_count=result_tokens,
            )
            if result_context_digest == base_context_digest:
                raise ValueError("tail publication must extend the base context")

            reused_tokens = (
                base_committed_tokens // identity.chunk_tokens
            ) * identity.chunk_tokens
            base_descriptors = tuple(
                descriptor
                for descriptor in base._manifest["chunks"]
                if int(descriptor["logical_end"]) <= reused_tokens
            )
            if len(base_descriptors) != reused_tokens // identity.chunk_tokens:
                raise CacheFormatError("base descriptor geometry is incomplete")
            expected_start = reused_tokens
            tail_descriptors: list[dict[str, Any]] = []
            tail_objects: list[tuple[Path, bytes]] = []
            for index, chunk in enumerate(tail_chunks):
                if chunk.logical_start != expected_start:
                    raise ValueError("tail chunk ranges must be contiguous")
                token_count = chunk.logical_end - chunk.logical_start
                if token_count > identity.chunk_tokens or (
                    index < len(tail_chunks) - 1
                    and token_count != identity.chunk_tokens
                ):
                    raise ValueError(
                        "tail chunk range disagrees with identity geometry"
                    )
                _require_complete_chunk(chunk, identity.required_records)
                encoded = _encode_chunk(chunk)
                digest = _sha256(encoded)
                tail_descriptors.append(
                    {
                        "sha256": digest,
                        "bytes": len(encoded),
                        "logical_start": chunk.logical_start,
                        "logical_end": chunk.logical_end,
                    }
                )
                tail_objects.append((self.root / "chunks" / f"{digest}.spcc", encoded))
                expected_start = chunk.logical_end
            if (
                expected_start != result_tokens
                or result_tokens <= base_committed_tokens
            ):
                raise ValueError("tail chunks do not extend the base context")

            (
                segment_digest,
                _segment_count,
                segment_allocated_bytes,
            ) = self._publish_descriptor_chain(identity, base_descriptors)
            publication = self._active_publication()
            publication.describe_payload(
                sum(int(item["bytes"]) for item in tail_descriptors),
                sum(int(item["bytes"]) for item in base_descriptors),
            )
            _publish_immutable_batch(tail_objects)
            root = {
                "schema": _TAIL_MANIFEST_SCHEMA,
                "format_abi": FORMAT_ABI,
                "identity": identity.to_wire(),
                "context_digest": result_context_digest,
                "committed_tokens": result_tokens,
                "base_context_digest": base_context_digest,
                "base_manifest_sha256": base.manifest_digest,
                "base_committed_tokens": base_committed_tokens,
                "reused_tokens": reused_tokens,
                "base_chunk_count": len(base_descriptors),
                "base_tail_segment_sha256": segment_digest,
                "tail_chunks": tail_descriptors,
            }
            root["metadata_sha256"] = _sha256(_canonical_json(root))
            encoded_root = _canonical_json(root)
            _publish_immutable(
                self._manifest_path(identity, result_context_digest),
                encoded_root,
            )
            return CommitReceipt(
                manifest_digest=_sha256(encoded_root),
                committed_tokens=result_tokens,
                encoded_bytes=(
                    len(encoded_root)
                    + sum(int(item["bytes"]) for item in tail_descriptors)
                ),
                allocated_bytes_upper_bound=(
                    segment_allocated_bytes
                    + sum(
                        (size + 4095) // 4096 * 4096
                        for size in (
                            len(encoded_root),
                            *(int(item["bytes"]) for item in tail_descriptors),
                        )
                    )
                ),
            )

    @_tracked_publication("page_snapshot")
    def commit_page_snapshot(
        self,
        *,
        identity: CacheIdentity,
        context_digest: str,
        span_tokens: int,
        snapshot: Any,
    ) -> CommitReceipt:
        """Publish one flat opaque page snapshot as authenticated extents.

        Physical extents are independent of the identity's logical chunk
        geometry. The root retains the exact 256-token boundary and chunk
        count used by lookup/admission. Restore authenticates the root metadata,
        contiguous descriptor geometry, and each extent before placement.
        """

        _validate_digest(context_digest, "context_digest")
        if identity.publication_schema not in ("", *_PAGE_TAIL_SCHEMAS):
            raise ValueError(
                "page snapshot publication requires the flat or page-tail"
                " publication schema"
            )
        if identity.required_records != frozenset(
            (StateRecord.TARGET_CKV, StateRecord.LOGICAL_POSITIONS)
        ):
            raise ValueError(
                "page snapshot publication requires target and logical-position records"
            )
        if (
            type(span_tokens) is not int
            or span_tokens <= 0
            or span_tokens % identity.chunk_tokens
        ):
            raise ValueError("page snapshot span must cover complete logical chunks")
        if isinstance(snapshot, (bytes, bytearray)):
            snapshot_bytes = len(snapshot)
            snapshot_view = memoryview(snapshot)

            def read_range(start: int, end: int) -> bytes:
                return snapshot_view[start:end].tobytes()

        else:
            snapshot_bytes = getattr(snapshot, "total_bytes", None)
            read_range = getattr(snapshot, "read_range", None)
            if (
                type(snapshot_bytes) is not int
                or snapshot_bytes <= 0
                or not callable(read_range)
            ):
                raise ValueError(
                    "page snapshot payload must be bytes or a bounded scatter view"
                )

        integrity = _payload_integrity(identity.publication_schema)
        nonce = uuid.uuid4().hex
        with _RootGuard(self.root, shared=True, blocking=True):
            publication = self._active_publication()
            publication.describe_payload(snapshot_bytes)
            descriptors: list[dict[str, Any]] = []
            objects: list[tuple[Path, bytes]] = []
            snapshot_digest = hashlib.sha256()
            for index, start in enumerate(
                range(0, snapshot_bytes, _PAGE_SNAPSHOT_OBJECT_BYTES)
            ):
                end = min(snapshot_bytes, start + _PAGE_SNAPSHOT_OBJECT_BYTES)
                encoded = read_range(start, end)
                if not isinstance(encoded, bytes) or len(encoded) != end - start:
                    raise ValueError("page snapshot scatter range differs")
                if integrity == "sha256":
                    snapshot_digest.update(encoded)
                    object_name = _sha256(encoded)
                else:
                    object_name = _derived_object_name(
                        identity, context_digest, "snapshot", index, nonce
                    )
                descriptors.append(
                    _object_descriptor(integrity, object_name, encoded, start, end)
                )
                objects.append(
                    (
                        self.root / "chunks" / f"{object_name}.spcc",
                        encoded,
                    )
                )
                if len(objects) == _PAGE_SNAPSHOT_WRITE_BATCH_SIZE:
                    _publish_immutable_batch(objects)
                    objects.clear()
            if objects:
                _publish_immutable_batch(objects)

            root = {
                "schema": _PAGE_SNAPSHOT_MANIFEST_SCHEMA,
                "format_abi": FORMAT_ABI,
                "storage_mode": "block_pages_v1",
                "identity": identity.to_wire(),
                "context_digest": context_digest,
                "committed_tokens": span_tokens,
                "snapshot_encoded_bytes": snapshot_bytes,
                "snapshot_object_bytes": _PAGE_SNAPSHOT_OBJECT_BYTES,
                "snapshot_objects": descriptors,
                "snapshot_sha256": (
                    snapshot_digest.hexdigest()
                    if integrity == "sha256"
                    else _boundary_token(context_digest, span_tokens)
                ),
                "logical_chunk_tokens": identity.chunk_tokens,
                "logical_chunk_count": span_tokens // identity.chunk_tokens,
            }
            root["metadata_sha256"] = _sha256(_canonical_json(root))
            encoded_root = _canonical_json(root)
            _publish_immutable(
                self._manifest_path(identity, context_digest),
                encoded_root,
            )
            # This process hashed the complete snapshot to the digest it just
            # bound into the manifest. Retain immutable bytes only; a mutable
            # buffer is copied so a caller cannot change the retained value.
            if isinstance(snapshot, bytes):
                _remember_hot_page_snapshot(root["snapshot_sha256"], snapshot)
            elif isinstance(snapshot, bytearray) and _hot_page_snapshot_fits(
                snapshot_bytes
            ):
                _remember_hot_page_snapshot(root["snapshot_sha256"], bytes(snapshot))
            return CommitReceipt(
                manifest_digest=_sha256(encoded_root),
                committed_tokens=span_tokens,
                encoded_bytes=len(encoded_root)
                + sum(int(item["bytes"]) for item in descriptors),
                allocated_bytes_upper_bound=sum(
                    (size + 4095) // 4096 * 4096
                    for size in (
                        len(encoded_root),
                        *(int(item["bytes"]) for item in descriptors),
                    )
                ),
            )

    @_tracked_publication("page_delta")
    def commit_page_extension(
        self,
        *,
        identity: CacheIdentity,
        base_context_digest: str,
        token_ids: Sequence[int],
        identity_salt: str,
        layout: Any,
        base_block_counts: Sequence[int],
        result_block_counts: Sequence[int],
        base_boundary_tokens: int,
        result_boundary_tokens: int,
        result_snapshot: object,
        verified_base_snapshot: bytes | bytearray | None = None,
    ) -> CommitReceipt:
        """Publish a page-semantic delta over one verified opaque snapshot."""

        if identity.publication_schema not in _PAGE_TAIL_SCHEMAS:
            raise ValueError(
                "page tail publication requires a page-tail-cow schema"
            )
        from sparkcache.spark_context_cache_codec import (
            context_prefix_digest,
        )
        from sparkcache.spark_context_cache_hybrid import (
            encode_page_delta,
            encode_page_delta_from_capture,
        )

        with _RootGuard(self.root, shared=True, blocking=True):
            base = self.lookup(
                identity,
                base_context_digest,
                # Materialization authenticates every payload it consumes.
                # The probe checks metadata and sizes without reading the
                # complete base a second time before that restore.
                verify_chunks=False,
                verify_chunk_metadata=True,
            )
            if not base.is_hit or base._manifest is None:
                raise CacheFormatError(f"base context is not reusable: {base.reason}")
            if int(base._manifest["committed_tokens"]) != base_boundary_tokens:
                raise CacheFormatError("base context boundary differs")
            if (
                context_prefix_digest(
                    token_ids,
                    identity_salt,
                    token_count=base_boundary_tokens,
                )
                != base_context_digest
            ):
                raise CommitConflict(
                    "base digest disagrees with supplied token sequence"
                )
            publication_base = base
            publication_base_digest = base_context_digest
            publication_base_boundary = base_boundary_tokens
            publication_base_block_counts = tuple(base_block_counts)
            if (
                identity.publication_schema == "page-tail-cow-v1"
                and self._page_delta_root_count(base._manifest)
                >= _MAX_PAGE_DELTA_DEPTH
            ):
                terminal_delta = base._manifest
                while _is_page_delta_root(terminal_delta["base_root"]):
                    terminal_delta = terminal_delta["base_root"]
                terminal_root = terminal_delta["base_root"]
                publication_base_digest = terminal_delta["base_context_digest"]
                publication_base_boundary = terminal_delta["base_committed_tokens"]
                publication_base_block_counts = tuple(
                    terminal_delta["base_block_counts"]
                )
                if (
                    context_prefix_digest(
                        token_ids,
                        identity_salt,
                        token_count=publication_base_boundary,
                    )
                    != publication_base_digest
                ):
                    raise CommitConflict(
                        "flattened base digest disagrees with supplied token sequence"
                    )
                publication_base = LookupResult(
                    True,
                    "hit",
                    manifest_digest=terminal_delta["base_root_sha256"],
                    _manifest=terminal_root,
                    root_kind=(
                        "page_snapshot"
                        if _is_page_snapshot_root(terminal_root)
                        else "manifest"
                    ),
                )
            result_context_digest = context_prefix_digest(
                token_ids,
                identity_salt,
                token_count=result_boundary_tokens,
            )
            if (
                verified_base_snapshot is not None
                and publication_base is base
            ):
                from sparkcache.spark_context_cache_hybrid import (
                    _verify_page_snapshot_bytes, plan_page_snapshot,
                )

                plan_page_snapshot(
                    layout,
                    verified_base_snapshot,
                    publication_base_block_counts,
                )
                expected = self._publication_snapshot_digest(
                    publication_base, layout=layout,
                    block_counts=publication_base_block_counts,
                    boundary_tokens=publication_base_boundary,
                )
                if _payload_integrity(identity.publication_schema) == "sha256":
                    base_proof = _verify_page_snapshot_bytes(verified_base_snapshot)
                    if base_proof.sha256 != expected:
                        raise CacheFormatError(
                            "publication base snapshot digest differs"
                        )
                else:
                    # Payload bytes are not hashed under the v3 schemas: the
                    # supplied base is bound to the boundary token by the
                    # caller having restored that boundary.
                    from sparkcache.spark_context_cache_hybrid import (
                        _VerifiedPageSnapshot,
                    )

                    base_proof = (
                        verified_base_snapshot
                        if isinstance(verified_base_snapshot, _VerifiedPageSnapshot)
                        else _VerifiedPageSnapshot(
                            verified_base_snapshot
                            if isinstance(verified_base_snapshot, bytes)
                            else bytes(verified_base_snapshot),
                            expected,
                        )
                    )
            else:
                with _PublicationReadFiles():
                    base_proof = self.restore_page_snapshot(
                        publication_base,
                        layout=layout,
                        result_block_counts=publication_base_block_counts,
                        result_boundary_tokens=publication_base_boundary,
                        _return_verified=True,
                    )
            base_snapshot = base_proof.payload
            captured_reuse = getattr(
                result_snapshot,
                "reused_pages_by_group",
                None,
            )
            if (
                identity.publication_schema in _FLAT_PAGE_SCHEMAS
                and self._page_delta_stage_limit_count
                and len(base._manifest.get("delta_stages", ()))
                >= self._page_delta_stage_limit_count
            ):
                from sparkcache.spark_context_cache_hybrid import (
                    materialize_page_extension_capture, plan_page_snapshot,
                )

                # The captured tail belongs to the result boundary. Publishing
                # the base here would silently discard this turn's state.
                complete = (
                    materialize_page_extension_capture(
                        layout, base_snapshot, result_snapshot,
                        base_block_counts=publication_base_block_counts,
                        result_block_counts=result_block_counts,
                        reused_pages_by_group=captured_reuse,
                    ) if captured_reuse is not None else result_snapshot
                )
                plan_page_snapshot(layout, complete, result_block_counts)
                return self.commit_page_snapshot(
                    identity=identity, context_digest=result_context_digest,
                    span_tokens=result_boundary_tokens, snapshot=complete,
                )
            # The base proof carries the digest the base bytes were verified
            # against, so the encoders bind it instead of hashing the base.
            base_snapshot_sha256 = base_proof.sha256
            integrity = _payload_integrity(identity.publication_schema)
            result_token = (
                None
                if integrity == "sha256"
                else _boundary_token(result_context_digest, result_boundary_tokens)
            )
            hot_result: bytes | bytearray | None = None
            if captured_reuse is not None:
                if identity.publication_schema not in _FLAT_PAGE_SCHEMAS:
                    raise ValueError(
                        "direct sparse page deltas require a flat page schema"
                    )
                from sparkcache.spark_context_cache_hybrid import (
                    encode_page_delta_from_capture_with_result,
                    page_snapshot_encoded_size,
                )

                if _hot_page_snapshot_fits(
                    page_snapshot_encoded_size(layout, result_block_counts)
                ):
                    delta, hot_result = encode_page_delta_from_capture_with_result(
                        layout,
                        base_proof,
                        result_snapshot,
                        base_block_counts=publication_base_block_counts,
                        result_block_counts=result_block_counts,
                        reused_pages_by_group=captured_reuse,
                        base_boundary_tokens=publication_base_boundary,
                        result_boundary_tokens=result_boundary_tokens,
                        base_snapshot_sha256=base_snapshot_sha256,
                        integrity=integrity,
                        result_snapshot_sha256=result_token,
                    )
                else:
                    delta = encode_page_delta_from_capture(
                        layout,
                        base_proof,
                        result_snapshot,
                        base_block_counts=publication_base_block_counts,
                        result_block_counts=result_block_counts,
                        reused_pages_by_group=captured_reuse,
                        base_boundary_tokens=publication_base_boundary,
                        result_boundary_tokens=result_boundary_tokens,
                        base_snapshot_sha256=base_snapshot_sha256,
                        integrity=integrity,
                        result_snapshot_sha256=result_token,
                    )
            else:
                if not isinstance(result_snapshot, (bytes, bytearray)):
                    raise ValueError("page delta result snapshot is invalid")
                delta = encode_page_delta(
                    layout,
                    base_snapshot,
                    result_snapshot,
                    base_block_counts=publication_base_block_counts,
                    result_block_counts=result_block_counts,
                    base_boundary_tokens=publication_base_boundary,
                    result_boundary_tokens=result_boundary_tokens,
                    base_snapshot_sha256=base_snapshot_sha256,
                    integrity=integrity,
                    result_snapshot_sha256=result_token,
                )
                if isinstance(result_snapshot, bytes):
                    hot_result = result_snapshot
            publication = self._active_publication()
            publication.describe_payload(len(delta), len(base_snapshot))
            descriptors: list[dict[str, Any]] = []
            objects: list[tuple[Path, bytes]] = []
            delta_view = memoryview(delta)
            nonce = uuid.uuid4().hex
            for index, start in enumerate(range(0, len(delta), _PAGE_DELTA_OBJECT_BYTES)):
                end = min(len(delta), start + _PAGE_DELTA_OBJECT_BYTES)
                encoded = delta_view[start:end].tobytes()
                if integrity == "sha256":
                    object_name = _sha256(encoded)
                else:
                    object_name = _derived_object_name(
                        identity, result_context_digest, "delta", index, nonce
                    )
                descriptors.append(
                    _object_descriptor(integrity, object_name, encoded, start, end)
                )
                objects.append(
                    (
                        self.root / "chunks" / f"{object_name}.spcc",
                        encoded,
                    )
                )
                if len(objects) == _PAGE_DELTA_WRITE_BATCH_SIZE:
                    _publish_immutable_batch(objects)
                    objects.clear()
            if objects:
                _publish_immutable_batch(objects)
            from sparkcache.spark_context_cache_hybrid import (
                page_delta_header_prefix,
            )

            # This process encoded the delta and wrote its first object under
            # this digest, so the header is authenticated for later restores.
            header_prefix = page_delta_header_prefix(delta)
            _remember_authenticated_delta_header(
                descriptors[0]["sha256"],
                descriptors[0]["bytes"],
                header_prefix,
            )
            if hot_result is not None:
                # The encoder hashed exactly this buffer to the result digest
                # it bound into the header; the buffer has no other owner.
                from sparkcache.spark_context_cache_hybrid import plan_page_delta

                result_sha256 = plan_page_delta(
                    layout,
                    header_prefix,
                    base_block_counts=publication_base_block_counts,
                    result_block_counts=result_block_counts,
                    base_boundary_tokens=publication_base_boundary,
                    result_boundary_tokens=result_boundary_tokens,
                    total_bytes=len(delta),
                ).result_snapshot_sha256
                _remember_hot_page_snapshot(result_sha256, hot_result)
            base_root = dict(publication_base._manifest)
            if identity.publication_schema in _FLAT_PAGE_SCHEMAS:
                if base._manifest.get("schema") == _PAGE_DELTA_MANIFEST_SCHEMA_V3:
                    _validate_page_delta_root(
                        base._manifest,
                        identity=identity,
                        context_digest=base_context_digest,
                    )
                    flat_base_root = dict(base._manifest["base_root"])
                    flat_base_digest = base._manifest["base_context_digest"]
                    flat_base_boundary = base._manifest["base_committed_tokens"]
                    flat_base_counts = list(base._manifest["base_block_counts"])
                    stages = [dict(stage) for stage in base._manifest["delta_stages"]]
                else:
                    flat_base_root = dict(base._manifest)
                    flat_base_digest = base_context_digest
                    flat_base_boundary = base_boundary_tokens
                    flat_base_counts = list(base_block_counts)
                    stages = []
                stage_entry = {
                    "context_digest": result_context_digest,
                    "committed_tokens": result_boundary_tokens,
                    "base_context_digest": base_context_digest,
                    "base_committed_tokens": base_boundary_tokens,
                    "base_block_counts": list(base_block_counts),
                    "result_block_counts": list(result_block_counts),
                    "delta_encoded_bytes": len(delta),
                    "delta_object_bytes": _PAGE_DELTA_OBJECT_BYTES,
                    "delta_objects": descriptors,
                    "delta_sha256": (
                        _sha256(delta)
                        if integrity == "sha256"
                        else _boundary_token(result_context_digest, -len(delta))
                    ),
                }
                if integrity != "sha256":
                    stage_entry["header_sha256"] = _sha256(header_prefix)
                stages.append(stage_entry)
                root = {
                    "schema": _PAGE_DELTA_MANIFEST_SCHEMA_V3,
                    "format_abi": FORMAT_ABI,
                    "identity": identity.to_wire(),
                    "context_digest": result_context_digest,
                    "committed_tokens": result_boundary_tokens,
                    "base_context_digest": flat_base_digest,
                    "base_committed_tokens": flat_base_boundary,
                    "base_root": flat_base_root,
                    "base_root_sha256": _sha256(_canonical_json(flat_base_root)),
                    "layout_sha256": layout.digest,
                    "base_block_counts": flat_base_counts,
                    "result_block_counts": list(result_block_counts),
                    "delta_stages": stages,
                    "logical_chunk_tokens": identity.chunk_tokens,
                }
            else:
                root = {
                    "schema": _PAGE_DELTA_MANIFEST_SCHEMA_V2,
                    "format_abi": FORMAT_ABI,
                    "identity": identity.to_wire(),
                    "context_digest": result_context_digest,
                    "committed_tokens": result_boundary_tokens,
                    "base_context_digest": publication_base_digest,
                    "base_committed_tokens": publication_base_boundary,
                    "base_root": base_root,
                    "base_root_sha256": _sha256(_canonical_json(base_root)),
                    "layout_sha256": layout.digest,
                    "base_block_counts": list(publication_base_block_counts),
                    "result_block_counts": list(result_block_counts),
                    "delta_encoded_bytes": len(delta),
                    "delta_object_bytes": _PAGE_DELTA_OBJECT_BYTES,
                    "delta_objects": descriptors,
                    "delta_sha256": _sha256(delta),
                    "logical_chunk_tokens": identity.chunk_tokens,
                }
            root["metadata_sha256"] = _sha256(_canonical_json(root))
            encoded_root = _canonical_json(root)
            _publish_immutable(
                self._manifest_path(identity, result_context_digest),
                encoded_root,
            )
            return CommitReceipt(
                manifest_digest=_sha256(encoded_root),
                committed_tokens=result_boundary_tokens,
                encoded_bytes=len(encoded_root)
                + sum(int(item["bytes"]) for item in descriptors),
                allocated_bytes_upper_bound=sum(
                    (size + 4095) // 4096 * 4096
                    for size in (
                        len(encoded_root),
                        *(int(item["bytes"]) for item in descriptors),
                    )
                ),
            )

    def _publication_snapshot_digest(
        self, lookup: LookupResult, *, layout: Any,
        block_counts: Sequence[int], boundary_tokens: int,
    ) -> str:
        """Bind supplied base bytes to the authenticated publication boundary."""
        manifest = lookup._manifest
        if _is_page_snapshot_root(manifest):
            return manifest["snapshot_sha256"]
        if _is_page_delta_root(manifest) and manifest.get("schema") in (
            _PAGE_DELTA_MANIFEST_SCHEMA_V2, _PAGE_DELTA_MANIFEST_SCHEMA_V3,
        ):
            from sparkcache.spark_context_cache_hybrid import plan_page_delta

            stage = (manifest["delta_stages"][-1]
                     if manifest["schema"] == _PAGE_DELTA_MANIFEST_SCHEMA_V3 else manifest)
            header = _authenticated_page_delta_header(
                self.root / "chunks", stage["delta_objects"][0], stage.get("header_sha256"),
            )
            plan = plan_page_delta(
                layout, header, base_block_counts=stage["base_block_counts"],
                result_block_counts=block_counts,
                base_boundary_tokens=stage["base_committed_tokens"],
                result_boundary_tokens=boundary_tokens,
                total_bytes=stage["delta_encoded_bytes"],
            )
            # A delta base denotes its composed result, never its root snapshot.
            return plan.result_snapshot_sha256
        proof = self.restore_page_snapshot(
            lookup, layout=layout, result_block_counts=block_counts,
            result_boundary_tokens=boundary_tokens, _return_verified=True,
        )
        return proof.sha256

    def restore_page_snapshot(
        self,
        lookup: LookupResult,
        *,
        layout: Any,
        result_block_counts: Sequence[int],
        result_boundary_tokens: int,
        base_reader: Callable[
            [PageBaseReadEvidence, Callable[[], bytes | bytearray]],
            bytes | bytearray,
        ]
        | None = None,
        _depth: int = 0,
        _return_verified: bool = False,
    ) -> bytes | bytearray | _VerifiedPageSnapshot:
        """Materialize an authenticated flat or delta-backed page snapshot.

        Flat histories verify surviving tails and the final composed digest.
        The private publication path retains that proof for sparse encoding.
        """

        if not lookup.is_hit or lookup._manifest is None:
            raise ValueError("cannot restore a cache miss")
        from sparkcache.spark_context_cache_hybrid import (
            _VerifiedPageSnapshot, _verify_page_snapshot_bytes,
        )

        manifest = lookup._manifest
        if _is_page_snapshot_root(manifest):
            identity_wire = dict(manifest["identity"])
            if "record_schema" in identity_wire:
                identity_wire["record_schema"] = tuple(identity_wire["record_schema"])
            identity = CacheIdentity(**identity_wire)
            descriptors = _validate_page_snapshot_root(
                manifest,
                identity=identity,
                context_digest=manifest["context_digest"],
            )
            if manifest["committed_tokens"] != result_boundary_tokens:
                raise CacheFormatError("page snapshot restore boundary differs")
            snapshot_sha256 = manifest["snapshot_sha256"]
            hot = _hot_page_snapshot(snapshot_sha256, manifest["snapshot_encoded_bytes"])
            if hot is not None:
                # Hot buffers are never modified after insertion, so the
                # carrier may wrap one without copying it.
                return _VerifiedPageSnapshot(hot, snapshot_sha256) if _return_verified else hot
            snapshot = self._read_page_snapshot_objects(
                descriptors,
                encoded_bytes=manifest["snapshot_encoded_bytes"],
                encoded_sha256=snapshot_sha256,
            )
            # _read_page_snapshot_objects hashed exactly these bytes to the
            # manifest-bound digest; the buffer is private to this call.
            if _return_verified:
                immutable = bytes(snapshot)
                _remember_hot_page_snapshot(snapshot_sha256, immutable)
                return _VerifiedPageSnapshot(immutable, snapshot_sha256)
            _remember_hot_page_snapshot(snapshot_sha256, snapshot)
            return snapshot
        if not _is_page_delta_root(manifest):
            chunks = self.restore(lookup)
            if chunks is None:
                raise CacheFormatError("page snapshot restore failed")
            snapshot = b"".join(chunk.records[StateRecord.TARGET_CKV] for chunk in chunks)
            return _verify_page_snapshot_bytes(snapshot) if _return_verified else snapshot
        if manifest.get("schema") == _PAGE_DELTA_MANIFEST_SCHEMA_V3:
            identity_wire = dict(manifest["identity"])
            if "record_schema" in identity_wire:
                identity_wire["record_schema"] = tuple(
                    identity_wire["record_schema"]
                )
            identity = CacheIdentity(**identity_wire)
            base_root, _descriptors = _validate_page_delta_root(
                manifest,
                identity=identity,
                context_digest=manifest["context_digest"],
            )
            if (
                manifest["layout_sha256"] != layout.digest
                or tuple(manifest["result_block_counts"])
                != tuple(result_block_counts)
                or manifest["committed_tokens"] != result_boundary_tokens
            ):
                raise CacheFormatError("flat page delta restore geometry differs")
            base_lookup = LookupResult(
                True,
                "hit",
                manifest_digest=manifest["base_root_sha256"],
                _manifest=base_root,
                root_kind=(
                    "page_snapshot"
                    if _is_page_snapshot_root(base_root)
                    else "manifest"
                ),
            )
            from sparkcache.spark_context_cache_hybrid import (
                PageDeltaPayloadChecksumError,
                compose_page_history,
                page_snapshot_encoded_size,
                plan_page_delta,
            )

            # Authenticate each stage header through its first macro object,
            # then read every layer tail at most once. A tail superseded by a
            # later stage is never read; every byte that is read is checked
            # against its header-bound tail digest before it is copied, and
            # the complete result is checked against the newest stage's
            # result digest before it escapes.
            object_root = self.root / "chunks"
            stages = manifest["delta_stages"]
            plans = []
            for stage in stages:
                header = _authenticated_page_delta_header(
                    object_root, stage["delta_objects"][0], stage.get("header_sha256")
                )
                plans.append(
                    plan_page_delta(
                        layout,
                        header,
                        base_block_counts=stage["base_block_counts"],
                        result_block_counts=stage["result_block_counts"],
                        base_boundary_tokens=stage["base_committed_tokens"],
                        result_boundary_tokens=stage["committed_tokens"],
                        total_bytes=stage["delta_encoded_bytes"],
                    )
                )
            # The newest authenticated header binds the result digest. A
            # process-local copy that hashed to it is the result itself and
            # needs neither the base nor any tail.
            result_sha256 = plans[-1].result_snapshot_sha256
            result_bytes = page_snapshot_encoded_size(layout, result_block_counts)
            hot = _hot_page_snapshot(result_sha256, result_bytes)
            if hot is not None:
                return _VerifiedPageSnapshot(hot, result_sha256) if _return_verified else hot
            snapshot = self.restore_page_snapshot(
                base_lookup,
                layout=layout,
                result_block_counts=manifest["base_block_counts"],
                result_boundary_tokens=manifest["base_committed_tokens"],
                _return_verified=True,
            )

            def read_tail(stage_index: int, tail: Any) -> bytearray:
                return _read_object_extent(
                    _object_extent_pieces(
                        object_root,
                        stages[stage_index]["delta_objects"],
                        tail.source_start,
                        tail.source_end,
                    )
                )

            try:
                composed = compose_page_history(
                    layout,
                    snapshot,
                    manifest["base_block_counts"],
                    plans,
                    read_tail,
                    _return_verified=_return_verified,
                )
            except PageDeltaPayloadChecksumError as error:
                # Stored bytes differ from authenticated metadata: corruption.
                raise CacheFormatError(str(error)) from error
            # compose_page_history hashed the immutable result to the newest
            # header's digest before returning it.
            _remember_hot_page_snapshot(
                result_sha256,
                composed.payload if isinstance(composed, _VerifiedPageSnapshot) else composed,
            )
            return composed
        if _depth >= _MAX_PAGE_DELTA_DEPTH:
            raise CacheFormatError("page delta graph exceeds the depth limit")
        evidence = self.page_delta_base_read_evidence(
            lookup,
            layout=layout,
            result_block_counts=result_block_counts,
            result_boundary_tokens=result_boundary_tokens,
        )
        identity_wire = dict(manifest["identity"])
        if "record_schema" in identity_wire:
            identity_wire["record_schema"] = tuple(identity_wire["record_schema"])
        identity = CacheIdentity(**identity_wire)
        base_root, delta_descriptors = _validate_page_delta_root(
            manifest,
            identity=identity,
            context_digest=manifest["context_digest"],
        )
        if (
            manifest["layout_sha256"] != layout.digest
            or tuple(manifest["result_block_counts"]) != tuple(result_block_counts)
            or manifest["committed_tokens"] != result_boundary_tokens
        ):
            raise CacheFormatError("page delta restore geometry differs")
        base_lookup = LookupResult(
            True,
            "hit",
            manifest_digest=manifest["base_root_sha256"],
            _manifest=base_root,
            root_kind=(
                "page_delta"
                if _is_page_delta_root(base_root)
                else (
                    "page_snapshot" if _is_page_snapshot_root(base_root) else "manifest"
                )
            ),
        )
        def read_base() -> bytes | bytearray:
            return self.restore_page_snapshot(
                base_lookup,
                layout=layout,
                result_block_counts=manifest["base_block_counts"],
                result_boundary_tokens=manifest["base_committed_tokens"],
                base_reader=None,
                _depth=_depth + 1,
            )

        base_snapshot = (
            base_reader(evidence, read_base)
            if base_reader is not None and _depth == 0
            else read_base()
        )
        if manifest["schema"] == _PAGE_DELTA_MANIFEST_SCHEMA_V2:
            encoded_delta = self._read_page_delta_objects(
                delta_descriptors,
                encoded_bytes=manifest["delta_encoded_bytes"],
                encoded_sha256=manifest["delta_sha256"],
            )
        else:
            delta_chunks = self._read_context_chunks(
                delta_descriptors,
                identity.required_records,
            )
            encoded_delta = b"".join(
                chunk.records[StateRecord.TARGET_CKV] for chunk in delta_chunks
            )
        from sparkcache.spark_context_cache_hybrid import apply_page_delta

        snapshot = apply_page_delta(
            layout,
            base_snapshot,
            encoded_delta,
            base_block_counts=manifest["base_block_counts"],
            result_block_counts=manifest["result_block_counts"],
            base_boundary_tokens=manifest["base_committed_tokens"],
            result_boundary_tokens=manifest["committed_tokens"],
        )
        return _verify_page_snapshot_bytes(snapshot) if _return_verified else snapshot

    def page_delta_base_read_evidence(
        self,
        lookup: LookupResult,
        *,
        layout: Any,
        result_block_counts: Sequence[int],
        result_boundary_tokens: int,
    ) -> PageBaseReadEvidence:
        """Authenticate metadata that permits sharing one immutable page base.

        This method reads no payload object. Each result root still authenticates
        and reads its private delta inside ``restore_page_snapshot``.
        """

        if (
            not lookup.is_hit
            or lookup.root_kind != "page_delta"
            or lookup._manifest is None
        ):
            raise ValueError("page-base evidence requires a page-delta cache hit")
        manifest = lookup._manifest
        identity_wire = dict(manifest["identity"])
        if "record_schema" in identity_wire:
            identity_wire["record_schema"] = tuple(identity_wire["record_schema"])
        identity = CacheIdentity(**identity_wire)
        base_root, _delta_descriptors = _validate_page_delta_root(
            manifest,
            identity=identity,
            context_digest=manifest["context_digest"],
        )
        result_counts = tuple(int(value) for value in result_block_counts)
        base_counts = tuple(int(value) for value in manifest["base_block_counts"])
        if (
            manifest["layout_sha256"] != layout.digest
            or tuple(manifest["result_block_counts"]) != result_counts
            or manifest["committed_tokens"] != result_boundary_tokens
            or base_root.get("committed_tokens")
            != manifest["base_committed_tokens"]
        ):
            raise CacheFormatError("page delta restore geometry differs")
        base_digest = manifest["base_context_digest"]
        from sparkcache.spark_context_cache_hybrid import page_snapshot_encoded_size

        base_encoded_bytes = page_snapshot_encoded_size(layout, base_counts)
        if _is_page_delta_root(base_root):
            self._page_graph_descriptors(
                base_root,
                identity=identity,
                context_digest=base_digest,
                depth=1,
            )
            if (
                base_root["layout_sha256"] != layout.digest
                or tuple(base_root["result_block_counts"]) != base_counts
            ):
                raise CacheFormatError("page delta base geometry differs")
            base_root_kind = "page_delta"
        elif _is_page_snapshot_root(base_root):
            _validate_page_snapshot_root(
                base_root,
                identity=identity,
                context_digest=base_digest,
            )
            if base_root["snapshot_encoded_bytes"] != base_encoded_bytes:
                raise CacheFormatError("page snapshot base geometry differs")
            base_root_kind = "page_snapshot"
        else:
            _validate_manifest_metadata(
                base_root,
                EntryKey(identity.storage_key, base_digest),
                expected_identity=identity,
            )
            base_root_kind = "manifest"
        return PageBaseReadEvidence(
            identity_storage_key=identity.storage_key,
            base_context_digest=base_digest,
            base_root_sha256=manifest["base_root_sha256"],
            base_root_kind=base_root_kind,
            layout_sha256=manifest["layout_sha256"],
            base_block_counts=base_counts,
            base_boundary_tokens=manifest["base_committed_tokens"],
            base_encoded_bytes=base_encoded_bytes,
        )

    def begin(
        self,
        *,
        identity: CacheIdentity,
        context_digest: str,
        span_tokens: int | None = None,
    ) -> ManifestTransaction:
        """Begin an invisible, incrementally written context transaction."""

        return ManifestTransaction(
            store=self,
            identity=identity,
            context_digest=context_digest,
            span_tokens=span_tokens,
        )

    def begin_context(
        self,
        *,
        identity: CacheIdentity,
        context_digest: str,
        span_tokens: int | None = None,
    ) -> ManifestTransaction:
        """Named alias for callers that manage more than one transaction type."""

        return self.begin(
            identity=identity,
            context_digest=context_digest,
            span_tokens=span_tokens,
        )

    def commit(
        self,
        *,
        identity: CacheIdentity,
        context_digest: str,
        chunks: Sequence[ContextChunk],
        span_tokens: int | None = None,
    ) -> CommitReceipt:
        """Commit chunks as one transaction. When span_tokens is given,
        commit_manifest additionally requires the chunks to cover exactly
        that span; a truncated chunk sequence then fails the commit instead
        of publishing a short manifest that every restore would reject."""

        transaction = self.begin(
            identity=identity,
            context_digest=context_digest,
            span_tokens=span_tokens,
        )
        try:
            # The transaction's batch path fsyncs file data concurrently and
            # uses one chunk-directory metadata barrier for the complete
            # batch. Keep the batch bounded because each encoded chunk remains
            # live until that barrier completes.
            for start in range(0, len(chunks), _COMMIT_CHUNK_BATCH_SIZE):
                transaction.append_chunks(
                    chunks[start : start + _COMMIT_CHUNK_BATCH_SIZE]
                )
            return transaction.commit_manifest()
        except BaseException:
            transaction.abort()
            raise

    def lookup(
        self,
        identity: CacheIdentity,
        context_digest: str,
        *,
        verify_chunks: bool = True,
        verify_chunk_metadata: bool = False,
        storage_mode: str | None = None,
    ) -> LookupResult:
        """With verify_chunks=False only the manifest itself is validated
        (existence, identity, descriptor structure). Setting
        verify_chunk_metadata also requires each referenced chunk file to
        exist at its declared size, but still does not read payload bytes.
        Restore always re-reads and re-hashes every chunk, so a probe-mode hit
        can still degrade to a clean miss at restore.

        An exact manifest always takes precedence. When no exact manifest
        exists, callers may explicitly select ``per_token_rows`` to consult a
        compact prefix alias. No other storage mode can consume aliases.
        """
        try:
            _validate_digest(context_digest, "context_digest")
        except ValueError:
            return LookupResult(False, "corrupt")
        manifest_path = self._manifest_path(identity, context_digest)
        try:
            encoded = manifest_path.read_bytes()
        except FileNotFoundError:
            if storage_mode != "per_token_rows":
                return LookupResult(False, "absent")
            return self._lookup_prefix_alias(
                identity,
                context_digest,
                verify_chunks=verify_chunks,
                verify_chunk_metadata=verify_chunk_metadata,
            )
        except OSError:
            return LookupResult(False, "corrupt")
        try:
            manifest = json.loads(encoded)
            if (
                isinstance(manifest, dict)
                and manifest.get("schema") == _TAIL_MANIFEST_SCHEMA
            ):
                manifest, _segments = self._resolve_tail_manifest(
                    manifest,
                    identity=identity,
                    context_digest=context_digest,
                )
            is_page_delta = _is_page_delta_root(manifest)
            is_page_snapshot = _is_page_snapshot_root(manifest)
            if is_page_delta:
                chunks = self._page_graph_descriptors(
                    manifest,
                    identity=identity,
                    context_digest=context_digest,
                )
            elif is_page_snapshot:
                chunks = _validate_page_snapshot_root(
                    manifest,
                    identity=identity,
                    context_digest=context_digest,
                )
            else:
                chunks = _validate_manifest_metadata(
                    manifest,
                    EntryKey(identity.storage_key, context_digest),
                    expected_identity=identity,
                )
            for descriptor in chunks:
                digest = descriptor["sha256"]
                encoded_bytes = descriptor["bytes"]
                if verify_chunks:
                    encoded_chunk = (
                        self.root / "chunks" / f"{digest}.spcc"
                    ).read_bytes()
                    if not _payload_matches_descriptor(descriptor, encoded_chunk):
                        raise CacheFormatError("chunk checksum mismatch")
                    if "encoded_start" in descriptor:
                        continue
                    logical_start = descriptor["logical_start"]
                    logical_end = descriptor["logical_end"]
                    # The descriptor digest authenticates the complete encoded
                    # chunk: prefix, header (including record digests and
                    # offsets), and every payload byte. Re-hashing each record
                    # after that whole-chunk match is a redundant full-data
                    # pass. Standalone _decode_chunk callers remain strict by
                    # default.
                    chunk = _decode_chunk(encoded_chunk, verify_record_checksums=False)
                    try:
                        _require_complete_chunk(chunk, identity.required_records)
                    except IncompleteEntry as error:
                        raise CacheFormatError(str(error)) from error
                    if (
                        chunk.logical_start != logical_start
                        or chunk.logical_end != logical_end
                    ):
                        raise CacheFormatError("chunk range disagrees with descriptor")
                elif verify_chunk_metadata:
                    chunk_path = self.root / "chunks" / f"{digest}.spcc"
                    if chunk_path.stat().st_size != encoded_bytes:
                        raise CacheFormatError(
                            "chunk file size disagrees with descriptor"
                        )
            return LookupResult(
                True,
                "hit",
                manifest_digest=_sha256(encoded),
                _manifest=manifest,
                root_kind=(
                    "page_delta"
                    if is_page_delta
                    else ("page_snapshot" if is_page_snapshot else "manifest")
                ),
            )
        except _IncompatibleManifestError:
            return LookupResult(False, "incompatible")
        except (CacheFormatError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return LookupResult(False, "corrupt")

    def _lookup_prefix_alias(
        self,
        identity: CacheIdentity,
        context_digest: str,
        *,
        verify_chunks: bool,
        verify_chunk_metadata: bool,
    ) -> LookupResult:
        try:
            alias_encoded = self._prefix_alias_path(
                identity, context_digest
            ).read_bytes()
            alias = json.loads(alias_encoded)
            committed_tokens, chunk_count, segment_digest = _validate_prefix_alias(
                alias,
                identity=identity,
                context_digest=context_digest,
            )
            segment_count = (
                chunk_count + _PREFIX_SEGMENT_DESCRIPTORS - 1
            ) // _PREFIX_SEGMENT_DESCRIPTORS
            visited: set[str] = set()
            reversed_segments: list[tuple[Mapping[str, Any], ...]] = []
            for segment_index in range(segment_count - 1, -1, -1):
                if segment_digest in visited:
                    raise CacheFormatError("prefix descriptor chain contains a cycle")
                visited.add(segment_digest)
                segment_encoded = self._prefix_segment_path(
                    identity, segment_digest
                ).read_bytes()
                if _sha256(segment_encoded) != segment_digest:
                    raise CacheFormatError(
                        "prefix descriptor segment checksum mismatch"
                    )
                segment = json.loads(segment_encoded)
                descriptors = _validate_prefix_segment(
                    segment,
                    storage_key=identity.storage_key,
                    expected_first_chunk=(segment_index * _PREFIX_SEGMENT_DESCRIPTORS),
                )
                if segment_index < segment_count - 1 and (
                    len(descriptors) != _PREFIX_SEGMENT_DESCRIPTORS
                ):
                    raise CacheFormatError("non-tail prefix segment is incomplete")
                reversed_segments.append(descriptors)
                parent = segment["parent_sha256"]
                if segment_index == 0:
                    if parent is not None:
                        raise CacheFormatError(
                            "prefix descriptor chain has extra parent"
                        )
                elif parent is None:
                    raise CacheFormatError("prefix descriptor chain is truncated")
                else:
                    segment_digest = parent

            ordered_segments = list(reversed(reversed_segments))
            descriptors: list[Mapping[str, Any]] = []
            for index, segment_descriptors in enumerate(ordered_segments):
                remaining = chunk_count - len(descriptors)
                take = min(len(segment_descriptors), remaining)
                if index < len(ordered_segments) - 1:
                    take = len(segment_descriptors)
                descriptors.extend(segment_descriptors[:take])
            if len(descriptors) != chunk_count:
                raise CacheFormatError("prefix descriptor chain length differs")
            manifest = {
                "format_abi": FORMAT_ABI,
                "identity": identity.to_wire(),
                "context_digest": context_digest,
                "committed_tokens": committed_tokens,
                "chunks": descriptors,
            }
            chunks = _validate_manifest_metadata(
                manifest,
                EntryKey(identity.storage_key, context_digest),
                expected_identity=identity,
            )
            for descriptor in chunks:
                digest = descriptor["sha256"]
                encoded_bytes = descriptor["bytes"]
                logical_start = descriptor["logical_start"]
                logical_end = descriptor["logical_end"]
                chunk_path = self.root / "chunks" / f"{digest}.spcc"
                if verify_chunks:
                    encoded_chunk = chunk_path.read_bytes()
                    if (
                        len(encoded_chunk) != encoded_bytes
                        or _sha256(encoded_chunk) != digest
                    ):
                        raise CacheFormatError("chunk checksum mismatch")
                    chunk = _decode_chunk(encoded_chunk, verify_record_checksums=False)
                    try:
                        _require_complete_chunk(chunk, identity.required_records)
                    except IncompleteEntry as error:
                        raise CacheFormatError(str(error)) from error
                    if (
                        chunk.logical_start != logical_start
                        or chunk.logical_end != logical_end
                    ):
                        raise CacheFormatError("chunk range disagrees with descriptor")
                elif verify_chunk_metadata:
                    if chunk_path.stat().st_size != encoded_bytes:
                        raise CacheFormatError(
                            "chunk file size disagrees with descriptor"
                        )
            return LookupResult(
                True,
                "hit",
                manifest_digest=_sha256(alias_encoded),
                _manifest=manifest,
                root_kind="prefix_alias",
            )
        except _IncompatibleManifestError:
            return LookupResult(False, "incompatible")
        except FileNotFoundError:
            return LookupResult(False, "absent")
        except (CacheFormatError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return LookupResult(False, "corrupt")

    def invalidate(
        self,
        identity: CacheIdentity,
        context_digest: str,
        *,
        verify_chunk_payloads: bool = True,
    ) -> bool:
        """Remove a manifest so a damaged entry can be republished.

        Chunks whose bytes no longer match their content address are also
        removed: because publication is content-addressed and idempotent,
        a corrupt file sitting at the correct-hash path would otherwise
        make every future publish of that content raise CommitConflict,
        and the entry could never repair itself. Chunks that still verify
        are left in place - they are valid, shared, and reusable.

        Metadata-only callers may set verify_chunk_payloads=False. That mode
        removes only the manifest: an unverified descriptor is never
        sufficient authority to delete a content-addressed chunk that may be
        shared by another healthy manifest.
        """
        manifest_path = self._manifest_path(identity, context_digest)
        is_prefix_alias = False
        try:
            _validate_digest(context_digest, "context_digest")
            raw = manifest_path.read_bytes()
        except FileNotFoundError:
            manifest_path = self._prefix_alias_path(identity, context_digest)
            is_prefix_alias = True
            try:
                raw = manifest_path.read_bytes()
            except (OSError, ValueError):
                return False
        except (OSError, ValueError):
            return False
        descriptors: Any = []
        if not is_prefix_alias:
            try:
                manifest = json.loads(raw)
                schema_name = manifest.get("schema")
                if (
                    schema_name == _TAIL_MANIFEST_SCHEMA
                    or schema_name in _PAGE_DELTA_MANIFEST_SCHEMAS
                    or schema_name == _PAGE_SNAPSHOT_MANIFEST_SCHEMA
                ):
                    identity_wire = dict(manifest["identity"])
                    if "record_schema" in identity_wire:
                        identity_wire["record_schema"] = tuple(
                            identity_wire["record_schema"]
                        )
                    identity = CacheIdentity(**identity_wire)
                    if schema_name == _TAIL_MANIFEST_SCHEMA:
                        resolved, _segments = self._resolve_tail_manifest(
                            manifest,
                            identity=identity,
                            context_digest=context_digest,
                        )
                        descriptors = resolved["chunks"]
                    elif schema_name in _PAGE_DELTA_MANIFEST_SCHEMAS:
                        descriptors = self._page_graph_descriptors(
                            manifest,
                            identity=identity,
                            context_digest=context_digest,
                        )
                    else:
                        descriptors = _validate_page_snapshot_root(
                            manifest,
                            identity=identity,
                            context_digest=context_digest,
                        )
                else:
                    descriptors = manifest.get("chunks", [])
            except (
                CacheFormatError,
                json.JSONDecodeError,
                AttributeError,
                KeyError,
                TypeError,
                ValueError,
            ):
                descriptors = []
        for descriptor in descriptors:
            if not isinstance(descriptor, dict):
                continue
            digest = descriptor.get("sha256")
            if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
                continue
            chunk_path = self.root / "chunks" / f"{digest}.spcc"
            if not verify_chunk_payloads:
                continue
            try:
                healthy = _payload_matches_descriptor(descriptor, chunk_path.read_bytes())
            except (KeyError, OSError, TypeError, ValueError):
                healthy = False
            if not healthy:
                try:
                    chunk_path.unlink()
                except OSError:
                    pass
        try:
            manifest_path.unlink()
            return True
        except (OSError, ValueError):
            return False

    def restore(self, lookup: LookupResult) -> tuple[ContextChunk, ...] | None:
        if not lookup.is_hit or lookup._manifest is None:
            raise ValueError("cannot restore a cache miss")
        if lookup.root_kind in ("page_delta", "page_snapshot"):
            raise ValueError("page restore requires restore_page_snapshot")
        required = _required_records_for_identity_wire(
            lookup._manifest.get("identity", {})
        )

        def _restore_one(descriptor: Any) -> ContextChunk:
            if not isinstance(descriptor, dict):
                raise CacheFormatError("chunk descriptor is not an object")
            _strict_keys(
                descriptor,
                {"sha256", "bytes", "logical_start", "logical_end"},
                "chunk descriptor",
            )
            digest = descriptor["sha256"]
            _validate_digest(digest, "chunk sha256")
            encoded = (self.root / "chunks" / f"{digest}.spcc").read_bytes()
            if len(encoded) != descriptor["bytes"] or _sha256(encoded) != digest:
                raise CacheFormatError("chunk checksum mismatch")
            # The outer descriptor digest above already covers the complete
            # encoded chunk. Avoid hashing the same payload a second time.
            chunk = _decode_chunk(encoded, verify_record_checksums=False)
            _require_complete_chunk(chunk, required)
            if (
                chunk.logical_start != descriptor["logical_start"]
                or chunk.logical_end != descriptor["logical_end"]
            ):
                raise CacheFormatError("chunk range disagrees with descriptor")
            return chunk

        try:
            descriptors = list(lookup._manifest["chunks"])
            with ThreadPoolExecutor(max_workers=8) as pool:
                result = list(pool.map(_restore_one, descriptors))
            return tuple(result)
        except (
            CacheFormatError,
            OSError,
            TypeError,
            ValueError,
        ):
            return None
