"""Model-free storage ABI probe for persistent GLM-5.2 context state.

This module deliberately treats tensor records as opaque bytes.  It proves the
identity, completeness, atomic-publication, and corruption semantics before a
vLLM adapter is allowed to supply real CKV or MTP buffers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType
from typing import Any, Mapping, Sequence


FORMAT_ABI = 1
_CHUNK_MAGIC = b"SPCKV001"
_CHUNK_PREFIX = struct.Struct("<8sII")
_DIGEST = re.compile(r"[0-9a-f]{64}")


class StateRecord(str, Enum):
    TARGET_CKV = "target_ckv"
    SPARSE_INDEXER = "sparse_indexer"
    MTP_DRAFT_KV = "mtp_draft_kv"
    BOUNDARY_HIDDEN = "boundary_hidden"
    LOGICAL_POSITIONS = "logical_positions"


_REQUIRED_RECORDS = frozenset(StateRecord)


class CacheFormatError(ValueError):
    """Internal format failure that public lookup converts to a clean miss."""


class CommitConflict(RuntimeError):
    """A different immutable object already owns the cache key."""


class IncompleteEntry(ValueError):
    """Required target, indexer, draft, or boundary state is absent."""


@dataclass(frozen=True)
class CacheIdentity:
    target_checkpoint: str
    draft_checkpoint: str
    quantization_layout: str
    rope_layout: str
    tp_degree: int
    dcp_degree: int
    chunk_tokens: int = 256
    # DCP shard ownership: entries written by one rank must never restore
    # into another. -1 means "not sharded" (DCP1 whole-context entries).
    dcp_shard_rank: int = -1
    # "persisted": every chunk carries a boundary_hidden record (original
    # tracer contract). "live_forward": boundary hidden state is not
    # persisted; the first post-restore forward regenerates it.
    boundary_hidden_policy: str = "persisted"
    # "separate": chunks carry a distinct mtp_draft_kv record.
    # "colocated_target": the runtime registers drafter KV layers in the
    # same cache pool without a distinguishing name, so draft state is
    # persisted and restored inside target_ckv records and no separate
    # mtp_draft_kv record exists.
    draft_kv_policy: str = "separate"

    def __post_init__(self) -> None:
        for field in ("target_checkpoint", "draft_checkpoint"):
            value = getattr(self, field)
            if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
                raise ValueError(f"{field} must be a 64-character lowercase SHA-256")
        for field in ("quantization_layout", "rope_layout"):
            if not getattr(self, field):
                raise ValueError(f"{field} must be non-empty")
        for field in ("tp_degree", "dcp_degree", "chunk_tokens"):
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

    @property
    def required_records(self) -> frozenset["StateRecord"]:
        dropped: set[StateRecord] = set()
        if self.boundary_hidden_policy == "live_forward":
            dropped.add(StateRecord.BOUNDARY_HIDDEN)
        if self.draft_kv_policy == "colocated_target":
            dropped.add(StateRecord.MTP_DRAFT_KV)
        return frozenset(_REQUIRED_RECORDS - dropped)

    def to_wire(self) -> dict[str, Any]:
        return {
            "target_checkpoint": self.target_checkpoint,
            "draft_checkpoint": self.draft_checkpoint,
            "quantization_layout": self.quantization_layout,
            "rope_layout": self.rope_layout,
            "tp_degree": self.tp_degree,
            "dcp_degree": self.dcp_degree,
            "chunk_tokens": self.chunk_tokens,
            "dcp_shard_rank": self.dcp_shard_rank,
            "boundary_hidden_policy": self.boundary_hidden_policy,
            "draft_kv_policy": self.draft_kv_policy,
        }

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


@dataclass(frozen=True)
class ChunkReceipt:
    chunk_digest: str
    encoded_bytes: int
    logical_start: int
    logical_end: int


@dataclass(frozen=True)
class LookupResult:
    is_hit: bool
    reason: str
    manifest_digest: str = ""
    _manifest: Mapping[str, Any] | None = None


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _sha256(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _publish_immutable(path: Path, payload: bytes) -> None:
    """Durably publish complete bytes once without an overwrite race."""

    _ensure_durable_directory(path.parent)
    temporary = path.with_name(f".{path.name}.writing-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            try:
                existing = path.read_bytes()
            except OSError as error:
                raise CommitConflict(
                    f"cannot verify existing immutable object {path}"
                ) from error
            if existing != payload:
                raise CommitConflict(
                    f"different immutable object already committed at {path}"
                )
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
    """Durably publish one directory-local immutable-object macro-batch.

    Every object's data reaches stable storage before any descriptor can be
    appended to a transaction. File-data barriers run concurrently; all hard
    links and temporary-name removals share one final directory barrier.
    """

    if not objects:
        return
    parent = objects[0][0].parent
    if any(path.parent != parent for path, _payload in objects):
        raise ValueError("immutable macro-batch must share one directory")
    _ensure_durable_directory(parent)
    staged = [
        (
            path,
            payload,
            path.with_name(f".{path.name}.writing-{uuid.uuid4().hex}"),
        )
        for path, payload in objects
    ]

    def stage(item: tuple[Path, bytes, Path]) -> None:
        _path, payload, temporary = item
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    try:
        worker_count = min(8, len(staged))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            tuple(pool.map(stage, staged))
        for path, payload, temporary in staged:
            try:
                os.link(temporary, path)
            except FileExistsError:
                try:
                    existing = path.read_bytes()
                except OSError as error:
                    raise CommitConflict(
                        f"cannot verify existing immutable object {path}"
                    ) from error
                if existing != payload:
                    raise CommitConflict(
                        f"different immutable object already committed at {path}"
                    )
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
        self._receipt: CommitReceipt | None = None
        self._lock = threading.RLock()

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
                    if (
                        existing["sha256"] != chunk_digest
                        or existing["bytes"] != len(encoded)
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

            _publish_immutable_batch(pending_objects)
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
                    "streaming transaction does not cover the declared context span"
                )
            manifest = {
                "format_abi": FORMAT_ABI,
                "identity": self._identity.to_wire(),
                "context_digest": self._context_digest,
                "committed_tokens": self._expected_start,
                "chunks": list(self._descriptors),
            }
            encoded_manifest = _canonical_json(manifest)
            receipt = CommitReceipt(
                manifest_digest=_sha256(encoded_manifest),
                committed_tokens=self._expected_start,
            )
            _publish_immutable(
                self._store._manifest_path(
                    self._identity,
                    self._context_digest,
                ),
                encoded_manifest,
            )
            self._receipt = receipt
            self._state = "committed"
            return receipt

    def abort(self) -> None:
        """Make the transaction terminal without publishing a manifest."""

        with self._lock:
            if self._state == "committed":
                raise RuntimeError("context transaction is committed")
            if self._state == "aborted":
                return
            self._state = "aborted"
            self._descriptors.clear()


class ManifestStore:
    """Atomic local-NVMe manifest publisher and fail-closed reader."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _manifest_path(self, identity: CacheIdentity, context_digest: str) -> Path:
        return self.root / "manifests" / identity.storage_key / f"{context_digest}.json"

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
    ) -> CommitReceipt:
        transaction = self.begin(
            identity=identity,
            context_digest=context_digest,
        )
        try:
            for chunk in chunks:
                transaction.append_chunk(chunk)
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
    ) -> LookupResult:
        """With verify_chunks=False only the manifest itself is validated
        (existence, identity, descriptor structure). Setting
        verify_chunk_metadata also requires each referenced chunk file to
        exist at its declared size, but still does not read payload bytes.
        Restore always re-reads and re-hashes every chunk, so a probe-mode hit
        can still degrade to a clean miss at restore."""
        try:
            _validate_digest(context_digest, "context_digest")
            encoded = self._manifest_path(identity, context_digest).read_bytes()
            manifest = json.loads(encoded)
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
            expected_identity = identity.to_wire()
            if (
                type(manifest["format_abi"]) is not int
                or manifest["format_abi"] != FORMAT_ABI
                or manifest["identity"] != expected_identity
                or _canonical_json(manifest["identity"])
                != _canonical_json(expected_identity)
                or manifest["context_digest"] != context_digest
            ):
                return LookupResult(False, "incompatible")
            chunks = manifest["chunks"]
            if not isinstance(chunks, list) or not chunks:
                raise CacheFormatError("manifest has no chunks")
            committed_tokens = manifest["committed_tokens"]
            if type(committed_tokens) is not int or committed_tokens <= 0:
                raise CacheFormatError("committed_tokens must be a positive integer")
            expected_start = 0
            for chunk_index, descriptor in enumerate(chunks):
                if not isinstance(descriptor, dict):
                    raise CacheFormatError("chunk descriptor is not an object")
                _strict_keys(
                    descriptor,
                    {
                        "sha256",
                        "bytes",
                        "logical_start",
                        "logical_end",
                    },
                    "chunk descriptor",
                )
                digest = descriptor["sha256"]
                _validate_digest(digest, "chunk sha256")
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
                    chunk_index != len(chunks) - 1
                    and token_count != identity.chunk_tokens
                ):
                    raise CacheFormatError(
                        "chunk range disagrees with identity geometry"
                    )
                if verify_chunks:
                    encoded_chunk = (
                        self.root / "chunks" / f"{digest}.spcc"
                    ).read_bytes()
                    if (
                        len(encoded_chunk) != encoded_bytes
                        or _sha256(encoded_chunk) != digest
                    ):
                        raise CacheFormatError("chunk checksum mismatch")
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
                expected_start = logical_end
            if expected_start != committed_tokens:
                raise CacheFormatError("committed token count mismatch")
            return LookupResult(
                True,
                "hit",
                manifest_digest=_sha256(encoded),
                _manifest=manifest,
            )
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
        try:
            _validate_digest(context_digest, "context_digest")
            raw = manifest_path.read_bytes()
        except FileNotFoundError:
            return False
        except (OSError, ValueError):
            return False
        try:
            manifest = json.loads(raw)
            descriptors = manifest.get("chunks", [])
        except (json.JSONDecodeError, AttributeError):
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
                healthy = _sha256(chunk_path.read_bytes()) == digest
            except OSError:
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
