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
        for field in (
            "target_checkpoint",
            "draft_checkpoint",
            "quantization_layout",
            "rope_layout",
        ):
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
            raise ValueError(
                "draft_kv_policy must be 'separate' or 'colocated_target'"
            )
        if not -1 <= self.dcp_shard_rank < self.dcp_degree:
            raise ValueError(
                "dcp_shard_rank must be -1 or in [0, dcp_degree)"
            )

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
        raise IncompleteEntry(
            f"incomplete speculative cache chunk: missing {names}"
        )


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
class LookupResult:
    is_hit: bool
    reason: str
    manifest_digest: str = ""
    _manifest: Mapping[str, Any] | None = None


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _publish_immutable(path: Path, payload: bytes) -> None:
    """Publish complete bytes once without an overwrite race."""

    path.parent.mkdir(parents=True, exist_ok=True)
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


def _validate_digest(value: str, field: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _encode_chunk(chunk: ContextChunk) -> bytes:
    payload = bytearray()
    records: list[dict[str, Any]] = []
    for kind in sorted(chunk.records, key=lambda item: item.value):
        value = chunk.records[kind]
        offset = len(payload)
        payload.extend(value)
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
    return _CHUNK_PREFIX.pack(_CHUNK_MAGIC, FORMAT_ABI, len(header)) + header + payload


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise CacheFormatError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _decode_chunk(encoded: bytes) -> ContextChunk:
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
    if header["format_abi"] != FORMAT_ABI or not isinstance(
        header["records"], list
    ):
        raise CacheFormatError("unsupported chunk header")
    raw_payload = encoded[header_end:]
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
        if (
            kind in records
            or offset < 0
            or length < 0
            or offset != expected_offset
        ):
            raise CacheFormatError("duplicate or invalid record descriptor")
        value = raw_payload[offset : offset + length]
        if len(value) != length or _sha256(value) != item["sha256"]:
            raise CacheFormatError("record payload checksum mismatch")
        records[kind] = bytes(value)
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


class ManifestStore:
    """Atomic local-NVMe manifest publisher and fail-closed reader."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _manifest_path(
        self, identity: CacheIdentity, context_digest: str
    ) -> Path:
        return (
            self.root
            / "manifests"
            / identity.storage_key
            / f"{context_digest}.json"
        )

    def commit(
        self,
        *,
        identity: CacheIdentity,
        context_digest: str,
        chunks: Sequence[ContextChunk],
    ) -> CommitReceipt:
        _validate_digest(context_digest, "context_digest")
        if not chunks:
            raise ValueError("at least one context chunk is required")
        expected_start = 0
        descriptors: list[dict[str, Any]] = []
        for chunk_index, chunk in enumerate(chunks):
            if chunk.logical_start != expected_start:
                raise ValueError("chunk logical ranges must be contiguous from zero")
            token_count = chunk.logical_end - chunk.logical_start
            if token_count > identity.chunk_tokens:
                raise ValueError("chunk exceeds identity chunk_tokens")
            if (
                chunk_index != len(chunks) - 1
                and token_count != identity.chunk_tokens
            ):
                raise ValueError("only the final context chunk may be partial")
            _require_complete_chunk(chunk, identity.required_records)
            encoded = _encode_chunk(chunk)
            chunk_digest = _sha256(encoded)
            chunk_path = self.root / "chunks" / f"{chunk_digest}.spcc"
            _publish_immutable(chunk_path, encoded)
            descriptors.append(
                {
                    "sha256": chunk_digest,
                    "bytes": len(encoded),
                    "logical_start": chunk.logical_start,
                    "logical_end": chunk.logical_end,
                }
            )
            expected_start = chunk.logical_end
        manifest = {
            "format_abi": FORMAT_ABI,
            "identity": identity.to_wire(),
            "context_digest": context_digest,
            "committed_tokens": expected_start,
            "chunks": descriptors,
        }
        encoded_manifest = _canonical_json(manifest)
        manifest_digest = _sha256(encoded_manifest)
        _publish_immutable(
            self._manifest_path(identity, context_digest), encoded_manifest
        )
        return CommitReceipt(
            manifest_digest=manifest_digest, committed_tokens=expected_start
        )

    def lookup(
        self,
        identity: CacheIdentity,
        context_digest: str,
        *,
        verify_chunks: bool = True,
    ) -> LookupResult:
        """With verify_chunks=False only the manifest itself is validated
        (existence, identity, descriptor structure); chunk payloads are not
        read. Restore always re-reads and re-hashes every chunk, so a
        probe-mode hit can still degrade to a clean miss at restore."""
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
            if (
                manifest["format_abi"] != FORMAT_ABI
                or manifest["identity"] != identity.to_wire()
                or manifest["context_digest"] != context_digest
            ):
                return LookupResult(False, "incompatible")
            chunks = manifest["chunks"]
            if not isinstance(chunks, list) or not chunks:
                raise CacheFormatError("manifest has no chunks")
            expected_start = 0
            for descriptor in chunks:
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
                if (
                    descriptor["logical_start"] != expected_start
                    or descriptor["logical_end"] <= expected_start
                ):
                    raise CacheFormatError(
                        "non-contiguous logical chunk range"
                    )
                if verify_chunks:
                    encoded_chunk = (
                        self.root / "chunks" / f"{digest}.spcc"
                    ).read_bytes()
                    if (
                        len(encoded_chunk) != descriptor["bytes"]
                        or _sha256(encoded_chunk) != digest
                    ):
                        raise CacheFormatError("chunk checksum mismatch")
                    chunk = _decode_chunk(encoded_chunk)
                    try:
                        _require_complete_chunk(
                            chunk, identity.required_records
                        )
                    except IncompleteEntry as error:
                        raise CacheFormatError(str(error)) from error
                    if (
                        chunk.logical_start != descriptor["logical_start"]
                        or chunk.logical_end != descriptor["logical_end"]
                    ):
                        raise CacheFormatError(
                            "chunk range disagrees with descriptor"
                        )
                expected_start = descriptor["logical_end"]
            if expected_start != manifest["committed_tokens"]:
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
        self, identity: CacheIdentity, context_digest: str
    ) -> bool:
        """Remove a manifest so a damaged entry can be republished.

        Chunks whose bytes no longer match their content address are also
        removed: because publication is content-addressed and idempotent,
        a corrupt file sitting at the correct-hash path would otherwise
        make every future publish of that content raise CommitConflict,
        and the entry could never repair itself. Chunks that still verify
        are left in place - they are valid, shared, and reusable.
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

    def restore(
        self, lookup: LookupResult
    ) -> tuple[ContextChunk, ...] | None:
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
            encoded = (
                self.root / "chunks" / f"{digest}.spcc"
            ).read_bytes()
            if (
                len(encoded) != descriptor["bytes"]
                or _sha256(encoded) != digest
            ):
                raise CacheFormatError("chunk checksum mismatch")
            chunk = _decode_chunk(encoded)
            _require_complete_chunk(chunk, required)
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
