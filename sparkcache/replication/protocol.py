"""Wire codec for bounded SparkCache buddy replication.

The codec is deliberately transport-neutral. A caller may carry these frames
over TCP first and replace that carrier later without changing transaction
semantics or the on-disk SparkCache format.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Mapping, Sequence


MAGIC = b"SPRP"
VERSION = 1
PREFIX = struct.Struct("!4sBBHII")
MAX_HEADER_BYTES = 1 << 20
MAX_PAYLOAD_BYTES = 16 << 20
MAX_FRAME_BYTES = PREFIX.size + MAX_HEADER_BYTES + MAX_PAYLOAD_BYTES
_DIGEST = re.compile(r"[0-9a-f]{64}")
_TRANSACTION_ID = re.compile(r"[0-9a-f]{32}")


class ProtocolError(ValueError):
    """Malformed or semantically invalid replication input."""


class FrameType(IntEnum):
    BEGIN = 1
    PUT_CHUNK = 2
    COMMIT = 3
    ABORT = 4
    ACK = 5
    CREDIT = 6


_FIELDS: dict[FrameType, frozenset[str]] = {
    FrameType.BEGIN: frozenset(
        {
            "transaction_id",
            "generation",
            "sequence",
            "context_digest",
            "identity_digest",
        }
    ),
    FrameType.PUT_CHUNK: frozenset(
        {
            "transaction_id",
            "generation",
            "sequence",
            "chunk_index",
            "chunk_digest",
        }
    ),
    FrameType.COMMIT: frozenset(
        {
            "transaction_id",
            "generation",
            "sequence",
            "manifest_digest",
            "chunk_count",
        }
    ),
    FrameType.ABORT: frozenset(
        {
            "transaction_id",
            "generation",
            "sequence",
            "reason",
        }
    ),
    FrameType.ACK: frozenset(
        {
            "transaction_id",
            "generation",
            "sequence",
            "ack_sequence",
            "status",
        }
    ),
    FrameType.CREDIT: frozenset(
        {
            "transaction_id",
            "generation",
            "sequence",
            "available_bytes",
            "available_frames",
        }
    ),
}
_EMPTY_PAYLOAD = {
    FrameType.BEGIN,
    FrameType.ABORT,
    FrameType.ACK,
    FrameType.CREDIT,
}
_ACK_STATUSES = {
    "ok",
    "duplicate",
    "committed",
    "aborted",
    "rejected",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def sha256(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _validate_header(frame_type: FrameType, header: Mapping[str, Any]) -> None:
    expected = _FIELDS[frame_type]
    if set(header) != expected:
        raise ProtocolError(
            f"{frame_type.name} header fields differ: "
            f"missing={sorted(expected - set(header))}, "
            f"unexpected={sorted(set(header) - expected)}"
        )
    transaction_id = header["transaction_id"]
    if (
        not isinstance(transaction_id, str)
        or _TRANSACTION_ID.fullmatch(transaction_id) is None
    ):
        raise ProtocolError("transaction_id must be 32 lowercase hex characters")
    for field in ("generation", "sequence"):
        if not _is_nonnegative_int(header[field]):
            raise ProtocolError(f"{field} must be a nonnegative integer")

    if frame_type is FrameType.BEGIN:
        for field in ("context_digest", "identity_digest"):
            value = header[field]
            if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
                raise ProtocolError(f"{field} must be a lowercase SHA-256 digest")
    elif frame_type is FrameType.PUT_CHUNK:
        if not _is_nonnegative_int(header["chunk_index"]):
            raise ProtocolError("chunk_index must be a nonnegative integer")
        digest = header["chunk_digest"]
        if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
            raise ProtocolError("chunk_digest must be a lowercase SHA-256 digest")
    elif frame_type is FrameType.COMMIT:
        digest = header["manifest_digest"]
        if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
            raise ProtocolError("manifest_digest must be a lowercase SHA-256 digest")
        if not _is_nonnegative_int(header["chunk_count"]) or not header["chunk_count"]:
            raise ProtocolError("chunk_count must be a positive integer")
    elif frame_type is FrameType.ABORT:
        reason = header["reason"]
        if not isinstance(reason, str) or len(reason.encode("utf-8")) > 256:
            raise ProtocolError("reason must be a UTF-8 string of at most 256 bytes")
    elif frame_type is FrameType.ACK:
        if not _is_nonnegative_int(header["ack_sequence"]):
            raise ProtocolError("ack_sequence must be a nonnegative integer")
        if header["status"] not in _ACK_STATUSES:
            raise ProtocolError("invalid ACK status")
    elif frame_type is FrameType.CREDIT:
        for field in ("available_bytes", "available_frames"):
            if not _is_nonnegative_int(header[field]):
                raise ProtocolError(f"{field} must be a nonnegative integer")


@dataclass(frozen=True)
class Frame:
    frame_type: FrameType
    header: Mapping[str, Any]
    payload: bytes = b""

    def __post_init__(self) -> None:
        try:
            frame_type = FrameType(self.frame_type)
        except ValueError as error:
            raise ProtocolError("unknown frame type") from error
        if not isinstance(self.payload, bytes):
            raise TypeError("frame payload must be bytes")
        normalized = dict(self.header)
        _validate_header(frame_type, normalized)
        if frame_type in _EMPTY_PAYLOAD and self.payload:
            raise ProtocolError(f"{frame_type.name} must not carry a payload")
        if frame_type in (FrameType.PUT_CHUNK, FrameType.COMMIT) and not self.payload:
            raise ProtocolError(f"{frame_type.name} requires a payload")
        if len(self.payload) > MAX_PAYLOAD_BYTES:
            raise ProtocolError("frame payload exceeds protocol maximum")
        object.__setattr__(self, "frame_type", frame_type)
        object.__setattr__(self, "header", MappingProxyType(normalized))

    def encode(self) -> bytes:
        encoded_header = _canonical_json(dict(self.header))
        if len(encoded_header) > MAX_HEADER_BYTES:
            raise ProtocolError("frame header exceeds protocol maximum")
        prefix = PREFIX.pack(
            MAGIC,
            VERSION,
            int(self.frame_type),
            0,
            len(encoded_header),
            len(self.payload),
        )
        return b"".join((prefix, encoded_header, self.payload))


class FrameStreamDecoder:
    """Fragment-tolerant decoder with a strict total-buffer bound."""

    def __init__(
        self,
        *,
        max_frame_bytes: int = MAX_FRAME_BYTES,
        max_buffer_bytes: int | None = None,
    ) -> None:
        if max_frame_bytes < PREFIX.size:
            raise ValueError("max_frame_bytes is smaller than the frame prefix")
        self.max_frame_bytes = max_frame_bytes
        self.max_buffer_bytes = (
            max_buffer_bytes if max_buffer_bytes is not None else max_frame_bytes * 2
        )
        if self.max_buffer_bytes < self.max_frame_bytes:
            raise ValueError("max_buffer_bytes must cover one maximum frame")
        self._buffer = bytearray()

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    def feed(self, data: bytes | bytearray | memoryview) -> tuple[Frame, ...]:
        if len(self._buffer) + len(data) > self.max_buffer_bytes:
            raise ProtocolError("replication receive buffer limit exceeded")
        self._buffer.extend(data)
        frames: list[Frame] = []
        while len(self._buffer) >= PREFIX.size:
            magic, version, raw_type, flags, header_bytes, payload_bytes = (
                PREFIX.unpack_from(self._buffer)
            )
            if magic != MAGIC or version != VERSION or flags != 0:
                raise ProtocolError("invalid replication frame prefix")
            if header_bytes > MAX_HEADER_BYTES or payload_bytes > MAX_PAYLOAD_BYTES:
                raise ProtocolError("declared replication frame exceeds protocol maximum")
            total = PREFIX.size + header_bytes + payload_bytes
            if total > self.max_frame_bytes:
                raise ProtocolError("declared replication frame exceeds configured maximum")
            if len(self._buffer) < total:
                break
            encoded_header = bytes(self._buffer[PREFIX.size : PREFIX.size + header_bytes])
            payload = bytes(self._buffer[PREFIX.size + header_bytes : total])
            del self._buffer[:total]
            try:
                decoded = json.loads(encoded_header)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ProtocolError("invalid frame header JSON") from error
            if not isinstance(decoded, dict):
                raise ProtocolError("frame header must be a JSON object")
            if encoded_header != _canonical_json(decoded):
                raise ProtocolError("frame header is not canonical JSON")
            try:
                frame_type = FrameType(raw_type)
            except ValueError as error:
                raise ProtocolError("unknown frame type") from error
            frames.append(Frame(frame_type, decoded, payload))
        return tuple(frames)


def encode_commit_record(
    *,
    transaction_id: str,
    generation: int,
    context_digest: str,
    chunk_digests: Sequence[str],
) -> bytes:
    if not chunk_digests:
        raise ProtocolError("commit record requires at least one chunk")
    record = {
        "transaction_id": transaction_id,
        "generation": generation,
        "context_digest": context_digest,
        "chunk_digests": list(chunk_digests),
    }
    # Reuse BEGIN validation for all common identity fields.
    _validate_header(
        FrameType.BEGIN,
        {
            "transaction_id": transaction_id,
            "generation": generation,
            "sequence": 0,
            "context_digest": context_digest,
            "identity_digest": "0" * 64,
        },
    )
    for digest in chunk_digests:
        if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
            raise ProtocolError("commit chunk digest must be lowercase SHA-256")
    return _canonical_json(record)


def decode_commit_record(payload: bytes) -> Mapping[str, Any]:
    try:
        record = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError("invalid commit record JSON") from error
    if not isinstance(record, dict) or set(record) != {
        "transaction_id",
        "generation",
        "context_digest",
        "chunk_digests",
    }:
        raise ProtocolError("invalid commit record fields")
    chunks = record["chunk_digests"]
    if not isinstance(chunks, list) or not chunks:
        raise ProtocolError("commit record requires chunk digests")
    # Validation is centralized in the encoder and is deterministic.
    expected = encode_commit_record(
        transaction_id=record["transaction_id"],
        generation=record["generation"],
        context_digest=record["context_digest"],
        chunk_digests=chunks,
    )
    if expected != payload:
        raise ProtocolError("commit record is not canonical")
    return MappingProxyType(record)
