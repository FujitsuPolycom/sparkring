"""Pure codec for the SparkRing persistent context cache.

DCP shard math and byte-record packing shared by the vLLM connector and its
tests. This module must not import vllm or torch: everything operates on
plain integers and bytes so the contract is testable model-free.

Layout facts this codec encodes (v40 runtime, attested):
- DCP token interleave size 1: global position ``p`` is owned by DCP rank
  ``p % dcp_degree``.
- Dense local prefix: owned position ``p`` sits at local ordinal
  ``p // dcp_degree``; the local KV slot is
  ``block_ids[ordinal // block_size] * block_size + ordinal % block_size``.
- Record byte order within a chunk is the sorted layer-name order captured
  at registration time; each layer contributes ``rows_per_chunk`` rows of
  its fixed per-token record width.
"""

from __future__ import annotations

import hashlib
import struct
import sys
from array import array
from dataclasses import dataclass
from typing import Mapping, Sequence

CHUNK_TOKENS = 256
_POSITION_STRUCT = struct.Struct("<I")


class CodecError(ValueError):
    """Deterministic packing/unpacking failure. Callers convert this to a
    fail-closed cache miss; it must never crash a serving process."""


def _pack_u32(values: Sequence[int], label: str) -> bytes:
    """Pack unsigned 32-bit integers in the frozen little-endian ABI.

    ``array`` performs the conversion in C, avoiding one Python ``bytes``
    allocation per token/position on large contexts.
    """
    try:
        packed = array("I", values)
    except (OverflowError, TypeError, ValueError) as error:
        raise CodecError(f"{label} must contain unsigned 32-bit integers") from error
    if packed.itemsize != _POSITION_STRUCT.size:
        raise CodecError("platform unsigned-int width does not match cache ABI")
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tobytes()


def owned_positions(
    span_tokens: int, dcp_degree: int, dcp_rank: int
) -> tuple[int, ...]:
    """Global positions in [0, span) owned by this rank (interleave 1)."""
    if span_tokens <= 0:
        raise CodecError("span must be positive")
    if not 0 <= dcp_rank < dcp_degree:
        raise CodecError("dcp_rank out of range")
    return tuple(range(dcp_rank, span_tokens, dcp_degree))


def local_slots_for_positions(
    positions: Sequence[int],
    block_ids: Sequence[int],
    block_size: int,
    dcp_degree: int,
) -> tuple[int, ...]:
    """Map owned global positions to this rank's local KV slots."""
    if block_size <= 0:
        raise CodecError("block_size must be positive")
    capacity = len(block_ids) * block_size
    slots = []
    for p in positions:
        ordinal = p // dcp_degree
        if ordinal >= capacity:
            raise CodecError(
                f"position {p} ordinal {ordinal} exceeds allocated capacity {capacity}"
            )
        block = block_ids[ordinal // block_size]
        slots.append(block * block_size + ordinal % block_size)
    return tuple(slots)


def context_digest(token_ids: Sequence[int], identity_salt: str) -> str:
    """Content digest for a block-aligned prompt span. The identity salt
    binds the digest to the cache identity so distinct configurations can
    never alias to the same key even inside one store root."""
    digest = hashlib.sha256()
    digest.update(identity_salt.encode("ascii"))
    digest.update(b"\x00")
    digest.update(_pack_u32(token_ids, "token_ids"))
    return digest.hexdigest()


def chunk_count(span_tokens: int) -> int:
    if span_tokens <= 0 or span_tokens % CHUNK_TOKENS:
        raise CodecError("span must be a positive multiple of CHUNK_TOKENS")
    return span_tokens // CHUNK_TOKENS


@dataclass(frozen=True)
class LayerPlan:
    """Byte layout of one registered cache layer inside a record kind."""

    name: str
    record_kind: str
    bytes_per_token: int

    def __post_init__(self) -> None:
        if self.bytes_per_token <= 0:
            raise CodecError(f"layer {self.name} has no per-token bytes")


def classify_layer(name: str) -> str:
    """Map a registered kv-cache layer name onto a persistent record kind."""
    lowered = name.lower()
    if "indexer" in lowered:
        return "sparse_indexer"
    if "draft" in lowered or "mtp" in lowered or "spec" in lowered:
        return "mtp_draft_kv"
    return "target_ckv"


def build_layer_plans(
    layer_bytes_per_token: Mapping[str, int],
    *,
    allow_missing: frozenset[str] = frozenset(),
) -> tuple[LayerPlan, ...]:
    """Stable, sorted plan over every registered layer. Raises if any
    required record kind has no contributing layer, because storing a chunk
    that silently omits state is exactly the failure mode this cache exists
    to prevent. A kind may be exempted only through an explicit declared
    policy (``allow_missing``), e.g. draft KV colocated in the target pool
    when the runtime registers drafter layers without a naming marker."""
    plans = tuple(
        LayerPlan(name, classify_layer(name), layer_bytes_per_token[name])
        for name in sorted(layer_bytes_per_token)
    )
    kinds = {plan.record_kind for plan in plans}
    missing = (
        {"target_ckv", "sparse_indexer", "mtp_draft_kv"} - kinds - set(allow_missing)
    )
    if missing:
        raise CodecError(
            "no registered cache layer for record kinds: " + ", ".join(sorted(missing))
        )
    return plans


def pack_positions(positions: Sequence[int]) -> bytes:
    if not positions:
        raise CodecError("a chunk shard must own at least one position")
    return _pack_u32(positions, "positions")


def unpack_positions(payload: bytes) -> tuple[int, ...]:
    if not payload or len(payload) % _POSITION_STRUCT.size:
        raise CodecError("malformed positions payload")
    unpacked = array("I")
    unpacked.frombytes(payload)
    if unpacked.itemsize != _POSITION_STRUCT.size:
        raise CodecError("platform unsigned-int width does not match cache ABI")
    if sys.byteorder != "little":
        unpacked.byteswap()
    return tuple(unpacked)


def pack_record(
    plans: Sequence[LayerPlan],
    record_kind: str,
    layer_rows: Mapping[str, bytes],
    rows: int,
) -> bytes:
    """Concatenate per-layer row bytes for one record kind, sorted order.

    ``layer_rows`` maps layer name to the raw bytes of exactly ``rows``
    per-token records for that layer.
    """
    parts = []
    for plan in plans:
        if plan.record_kind != record_kind:
            continue
        payload = layer_rows.get(plan.name)
        if payload is None:
            raise CodecError(f"missing rows for layer {plan.name}")
        expected = plan.bytes_per_token * rows
        if len(payload) != expected:
            raise CodecError(
                f"layer {plan.name} rows are {len(payload)} bytes, expected {expected}"
            )
        parts.append(payload)
    if not parts:
        raise CodecError(f"no layers contribute to {record_kind}")
    return b"".join(parts)


def unpack_record(
    plans: Sequence[LayerPlan],
    record_kind: str,
    payload: bytes,
    rows: int,
) -> dict[str, bytes]:
    """Split a packed record back into per-layer row bytes, verifying the
    total length exactly; trailing bytes are a hard error."""
    out: dict[str, bytes] = {}
    offset = 0
    for plan in plans:
        if plan.record_kind != record_kind:
            continue
        size = plan.bytes_per_token * rows
        if offset + size > len(payload):
            raise CodecError(f"record {record_kind} truncated at layer {plan.name}")
        out[plan.name] = payload[offset : offset + size]
        offset += size
    if not out:
        raise CodecError(f"no layers contribute to {record_kind}")
    if offset != len(payload):
        raise CodecError(
            f"record {record_kind} carries {len(payload) - offset}"
            " unclaimed trailing bytes"
        )
    return out
