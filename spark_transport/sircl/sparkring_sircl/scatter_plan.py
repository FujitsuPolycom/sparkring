"""Geometry and op plans of the scatter collectives (torch-free).

The scatter collectives of a ring session move one message of ``W`` chunks,
``W`` the group size: chunk ``j`` of every rank's input travels to rank ``j``.

- Reduce-scatter: rank ``j`` stores the sum of chunk ``j`` of every rank's
  input, added in rank order ``0 .. W-1`` in float32 and rounded once to the
  dtype. These are the bits that the one-shot and two-shot all-reduce give for
  the same elements.
- All-to-all: rank ``j`` stores the chunk that rank ``s`` sent at output chunk
  ``s``, bytes unchanged.

Input geometry (:func:`scatter_geometry`). Without an explicit chunk size the
input is ``W`` contiguous equal chunks of whole 16-byte packs. With
``chunk_bytes`` ``c`` and ``src_stride_bytes`` ``s`` (default ``c``), chunk
``j`` is the ``c`` bytes that start ``j * s`` bytes into the input; ``c`` and
``s`` are multiples of 16, ``s >= c``, and every chunk lies inside the input.
When a capacity is given, ``W * c`` must not exceed it; without one, any
size is eligible and the op plan below cuts the message into ops that fit a
slot. The all-to-all output holds chunk
``s`` at ``s * d`` bytes, with a destination stride ``d`` (default ``c``) that
is a multiple of 16 and at least ``c`` (:func:`destination_stride`). Every
rule reads sizes only, never pointers, so every rank of a group decides alike.

Op plan (:func:`scatter_plan`). A message whose chunks are larger than the
per-peer op size travels as several ops: op ``i`` carries bytes
``[i * piece, (i + 1) * piece)`` of every chunk. Each op is a strided scatter
with the same strides, so the split changes no bit of the result: every
element is still the rank-ordered sum of its own inputs. The per-peer op size
(:func:`piece_bytes`) is the chunk size, capped by the op limit per peer (an
op's ``W`` pieces fill at most one op of the session's large-message piece
size, :func:`op_peer_bytes`) and by the relay-safe per-peer size when the
session reports one: the largest per-peer op that keeps the busiest relay
hairpin queue within its share when no forward window paces the relayed lanes
(``sparkring_sircl.pieces.relay_safe_bytes``). With forward windows the native
layer already bounds the bytes in flight on every relayed lane.

Status: implemented; CPU tests in ``tests/test_scatter_plan.py``.
"""

from __future__ import annotations

import dataclasses

from .protocol import PACK_BYTES

MODES = ("reduce", "copy")


class ScatterError(ValueError):
    """A scatter geometry or output that the collective cannot carry."""


@dataclasses.dataclass(frozen=True)
class ScatterGeometry:
    """Chunk size and source stride, in bytes, of one rank's scatter input."""

    chunk_bytes: int
    src_stride_bytes: int

    def message_bytes(self, world: int) -> int:
        """Bytes of the whole message (``W`` chunks) as one op carries it."""
        return int(world) * self.chunk_bytes

    def chunk_offset(self, chunk: int) -> int:
        """Byte offset of chunk ``chunk`` in the input."""
        return int(chunk) * self.src_stride_bytes


def scatter_geometry(total_bytes: int, world: int, capacity: int | None = None, chunk_bytes: int | None = None,
                     src_stride_bytes: int | None = None) -> ScatterGeometry | None:
    """The geometry of a scatter over an input of ``total_bytes``, or None when ineligible.

    With a ``capacity`` (bytes), ``W * chunk_bytes`` must fit it; with None,
    every size is eligible.
    """
    world = int(world)
    total_bytes = int(total_bytes)
    if world < 2 or total_bytes <= 0:
        return None
    if chunk_bytes is None:
        if total_bytes % (world * PACK_BYTES):
            return None
        chunk = total_bytes // world
        stride = chunk
    else:
        chunk = int(chunk_bytes)
        stride = chunk if src_stride_bytes is None else int(src_stride_bytes)
    if chunk <= 0 or chunk % PACK_BYTES or stride % PACK_BYTES or stride < chunk:
        return None
    if (world - 1) * stride + chunk > total_bytes:
        return None
    if capacity is not None and world * chunk > int(capacity):
        return None
    return ScatterGeometry(chunk, stride)


def destination_stride(chunk_bytes: int, world: int, out_bytes: int, dst_stride_bytes: int | None = None) -> int:
    """The all-to-all destination stride; raises :class:`ScatterError` when the output cannot hold the chunks."""
    stride = int(chunk_bytes) if dst_stride_bytes is None else int(dst_stride_bytes)
    if stride < chunk_bytes or stride % PACK_BYTES:
        raise ScatterError(f"dst_stride_bytes {stride} must be a multiple of {PACK_BYTES} of at least "
                           f"one chunk ({chunk_bytes} bytes)")
    if (int(world) - 1) * stride + int(chunk_bytes) > int(out_bytes):
        raise ScatterError(f"an all-to-all output of {out_bytes} bytes cannot hold {world} chunks of "
                           f"{chunk_bytes} bytes {stride} bytes apart")
    return stride


def op_peer_bytes(op_bytes: int, world: int) -> int:
    """Largest per-peer piece of an op whose ``W`` pieces fill at most ``op_bytes``."""
    piece = int(op_bytes) // int(world) // PACK_BYTES * PACK_BYTES
    if piece < PACK_BYTES:
        raise ScatterError(f"an op of {op_bytes} bytes cannot carry one {PACK_BYTES}-byte pack to each of {world} "
                           "ranks")
    return piece


def piece_bytes(chunk_bytes: int, relay_safe_bytes: int | None, op_peer_limit: int | None = None) -> int:
    """Per-peer bytes of one op: the chunk, capped by the per-peer op limit and the relay-safe size."""
    piece = int(chunk_bytes)
    for name, cap in (("per-peer op limit", op_peer_limit), ("relay-safe per-peer size", relay_safe_bytes)):
        if cap is None or piece <= cap:
            continue
        capped = int(cap) // PACK_BYTES * PACK_BYTES
        if capped < PACK_BYTES:
            raise ScatterError(f"{name} {cap} is below one {PACK_BYTES}-byte pack")
        piece = capped
    return piece


@dataclasses.dataclass(frozen=True)
class ScatterPiece:
    """Bytes ``[offset, offset + nbytes)`` of every chunk, carried by one op."""

    offset: int
    nbytes: int


def scatter_plan(chunk_bytes: int, piece: int | None = None) -> tuple[ScatterPiece, ...]:
    """The ops of one scatter message: pieces of at most ``piece`` bytes of every chunk."""
    chunk = int(chunk_bytes)
    if chunk <= 0 or chunk % PACK_BYTES:
        raise ScatterError(f"chunks are positive multiples of {PACK_BYTES} bytes, got {chunk}")
    size = chunk if piece is None else int(piece)
    if size < PACK_BYTES or size % PACK_BYTES:
        raise ScatterError(f"scatter pieces are positive multiples of {PACK_BYTES} bytes, got {size}")
    return tuple(ScatterPiece(offset, min(size, chunk - offset)) for offset in range(0, chunk, size))


__all__ = ["MODES", "ScatterError", "ScatterGeometry", "ScatterPiece", "destination_stride", "op_peer_bytes",
           "piece_bytes", "scatter_geometry", "scatter_plan"]
