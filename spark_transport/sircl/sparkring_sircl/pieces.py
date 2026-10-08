"""Piece plans of the large-message collectives (torch-free).

A ring session moves at most one slot per op. ``all_reduce_large`` and
``all_gather_large`` split a message of any size into ops that every rank
issues in the same order; the plans here depend only on sizes and agreed
settings, never on pointers, so every rank makes the same ops.

All-reduce (:func:`reduce_plan`): the message is ``total`` bytes; pieces of
``piece`` bytes (a multiple of 16) cover the 16-byte-aligned prefix, and a
tail of fewer than 16 bytes travels as one zero-padded 16-byte op. The sum of
every element is computed independently (float32 in rank order, one
rounding), so the piece boundaries do not change any bit.

All-gather (:func:`gather_plan`): the shard is viewed as ``outer`` rows of
``inner`` bytes (``outer`` is the product of the extents before the gathered
dimension); shard ``s`` of every row lands at byte column ``s * inner`` of an
output row of ``W * inner`` bytes. Rows of at most ``piece`` bytes travel
several to an op; a longer row travels in column tiles of ``piece`` bytes, one
row per op. Every op's source bytes are contiguous (whole rows, or part of one
row), which the padded path for rows that are not 16-byte multiples needs.
"""

from __future__ import annotations

import dataclasses

from .protocol import PACK_BYTES


class PieceError(ValueError):
    """A piece size that cannot carry the message."""


def _check_piece(piece: int) -> None:
    if type(piece) is not int or piece < PACK_BYTES or piece % PACK_BYTES:
        raise PieceError(f"piece size must be a positive multiple of {PACK_BYTES} bytes, got {piece}")


@dataclasses.dataclass(frozen=True)
class ReducePiece:
    """Bytes ``[offset, offset + nbytes)`` of the message; ``padded`` pieces are the sub-16-byte tail,
    a ``chain`` piece is one chain op (``oneshot/_chain_cute.py``), a ``ring`` piece one ring all-reduce
    (``oneshot/_links_cute.py``)."""

    offset: int
    nbytes: int
    padded: bool = False
    chain: bool = False
    ring: bool = False


def reduce_plan(total: int, piece: int, chain_from: int | None = None, *, ring_from: int | None = None,
                ring_world: int | None = None) -> tuple[ReducePiece, ...]:
    """The ops of an all-reduce of ``total`` bytes, in message order: with ``ring_from``, one ring
    op for the largest prefix of ``ring_world`` equal chunks of whole 16-byte packs when it holds at
    least ``ring_from`` bytes; else, with ``chain_from``, one chain op for the 16-byte-aligned prefix
    when it holds at least that many bytes; the rest in pieces of at most ``piece`` bytes; then the
    zero-padded tail."""
    _check_piece(piece)
    if total < 0:
        raise PieceError(f"message size {total} is negative")
    aligned = total // PACK_BYTES * PACK_BYTES
    ops: list[ReducePiece] = []
    start = 0
    if ring_from is not None:
        if ring_world is None or ring_world < 2:
            raise PieceError("a ring op needs the ring's rank count")
        ring_bytes = aligned // (ring_world * PACK_BYTES) * ring_world * PACK_BYTES
        if ring_bytes and ring_bytes >= ring_from:
            ops.append(ReducePiece(0, ring_bytes, ring=True))
            start = ring_bytes
    if not ops and chain_from is not None and aligned and aligned >= chain_from:
        ops.append(ReducePiece(0, aligned, chain=True))
        start = aligned
    ops.extend(ReducePiece(offset, min(piece, aligned - offset)) for offset in range(start, aligned, piece))
    if aligned < total:
        ops.append(ReducePiece(aligned, total - aligned, padded=True))
    return tuple(ops)


def chain_reference_halves(nbytes: int) -> tuple[int, int]:
    """Bytes of half A and half B of a chain op of ``nbytes`` (a multiple of 16): half A is the
    first ``floor(packs / 2)`` packs."""
    packs = nbytes // PACK_BYTES
    return packs // 2 * PACK_BYTES, (packs - packs // 2) * PACK_BYTES


@dataclasses.dataclass(frozen=True)
class GatherTile:
    """Rows ``[row, row + rows)`` and byte columns ``[col, col + cols)`` of the shard."""

    row: int
    rows: int
    col: int
    cols: int

    @property
    def nbytes(self) -> int:
        return self.rows * self.cols

    def source_offset(self, inner: int) -> int:
        """Byte offset of the tile's first byte in the contiguous shard."""
        return self.row * inner + self.col

    def output_offset(self, inner: int, world: int, source: int = 0) -> int:
        """Byte offset in the gathered output of shard ``source``'s first tile byte."""
        return self.row * world * inner + source * inner + self.col


def gather_plan(outer: int, inner: int, piece: int) -> tuple[GatherTile, ...]:
    """The ops of an all-gather of ``outer`` rows of ``inner`` bytes, at most ``piece`` bytes per op."""
    _check_piece(piece)
    if outer < 0 or inner < 0:
        raise PieceError(f"shard view {outer} x {inner} has a negative extent")
    if outer == 0 or inner == 0:
        return ()
    if inner <= piece:
        rows = piece // inner
        return tuple(GatherTile(row, min(rows, outer - row), 0, inner) for row in range(0, outer, rows))
    return tuple(GatherTile(row, 1, col, min(piece, inner - col))
                 for row in range(outer) for col in range(0, inner, piece))


def gather_view(shape: tuple[int, ...], dim: int, itemsize: int) -> tuple[int, int]:
    """``(outer, inner bytes)`` of a contiguous shard gathered along ``dim``."""
    if not shape:
        raise PieceError("a scalar has no dimension to gather along")
    dim = dim % len(shape)
    outer = 1
    for extent in shape[:dim]:
        outer *= int(extent)
    inner = int(itemsize)
    for extent in shape[dim:]:
        inner *= int(extent)
    return outer, inner


def padded(nbytes: int) -> int:
    """``nbytes`` rounded up to whole 16-byte packs."""
    return -(-int(nbytes) // PACK_BYTES) * PACK_BYTES


def relay_safe_bytes(busiest_lanes: int, lane_count: int, queue_bytes: int, share: float) -> int | None:
    """Largest per-peer op size, a multiple of 16, that keeps the busiest relay queue within
    ``share * queue_bytes`` when no forward windows pace the relayed lanes (``busiest_lanes``
    lane paths share that queue, each carrying ``1 / lane_count`` of a peer's bytes);
    None when no lane crosses a relay."""
    if busiest_lanes <= 0:
        return None
    factor = busiest_lanes / lane_count
    return max(PACK_BYTES, int(share * queue_bytes / factor) // PACK_BYTES * PACK_BYTES)


__all__ = ["GatherTile", "PieceError", "ReducePiece", "chain_reference_halves", "gather_plan", "gather_view",
           "padded", "reduce_plan", "relay_safe_bytes"]
