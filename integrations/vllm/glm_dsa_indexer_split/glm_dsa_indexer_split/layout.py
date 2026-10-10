"""Row partition of one prefill step's DSA indexer rows over the KV-shard copies of a TP group.

Pure arithmetic, no torch: ``runtime.py`` and the tests use it.

Geometry. A tensor-parallel (TP) group of ``T`` ranks with decode context
parallelism (DCP) ``d`` forms ``T / d`` DCP groups of ``d`` consecutive TP
ranks; group ``g`` is TP ranks ``g d`` to ``g d + d - 1`` and holds one full
copy of the KV cache, spread over its ``d`` members. The ``P`` prefill rows of
a step are padded to ``P8``, the next multiple of ``T``; every rank owns
``S = P8 / T`` consecutive rows (rank ``t``: ``[t S, (t + 1) S)``), and group
``g`` scores the rows its members own, the block ``[g d S, (g + 1) d S)``
clipped to ``P``. After the group's candidate merge every member holds the
merged top-k of the whole block; one all-gather over the TP group, in which
rank ``t`` sends its own ``S`` rows, returns rows ``[0, P8)`` in rank order on
every rank. Rows at or beyond ``P`` are padding: they are sent but never
copied back.

Launches. The image scores a prefill chunk in launches (``chunk`` metadata
objects, each one request's consecutive rows). A group runs every launch
clipped to its block. With full launches on, each request's consecutive rows
inside the block are re-cut into launches of ``cap`` rows (the prepared
prefill plan's row capacity, 4,096), so the launch boundaries are
independent of the image's logits-budget split.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RowLayout:
    """Where one rank's rows lie in a step of ``rows`` prefill rows."""

    rows: int
    tp_size: int
    dcp_size: int
    tp_rank: int
    padded: int  # rows rounded up to a multiple of tp_size
    per_rank: int  # padded // tp_size
    copies: int  # tp_size // dcp_size: DCP groups, one KV copy each
    group: int  # tp_rank // dcp_size
    block_start: int  # first row this rank's group scores; equals block_stop for an empty block
    block_stop: int
    send_start: int  # rows this rank sends in the TP all-gather, in padded coordinates
    send_stop: int

    @property
    def block_rows(self) -> int:
        return self.block_stop - self.block_start

    def owner(self, row: int) -> int:
        """TP rank whose all-gather slot carries ``row``."""
        return row // self.per_rank

    def scoring_group(self, row: int) -> int:
        """DCP group that scores ``row``."""
        return row // (self.per_rank * self.dcp_size)


def check_geometry(tp_size: int, dcp_size: int) -> int:
    """Number of KV copies; raises ValueError for a TP / DCP pair the split cannot use."""
    if tp_size < 1 or dcp_size < 1:
        raise ValueError(f"TP {tp_size} and DCP {dcp_size} must be positive")
    if tp_size % dcp_size:
        raise ValueError(f"DCP {dcp_size} does not divide TP {tp_size}")
    copies = tp_size // dcp_size
    if copies < 2:
        raise ValueError(f"DCP {dcp_size} at TP {tp_size} leaves a single KV copy; the row split needs two or more")
    return copies


def row_layout(rows: int, tp_size: int, dcp_size: int, tp_rank: int) -> RowLayout:
    """The partition of ``rows`` prefill rows for TP rank ``tp_rank``."""
    copies = check_geometry(tp_size, dcp_size)
    if rows < 1:
        raise ValueError(f"rows must be positive, got {rows}")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"TP rank {tp_rank} outside 0..{tp_size - 1}")
    padded = -(-rows // tp_size) * tp_size
    per_rank = padded // tp_size
    group = tp_rank // dcp_size
    block = per_rank * dcp_size
    start = min(group * block, rows)
    stop = min((group + 1) * block, rows)
    return RowLayout(rows=rows, tp_size=tp_size, dcp_size=dcp_size, tp_rank=tp_rank, padded=padded,
                     per_rank=per_rank, copies=copies, group=group, block_start=start, block_stop=stop,
                     send_start=tp_rank * per_rank, send_stop=(tp_rank + 1) * per_rank)


@dataclass(frozen=True)
class Fragment:
    """Rows ``[start, stop)`` (step coordinates) taken from image launch ``chunk``."""

    chunk: int
    start: int
    stop: int


@dataclass(frozen=True)
class Launch:
    """One indexer launch of the split path: consecutive rows of one request."""

    start: int
    stop: int
    fragments: tuple[Fragment, ...]

    @property
    def rows(self) -> int:
        return self.stop - self.start


def block_launches(spans: Sequence[tuple[int, int, Hashable]], start: int, stop: int, *,
                   full: bool, cap: int) -> list[Launch]:
    """The launches that score rows ``[start, stop)``.

    ``spans`` lists the image's launches in order as ``(first row, end row,
    request key)``; consecutive spans must tile the step's rows. Without
    ``full`` every span clipped to the block is one launch. With ``full``
    every maximal run of consecutive clipped spans of one request (equal keys)
    is cut into launches of ``cap`` rows, the last one shorter.
    """
    if cap < 1:
        raise ValueError(f"cap must be positive, got {cap}")
    pieces: list[tuple[Fragment, Hashable]] = []
    for index, (lo, hi, key) in enumerate(spans):
        a, b = max(lo, start), min(hi, stop)
        if a < b:
            pieces.append((Fragment(index, a, b), key))
    if not full:
        return [Launch(f.start, f.stop, (f,)) for f, _ in pieces]
    runs: list[list[Fragment]] = []
    last_key: object = object()
    for fragment, key in pieces:
        if runs and key == last_key and runs[-1][-1].stop == fragment.start:
            runs[-1].append(fragment)
        else:
            runs.append([fragment])
        last_key = key
    launches: list[Launch] = []
    for run in runs:
        lo, hi = run[0].start, run[-1].stop
        for a in range(lo, hi, cap):
            b = min(a + cap, hi)
            parts = tuple(Fragment(f.chunk, max(f.start, a), min(f.stop, b)) for f in run
                          if f.start < b and f.stop > a)
            launches.append(Launch(a, b, parts))
    return launches


__all__ = ["Fragment", "Launch", "RowLayout", "block_launches", "check_geometry", "row_layout"]
