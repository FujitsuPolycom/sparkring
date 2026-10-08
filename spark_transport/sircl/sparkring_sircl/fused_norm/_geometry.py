"""Index arithmetic of the fused all-reduce + residual add + RMSNorm kernels.

Every function here is written with ``+``, ``-``, ``*``, ``//``, comparisons
and an injectable ``minimum``, so the same source serves two callers: the CPU
tests call it with Python integers, and the CuTe DSL kernel calls it while it
is traced, with ``Int32`` values and a ``min.s32`` instruction as ``minimum``.
The kernel therefore waits for exactly the flags that the tests prove
sufficient.

Terms. A message is ``rows`` rows of ``row_packs`` 16-byte packs (eight BF16
values each), ``packs = rows * row_packs`` in total. The kernel runs one
cooperative thread array (CTA) per row and one thread per pack of the row:
thread ``t`` of CTA ``row`` owns pack ``row * row_packs + t``.

Wire facts of the ring session that this module must agree with (the native
progress thread ``sparkring_sircl/oneshot/_roce_proxy.c`` and the reference
arithmetic ``sparkring_sircl.protocol.stripe`` and ``protocol.chunk``):

* a payload of ``count`` packs posted to one peer is split into ``lanes``
  stripes; stripe ``lane`` starts ``lane * (count // lanes) + min(lane,
  count % lanes)`` packs into the payload and holds ``count // lanes`` packs,
  plus one when ``lane < count % lanes``. Each stripe is followed on the same
  reliable queue pair by its own flag write, so a flag that shows the
  sequence number proves that its stripe's bytes are in memory;
* a one-shot payload is the whole message;
* chunk ``j`` of a two-shot message is the pack range
  ``[j * packs // world, (j + 1) * packs // world)``. In the scatter phase
  every peer writes its input's chunk ``rank`` to this rank; in the gather
  phase owner ``j`` writes its reduced chunk ``j`` to every peer. Each of
  those payloads is striped like a one-shot payload of ``count`` packs.

Origin: SparkRing's fused-norm kernels.
"""

from __future__ import annotations

from typing import Callable

ONESHOT = "oneshot"
TWOSHOT = "twoshot"
ALGORITHMS = (ONESHOT, TWOSHOT)
PACK_BYTES = 16
BF16_PER_PACK = 8
WARP = 32
SCATTER = 0  # flag namespace of the first network phase (one-shot payloads and two-shot scatter)
GATHER = 1  # flag namespace of the two-shot gather phase


def _python_min(a: int, b: int) -> int:
    return a if a < b else b


def stripe_bounds(start, count, lanes: int, lane, minimum: Callable = _python_min):
    """Pack range ``[lo, hi)`` of stripe ``lane`` of a payload of ``count`` packs starting at ``start``."""
    base = count // lanes
    remainder = count - base * lanes
    lo = start + lane * base + minimum(lane, remainder)
    hi = lo + base + (minimum(lane + 1, remainder) - minimum(lane, remainder))
    return lo, hi


def chunk_bounds(packs, world: int, owner):
    """Pack range ``[lo, hi)`` of two-shot chunk ``owner`` of a ``packs``-pack message."""
    return (owner * packs) // world, ((owner + 1) * packs) // world


def row_bounds(row, row_packs: int):
    """Pack range ``[lo, hi)`` of one row."""
    lo = row * row_packs
    return lo, lo + row_packs


def overlaps(a_lo, a_hi, b_lo, b_hi):
    """Whether the half-open ranges intersect (Python ``bool`` or DSL ``Boolean``)."""
    return (a_lo < b_hi) & (b_lo < a_hi)


def oneshot_wait(row, row_packs: int, packs, peer, lane, rank: int, lanes: int,
                 minimum: Callable = _python_min):
    """Whether CTA ``row`` waits for the one-shot flag of ``(peer, lane)``.

    Peer ``peer`` sends the whole message; stripe ``lane`` must arrive before
    the CTA reads any pack of its row inside that stripe. The own rank never
    sends to itself.
    """
    r_lo, r_hi = row_bounds(row, row_packs)
    s_lo, s_hi = stripe_bounds(0, packs, lanes, lane, minimum)
    return (peer != rank) & overlaps(s_lo, s_hi, r_lo, r_hi)


def twoshot_scatter_wait(row, row_packs: int, packs, peer, lane, rank: int, world: int,
                         lanes: int, minimum: Callable = _python_min):
    """Whether CTA ``row`` waits for the scatter-phase flag of ``(peer, lane)``.

    The CTA reduces the packs of its row that lie in this rank's own chunk;
    each peer's contribution to that chunk arrives striped over ``lanes``.
    """
    r_lo, r_hi = row_bounds(row, row_packs)
    c_lo, c_hi = chunk_bounds(packs, world, rank)
    s_lo, s_hi = stripe_bounds(c_lo, c_hi - c_lo, lanes, lane, minimum)
    return (peer != rank) & overlaps(s_lo, s_hi, r_lo, r_hi)


def twoshot_gather_wait(row, row_packs: int, packs, owner, lane, rank: int, world: int,
                        lanes: int, minimum: Callable = _python_min):
    """Whether CTA ``row`` waits for the gather-phase flag of ``(owner, lane)``.

    Owner ``owner`` sends its reduced chunk; the CTA needs the stripes of
    every other owner's chunk that intersect its row.
    """
    r_lo, r_hi = row_bounds(row, row_packs)
    c_lo, c_hi = chunk_bounds(packs, world, owner)
    s_lo, s_hi = stripe_bounds(c_lo, c_hi - c_lo, lanes, lane, minimum)
    return (owner != rank) & overlaps(s_lo, s_hi, r_lo, r_hi)


def flag_index(namespace: int, source, slot, lane, world: int, slots: int, lanes: int):
    """Index of the flag line for ``(namespace, source, slot, lane)`` in the flag area.

    Matches the native layer's addressing: namespace ``SCATTER`` holds the
    one-shot and scatter flags, ``GATHER`` starts after the first
    ``world * slots * lanes`` lines.
    """
    return namespace * world * slots * lanes + (source * slots + slot) * lanes + lane


# --------------------------------------------------------------------------- plain-Python checks


def stripes_covering(start: int, count: int, lanes: int, lo: int, hi: int) -> set[int]:
    """Stripes of a payload ``[start, start + count)`` that hold at least one pack of ``[lo, hi)``."""
    found = set()
    for pack in range(max(lo, start), min(hi, start + count)):
        for lane in range(lanes):
            s_lo, s_hi = stripe_bounds(start, count, lanes, lane)
            if s_lo <= pack < s_hi:
                found.add(lane)
                break
    return found


def check_launch_geometry(rows: int, row_packs: int, world: int, lanes: int) -> None:
    """Raise ``ValueError`` unless the kernels' integer assumptions hold for this launch."""
    if rows < 1 or row_packs < WARP or row_packs > 1024 or row_packs % WARP:
        raise ValueError(
            f"rows={rows} row_packs={row_packs}: the kernels need rows >= 1 and a row of "
            f"32..1024 packs in whole warps (hidden a multiple of {WARP * BF16_PER_PACK}, at most 8192)"
        )
    if (rows * row_packs) % world:
        raise ValueError(f"{rows * row_packs} packs do not split into {world} equal two-shot chunks")
    if lanes not in (1, 2):
        raise ValueError(f"the fused kernels support one or two lanes per peer, not {lanes}")
    if world * lanes > row_packs:
        raise ValueError("the flag waiters need world * lanes <= threads per CTA")


__all__ = [
    "ALGORITHMS",
    "BF16_PER_PACK",
    "GATHER",
    "ONESHOT",
    "PACK_BYTES",
    "SCATTER",
    "TWOSHOT",
    "WARP",
    "check_launch_geometry",
    "chunk_bounds",
    "flag_index",
    "oneshot_wait",
    "overlaps",
    "row_bounds",
    "stripe_bounds",
    "stripes_covering",
    "twoshot_gather_wait",
    "twoshot_scatter_wait",
]
