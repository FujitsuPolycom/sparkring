"""CuTe DSL kernels of the chain and ring all-gather, reduce-scatter and all-reduce (one launch per op).

The ranks of the group form a chain of cable neighbors (chain index 0 to
``W - 1``; ``order[i]`` is the rank at chain index ``i``). Both collectives run
over the session's two chain links (``protocol.LinkLayout``): link 0 carries
items toward higher chain indices and link 1 toward lower ones. A rank's
block of bytes travels in pieces; round ``p`` of a link carries piece ``p``
of every block the link moves at this rank (``protocol.link_rounds``). An
item's tag is its index on its link plus one, counted over the session; the
device counters hold the own and inbound items of both links before this op,
and the last block to finish advances them unless a wait timed out. A wait
longer than the session's wait limit records what it waited for (command
ring words 2, 3, 6 and 8) and poisons the session. In every block the lane
flags are awaited by warp 0 and the own slot by the block's last thread, so
two different wait loops never run in divergent threads of one warp.

Ring collectives (:class:`LinkRing`) run over links 2 and 3 toward the next
rank of the ring that closes the chain (its last rank reaches its first over
the closing cable of a cycle, or through relays on a path).

All-gather (:class:`LinkGather`): every rank's block travels toward both ends.
On link 0, rank ``j`` stages its own piece in the link's own slots and the
native progress thread writes it downstream, followed by the pieces of
owners ``j - 1`` down to 0, which the progress thread forwards straight from
the receive slots. Link 1 mirrors it. The output holds the block of chain
index ``i`` at byte ``order[i] * S``: rank order, whatever the chain order.
Bytes are copied unchanged.

Reduce-scatter (:class:`LinkScatter`): the rank at chain index ``j`` owns
chunk ``order[j]`` of the input. Partial sums of every owner's chunk travel
toward it from both ends: on link 0, rank ``j`` sends the partials for owners
``W - 1`` down to ``j + 1`` (farthest first), each the dtype rounding of the
float32 sum of the partial it received and its own values; link 1 mirrors
it. The owner stores ``round((L + x) + R)`` (float32 additions in that order,
one rounding), ``L`` the link-0 partial of the indices below it, ``x`` its
own values and ``R`` the link-1 partial of the indices above it; an end of
the chain has one of them. Every rank's rows are identical across ranks and
deterministic for a given size and chain order; they can differ from the
chain all-reduce's rows and from the one-shot rows in the last place
(``references.chain_reduce_scatter`` reproduces them bit for bit).

Copies and sums move ``unroll`` packs per thread per pass, with the loads of
a pass issued together (``_cute_batch.ld_v4_u32_batch``), so a pass over
pinned host memory pays one memory latency. A ring rank stores its link-3 own
item (the all-gather's own piece, the all-reduce's own result) to the link's
own slot and to the output in the same pass and publishes the item after that
pass, so the item leaves one pass after its sources were read.
"""

# Annotations stay evaluated (no postponed evaluation): the CuTe DSL reads each
# launch parameter's type from its annotation object, and some DSL releases
# type string annotations from the example arguments instead, which makes
# 64-bit address parameters 32-bit.

from collections.abc import Callable, Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

from ..protocol import (RING_GATHER_STAGGER_SHIFT, RING_STAGGER_SHIFT, TRACE_LINK_STREAM, FLAG_STRIDE, PACK_BYTES, Ctrl, ErrorKind, LinkLayout,
                        LinkOp, TraceEvent)
from ._compile import compile_launcher, current_cuda_stream, make_pointer
from ._cute_batch import ld_v4_u32_batch
from ._cute_intrinsics import (
    atomic_add_relaxed_gpu_u32,
    f32_as_u32,
    fence_sc_gpu,
    fence_sc_sys,
    globaltimer_u32x2,
    ld_relaxed_gpu_u32,
    ld_relaxed_sys_u32,
    min_s32,
    pack_f32x2_to_bf16x2,
    pack_f32x2_to_f16x2,
    st_global_v4_u32,
    st_relaxed_sys_u32,
    st_release_gpu_u32,
    u32_as_f32,
    unpack_bf16x2,
    unpack_f16x2,
)
from ._timed_wait import spin_until_eq_timed_sys, spin_until_ge_timed_sys

ROLES = 4                    # all-gather roles: own link 0, own link 1, inbound link 0, inbound link 1
# Device counters shared by every link collective (one item numbering per link): own items of
# links 0 and 1, inbound items of links 0 and 1, the link op sequence, then the tail arrivals of
# the chain all-gather (word 5), the chain reduce-scatter (word 6) and the ring reduce-scatter
# (word 7), the own and inbound items of links 2 and 3 (words 8-11), and the tail arrivals of the
# ring all-gather and all-reduce (words 12 and 13). Each kernel type counts its blocks' arrivals on
# a word of its own; the block whose arrival completes the launch's grid (prior + 1 == grid size)
# is the last one, and it returns the word to 0, so consecutive launches of one kernel type may use
# different grids (a tuning table's blocks per op) and every launch starts its count at 0.
COUNTER_WORDS = 16
DTYPE_NAMES = ("float16", "bfloat16", "float32")
PIECE_COUNTERS = 65536       # arrival counters of an owner's pieces: the most pieces of one reduce-scatter
DECISION_WORDS = 256         # one word per block: whether the block combines its piece
_ERROR_SEQ = 4 * int(Ctrl.ERROR_SEQ)
_MISSING_PEER = 4 * int(Ctrl.MISSING_PEER)
_MISSING_LANE = 4 * int(Ctrl.MISSING_LANE)
_ERROR_KIND = 4 * int(Ctrl.ERROR_KIND)
_WAIT_LIMIT = 4 * int(Ctrl.WAIT_LIMIT_US)


class _LinkKernel:
    """Specialization, link-area addresses and failure records shared by the link kernels."""

    def _setup(self, world_size: int, chain_index: int, prev_rank: int, next_rank: int, rank: int,
               order: Sequence[int], threads: int, lanes: int, slots: int, slot_bytes: int, blocks_per_role: int,
               unroll: int) -> None:
        if not 0 <= chain_index < world_size or world_size < 2 or len(order) != world_size:
            raise ValueError(f"chain index {chain_index} of a chain of {world_size} (order {list(order)})")
        if int(threads) < 64 or int(threads) % 32 or not 1 <= int(lanes) <= 32:
            raise ValueError(f"link kernels need a multiple of 32 threads, at least 64, so the slot wait runs "
                             f"in a warp apart from the lane waits; got {threads} threads and {lanes} lanes")
        if not 1 <= int(unroll) <= 8 or not 1 <= int(blocks_per_role) <= DECISION_WORDS // 2:
            raise ValueError(f"link kernels move 1 to 8 packs per thread per pass and run 1 to "
                             f"{DECISION_WORDS // 2} blocks per role; got {unroll} and {blocks_per_role}")
        self._world = int(world_size)
        self._index = int(chain_index)
        self._prev = int(prev_rank)
        self._next = int(next_rank)
        self._rank = int(rank)
        self._order = tuple(int(r) for r in order)
        self._threads = int(threads)
        self._lanes = int(lanes)
        self._slots = int(slots)
        self._slot_bytes = int(slot_bytes)
        self._nb = int(blocks_per_role)
        self._unroll = int(unroll)
        self._layout = LinkLayout(int(lanes), int(slots), int(slot_bytes))
        self._trace_capacity = 0

    @cute.jit
    def _slot(self, link_base: Int64, area_off: cutlass.Constexpr[int], link: cutlass.Constexpr[int],
              m: Uint32) -> Int64:
        return link_base + Int64(area_off) + (Int64(link * self._slots) + Int64(m)) * Int64(self._slot_bytes)

    @cute.jit
    def _flag(self, link_base: Int64, link: cutlass.Constexpr[int], m: Uint32, lane: Int32) -> Int64:
        line = (Int64(link * self._slots) + Int64(m)) * Int64(self._lanes) + Int64(lane)
        return link_base + Int64(self._layout.rflag_off) + line * Int64(FLAG_STRIDE)

    @cute.jit
    def _word(self, link_base: Int64, area_off: cutlass.Constexpr[int], link: cutlass.Constexpr[int],
              m: Uint32) -> Int64:
        return link_base + Int64(area_off + link * FLAG_STRIDE) + Int64(m) * Int64(4)

    @cute.jit
    def _trace(self, trace_base: Int64, event: cutlass.Constexpr[int], link: cutlass.Constexpr[int],
               tag: Uint32) -> None:
        """Thread 0 appends one event record of link ``link`` (stream ``TRACE_LINK_STREAM + link``) when the
        specialization has a trace (the record layout of the chain kernel's trace buffer)."""
        if cutlass.const_expr(self._trace_capacity > 0):
            tidx, _, _ = cute.arch.thread_idx()
            if Int32(tidx) == Int32(0):
                index = atomic_add_relaxed_gpu_u32(trace_base, Uint32(1))
                if index < Uint32(self._trace_capacity):
                    lo, hi = globaltimer_u32x2()
                    record = trace_base + Int64(16) + Int64(index) * Int64(16)
                    st_global_v4_u32(record, lo, hi, tag, Uint32(event | ((TRACE_LINK_STREAM + link) << 16)))

    @cute.jit
    def _trace_flags(self, trace_base: Int64, link: cutlass.Constexpr[int], tag: Uint32) -> None:
        """After the lane-flag waits: warp 0 converges and thread 0 records ``KERNEL_FLAG``."""
        if cutlass.const_expr(self._trace_capacity > 0):
            tidx, _, _ = cute.arch.thread_idx()
            if Int32(tidx) < Int32(32):
                cute.arch.sync_warp()
            self._trace(trace_base, int(TraceEvent.KERNEL_FLAG), link, tag)

    @cute.jit
    def _fail(self, ctrl_base: Int64, poison_ptr: Int64, peer: Int32, lane: Int32, kind: cutlass.Constexpr[int],
              tag: Uint32) -> None:
        st_relaxed_sys_u32(ctrl_base + Int64(_MISSING_PEER), Uint32(peer))
        st_relaxed_sys_u32(ctrl_base + Int64(_MISSING_LANE), Uint32(lane))
        st_relaxed_sys_u32(ctrl_base + Int64(_ERROR_KIND), Uint32(kind))
        fence_sc_sys()
        st_relaxed_sys_u32(ctrl_base + Int64(_ERROR_SEQ), tag)
        st_release_gpu_u32(poison_ptr, Uint32(1))


class LinkGather(_LinkKernel):
    """One kernel specialization of the chain all-gather: chain position, neighbors, rank order, block
    size, ring geometry. Four roles of ``blocks_per_role`` blocks; block ``b`` of a role takes every
    ``blocks_per_role``-th item: own link 0 and own link 1 stage the own piece (the first of them that
    exists also copies it to the output), inbound link 0 and inbound link 1 copy received pieces to
    their owners' places in the output."""

    def __init__(self, world_size: int, chain_index: int, prev_rank: int, next_rank: int, rank: int,
                 order: Sequence[int], threads: int, lanes: int, slots: int, slot_bytes: int,
                 blocks_per_role: int, unroll: int) -> None:
        self._setup(world_size, chain_index, prev_rank, next_rank, rank, order, threads, lanes, slots, slot_bytes,
                    blocks_per_role, unroll)

    # -- copies -------------------------------------------------------------------------------

    @cute.jit
    def _copy(self, src: Int64, system: cutlass.Constexpr[bool], dst: Int64, dst2: Int64,
              two: cutlass.Constexpr[bool], packs: Int32, zero: Uint32) -> None:
        """Copy ``packs`` packs from ``src`` to ``dst`` (and to ``dst2`` when ``two``), ``unroll``
        packs per thread per pass; ``system`` reads pinned host memory at system scope."""
        tidx, _, _ = cute.arch.thread_idx()
        thread = Int32(tidx)
        last = packs - Int32(1)
        base = Int32(0)
        while base < packs:
            addrs = []
            for u in cutlass.range_constexpr(self._unroll):
                index = min_s32(base + Int32(u * self._threads) + thread, last)
                addrs.append(src + Int64(index) * Int64(PACK_BYTES))
            words = ld_v4_u32_batch(addrs, [system] * self._unroll, zero)
            for u in cutlass.range_constexpr(self._unroll):
                index = base + Int32(u * self._threads) + thread
                if index < packs:
                    w = words[u]
                    st_global_v4_u32(dst + Int64(index) * Int64(PACK_BYTES), w[0], w[1], w[2], w[3])
                    if cutlass.const_expr(two):
                        st_global_v4_u32(dst2 + Int64(index) * Int64(PACK_BYTES), w[0], w[1], w[2], w[3])
            base = base + Int32(self._unroll * self._threads)

    # -- roles --------------------------------------------------------------------------------

    @cute.jit
    def _own(self, link: cutlass.Constexpr[int], to_output: cutlass.Constexpr[bool], sub: Int32,
             input_base: Int64, output_base: Int64, shard_packs: Int32, piece_packs: Int32, base: Uint32,
             link_base: Int64, ctrl_base: Int64, poison_ptr: Int64, spin_limit: Uint32, limit_us: Uint32,
             zero: Uint32) -> None:
        """Stage every own piece of ``link`` in its own slots (and copy it to the output)."""
        tidx, _, _ = cute.arch.thread_idx()
        thread = Int32(tidx)
        pieces = (shard_packs + piece_packs - Int32(1)) // piece_packs
        own_pos = self._order[self._index]
        p = sub
        while p < pieces:
            g = base + Uint32(p)
            m = g % Uint32(self._slots)
            tag = g + Uint32(1)
            if thread == Int32(self._threads - 1):
                sent = link_base + Int64(self._layout.sent_off + link * FLAG_STRIDE)
                if spin_until_ge_timed_sys(sent, tag - Uint32(self._slots), spin_limit, limit_us) != Uint32(0):
                    self._fail(ctrl_base, poison_ptr, Int32(self._rank), Int32(255), int(ErrorKind.CHAIN_SLOT), tag)
            cute.arch.sync_threads()
            if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                first = p * piece_packs
                count = shard_packs - first
                if piece_packs < count:
                    count = piece_packs
                src = input_base + Int64(first) * Int64(PACK_BYTES)
                slot = self._slot(link_base, self._layout.own_off, link, m)
                out = output_base + (Int64(own_pos) * Int64(shard_packs) + Int64(first)) * Int64(PACK_BYTES)
                self._copy(src, False, slot, out, to_output, count, zero)
                cute.arch.sync_threads()
                if thread == Int32(0):
                    fence_sc_sys()
                    st_relaxed_sys_u32(self._word(link_base, self._layout.ready_off, link, m), tag)
                p = p + Int32(self._nb)
            else:
                p = pieces

    @cute.jit
    def _inbound(self, link: cutlass.Constexpr[int], sub: Int32, output_base: Int64, shard_packs: Int32,
                 piece_packs: Int32, base: Uint32, link_base: Int64, ctrl_base: Int64, poison_ptr: Int64,
                 spin_limit: Uint32, limit_us: Uint32, zero: Uint32) -> None:
        """Copy every inbound piece of ``link`` to its owner's place in the output."""
        tidx, _, _ = cute.arch.thread_idx()
        thread = Int32(tidx)
        # Owners per inbound round: link 0 brings owners j - 1 down to 0, link 1 owners j + 1 up to W - 1.
        per_round = self._index if link == 0 else self._world - 1 - self._index
        peer = self._prev if link == 0 else self._next
        pieces = (shard_packs + piece_packs - Int32(1)) // piece_packs
        items = pieces * Int32(per_round)
        i = sub
        while i < items:
            g = base + Uint32(i)
            m = g % Uint32(self._slots)
            tag = g + Uint32(1)
            if thread < Int32(self._lanes):
                if spin_until_eq_timed_sys(self._flag(link_base, link, m, thread), tag, spin_limit,
                                           limit_us) != Uint32(0):
                    self._fail(ctrl_base, poison_ptr, Int32(peer), thread, int(ErrorKind.CHAIN_CHUNK), tag)
            cute.arch.sync_threads()
            if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                p = i // Int32(per_round)
                r = i - p * Int32(per_round)
                first = p * piece_packs
                count = shard_packs - first
                if piece_packs < count:
                    count = piece_packs
                # The output place of the owner at chain index j - 1 - r (link 0) or j + 1 + r (link 1).
                position = Int32(0)
                for k in cutlass.range_constexpr(per_round):
                    owner = self._index - 1 - k if link == 0 else self._index + 1 + k
                    if r == Int32(k):
                        position = Int32(self._order[owner])
                out = output_base + (Int64(position) * Int64(shard_packs) + Int64(first)) * Int64(PACK_BYTES)
                slot = self._slot(link_base, self._layout.recv_off, link, m)
                self._copy(slot, True, out, out, False, count, zero)
                cute.arch.sync_threads()
                if thread == Int32(0):
                    fence_sc_sys()
                    st_relaxed_sys_u32(self._word(link_base, self._layout.consumed_off, link, m), tag)
                i = i + Int32(self._nb)
            else:
                i = items

    # -- launch -------------------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        shard_packs: Int32,
        piece_packs: Int32,
        link_base: Int64,
        counters: Int64,
        ctrl_base: Int64,
        poison_ptr: Int64,
        spin_limit: Uint32,
        stream: cuda.CUstream,
    ) -> None:
        """Host entry: one block per (role, item lane)."""
        self.kernel(
            input_ptr, output_ptr, shard_packs, piece_packs, link_base, counters, ctrl_base, poison_ptr, spin_limit,
        ).launch(grid=(ROLES * self._nb, 1, 1), block=[self._threads, 1, 1], cluster=(1, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        shard_packs: Int32,
        piece_packs: Int32,
        link_base: Int64,
        counters: Int64,
        ctrl_base: Int64,
        poison_ptr: Int64,
        spin_limit: Uint32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        input_base = Int64(input_ptr.toint())
        output_base = Int64(output_ptr.toint())
        role = Int32(bidx) // Int32(self._nb)
        sub = Int32(bidx) - role * Int32(self._nb)
        zero = Uint32(shard_packs) >> Uint32(31)
        sends_next = self._index < self._world - 1
        sends_prev = self._index > 0
        if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
            # Every block reads the counters before any block can advance them:
            # the advance happens only after every block arrived at the tail.
            own0 = ld_relaxed_gpu_u32(counters)
            own1 = ld_relaxed_gpu_u32(counters + Int64(4))
            in0 = ld_relaxed_gpu_u32(counters + Int64(8))
            in1 = ld_relaxed_gpu_u32(counters + Int64(12))
            seq = ld_relaxed_gpu_u32(counters + Int64(16)) + Uint32(1)
            if Int32(bidx) == Int32(0):
                if Int32(tidx) == Int32(0):
                    params = link_base + Int64(self._layout.ctrl_off + 4) + Int64(seq & Uint32(1)) * Int64(16)
                    st_relaxed_sys_u32(params, Uint32(int(LinkOp.ALL_GATHER)))
                    st_relaxed_sys_u32(params + Int64(4), Uint32(shard_packs * Int32(PACK_BYTES)))
                    st_relaxed_sys_u32(params + Int64(8), Uint32(piece_packs * Int32(PACK_BYTES)))
                    fence_sc_sys()
                    st_relaxed_sys_u32(link_base + Int64(self._layout.ctrl_off), seq)
            limit_us = ld_relaxed_sys_u32(ctrl_base + Int64(_WAIT_LIMIT))
            if cutlass.const_expr(sends_next):
                if role == Int32(0):
                    self._own(0, True, sub, input_base, output_base, shard_packs, piece_packs, own0, link_base,
                              ctrl_base, poison_ptr, spin_limit, limit_us, zero)
            if cutlass.const_expr(sends_prev):
                if role == Int32(1):
                    self._own(1, not sends_next, sub, input_base, output_base, shard_packs, piece_packs, own1,
                              link_base, ctrl_base, poison_ptr, spin_limit, limit_us, zero)
            if cutlass.const_expr(sends_prev):
                if role == Int32(2):
                    self._inbound(0, sub, output_base, shard_packs, piece_packs, in0, link_base, ctrl_base,
                                  poison_ptr, spin_limit, limit_us, zero)
            if cutlass.const_expr(sends_next):
                if role == Int32(3):
                    self._inbound(1, sub, output_base, shard_packs, piece_packs, in1, link_base, ctrl_base,
                                  poison_ptr, spin_limit, limit_us, zero)
            # The last block to finish advances the item bases and the sequence.
            fence_sc_gpu()
            cute.arch.sync_threads()
            if Int32(tidx) == Int32(0):
                prior = atomic_add_relaxed_gpu_u32(counters + Int64(20), Uint32(1))
                if prior + Uint32(1) == Uint32(gdim):
                    # Every block of this launch arrived: return the tail to 0, so the next launch of this
                    # kernel finds its last block by the same count whatever its grid.
                    st_release_gpu_u32(counters + Int64(20), Uint32(0))
                    fence_sc_gpu()
                    if ld_relaxed_sys_u32(ctrl_base + Int64(_ERROR_SEQ)) == Uint32(0):
                        pieces = Uint32((shard_packs + piece_packs - Int32(1)) // piece_packs)
                        if cutlass.const_expr(sends_next):
                            st_release_gpu_u32(counters, own0 + pieces)
                        if cutlass.const_expr(sends_prev):
                            st_release_gpu_u32(counters + Int64(4), own1 + pieces)
                        st_release_gpu_u32(counters + Int64(8), in0 + pieces * Uint32(self._index))
                        st_release_gpu_u32(counters + Int64(12),
                                           in1 + pieces * Uint32(self._world - 1 - self._index))
                        st_release_gpu_u32(counters + Int64(16), seq)


class LinkScatter(_LinkKernel):
    """One kernel specialization of the chain reduce-scatter: dtype, chain position, neighbors, rank
    order, block size, ring geometry. Two roles of ``blocks_per_role`` blocks, one per link: the
    first rank of a link stages its own values as the partials; every other rank adds its own
    values to each inbound partial and stages it for the next rank, or keeps the partials of its
    own chunk.

    The two partials of an owner's piece arrive in either order. Each is first copied out of its
    receive slot (``L`` to the output, ``R`` to a scratch buffer), so no slot waits for the other
    link, and the block that arrives second (a per-piece counter that every op raises by two)
    computes the result from both copies.
    """

    def __init__(self, dtype_name: str, world_size: int, chain_index: int, prev_rank: int, next_rank: int,
                 rank: int, order: Sequence[int], threads: int, lanes: int, slots: int, slot_bytes: int,
                 blocks_per_role: int, unroll: int) -> None:
        if dtype_name not in DTYPE_NAMES:
            raise ValueError(f"unsupported chain reduce-scatter dtype {dtype_name!r}")
        self._dtype_name = dtype_name
        self._setup(world_size, chain_index, prev_rank, next_rank, rank, order, threads, lanes, slots, slot_bytes,
                    blocks_per_role, unroll)

    # -- arithmetic ---------------------------------------------------------------------------

    def _sum(self, group):
        """The dtype rounding of the float32 sum of the packs of ``group``, added in order (four
        words). Plain Python evaluated while the kernel is traced."""
        if len(group) == 1:
            return group[0]
        packed = []
        for w in range(4):
            if self._dtype_name == "float32":
                total = u32_as_f32(group[0][w])
                for words in group[1:]:
                    total = total + u32_as_f32(words[w])
                packed.append(f32_as_u32(total))
                continue
            unpack = unpack_f16x2 if self._dtype_name == "float16" else unpack_bf16x2
            lo, hi = unpack(group[0][w])
            for words in group[1:]:
                a, b = unpack(words[w])
                lo = lo + a
                hi = hi + b
            pack = pack_f32x2_to_f16x2 if self._dtype_name == "float16" else pack_f32x2_to_bf16x2
            packed.append(pack(lo, hi))
        return packed

    @cute.jit
    def _pass(self, s0: Int64, s1: Int64, s2: Int64, y0: cutlass.Constexpr[bool], y1: cutlass.Constexpr[bool],
              y2: cutlass.Constexpr[bool], count: cutlass.Constexpr[int], dst: Int64, packs: Int32,
              zero: Uint32) -> None:
        """``dst`` = the sum (the copy, for one source) of the first ``count`` sources' first
        ``packs`` packs, in source order; ``unroll`` packs per thread per pass with every load of a
        pass issued together. ``y<k>`` reads source ``k`` at system scope (pinned host memory, or
        another block's writes)."""
        tidx, _, _ = cute.arch.thread_idx()
        thread = Int32(tidx)
        sources = ((s0, y0), (s1, y1), (s2, y2))[:count]
        last = packs - Int32(1)
        base = Int32(0)
        while base < packs:
            addrs = []
            flags = []
            for u in cutlass.range_constexpr(self._unroll):
                index = min_s32(base + Int32(u * self._threads) + thread, last)
                for source, system in sources:
                    addrs.append(source + Int64(index) * Int64(PACK_BYTES))
                    flags.append(system)
            words = ld_v4_u32_batch(addrs, flags, zero)
            for u in cutlass.range_constexpr(self._unroll):
                index = base + Int32(u * self._threads) + thread
                if index < packs:
                    packed = self._sum(words[u * count:(u + 1) * count])
                    st_global_v4_u32(dst + Int64(index) * Int64(PACK_BYTES), packed[0], packed[1], packed[2],
                                     packed[3])
            base = base + Int32(self._unroll * self._threads)

    @cute.jit
    def _pass_two(self, s0: Int64, s1: Int64, y0: cutlass.Constexpr[bool], y1: cutlass.Constexpr[bool],
                  count: cutlass.Constexpr[int], dst: Int64, dst2: Int64, packs: Int32, zero: Uint32) -> None:
        """``_pass`` of the first ``count`` (1 or 2) sources with every result pack stored at ``dst`` and at
        ``dst2``: one read of the sources serves both destinations."""
        tidx, _, _ = cute.arch.thread_idx()
        thread = Int32(tidx)
        sources = ((s0, y0), (s1, y1))[:count]
        last = packs - Int32(1)
        base = Int32(0)
        while base < packs:
            addrs = []
            flags = []
            for u in cutlass.range_constexpr(self._unroll):
                index = min_s32(base + Int32(u * self._threads) + thread, last)
                for source, system in sources:
                    addrs.append(source + Int64(index) * Int64(PACK_BYTES))
                    flags.append(system)
            words = ld_v4_u32_batch(addrs, flags, zero)
            for u in cutlass.range_constexpr(self._unroll):
                index = base + Int32(u * self._threads) + thread
                if index < packs:
                    packed = self._sum(words[u * count:(u + 1) * count])
                    offset = Int64(index) * Int64(PACK_BYTES)
                    st_global_v4_u32(dst + offset, packed[0], packed[1], packed[2], packed[3])
                    st_global_v4_u32(dst2 + offset, packed[0], packed[1], packed[2], packed[3])
            base = base + Int32(self._unroll * self._threads)

    # -- roles --------------------------------------------------------------------------------

    @cute.jit
    def _own_partials(self, link: cutlass.Constexpr[int], sub: Int32, input_base: Int64, chunk_packs: Int32,
                      stride_packs: Int32, piece_packs: Int32, base_own: Uint32, link_base: Int64, ctrl_base: Int64,
                      poison_ptr: Int64, spin_limit: Uint32, limit_us: Uint32, zero: Uint32) -> None:
        """The first rank of a link: stage its own values of every owner's piece as the partials."""
        tidx, _, _ = cute.arch.thread_idx()
        thread = Int32(tidx)
        per_round = self._world - 1
        pieces = (chunk_packs + piece_packs - Int32(1)) // piece_packs
        items = pieces * Int32(per_round)
        q = sub
        while q < items:
            g = base_own + Uint32(q)
            m = g % Uint32(self._slots)
            tag = g + Uint32(1)
            if thread == Int32(self._threads - 1):
                sent = link_base + Int64(self._layout.sent_off + link * FLAG_STRIDE)
                if spin_until_ge_timed_sys(sent, tag - Uint32(self._slots), spin_limit, limit_us) != Uint32(0):
                    self._fail(ctrl_base, poison_ptr, Int32(self._rank), Int32(255), int(ErrorKind.CHAIN_SLOT), tag)
            cute.arch.sync_threads()
            if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                p = q // Int32(per_round)
                r = q - p * Int32(per_round)
                first = p * piece_packs
                count = chunk_packs - first
                if piece_packs < count:
                    count = piece_packs
                # Owner chain index W - 1 - r (link 0, from index 0) or r (link 1, from index W - 1).
                position = Int32(0)
                for k in cutlass.range_constexpr(per_round):
                    owner = self._world - 1 - k if link == 0 else k
                    if r == Int32(k):
                        position = Int32(self._order[owner])
                src = input_base + (Int64(position) * Int64(stride_packs) + Int64(first)) * Int64(PACK_BYTES)
                self._pass(src, Int64(0), Int64(0), False, False, False, 1,
                           self._slot(link_base, self._layout.own_off, link, m), count, zero)
                cute.arch.sync_threads()
                if thread == Int32(0):
                    fence_sc_sys()
                    st_relaxed_sys_u32(self._word(link_base, self._layout.ready_off, link, m), tag)
                q = q + Int32(self._nb)
            else:
                q = items

    @cute.jit
    def _inbound(self, link: cutlass.Constexpr[int], sub: Int32, input_base: Int64, output_base: Int64,
                 scratch_base: Int64, chunk_packs: Int32, stride_packs: Int32, piece_packs: Int32, base_in: Uint32,
                 base_own: Uint32, link_base: Int64, piece_counters: Int64, decisions: Int64, ctrl_base: Int64,
                 poison_ptr: Int64, spin_limit: Uint32, limit_us: Uint32, zero: Uint32) -> None:
        """Every inbound partial of ``link``: add the own values and stage it for the next rank, or,
        for this rank's own chunk, keep it for the owner's result."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        thread = Int32(tidx)
        j = self._index
        # Inbound round: link 0 brings owners W - 1 down to j, link 1 owners 0 up to j; the last
        # item of a round is this rank's own chunk.
        per_round = self._world - j if link == 0 else j + 1
        out_round = per_round - 1
        both_sides = 0 < j < self._world - 1
        peer = self._prev if link == 0 else self._next
        pieces = (chunk_packs + piece_packs - Int32(1)) // piece_packs
        items = pieces * Int32(per_round)
        own_pos = self._order[j]
        i = sub
        while i < items:
            g = base_in + Uint32(i)
            m = g % Uint32(self._slots)
            tag = g + Uint32(1)
            p = i // Int32(per_round)
            r = i - p * Int32(per_round)
            forward = r < Int32(out_round)
            q = p * Int32(out_round) + r
            g_own = base_own + Uint32(q)
            if thread < Int32(self._lanes):
                if spin_until_eq_timed_sys(self._flag(link_base, link, m, thread), tag, spin_limit,
                                           limit_us) != Uint32(0):
                    self._fail(ctrl_base, poison_ptr, Int32(peer), thread, int(ErrorKind.CHAIN_CHUNK), tag)
            if cutlass.const_expr(out_round > 0):
                if forward:
                    if thread == Int32(self._threads - 1):
                        sent = link_base + Int64(self._layout.sent_off + link * FLAG_STRIDE)
                        if spin_until_ge_timed_sys(sent, g_own + Uint32(1) - Uint32(self._slots), spin_limit,
                                                   limit_us) != Uint32(0):
                            self._fail(ctrl_base, poison_ptr, Int32(self._rank), Int32(255),
                                       int(ErrorKind.CHAIN_SLOT), g_own + Uint32(1))
            cute.arch.sync_threads()
            if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                first = p * piece_packs
                count = chunk_packs - first
                if piece_packs < count:
                    count = piece_packs
                slot = self._slot(link_base, self._layout.recv_off, link, m)
                piece_out = output_base + Int64(first) * Int64(PACK_BYTES)
                own_values = input_base + (Int64(own_pos) * Int64(stride_packs) + Int64(first)) * Int64(PACK_BYTES)
                if forward:
                    position = Int32(0)
                    for k in cutlass.range_constexpr(out_round):
                        owner = self._world - 1 - k if link == 0 else k
                        if r == Int32(k):
                            position = Int32(self._order[owner])
                    src = input_base + (Int64(position) * Int64(stride_packs) + Int64(first)) * Int64(PACK_BYTES)
                    m_own = g_own % Uint32(self._slots)
                    self._pass(slot, src, Int64(0), True, False, False, 2,
                               self._slot(link_base, self._layout.own_off, link, m_own), count, zero)
                    cute.arch.sync_threads()
                    if thread == Int32(0):
                        fence_sc_sys()
                        st_relaxed_sys_u32(self._word(link_base, self._layout.ready_off, link, m_own),
                                           g_own + Uint32(1))
                        st_relaxed_sys_u32(self._word(link_base, self._layout.consumed_off, link, m), tag)
                else:
                    if cutlass.const_expr(not both_sides):
                        # An end of the chain: the one partial completes the sum (L + x, or x + R).
                        if cutlass.const_expr(link == 0):
                            self._pass(slot, own_values, Int64(0), True, False, False, 2, piece_out, count, zero)
                        else:
                            self._pass(own_values, slot, Int64(0), False, True, False, 2, piece_out, count, zero)
                        cute.arch.sync_threads()
                        if thread == Int32(0):
                            fence_sc_sys()
                            st_relaxed_sys_u32(self._word(link_base, self._layout.consumed_off, link, m), tag)
                    else:
                        piece_scratch = scratch_base + Int64(first) * Int64(PACK_BYTES)
                        keep = piece_out if link == 0 else piece_scratch
                        self._pass(slot, Int64(0), Int64(0), True, False, False, 1, keep, count, zero)
                        cute.arch.sync_threads()
                        decision = decisions + Int64(4) * Int64(bidx)
                        if thread == Int32(0):
                            # The staged copy is complete before the slot is released and before the
                            # other link's block can see this arrival.
                            fence_sc_sys()
                            st_relaxed_sys_u32(self._word(link_base, self._layout.consumed_off, link, m), tag)
                            prior = atomic_add_relaxed_gpu_u32(piece_counters + Int64(4) * Int64(p), Uint32(1))
                            fence_sc_gpu()
                            st_release_gpu_u32(decision, prior & Uint32(1))
                        cute.arch.sync_threads()
                        if ld_relaxed_gpu_u32(decision) == Uint32(1):
                            fence_sc_gpu()
                            # Both partials are kept: L in the output, R in the scratch buffer.
                            self._pass(piece_out, own_values, piece_scratch, True, False, True, 3, piece_out,
                                       count, zero)
                        cute.arch.sync_threads()
                i = i + Int32(self._nb)
            else:
                i = items

    # -- launch -------------------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        scratch_ptr: cute.Pointer,
        chunk_packs: Int32,
        stride_packs: Int32,
        piece_packs: Int32,
        link_base: Int64,
        counters: Int64,
        piece_counters: Int64,
        ctrl_base: Int64,
        poison_ptr: Int64,
        spin_limit: Uint32,
        stream: cuda.CUstream,
    ) -> None:
        """Host entry: two roles (link 0 and link 1) of ``blocks_per_role`` blocks."""
        self.kernel(
            input_ptr, output_ptr, scratch_ptr, chunk_packs, stride_packs, piece_packs, link_base, counters,
            piece_counters, ctrl_base, poison_ptr, spin_limit,
        ).launch(grid=(2 * self._nb, 1, 1), block=[self._threads, 1, 1], cluster=(1, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        scratch_ptr: cute.Pointer,
        chunk_packs: Int32,
        stride_packs: Int32,
        piece_packs: Int32,
        link_base: Int64,
        counters: Int64,
        piece_counters: Int64,
        ctrl_base: Int64,
        poison_ptr: Int64,
        spin_limit: Uint32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        input_base = Int64(input_ptr.toint())
        output_base = Int64(output_ptr.toint())
        scratch_base = Int64(scratch_ptr.toint())
        role = Int32(bidx) // Int32(self._nb)
        sub = Int32(bidx) - role * Int32(self._nb)
        zero = Uint32(chunk_packs) >> Uint32(31)
        j = self._index
        last = self._world - 1
        decisions = piece_counters + Int64(4 * PIECE_COUNTERS)
        if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
            own0 = ld_relaxed_gpu_u32(counters)
            own1 = ld_relaxed_gpu_u32(counters + Int64(4))
            in0 = ld_relaxed_gpu_u32(counters + Int64(8))
            in1 = ld_relaxed_gpu_u32(counters + Int64(12))
            seq = ld_relaxed_gpu_u32(counters + Int64(16)) + Uint32(1)
            if Int32(bidx) == Int32(0):
                if Int32(tidx) == Int32(0):
                    params = link_base + Int64(self._layout.ctrl_off + 4) + Int64(seq & Uint32(1)) * Int64(16)
                    st_relaxed_sys_u32(params, Uint32(int(LinkOp.REDUCE_SCATTER)))
                    st_relaxed_sys_u32(params + Int64(4), Uint32(chunk_packs * Int32(PACK_BYTES)))
                    st_relaxed_sys_u32(params + Int64(8), Uint32(piece_packs * Int32(PACK_BYTES)))
                    fence_sc_sys()
                    st_relaxed_sys_u32(link_base + Int64(self._layout.ctrl_off), seq)
            limit_us = ld_relaxed_sys_u32(ctrl_base + Int64(_WAIT_LIMIT))
            if role == Int32(0):
                if cutlass.const_expr(j == 0):
                    self._own_partials(0, sub, input_base, chunk_packs, stride_packs, piece_packs, own0, link_base,
                                       ctrl_base, poison_ptr, spin_limit, limit_us, zero)
                else:
                    self._inbound(0, sub, input_base, output_base, scratch_base, chunk_packs, stride_packs,
                                  piece_packs, in0, own0, link_base, piece_counters, decisions, ctrl_base,
                                  poison_ptr, spin_limit, limit_us, zero)
            if role == Int32(1):
                if cutlass.const_expr(j == last):
                    self._own_partials(1, sub, input_base, chunk_packs, stride_packs, piece_packs, own1, link_base,
                                       ctrl_base, poison_ptr, spin_limit, limit_us, zero)
                else:
                    self._inbound(1, sub, input_base, output_base, scratch_base, chunk_packs, stride_packs,
                                  piece_packs, in1, own1, link_base, piece_counters, decisions, ctrl_base,
                                  poison_ptr, spin_limit, limit_us, zero)
            # The last block to finish advances the item bases and the sequence.
            fence_sc_gpu()
            cute.arch.sync_threads()
            if Int32(tidx) == Int32(0):
                prior = atomic_add_relaxed_gpu_u32(counters + Int64(24), Uint32(1))
                if prior + Uint32(1) == Uint32(gdim):
                    # Every block of this launch arrived: return the tail to 0, so the next launch of this
                    # kernel finds its last block by the same count whatever its grid.
                    st_release_gpu_u32(counters + Int64(24), Uint32(0))
                    fence_sc_gpu()
                    if ld_relaxed_sys_u32(ctrl_base + Int64(_ERROR_SEQ)) == Uint32(0):
                        pieces = Uint32((chunk_packs + piece_packs - Int32(1)) // piece_packs)
                        if cutlass.const_expr(j < last):
                            st_release_gpu_u32(counters, own0 + pieces * Uint32(last - j))
                            st_release_gpu_u32(counters + Int64(12), in1 + pieces * Uint32(j + 1))
                        if cutlass.const_expr(j > 0):
                            st_release_gpu_u32(counters + Int64(4), own1 + pieces * Uint32(j))
                            st_release_gpu_u32(counters + Int64(8), in0 + pieces * Uint32(self._world - j))
                        st_release_gpu_u32(counters + Int64(16), seq)


# -- ring collectives ---------------------------------------------------------------------------

RING_MODES = ("gather", "scatter", "reduce")
_RING_TAILS = {"scatter": 28, "gather": 48, "reduce": 52}      # byte offsets of each mode's tail word


class LinkRing(_LinkKernel):
    """One kernel specialization of a ring collective over the chain closed by its last rank's
    lanes to its first: ``mode`` ``gather`` (all-gather), ``scatter`` (reduce-scatter) or ``reduce``
    (all-reduce: the reduce-scatter, then each owner's result travels around as the all-gather).

    The rank at ring index ``i`` (its chain index) owns chunk ``order[i]`` of the input (and of the
    all-reduce's message). Link 2 carries partial sums to the ring's next rank: in round ``t`` the
    rank at ``i`` starts the partial of piece ``t`` for the owner at ``i - 1`` from its own values, and
    sends its relay ``r`` of piece ``t - (r + 1) D``: its own values added to the partial for the owner
    at ``i - 2 - r`` it received ``D`` rounds earlier (``D``, the stagger, a launch argument; the op has
    ``(W - 2) D`` more rounds, whose items outside a type's pieces are empty). The last inbound partial
    of a piece is its own chunk's, which it completes. Owner ``k``'s piece is
    therefore the dtype rounding at every hop of the values of ring indices ``k + 1``, ``k + 2``,
    ..., ``k`` added in that order. Link 3 carries finished pieces: in round ``t`` the own piece ``t``
    (the all-gather's input, or the all-reduce's own result), then the native progress thread forwards
    the pieces of the owners at ``i - 1`` down to ``i - W + 2``, forward ``r`` carrying piece
    ``t - (r + 1) D3`` received ``D3`` rounds earlier (``D3``, the all-gather's stagger, a launch
    argument; link 3 has ``(W - 2) D3`` more rounds); the kernel copies every received piece to its
    owner's place.

    Roles of ``blocks_per_role`` blocks: start (link 2, the partial for owner ``i - 1``), relay
    (link 2 inbound: add and pass on, or complete), own (link 3, all-gather only) and copy (link 3
    inbound).
    """

    def __init__(self, mode: str, dtype_name: str, world_size: int, chain_index: int, rank: int,
                 order: Sequence[int], threads: int, lanes: int, slots: int, slot_bytes: int,
                 blocks_per_role: int, unroll: int, trace_capacity: int = 0) -> None:
        if mode not in RING_MODES:
            raise ValueError(f"unknown ring mode {mode!r}")
        if dtype_name not in DTYPE_NAMES and not (mode == "gather" and dtype_name == "bytes"):
            raise ValueError(f"unsupported ring {mode} dtype {dtype_name!r}")
        self._mode = mode
        self._dtype_name = dtype_name
        world = int(world_size)
        prev_rank = order[(chain_index - 1) % world]
        next_rank = order[(chain_index + 1) % world]
        self._setup(world, chain_index, prev_rank, next_rank, rank, order, threads, lanes, slots, slot_bytes,
                    blocks_per_role, unroll)
        self._roles = (("start", "relay") if mode == "scatter" else ("own", "copy") if mode == "gather"
                       else ("start", "relay", "copy"))
        if not 0 <= int(trace_capacity) < 1 << 31:
            raise ValueError(f"ring kernel trace capacity {trace_capacity}")
        self._trace_capacity = int(trace_capacity)

    _sum = LinkScatter._sum
    _pass = LinkScatter._pass
    _pass_two = LinkScatter._pass_two

    def _position(self, offset: int) -> int:
        """The output place (rank) of the owner at ring index ``i + offset``."""
        return self._order[(self._index + offset) % self._world]

    @cute.jit
    def _owner_place(self, r: Int32, first: cutlass.Constexpr[int], step: cutlass.Constexpr[int],
                     count: cutlass.Constexpr[int]) -> Int32:
        """``order[i + first + step * r]`` for a runtime ``r`` below ``count``."""
        place = Int32(0)
        for k in cutlass.range_constexpr(count):
            if r == Int32(k):
                place = Int32(self._position(first + step * k))
        return place

    @cute.jit
    def _wait_own_slot(self, link: cutlass.Constexpr[int], item: Uint32, link_base: Int64, ctrl_base: Int64,
                       poison_ptr: Int64, spin_limit: Uint32, limit_us: Uint32) -> None:
        """The block's last thread waits until own slot ``item % slots`` of ``link`` is free."""
        tidx, _, _ = cute.arch.thread_idx()
        if Int32(tidx) == Int32(self._threads - 1):
            sent = link_base + Int64(self._layout.sent_off + link * FLAG_STRIDE)
            tag = item + Uint32(1)
            if spin_until_ge_timed_sys(sent, tag - Uint32(self._slots), spin_limit, limit_us) != Uint32(0):
                self._fail(ctrl_base, poison_ptr, Int32(self._rank), Int32(255), int(ErrorKind.CHAIN_SLOT), tag)

    @cute.jit
    def _wait_inbound(self, link: cutlass.Constexpr[int], g: Uint32, link_base: Int64, ctrl_base: Int64,
                      poison_ptr: Int64, spin_limit: Uint32, limit_us: Uint32) -> None:
        """Threads ``0 .. L - 1`` (warp 0) wait for every lane's flag of inbound item ``g``."""
        tidx, _, _ = cute.arch.thread_idx()
        thread = Int32(tidx)
        if thread < Int32(self._lanes):
            m = g % Uint32(self._slots)
            if spin_until_eq_timed_sys(self._flag(link_base, link, m, thread), g + Uint32(1), spin_limit,
                                       limit_us) != Uint32(0):
                self._fail(ctrl_base, poison_ptr, Int32(self._prev), thread, int(ErrorKind.CHAIN_CHUNK),
                           g + Uint32(1))

    @cute.jit
    def _publish(self, link: cutlass.Constexpr[int], area_off: cutlass.Constexpr[int], item: Uint32,
                 link_base: Int64) -> None:
        """Thread 0 publishes ``item`` in the ready (own slots) or consumed (receive slots) words."""
        tidx, _, _ = cute.arch.thread_idx()
        if Int32(tidx) == Int32(0):
            fence_sc_sys()
            st_relaxed_sys_u32(self._word(link_base, area_off, link, item % Uint32(self._slots)),
                               item + Uint32(1))

    # -- roles --------------------------------------------------------------------------------

    @cute.jit
    def _start(self, sub: Int32, input_base: Int64, chunk_packs: Int32, stride_packs: Int32, piece_packs: Int32,
               own2: Uint32, link_base: Int64, ctrl_base: Int64, poison_ptr: Int64, spin_limit: Uint32,
               limit_us: Uint32, zero: Uint32, trace_base: Int64) -> None:
        """Link 2, outbound item 0 of every round: this rank's values of owner ``i - 1``'s piece."""
        pieces = (chunk_packs + piece_packs - Int32(1)) // piece_packs
        per_round = self._world - 1
        place = self._position(-1)
        p = sub
        while p < pieces:
            item = own2 + Uint32(p * Int32(per_round))
            self._wait_own_slot(2, item, link_base, ctrl_base, poison_ptr, spin_limit, limit_us)
            cute.arch.sync_threads()
            if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                self._trace(trace_base, int(TraceEvent.KERNEL_SLOT), 2, item + Uint32(1))
                first = p * piece_packs
                count = min_s32(piece_packs, chunk_packs - first)
                src = input_base + (Int64(place) * Int64(stride_packs) + Int64(first)) * Int64(PACK_BYTES)
                self._pass(src, Int64(0), Int64(0), False, False, False, 1,
                           self._slot(link_base, self._layout.own_off, 2, item % Uint32(self._slots)), count, zero)
                cute.arch.sync_threads()
                self._publish(2, self._layout.ready_off, item, link_base)
                self._trace(trace_base, int(TraceEvent.KERNEL_READY), 2, item + Uint32(1))
                p = p + Int32(self._nb)
            else:
                p = pieces

    @cute.jit
    def _relay(self, sub: Int32, input_base: Int64, output_base: Int64, chunk_packs: Int32, stride_packs: Int32,
               piece_packs: Int32, own2: Uint32, in2: Uint32, own3: Uint32, in3: Uint32, link_base: Int64,
               ctrl_base: Int64, poison_ptr: Int64, spin_limit: Uint32, limit_us: Uint32, zero: Uint32,
               trace_base: Int64, stagger: Int32) -> None:
        """Link 2 inbound: add this rank's values and pass the partial on, or complete its own piece
        (the reduce-scatter's output; the all-reduce's own result, also staged on link 3)."""
        per_round = self._world - 1
        pieces = (chunk_packs + piece_packs - Int32(1)) // piece_packs
        items = (pieces + stagger * Int32(per_round - 1)) * Int32(per_round)
        own_place = self._position(0)
        reduce = self._mode == "reduce"
        i = sub
        while i < items:
            g = in2 + Uint32(i)
            t = i // Int32(per_round)
            r = i - t * Int32(per_round)
            p = t - r * stagger                              # the piece inbound item i carries
            full = Uint32(p) < Uint32(pieces)                # an empty item carries flags only
            last = r == Int32(per_round - 1)
            # The relay of inbound type r leaves D rounds later as outbound type r + 1.
            out_item = own2 + Uint32(i + stagger * Int32(per_round) + Int32(1))
            self._wait_inbound(2, g, link_base, ctrl_base, poison_ptr, spin_limit, limit_us)
            self._trace_flags(trace_base, 2, g + Uint32(1))
            if full:
                if last:
                    if cutlass.const_expr(reduce):
                        self._wait_own_slot(3, own3 + Uint32(p), link_base, ctrl_base, poison_ptr, spin_limit,
                                            limit_us)
                else:
                    self._wait_own_slot(2, out_item, link_base, ctrl_base, poison_ptr, spin_limit, limit_us)
            cute.arch.sync_threads()
            if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                if full:
                    # The link-3 outbound index of this rank's own item of piece p (link 3's outbound and
                    # inbound items advance together, so in3 is its first outbound item of the op).
                    tag3 = in3 + Uint32(p * Int32(per_round)) + Uint32(1)
                    if last:
                        if cutlass.const_expr(reduce):
                            self._trace(trace_base, int(TraceEvent.KERNEL_SLOT), 3, tag3)
                    else:
                        self._trace(trace_base, int(TraceEvent.KERNEL_SLOT), 2, out_item + Uint32(1))
                    first = p * piece_packs
                    count = min_s32(piece_packs, chunk_packs - first)
                    slot = self._slot(link_base, self._layout.recv_off, 2, g % Uint32(self._slots))
                    if last:
                        own_values = input_base + (Int64(own_place) * Int64(stride_packs) + Int64(first)) * Int64(PACK_BYTES)
                        if cutlass.const_expr(reduce):
                            # The result goes to its place in the message and, as link 3's own item, around:
                            # one pass stores both, and the item is published right after it.
                            result = output_base + (Int64(own_place) * Int64(stride_packs) + Int64(first)) * Int64(PACK_BYTES)
                            staged = self._slot(link_base, self._layout.own_off, 3, (own3 + Uint32(p)) % Uint32(self._slots))
                            self._pass_two(slot, own_values, True, False, 2, staged, result, count, zero)
                            cute.arch.sync_threads()
                            self._publish(3, self._layout.ready_off, own3 + Uint32(p), link_base)
                            self._trace(trace_base, int(TraceEvent.KERNEL_READY), 3, tag3)
                        else:
                            result = output_base + Int64(first) * Int64(PACK_BYTES)
                            self._pass(slot, own_values, Int64(0), True, False, False, 2, result, count, zero)
                            cute.arch.sync_threads()
                    else:
                        # Inbound item r carries owner i - 2 - r.
                        place = self._owner_place(r, -2, -1, per_round - 1)
                        values = input_base + (Int64(place) * Int64(stride_packs) + Int64(first)) * Int64(PACK_BYTES)
                        staged = self._slot(link_base, self._layout.own_off, 2, out_item % Uint32(self._slots))
                        self._pass(slot, values, Int64(0), True, False, False, 2, staged, count, zero)
                        cute.arch.sync_threads()
                        self._publish(2, self._layout.ready_off, out_item, link_base)
                        self._trace(trace_base, int(TraceEvent.KERNEL_READY), 2, out_item + Uint32(1))
                self._publish(2, self._layout.consumed_off, g, link_base)
                self._trace(trace_base, int(TraceEvent.KERNEL_CONSUMED), 2, g + Uint32(1))
                i = i + Int32(self._nb)
            else:
                i = items

    @cute.jit
    def _own(self, sub: Int32, input_base: Int64, output_base: Int64, chunk_packs: Int32, piece_packs: Int32,
             own3: Uint32, in3: Uint32, link_base: Int64, ctrl_base: Int64, poison_ptr: Int64, spin_limit: Uint32,
             limit_us: Uint32, zero: Uint32, trace_base: Int64) -> None:
        """Link 3, outbound item 0 of every round (all-gather): the own piece, also to the output."""
        pieces = (chunk_packs + piece_packs - Int32(1)) // piece_packs
        own_place = self._position(0)
        p = sub
        while p < pieces:
            item = own3 + Uint32(p)
            self._wait_own_slot(3, item, link_base, ctrl_base, poison_ptr, spin_limit, limit_us)
            cute.arch.sync_threads()
            if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                tag3 = in3 + Uint32(p * Int32(self._world - 1)) + Uint32(1)
                self._trace(trace_base, int(TraceEvent.KERNEL_SLOT), 3, tag3)
                first = p * piece_packs
                count = min_s32(piece_packs, chunk_packs - first)
                src = input_base + Int64(first) * Int64(PACK_BYTES)
                staged = self._slot(link_base, self._layout.own_off, 3, item % Uint32(self._slots))
                out = output_base + (Int64(own_place) * Int64(chunk_packs) + Int64(first)) * Int64(PACK_BYTES)
                self._pass_two(src, Int64(0), False, False, 1, staged, out, count, zero)
                cute.arch.sync_threads()
                self._publish(3, self._layout.ready_off, item, link_base)
                self._trace(trace_base, int(TraceEvent.KERNEL_READY), 3, tag3)
                p = p + Int32(self._nb)
            else:
                p = pieces

    @cute.jit
    def _copy(self, sub: Int32, output_base: Int64, chunk_packs: Int32, stride_packs: Int32, piece_packs: Int32,
              in3: Uint32, link_base: Int64, ctrl_base: Int64, poison_ptr: Int64, spin_limit: Uint32,
              limit_us: Uint32, zero: Uint32, trace_base: Int64, gather_stagger: Int32) -> None:
        """Link 3 inbound: copy every finished piece (owner ``i - 1 - r``) to its owner's place; inbound
        type ``r`` of round ``t`` carries piece ``t - r D3``, and items outside the pieces are empty."""
        per_round = self._world - 1
        pieces = (chunk_packs + piece_packs - Int32(1)) // piece_packs
        items = (pieces + gather_stagger * Int32(per_round - 1)) * Int32(per_round)
        i = sub
        while i < items:
            g = in3 + Uint32(i)
            t = i // Int32(per_round)
            r = i - t * Int32(per_round)
            p = t - r * gather_stagger                       # the piece inbound item i carries
            full = Uint32(p) < Uint32(pieces)                # an empty item carries flags only
            self._wait_inbound(3, g, link_base, ctrl_base, poison_ptr, spin_limit, limit_us)
            self._trace_flags(trace_base, 3, g + Uint32(1))
            cute.arch.sync_threads()
            if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                if full:
                    first = p * piece_packs
                    count = min_s32(piece_packs, chunk_packs - first)
                    place = self._owner_place(r, -1, -1, per_round)
                    out = output_base + (Int64(place) * Int64(stride_packs) + Int64(first)) * Int64(PACK_BYTES)
                    slot = self._slot(link_base, self._layout.recv_off, 3, g % Uint32(self._slots))
                    self._pass(slot, Int64(0), Int64(0), True, False, False, 1, out, count, zero)
                    cute.arch.sync_threads()
                self._publish(3, self._layout.consumed_off, g, link_base)
                self._trace(trace_base, int(TraceEvent.KERNEL_CONSUMED), 3, g + Uint32(1))
                i = i + Int32(self._nb)
            else:
                i = items

    # -- launch -------------------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        chunk_packs: Int32,
        stride_packs: Int32,
        piece_packs: Int32,
        link_base: Int64,
        counters: Int64,
        ctrl_base: Int64,
        poison_ptr: Int64,
        spin_limit: Uint32,
        trace_base: Int64,
        stagger: Int32,
        gather_stagger: Int32,
        stream: cuda.CUstream,
    ) -> None:
        """Host entry: one block per (role, item lane)."""
        self.kernel(
            input_ptr, output_ptr, chunk_packs, stride_packs, piece_packs, link_base, counters, ctrl_base,
            poison_ptr, spin_limit, trace_base, stagger, gather_stagger,
        ).launch(grid=(len(self._roles) * self._nb, 1, 1), block=[self._threads, 1, 1], cluster=(1, 1, 1),
                 stream=stream)

    @cute.kernel
    def kernel(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        chunk_packs: Int32,
        stride_packs: Int32,
        piece_packs: Int32,
        link_base: Int64,
        counters: Int64,
        ctrl_base: Int64,
        poison_ptr: Int64,
        spin_limit: Uint32,
        trace_base: Int64,
        stagger: Int32,
        gather_stagger: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        input_base = Int64(input_ptr.toint())
        output_base = Int64(output_ptr.toint())
        role = Int32(bidx) // Int32(self._nb)
        sub = Int32(bidx) - role * Int32(self._nb)
        zero = Uint32(chunk_packs) >> Uint32(31)
        op = {"gather": LinkOp.RING_GATHER, "scatter": LinkOp.RING_SCATTER, "reduce": LinkOp.RING_REDUCE}[self._mode]
        if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
            # Every block reads the counters before any block can advance them.
            own2 = ld_relaxed_gpu_u32(counters + Int64(32))
            in2 = ld_relaxed_gpu_u32(counters + Int64(36))
            own3 = ld_relaxed_gpu_u32(counters + Int64(40))
            in3 = ld_relaxed_gpu_u32(counters + Int64(44))
            seq = ld_relaxed_gpu_u32(counters + Int64(16)) + Uint32(1)
            self._trace(trace_base, int(TraceEvent.KERNEL_START), 0, Uint32(bidx))
            if Int32(bidx) == Int32(0):
                if Int32(tidx) == Int32(0):
                    params = link_base + Int64(self._layout.ctrl_off + 4) + Int64(seq & Uint32(1)) * Int64(16)
                    word = Uint32(int(op))
                    if cutlass.const_expr(self._mode != "gather"):
                        # The all-gather has no partials to stagger.
                        word = word | (Uint32(stagger) << Uint32(RING_STAGGER_SHIFT))
                    if cutlass.const_expr(self._mode != "scatter"):
                        # The reduce-scatter has no finished pieces to forward.
                        word = word | (Uint32(gather_stagger) << Uint32(RING_GATHER_STAGGER_SHIFT))
                    st_relaxed_sys_u32(params, word)
                    st_relaxed_sys_u32(params + Int64(4), Uint32(chunk_packs * Int32(PACK_BYTES)))
                    st_relaxed_sys_u32(params + Int64(8), Uint32(piece_packs * Int32(PACK_BYTES)))
                    fence_sc_sys()
                    st_relaxed_sys_u32(link_base + Int64(self._layout.ctrl_off), seq)
                    self._trace(trace_base, int(TraceEvent.KERNEL_BELL), 0, seq)
            limit_us = ld_relaxed_sys_u32(ctrl_base + Int64(_WAIT_LIMIT))
            for index, name in enumerate(self._roles):
                if role == Int32(index):
                    if cutlass.const_expr(name == "start"):
                        self._start(sub, input_base, chunk_packs, stride_packs, piece_packs, own2, link_base,
                                    ctrl_base, poison_ptr, spin_limit, limit_us, zero, trace_base)
                    if cutlass.const_expr(name == "relay"):
                        self._relay(sub, input_base, output_base, chunk_packs, stride_packs, piece_packs, own2, in2,
                                    own3, in3, link_base, ctrl_base, poison_ptr, spin_limit, limit_us, zero,
                                    trace_base, stagger)
                    if cutlass.const_expr(name == "own"):
                        self._own(sub, input_base, output_base, chunk_packs, piece_packs, own3, in3, link_base,
                                  ctrl_base, poison_ptr, spin_limit, limit_us, zero, trace_base)
                    if cutlass.const_expr(name == "copy"):
                        self._copy(sub, output_base, chunk_packs, stride_packs if self._mode == "reduce"
                                   else chunk_packs, piece_packs, in3, link_base, ctrl_base, poison_ptr, spin_limit,
                                   limit_us, zero, trace_base, gather_stagger)
            # The last block to finish advances the item bases and the sequence.
            fence_sc_gpu()
            cute.arch.sync_threads()
            if Int32(tidx) == Int32(0):
                prior = atomic_add_relaxed_gpu_u32(counters + Int64(_RING_TAILS[self._mode]), Uint32(1))
                if prior + Uint32(1) == Uint32(gdim):
                    # Every block of this launch arrived: return the tail to 0, so the next launch of this
                    # kernel finds its last block by the same count whatever its grid.
                    st_release_gpu_u32(counters + Int64(_RING_TAILS[self._mode]), Uint32(0))
                    fence_sc_gpu()
                    if ld_relaxed_sys_u32(ctrl_base + Int64(_ERROR_SEQ)) == Uint32(0):
                        pieces = Uint32((chunk_packs + piece_packs - Int32(1)) // piece_packs)
                        steps = Uint32(self._world - 1)
                        if cutlass.const_expr(self._mode != "gather"):
                            rounds = pieces + Uint32(stagger) * Uint32(self._world - 2)
                            st_release_gpu_u32(counters + Int64(32), own2 + rounds * steps)
                            st_release_gpu_u32(counters + Int64(36), in2 + rounds * steps)
                        if cutlass.const_expr(self._mode != "scatter"):
                            rounds3 = pieces + Uint32(gather_stagger) * Uint32(self._world - 2)
                            st_release_gpu_u32(counters + Int64(40), own3 + rounds3)
                            st_release_gpu_u32(counters + Int64(44), in3 + rounds3 * steps)
                        st_release_gpu_u32(counters + Int64(16), seq)


_LAUNCHERS: dict[tuple, Callable[..., None]] = {}


def launcher_key(world_size: int, chain_index: int, prev_rank: int, next_rank: int, rank: int,
                 order: Sequence[int], threads: int, lanes: int, slots: int, slot_bytes: int, blocks_per_role: int,
                 unroll: int, device_index: int) -> tuple:
    return ("link-gather", int(world_size), int(chain_index), int(prev_rank), int(next_rank), int(rank),
            tuple(int(r) for r in order), int(threads), int(lanes), int(slots), int(slot_bytes),
            int(blocks_per_role), int(unroll), int(device_index))


def get_gather_launcher(world_size: int, chain_index: int, prev_rank: int, next_rank: int, rank: int,
                        order: Sequence[int], threads: int, lanes: int, slots: int, slot_bytes: int,
                        blocks_per_role: int, unroll: int, device_index: int) -> Callable[..., None]:
    """The compiled chain all-gather launcher of one specialization (compiled once per process).

    Launch arguments: input address, output address, packs per rank, packs per
    piece, link area address, device counter address, control line address,
    poison address, spin limit.
    """
    key = launcher_key(world_size, chain_index, prev_rank, next_rank, rank, order, threads, lanes, slots,
                       slot_bytes, blocks_per_role, unroll, device_index)
    cached = _LAUNCHERS.get(key)
    if cached is not None:
        return cached
    launch = LinkGather(world_size, chain_index, prev_rank, next_rank, rank, order, threads, lanes, slots,
                        slot_bytes, blocks_per_role, unroll)
    compiled = compile_launcher(
        launch, make_pointer(16), make_pointer(16), 1, 1, 16, 16, 16, 16, 1, current_cuda_stream(),
        name="sircl chain all-gather", cache_key=key,
    )

    def run(input_address: int, output_address: int, shard_packs: int, piece_packs: int, link_base: int,
            counters: int, ctrl_base: int, poison_address: int, spin_limit: int) -> None:
        compiled(
            make_pointer(input_address), make_pointer(output_address), int(shard_packs), int(piece_packs),
            int(link_base), int(counters), int(ctrl_base), int(poison_address), int(spin_limit),
            current_cuda_stream(),
        )

    _LAUNCHERS[key] = run
    return run


def scatter_launcher_key(dtype_name: str, world_size: int, chain_index: int, prev_rank: int, next_rank: int,
                         rank: int, order: Sequence[int], threads: int, lanes: int, slots: int, slot_bytes: int,
                         blocks_per_role: int, unroll: int, device_index: int) -> tuple:
    return ("link-scatter", str(dtype_name), int(world_size), int(chain_index), int(prev_rank), int(next_rank),
            int(rank), tuple(int(r) for r in order), int(threads), int(lanes), int(slots), int(slot_bytes),
            int(blocks_per_role), int(unroll), int(device_index))


def get_scatter_launcher(dtype_name: str, world_size: int, chain_index: int, prev_rank: int, next_rank: int,
                         rank: int, order: Sequence[int], threads: int, lanes: int, slots: int, slot_bytes: int,
                         blocks_per_role: int, unroll: int, device_index: int) -> Callable[..., None]:
    """The compiled chain reduce-scatter launcher of one specialization (compiled once per process).

    Launch arguments: input address, output address, scratch address (one chunk), packs per chunk,
    packs between chunks of the input, packs per piece, link area address, device counter address,
    piece counter address (``PIECE_COUNTERS + DECISION_WORDS`` words), control line address,
    poison address, spin limit.
    """
    key = scatter_launcher_key(dtype_name, world_size, chain_index, prev_rank, next_rank, rank, order, threads,
                               lanes, slots, slot_bytes, blocks_per_role, unroll, device_index)
    cached = _LAUNCHERS.get(key)
    if cached is not None:
        return cached
    launch = LinkScatter(dtype_name, world_size, chain_index, prev_rank, next_rank, rank, order, threads, lanes,
                         slots, slot_bytes, blocks_per_role, unroll)
    compiled = compile_launcher(
        launch, make_pointer(16), make_pointer(16), make_pointer(16), 1, 1, 1, 16, 16, 16, 16, 16, 1,
        current_cuda_stream(), name="sircl chain reduce-scatter", cache_key=key,
    )

    def run(input_address: int, output_address: int, scratch_address: int, chunk_packs: int, stride_packs: int,
            piece_packs: int, link_base: int, counters: int, piece_counters: int, ctrl_base: int,
            poison_address: int, spin_limit: int) -> None:
        compiled(
            make_pointer(input_address), make_pointer(output_address), make_pointer(scratch_address),
            int(chunk_packs), int(stride_packs), int(piece_packs), int(link_base), int(counters),
            int(piece_counters), int(ctrl_base), int(poison_address), int(spin_limit), current_cuda_stream(),
        )

    _LAUNCHERS[key] = run
    return run


def ring_launcher_key(mode: str, dtype_name: str, world_size: int, chain_index: int, rank: int,
                      order: Sequence[int], threads: int, lanes: int, slots: int, slot_bytes: int,
                      blocks_per_role: int, unroll: int, device_index: int, trace_capacity: int = 0) -> tuple:
    return ("link-ring", str(mode), str(dtype_name), int(world_size), int(chain_index), int(rank),
            tuple(int(r) for r in order), int(threads), int(lanes), int(slots), int(slot_bytes),
            int(blocks_per_role), int(unroll), int(device_index), int(trace_capacity))


def get_ring_launcher(mode: str, dtype_name: str, world_size: int, chain_index: int, rank: int,
                      order: Sequence[int], threads: int, lanes: int, slots: int, slot_bytes: int,
                      blocks_per_role: int, unroll: int, device_index: int,
                      trace_capacity: int = 0) -> Callable[..., None]:
    """The compiled ring launcher of one specialization (compiled once per process).

    Launch arguments: input address, output address, packs per chunk (per rank's block for the
    all-gather), packs between chunks (the all-gather: its packs per rank), packs per piece, link
    area address, device counter address, control line address, poison address, spin limit, the trace
    buffer's address (0 without a trace), the stagger of the partials (link 2: reduce-scatter,
    all-reduce) and the stagger of the forwarded pieces (link 3: all-gather, all-reduce).
    """
    key = ring_launcher_key(mode, dtype_name, world_size, chain_index, rank, order, threads, lanes, slots,
                            slot_bytes, blocks_per_role, unroll, device_index, trace_capacity)
    cached = _LAUNCHERS.get(key)
    if cached is not None:
        return cached
    launch = LinkRing(mode, dtype_name, world_size, chain_index, rank, order, threads, lanes, slots, slot_bytes,
                      blocks_per_role, unroll, trace_capacity)
    compiled = compile_launcher(
        launch, make_pointer(16), make_pointer(16), 1, 1, 1, 16, 16, 16, 16, 1, 16, 0, 0, current_cuda_stream(),
        name=f"sircl ring {mode}", cache_key=key,
    )

    def run(input_address: int, output_address: int, chunk_packs: int, stride_packs: int, piece_packs: int,
            link_base: int, counters: int, ctrl_base: int, poison_address: int, spin_limit: int,
            trace_address: int = 0, stagger: int = 0, gather_stagger: int = 0) -> None:
        compiled(
            make_pointer(input_address), make_pointer(output_address), int(chunk_packs), int(stride_packs),
            int(piece_packs), int(link_base), int(counters), int(ctrl_base), int(poison_address), int(spin_limit),
            int(trace_address), int(stagger), int(gather_stagger), current_cuda_stream(),
        )

    _LAUNCHERS[key] = run
    return run


__all__ = ["COUNTER_WORDS", "DECISION_WORDS", "PIECE_COUNTERS", "RING_MODES", "LinkGather", "LinkRing", "LinkScatter",
           "get_gather_launcher", "get_ring_launcher", "get_scatter_launcher", "launcher_key", "ring_launcher_key",
           "scatter_launcher_key"]
