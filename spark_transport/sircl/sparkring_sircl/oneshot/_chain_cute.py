"""CuTe DSL kernel of the chain all-reduce: one launch performs one chain op of any size.

The ranks of the group form a chain of cable neighbors (``chain_index`` 0 to
``W - 1``). The message's first ``a_packs`` 16-byte packs are half A and the
rest half B. Half A reduces along the chain from index 0 to ``W - 1``: index 0
sends its own values, every later index adds its own values to the partial it
received (float32 addition of the two dtype values, rounded once to the dtype)
and sends the result on; index ``W - 1`` holds the final values, stores them and
sends them back along the chain. Half B runs the same way from index ``W - 1``
to index 0. Every rank stores the final values the end rank computed, so all
ranks hold identical bits; they can differ in the last place from the one-shot
all-reduce (one rounding of the float32 sum in rank order), because every hop
rounds.

Each half travels in chunks of ``chunk_packs`` packs through the session's chain
rings (``protocol.ChainLayout``; streams ``protocol.ChainStream``). The native
progress thread writes chunks to the neighbors, forwards results it receives
along the chain without the kernel, and returns credit for consumed slots. The
grid has four roles of ``blocks_per_role`` blocks, each block taking every
``blocks_per_role``-th chunk of its role:

- A reduce (and B reduce): wait for the inbound partial chunk (except at the
  half's first rank) and for a free send slot, add the own values, stage the
  result for the next rank, or, at the half's last rank, store it to the output
  and stage it as a result; publish the inbound slot as consumed and the send
  slot as ready;
- A results (and B results): wait for a result chunk, copy it to the output,
  publish the slot as consumed (absent at the rank that computes the half).

A chunk's tag is its global index plus one; flags, ready and consumed words carry
it, so slots need no resetting between ops. In a reduce block, threads 0 to
``L - 1`` (warp 0) wait for the inbound lane flags and the block's last thread
(the last warp) waits for the free send slot: two different wait loops never run
in divergent threads of one warp. With both loops in one warp, the send-slot
wait has been observed to report a timeout while its condition held. The device counters hold the chunks
of each half before this op and the chain op sequence; the last block to finish
advances them unless a wait timed out. A wait longer than the session's wait
limit records what it waited for (command ring words 2, 3, 6 and 8) and poisons
the session.

A reduce or result pass moves ``unroll`` packs per thread with the loads of
the pass issued together (``_cute_batch.ld_v4_u32_batch``): the inbound slots
live in pinned host memory, where each load waits about a microsecond, so a
pass pays one memory latency instead of one per pack.

A specialization compiled with a trace capacity records events in a device
buffer (``protocol.TraceEvent``): warp 0 of a block that saw every lane flag of
an inbound chunk (``KERNEL_FLAG``), and the block that published a chunk ready
or consumed (``KERNEL_READY``, ``KERNEL_CONSUMED``), each with the GPU's
``%globaltimer``. The buffer's first word counts the records claimed; records
of 16 bytes (time, tag, event and stream) start at byte ``TRACE_HEADER_BYTES``,
and records past the capacity are counted but not written.
"""

# Annotations stay evaluated (no postponed evaluation): the CuTe DSL reads each
# launch parameter's type from its annotation object, and some DSL releases
# type string annotations from the example arguments instead, which makes
# 64-bit address parameters 32-bit.

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

from ..protocol import FLAG_STRIDE, PACK_BYTES, ChainLayout, Ctrl, ErrorKind, TraceEvent
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

DTYPE_PACK_ELEMS = {"float32": 4, "float16": 8, "bfloat16": 8}
ROLES = 4                    # A reduce, B reduce, A results, B results
COUNTER_WORDS = 4            # chunks of half A before this op, of half B, chain sequence, tail arrivals
TRACE_HEADER_BYTES = 16      # the trace buffer's claimed-record counter, then its records
TRACE_RECORD_BYTES = 16
_ERROR_SEQ = 4 * int(Ctrl.ERROR_SEQ)
_MISSING_PEER = 4 * int(Ctrl.MISSING_PEER)
_MISSING_LANE = 4 * int(Ctrl.MISSING_LANE)
_ERROR_KIND = 4 * int(Ctrl.ERROR_KIND)
_WAIT_LIMIT = 4 * int(Ctrl.WAIT_LIMIT_US)


class ChainAllReduce:
    """One kernel specialization: dtype, chain position and neighbors, block size, ring geometry."""

    def __init__(self, dtype_name: str, world_size: int, chain_index: int, prev_rank: int, next_rank: int,
                 rank: int, threads: int, lanes: int, slots: int, slot_bytes: int, blocks_per_role: int,
                 unroll: int = 4, trace_capacity: int = 0) -> None:
        if dtype_name not in DTYPE_PACK_ELEMS:
            raise ValueError(f"unsupported chain all-reduce dtype {dtype_name!r}")
        if not 0 <= chain_index < world_size or world_size < 2:
            raise ValueError(f"chain index {chain_index} of a chain of {world_size}")
        if int(threads) < 64 or int(threads) % 32 or not 1 <= int(lanes) <= 32:
            raise ValueError(f"chain kernels need a multiple of 32 threads, at least 64, so the send-slot wait "
                             f"runs in a warp apart from the lane waits; got {threads} threads and {lanes} lanes")
        if not 1 <= int(unroll) <= 8:
            raise ValueError(f"chain kernels move 1 to 8 packs per thread per pass; got {unroll}")
        if not 0 <= int(trace_capacity) < 1 << 31:
            raise ValueError(f"chain kernel trace capacity {trace_capacity}")
        self._dtype_name = dtype_name
        self._pack_elems = DTYPE_PACK_ELEMS[dtype_name]
        self._world = int(world_size)
        self._index = int(chain_index)
        self._prev = int(prev_rank)
        self._next = int(next_rank)
        self._rank = int(rank)
        self._threads = int(threads)
        self._lanes = int(lanes)
        self._slots = int(slots)
        self._slot_bytes = int(slot_bytes)
        self._nb = int(blocks_per_role)
        self._unroll = int(unroll)
        self._trace_capacity = int(trace_capacity)
        self._layout = ChainLayout(int(lanes), int(slots), int(slot_bytes))

    # -- chain area addresses -------------------------------------------------------------

    @cute.jit
    def _slot(self, chain_base: Int64, area_off: cutlass.Constexpr[int], stream: cutlass.Constexpr[int],
              m: Uint32) -> Int64:
        return chain_base + Int64(area_off) + (Int64(stream * self._slots) + Int64(m)) * Int64(self._slot_bytes)

    @cute.jit
    def _flag(self, chain_base: Int64, stream: cutlass.Constexpr[int], m: Uint32, lane: Int32) -> Int64:
        line = (Int64(stream * self._slots) + Int64(m)) * Int64(self._lanes) + Int64(lane)
        return chain_base + Int64(self._layout.rflag_off) + line * Int64(FLAG_STRIDE)

    @cute.jit
    def _word(self, chain_base: Int64, area_off: cutlass.Constexpr[int], stream: cutlass.Constexpr[int],
              m: Uint32) -> Int64:
        return chain_base + Int64(area_off + stream * FLAG_STRIDE) + Int64(m) * Int64(4)

    # -- arithmetic -----------------------------------------------------------------------

    def _add_packs(self, partial, own):
        """The dtype rounding of float32(partial) + float32(own), per element (four words). Plain
        Python evaluated while the kernel is traced."""
        packed = []
        for word in range(4):
            if self._dtype_name == "float32":
                packed.append(f32_as_u32(u32_as_f32(partial[word]) + u32_as_f32(own[word])))
            elif self._dtype_name == "float16":
                p_lo, p_hi = unpack_f16x2(partial[word])
                o_lo, o_hi = unpack_f16x2(own[word])
                packed.append(pack_f32x2_to_f16x2(p_lo + o_lo, p_hi + o_hi))
            else:
                p_lo, p_hi = unpack_bf16x2(partial[word])
                o_lo, o_hi = unpack_bf16x2(own[word])
                packed.append(pack_f32x2_to_bf16x2(p_lo + o_lo, p_hi + o_hi))
        return packed

    def _combine(self, group):
        """One source's words unchanged, or the rounded sum of an inbound partial and the own values."""
        if len(group) == 1:
            return group[0]
        return self._add_packs(group[0], group[1])

    @cute.jit
    def _pass(self, thread: Int32, inbound: Int64, has_in: cutlass.Constexpr[bool], own: Int64,
              has_own: cutlass.Constexpr[bool], dst: Int64, dst2: Int64, to_dst2: cutlass.Constexpr[bool],
              packs: Int32, zero: Uint32) -> None:
        """Store, for each of the first ``packs`` packs, the inbound partial plus the own values (a
        reduce with an inbound partial), the own values (the half's first rank) or the inbound pack (a
        result copy) to ``dst`` and, with ``to_dst2``, to ``dst2``. ``unroll`` packs per thread per
        pass with every load of a pass issued together; inbound packs are read at system scope."""
        last = packs - Int32(1)
        base = Int32(0)
        while base < packs:
            addrs = []
            flags = []
            for u in cutlass.range_constexpr(self._unroll):
                index = Int64(min_s32(base + Int32(u * self._threads) + thread, last)) * Int64(PACK_BYTES)
                if cutlass.const_expr(has_in):
                    addrs.append(inbound + index)
                    flags.append(True)
                if cutlass.const_expr(has_own):
                    addrs.append(own + index)
                    flags.append(False)
            words = ld_v4_u32_batch(addrs, flags, zero)
            per = len(addrs) // self._unroll
            for u in cutlass.range_constexpr(self._unroll):
                index = base + Int32(u * self._threads) + thread
                if index < packs:
                    packed = self._combine(words[u * per:(u + 1) * per])
                    offset = Int64(index) * Int64(PACK_BYTES)
                    st_global_v4_u32(dst + offset, packed[0], packed[1], packed[2], packed[3])
                    if cutlass.const_expr(to_dst2):
                        st_global_v4_u32(dst2 + offset, packed[0], packed[1], packed[2], packed[3])
            base = base + Int32(self._unroll * self._threads)

    # -- trace ------------------------------------------------------------------------------

    @cute.jit
    def _trace(self, thread: Int32, trace_base: Int64, event: cutlass.Constexpr[int],
               stream: cutlass.Constexpr[int], tag: Uint32) -> None:
        """Thread 0 appends one event record when the specialization has a trace."""
        if cutlass.const_expr(self._trace_capacity > 0):
            if thread == Int32(0):
                index = atomic_add_relaxed_gpu_u32(trace_base, Uint32(1))
                if index < Uint32(self._trace_capacity):
                    lo, hi = globaltimer_u32x2()
                    record = trace_base + Int64(TRACE_HEADER_BYTES) + Int64(index) * Int64(TRACE_RECORD_BYTES)
                    st_global_v4_u32(record, lo, hi, tag, Uint32(event | (stream << 16)))

    @cute.jit
    def _trace_flags(self, thread: Int32, trace_base: Int64, stream: cutlass.Constexpr[int], tag: Uint32) -> None:
        """After the lane-flag waits: warp 0 converges and thread 0 records ``KERNEL_FLAG``."""
        if cutlass.const_expr(self._trace_capacity > 0):
            if thread < Int32(32):
                cute.arch.sync_warp()
            self._trace(thread, trace_base, int(TraceEvent.KERNEL_FLAG), stream, tag)

    # -- waits ------------------------------------------------------------------------------

    @cute.jit
    def _fail(self, ctrl_base: Int64, poison_ptr: Int64, peer: Int32, lane: Int32, kind: cutlass.Constexpr[int],
              tag: Uint32) -> None:
        st_relaxed_sys_u32(ctrl_base + Int64(_MISSING_PEER), Uint32(peer))
        st_relaxed_sys_u32(ctrl_base + Int64(_MISSING_LANE), Uint32(lane))
        st_relaxed_sys_u32(ctrl_base + Int64(_ERROR_KIND), Uint32(kind))
        fence_sc_sys()
        st_relaxed_sys_u32(ctrl_base + Int64(_ERROR_SEQ), tag)
        st_release_gpu_u32(poison_ptr, Uint32(1))

    @cute.jit
    def _wait_inbound(self, thread: Int32, chain_base: Int64, ctrl_base: Int64, poison_ptr: Int64,
                      stream: cutlass.Constexpr[int], peer: cutlass.Constexpr[int], m: Uint32, tag: Uint32,
                      spin_limit: Uint32, limit_us: Uint32) -> None:
        """Thread ``t < L`` waits for lane ``t``'s flag of the inbound chunk."""
        if thread < Int32(self._lanes):
            timed_out = spin_until_eq_timed_sys(self._flag(chain_base, stream, m, thread), tag, spin_limit,
                                                limit_us)
            if timed_out != Uint32(0):
                self._fail(ctrl_base, poison_ptr, Int32(peer), thread, int(ErrorKind.CHAIN_CHUNK), tag)

    @cute.jit
    def _wait_send_slot(self, thread: Int32, chain_base: Int64, ctrl_base: Int64, poison_ptr: Int64,
                        stream: cutlass.Constexpr[int], tag: Uint32, spin_limit: Uint32, limit_us: Uint32) -> None:
        """The block's last thread waits until the slot's previous chunk was written downstream."""
        if thread == Int32(self._threads - 1):
            sent = chain_base + Int64(self._layout.sent_off + stream * FLAG_STRIDE)
            timed_out = spin_until_ge_timed_sys(sent, tag - Uint32(self._slots), spin_limit, limit_us)
            if timed_out != Uint32(0):
                self._fail(ctrl_base, poison_ptr, Int32(self._rank), Int32(255), int(ErrorKind.CHAIN_SLOT), tag)

    # -- roles ------------------------------------------------------------------------------

    @cute.jit
    def _reduce(self, half: cutlass.Constexpr[int], sub: Int32, input_base: Int64, output_base: Int64,
                half_packs: Int32, half_offset: Int32, chunk_packs: Int32, base: Uint32, chain_base: Int64,
                ctrl_base: Int64, poison_ptr: Int64, spin_limit: Uint32, limit_us: Uint32,
                trace_base: Int64) -> None:
        """Reduce role of one half: add the own values to the inbound partial, send it on."""
        tidx, _, _ = cute.arch.thread_idx()
        thread = Int32(tidx)
        position = self._index if half == 0 else self._world - 1 - self._index
        has_in = position > 0
        last = position == self._world - 1
        in_stream = 0 if half == 0 else 2
        out_stream = (1 if half == 0 else 3) if last else in_stream
        in_peer = self._prev if half == 0 else self._next
        n_chunks = (half_packs + chunk_packs - Int32(1)) // chunk_packs
        j = sub
        while j < n_chunks:
            g = base + Uint32(j)
            m = g % Uint32(self._slots)
            tag = g + Uint32(1)
            if cutlass.const_expr(has_in):
                self._wait_inbound(thread, chain_base, ctrl_base, poison_ptr, in_stream, in_peer, m, tag,
                                   spin_limit, limit_us)
                self._trace_flags(thread, trace_base, in_stream, tag)
            self._wait_send_slot(thread, chain_base, ctrl_base, poison_ptr, out_stream, tag, spin_limit, limit_us)
            cute.arch.sync_threads()
            failed = ld_relaxed_gpu_u32(poison_ptr)
            if failed == Uint32(0):
                remaining = half_packs - j * chunk_packs
                count = chunk_packs
                if remaining < chunk_packs:
                    count = remaining
                first = half_offset + j * chunk_packs
                inbound = self._slot(chain_base, self._layout.recv_off, in_stream, m)
                outbound = self._slot(chain_base, self._layout.send_off, out_stream, m)
                self._pass(thread, inbound, has_in, input_base + Int64(first) * Int64(PACK_BYTES), True, outbound,
                           output_base + Int64(first) * Int64(PACK_BYTES), last, count,
                           Uint32(chunk_packs) >> Uint32(31))
                cute.arch.sync_threads()
                if thread == Int32(0):
                    fence_sc_sys()
                    if cutlass.const_expr(has_in):
                        st_relaxed_sys_u32(self._word(chain_base, self._layout.consumed_off, in_stream, m), tag)
                    st_relaxed_sys_u32(self._word(chain_base, self._layout.ready_off, out_stream, m), tag)
                if cutlass.const_expr(has_in):
                    self._trace(thread, trace_base, int(TraceEvent.KERNEL_CONSUMED), in_stream, tag)
                self._trace(thread, trace_base, int(TraceEvent.KERNEL_READY), out_stream, tag)
                j = j + Int32(self._nb)
            else:
                j = n_chunks

    @cute.jit
    def _results(self, half: cutlass.Constexpr[int], sub: Int32, output_base: Int64, half_packs: Int32,
                 half_offset: Int32, chunk_packs: Int32, base: Uint32, chain_base: Int64, ctrl_base: Int64,
                 poison_ptr: Int64, spin_limit: Uint32, limit_us: Uint32, trace_base: Int64) -> None:
        """Result role of one half: copy every inbound result chunk to the output."""
        tidx, _, _ = cute.arch.thread_idx()
        thread = Int32(tidx)
        stream = 1 if half == 0 else 3
        peer = self._next if half == 0 else self._prev
        n_chunks = (half_packs + chunk_packs - Int32(1)) // chunk_packs
        j = sub
        while j < n_chunks:
            g = base + Uint32(j)
            m = g % Uint32(self._slots)
            tag = g + Uint32(1)
            self._wait_inbound(thread, chain_base, ctrl_base, poison_ptr, stream, peer, m, tag, spin_limit,
                               limit_us)
            self._trace_flags(thread, trace_base, stream, tag)
            cute.arch.sync_threads()
            failed = ld_relaxed_gpu_u32(poison_ptr)
            if failed == Uint32(0):
                remaining = half_packs - j * chunk_packs
                count = chunk_packs
                if remaining < chunk_packs:
                    count = remaining
                first = half_offset + j * chunk_packs
                inbound = self._slot(chain_base, self._layout.recv_off, stream, m)
                self._pass(thread, inbound, True, Int64(0), False, output_base + Int64(first) * Int64(PACK_BYTES),
                           Int64(0), False, count, Uint32(chunk_packs) >> Uint32(31))
                cute.arch.sync_threads()
                if thread == Int32(0):
                    fence_sc_sys()
                    st_relaxed_sys_u32(self._word(chain_base, self._layout.consumed_off, stream, m), tag)
                self._trace(thread, trace_base, int(TraceEvent.KERNEL_CONSUMED), stream, tag)
                j = j + Int32(self._nb)
            else:
                j = n_chunks

    # -- launch -----------------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        a_packs: Int32,
        b_packs: Int32,
        chunk_packs: Int32,
        chain_base: Int64,
        counters: Int64,
        ctrl_base: Int64,
        poison_ptr: Int64,
        spin_limit: Uint32,
        trace_base: Int64,
        stream: cuda.CUstream,
    ) -> None:
        """Host entry: one block per (role, chunk lane)."""
        self.kernel(
            input_ptr, output_ptr, a_packs, b_packs, chunk_packs, chain_base, counters, ctrl_base, poison_ptr,
            spin_limit, trace_base,
        ).launch(grid=(ROLES * self._nb, 1, 1), block=[self._threads, 1, 1], cluster=(1, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        a_packs: Int32,
        b_packs: Int32,
        chunk_packs: Int32,
        chain_base: Int64,
        counters: Int64,
        ctrl_base: Int64,
        poison_ptr: Int64,
        spin_limit: Uint32,
        trace_base: Int64,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        input_base = Int64(input_ptr.toint())
        output_base = Int64(output_ptr.toint())
        role = Int32(bidx) // Int32(self._nb)
        sub = Int32(bidx) - role * Int32(self._nb)
        poisoned = ld_relaxed_gpu_u32(poison_ptr)
        if poisoned == Uint32(0):
            # Every block reads the counters before any block can advance them:
            # the advance happens only after every block arrived at the tail.
            base_a = ld_relaxed_gpu_u32(counters)
            base_b = ld_relaxed_gpu_u32(counters + Int64(4))
            seq = ld_relaxed_gpu_u32(counters + Int64(8)) + Uint32(1)
            if Int32(bidx) == Int32(0):
                if Int32(tidx) == Int32(0):
                    params = chain_base + Int64(self._layout.ctrl_off + 4) + Int64(seq & Uint32(1)) * Int64(16)
                    st_relaxed_sys_u32(params, Uint32(a_packs * Int32(PACK_BYTES)))
                    st_relaxed_sys_u32(params + Int64(4), Uint32(b_packs * Int32(PACK_BYTES)))
                    st_relaxed_sys_u32(params + Int64(8), Uint32(chunk_packs * Int32(PACK_BYTES)))
                    fence_sc_sys()
                    st_relaxed_sys_u32(chain_base + Int64(self._layout.ctrl_off), seq)
            limit_us = ld_relaxed_sys_u32(ctrl_base + Int64(_WAIT_LIMIT))
            if role == Int32(0):
                self._reduce(0, sub, input_base, output_base, a_packs, Int32(0), chunk_packs, base_a, chain_base,
                             ctrl_base, poison_ptr, spin_limit, limit_us, trace_base)
            if role == Int32(1):
                self._reduce(1, sub, input_base, output_base, b_packs, a_packs, chunk_packs, base_b, chain_base,
                             ctrl_base, poison_ptr, spin_limit, limit_us, trace_base)
            if cutlass.const_expr(self._index < self._world - 1):
                if role == Int32(2):
                    self._results(0, sub, output_base, a_packs, Int32(0), chunk_packs, base_a, chain_base,
                                  ctrl_base, poison_ptr, spin_limit, limit_us, trace_base)
            if cutlass.const_expr(self._index > 0):
                if role == Int32(3):
                    self._results(1, sub, output_base, b_packs, a_packs, chunk_packs, base_b, chain_base,
                                  ctrl_base, poison_ptr, spin_limit, limit_us, trace_base)
            # The last block to finish advances the chunk bases and the sequence.
            fence_sc_gpu()
            cute.arch.sync_threads()
            if Int32(tidx) == Int32(0):
                prior = atomic_add_relaxed_gpu_u32(counters + Int64(12), Uint32(1))
                if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                    fence_sc_gpu()
                    if ld_relaxed_sys_u32(ctrl_base + Int64(_ERROR_SEQ)) == Uint32(0):
                        n_a = (a_packs + chunk_packs - Int32(1)) // chunk_packs
                        n_b = (b_packs + chunk_packs - Int32(1)) // chunk_packs
                        st_release_gpu_u32(counters, base_a + Uint32(n_a))
                        st_release_gpu_u32(counters + Int64(4), base_b + Uint32(n_b))
                        st_release_gpu_u32(counters + Int64(8), seq)


def _pointer(address: int):
    return make_pointer(address)


_LAUNCHERS: dict[tuple, Callable[..., None]] = {}


def launcher_key(dtype_name: str, world_size: int, chain_index: int, prev_rank: int, next_rank: int, rank: int,
                 threads: int, lanes: int, slots: int, slot_bytes: int, blocks_per_role: int,
                 device_index: int, unroll: int = 4, trace_capacity: int = 0) -> tuple:
    return ("chain", str(dtype_name), int(world_size), int(chain_index), int(prev_rank), int(next_rank), int(rank),
            int(threads), int(lanes), int(slots), int(slot_bytes), int(blocks_per_role), int(device_index),
            int(unroll), int(trace_capacity))


def get_launcher(dtype_name: str, world_size: int, chain_index: int, prev_rank: int, next_rank: int, rank: int,
                 threads: int, lanes: int, slots: int, slot_bytes: int, blocks_per_role: int,
                 device_index: int, unroll: int = 4, trace_capacity: int = 0) -> Callable[..., None]:
    """The compiled chain launcher of one specialization (compiled once per process).

    Launch arguments: input address, output address, packs of half A, packs of
    half B, packs per chunk, chain area address, device counter address,
    control line address, poison address, spin limit.
    """
    key = launcher_key(dtype_name, world_size, chain_index, prev_rank, next_rank, rank, threads, lanes, slots,
                       slot_bytes, blocks_per_role, device_index, unroll, trace_capacity)
    cached = _LAUNCHERS.get(key)
    if cached is not None:
        return cached
    launch = ChainAllReduce(dtype_name, world_size, chain_index, prev_rank, next_rank, rank, threads, lanes, slots,
                            slot_bytes, blocks_per_role, unroll, trace_capacity)
    compiled = compile_launcher(
        launch, make_pointer(16), make_pointer(16), 1, 1, 1, 16, 16, 16, 16, 1, 16, current_cuda_stream(),
        name="sircl chain all-reduce", cache_key=key,
    )

    def run(input_address: int, output_address: int, a_packs: int, b_packs: int, chunk_packs: int,
            chain_base: int, counters: int, ctrl_base: int, poison_address: int, spin_limit: int,
            trace_address: int = 0) -> None:
        compiled(
            _pointer(input_address), _pointer(output_address), int(a_packs), int(b_packs), int(chunk_packs),
            int(chain_base), int(counters), int(ctrl_base), int(poison_address), int(spin_limit),
            int(trace_address), current_cuda_stream(),
        )

    _LAUNCHERS[key] = run
    return run
