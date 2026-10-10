"""CuTe DSL kernel of the ring session's scatter collectives: reduce-scatter and all-to-all.

One launch performs one scatter op of a reduce-scatter (``mode="reduce"``) or
an all-to-all (``mode="copy"``): ``W`` chunks of ``chunk_packs`` 16-byte
packs (``size_packs = W * chunk_packs``). Chunk ``j`` of the input starts
``j * src_stride`` bytes after the input pointer, so the chunks may be strided
(a head range of a ``[W * H, rows, D]`` tensor, or a row range of a
``[W, rows, ...]`` buffer); contiguous chunks are the case
``src_stride == chunk_packs * 16``.

1. stage: copy every chunk ``j != rank`` into packs
   ``[j * chunk_packs, (j + 1) * chunk_packs)`` of ``send[seq & 1]``; the own
   chunk never leaves the GPU;
2. doorbell: the last block to finish staging writes the byte count (command
   ring word 1 and the slot's op word with op code 3) and then ``seq`` into
   the doorbell. The progress thread writes chunk ``p`` to rank ``p`` at the
   same offsets of ``p``'s ``recv[own rank][slot]``, striped over ``p``'s
   lanes, each stripe followed by its namespace-0 flag;
3. wait: thread ``t < W * L`` waits for lane ``t % L`` of rank ``t // L``. A
   wait that lasts longer than the session's wait limit (command ring word 7,
   microseconds; ``spin_limit`` polls when the word is 0) writes the missing
   peer and lane, then the sequence, into the command ring and poisons the
   session;
4. reduce: sum the own chunk of the input and every peer's copy of it in rank
   order ``0 .. W-1`` in float32 and round once (the bits of the one-shot and
   two-shot all-reduce for those elements), storing ``chunk_packs`` packs at
   the output; or copy: store the chunk received from source ``s`` at
   ``output + s * dst_stride`` (the own one straight from the input);
5. epoch: the last block to finish advances the device-resident epoch unless a
   wait timed out; a poisoned session's later launches do nothing.

There is no later network phase: the progress thread is done after the first
one, and the kernel rings no phase doorbell.

A slot is reused two ops later without extra synchronization. A peer writes
this rank's receive slot for op ``seq + 2`` only after it finished op
``seq + 1``, which needed this rank's data of op ``seq + 1``, staged after this
rank finished reading op ``seq``. This rank stages op ``seq + 2`` only after it
finished op ``seq + 1``, which needed every peer's data of op ``seq + 1``, sent
after each peer finished op ``seq`` and so after this rank's data of op ``seq``
had arrived.

Every pass of the reduce and copy phases loads one pack (two when the second
exists) from every source in one gated batch (``_cute_batch.ld_v4_u32_batch``),
so a pass pays one host-memory latency instead of one per peer. Grid size and
message size are runtime values; each power-of-two grid has its own staging
and tail counters, so scatter ops interleave with the session's other
collectives of any size inside one CUDA graph.

Origin: SparkRing's scatter kernel; the dtype pack arithmetic and the staging,
doorbell, wait and epoch pattern follow b12x RoCEnante
``b12x/comm/roce/_oneshot_cute.py`` (Apache-2.0, Local Inference Lab).
"""

# Annotations stay evaluated (no postponed evaluation): the CuTe DSL reads each
# launch parameter's type from its annotation object.

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

from ..protocol import OP_SHIFT, PACK_BYTES, Ctrl, Op
from ..scatter_plan import MODES
from ._compile import compile_launcher, current_cuda_stream, make_pointer
from ._cute_batch import ld_v4_u32_batch
from ._timed_wait import spin_until_eq_timed_sys
from ._cute_intrinsics import (
    atomic_add_relaxed_gpu_u32,
    f32_as_u32,
    fence_sc_gpu,
    fence_sc_sys,
    ld_global_v4_u32,
    ld_relaxed_gpu_u32,
    ld_relaxed_sys_u32,
    pack_f32x2_to_bf16x2,
    pack_f32x2_to_f16x2,
    st_global_v4_u32,
    st_release_gpu_u32,
    st_relaxed_sys_u32,
    u32_as_f32,
    unpack_bf16x2,
    unpack_f16x2,
)

DTYPE_PACK_ELEMS = {"float32": 4, "float16": 8, "bfloat16": 8}
SCATTER_CODE = int(Op.SCATTER) << OP_SHIFT
_NBYTES = 4 * int(Ctrl.NBYTES)
_OP_WORD = 4 * int(Ctrl.OP_WORD)
_ERROR_SEQ = 4 * int(Ctrl.ERROR_SEQ)
_MISSING_PEER = 4 * int(Ctrl.MISSING_PEER)
_MISSING_LANE = 4 * int(Ctrl.MISSING_LANE)
_WAIT_LIMIT = 4 * int(Ctrl.WAIT_LIMIT_US)


def _source_pack_addresses(world_size, rank, slots, input_base, recv_base, slot, slot_bytes, offsets):
    """Addresses of one 16-byte pack per source in rank order, for each offset.

    Plain Python evaluated while the kernel is traced: returns the address list
    (offset-major, then source rank) and, per address, whether it is a
    NIC-written peer slot (system-scope load) or this rank's own input.
    """
    addrs, system = [], []
    for offset in offsets:
        for source in range(world_size):
            if source == rank:
                addrs.append(input_base + offset)
                system.append(False)
            else:
                addrs.append(recv_base + (Int64(source) * Int64(slots) + slot) * slot_bytes + offset)
                system.append(True)
    return addrs, system


def _reduce_and_store(launch, offsets, input_base, recv_base, slot, slot_bytes, zero, output_base):
    """Rank-order float32 sums of the packs at ``offsets``, loaded in one batch, stored at the output.

    Plain Python evaluated while the kernel is traced, so a dynamic branch can
    call it without assigning anything in the branch. Every address of a batch
    is a distinct pack: padding a batch with a repeated pack would make every
    warp read the same host line, and those system-scope loads queue behind
    each other at the coherence point.
    """
    addrs, system = _source_pack_addresses(
        launch._world_size, launch._rank, launch._slots, input_base, recv_base, slot, slot_bytes, offsets,
    )
    words = ld_v4_u32_batch(addrs, system, zero)
    for k, offset in enumerate(offsets):
        accumulator = cute.make_rmem_tensor((launch._pack_elems,), cutlass.Float32)
        for source in range(launch._world_size):
            launch._accumulate_words(accumulator, words[k * launch._world_size + source], source == 0)
        launch._store_accumulator(output_base + offset, accumulator)


def _copy_chunks(launch, packs, input_own_base, recv_base, slot, slot_bytes, zero, output_base, own_lo,
                 dst_stride):
    """Copy pack ``q`` (for each ``q`` in ``packs``) of every source's chunk for this rank.

    Plain Python evaluated while the kernel is traced. ``input_own_base`` is the
    input address of the own chunk minus ``own_lo`` packs, so that
    :func:`_source_pack_addresses` yields the own chunk's pack ``q`` for source
    ``rank`` and pack ``own_lo + q`` of ``recv[source][slot]`` for every peer,
    which is where the progress thread of ``source`` wrote the chunk meant for
    this rank. Source ``s`` lands at ``output_base + s * dst_stride``.
    """
    offsets = [Int64(own_lo + q) * Int64(PACK_BYTES) for q in packs]
    addrs, system = _source_pack_addresses(
        launch._world_size, launch._rank, launch._slots, input_own_base, recv_base, slot, slot_bytes, offsets,
    )
    words = ld_v4_u32_batch(addrs, system, zero)
    for k, q in enumerate(packs):
        for source in range(launch._world_size):
            w = words[k * launch._world_size + source]
            st_global_v4_u32(output_base + Int64(source) * dst_stride + Int64(q) * Int64(PACK_BYTES),
                             w[0], w[1], w[2], w[3])


class ScatterLaunch:
    """One kernel specialization: mode, dtype (reduce only), group size, rank, layout constants."""

    def __init__(self, mode: str, dtype_name: str, world_size: int, rank: int, threads: int, slots: int,
                 flag_stride: int, lane_count: int) -> None:
        if mode not in MODES:
            raise ValueError(f"unsupported scatter mode {mode!r}; choose one of {MODES}")
        if mode == "reduce" and dtype_name not in DTYPE_PACK_ELEMS:
            raise ValueError(f"unsupported reduce-scatter dtype {dtype_name!r}")
        if int(threads) < int(world_size) * int(lane_count):
            raise ValueError(
                "scatter kernels need threads >= world_size * lane_count (one waiter per lane flag), "
                f"got threads={threads} world_size={world_size} lane_count={lane_count}"
            )
        self._mode = mode
        self._dtype_name = dtype_name if mode == "reduce" else "bytes"
        self._pack_elems = DTYPE_PACK_ELEMS.get(dtype_name, 4)
        self._world_size = int(world_size)
        self._rank = int(rank)
        self._threads = int(threads)
        self._slots = int(slots)
        self._flag_stride = int(flag_stride)
        self._lanes = int(lane_count)

    @cute.jit
    def _accumulate_words(self, accumulator: cute.Tensor, words, initialize: cutlass.Constexpr[bool]) -> None:
        """Add one 16-byte pack of the dtype to the float32 accumulator."""
        if cutlass.const_expr(self._dtype_name == "float32"):
            for word in cutlass.range_constexpr(4):
                value = u32_as_f32(words[word])
                if cutlass.const_expr(initialize):
                    accumulator[word] = value
                else:
                    accumulator[word] = accumulator[word] + value
        else:
            for word in cutlass.range_constexpr(4):
                if cutlass.const_expr(self._dtype_name == "float16"):
                    lo, hi = unpack_f16x2(words[word])
                else:
                    lo, hi = unpack_bf16x2(words[word])
                lane = word * 2
                if cutlass.const_expr(initialize):
                    accumulator[lane] = lo
                    accumulator[lane + 1] = hi
                else:
                    accumulator[lane] = accumulator[lane] + lo
                    accumulator[lane + 1] = accumulator[lane + 1] + hi

    @cute.jit
    def _store_accumulator(self, address: Int64, accumulator: cute.Tensor) -> None:
        """Round the accumulator once to the dtype and store one 16-byte pack."""
        packed = cute.make_rmem_tensor((4,), cutlass.Uint32)
        if cutlass.const_expr(self._dtype_name == "float32"):
            for word in cutlass.range_constexpr(4):
                packed[word] = f32_as_u32(accumulator[word])
        else:
            for word in cutlass.range_constexpr(4):
                lane = word * 2
                if cutlass.const_expr(self._dtype_name == "float16"):
                    packed[word] = pack_f32x2_to_f16x2(accumulator[lane], accumulator[lane + 1])
                else:
                    packed[word] = pack_f32x2_to_bf16x2(accumulator[lane], accumulator[lane + 1])
        st_global_v4_u32(address, packed[0], packed[1], packed[2], packed[3])

    @cute.jit
    def __call__(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        size_packs: Int32,
        nbytes: Int32,
        chunk_packs: Int32,
        src_stride: Int64,
        dst_stride: Int64,
        recv_base: Int64,
        flag_base: Int64,
        send_base: Int64,
        ctrl_base: Int64,
        slot_bytes: Int64,
        epoch_ptr: Int64,
        stage_counter_ptr: Int64,
        tail_counter_ptr: Int64,
        poison_ptr: Int64,
        spin_limit: Uint32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        """Host entry: launch the scatter kernel with runtime scalars."""
        self.kernel(
            input_ptr, output_ptr, size_packs, nbytes, chunk_packs, src_stride, dst_stride, recv_base, flag_base,
            send_base, ctrl_base, slot_bytes, epoch_ptr, stage_counter_ptr, tail_counter_ptr, poison_ptr,
            spin_limit,
        ).launch(grid=(grid_x, 1, 1), block=[self._threads, 1, 1], cluster=(1, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        size_packs: Int32,
        nbytes: Int32,
        chunk_packs: Int32,
        src_stride: Int64,
        dst_stride: Int64,
        recv_base: Int64,
        flag_base: Int64,
        send_base: Int64,
        ctrl_base: Int64,
        slot_bytes: Int64,
        epoch_ptr: Int64,
        stage_counter_ptr: Int64,
        tail_counter_ptr: Int64,
        poison_ptr: Int64,
        spin_limit: Uint32,
    ) -> None:
        """Device kernel: stage the peers' chunks, doorbell, wait, reduce or copy, advance the epoch."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        input_base = Int64(input_ptr.toint())
        output_base = Int64(output_ptr.toint())
        # Every block reads the epoch before any block can advance it: the
        # advance happens only after every block arrived at the tail counter.
        epoch = ld_relaxed_gpu_u32(epoch_ptr)
        seq = epoch + Uint32(1)
        slot = Int64(seq & Uint32(self._slots - 1))
        send_slot = send_base + slot * slot_bytes

        index = Int32(bidx) * Int32(self._threads) + Int32(tidx)
        stride = Int32(gdim) * Int32(self._threads)
        # Zero at run time (sizes are below 2**31) but opaque to ptxas: the
        # dependency gate of ld_v4_u32_batch.
        zero = Uint32(size_packs) >> Uint32(31)
        own_lo = Int32(self._rank) * chunk_packs
        own_hi = own_lo + chunk_packs
        # The own chunk's pack q is at input_base + rank * src_stride + q * 16;
        # shifting the base by own_lo packs lets the reduce and copy helpers
        # address it as pack own_lo + q, like the peer slots.
        input_own_base = input_base + Int64(self._rank) * src_stride - Int64(own_lo) * Int64(PACK_BYTES)
        output_own_base = output_base - Int64(own_lo) * Int64(PACK_BYTES)

        # A recorded timeout poisons the session: later launches do nothing, so
        # the host sees the failure without waiting another spin limit per op.
        poisoned = ld_relaxed_gpu_u32(poison_ptr)
        if poisoned == Uint32(0):
            # 1. stage every chunk but the own one into its pack range of the send slot
            stage_index = index
            while stage_index < size_packs:
                chunk = stage_index // chunk_packs
                within = stage_index - chunk * chunk_packs
                if chunk != Int32(self._rank):
                    words = ld_global_v4_u32(input_base + Int64(chunk) * src_stride + Int64(within) * Int64(PACK_BYTES))
                    st_global_v4_u32(send_slot + Int64(stage_index) * Int64(PACK_BYTES),
                                     words[0], words[1], words[2], words[3])
                stage_index += stride
            cute.arch.sync_threads()

            # 2. the last block to finish staging rings the doorbell (op code 3)
            if Int32(tidx) == Int32(0):
                fence_sc_sys()
                prior = atomic_add_relaxed_gpu_u32(stage_counter_ptr, Uint32(1))
                if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                    st_relaxed_sys_u32(ctrl_base + Int64(_NBYTES), Uint32(nbytes))
                    st_relaxed_sys_u32(ctrl_base + Int64(_OP_WORD) + slot * Int64(4),
                                       Uint32(nbytes) | Uint32(SCATTER_CODE))
                    fence_sc_sys()
                    st_relaxed_sys_u32(ctrl_base, seq)

            # 3. wait for every peer's stripes of the own chunk (namespace 0)
            if Int32(tidx) < Int32(self._world_size * self._lanes):
                peer = Int32(tidx) // Int32(self._lanes)
                lane = Int32(tidx) - peer * Int32(self._lanes)
                if peer != Int32(self._rank):
                    flag_addr = flag_base + (
                        (Int64(peer) * Int64(self._slots) + slot) * Int64(self._lanes) + Int64(lane)
                    ) * Int64(self._flag_stride)
                    limit_us = ld_relaxed_sys_u32(ctrl_base + Int64(_WAIT_LIMIT))
                    timed_out = spin_until_eq_timed_sys(flag_addr, seq, spin_limit, limit_us)
                    if timed_out != Uint32(0):
                        st_relaxed_sys_u32(ctrl_base + Int64(_MISSING_PEER), Uint32(peer))
                        st_relaxed_sys_u32(ctrl_base + Int64(_MISSING_LANE), Uint32(lane))
                        fence_sc_sys()
                        st_relaxed_sys_u32(ctrl_base + Int64(_ERROR_SEQ), seq)
                        st_release_gpu_u32(poison_ptr, Uint32(1))
            cute.arch.sync_threads()
            # A wait that timed out in this block leaves the peer slots
            # unreliable: skip the data phase so nothing derived from it is stored.
            failed = ld_relaxed_gpu_u32(poison_ptr)
            if failed == Uint32(0):
                if cutlass.const_expr(self._mode == "reduce"):
                    # 4a. reduce the own chunk in fixed rank order into the
                    # output; two packs per thread when the second exists.
                    reduce_index = own_lo + index
                    while reduce_index < own_hi:
                        second = reduce_index + stride
                        first_offset = Int64(reduce_index) * Int64(PACK_BYTES)
                        if second < own_hi:
                            _reduce_and_store(self, (first_offset, Int64(second) * Int64(PACK_BYTES)),
                                              input_own_base, recv_base, slot, slot_bytes, zero, output_own_base)
                        else:
                            _reduce_and_store(self, (first_offset,), input_own_base, recv_base, slot, slot_bytes,
                                              zero, output_own_base)
                        reduce_index += stride + stride
                else:
                    # 4b. copy every source's chunk for this rank to its output
                    # chunk; the own chunk comes from the input.
                    copy_index = index
                    while copy_index < chunk_packs:
                        second = copy_index + stride
                        if second < chunk_packs:
                            _copy_chunks(self, (copy_index, second), input_own_base, recv_base, slot, slot_bytes,
                                         zero, output_base, own_lo, dst_stride)
                        else:
                            _copy_chunks(self, (copy_index,), input_own_base, recv_base, slot, slot_bytes, zero,
                                         output_base, own_lo, dst_stride)
                        copy_index += stride + stride

            # 5. the last block to finish publishes the next epoch
            fence_sc_gpu()
            cute.arch.sync_threads()
            if Int32(tidx) == Int32(0):
                prior = atomic_add_relaxed_gpu_u32(tail_counter_ptr, Uint32(1))
                if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                    fence_sc_gpu()
                    # Every block's timeout stores precede its tail arrival, so
                    # the error word is final here; a failed sequence keeps the
                    # epoch and every later launch does nothing.
                    if ld_relaxed_sys_u32(ctrl_base + Int64(_ERROR_SEQ)) == Uint32(0):
                        st_release_gpu_u32(epoch_ptr, seq)


_LAUNCHERS: dict[tuple, Callable[..., None]] = {}


def launcher_key(mode: str, dtype_name: str, world_size: int, rank: int, threads: int, slots: int,
                 flag_stride: int, lane_count: int, device_index: int) -> tuple:
    """Cache key: every value the kernel bakes in (the dtype is ``bytes`` for the all-to-all)."""
    return (f"scatter-{mode}", str(dtype_name) if mode == "reduce" else "bytes", int(world_size), int(rank),
            int(threads), int(slots), int(flag_stride), int(lane_count), int(device_index))


def is_launcher_prepared(*key) -> bool:
    return launcher_key(*key) in _LAUNCHERS


def get_launcher(mode: str, dtype_name: str, world_size: int, rank: int, threads: int, slots: int,
                 flag_stride: int, lane_count: int, device_index: int) -> Callable[..., None]:
    """The compiled launcher of one specialization (compiled once per process).

    Launch arguments: input address, output address, packs, bytes, chunk
    packs, source stride, destination stride (bytes), receive base, flag base,
    send base, control base, slot bytes, epoch address, stage counter, tail
    counter, poison address, spin limit, grid blocks.
    """
    key = launcher_key(mode, dtype_name, world_size, rank, threads, slots, flag_stride, lane_count, device_index)
    cached = _LAUNCHERS.get(key)
    if cached is not None:
        return cached
    launch = ScatterLaunch(mode, dtype_name, world_size, rank, threads, slots, flag_stride, lane_count)
    compiled = compile_launcher(
        launch, make_pointer(16), make_pointer(16),
        1, 16, 1, 16, 16, 16, 16, 16, 16, 4096, 16, 16, 16, 16, 1, 1, current_cuda_stream(),
        name=f"sircl {mode} scatter", cache_key=key,
    )

    def run(input_address: int, output_address: int, size_packs: int, nbytes: int, chunk_packs: int,
            src_stride: int, dst_stride: int, recv_base: int, flag_base: int, send_base: int, ctrl_base: int,
            slot_bytes: int, epoch_address: int, stage_counter: int, tail_counter: int, poison_address: int,
            spin_limit: int, grid_blocks: int) -> None:
        compiled(
            make_pointer(input_address), make_pointer(output_address), int(size_packs), int(nbytes),
            int(chunk_packs), int(src_stride), int(dst_stride), int(recv_base), int(flag_base), int(send_base),
            int(ctrl_base), int(slot_bytes), int(epoch_address), int(stage_counter), int(tail_counter),
            int(poison_address), int(spin_limit), int(grid_blocks), current_cuda_stream(),
        )

    _LAUNCHERS[key] = run
    return run


__all__ = ["DTYPE_PACK_ELEMS", "MODES", "SCATTER_CODE", "ScatterLaunch", "get_launcher", "is_launcher_prepared",
           "launcher_key"]
