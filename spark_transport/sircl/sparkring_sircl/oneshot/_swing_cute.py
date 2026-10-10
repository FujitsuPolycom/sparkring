"""CuTe DSL kernel of the ring session's Swing all-reduce.

One launch performs one complete Swing all-reduce (De Sensi et al., NSDI
2024) of up to one slot on a power-of-two group, in ``2 * log2(W)`` network
phases that each write one contiguous chunk-position range to one peer
(``sparkring_sircl.swing_plan``): the reduce-scatter steps exchange halves,
quarters, ... with peers at offsets +-1, -+1, +-3, ...; the all-gather runs the
steps in reverse. Every phase is one phase of a described op (op code 2): the
kernel writes the phase's descriptor (position range, peer, flag namespace)
and then rings the phase's doorbell.

1. stage: copy the positions that phase 0 sends from the input into
   ``send[seq & 1]``; the last block to finish writes the byte count, the
   slot's op word (op code 2), the phase-0 descriptor and the doorbell;
2. per network phase ``p`` of this rank:

   - reduce-scatter step ``p``: wait for the step's peer in flag namespace 0,
     add the peer's partial sums of the kept positions (from
     ``recv[peer][seq & 1]``) to this rank's own (the input at step 0, the send
     slot after it) in float32, and store the result, rounded to the dtype, in
     the send slot; the last step's sums are final and also go to the output;
   - all-gather step: wait for the peer in namespace 1 and copy the peer's
     reduced range to the output, and to the send slot while a later
     all-gather phase still sends it;

   then the last block to finish the phase writes the next phase's descriptor
   and rings its doorbell, unless a wait timed out. A wait that lasts longer
   than the session's wait limit (command ring word 7, microseconds;
   ``spin_limit`` polls when the word is 0) writes the missing peer and lane,
   then the sequence, into the command ring and poisons the session; later
   phases and launches then do nothing;
3. epoch: the last block to finish advances the device-resident epoch.

Partial sums travel rounded to the dtype. The results are bit-identical on
every rank (each position is reduced by one rank and copied to the others) but
can differ in the last bits from the one-shot and two-shot all-reduce, which
add every rank's value in rank order in float32 and round once.

Every pack keeps the same thread in every phase (thread ``i mod stride`` owns
pack ``i``), so a reduction reads only partial sums its own thread stored in
the previous phase; only the NIC reads ranges written by other blocks, after
the phase doorbell, which follows a system fence. Every block of a launch must
be resident at once: a phase doorbell waits for every block's arrival.

A slot is reused two ops later without extra synchronization: a rank starts op
``seq + 2`` only after op ``seq + 1`` completed, and op ``seq + 1`` completes
only after every rank started it (every final position depends on every rank's
reduce-scatter data); so every rank has finished op ``seq``, received this
rank's data of it and read its own receive slots. On every queue pair the
writes of one op precede those of the next.

Origin: SparkRing's Swing kernel; the dtype pack arithmetic and the staging,
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

from .. import swing_plan
from ..protocol import FLAG_LINES, MAX_PHASES, OP_SHIFT, PACK_BYTES, Ctrl, Op, descriptor_word, phase_doorbell_word
from ._compile import compile_launcher, current_cuda_stream, make_pointer
from ._timed_wait import spin_until_eq_timed_sys
from ._cute_intrinsics import (
    atomic_add_relaxed_gpu_u32,
    f32_as_u32,
    fence_sc_gpu,
    fence_sc_sys,
    ld_global_v4_u32,
    ld_relaxed_gpu_u32,
    ld_relaxed_sys_u32,
    ld_relaxed_sys_v4_u32,
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
DESCRIBED_CODE = int(Op.DESCRIBED) << OP_SHIFT
_NBYTES = 4 * int(Ctrl.NBYTES)
_OP_WORD = 4 * int(Ctrl.OP_WORD)
_ERROR_SEQ = 4 * int(Ctrl.ERROR_SEQ)
_MISSING_PEER = 4 * int(Ctrl.MISSING_PEER)
_MISSING_LANE = 4 * int(Ctrl.MISSING_LANE)
_WAIT_LIMIT = 4 * int(Ctrl.WAIT_LIMIT_US)


class SwingAllReduce:
    """One kernel specialization: dtype, group size, rank (its schedule), block size, layout constants."""

    def __init__(self, dtype_name: str, world_size: int, rank: int, threads: int, slots: int, flag_stride: int,
                 lane_count: int) -> None:
        if dtype_name not in DTYPE_PACK_ELEMS:
            raise ValueError(f"unsupported Swing all-reduce dtype {dtype_name!r}")
        if not swing_plan.available(world_size):
            raise ValueError(f"the Swing all-reduce needs a power-of-two group of 2 to 16 ranks, got {world_size}")
        if int(threads) < int(lane_count):
            raise ValueError("Swing kernels need one thread per lane flag of a peer")
        if 2 * int(lane_count) > FLAG_LINES:
            raise ValueError(f"Swing ops need at most {FLAG_LINES // 2} lanes per peer, got {lane_count}")
        self._steps = swing_plan.swing_steps(int(world_size), int(rank))
        self._descriptors = swing_plan.descriptor_words(int(world_size), int(rank))
        if len(self._descriptors) > MAX_PHASES:
            raise ValueError(f"a described op has at most {MAX_PHASES} network phases")
        self._dtype_name = dtype_name
        self._pack_elems = DTYPE_PACK_ELEMS[dtype_name]
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
    def _store(self, address: Int64, mirror: Int64, accumulator: cute.Tensor,
               twice: cutlass.Constexpr[bool]) -> None:
        """Round the accumulator once to the dtype; store at ``address`` (and at ``mirror``)."""
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
        if cutlass.const_expr(twice):
            st_global_v4_u32(mirror, packed[0], packed[1], packed[2], packed[3])

    @cute.jit
    def _wait_peer(self, tidx: Int32, first_flag: Int64, seq: Uint32, spin_limit: Uint32, ctrl_base: Int64,
                   poison_ptr: Int64, peer_rank: cutlass.Constexpr[int]) -> None:
        """Threads ``0 .. L-1`` each wait for one lane flag of ``peer_rank``."""
        if tidx < Int32(self._lanes):
            if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                flag = first_flag + Int64(tidx) * Int64(self._flag_stride)
                limit_us = ld_relaxed_sys_u32(ctrl_base + Int64(_WAIT_LIMIT))
                expired = spin_until_eq_timed_sys(flag, seq, spin_limit, limit_us)
                if expired != Uint32(0):
                    st_relaxed_sys_u32(ctrl_base + Int64(_MISSING_PEER), Uint32(peer_rank))
                    st_relaxed_sys_u32(ctrl_base + Int64(_MISSING_LANE), Uint32(tidx))
                    fence_sc_sys()
                    st_relaxed_sys_u32(ctrl_base + Int64(_ERROR_SEQ), seq)
                    st_release_gpu_u32(poison_ptr, Uint32(1))

    @cute.jit
    def _reduce_range(self, lo: Int32, hi: Int32, index: Int32, stride: Int32, mine: Int64, theirs: Int64,
                      output_base: Int64, send_slot: Int64, final: cutlass.Constexpr[bool]) -> None:
        """Partial sums of packs ``[lo, hi)``: mine + theirs, to the send slot (and the output)."""
        pack = lo + (index - lo % stride + stride) % stride
        while pack < hi:
            offset = Int64(pack) * Int64(PACK_BYTES)
            own = ld_global_v4_u32(mine + offset)
            received = ld_relaxed_sys_v4_u32(theirs + offset)
            accumulator = cute.make_rmem_tensor((self._pack_elems,), cutlass.Float32)
            self._accumulate_words(accumulator, own, True)
            self._accumulate_words(accumulator, received, False)
            if cutlass.const_expr(final):
                self._store(output_base + offset, send_slot + offset, accumulator, True)
            else:
                self._store(send_slot + offset, send_slot + offset, accumulator, False)
            pack += stride

    @cute.jit
    def _copy_range(self, lo: Int32, hi: Int32, index: Int32, stride: Int32, source: Int64, output_base: Int64,
                    send_slot: Int64, mirror: cutlass.Constexpr[bool]) -> None:
        """Copy a peer's reduced packs ``[lo, hi)`` to the output (and to the send slot)."""
        pack = lo + (index - lo % stride + stride) % stride
        while pack < hi:
            offset = Int64(pack) * Int64(PACK_BYTES)
            words = ld_relaxed_sys_v4_u32(source + offset)
            st_global_v4_u32(output_base + offset, words[0], words[1], words[2], words[3])
            if cutlass.const_expr(mirror):
                st_global_v4_u32(send_slot + offset, words[0], words[1], words[2], words[3])
            pack += stride

    @cute.jit
    def _ring_phase(self, tidx: Int32, gdim: Int32, counter: Int64, ctrl_base: Int64, seq: Uint32,
                    phase: cutlass.Constexpr[int]) -> None:
        """The last block to arrive writes phase ``phase``'s descriptor and rings its doorbell."""
        if tidx == Int32(0):
            fence_sc_sys()
            prior = atomic_add_relaxed_gpu_u32(counter, Uint32(1))
            if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                fence_sc_sys()
                if ld_relaxed_sys_u32(ctrl_base + Int64(_ERROR_SEQ)) == Uint32(0):
                    st_relaxed_sys_u32(ctrl_base + Int64(4 * descriptor_word(phase)),
                                       Uint32(self._descriptors[phase]))
                    fence_sc_sys()
                    st_relaxed_sys_u32(ctrl_base + Int64(4 * phase_doorbell_word(phase)), seq)

    @cute.jit
    def __call__(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        size_packs: Int32,
        nbytes: Int32,
        recv_base: Int64,
        flag_base: Int64,
        send_base: Int64,
        ctrl_base: Int64,
        slot_bytes: Int64,
        epoch_ptr: Int64,
        stage_counter_ptr: Int64,
        phase_counter_ptr: Int64,
        phase_counter_stride: Int64,
        tail_counter_ptr: Int64,
        poison_ptr: Int64,
        spin_limit: Uint32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        """Host entry: launch the Swing kernel with runtime scalars."""
        self.kernel(
            input_ptr, output_ptr, size_packs, nbytes, recv_base, flag_base, send_base, ctrl_base, slot_bytes,
            epoch_ptr, stage_counter_ptr, phase_counter_ptr, phase_counter_stride, tail_counter_ptr, poison_ptr,
            spin_limit,
        ).launch(grid=(grid_x, 1, 1), block=[self._threads, 1, 1], cluster=(1, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        size_packs: Int32,
        nbytes: Int32,
        recv_base: Int64,
        flag_base: Int64,
        send_base: Int64,
        ctrl_base: Int64,
        slot_bytes: Int64,
        epoch_ptr: Int64,
        stage_counter_ptr: Int64,
        phase_counter_ptr: Int64,
        phase_counter_stride: Int64,
        tail_counter_ptr: Int64,
        poison_ptr: Int64,
        spin_limit: Uint32,
    ) -> None:
        """Device kernel: stage, the reduce-scatter and all-gather phases, then the epoch."""
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
        thread = Int32(tidx)
        index = Int32(bidx) * Int32(self._threads) + thread
        stride = Int32(gdim) * Int32(self._threads)
        world = Int32(self._world_size)
        steps = len(self._steps)
        first_send = self._steps[0].send

        poisoned = ld_relaxed_gpu_u32(poison_ptr)
        if poisoned == Uint32(0):
            # 1. stage the positions written in phase 0 (reduce-scatter step 0)
            stage_lo = (Int32(first_send[0]) * size_packs) // world
            stage_hi = (Int32(first_send[1]) * size_packs) // world
            stage_index = stage_lo + index
            while stage_index < stage_hi:
                staged = ld_global_v4_u32(input_base + Int64(stage_index) * Int64(PACK_BYTES))
                st_global_v4_u32(send_slot + Int64(stage_index) * Int64(PACK_BYTES),
                                 staged[0], staged[1], staged[2], staged[3])
                stage_index += stride
            cute.arch.sync_threads()
            if thread == Int32(0):
                fence_sc_sys()
                prior = atomic_add_relaxed_gpu_u32(stage_counter_ptr, Uint32(1))
                if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                    st_relaxed_sys_u32(ctrl_base + Int64(_NBYTES), Uint32(nbytes))
                    st_relaxed_sys_u32(ctrl_base + Int64(_OP_WORD) + slot * Int64(4),
                                       Uint32(nbytes) | Uint32(DESCRIBED_CODE))
                    st_relaxed_sys_u32(ctrl_base + Int64(4 * descriptor_word(0)), Uint32(self._descriptors[0]))
                    fence_sc_sys()
                    st_relaxed_sys_u32(ctrl_base, seq)

            # 2. reduce-scatter and all-gather phases
            for phase in cutlass.range_constexpr(2 * steps):
                gather = phase >= steps
                step = self._steps[phase if phase < steps else 2 * steps - 1 - phase]
                namespace = 1 if gather else 0
                first_flag = flag_base + (
                    (Int64((namespace * self._world_size + step.peer) * self._slots) + slot) * Int64(self._lanes)
                ) * Int64(self._flag_stride)
                self._wait_peer(thread, first_flag, seq, spin_limit, ctrl_base, poison_ptr, step.peer)
                cute.arch.sync_threads()
                healthy = ld_relaxed_gpu_u32(poison_ptr)
                peer_slot = recv_base + (Int64(step.peer) * Int64(self._slots) + slot) * slot_bytes
                if healthy == Uint32(0):
                    if cutlass.const_expr(not gather):
                        self._reduce_range(
                            (Int32(step.keep[0]) * size_packs) // world,
                            (Int32(step.keep[1]) * size_packs) // world,
                            index, stride,
                            input_base if step.step == 0 else send_slot,
                            peer_slot, output_base, send_slot,
                            step.step == steps - 1,
                        )
                    else:
                        self._copy_range(
                            (Int32(step.send[0]) * size_packs) // world,
                            (Int32(step.send[1]) * size_packs) // world,
                            index, stride, peer_slot, output_base, send_slot,
                            phase < 2 * steps - 1,
                        )
                cute.arch.sync_threads()
                if cutlass.const_expr(phase < 2 * steps - 1):
                    self._ring_phase(thread, Int32(gdim), phase_counter_ptr + Int64(phase) * phase_counter_stride,
                                     ctrl_base, seq, phase + 1)

            # 3. the last block to finish publishes the next epoch
            fence_sc_gpu()
            cute.arch.sync_threads()
            if thread == Int32(0):
                prior = atomic_add_relaxed_gpu_u32(tail_counter_ptr, Uint32(1))
                if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                    fence_sc_gpu()
                    # Every block's timeout stores precede its tail arrival, so
                    # the error word is final here; a failed sequence keeps the
                    # epoch and every later launch does nothing.
                    if ld_relaxed_sys_u32(ctrl_base + Int64(_ERROR_SEQ)) == Uint32(0):
                        st_release_gpu_u32(epoch_ptr, seq)


_LAUNCHERS: dict[tuple, Callable[..., None]] = {}


def launcher_key(dtype_name: str, world_size: int, rank: int, threads: int, slots: int, flag_stride: int,
                 lane_count: int, device_index: int) -> tuple:
    return ("swing", str(dtype_name), int(world_size), int(rank), int(threads), int(slots), int(flag_stride),
            int(lane_count), int(device_index))


def is_launcher_prepared(*key) -> bool:
    return launcher_key(*key) in _LAUNCHERS


def get_launcher(dtype_name: str, world_size: int, rank: int, threads: int, slots: int, flag_stride: int,
                 lane_count: int, device_index: int) -> Callable[..., None]:
    """The compiled launcher of one specialization (compiled once per process).

    Launch arguments: input address, output address, packs, bytes, receive
    base, flag base, send base, control base, slot bytes, epoch address, stage
    counter, phase-1 arrival counter, byte stride between the phase counters,
    tail counter, poison address, spin limit, grid blocks.
    """
    key = launcher_key(dtype_name, world_size, rank, threads, slots, flag_stride, lane_count, device_index)
    cached = _LAUNCHERS.get(key)
    if cached is not None:
        return cached
    launch = SwingAllReduce(dtype_name, world_size, rank, threads, slots, flag_stride, lane_count)
    compiled = compile_launcher(
        launch, make_pointer(16), make_pointer(16),
        1, 16, 16, 16, 16, 16, 4096, 16, 16, 16, 4, 16, 16, 1, 1, current_cuda_stream(),
        name="sircl Swing all-reduce", cache_key=key,
    )

    def run(input_address: int, output_address: int, size_packs: int, nbytes: int, recv_base: int,
            flag_base: int, send_base: int, ctrl_base: int, slot_bytes: int, epoch_address: int,
            stage_counter: int, phase_counter: int, phase_stride: int, tail_counter: int, poison_address: int,
            spin_limit: int, grid_blocks: int) -> None:
        compiled(
            make_pointer(input_address), make_pointer(output_address), int(size_packs), int(nbytes),
            int(recv_base), int(flag_base), int(send_base), int(ctrl_base), int(slot_bytes), int(epoch_address),
            int(stage_counter), int(phase_counter), int(phase_stride), int(tail_counter), int(poison_address),
            int(spin_limit), int(grid_blocks), current_cuda_stream(),
        )

    _LAUNCHERS[key] = run
    return run


__all__ = ["DESCRIBED_CODE", "DTYPE_PACK_ELEMS", "SwingAllReduce", "get_launcher", "is_launcher_prepared",
           "launcher_key"]
