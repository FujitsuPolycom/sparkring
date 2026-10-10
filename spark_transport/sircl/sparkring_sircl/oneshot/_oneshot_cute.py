"""CuTe DSL kernel of the ring session's one-shot all-reduce.

One launch performs one complete all-reduce:

1. stage: copy the device input into ``send[seq & 1]`` of the pinned arena;
2. doorbell: the last block to finish staging writes the byte count (command
   ring word 1 and the slot's op word, op code 0) and then ``seq`` into the
   doorbell word that the native progress thread polls;
3. wait: thread ``t < world * L`` waits for lane ``t % L`` of source rank
   ``t // L`` (every peer has ``L`` lanes); flag line
   ``(source * SLOTS + slot) * L + lane`` of namespace 0. A peer's progress
   thread writes that flag after the lane's stripe on the same queue pair. A
   wait that lasts longer than the session's wait limit (command ring word
   7, microseconds, read at launch; ``spin_limit`` polls when the word is 0)
   writes the missing peer and lane (words 3 and 6), then the sequence (word
   2), and poisons the session;
4. reduce: sum the local input and every peer slot in rank order 0..W-1 in
   float32 and round once, so every rank stores identical bits;
5. epoch: the last block to finish advances the device-resident epoch unless
   a wait timed out, so the sequence is a runtime value and CUDA graph replay
   stays correct; a poisoned session's later launches do nothing.

Staging and tail arrivals use separate counters per power-of-two grid size,
so launches of different sizes may interleave in one graph.

Origin: b12x RoCEnante ``b12x/comm/roce/_oneshot_cute.py`` (Apache-2.0, Local
Inference Lab); the lane-count wait and the command-ring words of SIRCL's
ring sessions are SIRCL's.

Flag polling: the flags of every peer's lanes live in pinned host memory, which the GPU reads with
system-scope loads. Loads of one host line from many blocks queue behind each other, so a flag that every
block polls is seen by the last block about one load time per polling thread after it landed. With
``one_block_polls`` only block 0's threads poll the flags; block 0 then stores ``seq`` into the arrival
word of the namespace (device memory, ``arrival_ptr``: namespace 0, then namespace 1) and the other blocks
wait for that word (``spin_until_eq_or_poison_gpu``), so the host loads per wait do not grow with the grid.
A word holds the sequence of the newest op that wrote it, and every op of these kernels writes both, so it
equals the current sequence early only when 2^32 ops of other kernels ran between two ops of these
kernels.
The one-shot op uses the namespace-0 arrival word and also stores ``seq`` into the namespace-1 word.
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

from ..protocol import Ctrl, PACK_BYTES
from ._compile import compile_launcher, current_cuda_stream, current_stream_handle, fast_launch, make_pointer
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
    st_relaxed_sys_u32,
    spin_until_eq_or_poison_gpu,
    st_release_gpu_u32,
    u32_as_f32,
    unpack_bf16x2,
    unpack_f16x2,
)

DTYPE_PACK_ELEMS = {"float32": 4, "float16": 8, "bfloat16": 8}
_NBYTES = 4 * int(Ctrl.NBYTES)
_OP_WORD = 4 * int(Ctrl.OP_WORD)
_ERROR_SEQ = 4 * int(Ctrl.ERROR_SEQ)
_MISSING_PEER = 4 * int(Ctrl.MISSING_PEER)
_MISSING_LANE = 4 * int(Ctrl.MISSING_LANE)
_WAIT_LIMIT = 4 * int(Ctrl.WAIT_LIMIT_US)


class OneShotAllReduce:
    """One kernel specialization: dtype, group size, rank, block size, layout constants."""

    def __init__(self, dtype_name: str, world_size: int, rank: int, threads: int, slots: int,
                 flag_stride: int, lane_count: int, one_block_polls: bool = False) -> None:
        if dtype_name not in DTYPE_PACK_ELEMS:
            raise ValueError(f"unsupported one-shot all-reduce dtype {dtype_name!r}")
        if int(threads) < int(world_size) * int(lane_count):
            raise ValueError(
                "one-shot kernels need threads >= world_size * lane_count (one waiter per lane "
                f"flag), got threads={threads} world_size={world_size} lane_count={lane_count}"
            )
        self._dtype_name = dtype_name
        self._pack_elems = DTYPE_PACK_ELEMS[dtype_name]
        self._world_size = int(world_size)
        self._rank = int(rank)
        self._threads = int(threads)
        self._slots = int(slots)
        self._flag_stride = int(flag_stride)
        self._lanes = int(lane_count)
        self._one_block_polls = bool(one_block_polls)

    @cute.jit
    def _poll_lanes(self, thread: Int32, flag_base: Int64, ctrl_base: Int64, poison_ptr: Int64, slot: Int64,
                    seq: Uint32, spin_limit: Uint32) -> None:
        """Thread ``t < W * L`` waits for lane ``t % L`` of rank ``t // L``."""
        if thread < Int32(self._world_size * self._lanes):
            source = thread // Int32(self._lanes)
            lane = thread - source * Int32(self._lanes)
            if source != Int32(self._rank):
                flag_addr = flag_base + (
                    (Int64(source) * Int64(self._slots) + slot) * Int64(self._lanes) + Int64(lane)
                ) * Int64(self._flag_stride)
                limit_us = ld_relaxed_sys_u32(ctrl_base + Int64(_WAIT_LIMIT))
                timed_out = spin_until_eq_timed_sys(flag_addr, seq, spin_limit, limit_us)
                if timed_out != Uint32(0):
                    st_relaxed_sys_u32(ctrl_base + Int64(_MISSING_PEER), Uint32(source))
                    st_relaxed_sys_u32(ctrl_base + Int64(_MISSING_LANE), Uint32(lane))
                    fence_sc_sys()
                    st_relaxed_sys_u32(ctrl_base + Int64(_ERROR_SEQ), seq)
                    st_release_gpu_u32(poison_ptr, Uint32(1))

    @cute.jit
    def _wait_lanes(self, thread: Int32, block: Int32, flag_base: Int64, ctrl_base: Int64, poison_ptr: Int64,
                    arrival_ptr: Int64, slot: Int64, seq: Uint32, spin_limit: Uint32) -> None:
        """Every lane flag of every peer: polled by this block, or with one polling block by block 0, which
        then stores ``seq`` into both arrival words for the other blocks."""
        if cutlass.const_expr(self._one_block_polls):
            if block == Int32(0):
                self._poll_lanes(thread, flag_base, ctrl_base, poison_ptr, slot, seq, spin_limit)
                cute.arch.sync_threads()
                if thread == Int32(0):
                    st_release_gpu_u32(arrival_ptr + Int64(4), seq)
                    st_release_gpu_u32(arrival_ptr, seq)
            else:
                if thread == Int32(0):
                    spin_until_eq_or_poison_gpu(arrival_ptr, seq, poison_ptr)
        else:
            self._poll_lanes(thread, flag_base, ctrl_base, poison_ptr, slot, seq, spin_limit)

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
        recv_base: Int64,
        flag_base: Int64,
        send_base: Int64,
        ctrl_base: Int64,
        slot_bytes: Int64,
        epoch_ptr: Int64,
        stage_counter_ptr: Int64,
        tail_counter_ptr: Int64,
        poison_ptr: Int64,
        arrival_ptr: Int64,
        spin_limit: Uint32,
        grid_x: Int32,
        trace_ring: Int64,
        captured: Int32,
        stream: cuda.CUstream,
    ) -> None:
        """Host entry: launch with runtime scalars (``trace_ring`` and ``captured`` are unused)."""
        self.kernel(
            input_ptr, output_ptr, size_packs, nbytes, recv_base, flag_base, send_base, ctrl_base,
            slot_bytes, epoch_ptr, stage_counter_ptr, tail_counter_ptr, poison_ptr, arrival_ptr, spin_limit,
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
        tail_counter_ptr: Int64,
        poison_ptr: Int64,
        arrival_ptr: Int64,
        spin_limit: Uint32,
    ) -> None:
        """Device kernel: stage, doorbell, wait for lane flags, reduce, advance the epoch."""
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

        poisoned = ld_relaxed_gpu_u32(poison_ptr)
        if poisoned == Uint32(0):
            # 1. stage the input into the pinned send slot
            stage_index = index
            while stage_index < size_packs:
                words = ld_global_v4_u32(input_base + Int64(stage_index) * Int64(PACK_BYTES))
                st_global_v4_u32(send_slot + Int64(stage_index) * Int64(PACK_BYTES),
                                 words[0], words[1], words[2], words[3])
                stage_index += stride
            cute.arch.sync_threads()

            # 2. the last block to finish staging rings the doorbell
            if Int32(tidx) == Int32(0):
                fence_sc_sys()
                prior = atomic_add_relaxed_gpu_u32(stage_counter_ptr, Uint32(1))
                if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                    st_relaxed_sys_u32(ctrl_base + Int64(_NBYTES), Uint32(nbytes))
                    st_relaxed_sys_u32(ctrl_base + Int64(_OP_WORD) + slot * Int64(4), Uint32(nbytes))
                    fence_sc_sys()
                    st_relaxed_sys_u32(ctrl_base, seq)

            # 3. wait for every lane flag of every peer
            self._wait_lanes(Int32(tidx), Int32(bidx), flag_base, ctrl_base, poison_ptr, arrival_ptr, slot, seq,
                             spin_limit)
            cute.arch.sync_threads()
            # A timed-out wait in this block leaves a peer slot unreliable:
            # skip the data phase so nothing derived from it is stored.
            failed = ld_relaxed_gpu_u32(poison_ptr)
            if failed == Uint32(0):
                # 4. reduce in fixed rank order so every rank stores identical bits
                reduce_index = index
                while reduce_index < size_packs:
                    accumulator = cute.make_rmem_tensor((self._pack_elems,), cutlass.Float32)
                    offset = Int64(reduce_index) * Int64(PACK_BYTES)
                    for src in cutlass.range_constexpr(self._world_size):
                        if cutlass.const_expr(src == self._rank):
                            words = ld_global_v4_u32(input_base + offset)
                        else:
                            peer_slot = recv_base + (Int64(src) * Int64(self._slots) + slot) * slot_bytes
                            words = ld_relaxed_sys_v4_u32(peer_slot + offset)
                        self._accumulate_words(accumulator, words, src == 0)
                    self._store_accumulator(output_base + offset, accumulator)
                    reduce_index += stride

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


def _example_pointer():
    return make_pointer(16)


def _pointer(address: int):
    return make_pointer(address)


_LAUNCHERS: dict[tuple, Callable[..., None]] = {}


def launcher_key(dtype_name: str, world_size: int, rank: int, threads: int, slots: int,
                 flag_stride: int, lane_count: int, device_index: int, trace: bool = False,
                 one_block_polls: bool = False) -> tuple:
    return ("oneshot", str(dtype_name), int(world_size), int(rank), int(threads), int(slots),
            int(flag_stride), int(lane_count), int(device_index), bool(trace), bool(one_block_polls))


def is_launcher_prepared(*key) -> bool:
    return launcher_key(*key) in _LAUNCHERS


def get_launcher(dtype_name: str, world_size: int, rank: int, threads: int, slots: int,
                 flag_stride: int, lane_count: int, device_index: int,
                 trace: bool = False, one_block_polls: bool = False) -> Callable[..., None]:
    """The compiled launcher of one specialization (compiled once per process). With
    ``one_block_polls`` a launch also takes the address of the two arrival words (``arrival_address``,
    device memory)."""
    key = launcher_key(dtype_name, world_size, rank, threads, slots, flag_stride, lane_count,
                       device_index, trace, one_block_polls)
    cached = _LAUNCHERS.get(key)
    if cached is not None:
        return cached
    if trace:
        raise ValueError("one-shot kernel tracing is unsupported by this SIRCL build")
    launch = OneShotAllReduce(dtype_name, world_size, rank, threads, slots, flag_stride, lane_count,
                              one_block_polls)
    compiled = compile_launcher(
        launch, _example_pointer(), _example_pointer(),
        1, 16, 16, 16, 16, 16, 4096, 16, 16, 16, 16, 16, 1, 1, 0, 0, current_cuda_stream(),
        name="sircl one-shot all-reduce", cache_key=key,
    )
    fast = fast_launch(compiled, "sircl one-shot all-reduce")
    device = int(device_index)

    def run(input_address: int, output_address: int, size_packs: int, nbytes: int, recv_base: int,
            flag_base: int, send_base: int, ctrl_base: int, slot_bytes: int, epoch_address: int,
            stage_counter: int, tail_counter: int, poison_address: int, spin_limit: int,
            grid_blocks: int, trace_ring: int = 0, captured: int = 0, arrival_address: int = 0) -> None:
        if one_block_polls and not arrival_address:
            raise ValueError("a launcher with one polling block needs the arrival words' address")
        if fast is not None:
            fast(input_address, output_address, size_packs, nbytes, recv_base, flag_base, send_base, ctrl_base,
                 slot_bytes, epoch_address, stage_counter, tail_counter, poison_address, arrival_address,
                 spin_limit, grid_blocks, trace_ring, captured, current_stream_handle(device))
            return
        compiled(
            _pointer(input_address), _pointer(output_address), int(size_packs), int(nbytes),
            int(recv_base), int(flag_base), int(send_base), int(ctrl_base), int(slot_bytes),
            int(epoch_address), int(stage_counter), int(tail_counter), int(poison_address),
            int(arrival_address), int(spin_limit), int(grid_blocks), int(trace_ring), int(captured),
            current_cuda_stream(),
        )

    _LAUNCHERS[key] = run
    return run
