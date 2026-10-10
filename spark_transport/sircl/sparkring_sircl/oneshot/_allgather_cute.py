"""CuTe DSL kernel of the ring session's one-shot all-gather.

Same wire protocol as the one-shot all-reduce (stage, doorbell with op code
0, wait on every lane flag of every peer within the session's wait limit,
advance the epoch), with the reduction replaced by a strided copy that writes
the concatenated output.

Concatenating contiguous shards along a dimension is concatenating rows: view
each shard as rows of 16-byte packs; shard ``s`` lands at column block ``s`` of
every output row. Dimension 0 is the case of one row; the last dimension is a
row width of the shard's last extent. The local shard is copied from the
input; peer shards are read in place from the NIC-written receive slots with
system-scope loads.

One op may also move a tile of a larger gather (``all_gather_large``): pack
``i`` of the op is column ``i % tile_cols`` of tile row ``i // tile_cols``;
it is read from the input at ``row * in_row_stride + col`` packs, travels
compactly at pack ``i`` of the slot, and shard ``s`` lands in the output at
``row * out_row_stride + s * out_src_stride + col`` packs. The plain launcher
(arguments without the three strides) is the tile of every row of a
contiguous shard:
``tile_cols = in_row_stride = out_src_stride = row_packs`` and
``out_row_stride = W * row_packs``.

Origin: b12x RoCEnante ``b12x/comm/roce/_allgather_cute.py`` (Apache-2.0, Local
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
The all-gather uses the namespace-0 arrival word and also stores ``seq`` into the namespace-1 word.
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
    fence_sc_gpu,
    fence_sc_sys,
    ld_global_v4_u32,
    ld_relaxed_gpu_u32,
    ld_relaxed_sys_u32,
    ld_relaxed_sys_v4_u32,
    st_global_v4_u32,
    spin_until_eq_or_poison_gpu,
    st_relaxed_sys_u32,
    st_release_gpu_u32,
)

_NBYTES = 4 * int(Ctrl.NBYTES)
_OP_WORD = 4 * int(Ctrl.OP_WORD)
_ERROR_SEQ = 4 * int(Ctrl.ERROR_SEQ)
_MISSING_PEER = 4 * int(Ctrl.MISSING_PEER)
_MISSING_LANE = 4 * int(Ctrl.MISSING_LANE)
_WAIT_LIMIT = 4 * int(Ctrl.WAIT_LIMIT_US)


class OneShotAllGather:
    """One kernel specialization: group size, rank, block size, layout constants."""

    def __init__(self, world_size: int, rank: int, threads: int, slots: int, flag_stride: int,
                 lane_count: int, one_block_polls: bool = False) -> None:
        if int(threads) < int(world_size) * int(lane_count):
            raise ValueError(
                "one-shot kernels need threads >= world_size * lane_count, got "
                f"threads={threads} world_size={world_size} lane_count={lane_count}"
            )
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
            source_rank = thread // Int32(self._lanes)
            lane = thread - source_rank * Int32(self._lanes)
            if source_rank != Int32(self._rank):
                flag_addr = flag_base + (
                    (Int64(source_rank) * Int64(self._slots) + slot) * Int64(self._lanes) + Int64(lane)
                ) * Int64(self._flag_stride)
                limit_us = ld_relaxed_sys_u32(ctrl_base + Int64(_WAIT_LIMIT))
                timed_out = spin_until_eq_timed_sys(flag_addr, seq, spin_limit, limit_us)
                if timed_out != Uint32(0):
                    st_relaxed_sys_u32(ctrl_base + Int64(_MISSING_PEER), Uint32(source_rank))
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
    def __call__(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        shard_packs: Int32,
        nbytes: Int32,
        tile_cols: Int32,
        in_row_stride: Int64,
        out_row_stride: Int64,
        out_src_stride: Int64,
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
        stream: cuda.CUstream,
    ) -> None:
        """Host entry: launch the all-gather kernel with runtime scalars."""
        self.kernel(
            input_ptr, output_ptr, shard_packs, nbytes, tile_cols, in_row_stride, out_row_stride,
            out_src_stride, recv_base, flag_base, send_base, ctrl_base, slot_bytes, epoch_ptr,
            stage_counter_ptr, tail_counter_ptr, poison_ptr, arrival_ptr, spin_limit,
        ).launch(grid=(grid_x, 1, 1), block=[self._threads, 1, 1], cluster=(1, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        shard_packs: Int32,
        nbytes: Int32,
        tile_cols: Int32,
        in_row_stride: Int64,
        out_row_stride: Int64,
        out_src_stride: Int64,
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
        """Device kernel: stage, doorbell, wait for lane flags, strided copy, advance the epoch."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        input_base = Int64(input_ptr.toint())
        output_base = Int64(output_ptr.toint())
        epoch = ld_relaxed_gpu_u32(epoch_ptr)
        seq = epoch + Uint32(1)
        slot = Int64(seq & Uint32(self._slots - 1))
        send_slot = send_base + slot * slot_bytes
        index = Int32(bidx) * Int32(self._threads) + Int32(tidx)
        stride = Int32(gdim) * Int32(self._threads)

        poisoned = ld_relaxed_gpu_u32(poison_ptr)
        if poisoned == Uint32(0):
            # 1. stage the local tile compactly into the pinned send slot
            stage_index = index
            while stage_index < shard_packs:
                row = stage_index // tile_cols
                col = stage_index - row * tile_cols
                source = input_base + (Int64(row) * in_row_stride + Int64(col)) * Int64(PACK_BYTES)
                words = ld_global_v4_u32(source)
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
            failed = ld_relaxed_gpu_u32(poison_ptr)
            if failed == Uint32(0):
                # 4. scatter: shard s of tile row r lands at r * out_row_stride + s * out_src_stride
                for src in cutlass.range_constexpr(self._world_size):
                    copy_index = index
                    while copy_index < shard_packs:
                        row = copy_index // tile_cols
                        col = copy_index - row * tile_cols
                        dest = output_base + (
                            Int64(row) * out_row_stride + Int64(src) * out_src_stride + Int64(col)
                        ) * Int64(PACK_BYTES)
                        if cutlass.const_expr(src == self._rank):
                            words = ld_global_v4_u32(
                                input_base + (Int64(row) * in_row_stride + Int64(col)) * Int64(PACK_BYTES))
                        else:
                            peer_slot = recv_base + (Int64(src) * Int64(self._slots) + slot) * slot_bytes
                            words = ld_relaxed_sys_v4_u32(peer_slot + Int64(copy_index) * Int64(PACK_BYTES))
                        st_global_v4_u32(dest, words[0], words[1], words[2], words[3])
                        copy_index += stride

            # 5. the last block to finish publishes the next epoch
            fence_sc_gpu()
            cute.arch.sync_threads()
            if Int32(tidx) == Int32(0):
                prior = atomic_add_relaxed_gpu_u32(tail_counter_ptr, Uint32(1))
                if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                    fence_sc_gpu()
                    if ld_relaxed_sys_u32(ctrl_base + Int64(_ERROR_SEQ)) == Uint32(0):
                        st_release_gpu_u32(epoch_ptr, seq)


def _example_pointer():
    return make_pointer(16)


def _pointer(address: int):
    return make_pointer(address)


_COMPILED: dict[tuple, object] = {}
_FAST: dict[tuple, object] = {}
_LAUNCHERS: dict[tuple, Callable[..., None]] = {}
_TILED: dict[tuple, Callable[..., None]] = {}


def launcher_key(world_size: int, rank: int, threads: int, slots: int, flag_stride: int,
                 lane_count: int, device_index: int, one_block_polls: bool = False) -> tuple:
    return ("allgather", int(world_size), int(rank), int(threads), int(slots), int(flag_stride),
            int(lane_count), int(device_index), bool(one_block_polls))


def _compiled(key: tuple, world_size: int, rank: int, threads: int, slots: int, flag_stride: int,
              lane_count: int, one_block_polls: bool):
    compiled = _COMPILED.get(key)
    if compiled is None:
        launch = OneShotAllGather(world_size, rank, threads, slots, flag_stride, lane_count, one_block_polls)
        compiled = compile_launcher(
            launch, _example_pointer(), _example_pointer(),
            1, 16, 1, 1, 1, 1, 16, 16, 16, 16, 4096, 16, 16, 16, 16, 16, 1, 1, current_cuda_stream(),
            name="sircl one-shot all-gather", cache_key=key,
        )
        _COMPILED[key] = compiled
        _FAST[key] = fast_launch(compiled, "sircl one-shot all-gather")
    return compiled


def get_tiled_launcher(world_size: int, rank: int, threads: int, slots: int, flag_stride: int,
                       lane_count: int, device_index: int, one_block_polls: bool = False) -> Callable[..., None]:
    """The tiled launcher of one specialization (compiled once per process, shared with
    :func:`get_launcher`).

    Arguments: input address, output address, packs, bytes, tile columns, input row
    stride, output row stride, output shard stride (strides in packs), then the plain
    launcher's arguments from the receive base on; with ``one_block_polls`` also the
    address of the two arrival words (``arrival_address``, device memory).
    """
    key = launcher_key(world_size, rank, threads, slots, flag_stride, lane_count, device_index, one_block_polls)
    cached = _TILED.get(key)
    if cached is not None:
        return cached
    compiled = _compiled(key, world_size, rank, threads, slots, flag_stride, lane_count, one_block_polls)
    fast = _FAST.get(key)
    device = int(device_index)

    def run_tiled(input_address: int, output_address: int, shard_packs: int, nbytes: int, tile_cols: int,
                  in_row_stride: int, out_row_stride: int, out_src_stride: int, recv_base: int,
                  flag_base: int, send_base: int, ctrl_base: int, slot_bytes: int, epoch_address: int,
                  stage_counter: int, tail_counter: int, poison_address: int, spin_limit: int,
                  grid_blocks: int, arrival_address: int = 0) -> None:
        if one_block_polls and not arrival_address:
            raise ValueError("a launcher with one polling block needs the arrival words' address")
        if fast is not None:
            fast(input_address, output_address, shard_packs, nbytes, tile_cols, in_row_stride, out_row_stride,
                 out_src_stride, recv_base, flag_base, send_base, ctrl_base, slot_bytes, epoch_address,
                 stage_counter, tail_counter, poison_address, arrival_address, spin_limit, grid_blocks,
                 current_stream_handle(device))
            return
        compiled(
            _pointer(input_address), _pointer(output_address), int(shard_packs), int(nbytes),
            int(tile_cols), int(in_row_stride), int(out_row_stride), int(out_src_stride),
            int(recv_base), int(flag_base), int(send_base), int(ctrl_base), int(slot_bytes),
            int(epoch_address), int(stage_counter), int(tail_counter), int(poison_address),
            int(arrival_address), int(spin_limit), int(grid_blocks), current_cuda_stream(),
        )

    _TILED[key] = run_tiled
    return run_tiled


def get_launcher(world_size: int, rank: int, threads: int, slots: int, flag_stride: int,
                 lane_count: int, device_index: int, one_block_polls: bool = False) -> Callable[..., None]:
    """The plain all-gather launcher: input, output, packs, bytes, row packs, then the
    arena, counter and limit arguments, and ``arrival_address`` as for the tiled launcher
    (compiled once per process)."""
    key = launcher_key(world_size, rank, threads, slots, flag_stride, lane_count, device_index, one_block_polls)
    cached = _LAUNCHERS.get(key)
    if cached is not None:
        return cached
    tiled = get_tiled_launcher(world_size, rank, threads, slots, flag_stride, lane_count, device_index,
                               one_block_polls)
    world = int(world_size)

    def run(input_address: int, output_address: int, shard_packs: int, nbytes: int, row_packs: int,
            recv_base: int, flag_base: int, send_base: int, ctrl_base: int, slot_bytes: int,
            epoch_address: int, stage_counter: int, tail_counter: int, poison_address: int,
            spin_limit: int, grid_blocks: int, arrival_address: int = 0) -> None:
        tiled(input_address, output_address, shard_packs, nbytes, row_packs, row_packs, world * row_packs,
              row_packs, recv_base, flag_base, send_base, ctrl_base, slot_bytes, epoch_address,
              stage_counter, tail_counter, poison_address, spin_limit, grid_blocks,
              arrival_address=arrival_address)

    _LAUNCHERS[key] = run
    return run
