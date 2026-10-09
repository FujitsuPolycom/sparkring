"""CuTe DSL kernel: the DCP all-to-all combine's exchange with the pack fused into the scatter's staging.

The image's DCP all-to-all combine on SIRCL (``dcp_a2a_lse_reduce`` as SIRCL's
``dcp_all_to_all`` shim runs it) is three launches per attention layer: vLLM's
Triton pack of the attention output and LSE into a ``[world, rows, heads, 514]``
BF16 send buffer, the DCP session's all-to-all (one scatter op), and vLLM's
Triton unpack-and-combine. This kernel is that scatter op with the pack inside
it: the pinned SIRCL build's scatter kernel in copy mode (``sparkring_sircl/oneshot/
_scatter_cute.py``: the op word with op code 3, the doorbell, the timed flag
waits, the poison word, the per-grid arrival counters and the epoch, so the
progress thread and every other collective of the session see an ordinary
scatter op), with two differences:

1. staging reads each destination ``j``'s share straight from the attention
   output ``out[rows, world * heads, 512]`` (BF16) and the LSE
   ``lse[rows, world * heads]`` (FP32) and writes chunk ``j`` in the wire
   format of ``layout`` (data packs, then LSE packs): the same
   ``rows * heads * 1028`` bytes per chunk as vLLM's records;
2. the copy phase stores only the peers' chunks (source ``s`` at
   ``recv + s * dst_stride``); ``kernels.wire_combine`` reads this rank's own
   share from ``out`` and ``lse`` in place.

Every staged and copied pack is 16 bytes moved unchanged, so the combine sees
the bits vLLM's pack would have sent. The launch geometry, the counters and
the stream rules are the session's (``runtime._launch_scatter_pack``).
"""

# Annotations stay evaluated (no postponed evaluation): the CuTe DSL reads each
# launch parameter's type from its annotation object.

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
from sparkring_sircl.oneshot._compile import compile_launcher, current_cuda_stream, make_pointer
from sparkring_sircl.oneshot._cute_batch import ld_v4_u32_batch
from sparkring_sircl.oneshot._cute_intrinsics import (
    atomic_add_relaxed_gpu_u32,
    fence_sc_gpu,
    fence_sc_sys,
    ld_global_v4_u32,
    ld_relaxed_gpu_u32,
    ld_relaxed_sys_u32,
    st_global_v4_u32,
    st_release_gpu_u32,
    st_relaxed_sys_u32,
)
from sparkring_sircl.oneshot._timed_wait import spin_until_eq_timed_sys
from sparkring_sircl.protocol import OP_SHIFT, PACK_BYTES, Ctrl, Op

from . import layout as L

SCATTER_CODE = int(Op.SCATTER) << OP_SHIFT
_NBYTES = 4 * int(Ctrl.NBYTES)
_OP_WORD = 4 * int(Ctrl.OP_WORD)
_ERROR_SEQ = 4 * int(Ctrl.ERROR_SEQ)
_MISSING_PEER = 4 * int(Ctrl.MISSING_PEER)
_MISSING_LANE = 4 * int(Ctrl.MISSING_LANE)
_WAIT_LIMIT = 4 * int(Ctrl.WAIT_LIMIT_US)


def _copy_peer_chunks(launch, packs, recv_base, slot, slot_bytes, zero, output_base, own_lo, dst_stride):
    """Copy pack ``q`` (for each ``q`` in ``packs``) of every peer's chunk for this rank.

    Plain Python evaluated while the kernel is traced. The chunk that peer ``s``
    sent to this rank lies at pack ``own_lo + q`` of ``recv[s][slot]``
    (system-scope loads of NIC-written memory, in one batch); it lands at
    ``output_base + s * dst_stride``. The own source is skipped.
    """
    addrs, system, sources = [], [], []
    for q in packs:
        offset = Int64(own_lo + q) * Int64(PACK_BYTES)
        for source in range(launch._world_size):
            if source == launch._rank:
                continue
            addrs.append(recv_base + (Int64(source) * Int64(launch._slots) + slot) * slot_bytes + offset)
            system.append(True)
            sources.append((q, source))
    words = ld_v4_u32_batch(addrs, system, zero)
    for (q, source), w in zip(sources, words):
        st_global_v4_u32(output_base + Int64(source) * dst_stride + Int64(q) * Int64(PACK_BYTES),
                         w[0], w[1], w[2], w[3])


class ScatterPackLaunch:
    """One kernel specialization: group size, rank, the session's layout constants and the head count."""

    def __init__(self, world_size: int, rank: int, threads: int, slots: int, flag_stride: int, lane_count: int,
                 heads: int) -> None:
        if int(world_size) < 2:
            raise ValueError("the packed all-to-all needs at least two ranks")
        if int(threads) < int(world_size) * int(lane_count):
            raise ValueError(f"the packed all-to-all needs threads >= world_size * lane_count (one waiter per lane "
                             f"flag), got threads={threads} world_size={world_size} lane_count={lane_count}")
        if int(slots) < 1 or int(slots) & (int(slots) - 1):
            raise ValueError(f"the session's slot count must be a power of two, got {slots}")
        geometry = L.WireGeometry(1, heads)       # validates the head count
        self._world_size = int(world_size)
        self._rank = int(rank)
        self._threads = int(threads)
        self._slots = int(slots)
        self._flag_stride = int(flag_stride)
        self._lanes = int(lane_count)
        # Compile-time head geometry: every index split is a shift or a mask (layout's helpers).
        self._heads = int(heads)
        self._row_packs = geometry.row_packs
        self._row_shift = geometry.row_shift
        self._head_shift = geometry.head_shift
        self._lse_row_packs = geometry.lse_row_packs
        self._lse_shift = geometry.lse_shift

    @cute.jit
    def __call__(
        self,
        out_ptr: cute.Pointer,
        lse_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        size_packs: Int32,
        nbytes: Int32,
        chunk_packs: Int32,
        rows: Int32,
        out_row_stride: Int64,
        out_head_stride: Int64,
        lse_row_stride: Int64,
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
        """Host entry: launch the kernel with runtime scalars."""
        self.kernel(
            out_ptr, lse_ptr, output_ptr, size_packs, nbytes, chunk_packs, rows, out_row_stride, out_head_stride,
            lse_row_stride, dst_stride, recv_base, flag_base, send_base, ctrl_base, slot_bytes, epoch_ptr,
            stage_counter_ptr, tail_counter_ptr, poison_ptr, spin_limit,
        ).launch(grid=(grid_x, 1, 1), block=[self._threads, 1, 1], cluster=(1, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        out_ptr: cute.Pointer,
        lse_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        size_packs: Int32,
        nbytes: Int32,
        chunk_packs: Int32,
        rows: Int32,
        out_row_stride: Int64,
        out_head_stride: Int64,
        lse_row_stride: Int64,
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
        """Stage the peers' shares in the wire format, doorbell, wait, copy the peers' chunks, advance the epoch."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        out_base = Int64(out_ptr.toint())
        lse_base = Int64(lse_ptr.toint())
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

        # A recorded timeout poisons the session: later launches do nothing.
        poisoned = ld_relaxed_gpu_u32(poison_ptr)
        if poisoned == Uint32(0):
            # 1. stage every peer's share into its chunk of the send slot: per
            #    destination j, the data packs (b, h, k) and then the LSE packs
            #    (b, q), each read from out / lse in place.
            data_packs = rows * Int32(self._row_packs)
            lse_packs = rows * Int32(self._lse_row_packs)
            for j in cutlass.range_constexpr(self._world_size):
                if cutlass.const_expr(j != self._rank):
                    chunk_slot = send_slot + Int64(j) * Int64(chunk_packs) * Int64(PACK_BYTES)
                    head_base = out_base + Int64(j * self._heads) * out_head_stride
                    within = index
                    while within < data_packs:
                        b, h, k = L.data_pack_source(within, Int32(self._row_shift), Int32(self._row_packs - 1),
                                                     Int32(self._head_shift), Int32(L.HEAD_PACKS - 1))
                        words = ld_global_v4_u32(head_base + Int64(b) * out_row_stride + Int64(h) * out_head_stride
                                                 + Int64(k) * Int64(PACK_BYTES))
                        st_global_v4_u32(chunk_slot + Int64(within) * Int64(PACK_BYTES),
                                         words[0], words[1], words[2], words[3])
                        within += stride
                    lse_slot = chunk_slot + Int64(data_packs) * Int64(PACK_BYTES)
                    lse_head = lse_base + Int64(j * self._heads * L.LSE_BYTES)
                    within = index
                    while within < lse_packs:
                        b, q = L.lse_pack_source(within, Int32(self._lse_shift), Int32(self._lse_row_packs - 1))
                        words = ld_global_v4_u32(lse_head + Int64(b) * lse_row_stride + Int64(q) * Int64(PACK_BYTES))
                        st_global_v4_u32(lse_slot + Int64(within) * Int64(PACK_BYTES),
                                         words[0], words[1], words[2], words[3])
                        within += stride
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
            # A wait that timed out leaves the peer slots unreliable: skip the
            # copy so nothing derived from them is stored.
            failed = ld_relaxed_gpu_u32(poison_ptr)
            if failed == Uint32(0):
                # 4. copy every peer's chunk for this rank; the own share stays in out / lse
                copy_index = index
                while copy_index < chunk_packs:
                    second = copy_index + stride
                    if second < chunk_packs:
                        _copy_peer_chunks(self, (copy_index, second), recv_base, slot, slot_bytes, zero,
                                          output_base, own_lo, dst_stride)
                    else:
                        _copy_peer_chunks(self, (copy_index,), recv_base, slot, slot_bytes, zero,
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


def launcher_key(world_size: int, rank: int, threads: int, slots: int, flag_stride: int, lane_count: int,
                 heads: int, device_index: int) -> tuple:
    """Cache key: every value the kernel bakes in."""
    return ("dcp-scatter-pack", int(world_size), int(rank), int(threads), int(slots), int(flag_stride),
            int(lane_count), int(heads), int(device_index))


def is_prepared(*key) -> bool:
    """Whether :func:`get_launcher` compiled this specialization in this process."""
    return launcher_key(*key) in _LAUNCHERS


def get_launcher(world_size: int, rank: int, threads: int, slots: int, flag_stride: int, lane_count: int,
                 heads: int, device_index: int) -> Callable[..., None]:
    """The compiled launcher of one specialization (compiled once per process).

    Launch arguments: attention output, LSE and receive buffer addresses, packs,
    bytes, chunk packs, rows, the output's row and head strides and the LSE's
    row stride (bytes), destination stride (bytes), receive base, flag base,
    send base, control base, slot bytes, epoch address, stage counter, tail
    counter, poison address, spin limit, grid blocks.
    """
    key = launcher_key(world_size, rank, threads, slots, flag_stride, lane_count, heads, device_index)
    cached = _LAUNCHERS.get(key)
    if cached is not None:
        return cached
    launch = ScatterPackLaunch(world_size, rank, threads, slots, flag_stride, lane_count, heads)
    compiled = compile_launcher(
        launch, make_pointer(16), make_pointer(16), make_pointer(16),
        1, 16, 1, 1, 16, 16, 16, 16, 16, 16, 16, 16, 4096, 16, 16, 16, 16, 1, 1, current_cuda_stream(),
        name="glm_dcp_decode_comm packed all-to-all", cache_key=key,
    )
    compiled_heads = int(heads)
    row_records = compiled_heads * L.RECORD_BYTES // PACK_BYTES

    def run(out_address: int, lse_address: int, output_address: int, size_packs: int, nbytes: int,
            chunk_packs: int, rows: int, out_row_stride: int, out_head_stride: int, lse_row_stride: int,
            dst_stride: int, recv_base: int, flag_base: int, send_base: int, ctrl_base: int, slot_bytes: int,
            epoch_address: int, stage_counter: int, tail_counter: int, poison_address: int, spin_limit: int,
            grid_blocks: int) -> None:
        if int(chunk_packs) != int(rows) * row_records or int(size_packs) != int(world_size) * int(chunk_packs):
            raise ValueError(f"packed all-to-all launcher compiled for {compiled_heads} heads on {world_size} ranks, "
                             f"called with {chunk_packs} chunk packs for {rows} rows and {size_packs} packs")
        compiled(
            make_pointer(out_address), make_pointer(lse_address), make_pointer(output_address), int(size_packs),
            int(nbytes), int(chunk_packs), int(rows), int(out_row_stride), int(out_head_stride), int(lse_row_stride),
            int(dst_stride), int(recv_base), int(flag_base), int(send_base), int(ctrl_base), int(slot_bytes),
            int(epoch_address), int(stage_counter), int(tail_counter), int(poison_address), int(spin_limit),
            int(grid_blocks), current_cuda_stream(),
        )

    _LAUNCHERS[key] = run
    return run


__all__ = ["SCATTER_CODE", "ScatterPackLaunch", "get_launcher", "is_prepared", "launcher_key"]
