"""CuTe DSL kernels: all-reduce fused with the residual add and RMSNorm, one CTA per row.

One launch performs, for a BF16 message of ``rows x hidden``, the all-reduce
of every rank's partial sum, the residual add and the RMSNorm that vLLM's
decoder runs after it (``fused_add_rms_norm``), and writes the normalized
rows and the new residual. The launch grid is one cooperative thread array
(CTA) per row with one thread per 16-byte pack of the row (768 threads for a
hidden size of 6144), so a CTA holds its whole row in registers and finishes
the norm with a block reduction; no CTA reads another CTA's data and no
grid-wide barrier exists. The layout follows b12x's fused one-shot all-reduce
+ RMSNorm for PCIe peer access (one wide CTA per row; ``b12x/comm/pcie``,
Apache-2.0, Local Inference Lab), carried over to the ring session's RDMA
wire protocol.

The wire protocol is the ring session's (``sparkring_sircl/oneshot/_oneshot_cute.py``
and ``_twoshot_cute.py``): the same send and receive slots, byte-count and op
words, op codes, doorbells, flags, sequence numbers and epoch. A rank running
these kernels exchanges data with ranks running the plain all-reduce kernels
for the same message and algorithm.

One-shot (``algorithm="oneshot"``):

1. every thread copies its input pack into ``send[seq & 1]``; the last CTA
   to arrive at the stage counter publishes the byte count and rings the
   doorbell;
2. the first ``world * lanes`` threads wait for the flags of the peer
   stripes that hold packs of the CTA's row (``_geometry.oneshot_wait``);
3. each thread loads its pack from every peer slot in one gated batch,
   adds the sources in rank order 0..W-1 in float32 and rounds to BF16
   (the all-reduce bits), adds the residual, stores the new residual, and the
   CTA reduces the row's sum of squares;
4. each thread normalizes its pack and stores it;
5. the last CTA to arrive at the tail counter advances the epoch.

Two-shot (``algorithm="twoshot"``): chunk ``j`` of the message is reduced by
rank ``j``.

1. every thread copies its input pack into the send slot unless the pack is
   in this rank's own chunk; the last CTA rings the scatter doorbell (op code
   1 in the slot's op word);
2. CTAs whose row intersects the own chunk wait for the scatter stripes that
   hold their packs (``_geometry.twoshot_scatter_wait``); their own-chunk
   threads sum every source in rank order and store the BF16 result into the
   own chunk of the send slot; the last CTA to arrive at the mid counter rings
   the gather doorbell;
3. every CTA waits for the gather stripes of the other owners' chunks that
   intersect its row (``_geometry.twoshot_gather_wait``) and every thread
   loads its reduced pack (a peer's slot, or the own chunk of the send slot):
   this unpack of the all-gather is where the residual add and the norm run,
   because here each CTA holds its complete reduced row;
4. normalize, store, advance the epoch as in the one-shot kernel.

The arithmetic is that of ``_reference`` operation for operation: float32
sums in rank order, one BF16 rounding (the plain all-reduce's bits), the
BF16 residual sum, the sum of squares in the order of vLLM's
``fused_add_rms_norm_kernel`` launched with 1024 threads (per-thread pack
sum, ``shfl.down`` tree per warp, warps added in order), IEEE division,
``rsqrt.approx.f32``, two float32 products and one BF16 rounding. Every
thread of a CTA computes the row's reciprocal RMS from the same shared warp
sums in the same order, and every rank reduces the same bits in the same
order, so all ranks store identical outputs independent of arrival order.

Synchronization. Arrival counters are three words of the caller's own
device buffer (stage, mid, tail), advanced with ``atom.inc`` bounded by the
grid size, so they return to 0 after every launch whatever the row count.
The epoch and poison words are the session's: a rank's collectives share
one sequence. Entry state (epoch, poison) is read by thread 0 and shared
through shared memory, so every thread of a CTA takes the same branch
around its barriers. A flag wait that lasts longer than the session's wait
limit (command ring word 7, microseconds; ``spin_limit`` polls when the word
is 0) records the sequence, peer and lane in the command ring and poisons the
session; a thread that sees the poison after its CTA's waits skips its stores,
the error word blocks the epoch advance, later launches do nothing and the
host raises, as with the plain kernels.

Every CTA of a launch must be resident at once: a CTA that waits for peers
depends on this rank's doorbell, which the last CTA to stage rings. The host
glue limits ``rows`` to the GPU's multiprocessor count; the launch bounds
(``max_number_threads``, one CTA per multiprocessor) keep the register use
within one CTA per multiprocessor.

Origin: SparkRing's fused-norm kernels, after b12x's PCIe fused all-reduce +
RMSNorm (see above).
"""

# Annotations stay evaluated (no postponed evaluation): the CuTe DSL reads each
# launch parameter's type from its annotation object.

import threading
from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from ..oneshot._compile import compile_launcher, current_cuda_stream
from ..oneshot._timed_wait import spin_until_eq_timed_sys
from ..protocol import OP_SHIFT, Ctrl, Op, phase_doorbell_word
from . import _geometry as geo
from ._ptx import (
    add_rn,
    atom_inc_relaxed_gpu_u32,
    bf16x2_to_f32,
    div_rn,
    f32x2_to_bf16x2,
    fence_sc_gpu,
    fence_sc_sys,
    ld_global_v4,
    ld_relaxed_gpu_u32,
    ld_relaxed_sys_u32,
    ld_sys_v4,
    ld_sys_v4_gated,
    min_s32,
    mul_rn,
    rsqrt_approx,
    shfl_down_f32,
    st_global_v4,
    st_relaxed_sys_u32,
    st_release_gpu_u32,
)

PACK_BYTES = geo.PACK_BYTES
TWO_SHOT_OP = int(Op.TWOSHOT) << OP_SHIFT  # op code 1 in the top bits of the slot's op word
CTRL_NBYTES = 4 * int(Ctrl.NBYTES)  # byte offsets in the command ring
CTRL_ERROR_SEQ = 4 * int(Ctrl.ERROR_SEQ)
CTRL_ERROR_PEER = 4 * int(Ctrl.MISSING_PEER)
CTRL_SLOT_WORDS = 4 * int(Ctrl.OP_WORD)
CTRL_ERROR_LANE = 4 * int(Ctrl.MISSING_LANE)
CTRL_WAIT_LIMIT = 4 * int(Ctrl.WAIT_LIMIT_US)
CTRL_PHASE1 = 4 * phase_doorbell_word(1)  # the gather doorbell
COUNTER_WORDS = 4  # stage, mid, tail, unused
SHUFFLE_OFFSETS = (1, 2, 4, 8, 16)  # CUB WarpReduceShfl order for a full warp


# --------------------------------------------------------------------------- per-pack arithmetic
# Plain Python evaluated while the kernel is traced: straight-line code, no
# branches, so the caller may run it unconditionally.


def _values(words):
    """Eight float32 values of one pack (four BF16 pairs), element order."""
    out = []
    for word in words:
        lo, hi = bf16x2_to_f32(word)
        out.append(lo)
        out.append(hi)
    return out


def _round(values):
    """Eight float32 values rounded to BF16 and packed into four words."""
    return tuple(f32x2_to_bf16x2(values[2 * i], values[2 * i + 1]) for i in range(4))


def _sum_in_rank_order(world: int, rank: int, own_words, peer_words):
    """Float32 sum of the eight values of one pack over every source, rank 0 first."""
    peers = iter(peer_words)
    acc = None
    for source in range(world):
        values = _values(own_words if source == rank else next(peers))
        acc = values if acc is None else [add_rn(a, v) for a, v in zip(acc, values)]
    return acc


def _peer_addresses(world: int, rank: int, slots: int, recv_base, slot, slot_bytes, offset):
    """Address of one pack in every peer's receive slot, in rank order (the own rank skipped)."""
    addresses = []
    for source in range(world):
        if source != rank:
            addresses.append(recv_base + (Int64(source) * Int64(slots) + slot) * slot_bytes + offset)
    return addresses


def _residual_add(reduced_words, residual_words):
    """``bf16(reduced + residual)`` per element: the packed new residual and its float32 values."""
    summed = []
    for a, b in zip(_values(reduced_words), _values(residual_words)):
        summed.append(add_rn(a, b))
    words = _round(summed)
    return words, _values(words)


def _pack_sum_of_squares(z):
    """``((p01 + p23) + p45) + p67`` with ``pij = zi*zi + zj*zj``, as vLLM's ``sum_squares``."""
    pairs = [add_rn(mul_rn(z[2 * k], z[2 * k]), mul_rn(z[2 * k + 1], z[2 * k + 1])) for k in range(4)]
    acc = pairs[0]
    for k in range(1, 4):
        acc = add_rn(acc, pairs[k])
    return acc


def _normalize(z, inv, weight_words):
    """``bf16((z * inv) * weight)`` per element, packed."""
    weights = _values(weight_words)
    return _round([mul_rn(mul_rn(z[i], inv), weights[i]) for i in range(8)])


def _reduce_own_pack(world: int, rank: int, slots: int, recv_base, slot, slot_bytes, offset, own_words,
                     zero, destination) -> None:
    """Two-shot reduction of one own-chunk pack: rank-order sum of every source, stored for the gather.

    A plain function, so the kernel's dynamic branch around it assigns nothing.
    """
    peer_words = ld_sys_v4_gated(
        _peer_addresses(world, rank, slots, recv_base, slot, slot_bytes, offset), zero)
    mine = _round(_sum_in_rank_order(world, rank, own_words, peer_words))
    st_global_v4(destination, mine[0], mine[1], mine[2], mine[3])


def _record_timeout(ctrl_base, poison_addr, peer, lane, seq) -> None:
    """The missing peer and lane, then the sequence, then the poison word."""
    st_relaxed_sys_u32(ctrl_base + Int64(CTRL_ERROR_PEER), Uint32(peer))
    st_relaxed_sys_u32(ctrl_base + Int64(CTRL_ERROR_LANE), Uint32(lane))
    fence_sc_sys()
    st_relaxed_sys_u32(ctrl_base + Int64(CTRL_ERROR_SEQ), seq)
    st_release_gpu_u32(poison_addr, Uint32(1))


class FusedAddRmsNormLaunch:
    """One compiled specialization: algorithm, world size, rank, row length and arena layout."""

    def __init__(self, algorithm: str, world_size: int, rank: int, row_packs: int, lanes: int,
                 slots: int, flag_stride: int) -> None:
        if algorithm not in geo.ALGORITHMS:
            raise ValueError(f"unknown algorithm {algorithm!r}")
        geo.check_launch_geometry(1, row_packs, 1, lanes)
        if not 2 <= world_size <= 16 or not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} of world size {world_size} is outside 2..16 ranks")
        if world_size * lanes > row_packs:
            raise ValueError("the flag waiters need world_size * lanes <= threads per CTA")
        self._algorithm = algorithm
        self._world = int(world_size)
        self._rank = int(rank)
        self._row_packs = int(row_packs)
        self._threads = int(row_packs)
        self._warps = self._threads // geo.WARP
        self._hidden = self._row_packs * geo.BF16_PER_PACK
        self._lanes = int(lanes)
        self._slots = int(slots)
        self._flag_stride = int(flag_stride)

    @cute.jit
    def __call__(
        self,
        input_addr: Int64,
        residual_addr: Int64,
        weight_addr: Int64,
        output_addr: Int64,
        rows: Int32,
        nbytes: Int32,
        recv_base: Int64,
        flag_base: Int64,
        send_base: Int64,
        ctrl_base: Int64,
        slot_bytes: Int64,
        epoch_addr: Int64,
        poison_addr: Int64,
        counter_addr: Int64,
        spin_limit: Uint32,
        eps: Float32,
        stream: cuda.CUstream,
    ) -> None:
        """Host entry: one CTA per row, one thread per pack, at most one CTA per multiprocessor."""
        self.kernel(
            input_addr, residual_addr, weight_addr, output_addr, rows, nbytes, recv_base, flag_base,
            send_base, ctrl_base, slot_bytes, epoch_addr, poison_addr, counter_addr, spin_limit, eps,
        ).launch(
            grid=(rows, 1, 1),
            block=[self._threads, 1, 1],
            cluster=(1, 1, 1),
            max_number_threads=(self._threads, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        input_addr: Int64,
        residual_addr: Int64,
        weight_addr: Int64,
        output_addr: Int64,
        rows: Int32,
        nbytes: Int32,
        recv_base: Int64,
        flag_base: Int64,
        send_base: Int64,
        ctrl_base: Int64,
        slot_bytes: Int64,
        epoch_addr: Int64,
        poison_addr: Int64,
        counter_addr: Int64,
        spin_limit: Uint32,
        eps: Float32,
    ) -> None:
        """Device kernel; the module docstring lists the phases."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        thread = Int32(tidx)
        row = Int32(bidx)
        warp = thread // Int32(geo.WARP)
        lane_id = thread - warp * Int32(geo.WARP)

        smem = cutlass.utils.SmemAllocator()
        entry = smem.allocate_tensor(
            element_type=cutlass.Uint32, layout=cute.make_layout((4,)), byte_alignment=16
        )
        warp_sums = smem.allocate_tensor(
            element_type=cutlass.Float32, layout=cute.make_layout((32,)), byte_alignment=16
        )
        # One read of the entry state per CTA keeps every branch around a
        # barrier uniform within the CTA, even if another CTA poisons the
        # session meanwhile. The epoch cannot advance before every CTA of this
        # launch arrived at the tail counter, which follows this read.
        if thread == Int32(0):
            entry[0] = ld_relaxed_gpu_u32(epoch_addr)
            entry[1] = ld_relaxed_gpu_u32(poison_addr)
        cute.arch.sync_threads()
        seq = entry[0] + Uint32(1)
        poisoned = entry[1]

        slot = Int64(seq & Uint32(1))
        send_slot = send_base + slot * slot_bytes
        packs = rows * Int32(self._row_packs)
        pack = row * Int32(self._row_packs) + thread
        offset = Int64(pack) * Int64(PACK_BYTES)
        # Zero at run time (sizes stay below 2**31) but opaque to ptxas: the gate of ld_sys_v4_gated.
        zero = Uint32(packs) >> Uint32(31)
        bound = Uint32(gdim) - Uint32(1)
        stage_counter = counter_addr
        mid_counter = counter_addr + Int64(4)
        tail_counter = counter_addr + Int64(8)

        if poisoned == Uint32(0):
            # Device-memory loads that do not depend on the network, issued first.
            residual = ld_global_v4(residual_addr + offset)
            weight = ld_global_v4(weight_addr + Int64(thread) * Int64(PACK_BYTES))
            own = ld_global_v4(input_addr + offset)

            if cutlass.const_expr(self._algorithm == geo.ONESHOT):
                # 1. stage the whole row and ring the doorbell
                st_global_v4(send_slot + offset, own[0], own[1], own[2], own[3])
                cute.arch.sync_threads()
                if thread == Int32(0):
                    fence_sc_sys()
                    prior = atom_inc_relaxed_gpu_u32(stage_counter, bound)
                    if prior == bound:
                        st_relaxed_sys_u32(ctrl_base + Int64(CTRL_NBYTES), Uint32(nbytes))
                        st_relaxed_sys_u32(ctrl_base + Int64(CTRL_SLOT_WORDS) + slot * Int64(4), Uint32(nbytes))
                        fence_sc_sys()
                        st_relaxed_sys_u32(ctrl_base, seq)

                # 2. wait for the peer stripes that hold packs of this row
                if thread < Int32(self._world * self._lanes):
                    peer = thread // Int32(self._lanes)
                    lane = thread - peer * Int32(self._lanes)
                    if geo.oneshot_wait(row, self._row_packs, packs, peer, lane, self._rank, self._lanes,
                                        minimum=min_s32):
                        flag = flag_base + Int64(self._flag_stride) * geo.flag_index(
                            geo.SCATTER, Int64(peer), slot, Int64(lane), self._world, self._slots, self._lanes)
                        limit_us = ld_relaxed_sys_u32(ctrl_base + Int64(CTRL_WAIT_LIMIT))
                        if spin_until_eq_timed_sys(flag, seq, spin_limit, limit_us) != Uint32(0):
                            _record_timeout(ctrl_base, poison_addr, peer, lane, seq)
                cute.arch.sync_threads()
                failed = ld_relaxed_gpu_u32(poison_addr)

                # 3. reduce in rank order (every peer pack in one batch)
                peer_words = ld_sys_v4_gated(
                    _peer_addresses(self._world, self._rank, self._slots, recv_base, slot, slot_bytes, offset),
                    zero)
                reduced = _round(_sum_in_rank_order(self._world, self._rank, own, peer_words))
            else:
                own_lo, own_hi = geo.chunk_bounds(packs, self._world, Int32(self._rank))
                # 1. stage every pack outside the own chunk and ring the scatter doorbell
                if (pack < own_lo) | (pack >= own_hi):
                    st_global_v4(send_slot + offset, own[0], own[1], own[2], own[3])
                cute.arch.sync_threads()
                if thread == Int32(0):
                    fence_sc_sys()
                    prior = atom_inc_relaxed_gpu_u32(stage_counter, bound)
                    if prior == bound:
                        st_relaxed_sys_u32(ctrl_base + Int64(CTRL_NBYTES), Uint32(nbytes))
                        st_relaxed_sys_u32(ctrl_base + Int64(CTRL_SLOT_WORDS) + slot * Int64(4),
                                           Uint32(nbytes) | Uint32(TWO_SHOT_OP))
                        fence_sc_sys()
                        st_relaxed_sys_u32(ctrl_base, seq)

                # 2. scatter phase: wait for the stripes of the own chunk that this row needs
                if thread < Int32(self._world * self._lanes):
                    peer = thread // Int32(self._lanes)
                    lane = thread - peer * Int32(self._lanes)
                    if geo.twoshot_scatter_wait(row, self._row_packs, packs, peer, lane, self._rank,
                                                self._world, self._lanes, minimum=min_s32):
                        flag = flag_base + Int64(self._flag_stride) * geo.flag_index(
                            geo.SCATTER, Int64(peer), slot, Int64(lane), self._world, self._slots, self._lanes)
                        limit_us = ld_relaxed_sys_u32(ctrl_base + Int64(CTRL_WAIT_LIMIT))
                        if spin_until_eq_timed_sys(flag, seq, spin_limit, limit_us) != Uint32(0):
                            _record_timeout(ctrl_base, poison_addr, peer, lane, seq)
                cute.arch.sync_threads()
                scattered = ld_relaxed_gpu_u32(poison_addr)
                if (scattered == Uint32(0)) & (pack >= own_lo) & (pack < own_hi):
                    _reduce_own_pack(self._world, self._rank, self._slots, recv_base, slot, slot_bytes, offset,
                                     own, zero, send_slot + offset)
                cute.arch.sync_threads()
                # the last CTA to finish its reduction rings the gather doorbell,
                # unless a scatter wait timed out (its error store precedes its arrival)
                if thread == Int32(0):
                    fence_sc_sys()
                    prior = atom_inc_relaxed_gpu_u32(mid_counter, bound)
                    if prior == bound:
                        fence_sc_sys()
                        if ld_relaxed_sys_u32(ctrl_base + Int64(CTRL_ERROR_SEQ)) == Uint32(0):
                            st_relaxed_sys_u32(ctrl_base + Int64(CTRL_PHASE1), seq)

                # 3. gather phase: wait for the other owners' stripes that hold packs of this row
                if thread < Int32(self._world * self._lanes):
                    owner = thread // Int32(self._lanes)
                    lane = thread - owner * Int32(self._lanes)
                    if geo.twoshot_gather_wait(row, self._row_packs, packs, owner, lane, self._rank,
                                               self._world, self._lanes, minimum=min_s32):
                        if ld_relaxed_gpu_u32(poison_addr) == Uint32(0):
                            flag = flag_base + Int64(self._flag_stride) * geo.flag_index(
                                geo.GATHER, Int64(owner), slot, Int64(lane), self._world, self._slots,
                                self._lanes)
                            limit_us = ld_relaxed_sys_u32(ctrl_base + Int64(CTRL_WAIT_LIMIT))
                            if spin_until_eq_timed_sys(flag, seq, spin_limit, limit_us) != Uint32(0):
                                _record_timeout(ctrl_base, poison_addr, owner, lane, seq)
                cute.arch.sync_threads()
                failed = ld_relaxed_gpu_u32(poison_addr)
                # the unpack: every pack of the row is a reduced chunk pack
                pack_owner = pack // (packs // Int32(self._world))
                source_addr = recv_base + (Int64(pack_owner) * Int64(self._slots) + slot) * slot_bytes + offset
                if pack_owner == Int32(self._rank):
                    source_addr = send_slot + offset
                reduced = ld_sys_v4(source_addr)

            # 4. residual add, sum of squares, norm (both algorithms)
            new_residual, z = _residual_add(reduced, residual)
            if failed == Uint32(0):
                st_global_v4(residual_addr + offset, new_residual[0], new_residual[1], new_residual[2],
                             new_residual[3])
            partial = _pack_sum_of_squares(z)
            for shift in cutlass.range_constexpr(len(SHUFFLE_OFFSETS)):
                partial = add_rn(partial, shfl_down_f32(partial, Uint32(SHUFFLE_OFFSETS[shift])))
            if lane_id == Int32(0):
                warp_sums[warp] = partial
            cute.arch.sync_threads()
            total = warp_sums[0]
            for w in cutlass.range_constexpr(1, self._warps):
                total = add_rn(total, warp_sums[w])
            inv = rsqrt_approx(add_rn(div_rn(total, Float32(float(self._hidden))), eps))
            normed = _normalize(z, inv, weight)
            if failed == Uint32(0):
                st_global_v4(output_addr + offset, normed[0], normed[1], normed[2], normed[3])

            # 5. the last CTA to finish publishes the next epoch
            fence_sc_gpu()
            cute.arch.sync_threads()
            if thread == Int32(0):
                prior = atom_inc_relaxed_gpu_u32(tail_counter, bound)
                if prior == bound:
                    fence_sc_gpu()
                    # Every CTA's timeout store precedes its tail arrival, so the
                    # error word is final here; a failed sequence keeps the epoch.
                    if ld_relaxed_sys_u32(ctrl_base + Int64(CTRL_ERROR_SEQ)) == Uint32(0):
                        st_release_gpu_u32(epoch_addr, seq)


# --------------------------------------------------------------------------- compilation

_LAUNCHERS: dict[tuple, Callable[..., None]] = {}
_LOCK = threading.Lock()


def launcher_key(algorithm: str, world_size: int, rank: int, row_packs: int, lanes: int, slots: int,
                 flag_stride: int, device_index: int) -> tuple:
    """Process-local identity of one compiled specialization."""
    return ("fused-add-rms-norm", algorithm, int(world_size), int(rank), int(row_packs), int(lanes),
            int(slots), int(flag_stride), int(device_index))


def is_prepared(*key) -> bool:
    """Whether ``get_launcher`` already compiled this specialization in this process."""
    with _LOCK:
        return launcher_key(*key) in _LAUNCHERS


def get_launcher(algorithm: str, world_size: int, rank: int, row_packs: int, lanes: int, slots: int,
                 flag_stride: int, device_index: int) -> Callable[..., None]:
    """Compile once per process and return ``run(...)`` taking addresses and scalars.

    Launch arguments: input, residual, weight and output addresses, rows,
    bytes, receive base, flag base, send base, control base, slot bytes, epoch
    address, poison address, counter address (three words of the caller's
    device buffer), spin limit, epsilon, and optionally a driver stream (the
    caller's current torch stream by default).
    """
    key = launcher_key(algorithm, world_size, rank, row_packs, lanes, slots, flag_stride, device_index)
    with _LOCK:
        cached = _LAUNCHERS.get(key)
    if cached is not None:
        return cached
    launch = FusedAddRmsNormLaunch(algorithm, world_size, rank, row_packs, lanes, slots, flag_stride)
    compiled = compile_launcher(
        launch,
        16, 16, 16, 16,  # input, residual, weight, output addresses
        1, 16,  # rows, nbytes
        16, 16, 16, 16, 4096,  # recv, flag, send, ctrl bases, slot bytes
        16, 16, 16,  # epoch, poison, counter addresses
        1,  # spin limit
        1e-5,  # eps
        current_cuda_stream(),
        name=f"sircl fused add + RMSNorm ({algorithm})", cache_key=key,
    )

    def run(input_addr: int, residual_addr: int, weight_addr: int, output_addr: int, rows: int,
            nbytes: int, recv_base: int, flag_base: int, send_base: int, ctrl_base: int,
            slot_bytes: int, epoch_addr: int, poison_addr: int, counter_addr: int, spin_limit: int,
            eps: float, stream=None) -> None:
        compiled(int(input_addr), int(residual_addr), int(weight_addr), int(output_addr), int(rows),
                 int(nbytes), int(recv_base), int(flag_base), int(send_base), int(ctrl_base),
                 int(slot_bytes), int(epoch_addr), int(poison_addr), int(counter_addr),
                 int(spin_limit), float(eps), current_cuda_stream() if stream is None else stream)

    with _LOCK:
        _LAUNCHERS[key] = run
    return run


__all__ = [
    "COUNTER_WORDS",
    "FusedAddRmsNormLaunch",
    "PACK_BYTES",
    "TWO_SHOT_OP",
    "get_launcher",
    "is_prepared",
    "launcher_key",
]
