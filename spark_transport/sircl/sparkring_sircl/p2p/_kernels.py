"""CuTe DSL kernels of SIRCL's point-to-point channels: one launch per message.

A message of ``packs`` 16-byte packs travels as ``items`` items of at most one
slot (``slot_bytes``) on its channel; item ``i`` is item ``first + i`` of the
channel, in slot ``(first + i) % slots`` with tag ``first + i + 1``
(:mod:`sparkring_sircl.p2p.protocol`). Block ``b`` of a launch takes items
``b``, ``b + B``, ... (``B`` blocks).

- :class:`P2PSend`: per item, the block's last thread waits until the
  channel's sent word reaches ``tag - slots`` (the slot's previous item left
  the slot), the block copies the item from the source tensor into its send
  slot, and thread 0 writes the item's header into the desc line and then
  the tag into the ready line, after a system-scope fence; the progress
  thread posts it.
- :class:`P2PRecv`: per item, threads ``0 .. lanes - 1`` wait for their lane
  flag of the slot to hold the tag, thread 0 compares the item's header with
  the receive's, the block copies the slot into the output, and thread 0
  writes the tag into the consumed line after a system-scope fence; the
  progress thread then returns the slot to the sender.

Every wait is limited by the context's wait limit (control word
``WAIT_LIMIT_US``, read when the launch starts) on the GPU's nanosecond clock
and watches the poison word. A timeout or a header that differs records the
failure in the control line (peer, lane, kind, expected and received header,
then the tag) and sets the poison word; every later item and launch of the
context returns at once, and the progress thread stops every rank's channels.
"""

# Annotations stay evaluated (no postponed evaluation): the CuTe DSL reads each
# launch parameter's type from its annotation object.

import threading
from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ..oneshot._compile import compile_launcher, current_cuda_stream, make_pointer
from ..oneshot._cute_batch import ld_v4_u32_batch
from ..oneshot._cute_intrinsics import fence_sc_sys, ld_relaxed_sys_u32, min_s32, st_global_v4_u32, st_relaxed_sys_u32
from .protocol import LAST, LINE, Control, ErrorKind, P2PLayout

PACK = 16
POLLS_PER_POISON_CHECK = 64
POLLS_PER_CLOCK_CHECK = 1024
_WAIT_LIMIT = 4 * int(Control.WAIT_LIMIT_US)
_ERROR_TAG = 4 * int(Control.ERROR_TAG)
_ERROR_PEER = 4 * int(Control.ERROR_PEER)
_ERROR_LANE = 4 * int(Control.ERROR_LANE)
_ERROR_KIND = 4 * int(Control.ERROR_KIND)
_ERROR_EXPECTED = 4 * int(Control.ERROR_EXPECTED)
_ERROR_GOT = 4 * int(Control.ERROR_GOT)
_POISON = 4 * int(Control.POISON)
_NO_LANE = 255


def _asm(result_type, operands, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        result_type,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# Both waits compute their result in a register of their own and write the output operand only after
# their last read of an input: an inline-assembly output without an early-clobber marker may share a
# register with an input.


@dsl_user_op
def wait_eq_or_poison(addr: Int64, expected: Uint32, poison: Int64, limit_us: Uint32, *, loc=None,
                      ip=None) -> Uint32:
    """Spin until the word at ``addr`` equals ``expected`` (system-scope acquire loads).

    Returns 0 on a match, 1 after ``limit_us`` microseconds (``%globaltimer``
    checked every 1,024 polls; 0: no limit) and 2 as soon as the word at
    ``poison`` is nonzero (checked every 64 polls).
    """
    return Uint32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(expected).ir_value(loc=loc, ip=ip),
             Int64(poison).ir_value(loc=loc, ip=ip), Uint32(limit_us).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred pending, skip, timed, bad;
                .reg .b32 seen, polls, low, result, flag;
                .reg .b64 start, now, elapsed, budget;
                mov.u32 polls, 0;
                mov.u32 result, 0;
                setp.ne.u32 timed, $4, 0;
                mul.wide.u32 budget, $4, 1000;
                mov.u64 start, %globaltimer;
            p2p_eq_wait:
                ld.acquire.sys.global.u32 seen, [$1];
                setp.ne.u32 pending, seen, $2;
                @!pending bra p2p_eq_done;
                add.u32 polls, polls, 1;
                and.b32 low, polls, 63;
                setp.ne.u32 skip, low, 0;
                @skip bra p2p_eq_wait;
                ld.relaxed.sys.global.u32 flag, [$3];
                setp.ne.u32 bad, flag, 0;
                @bad bra p2p_eq_poisoned;
                and.b32 low, polls, 1023;
                setp.ne.u32 skip, low, 0;
                @skip bra p2p_eq_wait;
                @!timed bra p2p_eq_wait;
                mov.u64 now, %globaltimer;
                sub.u64 elapsed, now, start;
                setp.lt.u64 skip, elapsed, budget;
                @skip bra p2p_eq_wait;
                mov.u32 result, 1;
                bra p2p_eq_done;
            p2p_eq_poisoned:
                mov.u32 result, 2;
            p2p_eq_done:
                mov.u32 $0, result;
            }
            """,
            "=r,l,r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def wait_ge_or_poison(addr: Int64, target: Uint32, poison: Int64, limit_us: Uint32, *, loc=None,
                      ip=None) -> Uint32:
    """Spin until the counter at ``addr`` reaches ``target`` (``(int32)(value - target) >= 0``, so
    counters may wrap), with the results and limits of :func:`wait_eq_or_poison`."""
    return Uint32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(target).ir_value(loc=loc, ip=ip),
             Int64(poison).ir_value(loc=loc, ip=ip), Uint32(limit_us).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred pending, skip, timed, bad;
                .reg .b32 seen, polls, low, result, flag, diff;
                .reg .b64 start, now, elapsed, budget;
                mov.u32 polls, 0;
                mov.u32 result, 0;
                setp.ne.u32 timed, $4, 0;
                mul.wide.u32 budget, $4, 1000;
                mov.u64 start, %globaltimer;
            p2p_ge_wait:
                ld.acquire.sys.global.u32 seen, [$1];
                sub.u32 diff, seen, $2;
                setp.lt.s32 pending, diff, 0;
                @!pending bra p2p_ge_done;
                add.u32 polls, polls, 1;
                and.b32 low, polls, 63;
                setp.ne.u32 skip, low, 0;
                @skip bra p2p_ge_wait;
                ld.relaxed.sys.global.u32 flag, [$3];
                setp.ne.u32 bad, flag, 0;
                @bad bra p2p_ge_poisoned;
                and.b32 low, polls, 1023;
                setp.ne.u32 skip, low, 0;
                @skip bra p2p_ge_wait;
                @!timed bra p2p_ge_wait;
                mov.u64 now, %globaltimer;
                sub.u64 elapsed, now, start;
                setp.lt.u64 skip, elapsed, budget;
                @skip bra p2p_ge_wait;
                mov.u32 result, 1;
                bra p2p_ge_done;
            p2p_ge_poisoned:
                mov.u32 result, 2;
            p2p_ge_done:
                mov.u32 $0, result;
            }
            """,
            "=r,l,r,l,r",
            loc=loc,
            ip=ip,
        )
    )


class _P2PKernel:
    """Geometry and failure records shared by the send and receive kernels."""

    def _setup(self, threads: int, lanes: int, slots: int, slot_bytes: int, blocks: int, unroll: int) -> None:
        if int(threads) < 64 or int(threads) % 32 or int(threads) > 1024:
            raise ValueError(f"point-to-point kernels run a multiple of 32 threads from 64 to 1024, got {threads}")
        if not 1 <= int(blocks) <= 64 or not 1 <= int(unroll) <= 8:
            raise ValueError(f"point-to-point kernels run 1 to 64 blocks moving 1 to 8 packs per thread per pass; got "
                             f"{blocks} blocks and {unroll}")
        # Two lanes of a 32-entry ring: the layout validates lanes, slots and slot bytes.
        self._layout = P2PLayout(2, int(lanes), int(slots), int(slot_bytes))
        self._threads = int(threads)
        self._lanes = int(lanes)
        self._slots = int(slots)
        self._slot_bytes = int(slot_bytes)
        self._piece_packs = int(slot_bytes) // PACK
        self._blocks = int(blocks)
        self._unroll = int(unroll)

    @cute.jit
    def _fail(self, ctrl_base: Int64, peer: Int32, lane: Int32, kind: cutlass.Constexpr[int], tag: Uint32,
              expected: Uint32, got: Uint32) -> None:
        st_relaxed_sys_u32(ctrl_base + Int64(_ERROR_PEER), Uint32(peer))
        st_relaxed_sys_u32(ctrl_base + Int64(_ERROR_LANE), Uint32(lane))
        st_relaxed_sys_u32(ctrl_base + Int64(_ERROR_KIND), Uint32(kind))
        st_relaxed_sys_u32(ctrl_base + Int64(_ERROR_EXPECTED), expected)
        st_relaxed_sys_u32(ctrl_base + Int64(_ERROR_GOT), got)
        fence_sc_sys()
        st_relaxed_sys_u32(ctrl_base + Int64(_ERROR_TAG), tag)
        fence_sc_sys()
        st_relaxed_sys_u32(ctrl_base + Int64(_POISON), Uint32(1))

    @cute.jit
    def _copy(self, src: Int64, system: cutlass.Constexpr[bool], dst: Int64, packs: Int32, zero: Uint32) -> None:
        """Copy ``packs`` packs from ``src`` to ``dst``, ``unroll`` packs per thread per pass with the loads of
        a pass issued together; ``system`` reads pinned host memory at system scope."""
        tidx, _, _ = cute.arch.thread_idx()
        thread = Int32(tidx)
        last = packs - Int32(1)
        base = Int32(0)
        while base < packs:
            addrs = []
            for u in cutlass.range_constexpr(self._unroll):
                index = min_s32(base + Int32(u * self._threads) + thread, last)
                addrs.append(src + Int64(index) * Int64(PACK))
            words = ld_v4_u32_batch(addrs, [system] * self._unroll, zero)
            for u in cutlass.range_constexpr(self._unroll):
                index = base + Int32(u * self._threads) + thread
                if index < packs:
                    w = words[u]
                    st_global_v4_u32(dst + Int64(index) * Int64(PACK), w[0], w[1], w[2], w[3])
            base = base + Int32(self._unroll * self._threads)

    @cute.jit
    def _header(self, item: Int32, items: Int32, count: Int32, tail: Int32) -> Uint32:
        header = Uint32(count * Int32(PACK))
        if item == items - Int32(1):
            header = header | Uint32(LAST) | Uint32(tail)
        return header


class P2PSend(_P2PKernel):
    """The send kernel of one geometry: threads, lanes, slots, slot bytes, blocks, unroll."""

    def __init__(self, threads: int, lanes: int, slots: int, slot_bytes: int, blocks: int, unroll: int) -> None:
        self._setup(threads, lanes, slots, slot_bytes, blocks, unroll)

    @cute.jit
    def __call__(self, data: cute.Pointer, packs: Int32, tail: Int32, first: Uint32, items: Int32, block_base: Int64,
                 ctrl_base: Int64, peer: Int32, stream: cuda.CUstream) -> None:
        self.kernel(data, packs, tail, first, items, block_base, ctrl_base, peer).launch(
            grid=(self._blocks, 1, 1), block=[self._threads, 1, 1], cluster=(1, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, data: cute.Pointer, packs: Int32, tail: Int32, first: Uint32, items: Int32, block_base: Int64,
               ctrl_base: Int64, peer: Int32) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        thread = Int32(tidx)
        data_base = Int64(data.toint())
        poison = ctrl_base + Int64(_POISON)
        zero = Uint32(items) >> Uint32(31)
        layout = self._layout
        # Thread 0's view of the poison word for the whole block, so every branch around a barrier is uniform.
        smem = cutlass.utils.SmemAllocator()
        state = smem.allocate_tensor(element_type=cutlass.Uint32, layout=cute.make_layout((4,)), byte_alignment=16)
        limit_us = ld_relaxed_sys_u32(ctrl_base + Int64(_WAIT_LIMIT))
        item = Int32(bidx)
        while item < items:
            g = first + Uint32(item)
            m = g % Uint32(self._slots)
            tag = g + Uint32(1)
            if thread == Int32(self._threads - 1):
                sent = block_base + Int64(layout.sent_off)
                if wait_ge_or_poison(sent, tag - Uint32(self._slots), poison, limit_us) == Uint32(1):
                    self._fail(ctrl_base, peer, Int32(_NO_LANE), int(ErrorKind.SLOT), tag, Uint32(0), Uint32(0))
            cute.arch.sync_threads()
            if thread == Int32(0):
                state[0] = ld_relaxed_sys_u32(poison)
            cute.arch.sync_threads()
            healthy = state[0] == Uint32(0)
            start = item * Int32(self._piece_packs)
            count = packs - start
            if Int32(self._piece_packs) < count:
                count = Int32(self._piece_packs)
            if healthy:
                slot = block_base + Int64(layout.send_off) + Int64(m) * Int64(self._slot_bytes)
                self._copy(data_base + Int64(start) * Int64(PACK), False, slot, count, zero)
            cute.arch.sync_threads()
            if healthy:
                if thread == Int32(0):
                    fence_sc_sys()
                    st_relaxed_sys_u32(block_base + Int64(layout.desc_off) + Int64(m) * Int64(4),
                                       self._header(item, items, count, tail))
                    fence_sc_sys()
                    st_relaxed_sys_u32(block_base + Int64(layout.ready_off) + Int64(m) * Int64(4), tag)
                item = item + Int32(self._blocks)
            else:
                item = items


class P2PRecv(_P2PKernel):
    """The receive kernel of one geometry: threads, lanes, slots, slot bytes, blocks, unroll."""

    def __init__(self, threads: int, lanes: int, slots: int, slot_bytes: int, blocks: int, unroll: int) -> None:
        self._setup(threads, lanes, slots, slot_bytes, blocks, unroll)

    @cute.jit
    def __call__(self, data: cute.Pointer, packs: Int32, tail: Int32, first: Uint32, items: Int32, block_base: Int64,
                 ctrl_base: Int64, peer: Int32, stream: cuda.CUstream) -> None:
        self.kernel(data, packs, tail, first, items, block_base, ctrl_base, peer).launch(
            grid=(self._blocks, 1, 1), block=[self._threads, 1, 1], cluster=(1, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, data: cute.Pointer, packs: Int32, tail: Int32, first: Uint32, items: Int32, block_base: Int64,
               ctrl_base: Int64, peer: Int32) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        thread = Int32(tidx)
        data_base = Int64(data.toint())
        poison = ctrl_base + Int64(_POISON)
        zero = Uint32(items) >> Uint32(31)
        layout = self._layout
        smem = cutlass.utils.SmemAllocator()
        state = smem.allocate_tensor(element_type=cutlass.Uint32, layout=cute.make_layout((4,)), byte_alignment=16)
        limit_us = ld_relaxed_sys_u32(ctrl_base + Int64(_WAIT_LIMIT))
        item = Int32(bidx)
        while item < items:
            g = first + Uint32(item)
            m = g % Uint32(self._slots)
            tag = g + Uint32(1)
            line = block_base + Int64(layout.flag_off) + Int64(m) * Int64(self._lanes * LINE)
            if thread < Int32(self._lanes):
                if wait_eq_or_poison(line + Int64(thread) * Int64(LINE), tag, poison, limit_us) == Uint32(1):
                    self._fail(ctrl_base, peer, thread, int(ErrorKind.FLAG), tag, Uint32(0), Uint32(0))
            cute.arch.sync_threads()
            start = item * Int32(self._piece_packs)
            count = packs - start
            if Int32(self._piece_packs) < count:
                count = Int32(self._piece_packs)
            if thread == Int32(0):
                # Thread 0 waited for lane 0's flag, which follows the header on the same queue pair.
                if ld_relaxed_sys_u32(poison) == Uint32(0):
                    expected = self._header(item, items, count, tail)
                    got = ld_relaxed_sys_u32(line + Int64(4))
                    if got != expected:
                        self._fail(ctrl_base, peer, Int32(_NO_LANE), int(ErrorKind.SIZE), tag, expected, got)
                state[0] = ld_relaxed_sys_u32(poison)
            cute.arch.sync_threads()
            healthy = state[0] == Uint32(0)
            if healthy:
                slot = block_base + Int64(layout.recv_off) + Int64(m) * Int64(self._slot_bytes)
                self._copy(slot, True, data_base + Int64(start) * Int64(PACK), count, zero)
            cute.arch.sync_threads()
            if healthy:
                if thread == Int32(0):
                    fence_sc_sys()
                    st_relaxed_sys_u32(block_base + Int64(layout.consumed_off) + Int64(m) * Int64(4), tag)
                item = item + Int32(self._blocks)
            else:
                item = items


_LAUNCHERS: dict[tuple, Callable[..., None]] = {}
_GETTER_LOCK = threading.Lock()


def launcher_key(kind: str, threads: int, lanes: int, slots: int, slot_bytes: int, blocks: int, unroll: int,
                 device_index: int) -> tuple:
    return ("p2p", str(kind), int(threads), int(lanes), int(slots), int(slot_bytes), int(blocks), int(unroll),
            int(device_index))


def is_prepared(kind: str, threads: int, lanes: int, slots: int, slot_bytes: int, blocks: int, unroll: int,
                device_index: int) -> bool:
    return launcher_key(kind, threads, lanes, slots, slot_bytes, blocks, unroll, device_index) in _LAUNCHERS


def get_launcher(kind: str, threads: int, lanes: int, slots: int, slot_bytes: int, blocks: int, unroll: int,
                 device_index: int) -> Callable[..., None]:
    """The compiled ``send`` or ``recv`` launcher of one geometry (compiled once per process).

    Launch arguments: data address (16-byte aligned), packs, ``bytes % 16`` of the message, the channel's
    first item, items, the peer's block address, the control line address, the peer.
    """
    if kind not in ("send", "recv"):
        raise ValueError(f"a point-to-point launcher is send or recv, got {kind!r}")
    key = launcher_key(kind, threads, lanes, slots, slot_bytes, blocks, unroll, device_index)
    with _GETTER_LOCK:
        cached = _LAUNCHERS.get(key)
        if cached is not None:
            return cached
        launch = (P2PSend if kind == "send" else P2PRecv)(threads, lanes, slots, slot_bytes, blocks, unroll)
        compiled = compile_launcher(launch, make_pointer(16), 1, 0, 1, 1, 16, 16, 0, current_cuda_stream(),
                                    name=f"sircl point-to-point {kind}", cache_key=key)

        launch_lock = threading.Lock()

        def run(data_address: int, packs: int, tail: int, first: int, items: int, block_base: int, ctrl_base: int,
                peer: int) -> None:
            # One executor serves every context of the process, and it keeps one CUDA result word for all
            # its calls, so calls from several threads are serialized.
            with launch_lock:
                compiled(make_pointer(data_address), int(packs), int(tail), int(first) & 0xFFFFFFFF, int(items),
                         int(block_base), int(ctrl_base), int(peer), current_cuda_stream())

        _LAUNCHERS[key] = run
        return run


__all__ = ["P2PRecv", "P2PSend", "get_launcher", "is_prepared", "launcher_key", "wait_eq_or_poison",
           "wait_ge_or_poison"]
