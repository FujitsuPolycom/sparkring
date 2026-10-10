"""Flag waits with a time limit, and the poll-rate probe, for the ring-session kernels.

A ring session's kernels wait for peers' flags in pinned host memory. How
long a wait may last depends on what the group is doing: while a peer
compiles kernels at startup it can lag for minutes, while serving a lag of
more than seconds means the peer is gone. A poll budget cannot express
either, because the time one poll takes depends on the GPU and the memory
path, so the session writes a wait limit in microseconds into its command
ring (``protocol.Ctrl.WAIT_LIMIT_US``) and the kernels read it when they
start. The host can change the limit at any time, also between replays of a
captured CUDA graph.

:func:`spin_until_eq_timed_sys` polls with system-scope acquire loads and
reads the GPU's nanosecond global timer every 1,024 polls; a limit of 0
falls back to a budget of ``poll_limit`` polls. :func:`spin_until_ge_timed_sys`
waits the same way for a counter to reach a value. :func:`probe_poll_rate`
times a fixed number of polls of one word, so a session can report its
polls per second (``stats()["poll_rate_per_s"]``). :class:`ClockProbe`
answers the host's pings with the GPU's ``%globaltimer``, so a session can map
kernel trace times to the host clock.
"""

import threading
from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ._compile import compile_launcher, current_cuda_stream

# Annotations stay evaluated (no postponed evaluation): the CuTe DSL reads each
# launch parameter's type from its annotation object.
#
# Every wait computes its result in a register of its own and writes the output
# operand only after its last read of an input: an inline-assembly output
# without an early-clobber marker may share a register with an input, so an
# output written first could overwrite the target, the poll budget or the
# time limit of the wait.

POLLS_PER_CLOCK_CHECK = 1024


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


@dsl_user_op
def spin_until_eq_timed_sys(
    addr: Int64, expected: Uint32, poll_limit: Uint32, limit_us: Uint32, *, loc=None, ip=None
) -> Uint32:
    """Spin until the word at ``addr`` equals ``expected`` (system-scope acquire loads).

    Returns 0 on a match and 1 after ``limit_us`` microseconds (checked every
    1,024 polls on ``%globaltimer``), or, when ``limit_us`` is 0, after
    ``max(1, poll_limit)`` polls.
    """
    return Uint32(
        _asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Uint32(expected).ir_value(loc=loc, ip=ip),
                Uint32(poll_limit).ir_value(loc=loc, ip=ip),
                Uint32(limit_us).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .pred pending, expired, timed, skip;
                .reg .b32 seen, polls, low, result;
                .reg .b64 start, now, elapsed, budget;
                mov.u32 polls, 0;
                mov.u32 result, 0;
                setp.ne.u32 timed, $4, 0;
                mul.wide.u32 budget, $4, 1000;
                mov.u64 start, %globaltimer;
            timed_wait:
                ld.acquire.sys.global.u32 seen, [$1];
                setp.ne.u32 pending, seen, $2;
                @!pending bra timed_done;
                add.u32 polls, polls, 1;
                @timed bra timed_clock;
                setp.ge.u32 expired, polls, $3;
                @!expired bra timed_wait;
                bra timed_expired;
            timed_clock:
                and.b32 low, polls, 1023;
                setp.ne.u32 skip, low, 0;
                @skip bra timed_wait;
                mov.u64 now, %globaltimer;
                sub.u64 elapsed, now, start;
                setp.lt.u64 skip, elapsed, budget;
                @skip bra timed_wait;
            timed_expired:
                mov.u32 result, 1;
            timed_done:
                mov.u32 $0, result;
            }
            """,
            "=r,l,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def spin_until_ge_timed_sys(
    addr: Int64, target: Uint32, poll_limit: Uint32, limit_us: Uint32, *, loc=None, ip=None
) -> Uint32:
    """Spin until the counter at ``addr`` reaches ``target`` (``(int32)(value - target) >= 0``,
    so counters may wrap), with the limits of :func:`spin_until_eq_timed_sys`."""
    return Uint32(
        _asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Uint32(target).ir_value(loc=loc, ip=ip),
                Uint32(poll_limit).ir_value(loc=loc, ip=ip),
                Uint32(limit_us).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .pred pending, expired, timed, skip;
                .reg .b32 seen, polls, low, diff, result;
                .reg .b64 start, now, elapsed, budget;
                mov.u32 polls, 0;
                mov.u32 result, 0;
                setp.ne.u32 timed, $4, 0;
                mul.wide.u32 budget, $4, 1000;
                mov.u64 start, %globaltimer;
            reach_wait:
                ld.acquire.sys.global.u32 seen, [$1];
                sub.u32 diff, seen, $2;
                setp.lt.s32 pending, diff, 0;
                @!pending bra reach_done;
                add.u32 polls, polls, 1;
                @timed bra reach_clock;
                setp.ge.u32 expired, polls, $3;
                @!expired bra reach_wait;
                bra reach_expired;
            reach_clock:
                and.b32 low, polls, 1023;
                setp.ne.u32 skip, low, 0;
                @skip bra reach_wait;
                mov.u64 now, %globaltimer;
                sub.u64 elapsed, now, start;
                setp.lt.u64 skip, elapsed, budget;
                @skip bra reach_wait;
            reach_expired:
                mov.u32 result, 1;
            reach_done:
                mov.u32 $0, result;
            }
            """,
            "=r,l,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def probe_poll_rate(addr: Int64, polls: Uint32, result: Int64, *, loc=None, ip=None) -> None:
    """Load the word at ``addr`` ``polls`` times (system-scope acquire) and store the
    elapsed ``%globaltimer`` nanoseconds as a 64-bit value at ``result``."""
    _asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(polls).ir_value(loc=loc, ip=ip),
            Int64(result).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .pred more;
            .reg .b32 seen, count;
            .reg .b64 start, stop, elapsed;
            mov.u32 count, 0;
            mov.u64 start, %globaltimer;
        probe_loop:
            ld.acquire.sys.global.u32 seen, [$0];
            add.u32 count, count, 1;
            setp.lt.u32 more, count, $1;
            @more bra probe_loop;
            mov.u64 stop, %globaltimer;
            sub.u64 elapsed, stop, start;
            st.global.u64 [$2], elapsed;
        }
        """,
        "l,r,l",
        loc=loc,
        ip=ip,
    )


class PollRateProbe:
    """One thread times ``polls`` loads of one pinned word."""

    @cute.jit
    def __call__(self, addr: Int64, polls: Uint32, result: Int64, stream: cuda.CUstream) -> None:
        self.kernel(addr, polls, result).launch(grid=(1, 1, 1), block=[32, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, addr: Int64, polls: Uint32, result: Int64) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        if Int32(tidx) == Int32(0):
            probe_poll_rate(addr, polls, result)


_PROBES: dict[int, Callable[..., None]] = {}
# Each probe compiles once per device, also when several sessions of one process ask at once: a
# second executor of the same probe would be dropped, and its finalizer unloads its module whenever
# the garbage collector runs, which waits for the device and so for every kernel then running.
_PROBE_LOCK = threading.Lock()


def get_probe(device_index: int) -> Callable[..., None]:
    """The compiled poll-rate probe: ``run(word_address, polls, result_address)``."""
    with _PROBE_LOCK:
        probe = _PROBES.get(int(device_index))
        if probe is None:
            compiled = compile_launcher(PollRateProbe(), 16, 1, 16, current_cuda_stream(),
                                        name="sircl poll-rate probe", cache_key=("poll-rate", int(device_index)))

            def probe(word_address: int, polls: int, result_address: int) -> None:
                compiled(int(word_address), int(polls), int(result_address), current_cuda_stream())

            _PROBES[int(device_index)] = probe
    return probe


@dsl_user_op
def clock_pingpong(flag: Int64, out: Int64, rounds: Uint32, limit_us: Uint32, *, loc=None, ip=None) -> None:
    """Round ``r`` of ``rounds``: wait until the word at ``flag`` holds ``r + 1``, then store the
    GPU's ``%globaltimer`` as a 64-bit value at ``out + 8 r`` (system scope). A round that waits
    longer than ``limit_us`` microseconds ends the probe."""
    _asm(
        None,
        [
            Int64(flag).ir_value(loc=loc, ip=ip),
            Int64(out).ir_value(loc=loc, ip=ip),
            Uint32(rounds).ir_value(loc=loc, ip=ip),
            Uint32(limit_us).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .pred more, pending, skip;
            .reg .b32 r, want, seen, polls, low;
            .reg .b64 start, now, elapsed, budget, addr, stamp;
            mov.u32 r, 0;
            mul.wide.u32 budget, $3, 1000;
        ping_round:
            setp.lt.u32 more, r, $2;
            @!more bra ping_done;
            add.u32 want, r, 1;
            mov.u32 polls, 0;
            mov.u64 start, %globaltimer;
        ping_wait:
            ld.acquire.sys.global.u32 seen, [$0];
            setp.ne.u32 pending, seen, want;
            @!pending bra ping_stamp;
            add.u32 polls, polls, 1;
            and.b32 low, polls, 1023;
            setp.ne.u32 skip, low, 0;
            @skip bra ping_wait;
            mov.u64 now, %globaltimer;
            sub.u64 elapsed, now, start;
            setp.lt.u64 skip, elapsed, budget;
            @skip bra ping_wait;
            bra ping_done;
        ping_stamp:
            mov.u64 stamp, %globaltimer;
            mul.wide.u32 addr, r, 8;
            add.u64 addr, addr, $1;
            st.relaxed.sys.global.u64 [addr], stamp;
            fence.sc.sys;
            add.u32 r, r, 1;
            bra ping_round;
        ping_done:
        }
        """,
        "l,l,r,r",
        loc=loc,
        ip=ip,
    )


class ClockProbe:
    """One thread answers the host's pings with ``%globaltimer`` (:func:`clock_pingpong`)."""

    @cute.jit
    def __call__(self, flag: Int64, out: Int64, rounds: Uint32, limit_us: Uint32, stream: cuda.CUstream) -> None:
        self.kernel(flag, out, rounds, limit_us).launch(grid=(1, 1, 1), block=[32, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, flag: Int64, out: Int64, rounds: Uint32, limit_us: Uint32) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        if Int32(tidx) == Int32(0):
            clock_pingpong(flag, out, rounds, limit_us)


_CLOCK_PROBES: dict[int, Callable[..., None]] = {}


def get_clock_probe(device_index: int) -> Callable[..., None]:
    """The compiled clock probe: ``run(flag_address, out_address, rounds, limit_us)``."""
    with _PROBE_LOCK:
        probe = _CLOCK_PROBES.get(int(device_index))
        if probe is None:
            compiled = compile_launcher(ClockProbe(), 16, 16, 1, 1, current_cuda_stream(),
                                        name="sircl clock probe", cache_key=("clock-probe", int(device_index)))

            def probe(flag_address: int, out_address: int, rounds: int, limit_us: int) -> None:
                compiled(int(flag_address), int(out_address), int(rounds), int(limit_us), current_cuda_stream())

            _CLOCK_PROBES[int(device_index)] = probe
    return probe
