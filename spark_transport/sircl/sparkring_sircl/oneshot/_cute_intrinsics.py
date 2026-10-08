"""PTX intrinsics for the RoCE one-shot all-reduce kernel.

Payload slots and flags live in pinned host memory that the NIC writes by
RDMA and the GPU reads in place, so every access to them is spelled out with
system scope: ``ld.relaxed.sys`` for payload packs, ``ld.acquire.sys`` for the
arrival flag, ``st.relaxed.sys`` plus ``fence.sc.sys`` for the doorbell the
proxy thread polls.  Keeping them as small user ops makes the protocol
explicit and independent of compiler defaults.
"""

from __future__ import annotations

from typing import Tuple

from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


def _asm(result_type, operands, text, constraints, *, side_effects=True, loc=None, ip=None):
    """Emit one inline PTX statement through the CuTe DSL and return its result."""
    return llvm.inline_asm(
        result_type,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_relaxed_gpu_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    """GPU-scope relaxed 32-bit load."""
    return Uint32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.relaxed.gpu.global.u32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_relaxed_sys_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    """System-scope relaxed 32-bit load (sees host and NIC writes)."""
    return Uint32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.relaxed.sys.global.u32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def atomic_add_relaxed_gpu_u32(addr: Int64, value: Uint32, *, loc=None, ip=None) -> Uint32:
    """GPU-scope relaxed atomic add; returns the prior value."""
    return Uint32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
            "atom.relaxed.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_release_gpu_u32(addr: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    """GPU-scope release 32-bit store."""
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
        "st.release.gpu.global.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_relaxed_sys_u32(addr: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    """System-scope relaxed 32-bit store, visible to the host and the NIC."""
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
        "st.relaxed.sys.global.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fence_sc_sys(*, loc=None, ip=None) -> None:
    """Sequentially consistent system-scope fence."""
    _asm(None, [], "fence.sc.sys;", "", loc=loc, ip=ip)


@dsl_user_op
def fence_sc_gpu(*, loc=None, ip=None) -> None:
    """Sequentially consistent GPU-scope fence."""
    _asm(None, [], "fence.sc.gpu;", "", loc=loc, ip=ip)


@dsl_user_op
def spin_until_eq_acquire_sys(
    addr: Int64, expected: Uint32, limit: Uint32, *, loc=None, ip=None
) -> Uint32:
    """Spin until the word at ``addr`` equals ``expected`` (system scope).

    Checks at least once. Returns 0 on success and 1 after ``max(1, limit)``
    polls without a match, so a dead peer or proxy surfaces as an error instead
    of a hung kernel. The output operand is written after the last read of the
    inputs, since it may share a register with one of them.
    """
    return Uint32(
        _asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Uint32(expected).ir_value(loc=loc, ip=ip),
                Uint32(limit).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .pred pending, expired;
                .reg .b32 seen, polls, result;
                mov.u32 polls, 0;
                mov.u32 result, 0;
            roce_wait:
                ld.acquire.sys.global.u32 seen, [$1];
                setp.ne.u32 pending, seen, $2;
                @!pending bra roce_done;
                add.u32 polls, polls, 1;
                setp.ge.u32 expired, polls, $3;
                @!expired bra roce_wait;
                mov.u32 result, 1;
            roce_done:
                mov.u32 $0, result;
            }
            """,
            "=r,l,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def spin_until_eq_or_poison_gpu(
    addr: Int64, expected: Uint32, poison_addr: Int64, *, loc=None, ip=None
) -> Uint32:
    """Spin until the device word at ``addr`` equals ``expected`` (GPU-scope acquire loads) or the
    poison word at ``poison_addr`` is nonzero.

    Returns 0 on a match and 1 once poisoned. It has no poll limit of its own: the word is written by
    another block of the same launch whose own waits are timed and poison the session when they expire.
    The output operand is written after the last read of the inputs, since it may share a register with
    one of them.
    """
    return Uint32(
        _asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Uint32(expected).ir_value(loc=loc, ip=ip),
                Int64(poison_addr).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .pred pending, poisoned;
                .reg .b32 seen, bad, result;
                mov.u32 result, 0;
            arrival_wait:
                ld.acquire.gpu.global.u32 seen, [$1];
                setp.ne.u32 pending, seen, $2;
                @!pending bra arrival_done;
                ld.relaxed.gpu.global.u32 bad, [$3];
                setp.ne.u32 poisoned, bad, 0;
                @!poisoned bra arrival_wait;
                mov.u32 result, 1;
            arrival_done:
                mov.u32 $0, result;
            }
            """,
            "=r,l,r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_relaxed_sys_v4_u32(
    addr: Int64, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """System-scope relaxed load of one 16-byte pack from NIC-written pinned memory."""
    result = _asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.relaxed.sys.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4)
    )


@dsl_user_op
def ld_global_v4_u32(addr: Int64, *, loc=None, ip=None) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Plain global 16-byte load as four 32-bit words."""
    result = _asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4)
    )


@dsl_user_op
def st_global_v4_u32(
    addr: Int64, v0: Uint32, v1: Uint32, v2: Uint32, v3: Uint32, *, loc=None, ip=None
) -> None:
    """Plain global 16-byte store of four 32-bit words."""
    _asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(v0).ir_value(loc=loc, ip=ip),
            Uint32(v1).ir_value(loc=loc, ip=ip),
            Uint32(v2).ir_value(loc=loc, ip=ip),
            Uint32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def globaltimer_u32x2(*, loc=None, ip=None) -> Tuple[Uint32, Uint32]:
    """The GPU's global nanosecond timer (``%globaltimer``) as its low and high 32-bit words."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [],
        "{ .reg .b64 t; mov.u64 t, %globaltimer; mov.b64 {$0, $1}, t; }",
        "=r,=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)))


@dsl_user_op
def st_relaxed_sys_u32x2(addr: Int64, lo: Uint32, hi: Uint32, *, loc=None, ip=None) -> None:
    """System-scope relaxed 64-bit store of ``hi:lo`` (one access, so the host never sees half of it)."""
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(lo).ir_value(loc=loc, ip=ip),
         Uint32(hi).ir_value(loc=loc, ip=ip)],
        "{ .reg .b64 t; mov.b64 t, {$1, $2}; st.relaxed.sys.global.u64 [$0], t; }",
        "l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def min_s32(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """Signed 32-bit minimum."""
    return Int32(
        _asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "min.s32 $0, $1, $2;",
            "=r,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def u32_as_f32(value: Uint32, *, loc=None, ip=None) -> Float32:
    """Reinterpret the bits of a 32-bit word as float32."""
    return Float32(
        _asm(
            T.f32(),
            [Uint32(value).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def f32_as_u32(value: Float32, *, loc=None, ip=None) -> Uint32:
    """Reinterpret the bits of a float32 as a 32-bit word."""
    return Uint32(
        _asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def unpack_bf16x2(value: Uint32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """Split a packed bf16 pair into two float32 values, low half first."""
    result = _asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(value).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 lo, hi;
            mov.b32 {lo, hi}, $2;
            cvt.f32.bf16 $0, lo;
            cvt.f32.bf16 $1, hi;
        }
        """,
        "=f,=f,r",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def unpack_f16x2(value: Uint32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """Split a packed fp16 pair into two float32 values, low half first."""
    result = _asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(value).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 lo, hi;
            mov.b32 {lo, hi}, $2;
            cvt.f32.f16 $0, lo;
            cvt.f32.f16 $1, hi;
        }
        """,
        "=f,=f,r",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def pack_f32x2_to_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """Match two scalar ``__float2bfloat16`` conversions without saturation."""
    return Uint32(
        _asm(
            T.i32(),
            [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 blo, bhi;
                cvt.rn.bf16.f32 blo, $1;
                cvt.rn.bf16.f32 bhi, $2;
                mov.b32 $0, {blo, bhi};
            }
            """,
            "=r,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def pack_f32x2_to_f16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """Round two float32 values into a packed fp16 pair with ``lo`` in the low half."""
    return Uint32(
        _asm(
            T.i32(),
            [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
            "cvt.rn.f16x2.f32 $0, $2, $1;",
            "=r,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


# The batched 16-byte loads of the scatter kernel, importable from here as well.
from ._cute_batch import ld_v4_u32_batch  # noqa: E402,F401
