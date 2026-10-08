"""Inline PTX for the fused all-reduce + residual add + RMSNorm kernels.

Two groups of instructions are spelled out instead of left to the CuTe DSL's
code generator:

* memory operations whose scope is part of the protocol: the receive slots
  and flags live in pinned host memory that the NIC writes and the GPU reads
  in place (``ld.relaxed.sys`` for payload), the doorbell words are read by a
  CPU thread (``st.relaxed.sys`` after ``fence.sc.sys``), and the arrival
  counters and epoch are device words (GPU scope). The flag waits use the
  session's timed wait (``sparkring_sircl/oneshot/_timed_wait.py``);
* every floating-point operation of the numerics: ``add.rn.f32``,
  ``mul.rn.f32``, ``div.rn.f32``, ``rsqrt.approx.f32`` and the BF16
  conversions. An explicit ``.rn`` keeps ``ptxas`` from contracting a
  multiply and an add into one fused multiply-add, so the operation order
  documented in ``_reference`` is the order the GPU executes.

The gated batch load (:func:`ld_sys_v4_gated`) issues every load of a pass
before any result is used, so a pass over pinned host memory pays one load
latency (the technique of ``sparkring_sircl/oneshot/_cute_batch.py``).

Origin: SparkRing's fused-norm kernels.
"""

# Annotations stay evaluated (no postponed evaluation): the CuTe DSL reads
# parameter types from annotation objects.

from typing import Tuple

from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


def _asm(result_type, operands, text, constraints, *, side_effects=True, loc=None, ip=None):
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


def _words(result, count: int, *, loc=None, ip=None):
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(count)
    )


# --------------------------------------------------------------------------- scalar memory


@dsl_user_op
def ld_relaxed_gpu_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    """32-bit relaxed load at GPU scope (device words: epoch, poison)."""
    return Uint32(_asm(T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip)],
                       "ld.relaxed.gpu.global.u32 $0, [$1];", "=r,l", loc=loc, ip=ip))


@dsl_user_op
def ld_relaxed_sys_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    """32-bit relaxed load at system scope (command ring words written by the host)."""
    return Uint32(_asm(T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip)],
                       "ld.relaxed.sys.global.u32 $0, [$1];", "=r,l", loc=loc, ip=ip))


@dsl_user_op
def st_relaxed_sys_u32(addr: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    """32-bit relaxed store at system scope (doorbell and error words read by the host)."""
    _asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
         "st.relaxed.sys.global.u32 [$0], $1;", "l,r", loc=loc, ip=ip)


@dsl_user_op
def st_release_gpu_u32(addr: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    """32-bit release store at GPU scope (epoch and poison words)."""
    _asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
         "st.release.gpu.global.u32 [$0], $1;", "l,r", loc=loc, ip=ip)


@dsl_user_op
def atom_inc_relaxed_gpu_u32(addr: Int64, bound: Uint32, *, loc=None, ip=None) -> Uint32:
    """``atom.inc``: returns the old value and stores ``old >= bound ? 0 : old + 1``.

    With ``bound = arrivals - 1`` a counter that starts at 0 returns to 0 after
    exactly ``arrivals`` increments, and the last arrival reads ``bound``. The
    counter is therefore valid for any grid size, not only powers of two.
    """
    return Uint32(_asm(T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(bound).ir_value(loc=loc, ip=ip)],
                       "atom.relaxed.gpu.global.inc.u32 $0, [$1], $2;", "=r,l,r", loc=loc, ip=ip))


@dsl_user_op
def fence_sc_sys(*, loc=None, ip=None) -> None:
    """Sequentially consistent fence at system scope."""
    _asm(None, [], "fence.sc.sys;", "", loc=loc, ip=ip)


@dsl_user_op
def fence_sc_gpu(*, loc=None, ip=None) -> None:
    """Sequentially consistent fence at GPU scope."""
    _asm(None, [], "fence.sc.gpu;", "", loc=loc, ip=ip)


# --------------------------------------------------------------------------- 16-byte memory


@dsl_user_op
def ld_global_v4(addr: Int64, *, loc=None, ip=None) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """16-byte load of device memory as four 32-bit words."""
    result = _asm(llvm.StructType.get_literal([T.i32()] * 4), [Int64(addr).ir_value(loc=loc, ip=ip)],
                  "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", loc=loc, ip=ip)
    return _words(result, 4, loc=loc, ip=ip)


@dsl_user_op
def ld_sys_v4(addr: Int64, *, loc=None, ip=None) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """16-byte relaxed load at system scope (one NIC-written pack)."""
    result = _asm(llvm.StructType.get_literal([T.i32()] * 4), [Int64(addr).ir_value(loc=loc, ip=ip)],
                  "ld.relaxed.sys.global.v4.u32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", loc=loc, ip=ip)
    return _words(result, 4, loc=loc, ip=ip)


@dsl_user_op
def ld_sys_v4_gated(addrs, zero: Uint32, *, loc=None, ip=None):
    """One system-scope 16-byte load per address, all issued before any result is used.

    ``zero`` must be 0 at run time but unknown to the compiler. Every returned
    word is XORed with ``zero AND (XOR of one word of every load)``, an
    identity at run time that makes every use depend on every load, so
    ``ptxas`` cannot interleave uses of early loads with the issue of later
    ones. Returns one 4-word tuple per address.
    """
    count = len(addrs)
    if count == 0:
        raise ValueError("ld_sys_v4_gated needs at least one address")
    words = 4 * count
    lines = ["{", ".reg .b32 gate;"]
    for i in range(count):
        lines.append(
            f"ld.relaxed.sys.global.v4.u32 {{${4 * i}, ${4 * i + 1}, ${4 * i + 2}, ${4 * i + 3}}}, [${words + i}];"
        )
    lines.append("mov.b32 gate, $3;")
    for i in range(1, count):
        lines.append(f"xor.b32 gate, gate, ${4 * i + 3};")
    lines.append(f"and.b32 gate, gate, ${words + count};")
    for w in range(words):
        lines.append(f"xor.b32 ${w}, ${w}, gate;")
    lines.append("}")
    result = _asm(
        llvm.StructType.get_literal([T.i32()] * words),
        [Int64(a).ir_value(loc=loc, ip=ip) for a in addrs] + [Uint32(zero).ir_value(loc=loc, ip=ip)],
        "\n".join(lines),
        ",".join(["=r"] * words + ["l"] * count + ["r"]),
        loc=loc, ip=ip,
    )
    flat = _words(result, words, loc=loc, ip=ip)
    return [flat[4 * i: 4 * i + 4] for i in range(count)]


@dsl_user_op
def st_global_v4(addr: Int64, w0: Uint32, w1: Uint32, w2: Uint32, w3: Uint32, *, loc=None, ip=None) -> None:
    """16-byte store of four 32-bit words."""
    _asm(None,
         [Int64(addr).ir_value(loc=loc, ip=ip)] + [Uint32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
         "st.global.v4.u32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc=loc, ip=ip)


# --------------------------------------------------------------------------- arithmetic


@dsl_user_op
def bf16x2_to_f32(word: Uint32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """The two BF16 values of a 32-bit word as float32 (exact): low half, then high half."""
    result = _asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(word).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 lo, hi;
            shl.b32 lo, $2, 16;
            and.b32 hi, $2, 0xffff0000;
            mov.b32 $0, lo;
            mov.b32 $1, hi;
        }
        """,
        "=f,=f,r", side_effects=False, loc=loc, ip=ip)
    return (Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)))


@dsl_user_op
def f32x2_to_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """Round two float32 values to BF16 (round to nearest even) and pack them, ``lo`` in the low half."""
    return Uint32(_asm(T.i32(), [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
                       "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f", side_effects=False, loc=loc, ip=ip))


def _binary(op: str):
    @dsl_user_op
    def apply(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
        return Float32(_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
                            f"{op} $0, $1, $2;", "=f,f,f", side_effects=False, loc=loc, ip=ip))

    apply.__name__ = op.replace(".", "_")
    apply.__doc__ = f"``{op}``: one correctly rounded float32 operation, never contracted."
    return apply


add_rn = _binary("add.rn.f32")
mul_rn = _binary("mul.rn.f32")
div_rn = _binary("div.rn.f32")


@dsl_user_op
def rsqrt_approx(value: Float32, *, loc=None, ip=None) -> Float32:
    """``rsqrt.approx.f32``, the instruction CUDA's ``rsqrtf`` compiles to without ``-ftz``."""
    return Float32(_asm(T.f32(), [Float32(value).ir_value(loc=loc, ip=ip)],
                        "rsqrt.approx.f32 $0, $1;", "=f,f", side_effects=False, loc=loc, ip=ip))


@dsl_user_op
def shfl_down_f32(value: Float32, offset: Uint32, *, loc=None, ip=None) -> Float32:
    """``shfl.sync.down.b32`` over the full warp; a lane past the end receives its own value."""
    return Float32(_asm(
        T.f32(),
        [Float32(value).ir_value(loc=loc, ip=ip), Uint32(offset).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 moved;
            mov.b32 moved, $1;
            shfl.sync.down.b32 moved, moved, $2, 0x1f, 0xffffffff;
            mov.b32 $0, moved;
        }
        """,
        "=f,f,r", loc=loc, ip=ip))


@dsl_user_op
def min_s32(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """Signed 32-bit minimum."""
    return Int32(_asm(T.i32(), [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
                      "min.s32 $0, $1, $2;", "=r,r,r", side_effects=False, loc=loc, ip=ip))


__all__ = [
    "add_rn",
    "atom_inc_relaxed_gpu_u32",
    "bf16x2_to_f32",
    "div_rn",
    "f32x2_to_bf16x2",
    "fence_sc_gpu",
    "fence_sc_sys",
    "ld_global_v4",
    "ld_relaxed_gpu_u32",
    "ld_relaxed_sys_u32",
    "ld_sys_v4",
    "ld_sys_v4_gated",
    "min_s32",
    "mul_rn",
    "rsqrt_approx",
    "shfl_down_f32",
    "st_global_v4",
    "st_relaxed_sys_u32",
    "st_release_gpu_u32",
]
