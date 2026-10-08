"""Batched 16-byte loads for kernels that reduce or copy one pack from every rank.

:func:`ld_v4_u32_batch` issues one 16-byte load per address back to back and
returns four 32-bit words per address. Receive slots live in pinned host
memory that the NIC writes; the GPU reads them at system scope, and each such
load waits about a microsecond. ``ptxas`` schedules for ordinary global-memory
latency, so left alone it starts using the first loads before it issues the
later ones, and a pass over ``W`` sources pays ``W`` host-memory latencies.
Every returned word is therefore XORed with ``zero & (XOR of one word of every
load)``: an identity at run time (``zero`` is 0 but unknown to the compiler,
for example the sign bit of a size), which no use can start before every load
returned. ptxas then issues the loads together and a pass pays one latency.
The loads must follow the acquire of every flag they depend on, as single
loads do.

Used by the scatter kernel (``_scatter_cute.py``). Annotations stay evaluated
(no postponed evaluation), as in every module the CuTe DSL traces.
"""

from cutlass import Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def ld_v4_u32_batch(addrs, system, zero: Uint32, *, loc=None, ip=None):
    """One 16-byte load per address, issued together; a 4-word tuple per address.

    ``system[i]`` (a Python bool) selects ``ld.relaxed.sys`` for NIC-written
    pinned memory, otherwise a plain ``ld.global``.
    """
    count = len(addrs)
    if count != len(system) or count == 0:
        raise ValueError("ld_v4_u32_batch needs one scope flag per address")
    words = 4 * count
    lines = ["{", ".reg .b32 gate;"]
    for i in range(count):
        op = "ld.relaxed.sys.global.v4.u32" if system[i] else "ld.global.v4.u32"
        lines.append(f"{op} {{${4 * i}, ${4 * i + 1}, ${4 * i + 2}, ${4 * i + 3}}}, [${words + i}];")
    lines.append("mov.b32 gate, $3;")
    for i in range(1, count):
        lines.append(f"xor.b32 gate, gate, ${4 * i + 3};")
    lines.append(f"and.b32 gate, gate, ${words + count};")
    for w in range(words):
        lines.append(f"xor.b32 ${w}, ${w}, gate;")
    lines.append("}")
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * words),
        [Int64(addr).ir_value(loc=loc, ip=ip) for addr in addrs] + [Uint32(zero).ir_value(loc=loc, ip=ip)],
        "\n".join(lines),
        ",".join(["=r"] * words + ["l"] * count + ["r"]),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [
        tuple(Uint32(llvm.extractvalue(T.i32(), result, [4 * i + j], loc=loc, ip=ip)) for j in range(4))
        for i in range(count)
    ]


__all__ = ["ld_v4_u32_batch"]
