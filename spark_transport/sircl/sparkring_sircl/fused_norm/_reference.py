"""NumPy reference of the fused all-reduce + residual add + RMSNorm, operation for operation.

The fused operation of rank ``r`` takes every rank's partial sum ``x[0..W-1]``
(BF16, ``rows x hidden``), the residual stream (BF16, the same on every rank)
and the RMSNorm weight (BF16) and returns ``(normed, new_residual)``:

1. ``reduced = bf16(((x[0] + x[1]) + x[2]) + ... + x[W-1])``: float32
   additions in fixed rank order 0..W-1, each correctly rounded, then one
   round-to-nearest-even conversion to BF16. These are the bits of the ring
   session's one-shot and two-shot all-reduce
   (``sparkring_sircl/oneshot/_oneshot_cute.py``, ``_twoshot_cute.py``)
   whatever order the peers' data arrives in.
2. ``z = bf16(float32(reduced) + float32(residual))``, the new residual. The
   float32 sum of two BF16 values rounded once to BF16 equals the correctly
   rounded BF16 sum, which is what vLLM's ``__hadd2`` of two
   ``__nv_bfloat162`` computes.
3. The sum of squares of ``z`` (float32; squares of BF16 values are exact in
   float32) in the order of vLLM's ``fused_add_rms_norm_kernel`` for a row of
   at most 8192 values launched with 1024 threads
   (``csrc/libtorch_stable/layernorm_kernels.cu`` and ``type_convert.cuh`` at
   vLLM revision ``ab86b7073``):

   * thread ``t`` owns pack ``t`` (8 values ``z0..z7``) and computes
     ``((p01 + p23) + p45) + p67`` with ``pij = zi*zi + zj*zj`` rounded once;
   * each warp of 32 threads adds with ``shfl.down`` offsets 1, 2, 4, 8 and
     16 (CUB ``WarpReduceShfl``), so lane 0 holds the balanced binary tree
     of its 32 values in lane order;
   * the warp sums are added sequentially, warp 0 first (CUB
     ``BlockReduceWarpReductions``); warps without packs contribute +0.0,
     which changes no bit of a non-negative sum.
4. ``inv = rsqrt(variance / hidden + eps)``: IEEE float32 division
   (``div.rn.f32``, vLLM's layernorm is compiled without ``--use_fast_math``),
   IEEE addition, and the GPU's ``rsqrt.approx.f32``.
5. ``normed = bf16((float32(z) * inv) * float32(weight))``: two correctly
   rounded float32 products, one rounding to BF16, as vLLM's
   ``Converter::convert(x * s_variance * wf)``.

``rsqrt.approx.f32`` is a hardware approximation that NumPy cannot reproduce
bit for bit. ``fused_add_rms_norm`` therefore takes the reciprocal square
root as a function: the default is the correctly rounded float32 value
computed in float64, which differs from the GPU instruction by at most a few
float32 ulps; tests that need GPU-exact bits pass values measured on a GPU.
Every other step is exact IEEE arithmetic.

Origin: SparkRing's fused-norm kernels.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

WARP = 32
PACK = 8  # BF16 values per 16-byte pack


# --------------------------------------------------------------------------- BF16 bit helpers


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    """BF16 values stored as ``uint16`` to float32 (exact)."""
    bits = np.asarray(bits, dtype=np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(values: np.ndarray) -> np.ndarray:
    """Round float32 values to BF16 (round to nearest, ties to even; ``cvt.rn.bf16.f32``) as ``uint16``."""
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32).astype(np.uint64)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(values)
    if nan.any():
        rounded = np.where(nan, np.uint16(0x7FFF), rounded)
    return rounded


def random_bf16(rng: np.random.Generator, shape, scale: float = 1.0) -> np.ndarray:
    """BF16 bits of normally distributed values times ``scale``."""
    return f32_to_bf16((rng.standard_normal(shape) * scale).astype(np.float32))


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance in BF16 units in the last place between two BF16 bit arrays of the same sign."""
    return np.abs(a.astype(np.int32) - b.astype(np.int32))


# --------------------------------------------------------------------------- the operation


def rank_order_sum(partials: Sequence[np.ndarray]) -> np.ndarray:
    """``bf16(((x0 + x1) + x2) + ...)`` with float32 additions in rank order: the all-reduce bits."""
    acc = bf16_to_f32(partials[0]).copy()
    for part in partials[1:]:
        acc = (acc + bf16_to_f32(part)).astype(np.float32)
    return f32_to_bf16(acc)


def residual_add(reduced: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """``bf16(float32(reduced) + float32(residual))``."""
    return f32_to_bf16((bf16_to_f32(reduced) + bf16_to_f32(residual)).astype(np.float32))


def thread_partials(z: np.ndarray) -> np.ndarray:
    """Per-thread sums of squares ``((p01 + p23) + p45) + p67``; ``z`` is ``[rows, hidden]`` BF16 bits."""
    rows, hidden = z.shape
    values = bf16_to_f32(z).reshape(rows, hidden // PACK, PACK)
    squares = (values * values).astype(np.float32)  # exact: a BF16 square fits in float32
    pairs = (squares[..., 0::2] + squares[..., 1::2]).astype(np.float32)  # p01, p23, p45, p67
    acc = pairs[..., 0]
    for k in range(1, PACK // 2):
        acc = (acc + pairs[..., k]).astype(np.float32)
    return acc  # [rows, hidden // 8]


def block_sum_of_squares(z: np.ndarray, threads: int = 1024) -> np.ndarray:
    """Row sums of squares in the order of vLLM's ``fused_add_rms_norm_kernel`` (see the module docstring).

    ``threads`` is the kernel's block size; vLLM launches ``min(hidden, 1024)``
    threads for fewer than 256 rows. Rows of more than ``threads`` packs are
    outside the decode range this module models and are refused.
    """
    rows, hidden = z.shape
    packs = hidden // PACK
    if packs > threads or threads % WARP:
        raise ValueError("the modeled reduction needs at most one pack per thread and whole warps")
    per_thread = np.zeros((rows, threads), dtype=np.float32)
    per_thread[:, :packs] = thread_partials(z)
    lanes = per_thread.reshape(rows, threads // WARP, WARP)
    while lanes.shape[-1] > 1:  # shfl.down offsets 1, 2, 4, 8, 16: lane 2i adds lane 2i + 1, ...
        lanes = (lanes[..., 0::2] + lanes[..., 1::2]).astype(np.float32)
    warp_sums = lanes[..., 0]
    total = warp_sums[:, 0].copy()
    for warp in range(1, warp_sums.shape[1]):
        total = (total + warp_sums[:, warp]).astype(np.float32)
    return total


def rsqrt_correctly_rounded(x: np.ndarray) -> np.ndarray:
    """Float32 reciprocal square root rounded once from float64 (stand-in for ``rsqrt.approx.f32``)."""
    return (1.0 / np.sqrt(np.asarray(x, dtype=np.float64))).astype(np.float32)


def inverse_rms(variance_sum: np.ndarray, hidden: int, eps: float,
                rsqrt: Callable[[np.ndarray], np.ndarray] = rsqrt_correctly_rounded) -> np.ndarray:
    """``rsqrt(variance_sum / hidden + eps)`` with IEEE float32 division and addition."""
    mean = (np.asarray(variance_sum, dtype=np.float32) / np.float32(hidden)).astype(np.float32)
    return np.asarray(rsqrt((mean + np.float32(eps)).astype(np.float32)), dtype=np.float32)


def normalize(z: np.ndarray, inv: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """``bf16((float32(z) * inv) * float32(weight))`` per element; ``inv`` has one value per row."""
    scaled = (bf16_to_f32(z) * inv.reshape(-1, 1).astype(np.float32)).astype(np.float32)
    return f32_to_bf16((scaled * bf16_to_f32(weight).reshape(1, -1)).astype(np.float32))


def fused_add_rms_norm(partials: Sequence[np.ndarray], residual: np.ndarray, weight: np.ndarray,
                       eps: float, *, rsqrt: Callable[[np.ndarray], np.ndarray] = rsqrt_correctly_rounded,
                       threads: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """The fused operation on BF16 bit arrays: returns ``(normed, new_residual)``."""
    reduced = rank_order_sum(partials)
    z = residual_add(reduced, residual)
    variance_sum = block_sum_of_squares(z, threads)
    inv = inverse_rms(variance_sum, z.shape[1], eps, rsqrt)
    return normalize(z, inv, weight), z


def unfused_reference(reduced: np.ndarray, residual: np.ndarray, weight: np.ndarray, eps: float,
                      *, rsqrt: Callable[[np.ndarray], np.ndarray] = rsqrt_correctly_rounded,
                      threads: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """vLLM's ``fused_add_rms_norm`` applied to an already reduced tensor (steps 2-5)."""
    z = residual_add(reduced, residual)
    inv = inverse_rms(block_sum_of_squares(z, threads), z.shape[1], eps, rsqrt)
    return normalize(z, inv, weight), z


__all__ = [
    "bf16_to_f32",
    "block_sum_of_squares",
    "f32_to_bf16",
    "fused_add_rms_norm",
    "inverse_rms",
    "normalize",
    "random_bf16",
    "rank_order_sum",
    "residual_add",
    "rsqrt_correctly_rounded",
    "thread_partials",
    "ulp_distance",
    "unfused_reference",
]
