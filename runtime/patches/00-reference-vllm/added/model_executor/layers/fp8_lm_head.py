# SPDX-License-Identifier: Apache-2.0
"""Env-gated W8A16 fp8 lm_head for GLM-5.2 on DGX Spark GB10 (SM121, aarch64).

Weight-only fp8-E4M3 quantization of the (vocab-sharded) lm_head matrix with
per-output-channel (per vocab row) fp32 scales, consumed by a Triton GEMM/GEMV
kernel with bf16 activations and fp32 accumulation. Activations and logits
stay bf16 -- nothing about the numerics of the rest of the model changes.

Why Triton and not an existing vLLM path on this platform:
  - CUTLASS c3x w8a8 is broken on GB10.
  - The fork's only fp8 W8A16 scheme (compressed_tensors_w8a16_fp8) dispatches
    to Marlin (MarlinFP8ScaledMMLinearKernel), untested on SM121/aarch64.
  - deep_gemm fp8 quantizes activations (fp8 x fp8), which is not W8A16.
Triton compiles fine on SM121 and fp8e4m3 -> bf16 conversion is exact
(3-bit mantissa fits in bf16's 8, e4m3's exponent range is covered), so the
only numerical deltas vs a bf16 GEMM come from the weight round-trip itself
plus fp32 summation order.

CUDA-graph capture safety (compute_logits of the MTP draft runs INSIDE a FULL
cudagraph in this deployment -- speculator.py captures "model forward +
compute_logits + sample" as one graph):
  - quantization happens once at load time (process_weights_after_loading);
  - no @triton.autotune (autotune would launch candidate kernels at capture
    time); kernel config is a pure function of tensor shapes, so a given
    capture shape always takes the identical launch;
  - no data-dependent host branching, no syncs, no .item()/.cpu() calls;
  - split-K reduction is a deterministic partial-buffer sum (no atomics).

This module intentionally imports only torch/triton (no vllm) so the offline
precision gate (/home/tn/glm-hybrid/test_fp8_lmhead.py) can load it standalone
via importlib and validate the *exact* production quant + kernel code.
"""

import functools

import torch
import triton
import triton.language as tl

FP8_E4M3_MAX = 448.0
# Floor for per-channel scales: all-zero rows (vocab padding) quantize to 0
# with any positive scale; the floor only prevents 0/0.
_SCALE_EPS = 2.0**-24


def quantize_weight_fp8_per_channel(
    weight: torch.Tensor, chunk_rows: int = 4096
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a [N, K] bf16/fp16 weight to fp8-E4M3 with per-row fp32 scales.

    scale[n] = max(amax(|W[n, :]|) / 448, 2^-24)   (fp32)
    q[n, k]  = round_to_fp8_e4m3( W[n, k] / scale[n] )

    Chunked over rows to bound the fp32 transient (full-matrix fp32 would be
    ~0.9 GB for the 38720x6144 production shard -- unfriendly on UMA GB10).

    Returns (qweight fp8_e4m3fn [N, K], scale fp32 [N]).
    """
    assert weight.dim() == 2, f"expected 2D weight, got {tuple(weight.shape)}"
    n_rows = weight.shape[0]
    qweight = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scale = torch.empty(n_rows, dtype=torch.float32, device=weight.device)
    for i in range(0, n_rows, chunk_rows):
        w = weight[i : i + chunk_rows].to(torch.float32)
        s = torch.clamp(w.abs().amax(dim=1) / FP8_E4M3_MAX, min=_SCALE_EPS)
        q = (w / s[:, None]).clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX)
        qweight[i : i + chunk_rows] = q.to(torch.float8_e4m3fn)
        scale[i : i + chunk_rows] = s
    return qweight, scale


@triton.jit
def _w8a16_lmhead_kernel(
    x_ptr,  # [M, K] bf16/fp16 activations
    w_ptr,  # [N, K] uint8 view of fp8-E4M3 weight
    s_ptr,  # [N] fp32 per-output-channel scales
    out_ptr,  # [M, N] output, x dtype (used when SPLIT_K == 1)
    part_ptr,  # [SPLIT_K, M, N] fp32 partials (used when SPLIT_K > 1)
    M,
    N,
    K,
    k_chunk,  # ceil(K / SPLIT_K), computed on host
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    stride_pk,
    stride_pm,
    stride_pn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = rm < M
    n_mask = rn < N

    k_start = pid_k * k_chunk
    k_end = tl.minimum(k_start + k_chunk, K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_start, k_end, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        k_mask = rk < k_end
        a = tl.load(
            x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        wq = tl.load(
            w_ptr + rn[:, None] * stride_wn + rk[None, :] * stride_wk,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0,
        )
        # fp8e4m3 -> bf16/fp16 is exact (3-bit mantissa, covered exponent
        # range); dequant (scale) is deferred to the epilogue since the
        # scale is constant along K. Match a's dtype for tl.dot.
        w = wq.to(tl.float8e4nv, bitcast=True).to(a.dtype)
        acc = tl.dot(a, tl.trans(w), acc)

    s = tl.load(s_ptr + rn, mask=n_mask, other=0.0)
    acc = acc * s[None, :]

    out_mask = m_mask[:, None] & n_mask[None, :]
    if SPLIT_K == 1:
        tl.store(
            out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on,
            acc.to(out_ptr.dtype.element_ty),
            mask=out_mask,
        )
    else:
        tl.store(
            part_ptr
            + pid_k * stride_pk
            + rm[:, None] * stride_pm
            + rn[None, :] * stride_pn,
            acc,
            mask=out_mask,
        )


@functools.lru_cache(maxsize=None)
def _num_sms() -> int:
    if not torch.cuda.is_available():
        return 48
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


def _pick_config(m: int, n: int, k: int) -> tuple[int, int, int, int, int, int]:
    """Static, shape-only launch config (capture-safe: same shape -> same
    launch, no autotuning, no device queries beyond a cached SM count).

    Returns (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages).
    """
    if m <= 16:
        block_m = 16
    elif m <= 32:
        block_m = 32
    else:
        # Cap at 64: each additional m-block re-streams the fp8 weight, so
        # cover moderate M in one block; M > 64 (rare for logits) tiles.
        block_m = 64
    block_n = 128
    block_k = 128
    num_warps = 4
    num_stages = 3 if block_m <= 32 else 2

    # Split-K only for GEMV-shaped calls that would otherwise underfill the
    # GPU. Production lm_head shard (N=38720 -> 303 n-blocks) saturates GB10
    # without it; small-N or tiny-vocab cases benefit.
    split_k = 1
    n_blocks = triton.cdiv(n, block_n)
    m_blocks = triton.cdiv(m, block_m)
    if m <= 16:
        target = 2 * _num_sms()
        while (
            n_blocks * m_blocks * split_k < target
            and split_k < 8
            and (k // (2 * split_k)) >= block_k
        ):
            split_k *= 2
    return block_m, block_n, block_k, split_k, num_warps, num_stages


def w8a16_fp8_matmul(
    x: torch.Tensor,
    qweight_u8: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    _force_split_k: int | None = None,
) -> torch.Tensor:
    """out = x @ dequant(qweight).T  (+ bias), returned in x.dtype.

    x:          [..., K] bf16/fp16 (any leading shape; flattened internally)
    qweight_u8: [N, K] uint8 view of an fp8-E4M3 tensor, contiguous
    scale:      [N] fp32 per-output-channel scales
    _force_split_k: test-only override used by the offline precision gate.
    """
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    m, k = x2.shape
    n = qweight_u8.shape[0]
    assert qweight_u8.shape[1] == k, (qweight_u8.shape, k)
    if m == 0:  # shape-based, capture-safe
        return torch.empty((*orig_shape[:-1], n), dtype=x2.dtype, device=x2.device)

    block_m, block_n, block_k, split_k, num_warps, num_stages = _pick_config(m, n, k)
    if _force_split_k is not None:
        split_k = _force_split_k

    k_chunk = triton.cdiv(k, split_k)
    grid = (triton.cdiv(n, block_n), split_k, triton.cdiv(m, block_m))

    out = torch.empty((m, n), dtype=x2.dtype, device=x2.device)
    if split_k == 1:
        part = out  # unused dummy; dead branch is compiled out
        part_strides = (0, 0, 0)
    else:
        part = torch.empty((split_k, m, n), dtype=torch.float32, device=x2.device)
        part_strides = part.stride()

    _w8a16_lmhead_kernel[grid](
        x2,
        qweight_u8,
        scale,
        out,
        part,
        m,
        n,
        k,
        k_chunk,
        x2.stride(0),
        x2.stride(1),
        qweight_u8.stride(0),
        qweight_u8.stride(1),
        out.stride(0),
        out.stride(1),
        part_strides[0],
        part_strides[1],
        part_strides[2],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT_K=split_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if split_k > 1:
        # Deterministic reduction (no atomics): fp32 partial sum, then cast.
        out = part.sum(dim=0).to(x2.dtype)
    if bias is not None:
        out = out + bias
    return out.view(*orig_shape[:-1], n)
