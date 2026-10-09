"""Triton kernels of glm_dcp_decode_comm: the DCP query and the combine of the packed all-to-all.

Both restate statements of the image's vLLM kernels (pinned in ``FILE_CHECKS``)
with only their addressing changed, so each output word equals the image's:

- :func:`rope_cat` builds this rank's DCP query ``q_cat[b, h] = [ql_nope[b, h] |
  RoPE(q_pe[b, h])]`` in one launch. The ``ql_nope`` values are copied
  unchanged (the image's ``torch.cat``); the RoPE is program 0 of
  ``_fused_q_kernel`` (``vllm/models/deepseek_v32/common/kernels.py``) on the
  BF16 query path: ``cos`` and ``sin`` loaded from the cache row of the token's
  position and converted to float32, the interleaved pair ``x1 = q_pe[2i]``,
  ``x2 = q_pe[2i + 1]`` converted to float32, ``r1 = x1 * cos - x2 * sin`` and
  ``r2 = x2 * cos + x1 * sin`` stored in the output dtype. The image runs that
  program, not its CuTe DSL ``fused_q``, whenever the query is BF16
  (``is_fused_q_cutedsl_supported`` needs ``quantize_mqa``).
- :func:`wire_combine` is ``_dcp_a2a_unpack_combine_kernel``
  (``vllm/v1/attention/ops/dcp.py``) for one decode batch, reading source
  ``s``'s partial output and LSE from the packed all-to-all's receive buffer
  (``layout``: data region, then the FP32 LSE region) and this rank's own share
  from the attention output and LSE in place. The LSE words are the FP32 bits
  the image's pack splits into two BF16 slots and its combine reassembles; the
  three loops over sources, the clean-up of NaN and positive infinity, the
  maximum, the exponential sum, the weights and the float32 accumulation are
  the image's statements in the image's order.

Triton is imported with this module; the worker imports it on first use.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import layout as L


@triton.jit
def _rope_cat_kernel(
    pos_ptr,
    q_pe_ptr,
    q_pe_stride0,
    q_pe_stride1,
    cos_sin_ptr,
    cos_sin_stride,
    ql_nope_ptr,
    ql_nope_stride0,
    ql_nope_stride1,
    q_cat_ptr,
    q_cat_stride0,
    q_cat_stride1,
    HALF_ROT_DIM: tl.constexpr,
    QL_NOPE_DIM: tl.constexpr,
    QL_NOPE_BLOCK: tl.constexpr,
):
    tok_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    dst = q_cat_ptr + tok_idx * q_cat_stride0 + head_idx * q_cat_stride1

    # [ql_nope | ...]: the latent values unchanged.
    ql_nope_off = tl.arange(0, QL_NOPE_BLOCK)
    ql_nope_mask = ql_nope_off < QL_NOPE_DIM
    ql_nope = tl.load(
        ql_nope_ptr + tok_idx * ql_nope_stride0 + head_idx * ql_nope_stride1 + ql_nope_off,
        mask=ql_nope_mask,
    )
    tl.store(dst + ql_nope_off, ql_nope, mask=ql_nope_mask)

    # [... | RoPE(q_pe)]: the statements of _fused_q_kernel program 0, BF16 query path.
    pos = tl.load(pos_ptr + tok_idx)
    block = tl.arange(0, HALF_ROT_DIM)
    cos = tl.load(cos_sin_ptr + pos * cos_sin_stride + block)
    cos = cos.to(tl.float32)
    sin = tl.load(cos_sin_ptr + pos * cos_sin_stride + block + HALF_ROT_DIM)
    sin = sin.to(tl.float32)
    rot_off = tl.arange(0, HALF_ROT_DIM)
    x1 = tl.load(
        q_pe_ptr
        + tok_idx * q_pe_stride0
        + head_idx * q_pe_stride1
        + rot_off * 2,
    ).to(tl.float32)
    x2 = tl.load(
        q_pe_ptr
        + tok_idx * q_pe_stride0
        + head_idx * q_pe_stride1
        + rot_off * 2
        + 1
    ).to(tl.float32)
    r1 = x1 * cos - x2 * sin
    r2 = x2 * cos + x1 * sin
    out_ty = q_cat_ptr.dtype.element_ty
    q_pe_dst = dst + QL_NOPE_DIM
    tl.store(q_pe_dst + rot_off * 2, r1.to(out_ty))
    tl.store(q_pe_dst + rot_off * 2 + 1, r2.to(out_ty))


def rope_cat(positions: torch.Tensor, q_pe: torch.Tensor, cos_sin_cache: torch.Tensor, ql_nope: torch.Tensor,
             q_cat: torch.Tensor, rows: int) -> None:
    """Write ``[ql_nope | RoPE(q_pe)]`` of the first ``rows`` rows into ``q_cat`` (``[>= rows, heads, 576]``)."""
    heads = int(q_pe.shape[1])
    if rows <= 0:
        return
    if (positions.dtype != torch.int64 or q_pe.stride(2) != 1 or ql_nope.stride(2) != 1 or q_cat.stride(2) != 1
            or int(ql_nope.shape[2]) != L.QL_NOPE_DIM or int(q_pe.shape[2]) != L.ROPE_DIM
            or int(cos_sin_cache.shape[-1]) != L.ROPE_DIM or int(q_cat.shape[2]) != L.HEAD_DIM
            or int(ql_nope.shape[1]) != heads or int(q_cat.shape[1]) != heads):
        raise ValueError(f"rope_cat: unsupported shapes or strides: q_pe {tuple(q_pe.shape)} {q_pe.stride()}, "
                         f"ql_nope {tuple(ql_nope.shape)} {ql_nope.stride()}, q_cat {tuple(q_cat.shape)}, "
                         f"cache {tuple(cos_sin_cache.shape)}, positions {positions.dtype}")
    _rope_cat_kernel[(int(rows), heads)](
        positions,
        q_pe,
        q_pe.stride(0),
        q_pe.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        ql_nope,
        ql_nope.stride(0),
        ql_nope.stride(1),
        q_cat,
        q_cat.stride(0),
        q_cat.stride(1),
        HALF_ROT_DIM=L.ROPE_DIM // 2,
        QL_NOPE_DIM=L.QL_NOPE_DIM,
        QL_NOPE_BLOCK=triton.next_power_of_2(L.QL_NOPE_DIM),
        num_warps=1,
    )


@triton.jit
def _wire_lse(recv_f32_ptr, own_lse_ptr, peer_f32, lse_base, record, own_lse_offset,
              SOURCE: tl.constexpr, RANK: tl.constexpr):
    """Source ``SOURCE``'s LSE of this program's row and head, cleaned as the image's combine cleans it."""
    if SOURCE == RANK:
        lse_val = tl.load(own_lse_ptr + own_lse_offset).to(tl.float32)
    else:
        lse_val = tl.load(recv_f32_ptr + peer_f32 + lse_base + record).to(tl.float32)
    lse_val = tl.where(
        (lse_val != lse_val) | (lse_val == float("inf")),
        -float("inf"),
        lse_val,
    )
    return lse_val


@triton.jit
def _wire_combine_kernel(
    recv_bf16_ptr,
    recv_f32_ptr,
    own_out_ptr,
    own_lse_ptr,
    out_ptr,
    chunk_bf16,
    chunk_f32,
    lse_base,
    own_out_stride_B,
    own_out_stride_H,
    own_lse_stride_B,
    out_stride_B,
    out_stride_H,
    N: tl.constexpr,
    RANK: tl.constexpr,
    H_PER_RANK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)
    record = batch_idx * H_PER_RANK + head_idx
    own_head = RANK * H_PER_RANK + head_idx
    own_lse_offset = batch_idx * own_lse_stride_B + own_head
    chunk_bf16_wide = chunk_bf16.to(tl.int64)
    chunk_f32_wide = chunk_f32.to(tl.int64)

    lse_max = -float("inf")
    for rank_idx in tl.static_range(N):
        lse_val = _wire_lse(recv_f32_ptr, own_lse_ptr, rank_idx * chunk_f32_wide, lse_base, record,
                            own_lse_offset, rank_idx, RANK)
        lse_max = tl.maximum(lse_max, lse_val)

    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)

    lse_sum = 0.0
    for rank_idx in tl.static_range(N):
        lse_val = _wire_lse(recv_f32_ptr, own_lse_ptr, rank_idx * chunk_f32_wide, lse_base, record,
                            own_lse_offset, rank_idx, RANK)
        if IS_BASE_E:
            lse_sum += tl.exp(lse_val - lse_max)
        else:
            lse_sum += tl.exp2(lse_val - lse_max)

    if IS_BASE_E:  # noqa: SIM108
        global_lse = tl.log(lse_sum) + lse_max
    else:
        global_lse = tl.log2(lse_sum) + lse_max

    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for rank_idx in tl.static_range(N):
        lse_val = _wire_lse(recv_f32_ptr, own_lse_ptr, rank_idx * chunk_f32_wide, lse_base, record,
                            own_lse_offset, rank_idx, RANK)
        if IS_BASE_E:
            weight = tl.exp(lse_val - global_lse)
        else:
            weight = tl.exp2(lse_val - global_lse)
        weight = tl.where(weight != weight, 0.0, weight)
        if rank_idx == RANK:
            partial = tl.load(
                own_out_ptr + batch_idx * own_out_stride_B + own_head * own_out_stride_H + d_offsets
            ).to(tl.float32)
        else:
            partial = tl.load(
                recv_bf16_ptr + rank_idx * chunk_bf16_wide + record * HEAD_DIM + d_offsets
            ).to(tl.float32)
        partial = tl.where(weight == 0.0, 0.0, partial)
        acc += partial * weight

    final_offsets = batch_idx * out_stride_B + head_idx * out_stride_H + d_offsets
    tl.store(out_ptr + final_offsets, acc)


def wire_combine(recv: torch.Tensor, out: torch.Tensor, lse: torch.Tensor, world: int, rank: int, heads: int,
                 is_base_e: bool) -> torch.Tensor:
    """The combined output ``[rows, heads, 512]`` of this rank's heads.

    ``recv`` is the packed all-to-all's receive buffer ``[world, chunk bytes]``
    (uint8; row ``s`` from source ``s``, the own row unused); ``out`` and
    ``lse`` are this rank's ``[rows, world * heads, 512]`` attention output and
    ``[rows, world * heads]`` FP32 LSE, whose own heads are read in place.
    """
    rows = int(out.shape[0])
    chunk = L.wire_chunk_bytes(rows, heads)
    if (recv.dtype != torch.uint8 or tuple(recv.shape) != (int(world), chunk) or not recv.is_contiguous()
            or out.stride(2) != 1 or lse.stride(1) != 1 or int(out.shape[2]) != L.V_DIM
            or int(out.shape[1]) != int(world) * int(heads) or tuple(lse.shape) != (rows, int(world) * int(heads))
            or lse.dtype != torch.float32):
        raise ValueError(f"wire_combine: receive buffer {tuple(recv.shape)} {recv.dtype}, output "
                         f"{tuple(out.shape)} {out.stride()}, LSE {tuple(lse.shape)} {lse.dtype} do not fit "
                         f"{world} ranks of {heads} heads")
    result = torch.empty((rows, int(heads), L.V_DIM), dtype=out.dtype, device=out.device)
    _wire_combine_kernel[(rows, int(heads))](
        recv.view(torch.bfloat16),
        recv.view(torch.float32),
        out,
        lse,
        result,
        chunk // 2,
        chunk // 4,
        L.wire_data_bytes(rows, heads) // 4,
        out.stride(0),
        out.stride(1),
        lse.stride(0),
        result.stride(0),
        result.stride(1),
        N=int(world),
        RANK=int(rank),
        H_PER_RANK=int(heads),
        HEAD_DIM=L.V_DIM,
        IS_BASE_E=bool(is_base_e),
    )
    return result


__all__ = ["rope_cat", "wire_combine"]
