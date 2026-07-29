# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 DSpark draft module scaffolding.

DSpark checkpoints store their draft module under ``mtp.*`` but the module is
not the serial DeepSeek V4 MTP architecture.  This file provides a draft-only
module with parameter names and weight loading matching the DSpark checkpoint so
the runtime can be implemented incrementally without accidentally falling back
to serial MTP.
"""

import contextlib
import json
import os
import re
import time
import typing
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import safe_open

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_post_tilelang,
)
from vllm.model_executor.kernels.mhc.torch import (
    mhc_post_torch,
    mhc_pre_torch,
)
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _decode_e8m0_scales,
    _fused_inverse_rope_gptj,
    _get_cached_wo_a_bf16,
    rocm_inv_rope_einsum,
)
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

from .model import (
    DeepseekV4DecoderLayer,
    make_deepseek_v4_expert_params_mapping,
)
from .ops.dspark_sparse_attn_tilelang import (
    dspark_sparse_attn,
    dspark_sparse_attn_reference,
)

logger = init_logger(__name__)

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


def _small_tensor_stats(tensor: torch.Tensor) -> dict[str, object]:
    x = tensor.detach()
    xf = x.float()
    finite = torch.isfinite(xf)
    if bool(finite.any().item()):
        values = xf[finite]
        mean = float(values.mean().cpu())
        std = float(values.std(unbiased=False).cpu())
        absmax = float(values.abs().max().cpu())
    else:
        mean = float("nan")
        std = float("nan")
        absmax = float("nan")
    out = {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "finite": int(finite.sum().cpu()),
        "numel": x.numel(),
        "mean": mean,
        "std": std,
        "absmax": absmax,
    }
    if x.numel() <= 128:
        out["values"] = x.detach().to(torch.float32).cpu().reshape(-1).tolist()
    return out


def _maybe_dump_stage_debug(
    label: str,
    tensors: dict[str, torch.Tensor],
    extra: dict[str, object] | None = None,
) -> None:
    debug_dir = os.getenv("VLLM_DSPARK_DEBUG_DIR")
    if not debug_dir:
        return
    max_dumps = int(os.getenv("VLLM_DSPARK_STAGE_DEBUG_MAX_DUMPS", "16"))
    path = os.path.join(debug_dir, "stage_debug.jsonl")
    existing = 0
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = sum(1 for _ in f)
    if existing >= max_dumps:
        return
    os.makedirs(debug_dir, exist_ok=True)
    row: dict[str, object] = {
        "time": time.time(),
        "label": label,
        "tensors": {name: _small_tensor_stats(t) for name, t in tensors.items()},
    }
    if extra:
        row["extra"] = extra
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _maybe_save_tensor_debug(
    label: str,
    tensors: dict[str, torch.Tensor],
    extra: dict[str, object] | None = None,
) -> None:
    debug_dir = os.getenv("VLLM_DSPARK_TENSOR_DEBUG_DIR")
    if not debug_dir:
        return
    max_dumps = int(os.getenv("VLLM_DSPARK_TENSOR_DEBUG_MAX_DUMPS", "4"))
    os.makedirs(debug_dir, exist_ok=True)
    rank = get_tensor_model_parallel_rank()
    existing = sorted(
        name
        for name in os.listdir(debug_dir)
        if name.startswith(f"rank{rank:02d}_") and name.endswith(".pt")
    )
    if len(existing) >= max_dumps:
        return
    idx = len(existing)
    payload: dict[str, object] = {
        "time": time.time(),
        "label": label,
        "rank": rank,
        "extra": extra or {},
        "tensors": {
            name: tensor.detach().cpu()
            for name, tensor in tensors.items()
            if tensor is not None
        },
    }
    path = os.path.join(debug_dir, f"rank{rank:02d}_{idx:04d}_{label}.pt")
    torch.save(payload, path)


def _dspark_dump_layer_parts_enabled() -> bool:
    return os.getenv("VLLM_DSPARK_TENSOR_DEBUG_LAYER_PARTS") == "1"


def _build_dspark_topk_idxs(
    *,
    window_size: int,
    batch_size: int,
    block_size: int,
    positions: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build DSpark's rolling-window+draft attention index tensor.

    The public DSpark reference uses exactly this dense gather contract: all
    available target-window positions followed by the local draft block rows.
    """
    target = torch.arange(window_size, device=device, dtype=torch.int32)
    valid_target = target.view(1, 1, window_size) <= positions.to(torch.int32).view(
        batch_size, 1, 1
    )
    target_idxs = torch.where(
        valid_target,
        target.view(1, 1, window_size),
        target.new_full((1, 1, window_size), -1),
    )
    target_idxs = target_idxs.expand(batch_size, block_size, window_size)
    draft_idxs = (
        window_size + torch.arange(block_size, device=device, dtype=torch.int32)
    ).view(1, 1, block_size)
    draft_idxs = draft_idxs.expand(batch_size, block_size, block_size)
    return torch.cat([target_idxs, draft_idxs], dim=-1).contiguous()


@triton.jit
def _store_dspark_context_kv_kernel(
    cache_ptr,
    main_kv_ptr,
    positions_ptr,
    query_start_loc_ptr,
    num_rejected_ptr,
    cache_indices_ptr,
    cache_stride_batch: tl.constexpr,
    cache_stride_window: tl.constexpr,
    cache_stride_dim: tl.constexpr,
    main_kv_stride_tokens: tl.constexpr,
    main_kv_stride_dim: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    req_idx = tl.program_id(0)
    tail_idx = tl.program_id(1)
    dim_block = tl.program_id(2)
    cache_idx = tl.load(cache_indices_ptr + req_idx).to(tl.int64)

    start = tl.load(query_start_loc_ptr + req_idx)
    end = tl.load(query_start_loc_ptr + req_idx + 1)
    rejected = tl.load(num_rejected_ptr + req_idx)
    valid_end = end - rejected
    valid_len = tl.maximum(valid_end - start, 0)
    tail_len = tl.minimum(valid_len, WINDOW_SIZE)
    token_idx = valid_end - tail_len + tail_idx
    safe_token_idx = tl.maximum(token_idx, 0)

    dims = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM
    valid = tail_idx < tail_len

    pos = tl.load(positions_ptr + safe_token_idx, mask=valid, other=0)
    slot = pos % WINDOW_SIZE
    values = tl.load(
        main_kv_ptr
        + safe_token_idx * main_kv_stride_tokens
        + dims * main_kv_stride_dim,
        mask=valid & dim_mask,
        other=0.0,
    )
    tl.store(
        cache_ptr
        + cache_idx * cache_stride_batch
        + slot * cache_stride_window
        + dims * cache_stride_dim,
        values,
        mask=valid & dim_mask,
    )


def _fake_fp8_e4m3_mxfp_inplace(x: torch.Tensor, block_size: int = 64) -> None:
    """Mirror the public DSpark QAT KV fake-quant path.

    The HF DSpark inference code applies an in-place FP8 E4M3 quant+dequant to
    the non-RoPE KV dimensions with power-of-two scales.  This keeps the draft
    numerics aligned with the QAT-trained module without changing the storage
    format of the rolling KV cache.
    """
    if x.numel() == 0:
        return
    if x.shape[-1] % block_size != 0:
        raise ValueError(
            "DSpark fake-FP8 block size must divide the last dimension: "
            f"{x.shape[-1]} % {block_size} != 0."
        )
    view = x.reshape(-1, x.shape[-1] // block_size, block_size)
    amax = view.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    quant = torch.clamp(view / scale, -448.0, 448.0).to(torch.float8_e4m3fn)
    view.copy_(quant.to(view.dtype) * scale.to(view.dtype))


def _fake_fp8_e4m3_mxfp(x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Return a BF16 tensor after DSpark's MXFP8 activation fake-quant."""
    y = x.contiguous().clone()
    _fake_fp8_e4m3_mxfp_inplace(y, block_size=block_size)
    return y


_DSPARK_FP8_BATCH_ADAPTIVE = os.getenv(
    "VLLM_DSPARK_FP8_BATCH_ADAPTIVE", "0"
).lower() not in ("0", "false", "no", "off")
_DSPARK_MARKOV_SCALE = float(os.getenv("VLLM_DSPARK_MARKOV_SCALE", "1.0"))


def _dspark_main_proj(first_layer: torch.nn.Module, main_hidden: torch.Tensor) -> torch.Tensor:
    """main_proj via the checkpoint's native FP8 weight when stashed at load
    (VLLM_DSPARK_FP8_MAIN_PROJ), else the original bf16 ReplicatedLinear."""
    proj = first_layer.main_proj
    dg_w = getattr(proj, "_fp8_weight", None)
    if dg_w is not None:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8_packed_for_deepgemm,
        )
        from vllm.utils.deep_gemm import fp8_gemm_nt

        flat = main_hidden.reshape(-1, main_hidden.shape[-1])
        xq, xs = per_token_group_quant_fp8_packed_for_deepgemm(flat, 128)
        out = torch.empty(
            flat.shape[0], dg_w.shape[0], dtype=torch.bfloat16, device=flat.device
        )
        fp8_gemm_nt((xq, xs), (dg_w, proj._fp8_scale), out)
        return out.view(*main_hidden.shape[:-1], -1)
    return _linear_output(proj(_maybe_fake_fp8_main_proj_input(main_hidden)))


def _maybe_fake_fp8_main_proj_input(x: torch.Tensor) -> torch.Tensor:
    if os.getenv("VLLM_DSPARK_FAKE_FP8_MAIN_PROJ_INPUT") == "1":
        return _fake_fp8_e4m3_mxfp(x)
    return x


def _apply_dspark_kv_qat_(kv: torch.Tensor, rope_dim: int) -> None:
    non_rope = kv[..., :-rope_dim] if rope_dim > 0 else kv
    _fake_fp8_e4m3_mxfp_inplace(non_rope, block_size=64)


def _apply_dspark_rope_hf(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """Apply DSpark HF-style RoPE to the trailing RoPE lanes."""
    if rope_dim <= 0:
        return x
    out = x.clone()
    rope = x[..., -rope_dim:].float()
    cos_sin = cos_sin_cache.index_select(0, positions.reshape(-1)).float()
    cos, sin = cos_sin.chunk(2, dim=-1)
    rope_pairs = rope.view(*rope.shape[:-1], -1, 2)
    even = rope_pairs[..., 0]
    odd = rope_pairs[..., 1]
    rotated = torch.stack(
        (
            even * cos[:, None, :] - odd * sin[:, None, :],
            odd * cos[:, None, :] + even * sin[:, None, :],
        ),
        dim=-1,
    ).flatten(-2)
    out[..., -rope_dim:] = rotated.to(out.dtype)
    return out


def _apply_dspark_inverse_rope_hf(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """HF-style inverse RoPE on the trailing DSpark RoPE lanes.

    This mirrors the public DSpark reference's
    ``apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)`` and exists as
    an opt-in diagnostic to isolate fused inverse-RoPE numerics from the rest of
    the DSpark draft path.
    """
    if rope_dim <= 0:
        return x.to(torch.bfloat16)
    out = x.to(torch.bfloat16).clone()
    rope = x[..., -rope_dim:].float()
    cos_sin = cos_sin_cache.index_select(0, positions.reshape(-1)).float()
    cos, sin = cos_sin.chunk(2, dim=-1)
    rope_pairs = rope.view(*rope.shape[:-1], -1, 2)
    even = rope_pairs[..., 0]
    odd = rope_pairs[..., 1]
    rotated = torch.stack(
        (
            even * cos[:, None, :] + odd * sin[:, None, :],
            odd * cos[:, None, :] - even * sin[:, None, :],
        ),
        dim=-1,
    ).flatten(-2)
    out[..., -rope_dim:] = rotated.to(torch.bfloat16)
    return out


def _linear_output(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _hc_head_reference(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    shape = x.shape
    dtype = x.dtype
    x_float = x.flatten(1).float()
    rsqrt = torch.rsqrt(x_float.square().mean(-1, keepdim=True) + norm_eps)
    mixes = torch.nn.functional.linear(x_float, hc_fn.float()) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale.float() + hc_base.float()) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x_float.view(shape), dim=1)
    return y.to(dtype)


def _dequantize_e4m3_e8m0_block_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize DSpark's standalone FP8 block-scaled linears.

    The public DSpark inference path dispatches these weights through its own
    FP8 GEMM.  vLLM's DeepGEMM block-FP8 path currently illegal-accesses on the
    DSpark main projection shape, so keep this draft-only projection as a plain
    BF16 linear until a native safe kernel is wired in.
    """
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            "DSpark FP8 block dequant expects 2D weight and scale tensors, "
            f"got weight={tuple(weight.shape)} scale={tuple(scale.shape)}."
        )
    out_features, in_features = weight.shape
    expected_scale_shape = (
        (out_features + block_size - 1) // block_size,
        (in_features + block_size - 1) // block_size,
    )
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            "DSpark FP8 block scale shape mismatch: "
            f"expected {expected_scale_shape}, got {tuple(scale.shape)}."
        )
    expanded_scale = (
        scale.to(torch.float32)
        .repeat_interleave(block_size, dim=0)
        .repeat_interleave(block_size, dim=1)
    )
    expanded_scale = expanded_scale[:out_features, :in_features]
    return (weight.to(torch.float32) * expanded_scale).to(out_dtype)


def _fp8_activation_quant_dequant_reference(
    x: torch.Tensor,
    *,
    block_size: int = 128,
) -> torch.Tensor:
    """HF DSpark-style MXFP8 activation quantization, then dequantization."""
    if x.shape[-1] % block_size != 0:
        raise ValueError(
            "DSpark FP8 activation reference requires block-aligned K, got "
            f"{x.shape[-1]} with block={block_size}."
        )
    original_shape = x.shape
    xf = x.reshape(-1, x.shape[-1]).float()
    xb = xf.view(xf.shape[0], xf.shape[1] // block_size, block_size)
    amax = xb.abs().amax(dim=-1).clamp_min_(1e-4)
    scale = torch.pow(
        torch.full_like(amax, 2.0),
        torch.ceil(torch.log2(amax / 448.0)),
    )
    q = (xb / scale.unsqueeze(-1)).clamp_(-448.0, 448.0)
    dq = q.to(torch.float8_e4m3fn).float() * scale.unsqueeze(-1)
    return dq.view(original_shape)


def _fp8_block_linear_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Reference DSpark FP8 block linear matching HF's fp8_gemm contract."""
    weight_bf16 = _dequantize_e4m3_e8m0_block_weight(
        weight,
        scale,
        out_dtype=torch.bfloat16,
    )
    x_dq = _fp8_activation_quant_dequant_reference(x)
    out = x_dq.reshape(-1, x.shape[-1]).float().matmul(weight_bf16.float().t())
    return out.to(torch.bfloat16).view(*x.shape[:-1], weight.shape[0])


def _read_dspark_num_layers(model_path: str, default: int) -> int:
    """Infer DSpark stage count from local checkpoint metadata.

    The public DSpark top-level config currently keeps
    ``num_nextn_predict_layers=1`` even though the attached DSpark module has
    three ``mtp.N`` stages.  The HF inference directory carries
    ``n_mtp_layers``; if unavailable, fall back to the safetensors index.
    """
    inference_config = os.path.join(model_path, "inference", "config.json")
    if os.path.exists(inference_config):
        with open(inference_config, encoding="utf-8") as f:
            n_layers = json.load(f).get("n_mtp_layers")
        if n_layers:
            return int(n_layers)

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
        stage_ids: set[int] = set()
        for name in weight_map:
            parts = name.split(".")
            if len(parts) > 2 and parts[0] == "mtp":
                with contextlib.suppress(ValueError):
                    stage_ids.add(int(parts[1]))
        if stage_ids:
            return max(stage_ids) + 1
    return int(default)


def _iter_mtp_safetensors(model_path: str) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield only ``mtp.*`` tensors from a local safetensors checkpoint."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            "DSpark loader currently requires a local safetensors index at "
            f"{index_path!r}."
        )
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    files = sorted(
        {filename for name, filename in weight_map.items() if name.startswith("mtp.")}
    )
    if not files:
        raise ValueError(f"No mtp.* DSpark weights found in {index_path!r}.")

    for filename in files:
        path = os.path.join(model_path, filename)
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():  # noqa: SIM118
                if name.startswith("mtp."):
                    yield name, f.get_tensor(name)


def _read_bool_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


# STEP 3: fused DSpark Markov-head greedy argmax.
# These mirror the reference dspark_kernels.py / dspark.py: compute the greedy
# top-1 of (base_logits + low-rank Markov bias) WITHOUT materializing the full
# Markov-logit vector, and do the global top-1 across tensor parallel ranks
# from per-rank local candidates. All of this is opt-in and only used on the
# greedy draft path (see forward_head); the default materialized path is
# untouched.
_DSPARK_MARKOV_NEG_INF = -3.4028234663852886e38
_DSPARK_MARKOV_V_BLOCK = 256
_DSPARK_MARKOV_R_BLOCK = 32


@triton.jit
def _dspark_markov_block_argmax_kernel(
    base_logits_ptr,
    markov_embed_ptr,
    markov_w2_ptr,
    block_vals_ptr,
    block_indices_ptr,
    batch_size,
    local_vocab_size,
    markov_rank: tl.constexpr,
    num_pad,
    base_stride_b,
    base_stride_v,
    embed_stride_b,
    embed_stride_r,
    w2_stride_v,
    w2_stride_r,
    out_stride_b,
    out_stride_block,
    V_BLOCK: tl.constexpr,
    R_BLOCK: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    pid_b = tl.program_id(0).to(tl.int64)
    pid_block = tl.program_id(1).to(tl.int64)

    offs_v = pid_block * V_BLOCK + tl.arange(0, V_BLOCK)
    valid_vocab = offs_v < local_vocab_size
    padded_vocab = offs_v >= (local_vocab_size - num_pad)
    valid_vocab = valid_vocab & ~padded_vocab

    acc = tl.zeros((V_BLOCK,), dtype=tl.float32)
    for r_start in tl.static_range(0, markov_rank, R_BLOCK):
        offs_r = r_start + tl.arange(0, R_BLOCK)
        embed = tl.load(
            markov_embed_ptr + pid_b * embed_stride_b + offs_r * embed_stride_r,
            mask=offs_r < markov_rank,
            other=0.0,
        ).to(tl.float32)
        w2 = tl.load(
            markov_w2_ptr
            + offs_v[None, :] * w2_stride_v
            + offs_r[:, None] * w2_stride_r,
            mask=(offs_v[None, :] < local_vocab_size) & (offs_r[:, None] < markov_rank),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w2 * embed[:, None], axis=0)

    base = tl.load(
        base_logits_ptr + pid_b * base_stride_b + offs_v * base_stride_v,
        mask=offs_v < local_vocab_size,
        other=NEG_INF,
    ).to(tl.float32)
    scores = tl.where(valid_vocab, base + acc, NEG_INF)
    max_val = tl.max(scores, axis=0)
    local_idx = tl.argmax(scores, axis=0)
    token_idx = pid_block * V_BLOCK + local_idx

    tl.store(
        block_vals_ptr + pid_b * out_stride_b + pid_block * out_stride_block,
        max_val,
        mask=pid_b < batch_size,
    )
    tl.store(
        block_indices_ptr + pid_b * out_stride_b + pid_block * out_stride_block,
        token_idx,
        mask=pid_b < batch_size,
    )


def dspark_markov_argmax_torch(
    base_logits: torch.Tensor,
    markov_embed: torch.Tensor,
    markov_w2_weight: torch.Tensor,
    *,
    num_pad: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference local top-1 for ``base_logits + W1[token] @ W2``."""

    scores = base_logits.float() + torch.matmul(
        markov_embed.float(), markov_w2_weight.float().t()
    )
    if num_pad > 0:
        scores[..., -num_pad:] = -float("inf")
    local_max_vals, local_max_indices = scores.max(dim=-1)
    return local_max_vals, local_max_indices.to(torch.long)


def dspark_markov_argmax(
    base_logits: torch.Tensor,
    markov_embed: torch.Tensor,
    markov_w2_weight: torch.Tensor,
    *,
    num_pad: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused local top-1 for the DSpark Markov-head greedy step.

    Computes only the local maximum of the Markov-corrected logits instead of
    materializing the full Markov-logit vector. The caller still performs the
    tensor-parallel global top-1 reduction. Falls back to the torch reference
    when Triton/CUDA is unavailable or the shapes are unsupported.
    """

    if (
        not base_logits.is_cuda
        or not markov_embed.is_cuda
        or not markov_w2_weight.is_cuda
        or not HAS_TRITON
    ):
        return dspark_markov_argmax_torch(
            base_logits, markov_embed, markov_w2_weight, num_pad=num_pad
        )

    if base_logits.ndim != 2 or markov_embed.ndim != 2 or markov_w2_weight.ndim != 2:
        return dspark_markov_argmax_torch(
            base_logits, markov_embed, markov_w2_weight, num_pad=num_pad
        )

    batch_size, local_vocab_size = base_logits.shape
    markov_rank = markov_embed.shape[-1]
    if markov_w2_weight.shape != (local_vocab_size, markov_rank):
        return dspark_markov_argmax_torch(
            base_logits, markov_embed, markov_w2_weight, num_pad=num_pad
        )
    if markov_rank % _DSPARK_MARKOV_R_BLOCK != 0:
        return dspark_markov_argmax_torch(
            base_logits, markov_embed, markov_w2_weight, num_pad=num_pad
        )

    num_blocks = triton.cdiv(local_vocab_size, _DSPARK_MARKOV_V_BLOCK)
    block_vals = torch.empty(
        (batch_size, num_blocks),
        device=base_logits.device,
        dtype=torch.float32,
    )
    block_indices = torch.empty(
        (batch_size, num_blocks),
        device=base_logits.device,
        dtype=torch.int64,
    )
    _dspark_markov_block_argmax_kernel[(batch_size, num_blocks)](
        base_logits,
        markov_embed,
        markov_w2_weight,
        block_vals,
        block_indices,
        batch_size,
        local_vocab_size,
        markov_rank,
        int(num_pad),
        base_logits.stride(0),
        base_logits.stride(1),
        markov_embed.stride(0),
        markov_embed.stride(1),
        markov_w2_weight.stride(0),
        markov_w2_weight.stride(1),
        block_vals.stride(0),
        block_vals.stride(1),
        V_BLOCK=_DSPARK_MARKOV_V_BLOCK,
        R_BLOCK=_DSPARK_MARKOV_R_BLOCK,
        NEG_INF=_DSPARK_MARKOV_NEG_INF,
        num_warps=8,
        num_stages=4,
    )
    max_block = block_vals.argmax(dim=-1, keepdim=True)
    local_max_vals = block_vals.gather(dim=-1, index=max_block).squeeze(-1)
    local_max_indices = block_indices.gather(dim=-1, index=max_block).squeeze(-1)
    return local_max_vals, local_max_indices


def _vocab_parallel_argmax_from_local(
    local_max_vals: torch.Tensor,
    global_indices: torch.Tensor,
) -> torch.Tensor:
    """Return global token ids from per-rank local top-1 candidates."""

    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return global_indices.to(torch.long)

    local_pair = torch.stack(
        [local_max_vals.float(), global_indices.float()],
        dim=-1,
    )
    gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
    gathered = gathered.view(local_max_vals.shape[0], tp_size, 2)
    max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
    top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
    return top_tokens.squeeze(-1).to(torch.long)


def _vocab_parallel_markov_argmax(
    base_logits: torch.Tensor,
    markov_embed: torch.Tensor,
    markov_w2: ParallelLMHead,
    lm_head: VocabParallelEmbedding,
) -> torch.Tensor:
    """Return global greedy ids for base logits plus DSpark Markov bias."""

    num_pad = lm_head.shard_indices.num_org_vocab_padding
    local_max_vals, local_max_indices = dspark_markov_argmax(
        base_logits,
        markov_embed,
        markov_w2.weight,
        num_pad=num_pad,
    )
    global_indices = local_max_indices + lm_head.shard_indices.org_vocab_start_index
    return _vocab_parallel_argmax_from_local(local_max_vals, global_indices)


class DSparkMarkovHead(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        rank = int(getattr(config, "dspark_markov_rank", 256))
        # STEP 3: optional replicated Markov W1 (plain nn.Embedding, no
        # vocab-parallel all-reduce). Default keeps our VocabParallelEmbedding.
        self._replicate_w1 = _read_bool_env("VLLM_DSPARK_REPLICATE_MARKOV_W1")
        if self._replicate_w1:
            self.markov_w1 = nn.Embedding(
                config.vocab_size,
                rank,
                dtype=vllm_config.model_config.dtype,
            )
            self.markov_w1.weight.requires_grad_(False)
            logger.info_once(
                "DSpark replicated Markov W1 enabled (plain nn.Embedding, no "
                "vocab-parallel embedding all-reduce)."
            )
        else:
            self.markov_w1 = VocabParallelEmbedding(
                config.vocab_size,
                rank,
                prefix=f"{prefix}.markov_w1",
            )
        # STEP 5: optional BF16 Markov W2 - halves the per-position GEMV
        # weight traffic (66->33MB/rank); max bias error ~0.01 on a +-3 bias.
        self._w2_bf16 = _read_bool_env("VLLM_DSPARK_MARKOV_W2_BF16", "1")
        if self._w2_bf16:
            logger.info_once("DSpark Markov W2 stored in BF16.")
        # STEP 5b: replicated Markov W2 - full-vocab bias computed locally on
        # every rank (identical seeded gumbel => identical draft tokens), so
        # the 5 sequential per-position logits all-gathers disappear.
        self._replicate_w2 = _read_bool_env("VLLM_DSPARK_REPLICATE_MARKOV_W2", "1")
        if self._replicate_w2:
            self.markov_w2 = nn.Linear(
                rank,
                config.vocab_size,
                bias=False,
                dtype=torch.bfloat16 if self._w2_bf16 else torch.float32,
            )
            self.markov_w2.weight.requires_grad_(False)
            logger.info_once(
                "DSpark replicated Markov W2 (no per-position logits "
                "all-gather in the draft loop)."
            )
            # STEP 6: fp8 deepgemm for the per-position markov bias GEMV
            # (halves the 66MB/position replicated-W2 read). Lazily
            # quantized from the loaded weight on first forward.
            self._w2_fp8 = _read_bool_env("VLLM_DSPARK_MARKOV_W2_FP8", "1")
            if self._w2_fp8:
                logger.info_once("DSpark Markov W2 GEMV uses FP8 deepgemm.")
        else:
            self.markov_w2 = ParallelLMHead(
                config.vocab_size,
                rank,
                params_dtype=torch.bfloat16 if self._w2_bf16 else torch.float32,
                org_num_embeddings=config.vocab_size,
                prefix=f"{prefix}.markov_w2",
            )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeds = self.markov_w1(token_ids)
        flat = embeds.view(-1, embeds.shape[-1]).to(self.markov_w2.weight.dtype)
        if self._replicate_w2:
            if getattr(self, "_w2_fp8", False) and (
                not _DSPARK_FP8_BATCH_ADAPTIVE or flat.shape[0] > 1
            ):
                if not hasattr(self, "_w2_fp8_weight"):
                    from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: E501
                        deepgemm_post_process_fp8_weight_block,
                    )
                    from vllm.utils.deep_gemm import per_block_cast_to_fp8

                    with torch.no_grad():
                        wq0, ws0 = per_block_cast_to_fp8(
                            self.markov_w2.weight.detach().float(),
                            [128, 128],
                            use_ue8m0=True,
                        )
                        dg_w, dg_ws = deepgemm_post_process_fp8_weight_block(
                            wq0, ws0, (128, 128), use_e8m0=True
                        )
                    self._w2_fp8_weight = dg_w
                    self._w2_fp8_scale = dg_ws
                from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: E501
                    per_token_group_quant_fp8_packed_for_deepgemm,
                )
                from vllm.utils.deep_gemm import fp8_gemm_nt

                xq, xs = per_token_group_quant_fp8_packed_for_deepgemm(
                    flat.to(torch.bfloat16), 128
                )
                logits = torch.empty(
                    flat.shape[0],
                    self._w2_fp8_weight.shape[0],
                    dtype=torch.bfloat16,
                    device=flat.device,
                )
                fp8_gemm_nt(
                    (xq, xs), (self._w2_fp8_weight, self._w2_fp8_scale), logits
                )
            else:
                logits = F.linear(flat, self.markov_w2.weight)
        else:
            logits = self.logits_processor(self.markov_w2, flat)
        return logits.view(*embeds.shape[:-1], -1), embeds


class DSparkConfidenceHead(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        rank = int(getattr(config, "dspark_markov_rank", 256))
        self.proj = ReplicatedLinear(
            config.hidden_size + rank,
            1,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=f"{prefix}.proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        markov_embeds: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([hidden_states, markov_embeds], dim=-1)
        confidence = _linear_output(self.proj(x.float()))
        return confidence.squeeze(-1)


class DeepSeekV4DSparkLayer(DeepseekV4DecoderLayer):
    """One DSpark stage.

    It intentionally keeps the normal DeepSeek V4 decoder-layer parameter names
    for attention/MoE/HC weights, then adds the DSpark-only stage-0 and final
    stage heads with names matching the checkpoint.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        stage_id: int,
        num_dspark_layers: int,
        prefix: str,
        topk_indices_buffer: torch.Tensor,
        aux_stream_list: list[torch.cuda.Stream] | None,
    ) -> None:
        super().__init__(
            vllm_config,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )
        self.prefix = prefix
        config = vllm_config.model_config.hf_config
        self._dspark_use_reference_fp8_linears = bool(
            int(os.getenv("VLLM_DSPARK_REFERENCE_FP8_LINEARS", "0"))
        )
        self._dspark_use_separate_qkv = (
            bool(int(os.getenv("VLLM_DSPARK_SEPARATE_QKV", "0")))
            or self._dspark_use_reference_fp8_linears
        )
        # STEP 4c: KV-QAT fake-quant runs by default (current behavior).
        # VLLM_DSPARK_KV_QAT=0 disables it.
        self._dspark_kv_qat_enabled = (
            os.getenv("VLLM_DSPARK_KV_QAT", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        # STEP 4b: opt-in FP8 WO projection. Default (0) keeps the BF16 WO.
        self._dspark_fp8_wo = _read_bool_env("VLLM_DSPARK_FP8_WO")
        if self._dspark_use_separate_qkv:
            self.attn.wq_a = ReplicatedLinear(
                config.hidden_size,
                config.q_lora_rank,
                bias=False,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.attn.wq_a",
                return_bias=False,
            )
            self.attn.wkv = ReplicatedLinear(
                config.hidden_size,
                config.head_dim,
                bias=False,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.attn.wkv",
                return_bias=False,
            )
        if stage_id == 0:
            target_layer_ids = tuple(
                int(i) for i in getattr(config, "dspark_target_layer_ids", ())
            )
            if not target_layer_ids:
                raise ValueError("DSpark requires dspark_target_layer_ids.")
            self.main_proj = ReplicatedLinear(
                config.hidden_size * len(target_layer_ids),
                config.hidden_size,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.main_proj",
            )
            self.main_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        if stage_id == num_dspark_layers - 1:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.markov_head = DSparkMarkovHead(
                vllm_config,
                prefix=f"{prefix}.markov_head",
            )
            self.confidence_head = DSparkConfidenceHead(
                vllm_config,
                prefix=f"{prefix}.confidence_head",
            )
            self.hc_head_fn = nn.Parameter(
                torch.empty(
                    config.hc_mult,
                    config.hc_mult * config.hidden_size,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.hc_head_base = nn.Parameter(
                torch.empty(config.hc_mult, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_head_scale = nn.Parameter(
                torch.empty(1, dtype=torch.float32),
                requires_grad=False,
            )

        self.dspark_block_size = int(getattr(config, "dspark_block_size", 0) or 0)
        self.dspark_noise_token_id = int(getattr(config, "dspark_noise_token_id", -1))
        if self.dspark_block_size <= 0:
            raise ValueError("DSpark requires dspark_block_size in the config.")
        if self.dspark_noise_token_id < 0:
            raise ValueError("DSpark requires dspark_noise_token_id in the config.")
        self.register_buffer(
            "dspark_kv_cache",
            torch.zeros(
                vllm_config.scheduler_config.max_num_seqs,
                config.sliding_window,
                config.head_dim,
                dtype=vllm_config.model_config.dtype,
            ),
            persistent=False,
        )
        # DSpark reuses the DeepSeek V4 attention linears/output projection, but
        # not the normal vLLM paged MLA/SWA cache path.  Remove those attention
        # layers from static_forward_context so DraftModelSpeculator.set_attn()
        # does not allocate/build unused draft attention metadata for them.
        static_forward_context = vllm_config.compilation_config.static_forward_context
        static_forward_context.pop(self.attn.prefix, None)
        static_forward_context.pop(self.attn.swa_cache_layer.prefix, None)

    def _compute_main_kv(
        self,
        main_x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = main_x.shape[0]
        if self._dspark_use_reference_fp8_linears:
            kv = self._apply_reference_fp8_linear("wkv", main_x.view(batch_size, -1))
        elif self._dspark_use_separate_qkv:
            kv = _linear_output(self.attn.wkv(main_x.view(batch_size, -1)))
        else:
            qr_kv = _linear_output(self.attn.fused_wqa_wkv(main_x.view(batch_size, -1)))
            _, kv = qr_kv.split([self.attn.q_lora_rank, self.attn.head_dim], dim=-1)
        if _dspark_dump_layer_parts_enabled():
            _maybe_save_tensor_debug(
                "dspark_main_kv_linear",
                {
                    "main_x": main_x.clone(),
                    "main_kv_linear": kv.clone(),
                },
                {"layer": self.prefix},
            )
        kv_normed = self.attn.kv_norm(kv)
        if _dspark_dump_layer_parts_enabled():
            _maybe_save_tensor_debug(
                "dspark_main_kv_norm",
                {
                    "main_kv_linear": kv.clone(),
                    "main_kv_norm": kv_normed.clone(),
                    "kv_norm_weight": self.attn.kv_norm.weight.clone(),
                },
                {"layer": self.prefix},
            )
        kv = _apply_dspark_rope_hf(
            kv_normed.view(batch_size, 1, self.attn.head_dim),
            positions.view(-1),
            self.attn.rotary_emb.cos_sin_cache,
            self.attn.rope_head_dim,
        )
        if self._dspark_kv_qat_enabled:
            _apply_dspark_kv_qat_(kv, self.attn.rope_head_dim)
        return kv.view(batch_size, self.attn.head_dim)

    def precompute_dspark_context_kv(
        self,
        main_x: torch.Tensor,
        positions: torch.Tensor,
        cache_start: int = 0,
    ) -> None:
        """Populate this stage's rolling target-KV window from target hidden states."""
        batch_size, seq_len, _ = main_x.shape
        window_size = self.attn.window_size
        cache = self.dspark_kv_cache[cache_start : cache_start + batch_size]
        main_kv = self._compute_main_kv(
            main_x.reshape(-1, main_x.shape[-1]), positions.reshape(-1)
        )
        main_kv = main_kv.view(batch_size, seq_len, self.attn.head_dim)
        tail_len = min(seq_len, window_size)
        tail_kv = main_kv[:, -tail_len:]
        tail_positions = positions[:, -tail_len:]
        slots = tail_positions.remainder(window_size).long()
        batch_indices = torch.arange(
            batch_size,
            dtype=torch.long,
            device=main_x.device,
        ).view(batch_size, 1)
        cache[batch_indices, slots] = tail_kv

    def precompute_dspark_context_kv_flat(
        self,
        main_x: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_rejected: torch.Tensor,
        cache_indices: torch.Tensor,
        num_reqs: int,
    ) -> None:
        """Populate this stage's rolling target-KV window from flat target rows."""
        if num_reqs <= 0:
            return
        main_kv = self._compute_main_kv(main_x, positions).contiguous()
        block_d = 64
        grid = (
            num_reqs,
            self.attn.window_size,
            triton.cdiv(self.attn.head_dim, block_d),
        )
        _store_dspark_context_kv_kernel[grid](
            self.dspark_kv_cache,
            main_kv,
            positions,
            query_start_loc,
            num_rejected,
            cache_indices,
            self.dspark_kv_cache.stride(0),
            self.dspark_kv_cache.stride(1),
            self.dspark_kv_cache.stride(2),
            main_kv.stride(0),
            main_kv.stride(1),
            HEAD_DIM=self.attn.head_dim,
            WINDOW_SIZE=self.attn.window_size,
            BLOCK_D=block_d,
        )

    def cache_dspark_wo_a_bf16(self) -> None:
        """Preserve DSpark's HF-style BF16 WO-A before FP8 kernel repacking.

        DSpark attention uses the public reference path:
        inverse-RoPE -> BF16 WO-A einsum -> WO-B.  The target DeepSeek V4
        linears are repacked by process_weights_after_loading for DeepGEMM, so
        after that point wo_a.weight_scale_inv no longer has checkpoint block
        layout. Cache the canonical BF16 WO-A while the loaded params still
        match the checkpoint layout.
        """
        if hasattr(self.attn.wo_a, "weight_scale_inv"):
            flat_weight = self.attn.wo_a.weight.reshape(
                self.attn.n_local_groups * self.attn.o_lora_rank,
                -1,
            )
            scale = _decode_e8m0_scales(self.attn.wo_a.weight_scale_inv).reshape(
                -1, self.attn.wo_a.weight_scale_inv.shape[-1]
            )
            cached = flat_weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).to(
                torch.float32
            ) * scale[:, None, :, None].to(torch.float32)
            cached = (
                cached.flatten(2, 3)
                .flatten(0, 1)
                .to(torch.bfloat16)
                .view(
                    self.attn.n_local_groups,
                    self.attn.o_lora_rank,
                    flat_weight.shape[-1],
                )
            )
        else:
            cached = self.attn.wo_a.weight.view(
                self.attn.n_local_groups,
                self.attn.o_lora_rank,
                -1,
            ).to(torch.bfloat16)

        if "_dsv4_wo_a_bf16" in self.attn.wo_a._buffers:
            self.attn.wo_a._buffers["_dsv4_wo_a_bf16"] = cached
        else:
            self.attn.wo_a.register_buffer(
                "_dsv4_wo_a_bf16",
                cached,
                persistent=False,
            )

    def cache_dspark_reference_fp8_linears(self) -> None:
        if not self._dspark_use_reference_fp8_linears:
            return
        for attr in ("wq_a", "wkv", "wq_b"):
            layer = getattr(self.attn, attr)
            if not hasattr(layer, "weight_scale_inv"):
                raise ValueError(
                    f"DSpark reference FP8 linear {attr} has no block scale."
                )
            weight = layer.weight.detach().clone()
            scale = layer.weight_scale_inv.detach().clone()
            for name, tensor in (
                (f"_dspark_{attr}_weight_fp8", weight),
                (f"_dspark_{attr}_scale_e8m0", scale),
            ):
                if name in self.attn._buffers:
                    self.attn._buffers[name] = tensor
                else:
                    self.attn.register_buffer(name, tensor, persistent=False)

    def _apply_reference_fp8_linear(
        self,
        attr: str,
        x: torch.Tensor,
    ) -> torch.Tensor:
        weight = getattr(self.attn, f"_dspark_{attr}_weight_fp8")
        scale = getattr(self.attn, f"_dspark_{attr}_scale_e8m0")
        return _fp8_block_linear_reference(x, weight, scale)

    def _dspark_output_projection(
        self,
        out: torch.Tensor,
        draft_positions: torch.Tensor,
        batch_size: int,
        block_size: int,
        hidden_size: int,
        start_pos: int | None,
    ) -> torch.Tensor:
        flat_positions = draft_positions.reshape(-1)
        if getattr(self.attn, "_use_b12x_wo", False):
            # B12X owns WO-A/WO-B through a fused helper and marks the generic
            # FP8 linears as unavailable.  DSpark attention layers are removed
            # from the no-compile context, so call the helper directly instead
            # of the target attention custom op that looks layers up by name.
            out_final = self.attn._apply_b12x_wo_projection(
                out,
                flat_positions,
                o_storage=out,
                o_storage_offset=out.storage_offset(),
                o_stride_0=out.stride(0),
                o_stride_1=out.stride(1),
                o_stride_2=out.stride(2),
            ).view(batch_size, block_size, hidden_size)
            if _dspark_dump_layer_parts_enabled():
                _maybe_save_tensor_debug(
                    "dspark_attention_final",
                    {"out": out_final},
                    {"layer": self.prefix, "start_pos": start_pos},
                )
            return out_final

        # STEP 4b: VLLM_DSPARK_FP8_WO opt-in. A faithful FP8 WO einsum (as in
        # the reference dspark.py) is not reconstructable for our draft: our
        # wo_a is repacked for DeepGEMM by process_weights_after_loading (only a
        # checkpoint-layout BF16 wo_a is cached beforehand), so wo_a.weight /
        # weight_scale_inv no longer match the block-FP8 layout the reference
        # FP8 o_proj einsum expects. Running it would illegal-access or corrupt
        # draft numerics (the exact reason the BF16 WO is the default). So this
        # flag warns once and falls back to the current BF16 WO path.
        if self._dspark_fp8_wo and not getattr(
            self, "_dspark_fp8_wo_warned", False
        ):
            logger.warning(
                "VLLM_DSPARK_FP8_WO=1 requested, but a native FP8 WO "
                "projection is unavailable for the DSpark draft in this build "
                "(wo_a is repacked for DeepGEMM after loading; only a BF16 "
                "wo_a is cached). Falling back to the BF16 WO path (current "
                "default behavior)."
            )
            self._dspark_fp8_wo_warned = True

        # DSpark's public reference uses a BF16 inverse-RoPE + WO-A/WO-B path
        # here.  The target DeepSeek V4 CUDA path uses a fused FP8 _o_proj that
        # assumes target-layer RoPE/cache invariants and corrupts DSpark draft
        # numerics for compress_ratio=0.
        if os.getenv("VLLM_DSPARK_REFERENCE_WO") == "1":
            inverse_rope = (
                _apply_dspark_inverse_rope_hf
                if os.getenv("VLLM_DSPARK_HF_INVERSE_ROPE") == "1"
                else _fused_inverse_rope_gptj
            )
            out_ref = inverse_rope(
                out,
                flat_positions,
                self.attn.rotary_emb.cos_sin_cache,
                self.attn.rope_head_dim,
            )
            out_ref = out_ref.view(
                batch_size * block_size,
                self.attn.n_local_groups,
                -1,
            )
            if not getattr(self, "_dspark_reference_wo_shape_logged", False):
                logger.warning(
                    "DSpark reference WO shapes: weight=%s scale=%s "
                    "groups=%s rank=%s out_width=%s",
                    tuple(self.attn.wo_a.weight.shape),
                    tuple(getattr(self.attn.wo_a, "weight_scale_inv", ()).shape),
                    self.attn.n_local_groups,
                    self.attn.o_lora_rank,
                    out_ref.shape[-1],
                )
                self._dspark_reference_wo_shape_logged = True
            wo_a = _get_cached_wo_a_bf16(
                self.attn.wo_a,
                self.attn.n_local_groups,
                self.attn.o_lora_rank,
                out_ref.shape[-1],
            )
            z = torch.einsum("tgd,grd->tgr", out_ref, wo_a)
            if _dspark_dump_layer_parts_enabled():
                _maybe_save_tensor_debug(
                    "dspark_attention_reference_wo",
                    {
                        "out_ref": out_ref,
                        "wo_a": wo_a,
                        "z": z,
                    },
                    {"layer": self.prefix, "start_pos": start_pos},
                )
        else:
            z = rocm_inv_rope_einsum(
                self.attn.rotary_emb,
                out,
                flat_positions,
                self.attn.rope_head_dim,
                self.attn.n_local_groups,
                self.attn.o_lora_rank,
                self.attn.wo_a,
            )
        out_final = self.attn.wo_b(z.flatten(1)).view(
            batch_size,
            block_size,
            hidden_size,
        )
        if _dspark_dump_layer_parts_enabled():
            _maybe_save_tensor_debug(
                "dspark_attention_final",
                {"out": out_final},
                {"layer": self.prefix, "start_pos": start_pos},
            )
        return out_final

    def _dspark_attention(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        main_x: torch.Tensor,
        cache_indices: torch.Tensor,
        start_pos: int | None = None,
    ) -> torch.Tensor:
        batch_size, block_size, hidden_size = x.shape
        if block_size != self.dspark_block_size:
            raise ValueError(
                "DSpark draft block size mismatch: "
                f"expected {self.dspark_block_size}, got {block_size}."
            )
        if main_x.shape[1] != 1:
            raise ValueError(
                "DSpark decode currently expects one target hidden state per request."
            )

        main_positions = positions.view(batch_size)
        main_kv = self._compute_main_kv(main_x[:, 0], main_positions)

        flat_x = x.reshape(batch_size * block_size, hidden_size)
        if self._dspark_use_reference_fp8_linears:
            qr = self._apply_reference_fp8_linear("wq_a", flat_x)
            kv = self._apply_reference_fp8_linear("wkv", flat_x)
        elif self._dspark_use_separate_qkv:
            qr = _linear_output(self.attn.wq_a(flat_x))
            kv = _linear_output(self.attn.wkv(flat_x))
        else:
            qr_kv = _linear_output(self.attn.fused_wqa_wkv(flat_x))
            qr, kv = qr_kv.split([self.attn.q_lora_rank, self.attn.head_dim], dim=-1)
        q_normed = self.attn.q_norm(qr)
        if self._dspark_use_reference_fp8_linears:
            q = self._apply_reference_fp8_linear("wq_b", q_normed)
        else:
            q = _linear_output(self.attn.wq_b(q_normed))
        kv_normed = self.attn.kv_norm(kv)
        if _dspark_dump_layer_parts_enabled():
            _maybe_save_tensor_debug(
                "dspark_attention_linear_outputs",
                {
                    "flat_x": flat_x.clone(),
                    "qr_linear": qr.clone(),
                    "q_norm": q_normed.clone(),
                    "kv_linear": kv.clone(),
                    "kv_norm": kv_normed.clone(),
                    "q_after_wqb_linear": q.clone(),
                    "q_norm_weight": self.attn.q_norm.weight.clone(),
                    "kv_norm_weight": self.attn.kv_norm.weight.clone(),
                },
                {"layer": self.prefix, "start_pos": start_pos},
            )
        q = q.view(batch_size * block_size, self.attn.n_local_heads, self.attn.head_dim)
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.attn.eps)
        q_after_scale_debug = q.clone() if _dspark_dump_layer_parts_enabled() else None
        kv = kv_normed.view(batch_size * block_size, 1, self.attn.head_dim)

        draft_positions = positions.view(batch_size, 1) + torch.arange(
            1,
            block_size + 1,
            dtype=positions.dtype,
            device=positions.device,
        ).view(1, block_size)
        q = _apply_dspark_rope_hf(
            q,
            draft_positions.reshape(-1),
            self.attn.rotary_emb.cos_sin_cache,
            self.attn.rope_head_dim,
        )
        kv = _apply_dspark_rope_hf(
            kv,
            draft_positions.reshape(-1),
            self.attn.rotary_emb.cos_sin_cache,
            self.attn.rope_head_dim,
        )
        q_after_rope_debug = q.clone() if _dspark_dump_layer_parts_enabled() else None
        kv_after_rope_debug = kv.clone() if _dspark_dump_layer_parts_enabled() else None
        if self._dspark_kv_qat_enabled:
            _apply_dspark_kv_qat_(kv, self.attn.rope_head_dim)
        if _dspark_dump_layer_parts_enabled():
            _maybe_save_tensor_debug(
                "dspark_attention_post_rope",
                {
                    "q_after_scale": q_after_scale_debug,
                    "q_after_rope": q_after_rope_debug,
                    "kv_after_rope": kv_after_rope_debug,
                    "kv_after_qat": kv.clone(),
                },
                {"layer": self.prefix, "start_pos": start_pos},
            )
        q = q.view(batch_size, block_size, self.attn.n_local_heads, self.attn.head_dim)
        kv = kv.view(batch_size, block_size, self.attn.head_dim)

        window_size = self.attn.window_size
        cache_rows = cache_indices.view(batch_size).long()
        cache_slots = positions.view(batch_size).remainder(window_size).long()
        self.dspark_kv_cache[cache_rows, cache_slots].copy_(main_kv)
        cache_window = self.dspark_kv_cache[cache_rows]
        all_kv = torch.cat([cache_window, kv], dim=1)
        topk_idxs = _build_dspark_topk_idxs(
            window_size=window_size,
            batch_size=batch_size,
            block_size=block_size,
            positions=positions.view(batch_size),
            device=x.device,
        )
        if _dspark_dump_layer_parts_enabled():
            _maybe_save_tensor_debug(
                "dspark_attention_inputs",
                {
                    "positions": positions,
                    "main_x": main_x,
                    "main_kv": main_kv,
                    "q": q,
                    "kv": kv,
                    "cache_window": cache_window,
                    "topk_idxs": topk_idxs,
                },
                {"layer": self.prefix, "start_pos": start_pos},
            )
        sparse_attn_fn = (
            dspark_sparse_attn_reference
            if os.getenv("VLLM_DSPARK_REFERENCE_ATTN") == "1"
            else dspark_sparse_attn
        )
        out = sparse_attn_fn(
            q,
            all_kv,
            self.attn.attn_sink[: self.attn.n_local_heads],
            topk_idxs,
            self.attn.scale,
        )
        out = out.reshape(
            batch_size * block_size,
            self.attn.n_local_heads,
            self.attn.head_dim,
        )
        if _dspark_dump_layer_parts_enabled():
            _maybe_save_tensor_debug(
                "dspark_attention_sparse_out",
                {"out": out},
                {"layer": self.prefix, "start_pos": start_pos},
            )
        return self._dspark_output_projection(
            out,
            draft_positions,
            batch_size,
            block_size,
            hidden_size,
            start_pos,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        main_x: torch.Tensor,
        cache_indices: torch.Tensor,
        start_pos: int | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, block_size = input_ids.shape
        hidden_size = self.hidden_size
        if residual is None:
            if x.ndim != 4:
                raise ValueError(
                    "First DSpark stage expects [batch, block, hc, hidden] input."
                )
            hc_mult = x.shape[2]
            x_current = x.reshape(batch_size * block_size, hc_mult, hidden_size)
        else:
            if x.ndim != 3:
                raise ValueError(
                    "Subsequent DSpark stages expect [batch, block, hidden] input."
                )
            x_current = x.reshape(batch_size * block_size, hidden_size)
        input_ids_flat = input_ids.reshape(-1)

        attn_norm_weight = self.attn_norm.weight.data
        attn_norm_eps = self.attn_norm.variance_epsilon
        if residual is None:
            residual = x_current
            x_attn, post_mix, res_mix = self.hc_pre(
                residual,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=attn_norm_weight,
                norm_eps=attn_norm_eps,
            )
        else:
            assert post_mix is not None
            assert res_mix is not None
            residual, post_mix, res_mix, x_attn = self.hc_post_pre(
                x_current,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=attn_norm_weight,
                norm_eps=attn_norm_eps,
            )

        x_attn = self._dspark_attention(
            positions,
            x_attn.view(batch_size, block_size, hidden_size),
            main_x,
            cache_indices,
            start_pos,
        ).reshape(batch_size * block_size, hidden_size)

        ffn_norm_weight = self.ffn_norm.weight.data
        ffn_norm_eps = self.ffn_norm.variance_epsilon
        residual, post_mix, res_mix, x_ffn = self.hc_post_pre(
            x_attn,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            norm_weight=ffn_norm_weight,
            norm_eps=ffn_norm_eps,
            hc_fn_bf16=self.hc_ffn_fn_bf16,
        )
        x_ffn = self.ffn(x_ffn, input_ids_flat)
        return (
            x_ffn.view(batch_size, block_size, hidden_size),
            residual,
            post_mix,
            res_mix,
        )

    def forward_reference_hc(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        main_x: torch.Tensor,
        cache_indices: torch.Tensor,
        start_pos: int | None = None,
    ) -> torch.Tensor:
        """DSpark-only HF-style HC reference path.

        This intentionally materializes the full HC state after each post-mix
        instead of using the production fused post-pre chaining. It is guarded
        by an env flag and exists to isolate DSpark numerical mismatches.
        """
        batch_size, block_size = input_ids.shape
        hidden_size = self.hidden_size
        if x.ndim != 4:
            raise ValueError("DSpark reference HC expects [batch, block, hc, hidden].")
        x_current = x.reshape(batch_size * block_size, x.shape[2], hidden_size)
        input_ids_flat = input_ids.reshape(-1)

        post_mix, res_mix, x_attn = mhc_pre_torch(
            x_current,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
        )
        if _dspark_dump_layer_parts_enabled():
            _maybe_save_tensor_debug(
                "dspark_layer_attn_pre",
                {
                    "x_current": x_current,
                    "post_mix": post_mix,
                    "res_mix": res_mix,
                    "x_attn_pre_norm": x_attn,
                },
                {"layer": self.prefix, "start_pos": start_pos},
            )
        x_attn = self.attn_norm(x_attn)
        if _dspark_dump_layer_parts_enabled():
            _maybe_save_tensor_debug(
                "dspark_layer_attn_norm",
                {"x_attn": x_attn},
                {"layer": self.prefix, "start_pos": start_pos},
            )
        x_attn = self._dspark_attention(
            positions,
            x_attn.view(batch_size, block_size, hidden_size),
            main_x,
            cache_indices,
            start_pos,
        ).reshape(batch_size * block_size, hidden_size)
        if _dspark_dump_layer_parts_enabled():
            _maybe_save_tensor_debug(
                "dspark_layer_attn_out",
                {"x_attn": x_attn},
                {"layer": self.prefix, "start_pos": start_pos},
            )
        x_current = mhc_post_torch(x_attn, x_current, post_mix, res_mix)
        if _dspark_dump_layer_parts_enabled():
            _maybe_save_tensor_debug(
                "dspark_layer_after_attn_post",
                {"x_current": x_current},
                {"layer": self.prefix, "start_pos": start_pos},
            )

        post_mix, res_mix, x_ffn = mhc_pre_torch(
            x_current,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
        )
        if _dspark_dump_layer_parts_enabled():
            _maybe_save_tensor_debug(
                "dspark_layer_ffn_pre",
                {
                    "post_mix": post_mix,
                    "res_mix": res_mix,
                    "x_ffn_pre_norm": x_ffn,
                },
                {"layer": self.prefix, "start_pos": start_pos},
            )
        x_ffn = self.ffn_norm(x_ffn)
        if _dspark_dump_layer_parts_enabled():
            _maybe_save_tensor_debug(
                "dspark_layer_ffn_norm",
                {"x_ffn": x_ffn},
                {"layer": self.prefix, "start_pos": start_pos},
            )
        x_ffn = self.ffn(x_ffn, input_ids_flat)
        if _dspark_dump_layer_parts_enabled():
            _maybe_save_tensor_debug(
                "dspark_layer_ffn_out",
                {"x_ffn": x_ffn},
                {"layer": self.prefix, "start_pos": start_pos},
            )
        x_current = mhc_post_torch(x_ffn, x_current, post_mix, res_mix)
        if _dspark_dump_layer_parts_enabled():
            _maybe_save_tensor_debug(
                "dspark_layer_after_ffn_post",
                {"x_current": x_current},
                {"layer": self.prefix, "start_pos": start_pos},
            )
        return x_current.view(batch_size, block_size, x.shape[2], hidden_size)


class DeepSeekV4DSparkDraft(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model_path = vllm_config.model_config.model
        self.num_dspark_layers = _read_dspark_num_layers(
            self.model_path,
            getattr(self.config, "num_nextn_predict_layers", 1),
        )
        self.dspark_start_layer_idx = self.config.num_hidden_layers
        self.checkpoint_weight_name_prefixes = tuple(
            f"mtp.{idx}." for idx in range(self.num_dspark_layers)
        )

        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            self.config.index_topk,
            dtype=torch.int32,
        )
        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]

        self.layers = nn.ModuleDict()
        for idx in range(self.num_dspark_layers):
            layer_idx = self.dspark_start_layer_idx + idx
            layer_prefix = (
                f"{prefix}.layers.{layer_idx}" if prefix else f"layers.{layer_idx}"
            )
            self.layers[str(layer_idx)] = DeepSeekV4DSparkLayer(
                vllm_config,
                stage_id=idx,
                num_dspark_layers=self.num_dspark_layers,
                prefix=layer_prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            )
        self.embed_tokens: VocabParallelEmbedding | None = None
        self.lm_head: ParallelLMHead | None = None
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        # STEP 4a: greedy draft (argmax instead of gumbel). Default off.
        self._greedy_draft = _read_bool_env("VLLM_DSPARK_GREEDY_DRAFT")
        # STEP 3: fused Markov greedy argmax. Default off; only used when the
        # greedy draft path is also on (the fused argmax is greedy-only).
        self._fused_markov_argmax = _read_bool_env("VLLM_DSPARK_FUSED_MARKOV_ARGMAX")
        if self._greedy_draft:
            logger.info_once(
                "DSpark greedy draft enabled: draft block tokens use argmax "
                "instead of gumbel sampling."
            )
        if self._fused_markov_argmax and not self._greedy_draft:
            logger.warning(
                "VLLM_DSPARK_FUSED_MARKOV_ARGMAX=1 requires "
                "VLLM_DSPARK_GREEDY_DRAFT=1 (the fused argmax is only valid on "
                "the greedy draft path); the default materialized Markov path "
                "will be used while greedy draft is off."
            )
        elif self._fused_markov_argmax:
            logger.info_once(
                "DSpark fused Markov argmax enabled: greedy draft computes the "
                "base+Markov top-1 without materializing the full Markov-logit "
                "vector."
            )

    def attach_target_modules(
        self,
        embed_tokens: VocabParallelEmbedding,
        lm_head: ParallelLMHead,
    ) -> None:
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        if _read_bool_env("VLLM_DSPARK_FP8_LM_HEAD", "1"):
            # STEP 5: FP8 deepgemm copy of target lm_head for DRAFT logits
            # only - halves the 530MB/rank weight read per draft block.
            # Verification stays lossless (rejection sampling uses the
            # actual draft distribution q); only proposal quality shifts
            # marginally (top6 overlap vs bf16 ~93%).
            from vllm.model_executor.layers.quantization.utils.fp8_utils import (
                deepgemm_post_process_fp8_weight_block,
            )
            from vllm.utils.deep_gemm import per_block_cast_to_fp8

            with torch.no_grad():
                w = lm_head.weight.detach()
                chunks_q, chunks_s = [], []
                rows = w.shape[0]
                step = max(128, (rows // 8) // 128 * 128)
                for r0 in range(0, rows, step):
                    cq, cs = per_block_cast_to_fp8(
                        w[r0 : r0 + step].float(), [128, 128], use_ue8m0=True
                    )
                    chunks_q.append(cq)
                    chunks_s.append(cs)
                dg_w, dg_ws = deepgemm_post_process_fp8_weight_block(
                    torch.cat(chunks_q, dim=0),
                    torch.cat(chunks_s, dim=0),
                    (128, 128),
                    use_e8m0=True,
                )
            self.register_buffer("lm_head_fp8_weight", dg_w, persistent=False)
            self.register_buffer("lm_head_fp8_scale", dg_ws, persistent=False)
            logger.info_once(
                "DSpark draft logits use an FP8 deepgemm copy of target "
                "lm_head (draft-only)."
            )
        if os.getenv("VLLM_DSPARK_FP32_LM_HEAD") == "1":
            self.register_buffer(
                "lm_head_fp32_weight",
                lm_head.weight.detach().to(torch.float32).clone(),
                persistent=False,
            )
            logger.info_once(
                "DSpark draft logits use a local FP32 copy of target lm_head."
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.embed_tokens is None:
            raise RuntimeError("DSpark draft has no shared target embedding.")
        return self.embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            raise RuntimeError("DSpark draft has no shared target lm_head.")
        lm_head_fp8_weight = getattr(self, "lm_head_fp8_weight", None)
        flat_rows = hidden_states.numel() // hidden_states.shape[-1]
        block = int(getattr(self.config, "dspark_block_size", 5) or 5)
        if lm_head_fp8_weight is not None and (
            not _DSPARK_FP8_BATCH_ADAPTIVE or flat_rows > block
        ):
            from vllm.model_executor.layers.quantization.utils.fp8_utils import (
                per_token_group_quant_fp8_packed_for_deepgemm,
            )
            from vllm.utils.deep_gemm import fp8_gemm_nt

            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            xq, xs = per_token_group_quant_fp8_packed_for_deepgemm(flat, 128)
            local = torch.empty(
                flat.shape[0],
                lm_head_fp8_weight.shape[0],
                dtype=torch.bfloat16,
                device=flat.device,
            )
            fp8_gemm_nt(
                (xq, xs), (lm_head_fp8_weight, self.lm_head_fp8_scale), local
            )
            logits = self.logits_processor._gather_logits(local)
            if logits is not None:
                logits = logits[..., : self.config.vocab_size]
                logits = logits.view(*hidden_states.shape[:-1], -1)
            return logits
        lm_head_fp32_weight = getattr(self, "lm_head_fp32_weight", None)
        if lm_head_fp32_weight is not None:
            logits = F.linear(hidden_states.float(), lm_head_fp32_weight)
            logits = self.logits_processor._gather_logits(logits)
            if logits is not None:
                logits = logits[..., : self.config.vocab_size]
            return logits
        return self.logits_processor(
            self.lm_head,
            hidden_states.view(-1, hidden_states.shape[-1]),
        ).view(*hidden_states.shape[:-1], -1)

    def forward_embed(
        self,
        main_hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first_layer = self.layers[str(self.dspark_start_layer_idx)]
        assert isinstance(first_layer, DeepSeekV4DSparkLayer)
        main_x = _dspark_main_proj(first_layer, main_hidden)
        main_x = first_layer.main_norm(main_x)
        batch_size = input_ids.shape[0]
        draft_input_ids = input_ids.new_full(
            (batch_size, first_layer.dspark_block_size),
            first_layer.dspark_noise_token_id,
        )
        draft_input_ids[:, 0] = input_ids
        x = self.embed_input_ids(draft_input_ids.reshape(-1))
        x = x.view(batch_size, first_layer.dspark_block_size, self.config.hidden_size)
        x = x.unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)
        return x, main_x.unsqueeze(1), draft_input_ids

    def precompute_context_kv(
        self,
        main_hidden: torch.Tensor,
        positions: torch.Tensor,
        context_ranges: list[tuple[int, int]],
    ) -> None:
        """Populate DSpark rolling target-KV caches for scheduled target tokens."""
        if not context_ranges:
            return
        first_layer = self.layers[str(self.dspark_start_layer_idx)]
        assert isinstance(first_layer, DeepSeekV4DSparkLayer)
        main_x = _dspark_main_proj(first_layer, main_hidden)
        main_x = first_layer.main_norm(main_x)

        for req_idx, (start, end) in enumerate(context_ranges):
            if end <= start:
                continue
            req_main_x = main_x[start:end].unsqueeze(0)
            req_positions = positions[start:end].unsqueeze(0)
            for layer in self.layers.values():
                assert isinstance(layer, DeepSeekV4DSparkLayer)
                layer.precompute_dspark_context_kv(
                    req_main_x,
                    req_positions,
                    cache_start=req_idx,
                )

    def precompute_context_kv_flat(
        self,
        main_hidden: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_rejected: torch.Tensor,
        cache_indices: torch.Tensor,
        num_reqs: int,
    ) -> None:
        """Populate DSpark rolling target-KV caches from flat target rows."""
        if num_reqs <= 0 or main_hidden.numel() == 0:
            return
        first_layer = self.layers[str(self.dspark_start_layer_idx)]
        assert isinstance(first_layer, DeepSeekV4DSparkLayer)
        main_x = _dspark_main_proj(first_layer, main_hidden)
        main_x = first_layer.main_norm(main_x)

        for layer in self.layers.values():
            assert isinstance(layer, DeepSeekV4DSparkLayer)
            layer.precompute_dspark_context_kv_flat(
                main_x,
                positions,
                query_start_loc,
                num_rejected,
                cache_indices,
                num_reqs,
            )

    def reset_dspark_kv_cache(self) -> None:
        for layer in self.layers.values():
            assert isinstance(layer, DeepSeekV4DSparkLayer)
            layer.dspark_kv_cache.zero_()

    def forward_head(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        idx_mapping: torch.Tensor | None = None,
        temperature: torch.Tensor | None = None,
        seeds: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        draft_logits: torch.Tensor | None = None,
        draft_step_cols: torch.Tensor | None = None,
        active_num_reqs: torch.Tensor | None = None,
        use_fp64_gumbel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        final_layer = self.layers[
            str(self.dspark_start_layer_idx + self.num_dspark_layers - 1)
        ]
        assert isinstance(final_layer, DeepSeekV4DSparkLayer)
        batch_size, block_size, hc_mult, hidden_size = x.shape
        x_flat = x.reshape(batch_size * block_size, hc_mult, hidden_size)
        if os.getenv("VLLM_DSPARK_REFERENCE_HC_HEAD") == "1":
            hidden = _hc_head_reference(
                x_flat,
                final_layer.hc_head_fn,
                final_layer.hc_head_scale,
                final_layer.hc_head_base,
                final_layer.rms_norm_eps,
                final_layer.hc_eps,
            ).view(batch_size, block_size, hidden_size)
        else:
            hidden = hc_head_fused_kernel_tilelang(
                x_flat,
                final_layer.hc_head_fn,
                final_layer.hc_head_scale,
                final_layer.hc_head_base,
                final_layer.rms_norm_eps,
                final_layer.hc_eps,
            ).view(batch_size, block_size, hidden_size)
        logits_hidden = final_layer.norm(hidden)
        # STEP 3: fused Markov greedy argmax (opt-in, greedy-only). Compute the
        # base + low-rank Markov top-1 directly from LOCAL (vocab-parallel)
        # logits without materializing the gathered base logits or the full
        # Markov-logit vector. Only taken on the greedy draft path; falls back
        # to the materialized path below otherwise, and when the FP32 lm_head
        # override is active (which the fused local path cannot reproduce).
        if (
            self._greedy_draft
            and self._fused_markov_argmax
            and getattr(self, "lm_head_fp32_weight", None) is None
        ):
            assert self.lm_head is not None
            local_logits = self.lm_head.quant_method.apply(
                self.lm_head,
                logits_hidden.reshape(batch_size * block_size, hidden_size),
                bias=None,
            ).view(batch_size, block_size, -1)
            output_ids = input_ids.new_empty(batch_size, block_size + 1)
            output_ids[:, 0] = input_ids
            markov_head = final_layer.markov_head
            markov_embeds = []
            for idx in range(block_size):
                markov_embed = markov_head.markov_w1(output_ids[:, idx])
                markov_embeds.append(markov_embed)
                output_ids[:, idx + 1] = _vocab_parallel_markov_argmax(
                    local_logits[:, idx],
                    markov_embed,
                    markov_head.markov_w2,
                    self.lm_head,
                )
            confidence = final_layer.confidence_head(
                hidden,
                torch.stack(markov_embeds, dim=1),
            )
            return output_ids, logits_hidden.new_empty((0, 0, 0)), confidence
        logits = self.compute_logits(logits_hidden)
        if os.getenv("VLLM_DSPARK_TENSOR_DEBUG_HEAD_PARTS") == "1":
            _maybe_save_tensor_debug(
                "head_base_logits",
                {
                    "input_ids": input_ids,
                    "positions": positions,
                    "hidden": hidden,
                    "logits_hidden": logits_hidden,
                    "base_logits": logits,
                },
                {
                    "start_pos": (
                        int(positions[0].item())
                        if positions is not None and positions.numel()
                        else None
                    ),
                },
            )

        output_ids = input_ids.new_empty(batch_size, block_size + 1)
        output_ids[:, 0] = input_ids
        markov_embeds = []
        disable_markov_bias = os.getenv("VLLM_DSPARK_DISABLE_MARKOV_BIAS") == "1"
        for idx in range(block_size):
            logits_bias, markov_embed = final_layer.markov_head(output_ids[:, idx])
            if os.getenv("VLLM_DSPARK_TENSOR_DEBUG_HEAD_PARTS") == "1":
                _maybe_save_tensor_debug(
                    f"head_markov_{idx}",
                    {
                        "step_input_ids": output_ids[:, idx],
                        "logits_bias": logits_bias,
                        "base_logits_step": logits[:, idx],
                    },
                    {
                        "step": idx,
                        "start_pos": (
                            int(positions[0].item())
                            if positions is not None and positions.numel()
                            else None
                        ),
                    },
                )
            if not disable_markov_bias:
                if _DSPARK_MARKOV_SCALE != 1.0:
                    logits[:, idx].add_(logits_bias, alpha=_DSPARK_MARKOV_SCALE)
                else:
                    logits[:, idx].add_(logits_bias)
            markov_embeds.append(markov_embed)
            # STEP 4a: greedy draft uses argmax (skips gumbel) when enabled.
            if draft_logits is not None and not self._greedy_draft:
                assert idx_mapping is not None
                assert temperature is not None
                assert seeds is not None
                assert positions is not None
                assert draft_step_cols is not None
                output_ids[:, idx + 1] = gumbel_sample(
                    logits[:, idx],
                    idx_mapping,
                    temperature,
                    seeds,
                    positions + idx + 1,
                    apply_temperature=True,
                    output_processed_logits=draft_logits,
                    output_processed_logits_col=draft_step_cols[idx],
                    output_processed_logits_active_rows=active_num_reqs,
                    use_fp64=use_fp64_gumbel,
                )
            else:
                output_ids[:, idx + 1] = logits[:, idx].argmax(dim=-1)
        confidence = final_layer.confidence_head(
            hidden,
            torch.stack(markov_embeds, dim=1),
        )
        return output_ids, logits, confidence

    def forward_spec(
        self,
        input_ids: torch.Tensor,
        main_hidden: torch.Tensor,
        positions: torch.Tensor,
        *,
        idx_mapping: torch.Tensor | None = None,
        temperature: torch.Tensor | None = None,
        seeds: torch.Tensor | None = None,
        draft_logits: torch.Tensor | None = None,
        draft_step_cols: torch.Tensor | None = None,
        active_num_reqs: torch.Tensor | None = None,
        use_fp64_gumbel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if positions.numel() == 0:
            raise ValueError("DSpark forward_spec requires at least one position.")
        if idx_mapping is None:
            raise ValueError("DSpark forward_spec requires idx_mapping.")
        debug_start_pos = None
        if os.getenv("VLLM_DSPARK_DEBUG_DIR") or os.getenv(
            "VLLM_DSPARK_TENSOR_DEBUG_DIR"
        ):
            debug_start_pos = int(positions[0].detach().cpu())
        x, main_x, draft_input_ids = self.forward_embed(main_hidden, input_ids)
        if os.getenv("VLLM_DSPARK_DEBUG_DIR"):
            _maybe_dump_stage_debug(
                "forward_embed",
                {
                    "main_hidden": main_hidden,
                    "main_x": main_x,
                    "x": x,
                },
                {
                    "start_pos": debug_start_pos,
                    "input_ids": [
                        int(v) for v in input_ids.detach().cpu().tolist()[:8]
                    ],
                    "positions": [
                        int(v) for v in positions.detach().cpu().tolist()[:8]
                    ],
                },
            )
        _maybe_save_tensor_debug(
            "forward_embed",
            {
                "input_ids": input_ids,
                "positions": positions,
                "main_hidden": main_hidden,
                "main_x": main_x,
                "draft_input_ids": draft_input_ids,
                "x": x,
            },
            {
                "start_pos": debug_start_pos,
            },
        )

        final_layer = self.layers[
            str(self.dspark_start_layer_idx + self.num_dspark_layers - 1)
        ]
        assert isinstance(final_layer, DeepSeekV4DSparkLayer)
        if os.getenv("VLLM_DSPARK_REFERENCE_HC") == "1":
            for layer in self.layers.values():
                assert isinstance(layer, DeepSeekV4DSparkLayer)
                x = layer.forward_reference_hc(
                    x,
                    positions,
                    draft_input_ids,
                    main_x,
                    idx_mapping,
                    debug_start_pos,
                )
                _maybe_dump_stage_debug(
                    "layer_reference_hc",
                    {"x": x},
                    {"layer": layer.prefix, "start_pos": debug_start_pos},
                )
                _maybe_save_tensor_debug(
                    "layer_reference_hc",
                    {"x": x},
                    {"layer": layer.prefix, "start_pos": debug_start_pos},
                )
        else:
            residual, post_mix, res_mix = None, None, None
            for layer in self.layers.values():
                assert isinstance(layer, DeepSeekV4DSparkLayer)
                x, residual, post_mix, res_mix = layer(
                    x,
                    positions,
                    draft_input_ids,
                    main_x,
                    idx_mapping,
                    debug_start_pos,
                    post_mix,
                    res_mix,
                    residual,
                )
                _maybe_dump_stage_debug(
                    "layer_fused_hc",
                    {
                        "x": x,
                        "residual": residual,
                        "post_mix": post_mix,
                        "res_mix": res_mix,
                    },
                    {"layer": layer.prefix, "start_pos": debug_start_pos},
                )
                _maybe_save_tensor_debug(
                    "layer_fused_hc",
                    {
                        "x": x,
                        "residual": residual,
                        "post_mix": post_mix,
                        "res_mix": res_mix,
                    },
                    {"layer": layer.prefix, "start_pos": debug_start_pos},
                )
            assert residual is not None
            assert post_mix is not None
            assert res_mix is not None
            if final_layer._should_run_b12x_mhc(int(x.shape[0] * x.shape[1])):
                from b12x.integration.residual import b12x_mhc_post

                x_flat = b12x_mhc_post(
                    x.reshape(-1, self.config.hidden_size),
                    residual,
                    post_mix,
                    res_mix,
                )
            else:
                x_flat = mhc_post_tilelang(
                    x.reshape(-1, self.config.hidden_size),
                    residual,
                    post_mix,
                    res_mix,
                )
            x = x_flat.view(
                input_ids.shape[0],
                draft_input_ids.shape[1],
                self.config.hc_mult,
                self.config.hidden_size,
            )
            _maybe_dump_stage_debug(
                "final_mhc_post",
                {"x": x},
                {"start_pos": debug_start_pos},
            )
            _maybe_save_tensor_debug(
                "final_mhc_post",
                {"x": x},
                {"start_pos": debug_start_pos},
            )
        output_ids, logits, confidence = self.forward_head(
            x,
            input_ids,
            idx_mapping=idx_mapping,
            temperature=temperature,
            seeds=seeds,
            positions=positions,
            draft_logits=draft_logits,
            draft_step_cols=draft_step_cols,
            active_num_reqs=active_num_reqs,
            use_fp64_gumbel=use_fp64_gumbel,
        )
        if os.getenv("VLLM_DSPARK_DEBUG_DIR"):
            _maybe_dump_stage_debug(
                "forward_head",
                {
                    "logits": logits,
                    "confidence": confidence,
                },
                {
                    "start_pos": debug_start_pos,
                    "output_ids": [
                        [int(v) for v in row]
                        for row in output_ids.detach().cpu().tolist()[:2]
                    ],
                },
            )
        _maybe_save_tensor_debug(
            "forward_head",
            {
                "input_ids": input_ids,
                "positions": positions,
                "output_ids": output_ids,
                "logits": logits,
                "confidence": confidence,
            },
            {
                "start_pos": debug_start_pos,
            },
        )
        return output_ids, logits, confidence

    def _get_dspark_layer_idx_from_weight_name(self, name: str) -> int | None:
        for idx in range(self.num_dspark_layers):
            layer_idx = self.dspark_start_layer_idx + idx
            if name.startswith(f"layers.{layer_idx}.") or name.startswith(
                f"model.layers.{layer_idx}."
            ):
                return layer_idx
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        first_layer = next(iter(self.layers.values()))
        if first_layer.ffn.use_mega_moe:
            expert_mapping = make_deepseek_v4_expert_params_mapping(
                self.config.n_routed_experts
            )
        else:
            expert_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.n_routed_experts,
            )

        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )

        pending_main_proj: dict[str, dict[str, torch.Tensor]] = {}

        def maybe_load_main_proj(
            name: str,
            loaded_weight: torch.Tensor,
        ) -> bool:
            if ".main_proj." not in name:
                return False
            base_name, suffix = name.rsplit(".", 1)
            if suffix not in ("weight", "scale"):
                return False
            entry = pending_main_proj.setdefault(base_name, {})
            entry[suffix] = loaded_weight
            if "weight" not in entry or "scale" not in entry:
                return True
            param_name = f"{base_name}.weight"
            param = params_dict[param_name]
            dequant_weight = _dequantize_e4m3_e8m0_block_weight(
                entry["weight"],
                entry["scale"],
                out_dtype=param.dtype,
            )
            dequant_weight = dequant_weight.to(device=param.device, dtype=param.dtype)
            with torch.no_grad():
                param.data.copy_(dequant_weight)
            if _read_bool_env("VLLM_DSPARK_FP8_MAIN_PROJ", "1"):
                from vllm.model_executor.layers.quantization.utils.fp8_utils import (
                    deepgemm_post_process_fp8_weight_block,
                )

                module = self.get_submodule(base_name)
                with torch.no_grad():
                    dg_w, dg_ws = deepgemm_post_process_fp8_weight_block(
                        entry["weight"].to(param.device),
                        entry["scale"].to(param.device),
                        (128, 128),
                        use_e8m0=True,
                    )
                module._fp8_weight = dg_w
                module._fp8_scale = dg_ws
                logger.info_once(
                    "DSpark main_proj runs natively FP8 (checkpoint weight)."
                )
            loaded_params.add(param_name)
            del pending_main_proj[base_name]
            return True

        for name, loaded_weight in weights:
            if not name.startswith("mtp."):
                continue
            mtp_layer_idx = int(name.split(".", 2)[1])
            name = name.replace(
                f"mtp.{mtp_layer_idx}.",
                f"layers.{self.dspark_start_layer_idx + mtp_layer_idx}.",
            )
            spec_layer = self._get_dspark_layer_idx_from_weight_name(name)
            if spec_layer is None:
                continue

            if maybe_load_main_proj(name, loaded_weight):
                continue

            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else ".weight_scale_inv"
                )
                name = name.removesuffix(".scale") + suffix

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if ".experts." in name:
                    continue
                if self.layers[
                    str(spec_layer)
                ]._dspark_use_separate_qkv and weight_name in ("attn.wq_a", "attn.wkv"):
                    continue
                if weight_name in ("w1", "w3") and ".shared_experts." not in name:
                    continue
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                param = params_dict[mapped_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped_name)
                break
            else:
                if ".experts." in name:
                    # E8M0 expert scales are stored as float8_e8m0fnu in the
                    # checkpoint, but MXFP4 MoE scale params keep raw exponent
                    # bytes. Non-expert FP8 linears use float8 scale params, so
                    # keep this conversion scoped to experts only.
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        mapped_name = name.replace(weight_name, param_name)
                        param = params_dict[mapped_name]
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        if weight_loader(
                            param,
                            loaded_weight,
                            mapped_name,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        ):
                            loaded_params.add(mapped_name)
                            break
                    continue
                if "attn_sink" in name:
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                if ".shared_experts.w2" in name:
                    name = name.replace(
                        ".shared_experts.w2", ".shared_experts.down_proj"
                    )
                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias",
                        ".ffn.gate.e_score_correction_bias",
                    )
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        if pending_main_proj:
            raise ValueError(
                "Incomplete DSpark main_proj FP8 block weights: "
                f"{sorted(pending_main_proj)}"
            )

        expected_layers = set(
            range(
                self.dspark_start_layer_idx,
                self.dspark_start_layer_idx + self.num_dspark_layers,
            )
        )
        loaded_layers = {
            layer
            for param_name in loaded_params
            if (layer := self._get_dspark_layer_idx_from_weight_name(param_name))
            is not None
        }
        missing_layers = expected_layers - loaded_layers
        if missing_layers:
            raise ValueError(
                "DSpark draft layer weights missing from checkpoint: "
                f"{sorted(missing_layers)}"
            )

        for layer in self.layers.values():
            layer.cache_dspark_reference_fp8_linears()
            layer.cache_dspark_wo_a_bf16()
            layer.refresh_b12x_mhc_bf16_weights()
            layer.ffn.finalize_mega_moe_weights()

        logger.info_once(
            "DSpark draft weights loaded: %d params across %d stages",
            len(loaded_params),
            self.num_dspark_layers,
        )
        return loaded_params


def load_dspark_model(vllm_config: VllmConfig) -> DeepSeekV4DSparkDraft:
    target_device = vllm_config.device_config.device
    with set_default_torch_dtype(vllm_config.model_config.dtype):
        model = DeepSeekV4DSparkDraft(vllm_config=vllm_config)
    model.load_weights(_iter_mtp_safetensors(vllm_config.model_config.model))
    process_weights_after_loading(
        model,
        vllm_config.model_config,
        target_device,
    )
    model.to(target_device)
    return model
