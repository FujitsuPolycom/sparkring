# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.eagle.utils import _should_share


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from vllm.compilation.backends import set_model_tag

    # DSpark uses non-causal attention.
    causal = False
    # Inherit the target's attention backend unless one was explicitly set for
    # the draft. Upstream unconditionally overrode with
    # speculative_config.attention_backend (None here), which resolved to the
    # platform default (flashinfer_sparse) while the target runs b12x — the
    # mismatched draft layers then hit fp32 rope-cache asserts and would use a
    # different KV layout than the sparse-SWA metadata expects.
    draft_backend = (
        speculative_config.attention_backend
        if speculative_config.attention_backend is not None
        else vllm_config.attention_config.backend
    )
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=not causal,
            backend=draft_backend,
        ),
    )

    with set_model_tag("dspark_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    # The draft's rope instances are built fresh under the draft's dtype
    # context, but every fused DSV4 op (deep_gemm_fp8_o_proj, the fused
    # rope/quant KV inserts) requires an fp32 cos_sin_cache — which is what
    # the target's layers carry. Normalize the draft to match.
    import torch

    for module in draft_model.modules():
        cache = getattr(module, "cos_sin_cache", None)
        if cache is not None and cache.dtype != torch.float32:
            module.cos_sin_cache = cache.float()

    if get_pp_group().world_size != 1:
        raise NotImplementedError("DSpark does not support pipeline parallelism.")

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model

    target_embed = getattr(target_inner, "embed_tokens", None)
    draft_embed = getattr(draft_inner, "embed_tokens", None)
    if target_embed is not None and _should_share(
        draft_model, "has_own_embed_tokens", draft_embed, target_embed
    ):
        if draft_embed is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed

    target_lm_head = getattr(target_model, "lm_head", None)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        draft_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    # Fork optimization (VLLM_DSPARK_FP8_LM_HEAD): build the draft-only fp8
    # copy of the just-aliased target lm_head (eager, pre-cudagraph-capture).
    if hasattr(draft_model, "maybe_build_fp8_lm_head"):
        draft_model.maybe_build_fp8_lm_head()

    return draft_model
