# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Sequence

import torch.nn as nn

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger
from vllm.models.deepseek_v4.nvidia.dspark import load_dspark_model

logger = init_logger(__name__)


def _get_target_layer_ids(spec_config: SpeculativeConfig) -> tuple[int, ...]:
    override = os.getenv("VLLM_DSPARK_TARGET_LAYER_IDS")
    if override:
        try:
            result = tuple(int(part.strip()) for part in override.split(","))
        except ValueError as err:
            raise RuntimeError(
                "VLLM_DSPARK_TARGET_LAYER_IDS must be comma-separated integers, "
                f"got {override!r}."
            ) from err
        if not result:
            raise RuntimeError("VLLM_DSPARK_TARGET_LAYER_IDS must not be empty.")
        return result

    if not (spec_config and spec_config.draft_model_config):
        raise RuntimeError("DSpark requires a draft model config.")
    layer_ids = getattr(
        spec_config.draft_model_config.hf_config,
        "dspark_target_layer_ids",
        None,
    )
    if not isinstance(layer_ids, Sequence) or isinstance(layer_ids, str):
        raise RuntimeError("DSpark model config must define dspark_target_layer_ids.")
    result = tuple(int(i) for i in layer_ids)
    if not result:
        raise RuntimeError("DSpark target layer list must not be empty.")
    return result


def set_dspark_aux_hidden_state_layers(
    model: nn.Module,
    spec_config: SpeculativeConfig,
) -> None:
    """Configure target-model auxiliary hidden-state capture for DSpark.

    DeepSeek's reference DSpark code concatenates mean(HC-stream) hidden states
    after the configured target layers. This is not EAGLE3's pre-layer residual
    capture, so keep it as a separate mode.
    """
    layer_ids = _get_target_layer_ids(spec_config)
    parent_ref = (
        model.get_language_model() if hasattr(model, "get_language_model") else model
    )
    if not hasattr(parent_ref, "model"):
        raise RuntimeError("DSpark target model must expose a .model module.")
    inner = parent_ref.model
    if not hasattr(inner, "set_dspark_aux_hidden_state_layers"):
        raise RuntimeError(
            "Target model does not support DSpark auxiliary hidden-state capture."
        )
    inner.set_dspark_aux_hidden_state_layers(layer_ids)
    logger.info("Using DSpark auxiliary target layers: %s", layer_ids)


def _load_embed_tokens_for_last_stage(vllm_config):
    """Build the draft's embedding on a PP stage that has no target embed.

    Under PP, the target keeps embed_tokens on the first stage only, but the
    DSpark draft runs on the LAST stage. Reconstruct an identical
    VocabParallelEmbedding there and stream its weight from the checkpoint
    (embeddings are stored unquantized; the param's weight_loader applies the
    local TP shard).
    """
    import json
    import os as _os

    import torch
    from safetensors import safe_open

    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.models.deepseek_v4.nvidia.model import (
        _get_virtual_tp_vocab_padding_size,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    hf_config = vllm_config.model_config.hf_config
    model_path = vllm_config.model_config.model
    with set_default_torch_dtype(vllm_config.model_config.dtype):
        embed = VocabParallelEmbedding(
            hf_config.vocab_size,
            hf_config.hidden_size,
            padding_size=_get_virtual_tp_vocab_padding_size(hf_config),
            quant_config=None,
            prefix="model.embed_tokens",
        )
    index_path = _os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    for tensor_name in (
        "model.embed_tokens.weight",  # HF-style
        "embed.weight",               # DeepSeek V4 native naming
        "embed_tokens.weight",
    ):
        if tensor_name in weight_map:
            break
    else:
        raise RuntimeError(
            "No embedding tensor found in checkpoint index for DSpark PP "
            f"draft (looked for embed_tokens/embed variants) at {index_path}."
        )
    shard_file = weight_map[tensor_name]
    with safe_open(
        _os.path.join(model_path, shard_file), framework="pt", device="cpu"
    ) as f:
        full_weight = f.get_tensor(tensor_name)
    param = embed.weight
    loader = getattr(param, "weight_loader", None)
    if loader is not None:
        loader(param, full_weight)
    else:
        param.data.copy_(full_weight)
    embed = embed.to(vllm_config.device_config.device)
    logger.info(
        "DSpark PP: rebuilt embed_tokens on the last stage from %s "
        "(%s, dtype=%s).", shard_file, tuple(param.shape), param.dtype,
    )
    return embed


def load_deepseek_v4_dspark_model(target_model: nn.Module, vllm_config):
    """Load the DeepSeek V4 DSpark draft module from the target checkpoint.

    DSpark shares the target embedding table and lm_head, but its ``mtp.*``
    tensors are a draft-only block model rather than serial MTP layers.
    """
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.model_executor.models.utils import PPMissingLayer

    dspark_model = load_dspark_model(vllm_config)
    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
        target_inner, "embedding", None
    )
    target_lm_head = getattr(target_language_model, "lm_head", None)
    if isinstance(target_embed, PPMissingLayer):
        # PP: embed lives on the first stage; the draft runs here on the
        # last. A PPMissingLayer would silently echo token ids as embeds.
        target_embed = None
    if target_embed is None:
        if get_pp_group().world_size > 1 and get_pp_group().is_last_rank:
            target_embed = _load_embed_tokens_for_last_stage(vllm_config)
        else:
            raise RuntimeError(
                "DSpark target model does not expose embed_tokens."
            )
    if target_lm_head is None or isinstance(target_lm_head, PPMissingLayer):
        raise RuntimeError(
            "DSpark target model does not expose lm_head (draft must run "
            "on the last PP stage)."
        )
    dspark_model.attach_target_modules(target_embed, target_lm_head)
    return dspark_model
