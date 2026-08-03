# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import typing
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import PretrainedConfig

from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_dequantize,
)
from vllm.model_executor.layers.sparse_attn_indexer import use_b12x_sparse_indexer
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2MixtureOfExperts,
    DeepseekV2MoE,
    _try_load_fp8_indexer_wk,
    get_spec_layer_idx_from_weight_name,
)
from .utils import get_draft_quant_config, get_pp_missing_layer_names, maybe_prefix

logger = init_logger(__name__)


def _resolve_cached_hf_model_path(model_path: str | None) -> str | None:
    if not model_path or os.path.isdir(model_path):
        return model_path
    try:
        from huggingface_hub import try_to_load_from_cache

        cached_index = try_to_load_from_cache(
            model_path, "model.safetensors.index.json"
        )
        if isinstance(cached_index, str):
            return os.path.dirname(cached_index)
    except Exception:
        return None
    return None


def _get_local_model_path(
    config: PretrainedConfig, vllm_config: VllmConfig
) -> str | None:
    for attr in ("_name_or_path", "name_or_path"):
        model_path = getattr(config, attr, None)
        resolved_model_path = _resolve_cached_hf_model_path(model_path)
        if resolved_model_path:
            return resolved_model_path

    speculative_config = getattr(vllm_config, "speculative_config", None)
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    for model_path in (
        getattr(draft_model_config, "model", None),
        getattr(draft_model_config, "model_path", None),
        getattr(model_config, "model", None),
        getattr(model_config, "model_path", None),
    ):
        resolved_model_path = _resolve_cached_hf_model_path(model_path)
        if resolved_model_path:
            return resolved_model_path
    return None


def _has_serialized_modelopt_fp4_nextn_experts(
    config: PretrainedConfig, vllm_config: VllmConfig
) -> bool:
    model_path = _get_local_model_path(config, vllm_config)
    if model_path is None:
        return False

    nextn_layer_id = getattr(config, "num_hidden_layers", None)
    if nextn_layer_id is None:
        return False

    probe_prefix = f"model.layers.{nextn_layer_id}.mlp.experts.0.down_proj"
    probe_weight = f"{probe_prefix}.weight"
    required_keys = {
        probe_weight,
        f"{probe_prefix}.weight_scale",
        f"{probe_prefix}.weight_scale_2",
        f"{probe_prefix}.input_scale",
    }
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    single_shard_path = os.path.join(model_path, "model.safetensors")

    try:
        if os.path.exists(index_path):
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            if not required_keys.issubset(weight_map):
                return False
            shard_path = os.path.join(model_path, weight_map[probe_weight])
        elif os.path.exists(single_shard_path):
            shard_path = single_shard_path
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                if not required_keys.issubset(set(f.keys())):
                    return False
        else:
            return False

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            return f.get_slice(probe_weight).get_dtype() == "U8"
    except Exception as err:
        logger.warning(
            "Failed to inspect serialized NextN expert quantization metadata: %s",
            err,
        )
        return False


def _maybe_disable_unserialized_modelopt_fp4_nextn(
    config: PretrainedConfig,
    vllm_config: VllmConfig,
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    if quant_config is None or quant_config.get_name() != "modelopt_fp4":
        return quant_config

    if not _has_serialized_modelopt_fp4_nextn_experts(config, vllm_config):
        logger.warning_once(
            "Disabling DeepSeek/GLM NextN modelopt_fp4 quant config because "
            "serialized NextN FP4 expert weights were not found."
        )
        return None

    # SpeculativeConfig.hf_config_override defensively adds whole-MTP-layer
    # ignore entries for GLM checkpoints whose config does not explicitly target
    # the MTP prefix. That is correct for BF16 MTP checkpoints, but serialized
    # NVFP4 NextN experts must stay quantized. Remove only those synthetic
    # whole-layer entries and keep the checkpoint's finer-grained ignores such
    # as self_attn/indexer/shared_experts.
    unquantized_prefixes = getattr(
        config, "vllm_unquantized_mtp_layer_prefixes", None
    )
    exclude_modules = getattr(quant_config, "exclude_modules", None)
    if isinstance(unquantized_prefixes, list) and isinstance(exclude_modules, list):
        synthetic_ignores = {
            pattern
            for prefix in unquantized_prefixes
            for pattern in (prefix, f"{prefix}.*")
        }
        quant_config.exclude_modules = [
            pattern for pattern in exclude_modules if pattern not in synthetic_ignores
        ]
        config.vllm_unquantized_mtp_layer_prefixes = []
    return quant_config


def _maybe_remap_fp8_scale_inv_name(
    name: str, params_dict: dict[str, torch.nn.Parameter]
) -> str:
    """Map FP8 checkpoint scale_inv names to CT runtime scale params."""
    if name in params_dict:
        return name
    if "weight_scale_inv" not in name:
        return name
    alt_name = name.replace("weight_scale_inv", "weight_scale")
    return alt_name if alt_name in params_dict else name


def _maybe_pad_glm_mtp_fused_qkv_fp8_weight(
    name: str,
    tensor: torch.Tensor,
    param: torch.nn.Parameter,
    shard_id: int | str | None,
) -> torch.Tensor:
    """Pad GLM FP8 MTP fused KV-A rows to CUTLASS block shape.

    The checkpoint stores q_a and kv_a separately as 2048 and 576 rows. The
    runtime fused block-FP8 module pads kv_a to 640 rows so the physical output
    dimension is a full 128-row block. Do the corresponding zero padding before
    vLLM's generic merged-column loader narrows by the padded shard size.
    """
    if (
        shard_id != 1
        or not name.endswith("self_attn.fused_qkv_a_proj.weight")
        or tensor.dtype != torch.float8_e4m3fn
    ):
        return tensor

    output_dim = getattr(param, "output_dim", 0)
    if tensor.shape[output_dim] != 576:
        return tensor

    padded_size = 640
    pad_shape = list(tensor.shape)
    pad_shape[output_dim] = padded_size - tensor.shape[output_dim]
    padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    logger.info_once(
        "Padding GLM FP8 MTP fused_qkv_a_proj kv_a checkpoint shard for %s: "
        "loaded shape %s -> output-dim %d padded to %d rows",
        name,
        tuple(tensor.shape),
        output_dim,
        padded_size,
    )
    return torch.cat((tensor, padding), dim=output_dim)


def _try_load_fp8_linear_as_bf16(
    name: str,
    tensor: torch.Tensor,
    buf: dict[str, dict[str, torch.Tensor]],
    params_dict: dict[str, torch.nn.Parameter],
    loaded_params: set[str],
    shard_id: int | str | None = None,
) -> bool:
    is_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
    is_scale_inv = name.endswith(".weight_scale_inv")
    is_mxfp8_scale = name.endswith(".weight_scale") and tensor.dtype == torch.uint8
    if not is_weight and not is_scale_inv and not is_mxfp8_scale:
        return False

    if is_weight:
        base_name = name.rsplit(".", 1)[0]
    elif is_scale_inv:
        base_name = name.removesuffix(".weight_scale_inv")
    else:
        base_name = name.removesuffix(".weight_scale")
    weight_name = f"{base_name}.weight"
    scale_inv_name = f"{base_name}.weight_scale_inv"
    mxfp8_scale_name = f"{base_name}.weight_scale"

    # If the runtime module registered an FP8 scale parameter, let the normal
    # quantized loader handle it.
    if (
        _maybe_remap_fp8_scale_inv_name(scale_inv_name, params_dict) in params_dict
        or mxfp8_scale_name in params_dict
    ):
        return False
    if weight_name not in params_dict:
        return False

    buffer_key = base_name if shard_id is None else f"{base_name}:{shard_id}"
    entry = buf.setdefault(buffer_key, {})
    if is_weight:
        entry["weight"] = tensor
    elif is_scale_inv:
        entry["scale_inv"] = tensor
    else:
        entry["mxfp8_scale"] = tensor
    if "weight" not in entry or (
        "scale_inv" not in entry and "mxfp8_scale" not in entry
    ):
        return True

    weight_fp8 = entry["weight"]
    del buf[buffer_key]

    if "scale_inv" in entry:
        scale_inv = entry["scale_inv"]
        block_size = weight_fp8.shape[1] // scale_inv.shape[1]
        weight_bf16 = scaled_dequantize(
            weight_fp8,
            scale_inv,
            group_shape=GroupShape(block_size, block_size),
            out_dtype=torch.bfloat16,
        )
    else:
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            dequant_mxfp8_to_bf16,
        )

        weight_bf16 = dequant_mxfp8_to_bf16(weight_fp8, entry["mxfp8_scale"])

    param = params_dict[weight_name]
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    if shard_id is None:
        weight_loader(param, weight_bf16)
    else:
        weight_loader(param, weight_bf16, shard_id)
    loaded_params.add(weight_name)
    return True


class SharedHead(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "head"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class DeepSeekMultiTokenPredictorLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()

        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = _maybe_disable_unserialized_modelopt_fp4_nextn(
            config, vllm_config, get_draft_quant_config(vllm_config)
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        self.device = current_platform.device_type

        self.is_v32 = getattr(config, "index_topk", 0) > 0
        if self.is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            topk_scores_buffer = None
            if (
                vllm_config.parallel_config.decode_context_parallel_size > 1
                and use_b12x_sparse_indexer()
            ):
                topk_scores_buffer = torch.empty(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    topk_tokens,
                    dtype=torch.float32,
                    device=self.device,
                )
        else:
            topk_indices_buffer = None
            topk_scores_buffer = None

        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=quant_config
        )
        self.mtp_block = DeepseekV2DecoderLayer(
            vllm_config,
            prefix,
            config=self.config,
            topk_indices_buffer=topk_indices_buffer,
            topk_scores_buffer=topk_scores_buffer,
            quant_config=quant_config,
            layer_idx_override=0,
            is_nextn=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # masking inputs at position 0, as not needed by MTP
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )

        hidden_states, residual = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=None,
        )
        hidden_states = residual + hidden_states  # pre-final-norm (logits hidden)
        # Recycle the post-final-norm hidden into the next draft step.
        # compute_logits applies shared_head (== final norm) to the pre-norm
        # element, so logits and the recycle each get exactly one final-norm.
        # Matches SGLang's deepseek_nextn.
        return hidden_states, self.shared_head(hidden_states)


class DeepSeekMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        # to map the exact layer index from weights

        self.layers = torch.nn.ModuleDict(
            {
                str(idx): DeepSeekMultiTokenPredictorLayer(
                    vllm_config, f"{prefix}.layers.{idx}"
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def set_skip_topk(self, skip: bool):
        """Toggle skip_topk on all MTP layers with sparse attention.

        Called by the proposer to implement index_share_for_mtp_iteration:
        step 0 sets skip=False (compute own indices), steps 1+ set skip=True
        (reuse step 0's indices).
        """
        for layer in self.layers.values():
            mtp_block = getattr(layer, "mtp_block", None)
            if mtp_block is not None:
                self_attn = getattr(mtp_block, "self_attn", None)
                if self_attn is not None:
                    mla_attn = getattr(self_attn, "mla_attn", None)
                    if mla_attn is not None and hasattr(mla_attn, "skip_topk"):
                        mla_attn.skip_topk = skip

    def compact_topk_indices(self, slot_ids: torch.Tensor):
        """Gather each request's final-token top-k row to the buffer front."""
        num_slots = slot_ids.numel()
        for layer in self.layers.values():
            mtp_block = getattr(layer, "mtp_block", None)
            if mtp_block is not None:
                self_attn = getattr(mtp_block, "self_attn", None)
                if self_attn is not None:
                    mla_attn = getattr(self_attn, "mla_attn", None)
                    if mla_attn is not None and hasattr(
                        mla_attn, "topk_indices_buffer"
                    ):
                        topk_indices_buffer = mla_attn.topk_indices_buffer
                        topk_indices_buffer[:num_slots] = topk_indices_buffer[
                            slot_ids
                        ]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        logits = self.logits_processor(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
        )
        return logits

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Vocab-parallel argmax without all-gathering full logits.

        Avoids the per-draft NCCL AllGather inside `compute_logits` by
        running the local lm_head + argmax + tiny (max_value, max_index)
        AllReduce. Selects the right MTP layer's shared head based on
        `spec_step_idx`, mirroring `compute_logits`. Returns full-vocab
        token ids (the MTP head spans the full target vocab, no remap
        needed).
        """
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        return self.logits_processor.get_top_tokens(
            mtp_layer.shared_head.head,
            mtp_layer.shared_head(hidden_states),
        )


@support_torch_compile
class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = _maybe_disable_unserialized_modelopt_fp4_nextn(
            self.config, vllm_config, get_draft_quant_config(vllm_config)
        )
        self.checkpoint_weight_name_prefixes = self._checkpoint_weight_name_prefixes()
        self._exclude_unquantized_mtp_layers_from_quant_config()
        self.model = DeepSeekMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Set MoE hyperparameters
        self.set_moe_parameters()

    def _checkpoint_weight_name_prefixes(self) -> tuple[str, ...]:
        return tuple(
            f"model.layers.{layer_idx}."
            for layer_idx in range(
                self.config.num_hidden_layers,
                self.config.num_hidden_layers + self.config.num_nextn_predict_layers,
            )
        )

    def _exclude_unquantized_mtp_layers_from_quant_config(self) -> None:
        unquantized_prefixes = getattr(
            self.config, "vllm_unquantized_mtp_layer_prefixes", None
        )
        exclude_modules = getattr(self.quant_config, "exclude_modules", None)
        if not unquantized_prefixes or not isinstance(exclude_modules, list):
            return

        added_patterns = []
        for prefix in unquantized_prefixes:
            for pattern in (prefix, f"{prefix}.*"):
                if pattern not in exclude_modules:
                    exclude_modules.append(pattern)
                    added_patterns.append(pattern)

        if added_patterns:
            logger.info(
                "Excluding MTP layers from checkpoint quantization: %s",
                added_patterns,
            )

    def set_moe_parameters(self):
        self.expert_weights = []
        self.num_moe_layers = self.config.num_nextn_predict_layers
        self.num_expert_groups = self.config.n_group

        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers.values():
            assert isinstance(layer, DeepSeekMultiTokenPredictorLayer)
            layer = layer.mtp_block
            assert isinstance(layer, DeepseekV2DecoderLayer)
            if isinstance(layer.mlp, DeepseekV2MoE):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)
        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            hidden_states,
            inputs_embeds,
            spec_step_idx,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Delegate to the inner predictor's vocab-parallel argmax.

        Used by the spec-decode proposer's `_greedy_sample` when
        `use_local_argmax_reduction=True`, replacing the full-vocab
        AllGather with an O(2 * tp_size) reduction.
        """
        return self.model.get_top_tokens(hidden_states, spec_step_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        rocm_aiter_moe_shared_expert_enabled = (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        )
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
        ]

        # Fused indexer wk + weights_proj (shard 0 = wk, shard 1 = weights_proj)
        indexer_fused_mapping = [
            ("wk_weights_proj", "wk", 0),
            ("wk_weights_proj", "weights_proj", 1),
        ]
        stacked_params_mapping.extend(indexer_fused_mapping)

        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (
                self.config.n_shared_experts
                if rocm_aiter_moe_shared_expert_enabled
                else 0
            ),
        )

        pp_missing_layer_names = get_pp_missing_layer_names(self)
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        _pending_wk_fp8: dict = {}  # FP8 indexer wk dequant buffer
        _pending_fp8_linear: dict = {}
        # Rank-sliced EXL3 checkpoints ship one tensor per TP rank
        # (`....rank{r}.{trellis|suh|svh|mcg}`), while Exl3MoEMethod registers a
        # single fused param per projection group (`w13_mcg`, `w2_trellis`, ...).
        # Strip the `.rank{r}` segment before expert mapping, exactly as the
        # target model (deepseek_v2.load_weights) does; otherwise the MTP loader
        # looks up `...routed_experts.w2_rank0.mcg` and KeyErrors.
        rank_sliced_name = getattr(
            self.quant_config,
            "normalize_rank_sliced_weight_name",
            None,
        )
        for name, loaded_weight in weights:
            if rank_sliced_name is not None:
                name = rank_sliced_name(name)
                if name is None:
                    continue
            if "rotary_emb.inv_freq" in name:
                continue
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue
            is_fusion_moe_shared_experts_layer = (
                rocm_aiter_moe_shared_expert_enabled and ("mlp.shared_experts" in name)
            )
            name = self._rewrite_spec_layer_name(spec_layer, name)

            if _try_load_fp8_indexer_wk(
                name,
                loaded_weight,
                _pending_wk_fp8,
                params_dict,
                loaded_params,
                pp_missing_layer_names,
            ):
                continue
            if _try_load_fp8_linear_as_bf16(
                name,
                loaded_weight,
                _pending_fp8_linear,
                params_dict,
                loaded_params,
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                if is_fusion_moe_shared_experts_layer:
                    continue
                name_mapped = name.replace(weight_name, param_name)

                # QKV fusion is optional, fall back to normal
                # weight loading if it's not enabled
                if param_name == "fused_qkv_a_proj":
                    # FP8 checkpoints provide scale tensors as
                    # *.weight_scale_inv or MXFP8 *.weight_scale, while the
                    # fused runtime module may only expose the fused BF16
                    # weight. Do not skip those scale tensors before the
                    # fallback loader has a chance to pair them with the fused
                    # weight.
                    has_fused_target = name_mapped in params_dict
                    if name_mapped.endswith((".weight_scale_inv", ".weight_scale")):
                        if name_mapped.endswith(".weight_scale_inv"):
                            fused_weight = (
                                name_mapped.removesuffix(".weight_scale_inv")
                                + ".weight"
                            )
                        else:
                            fused_weight = (
                                name_mapped.removesuffix(".weight_scale") + ".weight"
                            )
                        has_fused_target = (
                            has_fused_target
                            or fused_weight in params_dict
                            or _maybe_remap_fp8_scale_inv_name(
                                name_mapped, params_dict
                            )
                            in params_dict
                        )
                    if not has_fused_target:
                        continue
                name = name_mapped

                if _try_load_fp8_linear_as_bf16(
                    name,
                    loaded_weight,
                    _pending_fp8_linear,
                    params_dict,
                    loaded_params,
                    shard_id=shard_id,
                ):
                    break

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                name = _maybe_remap_fp8_scale_inv_name(name, params_dict)
                param = params_dict[name]
                loaded_weight = _maybe_pad_glm_mtp_fused_qkv_fp8_weight(
                    name, loaded_weight, param, shard_id
                )
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Special handling: when AITER fusion_shared_experts is enabled,
                # checkpoints may provide a single widened shared_experts tensor
                # without explicit expert indices
                # (e.g. ...mlp.shared_experts.gate_proj.weight).
                # For models with multiple shared experts, split that tensor
                # evenly into per-shared-expert slices and load them into
                # appended expert slots mlp.experts.{n_routed_experts + j}.*
                # accordingly.
                num_chunks = 1
                if is_fusion_moe_shared_experts_layer:
                    num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                    # Determine split axis based on op type
                    # gate/up: ColumnParallel → split along dim 0
                    # down: RowParallel → split along dim 1
                    split_dim = (
                        1
                        if ("down_proj.weight" in name and loaded_weight.ndim > 1)
                        else 0
                    )
                    total = loaded_weight.shape[split_dim]
                    assert total % num_chunks == 0, (
                        f"Shared expert weight dim {total} "
                        f"not divisible by num_chunks {num_chunks}"
                    )
                    chunk_size = total // num_chunks

                for j in range(num_chunks):
                    chunk_name = name
                    weight_to_load = loaded_weight

                    if is_fusion_moe_shared_experts_layer:
                        chunk_slice = slice(j * chunk_size, (j + 1) * chunk_size)
                        if loaded_weight.ndim == 1:
                            weight_to_load = loaded_weight[chunk_slice]
                        elif split_dim == 0:
                            weight_to_load = loaded_weight[chunk_slice, :]
                        else:
                            weight_to_load = loaded_weight[:, chunk_slice]
                        # Synthesize an expert-style name so expert mapping
                        # can route it
                        chunk_name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts + j}",
                        )

                    # Use expert_params_mapping to locate the destination
                    # param and delegate to its expert-aware weight_loader
                    # with expert_id.
                    is_expert_weight = False
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in chunk_name:
                            continue

                        # Anyway, this is an expert weight and should not be
                        # attempted to load as other weights later
                        is_expert_weight = True

                        # Do not modify `name` since the loop may continue here
                        # Instead, create a new variable
                        name_mapped = chunk_name.replace(weight_name, param_name)
                        name_mapped = _maybe_remap_fp8_scale_inv_name(
                            name_mapped, params_dict
                        )

                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # other available replicas.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            weight_to_load,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            if not is_fusion_moe_shared_experts_layer:
                                name = name_mapped
                            else:
                                loaded_params.add(name_mapped)
                            break
                    else:
                        if is_expert_weight:
                            # We've checked that this is an expert weight
                            # However it's not mapped locally to this rank
                            # So we simply skip it
                            continue

                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue

                        name = maybe_remap_kv_scale_name(name, params_dict)
                        if name is None:
                            continue

                        name = _maybe_remap_fp8_scale_inv_name(name, params_dict)
                        # According to DeepSeek-V3 Technical Report, MTP modules
                        # shares embedding layer. We only load the first weights.
                        if (
                            spec_layer != self.model.mtp_start_layer_idx
                            and ".layers" not in name
                        ):
                            continue

                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        try:
                            weight_loader(param, loaded_weight)
                        except AssertionError as e:
                            raise AssertionError(
                                "MTP weight shape mismatch while loading "
                                f"{name}: param={tuple(param.shape)} "
                                f"loaded={tuple(loaded_weight.shape)}"
                            ) from e
            if not is_fusion_moe_shared_experts_layer:
                loaded_params.add(name)

        # Validate that weights were loaded for each expected MTP layer.
        loaded_layers: set[int] = set()
        for param_name in loaded_params:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, param_name)
            if spec_layer is not None:
                loaded_layers.add(spec_layer)
        for layer_idx in range(
            self.model.mtp_start_layer_idx,
            self.model.mtp_start_layer_idx + self.model.num_mtp_layers,
        ):
            if layer_idx not in loaded_layers:
                raise ValueError(
                    f"MTP speculative decoding layer {layer_idx} weights "
                    f"missing from checkpoint. The checkpoint may have "
                    f"been quantized without including the MTP layers. "
                    f"Use a checkpoint that includes MTP layer weights, "
                    f"or disable speculative decoding."
                )

        return loaded_params

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """
        Rewrite the weight name to match the format of the original model.
        Add .mtp_block for modules in transformer layer block for spec layer
        and rename shared layer weights to be top level.
        """
        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head",
        ]
        shared_weight_names = ["embed_tokens"]
        spec_layer_weight = False
        shared_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                if weight_name in shared_weight_names:
                    shared_weight = True
                break
        if not spec_layer_weight:
            # treat rest weights as weights for transformer layer block
            name = name.replace(
                f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
            )
        elif shared_weight:
            # treat shared weights as top level weights
            name = name.replace(f"model.layers.{spec_layer}.", "model.")
        return name

