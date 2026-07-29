# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft model for DeepSeek-V4 (semi-autoregressive speculative decoding).

See: qwen3_dspark.py for base architecture. This one is specialized to the DSV4 DSpark,
which reuses the target model's architecture similarly to MTP.

To implement non-causal attention, we leverage the sparse attention implementation to
include the future query tokens in the top-k indices for each query token.
"""

import os
from collections.abc import Iterable

import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_post_tilelang,
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
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix

from .model import (
    DeepseekV4DecoderLayer,
    make_deepseek_v4_expert_params_mapping,
)

logger = init_logger(__name__)

# MoE expert scale suffix differs by expert dtype (mirrors deepseek_v4 loaders):
# fp4 experts register ``.weight_scale``; block-fp8 experts ``.weight_scale_inv``.
_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


def _read_bool_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _quantize_fp8_deepgemm(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-quantize a bf16/fp32 weight to the deepgemm fp8 layout (chunked
    over rows so the fp32 staging copy never exceeds ~1/8 of the weight)."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    with torch.no_grad():
        w = weight.detach()
        rows = w.shape[0]
        step = max(128, (rows // 8) // 128 * 128)
        chunks_q, chunks_s = [], []
        for r0 in range(0, rows, step):
            cq, cs = per_block_cast_to_fp8(
                w[r0 : r0 + step].float(), [128, 128], use_ue8m0=True
            )
            chunks_q.append(cq)
            chunks_s.append(cs)
        return deepgemm_post_process_fp8_weight_block(
            torch.cat(chunks_q, dim=0),
            torch.cat(chunks_s, dim=0),
            (128, 128),
            use_e8m0=True,
        )


def _fp8_gemm(
    x: torch.Tensor, dg_w: torch.Tensor, dg_ws: torch.Tensor
) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8_packed_for_deepgemm,
    )
    from vllm.utils.deep_gemm import fp8_gemm_nt

    xq, xs = per_token_group_quant_fp8_packed_for_deepgemm(
        x.to(torch.bfloat16), 128
    )
    out = torch.empty(
        x.shape[0], dg_w.shape[0], dtype=torch.bfloat16, device=x.device
    )
    fp8_gemm_nt((xq, xs), (dg_w, dg_ws), out)
    return out


class DSparkMarkovHeadOptimized(nn.Module):
    """Drop-in for qwen3_dspark.DSparkMarkovHead carrying the fork's shipped
    draft-tail optimizations (same env gates, same defaults as production):

    - VLLM_DSPARK_REPLICATE_MARKOV_W2 (default on): markov_w2 is a plain
      replicated Linear so the full-vocab bias is computed locally on every
      rank — the per-position logits all-gathers in the sequential Markov
      loop disappear (both ranks then sample identical seeded-gumbel tokens).
    - VLLM_DSPARK_MARKOV_W2_BF16 (default on): store W2 bf16 (halves GEMV
      bytes; bias err <= 0.013 on a +-3 bias).
    - VLLM_DSPARK_MARKOV_W2_FP8 (default on): run the bias GEMV via fp8
      deepgemm, lazily quantized from the loaded weight on first use.
    """

    def __init__(self, vocab_size: int, markov_rank: int, prefix: str) -> None:
        super().__init__()
        # VLLM_DSPARK_REPLICATE_MARKOV_W1: plain nn.Embedding, no
        # vocab-parallel all-reduce per markov step (production runs this on).
        self._replicate_w1 = _read_bool_env("VLLM_DSPARK_REPLICATE_MARKOV_W1")
        if self._replicate_w1:
            self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
            self.markov_w1.weight.requires_grad_(False)
            logger.info_once("DSpark(v2) replicated Markov W1 (plain embedding).")
        else:
            self.markov_w1 = VocabParallelEmbedding(
                vocab_size, markov_rank, prefix=maybe_prefix(prefix, "markov_w1")
            )
        self._w2_bf16 = _read_bool_env("VLLM_DSPARK_MARKOV_W2_BF16", "1")
        self._replicate_w2 = _read_bool_env("VLLM_DSPARK_REPLICATE_MARKOV_W2", "1")
        if self._replicate_w2:
            self.markov_w2 = nn.Linear(
                markov_rank,
                vocab_size,
                bias=False,
                dtype=torch.bfloat16 if self._w2_bf16 else torch.float32,
            )
            self.markov_w2.weight.requires_grad_(False)
            self._w2_fp8 = _read_bool_env("VLLM_DSPARK_MARKOV_W2_FP8", "1")
            logger.info_once(
                "DSpark(v2) replicated Markov W2 (bf16=%s, fp8=%s).",
                self._w2_bf16,
                self._w2_fp8,
            )
        else:
            self._w2_fp8 = False
            self.markov_w2 = ParallelLMHead(
                vocab_size,
                markov_rank,
                params_dtype=torch.bfloat16 if self._w2_bf16 else torch.float32,
                org_num_embeddings=vocab_size,
                prefix=maybe_prefix(prefix, "markov_w2"),
            )

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def bias(self, markov_embed: torch.Tensor, logits_processor) -> torch.Tensor:
        flat = markov_embed.view(-1, markov_embed.shape[-1])
        if self._replicate_w2:
            if self._w2_fp8:
                if not hasattr(self, "_w2_fp8_weight"):
                    dg_w, dg_ws = _quantize_fp8_deepgemm(self.markov_w2.weight)
                    self._w2_fp8_weight = dg_w
                    self._w2_fp8_scale = dg_ws
                out = _fp8_gemm(flat, self._w2_fp8_weight, self._w2_fp8_scale)
            else:
                out = F.linear(
                    flat.to(self.markov_w2.weight.dtype), self.markov_w2.weight
                )
            return out.view(*markov_embed.shape[:-1], -1)
        return logits_processor(self.markov_w2, flat).view(
            *markov_embed.shape[:-1], -1
        )


class DSparkDeepseekV4Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.num_hidden_layers = config.num_hidden_layers
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)

        self.num_dspark_layers = getattr(config, "n_mtp_layers", None) or 3

        # Shared with the target (aliased by the speculator's loading utility).
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        current_vllm_config = get_current_vllm_config()
        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(
                    current_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{self.num_hidden_layers + i}"),
                )
                for i in range(self.num_dspark_layers)
            ]
        )

        # Heads: final norm + hc_head, and the Markov head
        # Loaded from the "final" MTP layer weights (mtp.*) in the target checkpoint
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )
        self.markov_head = DSparkMarkovHeadOptimized(
            config.vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """main_x = main_norm(main_proj(concat of target aux hidden states)).

        ``aux_hidden_states`` is [T, hidden_size * len(target_layer_ids)].
        """
        return self.main_norm(self.main_proj(aux_hidden_states))

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        main_x: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        """Insert the sliding-window context KV for every draft layer.

        Mirrors the reference DSparkAttention: each layer derives its context KV
        from the SAME projected target hidden ``main_x``, via that layer's own
        ``wkv`` + ``kv_norm`` + RoPE + quant, then writes it at the
        layer's context slots.

        ``context_slot_mappings`` is a per-layer list (each entry is the context
        slot mapping for that layer's kv-cache group, since the hybrid manager may
        place draft layers in different groups). ``None`` (or a ``None`` entry)
        runs the projection to reserve workspace but writes nothing (profiling).
        """
        for i, layer in enumerate(self.layers):
            slot_mapping = (
                None if context_slot_mappings is None else context_slot_mappings[i]
            )
            attn = layer.attn
            # Optimized DSV4 MLA path: wkv part of the fused wq_a|wkv projection
            # (q_lora part discarded), then RoPE/quant/insert via the fused op.
            qr_kv, _ = attn.fused_wqa_wkv(main_x)
            kv = qr_kv[..., attn.q_lora_rank :]
            kv = attn.kv_norm(kv)
            if slot_mapping is None:
                continue
            _insert_context_kv(attn, kv, context_positions, slot_mapping)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        # Expand to hc_mult copies for hyper-connections ([T, H] -> [T, hc, H]).
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)

        residual = post_mix = res_mix = None
        for layer in self.layers:
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )
        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
        # hc_head reduces the hc copies; return the PRE-norm head hidden
        hidden_states = hc_head_fused_kernel_tilelang(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        return hidden_states


def _insert_context_kv(
    attn: nn.Module,
    kv: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """RoPE + quant + paged-cache insert of (already kv_norm'd) context KV.

    Reuses the DSV4 fused insert ops (which also process a query; we pass a dummy
    query and discard it, since context tokens have no query). Mirrors
    ``DeepseekV4Attention._fused_qnorm_rope_kv_insert``.
    """
    swa_cache = attn.swa_cache_layer.kv_cache
    block_size = attn.swa_cache_layer.block_size
    cos_sin_cache = attn.rotary_emb.cos_sin_cache
    cache_dtype = swa_cache.dtype
    n_ctx = kv.shape[0]
    dummy_q = torch.zeros(
        (n_ctx, attn.n_local_heads, attn.head_dim),
        dtype=kv.dtype,
        device=kv.device,
    )
    if cache_dtype == torch.uint8:
        # fp8_ds_mla UE8M0 paged layout
        swa_2d = swa_cache.view(swa_cache.shape[0], -1)
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            dummy_q,
            kv,
            swa_2d,
            slot_mapping,
            positions,
            cos_sin_cache,
            attn.padded_heads,
            attn.eps,
            block_size,
        )
    elif cache_dtype == torch.bfloat16:
        swa_3d = swa_cache.view(-1, block_size, attn.head_dim)
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
            dummy_q,
            kv,
            swa_3d,
            slot_mapping,
            positions,
            cos_sin_cache,
            attn.eps,
            block_size,
        )
    else:  # per-tensor fp8 (torch.float8_e4m3fn)
        # TODO(ben): double-check if this is being dispatched correctly for FI backend
        swa_3d = swa_cache.view(-1, block_size, attn.head_dim)
        dummy_q_fp8 = torch.zeros_like(dummy_q, dtype=torch.float8_e4m3fn)
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
            dummy_q,
            kv,
            dummy_q_fp8,
            swa_3d,
            slot_mapping,
            positions,
            cos_sin_cache,
            attn._flashinfer_fp8_kv_scale,
            attn._flashinfer_fp8_q_scale_inv,
            attn.eps,
            block_size,
        )


class DSparkDeepseekV4ForCausalLM(nn.Module):
    # Draft weights ship in the target checkpoint (mtp.*) without embed/head, so
    # load_dspark_model always aliases the target's.
    has_own_embed_tokens = False
    has_own_lm_head = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.model = DSparkDeepseekV4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Shared with the target (aliased by the speculator's load utility).
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    # --- Hooks used by the speculator -------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        # DSV4 MLA path: each draft layer's sliding-window cache is a separate
        # layer, named by its prefix.
        return [layer.attn.swa_cache_layer.prefix for layer in self.model.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mappings
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Returns the pre-norm hc_head hidden ([T, hidden_size]).
        return self.model(input_ids, positions, inputs_embeds)

    def maybe_build_fp8_lm_head(self) -> None:
        """FP8 deepgemm copy of the (target-aliased) lm_head for DRAFT logits
        only — halves the per-draft-block lm_head weight read. Verification
        stays lossless (rejection sampling uses the actual draft distribution
        q); only proposal quality shifts marginally (top6 overlap ~93%).
        Called by load_dspark_model after the target lm_head is aliased in.
        """
        if not _read_bool_env("VLLM_DSPARK_FP8_LM_HEAD", "1"):
            return
        dg_w, dg_ws = _quantize_fp8_deepgemm(self.lm_head.weight)
        self.register_buffer("lm_head_fp8_weight", dg_w, persistent=False)
        self.register_buffer("lm_head_fp8_scale", dg_ws, persistent=False)
        logger.info_once(
            "DSpark(v2) draft logits use an FP8 deepgemm copy of target lm_head."
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Base logits U_k = lm_head(norm(head_hidden))."""
        lm_head_fp8_weight = getattr(self, "lm_head_fp8_weight", None)
        if lm_head_fp8_weight is not None:
            normed = self.model.norm(hidden_states)
            local = _fp8_gemm(
                normed.reshape(-1, normed.shape[-1]),
                lm_head_fp8_weight,
                self.lm_head_fp8_scale,
            )
            logits = self.logits_processor._gather_logits(local)
            if logits is not None:
                logits = logits[..., : self.config.vocab_size]
                logits = logits.view(*hidden_states.shape[:-1], -1)
            return logits
        return self.logits_processor(self.lm_head, self.model.norm(hidden_states))

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    # --- Weight loading ----------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the ``mtp.{0,1,2}.*`` draft weights from the target checkpoint.

        Non-mtp weights (embed/head/main layers) belong to the target model and
        are skipped here. ``embed_tokens``/``lm_head`` are aliased from the target.
        """
        first_layer = self.model.layers[0]
        use_mega_moe = first_layer.ffn.use_mega_moe
        if use_mega_moe:
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

        # (param_name, ckpt_shard_name, shard_id) for non-expert stacked params.
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
        head_start = n_local_head * tp_rank
        head_end = n_local_head * (tp_rank + 1)

        for name, loaded_weight in weights:
            mapped = self._remap_dspark_name(name)
            if mapped is None:
                continue
            name = mapped

            # ``.scale`` -> per-method scale suffix.
            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else ".weight_scale_inv"
                )
                name = name.removesuffix(".scale") + suffix

            # E8M0 expert scales: keep raw exponent bytes.
            if ".experts." in name:
                if (
                    "weight_scale" in name
                    and loaded_weight.dtype == torch.float8_e8m0fnu
                ):
                    loaded_weight = loaded_weight.view(torch.uint8)
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name not in name:
                        continue
                    name_mapped = name.replace(weight_name, param_name)
                    param = params_dict[name_mapped]
                    success = param.weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                        break
                continue

            # Stacked rules only apply to decoder-layer weights. Head-stack params
            # (main_proj/norm/hc_head/markov_head) load directly — otherwise e.g.
            # "markov_w1" would collide with the "w1" shard rule.
            is_layer_param = name.startswith("model.layers.")
            for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
                if not is_layer_param or weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, stacked_shard_id)
                loaded_params.add(name)
                break
            else:
                if "attn_sink" in name:
                    narrow = loaded_weight[head_start:head_end]
                    params_dict[name][: narrow.shape[0]].copy_(narrow)
                    loaded_params.add(name)
                    continue
                if ".shared_experts.w2" in name:
                    name = name.replace(
                        ".shared_experts.w2", ".shared_experts.down_proj"
                    )
                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                    )
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        self._finalize_moe()
        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def _finalize_moe(self) -> None:
        for layer in self.model.layers:
            layer.ffn.finalize_mega_moe_weights()

    def _remap_dspark_name(self, name: str) -> str | None:
        """Map a checkpoint ``mtp.{i}.*`` name to this model's parameter path.

        Returns None for non-mtp weights (owned by the target model).
        """
        m = re.match(r"mtp\.(\d+)\.(.*)", name)
        if m is None:
            return None
        stage = int(m.group(1))
        rest = m.group(2)
        # The confidence head is not wired into inference yet; drop its weights.
        if rest.startswith("confidence_head."):
            return None
        # Head-stack params live at model level (mtp.last), context combiner at
        # model level (mtp.0); everything else is a per-layer decoder block.
        head_prefixes = (
            "norm.",
            "hc_head_fn",
            "hc_head_base",
            "hc_head_scale",
            "markov_head.",
        )
        if rest.startswith(("main_proj.", "main_norm.")) or rest.startswith(
            head_prefixes
        ):
            return f"model.{rest}"
        return f"model.layers.{stage}.{rest}"
