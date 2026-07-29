# SPDX-License-Identifier: Apache-2.0
"""Hybrid quantization config: festr2 MXFP4 routed experts + RedHat
compressed-tensors (block-FP8 self_attn / NVFP4 mlp+shared) everything-else,
in one GLM-5.2 checkpoint.

Selected when config.json declares quant_method: "hybrid_mxfp4_ct".
Mounted at vllm/model_executor/layers/quantization/hybrid_mxfp4_ct.py, imported
via a 2-line append to that package's __init__ overlay.

Design (verified against production-3.75 sources):
- RoutedExperts -> Mxfp4Config.get_quant_method -> Mxfp4MoEMethod (b12x
  fp4_e8m0_k32 kernel via VLLM_USE_B12X_MOE, backend auto).
- Everything else (LinearBase incl. MLA projections, Attention KV method,
  dense/shared mlp) -> CompressedTensorsConfig, fed RedHat's verbatim
  quantization_config as the "linear" sub-dict.
- Branch order matters: intercept RoutedExperts BEFORE delegating, else the CT
  half would claim the MoE and build an NVFP4 MoE method.
"""

import torch

from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


def _routed_experts_cls():
    # Lazy import (avoid import cycles at package-init time).
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    return RoutedExperts


@register_quantization_config("hybrid_mxfp4_ct")
class HybridMxfp4CtConfig(QuantizationConfig):
    """Compose Mxfp4 (routed experts) with compressed-tensors (the rest)."""

    def __init__(self, moe, linear):
        super().__init__()
        self.moe = moe
        self.linear = linear

    @classmethod
    def get_name(cls) -> str:
        return "hybrid_mxfp4_ct"

    @classmethod
    def get_min_capability(cls) -> int:
        return 80  # mxfp4 floor; CT is lower

    def get_supported_act_dtypes(self):
        return [torch.bfloat16]

    @classmethod
    def get_config_filenames(cls):
        return []

    def is_mxfp4_quant(self, prefix, layer):
        # hidden-size rounding helpers treat MoE as mxfp4
        return isinstance(layer, _routed_experts_cls())

    @classmethod
    def from_config(cls, config):
        from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
            CompressedTensorsConfig,
        )

        moe = Mxfp4Config.from_config(config.get("moe", {}))
        lin_cfg = dict(config["linear"])
        # CT head-piping is skipped when top-level quant_method != CT
        # (weight_utils.py:259-274); forward head counts defensively.
        for k in ("total_num_heads", "total_num_kv_heads"):
            if k in config:
                lin_cfg.setdefault(k, config[k])
        return cls(moe, CompressedTensorsConfig.from_config(lin_cfg))

    def get_quant_method(self, layer, prefix):
        # Propagate the model-supplied fused-module map to both halves.
        self.moe.packed_modules_mapping = self.packed_modules_mapping
        self.linear.packed_modules_mapping = self.packed_modules_mapping
        if isinstance(layer, _routed_experts_cls()):
            return self.moe.get_quant_method(layer, prefix)
        return self.linear.get_quant_method(layer, prefix)

    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        # Keep CT target/ignore remapping intact.
        self.linear.apply_vllm_mapper(hf_to_vllm_mapper)
