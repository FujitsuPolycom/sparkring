# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""EXL3 (ExLlamaV3 trellis) quantization support.

Rank-sliced routed-expert checkpoints use Sparkinfer's unified planned
``fused_moe`` API for the Trellis decode/prefill windows and the ExLlamaV3
extension for the small eager parity window. Generic dense and non-rank-sliced
MoE checkpoints use the
bit-faithful ``exllamav3_ext.exl3_gemm`` parity path. Every logical checkpoint
matrix is dispatched independently: vLLM's packed QKV and gate/up modules are
not treated as one EXL3 matrix because each source matrix owns its Hadamard
vectors and codebook marker.

Both dependencies are imported lazily. Importing this module, parsing
checkpoint metadata, or compiling it with ``py_compile`` does not load either
one or initialize CUDA.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import re
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from transformers import PretrainedConfig

from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    MoEActivation,
    RoutedExperts,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    QKVParallelLinear,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

_MCG_SENTINEL = 0xCBAC1FED
_MUL1_SENTINEL = 0x83DCD12D
_HADAMARD_BLOCK = 128
_EXL3_EXT: Any | None = None
_SPARKINFER_FUSED_MOE_API: Any | None = None
_SPARKINFER_MIXED_TRELLIS_API: Any | None = None
_RANK_SLICED_RUNTIMES: dict[tuple[Any, ...], dict[str, Any]] = {}
_MIXED_TRELLIS_RUNTIMES: dict[tuple[Any, ...], dict[str, Any]] = {}
_NEXT_RUNTIME_SCOPE_ID = 0
_MIXED_TRELLIS_ROUTE_BLOCK_SIZE = 8


# Smallest m the Trellis kernel path can service, and therefore the smallest
# row count an EXL3 rank-sliced MoE layer can be CUDA-graph captured at. A
# capture-size selector may read this to align its sizes with the backend
# instead of failing at capture time.
MIN_CAPTURABLE_TRELLIS_M = 1

# Target execution also reaches m=1..3 during profiling and small-batch decode.
# Keep every supported row count on the native path by default; operators may
# still raise the threshold explicitly as a diagnostic kill switch.
_DEFAULT_TRELLIS_MIN_M = MIN_CAPTURABLE_TRELLIS_M


def _is_draft_layer(layer: Any) -> bool:
    """True for a rank-sliced MTP/nextn/eagle draft MoE layer.

    The role is stamped (``exl3_is_draft = True``) on every draft-owned module
    by ``load_eagle_model`` at draft construction, which is the single funnel
    every speculator draft passes through. It is NOT inferable here: a
    GLM-5.2-style MTP head is an extra decoder layer named exactly like a
    target layer (``model.layers.78.*`` with ``num_hidden_layers = 78``), and
    this function runs at plan/capture/forward time, when
    ``set_current_vllm_config`` has already exited and
    ``get_current_vllm_config_or_none()`` returns None -- so both name and
    layer-index inference silently fail. The substring fallback below covers
    drafts built outside ``load_eagle_model``.
    """
    stamped = getattr(layer, "exl3_is_draft", None)
    if stamped is not None:
        return bool(stamped)
    name = str(getattr(layer, "layer_name", "") or getattr(layer, "prefix", ""))
    return any(
        token in name
        for token in (".mtp", "mtp.", "nextn", "eagle", "draft", "speculator")
    )


def _runtime_owner_token(quant_config: Any, layer: Any) -> tuple[int, bool]:
    """Runtime-cache owner identity: (config scope, is_draft).

    Adding the role makes target/draft isolation independent of whether the model
    file happened to mint a separate quant config.
    """
    return (_runtime_scope_id(quant_config), _is_draft_layer(layer))


def _runtime_scope_id(quant_config: Any) -> int:
    """Stable identity for the model that owns a rank-sliced runtime.

    A cached runtime owns mutable Trellis/prefill scratch plus parity staging and
    sort buffers, so an entry must never be shared across models. A target MoE
    layer and a rank-sliced MTP draft layer have identical shapes, topk and
    planner settings -- both read ``max_num_batched_tokens`` from the same
    scheduler config -- so a shape-only key makes the draft reuse the target's
    scratch. That defeats the target/draft resource isolation their
    independently captured CUDA graphs rely on.

    Scoping by the owning quant config is deliberately coarser than per-layer:
    the draft is built with its own ``Exl3Config`` while every layer of one model
    shares a single config, so each model gets exactly one runtime. The prefill
    arena alone is ~1 GiB, so per-layer runtimes would cost tens of GiB per rank
    on a 75+ layer model and are not affordable.
    """
    global _NEXT_RUNTIME_SCOPE_ID
    scope = getattr(quant_config, "_exl3_runtime_scope_id", None)
    if scope is not None:
        return scope
    scope = _NEXT_RUNTIME_SCOPE_ID
    _NEXT_RUNTIME_SCOPE_ID += 1
    try:
        quant_config._exl3_runtime_scope_id = scope  # noqa: SLF001
    except AttributeError:
        # Frozen/slotted config: fall back to object identity. Configs live for
        # the process lifetime, so reuse-after-GC aliasing is not a concern here.
        return id(quant_config)
    return scope


_RANK_SLICED_FORMAT = "exl3-trellis"
_RANK_SLICED_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.+)\.rank(?P<rank>\d+)\."
    r"(?P<field>trellis|suh|svh|mcg|mul1)$"
)

ShardId = str | int | tuple[int, ...] | None


def _load_exl3_ext() -> Any:
    """Load the existing ExLlamaV3 extension only from an actual CUDA call."""

    global _EXL3_EXT
    if _EXL3_EXT is not None:
        return _EXL3_EXT

    shim = os.environ.get("VLLM_EXL3_ABI_SHIM")
    if shim:
        ctypes.CDLL(shim, mode=ctypes.RTLD_GLOBAL)

    ext_path = os.environ.get("VLLM_EXL3_EXT_PATH")
    if ext_path:
        search_dir = ext_path if os.path.isdir(ext_path) else os.path.dirname(ext_path)
        if search_dir and search_dir not in sys.path:
            sys.path.insert(0, search_dir)

    try:
        ext = importlib.import_module("exllamav3_ext")
    except Exception as exc:
        hint = (
            "Set VLLM_EXL3_EXT_PATH to the directory containing "
            "exllamav3_ext*.so (and VLLM_EXL3_ABI_SHIM when the local "
            "PyTorch ABI shim is required)."
        )
        raise RuntimeError(f"Unable to import exllamav3_ext. {hint}") from exc

    if not hasattr(ext, "exl3_gemm"):
        raise RuntimeError(
            "The imported exllamav3_ext does not export exl3_gemm; rebuild the "
            "track_a_retile extension used by this overlay."
        )
    _EXL3_EXT = ext
    return ext


def _load_sparkinfer_fused_moe() -> Any:
    """Resolve the public unified MoE API lazily."""

    global _SPARKINFER_FUSED_MOE_API
    if _SPARKINFER_FUSED_MOE_API is not None:
        return _SPARKINFER_FUSED_MOE_API
    try:
        from sparkinfer.moe import fused_moe
    except Exception as exc:
        raise RuntimeError(
            "Rank-sliced EXL3 requires the exl3_trellis_mcg source in "
            "sparkinfer.moe.fused_moe. Install a matching Sparkinfer build."
        ) from exc
    _SPARKINFER_FUSED_MOE_API = fused_moe
    return fused_moe


def _load_sparkinfer_mixed_trellis() -> Any:
    """Resolve the one-grid mixed-bitrate Trellis API lazily."""

    global _SPARKINFER_MIXED_TRELLIS_API
    if _SPARKINFER_MIXED_TRELLIS_API is not None:
        return _SPARKINFER_MIXED_TRELLIS_API
    try:
        module = importlib.import_module(
            "sparkinfer.moe._shared.kernels.w4a16.mixed_trellis"
        )
        prepare = importlib.import_module(
            "sparkinfer.moe._shared.kernels.w4a16.prepare"
        )
        host = importlib.import_module("sparkinfer.moe._shared.kernels.w4a16.host")
    except Exception as exc:
        raise RuntimeError(
            "Mixed-bitrate rank-sliced EXL3 requires the matching SparkInfer "
            "mixed_trellis implementation."
        ) from exc
    api = SimpleNamespace(
        build_tiered_maps=module.build_tiered_maps,
        combine_trellis_rotations=module.combine_trellis_rotations,
        compile_mixed_trellis=module.compile_mixed_trellis,
        make_mixed_trellis_buffers=module.make_mixed_trellis_buffers,
        max_packed_route_slots=host.max_packed_route_slots,
        prepare_weights=prepare.prepare_trellis256_moe_weights,
        run_mixed_trellis=module.run_mixed_trellis,
    )
    _SPARKINFER_MIXED_TRELLIS_API = api
    return api


def _unique_tensor_storage_bytes(*buffers: Any) -> int:
    """Count unique tensor storage while ignoring buffer metadata fields."""

    total = 0
    seen: set[tuple[int, int]] = set()
    for buffers_ in buffers:
        for value in vars(buffers_).values():
            if not isinstance(value, torch.Tensor):
                continue
            storage = value.untyped_storage()
            storage_key = (storage.data_ptr(), storage.nbytes())
            if storage_key not in seen:
                seen.add(storage_key)
                total += storage.nbytes()
    return total


def _positive_env_int(name: str, default: int) -> int:
    # An env var that is present but blank means "unset". Compose and Kubernetes
    # both render an unset variable as the empty string, so int("") would abort
    # engine startup with a bare
    #   ValueError: invalid literal for int() with base 10: ''
    # that names neither the variable nor the fix.
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(
            f"{name} must be a positive integer or unset, got {raw!r}"
        ) from None
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


@torch.library.custom_op(
    "vllm::exl3_gemm",
    mutates_args=(),
    device_types="cuda",
)
def _exl3_gemm(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    """Opaque torch op around the bit-faithful ExLlamaV3 dense call."""

    ext = _load_exl3_ext()
    output = torch.empty(
        (x.shape[0], trellis.shape[1] * 16),
        dtype=torch.float16,
        device=x.device,
    )
    x_had = torch.empty_like(x)
    ext.exl3_gemm(
        x,
        trellis,
        output,
        suh,
        x_had,
        svh,
        -1,
        mcg,
        mul1,
        0,
    )
    return output


@_exl3_gemm.register_fake
def _exl3_gemm_fake(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    del suh, svh, mcg, mul1
    return torch.empty(
        (x.shape[0], trellis.shape[1] * 16),
        dtype=torch.float16,
        device=x.device,
    )


class Exl3Config(QuantizationConfig):
    """Configuration for modern and legacy EXL3 trellis checkpoints."""

    def __init__(
        self,
        bits: float | None = None,
        head_bits: float | None = None,
        codebook: str | None = None,
        version: str | None = None,
        tensor_storage: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.bits = bits
        self.head_bits = head_bits
        self.codebook = codebook
        self.version = version
        self.tensor_storage = tensor_storage or {}
        self._eager_checked = False
        self.rank_sliced_metadata: dict[str, Any] | None = None
        self.rank_sliced_k_values: tuple[int, ...] | None = None
        self.rank_sliced_bits_by_layer: dict[int, tuple[int, ...]] = {}

    def get_name(self) -> str:
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # The kernel boundary is always fp16.  BF16 model activations are cast
        # in apply() and converted back after the fp16 bias addition.
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Exl3Config:
        return cls(
            bits=config.get("bits"),
            head_bits=config.get("head_bits"),
            codebook=config.get("codebook"),
            version=config.get("version"),
            tensor_storage=config.get("tensor_storage"),
        )

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: PretrainedConfig | None = None,
    ) -> str | None:
        del hf_quant_cfg
        if user_quant is not None and user_quant != "exl3":
            return None
        metadata = getattr(hf_config, "hybrid_tr3_tail", None)
        if isinstance(metadata, dict) and metadata.get("format") == _RANK_SLICED_FORMAT:
            return "exl3"
        return None

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: PretrainedConfig | None = None,
        revision: str | None = None,
    ) -> None:
        rank_sliced = getattr(hf_config, "hybrid_tr3_tail", None)
        if (
            isinstance(rank_sliced, dict)
            and rank_sliced.get("format") == _RANK_SLICED_FORMAT
        ):
            self._configure_rank_sliced(rank_sliced)
            if self.rank_sliced_k_values is not None:
                resolved_revision = revision
                if resolved_revision is None and hf_config is not None:
                    resolved_revision = getattr(hf_config, "_commit_hash", None)
                self._load_rank_sliced_bitrates(
                    model_name,
                    revision=resolved_revision,
                )
            return

        # vLLM returns the summary embedded in config.json without consulting
        # get_config_filenames().  Hydrate the per-module records explicitly.
        if not self.tensor_storage:
            resolved_revision = revision
            if resolved_revision is None and hf_config is not None:
                resolved_revision = getattr(hf_config, "_commit_hash", None)
            config = get_hf_file_to_dict(
                "quantization_config.json",
                model_name,
                revision=resolved_revision,
            )
            if not config or not config.get("tensor_storage"):
                raise ValueError(
                    "EXL3 requires quantization_config.json with a non-empty "
                    "tensor_storage map. For branch-indexed Hugging Face repos, "
                    "download/serve an actual bpw revision rather than main."
                )
            self.bits = config.get("bits", self.bits)
            self.head_bits = config.get("head_bits", self.head_bits)
            self.codebook = config.get("codebook", self.codebook)
            self.version = config.get("version", self.version)
            self.tensor_storage = config["tensor_storage"]

        self._validate_storage_metadata()
        self._force_independent_lm_head(hf_config)

    def _configure_rank_sliced(self, metadata: dict[str, Any]) -> None:
        required = {
            "bits",
            "codebook",
            "experts_per_layer",
            "moe_layers",
            "tensor_schema",
            "tp",
        }
        missing = sorted(required.difference(metadata))
        if missing:
            raise ValueError(
                "rank-sliced EXL3 metadata is missing: " + ", ".join(missing)
            )
        if metadata["codebook"] != "mcg":
            raise ValueError(
                "rank-sliced EXL3 currently requires the MCG codebook, got "
                f"{metadata['codebook']!r}"
            )
        layers = metadata["moe_layers"]
        if (
            not isinstance(layers, list)
            or len(layers) != 2
            or int(layers[0]) < 0
            or int(layers[1]) < int(layers[0])
        ):
            raise ValueError("rank-sliced EXL3 moe_layers must be [first, last]")
        expected_schema = (
            "model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}"
        )
        if metadata["tensor_schema"] != expected_schema:
            raise ValueError(
                "unsupported rank-sliced EXL3 tensor schema: "
                f"{metadata['tensor_schema']!r}"
            )
        self.rank_sliced_metadata = dict(metadata)
        bits_field = metadata["bits"]
        if isinstance(bits_field, str) and bits_field.strip().lower() == "mixed":
            k_values = tuple(
                sorted({int(value) for value in metadata.get("k_values", ())})
            )
            if not k_values or any(value not in (3, 4, 5, 6) for value in k_values):
                raise ValueError(
                    "mixed rank-sliced EXL3 requires k_values within 3..6, got "
                    f"{metadata.get('k_values')!r}"
                )
            if not isinstance(metadata.get("bits_per_expert"), str):
                raise ValueError(
                    "mixed rank-sliced EXL3 requires a bits_per_expert JSON reference"
                )
            self.bits = None
            self.rank_sliced_k_values = k_values
        else:
            self.bits = float(bits_field)
            self.rank_sliced_k_values = None
        self.codebook = str(metadata["codebook"])
        self.version = str(metadata.get("exllamav3_version", "rank-sliced"))

    def _load_rank_sliced_bitrates(
        self,
        model_name: str,
        *,
        revision: str | None,
    ) -> None:
        assert self.rank_sliced_metadata is not None
        reference = str(self.rank_sliced_metadata["bits_per_expert"])
        try:
            filename, field = reference.rsplit(":", 1)
        except ValueError as exc:
            raise ValueError(
                "rank-sliced EXL3 bits_per_expert must use 'file.json:field' syntax, "
                f"got {reference!r}"
            ) from exc
        payload = get_hf_file_to_dict(filename, model_name, revision=revision)
        if not isinstance(payload, dict):
            raise ValueError(f"rank-sliced EXL3 could not load {filename!r}")

        experts = int(self.rank_sliced_metadata["experts_per_layer"])
        first, last = (int(value) for value in self.rank_sliced_metadata["moe_layers"])
        allowed = set(self.rank_sliced_k_values or ())
        by_layer: dict[int, tuple[int, ...]] = {}
        for layer_index in range(first, last + 1):
            entry = payload.get(str(layer_index))
            if not isinstance(entry, dict):
                raise ValueError(
                    f"rank-sliced EXL3 bitrate map is missing layer {layer_index}"
                )
            raw = entry.get(field)
            # The GLM-5.2 MTP overlay records all routed experts under tail_tr3
            # instead of repeating a 256-entry K3 vector.
            if raw is None and len(entry.get("tail_tr3", ())) == experts:
                raw = [3] * experts
            if not isinstance(raw, list) or len(raw) != experts:
                raise ValueError(
                    "rank-sliced EXL3 bitrate map must contain one entry per expert: "
                    f"layer={layer_index}, field={field!r}, expected={experts}"
                )
            bitrates = tuple(int(value) for value in raw)
            unexpected = sorted(set(bitrates).difference(allowed))
            if unexpected:
                raise ValueError(
                    f"rank-sliced EXL3 layer {layer_index} uses undeclared bitrates "
                    f"{unexpected}; declared={sorted(allowed)}"
                )
            by_layer[layer_index] = bitrates
        self.rank_sliced_bits_by_layer = by_layer

    def rank_sliced_layer_bitrates(self, layer_name: str) -> tuple[int, ...]:
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
        if match is None:
            raise ValueError(
                f"cannot resolve rank-sliced EXL3 layer index from {layer_name!r}"
            )
        layer_index = int(match.group(1))
        if self.rank_sliced_k_values is None:
            if self.bits is None or float(self.bits) != int(self.bits):
                raise ValueError(f"invalid uniform EXL3 bitrate {self.bits!r}")
            experts = int(self.rank_sliced_metadata["experts_per_layer"])
            return (int(self.bits),) * experts
        try:
            return self.rank_sliced_bits_by_layer[layer_index]
        except KeyError as exc:
            raise ValueError(
                f"rank-sliced EXL3 bitrate map has no layer {layer_index}"
            ) from exc

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        # Keep both spellings: loader prefixes use vLLM names, while packed
        # source-matrix discovery intentionally refers to the unstacked HF name.
        mapped = hf_to_vllm_mapper.apply_dict(self.tensor_storage)
        self.tensor_storage = {**self.tensor_storage, **mapped}

    def _validate_storage_metadata(self) -> None:
        bad: list[str] = []
        exl3_count = 0
        for prefix, entry in self.tensor_storage.items():
            if entry.get("quant_format") != "exl3":
                continue
            exl3_count += 1
            stored = entry.get("stored_tensors", {})
            suffixes = {name.rsplit(".", 1)[-1] for name in stored}
            required = {"trellis"}
            if not ({"suh", "su"} & suffixes):
                required.add("suh|su")
            if not ({"svh", "sv"} & suffixes):
                required.add("svh|sv")
            missing = [name for name in required if name not in suffixes]
            if missing:
                bad.append(f"{prefix}: missing {','.join(sorted(missing))}")
            if {"mcg", "mul1"} <= suffixes:
                bad.append(f"{prefix}: both mcg and mul1 are present")
        if not exl3_count:
            raise ValueError("quantization_config.json has no EXL3 tensor records")
        if bad:
            raise ValueError("Invalid EXL3 tensor metadata: " + "; ".join(bad[:16]))

    def _force_independent_lm_head(self, hf_config: PretrainedConfig | None) -> None:
        if hf_config is None or not self.has_quantized_lm_head():
            return
        configs: list[Any] = [hf_config]
        try:
            text_config = hf_config.get_text_config()
        except (AttributeError, TypeError):
            text_config = None
        if text_config is not None and text_config is not hf_config:
            configs.append(text_config)
        changed = False
        for config in configs:
            if getattr(config, "tie_word_embeddings", False):
                config.tie_word_embeddings = False
                changed = True
        if changed:
            logger.warning_once(
                "EXL3 metadata contains an independently quantized lm_head; "
                "overriding tie_word_embeddings so vLLM instantiates it."
            )

    def _require_enforce_eager(self) -> None:
        if self.rank_sliced_metadata is not None:
            # The routed-expert fast path is eagerly planned before graph
            # capture. Only its large-M parity fallback remains eager.
            return
        # exllamav3_ext's exl3_gemm autotunes with timing launches on the first
        # call per (m-bucket, k, n, K) shape hash; under CUDA-graph capture
        # those launches fault, and m-bucketing means a warmup pass cannot
        # reliably cover every bucket. Fail fast at build time instead of
        # faulting mid-capture.
        if self._eager_checked:
            return
        self._eager_checked = True
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None:
            return
        if not vllm_config.model_config.enforce_eager:
            raise ValueError(
                "The EXL3 quantization backend requires eager execution: "
                "pass --enforce-eager (enforce_eager=True). exl3_gemm "
                "autotunes with timing launches on first use per shape "
                "bucket, which is incompatible with CUDA-graph capture."
            )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        self._require_enforce_eager()
        is_lm_head = layer.__class__.__name__ == "ParallelLMHead"
        if is_lm_head and not prefix:
            prefix = "lm_head"
        if isinstance(layer, LinearBase) or is_lm_head:
            if not self._linear_prefix_is_exl3(prefix):
                return UnquantizedLinearMethod()
            return Exl3LinearMethod(self)
        if isinstance(layer, RoutedExperts):
            if not self._moe_prefix_is_exl3(prefix, layer):
                return None
            return Exl3MoEMethod(self, layer.moe_config)
        return None

    def _storage_entry(self, prefix: str) -> dict[str, Any] | None:
        candidates = [prefix]
        if prefix.startswith("model."):
            candidates.append(prefix.removeprefix("model."))
        else:
            candidates.append(f"model.{prefix}")

        # Multimodal wrappers often add an extra `model` or `language_model`
        # segment relative to vLLM's text-only module — interior
        # (`model.language_model.layers...`) or leading
        # (`language_model.lm_head`), so leading segments collapse too.
        parts = prefix.split(".")
        for removable in ("model", "language_model"):
            for idx in range(0, len(parts) - 1):
                if parts[idx] != removable:
                    continue
                collapsed = ".".join(parts[:idx] + parts[idx + 1 :])
                candidates.extend((collapsed, f"model.{collapsed}"))
                if collapsed.startswith("model."):
                    candidates.append(collapsed.removeprefix("model."))

        for candidate in dict.fromkeys(candidates):
            entry = self.tensor_storage.get(candidate)
            if entry is not None:
                return entry
        return None

    def _is_exl3_prefix(self, prefix: str) -> bool:
        entry = self._storage_entry(prefix)
        return entry is not None and entry.get("quant_format") == "exl3"

    def _linear_prefix_is_exl3(self, prefix: str) -> bool:
        if self._is_exl3_prefix(prefix):
            return True
        leaf = prefix.rsplit(".", 1)[-1]
        source_leaves = self.packed_modules_mapping.get(leaf)
        if not source_leaves:
            return False
        base = prefix.rsplit(".", 1)[0] if "." in prefix else ""
        return all(
            self._is_exl3_prefix(f"{base}.{source}" if base else source)
            for source in source_leaves
        )

    def _moe_prefix_is_exl3(
        self, prefix: str, layer: torch.nn.Module | None = None
    ) -> bool:
        if self.rank_sliced_metadata is not None:
            match = re.search(r"layers\.(\d+)\b", prefix)
            if match is None:
                return False
            first, last = (int(v) for v in self.rank_sliced_metadata["moe_layers"])
            return first <= int(match.group(1)) <= last
        # Use the layer's checkpoint projection names (the same fields
        # _validate_codebooks keys off) so remapped-projection MoE
        # checkpoints are still detected; fall back to the defaults when the
        # layer variant does not carry them.
        projections = tuple(
            getattr(layer, attr, default)
            for attr, default in (
                ("ckpt_gate_proj_name", "gate_proj"),
                ("ckpt_up_proj_name", "up_proj"),
                ("ckpt_down_proj_name", "down_proj"),
            )
        )
        expert_prefixes = (f"{prefix}.0", f"{prefix}.experts.0")
        return any(
            all(
                self._is_exl3_prefix(f"{expert}.{projection}")
                for projection in projections
            )
            for expert in expert_prefixes
        )

    def codebook_for_prefix(self, prefix: str) -> str | None:
        if self.rank_sliced_metadata is not None:
            match = re.search(r"layers\.(\d+)\b", prefix)
            if match is None:
                return None
            first, last = (int(v) for v in self.rank_sliced_metadata["moe_layers"])
            return "mcg" if first <= int(match.group(1)) <= last else None
        entry = self._storage_entry(prefix)
        if entry is None:
            return None
        suffixes = {name.rsplit(".", 1)[-1] for name in entry.get("stored_tensors", {})}
        if "mcg" in suffixes:
            return "mcg"
        if "mul1" in suffixes:
            return "mul1"
        return None

    def has_quantized_lm_head(self) -> bool:
        return self._is_exl3_prefix("lm_head")

    def normalize_rank_sliced_weight_name(self, name: str) -> str | None:
        """Drop non-local TP payloads and remove the serialized rank segment."""
        if self.rank_sliced_metadata is None:
            return name
        match = _RANK_SLICED_WEIGHT_RE.match(name)
        if match is None:
            return name
        if int(match.group("rank")) != get_tensor_model_parallel_rank():
            return None
        return f"{match.group('prefix')}.{match.group('field')}"


class Exl3Parameter(BasevLLMParameter):
    """Zero-sized parameter holding independently shaped EXL3 components."""

    def __new__(cls, *, weight_loader):
        data = torch.empty(0, dtype=torch.uint8)
        return super().__new__(cls, data=data, weight_loader=weight_loader)

    def __init__(self, *, weight_loader):
        self.exl3_tensors: dict[ShardId, torch.Tensor] = {}
        super().__init__(data=self.data, weight_loader=weight_loader)

    def load_exl3_weight(
        self,
        loaded_weight: torch.Tensor,
        shard_id: ShardId = None,
    ) -> None:
        self.exl3_tensors[shard_id] = loaded_weight.contiguous()


def _exl3_weight_loader(
    param: Exl3Parameter,
    loaded_weight: torch.Tensor,
    loaded_shard_id: ShardId = None,
) -> None:
    param.load_exl3_weight(loaded_weight, loaded_shard_id)


class Exl3LinearMethod(LinearMethodBase):
    def __init__(self, quant_config: Exl3Config) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del params_dtype, extra_weight_attrs
        if layer.__class__.__name__ == "ParallelLMHead":
            org = getattr(layer, "org_vocab_size", None)
            total = getattr(layer, "num_embeddings", None)
            if org is not None and total is not None and org != total:
                raise NotImplementedError(
                    "EXL3 lm_head with added vocabulary is unsupported: the "
                    f"trellis tensor covers the original {org} rows but the "
                    f"layer allocates {total}; TP slicing would silently "
                    "misalign. Strip --lora-extra-vocab-size / added tokens "
                    "or leave lm_head unquantized."
                )
        # Respect the layer's effective topology. disable_tp linears set their
        # own tp_size=1, while ReplicatedLinear carries full weights even when
        # the process-wide tensor group is larger than one.
        if isinstance(layer, ReplicatedLinear):
            layer.exl3_tp_rank = 0
            layer.exl3_tp_size = 1
        else:
            layer.exl3_tp_rank = getattr(
                layer, "tp_rank", get_tensor_model_parallel_rank()
            )
            layer.exl3_tp_size = getattr(
                layer, "tp_size", get_tensor_model_parallel_world_size()
            )
        layer.exl3_input_size = input_size
        layer.exl3_input_size_per_partition = input_size_per_partition
        layer.exl3_output_size = output_size
        layer.exl3_output_partition_sizes = output_partition_sizes
        layer.exl3_shard_ids = self._shard_ids_for_layer(layer, output_partition_sizes)
        layer.exl3_parallel_mode = (
            "row" if input_size_per_partition != input_size else "column"
        )
        source_prefixes = self._source_prefixes_for_layer(layer, layer.exl3_shard_ids)
        layer.exl3_expected_codebooks = {
            shard_id: self.quant_config.codebook_for_prefix(source_prefix)
            for shard_id, source_prefix in zip(
                layer.exl3_shard_ids, source_prefixes, strict=True
            )
        }

        # su/sv are legacy packed sign bitfields.  Modern checkpoints load
        # suh/svh directly.
        for name in ("suh", "svh", "su", "sv", "trellis", "mcg", "mul1"):
            layer.register_parameter(
                name,
                Exl3Parameter(weight_loader=_exl3_weight_loader),
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._materialize_legacy_hadamard(layer)
        missing: list[str] = []
        for attr in ("suh", "svh", "trellis"):
            param = getattr(layer, attr)
            for shard_id in layer.exl3_shard_ids:
                if shard_id not in param.exl3_tensors:
                    missing.append(f"{attr}[{shard_id!r}]")
        for shard_id in layer.exl3_shard_ids:
            expected = layer.exl3_expected_codebooks[shard_id]
            has_mcg = shard_id in layer.mcg.exl3_tensors
            has_mul1 = shard_id in layer.mul1.exl3_tensors
            if has_mcg and has_mul1:
                missing.append(f"codebook[{shard_id!r}]=both mcg and mul1")
            elif expected == "mcg" and not has_mcg:
                missing.append(f"mcg[{shard_id!r}]")
            elif expected == "mul1" and not has_mul1:
                missing.append(f"mul1[{shard_id!r}]")
            elif expected is None and (has_mcg or has_mul1):
                missing.append(f"unexpected codebook[{shard_id!r}]")
        if missing:
            prefix = getattr(layer, "prefix", layer.__class__.__name__)
            raise ValueError(
                f"Missing or inconsistent EXL3 tensors for {prefix}: "
                + ", ".join(missing)
            )

        self._validate_loaded_tensors(layer)
        self._shard_tensors_for_tensor_parallel(layer)
        self._validate_loaded_tensors(layer)

        # device_loading_context has moved the zero-sized registered parameter
        # to the model target device.  Its device is the safest destination for
        # the tensors kept in the side dictionaries.
        device = layer.trellis.device
        for attr in ("suh", "svh", "trellis", "mcg", "mul1"):
            param = getattr(layer, attr)
            for shard_id, tensor in list(param.exl3_tensors.items()):
                param.exl3_tensors[shard_id] = tensor.to(
                    device=device, non_blocking=True
                ).contiguous()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_shape = x.shape[:-1]
        original_dtype = x.dtype
        x_2d = x.reshape(-1, x.shape[-1]).to(torch.float16).contiguous()
        outputs = [
            self._apply_one(layer, x_2d, shard_id) for shard_id in layer.exl3_shard_ids
        ]
        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        if bias is not None:
            output = output + bias.to(dtype=output.dtype)
        output = output.reshape(*original_shape, output.shape[-1])
        return output if output.dtype == original_dtype else output.to(original_dtype)

    @staticmethod
    def _unpack_signs(bitfield: torch.Tensor) -> torch.Tensor:
        words = bitfield.contiguous().view(torch.uint16).to(torch.int32)
        masks = 1 << torch.arange(16, device=words.device, dtype=torch.int32)
        negative = (words.unsqueeze(-1) & masks) != 0
        return (
            torch.where(
                negative,
                torch.tensor(-1.0, device=words.device, dtype=torch.float16),
                torch.tensor(1.0, device=words.device, dtype=torch.float16),
            )
            .flatten()
            .contiguous()
        )

    @classmethod
    def _materialize_legacy_hadamard(cls, layer: torch.nn.Module) -> None:
        for packed_name, half_name in (("su", "suh"), ("sv", "svh")):
            packed = getattr(layer, packed_name).exl3_tensors
            half = getattr(layer, half_name).exl3_tensors
            for shard_id in layer.exl3_shard_ids:
                if shard_id not in half and shard_id in packed:
                    half[shard_id] = cls._unpack_signs(packed[shard_id])

    @staticmethod
    def _validate_marker(tensor: torch.Tensor, expected: int, name: str) -> None:
        if tensor.dtype != torch.int32 or tensor.numel() != 1:
            raise ValueError(f"EXL3 {name} must be a scalar int32 sentinel")
        value = int(tensor.reshape(()).item()) & 0xFFFFFFFF
        if value != expected:
            raise ValueError(
                f"Invalid EXL3 {name} sentinel 0x{value:08x}; expected 0x{expected:08x}"
            )

    @classmethod
    def _validate_loaded_tensors(cls, layer: torch.nn.Module) -> None:
        for shard_id in layer.exl3_shard_ids:
            trellis = layer.trellis.exl3_tensors[shard_id]
            suh = layer.suh.exl3_tensors[shard_id]
            svh = layer.svh.exl3_tensors[shard_id]
            if trellis.dtype != torch.int16 or trellis.ndim != 3:
                raise ValueError("EXL3 trellis must be rank-3 int16")
            if trellis.shape[2] % 16 or not 1 <= trellis.shape[2] // 16 <= 8:
                raise ValueError(
                    f"Invalid EXL3 trellis bit width {trellis.shape[2]} / 16"
                )
            if suh.dtype != torch.float16 or suh.ndim != 1:
                raise ValueError("EXL3 suh must be rank-1 float16")
            if svh.dtype != torch.float16 or svh.ndim != 1:
                raise ValueError("EXL3 svh must be rank-1 float16")
            k = trellis.shape[0] * 16
            n = trellis.shape[1] * 16
            if suh.numel() != k or svh.numel() != n:
                raise ValueError(
                    "EXL3 dimensions disagree: "
                    f"trellis={tuple(trellis.shape)}, suh={suh.numel()}, "
                    f"svh={svh.numel()}"
                )
            if k % _HADAMARD_BLOCK or n % _HADAMARD_BLOCK:
                raise ValueError(
                    f"EXL3 kernel dimensions must be {_HADAMARD_BLOCK}-aligned, "
                    f"got K={k}, N={n}"
                )
            if shard_id in layer.mcg.exl3_tensors:
                cls._validate_marker(
                    layer.mcg.exl3_tensors[shard_id], _MCG_SENTINEL, "mcg"
                )
            if shard_id in layer.mul1.exl3_tensors:
                cls._validate_marker(
                    layer.mul1.exl3_tensors[shard_id], _MUL1_SENTINEL, "mul1"
                )

    @staticmethod
    def _slice_exl3_tensor(
        tensor: torch.Tensor,
        *,
        dim: int,
        start: int,
        size: int,
    ) -> torch.Tensor:
        if start % _HADAMARD_BLOCK or size % _HADAMARD_BLOCK:
            axis = "output" if dim == 1 else "input"
            raise ValueError(
                f"EXL3 TP {axis} slice must be {_HADAMARD_BLOCK}-aligned, "
                f"got start={start}, size={size}"
            )
        return tensor.narrow(dim, start // 16, size // 16).contiguous()

    @staticmethod
    def _output_shard_size(layer: torch.nn.Module, shard_id: ShardId) -> int:
        if shard_id is None:
            return layer.exl3_output_partition_sizes[0]
        if isinstance(shard_id, str) and shard_id in ("q", "k", "v"):
            return layer.exl3_output_partition_sizes[{"q": 0, "k": 1, "v": 2}[shard_id]]
        if isinstance(shard_id, tuple):
            return sum(layer.exl3_output_partition_sizes[idx] for idx in shard_id)
        if isinstance(shard_id, int):
            return layer.exl3_output_partition_sizes[shard_id]
        return layer.exl3_output_partition_sizes[layer.exl3_shard_ids.index(shard_id)]

    @staticmethod
    def _qkv_output_start(
        layer: torch.nn.Module, shard_id: ShardId, shard_size: int
    ) -> int:
        if shard_id in ("k", "v"):
            shard_rank = layer.exl3_tp_rank // layer.num_kv_head_replicas
        else:
            shard_rank = layer.exl3_tp_rank
        return shard_rank * shard_size

    @classmethod
    def _shard_tensors_for_tensor_parallel(cls, layer: torch.nn.Module) -> None:
        if layer.exl3_tp_size == 1:
            return
        if layer.exl3_parallel_mode == "row":
            start = layer.exl3_tp_rank * layer.exl3_input_size_per_partition
            size = layer.exl3_input_size_per_partition
            for shard_id in layer.exl3_shard_ids:
                layer.suh.exl3_tensors[shard_id] = (
                    layer.suh.exl3_tensors[shard_id].narrow(0, start, size).contiguous()
                )
                layer.trellis.exl3_tensors[shard_id] = cls._slice_exl3_tensor(
                    layer.trellis.exl3_tensors[shard_id],
                    dim=0,
                    start=start,
                    size=size,
                )
            return

        already_sharded = cls._expand_tuple_output_shards(layer)
        for shard_id in layer.exl3_shard_ids:
            if shard_id in already_sharded:
                continue
            size = cls._output_shard_size(layer, shard_id)
            start = cls._qkv_output_start(layer, shard_id, size)
            layer.svh.exl3_tensors[shard_id] = (
                layer.svh.exl3_tensors[shard_id].narrow(0, start, size).contiguous()
            )
            layer.trellis.exl3_tensors[shard_id] = cls._slice_exl3_tensor(
                layer.trellis.exl3_tensors[shard_id],
                dim=1,
                start=start,
                size=size,
            )

    @classmethod
    def _expand_tuple_output_shards(cls, layer: torch.nn.Module) -> set[int]:
        tuples = [sid for sid in layer.exl3_shard_ids if isinstance(sid, tuple)]
        if not tuples:
            return set()

        expanded_ids: list[ShardId] = []
        component_ids: set[int] = set()
        for shard_id in layer.exl3_shard_ids:
            if isinstance(shard_id, tuple):
                expanded_ids.extend(shard_id)
                component_ids.update(shard_id)
            else:
                expanded_ids.append(shard_id)

        for tuple_id in tuples:
            full_offsets: dict[int, int] = {}
            offset = 0
            for idx in tuple_id:
                full_offsets[idx] = offset
                offset += layer.exl3_output_partition_sizes[idx] * layer.exl3_tp_size
            for idx in tuple_id:
                size = layer.exl3_output_partition_sizes[idx]
                start = full_offsets[idx] + layer.exl3_tp_rank * size
                layer.suh.exl3_tensors[idx] = layer.suh.exl3_tensors[tuple_id]
                layer.svh.exl3_tensors[idx] = (
                    layer.svh.exl3_tensors[tuple_id].narrow(0, start, size).contiguous()
                )
                layer.trellis.exl3_tensors[idx] = cls._slice_exl3_tensor(
                    layer.trellis.exl3_tensors[tuple_id],
                    dim=1,
                    start=start,
                    size=size,
                )
                layer.exl3_expected_codebooks[idx] = layer.exl3_expected_codebooks[
                    tuple_id
                ]
                for marker in ("mcg", "mul1"):
                    tensors = getattr(layer, marker).exl3_tensors
                    if tuple_id in tensors:
                        tensors[idx] = tensors[tuple_id]
            for attr in ("suh", "svh", "trellis", "mcg", "mul1"):
                getattr(layer, attr).exl3_tensors.pop(tuple_id, None)
            layer.exl3_expected_codebooks.pop(tuple_id, None)

        layer.exl3_shard_ids = expanded_ids
        return component_ids

    @staticmethod
    def _shard_ids_for_layer(
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
    ) -> list[ShardId]:
        if len(output_partition_sizes) == 1:
            return [None]
        prefix = getattr(layer, "prefix", "")
        if isinstance(layer, QKVParallelLinear) and len(output_partition_sizes) == 3:
            return ["q", "k", "v"]
        if prefix.endswith("in_proj_qkvz"):
            return [(0, 1, 2), 3]
        return list(range(len(output_partition_sizes)))

    def _source_prefixes_for_layer(
        self, layer: torch.nn.Module, shard_ids: list[ShardId]
    ) -> list[str]:
        prefix = getattr(layer, "prefix", "")
        if len(shard_ids) == 1:
            return [prefix or "lm_head"]
        leaf = prefix.rsplit(".", 1)[-1]
        base = prefix.rsplit(".", 1)[0] if "." in prefix else ""
        sources = self.quant_config.packed_modules_mapping.get(leaf)
        if sources and len(sources) == len(shard_ids):
            return [f"{base}.{source}" if base else source for source in sources]
        raise ValueError(
            f"EXL3 does not know the source matrices for packed layer {prefix}; "
            "add it to the model's packed_modules_mapping."
        )

    @staticmethod
    def _apply_one(
        layer: torch.nn.Module, x: torch.Tensor, shard_id: ShardId
    ) -> torch.Tensor:
        trellis = layer.trellis.exl3_tensors[shard_id]
        packed_k = trellis.shape[0] * 16
        if x.shape[-1] > packed_k:
            raise ValueError(
                f"EXL3 input width {x.shape[-1]} exceeds packed K={packed_k}"
            )
        if x.shape[-1] < packed_k:
            x = torch.nn.functional.pad(x, (0, packed_k - x.shape[-1]))
        output = _exl3_gemm(
            x,
            trellis,
            layer.suh.exl3_tensors[shard_id],
            layer.svh.exl3_tensors[shard_id],
            shard_id in layer.mcg.exl3_tensors,
            shard_id in layer.mul1.exl3_tensors,
        )
        logical_n = Exl3LinearMethod._output_shard_size(layer, shard_id)
        if output.shape[-1] < logical_n:
            raise ValueError(
                f"EXL3 packed N={output.shape[-1]} is below logical N={logical_n}"
            )
        return output[..., :logical_n]


class Exl3MoEParameter(BasevLLMParameter):
    """EXL3 tensors keyed by expert/projection, optionally in one GPU slab."""

    def __new__(
        cls,
        *,
        weight_loader,
        num_experts: int = 0,
        shard_ids: tuple[str, ...] = (),
        preallocate: bool = False,
    ):
        del num_experts, shard_ids, preallocate
        data = torch.empty(0, dtype=torch.uint8)
        return super().__new__(cls, data=data, weight_loader=weight_loader)

    def __init__(
        self,
        *,
        weight_loader,
        num_experts: int = 0,
        shard_ids: tuple[str, ...] = (),
        preallocate: bool = False,
    ):
        self.exl3_tensors: dict[tuple[int, str], torch.Tensor] = {}
        self.exl3_backing: torch.Tensor | None = None
        self.exl3_num_experts = int(num_experts)
        self.exl3_shard_ids = tuple(shard_ids)
        self.exl3_preallocate = bool(preallocate)
        super().__init__(data=self.data, weight_loader=weight_loader)

    def load_exl3_weight(
        self,
        loaded_weight: torch.Tensor,
        *,
        expert_id: int,
        shard_id: str,
    ) -> None:
        key = (int(expert_id), str(shard_id))
        if not self.exl3_preallocate:
            self.exl3_tensors[key] = loaded_weight.contiguous()
            return
        if self.exl3_num_experts <= 0 or shard_id not in self.exl3_shard_ids:
            raise ValueError(
                f"invalid EXL3 slab key expert={expert_id}, shard={shard_id!r}"
            )
        if not 0 <= int(expert_id) < self.exl3_num_experts:
            raise ValueError(
                f"EXL3 expert {expert_id} is outside [0, {self.exl3_num_experts})"
            )
        if self.device.type == "meta":
            raise RuntimeError("rank-sliced EXL3 slabs cannot be allocated on meta")
        if self.exl3_backing is None:
            prefix = (
                (len(self.exl3_shard_ids), self.exl3_num_experts)
                if len(self.exl3_shard_ids) > 1
                else (self.exl3_num_experts,)
            )
            self.exl3_backing = torch.empty(
                prefix + tuple(loaded_weight.shape),
                dtype=loaded_weight.dtype,
                device=self.device,
            )
        shard_index = self.exl3_shard_ids.index(shard_id)
        target = (
            self.exl3_backing[shard_index, expert_id]
            if len(self.exl3_shard_ids) > 1
            else self.exl3_backing[expert_id]
        )
        if tuple(target.shape) != tuple(loaded_weight.shape):
            raise ValueError(
                "rank-sliced EXL3 tensor shape changed within one slab: "
                f"expected={tuple(target.shape)}, got={tuple(loaded_weight.shape)}"
            )
        target.copy_(loaded_weight, non_blocking=True)
        self.exl3_tensors[key] = target


def _exl3_moe_weight_loader(
    param: Exl3MoEParameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    expert_id: int,
    return_success: bool = False,
) -> bool | None:
    del weight_name
    param.load_exl3_weight(
        loaded_weight,
        expert_id=expert_id,
        shard_id=shard_id,
    )
    return True if return_success else None


# Model loaders (e.g. llama4-style paths) check this attribute before routing
# expert tensors through a param's weight_loader with MoE kwargs.
_exl3_moe_weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]


class Exl3MoEMethod(FusedMoEMethodBase):
    """Correctness MoE path: route, then use three dense EXL3 GEMMs/expert."""

    def __init__(self, quant_config: Exl3Config, moe) -> None:
        super().__init__(moe)
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del extra_weight_attrs
        if params_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"EXL3 MoE requires BF16 or FP16 activations, got {params_dtype}"
            )
        if self.moe.moe_parallel_config.use_ep:
            raise NotImplementedError(
                "EXL3 correctness MoE currently supports TP but not expert parallelism"
            )
        if self.moe.has_bias:
            raise NotImplementedError(
                "EXL3 correctness MoE does not yet support expert biases"
            )
        layer.exl3_tp_rank = self.moe.moe_parallel_config.tp_rank
        layer.exl3_tp_size = self.moe.moe_parallel_config.tp_size
        layer.exl3_hidden_size = hidden_size
        layer.exl3_intermediate_size_per_partition = intermediate_size_per_partition
        layer.exl3_params_dtype = params_dtype
        rank_sliced = self.quant_config.rank_sliced_metadata is not None
        if rank_sliced:
            checkpoint_tp = int(self.quant_config.rank_sliced_metadata["tp"])
            if checkpoint_tp != layer.exl3_tp_size:
                raise ValueError(
                    "rank-sliced EXL3 checkpoint TP does not match runtime: "
                    f"checkpoint={checkpoint_tp}, runtime={layer.exl3_tp_size}"
                )
            expected_experts = int(
                self.quant_config.rank_sliced_metadata["experts_per_layer"]
            )
            if expected_experts != num_experts:
                raise ValueError(
                    "rank-sliced EXL3 expert count does not match the model: "
                    f"checkpoint={expected_experts}, model={num_experts}"
                )
            vllm_config = get_current_vllm_config_or_none()
            scheduler_config = (
                vllm_config.scheduler_config if vllm_config is not None else None
            )
            # No silent fallback: a wrong capacity here puts the target and the
            # rank-sliced MTP draft on different plans with no error, which is
            # exactly the class of mismatch that corrupts only at scale.
            if (
                scheduler_config is None
                or getattr(scheduler_config, "max_num_batched_tokens", None) is None
            ):
                raise ValueError(
                    "EXL3 rank-sliced MoE requires scheduler_config."
                    "max_num_batched_tokens to plan its Trellis arena; refusing to "
                    "guess a capacity."
                )
            layer.exl3_max_num_batched_tokens = int(
                scheduler_config.max_num_batched_tokens
            )
            # Stamp the layer role while the model-construction config context
            # is still active. Forward time (the profile pass and CUDA graph
            # capture) runs with no current vllm config, so neither name- nor
            # index-based detection can resolve the role there -- a GLM-5.2
            # style MTP head is named exactly like a target layer
            # (model.layers.<num_hidden_layers>.*). runner_type is minted as
            # "draft" for every model-backed speculator draft
            # (SpeculativeConfig.__post_init__ -> ModelConfig(runner="draft")),
            # which also covers speculators that bypass load_eagle_model.
            layer.exl3_is_draft = (
                getattr(vllm_config.model_config, "runner_type", None) == "draft"
            )
            layer.exl3_layer_bitrates = self.quant_config.rank_sliced_layer_bitrates(
                str(layer.layer_name)
            )
            layer.exl3_mixed_bitrate = len(set(layer.exl3_layer_bitrates)) > 1
        for prefix, shard_ids in (("w13", ("w1", "w3")), ("w2", ("w2",))):
            for suffix in ("suh", "svh", "trellis", "mcg", "mul1"):
                layer.register_parameter(
                    f"{prefix}_{suffix}",
                    Exl3MoEParameter(
                        weight_loader=_exl3_moe_weight_loader,
                        num_experts=num_experts,
                        shard_ids=shard_ids,
                        preallocate=rank_sliced
                        and suffix
                        in (
                            {"suh", "svh"}
                            if getattr(layer, "exl3_mixed_bitrate", False)
                            else {"suh", "svh", "trellis"}
                        ),
                    ),
                )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        required = {"w13": ("w1", "w3"), "w2": ("w2",)}
        missing: list[str] = []
        for prefix, shard_ids in required.items():
            for attr in ("suh", "svh", "trellis"):
                tensors = getattr(layer, f"{prefix}_{attr}").exl3_tensors
                for expert_id in range(layer.local_num_experts):
                    for shard_id in shard_ids:
                        if (expert_id, shard_id) not in tensors:
                            missing.append(f"{prefix}_{attr}[{expert_id},{shard_id}]")
        if missing:
            raise ValueError(
                f"Missing EXL3 MoE tensors for {layer.layer_name}: "
                + ", ".join(missing[:32])
                + (" ..." if len(missing) > 32 else "")
            )
        self._validate_codebooks(layer)
        if self.quant_config.rank_sliced_metadata is None:
            self._shard_tensors_for_tensor_parallel(layer)
        device = layer.w13_trellis.device
        for prefix in ("w13", "w2"):
            for attr in ("suh", "svh", "trellis", "mcg", "mul1"):
                param = getattr(layer, f"{prefix}_{attr}")
                for key, tensor in list(param.exl3_tensors.items()):
                    param.exl3_tensors[key] = tensor.to(
                        device=device, non_blocking=True
                    ).contiguous()
        self._validate_moe_shapes(layer)

        if self.quant_config.rank_sliced_metadata is not None:
            self._prepare_rank_sliced_weights(layer)
            return

    def _validate_codebooks(self, layer: RoutedExperts) -> None:
        projections = {
            "w1": layer.ckpt_gate_proj_name,
            "w2": layer.ckpt_down_proj_name,
            "w3": layer.ckpt_up_proj_name,
        }
        for expert_id in range(layer.local_num_experts):
            for shard_id, projection in projections.items():
                prefix = f"{layer.layer_name}.{expert_id}.{projection}"
                expected = self.quant_config.codebook_for_prefix(prefix)
                group = "w2" if shard_id == "w2" else "w13"
                key = (expert_id, shard_id)
                has_mcg = key in getattr(layer, f"{group}_mcg").exl3_tensors
                has_mul1 = key in getattr(layer, f"{group}_mul1").exl3_tensors
                if has_mcg and has_mul1:
                    raise ValueError(f"EXL3 MoE {prefix} has both codebooks")
                if expected == "mcg" and not has_mcg:
                    raise ValueError(f"EXL3 MoE {prefix} is missing mcg")
                if expected == "mul1" and not has_mul1:
                    raise ValueError(f"EXL3 MoE {prefix} is missing mul1")
                if expected is None and (has_mcg or has_mul1):
                    raise ValueError(
                        f"EXL3 MoE {prefix} has an unexpected codebook marker"
                    )
                if has_mcg:
                    Exl3LinearMethod._validate_marker(
                        getattr(layer, f"{group}_mcg").exl3_tensors[key],
                        _MCG_SENTINEL,
                        "mcg",
                    )
                if has_mul1:
                    Exl3LinearMethod._validate_marker(
                        getattr(layer, f"{group}_mul1").exl3_tensors[key],
                        _MUL1_SENTINEL,
                        "mul1",
                    )

    @classmethod
    def _shard_tensors_for_tensor_parallel(cls, layer: RoutedExperts) -> None:
        if layer.exl3_tp_size == 1:
            return
        start = layer.exl3_tp_rank * layer.exl3_intermediate_size_per_partition
        size = layer.exl3_intermediate_size_per_partition
        for expert_id in range(layer.local_num_experts):
            for shard_id in ("w1", "w3"):
                key = (expert_id, shard_id)
                layer.w13_svh.exl3_tensors[key] = (
                    layer.w13_svh.exl3_tensors[key].narrow(0, start, size).contiguous()
                )
                layer.w13_trellis.exl3_tensors[key] = (
                    Exl3LinearMethod._slice_exl3_tensor(
                        layer.w13_trellis.exl3_tensors[key],
                        dim=1,
                        start=start,
                        size=size,
                    )
                )
            key = (expert_id, "w2")
            layer.w2_suh.exl3_tensors[key] = (
                layer.w2_suh.exl3_tensors[key].narrow(0, start, size).contiguous()
            )
            layer.w2_trellis.exl3_tensors[key] = Exl3LinearMethod._slice_exl3_tensor(
                layer.w2_trellis.exl3_tensors[key],
                dim=0,
                start=start,
                size=size,
            )

    @staticmethod
    def _validate_moe_shapes(layer: RoutedExperts) -> None:
        for expert_id in range(layer.local_num_experts):
            for group, shard_ids in (("w13", ("w1", "w3")), ("w2", ("w2",))):
                for shard_id in shard_ids:
                    key = (expert_id, shard_id)
                    trellis = getattr(layer, f"{group}_trellis").exl3_tensors[key]
                    suh = getattr(layer, f"{group}_suh").exl3_tensors[key]
                    svh = getattr(layer, f"{group}_svh").exl3_tensors[key]
                    if (
                        trellis.dtype != torch.int16
                        or trellis.ndim != 3
                        or trellis.shape[2] % 16
                        or not 1 <= trellis.shape[2] // 16 <= 8
                        or suh.dtype != torch.float16
                        or suh.ndim != 1
                        or svh.dtype != torch.float16
                        or svh.ndim != 1
                        or suh.numel() != trellis.shape[0] * 16
                        or svh.numel() != trellis.shape[1] * 16
                        or (trellis.shape[0] * 16) % _HADAMARD_BLOCK
                        or (trellis.shape[1] * 16) % _HADAMARD_BLOCK
                    ):
                        raise ValueError(
                            f"Invalid EXL3 MoE tensors for expert={expert_id}, "
                            f"projection={shard_id}"
                        )

    @staticmethod
    def _rank_sliced_backing(
        layer: RoutedExperts,
        param_name: str,
    ) -> torch.Tensor:
        param = getattr(layer, param_name)
        backing = param.exl3_backing
        if backing is None or not backing.is_contiguous():
            raise RuntimeError(
                f"rank-sliced EXL3 parameter {param_name} has no contiguous slab"
            )
        for (expert_id, shard_id), tensor in param.exl3_tensors.items():
            shard_index = param.exl3_shard_ids.index(shard_id)
            expected = (
                backing[shard_index, expert_id]
                if len(param.exl3_shard_ids) > 1
                else backing[expert_id]
            )
            if tensor.data_ptr() != expected.data_ptr():
                raise RuntimeError(
                    "rank-sliced EXL3 expert payload lost its slab alias: "
                    f"{param_name}[{expert_id},{shard_id}]"
                )
        return backing

    @staticmethod
    def _pointer_table(slab: torch.Tensor) -> torch.Tensor:
        if slab.ndim < 2 or not slab[0].is_contiguous():
            raise RuntimeError("EXL3 pointer-table rows must be contiguous")
        step = slab.stride(0) * slab.element_size()
        base = slab.data_ptr()
        return torch.tensor(
            [base + expert_id * step for expert_id in range(slab.shape[0])],
            dtype=torch.int64,
            device=slab.device,
        )

    @staticmethod
    def _trellis_tile_config(hidden_size: int, intermediate_size: int):
        if hidden_size % 128 or intermediate_size % 128:
            raise ValueError(
                "rank-sliced EXL3 full rotations require hidden and "
                "intermediate dimensions divisible by 128"
            )
        if hidden_size % 256 == 0 and intermediate_size % 256 == 0:
            return (64, 256, 64, 256)
        return (64, 128, 64, 128)

    @staticmethod
    def _mixed_trellis_tile_config(hidden_size: int, intermediate_size: int):
        if hidden_size % 128 or intermediate_size % 128:
            raise ValueError(
                "mixed rank-sliced EXL3 requires hidden and intermediate "
                "dimensions divisible by 128"
            )
        # The mixed K3/K4 megakernel needs a 128-wide FC1 K tile. The older
        # 64x256 FC1 geometry loses partial reductions at large prefill M. A
        # 512-wide FC2 tile then removes the second persistent wave on GLM-5.2.
        if hidden_size % 512 == 0:
            return (128, 128, 32, 512)
        if hidden_size % 256 == 0:
            return (128, 128, 64, 256)
        return (128, 128, 128, 128)

    def _prepare_mixed_rank_sliced_weights(self, layer: RoutedExperts) -> None:
        api = _load_sparkinfer_mixed_trellis()
        num_experts = int(layer.local_num_experts)
        hidden_size = int(layer.exl3_hidden_size)
        intermediate_size = int(layer.exl3_intermediate_size_per_partition)
        bitrates = tuple(int(value) for value in layer.exl3_layer_bitrates)
        if len(bitrates) != num_experts:
            raise ValueError(
                "mixed rank-sliced EXL3 bitrate count does not match experts: "
                f"bitrates={len(bitrates)}, experts={num_experts}"
            )
        tiers = {
            bits: tuple(
                expert_id
                for expert_id, expert_bits in enumerate(bitrates)
                if expert_bits == bits
            )
            for bits in sorted(set(bitrates))
        }
        if len(tiers) != 2:
            raise ValueError(
                "the one-grid mixed Trellis path currently requires exactly two "
                f"bitrates, got {tuple(tiers)}"
            )

        w13_param = layer.w13_trellis
        w2_param = layer.w2_trellis
        w13_shards = tuple(w13_param.exl3_shard_ids)
        w2_shards = tuple(w2_param.exl3_shard_ids)
        if w13_shards != ("w1", "w3") or w2_shards != ("w2",):
            raise ValueError(
                "mixed rank-sliced EXL3 requires w13=(w1,w3) and w2=(w2), got "
                f"{w13_shards}/{w2_shards}"
            )

        gate_suh, up_suh = self._rank_sliced_backing(layer, "w13_suh")
        gate_svh, up_svh = self._rank_sliced_backing(layer, "w13_svh")
        down_suh = self._rank_sliced_backing(layer, "w2_suh")
        down_svh = self._rank_sliced_backing(layer, "w2_svh")
        device = gate_suh.device
        tile_config = self._mixed_trellis_tile_config(hidden_size, intermediate_size)
        prepared_tiers = []
        tier_ids = []
        for bits, expert_ids in tiers.items():
            expected_last = 16 * bits
            w13 = torch.stack(
                tuple(
                    torch.stack(
                        tuple(
                            w13_param.exl3_tensors[(expert_id, shard_id)]
                            for expert_id in expert_ids
                        )
                    )
                    for shard_id in w13_shards
                )
            ).contiguous()
            w2 = torch.stack(
                tuple(
                    w2_param.exl3_tensors[(expert_id, "w2")] for expert_id in expert_ids
                )
            ).contiguous()
            expected_w13 = (
                2,
                len(expert_ids),
                hidden_size // 16,
                intermediate_size // 16,
                expected_last,
            )
            expected_w2 = (
                len(expert_ids),
                intermediate_size // 16,
                hidden_size // 16,
                expected_last,
            )
            if tuple(w13.shape) != expected_w13 or tuple(w2.shape) != expected_w2:
                raise ValueError(
                    f"mixed EXL3 K{bits} slab geometry mismatch: "
                    f"w13={tuple(w13.shape)}, w2={tuple(w2.shape)}, "
                    f"expected={expected_w13}/{expected_w2}"
                )

            index = torch.tensor(expert_ids, dtype=torch.long, device=device)
            tier_gate_suh = gate_suh.index_select(0, index).contiguous()
            tier_up_suh = up_suh.index_select(0, index).contiguous()
            intermediate_rotations = torch.cat(
                (
                    gate_svh.index_select(0, index),
                    up_svh.index_select(0, index),
                    down_suh.index_select(0, index),
                ),
                dim=1,
            ).contiguous()
            tier_down_svh = down_svh.index_select(0, index).contiguous()
            prepared_tiers.append(
                api.prepare_weights(
                    w13=w13,
                    w2=w2,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_experts=len(expert_ids),
                    activation=layer.activation.value,
                    fc1_tile_n=tile_config[1],
                    fc2_tile_n=tile_config[3],
                    params_dtype=torch.float16,
                    w13_layout="trellis3_t256_proj",
                    trellis_bits=bits,
                    codebook="mcg",
                    gate_suh=tier_gate_suh,
                    up_suh=tier_up_suh,
                    intermediate_rotations=intermediate_rotations,
                    down_svh=tier_down_svh,
                    tile_config=tile_config,
                    workspace=w13.view(torch.int32).reshape(-1)[:1],
                )
            )
            tier_ids.append(expert_ids)

        global_to_combined, descriptor_map = api.build_tiered_maps(
            tier_ids[0], tier_ids[1], device=device
        )
        layer.exl3_mixed_trellis = {
            "tiers": tuple(prepared_tiers),
            "tier_ids": tuple(tier_ids),
            "tier_bits": tuple(tiers),
            "global_to_combined": global_to_combined,
            "descriptor_map": descriptor_map,
            "rotations": api.combine_trellis_rotations(*prepared_tiers),
            "tile_config": tile_config,
        }
        layer.exl3_trellis_tile_config = tile_config

        # The prepared tier objects own compact, tier-ordered copies. Release
        # per-expert source tensors and the original rotation slabs now rather
        # than retaining both representations for every layer.
        for prefix in ("w13", "w2"):
            for suffix in ("suh", "svh", "trellis", "mcg", "mul1"):
                param = getattr(layer, f"{prefix}_{suffix}")
                param.exl3_tensors.clear()
                param.exl3_backing = None
        logger.info(
            "EXL3 mixed Trellis %s: tiers=%s",
            layer.layer_name,
            tuple((bits, len(ids)) for bits, ids in zip(tiers, tier_ids, strict=True)),
        )

    def _prepare_rank_sliced_weights(self, layer: RoutedExperts) -> None:
        if getattr(layer, "exl3_mixed_bitrate", False):
            self._prepare_mixed_rank_sliced_weights(layer)
            return
        api = _load_sparkinfer_fused_moe()
        num_experts = int(layer.local_num_experts)
        hidden_size = int(layer.exl3_hidden_size)
        intermediate_size = int(layer.exl3_intermediate_size_per_partition)
        layer_bitrates = tuple(getattr(layer, "exl3_layer_bitrates", ()))
        if len(set(layer_bitrates)) != 1:
            raise ValueError(
                "uniform rank-sliced EXL3 layer has invalid bitrates "
                f"{layer_bitrates!r}"
            )
        bits = int(layer_bitrates[0])
        if bits not in (3, 4, 5, 6):
            raise ValueError(
                f"rank-sliced EXL3 requires an integral 3/4/5/6 bitrate, got {bits!r}"
            )

        w13 = self._rank_sliced_backing(layer, "w13_trellis")
        w2 = self._rank_sliced_backing(layer, "w2_trellis")
        gate_suh, up_suh = self._rank_sliced_backing(layer, "w13_suh")
        gate_svh, up_svh = self._rank_sliced_backing(layer, "w13_svh")
        down_suh = self._rank_sliced_backing(layer, "w2_suh")
        down_svh = self._rank_sliced_backing(layer, "w2_svh")
        expected_w13 = (
            2,
            num_experts,
            hidden_size // 16,
            intermediate_size // 16,
            16 * bits,
        )
        expected_w2 = (
            num_experts,
            intermediate_size // 16,
            hidden_size // 16,
            16 * bits,
        )
        if tuple(w13.shape) != expected_w13 or tuple(w2.shape) != expected_w2:
            raise ValueError(
                "rank-sliced EXL3 slab geometry mismatch: "
                f"w13={tuple(w13.shape)}, w2={tuple(w2.shape)}, "
                f"expected={expected_w13}/{expected_w2}"
            )

        intermediate_rotations = torch.empty(
            (num_experts, 3 * intermediate_size),
            dtype=torch.float16,
            device=w13.device,
        )
        intermediate_rotations[:, :intermediate_size].copy_(gate_svh)
        intermediate_rotations[:, intermediate_size : 2 * intermediate_size].copy_(
            up_svh
        )
        intermediate_rotations[:, 2 * intermediate_size :].copy_(down_suh)
        tile_config = self._trellis_tile_config(hidden_size, intermediate_size)
        marker = layer.w13_mcg.exl3_tensors[(0, "w1")]
        weight_plan = api.plan_weights(
            quant_modes="w4a16",
            source_format="exl3_trellis_mcg",
            activation=layer.activation.value,
            params_dtype=layer.exl3_params_dtype,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            w13_layout="w13",
            trellis_bits=bits,
            trellis_tile_config=tile_config,
        )
        layer.exl3_trellis_weights = api.prepare_weights(
            plan=weight_plan,
            params_dtype=layer.exl3_params_dtype,
            w1_fp4=w13,
            w2_fp4=w2,
            gate_suh=gate_suh,
            up_suh=up_suh,
            intermediate_rotations=intermediate_rotations,
            down_svh=down_svh,
            trellis_mcg=marker,
        )

        slabs = (
            w13[0],
            gate_suh,
            gate_svh,
            w13[1],
            up_suh,
            up_svh,
            w2,
            down_suh,
            down_svh,
        )
        layer.exl3_pointer_tables = tuple(self._pointer_table(slab) for slab in slabs)
        layer.exl3_expert_map = torch.arange(
            num_experts,
            dtype=torch.int64,
            device=w13.device,
        )
        layer.exl3_trellis_tile_config = tile_config

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        del layer
        return None

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.long

    def _mixed_rank_sliced_runtime(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> dict[str, Any]:
        mixed = layer.exl3_mixed_trellis
        max_decode_m = _positive_env_int("VLLM_EXL3_TRELLIS_MAX_M", 32)
        max_batched_tokens = int(layer.exl3_max_num_batched_tokens)
        if max_decode_m > max_batched_tokens:
            max_decode_m = max_batched_tokens
        topk = int(topk_ids.shape[1])
        tier_signature = tuple(
            (int(bits), len(ids))
            for bits, ids in zip(mixed["tier_bits"], mixed["tier_ids"], strict=True)
        )
        key = (
            _runtime_owner_token(self.quant_config, layer),
            x.device.index,
            x.dtype,
            topk_ids.dtype,
            int(layer.exl3_hidden_size),
            int(layer.exl3_intermediate_size_per_partition),
            tier_signature,
            topk,
            max_decode_m,
            max_batched_tokens,
            mixed["tile_config"],
        )
        runtime = _MIXED_TRELLIS_RUNTIMES.get(key)
        if runtime is not None:
            return runtime
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Mixed-bitrate EXL3 runtime must be compiled during the eager "
                "profile pass before CUDA graph capture"
            )
        if topk_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "mixed-bitrate EXL3 requires int32/int64 route IDs, got "
                f"{topk_ids.dtype}"
            )

        api = _load_sparkinfer_mixed_trellis()
        device = x.device
        props = torch.cuda.get_device_properties(device)
        total_experts = sum(experts for _, experts in tier_signature)

        def make_state(capacity: int) -> dict[str, Any]:
            route_slots = api.max_packed_route_slots(
                capacity * topk,
                _MIXED_TRELLIS_ROUTE_BLOCK_SIZE,
                total_experts,
            )
            launch = api.compile_mixed_trellis(
                size_m=capacity,
                hidden_size=int(layer.exl3_hidden_size),
                intermediate_size=int(layer.exl3_intermediate_size_per_partition),
                tier0_num_experts=tier_signature[0][1],
                tier1_num_experts=tier_signature[1][1],
                tier0_bits=tier_signature[0][0],
                tier1_bits=tier_signature[1][0],
                top_k=topk,
                max_m_blocks=(route_slots + _MIXED_TRELLIS_ROUTE_BLOCK_SIZE - 1)
                // _MIXED_TRELLIS_ROUTE_BLOCK_SIZE,
                moe_block_size=_MIXED_TRELLIS_ROUTE_BLOCK_SIZE,
                sms=int(props.multi_processor_count),
                max_shared_mem=int(props.shared_memory_per_block_optin),
                force_tile_config=mixed["tile_config"],
                rotation_input_dtype=("bf16" if x.dtype == torch.bfloat16 else "fp16"),
                route_ids_dtype=topk_ids.dtype,
            )
            return {
                "capacity": capacity,
                "launch": launch,
                "buffers": api.make_mixed_trellis_buffers(
                    launch,
                    device=device,
                    sms=int(props.multi_processor_count),
                ),
            }

        decode = make_state(max_decode_m)
        prefill = (
            make_state(max_batched_tokens)
            if max_batched_tokens > max_decode_m
            else decode
        )
        runtime = {
            "api": api,
            "decode": decode,
            "prefill": prefill,
            "max_decode_m": max_decode_m,
            "max_batched_tokens": max_batched_tokens,
        }
        _MIXED_TRELLIS_RUNTIMES[key] = runtime
        buffer_bytes = _unique_tensor_storage_bytes(
            decode["buffers"], prefill["buffers"]
        )
        logger.info_once(
            "EXL3 mixed Trellis runtime planned: tiers=%s decode_capacity=%d "
            "prefill_capacity=%d buffers=%.1f MiB",
            tier_signature,
            max_decode_m,
            max_batched_tokens,
            buffer_bytes / (1 << 20),
        )
        return runtime

    def _apply_mixed_rank_sliced(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        runtime = self._mixed_rank_sliced_runtime(layer, x, topk_ids)
        m = int(x.shape[0])
        if m > runtime["max_batched_tokens"]:
            raise ValueError(
                "mixed-bitrate EXL3 batch exceeds planned capacity: "
                f"m={m}, capacity={runtime['max_batched_tokens']}"
            )
        state = (
            runtime["decode"] if m <= runtime["max_decode_m"] else runtime["prefill"]
        )
        mixed = layer.exl3_mixed_trellis
        output = runtime["api"].run_mixed_trellis(
            x,
            mixed["tiers"][0],
            mixed["tiers"][1],
            topk_weights,
            topk_ids,
            mixed["global_to_combined"],
            mixed["descriptor_map"],
            mixed["rotations"],
            state["launch"],
            state["buffers"],
        )
        return output.to(x.dtype)

    def _rank_sliced_runtime(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> dict[str, Any]:
        # Rank-sliced target and draft layers both reach small row counts
        # (typically m=1..3) during profiling, decode, or CUDA-graph capture. If
        # m falls outside the Trellis window the eager parity path is reached;
        # under capture that is illegal and during eager execution it adds an
        # avoidable external-extension ABI dependency:
        #
        #   RuntimeError: EXL3 eager parity path entered during CUDA graph
        #   capture (m=3); capture sizes must lie inside the Trellis window
        #   [4, 32]
        #
        # The backend therefore owns one capability-based default for both
        # roles. An explicit env value remains authoritative for diagnostics.
        min_trellis_m = _positive_env_int(
            "VLLM_EXL3_TRELLIS_MIN_M", _DEFAULT_TRELLIS_MIN_M
        )
        max_trellis_m = _positive_env_int("VLLM_EXL3_TRELLIS_MAX_M", 32)
        block_m = _positive_env_int("VLLM_EXL3_TRELLIS_BLOCK_M", 8)
        chunk = _positive_env_int("VLLM_EXL3_PREFILL_CHUNK", 128)
        prefill_trellis = os.environ.get("VLLM_EXL3_PREFILL_TRELLIS", "1") == "1"
        prefill_block_m = _positive_env_int("VLLM_EXL3_PREFILL_BLOCK_M", 64)
        if min_trellis_m > max_trellis_m:
            raise ValueError(
                "VLLM_EXL3_TRELLIS_MIN_M cannot exceed VLLM_EXL3_TRELLIS_MAX_M"
            )
        # Batch-INVARIANT capacity. Including x.shape[0] here made the runtime
        # cache key depend on the live batch, so any m above the planned capacity
        # produced a cache MISS -- silently planning a fresh ~1 GiB arena mid-serve
        # and making the capacity guard in _apply_rank_sliced unreachable. The
        # planned capacity is a property of the layer, not of one forward pass.
        max_batched_tokens = int(layer.exl3_max_num_batched_tokens)
        prefill_plan_enabled = prefill_trellis and max_batched_tokens > max_trellis_m
        parity_rows = (
            min(chunk, max_batched_tokens)
            if prefill_plan_enabled
            else max_batched_tokens
        )
        max_parity_batch = min(max_batched_tokens, min_trellis_m - 1)
        if max_parity_batch > parity_rows:
            raise ValueError(
                "VLLM_EXL3_PREFILL_CHUNK cannot cover the EXL3 parity window: "
                f"chunk={chunk}, required_rows={max_parity_batch}. Increase "
                "VLLM_EXL3_PREFILL_CHUNK or lower VLLM_EXL3_TRELLIS_MIN_M."
            )
        topk = int(topk_ids.shape[1])
        device_index = x.device.index
        key = (
            # Owning model scope first: the cached runtime holds mutable scratch,
            # so a target layer and a same-shape rank-sliced MTP draft layer must
            # not share an entry (see _runtime_scope_id).
            _runtime_owner_token(self.quant_config, layer),
            device_index,
            x.dtype,
            int(layer.exl3_hidden_size),
            int(layer.exl3_intermediate_size_per_partition),
            int(layer.local_num_experts),
            topk,
            max_batched_tokens,
            min_trellis_m,
            max_trellis_m,
            block_m,
            chunk,
            prefill_trellis,
            prefill_block_m,
            layer.exl3_trellis_tile_config,
        )
        runtime = _RANK_SLICED_RUNTIMES.get(key)
        if runtime is not None:
            return runtime
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Rank-sliced EXL3 runtime must be planned during the eager "
                "profile pass before CUDA graph capture"
            )

        api = _load_sparkinfer_fused_moe()

        def _plan_with_scratch(plan_max_tokens: int, plan_block_m: int):
            caps = api.Caps(
                max_tokens=plan_max_tokens,
                num_topk=topk,
                # vLLM supplies final top-k IDs/weights to bind(); the fused-MoE
                # router workspace is unused. A zero route-workspace request
                # still lets the W4A16 core derive route_E from weight_E.
                route_num_experts=0,
                device=x.device,
                weight_plan=layer.exl3_trellis_weights.plan,
                quant_mode="w4a16",
                w4a16_block_size_m=plan_block_m,
            )
            plan = api.plan(caps)
            scratch_spec = plan.scratch_specs()[0]
            scratch = torch.empty(
                scratch_spec.shape,
                dtype=scratch_spec.dtype,
                device=scratch_spec.device,
            )
            return plan, scratch

        trellis_plan, trellis_scratch = _plan_with_scratch(max_trellis_m, block_m)
        # Prefill batches (m > max_trellis_m) run through a second planned
        # Trellis capacity instead of the eager ExLlamaV3 parity path. The
        # large block keeps expert tiles streamed once per block of routed
        # rows rather than once per 8-row block.
        prefill_plan = None
        prefill_scratch = None
        if prefill_plan_enabled:
            prefill_plan, prefill_scratch = _plan_with_scratch(
                max_batched_tokens, prefill_block_m
            )

        ext = _load_exl3_ext()
        required_ext = {
            "exl3_moe",
            "exl3_moe_max_concurrency",
        }
        missing_ext = sorted(name for name in required_ext if not hasattr(ext, name))
        if missing_ext:
            raise RuntimeError(
                "The EXL3 extension lacks routed-expert entry points: "
                + ", ".join(missing_ext)
            )
        concurrency = int(ext.exl3_moe_max_concurrency(torch.cuda.current_device()))
        hidden_size = int(layer.exl3_hidden_size)
        intermediate_size = int(layer.exl3_intermediate_size_per_partition)
        num_experts = int(layer.local_num_experts)
        device = x.device
        # With the prefill plan live, the parity path only ever serves
        # m < min_trellis_m, so its persistent staging shrinks to one chunk.
        runtime = {
            "api": api,
            "trellis_plan": trellis_plan,
            "trellis_scratch": trellis_scratch,
            "prefill_plan": prefill_plan,
            "prefill_scratch": prefill_scratch,
            "ext": ext,
            "min_trellis_m": min_trellis_m,
            "max_trellis_m": max_trellis_m,
            "max_batched_tokens": max_batched_tokens,
            "parity_rows": parity_rows,
            "topk": topk,
            "chunk": chunk,
            "xh": torch.empty(
                (parity_rows, hidden_size),
                dtype=torch.float16,
                device=device,
            ),
            "out32": torch.empty(
                (parity_rows, hidden_size),
                dtype=torch.float32,
                device=device,
            ),
            "tg": torch.empty(
                (concurrency, chunk, hidden_size),
                dtype=torch.float16,
                device=device,
            ),
            "tu": torch.empty(
                (concurrency, chunk, hidden_size),
                dtype=torch.float16,
                device=device,
            ),
            "ig": torch.empty(
                (concurrency, chunk, intermediate_size),
                dtype=torch.float16,
                device=device,
            ),
            "iu": torch.empty(
                (concurrency, chunk, intermediate_size),
                dtype=torch.float16,
                device=device,
            ),
            "expert_count": torch.empty(
                num_experts + 1,
                dtype=torch.int64,
                device=device,
            ),
            "expert_offsets": torch.empty(
                num_experts + 1,
                dtype=torch.int64,
                device=device,
            ),
            "token_sorted": torch.empty(
                parity_rows * topk,
                dtype=torch.int64,
                device=device,
            ),
            "weight_sorted": torch.empty(
                parity_rows * topk,
                dtype=torch.float16,
                device=device,
            ),
            "flat_token": torch.arange(
                chunk,
                dtype=torch.int64,
                device=device,
            ).repeat_interleave(topk),
            "ones": torch.ones(
                chunk * topk,
                dtype=torch.int64,
                device=device,
            ),
        }
        _RANK_SLICED_RUNTIMES[key] = runtime
        prefill_arena_mib = (
            0.0
            if prefill_scratch is None
            else prefill_scratch.numel() * prefill_scratch.element_size() / (1 << 20)
        )
        logger.info_once(
            "EXL3 rank-sliced runtime planned: Trellis m=%d..%d block_m=%d, "
            "prefill %s capacity=%d chunk=%d topk=%d",
            min_trellis_m,
            max_trellis_m,
            block_m,
            (
                f"trellis block_m={prefill_block_m} arena={prefill_arena_mib:.1f}MiB"
                if prefill_plan is not None
                else "parity"
            ),
            max_batched_tokens,
            chunk,
            topk,
        )
        return runtime

    def _apply_rank_sliced(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if getattr(layer, "exl3_mixed_bitrate", False):
            return self._apply_mixed_rank_sliced(
                layer,
                x,
                topk_weights,
                topk_ids,
            )
        runtime = self._rank_sliced_runtime(layer, x, topk_ids)
        m = int(x.shape[0])
        if runtime["min_trellis_m"] <= m <= runtime["max_trellis_m"]:
            binding = runtime["api"].bind(
                runtime["trellis_plan"],
                scratch=runtime["trellis_scratch"],
                a=x,
                experts=layer.exl3_trellis_weights,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
            )
            output = runtime["api"].run(binding=binding)
            return output.to(x.dtype)

        if runtime["prefill_plan"] is not None and m > runtime["max_trellis_m"]:
            if m > runtime["max_batched_tokens"]:
                raise ValueError(
                    "EXL3 batch exceeds its planned capacity: "
                    f"m={m}, capacity={runtime['max_batched_tokens']}"
                )
            binding = runtime["api"].bind(
                runtime["prefill_plan"],
                scratch=runtime["prefill_scratch"],
                a=x,
                experts=layer.exl3_trellis_weights,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
            )
            output = runtime["api"].run(binding=binding)
            return output.to(x.dtype)

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "EXL3 eager parity path entered during CUDA graph capture "
                f"(m={m}); capture sizes must lie inside the Trellis window "
                f"[{runtime['min_trellis_m']}, {runtime['max_trellis_m']}]. "
                f"Layer {getattr(layer, 'layer_name', '<unknown>')!r} was "
                f"classified as {'draft' if _is_draft_layer(layer) else 'target'}. "
                "A rank-sliced draft layer should have had its window widened to "
                f"MIN_CAPTURABLE_TRELLIS_M={MIN_CAPTURABLE_TRELLIS_M} "
                "automatically; if VLLM_EXL3_TRELLIS_MIN_M is set explicitly, "
                f"lower it to {MIN_CAPTURABLE_TRELLIS_M} or unset it."
            )
        if m > runtime["parity_rows"]:
            raise ValueError(
                "EXL3 batch exceeds its planned parity capacity: "
                f"m={m}, capacity={runtime['parity_rows']}"
            )
        ext = runtime["ext"]
        xh = runtime["xh"][:m]
        xh.copy_(x)
        out32 = runtime["out32"][:m]
        out32.zero_()
        chunk = int(runtime["chunk"])
        pointer_args = layer.exl3_pointer_tables
        if m > chunk and hasattr(ext, "exl3_moe_fused"):
            ext.exl3_moe_fused(
                xh,
                out32,
                topk_ids,
                topk_weights,
                layer.exl3_expert_map,
                runtime["expert_count"],
                runtime["expert_offsets"],
                runtime["token_sorted"],
                runtime["weight_sorted"],
                runtime["tg"],
                runtime["tu"],
                runtime["ig"],
                runtime["iu"],
                0,
                3,
                3,
                3,
                *pointer_args,
                True,
                False,
                True,
                False,
                True,
                False,
                0.0,
                0,
            )
            return out32.to(x.dtype)

        local_ids = layer.exl3_expert_map[topk_ids.long()]
        half_weights = topk_weights.to(torch.float16)
        topk = int(runtime["topk"])
        for start in range(0, m, chunk):
            current_m = min(chunk, m - start)
            flat = local_ids[start : start + current_m].reshape(-1)
            order = torch.argsort(flat)
            route_count = current_m * topk
            token_ids = runtime["flat_token"][:route_count].index_select(0, order)
            route_weights = (
                half_weights[start : start + current_m]
                .reshape(-1)
                .index_select(0, order)
            )
            counts = runtime["expert_count"]
            counts.zero_()
            counts.scatter_add_(0, flat, runtime["ones"][:route_count])
            ext.exl3_moe(
                xh[start : start + current_m],
                out32[start : start + current_m],
                counts,
                token_ids,
                route_weights,
                runtime["tg"],
                runtime["tu"],
                runtime["ig"],
                runtime["iu"],
                0,
                3,
                3,
                3,
                *pointer_args,
                True,
                False,
                True,
                False,
                True,
                False,
                0.0,
            )
        return out32.to(x.dtype)

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        if layer.activation != MoEActivation.SILU:
            raise NotImplementedError(
                f"EXL3 correctness MoE supports SiLU only, got {layer.activation}"
            )
        if layer.expert_map is not None:
            raise NotImplementedError("EXL3 MoE expert maps/EPLB are not supported")
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "EXL3 MoE does not support router weights applied on input"
            )

        original_shape = x.shape[:-1]
        original_dtype = x.dtype
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        ids = topk_ids.reshape(x_2d.shape[0], -1).contiguous()
        weights = topk_weights.reshape_as(ids).to(torch.float32).contiguous()
        if self.quant_config.rank_sliced_metadata is not None:
            output = self._apply_rank_sliced(layer, x_2d, weights, ids)
            return output.reshape(*original_shape, output.shape[-1])

        x_2d = x_2d.to(torch.float16)
        ids = ids.to(torch.long)
        weights = weights.to(torch.float16)
        output = torch.zeros(
            (x_2d.shape[0], layer.hidden_size),
            dtype=torch.float32,
            device=x.device,
        )
        for expert_id in range(layer.local_num_experts):
            positions = (ids == expert_id).nonzero(as_tuple=False)
            if positions.shape[0] == 0:
                continue
            token_ids = positions[:, 0]
            route_ids = positions[:, 1]
            expert_input = x_2d.index_select(0, token_ids)
            gate = self._apply_expert(layer, "w13", expert_input, expert_id, "w1")
            up = self._apply_expert(layer, "w13", expert_input, expert_id, "w3")
            hidden = torch.nn.functional.silu(gate) * up
            expert_output = self._apply_expert(layer, "w2", hidden, expert_id, "w2")
            route_weight = weights[token_ids, route_ids].unsqueeze(-1)
            output.index_add_(
                0,
                token_ids,
                (expert_output * route_weight).to(torch.float32),
            )
        output = output.reshape(*original_shape, output.shape[-1])
        return output if output.dtype == original_dtype else output.to(original_dtype)

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer, x, router_logits, input_ids
        raise NotImplementedError("EXL3 MoE uses vLLM's external router")

    @staticmethod
    def _apply_expert(
        layer: RoutedExperts,
        group: str,
        x: torch.Tensor,
        expert_id: int,
        shard_id: str,
    ) -> torch.Tensor:
        key = (expert_id, shard_id)
        trellis = getattr(layer, f"{group}_trellis").exl3_tensors[key]
        packed_k = trellis.shape[0] * 16
        if x.shape[-1] > packed_k:
            raise ValueError(
                f"EXL3 MoE input width {x.shape[-1]} exceeds packed K={packed_k}"
            )
        if x.shape[-1] < packed_k:
            x = torch.nn.functional.pad(x, (0, packed_k - x.shape[-1]))
        output = _exl3_gemm(
            x,
            trellis,
            getattr(layer, f"{group}_suh").exl3_tensors[key],
            getattr(layer, f"{group}_svh").exl3_tensors[key],
            key in getattr(layer, f"{group}_mcg").exl3_tensors,
            key in getattr(layer, f"{group}_mul1").exl3_tensors,
        )
        logical_n = (
            layer.hidden_size
            if shard_id == "w2"
            else layer.exl3_intermediate_size_per_partition
        )
        if output.shape[-1] < logical_n:
            raise ValueError(
                f"EXL3 MoE packed N={output.shape[-1]} is below logical N={logical_n}"
            )
        return output[..., :logical_n]


__all__ = ["Exl3Config", "Exl3LinearMethod", "Exl3MoEMethod"]
