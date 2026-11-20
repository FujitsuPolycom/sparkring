"""vLLM general plugin for row-owned DeepSeek-V4.1-Flash prefill.

Status: research-only. The CPU suite in ``tests/`` checks the owned-row
forward against the image's own layer code on simulated ranks. It also checks
the source pins and attention edit against an extracted serving image.

The source pins target SparkRing installer image
``dev-20261013-dcp2prefill-status038-sircl-tp6`` with B12X attention and experts.

Flag: ``DS41_ROW_SHARD_PREFILL=1`` selects row ownership; unset or ``0``
leaves the image untouched (the plugin then installs nothing). Optional
``DS41_ROW_SHARD_MIN_OWNED_ROWS`` (default 512) is the smallest per-rank row
count admitted. Every rank must set the same values; startup votes on them.

What changes when a forward is admitted (``core`` explains the math):

* ``DeepseekV4Model.forward`` decides admission and holds the forward's
  ``Ownership``; ``DeepseekV4DecoderLayer.forward`` runs layers ``0`` through
  the CED boundary (layer ``ced_decoder_start``) on owned rows.
* ``DeepseekV4Attention._o_proj`` gets one line edited: its TP all-reduce
  goes through ``_ds41_row_shard_attention_reduce``, which skips the reduction
  only while an owned layer asks for the unreduced partial.
* The MoE runner is called with ``defer_tp_reduction=True``. The runner
  validates that request with GLM's ``validate_moe_deferral``, which admits
  TP2 and TP4 only; for the duration of a deferred call this plugin
  substitutes the same checks with TP8 admitted as well.

Admission (all ranks vote; one ineligible rank keeps every rank on the image
path): eager forward (no CUDA-graph replay, capture or compilation), no
micro-batches, first pipeline stage of a single-stage pipeline, CED indices
present, rows divisible by the TP size with at least the minimum per rank,
attention metadata for every row with no padding (``num_actual_tokens``
equal to the row count) and not a pure decode step. Mixed batches of prefill
and decode rows are admitted: owned rows only change which rank computes
row-local work, and attention still reads every row.

Mechanism: before any edit or wrapper is installed the SHA-256 of each file
in ``PINS`` must equal the recorded value; otherwise startup fails with
``PatchRefused``. No file is written. Modules not imported yet are patched
right after their first import (a ``sys.meta_path`` finder).
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import importlib.util
import logging
import os
import sys
import threading
from typing import Any

from .core import (
    MIN_OWNED_ROWS,
    STATE,
    SUPPORTED_WORLDS,
    Ownership,
    RowShardRefused,
    attention_reduce,
    gate_owned_routes,
    indexer_score_offsets,
    owned_layer_forward,
    owned_layers,
    publish_index_rows,
    row_geometry,
    select_owned_routes,
)
from .source_edit import PatchRefused, edit_function, file_sha256

PLUGIN_NAME = "ds41_row_shard"
PLUGIN_VERSION = "0.1.0"
FLAG = "DS41_ROW_SHARD_PREFILL"
MIN_ROWS_FLAG = "DS41_ROW_SHARD_MIN_OWNED_ROWS"
INDEXER_FLAG = "DS41_ROW_SHARD_INDEXER"
ROUTES_FLAG = "DS41_ROW_SHARD_COMPACT_ROUTES"
LOCAL_QUERIES_FLAG = "DS41_ROW_SHARD_LOCAL_QUERIES"
KV_LATENTS_FLAG = "DS41_ROW_SHARD_KV_LATENTS"
COMPRESSOR_LATENTS_FLAG = "DS41_ROW_SHARD_COMPRESSOR_LATENTS"
ENGRAM_BANDS_FLAG = "DS41_ROW_SHARD_ENGRAM_BANDS"
MARKER = "__ds41_row_shard__"
IMAGE = "dev-20261013-dcp2prefill-status038-sircl-tp6"

MOE_RUNNER_MODULE = "vllm.model_executor.layers.fused_moe.runner.moe_runner"
ATTENTION_MODULE = "vllm.models.deepseek_v4_1.attention"
COMPRESSOR_MODULE = "vllm.models.deepseek_v4_1.compressor"
MODEL_MODULE = "vllm.models.deepseek_v4_1.nvidia.model"
GLM_SHARDING_MODULE = "vllm.models.glm5next.nvidia.mhc_prefill_sharding"

PINS: dict[str, tuple[str, str]] = {
    COMPRESSOR_MODULE: (
        "45a881e593f3704e3cc6254a28d15bfc88ddb2d3d7c2494e4b145a4a45605c51",
        "compressor projection buffers and full-batch cache state contracts",
    ),
    # module: (SHA-256 in the image, why the plugin depends on it)
    MODEL_MODULE: (
        "dafb69b36e68c36ba8878c6033e1bc94439526c3d1d9c83edd7c153ad2ba88c8",
        "decoder layer and model loop that core.owned_layer_forward mirrors",
    ),
    ATTENTION_MODULE: (
        "ad71ccd505a368a88cfe366e26ae0d66e1ba83a94d860f7a2e0a29b7e984dc2d",
        "edited _o_proj; prepare_global_kv and forward call contracts",
    ),
    "vllm.models.deepseek_v4_1.common.engram": (
        "efda07bdd54d947852cdf75095905033038cf2a4425d7b234d092a9253c7c811",
        "Engram.forward, which core.owned_engram mirrors on owned rows",
    ),
    "vllm.models.deepseek_v4_1.ced": (
        "fe956af8afeb3c708b0d677d437d985974b556f4edf7f90b54a2dfa344568781",
        "gather_rows zeroes negative indices, which compaction relies on",
    ),
    "vllm.models.deepseek_v4_1.b12x_layers": (
        "d6e3352070bc69329ef0c09a20639c7f1165b02fd1f75e8a07133ef2fdd04503",
        "B12xMHC plans fall back to the capacity plan for fewer rows",
    ),
    MOE_RUNNER_MODULE: (
        "86aa7228e44c8dc6e5d3602bcfede3f4d009c935d97237d6a5ac42b2bcb3d3bc",
        "defer_tp_reduction skips only the final all-reduce",
    ),
    "vllm.models.deepseek_v4.nvidia.model": (
        "109236e3b5868f74922f9d270a2d82776f498855b284666279613002b65016ea",
        "DeepseekV4MoE forward, which core.moe_unreduced mirrors",
    ),
    GLM_SHARDING_MODULE: (
        "61d02201f0288bd58ef76687b8698900e45b45446f37a045c1703f8726c288e3",
        "validate_moe_deferral, substituted for TP8 during deferred calls",
    ),
}

_O_PROJ_EDIT = (
    "            local = get_tp_group().all_reduce(local)\n",
    "            local = _ds41_row_shard_attention_reduce(local)\n",
)
_INDEXER_SCORE_EDITS = (
    (
        "            for offset in range(0, rows, chunk_rows):\n",
        "            for offset in _ds41_indexer_score_offsets(self, rows, chunk_rows):\n",
    ),
    (
        "                dsa_indexer.select(binding)\n\n",
        "                dsa_indexer.select(binding)\n"
        "            _ds41_publish_index_rows(self, rows, chunk_rows)\n",
    ),
)

_LOG = logging.getLogger("vllm." + PLUGIN_NAME)
_LOCK = threading.Lock()
_REPORTS = 0


def _enabled_flag(flag: str) -> bool:
    value = os.environ.get(flag, "0").strip()
    if value not in ("0", "1", ""):
        raise PatchRefused(f"{flag} must be 0 or 1, not {value!r}")
    return value == "1"


def enabled() -> bool:
    return _enabled_flag(FLAG)


def min_owned_rows() -> int:
    value = os.environ.get(MIN_ROWS_FLAG, str(MIN_OWNED_ROWS))
    try:
        rows = int(value)
    except ValueError as exc:
        raise PatchRefused(f"{MIN_ROWS_FLAG} must be an integer") from exc
    if rows < 1:
        raise PatchRefused(f"{MIN_ROWS_FLAG} must be positive")
    return rows


# -- pins --------------------------------------------------------------------
def check_pins() -> None:
    """Refuse unless every pinned file has the recorded SHA-256."""
    for module, (sha256, reason) in PINS.items():
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise PatchRefused(f"{module} not found ({reason})")
        actual = file_sha256(spec.origin)
        if actual != sha256:
            raise PatchRefused(
                f"{module} ({spec.origin}) has SHA-256 {actual}, expected "
                f"{sha256} ({reason}); this plugin targets image {IMAGE}"
            )


# -- attention edit ------------------------------------------------------------
def _ds41_row_shard_attention_reduce(local):
    from vllm.distributed import get_tp_group

    return attention_reduce(local, get_tp_group().all_reduce)


def patch_attention(module) -> None:
    edit_function(
        module,
        "DeepseekV4Attention._o_proj",
        PINS[ATTENTION_MODULE][0],
        (_O_PROJ_EDIT,),
        {"_ds41_row_shard_attention_reduce": _ds41_row_shard_attention_reduce},
        MARKER,
    )
    edits = ()
    helpers = {}
    if _enabled_flag(INDEXER_FLAG):
        edits = _INDEXER_SCORE_EDITS
        helpers = {
            "_ds41_indexer_score_offsets": indexer_score_offsets,
            "_ds41_publish_index_rows": publish_index_rows,
        }
    if _enabled_flag(LOCAL_QUERIES_FLAG):
        from . import local_query

        local_query.install(module, PINS[ATTENTION_MODULE][0], MARKER)
        if _enabled_flag(COMPRESSOR_LATENTS_FLAG):
            local_query.install_compressor(
                module, sys.modules[COMPRESSOR_MODULE], PINS[ATTENTION_MODULE][0],
                PINS[COMPRESSOR_MODULE][0], MARKER,
            )
        edits += local_query.MQA_EDITS
        helpers.update(local_query.MQA_HELPERS)
    if edits:
        edit_function(
            module,
            "DeepseekV4Attention.forward_mqa",
            PINS[ATTENTION_MODULE][0],
            edits,
            helpers,
            MARKER,
            decorator="eager_break_during_capture",
        )


# -- MoE deferral at TP8 -----------------------------------------------------
def _validate_moe_deferral(runner: Any) -> None:
    """GLM's ``validate_moe_deferral`` with TP8 admitted beside TP2 and TP4."""
    config = runner.moe_config
    parallel = config.moe_parallel_config
    if (
        config.tp_size not in SUPPORTED_WORLDS
        or config.dp_size != 1
        or config.ep_size != 1
        or config.pcp_size != 1
        or config.is_sequence_parallel
        or config.skip_final_all_reduce
        or parallel.use_all2all_kernels
        or runner._fused_output_is_reduced
        or runner.routed_output_transform is not None
        or runner.routed_input_transform is not None
        or type(runner.router).__name__ == "ZeroExpertRouter"
    ):
        raise RuntimeError(
            "row-owned prefill requires one unreduced conventional TP MoE output"
        )


class _DeferralValidator:
    """Swap the runner's deferral validator while an owned forward runs."""

    def __enter__(self):
        module = importlib.import_module(GLM_SHARDING_MODULE)
        self._module = module
        self._saved = module.validate_moe_deferral
        module.validate_moe_deferral = _validate_moe_deferral
        return self

    def __exit__(self, *exc):
        self._module.validate_moe_deferral = self._saved
        return False


# -- admission -----------------------------------------------------------------
def _static_error(model, world: int) -> str | None:
    """Why this model cannot own rows, or None."""
    from vllm.distributed import get_pp_group

    try:
        decoder_layer = sys.modules[MODEL_MODULE].DeepseekV4DecoderLayer
        parallel = model.parallel_config
        if world not in SUPPORTED_WORLDS:
            raise RowShardRefused(f"TP size {world} is not one of {SUPPORTED_WORLDS}")
        if (
            get_pp_group().world_size != 1
            or parallel.data_parallel_size != 1
            or parallel.enable_expert_parallel
            or getattr(parallel, "enable_eplb", False)
            or model.use_sequence_parallel
            or model.use_mega_moe
        ):
            raise RowShardRefused(
                "row ownership requires PP1, DP1, no expert parallelism, "
                "no sequence parallelism and no mega-MoE"
            )
        layers = owned_layers(
            model.ced_decoder_start,
            len(model.layers),
            getattr(model, "aux_hidden_state_layers", ()),
        )
        if model.start_layer != 0 or model.end_layer <= layers[-1]:
            raise RowShardRefused("owned layers must be local to this stage")
        attention = sys.modules[ATTENTION_MODULE].DeepseekV4Attention
        if not getattr(attention._o_proj, MARKER, False):
            raise RowShardRefused("the WO projection edit is not installed")
        if _enabled_flag(ROUTES_FLAG):
            runner = sys.modules[MOE_RUNNER_MODULE].MoERunner
            if not all(getattr(method, MARKER, False) for method in (
                runner._forward_impl, runner._apply_quant_method,
            )):
                raise RowShardRefused("the compact MoE routing edits are not installed")
        for index in layers:
            layer = model.layers[index]
            if not isinstance(layer, decoder_layer):
                raise RowShardRefused(f"layer {index} is not a V4.1 decoder layer")
            ffn = layer.ffn
            if ffn.use_mega_moe:
                raise RowShardRefused(f"layer {index} uses mega-MoE")
            _validate_moe_deferral(ffn.experts)
            if ffn.experts.moe_config.tp_size != world:
                raise RowShardRefused(f"layer {index} MoE TP size differs")
            engram = getattr(layer, "engram", None)
            if engram is not None:
                kind = type(engram.wkv).__name__
                # Replicated: projected on owned rows. Column-parallel with a
                # gathered output (projection_tp): projected on all rows, then
                # narrowed. Anything else is unknown here.
                if kind == "ColumnParallelLinear":
                    if not getattr(engram.wkv, "gather_output", False):
                        raise RowShardRefused(f"layer {index} Engram wkv does not gather")
                elif kind != "ReplicatedLinear":
                    raise RowShardRefused(f"layer {index} Engram wkv is {kind}")
    except Exception as exc:  # any refusal becomes a vote, never a one-rank raise
        return f"{type(exc).__name__}: {exc}"
    return None


def _vote(group, item):
    import torch

    votes: list[Any] = [None] * group.world_size
    torch.distributed.all_gather_object(votes, item, group=group.cpu_group)
    return votes


def configure(model) -> None:
    """Agree on feature flags, the threshold and the static checks on all ranks."""
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    error = _static_error(model, group.world_size)
    settings = (
        enabled(), min_owned_rows(), _enabled_flag(INDEXER_FLAG),
        _enabled_flag(ROUTES_FLAG), _enabled_flag(LOCAL_QUERIES_FLAG),
        _enabled_flag(KV_LATENTS_FLAG),
        _enabled_flag(COMPRESSOR_LATENTS_FLAG),
        _enabled_flag(ENGRAM_BANDS_FLAG),
    )
    item = (settings, error)
    votes = _vote(group, item)
    if any(vote[0] != settings for vote in votes):
        raise PatchRefused(f"TP ranks disagree on row-owned prefill settings: {votes}")
    errors = [vote[1] for vote in votes if vote[1] is not None]
    if errors:
        raise PatchRefused("row-owned prefill admission failed: " + repr(errors))
    model._ds41_row_shard_layers = {
        id(layer): index for index, layer in enumerate(model.layers)
    }
    model._ds41_row_shard_min_rows = settings[1]
    model._ds41_row_shard_compact_routes = settings[3]
    model._ds41_row_shard_local_queries = settings[4]
    model._ds41_row_shard_kv_latents = settings[5]
    model._ds41_row_shard_compressor_latents = settings[6]
    model._ds41_row_shard_engram_bands = settings[7]
    _LOG.info(
        "ds41_row_shard %s enabled: TP%d, owned layers 0..%d, min %d rows per rank",
        PLUGIN_VERSION,
        group.world_size,
        model.ced_decoder_start,
        settings[1],
    )


def _local_eligible(model, positions, intermediate_tensors, ced_indices):
    """Return (geometry, eligible) without any collective."""
    import torch
    from vllm.distributed import get_tp_group
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return None, False
    if not is_forward_context_available():
        return None, False
    context = get_forward_context()
    if context.cudagraph_runtime_mode.name != "NONE" or context.ubatch_slices:
        return None, False
    group = get_tp_group()
    rows = positions.shape[0]
    geometry = row_geometry(
        rows, group.world_size, group.rank_in_group, model._ds41_row_shard_min_rows
    )
    if geometry is None:
        return None, False
    metadata = context.attn_metadata
    eligible = (
        ced_indices is not None
        and intermediate_tensors is None
        and isinstance(metadata, dict)
    )
    if eligible:
        first = model.layers[0].attn.swa_cache_layer.prefix
        swa = metadata.get(first)
        eligible = (
            swa is not None
            and type(swa.num_actual_tokens) is int
            and swa.num_actual_tokens == rows
            and not swa.is_decode
        )
    return geometry, eligible


def _model_forward(original):
    @functools.wraps(original)
    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        lookback_token_ids=None,
        ced_indices=None,
    ):
        global _REPORTS
        call = functools.partial(
            original,
            self,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids=lookback_token_ids,
            ced_indices=ced_indices,
        )
        if STATE.ownership is not None:
            raise RuntimeError("nested row-owned forward")
        import torch

        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            return call()
        if not hasattr(self, "_ds41_row_shard_layers"):
            with _LOCK:
                if not hasattr(self, "_ds41_row_shard_layers"):
                    configure(self)
        try:
            geometry, eligible = _local_eligible(
                self, positions, intermediate_tensors, ced_indices
            )
        except Exception:  # never leave the other ranks waiting in the vote
            _LOG.exception("ds41_row_shard: eligibility check failed; voting no")
            from vllm.distributed import get_tp_group

            tp = get_tp_group()
            # Geometry depends only on the shared row count, so every rank
            # still reaches (or skips) the vote together.
            geometry = row_geometry(
                positions.shape[0],
                tp.world_size,
                tp.rank_in_group,
                self._ds41_row_shard_min_rows,
            )
            eligible = False
        if geometry is None:
            return call()
        from vllm.distributed import get_tp_group

        group = get_tp_group()
        # Geometry follows from host values every rank shares; the vote keeps
        # one rank's metadata difference from splitting the reduction paths.
        if not all(_vote(group, eligible)):
            return call()
        ownership = Ownership(
            geometry=geometry,
            group=group,
            gather_rows=sys.modules["vllm.models.deepseek_v4_1.ced"].gather_rows,
            layer_index=self._ds41_row_shard_layers,
            boundary=self.ced_decoder_start,
            compact_routes=self._ds41_row_shard_compact_routes,
            local_queries=self._ds41_row_shard_local_queries,
            gather_kv_latents=self._ds41_row_shard_kv_latents,
            gather_compressor_latents=self._ds41_row_shard_compressor_latents,
            band_engram=self._ds41_row_shard_engram_bands,
        )
        STATE.ownership = ownership
        try:
            with _DeferralValidator():
                result = call()
            ownership.finish()
        finally:
            STATE.ownership = None
        if _REPORTS < 8:
            _REPORTS += 1
            _LOG.info(
                "DS41_ROW_SHARD rank=%d rows=%d owned=%d rs=%d ag=%d compact=%d "
                "engram=%d engram_rs=%d indexer_gathers=%d route_gathers=%d local_attention=%d "
                "kv_latent_gathers=%d compressor_latent_gathers=%d",
                geometry.rank,
                geometry.rows,
                geometry.owned,
                ownership.reduce_scatters,
                ownership.all_gathers,
                ownership.compactions,
                ownership.engram_layers,
                ownership.engram_reduce_scatters,
                ownership.indexer_gathers,
                ownership.route_gathers,
                ownership.local_attention_layers,
                ownership.kv_latent_gathers,
                ownership.compressor_latent_gathers,
            )
        return result

    setattr(forward, MARKER, True)
    return forward


def _layer_forward(original):
    @functools.wraps(original)
    def forward(
        self,
        x,
        positions,
        input_ids,
        pre_mix=None,
        post_mix=None,
        res_mix=None,
        residual=None,
        engram_hashes=None,
        engram_mask=None,
        ced_indices=None,
    ):
        ownership = STATE.ownership
        index = None if ownership is None else ownership.layer_index.get(id(self))
        if index is None or index > ownership.boundary:
            return original(
                self,
                x,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
                engram_hashes,
                engram_mask,
                ced_indices=ced_indices,
            )
        return owned_layer_forward(
            self,
            ownership,
            index,
            x,
            positions,
            input_ids,
            pre_mix,
            post_mix,
            res_mix,
            residual,
            engram_hashes,
            engram_mask,
            ced_indices,
        )

    setattr(forward, MARKER, True)
    return forward


def patch_model(module) -> None:
    for cls, wrap in (
        (module.DeepseekV4Model, _model_forward),
        (module.DeepseekV4DecoderLayer, _layer_forward),
    ):
        current = cls.__dict__["forward"]
        if not getattr(current, MARKER, False):
            cls.forward = wrap(current)


# -- installation ----------------------------------------------------------------
def patch_moe_runner(module) -> None:
    if not _enabled_flag(ROUTES_FLAG):
        return
    pin = PINS[MOE_RUNNER_MODULE][0]
    edit_function(
        module, "MoERunner._forward_impl", pin,
        (("                router_logits, _ = self.gate(hidden_states)\n",
          "                router_logits = _ds41_gate_owned_routes(self, hidden_states)\n"),),
        {"_ds41_gate_owned_routes": gate_owned_routes}, MARKER,
    )
    edit_function(
        module, "MoERunner._apply_quant_method", pin,
        (("            topk_weights, topk_ids = self.router.select_experts(\n",
          "            topk_weights, topk_ids = _ds41_select_owned_routes(self,\n"),),
        {"_ds41_select_owned_routes": select_owned_routes}, MARKER,
    )


_PATCHES = {
    MOE_RUNNER_MODULE: patch_moe_runner,
    ATTENTION_MODULE: patch_attention,
    MODEL_MODULE: patch_model,
}


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, loader, patch):
        self._loader = loader
        self._patch = patch

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        self._patch(module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        patch = _PATCHES.get(name)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchLoader(spec.loader, patch)
        return spec


def install() -> None:
    check_pins()
    with _LOCK:
        if not any(isinstance(f, _PatchFinder) for f in sys.meta_path):
            sys.meta_path.insert(0, _PatchFinder())
    # The attention module must be edited before the model module runs.
    for name in _PATCHES:
        module = sys.modules.get(name)
        if module is not None:
            _PATCHES[name](module)


def register() -> None:
    """vLLM general-plugin entry point; installs nothing unless the flag is 1."""
    if not enabled():
        return
    min_owned_rows()
    if _enabled_flag(COMPRESSOR_LATENTS_FLAG) and not (
        _enabled_flag(LOCAL_QUERIES_FLAG) and _enabled_flag(KV_LATENTS_FLAG)
    ):
        raise PatchRefused(f"{COMPRESSOR_LATENTS_FLAG} requires local queries and narrow KV gathers")
    install()
