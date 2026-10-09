"""Allowlisted, passive snapshots. Never call a model, plan, tensor or device."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from enum import Enum


SCHEMA = "sparkring-runtime-status/v1"
WORKER_SCHEMA = "sparkring-worker-status/v1"
MAX_PLANS = 32
MAX_RECEIPT_BYTES = 4 * 1024 * 1024
MISSING = object()

# These are stored config fields, not properties: reading status cannot resolve
# a platform, load a backend, or cause a lazy preparation.
CONFIG_FIELDS = {
    "tensor_parallel_size": "parallel_config.tensor_parallel_size",
    "decode_context_parallel_size": "parallel_config.decode_context_parallel_size",
    "prefill_context_parallel_size": "parallel_config.prefill_context_parallel_size",
    "pipeline_parallel_size": "parallel_config.pipeline_parallel_size",
    "data_parallel_size": "parallel_config.data_parallel_size",
    "data_parallel_rank": "parallel_config.data_parallel_rank",
    "enable_expert_parallel": "parallel_config.enable_expert_parallel",
    "max_model_len": "model_config.max_model_len",
    # The name clients put in requests. ModelConfig stores one string: the first
    # --served-model-name alias, or the --model value when no alias is given.
    "served_model_name": "model_config.served_model_name",
    # The checkpoint architecture, not a name clients can request.
    "model_type": "model_config.hf_config.model_type",
    "quantization": "model_config.quantization",
    "model_dtype": "model_config.dtype",
    "max_num_batched_tokens": "scheduler_config.max_num_batched_tokens",
    "max_num_seqs": "scheduler_config.max_num_seqs",
    "chunked_prefill_enabled": "scheduler_config.enable_chunked_prefill",
    "prefix_caching_enabled": "cache_config.enable_prefix_caching",
    "cache_dtype": "cache_config.cache_dtype",
    "block_size": "cache_config.block_size",
    "kv_cache_memory_bytes": "cache_config.kv_cache_memory_bytes",
    "mamba_cache_mode": "cache_config.mamba_cache_mode",
    "recurrent_checkpoint_policy": "cache_config.recurrent_checkpoint_policy",
    "kv_connector": "kv_transfer_config.kv_connector",
    "kv_role": "kv_transfer_config.kv_role",
    "speculative_method": "speculative_config.method",
    "num_speculative_tokens": "speculative_config.num_speculative_tokens",
    "cudagraph_mode": "compilation_config.cudagraph_mode",
    "max_cudagraph_capture_size": "compilation_config.max_cudagraph_capture_size",
    "fuse_act_quant": "compilation_config.pass_config.fuse_act_quant",
    "linear_backend": "kernel_config.linear_backend",
    "moe_backend": "kernel_config.moe_backend",
    "gdn_decode_kernel": "additional_config.gdn_decode_kernel",
    "load_format": "load_config.load_format",
    "loader_read_mode": "load_config.model_loader_extra_config.read_mode",
    "loader_io_threads": "load_config.model_loader_extra_config.io_threads",
    "attention_backend": "attention_config.backend",
    "draft_model_type": "speculative_config.draft_model_config.hf_config.model_type",
    "draft_kv_cache_dtype": "speculative_config.kv_cache_dtype",
    "draft_load_format": "speculative_config.draft_load_config.load_format",
    "draft_tensor_parallel_size": "speculative_config.draft_tensor_parallel_size",
    "draft_sample_method": "speculative_config.draft_sample_method",
    "rejection_sample_method": "speculative_config.rejection_sample_method",
    "adaptive_verification": "speculative_config.enable_adaptive_verification",
    # --reasoning-parser; vLLM stores "" when no parser is selected.
    "reasoning_parser": "structured_outputs_config.reasoning_parser",
}
ARG_FIELDS = {
    "tensor_parallel_size": "tensor_parallel_size",
    "decode_context_parallel_size": "decode_context_parallel_size",
    "pipeline_parallel_size": "pipeline_parallel_size",
    "max_model_len": "max_model_len",
    "max_num_batched_tokens": "max_num_batched_tokens",
    "max_num_seqs": "max_num_seqs",
    "block_size": "block_size",
    "prefix_caching_enabled": "enable_prefix_caching",
    "chunked_prefill_enabled": "enable_chunked_prefill",
    "load_format": "load_format",
    "quantization": "quantization",
    "model_dtype": "dtype",
    "cache_dtype": "kv_cache_dtype",
    "kv_cache_memory_bytes": "kv_cache_memory_bytes",
    "mamba_cache_mode": "mamba_cache_mode",
    "recurrent_checkpoint_policy": "recurrent_checkpoint_policy",
    "linear_backend": "linear_backend",
    "moe_backend": "moe_backend",
    "gdn_decode_kernel": "gdn_decode_kernel",
    "gdn_prefill_backend": "gdn_prefill_backend",
    "attention_backend": "attention_config.backend",
    "speculative_method": "speculative_config.method",
    "num_speculative_tokens": "speculative_config.num_speculative_tokens",
    "draft_tensor_parallel_size": "speculative_config.draft_tensor_parallel_size",
    "draft_load_format": "speculative_config.draft_load_config.load_format",
    "draft_kv_cache_dtype": "speculative_config.kv_cache_dtype",
    "draft_sample_method": "speculative_config.draft_sample_method",
    "rejection_sample_method": "speculative_config.rejection_sample_method",
    "adaptive_verification": "speculative_config.enable_adaptive_verification",
    "cudagraph_mode": "compilation_config.cudagraph_mode",
    "max_cudagraph_capture_size": "compilation_config.max_cudagraph_capture_size",
    "fuse_act_quant": "compilation_config.pass_config.fuse_act_quant",
    "loader_read_mode": "model_loader_extra_config.read_mode",
    "loader_io_threads": "model_loader_extra_config.io_threads",
    "reasoning_parser": "reasoning_parser",
    # The API server passes the next two launch arguments to its chat renderer
    # unchanged; VllmConfig has no copy of them, so workers cannot report them.
    "tool_call_parser": "tool_call_parser",
    # Summarized by template_arguments(), because the value is a JSON object.
    "default_chat_template_kwargs": "default_chat_template_kwargs",
}
MAX_TEMPLATE_ARGUMENTS = 16
MAX_TEMPLATE_TEXT = 512
BOOL_ENV = (
    "VLLM_QWEN3_8_FLASH_NEXT_HC_TP", "VLLM_QWEN3_8_PREFILL_COALESCE",
    "VLLM_B12X_KDA_PREFILL_COALESCING", "QWEN_HC_FUSION", "QWEN_MTP_GEMM",
    "SPARKCACHE_ENABLED", "VLLM_ENABLE_ROCE_ALLREDUCE", "B12X_AUTOTUNE",
    "VLLM_MXFP8_LM_HEAD", "VLLM_LM_HEAD_A16", "VLLM_MTP_NVFP4_LM_HEAD",
    "VLLM_QWEN3_8_FLASH_NEXT_OVERLAP", "VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT",
    "VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH", "NCCL_IB_PRESERVE_PCI_DOMAIN",
    "NCCL_IB_SUBNET_AWARE_ROUTING", "NCCL_IB_EXTENDED_IPV4_GIDS",
)
ENUM_ENV = {
    "VLLM_QWEN3_8_HC_PREFILL_MODE": {"off", "control", "shard"},
    "VLLM_USE_V2_MODEL_RUNNER": {"0", "1"},
    "NCCL_ALGO": {"Ring", "Tree", "Ring,Tree", "Tree,Ring"},
    "NCCL_PROTO": {"Simple", "LL", "LL128", "LL,LL128,Simple"},
    "SPARKRING_TRANSPORT_PROFILE": {"tp2-rocenante-adaptive", "tp2-rocenante-adaptive-prepared"},
    "QWEN_DISPATCH_MODE": {"both", "reduce", "nccl"},
    "VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE": {"0", "1", "all", "w13", "w2"},
    # SIRCL ring sessions (spark_transport/sircl/sparkring_sircl/vllm/settings.py).
    "SIRCL_MODE": {"custom", "disabled"},
    "SIRCL_NCCL": {"never", "auto", "topology"},
}
INTEGER_ENV = (
    "QWEN_DISPATCH_AR_BYTES", "VLLM_ROCE_ALLREDUCE_MAX_SIZE",
    "VLLM_ROCE_ALLGATHER_MAX_SIZE", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS",
    "SPARKCACHE_CUDA_ARENA_BYTES", "SPARKCACHE_MAX_PENDING_RESTORES",
    "OMP_NUM_THREADS",
    "NCCL_IB_GID_INDEX",
)
TEXT_ENV = {
    "B12X_ROCE_HCA": r"[A-Za-z0-9_.:,/-]{1,512}",
    "NCCL_IB_HCA": r"[A-Za-z0-9_.:,/\^=-]{1,512}",
    "NCCL_SOCKET_IFNAME": r"[A-Za-z0-9_.:,/\^=-]{1,512}",
    "SIRCL_FABRIC": r"(?:ring|path):[0-9]{1,2}|pair(?::[12])?",
    "SIRCL_RANK_POSITIONS": r"[0-9]{1,2}(?:,[0-9]{1,2}){0,15}",
}
# Durations in seconds, read as vLLM reads them: float(value).
# SPARKRING_SHM_BUSY_LOOP_S is how long vLLM's shared-memory readers poll after
# a read in images derived with runtime/images/derive_spin_wait.py.
SECONDS_ENV = ("SPARKRING_SHM_BUSY_LOOP_S",)
CHOICE_FIELDS = (
    "backend", "tile_m", "tile_n", "tile_k", "load_path", "swap_ab",
    "split_k_slices", "large_m_unroll", "target_occupancy", "num_warps",
    "num_stages", "block_m", "block_n", "block_k", "strategy",
    "mode", "split_k", "algorithm", "segment_tokens", "v_split", "k_split",
    "stages", "window_tiles",
)
QUERY_FIELDS = (
    "recipe", "entry_point", "weight_storage", "output_dtype", "batch",
    "max_rows", "in_features", "out_features", "expected_m",
    "num_tokens", "max_tokens", "max_seqs", "key_heads", "value_heads",
    "head_dim", "checkpoint_export",
)
OBSERVATIONS = (
    "prefill", "checkpoint_coalescing", "hyperconnection", "transport",
    "kernel_execution", "speculative_acceptance", "cache_publication",
)


def stored(obj, name, default=MISSING):
    """Bypass descriptors/__getattr__, including nn.Module's lazy accessors."""
    if type(obj) is dict:
        return obj.get(name, default)
    try:
        values = object.__getattribute__(obj, "__dict__")
    except (AttributeError, TypeError):
        return default
    if type(values) is not dict:
        return default
    return values.get(name, default)


def path(obj, dotted):
    for name in dotted.split("."):
        obj = stored(obj, name)
    return obj


def scalar(value):
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Enum):
        return value.name
    if type(value).__module__ == "torch" and type(value).__name__ == "dtype":
        # torch.dtype string conversion is metadata, never a tensor read.
        return str(value)
    if type(value) is str and len(value) <= 512 and re.fullmatch(r"[\w.+,:/^=-]*", value):
        return value
    return MISSING


def fact(value=MISSING, *, source, state="known", phase=None, reason=None):
    value = scalar(value)
    result = {"state": "unknown" if value is MISSING else state, "source": source}
    if value is not MISSING:
        result["value"] = value
    if phase:
        result["phase"] = phase
    if reason:
        result["reason"] = reason
    return result


def template_arguments(value, source):
    """Summarize --default-chat-template-kwargs as JSON text.

    The argument is a JSON object, which the scalar allowlist cannot carry. Its
    JSON text keeps strings and booleans apart ("false" is not false), is
    printable ASCII because json.dumps escapes every other character, and is
    bounded. None and an empty object both mean the server adds no defaults.
    """
    if value is MISSING:
        return fact(source=source, reason="not_supplied")
    if value is None or type(value) is dict and not value:
        return fact(None, source=source)
    try:
        # Only JSON types serialize; any other object raises TypeError.
        text = json.dumps(value, allow_nan=False) if (
            type(value) is dict and len(value) <= MAX_TEMPLATE_ARGUMENTS) else None
    except (TypeError, ValueError, RecursionError):
        text = None
    if text is None or len(text) > MAX_TEMPLATE_TEXT:
        return fact(source=source, reason="unsupported_value")
    return {"state": "known", "source": source, "value": text}


def configuration(config, fields=CONFIG_FIELDS):
    result = {key: fact(path(config, route), source="resolved_vllm_config",
                        reason=("not_supplied" if fields is ARG_FIELDS else "not_collected")
                        if path(config, route) is MISSING else None)
              for key, route in fields.items()}
    if fields is ARG_FIELDS:
        result["default_chat_template_kwargs"] = template_arguments(
            path(config, ARG_FIELDS["default_chat_template_kwargs"]), "resolved_vllm_config")
    if fields is CONFIG_FIELDS:
        for key, attr in (("speculative_enabled", "speculative_config"),
                           ("kv_transfer_enabled", "kv_transfer_config")):
            value = stored(config, attr)
            result[key] = fact(MISSING if value is MISSING else value is not None,
                               source="resolved_vllm_config")
        for parent, keys in (("kv_transfer_config", ("kv_connector", "kv_role")),
                             ("speculative_config", ("speculative_method", "num_speculative_tokens",
                              "draft_model_type", "draft_tensor_parallel_size", "draft_load_format",
                              "draft_kv_cache_dtype", "draft_sample_method", "rejection_sample_method",
                              "adaptive_verification"))):
            if stored(config, parent) is None:
                for key in keys:
                    result[key] = fact(None, source="resolved_vllm_config", reason="not_applicable")
    return result


def environment(environ):
    result = {}
    for name in BOOL_ENV:
        raw = environ.get(name)
        result[name] = fact({"0": False, "1": True}.get(raw, MISSING),
                            source="process_environment",
                            reason="not_set_in_environment" if raw is None else "invalid_value" if raw not in ("0", "1") else None)
    for name, domain in ENUM_ENV.items():
        raw = environ.get(name)
        result[name] = fact(raw if raw in domain else MISSING,
                            source="process_environment",
                            reason="not_set_in_environment" if raw is None else "invalid_value" if raw not in domain else None)
    for name in INTEGER_ENV:
        raw = environ.get(name)
        value = int(raw) if type(raw) is str and re.fullmatch(r"[0-9]{1,12}", raw) else MISSING
        result[name] = fact(value, source="process_environment",
                            reason="not_set_in_environment" if raw is None else "invalid_value" if value is MISSING else None)
    for name, pattern in TEXT_ENV.items():
        raw = environ.get(name)
        value = raw if type(raw) is str and re.fullmatch(pattern, raw) else MISSING
        result[name] = fact(value, source="process_environment",
                            reason="not_set_in_environment" if raw is None else "invalid_value" if value is MISSING else None)
    for name in SECONDS_ENV:
        raw = environ.get(name)
        try:
            value = float(raw) if type(raw) is str and len(raw) <= 32 else MISSING
        except ValueError:
            value = MISSING
        # fact() withholds inf and nan, which float() accepts.
        result[name] = fact(value, source="process_environment",
                            reason="not_set_in_environment" if raw is None else "invalid_value"
                            if value is MISSING or not math.isfinite(value) else None)
    return result


def observations():
    return {key: {"state": "not_observed", "source": "no_request_instrumentation"}
            for key in OBSERVATIONS}


def _module_child(obj, name):
    child = stored(obj, name)
    return stored(stored(obj, "_modules"), name) if child is MISSING else child


def _resident_model(worker):
    model = _module_child(stored(worker, "model_runner"), "model")
    # Fixed wrapper depth; do not scan named_modules or inspect tensor values.
    for _ in range(4):
        if stored(model, "hc_prefill_mode") is not MISSING:
            return model
        child = _module_child(model, "model")
        if child is MISSING:
            # Qwen's multimodal wrapper owns a causal LM under language_model;
            # inspect only these fixed wrapper edges, never the vision branch.
            child = _module_child(model, "language_model")
        if child is MISSING:
            break
        model = child
    return model


def _model_chain(model):
    chain, seen = [], set()
    for _ in range(6):
        if model is MISSING or model is None or id(model) in seen:
            break
        seen.add(id(model))
        chain.append(model)
        children = (_module_child(model, key) for key in ('language_model', 'model', 'module', '_orig_mod', 'runnable'))
        model = next((child for child in children if child is not MISSING and child is not None), MISSING)
    return chain


def _head(chain):
    return next((head for item in chain if (head := _module_child(item, 'lm_head')) is not MISSING), MISSING)


def model_choices(worker):
    """Read resident head precision and a bounded sample of attention modules."""
    runner = stored(worker, 'model_runner')
    target = _model_chain(_module_child(runner, 'model'))
    speculator = stored(runner, 'speculator')
    if speculator is MISSING:
        speculator = stored(runner, 'drafter')
    draft = _model_chain(_module_child(speculator, 'model'))
    target_head, draft_head = _head(target), _head(draft)
    result = {}
    for role, head in (('target', target_head), ('draft', draft_head)):
        quantization = stored(head, 'runtime_lm_head_quantization')
        if quantization is None:
            quantization = 'unquantized'
        result[role + '_head_quantization'] = fact(quantization, source='resident_lm_head', reason='head_not_exposed' if quantization is MISSING else None)
        a16 = stored(head, 'b12x_activation_mode')
        a16 = a16 == 'a16' if type(a16) is str and a16 in ('a16', 'quantized') else stored(_module_child(head, 'quant_method'), 'use_a16')
        result[role + '_head_a16'] = fact(a16, source='resident_lm_head_quant_method')
    for key, head, mode in (('target_head_mxfp8', target_head, 'mxfp8'), ('draft_head_nvfp4', draft_head, 'nvfp4')):
        quantization = stored(head, 'runtime_lm_head_quantization')
        resolved = quantization == mode if type(quantization) is str or quantization is None else MISSING
        result[key] = fact(resolved, source='resident_lm_head')
    result['draft_head_shared'] = fact(MISSING if target_head is MISSING or draft_head is MISSING else target_head is draft_head,
                                        source='resident_head_identity')
    names, qk, values, prefill, decode = set(), set(), set(), set(), set()
    for model in target:
        layers = stored(_module_child(model, 'layers'), '_modules')
        if type(layers) is not dict:
            continue
        for layer in list(layers.values())[:8]:
            attention = _module_child(layer, 'self_attn')
            if attention is MISSING:
                attention = _module_child(layer, 'linear_attn')
            if attention is MISSING or attention is None:
                continue
            names.add(type(attention).__name__)
            for destination, attributes in ((qk, ('head_k_dim', 'head_dim')), (values, ('head_v_dim', 'v_head_dim'))):
                value = next((stored(attention, attr) for attr in attributes if type(stored(attention, attr)) is int), MISSING)
                if type(value) is int:
                    destination.add(value)
            for destination, attr in ((prefill, 'gdn_prefill_backend'), (decode, 'gdn_decode_kernel')):
                value = scalar(stored(attention, attr))
                if type(value) is str:
                    destination.add(value)
    for key, items in (('decoder_attention_modules', names), ('sampled_qk_head_dims', qk),
                       ('sampled_value_head_dims', values), ('gdn_prefill_backend', prefill),
                       ('resident_gdn_decode_kernel', decode)):
        value = ','.join(str(item) for item in sorted(items)[:8]) if items else MISSING
        result[key] = fact(value, source='resident_first_eight_decoder_layers', reason='field_not_exposed' if value is MISSING else None)
    return result


def prepared_choices(worker):
    session = stored(worker, "_b12x_session")
    plans = stored(session, "_plans")
    if type(plans) not in (list, tuple):
        return {"state": "unknown", "source": "resident_b12x_session"}
    rows = []
    device = None
    for plan in plans[:MAX_PLANS]:
        prepared = stored(plan, "_prepared")
        if stored(prepared, "closed") is not False:
            continue
        selection = stored(prepared, "selection")
        config = stored(selection, "config")
        query = stored(plan, "query")
        row = {"component": scalar(stored(selection, "component_id")),
               "selection_source": scalar(stored(selection, "source")),
               "config": {}, "query": {}}
        for target, fields, obj in (("config", CHOICE_FIELDS, config),
                                     ("query", QUERY_FIELDS, query)):
            for key in fields:
                value = scalar(stored(obj, key))
                if value is not MISSING:
                    row[target][key] = value
        for key in ("component", "selection_source"):
            if row[key] is MISSING:
                row[key] = None
        rows.append(row)
        identity = path(prepared, "device.identity")
        capability = stored(identity, "compute_capability")
        if (type(capability) is tuple and len(capability) == 2
                and all(type(item) is int for item in capability)):
            device = {"compute_capability": list(capability),
                      "sm_count": scalar(stored(identity, "sm_count", None))}
            if device["sm_count"] is MISSING:
                device["sm_count"] = None
    session_state = scalar(stored(session, "state", None))
    return {"state": "known", "source": "resident_b12x_session",
            "phase": "preparation", "session_state": None if session_state is MISSING else session_state,
            "plan_count": len(plans), "inspected_count": min(len(plans), MAX_PLANS),
            "truncated": len(plans) > MAX_PLANS, "selections": rows,
            "device": device, "execution": "not_observed"}


def worker_snapshot(worker, *, environ=None, modules=None):
    """Worker RPC: only Python-owned CPU fields; no dynamic backend imports."""
    environ = os.environ if environ is None else environ
    modules = sys.modules if modules is None else modules
    result = {
        "schema": WORKER_SCHEMA, "collected_at_unix_ns": time.time_ns(),
        "identity": {key: fact(stored(worker, key), source="worker_instance")
                     for key in ("rank", "local_rank")},
        "configured": {"environment": environment(environ)},
        "effective": configuration(stored(worker, "vllm_config")),
        "observed": observations(),
    }
    result["identity"]["process_id"] = fact(os.getpid(), source="worker_process")
    from .binding import worker_identity
    result['identity'].update(worker_identity(stored(worker, 'rank')))
    result["effective"]["model_runner_v2"] = fact(
        stored(worker, "use_v2_model_runner"), source="worker_instance")
    model = _resident_model(worker)
    result["effective"]["hc_prefill_row_ownership"] = fact(
        stored(model, "hc_prefill_mode"), source="resident_model")
    workspace = _module_child(model, "hyper_connection_workspace")
    result["effective"]["hc_projection_tp_size"] = fact(
        stored(workspace, "tp_size"), source="resident_hc_workspace")
    result["effective"]["kernel_preparation"] = prepared_choices(worker)
    result['effective'].update(model_choices(worker))
    from . import resources, transport, versioning
    for name, collect in (
        ('transport', lambda: transport.snapshot(modules=modules, environ=environ)),
        ('resources', resources.snapshot),
        ('versions', lambda: versioning.snapshot(modules=modules)),
    ):
        try:
            result[name] = collect()
        except Exception:
            result[name] = {'state': 'unknown', 'reason': 'optional_snapshot_failed'}
    result['effective'].update(result['transport'].get('effective', {}))
    result['observed'].update(result['transport'].get('observed', {}))
    policy = modules.get("qwen38_collective_policy")
    result["effective"]["qwen_collective_routing_mode"] = fact(
        stored(policy, "MODE"), source="resident_qwen_collective_policy")
    result["effective"]["qwen_collective_allreduce_cutoff_bytes"] = fact(
        stored(policy, "LIMIT"), source="resident_qwen_collective_policy")
    # The hook's first-use markers include warmup. Do not turn them into a
    # real-request assertion or a count of executions.
    announced = stored(modules.get("qwen4_hc_fusion"), "announced")
    if type(announced) is set and announced.intersection({"prefill", "decode"}):
        result["observed"]["hyperconnection"] = {
            "state": "known", "source": "hc_fusion_first_use_markers",
            "phase": "unspecified_includes_warmup",
            "categories": sorted(announced.intersection({"prefill", "decode"})),
            "current_request_execution": "not_observed",
        }
    return result


def _hex(value, length=64):
    return value if type(value) is str and re.fullmatch(f"[0-9a-f]{{{length}}}", value) else None


def provenance(receipt_dir=Path("/opt/sparkring/receipts")):
    """Read a bounded receipt once at plugin startup, not audit runtime files."""
    for name in ("external-base-installed.json", "native-installed.json"):
        try:
            with (receipt_dir / name).open("rb") as handle:
                raw = handle.read(MAX_RECEIPT_BYTES + 1)
            if len(raw) > MAX_RECEIPT_BYTES:
                return {"state": "unknown", "reason": "receipt_too_large"}
            data = json.loads(raw)
            if type(data) is not dict:
                raise ValueError("receipt shape")
            expected_schema = {"external-base-installed.json": "sparkring-external-installed/v1",
                               "native-installed.json": "sparkring-native-installed/v1"}[name]
            if data.get("schema") != expected_schema:
                return {"state": "unknown", "reason": "unsupported_receipt_schema"}
        except FileNotFoundError:
            continue
        except (OSError, ValueError, RecursionError):
            return {"state": "unknown", "reason": "receipt_unreadable"}
        sources = {}
        for key in ("vllm", "b12x"):
            row = path(data, "sources." + key)
            sources[key] = {"commit": _hex(stored(row, "commit"), 40),
                            "archive_sha256": _hex(stored(row, "archive_sha256"))}
        base_id = path(data, "base.config_id")
        if base_id is MISSING:
            base_id = data.get("parent_image_id")
        if type(base_id) is not str or not re.fullmatch(r"sha256:[0-9a-f]{64}", base_id):
            base_id = None
        return {"state": "known", "source": "installed_receipt",
                "receipt_sha256": hashlib.sha256(raw).hexdigest(),
                "composition_sha256": _hex(data.get("composition_sha256")),
                "sources": sources, "parent_image_config_id": base_id,
                "runtime_image_id": {"state": "unknown", "reason": "not_available_inside_container"},
                # The b12x communication bundle the image carries (its published, content-bound name and
                # manifest), not the transport that carries the collectives: the workers report that as
                # tp_collective_transport.
                "b12x_comm_bundle": {
                    "name": fact(path(data, "capabilities.transport_profile"), source="installed_receipt"),
                    "manifest_sha256": _hex(path(data, "capabilities.transport_manifest_sha256")),
                    "role": "carried_by_image_not_the_active_transport"},
                "verification": "receipt_read_at_plugin_startup_no_fresh_file_audit"}
    return {"state": "unknown", "reason": "installed_receipt_missing"}
