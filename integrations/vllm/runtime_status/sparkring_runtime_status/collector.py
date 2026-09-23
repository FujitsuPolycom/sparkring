"""Allowlisted, passive snapshots. Never call a model, plan, tensor or device."""

from __future__ import annotations

import hashlib
import json
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
}
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
}
INTEGER_ENV = (
    "QWEN_DISPATCH_AR_BYTES", "VLLM_ROCE_ALLREDUCE_MAX_SIZE",
    "VLLM_ROCE_ALLGATHER_MAX_SIZE", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS",
    "SPARKCACHE_CUDA_ARENA_BYTES", "SPARKCACHE_MAX_PENDING_RESTORES",
    "OMP_NUM_THREADS",
)
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
    if isinstance(value, Enum):
        return value.name
    if type(value).__module__ == "torch" and type(value).__name__ == "dtype":
        # torch.dtype string conversion is metadata, never a tensor read.
        return str(value)
    if type(value) is str and len(value) <= 128 and re.fullmatch(r"[\w.+,:/-]*", value):
        return value
    return MISSING


def fact(value=MISSING, *, source, state="known", phase=None):
    value = scalar(value)
    result = {"state": "unknown" if value is MISSING else state, "source": source}
    if value is not MISSING:
        result["value"] = value
    if phase:
        result["phase"] = phase
    return result


def configuration(config, fields=CONFIG_FIELDS):
    result = {key: fact(path(config, route), source="resolved_vllm_config")
              for key, route in fields.items()}
    if fields is CONFIG_FIELDS:
        for key, attr in (("speculative_enabled", "speculative_config"),
                           ("kv_transfer_enabled", "kv_transfer_config")):
            value = stored(config, attr)
            result[key] = fact(MISSING if value is MISSING else value is not None,
                               source="resolved_vllm_config")
    return result


def environment(environ):
    result = {}
    for name in BOOL_ENV:
        raw = environ.get(name)
        result[name] = fact({"0": False, "1": True}.get(raw, MISSING),
                            source="process_environment")
    for name, domain in ENUM_ENV.items():
        raw = environ.get(name)
        result[name] = fact(raw if raw in domain else MISSING,
                            source="process_environment")
    for name in INTEGER_ENV:
        raw = environ.get(name)
        value = int(raw) if type(raw) is str and re.fullmatch(r"[0-9]{1,12}", raw) else MISSING
        result[name] = fact(value, source="process_environment")
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
    result["effective"]["model_runner_v2"] = fact(
        stored(worker, "use_v2_model_runner"), source="worker_instance")
    model = _resident_model(worker)
    result["effective"]["hc_prefill_row_ownership"] = fact(
        stored(model, "hc_prefill_mode"), source="resident_model")
    workspace = _module_child(model, "hyper_connection_workspace")
    result["effective"]["hc_projection_tp_size"] = fact(
        stored(workspace, "tp_size"), source="resident_hc_workspace")
    result["effective"]["kernel_preparation"] = prepared_choices(worker)
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
                "transport_profile": fact(path(data, "capabilities.transport_profile"), source="installed_receipt"),
                "transport_manifest_sha256": _hex(path(data, "capabilities.transport_manifest_sha256")),
                "verification": "receipt_read_at_plugin_startup_no_fresh_file_audit"}
    return {"state": "unknown", "reason": "installed_receipt_missing"}
