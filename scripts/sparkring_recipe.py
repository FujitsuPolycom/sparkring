#!/usr/bin/env python3
"""Offline entry point for SparkRing deployment recipes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "recipes"
NF3_RECIPE = "glm52-nf3-hybrid"
EXL3_RECIPE = "glm52-exl3-tr3-3.25bpw"
EXL3_R7_RECIPE = "glm52-exl3-r7-3.5bpw"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OCI_DIGEST_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
DEFAULT_RECIPE = EXL3_RECIPE
TOP_LEVEL_KEYS = {
    "schema",
    "recipe_id",
    "maturity",
    "default",
    "hardware",
    "model",
    "runtime",
    "serving",
    "publication",
}


class RecipeError(ValueError):
    """A recipe is internally inconsistent or unsafe to use."""


def _load(recipe_id: str) -> tuple[dict[str, Any], Path]:
    if not re.fullmatch(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", recipe_id):
        raise RecipeError(f"invalid recipe id: {recipe_id!r}")
    path = RECIPES / f"{recipe_id}.json"
    if not path.is_file():
        raise RecipeError(f"unknown recipe: {recipe_id}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise RecipeError("recipe root must be an object")
    unknown = set(document) - TOP_LEVEL_KEYS
    missing = TOP_LEVEL_KEYS - set(document)
    if unknown:
        raise RecipeError(f"unknown recipe fields: {sorted(unknown)}")
    if missing:
        raise RecipeError(f"missing recipe fields: {sorted(missing)}")
    _validate(document)
    return document, path


def _require_sha256(value: object, field: str) -> None:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise RecipeError(f"{field} must be lowercase SHA-256")


def _require_commit(value: object, field: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise RecipeError(f"{field} must be an immutable 40-hex commit")


def _validate(recipe: dict[str, Any]) -> None:
    if recipe["schema"] != "sparkring-recipe/v1":
        raise RecipeError("schema must be sparkring-recipe/v1")
    if recipe["recipe_id"] == NF3_RECIPE:
        _validate_nf3(recipe)
    elif recipe["recipe_id"] == EXL3_RECIPE:
        _validate_exl3(recipe)
    elif recipe["recipe_id"] == EXL3_R7_RECIPE:
        _validate_exl3_r7(recipe)
    else:
        raise RecipeError(f"unsupported recipe id: {recipe['recipe_id']}")

def _validate_hardware(recipe: dict[str, Any]) -> None:
    if recipe["hardware"] != {
        "platform": "linux/arm64",
        "cuda_arch": "sm_121",
        "ranks": 4,
        "topology": "direct-cycle-4",
    }:
        raise RecipeError("hardware contract drifted from four DGX Sparks")


def _validate_publication(recipe: dict[str, Any]) -> None:
    runtime = recipe["runtime"]
    publication = recipe["publication"]
    final_image = runtime.get("final_image")
    ready = publication.get("zero_build_ready")
    if final_image is None:
        if ready is not False:
            raise RecipeError("zero_build_ready cannot be true without final_image")
    elif not OCI_DIGEST_RE.fullmatch(final_image):
        raise RecipeError("runtime.final_image must be an immutable OCI digest")
    elif ready is not True:
        raise RecipeError("published final_image requires zero_build_ready=true")


def _validate_nf3(recipe: dict[str, Any]) -> None:
    if recipe["default"] is not False:
        raise RecipeError("the NF3 recipe must remain a non-default alternative")
    if recipe["maturity"] != "accepted":
        raise RecipeError(
            "the NF3 recipe maturity must match the accepted public live gate"
        )
    _validate_hardware(recipe)

    model = recipe["model"]
    if model["repository"] != (
        "madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid"
    ):
        raise RecipeError("model.repository is not the supported NF3 checkpoint")
    _require_commit(model["revision"], "model.revision")
    _require_sha256(model["config_sha256"], "model.config_sha256")
    _require_sha256(model["index_sha256"], "model.index_sha256")
    if model["shard_count"] != 184:
        raise RecipeError("model.shard_count must remain 184")
    draft = model["mtp_draft"]
    if draft["repository"] != "aidendle94/GLM-5.2-MXFP4-Experts-GPTQ":
        raise RecipeError("model.mtp_draft repository drifted")
    _require_commit(draft["revision"], "model.mtp_draft.revision")
    for field in (
        "config_sha256",
        "index_sha256",
        "weight_sha256",
        "inputscales_sha256",
    ):
        _require_sha256(draft[field], f"model.mtp_draft.{field}")

    runtime = recipe["runtime"]
    if not OCI_DIGEST_RE.fullmatch(runtime["base_image"]):
        raise RecipeError("runtime.base_image must be digest-pinned")
    _require_commit(runtime["sparkring_source_commit"], "runtime.sparkring_source_commit")
    _require_commit(runtime["b12x_commit"], "runtime.b12x_commit")
    _require_commit(runtime["spark_port_commit"], "runtime.spark_port_commit")
    local_id = runtime["validated_local_image_id"]
    if not isinstance(local_id, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", local_id
    ):
        raise RecipeError("runtime.validated_local_image_id is malformed")

    _validate_publication(recipe)
    local_build_ready = recipe["publication"].get("local_build_ready")
    if local_build_ready is not True:
        raise RecipeError("NF3 source bootstrap must remain local_build_ready")
    for field in ("bootstrap_script", "build_script", "build_containerfile"):
        source_path = runtime.get(field)
        if not isinstance(source_path, str) or not (ROOT / source_path).is_file():
            raise RecipeError(f"runtime.{field} must name a published file")

    serving = recipe["serving"]
    required_serving = {
        "tensor_parallel_size": 4,
        "decode_context_parallel_size": 4,
        "mtp_policy": "adaptive-2-4-window32",
        "kv_cache_dtype": "fp8",
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 8,
        "max_query_rows": 40,
        "workspace_reserve_bytes": 805306368,
        "startup_profile_max_tokens": 2,
        "default_kv_profile": "fp8",
    }
    for key, expected in required_serving.items():
        if serving.get(key) != expected:
            raise RecipeError(
                f"serving.{key} must be {expected!r}; got {serving.get(key)!r}"
            )
    expected_profiles = {
        "fp8": {
            "maturity": "public-source-bootstrap-ready",
            "kv_cache_dtype": "fp8",
            "reported_kv_tokens": 511488,
        },
        "nvfp4-rope8": {
            "maturity": "accepted",
            "kv_cache_dtype": "nvfp4_ds_mla",
            "scale_mode": "per-token",
            "reported_kv_tokens": 875520,
            "equivalent_live_startup_api_healthy": True,
            "public_bootstrap_live_validated": True,
        },
    }
    if serving.get("kv_profiles") != expected_profiles:
        raise RecipeError("serving.kv_profiles drifted from pinned contracts")


def _option_values(args: list[object], option: str) -> list[object]:
    return [args[index + 1] for index, item in enumerate(args[:-1]) if item == option]


def _validate_exl3(recipe: dict[str, Any]) -> None:
    if recipe["default"] is not True:
        raise RecipeError("the validated EXL3 recipe must remain the default")
    if recipe["maturity"] != "live-validated":
        raise RecipeError("the EXL3 recipe maturity must remain live-validated")
    _validate_hardware(recipe)

    model = recipe["model"]
    if model.get("repository") != "willfalco/GLM-5.2-EXL3-TR3-3.25bpw":
        raise RecipeError("model.repository is not the validated EXL3 checkpoint")
    _require_commit(model.get("revision"), "model.revision")
    for field in (
        "config_sha256",
        "index_sha256",
        "tier_bitmap_sha256",
        "manifest_sha256",
    ):
        _require_sha256(model.get(field), f"model.{field}")
    if model.get("shard_count") != 81:
        raise RecipeError("model.shard_count must remain 81")
    if model.get("expected_tiers") != [[3, 192], [4, 64]]:
        raise RecipeError("model.expected_tiers must remain K3x192 plus K4x64")

    runtime = recipe["runtime"]
    if runtime.get("base_recipe") != NF3_RECIPE:
        raise RecipeError("EXL3 must derive from the validated NF3 base recipe")
    local_id = runtime.get("validated_local_image_id")
    if not isinstance(local_id, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", local_id
    ):
        raise RecipeError("runtime.validated_local_image_id is malformed")
    sources = runtime.get("sources")
    if not isinstance(sources, dict) or set(sources) != {
        "sparkinfer",
        "exllamav3",
        "vllm_exl3_port",
    }:
        raise RecipeError("runtime.sources must pin all three EXL3 inputs")
    for name, source in sources.items():
        if not isinstance(source, dict):
            raise RecipeError(f"runtime.sources.{name} must be an object")
        _require_commit(source.get("commit"), f"runtime.sources.{name}.commit")
        _require_commit(source.get("tree"), f"runtime.sources.{name}.tree")
    lmcache_runtime = runtime.get("lmcache")
    if not isinstance(lmcache_runtime, dict):
        raise RecipeError("runtime.lmcache must pin the public LMCache source")
    if lmcache_runtime.get("repository") != (
        "https://github.com/local-inference-lab/LMCache.git"
    ):
        raise RecipeError("runtime.lmcache.repository drifted")
    for field in ("base_commit", "integration_tree", "composed_tree"):
        _require_commit(
            lmcache_runtime.get(field), f"runtime.lmcache.{field}"
        )
    if lmcache_runtime.get("version") != "0.5.2+glm52dcp4.1":
        raise RecipeError("runtime.lmcache.version drifted")
    if lmcache_runtime.get("topology") != (
        "tp4-dcp4-pp1-four-local-servers"
    ):
        raise RecipeError("runtime.lmcache.topology drifted")

    _validate_publication(recipe)
    publication = recipe["publication"]
    if publication.get("local_build_ready") is not True:
        raise RecipeError("EXL3 source bootstrap must remain local_build_ready")
    required_runtime_files = {
        "bootstrap_script": "scripts/bootstrap_exl3.py",
        "download_script": "scripts/download_exl3.py",
        "launcher": "scripts/sparkring_exl3_lmcache_launcher.py",
        "pins": "runtime/exl3/pins.json",
        "build_script": "runtime/exl3/build-image.sh",
        "build_containerfile": "runtime/exl3/Containerfile",
    }
    for field, expected in required_runtime_files.items():
        if runtime.get(field) != expected or not (ROOT / expected).is_file():
            raise RecipeError(
                f"EXL3 runtime.{field} must name the published {expected}"
            )
    if not publication.get("blocker"):
        raise RecipeError("EXL3 must state its public-bootstrap blocker")
    evidence = publication.get("evidence")
    if not isinstance(evidence, str) or not (ROOT / evidence).is_file():
        raise RecipeError("EXL3 publication.evidence must name a published file")

    serving = recipe["serving"]
    required_serving = {
        "served_model_name": "glm-5.2-exl3-tr3-3.25bpw",
        "tensor_parallel_size": 4,
        "decode_context_parallel_size": 4,
        "dcp_backend": "ag_rs",
        "mtp_policy": "fixed-2",
        "max_model_len": 524288,
        "kv_cache_dtype": "nvfp4_ds_mla",
        "kv_cache_bytes_per_rank": 4500000000,
        "reported_kv_tokens": 562688,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 8,
        "max_query_rows": 32,
        "load_format": "safetensors",
    }
    for key, expected in required_serving.items():
        if serving.get(key) != expected:
            raise RecipeError(
                f"serving.{key} must be {expected!r}; got {serving.get(key)!r}"
            )
    environment = serving.get("environment")
    if not isinstance(environment, dict):
        raise RecipeError("serving.environment must be an object")
    required_environment = {
        "SPARK_CONTEXT_CACHE_ENABLE": "0",
        "SPARK_ADAPTIVE_MTP_CONTROL": "0",
        "SPARK_GLM52_MTP_INDEX_REUSE": "0",
        "VLLM_B12X_MLA_CKV_GATHER": "1",
        "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS": "524288",
        "VLLM_DCP_GLOBAL_TOPK": "1",
        "VLLM_DCP_SHARD_DRAFT": "1",
        "VLLM_EXL3_PREFILL_TRELLIS": "1",
        "VLLM_EXL3_TRELLIS_MIN_M": "1",
        "VLLM_EXL3_TRELLIS_MAX_M": "32",
        "VLLM_NVFP4_MLA_PER_TOKEN_SCALE": "1",
        "VLLM_SPARK_KV_CACHE_MEMORY_BYTES": "4500000000",
        "VLLM_SPARK_MAX_MODEL_LEN": "524288",
        "VLLM_SPARK_MAX_NUM_BATCHED_TOKENS": "4096",
        "VLLM_SPARK_MAX_NUM_SEQS": "8",
        "VLLM_SPARK_MAX_QUERY_ROWS": "32",
        "VLLM_SPARK_MTP_MODE_ID": "fixed-mtp2",
        "VLLM_SPARK_MTP_TOKENS": "2",
        "VLLM_SPARK_TP4_MODE": "custom",
    }
    for key, expected in required_environment.items():
        if environment.get(key) != expected:
            raise RecipeError(
                f"serving.environment.{key} must be {expected!r}; "
                f"got {environment.get(key)!r}"
            )
    for key in (
        "HYBRID_NF3",
        "VLLM_PREFIX_CACHE_RETENTION_INTERVAL",
        "VLLM_SPARK_NF3_PROFILE",
        "VLLM_SPARK_NF3_SINGLE_COMPILE_RANGE",
        "VLLM_SPARK_NF3_STARTUP_PROFILE_MAX_TOKENS",
        "VLLM_SPARK_NF3_WORKSPACE_RESERVE_BYTES",
    ):
        if key not in environment or environment[key] is not None:
            raise RecipeError(f"serving.environment.{key} must be explicitly unset")

    args = serving.get("vllm_args")
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise RecipeError("serving.vllm_args must be a string list")
    required_options = {
        "--quantization": "exl3",
        "--moe-backend": "b12x",
        "--dcp-comm-backend": "ag_rs",
        "--attention-backend": "B12X_MLA_SPARSE",
        "--kv-cache-dtype": "nvfp4_ds_mla",
        "--max-cudagraph-capture-size": "32",
        "--max-num-batched-tokens": "4096",
        "--load-format": "safetensors",
    }
    for option, expected in required_options.items():
        if _option_values(args, option) != [expected]:
            raise RecipeError(f"serving.vllm_args requires {option} {expected}")
    if "--enforce-eager" in args:
        raise RecipeError("EXL3 recipe requires CUDA graphs")
    lmcache = serving.get("lmcache")
    expected_lmcache = {
        "connector": "LMCacheMPConnector",
        "connector_module": "lmcache.integration.vllm.lmcache_mp_connector",
        "server_topology": "one-local-server-per-rank",
        "server_port": 6556,
        "http_port": 18081,
        "chunk_size": 512,
        "parent_chunk_size": 256,
        "l1_size_gb": 1,
        "l1_init_size_gb": 0,
        "l1_lazy": True,
        "max_gpu_workers": 1,
        "max_cpu_workers": 2,
        "transfer_mode": "lmcache_driven",
        "eviction_policy": "LRU",
        "mq_timeout_seconds": 10,
        "heartbeat_interval_seconds": 10,
        "load_failure_policy": "recompute",
    }
    if lmcache != expected_lmcache:
        raise RecipeError("serving.lmcache drifted from the CS512 contract")


def _validate_exl3_r7(recipe: dict[str, Any]) -> None:
    """Validate the operator-accepted EXL3 R7 3.5-bpw recipe."""
    if recipe["default"] is not False:
        raise RecipeError("the R7 candidate must remain non-default")
    if recipe["maturity"] != "accepted":
        raise RecipeError("the R7 recipe maturity must remain accepted")
    _validate_hardware(recipe)

    model = recipe["model"]
    if model.get("repository") != (
        "brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78"
    ):
        raise RecipeError("R7 model.repository is not the qualified checkpoint")
    _require_commit(model.get("revision"), "model.revision")
    for field in ("config_sha256", "index_sha256"):
        _require_sha256(model.get(field), f"model.{field}")
    if model.get("shard_count") != 157:
        raise RecipeError("R7 model.shard_count must remain 157")
    if model.get("weight_count") != 186905:
        raise RecipeError("R7 model.weight_count must remain 186905")
    if model.get("index_total_size") != 346218639128:
        raise RecipeError("R7 model.index_total_size must remain 346218639128")

    runtime = recipe["runtime"]
    if runtime.get("final_image") is not None:
        raise RecipeError("R7 runtime.final_image must be null (no registry image)")
    local_id = runtime.get("validated_local_image_id")
    if not isinstance(local_id, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", local_id
    ):
        raise RecipeError("R7 runtime.validated_local_image_id is malformed")
    pins = runtime.get("pins")
    if pins != "runtime/exl3-r7/pins.json":
        raise RecipeError("R7 runtime.pins must point to the builder branch pins")
    if not (ROOT / "scripts" / "download_exl3_r7.py").is_file():
        raise RecipeError("R7 runtime.download_script must be published")
    if not (ROOT / "scripts" / "generate_exl3_r7_candidate.py").is_file():
        raise RecipeError("R7 runtime.profile_generator must be published")
    if not (ROOT / "scripts" / "prepare_exl3_r7_mtp4.py").is_file():
        raise RecipeError("R7 runtime.mtp4_generator must be published")
    required_runtime_files = {
        "nvfp4_generator": "scripts/prepare_exl3_r7_mtp4_nvfp4.py",
        "ckv_gather_generator": "scripts/prepare_exl3_r7_mtp4_ckv_gather.py",
        "sircl_tiered_generator": "scripts/prepare_exl3_r7_sircl_tiered.py",
        "exact_q40_profile_generator": (
            "spark_transport/experiments/moe_round_floor/"
            "prepare_q40_exact_state_serving.py"
        ),
        "exact_q40_patch": (
            "spark_transport/experiments/moe_round_floor/q40_exact_state.patch"
        ),
        "exact_q40_attestation_overlay": (
            "spark_transport/experiments/moe_round_floor/"
            "q40_exact_state_attestation_overlay.py"
        ),
    }
    for field, expected in required_runtime_files.items():
        if runtime.get(field) != expected or not (ROOT / expected).is_file():
            raise RecipeError(f"R7 runtime.{field} must name published source")

    _validate_publication(recipe)
    publication = recipe["publication"]
    if publication.get("zero_build_ready") is not False:
        raise RecipeError("R7 candidate must not claim zero_build_ready")
    if publication.get("local_build_ready") is not True:
        raise RecipeError("R7 public builder must remain local_build_ready")
    if not publication.get("remaining_gate"):
        raise RecipeError("R7 must state the clean-checkout live promotion gate")
    checklist = publication.get("promotion_checklist")
    if not isinstance(checklist, str) or not (ROOT / checklist).is_file():
        raise RecipeError("R7 promotion_checklist must name a published file")
    if publication.get("operator_default") is not True:
        raise RecipeError("R7 publication.operator_default must remain true")
    evidence = publication.get("evidence")
    if not isinstance(evidence, str) or not (ROOT / evidence).is_file():
        raise RecipeError("R7 publication.evidence must name a published file")

    serving = recipe["serving"]
    required_serving = {
        "served_model_name": "glm-5.2-exl3-r7-3.5bpw",
        "tensor_parallel_size": 4,
        "decode_context_parallel_size": 4,
        "dcp_backend": "ag_rs",
        "sircl_graph_protocol": "two_slot_deferred_ack",
        "sircl_kernel_strategy": "tiered_64k",
        "sircl_dual_port": False,
        "mtp_policy": "fixed-4",
        "max_model_len": 262144,
        "kv_cache_dtype": "nvfp4_ds_mla",
        "kv_dynamic_per_token_scale": True,
        "kv_fp8_rope": True,
        "kv_record_bytes": 368,
        "kv_cache_bytes_per_rank": 9250000000,
        "reported_kv_tokens": 1156864,
        "gpu_memory_utilization": 0.85,
        "max_num_batched_tokens": 4096,
        "exl3_prefill_capacity": 4096,
        "max_num_seqs": 8,
        "max_query_rows": 40,
        "load_format": "instanttensor",
        "online_quantization": "exl3-b6",
        "online_quantization_scope": "target-only",
        "online_cache_mode": "readwrite",
        "execution_mode": "FULL_AND_PIECEWISE",
        "native_prefix_caching": True,
        "lmcache": False,
        "sparkcache": False,
    }
    for key, expected in required_serving.items():
        if serving.get(key) != expected:
            raise RecipeError(
                f"R7 serving.{key} must be {expected!r}; got {serving.get(key)!r}"
            )
    for transport_field in (
        "tensor_parallel_transport",
        "dcp_transport",
        "indexer_transport",
    ):
        if transport_field not in serving:
            raise RecipeError(f"R7 serving.{transport_field} must be declared")
    if serving.get("tensor_parallel_transport") != (
        "sircl-with-nccl-fallback"
    ):
        raise RecipeError("R7 TP transport must be the qualified hybrid")
    if serving.get("dcp_transport") != "stock":
        raise RecipeError("R7 DCP transport must remain stock")
    if serving.get("indexer_transport") != "stock":
        raise RecipeError("R7 indexer transport must remain stock")
    expected_q40 = {
        "scope": "target-mixed-exl3-only",
        "query_rows": 40,
        "capacity_rows": 40,
        "route_block_rows": 8,
        "uses_prefill_tiers": True,
        "q1_q32_unchanged": True,
        "other_prefill_shapes_unchanged": True,
        "uniform_draft_unchanged": True,
    }
    if serving.get("exact_q40_policy") != expected_q40:
        raise RecipeError("R7 serving.exact_q40_policy drifted")
    expected_ckv = {
        "implementation": "b12x-transient-full-ckv-dcp-gather",
        "enabled": True,
        "maximum_logical_tokens": 262144,
        "workspace_mib_per_rank": 414.4,
        "spark_custom_ckv_allgather_enabled": False,
    }
    if serving.get("ckv_gather") != expected_ckv:
        raise RecipeError("R7 serving.ckv_gather drifted")


def _canonical_digest(recipe: dict[str, Any]) -> str:
    payload = json.dumps(
        recipe, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _plan(recipe: dict[str, Any], path: Path, as_json: bool) -> int:
    final_image = recipe["runtime"]["final_image"]
    result = {
        "schema": "sparkring-recipe-plan/v1",
        "recipe_id": recipe["recipe_id"],
        "recipe_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "recipe_sha256": _canonical_digest(recipe),
        "model": (
            f'{recipe["model"]["repository"]}@{recipe["model"]["revision"]}'
        ),
        "platform": recipe["hardware"]["platform"],
        "final_image": final_image,
        "zero_build_ready": recipe["publication"]["zero_build_ready"],
        "local_build_ready": recipe["publication"]["local_build_ready"],
        "blocker": recipe["publication"]["blocker"],
    }
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f'RECIPE: {result["recipe_id"]}')
        print(f'MODEL: {result["model"]}')
        print(f'PLATFORM: {result["platform"]}')
        print(f'RECIPE_SHA256: {result["recipe_sha256"]}')
        if final_image:
            print(f"IMAGE: {final_image}")
            print("NEXT: configure scripts/config/site.yaml and run preflight")
        elif recipe["publication"]["local_build_ready"]:
            print("IMAGE: built locally from pinned public inputs")
            print(
                f'NEXT: python {recipe["runtime"]["bootstrap_script"]} plan '
                "--site scripts/config/site.yaml"
            )
        else:
            print("IMAGE: no public reproducible image yet")
            print(f'BLOCKER: {recipe["publication"]["blocker"]}')
    return 0


def _list(as_json: bool) -> int:
    rows = []
    for path in sorted(RECIPES.glob("*.json")):
        recipe, _ = _load(path.stem)
        rows.append(
            {
                "recipe_id": recipe["recipe_id"],
                "maturity": recipe["maturity"],
                "default": recipe["default"],
                "zero_build_ready": recipe["publication"]["zero_build_ready"],
                "local_build_ready": recipe["publication"]["local_build_ready"],
            }
        )
    if as_json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for row in rows:
            if row["zero_build_ready"]:
                state = "ready"
            elif row["local_build_ready"]:
                state = "local-build-ready"
            else:
                state = "documented-live-candidate"
            print(f'{row["recipe_id"]}\t{state}\t{row["maturity"]}')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and plan SparkRing deployment recipes."
    )
    parser.add_argument("action", choices=("list", "plan"))
    parser.add_argument("--recipe", default=DEFAULT_RECIPE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "list":
            return _list(args.json)
        recipe, path = _load(args.recipe)
        return _plan(recipe, path, args.json)
    except (OSError, json.JSONDecodeError, RecipeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
