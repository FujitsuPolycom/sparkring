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
DEFAULT_RECIPE = "glm52-nf3-hybrid"
EXL3_RECIPE = "glm52-exl3-tr3-3.25bpw"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OCI_DIGEST_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")

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
    if recipe["recipe_id"] == DEFAULT_RECIPE:
        _validate_nf3(recipe)
    elif recipe["recipe_id"] == EXL3_RECIPE:
        _validate_exl3(recipe)
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
    if recipe["default"] is not True:
        raise RecipeError("the NF3 recipe must remain the default")
    if recipe["maturity"] != "public-clean-checkout-live-validated":
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
            "maturity": "public-clean-checkout-live-validated",
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
    if recipe["default"] is not False:
        raise RecipeError("the EXL3 recipe is experimental and cannot be default")
    if recipe["maturity"] != "live-validated":
        raise RecipeError("the EXL3 recipe maturity must remain live-validated")
    _validate_hardware(recipe)

    model = recipe["model"]
    if model.get("repository") != "willfalco/GLM-5.2-EXL3-TR3-3.25bpw":
        raise RecipeError("model.repository is not the validated EXL3 checkpoint")
    _require_commit(model.get("revision"), "model.revision")
    for field in ("config_sha256", "index_sha256", "tier_bitmap_sha256"):
        _require_sha256(model.get(field), f"model.{field}")
    if model.get("shard_count") != 81:
        raise RecipeError("model.shard_count must remain 81")
    if model.get("expected_tiers") != [[3, 192], [4, 64]]:
        raise RecipeError("model.expected_tiers must remain K3x192 plus K4x64")

    runtime = recipe["runtime"]
    if runtime.get("base_recipe") != DEFAULT_RECIPE:
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

    _validate_publication(recipe)
    publication = recipe["publication"]
    if publication.get("local_build_ready") is not False:
        raise RecipeError("EXL3 cannot claim public local-build readiness yet")
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
        "mtp_policy": "fixed-3",
        "max_model_len": 1048576,
        "kv_cache_dtype": "nvfp4_ds_mla",
        "kv_cache_bytes_per_rank": 9000000000,
        "reported_kv_tokens": 1125632,
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
        "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS": "1048576",
        "VLLM_DCP_GLOBAL_TOPK": "1",
        "VLLM_DCP_SHARD_DRAFT": "1",
        "VLLM_EXL3_PREFILL_TRELLIS": "1",
        "VLLM_EXL3_TRELLIS_MIN_M": "1",
        "VLLM_EXL3_TRELLIS_MAX_M": "32",
        "VLLM_NVFP4_MLA_PER_TOKEN_SCALE": "1",
        "VLLM_SPARK_KV_CACHE_MEMORY_BYTES": "9000000000",
        "VLLM_SPARK_MAX_MODEL_LEN": "1048576",
        "VLLM_SPARK_MAX_NUM_BATCHED_TOKENS": "4096",
        "VLLM_SPARK_MAX_NUM_SEQS": "8",
        "VLLM_SPARK_MAX_QUERY_ROWS": "32",
        "VLLM_SPARK_MTP_MODE_ID": "fixed-mtp3",
        "VLLM_SPARK_MTP_TOKENS": "3",
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
