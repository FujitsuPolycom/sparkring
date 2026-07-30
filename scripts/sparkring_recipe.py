#!/usr/bin/env python3
"""Offline entry point for the single supported SparkRing deployment recipe."""

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
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", recipe_id):
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
    if recipe["recipe_id"] != DEFAULT_RECIPE:
        raise RecipeError("the public deployment surface admits only NF3")
    if recipe["default"] is not True:
        raise RecipeError("the NF3 recipe must be the sole default")

    hardware = recipe["hardware"]
    if hardware != {
        "platform": "linux/arm64",
        "cuda_arch": "sm_121",
        "ranks": 4,
        "topology": "direct-cycle-4",
    }:
        raise RecipeError("hardware contract drifted from four DGX Sparks")

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

    final_image = runtime["final_image"]
    ready = recipe["publication"]["zero_build_ready"]
    local_build_ready = recipe["publication"].get("local_build_ready")
    if final_image is None:
        if ready is not False:
            raise RecipeError("zero_build_ready cannot be true without final_image")
    elif not OCI_DIGEST_RE.fullmatch(final_image):
        raise RecipeError("runtime.final_image must be an immutable OCI digest")
    elif ready is not True:
        raise RecipeError("published final_image requires zero_build_ready=true")
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
            "maturity": "offline-validated",
            "kv_cache_dtype": "nvfp4_ds_mla",
            "scale_mode": "per-token",
            "reported_kv_tokens": 875520,
            "equivalent_live_startup_api_healthy": True,
            "public_bootstrap_live_validated": False,
        },
    }
    if serving.get("kv_profiles") != expected_profiles:
        raise RecipeError("serving.kv_profiles drifted from pinned contracts")


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
        else:
            print("IMAGE: built locally from pinned public inputs")
            print(
                "NEXT: python scripts/bootstrap_nf3.py plan "
                "--site scripts/config/site.yaml"
            )
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
            }
        )
    if as_json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for row in rows:
            state = "ready" if row["zero_build_ready"] else "local-build-ready"
            print(f'{row["recipe_id"]}\t{state}\t{row["maturity"]}')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and plan the supported SparkRing NF3 recipe."
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
