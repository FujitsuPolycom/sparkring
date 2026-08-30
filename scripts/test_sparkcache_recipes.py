"""GPU-free contracts for SparkCache compositions and retained receipts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECIPE_DIR = ROOT / "recipes" / "sparkcache"
RECIPE_PATHS = sorted(RECIPE_DIR.glob("*.json"))
ARTIFACTS = {
    "deepseek-v4-flash-0731-tp2-dcp1.json": (
        "0.1.0a1",
        "87c17d8dab5052f5a7833349dc9b99b76a3b6531ca6f0d3deff812f724fecdcc",
    ),
    "deepseek-v4-flash-0731-tp4-dcp1.json": (
        "0.1.0a1",
        "87c17d8dab5052f5a7833349dc9b99b76a3b6531ca6f0d3deff812f724fecdcc",
    ),
    "glm52-exl3-r7-3.5bpw-tp4-dcp4.json": (
        "0.1.0a2",
        "3345b8c574951a8204377b0c27f53765c84b96ab4f5a8ec1ac147574dba7568b",
    ),
}
SOURCE_ARTIFACTS = {
    "glm53-flash-nvfp4-dflash2-bf16-tp4-dcp1.json": (
        "6210f439c64e4079ed3304c9cc181174abb3e6045de740ba7b7c2546bcaf6ac2"
    ),
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_composition_registry_has_all_profiles() -> None:
    assert [path.name for path in RECIPE_PATHS] == [
        "deepseek-v4-flash-0731-tp2-dcp1.json",
        "deepseek-v4-flash-0731-tp4-dcp1.json",
        "glm52-exl3-r7-3.5bpw-tp4-dcp4.json",
        "glm53-flash-nvfp4-dflash2-bf16-tp4-dcp1.json",
    ]


def test_compositions_pin_artifact_and_fail_closed_policy() -> None:
    for path in RECIPE_PATHS:
        recipe = _load(path)
        assert recipe["schema"] == "sparkring-sparkcache-composition/v1"
        assert recipe["status"] in {"implemented", "qualified"}
        assert (path.parent / recipe["base_recipe"]).resolve().is_file()
        artifact = recipe["runtime"]["sparkcache"]
        if path.name in ARTIFACTS:
            version, wheel_sha256 = ARTIFACTS[path.name]
            assert artifact["version"] == version
            assert artifact["wheel_sha256"] == wheel_sha256
            assert recipe["serving"]["max_num_batched_tokens"] == 4096
        else:
            assert artifact["artifact_kind"] == "OCI image overlay"
            assert artifact["source_sha256"] == SOURCE_ARTIFACTS[path.name]
            assert artifact["source_commit"] == (
                "3860a2250193a6679ac6bac857af53e0757841f8"
            )
            assert recipe["runtime"]["sparkcache_image"].startswith(
                "ghcr.io/fujitsupolycom/"
            )
            assert recipe["serving"]["max_num_batched_tokens"] == 8192
        if "deepseek" in path.name:
            assert recipe["serving"]["async_scheduling"] is True
            assert recipe["serving"]["scheduler_reserve_full_isl"] is True
        if "glm53" in path.name:
            assert recipe["serving"]["async_scheduling"] is True
            assert recipe["serving"]["native_prefix_caching"] is True
            assert recipe["serving"]["chunked_prefill"] is True
        assert recipe["serving"]["scheduler_budget_status"] == "qualified"
        assert recipe["sparkcache"]["kv_load_failure_policy"] == "recompute"
        assert recipe["sparkcache"]["streaming_snapshots"] is False
        assert recipe["sparkcache"]["cuda_restore"] is False
        assert "native_restore" not in recipe["sparkcache"]


def test_scheduler_budget_records_evidence_without_an_operator_ceiling() -> None:
    for path in RECIPE_PATHS:
        if path.name in SOURCE_ARTIFACTS:
            continue
        recipe = _load(path)
        limitations = " ".join(recipe["evidence"]["limitations"])
        assert "Operators may choose other values" in limitations
        assert "8192 is known to work" in limitations
        assert "8192 remains unsupported" not in limitations
        assert "only qualified budget" not in limitations


def test_parallelism_matches_physical_rank_count() -> None:
    for path in RECIPE_PATHS:
        recipe = _load(path)
        serving = recipe["serving"]
        assert serving["tensor_parallel_size"] == recipe["hardware"]["ranks"]
        assert serving["node_count"] == recipe["hardware"]["ranks"]
        assert serving["decode_context_parallel_size"] in (1, 4)


def test_composition_evidence_has_claim_shape_and_valid_base() -> None:
    """Each composition states a full claim and names its actual base recipe."""
    required = {"status", "conditions", "result", "conclusion", "limitations", "record"}
    for path in RECIPE_PATHS:
        recipe = _load(path)
        base = _load((path.parent / recipe["base_recipe"]).resolve())
        evidence = recipe["evidence"]
        assert required <= set(evidence), path.name
        assert evidence["status"] == "qualified"
        if path.name in SOURCE_ARTIFACTS:
            assert recipe["status"] == "qualified"
            assert "Four directly cabled" in evidence["conditions"]
        else:
            assert "Historical qualified receipt" in evidence["conditions"]
        assert evidence["conclusion"].strip()
        assert evidence["result"].strip()
        assert evidence["record"] != "README.md"
        assert evidence["artifact_scope"].strip()
        assert base["schema"] == "sparkring-recipe/v1"
        assert base["model"]["repository"] == recipe["model"]["repository"]
        assert base["hardware"]["ranks"] == recipe["hardware"]["ranks"]
        if "evidence" in base:
            assert required <= set(base["evidence"])
        else:
            assert {"status", "evidence"} <= set(base["publication"])
