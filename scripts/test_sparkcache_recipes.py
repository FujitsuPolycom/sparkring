"""GPU-free contracts for SparkCache compositions and retained receipts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_RECIPE_DIR = ROOT / "recipes"
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
    "glm53-flash-nvfp4-dflash2-bf16-tp4.json": (
        "f8adb4ecdadd524e79cf1ef14e7f3d83d1f20ff07c79333b2c7c0d9ea12919d5"
    ),
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_recipe_indexes_link_every_machine_readable_recipe() -> None:
    base_index = (BASE_RECIPE_DIR / "README.md").read_text(encoding="utf-8")
    composition_index = (RECIPE_DIR / "README.md").read_text(encoding="utf-8")

    for path in sorted(BASE_RECIPE_DIR.glob("*.json")):
        assert f"]({path.name})" in base_index, path.name
    for path in RECIPE_PATHS:
        assert f"]({path.name})" in composition_index, path.name


def test_composition_registry_has_all_profiles() -> None:
    assert [path.name for path in RECIPE_PATHS] == [
        "deepseek-v4-flash-0731-tp2-dcp1.json",
        "deepseek-v4-flash-0731-tp4-dcp1.json",
        "glm52-exl3-r7-3.5bpw-tp4-dcp4.json",
        "glm53-flash-nvfp4-dflash2-bf16-tp4.json",
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
            if "glm53" in path.name:
                assert artifact["artifact_kind"] == "source-pinned OCI image"
            else:
                assert artifact["artifact_kind"] == "OCI image overlay"
            assert artifact["source_sha256"] == SOURCE_ARTIFACTS[path.name]
            assert artifact["source_commit"] == (
                "9c6218c96f1db233c0d17691dbc32a7d9fb2c0e4"
            )
            assert recipe["runtime"]["image"].startswith("ghcr.io/fujitsupolycom/")
            assert recipe["serving_common"]["max_num_batched_tokens"] == 8192
        if "deepseek" in path.name:
            assert recipe["serving"]["async_scheduling"] is True
            assert recipe["serving"]["scheduler_reserve_full_isl"] is True
        if "glm53" in path.name:
            assert recipe["serving_common"]["async_scheduling"] is True
            assert recipe["serving_common"]["native_prefix_caching"] is True
            assert recipe["serving_common"]["chunked_prefill"] is True
            assert recipe["serving_common"]["scheduler_budget_status"] == "qualified"
        else:
            assert recipe["serving"]["scheduler_budget_status"] == "qualified"
        assert recipe["sparkcache"]["kv_load_failure_policy"] == "recompute"
        assert recipe["sparkcache"]["streaming_snapshots"] is False
        if "glm53" in path.name:
            assert recipe["sparkcache"]["cuda_restore"] is True
        else:
            assert recipe["sparkcache"]["native_restore"] is False


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
        serving = recipe.get("serving", recipe.get("serving_common"))
        assert serving is not None
        assert serving["tensor_parallel_size"] == recipe["hardware"]["ranks"]
        assert serving["node_count"] == recipe["hardware"]["ranks"]
        if "profiles" in recipe:
            assert {
                profile["decode_context_parallel_size"]
                for profile in recipe["profiles"].values()
            } == {1, 2, 4}
        else:
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
            assert "four directly connected" in evidence["conditions"].lower()
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


def test_glm53_composition_matches_the_operator_contract() -> None:
    recipe = _load(RECIPE_DIR / "glm53-flash-nvfp4-dflash2-bf16-tp4.json")
    base = _load(ROOT / "recipes" / "glm53-flash-nvfp4-dflash2-bf16-tp4.json")
    pins = _load(ROOT / "runtime" / "glm53-flash-jj-r8-gb10" / "pins.json")
    receipt = _load(
        ROOT
        / "runtime"
        / "glm53-flash-jj-r8-gb10"
        / "async-store-completion-public-image-receipt.json"
    )

    assert recipe["base_recipe"] == "../glm53-flash-nvfp4-dflash2-bf16-tp4.json"
    assert recipe["preferred_profile"] == base["preferred_profile"] == "dcp4"
    assert (
        set(recipe["profiles"])
        == set(base["profiles"])
        == {
            "dcp1",
            "dcp2",
            "dcp4",
        }
    )
    assert recipe["runtime"]["image"] == receipt["artifact"]["registry"]
    assert recipe["runtime"]["image_id"] == receipt["artifact"]["image_id"]
    assert (
        recipe["runtime"]["sparkcache"]["source_commit"] == pins["sparkcache"]["commit"]
    )
    assert (
        recipe["runtime"]["sparkcache"]["source_sha256"]
        == pins["sparkcache"]["source_tree_sha256"]
    )
    assert recipe["runtime"]["vllm"]["commit"] == pins["vllm"]["commit"]
    assert (
        recipe["serving_common"]["max_model_len"] == pins["defaults"]["max_model_len"]
    )
    assert (
        recipe["serving_common"]["max_num_batched_tokens"]
        == pins["defaults"]["max_num_batched_tokens"]
    )

    expected_bytes = pins["defaults"]["kv_cache_bytes_per_rank"]
    for name, profile in recipe["profiles"].items():
        assert profile["kv_cache_memory_bytes_per_rank"] == expected_bytes[name]
    assert recipe["profiles"]["dcp1"]["async_page_capture"] is False
    assert recipe["profiles"]["dcp2"]["async_page_capture"] is False
    assert recipe["profiles"]["dcp4"]["async_page_capture"] is True
    assert recipe["profiles"]["dcp4"]["status"] == "qualified"
    assert recipe["profiles"]["dcp4"]["preferred"] is True
    assert (
        recipe["profiles"]["dcp4"]["capture_slot_bytes"]
        == receipt["configuration"]["async_capture_slot_bytes"]
    )
    assert recipe["evidence"]["machine_receipt"] == (
        "runtime/glm53-flash-jj-r8-gb10/"
        "async-store-completion-public-image-receipt.json"
    )
