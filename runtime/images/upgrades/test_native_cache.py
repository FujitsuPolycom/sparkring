"""Native reuse includes build metadata and native-language files outside csrc."""

from .sources import native_digest
from runtime.images.upgrades.build_native import select_native_cache, validate_recipe
from runtime.images.upgrades.contracts import Refused, sha
import json
import pytest


def test_python_edits_do_not_alias_changed_native_inputs(tmp_path):
    (tmp_path / "csrc").mkdir()
    (tmp_path / "vllm").mkdir()
    (tmp_path / "csrc/op.cu").write_text("native")
    (tmp_path / "setup.py").write_text("build")
    (tmp_path / "vllm/model.py").write_text("model")
    before = native_digest(tmp_path, ["csrc", "setup.py"])
    (tmp_path / "vllm/model.py").write_text("updated Python")
    assert native_digest(tmp_path, ["csrc", "setup.py"]) == before
    (tmp_path / "vllm/helper.cpp").write_text("outside declared native directory")
    assert native_digest(tmp_path, ["csrc", "setup.py"]) != before


def test_build_metadata_changes_require_recompilation(tmp_path):
    (tmp_path / "setup.py").write_text("one")
    before = native_digest(tmp_path, ["setup.py"])
    (tmp_path / "setup.py").write_text("two")
    assert native_digest(tmp_path, ["setup.py"]) != before


@pytest.fixture
def cache_selection(tmp_path):
    inputs = {"vllm": "a" * 64, "b12x": "b" * 64}
    compiler = "sha256:" + "c" * 64
    recipe = {
        "architecture": "12.1a",
        "torch_version": "2.13.0",
        "build_type": "Release",
    }
    record = {
        "schema": "sparkring-native-cache/v1",
        "native_inputs": inputs,
        "compiler_image_id": compiler,
        **recipe,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(record))
    policy = {
        "foundation": {
            "native_cache": {"manifest": str(path), "sha256": sha(path.read_bytes())}
        },
        "build": {"supports_native_rebuild": True},
    }
    return policy, inputs, compiler, recipe, path


def test_exact_native_inputs_reuse_without_rebuild_permission(cache_selection):
    policy, inputs, compiler, recipe, path = cache_selection
    policy["build"]["supports_native_rebuild"] = False
    record, directory, decision = select_native_cache(policy, inputs, compiler, recipe)
    assert record == json.loads(path.read_text())
    assert directory == path.parent
    assert decision["mode"] == "reuse"


@pytest.mark.parametrize(
    "changed",
    [
        "native_inputs",
        "compiler_image_id",
        "architecture",
        "torch_version",
        "build_type",
    ],
)
def test_input_drift_requires_explicit_bounded_rebuild(cache_selection, changed):
    policy, inputs, compiler, recipe, _ = cache_selection
    if changed == "native_inputs":
        inputs = {**inputs, "vllm": "d" * 64}
    elif changed == "compiler_image_id":
        compiler = "sha256:" + "d" * 64
    else:
        recipe = {**recipe, changed: "different"}
    with pytest.raises(Refused, match="rebuild is required"):
        select_native_cache(policy, inputs, compiler, recipe)
    policy["foundation"]["native_cache"]["on_input_change"] = "rebuild"
    record, directory, decision = select_native_cache(policy, inputs, compiler, recipe)
    assert record is directory is None
    assert decision["mode"] == "compile"
    assert decision["changed_inputs"] == [changed]


def test_corrupt_manifest_never_becomes_automatic_rebuild(cache_selection):
    policy, inputs, compiler, recipe, path = cache_selection
    policy["foundation"]["native_cache"]["on_input_change"] = "rebuild"
    path.write_text("{}")
    with pytest.raises(Refused, match="manifest differs"):
        select_native_cache(policy, inputs, compiler, recipe)


def test_no_cache_selects_full_compile(cache_selection):
    policy, inputs, compiler, recipe, _ = cache_selection
    policy["foundation"].pop("native_cache")
    record, directory, decision = select_native_cache(policy, inputs, compiler, recipe)
    assert record is directory is None
    assert decision["mode"] == "compile"


def test_cache_without_build_mode_is_never_reused(cache_selection):
    policy, inputs, compiler, recipe, path = cache_selection
    record = json.loads(path.read_text())
    record.pop("build_type")
    path.write_text(json.dumps(record))
    policy["foundation"]["native_cache"]["sha256"] = sha(path.read_bytes())
    with pytest.raises(Refused, match="rebuild is required"):
        select_native_cache(policy, inputs, compiler, recipe)
    policy["foundation"]["native_cache"]["on_input_change"] = "rebuild"
    record, directory, decision = select_native_cache(policy, inputs, compiler, recipe)
    assert record is directory is None
    assert decision["changed_inputs"] == ["build_type"]


@pytest.mark.parametrize(
    "mode", ["Release", "RelWithDebInfo", None, "Debug", "release"]
)
def test_recipe_requires_an_explicit_supported_build_mode(mode):
    recipe = {
        "schema": "sparkring-native-recipe/v1",
        "architecture": "12.1a",
        "jobs": 1,
        "cpus": 1,
        "memory_bytes": 8 * 1024**3,
        "build_seconds": 60,
        "torch_version": "2.13.0",
        "network": "none",
    }
    if mode is not None:
        recipe["build_type"] = mode
    if mode in ("Release", "RelWithDebInfo"):
        assert validate_recipe(recipe)["build_type"] == mode
    else:
        with pytest.raises(Refused):
            validate_recipe(recipe)
