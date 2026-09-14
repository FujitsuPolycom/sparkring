"""CPU-only admission and activation checks for image-baked feature bundles."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from runtime.images import feature_bootstrap as bootstrap


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    # Activation may prepend bundle directories; keep test imports isolated.
    monkeypatch.setattr(bootstrap.sys, "path", list(bootstrap.sys.path))
    features = {}
    for name, directory, module in (
        ("qwen-collectives", "collectives", "qwen38_collective_policy.py"),
        ("qwen-prefill", "prefill", "prefill_bootstrap.py"),
    ):
        relative = f"{directory}/{module}"
        path = tmp_path / relative
        path.parent.mkdir()
        content = f'"""Fixture asset for {name}."""\n'.encode()
        path.write_bytes(content)
        features[name] = {
            "directory": directory,
            "files": {relative: hashlib.sha256(content).hexdigest()},
            "manifest_sha256": hashlib.sha256(name.encode()).hexdigest(),
        }
    record = {"schema": "sparkring-image-capabilities/v1", "features": features}
    (tmp_path / "capabilities.json").write_text(json.dumps(record), encoding="utf-8")
    return tmp_path, record


@pytest.mark.parametrize("selection", [None, "", "  ", " , , "])
def test_default_does_not_read_inventory_import_or_mutate(tmp_path, monkeypatch, selection):
    environment = {} if selection is None else {"SPARKRING_FEATURES": selection}
    before = dict(environment)
    paths = list(bootstrap.sys.path)
    reader = Mock(side_effect=AssertionError("No feature inventory should be read"))
    importer = Mock(side_effect=AssertionError("No feature hook should be imported"))
    monkeypatch.setattr(bootstrap, "description", reader)
    bootstrap.install(tmp_path / "absent", environment, importer)
    reader.assert_not_called()
    importer.assert_not_called()
    assert environment == before
    assert bootstrap.sys.path == paths


@pytest.mark.parametrize("selection", [
    "unregistered", "qwen-collectives,unregistered",
    "qwen-collectives,qwen-collectives", " qwen-prefill , qwen-prefill ",
])
def test_invalid_selection_fails_before_verification_or_activation(inventory, monkeypatch, selection):
    root, _ = inventory
    environment = {"SPARKRING_FEATURES": selection}
    paths = list(bootstrap.sys.path)
    verifier, importer = Mock(), Mock()
    monkeypatch.setattr(bootstrap, "verify_assets", verifier)
    with pytest.raises(ValueError, match="Unknown or duplicate"):
        bootstrap.install(root, environment, importer)
    verifier.assert_not_called()
    importer.assert_not_called()
    assert environment == {"SPARKRING_FEATURES": selection}
    assert bootstrap.sys.path == paths


@pytest.mark.parametrize("damage", ["changed", "missing"])
def test_all_selected_assets_are_checked_before_any_activation(inventory, damage):
    root, record = inventory
    relative = next(iter(record["features"]["qwen-prefill"]["files"]))
    if damage == "changed":
        (root / relative).write_bytes(b"changed fixture\n")
    else:
        (root / relative).unlink()
    environment = {"SPARKRING_FEATURES": "qwen-collectives,qwen-prefill"}
    before, paths = dict(environment), list(bootstrap.sys.path)
    importer = Mock()
    expected = ValueError if damage == "changed" else FileNotFoundError
    with pytest.raises(expected):
        bootstrap.install(root, environment, importer)
    importer.assert_not_called()
    assert environment == before
    assert bootstrap.sys.path == paths


def test_unselected_broken_bundle_does_not_prevent_selected_feature(inventory):
    root, record = inventory
    for relative in record["features"]["qwen-prefill"]["files"]:
        (root / relative).unlink()
    importer = Mock()
    bootstrap.install(root, {"SPARKRING_FEATURES": "qwen-collectives"}, importer)
    importer.assert_called_once_with("qwen38_collective_policy")


def test_collective_defaults_are_available_before_hook_import(inventory):
    root, record = inventory
    environment = {"SPARKRING_FEATURES": "qwen-collectives"}

    def importing(name):
        assert name == "qwen38_collective_policy"
        assert environment["QWEN_DISPATCH_MODE"] == "both"
        assert environment["QWEN_DISPATCH_AR_BYTES"] == "20480"
        assert environment["QWEN_DISPATCH_TRACE"] == "0"
        assert str(root / record["features"]["qwen-collectives"]["directory"]) in bootstrap.sys.path

    importer = Mock(side_effect=importing)
    bootstrap.install(root, environment, importer)
    importer.assert_called_once()
    assert "SPARKRING_QWEN_PREFILL_MANIFEST_SHA256" not in environment


@pytest.mark.parametrize("overrides", [
    {"QWEN_DISPATCH_MODE": "all-reduce", "QWEN_DISPATCH_AR_BYTES": "4096", "QWEN_DISPATCH_TRACE": "1"},
    {"QWEN_DISPATCH_MODE": "", "QWEN_DISPATCH_AR_BYTES": "0", "QWEN_DISPATCH_TRACE": "0"},
])
def test_collective_overrides_are_not_replaced(inventory, overrides):
    root, _ = inventory
    environment = {"SPARKRING_FEATURES": "qwen-collectives", **overrides}
    bootstrap.install(root, environment, Mock())
    assert all(environment[key] == value for key, value in overrides.items())


@pytest.mark.parametrize("cache_values,expected_roots", [
    ({}, {"VLLM_CACHE_ROOT": "/cache", "TORCHINDUCTOR_CACHE_DIR": "/cache"}),
    ({"VLLM_CACHE_ROOT": "/srv/model-qad/vllm", "TORCHINDUCTOR_CACHE_DIR": "/srv/model-qad/inductor"},
     {"VLLM_CACHE_ROOT": "/srv/model-qad", "TORCHINDUCTOR_CACHE_DIR": "/srv/model-qad"}),
    ({"VLLM_CACHE_ROOT": "/cache/site/model/vllm/", "TORCHINDUCTOR_CACHE_DIR": "/fast/site/model/torch/"},
     {"VLLM_CACHE_ROOT": "/cache/site/model", "TORCHINDUCTOR_CACHE_DIR": "/fast/site/model"}),
])
def test_prefill_cache_namespace_is_source_bound_and_idempotent(cache_values, expected_roots):
    digest = hashlib.sha256(b"prefill implementation identity").hexdigest()
    environment = {**cache_values, "HF_HOME": "/models/cache", "MODEL_PATH": "/models/target"}
    bootstrap.prefill_environment(environment, digest)
    for key, leaf in (("VLLM_CACHE_ROOT", "vllm"), ("TORCHINDUCTOR_CACHE_DIR", "inductor")):
        assert environment[key] == f"{expected_roots[key]}/qwen-prefill-{digest[:12]}/{leaf}"
    assert environment["SPARKRING_QWEN_PREFILL_MANIFEST_SHA256"] == digest
    assert environment["HF_HOME"] == "/models/cache"
    assert environment["MODEL_PATH"] == "/models/target"
    first = dict(environment)
    bootstrap.prefill_environment(environment, digest)
    assert environment == first


def test_different_prefill_sources_receive_distinct_compiler_namespaces():
    first = {"VLLM_CACHE_ROOT": "/cache/site/model/vllm"}
    second = dict(first)
    bootstrap.prefill_environment(first, "a" * 64)
    bootstrap.prefill_environment(second, "b" * 64)
    assert first["VLLM_CACHE_ROOT"] == "/cache/site/model/qwen-prefill-aaaaaaaaaaaa/vllm"
    assert second["VLLM_CACHE_ROOT"] == "/cache/site/model/qwen-prefill-bbbbbbbbbbbb/vllm"


@pytest.mark.parametrize("selection,expected_order", [
    ("qwen-collectives,qwen-prefill", ["qwen38_collective_policy", "prefill_bootstrap"]),
    (" qwen-prefill , qwen-collectives ", ["prefill_bootstrap", "qwen38_collective_policy"]),
])
def test_both_features_activate_with_prefill_environment_ready(inventory, selection, expected_order):
    root, record = inventory
    environment = {"SPARKRING_FEATURES": selection, "VLLM_CACHE_ROOT": "/cache/site/model/vllm"}
    digest = record["features"]["qwen-prefill"]["manifest_sha256"]
    events = []

    def activate_prefill():
        assert environment["VLLM_CACHE_ROOT"] == f"/cache/site/model/qwen-prefill-{digest[:12]}/vllm"
        assert environment["SPARKRING_QWEN_PREFILL_MANIFEST_SHA256"] == digest

    prefill_install = Mock(side_effect=activate_prefill)

    def importing(name):
        events.append(name)
        return SimpleNamespace(install=prefill_install)

    bootstrap.install(root, environment, importing)
    assert events == expected_order
    prefill_install.assert_called_once_with()
    assert environment["QWEN_DISPATCH_AR_BYTES"] == "20480"
    for feature in record["features"].values():
        assert bootstrap.sys.path.count(str(root / feature["directory"])) == 1


@pytest.mark.parametrize("relative", ["/outside.py", "../outside.py", "prefill/../outside.py", "prefill\\outside.py"])
def test_asset_paths_cannot_escape_bundle_root(tmp_path, relative):
    with pytest.raises(ValueError, match="Invalid feature asset path"):
        bootstrap.verify_assets(tmp_path, {"files": {relative: "a" * 64}})


def test_asset_symlink_is_rejected_before_reading_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "is_symlink", lambda self: True)
    with pytest.raises(ValueError, match="Feature asset differs"):
        bootstrap.verify_assets(tmp_path, {"files": {"prefill/hook.py": "a" * 64}})


def test_unsupported_inventory_schema_prevents_activation(inventory):
    root, record = inventory
    record["schema"] = "unrecognized"
    (root / "capabilities.json").write_text(json.dumps(record), encoding="utf-8")
    importer = Mock()
    with pytest.raises(ValueError, match="Unsupported SparkRing capability inventory"):
        bootstrap.install(root, {"SPARKRING_FEATURES": "qwen-prefill"}, importer)
    importer.assert_not_called()


def test_install_wrapper_converts_activation_error_into_fatal_exit(monkeypatch):
    failure = ValueError("selected asset changed")
    install = Mock(side_effect=failure)
    monkeypatch.setattr(bootstrap, "install", install)
    with pytest.raises(SystemExit, match="SparkRing feature activation failed: selected asset changed") as caught:
        bootstrap.install_or_exit()
    assert caught.value.__cause__ is failure
    install.assert_called_once_with()


def test_install_wrapper_returns_after_success(monkeypatch):
    install = Mock()
    monkeypatch.setattr(bootstrap, "install", install)
    assert bootstrap.install_or_exit() is None
    install.assert_called_once_with()
