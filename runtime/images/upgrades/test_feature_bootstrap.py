"""Feature migration must reject legacy selection before importing any hooks."""

import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from . import feature_bootstrap as module


def fixture(tmp_path):
    directory = tmp_path / "qwen4-prefill"
    directory.mkdir()
    payload = b"fixture source"
    (directory / "hook.py").write_bytes(payload)
    record = {
        "schema": "sparkring-image-capabilities/v1",
        "features": {
            "qwen4-prefill": {
                "directory": "qwen4-prefill",
                "manifest_sha256": "a" * 64,
                "files": {"qwen4-prefill/hook.py": hashlib.sha256(payload).hexdigest()},
            }
        },
        "unsupported_features": {"qwen-prefill": {"replacement": "qwen4-prefill"}},
    }
    (tmp_path / "capabilities.json").write_text(json.dumps(record))
    return record


def test_new_bundle_selects_its_own_bootstrap_and_idempotent_namespace(tmp_path):
    fixture(tmp_path)
    calls = []
    env = {"SPARKRING_FEATURES": "qwen4-prefill"}

    def load(name):
        calls.append(name)
        return NS(install=lambda: calls.append("installed"))

    module.install(tmp_path, env, load)
    cache = env["VLLM_CACHE_ROOT"]
    module.install(tmp_path, env, load)
    assert calls == ["qwen4_prefill_bootstrap", "installed"] * 2
    assert env["VLLM_CACHE_ROOT"] == cache
    assert env["SPARKRING_QWEN4_PREFILL_MANIFEST_SHA256"] == "a" * 64
    assert "SPARKRING_QWEN_PREFILL_MANIFEST_SHA256" not in env


@pytest.mark.parametrize(
    "selection",
    ["qwen-prefill", "qwen4-prefill,qwen-prefill", "qwen4-prefill,qwen4-prefill"],
)
def test_bad_selection_cannot_partially_activate(tmp_path, selection):
    fixture(tmp_path)
    calls = []
    env = {"SPARKRING_FEATURES": selection}
    with pytest.raises(ValueError):
        module.install(tmp_path, env, calls.append)
    assert calls == [] and set(env) == {"SPARKRING_FEATURES"}


def test_asset_drift_and_directory_escape_refuse(tmp_path):
    record = fixture(tmp_path)
    (tmp_path / "qwen4-prefill/hook.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        module.install(tmp_path, {"SPARKRING_FEATURES": "qwen4-prefill"})
    record["features"]["qwen4-prefill"]["directory"] = "../escape"
    (tmp_path / "capabilities.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="directory"):
        module.install(tmp_path, {"SPARKRING_FEATURES": "qwen4-prefill"})
