"""Registration in fresh interpreters: flag combinations and refusal paths.

Each case runs ``register`` and then, for the enabled cases, imports the
image's ``b12x_indexer`` module, exactly as a worker does: the wrapper
installs through the plugin's import hook. The helper prints one JSON line.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PROBE = HERE / "register_probe.py"


def run(env: dict[str, str]) -> dict:
    # The flags come from the case alone: the session defaults do not leak.
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("GLM_DSA_INDEXER_SPLIT")}
    environment["SPARKRING_GLM53_IMAGE_SOURCES"] = os.environ["SPARKRING_GLM53_IMAGE_SOURCES"]
    environment.update(env)
    for key, value in list(environment.items()):
        if value == "":
            del environment[key]
    result = subprocess.run([sys.executable, str(PROBE)], env=environment,
                            capture_output=True, text=True, timeout=300)
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    if result.returncode != 0 or not lines:
        raise AssertionError(f"probe failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    return json.loads(lines[-1])


def test_disabled_by_default_leaves_the_module_unpatched():
    state = run({"GLM_DSA_INDEXER_SPLIT": "0"})
    assert state["registered"] and state["installed"] == []
    assert state["forward_is_wrapper"] is False


@pytest.mark.parametrize("extra", [{}, {"GLM_DSA_INDEXER_SPLIT_FULL_LAUNCHES": "1"},
                                   {"GLM_DSA_INDEXER_SPLIT_MIXED": "1"}],
                         ids=["plain", "full-launches", "mixed"])
def test_enabled_installs_through_the_import_hook(extra):
    state = run({"GLM_DSA_INDEXER_SPLIT": "1", **extra})
    assert state["installed"] == ["__init__", "forward"]
    assert state["forward_is_wrapper"] is True
    assert state["pending"] == []  # the import hook finished


def test_imported_module_installs_at_once():
    state = run({"GLM_DSA_INDEXER_SPLIT": "1", "PROBE_IMPORT_FIRST": "1"})
    assert state["installed"] == ["__init__", "forward"]
    assert state["forward_is_wrapper"] is True


def test_unknown_flag_value_refuses():
    with pytest.raises(AssertionError, match="must be 0 or 1"):
        run({"GLM_DSA_INDEXER_SPLIT": "on", "PROBE_NO_DEFAULTS": "1"})


def test_sub_flag_without_enable_refuses():
    with pytest.raises(AssertionError, match="set without GLM_DSA_INDEXER_SPLIT"):
        run({"GLM_DSA_INDEXER_SPLIT_FULL_LAUNCHES": "1", "PROBE_NO_DEFAULTS": "1"})


def test_missing_sources_refuse(monkeypatch):
    environment = {**os.environ}
    environment.pop("SPARKRING_GLM53_IMAGE_SOURCES", None)
    result = subprocess.run([sys.executable, str(PROBE)], env=environment,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "SPARKRING_GLM53_IMAGE_SOURCES" in (result.stdout + result.stderr)


def test_tampered_sources_refuse(tmp_path, monkeypatch):
    import shutil

    copy = tmp_path / "image"
    shutil.copytree(Path(os.environ["SPARKRING_GLM53_IMAGE_SOURCES"]), copy,
                    ignore=shutil.ignore_patterns("__pycache__"))
    path = copy / "vllm/v1/attention/backends/mla/b12x_indexer.py"
    data = bytearray(path.read_bytes())
    data[0] = data[0] ^ 0x20
    path.write_bytes(bytes(data))
    monkeypatch.setenv("SPARKRING_GLM53_IMAGE_SOURCES", str(copy))
    with pytest.raises(AssertionError, match="SHA-256"):
        run({"GLM_DSA_INDEXER_SPLIT": "1"})
