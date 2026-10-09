"""Registration in fresh interpreters: flag combinations and refusal paths.

``register`` reads the flags, verifies the pins of every enabled item and
installs the import hook; nothing imports the image's modules here, so a
probe also runs without them and the pin replay happens at registration
(``prepare`` reads and compiles the pinned files).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PROBE = HERE / "register_probe.py"


def run(env: dict[str, str]) -> dict:
    environment = {**os.environ, "SPARKRING_GLM53_IMAGE_SOURCES": os.environ["SPARKRING_GLM53_IMAGE_SOURCES"]}
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


def test_disabled_by_default_registers_without_patching():
    state = run({})
    assert state["registered_flags"] == []
    assert state["pending_modules"] == []


@pytest.mark.parametrize("extra", [{"GLM53FULL_LATENT_SHARD": "1"},
                                   {"GLM53FULL_EH_PROJ_TP": "1"},
                                   {"GLM53FULL_LATENT_SHARD": "1", "GLM53FULL_EH_PROJ_TP": "1"}],
                         ids=["latent", "eh-proj", "both"])
def test_enabled_flags_prepare_their_patches(extra):
    state = run(extra)
    assert sorted(state["registered_flags"]) == sorted(k for k, v in extra.items() if v == "1")
    assert state["pending_modules"]


def test_unknown_flag_value_refuses():
    with pytest.raises(AssertionError, match="must be 0 or 1"):
        run({"GLM53FULL_LATENT_SHARD": "on"})


def test_missing_sources_refuse():
    environment = {**os.environ}
    environment.pop("SPARKRING_GLM53_IMAGE_SOURCES", None)
    environment["GLM53FULL_LATENT_SHARD"] = "1"
    result = subprocess.run([sys.executable, str(PROBE)], env=environment,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "SPARKRING_GLM53_IMAGE_SOURCES" in (result.stdout + result.stderr)


def test_tampered_sources_refuse(tmp_path, monkeypatch):
    copy = tmp_path / "image"
    shutil.copytree(Path(os.environ["SPARKRING_GLM53_IMAGE_SOURCES"]), copy,
                    ignore=shutil.ignore_patterns("__pycache__"))
    path = copy / "vllm/models/deepseek_v32/attention.py"
    data = bytearray(path.read_bytes())
    data[0] = data[0] ^ 0x20
    path.write_bytes(bytes(data))
    monkeypatch.setenv("SPARKRING_GLM53_IMAGE_SOURCES", str(copy))
    with pytest.raises(AssertionError, match="SHA-256"):
        run({"GLM53FULL_LATENT_SHARD": "1"})


def test_changed_dependency_file_refuses(tmp_path, monkeypatch):
    copy = tmp_path / "image"
    shutil.copytree(Path(os.environ["SPARKRING_GLM53_IMAGE_SOURCES"]), copy,
                    ignore=shutil.ignore_patterns("__pycache__"))
    path = copy / "vllm/model_executor/parameter.py"
    data = bytearray(path.read_bytes())
    data[0] = data[0] ^ 0x20
    path.write_bytes(bytes(data))
    monkeypatch.setenv("SPARKRING_GLM53_IMAGE_SOURCES", str(copy))
    with pytest.raises(AssertionError, match="relies on"):
        run({"GLM53FULL_LATENT_SHARD": "1"})
