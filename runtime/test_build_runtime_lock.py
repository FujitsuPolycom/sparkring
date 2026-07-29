"""Fail-closed tests for build-runtime.sh's patch-input lock."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

RUNTIME = Path(__file__).resolve().parent
REPO = RUNTIME.parent


def _bash() -> str:
    candidates = (
        [r"C:\Program Files\Git\bin\bash.exe", shutil.which("bash")]
        if os.name == "nt"
        else [shutil.which("bash")]
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    pytest.skip("bash is unavailable")


def _copy_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    runtime = root / "runtime"
    shutil.copytree(RUNTIME / "patches", runtime / "patches")
    shutil.copy2(RUNTIME / "build-runtime.sh", runtime / "build-runtime.sh")
    shutil.copy2(RUNTIME / "runtime-lock.json", runtime / "runtime-lock.json")
    for entry in json.loads((runtime / "runtime-lock.json").read_text())["nccl"][
        "patches"
    ]:
        source = REPO / entry["path"]
        target = root / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return root


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    script = root / "runtime" / "build-runtime.sh"
    # Git Bash needs a slash-form path; POSIX runners already have one.
    script_arg = str(script).replace("\\", "/")
    env = dict(os.environ)
    if os.name == "nt":
        shim_dir = root / ".test-bin"
        shim_dir.mkdir()
        shutil.copy2(sys.executable, shim_dir / "python3.exe")
        env["PATH"] = f"{shim_dir}{os.pathsep}{env['PATH']}"
    return subprocess.run(
        [_bash(), script_arg, "--verify-lock-only"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def test_exact_locked_patch_inputs_pass(tmp_path):
    result = _run(_copy_fixture(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "runtime lock and patch inputs verified" in result.stdout


def test_modified_patch_fails_before_build(tmp_path):
    root = _copy_fixture(tmp_path)
    patch = root / "runtime/patches/vllm/010-sparkcache-async-rollback.patch"
    patch.write_bytes(patch.read_bytes() + b"\n# tampered\n")

    result = _run(root)

    assert result.returncode != 0
    assert "overlays sha256 mismatch" in result.stderr


def test_modified_preimages_manifest_fails_before_build(tmp_path):
    root = _copy_fixture(tmp_path)
    preimages = root / "runtime/patches/vllm/preimages.json"
    data = json.loads(preimages.read_text())
    data["010-sparkcache-async-rollback.patch"]["preimage_sha256"] = "0" * 64
    preimages.write_text(json.dumps(data), encoding="utf-8")

    result = _run(root)

    assert result.returncode != 0
    assert "overlays sha256 mismatch" in result.stderr


def test_unlocked_extra_patch_fails_before_build(tmp_path):
    root = _copy_fixture(tmp_path)
    extra = root / "runtime/patches/vllm/999-unlocked.patch"
    extra.write_text("unlocked", encoding="utf-8")

    result = _run(root)

    assert result.returncode != 0
    assert "unlocked patch input(s)" in result.stderr


def test_locked_but_unused_overlay_fails_before_build(tmp_path):
    root = _copy_fixture(tmp_path)
    lock_path = root / "runtime/runtime-lock.json"
    lock = json.loads(lock_path.read_text())
    unused = root / "runtime/patches/vllm/unused-input.txt"
    unused.write_text("unused", encoding="utf-8")
    lock["overlays"].append(
        {
            "path": "runtime/patches/vllm/unused-input.txt",
            "sha256": hashlib.sha256(unused.read_bytes()).hexdigest(),
        }
    )
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    result = _run(root)

    assert result.returncode != 0
    assert "locked overlay(s) not consumed" in result.stderr


def test_retargeted_nccl_pin_fails_before_build(tmp_path):
    root = _copy_fixture(tmp_path)
    lock_path = root / "runtime/runtime-lock.json"
    lock = json.loads(lock_path.read_text())
    substitute = root / "runtime/substitute-nccl.patch"
    substitute.write_text("different but correctly hashed input", encoding="utf-8")
    lock["nccl"]["patches"][0] = {
        "path": "runtime/substitute-nccl.patch",
        "sha256": hashlib.sha256(substitute.read_bytes()).hexdigest(),
    }
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    result = _run(root)

    assert result.returncode != 0
    assert "nccl.patches paths do not match Containerfile COPY inputs" in result.stderr
