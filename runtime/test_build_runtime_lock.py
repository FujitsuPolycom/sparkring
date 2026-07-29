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
    shutil.copy2(
        RUNTIME / "prepare-public-requirements.py",
        runtime / "prepare-public-requirements.py",
    )
    shutil.copy2(RUNTIME / "pip-freeze.txt", runtime / "pip-freeze.txt")
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


def test_pending_model_revision_fails_before_build(tmp_path):
    root = _copy_fixture(tmp_path)
    lock_path = root / "runtime/runtime-lock.json"
    lock = json.loads(lock_path.read_text())
    lock["model"]["revision"] = "pending"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    result = _run(root)

    assert result.returncode != 0
    assert "model.revision must be an immutable 40-hex commit" in result.stderr


def test_unknown_lock_schema_fails_before_build(tmp_path):
    root = _copy_fixture(tmp_path)
    lock_path = root / "runtime/runtime-lock.json"
    lock = json.loads(lock_path.read_text())
    lock["schema"] = "sparkring-runtime-lock/v999"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    result = _run(root)

    assert result.returncode != 0
    assert "unsupported runtime lock schema" in result.stderr


def test_pending_base_digest_fails_before_build(tmp_path):
    root = _copy_fixture(tmp_path)
    lock_path = root / "runtime/runtime-lock.json"
    lock = json.loads(lock_path.read_text())
    lock["base_image"]["builder"]["digest"] = "pending-first-build"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    result = _run(root)

    assert result.returncode != 0
    assert "base_image.builder.digest must be sha256:<64 hex>" in result.stderr


def test_unresolved_deepgemm_commit_fails_before_build(tmp_path):
    root = _copy_fixture(tmp_path)
    lock_path = root / "runtime/runtime-lock.json"
    lock = json.loads(lock_path.read_text())
    lock["deep_gemm"]["commit_full"] = "resolve-pending"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    result = _run(root)

    assert result.returncode != 0
    assert "deep_gemm.commit_full must be an immutable 40-hex commit" in result.stderr


def test_unknown_lock_pin_fails_instead_of_becoming_decorative(tmp_path):
    root = _copy_fixture(tmp_path)
    lock_path = root / "runtime/runtime-lock.json"
    lock = json.loads(lock_path.read_text())
    lock["toolchain"]["unused_future_pin"] = "looks-pinned-but-is-not-consumed"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    result = _run(root)

    assert result.returncode != 0
    assert "unconsumed lock field: toolchain.unused_future_pin" in result.stderr


def test_unknown_patch_entry_pin_fails_instead_of_becoming_decorative(tmp_path):
    root = _copy_fixture(tmp_path)
    lock_path = root / "runtime/runtime-lock.json"
    lock = json.loads(lock_path.read_text())
    lock["overlays"][0]["upstream_revision"] = "looks-pinned-but-is-not-consumed"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    result = _run(root)

    assert result.returncode != 0
    assert "unconsumed lock field: overlays[0].upstream_revision" in result.stderr


def test_empty_consumed_pin_fails_lock_verification(tmp_path):
    root = _copy_fixture(tmp_path)
    lock_path = root / "runtime/runtime-lock.json"
    lock = json.loads(lock_path.read_text())
    lock["flashinfer"]["repository"] = ""
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    result = _run(root)

    assert result.returncode != 0
    assert "flashinfer.repository must be a non-empty string" in result.stderr


def test_invalid_flashinfer_wheel_hash_fails_before_build(tmp_path):
    root = _copy_fixture(tmp_path)
    lock_path = root / "runtime/runtime-lock.json"
    lock = json.loads(lock_path.read_text())
    lock["flashinfer"]["wheel_sha256"]["flashinfer_jit_cache"] = "not-a-hash"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    result = _run(root)

    assert result.returncode != 0
    assert "flashinfer JIT wheel sha256 must be 64 lowercase hex" in result.stderr


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
