"""Offline integrity tests for the public EXL3 derived-image lane."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PINS = json.loads((HERE / "pins.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pins_cover_every_published_overlay_byte():
    assert PINS["schema"] == "sparkring-public-exl3-pins/v1"
    expected = set(PINS["overlay_files"])
    observed = {
        path.relative_to(HERE / "overlay").as_posix()
        for path in (HERE / "overlay").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    assert observed == expected
    for relative, digest in PINS["overlay_files"].items():
        assert _sha256(HERE / "overlay" / relative) == digest


def test_source_patches_and_runtime_overlays_are_receipt_pinned():
    for source in ("sparkinfer", "exllamav3"):
        record = PINS["sources"][source]
        assert _sha256(HERE / record["patch"]) == record["patch_sha256"]
        assert len(record["base_commit"]) == 40
        assert len(record["tree"]) == 40
    for relative, record in PINS["spark_runtime_overlay_files"].items():
        assert _sha256(ROOT / record["source"]) == record["preimage"]
        assert _sha256(HERE / "runtime-overlay" / relative) == record["postimage"]


def test_builder_is_arm64_receipt_gated_and_reuses_exact_images():
    builder = (HERE / "build-image.sh").read_text(encoding="utf-8")
    containerfile = (HERE / "Containerfile").read_text(encoding="utf-8")
    for required in (
        '[[ "$(uname -m)" == "aarch64" ]]',
        "verify_build_context.py",
        "SPARKRING_EXL3_BASE_IMAGE_ID",
        "org.sparkring.parent.image-id",
        "org.sparkring.exl3.context-manifest-sha256",
        "PASS: exact EXL3 image exists; build skipped",
        "--platform linux/arm64",
        "verify_exl3_runtime.py --phase runtime",
    ):
        assert required in builder
    for required in (
        "ARG BASE_IMAGE",
        "FROM ${BASE_IMAGE}",
        "--require-hashes",
        "cutlass-requirements.txt",
        "COPY overlay/vllm/",
        "COPY runtime-overlay/",
        "verify_exl3_runtime.py",
        'ENTRYPOINT ["/opt/sparkring-exl3/entrypoint.sh"]',
    ):
        assert required in containerfile


def test_cutlass_wheels_and_installed_distribution_files_are_receipt_pinned():
    requirements_path = HERE / "cutlass-requirements.txt"
    requirements = requirements_path.read_text(encoding="utf-8")
    lock = PINS["cutlass_python_lock"]
    assert _sha256(requirements_path) == lock["requirements_sha256"]
    for name, record in lock["distributions"].items():
        assert f"{name}=={record['version']}" in requirements
        assert f"--hash=sha256:{record['wheel_sha256']}" in requirements
    composer = (HERE / "compose_runtime_manifest.py").read_text(encoding="utf-8")
    assert "distribution_files(tuple(cutlass_distributions))" in composer


def test_entrypoint_clears_explicit_unsets_and_runs_full_model_verifier():
    entrypoint = (HERE / "entrypoint.sh").read_text(encoding="utf-8")
    assert 'unset "${name}"' in entrypoint
    assert "verify_exl3_model.py" in entrypoint


def test_runtime_manifest_receipts_distribution_tools_outside_site_packages():
    composer = (HERE / "compose_runtime_manifest.py").read_text(encoding="utf-8")
    assert 'VENV_ROOT = Path("/opt/venv")' in composer
    assert '"public-exl3-venv-tools"' in composer
    assert "installed distribution file escapes venv" in composer


def test_runtime_verifier_uses_composed_lmcache_writer_contract():
    verifier = (HERE / "verify_exl3_runtime.py").read_text(encoding="utf-8")
    assert "strategy.is_kv_writer" in verifier
    assert "strategy.is_writer" not in verifier


def test_generated_context_verifier_rejects_byte_drift(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"exact")
    manifest = {
        "schema": "sparkring-public-exl3-build-context/v1",
        "profile_id": "glm52-exl3-tr3-3.25bpw",
        "files": {"payload.bin": _sha256(payload)},
    }
    (tmp_path / "context-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    command = [
        sys.executable,
        str(HERE / "verify_build_context.py"),
        "--context",
        str(tmp_path),
    ]
    passed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert passed.returncode == 0, passed.stderr
    payload.write_bytes(b"drift")
    failed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert failed.returncode != 0
    assert "changed=['payload.bin']" in failed.stderr


def test_recipe_and_runtime_pin_same_model_contract():
    recipe = json.loads(
        (ROOT / "recipes/glm52-exl3-tr3-3.25bpw.json").read_text(encoding="utf-8")
    )
    for key in (
        "repository",
        "revision",
        "config_sha256",
        "index_sha256",
        "tier_bitmap_sha256",
        "manifest_sha256",
        "weight_bytes",
        "repository_bytes",
        "shard_count",
        "expected_tiers",
    ):
        assert PINS["model"][key] == recipe["model"][key]
