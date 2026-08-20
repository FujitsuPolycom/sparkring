"""Offline contracts for the final NVFP4/FP8-RoPE image receipt."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "runtime/write-nf3-nvfp4-receipt.py"
SPEC = importlib.util.spec_from_file_location("nf3_nvfp4_receipt", MODULE_PATH)
assert SPEC and SPEC.loader
receipt_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(receipt_module)

INSTALLED_MODULE_PATH = ROOT / "runtime/write-nf3-installed-receipt.py"


def _verifier(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "sparkring-nf3-nvfp4-rope8-verification/v1",
                "passed": True,
                "hybrid_method": "hybrid_loader.HybridNvFp4MoE",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _installed_receipt(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "sparkring-nf3-bootstrap-input/v1",
                "profile": "nvfp4-rope8",
                "files": {"b12x/attention/mla/api.py": "a" * 64},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_receipt_is_deterministic_and_names_final_and_parent_images(tmp_path):
    verifier = _verifier(tmp_path / "verification.json")
    installed_receipt = _installed_receipt(tmp_path / "installed.json")
    arguments = {
        "image": "sparkring/gb10-vllm-base:test",
        "image_id": "sha256:" + "1" * 64,
        "nf3_image_id": "sha256:" + "2" * 64,
        "mla_image_id": "sha256:" + "3" * 64,
        "source_commit": "4" * 40,
        "verifier_report": verifier,
        "installed_receipt": installed_receipt,
    }
    first = receipt_module.build_receipt(**arguments)
    second = receipt_module.build_receipt(**arguments)
    assert first == second
    assert first["profile"] == "nvfp4-rope8"
    assert first["image_id"] == arguments["image_id"]
    assert first["parents"] == {
        "nf3_image_id": arguments["nf3_image_id"],
        "mla_image_id": arguments["mla_image_id"],
    }
    assert first["sparkring_source_commit"] == arguments["source_commit"]
    assert len(first["verifier_report"]["sha256"]) == 64
    assert first["installed_file_receipt"] == {
        "schema": "sparkring-nf3-bootstrap-input/v1",
        "profile": "nvfp4-rope8",
        "sha256": receipt_module._sha256(installed_receipt),
    }


def test_receipt_rejects_failed_or_wrong_schema_verifier(tmp_path):
    verifier = tmp_path / "verification.json"
    base_arguments = {
        "image": "sparkring/test:one",
        "image_id": "sha256:" + "1" * 64,
        "nf3_image_id": "sha256:" + "2" * 64,
        "mla_image_id": "sha256:" + "3" * 64,
        "source_commit": "4" * 40,
        "verifier_report": verifier,
        "installed_receipt": _installed_receipt(tmp_path / "installed.json"),
    }
    for document in (
        {"schema": "wrong", "passed": True},
        {
            "schema": "sparkring-nf3-nvfp4-rope8-verification/v1",
            "passed": False,
        },
    ):
        verifier.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError):
            receipt_module.build_receipt(**base_arguments)


def test_receipt_rejects_untrusted_installed_file_receipt(tmp_path):
    verifier = _verifier(tmp_path / "verification.json")
    installed = tmp_path / "installed.json"
    arguments = {
        "image": "sparkring/test:one",
        "image_id": "sha256:" + "1" * 64,
        "nf3_image_id": "sha256:" + "2" * 64,
        "mla_image_id": "sha256:" + "3" * 64,
        "source_commit": "4" * 40,
        "verifier_report": verifier,
        "installed_receipt": installed,
    }
    for document in (
        {"schema": "wrong", "profile": "nvfp4-rope8", "files": {"x": "y"}},
        {
            "schema": "sparkring-nf3-bootstrap-input/v1",
            "profile": "fp8",
            "files": {"x": "y"},
        },
        {
            "schema": "sparkring-nf3-bootstrap-input/v1",
            "profile": "nvfp4-rope8",
            "files": {},
        },
    ):
        installed.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError):
            receipt_module.build_receipt(**arguments)


def test_builder_writes_receipt_for_reused_and_new_images():
    text = (
        ROOT / "scripts/build-gb10-vllm-base-image.sh"
    ).read_text(encoding="utf-8")
    assert "write_final_receipt()" in text
    assert text.count('write_final_receipt "') == 2
    assert "nf3-nvfp4-rope8-runtime.json" in text
    assert "write-nf3-nvfp4-receipt.py" in text
    assert "--installed-receipt" in text


def test_installed_receipt_refreshes_changed_b12x_and_includes_new_files(
    tmp_path,
):
    specification = importlib.util.spec_from_file_location(
        "nf3_installed_receipt",
        INSTALLED_MODULE_PATH,
    )
    assert specification and specification.loader
    installed_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(installed_module)

    site_packages = tmp_path / "site-packages"
    spark_root = tmp_path / "spark-vllm"
    (site_packages / "b12x/attention/mla").mkdir(parents=True)
    (site_packages / "b12x/attention/mla/api.py").write_text(
        "final api\n",
        encoding="utf-8",
    )
    (site_packages / "b12x/attention/mla/new_reader.py").write_text(
        "new reader\n",
        encoding="utf-8",
    )
    (site_packages / "hybrid_loader.py").write_text(
        "hybrid\n",
        encoding="utf-8",
    )
    spark_root.mkdir()
    (spark_root / "sitecustomize.py").write_text(
        "startup\n",
        encoding="utf-8",
    )
    parent = {
        "schema": "sparkring-nf3-bootstrap-input/v1",
        "base_image": "base",
        "faststart_image_id": "sha256:" + "1" * 64,
        "b12x": {"repository": "example/b12x", "commit": "2" * 40},
        "spark_port": {"repository": "example/port", "commit": "3" * 40},
        "sparkring_source_commit": "4" * 40,
        "files": {
            "b12x/attention/mla/api.py": "0" * 64,
            "overlay/hybrid_loader.py": "0" * 64,
            "sparkring/sitecustomize.py": "0" * 64,
        },
    }
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent), encoding="utf-8")

    receipt = installed_module.build_installed_receipt(
        parent_receipt=parent_path,
        site_packages=site_packages,
        sparkring_root=spark_root,
        profile="nvfp4-rope8",
    )

    assert receipt["schema"] == "sparkring-nf3-bootstrap-input/v1"
    assert receipt["profile"] == "nvfp4-rope8"
    assert receipt["parent_input_receipt_sha256"] != "0" * 64
    assert receipt["files"]["b12x/attention/mla/api.py"] != "0" * 64
    assert "b12x/attention/mla/new_reader.py" in receipt["files"]
    assert receipt["files"]["overlay/hybrid_loader.py"] != "0" * 64
    assert receipt["files"]["sparkring/sitecustomize.py"] != "0" * 64


def test_final_layer_binds_installed_receipt_digest_and_reverifies():
    text = (
        ROOT / "runtime/Containerfile.nf3-nvfp4-final"
    ).read_text(encoding="utf-8")
    assert "ARG FINAL_RECEIPT_SHA256" in text
    assert (
        "COPY final-installed-receipt.json "
        "/opt/sparkring/nf3-bootstrap-input-receipt.json"
    ) in text
    assert (
        "SPARKRING_NF3_INPUT_RECEIPT_SHA256=${FINAL_RECEIPT_SHA256}"
        in text
    )
    assert "verify-nf3-bootstrap.py" in text
    assert 'org.sparkring.final_installed_receipt_sha256=' in text
    assert 'org.sparkring.input_receipt_sha256=' in text


def test_builder_generates_receipt_from_candidate_then_builds_final_layer():
    text = (
        ROOT / "scripts/build-gb10-vllm-base-image.sh"
    ).read_text(encoding="utf-8")
    assert 'CANDIDATE_IMAGE="${OUTPUT_IMAGE}-candidate"' in text
    assert "write-nf3-installed-receipt.py" in text
    assert "--parent-receipt" in text
    assert "--output /receipt-output/final-installed-receipt.json.tmp" in text
    assert '> "${FINAL_RECEIPT_TMP}"' not in text
    assert "FINAL_RECEIPT_SHA256=" in text
    assert "Containerfile.nf3-nvfp4-final" in text
    assert '--build-arg "FINAL_RECEIPT_SHA256=' in text
    normalized = " ".join(text.replace("\\\n", "").split())
    assert (
        "/opt/sparkring/verify-nf3-bootstrap.py "
        "--receipt /opt/sparkring/nf3-bootstrap-input-receipt.json"
        in normalized
    )


def test_entrypoint_hashes_composed_runtime_before_nvfp4_abi_import():
    text = (ROOT / "runtime/public-entrypoint.sh").read_text(encoding="utf-8")
    runtime_index = text.index("verify-runtime.py")
    installed_index = text.index("verify-nf3-bootstrap.py")
    nvfp4_index = text.index(
        "/opt/venv/bin/python /opt/sparkring/verify-nf3-nvfp4-rope8.py"
    )
    assert runtime_index < installed_index < nvfp4_index
