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


def test_receipt_is_deterministic_and_names_final_and_parent_images(tmp_path):
    verifier = _verifier(tmp_path / "verification.json")
    arguments = {
        "image": "sparkring/glm52-nf3-nvfp4-rope8:test",
        "image_id": "sha256:" + "1" * 64,
        "nf3_image_id": "sha256:" + "2" * 64,
        "mla_image_id": "sha256:" + "3" * 64,
        "source_commit": "4" * 40,
        "verifier_report": verifier,
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


def test_receipt_rejects_failed_or_wrong_schema_verifier(tmp_path):
    verifier = tmp_path / "verification.json"
    base_arguments = {
        "image": "sparkring/test:one",
        "image_id": "sha256:" + "1" * 64,
        "nf3_image_id": "sha256:" + "2" * 64,
        "mla_image_id": "sha256:" + "3" * 64,
        "source_commit": "4" * 40,
        "verifier_report": verifier,
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


def test_builder_writes_receipt_for_reused_and_new_images():
    text = (
        ROOT / "scripts/build-nf3-nvfp4-rope8-image.sh"
    ).read_text(encoding="utf-8")
    assert "write_final_receipt()" in text
    assert text.count('write_final_receipt "') == 2
    assert "nf3-nvfp4-rope8-runtime.json" in text
    assert "write-nf3-nvfp4-receipt.py" in text
