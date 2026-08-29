"""CPU-only contracts for the source-complete GLM-5.3 image builder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PINS = HERE / "pins.json"
PATCH = ROOT / "spark_transport" / "nccl" / "nccl-2.30.7-switchless-cycle.patch"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = _module("glm53_prepare_context", HERE / "prepare_context.py")
verify = _module("glm53_verify_image", HERE / "verify_image.py")


def test_public_builder_pins_complete_source_and_license_boundary() -> None:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    build = pins["public_image_build"]
    assert build["platform"] == "linux/arm64"
    assert build["status"] == "research-only"
    assert build["base_images"]["cuda_runtime"]["license"] == (
        "LicenseRef-NVIDIA-Deep-Learning-Container"
    )
    assert build["sources"]["vllm"]["commit"] == (
        "da4d7be6c97434f6942292ed8abbf4b32dc44355"
    )
    assert build["sources"]["b12x"]["commit"] == (
        "2fcf23a0ce269be27b2e03fece73d46e90e6aeea"
    )
    assert build["sources"]["nccl"]["commit"] == (
        "73cf112295c33aee2b895f329f592f2a9b4b0f97"
    )
    assert build["sources"]["nccl"]["patched_tree"] == (
        "abdeb053b94c3f6d472cd55ae2b79ca821299009"
    )
    assert build["instanttensor"]["license"] == "Apache-2.0"


def test_switchless_patch_is_hash_bound_and_uses_original_interface() -> None:
    pins = prepare.load_pins(PINS)
    patch_pin = pins["public_image_build"]["sources"]["nccl"]["patches"][0]
    assert hashlib.sha256(PATCH.read_bytes()).hexdigest() == patch_pin["sha256"]
    text = PATCH.read_text(encoding="utf-8")
    assert "NCCL_SWITCHLESS_RING_ONLY" in text
    assert "NCCL_SKIP_TREE_CONNECT" not in text
    assert "ncclNMergedIbDevs + devIdx" not in text
    assert "ncclIbMergedDevs + devIdx" in text


def test_image_inspection_requires_exact_platform_and_labels() -> None:
    pins = verify.load_pins(PINS)
    document = {
        "Id": "sha256:" + "a" * 64,
        "Architecture": "arm64",
        "Os": "linux",
        "Config": {"Labels": verify.expected_labels(pins)},
    }
    verify.validate_inspection(document, pins)
    document["Architecture"] = "amd64"
    with pytest.raises(verify.VerifyError, match="linux/arm64"):
        verify.validate_inspection(document, pins)


def test_builder_uses_public_context_and_source_built_nccl() -> None:
    script = (HERE / "build-image.sh").read_text(encoding="utf-8")
    containerfile = (HERE / "Containerfile").read_text(encoding="utf-8")
    assert "prepare_context.py" in script
    assert "sparkring-glm53-official-spark" not in script
    assert "make -C /build/nccl" in containerfile
    assert "org.opencontainers.image.licenses" in containerfile
    assert "COPY bundle/sources/vllm/LICENSE" in containerfile
