from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parent
PINS = HERE / "pins.json"
MANIFEST = HERE / "vllm-python-overlay.json"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = _module("glm53_public_python_overlay_contract", HERE / "overlay_contract.py")
verify = _module("glm53_public_python_overlay_verify", HERE / "verify_image.py")


def test_overlay_pins_public_base_and_mixed_vllm_provenance() -> None:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    assert pins["status"] == "implemented"
    assert pins["public_base"]["reference"] == (
        "ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:"
        "864adfe68f458223e186a19844ac80c7adc7365e5db1f25e109b85fc19850dcd"
    )
    assert pins["vllm"]["native_commit"] == (
        "da4d7be6c97434f6942292ed8abbf4b32dc44355"
    )
    assert pins["vllm"]["python_commit"] == (
        "0b67266a0f37d6146a8403fb8482403c62f412d5"
    )
    assert pins["b12x"]["commit"] == (
        "b1d541f9e71a35f030d45fae437630fff7507c2a"
    )
    assert pins["b12x"]["tree"] == "c69cdec1c59a08e8e0e549f930fa8abcfb5134ae"
    assert pins["b12x"]["package_version"] == "1.3.0"
    assert pins["b12x"]["commit"] != pins["b12x"]["base_commit"]


def test_overlay_manifest_is_the_exact_31_file_python_delta() -> None:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = contract.validate_overlay_manifest(manifest)
    assert len(records) == 31
    assert sum(record["target_bytes"] for record in records) == 1_567_308
    assert [record["operation"] for record in records].count("add") == 1
    assert next(record for record in records if record["operation"] == "add")[
        "path"
    ] == "vllm/v1/spec_decode/dynamic/acceptance_length.py"
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == pins["vllm"][
        "overlay_manifest_sha256"
    ]


def test_native_build_inputs_are_identical_git_objects() -> None:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    assert pins["vllm"]["native_source_objects"] == {
        "csrc": "9ada29088768f1bc08dadd2eed3c9738eb9ac8a1",
        "cmake": "5e5bbdbe1c1b3a479656d8d6a41cc32a1982c43d",
        "rust": "85c3cd52db223217d45377d3f7f884e756641de3",
        "CMakeLists.txt": "bb0f51b43ef4e1c57918b551b8cf213f9059b601",
        "setup.py": "ae64a13daa0f1facc255afcdb4ffcad264776b98",
        "pyproject.toml": "0766645fc7481da0ec439208128b838b3348d94c",
        "requirements": "d6e1c8e13cd4c4358ab422e3ef006d3f9f23e18b",
        "docker/Dockerfile": "6e20f6eab482782ed90a05d91b669b59641eaa46",
    }


def test_stale_b12x_caps_are_rejected_before_a_gpu_launch() -> None:
    @dataclass
    class StaleCaps:
        device: str

    def stale_bind(plan, *, mixed_qkv):
        return plan, mixed_qkv

    with pytest.raises(contract.ContractError, match="omits required field"):
        contract.validate_b12x_surface(
            StaleCaps,
            stale_bind,
            SimpleNamespace(plan=object()),
            required_field="kda_metadata_validation",
        )


def test_live_tensor_b12x_surface_requires_every_binding_input() -> None:
    @dataclass
    class Caps:
        kda_metadata_validation: str = "transactional"

    def incomplete_bind(plan, *, mixed_qkv, output):
        return plan, mixed_qkv, output

    with pytest.raises(contract.ContractError, match="omits live tensor parameters"):
        contract.validate_b12x_surface(
            Caps,
            incomplete_bind,
            SimpleNamespace(plan=object()),
            required_field="kda_metadata_validation",
        )


def test_elf_manifest_detects_retained_native_changes(tmp_path: Path) -> None:
    package = tmp_path / "vllm"
    package.mkdir()
    first = package / "_C_stable_libtorch.abi3.so"
    second = package / "vllm-rs"
    first.write_bytes(b"\x7fELFfirst")
    second.write_bytes(b"\x7fELFsecond")
    original = contract.elf_manifest(package)
    assert [record["path"] for record in original] == [
        "_C_stable_libtorch.abi3.so",
        "vllm-rs",
    ]
    second.write_bytes(b"\x7fELFchanged")
    assert contract.canonical_sha256(contract.elf_manifest(package)) != (
        contract.canonical_sha256(original)
    )


def test_containerfile_reuses_vllm_and_nccl_native_artifacts() -> None:
    recipe = (HERE / "Containerfile").read_text(encoding="utf-8")
    assert "FROM ${PUBLIC_BASE}" in recipe
    assert "COPY bundle/vllm-overlay/ /usr/local/lib/python3.12/dist-packages/" in recipe
    assert "--reinstall --no-deps" in recipe
    assert "record-base" in recipe and "verify-composed" in recipe
    assert "native-elf-manifest-sha256" in recipe
    assert "native-dispatch-manifest-sha256" in recipe
    assert "setup.py bdist_wheel" not in recipe
    assert "make -C /build/nccl" not in recipe


def test_build_prepares_context_below_the_temporary_workspace() -> None:
    script = (HERE / "build-image.sh").read_text(encoding="utf-8")
    assert 'workspace="$(mktemp -d)"' in script
    assert 'context="${workspace}/context"' in script
    assert 'rm -rf -- "${workspace}"' in script


def test_sparkcache_patches_and_contract_run_after_the_python_overlay() -> None:
    recipe = (HERE / "Containerfile").read_text(encoding="utf-8")
    overlay = recipe.index("COPY bundle/vllm-overlay/")
    patch_020 = recipe.index("020-sparkcache-vmm-exemption.patch")
    patch_030 = recipe.index("030-sparkcache-hma-load-failure.patch")
    patch_040 = recipe.index("040-sparkcache-shared-prefix-lease.patch")
    patch_041 = recipe.index("041-sparkcache-shared-prefix-attach.patch")
    lease = recipe.index("verify_lease_contract.py")
    assert overlay < patch_020 < patch_030 < patch_040 < patch_041 < lease


def test_output_labels_do_not_claim_a_source_built_0b_wheel() -> None:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    labels = verify.expected_output_labels(pins)
    assert labels["org.jovian.vllm.commit"] == pins["vllm"]["native_commit"]
    assert labels["org.sparkring.vllm.python.commit"] == pins["vllm"][
        "python_commit"
    ]
    assert labels["org.jovian.vllm.commit"] != labels[
        "org.sparkring.vllm.python.commit"
    ]
