from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
PINS = HERE / "pins.json"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = _module("glm53_dflash7_overlay_prepare", HERE / "prepare_context.py")
verify = _module("glm53_dflash7_overlay_verify", HERE / "verify_image.py")
removal = _module("glm53_dflash7_remove_distribution", HERE / "remove_distribution.py")


def test_pins_bind_the_exact_dflash7_composition() -> None:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    assert pins["vllm"]["native_commit"] == (
        "da4d7be6c97434f6942292ed8abbf4b32dc44355"
    )
    assert pins["vllm"]["python_commit"] == (
        "0b67266a0f37d6146a8403fb8482403c62f412d5"
    )
    assert pins["b12x"]["commit"] == (
        "b1d541f9e71a35f030d45fae437630fff7507c2a"
    )
    assert pins["sparkcache"]["commit"] == (
        "c56f77f97b3da907d32e888d82046359a62f0f88"
    )
    assert pins["sparkcache"]["tree"] == (
        "deac36758f86695cd13f07b2870c2e49842aed9c"
    )
    assert pins["sparkcache"]["source_tree_sha256"] == (
        "788686e858ba4af01f535e95122c7650f412fddc40cd221a0924f4ce2b32ff98"
    )
    assert pins["sparkcache"]["cuda_config_schema"] == "canonical-v1"
    assert pins["sparkcache"]["canonical_cuda_config_keys"] == [
        "spark_cache_cuda_restore",
        "spark_cache_cuda_placement_library",
        "spark_cache_cuda_placement_library_sha256",
        "spark_cache_cuda_placement_arena_bytes",
        "spark_cache_cuda_restore_io_workers",
    ]
    workload = pins["workload"]
    assert workload["role"] == "external-dflash7"
    assert workload["draft"]["weights_sha256"] == (
        "b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b"
    )
    assert workload["draft"]["speculative_tokens"] == 7
    assert workload["draft"]["tensor_parallel_size"] == 4
    assert workload["target_loaders"] == {
        "safetensors": "implemented",
        "fastsafetensors": "implemented",
    }
    cleanup = pins["runtime_cleanup"]["deep_ep"]
    assert cleanup["distribution"] == "deep_ep"
    assert cleanup["version"] == "2.0.0+local"
    assert cleanup["receipt_sha256"] == (
        "65514f44829e7d176b0b2cacc9559ed22724e525b7041a8bcd4d2e02d1f372e3"
    )
    patch = pins["vllm"]["runtime_patches"][0]
    assert patch["sha256"] == (
        "39b567013ee7aed79f63200ed460129587933dc77fb430decdf19f78178de279"
    )
    assert patch["postimage_sha256"] == (
        "98acbae2b3bb4482d83f9637c163ce7c92707ccdf6561b7e431f23337f151cf4"
    )


def test_rendered_image_metadata_names_dflash7_not_adaptive_mtp() -> None:
    recipe = prepare._render_containerfile()
    assert "SparkRing GLM-5.3 DFlash7 Python overlay" in recipe
    assert "glm53-flash-dflash7-python-overlay" in recipe
    assert 'org.sparkcache.cuda-config-schema="canonical-v1"' in recipe
    assert "010-dflash-draft-load-config.patch" in recipe
    assert "98acbae2b3bb4482d83f9637c163ce7c92707ccdf6561b7e431f23337f151cf4" in recipe
    assert "remove_distribution.py" in recipe
    assert "deep-ep-removal-receipt.json" in recipe
    assert "DEEP_EP_REMOVAL_RECEIPT_SHA256" in recipe
    assert 'find_spec("deep_ep") is None' in recipe
    label_section = recipe[recipe.index("LABEL org.opencontainers.image.title=") :]
    assert "adaptive-MTP" not in label_section
    verifier = prepare._render_verify_image()
    assert "sparkring-glm53-dflash7-python-overlay-image/v1" in verifier
    assert '"glm53-flash-dflash7-python-overlay"' in verifier


def test_verifier_requires_the_dflash7_deployment_label() -> None:
    pins = verify.shared.load_pins(PINS)
    labels = verify.expected_output_labels(pins)
    assert labels["org.sparkcache.deployment-profile"] == (
        "glm53-flash-dflash7-python-overlay"
    )
    assert labels["org.sparkcache.cuda-config-schema"] == "canonical-v1"
    assert labels["org.sparkcache.source-revision"] == (
        "c56f77f97b3da907d32e888d82046359a62f0f88"
    )
    assert labels["org.sparkcache.source-tree"] == (
        "deac36758f86695cd13f07b2870c2e49842aed9c"
    )
    assert labels["org.sparkcache.source-sha256"] == (
        "788686e858ba4af01f535e95122c7650f412fddc40cd221a0924f4ce2b32ff98"
    )
    assert labels["org.jovian.vllm.commit"] != labels[
        "org.sparkring.vllm.python.commit"
    ]
    assert labels["org.sparkring.vllm.dflash-draft-loader-patch-sha256"] == (
        "39b567013ee7aed79f63200ed460129587933dc77fb430decdf19f78178de279"
    )
    assert labels["org.sparkring.vllm.recurrent-boundary-patch-sha256"] == (
        "5a6561a5bbab990dcd03bfd6a485ea26c3b5a578c2fd61b76305767b16dbfba0"
    )
    assert labels["org.sparkring.runtime.removed-deep-ep-distribution"] == (
        "deep_ep==2.0.0+local"
    )
    assert labels["org.sparkring.runtime.deep-ep-removal-receipt-sha256"] == (
        "65514f44829e7d176b0b2cacc9559ed22724e525b7041a8bcd4d2e02d1f372e3"
    )


def test_builder_uses_the_dflash7_runtime_path_and_receipt() -> None:
    script = (HERE / "build-image.sh").read_text(encoding="utf-8")
    assert "runtime/glm53-flash-dflash7-python-overlay" in script
    assert "dflash7-vllm-python-0b67266-native-da4d7be" in script
    assert "glm53-dflash7-python-overlay-image-receipt.json" in script
    assert "dflash7" in script.lower()
    assert "runtime_cleanup.deep_ep.receipt_sha256" in script
    assert "DEEP_EP_REMOVAL_RECEIPT_SHA256" in script


def test_removal_receipt_is_content_addressed() -> None:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    receipt = prepare._deep_ep_removal_receipt(pins)
    encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()

    assert hashlib.sha256(encoded).hexdigest() == (
        pins["runtime_cleanup"]["deep_ep"]["receipt_sha256"]
    )
    assert receipt["postcondition"] == "module-absent"


def test_prepared_context_records_removal_inputs(tmp_path: Path) -> None:
    runtime = tmp_path / "bundle" / "runtime"
    runtime.mkdir(parents=True)
    (tmp_path / "receipt.json").write_text(
        json.dumps({"files": {}, "workload": {}}), encoding="utf-8"
    )

    prepare._replace_runtime_files(tmp_path)

    receipt_path = runtime / prepare.DEEP_EP_RECEIPT
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == (
        pins["runtime_cleanup"]["deep_ep"]["receipt_sha256"]
    )
    assert removal.load_receipt(receipt_path) == prepare._deep_ep_removal_receipt(pins)
    context_receipt = json.loads(
        (tmp_path / "receipt.json").read_text(encoding="utf-8")
    )
    assert (
        context_receipt["files"][
            f"bundle/runtime/{prepare.DEEP_EP_RECEIPT}"
        ]
        == pins["runtime_cleanup"]["deep_ep"]["receipt_sha256"]
    )
    assert "bundle/runtime/remove_distribution.py" in context_receipt["files"]


def test_image_verifier_requires_distribution_and_module_absence() -> None:
    pins = verify.shared.load_pins(PINS)
    artifacts = {
        "deep_ep_receipt": prepare._deep_ep_removal_receipt(pins),
        "deep_ep_receipt_sha256": pins["runtime_cleanup"]["deep_ep"][
            "receipt_sha256"
        ],
        "deep_ep_module_present": False,
        "deep_ep_owners": [],
        "deep_ep_distribution_present": False,
    }
    verify.shared.verify_runtime_cleanup(artifacts, pins)

    artifacts["deep_ep_module_present"] = True
    with pytest.raises(verify.shared.VerifyError, match="remains importable"):
        verify.shared.verify_runtime_cleanup(artifacts, pins)


def test_distribution_removal_proves_owner_and_uninstalls_exact_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = {
        "schema": removal.SCHEMA,
        "status": "implemented",
        "module": "deep_ep",
        "distribution": "deep_ep",
        "version": "2.0.0+local",
        "postcondition": "module-absent",
        "reason": "Unused by the selected collective backends.",
    }
    state = {"installed": True}
    commands: list[tuple[list[str], bool]] = []

    class Distribution:
        version = "2.0.0+local"
        metadata = {"Name": "deep_ep"}

    def packages_distributions():
        return {"deep_ep": ["deep_ep"]} if state["installed"] else {}

    def distribution(name: str):
        assert name == "deep_ep"
        if not state["installed"]:
            raise removal.importlib.metadata.PackageNotFoundError(name)
        return Distribution()

    def find_spec(name: str):
        assert name == "deep_ep"
        return object() if state["installed"] else None

    def run(command: list[str], *, check: bool):
        commands.append((command, check))
        state["installed"] = False
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        removal.importlib.metadata, "packages_distributions", packages_distributions
    )
    monkeypatch.setattr(removal.importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(removal.importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(removal.subprocess, "run", run)

    result = removal.remove_distribution(receipt)

    assert commands == [
        (
            [
                removal.sys.executable,
                "-m",
                "pip",
                "uninstall",
                "--yes",
                "deep_ep",
            ],
            True,
        )
    ]
    assert result["postcondition"] == "module-absent"


def test_distribution_removal_rejects_ambiguous_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = {
        "module": "deep_ep",
        "distribution": "deep_ep",
        "version": "2.0.0+local",
    }
    monkeypatch.setattr(
        removal.importlib.metadata,
        "packages_distributions",
        lambda: {"deep_ep": ["deep_ep", "another-owner"]},
    )

    with pytest.raises(removal.RemovalError, match="exactly one owner"):
        removal.verify_unique_owner(receipt)
