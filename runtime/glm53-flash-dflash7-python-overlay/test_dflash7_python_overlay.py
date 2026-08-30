from __future__ import annotations

import importlib.util
import json
from pathlib import Path


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
        "5d571018de5b63a9a90e5c11e6d6e86bbff4a957"
    )
    workload = pins["workload"]
    assert workload["role"] == "external-dflash7"
    assert workload["draft"]["weights_sha256"] == (
        "b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b"
    )
    assert workload["draft"]["speculative_tokens"] == 7
    assert workload["draft"]["tensor_parallel_size"] == 4
    assert workload["target_loaders"] == {
        "safetensors": "implemented",
        "fastsafetensors": "research-only",
    }


def test_rendered_image_metadata_names_dflash7_not_adaptive_mtp() -> None:
    recipe = prepare._render_containerfile()
    assert "SparkRing GLM-5.3 DFlash7 Python overlay" in recipe
    assert "glm53-flash-dflash7-python-overlay" in recipe
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
    assert labels["org.jovian.vllm.commit"] != labels[
        "org.sparkring.vllm.python.commit"
    ]


def test_builder_uses_the_dflash7_runtime_path_and_receipt() -> None:
    script = (HERE / "build-image.sh").read_text(encoding="utf-8")
    assert "runtime/glm53-flash-dflash7-python-overlay" in script
    assert "dflash7-vllm-python-0b67266-native-da4d7be" in script
    assert "glm53-dflash7-python-overlay-image-receipt.json" in script
    assert "dflash7" in script.lower()
