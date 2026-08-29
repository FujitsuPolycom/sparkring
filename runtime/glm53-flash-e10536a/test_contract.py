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


verify = _module("glm53_e105_verify", HERE / "verify_image.py")


def test_e105_source_build_is_exact_and_unqualified() -> None:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    assert pins["status"] == "implemented"
    build = pins["public_image_build"]
    assert build["status"] == "implemented"
    assert build["sources"]["vllm"] == {
        "repository": "https://github.com/local-inference-lab/vllm.git",
        "commit": "e10536aadf02a18fccddda7ec939c33147e8b0b3",
        "tree": "f7864d18865573dd162d3b850b4aa26acf320ab7",
        "license": "Apache-2.0",
    }
    assert build["outputs"]["runtime_image"] is None
    assert build["outputs"]["sparkcache_image"] is None


def test_e105_image_labels_bind_the_vllm_revision() -> None:
    pins = verify.load_pins(PINS)
    labels = verify.expected_labels(pins)
    assert labels["org.jovian.vllm.commit"] == (
        "e10536aadf02a18fccddda7ec939c33147e8b0b3"
    )


def test_e105_builder_uses_its_own_pins_and_image_name() -> None:
    script = (HERE / "build-image.sh").read_text(encoding="utf-8")
    assert "runtime/glm53-flash-e10536a" in script
    assert "sparkring-glm53-runtime:e10536a-source-arm64" in script
    assert "prepare_context.py" in script
