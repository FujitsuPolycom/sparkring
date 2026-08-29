from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PINS = HERE / "pins.json"
SPARKCACHE_COMMIT = "eb3690c1aac2b9e86be8d513799dbb64afa53f25"
SPARKCACHE_SOURCE_SHA256 = (
    "34108fb22ba95b457bf4b357407b176dcbf3a6db6227227b21ecee045502a16f"
)
OVERLAY_CONTAINERFILE_SHA256 = (
    "41d0f82a9c648664589a040153b91682720ae133719d86694fd4d028e2800f99"
)


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
    sparkcache = pins["sparkcache"]
    assert sparkcache["commit"] == SPARKCACHE_COMMIT
    assert sparkcache["source_tree_sha256"] == SPARKCACHE_SOURCE_SHA256
    assert (
        sparkcache["overlay_containerfile_sha256"]
        == OVERLAY_CONTAINERFILE_SHA256
    )


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


def test_e105_image_probe_requires_the_parallel_loader_dependency() -> None:
    script = (HERE / "verify_image.py").read_text(encoding="utf-8")
    assert "'fastsafetensors': importlib.metadata.version('fastsafetensors')" in script
