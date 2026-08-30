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


prepare = _module("glm53_pr42_prepare", HERE / "prepare_context.py")
verify = _module("glm53_pr42_verify", HERE / "verify_image.py")


def test_pins_bind_pr42_without_changing_runtime_contracts() -> None:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    assert pins["sparkcache"] == {
        **pins["sparkcache"],
        "commit": "9c2f6c8ac36e0aa5d134fbcd81e819db2ce63970",
        "tree": "e7ac2ef7a3180c5a83771edac44216c3325894e5",
        "source_tree_sha256": (
            "834ff02c235e3f3a3594cec31d0a83d981ac8d410d6482d062725fd9b846a95c"
        ),
        "cuda_placement_library_sha256": (
            "d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c"
        ),
    }
    assert pins["vllm"]["native_commit"] == (
        "da4d7be6c97434f6942292ed8abbf4b32dc44355"
    )
    assert pins["vllm"]["python_commit"] == (
        "0b67266a0f37d6146a8403fb8482403c62f412d5"
    )
    assert pins["b12x"]["commit"] == (
        "b1d541f9e71a35f030d45fae437630fff7507c2a"
    )
    assert pins["sparkcache"]["contract"]["sha256"] == (
        "8adbdfa3fd4b06b213c3aab45255a0b039f1c9940a4b1fad0efd004d263227c9"
    )
    assert pins["page_base_restore_flight"] == {
        "status": "implemented-unqualified",
        "summary_schema": "sparkcache-page-base-restore-flight/v1",
        "storage_mode": "block_pages_v1",
        "base_tokens": 98304,
        "result_tokens": 131072,
        "private_tail_tokens": 32768,
        "participants": 16,
        "physical_base_reads": 1,
        "avoided_base_reads": 15,
        "required_outcome": "verified",
        "maximum_simultaneous_flights": 2,
        "maximum_declared_bytes_per_flight": 1073741824,
        "maximum_peak_reserved_bytes": 2147483648,
        "artifact_receipt": {
            "schema": "sparkcache-diagnostic-image-receipt/v1",
            "sha256": "551072f700db6cf064a85242ed1c8fbd52c8cca870c20534a6d0c1b133ba6bc4",
            "builder_path": "/home/code/image-build-receipts/sparkring-glm53-sparkcache-dflash7-pr42-page-base-flight-arm64.json",
            "image": "sparkring-glm53-sparkcache:dflash7-pr42-page-base-flight-arm64",
            "image_id": "sha256:ba6ca684f5dbbe7fccac93aedce8abe907cf4cabcc4f5d2f481fad2706fdbfde",
            "builder": "spark-aa42",
            "feature_status": "implemented-gpu-free-tested",
            "parent_image_id": "sha256:ed60be066d6d9eadea267bc4597a0687869f3ddb95a3e5c6f86649893a838eb8",
            "cache_namespace_impact": "none",
        },
    }


def test_rendered_context_and_verifier_bind_feature_receipt_fields() -> None:
    containerfile = prepare._render_containerfile()
    assert 'org.sparkcache.feature.page-base-read-flight="implemented-gpu-free-tested"' in containerfile
    assert "org.sparkcache.feature.page-base-read-flight" in containerfile
    assert "implemented-gpu-free-tested" in containerfile
    assert "/cache/jit/vllm/dflash7-pr42-page-base-flight" in containerfile
    verifier = prepare._render_verify_image()
    assert "sparkring-glm53-dflash7-pr42-page-base-flight-image/v1" in verifier
    assert 'pins["page_base_restore_flight"]' in verifier
    labels = verify.expected_output_labels(json.loads(PINS.read_text()))
    assert labels["org.sparkcache.feature.page-base-read-flight"] == (
        "implemented-gpu-free-tested"
    )
    assert labels["org.sparkcache.feature.page-base-read-flight-pr"] == "42"


def test_build_script_uses_isolated_default_outputs() -> None:
    script = (HERE / "build-image.sh").read_text(encoding="utf-8")
    assert "dflash7-pr42-page-base-flight-arm64" in script
    assert "glm53-pr42-page-base-flight-image-receipt.json" in script
    assert "runtime/glm53-flash-dflash7-pr42-page-base-flight" in script
