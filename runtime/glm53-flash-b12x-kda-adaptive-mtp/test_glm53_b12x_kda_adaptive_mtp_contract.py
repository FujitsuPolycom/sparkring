from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
PINS = HERE / "pins.json"
SPARKCACHE_COMMIT = "5ec6a9953ad5d39120298bbfc26e95a6fa4b1dc3"
SPARKCACHE_TREE = "94c236b9dfbf5f70075eb47877fd9caaa5d8c249"
SPARKCACHE_SOURCE_SHA256 = (
    "bc238f96e550c7ec27d4081dd1f2e741d404aaf5c8572d89ccc5e76812be4d63"
)
OVERLAY_CONTAINERFILE_SHA256 = (
    "8e2377d034ba80b059f9a4387a6590a08e205313568ee1382e0e25342f8c5d40"
)


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = _module("glm53_b12x_kda_adaptive_mtp_verify", HERE / "verify_image.py")
prepare = _module("glm53_b12x_kda_adaptive_mtp_prepare", HERE / "prepare_context.py")


def test_glm53_b12x_kda_adaptive_mtp_source_build_is_exact_and_unqualified() -> None:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    assert pins["status"] == "implemented"
    build = pins["public_image_build"]
    assert build["status"] == "implemented"
    vllm = build["sources"]["vllm"]
    assert {key: vllm[key] for key in ("repository", "commit", "tree", "license")} == {
        "repository": "https://github.com/local-inference-lab/vllm.git",
        "commit": "0b67266a0f37d6146a8403fb8482403c62f412d5",
        "tree": "ba9484ccb33aa56e90ff2f447f15ca9b9da97639",
        "license": "Apache-2.0",
    }
    lineage = vllm["source_lineage"]
    assert lineage["included_commits"][-1] == vllm["commit"]
    assert lineage["included_commits"][-3:] == lineage["kda_live_tensor_commits"]
    assert len(lineage["included_commits"]) == 12
    assert build["outputs"]["runtime_image"] is None
    assert build["outputs"]["sparkcache_image"] is None
    sparkcache = pins["sparkcache"]
    assert sparkcache["commit"] == SPARKCACHE_COMMIT
    assert sparkcache["tree"] == SPARKCACHE_TREE
    assert sparkcache["source_tree_sha256"] == SPARKCACHE_SOURCE_SHA256
    assert sparkcache["cuda_config_schema"] == "canonical-v1"
    assert set(sparkcache["canonical_cuda_config_keys"]) == {
        "spark_cache_cuda_restore",
        "spark_cache_cuda_placement_library",
        "spark_cache_cuda_placement_library_sha256",
        "spark_cache_cuda_placement_arena_bytes",
        "spark_cache_cuda_restore_io_workers",
    }
    assert (
        sparkcache["overlay_containerfile_sha256"]
        == OVERLAY_CONTAINERFILE_SHA256
    )


def test_glm53_b12x_kda_adaptive_mtp_image_labels_bind_the_vllm_revision() -> None:
    pins = verify.load_pins(PINS)
    labels = verify.expected_labels(pins)
    assert labels["org.jovian.vllm.commit"] == (
        "0b67266a0f37d6146a8403fb8482403c62f412d5"
    )


def test_glm53_b12x_kda_adaptive_mtp_builder_uses_its_own_pins_and_image_name() -> None:
    script = (HERE / "build-image.sh").read_text(encoding="utf-8")
    assert "runtime/glm53-flash-b12x-kda-adaptive-mtp" in script
    assert "sparkring-glm53-runtime:b12x-kda-adaptive-mtp-0b67266a-arm64" in script
    assert "prepare_context.py" in script


def test_glm53_b12x_kda_adaptive_mtp_image_probe_requires_the_parallel_loader_dependency() -> None:
    script = (HERE / "verify_image.py").read_text(encoding="utf-8")
    assert "'fastsafetensors': importlib.metadata.version('fastsafetensors')" in script


def test_source_lineage_verifier_requires_the_exact_first_parent_sequence() -> None:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    lineage = pins["public_image_build"]["sources"]["vllm"]["source_lineage"]
    observed = "\n".join(lineage["included_commits"])
    with (
        mock.patch.object(prepare, "run", return_value=observed),
        mock.patch.object(
            prepare,
            "git_value",
            return_value=lineage["adaptive_mtp_tree"],
        ),
    ):
        prepare.verify_source_lineage(Path("/source/vllm"), lineage)

    with mock.patch.object(prepare, "run", return_value=observed.rsplit("\n", 1)[0]):
        try:
            prepare.verify_source_lineage(Path("/source/vllm"), lineage)
        except prepare.PrepareError as error:
            assert "first-parent lineage differs" in str(error)
        else:
            raise AssertionError("truncated vLLM lineage must be rejected")
