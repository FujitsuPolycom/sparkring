from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest import mock

import pytest


HERE = Path(__file__).resolve().parent
PINS = HERE / "pins.json"
SPARKCACHE_COMMIT = "20838ace3ebda570ca039cb7f1976c29da554b39"
SPARKCACHE_SOURCE_SHA256 = (
    "4998b24f4f504aeeb9bf92769ec720e282f546e6726d89fdfd06c4efa8d17c10"
)
OVERLAY_CONTAINERFILE_SHA256 = (
    "8e2377d034ba80b059f9a4387a6590a08e205313568ee1382e0e25342f8c5d40"
)
B12X_COMMIT = "b1d541f9e71a35f030d45fae437630fff7507c2a"
B12X_TREE = "c69cdec1c59a08e8e0e549f930fa8abcfb5134ae"
B12X_INCOMPATIBLE_COMMIT = "2fcf23a0ce269be27b2e03fece73d46e90e6aeea"
B12X_INCOMPATIBLE_SOURCE_SHA256 = (
    "b1f072405ad2f3bba83e720419a59ae791e1269dfccc9fc4a279889f8bd07d6e"
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
    b12x = build["sources"]["b12x"]
    assert {key: b12x[key] for key in ("repository", "commit", "tree", "license")} == {
        "repository": "https://github.com/local-inference-lab/b12x.git",
        "commit": B12X_COMMIT,
        "tree": B12X_TREE,
        "license": "Apache-2.0",
    }
    assert b12x["source_lineage"]["base_commit"] == B12X_INCOMPATIBLE_COMMIT
    assert b12x["source_lineage"]["included_commits"][-1] == B12X_COMMIT
    assert len(b12x["source_lineage"]["included_commits"]) == 16
    kda_contract = build["vllm_b12x_kda"]
    assert kda_contract["schema"] == "sparkring-vllm-b12x-live-tensor-kda/v1"
    assert kda_contract["metadata_validation"] == "trusted"
    assert kda_contract["tensor_binding"] == "request-sized-live-tensors"
    assert kda_contract["vllm_source"]["sha256"] == (
        "8bf8bc579dd4a80224dc1633e7513f2a0c58e07db72a736c7e41d28d3c35f3b9"
    )
    assert kda_contract["b12x_source"]["sha256"] == (
        "a2c81a486eb4d86a59f39e4f03855382583d39cdadd5fc8fd1c685ba28b1b56e"
    )
    assert build["outputs"]["runtime_image"] is None
    assert build["outputs"]["sparkcache_image"] is None
    sparkcache = pins["sparkcache"]
    assert sparkcache["commit"] == SPARKCACHE_COMMIT
    assert sparkcache["source_tree_sha256"] == SPARKCACHE_SOURCE_SHA256
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
    assert labels["org.jovian.b12x.commit"] == B12X_COMMIT
    assert labels["org.jovian.b12x.tree"] == B12X_TREE
    assert labels["org.jovian.b12x.kda-contract"] == (
        "sparkring-vllm-b12x-live-tensor-kda/v1"
    )


def test_image_inspection_rejects_the_incompatible_b12x_label() -> None:
    pins = verify.load_pins(PINS)
    labels = verify.expected_labels(pins)
    document = {
        "Id": "sha256:" + "a" * 64,
        "Architecture": "arm64",
        "Os": "linux",
        "Config": {"Labels": labels},
    }
    verify.validate_inspection(document, pins)
    labels["org.jovian.b12x.commit"] = B12X_INCOMPATIBLE_COMMIT
    with pytest.raises(verify.VerifyError, match="org.jovian.b12x.commit"):
        verify.validate_inspection(document, pins)


def test_glm53_b12x_kda_adaptive_mtp_builder_uses_its_own_pins_and_image_name() -> None:
    script = (HERE / "build-image.sh").read_text(encoding="utf-8")
    assert "runtime/glm53-flash-b12x-kda-adaptive-mtp" in script
    assert "sparkring-glm53-runtime:b12x-kda-adaptive-mtp-0b67266a-arm64" in script
    assert "prepare_context.py" in script
    assert "public_image_build.sources.b12x.tree" in script
    assert "public_image_build.vllm_b12x_kda.schema" in script


def test_glm53_b12x_kda_adaptive_mtp_image_probe_requires_the_parallel_loader_dependency() -> None:
    script = (HERE / "verify_image.py").read_text(encoding="utf-8")
    assert "'fastsafetensors': importlib.metadata.version('fastsafetensors')" in script
    assert "'request_sized_token_binding': 'mixed_qkv.shape[0]' in bind_source" in script
    assert "'trusted_metadata_dispatch'" in script


def test_runtime_probe_rejects_an_incompatible_installed_b12x_api() -> None:
    probe = {
        "nccl_sha256": "a" * 64,
        "source_receipt_present": True,
        "pins_present": True,
        "b12x_kda_contract": {
            "metadata_validation_default": None,
            "request_sized_token_binding": False,
            "request_sized_sequence_binding": False,
            "trusted_metadata_dispatch": False,
        },
    }
    with mock.patch.object(verify, "run", return_value=json.dumps(probe)):
        with pytest.raises(verify.VerifyError, match="request-sized live-tensor"):
            verify.runtime_probe("docker", "image")


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


def test_b12x_source_lineage_verifier_requires_the_exact_first_parent_sequence() -> None:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    lineage = pins["public_image_build"]["sources"]["b12x"]["source_lineage"]
    observed = "\n".join(lineage["included_commits"])
    with mock.patch.object(prepare, "run", return_value=observed):
        prepare.verify_b12x_source_lineage(Path("/source/b12x"), lineage)

    with mock.patch.object(prepare, "run", return_value=observed.rsplit("\n", 1)[0]):
        with pytest.raises(prepare.PrepareError, match="B12X first-parent lineage"):
            prepare.verify_b12x_source_lineage(Path("/source/b12x"), lineage)


def test_live_tensor_contract_rejects_b12x_2fcf23a_api() -> None:
    contract = json.loads(PINS.read_text(encoding="utf-8"))["public_image_build"][
        "vllm_b12x_kda"
    ]
    unsupported = contract["unsupported_b12x_sources"]
    assert unsupported == [
        {
            "status": "unsupported",
            "commit": B12X_INCOMPATIBLE_COMMIT,
            "tree": "58a046fc8faa747346f40f87166cda7e0f67ff47",
            "source_sha256": B12X_INCOMPATIBLE_SOURCE_SHA256,
            "limitation": (
                "Caps does not accept kda_metadata_validation, and bind_kda "
                "requires plan-sized tensors instead of request-sized live tensors."
            ),
        }
    ]
    with pytest.raises(prepare.PrepareError, match=B12X_INCOMPATIBLE_COMMIT):
        prepare.reject_unsupported_b12x_source(
            B12X_INCOMPATIBLE_SOURCE_SHA256,
            contract,
        )


def test_live_tensor_contract_rejects_an_incompatible_b12x_api(
    tmp_path: Path,
) -> None:
    contract = json.loads(PINS.read_text(encoding="utf-8"))["public_image_build"][
        "vllm_b12x_kda"
    ]
    vllm = tmp_path / "vllm-source"
    b12x = tmp_path / "b12x-source"
    vllm_path = vllm / contract["vllm_source"]["path"]
    b12x_path = b12x / contract["b12x_source"]["path"]
    vllm_path.parent.mkdir(parents=True)
    b12x_path.parent.mkdir(parents=True)
    vllm_path.write_text(
        '\n'.join(
            (
                'kda_metadata_validation="trusted"',
                "num_tokens = int(mixed_qkv.shape[0])",
                "query_start_loc = query_start_loc[: num_requests + 1]",
                "state_indices = state_indices[:num_requests, :state_columns]",
                "binding = api.bind_kda(",
                "api.run_kda(",
            )
        ),
        encoding="utf-8",
    )
    b12x_path.write_text(
        "class Caps:\n    pass\n\ndef bind_kda(plan, *, mixed_qkv):\n    "
        "shape = (plan.caps.max_tokens, plan.caps.packed_qkv_width)\n",
        encoding="utf-8",
    )
    contract = json.loads(json.dumps(contract))
    contract["vllm_source"]["sha256"] = prepare.sha256_file(vllm_path)
    contract["b12x_source"]["sha256"] = prepare.sha256_file(b12x_path)

    with pytest.raises(prepare.PrepareError, match="trusted KDA metadata policy"):
        prepare.verify_live_tensor_kda_contract(vllm, b12x, contract)
