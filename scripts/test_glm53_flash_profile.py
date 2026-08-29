"""CPU-only contracts for the GLM-5.3 Flash DFlash2 TP4 profiles."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sparkring_generic_launcher as launcher  # noqa: E402
from sparkring_site import load_site  # noqa: E402


PINS_PATH = ROOT / "runtime" / "glm53-flash" / "pins.json"
CONTRACT_PATH = (
    ROOT
    / "runtime"
    / "glm53-flash"
    / "vllm-kv-block-lease-contract-da4d7be.json"
)
SITE_PATH = ROOT / "scripts" / "config" / "glm53-flash-tp4-site.example.yaml"
CACHE_PROFILE_PATH = (
    ROOT
    / "scripts"
    / "config"
    / "glm53-flash-dflash2-bf16-tp4-dcp1-sparkcache.example.json"
)
BASE_PROFILE_PATH = (
    ROOT
    / "scripts"
    / "config"
    / "glm53-flash-dflash2-bf16-tp4-dcp1.example.json"
)
BASE_RECIPE_PATH = (
    ROOT / "recipes" / "glm53-flash-nvfp4-dflash2-bf16-tp4-dcp1.json"
)
CACHE_RECIPE_PATH = (
    ROOT
    / "recipes"
    / "sparkcache"
    / "glm53-flash-nvfp4-dflash2-bf16-tp4-dcp1.json"
)
CACHE_QUICKSTART_PATH = (
    ROOT / "docs" / "GLM53_FLASH_DFLASH2_BF16_SPARKCACHE_TP4_QUICKSTART.md"
)
BASE_QUICKSTART_PATH = (
    ROOT / "docs" / "GLM53_FLASH_DFLASH2_BF16_TP4_QUICKSTART.md"
)
PROFILE_RECORD_PATH = (
    ROOT / "docs" / "profiles" / "GLM53_FLASH_DFLASH2_BF16_TP4.md"
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _argument_value(arguments: tuple[str, ...], option: str) -> str:
    index = arguments.index(option)
    return arguments[index + 1]


def test_pins_record_complete_provenance_without_inferred_lineage() -> None:
    pins = _json(PINS_PATH)
    assert pins["schema"] == "sparkring-glm53-flash-runtime-lock/v1"
    assert pins["status"] == "qualified"

    target = pins["target_model"]
    assert target["repository"] == "local-inference-lab/GLM-5.3-Flash-NVFP4"
    assert target["revision"] == "520de24eabf507659eaef7c70f14fd584527facc"
    assert target["revision_uploader"] == "lukealonso"
    assert target["config_sha256"] == (
        "676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996"
    )
    assert target["weight_index_sha256"] == (
        "0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb"
    )
    assert target["quantization"]["tool_version"] == (
        "0.39.0.dev290+gf9d9a71de.d20260407"
    )
    assert target["quantization"]["base_checkpoint_revision"] is None
    assert "No base-checkpoint lineage is inferred" in (
        target["quantization"]["base_checkpoint_limitation"]
    )

    draft = pins["draft_model"]
    assert draft["revision_uploader"] == "zhijianliu"
    assert draft["dtype"] == "bfloat16"
    assert draft["license"] == "CC-BY-NC-ND-4.0"
    assert draft["config_sha256"] == (
        "c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573"
    )
    assert draft["weights_sha256"] == (
        "b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b"
    )

    vllm = pins["vllm"]
    assert vllm["branch"] == "dev/jovian-judgement"
    assert vllm["commit"] == "da4d7be6c97434f6942292ed8abbf4b32dc44355"
    assert {entry["commit"] for entry in vllm["direct_commits"]} == {
        "e0db84abedb4a85f93d130252e54b73c0f3ed695",
        "0c878821cf46c99729c7936bcbd4d868ad40e44e",
        "4dbd82b9ced13114f90e93b8b6fae0966c942a3b",
        "1036123e935177900122c14d3cf02ad67b5422aa",
        "e7097feb6fcdf57911cd68884420af2d80600dd7",
    }
    assert [entry["number"] for entry in vllm["merged_pull_requests"]] == [
        486, 489, 493, 494, 497, 499,
    ]
    assert vllm["merged_pull_requests"][-1]["merge_commit"] == vllm["commit"]
    assert "No relationship" in vllm["lineage_scope"]

    assert pins["b12x"]["commit"] == (
        "2fcf23a0ce269be27b2e03fece73d46e90e6aeea"
    )
    assert pins["b12x"]["pull_request"] is None
    assert pins["patched_nccl"]["qualified_binary_sha256"] == (
        "ccd57342449c3f680befcb379329b935746e5299dc4de5f2516146e0411bd85f"
    )
    assert "does not bind" in pins["patched_nccl"]["source_limitation"]
    assert pins["sparkcache"]["image_build"]["containerfile_sha256"] == (
        "ccc6b39173df80f604820959c3f19f8bc363f79d11f7d4f2d913054a4161b3f5"
    )
    assert pins["sparkcache"]["image_build"]["builder_sha256"] == (
        "c130e5c2fdd5f33e73f90f04ef85fa1247d93bfe6db409cd99508841f8d84547"
    )
    assert pins["sparkcache"]["commit"] == (
        "2d6a222f04fcb7b903cb899aba3ed3fdc75edc11"
    )
    assert pins["spark_ring_profile"]["publication_revision"] == (
        "d45572dbd2adc7afa1d3208fb801c8ad9eac7864"
    )


def test_vllm_lease_contract_matches_the_pinned_copy() -> None:
    pins = _json(PINS_PATH)
    expected = pins["sparkcache"]["vllm_lease_contract"]["sha256"]
    contract = _json(CONTRACT_PATH)

    assert _sha256(CONTRACT_PATH) == expected
    assert contract["vllm_commit"] == pins["vllm"]["commit"]
    assert len(contract["files"]) == 7
    assert {record["path"] for record in contract["files"]} >= {
        "vllm/v1/core/single_type_kv_cache_manager.py",
        "vllm/v1/kv_cache_interface.py",
    }


def test_site_template_encodes_the_qualified_tp4_geometry() -> None:
    site = load_site(SITE_PATH)

    assert len(site.ranks) == 4
    assert site.serving.tensor_parallel_size == 4
    assert site.serving.decode_context_parallel_size == 1
    assert site.serving.mtp_mode == "off"
    assert site.serving.mtp_tokens == 0
    assert site.serving.max_model_len == 524288
    assert site.serving.kv_cache_bytes_per_rank == 12884901888
    assert site.serving.max_num_seqs == 32
    assert str(site.runtime.model_revision) == (
        "520de24eabf507659eaef7c70f14fd584527facc"
    )


def test_profiles_preserve_scheduler_prefix_and_prefill_contracts() -> None:
    profiles = [
        launcher.load_profile(BASE_PROFILE_PATH),
        launcher.load_profile(CACHE_PROFILE_PATH),
    ]
    for profile in profiles:
        assert profile.init is True
        assert profile.security_opts == ("label=disable",)
        arguments = profile.extra_vllm_args
        assert "--async-scheduling" in arguments
        assert "--enable-prefix-caching" in arguments
        assert "--enable-chunked-prefill" in arguments
        assert _argument_value(arguments, "--max-num-batched-tokens") == "8192"
        assert _argument_value(arguments, "--mamba-cache-mode") == "align"
        assert _argument_value(arguments, "--kda-prefill-backend") == "flashkda"
        assert _argument_value(arguments, "--max-cudagraph-capture-size") == "256"
        assert json.loads(_argument_value(arguments, "--speculative-config")) == {
            "method": "dflash",
            "model": "/mtp-draft",
            "num_speculative_tokens": 7,
            "draft_tensor_parallel_size": 4,
            "kv_cache_dtype": "auto",
            "draft_sample_method": "probabilistic",
            "rejection_sample_method": "standard",
        }
        compilation = json.loads(
            _argument_value(arguments, "--compilation-config")
        )
        assert compilation["cudagraph_mode"] == "FULL_AND_PIECEWISE"
        assert compilation["cudagraph_capture_sizes"] == [8, 16, 32, 64, 128, 256]


def test_cache_profile_adds_only_the_external_connector_to_serving_arguments() -> None:
    base = launcher.load_profile(BASE_PROFILE_PATH)
    cache = launcher.load_profile(CACHE_PROFILE_PATH)

    assert "--kv-transfer-config" not in base.extra_vllm_args
    option_index = cache.extra_vllm_args.index("--kv-transfer-config")
    assert cache.extra_vllm_args[:option_index] == base.extra_vllm_args
    connector = json.loads(cache.extra_vllm_args[option_index + 1])
    assert connector["kv_connector"] == "SparkContextCacheConnector"
    assert connector["kv_load_failure_policy"] == "recompute"
    extra = connector["kv_connector_extra_config"]
    assert extra["spark_cache_model_profile"] == "glm53-flash-hybrid"
    assert extra["spark_cache_draft_policy"] == "separate"
    assert extra["spark_cache_streaming_snapshots"] is False
    assert extra["spark_cache_native_restore"] is False

    for attribute in (
        "image",
        "image_id",
        "model_family",
        "model_host_path",
        "model_container_path",
        "shm_size",
        "startup_timeout_seconds",
        "environment",
        "extra_volumes",
        "init",
        "security_opts",
        "privileged",
        "entrypoint",
        "confirmation",
        "identity",
        "required_image_labels",
        "attestation_hook",
        "health_check",
    ):
        assert getattr(base, attribute) == getattr(cache, attribute), attribute
    assert base.extra_labels["org.sparkring.model-profile"] == (
        cache.extra_labels["org.sparkring.model-profile"]
    )
    assert set(base.extra_labels) ^ set(cache.extra_labels) == set()
    assert base.extra_labels["org.sparkring.external-cache"] == "disabled"
    assert cache.extra_labels["org.sparkring.external-cache"] == "sparkcache"


def test_profiles_fail_closed_on_image_source_draft_contract_and_nccl() -> None:
    site = load_site(SITE_PATH)
    for path in (BASE_PROFILE_PATH, CACHE_PROFILE_PATH):
        profile = launcher.load_profile(path)
        assert launcher._is_template(profile) is True
        assert profile.required_image_labels["org.jovian.vllm.commit"] == (
            "da4d7be6c97434f6942292ed8abbf4b32dc44355"
        )
        assert profile.required_image_labels[
            "org.sparkcache.deployment-profile"
        ] == "glm53-flash-hybrid"
        assert profile.required_image_labels["org.sparkcache.parent-image-id"] == (
            "sha256:" + "0" * 64
        )
        command = launcher.build_actions(site, profile, "start")[0].argv[2]
        assert "org.sparkcache.source-sha256" in command
        assert "verify_lease_contract.py" in command
        assert "/opt/sparkcache-source-identity.py" in command
        assert "source_tree_sha256" in command
        assert "b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b" in command
        assert "676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996" in command
        assert "0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb" in command
        assert "ccd57342449c3f680befcb379329b935746e5299dc4de5f2516146e0411bd85f" in command
        assert "VLLM_HOST_IP=198.18.1.10" in command
        assert "--init" in command
        assert "--security-opt label=disable" in command


def test_site_and_runtime_profiles_have_one_identity_contract() -> None:
    site = load_site(SITE_PATH)
    for path in (BASE_PROFILE_PATH, CACHE_PROFILE_PATH):
        launcher._validate_site_profile_alignment(
            site, launcher.load_profile(path)
        )


def test_recipes_distinguish_qualified_cache_evidence_from_implementation() -> None:
    base = _json(BASE_RECIPE_PATH)
    cache = _json(CACHE_RECIPE_PATH)

    assert base["status"] == "implemented"
    assert base["serving"]["external_kv_cache"] is False
    assert base["evidence"]["status"] == "implemented"
    assert cache["status"] == "qualified"
    assert cache["evidence"]["status"] == "qualified"
    assert cache["serving"]["async_scheduling"] is True
    assert cache["serving"]["native_prefix_caching"] is True
    assert cache["serving"]["chunked_prefill"] is True
    assert cache["evidence"]["external_cache_hit_tokens"] == 8192
    assert cache["evidence"]["restore_request_seconds"] == 1.509
    assert cache["evidence"]["draft_tokens"] == 7 * cache["evidence"]["drafts"]
    assert [entry["rank"] for entry in cache["runtime"]["qualified_images_by_rank"]] == [
        0, 1, 2, 3,
    ]
    assert all(
        entry["base_image_id"].startswith("sha256:")
        and entry["derived_image_id"].startswith("sha256:")
        for entry in cache["runtime"]["qualified_images_by_rank"]
    )
    receipt_dir = (
        ROOT
        / "performance"
        / "receipts"
        / "glm53-flash"
        / "sparkcache-dflash2-bf16-tp4-20260828"
    )
    receipts = {path.name: _json(path) for path in receipt_dir.glob("*.json")}
    assert set(receipts) == {
        "cold.json",
        "post-restart-prime.json",
        "post-restart-restore.json",
        "post-restore-semantic.json",
    }
    assert receipts["post-restart-restore.json"]["elapsed_seconds"] == 1.509
    assert receipts["post-restore-semantic.json"]["semantic_match"] is True


def test_public_profile_files_do_not_embed_private_site_values_or_mutable_tags() -> None:
    paths = [
        PINS_PATH,
        SITE_PATH,
        CACHE_PROFILE_PATH,
        BASE_PROFILE_PATH,
        BASE_RECIPE_PATH,
        CACHE_RECIPE_PATH,
        CACHE_QUICKSTART_PATH,
        BASE_QUICKSTART_PATH,
        PROFILE_RECORD_PATH,
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert re.search(r"(?i)\b[A-Z]:\\(?:Users|home)\\", text) is None
    assert re.search(r"\b(?:10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)", text) is None
    assert "/home/" not in text
    assert "/var/tmp/" not in text
    assert "spark-r0" not in text
    assert "HF_TOKEN=" not in text
    assert "@sha256:" in text
    for profile_path in (CACHE_PROFILE_PATH, BASE_PROFILE_PATH):
        image = _json(profile_path)["image"]
        assert "@sha256:" in image
        assert not image.endswith(":latest")


def test_operator_docs_state_status_invariants_and_full_provenance() -> None:
    for path in (CACHE_QUICKSTART_PATH, BASE_QUICKSTART_PATH, PROFILE_RECORD_PATH):
        text = path.read_text(encoding="utf-8")
        assert "## Provenance" in text
        assert "local-inference-lab/GLM-5.3-Flash-NVFP4@520de24" in text
        assert "incoai/GLM-5.3-Flash-DFlash2@dc77ff1" in text
        assert "CC BY-NC-ND 4.0" in text
        assert "dev/jovian-judgement@da4d7be" in text
        assert "2fcf23a0ce269be27b2e03fece73d46e90e6aeea" in text
        assert "ccd57342449c3f680befcb379329b935746e5299dc4de5f2516146e0411bd85f" in text
        assert "6210f439c64e4079ed3304c9cc181174abb3e6045de740ba7b7c2546bcaf6ac2" in text
        assert "2d6a222f04fcb7b903cb899aba3ed3fdc75edc11" in text
        assert "base-checkpoint revision" in text
        assert "binary" in text and "source commit" in text
        assert "Docker image publication" in text
        assert "FujitsuPolycom community" in text
        assert "parent-image" in text
        assert "FujitsuPolycom/sparkring/issues" in text
        assert "FujitsuPolycom/sparkcache/issues" in text
        assert "announcement" in text.casefold()

    for path in (CACHE_QUICKSTART_PATH, BASE_QUICKSTART_PATH):
        text = path.read_text(encoding="utf-8")
        assert "--async-scheduling" in text
        assert "--enable-prefix-caching" in text
        assert "--enable-chunked-prefill" in text
        assert "docker logs --follow --tail 120" in text
        assert "org.sparkcache.parent-image-id" in text
        assert "Public reproduction: unsupported" in text
        assert "--strict-placeholders" in text
        assert "fatal_pattern=" in text
    cache_text = CACHE_QUICKSTART_PATH.read_text(encoding="utf-8")
    assert "deploy/glm53_flash/build_image.py" in cache_text
    assert '--base-image-id "${base_image_id}"' in cache_text
    assert "hf cache verify" in cache_text
    assert "metrics-before-restore.prom" in cache_text
    assert '${TARGET_MODEL_DIR:?' in cache_text
    assert "temporary_build_tag" in cache_text
    assert "inherits qualified" not in cache_text
