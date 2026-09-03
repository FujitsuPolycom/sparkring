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
    ROOT / "runtime" / "glm53-flash" / "vllm-kv-block-lease-contract-da4d7be.json"
)
SITE_PATH = ROOT / "scripts" / "config" / "glm53-flash-tp4-site.example.yaml"
CACHE_PROFILE_PATH = (
    ROOT
    / "scripts"
    / "config"
    / "glm53-flash-dflash2-bf16-tp4-dcp1-sparkcache.example.json"
)
BASE_PROFILE_PATH = (
    ROOT / "scripts" / "config" / "glm53-flash-dflash2-bf16-tp4-dcp1.example.json"
)
BASE_RECIPE_PATH = ROOT / "recipes" / "glm53-flash-nvfp4-dflash2-bf16-tp4.json"
CACHE_RECIPE_PATH = (
    ROOT / "recipes" / "sparkcache" / "glm53-flash-nvfp4-dflash2-bf16-tp4.json"
)
CACHE_QUICKSTART_PATH = (
    ROOT / "docs" / "GLM53_FLASH_DFLASH2_BF16_SPARKCACHE_TP4_QUICKSTART.md"
)
BASE_QUICKSTART_PATH = ROOT / "docs" / "GLM53_FLASH_DFLASH2_BF16_TP4_QUICKSTART.md"
PROFILE_RECORD_PATH = ROOT / "docs" / "profiles" / "GLM53_FLASH_DFLASH2_BF16_TP4.md"
PERFORMANCE_RECORD_PATH = (
    ROOT
    / "performance"
    / "records"
    / "glm53-flash"
    / "sparkcache-dflash2-bf16-tp4-16k-run1-20260829.md"
)
PERFORMANCE_RECEIPT_PATH = (
    ROOT
    / "performance"
    / "receipts"
    / "glm53-flash"
    / "sparkcache-dflash2-bf16-tp4-20260829"
    / "benchmark-16k-run1.json"
)
KV20_RECEIPT_PATH = (
    ROOT
    / "performance"
    / "receipts"
    / "glm53-flash"
    / "sparkcache-dflash2-bf16-tp4-20g-20260829"
    / "observation.json"
)
KV20_RECORD_PATH = (
    ROOT
    / "performance"
    / "records"
    / "glm53-flash"
    / "sparkcache-dflash2-bf16-tp4-20g-20260829.md"
)
README_PATH = ROOT / "README.md"


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
    assert (
        "No base-checkpoint lineage is inferred"
        in (target["quantization"]["base_checkpoint_limitation"])
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
        486,
        489,
        493,
        494,
        497,
        499,
    ]
    assert vllm["merged_pull_requests"][-1]["merge_commit"] == vllm["commit"]
    assert "No relationship" in vllm["lineage_scope"]

    assert pins["b12x"]["commit"] == ("2fcf23a0ce269be27b2e03fece73d46e90e6aeea")
    assert pins["b12x"]["pull_request"] is None
    assert pins["patched_nccl"]["qualified_binary_sha256"] == (
        "5f1c3f10d5ace66d4ba584415bbfe42b6ac1a0a9116a3b81dcbe50516ad924b3"
    )
    assert pins["patched_nccl"]["commit"] == (
        "73cf112295c33aee2b895f329f592f2a9b4b0f97"
    )
    assert pins["patched_nccl"]["patched_tree"] == (
        "abdeb053b94c3f6d472cd55ae2b79ca821299009"
    )
    assert pins["sparkcache"]["image_build"]["containerfile_sha256"] == (
        "a2b65f3600950855cbfa00d82d532de1fbced3f4fa26c4bf1e59c3b6a519abd9"
    )
    assert pins["sparkcache"]["image_build"]["builder_sha256"] == (
        "b72466799e2fe569ecdee3a536cfb4606d2599da0a20cd398ca03aa99e21a3e6"
    )
    assert pins["sparkcache"]["commit"] == ("3860a2250193a6679ac6bac857af53e0757841f8")
    assert pins["spark_ring_profile"]["runtime_image_source_revision"] == (
        "862db89b1dd905e0ce3197f1d7b64b8a5802dbf1"
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
        assert _argument_value(arguments, "--kda-prefill-backend") == "triton"
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
        compilation = json.loads(_argument_value(arguments, "--compilation-config"))
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
    assert (
        base.extra_labels["org.sparkring.model-profile"]
        == (cache.extra_labels["org.sparkring.model-profile"])
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
        assert (
            profile.required_image_labels["org.sparkcache.deployment-profile"]
            == "glm53-flash-hybrid"
        )
        assert profile.required_image_labels["org.sparkcache.parent-image-id"] == (
            "sha256:7e8c0ebcb2001efb4cdab0ec9d20d53972e62db3688230044e22e61ffb1d35d5"
        )
        command = launcher.build_actions(site, profile, "start")[0].argv[2]
        assert "org.sparkcache.source-sha256" in command
        assert "verify_lease_contract.py" in command
        assert "/opt/sparkcache-source-identity.py" in command
        assert "source_tree_sha256" in command
        assert (
            "b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b"
            in command
        )
        assert (
            "676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996"
            in command
        )
        assert (
            "0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb"
            in command
        )
        assert (
            "5f1c3f10d5ace66d4ba584415bbfe42b6ac1a0a9116a3b81dcbe50516ad924b3"
            in command
        )
        assert "VLLM_HOST_IP=198.18.1.10" in command
        assert "--init" in command
        assert "--security-opt label=disable" in command


def test_site_and_runtime_profiles_have_one_identity_contract() -> None:
    site = load_site(SITE_PATH)
    for path in (BASE_PROFILE_PATH, CACHE_PROFILE_PATH):
        launcher._validate_site_profile_alignment(site, launcher.load_profile(path))


def test_recipes_record_qualified_cached_and_cache_disabled_evidence() -> None:
    base = _json(BASE_RECIPE_PATH)
    cache = _json(CACHE_RECIPE_PATH)
    image_receipt = _json(
        ROOT
        / "runtime"
        / "glm53-flash-jj-r8-gb10"
        / "async-store-completion-public-image-receipt.json"
    )

    assert base["status"] == "implemented"
    assert base["serving_common"]["external_kv_cache"] is False
    assert base["runtime"]["environment_overrides"] == {"SPARKCACHE_ENABLED": "0"}
    assert base["evidence"]["status"] == "implemented"
    assert base["preferred_profile"] == "dcp4"
    assert set(base["profiles"]) == {"dcp1", "dcp2", "dcp4"}
    assert cache["status"] == "qualified"
    assert cache["evidence"]["status"] == "qualified"
    assert cache["serving_common"]["async_scheduling"] is True
    assert cache["serving_common"]["native_prefix_caching"] is True
    assert cache["serving_common"]["chunked_prefill"] is True
    assert cache["profiles"]["dcp4"]["status"] == "qualified"
    assert cache["profiles"]["dcp4"]["preferred"] is True
    assert cache["profiles"]["dcp1"]["async_page_capture"] is False
    assert cache["profiles"]["dcp2"]["async_page_capture"] is False
    assert cache["profiles"]["dcp4"]["async_page_capture"] is True
    assert cache["profiles"]["dcp1"]["publication_schema"] == "snapshot-v1"
    assert cache["profiles"]["dcp1"]["cache_namespace_default"] == (
        "glm53-flash-dcp1-snapshot-v1"
    )
    assert cache["profiles"]["dcp2"]["publication_schema"] == "snapshot-v1"
    assert cache["profiles"]["dcp2"]["cache_namespace_default"] == (
        "glm53-flash-dcp2-snapshot-v1"
    )
    assert cache["profiles"]["dcp4"]["publication_schema"] == "tail-cow-v2"
    assert cache["profiles"]["dcp4"]["cache_namespace_default"] == (
        "glm53-flash-dcp4-page-tail-cow-v2"
    )
    assert cache["runtime"]["image"].endswith(
        "@sha256:e34aa58fda32c2cc63bc70de680b50c5f2bb69c1e0ad3c5bce0782c6501f7d34"
    )
    assert cache["runtime"]["image_id"] == (
        "sha256:058b17b49ee3b5ffd805fa4a17e4d9efcb885f92349b98a8c8623bd7f0f96dd4"
    )
    assert cache["runtime"]["sparkcache"]["source_commit"] == (
        "9c6218c96f1db233c0d17691dbc32a7d9fb2c0e4"
    )
    assert base["runtime"]["sparkring_source_commit"] == (
        image_receipt["sources"]["sparkring_image_commit"]
    )


def test_historical_async_capture_record_discloses_unavailable_source() -> None:
    record = (
        ROOT
        / "runtime"
        / "glm53-flash-jj-r8-gb10"
        / "ASYNC_CAPTURE_IMAGE_VALIDATION.md"
    ).read_text(encoding="utf-8")
    assert "d2f8911427d64bbb89c275814777fc3f8112fd21" in record
    assert "not reachable from the public repository" in record
    assert "history." in record
    assert "cannot be independently reconstructed" in record


def test_public_glm53_benchmark_retains_only_valid_run1_cells() -> None:
    receipt = _json(PERFORMANCE_RECEIPT_PATH)

    assert receipt["status"] == "research-only"
    assert receipt["observation_count"] == 1
    assert receipt["comparison"]["ab_baseline_present"] is False
    assert receipt["conditions"]["benchmark"]["working_file_sha256"] == (
        "07aad353cd9c894e14e9d1392c8509d3af8999c4022d3d22b29423a4572f5851"
    )
    assert receipt["conditions"]["benchmark"]["source_snapshot_published"] is False
    assert receipt["conditions"]["image"]["registry_digest"].endswith(
        "@sha256:cd4045bba2a0f3dc55361560f8c3a3f171939854db28d48dfdae58eed9c44943"
    )
    assert receipt["conditions"]["image"]["image_id_on_every_rank"] == (
        "sha256:7c007cf673c35f5818da7fea8faa343304baed00f489efdcbd027d6616b8a290"
    )
    assert receipt["result"]["valid_cells"] == [
        {
            "cell": "16K integrated-scout prefill",
            "tokens_per_second": 2371.0,
        },
        {
            "cell": "16K sustained C1 decode",
            "tokens_per_second": 36.05648849867254,
        },
    ]
    assert [cell["cell"] for cell in receipt["result"]["excluded_cells"]] == [
        "16K sustained C4 decode",
        "16K sustained C8 decode",
    ]
    assert all(
        "tokens_per_second" not in cell for cell in receipt["result"]["excluded_cells"]
    )


def test_public_glm53_benchmark_is_sanitized_and_front_page_lists_dcp_profiles() -> None:
    paths = [PERFORMANCE_RECEIPT_PATH, PERFORMANCE_RECORD_PATH]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "http://<rank-0-address>" in text
    assert "--chat-template-kwargs" in text
    assert "exact benchmark source snapshot is not published" in text
    assert "no A/B baseline" in text
    assert "invalid/excluded: capacity-limited" in text
    assert re.search(r"(?i)\b[A-Z]:\\", text) is None
    assert (
        re.search(r"\b(?:10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)", text) is None
    )
    assert "DESKTOP-" not in text
    assert "api_key" not in text.lower()

    readme = README_PATH.read_text(encoding="utf-8")
    assert "GLM-5.3 Flash NVFP4 target, BF16 DFlash2" in readme
    assert "| DCP1 | 4 Sparks · TP4/DCP1 |" in readme
    assert "| DCP2 | 4 Sparks · TP4/DCP2 |" in readme
    assert "| **DCP4 preferred** | **4 Sparks · TP4/DCP4** |" in readme
    assert "| 1.30M tokens |" in readme
    assert "| 2.90M tokens |" in readme
    assert "| **4.32M tokens** |" in readme
    assert "| 2,513 | 40.20 | 116.73 | C16: 168.39 | 71.67 |" in readme
    quickstart = (
        ROOT / "docs" / "GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md"
    ).read_text(encoding="utf-8")
    assert (
        "The preferred launch is TP4/DCP4 with 24 GiB of FP8 KV"
        in quickstart
    )
    assert "942,898-token needle" in readme
    assert "GLM-5.3 Flash research observation" not in readme
    assert "IN PROGRESS" not in readme


def test_twenty_gib_kv_observation_is_research_only_and_sanitized() -> None:
    receipt = _json(KV20_RECEIPT_PATH)
    assert receipt["status"] == "research-only"
    assert receipt["conditions"]["kv_cache_memory_bytes_per_rank"] == 20 * 1024**3
    assert receipt["conditions"]["max_model_len"] == 524288
    assert receipt["result"]["kv_capacity_tokens"] == 916676
    assert receipt["result"]["external_restore_tokens"] == 8192
    assert receipt["result"]["dflash_draft_tokens"] == (
        7 * receipt["result"]["dflash_drafts"]
    )
    assert receipt["result"]["preemptions"] == 0
    assert receipt["result"]["cgroup_oom_events_by_rank"] == [0, 0, 0, 0]
    assert "does not qualify a 1048576-token model limit" in receipt["conclusion"]

    text = KV20_RECORD_PATH.read_text(encoding="utf-8") + json.dumps(receipt)
    assert re.search(r"(?i)\b[A-Z]:\\", text) is None
    assert (
        re.search(r"\b(?:10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)", text) is None
    )
    assert "SparkCache storage remains configured for a 48 GiB ceiling" in text


def test_public_profile_files_do_not_embed_private_site_values_or_mutable_tags() -> (
    None
):
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
    assert (
        re.search(r"\b(?:10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)", text) is None
    )
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
        assert "Provenance" in text
        assert "local-inference-lab/GLM-5.3-Flash-NVFP4@520de24" in text
        assert "incoai/GLM-5.3-Flash-DFlash2@dc77ff1" in text
        assert "CC BY-NC-ND 4.0" in text
        assert "dev/jovian-judgement@da4d7be" in text
        assert "2fcf23a0ce269be27b2e03fece73d46e90e6aeea" in text
        assert "base-checkpoint revision" in text
        assert "FujitsuPolycom/sparkring/issues" in text
        assert "FujitsuPolycom/sparkcache/issues" in text
        assert (
            "@sha256:cd4045bba2a0f3dc55361560f8c3a3f171939854db28d48dfdae58eed9c44943"
            in text
        )

    for path in (CACHE_QUICKSTART_PATH, BASE_QUICKSTART_PATH):
        text = path.read_text(encoding="utf-8")
        assert "asynchronous scheduling" in text
        assert "native prefix caching" in text
        assert "chunked prefill" in text
        assert "docker logs --follow --tail 120" in text
        assert "--strict-placeholders" in text
    cache_text = CACHE_QUICKSTART_PATH.read_text(encoding="utf-8")
    assert "deploy/glm53_flash/build_public_image.py" in cache_text
    assert "runtime/glm53-flash/BUILD.md" in cache_text
    assert (
        "git -C sparkcache checkout --detach 3860a2250193a6679ac6bac857af53e0757841f8"
    ) in cache_text
    assert 'git -C sparkring checkout --detach "${sparkring_revision}"' in cache_text
    assert "checkout codex/glm53-flash-sparkcache-tp4" not in cache_text
    assert "metrics-before-restore.prom" in cache_text
    assert "restored 8192 tokens async" in cache_text
    assert "A rebuilt image has **implemented** status" in cache_text

    base_text = BASE_QUICKSTART_PATH.read_text(encoding="utf-8")
    assert "qualification-client checkout" in base_text
    assert "${sparkcache_root}/deploy/glm53_flash/qualification_request.py" in base_text


def test_ci_runs_glm53_runtime_contracts() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    pytest_command = next(
        line for line in workflow.splitlines() if "python -m pytest spark_transport" in line
    )
    assert "runtime/exl3-r7" in pytest_command
    assert "runtime/glm53-flash" in pytest_command
    assert "runtime/glm53-flash-jj-r8-gb10" in pytest_command
    assert "runtime/deepseek0731-gb10" in pytest_command
