from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

import sparkring_generic_launcher as launcher
from prepare_glm53_dflash7_python_overlay_profile import (
    ALLOWED_RUNTIME_WARNINGS,
    B12X_COMMIT,
    DEEP_EP_DISTRIBUTION,
    DEEP_EP_REMOVAL_RECEIPT_SHA256,
    DFLASH_LOADER_PATCH_SHA256,
    DFLASH_LOADER_POSTIMAGE_SHA256,
    RECURRENT_BOUNDARY_PATCH_SHA256,
    DFLASH_WEIGHTS_SHA256,
    SAFE_CUDA_PLACEMENT_SHA256,
    SAFE_IMAGE,
    SAFE_IMAGE_ID,
    SAFE_SOURCE_RECEIPT_SHA256,
    SPARKCACHE_COMMIT,
    SPARKCACHE_SOURCE_SHA256,
    SPARKCACHE_TREE,
    VLLM_NATIVE_COMMIT,
    VLLM_PYTHON_COMMIT,
    ResolveError,
    resolve,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts/config"
FAST = (
    CONFIG
    / "glm53-flash-dflash7-python-overlay-fastsafetensors-sparkcache-tp4-dcp1.example.json"
)
SAFE = (
    CONFIG
    / "glm53-flash-dflash7-python-overlay-safetensors-sparkcache-tp4-dcp1.example.json"
)
SITE = CONFIG / "glm53-flash-tp4-site.example.yaml"
GUIDE = ROOT / "docs/GLM53_DFLASH7_PYTHON_OVERLAY_SPARKCACHE_TP4_QUICKSTART.md"
QUALIFIED_IMAGE_ID = (
    "sha256:35b58a7bf414059c65b8f74e4e4b17ee6a81b7008e1bffbc9bd298b5e08c739e"
)
QUALIFIED_SPARKCACHE_COMMIT = "a1511d26a1fe2b17b24561bc52e376bf7f54b06a"
DIGESTS = {
    "cuda_placement_library_sha256": SAFE_CUDA_PLACEMENT_SHA256,
    "native_elf_manifest_sha256": "2b" * 32,
    "native_dispatch_manifest_sha256": "3c" * 32,
    "source_receipt_sha256": SAFE_SOURCE_RECEIPT_SHA256,
}


def _argument(profile: dict, option: str) -> str:
    args = profile["extra_vllm_args"]
    return args[args.index(option) + 1]


def _resolved(path: Path) -> tuple[dict, dict]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    site = yaml.safe_load(SITE.read_text(encoding="utf-8"))
    profile["model_host_path"] = "/srv/models/glm53"
    profile["extra_volumes"][0]["host"] = "/srv/models/dflash2"
    profile["extra_volumes"][1]["host"] = "/srv/cache/glm53-dflash7"
    management = ["10.20.0.10", "10.20.0.11", "10.20.0.12", "10.20.0.13"]
    networks = ["10.30.1.0/24", "10.30.2.0/24", "10.30.3.0/24", "10.30.4.0/24"]
    ring = (
        ("10.30.1.10", "10.30.4.10"),
        ("10.30.1.11", "10.30.2.11"),
        ("10.30.2.12", "10.30.3.12"),
        ("10.30.3.13", "10.30.4.13"),
    )
    for edge, subnet in zip(site["topology"]["edges"], networks, strict=True):
        edge["subnet"] = subnet
    for rank, address, ports in zip(site["ranks"], management, ring, strict=True):
        rank["ssh_target"] = f"operator@{address}"
        rank["management"]["address"] = address
        for port, port_address in zip(rank["ring_ports"], ports, strict=True):
            port["address"] = port_address
        for peer in rank["transport_peers"]:
            peer["address"] = management[peer["rank"]]
    image = SAFE_IMAGE if path == FAST else "local/glm53-dflash7-overlay@sha256:" + "ab" * 32
    image_id = SAFE_IMAGE_ID if path == FAST else "sha256:" + "cd" * 32
    return resolve(
        profile,
        site,
        image=image,
        image_id=image_id,
        **DIGESTS,
    )


@pytest.mark.parametrize("path,loader", [(FAST, "fastsafetensors"), (SAFE, "safetensors")])
def test_profiles_pin_external_dflash7_and_tp4(path: Path, loader: str) -> None:
    profile = json.loads(path.read_text(encoding="utf-8"))
    speculative = json.loads(_argument(profile, "--speculative-config"))
    assert speculative["method"] == "dflash"
    assert speculative["num_speculative_tokens"] == 7
    assert speculative["draft_tensor_parallel_size"] == 4
    assert _argument(profile, "--load-format") == loader
    assert profile["identity"]["draft_weights_sha256"] == DFLASH_WEIGHTS_SHA256
    assert profile["identity"]["max_num_seqs"] == "32"
    assert profile["identity"]["vllm_block_size"] == "256"
    assert profile["identity"]["kv_cache_dtype"] == "fp8"
    assert not any(
        name.startswith("INSTANTTENSOR_") for name in profile["environment"]
    )
    environment = profile["environment"]
    assert "OMP_NUM_THREADS" not in environment
    assert "PYTHONWARNINGS" not in environment
    assert environment["VLLM_ALLREDUCE_USE_SYMM_MEM"] == "0"
    assert environment["VLLM_ALLREDUCE_USE_FLASHINFER"] == "0"
    assert environment["VLLM_NCCL_SO_PATH"] == "/opt/sparkring/nccl/libnccl.so.2"
    assert environment["LD_PRELOAD"] == environment["VLLM_NCCL_SO_PATH"]
    assert environment["VLLM_ENABLE_PCIE_ALLREDUCE"] == "0"
    assert profile["extra_vllm_args"].count("--disable-custom-all-reduce") == 1
    assert profile["extra_vllm_args"].count("--language-model-only") == 1
    assert _argument(profile, "--attention-backend") == "B12X"
    assert _argument(profile, "--moe-backend") == "b12x"
    assert _argument(profile, "--linear-backend") == "b12x"
    compilation = json.loads(_argument(profile, "--compilation-config"))
    assert compilation["pass_config"]["fuse_allreduce_rms"] is False
    assert profile["identity"]["deep_ep_removed_distribution"] == (
        DEEP_EP_DISTRIBUTION
    )
    assert profile["identity"]["deep_ep_module_status"] == "absent"
    assert profile["identity"]["allowed_runtime_warnings"] == (
        ALLOWED_RUNTIME_WARNINGS
    )


def test_fastsafetensors_profile_separates_the_draft_loader() -> None:
    profile = json.loads(FAST.read_text(encoding="utf-8"))
    assert profile["image"] == SAFE_IMAGE
    assert profile["image_id"] == SAFE_IMAGE_ID
    assert profile["environment"]["VLLM_FASTSAFETENSORS_QUEUE_SIZE"] == "1"
    speculative = json.loads(_argument(profile, "--speculative-config"))
    assert speculative["draft_load_config"] == {"load_format": "safetensors"}
    assert profile["identity"]["dflash_peak_gpu_memory_status"] == "implemented"
    assert profile["identity"]["sparkcache_publication_schema"] == "snapshot-v1"
    assert profile["identity"]["sparkcache_effective_publication_schema"] == (
        "page-snapshot-v1"
    )
    assert profile["extra_labels"]["org.sparkring.qualification-status"] == (
        "implemented"
    )


def test_fastsafetensors_resolver_rejects_predecessor_images_and_source() -> None:
    profile = json.loads(FAST.read_text(encoding="utf-8"))
    site = yaml.safe_load(SITE.read_text(encoding="utf-8"))
    kwargs = {"image": SAFE_IMAGE, "image_id": SAFE_IMAGE_ID, **DIGESTS}

    for predecessor in (
        "sha256:ed60be066d6d9eadea267bc4597a0687869f3ddb95a3e5c6f86649893a838eb8",
        "sha256:cc2c0e2f812f4b78d5b91f863aaf46fd8e8e505844245aa50911af1fb8e061c0",
    ):
        with pytest.raises(ResolveError, match="exact image35b"):
            resolve(
                copy.deepcopy(profile),
                copy.deepcopy(site),
                **{**kwargs, "image_id": predecessor},
            )

    changed = copy.deepcopy(profile)
    changed["identity"]["sparkcache_source_revision"] = (
        "65b6642df1afc64366430d3aef9aca01f5c5e1c3"
    )
    with pytest.raises(ResolveError, match="sparkcache_source_revision"):
        resolve(changed, copy.deepcopy(site), **kwargs)


def test_cuda_restore_uses_only_canonical_configuration_keys() -> None:
    profile = json.loads(FAST.read_text(encoding="utf-8"))
    transfer = json.loads(_argument(profile, "--kv-transfer-config"))
    extra = transfer["kv_connector_extra_config"]
    assert extra["spark_cache_cuda_restore"] is True
    assert extra["spark_cache_cuda_placement_library"].endswith(
        "libspark_cache_placement.so"
    )
    assert extra["spark_cache_cuda_placement_arena_bytes"] == 256 * 1024**2
    assert extra["spark_cache_cuda_restore_io_workers"] == 8
    assert extra["spark_cache_load_threads"] == 1
    assert extra["spark_cache_max_pending_restores"] == 1
    assert extra["spark_cache_root"].endswith("snapshot-v1-safe")
    assert extra["spark_cache_clear_once"].endswith("snapshot-v1-safe")
    assert not any(key.startswith("spark_cache_native_") for key in extra)


def test_loader_profiles_share_model_identity_but_isolate_test_roots() -> None:
    fast = json.loads(FAST.read_text(encoding="utf-8"))
    safe = json.loads(SAFE.read_text(encoding="utf-8"))
    fast_extra = json.loads(_argument(fast, "--kv-transfer-config"))[
        "kv_connector_extra_config"
    ]
    safe_extra = json.loads(_argument(safe, "--kv-transfer-config"))[
        "kv_connector_extra_config"
    ]
    assert fast_extra["spark_cache_draft_checkpoint_sha256"] == (
        safe_extra["spark_cache_draft_checkpoint_sha256"]
    )
    assert fast_extra["spark_cache_publication_schema"] == "snapshot-v1"
    assert safe_extra["spark_cache_publication_schema"] == "tail-cow-v1"
    assert fast_extra["spark_cache_root"] != safe_extra["spark_cache_root"]
    assert fast_extra["spark_cache_clear_once"] != safe_extra["spark_cache_clear_once"]


def test_resolved_profile_requires_dflash7_image_labels() -> None:
    profile, _ = _resolved(FAST)
    assert profile["image"] == SAFE_IMAGE
    assert profile["image_id"] == SAFE_IMAGE_ID
    labels = profile["required_image_labels"]
    assert labels["org.jovian.vllm.commit"] == VLLM_NATIVE_COMMIT
    assert labels["org.sparkring.vllm.python.commit"] == VLLM_PYTHON_COMMIT
    assert labels["org.jovian.b12x.commit"] == B12X_COMMIT
    assert labels["org.sparkcache.deployment-profile"] == (
        "glm53-flash-dflash7-python-overlay"
    )
    assert labels["org.sparkcache.cuda-config-schema"] == "canonical-v1"
    assert labels["org.sparkcache.source-revision"] == SPARKCACHE_COMMIT
    assert labels["org.sparkcache.source-tree"] == SPARKCACHE_TREE
    assert labels["org.sparkcache.source-sha256"] == SPARKCACHE_SOURCE_SHA256
    assert labels["org.sparkcache.cuda-placement-library-sha256"] == (
        SAFE_CUDA_PLACEMENT_SHA256
    )
    assert labels["org.sparkring.source-receipt-sha256"] == (
        SAFE_SOURCE_RECEIPT_SHA256
    )
    assert labels["org.sparkring.vllm.dflash-draft-loader-patch-sha256"] == (
        DFLASH_LOADER_PATCH_SHA256
    )
    assert labels["org.sparkring.vllm.dflash-draft-loader-postimage-sha256"] == (
        DFLASH_LOADER_POSTIMAGE_SHA256
    )
    assert labels["org.sparkring.vllm.recurrent-boundary-patch-sha256"] == (
        RECURRENT_BOUNDARY_PATCH_SHA256
    )
    assert labels["org.sparkring.runtime.removed-deep-ep-distribution"] == (
        DEEP_EP_DISTRIBUTION
    )
    assert labels["org.sparkring.runtime.deep-ep-removal-receipt-sha256"] == (
        DEEP_EP_REMOVAL_RECEIPT_SHA256
    )
    assert "adaptive" not in " ".join(labels.values()).lower()
    attestation = " ".join(profile["attestation_hook"])
    assert SPARKCACHE_SOURCE_SHA256 in attestation
    assert SAFE_SOURCE_RECEIPT_SHA256 in attestation
    assert SAFE_CUDA_PLACEMENT_SHA256 in attestation


def test_resolver_rejects_mtp_or_noncanonical_cuda_restore() -> None:
    profile = json.loads(FAST.read_text(encoding="utf-8"))
    site = yaml.safe_load(SITE.read_text(encoding="utf-8"))
    kwargs = {
        "image": SAFE_IMAGE,
        "image_id": SAFE_IMAGE_ID,
        **DIGESTS,
    }
    changed = copy.deepcopy(profile)
    changed["identity"]["speculator"] = "embedded_mtp"
    with pytest.raises(ResolveError, match="speculator"):
        resolve(changed, copy.deepcopy(site), **kwargs)

    changed = copy.deepcopy(profile)
    transfer = json.loads(_argument(changed, "--kv-transfer-config"))
    transfer["kv_connector_extra_config"].pop("spark_cache_cuda_restore_io_workers")
    args = changed["extra_vllm_args"]
    args[args.index("--kv-transfer-config") + 1] = json.dumps(transfer)
    with pytest.raises(ResolveError, match="cuda_restore_io_workers"):
        resolve(changed, copy.deepcopy(site), **kwargs)

    changed = copy.deepcopy(profile)
    transfer = json.loads(_argument(changed, "--kv-transfer-config"))
    transfer["kv_connector_extra_config"].pop("spark_cache_max_pending_restores")
    args = changed["extra_vllm_args"]
    args[args.index("--kv-transfer-config") + 1] = json.dumps(transfer)
    with pytest.raises(ResolveError, match="max_pending_restores"):
        resolve(changed, copy.deepcopy(site), **kwargs)

    changed = copy.deepcopy(profile)
    changed["identity"]["sparkcache_publication_schema"] = "tail-cow-v1"
    with pytest.raises(ResolveError, match="sparkcache_publication_schema"):
        resolve(changed, copy.deepcopy(site), **kwargs)

    changed = copy.deepcopy(profile)
    speculative = json.loads(_argument(changed, "--speculative-config"))
    speculative.pop("draft_load_config")
    args = changed["extra_vllm_args"]
    args[args.index("--speculative-config") + 1] = json.dumps(speculative)
    with pytest.raises(ResolveError, match="draft_load_config safetensors"):
        resolve(changed, copy.deepcopy(site), **kwargs)

    changed = copy.deepcopy(profile)
    changed["environment"]["OMP_NUM_THREADS"] = "16"
    with pytest.raises(ResolveError, match="OMP_NUM_THREADS"):
        resolve(changed, copy.deepcopy(site), **kwargs)

    changed = copy.deepcopy(profile)
    changed["required_image_labels"].pop(
        "org.sparkring.vllm.recurrent-boundary-patch-sha256"
    )
    with pytest.raises(ResolveError, match="recurrent-boundary-patch-sha256"):
        resolve(changed, copy.deepcopy(site), **kwargs)

    changed = copy.deepcopy(profile)
    changed["environment"].pop("VLLM_ALLREDUCE_USE_SYMM_MEM")
    with pytest.raises(ResolveError, match="VLLM_ALLREDUCE_USE_SYMM_MEM"):
        resolve(changed, copy.deepcopy(site), **kwargs)

    changed = copy.deepcopy(profile)
    compilation = json.loads(_argument(changed, "--compilation-config"))
    compilation["pass_config"]["fuse_allreduce_rms"] = True
    args = changed["extra_vllm_args"]
    args[args.index("--compilation-config") + 1] = json.dumps(compilation)
    with pytest.raises(ResolveError, match="all-reduce RMS fusion"):
        resolve(changed, copy.deepcopy(site), **kwargs)


def test_generic_launcher_emits_four_rank_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile, site = _resolved(SAFE)
    profile_path = tmp_path / "profile.json"
    site_path = tmp_path / "site.yaml"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    site_path.write_text(yaml.safe_dump(site, sort_keys=False), encoding="utf-8")
    assert launcher.main(
        ["--site", str(site_path), "--profile", str(profile_path), "plan"]
    ) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["actions"]) == 4
    rendered = json.dumps(plan)
    assert profile["container_name"] in rendered
    assert "--max-num-seqs" in rendered and "32" in rendered


def test_quickstart_names_both_loader_statuses_and_exact_builder() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    assert "runtime/glm53-flash-dflash7-python-overlay/build-image.sh" in guide
    assert FAST.name in guide and SAFE.name in guide
    assert "implemented" in guide and "qualified" in guide
    assert QUALIFIED_IMAGE_ID in guide
    assert QUALIFIED_SPARKCACHE_COMMIT in guide
    assert SPARKCACHE_COMMIT in guide
    assert "full `snapshot-v1` publication" in guide
    assert "13\n  authenticated macro objects per rank" in guide
    assert "1.552–1.700 seconds" in guide
    assert "expected and observed oracle `red`" in guide
    assert "C2 delta-restore" in guide
    assert "does not qualify C2 delta restore" in guide
    assert "host-base read coalescing" in guide
    assert "different-root concurrent restore" in guide
    assert "research-only" in guide
    assert "rebuilds are not qualified" in guide
    assert "twelve-file lease contract" in guide
    assert "command that changes\ncontainers" in guide
    assert "Response quality and public OCI publication are **unsupported**" in guide
    assert DFLASH_WEIGHTS_SHA256 in guide
    assert DFLASH_LOADER_PATCH_SHA256 in guide
    assert "formats, cache roots,\nand one-shot clear tokens are distinct" in guide
    assert "No legacy-key rewrite" in guide
    assert "DeepEP" in guide
    assert "ModelOpt" in guide
    assert "FP8 KV" in guide
    assert "START_GLM53_FLASH_DFLASH7_PYTHON_OVERLAY_FASTSAFETENSORS_TP4" in guide
    assert "docker logs --follow --tail 120" in guide
    assert "spark_cache_clear_once" in guide
    assert "prefill-schedule-interval" in guide
    shortest = guide.split("## Shortest qualified start", 1)[1].split(
        "### Research-only", 1
    )[0]
    assert FAST.name in shortest


def test_quickstart_has_one_copy_paste_four_rank_start_path() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    section = guide.split("## Shortest qualified start", 1)[1].split(
        "### Research-only", 1
    )[0]

    for required in (
        SAFE_IMAGE,
        SAFE_IMAGE_ID,
        SAFE_CUDA_PLACEMENT_SHA256,
        SAFE_SOURCE_RECEIPT_SHA256,
        "scripts/config/glm53-flash-tp4-site.example.yaml",
        "prepare_glm53_dflash7_python_overlay_profile.py",
        "--strict-placeholders",
        "verify-image.json",
        'until curl --fail --silent "${api_endpoint}/health"',
        "docker logs --follow --tail 120",
        "replacement image belongs here only after its own live receipt",
    ):
        assert required in section

    token = "START_GLM53_FLASH_DFLASH7_PYTHON_OVERLAY_FASTSAFETENSORS_TP4"
    assert section.count(f"--confirmation {token} start") == 1
    assert "\ndocker run " not in section
    normalized = " ".join(section.split())
    assert "four hand-maintained `docker run` commands" in normalized
    assert "df4e09a32cdb" not in section

    research = guide.split("### Research-only", 1)[1]
    assert "eabe7fd0c878db7384ef87fe80a1e96b9bedcf67" in research
    assert "df4e09a32cdbf1c0e69cc7c4c9e95d890d6c7a1e3eaac84f969912a16fd27dd3" in research
    assert "is not deployable" in research
    assert "rejected-four-reader-eabe7fd.json" in research

    routing = (ROOT / "docs/GLM53_FLASH_QUICKSTARTS.md").read_text(
        encoding="utf-8"
    )
    assert (
        "GLM53_DFLASH7_PYTHON_OVERLAY_SPARKCACHE_TP4_QUICKSTART.md"
        "#shortest-qualified-start"
    ) in routing
    assert (
        "docs/GLM53_DFLASH7_PYTHON_OVERLAY_SPARKCACHE_TP4_QUICKSTART.md"
        "#shortest-qualified-start"
    ) in (ROOT / "README.md").read_text(encoding="utf-8")
