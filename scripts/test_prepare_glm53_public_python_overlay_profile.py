from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

import sparkring_generic_launcher as launcher
from prepare_glm53_public_python_overlay_profile import (
    B12X_COMMIT,
    DFLASH_LOADER_PATCH_SHA256,
    DFLASH_LOADER_POSTIMAGE_SHA256,
    LEASE_CONTRACT_SHA256,
    RECURRENT_BOUNDARY_PATCH_SHA256,
    MTP_CACHE_IDENTITY_SHA256,
    OVERLAY_MANIFEST_SHA256,
    PUBLIC_BASE,
    SPARKCACHE_COMMIT,
    SPARKCACHE_SOURCE_SHA256,
    SPARKCACHE_TREE,
    VLLM_NATIVE_COMMIT,
    VLLM_PYTHON_COMMIT,
    ResolveError,
    composed_mtp_identity,
    resolve,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts/config"
PROFILE = (
    CONFIG
    / "glm53-flash-public-python-overlay-mtp5-adaptive-fastsafetensors-sparkcache-tp4-dcp1.example.json"
)
SITE = CONFIG / "glm53-flash-b12x-kda-adaptive-mtp-tp4-site.example.yaml"
QUICKSTART = ROOT / "docs/GLM53_B12X_KDA_ADAPTIVE_MTP_SPARKCACHE_TP4_QUICKSTART.md"
IMAGE_ID = "sha256:" + "ab" * 32
CUDA_PLACEMENT_LIBRARY = "1a" * 32
NATIVE_ELF = "2b" * 32
NATIVE_DISPATCH = "3c" * 32
SOURCE_RECEIPT = "4d" * 32


def _argument(profile: dict, option: str) -> str:
    arguments = profile["extra_vllm_args"]
    return arguments[arguments.index(option) + 1]


def _resolved() -> tuple[dict, dict]:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    site = yaml.safe_load(SITE.read_text(encoding="utf-8"))
    profile["model_host_path"] = "/srv/models/glm53"
    profile["extra_volumes"][0]["host"] = "/srv/cache/glm53-public-python-overlay"
    management = ["10.20.0.10", "10.20.0.11", "10.20.0.12", "10.20.0.13"]
    edge_networks = ["10.30.1.0/24", "10.30.2.0/24", "10.30.3.0/24", "10.30.4.0/24"]
    ring_addresses = (
        ("10.30.1.10", "10.30.4.10"),
        ("10.30.1.11", "10.30.2.11"),
        ("10.30.2.12", "10.30.3.12"),
        ("10.30.3.13", "10.30.4.13"),
    )
    for edge, subnet in zip(site["topology"]["edges"], edge_networks, strict=True):
        edge["subnet"] = subnet
    for rank, address, ports in zip(
        site["ranks"], management, ring_addresses, strict=True
    ):
        rank["ssh_target"] = f"operator@{address}"
        rank["management"]["address"] = address
        for port, ring_address in zip(rank["ring_ports"], ports, strict=True):
            port["address"] = ring_address
        for peer in rank["transport_peers"]:
            peer["address"] = management[peer["rank"]]
    return resolve(
        profile,
        site,
        image="local/glm53-public-python-overlay@sha256:" + "a" * 64,
        image_id=IMAGE_ID,
        cuda_placement_library_sha256=CUDA_PLACEMENT_LIBRARY,
        native_elf_manifest_sha256=NATIVE_ELF,
        native_dispatch_manifest_sha256=NATIVE_DISPATCH,
        source_receipt_sha256=SOURCE_RECEIPT,
    )


def test_profile_uses_distinct_composed_runtime_and_cache_identities() -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    identity = profile["identity"]
    assert identity["vllm_native_revision"] == VLLM_NATIVE_COMMIT
    assert identity["vllm_python_revision"] == VLLM_PYTHON_COMMIT
    assert identity["b12x_revision"] == B12X_COMMIT
    assert identity["vllm_python_overlay_manifest_sha256"] == (
        OVERLAY_MANIFEST_SHA256
    )
    assert identity["mtp_cache_identity_sha256"] == composed_mtp_identity()
    assert identity["mtp_cache_identity_sha256"] == MTP_CACHE_IDENTITY_SHA256
    assert "python-overlay" in _argument(profile, "--served-model-name")
    assert "public-python-overlay" in profile["container_name"]
    assert "py-0b67266-native-da4d7be" in profile["environment"]["VLLM_CACHE_ROOT"]
    assert "b1d541f9" in profile["environment"]["B12X_CUTE_COMPILE_CACHE_DIR"]


def test_profile_selects_opaque_page_tail_copy_on_write() -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    transfer = json.loads(_argument(profile, "--kv-transfer-config"))
    extra = transfer["kv_connector_extra_config"]
    assert extra["spark_cache_cuda_restore"] is True
    assert extra["spark_cache_cuda_placement_arena_bytes"] == 256 * 1024**2
    assert extra["spark_cache_cuda_restore_io_workers"] == 8
    assert not any(key.startswith("spark_cache_native_") for key in extra)
    assert extra["spark_cache_publication_schema"] == "tail-cow-v1"
    assert "tail-cow" in extra["spark_cache_root"]
    assert "tail-cow" in extra["spark_cache_clear_once"]
    assert profile["identity"]["sparkcache_effective_publication_schema"] == (
        "page-tail-cow-v1"
    )
    assert profile["extra_labels"]["org.sparkcache.publication-schema"] == (
        "tail-cow-v1"
    )
    attestation = " ".join(profile["attestation_hook"])
    assert '"tail-cow-v1"' in attestation
    assert "/opt/sparkring/runtime/python-overlay/sparkcache-source-tree.sha256" in (
        attestation
    )
    assert "source_tree_sha256(" not in attestation
    assert "/opt/sparkcache-source-identity.py" not in attestation


def test_resolver_requires_mixed_provenance_and_all_artifact_hashes() -> None:
    profile, _ = _resolved()
    labels = profile["required_image_labels"]
    assert labels["org.jovian.vllm.commit"] == VLLM_NATIVE_COMMIT
    assert labels["org.sparkring.vllm.python.commit"] == VLLM_PYTHON_COMMIT
    assert labels["org.jovian.b12x.commit"] == B12X_COMMIT
    assert labels["org.opencontainers.image.base.name"] == PUBLIC_BASE
    assert labels["org.sparkring.vllm.python-overlay-manifest-sha256"] == (
        OVERLAY_MANIFEST_SHA256
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
    assert labels["org.sparkring.vllm.native-elf-manifest-sha256"] == NATIVE_ELF
    assert labels["org.sparkring.vllm.native-dispatch-manifest-sha256"] == (
        NATIVE_DISPATCH
    )
    assert labels["org.sparkcache.cuda-placement-library-sha256"] == (
        CUDA_PLACEMENT_LIBRARY
    )
    assert "org.sparkcache.native-library-sha256" not in labels
    assert labels["org.sparkcache.source-tree"] == (
        "ab6e25fd1126405a94ce8735a6261f9dd08c0b5f"
    )
    assert labels["org.sparkcache.vllm-contract-sha256"] == LEASE_CONTRACT_SHA256
    assert labels["org.sparkring.source-receipt-sha256"] == SOURCE_RECEIPT
    assert labels["org.jovian.vllm.commit"] != labels[
        "org.sparkring.vllm.python.commit"
    ]


def test_resolver_rejects_snapshot_publication_or_source_built_labels() -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    site = yaml.safe_load(SITE.read_text(encoding="utf-8"))
    arguments = {
        "image": "image",
        "image_id": "sha256:" + "a" * 64,
        "cuda_placement_library_sha256": "b" * 64,
        "native_elf_manifest_sha256": "c" * 64,
        "native_dispatch_manifest_sha256": "d" * 64,
        "source_receipt_sha256": "e" * 64,
    }
    changed = copy.deepcopy(profile)
    transfer = json.loads(_argument(changed, "--kv-transfer-config"))
    transfer["kv_connector_extra_config"]["spark_cache_publication_schema"] = (
        "snapshot-v1"
    )
    arguments_list = changed["extra_vllm_args"]
    index = arguments_list.index("--kv-transfer-config") + 1
    arguments_list[index] = json.dumps(transfer, separators=(",", ":"))
    with pytest.raises(ResolveError, match="tail-cow-v1"):
        resolve(changed, copy.deepcopy(site), **arguments)

    changed = copy.deepcopy(profile)
    changed["required_image_labels"]["org.jovian.vllm.commit"] = VLLM_PYTHON_COMMIT
    with pytest.raises(ResolveError, match="org.jovian.vllm.commit"):
        resolve(changed, copy.deepcopy(site), **arguments)

    changed = copy.deepcopy(profile)
    changed["required_image_labels"].pop(
        "org.sparkring.vllm.recurrent-boundary-patch-sha256"
    )
    with pytest.raises(ResolveError, match="recurrent-boundary-patch-sha256"):
        resolve(changed, copy.deepcopy(site), **arguments)

    changed = copy.deepcopy(profile)
    transfer = json.loads(_argument(changed, "--kv-transfer-config"))
    transfer["kv_connector_extra_config"].pop(
        "spark_cache_cuda_restore_io_workers"
    )
    changed_args = changed["extra_vllm_args"]
    changed_args[changed_args.index("--kv-transfer-config") + 1] = json.dumps(
        transfer, separators=(",", ":")
    )
    with pytest.raises(ResolveError, match="omits canonical SparkCache CUDA keys"):
        resolve(changed, copy.deepcopy(site), **arguments)

    changed = copy.deepcopy(profile)
    changed["attestation_hook"][2] += " && source_tree_sha256("
    with pytest.raises(ResolveError, match="clean SparkCache source receipt"):
        resolve(changed, copy.deepcopy(site), **arguments)


def test_resolver_normalizes_legacy_placement_label_and_digest_alias() -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    site = yaml.safe_load(SITE.read_text(encoding="utf-8"))
    labels = profile["required_image_labels"]
    labels["org.sparkcache.native-library-sha256"] = labels.pop(
        "org.sparkcache.cuda-placement-library-sha256"
    )

    resolved, _ = resolve(
        profile,
        site,
        image="image",
        image_id="sha256:" + "a" * 64,
        native_library_sha256="b" * 64,
        native_elf_manifest_sha256="c" * 64,
        native_dispatch_manifest_sha256="d" * 64,
        source_receipt_sha256="e" * 64,
    )

    resolved_labels = resolved["required_image_labels"]
    assert resolved_labels["org.sparkcache.cuda-placement-library-sha256"] == (
        "b" * 64
    )
    assert "org.sparkcache.native-library-sha256" not in resolved_labels


def test_resolver_rejects_conflicting_placement_label_aliases() -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    site = yaml.safe_load(SITE.read_text(encoding="utf-8"))
    profile["required_image_labels"][
        "org.sparkcache.native-library-sha256"
    ] = "0" * 64

    with pytest.raises(ResolveError, match="conflicting values"):
        resolve(
            profile,
            site,
            image="image",
            image_id="sha256:" + "a" * 64,
            cuda_placement_library_sha256="b" * 64,
            native_elf_manifest_sha256="c" * 64,
            native_dispatch_manifest_sha256="d" * 64,
            source_receipt_sha256="e" * 64,
        )


def test_generic_launcher_builds_a_four_rank_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile, site = _resolved()
    profile_path = tmp_path / "profile.json"
    site_path = tmp_path / "site.yaml"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    site_path.write_text(yaml.safe_dump(site, sort_keys=False), encoding="utf-8")
    assert launcher.main(
        ["--site", str(site_path), "--profile", str(profile_path), "plan"]
    ) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["command"] == "plan"
    assert len(plan["actions"]) == 4
    assert {action["rank"] for action in plan["actions"]} == {0, 1, 2, 3}
    rendered = json.dumps(plan)
    assert profile["container_name"] in rendered
    assert profile["image"] in rendered


def test_quickstart_uses_the_recurrent_capable_python_overlay() -> None:
    guide = QUICKSTART.read_text(encoding="utf-8")
    assert "runtime/glm53-flash-adaptive-mtp-python-overlay/build-image.sh" in guide
    assert str(PROFILE.relative_to(ROOT)).replace("\\", "/") in guide
    assert "prepare_glm53_public_python_overlay_profile.py" in guide
    assert SPARKCACHE_COMMIT in guide
    assert SPARKCACHE_TREE in guide
    assert SPARKCACHE_SOURCE_SHA256 in guide
    assert RECURRENT_BOUNDARY_PATCH_SHA256 in guide
    assert "runtime/glm53-flash-b12x-kda-adaptive-mtp/build-image.sh" not in guide
