from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

import sparkring_generic_launcher as launcher
from prepare_glm53_b12x_kda_adaptive_mtp_profile import (
    LEASE_CONTRACT_SHA256,
    MTP_CACHE_IDENTITY_SHA256,
    PUBLICATION_SCHEMA,
    SPARKCACHE_COMMIT,
    SPARKCACHE_SOURCE_SHA256,
    VLLM_COMMIT,
    ResolveError,
    resolve,
)
from sparkring_site import load_site


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts/config"
PROFILE = (
    CONFIG
    / "glm53-flash-b12x-kda-mtp5-adaptive-fastsafetensors-sparkcache-tp4-dcp1.example.json"
)
SITE = CONFIG / "glm53-flash-b12x-kda-adaptive-mtp-tp4-site.example.yaml"
QUICKSTART = ROOT / "docs/GLM53_B12X_KDA_ADAPTIVE_MTP_SPARKCACHE_TP4_QUICKSTART.md"
TARGET = "a35e6bf2875c1875609b8deaec404c07c6cc80259e4222fc0b51e649498bd6b9"


def _argument(profile: dict, option: str) -> str:
    arguments = profile["extra_vllm_args"]
    return arguments[arguments.index(option) + 1]


def _mtp_identity() -> str:
    fields = (
        "glm53-embedded-mtp-runtime-v1",
        TARGET,
        VLLM_COMMIT,
        "5",
        "adaptive:3:32",
    )
    return hashlib.sha256("\0".join(fields).encode()).hexdigest()


def test_profile_pins_adaptive_mtp_fastsafetensors_and_sparkcache() -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    speculative = json.loads(_argument(profile, "--speculative-config"))
    assert speculative == {
        "method": "mtp",
        "num_speculative_tokens": 5,
        "moe_backend": "humming",
        "attention_backend": "B12X",
        "adaptive_speculative_tokens_initial": 3,
        "adaptive_speculative_tokens_window": 32,
    }
    assert _argument(profile, "--load-format") == "fastsafetensors"
    assert profile["environment"]["VLLM_FASTSAFETENSORS_QUEUE_SIZE"] == "1"
    assert profile["identity"]["weight_loader_tp_nogds"] == "true"

    identity = profile["identity"]
    assert identity["vllm_revision"] == VLLM_COMMIT
    assert identity["sparkcache_source_revision"] == SPARKCACHE_COMMIT
    assert identity["sparkcache_source_sha256"] == SPARKCACHE_SOURCE_SHA256
    assert identity["mtp_cache_identity_sha256"] == _mtp_identity()
    assert identity["mtp_cache_identity_sha256"] == MTP_CACHE_IDENTITY_SHA256

    transfer = json.loads(_argument(profile, "--kv-transfer-config"))
    extra = transfer["kv_connector_extra_config"]
    assert extra["spark_cache_draft_checkpoint_sha256"] == MTP_CACHE_IDENTITY_SHA256
    assert extra["spark_cache_native_restore"] is True
    assert extra["spark_cache_native_arena_bytes"] == 256 * 1024**2
    assert extra["spark_cache_native_io_workers"] == 8
    assert extra["spark_cache_load_threads"] == 2
    assert extra["spark_cache_publication_schema"] == PUBLICATION_SCHEMA
    assert extra["spark_cache_clear_once"] == (
        "sparkring-b12x-kda-adaptive-mtp-fastsafetensors-initialization"
    )

    attestation = " ".join(profile["attestation_hook"])
    assert SPARKCACHE_SOURCE_SHA256 in attestation
    assert LEASE_CONTRACT_SHA256 in attestation
    assert (
        "9f64f5041f7f9d953e9f6bc53de8733b3eb4035c0753056a1f646346702a0994"
        in attestation
    )


def test_runtime_bound_identity_does_not_alias_the_e105_adaptive_profile() -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    e105 = json.loads(
        (
            CONFIG
            / "glm53-flash-e10536a-mtp5-adaptive-fastsafetensors-sparkcache-tp4-dcp1.example.json"
        ).read_text(encoding="utf-8")
    )
    assert profile["identity"]["mtp_cache_identity_sha256"] != (
        e105["identity"]["mtp_cache_identity_sha256"]
    )
    assert profile["required_image_labels"]["org.jovian.vllm.commit"] == VLLM_COMMIT
    assert e105["required_image_labels"]["org.jovian.vllm.commit"] == (
        "e10536aadf02a18fccddda7ec939c33147e8b0b3"
    )


def test_resolver_produces_an_aligned_tp4_profile(tmp_path: Path) -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    site = yaml.safe_load(SITE.read_text(encoding="utf-8"))
    profile, site = resolve(
        profile,
        site,
        image="local/glm53-kda-sparkcache@sha256:" + "a" * 64,
        image_id="sha256:" + "b" * 64,
        parent_image="local/glm53-kda-runtime@sha256:" + "c" * 64,
        parent_image_id="sha256:" + "d" * 64,
        native_library_sha256="e" * 64,
    )
    assert site["topology"] and len(site["ranks"]) == 4
    assert site["serving"]["tensor_parallel_size"] == 4
    assert site["serving"]["kv_cache_bytes_per_rank"] == 20 * 1024**3
    assert "REPLACE_WITH" not in json.dumps(profile)

    profile_path = tmp_path / "profile.json"
    site_path = tmp_path / "site.yaml"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    site_path.write_text(yaml.safe_dump(site, sort_keys=False), encoding="utf-8")
    loaded_profile = launcher.load_profile(profile_path)
    loaded_site = load_site(site_path)
    launcher._validate_site_profile_alignment(loaded_site, loaded_profile)
    assert len(launcher.build_actions(loaded_site, loaded_profile, "plan")) == 4


def test_resolver_rejects_runtime_or_loader_identity_drift() -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    site = yaml.safe_load(SITE.read_text(encoding="utf-8"))
    arguments = {
        "image": "image",
        "image_id": "sha256:" + "a" * 64,
        "parent_image": "parent",
        "parent_image_id": "sha256:" + "b" * 64,
        "native_library_sha256": "c" * 64,
    }
    changed = copy.deepcopy(profile)
    changed["identity"]["vllm_revision"] = "0" * 40
    with pytest.raises(ResolveError, match="live-tensor B12X KDA runtime"):
        resolve(changed, copy.deepcopy(site), **arguments)

    changed = copy.deepcopy(profile)
    changed["environment"]["VLLM_FASTSAFETENSORS_QUEUE_SIZE"] = "2"
    with pytest.raises(ResolveError, match="queue size must be one"):
        resolve(changed, copy.deepcopy(site), **arguments)


def test_resolver_rejects_publication_schema_drift() -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    site = yaml.safe_load(SITE.read_text(encoding="utf-8"))
    transfer_index = profile["extra_vllm_args"].index("--kv-transfer-config") + 1
    transfer = json.loads(profile["extra_vllm_args"][transfer_index])
    transfer["kv_connector_extra_config"].pop("spark_cache_publication_schema")
    profile["extra_vllm_args"][transfer_index] = json.dumps(transfer)

    with pytest.raises(ResolveError, match="tail-cow-v1 publication"):
        resolve(
            profile,
            site,
            image="image",
            image_id="sha256:" + "a" * 64,
            parent_image="parent",
            parent_image_id="sha256:" + "b" * 64,
            native_library_sha256="c" * 64,
        )


def test_quickstart_names_the_executable_builder_and_profile_contracts() -> None:
    guide = QUICKSTART.read_text(encoding="utf-8")
    assert "runtime/glm53-flash-b12x-kda-adaptive-mtp/build-image.sh" in guide
    assert str(PROFILE.relative_to(ROOT)).replace("\\", "/") in guide
    assert str(SITE.relative_to(ROOT)).replace("\\", "/") in guide
    assert "prepare_glm53_b12x_kda_adaptive_mtp_profile.py" in guide
    assert SPARKCACHE_COMMIT in guide
    assert SPARKCACHE_SOURCE_SHA256 in guide
    assert VLLM_COMMIT in guide
    assert "START_GLM53_FLASH_MTP5_ADAPTIVE_FASTSAFETENSORS_TP4" in guide
