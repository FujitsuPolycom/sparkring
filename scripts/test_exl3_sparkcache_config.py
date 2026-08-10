import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from exl3_sparkcache_config import (  # noqa: E402
    CACHE_DESTINATION,
    PROFILE_ID,
    SparkCacheProfileError,
    build_candidate,
)
from checkpoint_manifest_generator import (  # noqa: E402
    FileEntry,
    build_receipt,
    compute_identity,
)


BUNDLE = "b" * 64


def checkpoint_receipt():
    receipt = build_receipt(
        [FileEntry("model.safetensors", 7, "a" * 64)],
        artifact_root_name="glm52-exl3",
    )
    receipt["checkpoint_identity_sha256"] = compute_identity(receipt)
    return receipt


TARGET = checkpoint_receipt()["checkpoint_identity_sha256"]


def source_profile() -> dict:
    profile = json.loads(
        (ROOT / "scripts/config/exl3-profile-fixture.json").read_text(
            encoding="utf-8"
        )
    )
    return profile


def build(source=None):
    return build_candidate(
        source_profile() if source is None else source,
        checkpoint_receipt=checkpoint_receipt(),
        connector_bundle_identity=BUNDLE,
        connector_staging_host="/srv/sparkcache/staging",
        cache_root_host="/srv/sparkcache/context",
    )


def test_candidate_changes_only_cache_composition_contract():
    source = source_profile()
    candidate = build(source)
    profile = candidate["profile"]
    assert candidate["execution_supported"] is False
    assert candidate["maturity"] == "offline-validated"
    assert candidate["configuration_status"] == "candidate"
    assert profile["profile_id"] == PROFILE_ID
    for field in (
        "image_id",
        "model_revision",
        "model_manifest_sha256",
        "model_weight_bytes",
    ):
        assert profile[field] == source[field]
    assert profile["environment"]["VLLM_SPARK_DCP_SIZE"] == "4"
    assert profile["environment"]["VLLM_SPARK_MTP_MODE_ID"] == "fixed-mtp2"
    assert profile["environment"]["SPARK_CONTEXT_CACHE_ENABLE"] is None
    assert (
        profile["environment"]["PYTORCH_CUDA_ALLOC_CONF"]
        == "expandable_segments:False"
    )


def test_candidate_emits_colocated_draft_and_first_gate_safety_defaults():
    arguments = build()["profile"]["extra_vllm_args"]
    index = arguments.index("--kv-transfer-config")
    config = json.loads(arguments[index + 1])
    extra = config["kv_connector_extra_config"]
    assert config["kv_connector"] == "SparkContextCacheConnector"
    assert config["kv_load_failure_policy"] == "recompute"
    assert extra["spark_cache_root"] == CACHE_DESTINATION
    assert extra["spark_cache_target_checkpoint_sha256"] == TARGET
    assert extra["spark_cache_draft_policy"] == "colocated_target"
    assert "spark_cache_draft_checkpoint_sha256" not in extra
    assert extra["spark_cache_streaming_snapshots"] is False
    assert extra["spark_cache_native_restore"] is False
    assert arguments.count("--disable-hybrid-kv-cache-manager") == 1


def test_candidate_declares_distinct_rank_local_mount_contract():
    mounts = build()["required_mounts"]
    assert mounts == [
        {
            "source": "/srv/sparkcache/staging",
            "destination": "/opt/sparkcache-staging",
            "read_only": True,
        },
        {
            "source": "/srv/sparkcache/context",
            "destination": "/cache/context",
            "read_only": False,
        },
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_receipt", {"checkpoint_identity_sha256": "A" * 64}),
        ("connector_bundle_identity", "short"),
        ("connector_staging_host", "relative/path"),
        ("cache_root_host", "/"),
        ("cache_root_host", "/srv/../cache"),
        ("cache_root_host", "/srv/cache\rbreak"),
    ],
)
def test_candidate_rejects_unattested_identity_or_unsafe_paths(field, value):
    kwargs = {
        "checkpoint_receipt": checkpoint_receipt(),
        "connector_bundle_identity": BUNDLE,
        "connector_staging_host": "/srv/sparkcache/staging",
        "cache_root_host": "/srv/sparkcache/context",
    }
    kwargs[field] = value
    with pytest.raises(SparkCacheProfileError):
        build_candidate(source_profile(), **kwargs)


def test_candidate_rejects_topology_or_cache_composition_drift():
    source = source_profile()
    source["environment"]["VLLM_SPARK_DCP_SIZE"] = "2"
    with pytest.raises(SparkCacheProfileError, match="contract drift"):
        build(source)

    source = source_profile()
    source["extra_vllm_args"].extend(["--kv-transfer-config", "{}"])
    with pytest.raises(SparkCacheProfileError, match="already contains"):
        build(source)


def test_builder_does_not_mutate_source_document():
    source = source_profile()
    original = copy.deepcopy(source)
    build(source)
    assert source == original


def test_candidate_identity_is_derived_from_complete_receipt():
    candidate = build()
    assert candidate["checkpoint_receipt"] == checkpoint_receipt()
    tampered = checkpoint_receipt()
    tampered["files"][0]["byte_size"] = 8
    with pytest.raises(SparkCacheProfileError, match="does not match inventory"):
        build_candidate(
            source_profile(),
            checkpoint_receipt=tampered,
            connector_bundle_identity=BUNDLE,
            connector_staging_host="/srv/staging",
            cache_root_host="/srv/cache",
        )


def test_cli_refuses_to_overwrite_existing_candidate(tmp_path):
    output = tmp_path / "candidate.json"
    receipt_path = tmp_path / "checkpoint.json"
    receipt_path.write_text(json.dumps(checkpoint_receipt()), encoding="utf-8")
    output.write_text("operator evidence\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/exl3_sparkcache_config.py"),
            "--profile",
            str(ROOT / "scripts/config/exl3-profile-fixture.json"),
            "--checkpoint-receipt",
            str(receipt_path),
            "--connector-bundle-identity",
            BUNDLE,
            "--connector-staging-host",
            "/srv/staging",
            "--cache-root-host",
            "/srv/cache",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "cannot create exclusive output" in result.stderr
    assert output.read_text(encoding="utf-8") == "operator evidence\n"
