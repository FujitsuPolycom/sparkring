"""Native GLM inventory admission refuses altered metadata and lease identities."""
import base64
import copy
import hashlib
import json

import pytest

from runtime.common import glm_native_candidate as glm
from runtime.common import native_candidate as native
from runtime.common.test_native_candidate import fixture as native_fixture


@pytest.fixture
def observations(monkeypatch):
    publication, image, inspection, original, verified = native_fixture()
    installed = json.loads(original)
    lease = "/opt/sparkring/contracts/vllm-connector-jobs-source-" + "1" * 16 + ".json"
    installed["active_contracts"] = [lease]
    installed["files"].update({path: "2" * 64 for path in (
        glm.MODEL, glm.MANIFEST, glm.SIRCL, glm.NCCL, lease,
        "/opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so",
        "/opt/sparkring/sparkcache/lib/libspark_cache_placement.so")})
    raw = json.dumps(installed).encode()
    publication["installed_receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    verified.update(receipt_sha256=publication["installed_receipt_sha256"], files_verified=len(installed["files"]))
    monkeypatch.setattr(native, "publication", lambda *args, **kwargs: publication)
    return dict(release=publication["release"], image_id=image, inspection=inspection,
                installed_bytes=raw, verification=verified)


def test_native_glm_receipt_is_reconstructed_from_authenticated_bytes(observations):
    record = glm.make_receipt(**observations)
    assert glm.validate_receipt(record) == record
    assert record["serving_qualified"] is False
    assert record["installed"]["files"][glm.SIRCL] == "2" * 64
    assert "/owned" not in record["installed"]["files"]


@pytest.mark.parametrize("field,value", [
    ("platform", "linux/amd64"), ("image_reference", "unregistered:tag"),
    ("bundle_manifest_sha256", "0" * 64), ("serving_qualified", True),
    ("raw_installed", "invalid-base64"),
])
def test_changed_host_fields_are_rejected(observations, field, value):
    record = glm.make_receipt(**observations)
    record[field] = value
    with pytest.raises(ValueError):
        glm.validate_receipt(record)


def test_compact_view_cannot_override_verified_library_hashes(observations):
    record = glm.make_receipt(**observations)
    record["installed"]["files"][glm.SIRCL] = "0" * 64
    with pytest.raises(ValueError, match="authenticated inventory"):
        glm.validate_receipt(record)


def test_changed_raw_receipt_is_not_admitted_by_its_claimed_view(observations):
    record = glm.make_receipt(**observations)
    record["raw_installed"] = base64.b64encode(observations["installed_bytes"] + b" ").decode()
    with pytest.raises(ValueError, match="publication"):
        glm.validate_receipt(record)


@pytest.mark.parametrize("field,value", [
    ("active_contracts", []), ("active_contracts", ["/tmp/unbound.json"]),
    ("files", {}),
])
def test_missing_glm_capability_inventory_is_rejected(observations, field, value):
    installed = json.loads(observations["installed_bytes"])
    installed[field] = value
    with pytest.raises(ValueError):
        glm._view(installed, observations["release"])


def test_unknown_receipt_fields_do_not_expand_admission(observations):
    record = copy.deepcopy(glm.make_receipt(**observations))
    record["allow_unverified"] = True
    with pytest.raises(ValueError):
        glm.validate_receipt(record)


@pytest.mark.parametrize("path", [glm.MODEL, glm.MANIFEST, glm.SIRCL, glm.NCCL])
def test_required_runtime_artifacts_cannot_be_omitted(observations, path):
    installed = json.loads(observations["installed_bytes"])
    del installed["files"][path]
    with pytest.raises(ValueError, match="missing"):
        glm._view(installed, observations["release"])


def test_profile_settings_use_native_bindings_without_retained_provenance(observations):
    record = glm.make_receipt(**observations)
    frozen = glm.ROOT / "runtime/sparkring/jovian-r33/profiles/profile-contract.json"
    before = frozen.read_bytes()
    contract = glm.contract_for_receipt(record)
    assert set(contract["profiles"]) == glm.PROFILES
    assert contract["sparkcache_native"]["lease_contract"] == record["installed"]["active_contracts"][0]
    assert contract["sparkcache_native"]["snapshot_sha256"] == "2" * 64
    assert "source_commit" not in contract["sparkcache_native"]
    assert "revision" not in contract["model"]
    assert contract["model"]["loader"]["load_format"] == "b12x"
    assert contract["common_environment"]["VLLM_SPARK_SHARED_CAPTURE_STREAM"] == "1"
    assert all(not profile["memory_guard_required"] for profile in contract["profiles"].values())
    assert contract["profiles"]["tp2-dcp1-sparkcache"]["kv_cache_memory_bytes"] == 8053063680
    assert contract["profiles"]["tp4-dcp1-sparkcache"]["kv_cache_memory_bytes"] == 25769803776
    assert frozen.read_bytes() == before


def test_unconfigured_topology_is_not_inferred_from_image_admission(observations):
    record = glm.make_receipt(**observations)
    glm.validate_profile_capabilities(record, "tp4-dcp1-sparkcache")
    with pytest.raises(ValueError, match="TP2/DCP1 and TP4/DCP1"):
        glm.validate_profile_capabilities(record, "tp4-dcp4-sparkcache")


@pytest.mark.parametrize("profile", sorted(glm.PROFILES))
@pytest.mark.parametrize("variant", ["nvfp4-spark", "nvfp4-qad"])
def test_native_arguments_preserve_profile_resources_and_select_draft_backend(observations, profile, variant):
    contract = glm.contract_for_receipt(glm.make_receipt(**observations))
    original = ["serve", "/models/target", "--load-format", "safetensors", "--mamba-block-size", "512",
                "--cp-kv-cache-interleave-size", "auto", "--cudagraph-metrics", "--async-scheduling",
                "--max-cudagraph-capture-size", "64", "--model-loader-extra-config", '{"allocation":"managed"}']
    before = list(original)
    args = glm.adapt_arguments(original, contract, profile, variant)
    assert original == before
    selected = contract["profiles"][profile]
    for flag, expected in (("--max-model-len", 1048576), ("--kv-cache-memory-bytes", selected["kv_cache_memory_bytes"]),
                           ("--max-num-seqs", selected["serving"]["max_num_seqs"]), ("--max-num-batched-tokens", 8192),
                           ("--load-format", "b12x"), ("--max-parallel-prefills", 1)):
        assert args.count(flag) == 1 and args[args.index(flag) + 1] == str(expected)
    flags = ("--mamba-block-size", "--max-cudagraph-capture-size", "--async-scheduling", "--cudagraph-metrics")
    assert all((flag in args) == (selected["node_count"] == 4) for flag in flags)
    speculation = json.loads(args[args.index("--speculative-config") + 1])
    assert speculation["num_speculative_tokens"] == 3
    assert speculation.get("moe_backend") == ("humming" if selected["node_count"] == 2 or variant == "nvfp4-qad" else None)
    if selected["node_count"] == 2:
        assert speculation["draft_load_config"]["load_format"] == "b12x"
    else:
        assert "draft_load_config" not in speculation and speculation["draft_tensor_parallel_size"] == 4
    graph = json.loads(args[args.index("--compilation-config") + 1])
    assert graph["cudagraph_capture_sizes"] == selected["cudagraph_capture_sizes"]
    if selected["node_count"] == 2:
        assert graph["mode"] == 0
    else:
        assert graph["custom_ops"] == ["all"] and graph["pass_config"] == {"fuse_allreduce_rms": False}


@pytest.mark.parametrize("args", [["serve", "--load-format"], ["serve", "--load-format", "a", "--load-format", "b"]])
def test_ambiguous_argument_translation_is_rejected(observations, args):
    contract = glm.contract_for_receipt(glm.make_receipt(**observations))
    with pytest.raises(ValueError):
        glm.adapt_arguments(args, contract, "tp2-dcp1-sparkcache")
