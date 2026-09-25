"""Native GLM inventory admission refuses altered metadata and lease identities."""
import base64
import copy
import hashlib
import json
from types import SimpleNamespace

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
    publication["transport"] = dict(profile="tp2-rocenante-adaptive-prepared", manifest_sha256="3" * 64)
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


def test_planning_completion_preserves_published_entries_and_matches_runtime(observations):
    source = glm.ROOT / "runtime/releases/shared-2026.09.3/glm-profile-contract.json"
    before = source.read_bytes()
    published = json.loads(before)
    expanded = glm.complete_planning_contract(published)
    runtime = glm.profile_contract(glm.make_receipt(**observations)["installed"])
    for name in ("tp2-dcp1", "tp4-dcp1"):
        assert expanded["profiles"][name] == runtime["profiles"][name]
        assert expanded["profiles"][name]["sparkcache"] is False
    for name, value in published["profiles"].items():
        assert expanded["profiles"][name] == value
    assert source.read_bytes() == before and json.loads(before) == published


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


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("variant", ["nvfp4-spark", "nvfp4-qad"])
def test_tp2_native_plan_selects_verified_entrypoint_loader_and_transport(observations, tmp_path, monkeypatch, cache_enabled, variant):
    from runtime.common import tp2, glm_targets
    record = glm.make_receipt(**observations)
    model, cache = tmp_path / "model", tmp_path / "cache"
    model.mkdir()
    cache.mkdir()
    (model / "config.json").write_bytes(b"config fixture")
    (model / "model.safetensors.index.json").write_bytes(b"index fixture")
    seen = []
    monkeypatch.setattr(glm_targets, "verified_override", lambda *args: seen.append(args))
    env = tmp_path / "rank.env"
    env.write_text("VLLM_HOST_IP=192.0.2.10\nNCCL_SOCKET_IFNAME=eth0\nGLOO_SOCKET_IFNAME=eth0\n")
    plan = tp2.render(0, "192.0.2.10", model, cache, env, record["image_id"], record,
                      r33_sparkcache=cache_enabled, target_model_variant=variant)
    assert seen == [(variant, b"config fixture", b"index fixture")]
    assert plan["container_args"][:2] == [glm.ENTRYPOINT, "serve"]
    assert plan["environment"]["SPARKRING_TRANSPORT_PROFILE"] == "tp2-rocenante-adaptive-prepared"
    assert plan["environment"]["SOURCE_IMAGE_PROFILE"] == ""
    assert plan["labels"]["org.sparkring.memory-guard"] == "false"
    assert plan["memory_guard_floor_bytes"] == 0
    assert plan["model"]["revision"] == glm_targets.target(variant)["revision"]
    assert plan["target_model_variant"] == variant
    assert plan["environment"]["SERVED_MODEL_NAME"].endswith(("QAD" if variant == "nvfp4-qad" else "Spark") + "-TP2")
    assert plan["qualification"]["gpu_qualified"] is False
    assert "reference_context_limit" not in plan["qualification"]
    assert plan["environment"]["LOAD_FORMAT"] == "b12x"
    assert ("--kv-transfer-config" in plan["container_args"]) == cache_enabled
    tp2.validate_runtime_receipt(record, plan)
    observations_seen = []
    monkeypatch.setattr(glm, "verify_local_image", lambda doc, **kwargs: observations_seen.append(doc))
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert argv[0] != "systemctl", "Native profiles must not enable or require an untested memory guard"
        assert argv == ["docker", "ps", "--quiet"] or argv == plan["command"]
        return SimpleNamespace(stdout="")

    tp2.execute(plan, "create", record, run=run)
    assert observations_seen == [record]
    assert calls == [["docker", "ps", "--quiet"], plan["command"]]
    if cache_enabled:
        config = json.loads(plan["container_args"][plan["container_args"].index("--kv-transfer-config") + 1])
        assert config["kv_connector_extra_config"]["spark_cache_async_page_capture_lease_contract"] == record["installed"]["active_contracts"][0]
        assert config["kv_connector_extra_config"]["spark_cache_target_checkpoint_sha256"] == glm_targets.target(variant)["checkpoint_identity"]


@pytest.mark.parametrize("variant", ["nvfp4-spark", "nvfp4-qad"])
@pytest.mark.parametrize("profile", ["tp4-dcp1", "tp4-dcp1-sparkcache"])
def test_managed_tp4_native_render_and_spec(observations, tmp_path, monkeypatch, variant, profile):
    from runtime.common import glm_targets, glm_tp4
    from runtime.common.test_glm_tp4 import mesh, example, module, MESH
    record = glm.make_receipt(**observations)
    image_path = tmp_path / "image.json"
    image_path.write_text(json.dumps(record))
    (tmp_path / "fabric.example.json").write_text(json.dumps(example.topology_example()))
    site = dict(example.site_example(), runtime_profile=profile, target_model_variant=variant)
    site_path = tmp_path / "site.json"
    site_path.write_text(json.dumps(site))
    monkeypatch.setattr(mesh, "verify_bundle", lambda *args: record["bundle_manifest_sha256"])
    seen = []
    monkeypatch.setattr(glm_targets, "verified_override", lambda *args: seen.append(args))
    output = tmp_path / "launch"
    mesh.render(site_path, tmp_path / "bundle", output, image_path)
    assert "structured container plan" in (output / "launch-rank.sh").read_text()
    readiness = module("native_glm_readiness_test", MESH / "wait_managed_ready.py")
    readiness_plan = readiness.load_launch(output)
    assert readiness_plan["timeout_seconds"] == 1800
    assert readiness.readiness_limit(readiness_plan) == 1800
    assert all(target["wrapper"] == glm.ENTRYPOINT and target["runtime_release"] == "native"
               for target in readiness_plan["containers"])
    contract = glm.contract_for_receipt(record)
    for rank in range(4):
        values = mesh.defaults(output / f"rank{rank}.env")
        spec = glm_tp4.build_spec(values, image_record=record, contract=contract,
                                  model_config=b"config fixture", model_index=b"index fixture")
        assert spec.command[:2] == (glm.ENTRYPOINT, "serve")
        assert spec.environment["SOURCE_IMAGE_PROFILE"] == ""
        assert spec.environment["VLLM_PLUGINS"] == "b12x_loader"
        assert spec.environment["SPARKRING_FEATURES"] == ""
        assert spec.environment["SPARK_TP4_GRAPH_DIRECT_DOORBELL"] == "1"
        assert spec.command[spec.command.index("--load-format") + 1] == "b12x"
        assert spec.command[spec.command.index("--kv-cache-memory-bytes") + 1] == "25769803776"
        if profile.endswith("-sparkcache"):
            config = json.loads(spec.command[spec.command.index("--kv-transfer-config") + 1])["kv_connector_extra_config"]
            assert config["spark_cache_async_page_capture_lease_contract"] == record["installed"]["active_contracts"][0]
            assert config["spark_cache_max_bytes"] == 8 * 1024**3
            assert config["spark_cache_async_page_capture_slot_bytes"] == 512 * 1024**2
            assert config["spark_cache_cuda_placement_arena_bytes"] == 64 * 1024**2
            assert "spark_cache_clear_once" not in config
            assert config["spark_cache_target_checkpoint_sha256"] == glm_targets.target(variant)["checkpoint_identity"]
    assert seen == [(variant, b"config fixture", b"index fixture")] * 4
