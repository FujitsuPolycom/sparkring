"""Local GLM selection verifies ancestry and gates isolated SparkCache trials."""
import copy
import json
import subprocess
import sys

import pytest

from runtime.common import cache_candidate, candidate, feature_candidate, glm_launch, glm_source_candidate as glm
from runtime.common import glm_targets, glm_tp4, source_candidate, tp2
from runtime.common.test_feature_candidate import encoded, sha
from runtime.common import test_feature_candidate

from runtime.common.test_source_candidate import inputs as inputs
from runtime.common.test_glm_tp4 import mesh, example
from runtime.images import feature_extension

original_feature_inputs = test_feature_candidate.inputs


@pytest.fixture
def feature_inputs(original_feature_inputs):
    """Extend the complete ancestry fixture with an inherited GLM transport manifest."""
    result = copy.deepcopy(original_feature_inputs)
    base = candidate._read(result["base_bytes"])
    parent = candidate._read(result["parent_bytes"])
    path = "/opt/sparkring/sircl/python/sparkring-overlay-manifest.json"
    for record in (base, parent):
        record["files"][path] = "8" * 64
    result["base_bytes"] = encoded(base)
    contract = cache_candidate.descriptor()
    contract["parent"]["receipt_sha256"] = sha(result["base_bytes"])
    cache_candidate.DESCRIPTOR.write_bytes(encoded(contract))
    parent["cache_extension"]["descriptor_sha256"] = sha(cache_candidate.DESCRIPTOR.read_bytes())
    result["parent_bytes"] = encoded(parent)
    contract = feature_candidate.descriptor()
    contract["parent"]["receipt_sha256"] = sha(result["parent_bytes"])
    feature_candidate.DESCRIPTOR.write_bytes(encoded(contract))
    payload = {name: item["text"].encode() for name, item in contract["assets"].items()}
    result["installed_bytes"] = encoded(feature_extension.expected_receipt(
        parent, contract, feature_candidate.DESCRIPTOR.read_bytes(), result["parent_bytes"],
        payload, b"fixture installer\n"))
    return result


@pytest.fixture
def document(inputs):
    return glm.make_receipt(local_source_extension=source_candidate.IDENTITY, **inputs)


def test_complete_chain_and_local_selection_are_required(document):
    assert glm.validate_receipt(document) == document
    assert document["admission"]["serving_qualified"] is False
    assert document["bundle_manifest_sha256"] == "8" * 64
    for key, value in (("local_source_extension", None), ("image_reference", "ghcr.io/example@sha256:" + "1" * 64)):
        changed = copy.deepcopy(document)
        changed[key] = value
        with pytest.raises(ValueError, match="explicit registered local"):
            glm.validate_receipt(changed)
    changed = copy.deepcopy(document)
    changed["raw_receipts"].pop("base_bytes")
    with pytest.raises(ValueError, match="every raw ancestry"):
        glm.validate_receipt(changed)
    changed = copy.deepcopy(document)
    changed["installed"]["files"][source_candidate.ENTRYPOINT] = "0" * 64
    with pytest.raises(ValueError, match="parsed inventory"):
        glm.validate_receipt(changed)


@pytest.mark.parametrize("profile", ["tp2-dcp1-sparkcache", "tp4-dcp1-sparkcache", "tp4-dcp4-sparkcache", "tp4-switched"])
def test_cache_and_other_profiles_remain_rejected(document, profile):
    with pytest.raises(ValueError, match="cache-disabled|supports only"):
        glm.validate_profile_capabilities(document, profile)


@pytest.mark.parametrize("flag", [0, 1, "true", "false", None, {}, []])
def test_trial_flag_requires_a_boolean(document, flag):
    document["allow_sparkcache_trial"] = flag
    with pytest.raises(ValueError, match="must be a boolean"):
        glm.validate_receipt(document)


def test_trial_contract_requires_opt_in_and_binds_child_lease_and_libraries(document):
    assert document["allow_sparkcache_trial"] is False
    old = copy.deepcopy(document)
    old.pop("allow_sparkcache_trial")
    assert glm.validate_receipt(old)["allow_sparkcache_trial"] is False
    for profile in glm.CACHE_PROFILES:
        with pytest.raises(ValueError, match="cache-disabled"):
            glm.validate_profile_capabilities(old, profile)
    selected = glm.validate_receipt(dict(document, allow_sparkcache_trial=True))
    contract = glm.contract_for_receipt(selected)
    assert set(contract["profiles"]) == glm.PROFILES | glm.CACHE_PROFILES
    native = contract["sparkcache_native"]
    assert native["lease_contract"] == source_candidate.LEASE_CONTRACT
    for stem in ("placement", "snapshot"):
        assert native[stem + "_sha256"] == document["installed"]["files"][native[stem + "_path"]]
    for profile in glm.CACHE_PROFILES:
        glm.validate_profile_capabilities(selected, profile)
    wrong = copy.deepcopy(document["installed"])
    wrong["files"][source_candidate.LEASE_CONTRACT] = "0" * 64
    with pytest.raises(ValueError, match="source-matched lease"):
        glm.profile_contract(wrong, allow_sparkcache_trial=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_trial_selection_is_explicit_and_saved(document, tmp_path, monkeypatch, enabled):
    output = tmp_path / "receipt.json"
    arguments = ["glm-source", "--local-source-extension", source_candidate.IDENTITY,
                 "--image-id", document["image_id"], "--output", str(output)]
    if enabled:
        arguments.append("--allow-sparkcache-trial")
    monkeypatch.setattr(sys, "argv", arguments)
    observed = []
    def observe(image_id, **options):
        observed.append(options)
        return glm.validate_receipt(dict(document, allow_sparkcache_trial=options["allow_sparkcache_trial"]))
    monkeypatch.setattr(glm, "observe", observe)
    glm.main()
    assert observed[0]["allow_sparkcache_trial"] is enabled
    assert json.loads(output.read_text())["allow_sparkcache_trial"] is enabled


def test_contract_keeps_only_three_profiles_and_disables_qwen(document):
    contract = glm.profile_contract(document["installed"])
    assert set(contract["profiles"]) == glm.PROFILES
    assert contract["sparkcache_native"]["lease_contract"] == ""
    assert all(contract["common_environment"][key] == value for key, value in glm.DISABLED.items())
    assert glm_targets.target_for_image(image=document) == glm_targets.target()
    with pytest.raises(ValueError):
        glm_targets.require_image("nvidia-nvfp4", document)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("cache_trial", [False, True])
def test_tp2_plan_binds_source_entrypoint_and_gates_cache(document, tmp_path, monkeypatch, rank, cache_trial):
    document = glm.validate_receipt(dict(document, allow_sparkcache_trial=cache_trial))
    model, cache = tmp_path / "model", tmp_path / "cache"
    model.mkdir()
    cache.mkdir()
    (model / "config.json").write_text("{}")
    site = tmp_path / "rank.env"
    site.write_text("VLLM_HOST_IP=192.0.2.10\nNCCL_SOCKET_IFNAME=eth0\nGLOO_SOCKET_IFNAME=eth0\n")
    arguments = (rank, "192.0.2.10", model, cache, site, document["image_id"], document)
    plan = tp2.render(*arguments, r33_sparkcache=cache_trial)
    assert plan["container_args"][:2] == [glm.ENTRYPOINT, "serve"]
    assert plan["environment"]["SPARKCACHE_ENABLED"] == str(int(cache_trial))
    assert all(plan["environment"][key] == value for key, value in glm.DISABLED.items())
    if cache_trial:
        args = plan["container_args"]
        extra = json.loads(args[args.index("--kv-transfer-config") + 1])["kv_connector_extra_config"]
        assert extra["spark_cache_async_page_capture_lease_contract"] == source_candidate.LEASE_CONTRACT
        namespace = plan["environment"]["SPARKCACHE_CACHE_NAMESPACE"]
        assert namespace == glm.cache_namespace(document["image_id"], "tp2-dcp1-sparkcache") + "-" + glm_targets.target()["revision"][:12]
        assert extra["spark_cache_root"].endswith(namespace)
        assert extra["spark_cache_async_page_capture_library_sha256"] == cache_candidate.descriptor()["native"]["sha256"]
    else:
        assert "--kv-transfer-config" not in plan["container_args"]
    assert plan["model"] == glm_targets.target()
    tp2.validate_runtime_receipt(document, plan)
    if not cache_trial:
        with pytest.raises(ValueError, match="cache-disabled"):
            tp2.render(*arguments, r33_sparkcache=True)
    def reject(*args, **kwargs):
        raise ValueError("source verification failed")
    monkeypatch.setattr(glm, "verify_local_image", reject)
    operations = []
    with pytest.raises(ValueError, match="source verification failed"):
        tp2.execute(plan, "create", document, run=lambda *args, **kwargs: operations.append(args))
    assert operations == []


@pytest.mark.parametrize("profile", ["tp4-dcp1", "tp4-dcp4", "tp4-dcp1-sparkcache", "tp4-dcp4-sparkcache"])
def test_managed_tp4_source_contract_and_container_spec(document, tmp_path, monkeypatch, profile):
    cache_trial = profile.endswith("-sparkcache")
    document = glm.validate_receipt(dict(document, allow_sparkcache_trial=cache_trial))
    image_path = tmp_path / "image.json"
    image_path.write_text(json.dumps(document))
    (tmp_path / "fabric.example.json").write_text(json.dumps(example.topology_example()))
    site_path = tmp_path / "site.json"
    site_path.write_text(json.dumps(dict(example.site_example(), runtime_profile=profile)))
    monkeypatch.setattr(mesh, "verify_bundle", lambda bundle, record: record["bundle_manifest_sha256"])
    output = tmp_path / "launch"
    mesh.render(site_path, tmp_path / "bundle", output, image_path)
    assert "exit 2" in (output / "launch-rank.sh").read_text()
    monkeypatch.syspath_prepend(str(mesh.HERE))
    import importlib
    managed = importlib.import_module("managed_install")
    assert managed.managed_image_attestation(document) == {}
    with pytest.raises(ValueError, match="structured creation plan"):
        managed.canonical_container_spec(output, image_path, 0, {})
    for rank in range(4):
        spec, record, resolved = glm_launch.resolve_spec(output, image_path, rank, owner=mesh)
        assert spec.command[0] == glm.ENTRYPOINT
        assert ("--kv-transfer-config" in spec.command) is cache_trial
        assert spec.environment["SPARKCACHE_ENABLED"] == str(int(cache_trial))
        assert all(spec.environment[key] == value for key, value in glm.DISABLED.items())
        if cache_trial:
            namespace = glm.cache_namespace(document["image_id"], profile) + "-" + glm_targets.target()["revision"][:12]
            assert spec.environment["SPARKCACHE_CACHE_NAMESPACE"] == namespace
            extra = json.loads(spec.command[spec.command.index("--kv-transfer-config") + 1])["kv_connector_extra_config"]
            assert extra["spark_cache_async_page_capture_lease_contract"] == source_candidate.LEASE_CONTRACT
            assert extra["spark_cache_async_page_capture_library_sha256"] == cache_candidate.descriptor()["native"]["sha256"]
            assert extra["spark_cache_root"].endswith(namespace)
            values = dict(resolved["environments"][rank], SPARKCACHE_CACHE_NAMESPACE="parent-cache")
            with pytest.raises(ValueError, match="SPARKCACHE_CACHE_NAMESPACE"):
                glm_tp4.build_spec(values, image_record=record, contract=glm.contract_for_receipt(record))
        values = dict(resolved["environments"][rank], SPARKRING_FEATURES="qwen-prefill")
        with pytest.raises(ValueError, match="SPARKRING_FEATURES"):
            glm_tp4.build_spec(values, image_record=record, contract=glm.contract_for_receipt(record))


@pytest.mark.parametrize("cache_trial", [False, True])
def test_fresh_image_observation_and_tampered_entrypoint(inputs, document, cache_trial):
    document = glm.validate_receipt(dict(document, allow_sparkcache_trial=cache_trial))
    def run(argv, **kwargs):
        if argv[1:3] == ["image", "inspect"]:
            info = {"Id": inputs["image_id"], "Os": "linux", "Architecture": "arm64",
                    "Config": {"Entrypoint": ["/opt/venv/bin/python", glm.ENTRYPOINT]}}
            return subprocess.CompletedProcess(argv, 0, json.dumps([info]))
        if argv[-1] == "verify":
            return subprocess.CompletedProcess(argv, 0, json.dumps(inputs["verification"]))
        raw = {source_candidate.PARENT_RECEIPT: inputs["parent_bytes"],
               feature_candidate.PARENT_RECEIPT: inputs["cache_parent_bytes"]}
        if argv[-1] == "/opt/sparkring/receipts/candidate-installed.json":
            value = inputs["installed_bytes"] if argv[-2] == inputs["image_id"] else inputs["base_bytes"]
        else:
            value = raw[argv[-1]]
        return subprocess.CompletedProcess(argv, 0, value)
    glm.verify_local_image(document, run=run)
    def wrong(argv, **kwargs):
        result = run(argv, **kwargs)
        if argv[1:3] == ["image", "inspect"]:
            value = json.loads(result.stdout)
            value[0]["Config"]["Entrypoint"] = ["/bin/sh"]
            result.stdout = json.dumps(value)
        return result
    with pytest.raises(ValueError, match="entrypoint"):
        glm.verify_local_image(document, run=wrong)
