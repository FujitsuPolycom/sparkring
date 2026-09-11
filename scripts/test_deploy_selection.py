"""Receipt selection stays identical through private deployment preparation."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.deploy_selection import profile_module, selection, validate_selected_file
from scripts.deploy_suite import PROFILE, create_spec
from scripts.deploy_runtime import build_runtime_plan
from scripts.deploy_engine import plan_digest
from scripts.test_deploy_suite import inventory
from scripts.test_deploy_runtime import prepared


def local_receipt():
    module = profile_module(PROFILE)
    lock = json.loads(module.SOURCE_LOCK.read_text())
    inside = {key: lock["runtime"][key] for key in (
        "bundle_manifest_sha256", "transport_sha256", "marker_binary_sha256",
        "marker_source_sha256", "nccl_sha256", "retained_vllm_native_sha256",
        "readiness_warmup")}
    inside.update(checks_passed=True, source_lock_sha256=module.sha(module.SOURCE_LOCK),
                  cuda_initialized=False, model_loaded=False)
    inside["inherited_runtime"] = lock["runtime"]["expected_distributions"]
    inside["packages"] = {name: {"revision": source["revision"],
                                "file_map_sha256": source["installed_file_map_sha256"],
                                "files": source["installed_file_count"]}
                          for name, source in lock["sources"].items()}
    return {"schema": "sparkring-source-image-receipt/v1", "checks_passed": True,
            "image_id": "sha256:" + "7" * 64, "image_reference": "sha256:" + "7" * 64,
            "platform": "linux/arm64", "profile": "tp4-dcp1-mtp3-prefill",
            "source_lock_sha256": module.sha(module.SOURCE_LOCK),
            "inside_image": inside}


def selected_spec(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(local_receipt()))
    return create_spec(inventory(), "test-mesh", "/srv/sparkring/test-mesh", image_receipt=path)


def test_source_receipt_loads_in_fresh_process_without_verifier_imports(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(local_receipt()), encoding="utf-8")
    script = """
import sys
from pathlib import Path
from scripts.deploy_selection import profile_module
from scripts.deploy_suite import PROFILE
assert 'native_files' not in sys.modules
assert 'archive_utils' not in sys.modules
result = profile_module(PROFILE).load_image_receipt(Path(sys.argv[1]))
assert result['image_id'] == 'sha256:' + '7' * 64
assert 'native_files' not in sys.modules
assert 'archive_utils' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script, str(path)],
                            cwd=Path(__file__).resolve().parents[1],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("field,value", [
    ("image_reference", "ghcr.io/example/runtime@sha256:" + "7" * 64),
    ("source_lock_sha256", "0" * 64), ("profile", "unknown"),
    ("checks_passed", False), ("platform", "linux/amd64"),
])
def test_local_receipt_rejects_identity_substitution(field, value):
    document = local_receipt()
    document[field] = value
    with pytest.raises(ValueError):
        profile_module(PROFILE).validate_image_receipt(document)


@pytest.mark.parametrize("mutation", ["missing-package", "package-count", "native-library", "dependency"])
def test_local_receipt_requires_complete_source_and_native_witnesses(mutation):
    document = local_receipt()
    inside = document["inside_image"]
    if mutation == "missing-package":
        inside["packages"].pop("vllm")
    elif mutation == "package-count":
        inside["packages"]["b12x"]["files"] -= 1
    elif mutation == "native-library":
        inside["nccl_sha256"] = "0" * 64
    else:
        inside["inherited_runtime"]["torch"] = "unsupported"
    with pytest.raises(ValueError):
        profile_module(PROFILE).validate_image_receipt(document)


def test_selected_receipt_is_bound_to_private_spec_and_staged_copy(tmp_path):
    spec = selected_spec(tmp_path)
    chosen = selection(spec, PROFILE)
    assert chosen["local"] and chosen["image_reference"] == chosen["config_image_id"]
    assert "public_reference" not in chosen
    assert "bundle_manifest_sha256" not in chosen["receipt"]
    assert "nccl_path" not in chosen["receipt"]["inside_image"]
    normalized = profile_module(PROFILE).validate_image_receipt(chosen["receipt"])
    assert normalized["bundle_manifest_sha256"] == chosen["inside_image"]["bundle_manifest_sha256"]
    assert spec["site"]["runtime_profile"] == "tp4-dcp1-mtp3-prefill"
    path = tmp_path / "selected-image-receipt.json"
    path.write_text(json.dumps(chosen["receipt"]))
    assert validate_selected_file(spec, tmp_path, PROFILE) == path
    changed = copy.deepcopy(chosen["receipt"])
    changed["inside_image"]["nccl_sha256"] = "0" * 64
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Staged image receipt"):
        validate_selected_file(spec, tmp_path, PROFILE)


@pytest.mark.parametrize("action", ["create", "install", "native-check"])
def test_runtime_plan_uses_selected_receipt_and_local_config_identity(tmp_path, action):
    preparation = prepared()
    preparation["spec"] = selected_spec(tmp_path)
    preparation["network_verification"]["spec_sha256"] = plan_digest(preparation["spec"])
    plan = build_runtime_plan(preparation, action)
    text = json.dumps(plan)
    if action in ("install", "native-check"):
        assert "selected-image-receipt.json" in text
        assert "source/runtime/glm53-spark-mtp3-mesh/image-receipt.json" not in text
    if action == "create":
        assert "sha256:" + "7" * 64 in text
        assert "ghcr.io" not in text


def test_public_performance_receipt_remains_exactly_pinned():
    module = profile_module(PROFILE)
    path = Path(PROFILE) / "performance/public-image.json"
    receipt = json.loads(path.read_text())
    module.validate_image_receipt(receipt)
    receipt["image_id"] = "sha256:" + "7" * 64
    with pytest.raises(ValueError, match="repository pin"):
        module.validate_image_receipt(receipt)


def test_actual_receipt_producer_flows_into_deployment_selection(tmp_path, monkeypatch):
    profile = profile_module(PROFILE)
    source = profile.SOURCE_LOCK.parent
    monkeypatch.syspath_prepend(str(source))
    spec = importlib.util.spec_from_file_location("source_receipt_producer_test", source / "verify_image.py")
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    from archive_utils import make_archive, sha

    inside = local_receipt()["inside_image"]
    image_id = "sha256:" + "7" * 64
    lock = json.loads(profile.SOURCE_LOCK.read_text())
    lock_bytes = profile.SOURCE_LOCK.read_bytes()
    manifest_bytes = json.dumps({
        "source_lock_sha256": sha(lock_bytes),
        "tool_hashes": {name: sha((source / name).read_bytes()) for name in producer.TOOLS},
    }, sort_keys=True).encode()
    closure = {name: (source / name).read_bytes() for name in producer.TOOLS}
    closure.update({"manifest.json": manifest_bytes, "source-lock.json": lock_bytes})
    payload = make_archive({producer.IMAGE_ROOT.lstrip("/") + "/" + name: (data, 0o644)
                            for name, data in closure.items()}, 0)
    context_bytes = make_archive({"Dockerfile": (b"FROM parent\n", 0o644),
                                  "payload.tar": (payload, 0o644)}, 0)
    context = tmp_path / "context"
    context.mkdir()
    (context / "image-context.tar").write_bytes(context_bytes)
    (context / "manifest.json").write_bytes(manifest_bytes)
    (context / "context-receipt.json").write_text(json.dumps({
        "context_sha256": sha(context_bytes), "payload_sha256": sha(payload),
        "manifest_sha256": sha(manifest_bytes),
    }))
    created_id = "a" * 64
    copied = set()
    def output(argv):
        if argv[:3] == ["docker", "image", "inspect"]:
            if argv[-1] == lock["parent"]["image_id"]:
                return json.dumps([{"RootFS": {"Layers": ["parent"]}}]).encode()
            return json.dumps([{"Id": image_id, "Os": "linux", "Architecture": "arm64",
                                "RootFS": {"Layers": ["parent", "source"]}}]).encode()
        if argv[:2] == ["docker", "create"]:
            return created_id.encode()
        if argv[:2] == ["docker", "cp"]:
            name = argv[2].rsplit("/", 1)[1]
            copied.add(name)
            return make_archive({name: (closure[name], 0o644)}, 0)
        assert argv[:2] == ["docker", "run"]
        assert copied == set(closure)
        assert all(flag in argv for flag in ("--mount", "-I", "-S", "-B"))
        return json.dumps(inside).encode()
    monkeypatch.setattr(producer.subprocess, "check_output", output)
    def cleanup(argv, **kwargs):
        assert argv == ["docker", "rm", created_id] and kwargs["check"]
    monkeypatch.setattr(producer.subprocess, "run", cleanup)
    receipt = tmp_path / "producer.json"
    produced = producer.verify(image_id, "tp4-dcp1-mtp3-prefill", profile.SOURCE_LOCK, receipt, context)
    assert "bundle_manifest_sha256" not in produced
    assert "nccl_path" not in produced["inside_image"]
    deployment = create_spec(inventory(), "test-mesh", "/srv/sparkring/test-mesh", image_receipt=receipt)
    selected = selection(deployment, PROFILE)
    assert selected["receipt"] == produced
    staged = tmp_path / "selected-image-receipt.json"
    staged.write_text(json.dumps(produced))
    assert validate_selected_file(deployment, tmp_path, PROFILE) == staged
    assert profile.load_image_receipt(staged)["bundle_manifest_sha256"] == inside["bundle_manifest_sha256"]


@pytest.mark.parametrize("mutation", ["missing-bundle", "missing-nccl", "wrong-path", "outer-bundle"])
def test_producer_receipt_missing_or_conflicting_runtime_identity_is_rejected(mutation):
    receipt = local_receipt()
    if mutation == "missing-bundle":
        receipt["inside_image"].pop("bundle_manifest_sha256")
    elif mutation == "missing-nccl":
        receipt["inside_image"].pop("nccl_sha256")
    elif mutation == "wrong-path":
        receipt["inside_image"]["nccl_path"] = "/unverified/libnccl.so"
    else:
        receipt["bundle_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        profile_module(PROFILE).validate_image_receipt(receipt)


def test_explicit_public_performance_selection_retains_registry_and_raw_receipt(tmp_path):
    path = Path(PROFILE) / "performance/public-image.json"
    raw = json.loads(path.read_text())
    spec = create_spec(inventory(), "test-mesh", "/srv/sparkring/test-mesh", image_receipt=path)
    chosen = selection(spec, PROFILE)
    assert not chosen["local"]
    assert chosen["image_reference"] == raw["image_reference"]
    assert "@sha256:" in chosen["image_reference"]
    assert chosen["config_image_id"] == raw["image_id"]
    assert chosen["receipt"] == raw and "inside_image" not in chosen["receipt"]
    assert "runtime_profile" not in spec["site"]
    staged = tmp_path / "selected-image-receipt.json"
    staged.write_text(json.dumps(raw))
    assert validate_selected_file(spec, tmp_path, PROFILE) == staged
    preparation = prepared()
    preparation["spec"] = spec
    plan = build_runtime_plan(preparation, "install")
    assert "selected-image-receipt.json" in json.dumps(plan)

def r33_public_receipt_path():
    return Path(__file__).resolve().parents[1] / 'runtime/sparkring/jovian-r33/public-image-receipt.json'


@pytest.mark.parametrize('runtime_profile', ['tp4-dcp1', 'tp4-dcp1-sparkcache'])
def test_r33_public_receipt_plans_explicit_profile(tmp_path, runtime_profile):
    spec = create_spec(inventory(), 'r33-ring', '/srv/sparkring/r33-ring',
                       image_receipt=r33_public_receipt_path(), runtime_profile=runtime_profile)
    chosen = selection(spec, PROFILE)
    assert not chosen['local']
    assert chosen['image_reference'].startswith('ghcr.io/fujitsupolycom/sparkring@sha256:')
    assert chosen['config_image_id'] == 'sha256:3c7779ad71dd0d5d6fae4c98e04b94c377429306158c2259fc44635892b8b8e4'
    assert spec['site']['runtime_profile'] == runtime_profile
    assert spec['site']['marker_binary_sha256'] == '2828c07e4255c4962c77425be2c88969e7eb7dd4b1bf9e36485bc705bb5d6d64'
    assert chosen['pins']['canonical_bundle_manifest_sha256'] == chosen['receipt']['bundle_manifest_sha256']
    preparation = prepared()
    preparation['spec'] = spec
    preparation['network_verification']['spec_sha256'] = plan_digest(spec)
    result = build_runtime_plan(preparation, 'create')
    assert chosen['config_image_id'] in json.dumps(result)
    receipt = tmp_path / 'selected-image-receipt.json'
    receipt.write_text(json.dumps(chosen['receipt']))
    assert validate_selected_file(spec, tmp_path, PROFILE) == receipt


@pytest.mark.parametrize('runtime_profile', [None, 'tp2-dcp1', 'tp4-dcp4'])
def test_r33_selection_requires_supported_explicit_profile(runtime_profile):
    with pytest.raises(ValueError, match='explicit TP4 runtime profile'):
        create_spec(inventory(), 'r33-ring', '/srv/sparkring/r33-ring',
                    image_receipt=r33_public_receipt_path(), runtime_profile=runtime_profile)


def test_r33_selection_rejects_mesh_attestation_substitution(tmp_path):
    document = json.loads(r33_public_receipt_path().read_text())
    document['bundle_manifest_sha256'] = '0' * 64
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='mesh manifest'):
        create_spec(inventory(), 'r33-ring', '/srv/sparkring/r33-ring',
                    image_receipt=path, runtime_profile='tp4-dcp1-sparkcache')
