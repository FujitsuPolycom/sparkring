"""Image switches bind receipts and preserve model, mesh and lifecycle ownership."""
import copy
import hashlib
import json
import subprocess
from types import SimpleNamespace
import zipfile

import pytest

from runtime.common import installer, installer_image, setup
from runtime.common.test_installer import site
from runtime.host import controller

PROFILE = "qwen38-flash-next-qad-tp4"


def image_lock(profile=PROFILE):
    return {"schema": "sparkring-installer-image/v1", "name": "qwen-cuda1342-status03",
            "profile": profile, "image_id": "sha256:" + "a" * 64, "image_reference": "sha256:" + "a" * 64,
            "parent_receipt_sha256": "b" * 64, "toolchain_receipt_sha256": "c" * 64,
            "composition_sha256": "d" * 64, "transport_profile": "tp2-rocenante-adaptive-prepared",
            "transport_manifest_sha256": "e" * 64, "status_version": "0.3.0"}


def ring_site():
    value = site(4)
    for row in value["hosts"]:
        row["fabric"] = {"site_path": "/etc/sparkring/site.json", "site_sha256": "1" * 64, "plan_sha256": "2" * 64}
    return value


def test_image_selection_preserves_profile_weights_network_and_model_arguments(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Offline render contacted a host"))
    original = setup.selection(PROFILE)
    baseline = installer.make_lock(PROFILE, ring_site(), "1" * 40, "2" * 64)
    selected = installer.make_lock(PROFILE, ring_site(), "1" * 40, "2" * 64, image_runtime=image_lock())
    assert installer.validate(selected) == selected
    assert setup.selection(PROFILE) == original
    assert selected["selection"]["profile_release"] == original["release"]
    assert selected["site"] == baseline["site"] and selected["id"] != baseline["id"]
    for before, after in zip(installer.specifications(baseline), installer.specifications(selected), strict=True):
        assert after.command == before.command[1:]
        assert after.entrypoint == installer_image.ENTRYPOINT
        assert after.mounts[:-1] == before.mounts and after.devices == before.devices
        assert after.mounts[-1].target == installer_image.BINDING_TARGET and after.mounts[-1].read_only
        assert after.environment["SPARKRING_RUNTIME_BINDING"] == installer_image.BINDING_TARGET
        assert after.environment["B12X_ROCE_HCA"] == before.environment["B12X_ROCE_HCA"]
        assert after.environment["B12X_ROCE_PEER_HCA_MAP"] == before.environment["B12X_ROCE_PEER_HCA_MAP"]
        assert after.environment["VLLM_PLUGINS"] == "b12x_loader,sparkring_status"
        assert after.environment["VLLM_QWEN3_8_FLASH_NEXT_HC_TP"] == "0"
        assert after.environment["VLLM_QWEN3_8_HC_PREFILL_MODE"] == "shard"
        assert "PYTHONPATH" not in after.environment and "LD_PRELOAD" not in after.environment
        assert "a" * 12 in after.environment["B12X_COMPILE_CACHE_DIR"]
        assert before.image_id[7:19] not in after.environment["B12X_COMPILE_CACHE_DIR"]
    rendered = installer.rendered(selected)
    assert len(rendered) == 8
    assert "toolchain/toolchain.py" in rendered["rank0/compose.yaml"]
    assert "python3" in rendered["rank0/compose.yaml"]
    assert "Development" in selected["selection"]["evidence_scope"]


@pytest.mark.parametrize("change", ["tag", "digest", "profile", "unknown", "version"])
def test_ambiguous_or_unsupported_image_inputs_are_rejected(change):
    value = image_lock()
    if change == "tag":
        value["image_reference"] = "example/image:latest"
    elif change == "digest":
        value["toolchain_receipt_sha256"] = "abc"
    elif change == "profile":
        value["profile"] = "qwen38-flash-next-qad-tp4-sparkcache"
    elif change == "unknown":
        value["environment"] = {"PYTHONPATH": "/unverified"}
    else:
        value["status_version"] = "future"
    with pytest.raises(ValueError):
        installer.make_lock(PROFILE, ring_site(), "1" * 40, "2" * 64, image_runtime=value)


def test_replaced_image_or_receipt_invalidates_saved_deployment():
    lock = installer.make_lock(PROFILE, ring_site(), "1" * 40, "2" * 64, image_runtime=image_lock())
    for field in ("image_id", "parent_receipt_sha256", "toolchain_receipt_sha256"):
        changed = copy.deepcopy(lock)
        changed["image_runtime"][field] = ("sha256:" if field == "image_id" else "") + "f" * 64
        with pytest.raises(ValueError):
            installer.validate(changed)


def admission_fixture():
    value = image_lock()
    parent = {"schema": "sparkring-external-installed/v1", "composition_sha256": value["composition_sha256"],
              "capabilities": {"transport_profile": value["transport_profile"],
                               "transport_manifest_sha256": value["transport_manifest_sha256"],
                               "runtime_status": {"version": "0.3.0"}, "features": ["qwen-collectives", "qwen4-prefill"],
                               "hc_supported_modes": {"4": [{"projection_tp": "0", "prefill_row_ownership": "shard"}]}}}
    raw_parent = json.dumps(parent).encode()
    value["parent_receipt_sha256"] = hashlib.sha256(raw_parent).hexdigest()
    toolchain = {"schema": "sparkring-toolchain-installed/v1", "variant": "combined",
                 "parent_receipt_sha256": value["parent_receipt_sha256"], "nccl_version": 23203, "nvcc": "V13.4.92"}
    raw_toolchain = json.dumps(toolchain).encode()
    value["toolchain_receipt_sha256"] = hashlib.sha256(raw_toolchain).hexdigest()
    image = {"Id": value["image_id"], "Os": "linux", "Architecture": "arm64", "Config": {"Entrypoint": list(installer_image.ENTRYPOINT)}}
    return value, image, {installer_image.PARENT_RECEIPT: raw_parent, installer_image.TOOLCHAIN_RECEIPT: raw_toolchain}


@pytest.mark.parametrize("change", [None, "architecture", "entrypoint", "parent", "toolchain", "installed-tree"])
def test_admission_verifies_receipts_and_installed_tree_without_gpu_or_network(change):
    value, image, receipts = admission_fixture()
    if change == "architecture":
        image["Architecture"] = "amd64"
    elif change == "entrypoint":
        image["Config"]["Entrypoint"] = ["python3", "/unverified.py"]
    elif change in ("parent", "toolchain"):
        path = installer_image.PARENT_RECEIPT if change == "parent" else installer_image.TOOLCHAIN_RECEIPT
        receipts[path] += b" "
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if argv[1] == "image":
            return SimpleNamespace(stdout=json.dumps([image]))
        assert "--gpus" not in argv
        assert argv[argv.index("--runtime") + 1] == "runc"
        assert argv[argv.index("--network") + 1] == "none" and "--read-only" in argv
        if argv[-1] in receipts:
            assert kwargs["text"] is False
            return SimpleNamespace(stdout=receipts[argv[-1]])
        assert argv[-2:] == [installer_image.ENTRYPOINT[1], "verify"]
        if change == "installed-tree":
            raise subprocess.CalledProcessError(1, argv)
        return SimpleNamespace(stdout="verified")
    if change:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            installer_image.admit(value, run=run)
    else:
        result = installer_image.admit(value, run=run)
        assert result["image_id"] == value["image_id"] and result["serving_qualified"] is False
        assert len(calls) == 4


def test_qwen_recipe_follows_the_profile_environment():
    assert installer_image.qwen_recipe({}) == ({"projection_tp": "1", "prefill_row_ownership": "off"}, set())
    shard = {"VLLM_QWEN3_8_HC_PREFILL_MODE": "shard", "SPARKRING_FEATURES": "qwen-collectives, qwen4-prefill"}
    assert installer_image.qwen_recipe(shard) == ({"projection_tp": "0", "prefill_row_ownership": "shard"},
                                                  {"qwen-collectives", "qwen4-prefill"})


@pytest.mark.parametrize("supported", [False, True])
def test_tp2_row_sharding_requires_an_image_that_declares_it(supported):
    value, image, receipts = admission_fixture()
    value["profile"] = "qwen38-flash-next-tp2"
    parent = json.loads(receipts[installer_image.PARENT_RECEIPT])
    modes = [{"projection_tp": "1", "prefill_row_ownership": "off"}]
    if supported:
        modes.append({"projection_tp": "0", "prefill_row_ownership": "shard"})
    parent["capabilities"]["hc_supported_modes"]["2"] = modes
    receipts[installer_image.PARENT_RECEIPT] = json.dumps(parent).encode()
    value["parent_receipt_sha256"] = hashlib.sha256(receipts[installer_image.PARENT_RECEIPT]).hexdigest()
    toolchain = json.loads(receipts[installer_image.TOOLCHAIN_RECEIPT])
    toolchain["parent_receipt_sha256"] = value["parent_receipt_sha256"]
    receipts[installer_image.TOOLCHAIN_RECEIPT] = json.dumps(toolchain).encode()
    value["toolchain_receipt_sha256"] = hashlib.sha256(receipts[installer_image.TOOLCHAIN_RECEIPT]).hexdigest()
    def run(argv, **kwargs):
        if argv[1] == "image":
            return SimpleNamespace(stdout=json.dumps([image]))
        if argv[-1] in receipts:
            return SimpleNamespace(stdout=receipts[argv[-1]])
        return SimpleNamespace(stdout="verified")
    environment = {"VLLM_QWEN3_8_HC_PREFILL_MODE": "shard", "SPARKRING_FEATURES": "qwen-collectives,qwen4-prefill"}
    admit = lambda: installer_image.admit(value, run=run, profile="qwen38-flash-next-tp2", nodes=2, environment=environment)
    if supported:
        assert admit()["image_id"] == value["image_id"]
    else:
        with pytest.raises(ValueError, match="Qwen topology"):
            admit()


def test_share_export_preserves_image_lock_without_private_registry(tmp_path):
    value = image_lock()
    value["image_reference"] = "registry.private.example:5000/rehearsal@sha256:" + "f" * 64
    bundle = b"offline fixture"
    lock = installer.make_lock(PROFILE, ring_site(), "1" * 40, hashlib.sha256(bundle).hexdigest(), image_runtime=value)
    directory = tmp_path / "deployment"
    installer.write(directory / "deployment.lock.json", lock)
    (directory / "source.bundle").write_bytes(bundle)
    output = tmp_path / "share.zip"
    installer.export(directory, output, share=True)
    with zipfile.ZipFile(output) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    assert b"registry.private.example" not in b"".join(files.values())
    exported = json.loads(files["image-lock.json"])
    assert exported["image_id"] == value["image_id"]
    assert exported["parent_receipt_sha256"] == value["parent_receipt_sha256"]
    assert b"--image-lock image-lock.json" in files["README.txt"]
    assert b"toolchain/toolchain.py" in files["rank0/compose.yaml"]


def test_controller_allows_preview_while_another_deployment_is_running(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "STATE", tmp_path)
    installer.write(tmp_path / "cluster.json", {"plan": {"nodes": [0, 1, 2, 3]}})
    installer.write(tmp_path / "active.json", {"path": str(tmp_path / "baseline")})
    (tmp_path / "deployments" / (PROFILE + "-candidate")).mkdir(parents=True)
    # Every installer profile runs on the shared image, so saved deployments carry its lock.
    monkeypatch.setattr(controller.installer, "load", lambda _: {"site": {"ranks": []}, "image_runtime": installer_image.default_lock()})
    monkeypatch.setattr(controller.installer, "apply", lambda *a, **k: {"profile": PROFILE, "hosts": [], "phases": []})
    monkeypatch.setattr(controller.installer, "status", lambda _: {"state": {"operation": "up", "complete": True}})
    assert controller.lifecycle(["up", PROFILE, "--instance", "candidate", "--plan"]) == 0
    with pytest.raises(ValueError, match="sparkring down"):
        controller.lifecycle(["up", PROFILE, "--instance", "candidate", "--execute"])
    assert installer.read(tmp_path / "active.json")["path"] == str(tmp_path / "baseline")


SHARED = ("glm53-flash-nvfp4-spark-tp2", "glm53-flash-nvfp4-spark-tp4", "mimo-v26-flash-rl-tp2",
          "mimo-v26-flash-rl-tp4", "qwen38-flash-next-qad-tp4", "qwen38-flash-next-tp2")


def test_release_lock_lists_every_installer_profile_on_one_image():
    lock = installer_image.default_lock()
    assert lock["schema"] == installer_image.SCHEMA and tuple(lock["profiles"]) == SHARED
    assert installer.INSTALLABLE == frozenset(SHARED)
    assert lock["image_reference"].startswith("ghcr.io/fujitsupolycom/sparkring@sha256:")
    for profile in SHARED:
        assert installer_image.for_profile(profile) == lock
        card = setup.selection(profile)
        if not profile.startswith("qwen"):
            # Shared-image profiles select the same image through their release.
            assert card["image_id"] == lock["image_id"] and card["image_reference"] == lock["image_reference"]
    with pytest.raises(ValueError, match="not admitted"):
        installer_image.for_profile("glm53-flash-spark-tp4-dcp1-sparkcache")


@pytest.mark.parametrize("change", ["unsorted", "unknown", "empty"])
def test_shared_lock_profile_list_is_exact(change):
    value = copy.deepcopy(installer_image.default_lock())
    if change == "unsorted":
        value["profiles"] = list(reversed(value["profiles"]))
    elif change == "unknown":
        value["profiles"] = sorted([*value["profiles"], "qwen38-flash-next-qad-tp4-sparkcache"])
    else:
        value["profiles"] = []
    with pytest.raises(ValueError):
        installer_image.validate(value, PROFILE)


@pytest.mark.parametrize("profile", ["glm53-flash-nvfp4-spark-tp4", "mimo-v26-flash-rl-tp4", "glm53-flash-nvfp4-spark-tp2"])
def test_glm_and_mimo_render_on_the_shared_image(monkeypatch, profile):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Offline render contacted a host"))
    lock_value = installer_image.default_lock()
    nodes = 4 if profile.endswith("tp4") else 2
    raw = ring_site() if nodes == 4 else site(2)
    lock = installer.make_lock(profile, raw, "1" * 40, "2" * 64, image_runtime=lock_value)
    assert lock["backend"] == "compose" and lock["selection"]["release"] == lock_value["name"]
    specs = installer.specifications(lock)
    assert len(specs) == nodes
    for rank, spec in enumerate(specs):
        assert spec.image_id == lock_value["image_id"] and spec.entrypoint == installer_image.ENTRYPOINT
        assert spec.command[0] == "serve" and spec.command[spec.command.index("--node-rank") + 1] == str(rank)
        assert ("--headless" in spec.command) == bool(rank)
        assert spec.environment["VLLM_PLUGINS"].split(",")[:2] == ["b12x_loader", "sparkring_status"]
        assert "VLLM_QWEN3_8_FLASH_NEXT_HC_TP" not in spec.environment
        assert not any(key.startswith(("VLLM_QWEN3_8_", "QWEN_")) for key in spec.environment)
        assert spec.environment["SPARKCACHE_ENABLED"] == "0" and "--kv-transfer-config" not in spec.command
        assert "--enable-prefix-caching" in spec.command
        assert spec.environment["VLLM_NCCL_SO_PATH"] == "/opt/sparkring/toolchain/nccl/lib/libnccl.so.2"
        family = "glm53-flash-nvfp4-spark" if profile.startswith("glm") else "mimo-v26-flash-rl"
        assert spec.environment["XDG_CACHE_HOME"] == f"/cache/{family}-{lock_value['image_id'][7:19]}-{lock['selection']['model_revision'][:12]}"
        assert spec.mounts[-1].target == installer_image.BINDING_TARGET
        if rank == 0:
            assert spec.health_command[0] == "python3"
    connection = installer.connection(lock)
    assert connection["model"].endswith(f"-TP{nodes}")


def test_qwen_admission_features_do_not_apply_to_other_models():
    value, image, receipts = admission_fixture()
    value = {**{k: v for k, v in value.items() if k != "profile"}, "schema": installer_image.SCHEMA,
             "profiles": ["glm53-flash-nvfp4-spark-tp4", PROFILE], "image_bytes": 1, "download_bytes": 1}
    parent = json.loads(receipts[installer_image.PARENT_RECEIPT])
    parent["capabilities"]["features"] = []
    raw = json.dumps(parent).encode()
    receipts[installer_image.PARENT_RECEIPT] = raw
    value["parent_receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    toolchain = json.loads(receipts[installer_image.TOOLCHAIN_RECEIPT])
    toolchain["parent_receipt_sha256"] = value["parent_receipt_sha256"]
    raw = json.dumps(toolchain).encode()
    receipts[installer_image.TOOLCHAIN_RECEIPT] = raw
    value["toolchain_receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    def run(argv, **kwargs):
        if argv[1] == "image":
            return SimpleNamespace(stdout=json.dumps([image]))
        if argv[-1] in receipts:
            return SimpleNamespace(stdout=receipts[argv[-1]])
        return SimpleNamespace(stdout="verified")
    assert installer_image.admit(value, run=run, profile="glm53-flash-nvfp4-spark-tp4", nodes=4)["serving_qualified"] is False
    with pytest.raises(ValueError, match="Qwen topology"):
        installer_image.admit(value, run=run, profile=PROFILE, nodes=4)
