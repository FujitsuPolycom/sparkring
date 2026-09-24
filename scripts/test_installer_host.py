"""Host-facing installer checks with fake Docker/RDMA observations."""
import copy
import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from runtime.common import compose, installer
from runtime.common.container_spec import expected_inspection
from runtime.common.test_installer import GLM, QWEN, site
from scripts import installer_host as host, installer_runner as runner


def facts(row):
    return {"platform": "Linux", "architecture": "aarch64", "tools": {"docker": True, "PyYAML": True},
            "gpu": "GPU 0: NVIDIA GB10 (UUID: test)",
            "management_ip": row["management_ip"], "fabric_ip": row["host_ip"], "interface": row["interface"],
            "rdma": [{"device": device, "active": True, "type": "RoCE v2", "mtu": 9000, "rdma_mtu": 4096,
                      "gid_ip": "198.18.20.1", "ips": ["198.18.20.1"]} for device in row["hcas"]]}


def test_image_prepare_accepts_verified_untagged_id_hidden_from_default_listing(tmp_path, monkeypatch):
    image = "sha256:" + "a" * 64
    lock = {"id": "fixture", "site": {"workspace": str(tmp_path), "ranks": [{}]},
            "selection": {"image_id": image, "image_reference": image}}
    (tmp_path / ".installer-owner.json").write_text(json.dumps({"deployment": "fixture"}))
    monkeypatch.setattr(installer, "validate", lambda _: lock)
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        assert argv in (["docker", "info"], ["docker", "image", "inspect", image])
        return SimpleNamespace(returncode=0, stdout=json.dumps([{"Id": image, "RepoTags": []}]))
    monkeypatch.setattr(host, "run", run)
    monkeypatch.setattr(host, "admit_image", lambda _: {"verified_image": image})
    assert host.perform("image", lock, 0) == {"ok": True}
    assert ["docker", "image", "inspect", image] in calls
    assert json.loads((tmp_path / "installer/image.json").read_text())["verified_image"] == image


@pytest.mark.parametrize("mutation", ["architecture", "missing-tool", "management", "wrong-gid", "mtu", "down-link"])
def test_prerequisites_fail_before_host_changes(mutation):
    row = installer.make_lock(GLM, site(), "1" * 40, "2" * 64)["site"]["ranks"][0]
    value = facts(row)
    runner.check_facts(value, row)
    if mutation == "architecture":
        value["architecture"] = "x86_64"
    elif mutation == "missing-tool":
        value["tools"]["PyYAML"] = False
    elif mutation == "management":
        value["management_ip"] = "192.0.2.99"
    elif mutation == "wrong-gid":
        value["rdma"][0]["gid_ip"] = "198.18.99.1"
    elif mutation == "mtu":
        value["rdma"][0]["rdma_mtu"] = 1024
    else:
        value["rdma"][0]["active"] = False
    with pytest.raises(ValueError):
        runner.check_facts(value, row)


def test_tp4_bootstrap_uses_management_while_rdma_devices_remain_independent():
    row = installer.make_lock(GLM, site(), "1" * 40, "2" * 64)["site"]["ranks"][0]
    row.update(fabric={"site_path": "/etc/sparkring/site.json"}, host_ip="192.0.2.50", interface="management0")
    value = facts(row)
    value.update(fabric_ip="198.18.20.1", interface="data0", ipv4={"management0": ["192.0.2.50"], "data0": ["198.18.20.1"]})
    for device in value["rdma"]:
        device["netdev"] = "data0"
    runner.check_facts(value, row)
    value["ipv4"]["management0"] = ["192.0.2.51"]
    with pytest.raises(ValueError, match="bootstrap address"):
        runner.check_facts(value, row)


def inspection(spec):
    image = {"Id": spec.image_id, "Os": "linux", "Architecture": "arm64", "Config": {"Env": [], "Labels": {}}}
    expected = expected_inspection(spec, image, backend="compose")
    info = {"Id": "container-id", "Image": spec.image_id, "State": {"Running": False},
            "Config": {"Cmd": expected["cmd"], "Entrypoint": expected["entrypoint"],
                       "Env": [key + "=" + value for key, value in expected["env"].items()],
                       "Labels": expected["labels"], "Healthcheck": expected["healthcheck"],
                       "WorkingDir": expected["working_dir"], "User": expected["user"]},
            "HostConfig": expected["host_config"],
            "Mounts": [dict(Destination=key, **value) for key, value in expected["mounts"].items()]}
    return info, image


@pytest.mark.parametrize("profile", [GLM, QWEN])
@pytest.mark.parametrize("change", ["none", "image", "label", "command", "mount", "extra-device", "privileged"])
def test_container_ownership_is_more_than_a_name(profile, change, monkeypatch):
    lock = installer.make_lock(profile, site(), "1" * 40, "2" * 64)
    spec = installer.specifications(lock)[0]
    info, image = inspection(spec)
    monkeypatch.setattr(compose, "check_project_containers", lambda *a, **k: None)
    if change == "none":
        assert host.owned(spec, info, image) == info
        return
    if change == "image":
        info["Image"] = "sha256:" + "0" * 64
    elif change == "label":
        info["Config"]["Labels"][compose.LABEL] = "foreign"
    elif change == "command":
        info["Config"]["Cmd"].append("--different-runtime")
    elif change == "mount":
        info["Mounts"][0]["RW"] = True
    elif change == "extra-device":
        info["HostConfig"]["Devices"].append({"PathOnHost": "/dev/other"})
    else:
        info["HostConfig"]["Privileged"] = True
    with pytest.raises(ValueError, match="refusing adoption"):
        host.owned(spec, info, image)


@pytest.mark.parametrize("selinux", [False, True])
def test_daemon_empty_capabilities_and_nvidia_selinux_default(selinux, monkeypatch):
    lock = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64)
    spec = installer.specifications(lock)[0]
    info, image = inspection(spec)
    info["HostConfig"].update(CapAdd=None, SecurityOpt=["label=disable"])
    monkeypatch.setattr(compose, "check_project_containers", lambda *a, **k: None)
    monkeypatch.setattr(host, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps(["name=selinux"] if selinux else ["name=apparmor", "name=seccomp,profile=builtin"])))
    if selinux:
        with pytest.raises(ValueError, match="refusing adoption"):
            host.owned(spec, info, image)
    else:
        assert host.owned(spec, info, image) is info
    # Extra policies remain a mismatch even on a host without SELinux.
    info["HostConfig"]["SecurityOpt"].append("apparmor=unconfined")
    with pytest.raises(ValueError, match="refusing adoption"):
        host.owned(spec, info, image)


def test_unchanged_verified_checkpoint_uses_file_identity_but_changes_rehash(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    for name, content in {"config.json": b"{}", "model.safetensors.index.json": b'{"weight_map":{"w":"weights.safetensors"}}', "weights.safetensors": b"original"}.items():
        (model / name).write_bytes(content)
    hashes = host.model_files(model)
    receipt = {"repository": "test/model", "revision": "a" * 40, "path": str(model), "files": hashes,
               "file_stats": host.model_file_stats(model)}
    lock = {"selection": {"profile": "fixture", "model_repository": "test/model", "model_revision": "a" * 40}}
    row = {"model": str(model)}
    monkeypatch.setattr(host, "POSIX_STATS", True)
    monkeypatch.setattr(installer, "checkpoint_contract", lambda _: {"config_sha256": hashes["config.json"], "index_sha256": hashes["model.safetensors.index.json"]})
    original = host.model_files
    calls = []
    monkeypatch.setattr(host, "model_files", lambda path: calls.append(path) or original(path))
    host.verify_model(lock, row, tmp_path / "receipt.json", receipt=receipt)
    assert calls == []
    (model / "weights.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checkpoint differs"):
        host.verify_model(lock, row, tmp_path / "receipt.json", receipt=receipt)
    assert calls


@pytest.mark.skipif(os.name != "posix", reason="Linux ctime is the change-detection contract")
def test_restoring_mtime_cannot_hide_a_linux_checkpoint_edit(tmp_path):
    path = tmp_path / "weights.safetensors"
    path.write_bytes(b"original")
    before = host.model_file_stats(tmp_path)
    metadata = path.stat()
    path.write_bytes(b"modified")
    os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    assert host.model_file_stats(tmp_path) != before


def test_native_receipt_reads_keep_binary_output(monkeypatch):
    calls = []
    def observe(image, release, *, run):
        result = run(["docker", "run", "--entrypoint", "/bin/cat", image, "/receipt"], capture_output=True)
        assert result.stdout == b"raw receipt"
        return {"verified": True}
    def command(argv, **kwargs):
        calls.append(kwargs)
        assert kwargs["text"] is False
        return SimpleNamespace(stdout=b"raw receipt")
    monkeypatch.setattr(host.glm_native_candidate, "observe", observe)
    monkeypatch.setattr(host, "run", command)
    assert host.admit_image(installer.make_lock(GLM, site(), "1" * 40, "2" * 64)) == {"verified": True}
    assert len(calls) == 1


def test_status_runner_reaches_verified_observation_producer(monkeypatch):
    current = object.__new__(runner.Runner)
    current.lock = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64)
    row = current.lock["site"]["ranks"][0]
    observed = {"schema": "sparkring-model-observation/v1", "container_id": "observed", "deployment_id": current.lock["id"]}
    calls = []
    monkeypatch.setattr(runner, "ssh", lambda *a, **kw: "True\n")
    current.remote = lambda rank, operation: calls.append((rank, operation)) or observed
    result = current._call(row["host"], ["installer", "status", "0"], 30)
    assert json.loads(result["stdout"]) == observed
    assert calls == [(0, "status")]


def test_status_ownership_failure_never_falls_back_to_a_name_lookup(monkeypatch):
    current = object.__new__(runner.Runner)
    current.lock = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64)
    row = current.lock["site"]["ranks"][0]
    calls = []
    monkeypatch.setattr(runner, "ssh", lambda *a, **kw: calls.append(a) or "True\n")
    def refuse(*args):
        raise ValueError("Container specification mismatch")
    current.remote = refuse
    result = current._call(row["host"], ["installer", "status", "0"], 30)
    assert result["returncode"] == 1 and len(calls) == 1


def test_existing_workload_is_rejected_before_download_but_own_container_can_resume():
    lock = installer.make_lock(GLM, site(), "1" * 40, "2" * 64)
    value = {"gpu_containers": [], "gpu_process_ancestors": []}
    runner.check_workloads(value, lock, 0)
    value["gpu_process_ancestors"] = [[123, 1]]
    with pytest.raises(ValueError, match="no assets were downloaded"):
        runner.check_workloads(value, lock, 0)
    value["gpu_containers"] = [{"name": "sr-local-test-r0", "image": lock["selection"]["image_id"],
                                "pid": 123, "labels": {compose.LABEL: lock["id"]}}]
    runner.check_workloads(value, lock, 0)
    value["gpu_process_ancestors"].append([999, 1])
    with pytest.raises(ValueError, match="Another GPU workload"):
        runner.check_workloads(value, lock, 0)


def test_model_inventory_requires_every_weight_shard_and_detects_changes(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"tensor": "part.safetensors"}}))
    with pytest.raises(ValueError, match="missing indexed"):
        host.model_files(tmp_path)
    (tmp_path / "part.safetensors").write_bytes(b"original")
    first = host.model_files(tmp_path)
    (tmp_path / "part.safetensors").write_bytes(b"changed")
    assert host.model_files(tmp_path)["part.safetensors"] != first["part.safetensors"]


def test_source_bootstrap_checks_bundle_commit_and_owner(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args, cwd=repo):
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()
    git("init", "-q")
    git("config", "user.name", "Offline test")
    git("config", "user.email", "offline@example.invalid")
    (repo / "tracked").write_text("fixture")
    git("add", "tracked")
    git("commit", "-qm", "fixture")
    bundle = tmp_path / "source.bundle"
    git("bundle", "create", str(bundle), "HEAD")
    data = bundle.read_bytes()
    workspace = tmp_path / "workspace"
    source = workspace / "source"
    argv = [sys.executable, "-I", "-c", runner.SOURCE, str(workspace), str(source), git("rev-parse", "HEAD"),
            "a" * 64, hashlib.sha256(data).hexdigest()]
    result = subprocess.run([*argv, "install"], input=data, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert subprocess.run([*argv, "check"], capture_output=True).returncode == 0
    # Repeated installation verifies and reuses the exact checkout.
    assert subprocess.run([*argv, "install"], input=data, capture_output=True).returncode == 0
    (source / "tracked").write_text("local edit")
    changed = subprocess.run([*argv, "check"], capture_output=True)
    assert changed.returncode != 0
    assert b"Host source changed" in changed.stderr
    wrong_owner = copy.copy(argv)
    wrong_owner[-2] = "b" * 64
    assert subprocess.run([*wrong_owner, "check"], capture_output=True).returncode != 0


def test_status_does_not_write_workspace(tmp_path, monkeypatch):
    # Stop early on a missing recorded image; status must not create state dirs.
    lock = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64)
    lock["site"]["workspace"] = str(tmp_path)
    (tmp_path / ".installer-owner.json").write_text(json.dumps({"deployment": lock["id"]}))
    monkeypatch.setattr(installer, "validate", lambda value: value)
    monkeypatch.setattr(host, "image_info", lambda *a: (_ for _ in ()).throw(ValueError("missing image")))
    with pytest.raises(ValueError, match="missing image"):
        host.perform("status", lock, 0)
    assert not (tmp_path / "installer").exists()
