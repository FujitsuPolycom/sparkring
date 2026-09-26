"""Host-facing installer checks with fake Docker/RDMA observations."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from runtime.common import compose, installer
from runtime.common.container_spec import expected_inspection
from runtime.common.test_installer import GLM, QWEN, site
from runtime.host import checkpoint_place as place
from scripts import installer_host as host, installer_runner as runner
from scripts.test_installer_adopt import (DEPLOYMENT, IMAGE, REPOSITORY, REVISION, SHARDS, changes, contents, entry,
                                          environment, journal, plain_folder, required, sha256, state_directory,
                                          tree_state, weights, write)

linux = pytest.mark.skipif(not sys.platform.startswith("linux"),
                           reason="SparkRing checkpoint directories use Linux hard links, /proc/self/fd and flock")


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
    if "io.sparkring.image-lock" in spec.labels:
        from runtime.common import loader_policy
        expected["host_config"]["SecurityOpt"] = [loader_policy.inspection_option() if option.startswith("seccomp=") else option
                                                  for option in spec.security_opt]
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


def served_copy(tmp_path, monkeypatch, name="models/qwen"):
    """A named exact copy served in place, verified once, with Linux-style stat fast paths."""
    data = contents()
    folder = plain_folder(tmp_path / name, data, required(data))
    env = environment(tmp_path, monkeypatch, reuse=True, model=folder)
    monkeypatch.setattr(host, "POSIX_STATS", True)
    assert env.call("model") == {"ok": True}
    return env, folder


def touch(path, seconds=1):
    info = os.stat(path)
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + seconds * 10**9))


def test_verify_rehashes_only_changed_entries(tmp_path, monkeypatch):
    env, folder = served_copy(tmp_path, monkeypatch)
    receipt_path = env.state / "model.json"
    hashed = []
    original = host._hash_files
    monkeypatch.setattr(host, "_hash_files", lambda root, recorded: hashed.append(sorted(recorded)) or original(root, recorded))
    host.verify_model(env.lock, env.row, receipt_path)
    assert hashed == []
    # A new modification time with unchanged content re-hashes only that file.
    touch(folder / "config.json")
    saved = receipt_path.read_bytes()
    receipt = host.verify_model(env.lock, env.row, receipt_path)
    assert hashed == [["config.json"]]
    assert receipt["file_stats"]["config.json"] == host.model_file_stats(folder, required(env.data), in_place=True)["config.json"]
    assert receipt_path.read_bytes() == saved
    # Changed content of the same size is found; the unchanged files are not read.
    (folder / SHARDS[0]).write_bytes(env.data[SHARDS[0]].upper())
    touch(folder / SHARDS[0], 2)
    with pytest.raises(ValueError, match=r"differs from its recorded files \(model-00001-of-00002.safetensors\); "
                                         "SparkRing does not change it"):
        host.verify_model(env.lock, env.row, receipt_path)
    assert hashed[-1] == ["config.json", SHARDS[0]]
    # Unknown receipt keys and receipt entries outside the pinned names are ignored.
    (folder / SHARDS[0]).write_bytes(env.data[SHARDS[0]])
    document = json.loads(receipt_path.read_text())
    document.update(sources=[{"path": "/elsewhere"}], extra="ignored")
    document["files"]["README.md"] = "0" * 64
    host.verify_model(env.lock, env.row, receipt_path, receipt=document)


def test_receipt_refresh_writes_only_the_ranks_own_receipt(tmp_path, monkeypatch):
    env, folder = served_copy(tmp_path, monkeypatch)
    receipt_path, record = env.state / "model.json", host._checkpoint_record(env.row["model"])
    touch(folder / "config.json")
    current = host.model_file_stats(folder, required(env.data), in_place=True)
    saved, saved_record = receipt_path.read_bytes(), record.read_bytes()
    host.verify_model(env.lock, env.row, receipt_path)
    assert (receipt_path.read_bytes(), record.read_bytes()) == (saved, saved_record)
    host.verify_model(env.lock, env.row, receipt_path, refresh=True)
    for path in (receipt_path, record):
        assert json.loads(path.read_text())["file_stats"] == current
    hashed = []
    original = host._hash_files
    monkeypatch.setattr(host, "_hash_files", lambda root, recorded: hashed.append(sorted(recorded)) or original(root, recorded))
    assert env.call("model-check") == {"ok": True} and hashed == []
    # Receipt reuse reads another deployment's receipt and never rewrites it.
    previous = tmp_path / "previous"
    (previous / "installer").mkdir(parents=True)
    (previous / ".installer-owner.json").write_text(json.dumps({"deployment": "e" * 64}))
    shutil.copyfile(receipt_path, previous / "installer/model.json")
    theirs = (previous / "installer/model.json").read_bytes()
    receipt_path.unlink()
    touch(folder / "tokenizer.json")
    assert env.call("model-reuse-receipt", {"workspace": str(previous), "deployment": "e" * 64}) == {"reused": True}
    assert (previous / "installer/model.json").read_bytes() == theirs
    assert json.loads(receipt_path.read_text())["file_stats"] == host.model_file_stats(folder, required(env.data), in_place=True)


def owned_workspace(root, deployment, receipt):
    """A retained deployment's workspace holding its checkpoint receipt; returns the receipt path."""
    path = root / "installer/model.json"
    path.parent.mkdir(parents=True)
    if deployment is not None:
        (root / ".installer-owner.json").write_text(json.dumps({"deployment": deployment}))
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return path


def receipt_for(data, folder, names, **changes):
    return {"repository": REPOSITORY, "revision": REVISION, "path": str(folder),
            "files": {name: changes.get(name, sha256(data[name])) for name in names},
            "file_stats": {name: place.stats(os.lstat(folder / name)) for name in names},
            "origin": "operator-declared-verified-copy"}


@linux
def test_other_receipts_are_refreshed_only_for_verified_inodes(tmp_path, monkeypatch):
    env = environment(tmp_path, monkeypatch)
    data = env.data
    user = plain_folder(tmp_path / "var/tmp/models/qwen", data)
    other = plain_folder(tmp_path / "var/tmp/models/other", data, weights(data))
    workspaces = tmp_path / "srv/sparkring/tp2"
    served = owned_workspace(workspaces / "qwen-a", "a" * 64, receipt_for(data, user, sorted(data)))
    stale = owned_workspace(workspaces / "qwen-b", "b" * 64, receipt_for(data, user, required(data), **{SHARDS[0]: "0" * 64}))
    elsewhere = owned_workspace(workspaces / "qwen-c", "c" * 64, receipt_for(data, other, weights(data)))
    unowned = owned_workspace(workspaces / "qwen-d", None, receipt_for(data, user, required(data)))
    claimed = owned_workspace(workspaces / "qwen-e", "e" * 64, receipt_for(data, user, required(data)))
    record = host._checkpoint_record(str(user))
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(receipt_for(data, user, sorted(data))))
    untouched = {path: path.read_bytes() for path in (elsewhere, unowned, claimed)}
    before = {path: json.loads(path.read_text()) for path in (served, stale, record)}
    files = {name: entry("link" if name in weights(data) else "copy", user / name, data, name) for name in required(data)}
    # Linking sets the change time from the kernel's coarse clock; let it move past the files' creation.
    time.sleep(0.05)
    listed = [str(served), str(stale), str(elsewhere), str(unowned), {"path": str(claimed), "deployment": "f" * 64}]
    result = env.call("model-adopt", {"files": files, "receipts": listed, "tolerance_bytes": 0})
    assert result["complete"] and sorted(result["refreshed"]) == sorted([str(served), str(stale), str(record)])
    for path, document in before.items():
        after = json.loads(path.read_text())
        assert {key: value for key, value in after.items() if key != "file_stats"} == {
            key: value for key, value in document.items() if key != "file_stats"}
        for name, recorded in document["file_stats"].items():
            refreshed = name in weights(data) and document["files"][name] == sha256(data[name])
            expected = place.stats(os.lstat(user / name)) if refreshed else recorded
            assert after["file_stats"][name] == expected, (path, name)
            assert (after["file_stats"][name] != recorded) is refreshed
    assert {path: path.read_bytes() for path in untouched} == untouched
    # The refreshed deployment's receipt again describes its files, so it verifies without hashing.
    row = {**env.row, "model": str(user), "reuse_verified_model": True}
    hashed = []
    original = host._hash_files
    monkeypatch.setattr(host, "_hash_files", lambda root, recorded: hashed.append(sorted(recorded)) or original(root, recorded))
    host.verify_model(env.lock, row, served)
    assert hashed == []


def test_reused_copy_that_differs_is_reported_not_repaired(tmp_path, monkeypatch):
    data = contents()
    changed = {"config.json": data["config.json"].upper()}
    folder = plain_folder(tmp_path / "models/qwen", data, required(data), changed=changed)
    env = environment(tmp_path, monkeypatch, reuse=True, model=folder)
    monkeypatch.setattr(host, "fetch_model", lambda *a, **k: pytest.fail("download into a copy SparkRing does not own"))
    monkeypatch.setattr(host, "adopt_model", lambda *a, **k: pytest.fail("placement into a copy SparkRing does not own"))
    monkeypatch.setattr(host, "run", lambda argv, **k: pytest.fail(f"command for a copy SparkRing does not own: {argv}"))
    before = tree_state(folder)
    with pytest.raises(ValueError, match=r"Checkpoint at .* differs from the pinned revision in config.json; "
                                         "SparkRing does not change it"):
        env.call("model")
    assert changes(before, tree_state(folder)) == {}
    assert not (env.state / "model.json").exists()


def test_missing_named_copy_is_not_created(tmp_path, monkeypatch):
    folder = tmp_path / "models/absent"
    env = environment(tmp_path, monkeypatch, reuse=True, model=folder)
    monkeypatch.setattr(host, "run", lambda argv, **k: pytest.fail(f"command for a missing copy: {argv}"))
    for operation in ("model", "model-transfer-prepare"):
        document = None if operation == "model" else {"repository": REPOSITORY, "revision": REVISION, "files": {}, "sizes": {}}
        with pytest.raises(ValueError, match=r"does not exist; SparkRing does not create it|serves .* in place; "
                                             "SparkRing never writes into a copy it did not create"):
            env.call(operation, document)
    assert not folder.exists() and not folder.parent.exists()


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
    # With the required names, a SparkRing directory holds exactly them; a copy served in
    # place may also hold its optional names and its download client's metadata.
    names = ["config.json", "model.safetensors.index.json", "part.safetensors"]
    assert sorted(host.model_files(tmp_path, names)) == names
    (tmp_path / "README.md").write_text("notes")
    (tmp_path / ".cache/huggingface/download").mkdir(parents=True)
    (tmp_path / ".cache/huggingface/download/part.safetensors.metadata").write_text("record")
    with pytest.raises(host.NameSetError, match="also holds .cache/huggingface/download/part.safetensors.metadata, README.md"):
        host.model_files(tmp_path, names)
    assert sorted(host.model_files(tmp_path, names, in_place=True, optional=["README.md"])) == names
    with pytest.raises(host.NameSetError, match="also holds README.md"):
        host.model_file_stats(tmp_path, names, in_place=True)
    (tmp_path / "part.safetensors").unlink()
    with pytest.raises(host.NameSetError, match="lacks part.safetensors"):
        host.model_file_stats(tmp_path, names, in_place=True, optional=["README.md"])


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


def test_smoke_request_settings_come_from_the_serving_profile():
    from scripts import installer_host
    assert installer_host.smoke_request({"profile": "glm53-flash-nvfp4-spark-tp4"}) == {
        "chat_template_kwargs": {"reasoning_effort": "low"}}
    assert installer_host.smoke_request({"profile": "qwen38-flash-next-qad-tp4"}) == {
        "chat_template_kwargs": {"enable_thinking": False}}


def test_pinned_differences_name_stale_or_missing_files():
    from scripts import installer_host
    manifest = installer_host.checksum_manifest("mimo-v26-flash-rl-tp4")
    pins = dict(reversed(line.split(maxsplit=1)) for line in manifest.read_text().splitlines())
    assert installer_host.pinned_differences("mimo-v26-flash-rl-tp4", pins) == []
    stale = {**pins, "dflash/config.json": "0" * 64}
    stale.pop("tokenizer.json")
    assert installer_host.pinned_differences("mimo-v26-flash-rl-tp4", stale) == ["dflash/config.json", "tokenizer.json"]
    assert installer_host.checksum_manifest("qwen38-flash-next-qad-tp4").name == "SHA256SUMS"
    # The SparkCache profiles pin their own revision; each keeps its own file.
    cache = installer_host.checksum_manifest("qwen38-flash-next-tp2-sparkcache")
    assert cache.parent.name == "qwen38-flash-next-tp2-sparkcache"


def test_checkpoint_hashes_are_remembered_only_for_an_unchanged_tree(tmp_path, monkeypatch):
    from scripts import installer_host
    monkeypatch.setattr(installer_host, "CHECKPOINTS", tmp_path / "records")
    monkeypatch.setattr(installer_host, "POSIX_STATS", True)
    card = {"model_repository": "owner/model", "model_revision": "a" * 40}
    receipt = {"repository": "owner/model", "revision": "a" * 40, "path": "/models/m",
               "files": {"config.json": "1" * 64}, "file_stats": {"config.json": [1, 2, 3, 4, 5]}}
    installer_host.remember_checkpoint(receipt)
    assert installer_host.remembered_checkpoint(card, "/models/m", receipt["file_stats"]) == receipt["files"]
    assert installer_host.remembered_checkpoint(card, "/models/m", {"config.json": [1, 2, 3, 4, 6]}) is None
    assert installer_host.remembered_checkpoint(card, "/models/other", receipt["file_stats"]) is None
    assert installer_host.remembered_checkpoint({**card, "model_revision": "b" * 40}, "/models/m", receipt["file_stats"]) is None


class Hub:
    """``host.run`` for Docker: inspections answer from ``containers``; ``run`` writes pinned files into the staging mount.

    ``fail_after`` makes a download stop after that many files, as a dropped
    connection does. Every ``docker run`` argv is kept in ``runs``.
    """

    def __init__(self, data, *, containers=(), fail_after=None):
        self.data, self.containers, self.fail_after = data, set(containers), fail_after
        self.commands, self.runs = [], []

    def __call__(self, argv, **kwargs):
        self.commands.append(argv)
        assert argv[0] == "docker"
        if argv[1:3] == ["container", "inspect"]:
            return SimpleNamespace(returncode=0 if argv[3] in self.containers else 1, stdout="")
        if argv[1:3] == ["image", "inspect"]:
            return SimpleNamespace(returncode=0, stdout="[]")
        assert argv[1] == "run"
        self.runs.append(argv)
        mount = dict(item.split("=", 1) for item in argv[argv.index("--mount") + 1].split(","))
        names = argv[argv.index(IMAGE) + 5:]
        for count, name in enumerate(names):
            if self.fail_after is not None and count >= self.fail_after:
                raise host.CommandError(1, argv, "", "connection reset")
            write(Path(mount["src"]) / name, self.data[name])
        return SimpleNamespace(returncode=0, stdout="")


@linux
def test_interrupted_checkpoint_download_resumes_in_its_staging_directory(tmp_path, monkeypatch):
    env = environment(tmp_path, monkeypatch)
    hub = Hub(env.data, fail_after=2)
    monkeypatch.setattr(host, "run", hub)
    # The failure names the client's last error line and what the next run does, not the docker argv.
    with pytest.raises(ValueError, match=r"^Downloading 8 files from huggingface\.co failed: connection reset\. "
                                         "Nothing was placed for those files; files the download completed are placed "
                                         r"by the next run\. Check that Node 0 reaches huggingface\.co, then repeat "
                                         r"the command\.$"):
        env.call("model")
    assert not (env.state / "model.json").exists() and os.listdir(env.model) == []
    hub.fail_after = None
    assert env.call("model") == {"ok": True}
    first = hub.runs[0][hub.runs[0].index(IMAGE) + 5:]
    assert first == required(env.data)
    # The two files the interrupted download completed are placed, not downloaded again.
    assert hub.runs[1][hub.runs[1].index(IMAGE) + 5:] == first[2:]
    receipt = json.loads((env.state / "model.json").read_text())
    assert receipt["origin"] == "pinned-hub-download" and receipt["fetched"] == required(env.data)
    assert not (state_directory(env.model) / "fetch").exists()
    assert sorted(journal(env.model)) == required(env.data)


@linux
@pytest.mark.parametrize("case", ["files", "state-without-owner", "download-record"])
def test_nonempty_checkpoint_destination_not_created_by_sparkring_is_refused(tmp_path, monkeypatch, case):
    env = environment(tmp_path, monkeypatch)
    env.model.mkdir(parents=True)
    (env.model / "notes.txt").write_text("not a download")
    if case == "state-without-owner":
        state_directory(env.model).mkdir(mode=0o700)
    if case == "download-record":
        env.state.mkdir()
        (env.state / "model-download.json").write_text(json.dumps(
            {"deployment": DEPLOYMENT, "path": str(env.model), "repository": REPOSITORY, "revision": REVISION}))
    monkeypatch.setattr(host, "run", lambda *a, **k: pytest.fail("download into a directory SparkRing did not create"))
    with pytest.raises(ValueError, match="is not empty and was not created by SparkRing"):
        env.call("model")
    assert os.listdir(env.model) == ["notes.txt"] and (env.model / "notes.txt").read_text() == "not a download"


@linux
@pytest.mark.parametrize("existing", [False, True])
def test_second_download_into_a_directory_is_refused_while_the_first_container_exists(tmp_path, monkeypatch, existing):
    env = environment(tmp_path, monkeypatch)
    name = host.fetch_container(env.model)
    hub = Hub(env.data, containers=[name] if existing else [], fail_after=0)
    monkeypatch.setattr(host, "run", hub)
    if existing:
        # An interrupted installer session leaves its download container running.
        with pytest.raises(ValueError, match="still in progress in container " + name):
            env.call("model")
        assert hub.runs == []
    else:
        with pytest.raises(ValueError, match="from huggingface.co failed: connection reset"):
            env.call("model")
        assert hub.runs[0][hub.runs[0].index("--name") + 1] == name
    inspected = [argv for argv in hub.commands if argv[1:3] == ["container", "inspect"]]
    assert inspected and all(argv[3] == name for argv in inspected)


@linux
@pytest.mark.parametrize("staged", [False, True])
def test_resumed_download_needs_only_the_rest_of_its_storage_allowance(tmp_path, monkeypatch, staged):
    from runtime.host import checkpoint_plan
    env = environment(tmp_path, monkeypatch)
    if staged:
        monkeypatch.setattr(host, "run", Hub(env.data, fail_after=2))
        with pytest.raises(ValueError, match="from huggingface.co failed: connection reset"):
            env.call("model")
    monkeypatch.setattr(host, "run", lambda argv, **k: pytest.fail("download without free space") if argv[1] == "run"
                        else SimpleNamespace(returncode=1, stdout=""))
    monkeypatch.setattr(host.shutil, "disk_usage", lambda path: SimpleNamespace(total=1 << 40, used=0, free=1000))
    observed = []
    original = checkpoint_plan.required_space
    monkeypatch.setattr(checkpoint_plan, "required_space",
                        lambda written, **kwargs: observed.append(sorted(written)) or original(written, **kwargs))
    with pytest.raises(ValueError, match="needs .* GiB free on the filesystem of .*; 0.0 GiB is free. Free space, then "
                                         "repeat sudo sparkring install. The running model has not been stopped."):
        env.call("model")
    names = required(env.data)[2:] if staged else required(env.data)
    # Only the files still to be written count, each at its pinned size.
    assert observed == [sorted(len(env.data[name]) for name in names)]
    if staged:
        assert sorted(journal(env.model)) == required(env.data)[:2]


def test_command_errors_name_the_program_and_its_last_output_lines_not_its_arguments():
    argv = ["docker", "--context", "default", "run", "--rm", "--mount", "type=bind,src=/srv/x/fetch,dst=/fetch",
            "sha256:" + "a" * 64, "-c", "print(1)", "owner/model", "a" * 40, *[f"model-{n:05d}.safetensors" for n in range(40)]]
    error = host.CommandError(1, argv, "", "Traceback (most recent call last):\n  ...\nhttpx.ConnectError: [Errno 101] "
                              "Network is unreachable\n")
    assert str(error) == ("docker run exited with status 1: Traceback (most recent call last): | ... | "
                          "httpx.ConnectError: [Errno 101] Network is unreachable")
    assert error.lines(1) == ["httpx.ConnectError: [Errno 101] Network is unreachable"]
    assert str(host.CommandError(2, ["nvidia-smi", "--query-compute-apps=pid"], "", "")) == \
        "nvidia-smi exited with status 2"


def test_rank_operations_run_with_umask_0022():
    # Files a rank operation copies into a checkpoint directory get mode 0644 whatever the SSH session's umask.
    assert runner.HOST.splitlines()[:2] == ["import base64,json,os,pathlib,subprocess,sys", "os.umask(0o022)"]


@pytest.mark.parametrize("user", ["code", "root", None])
def test_every_rank_operation_runs_under_sudo_when_the_ssh_user_is_not_root(user, monkeypatch, tmp_path):
    raw = site()
    for number, row in enumerate(raw["hosts"]):
        row["host"] = f"{user}@spark{number}" if user else f"spark{number}"
    lock = installer.make_lock(QWEN, raw, "1" * 40, "2" * 64)
    current = object.__new__(runner.Runner)
    current.lock, current.directory = lock, tmp_path
    (tmp_path / "source.bundle").write_bytes(b"bundle")
    calls = []

    def ssh(target, argv, **kwargs):
        calls.append(argv)
        if "is_dir()" in " ".join(argv):
            return "True\n"
        return "ok\n" if runner.SOURCE in argv else '{"ok": true}'
    monkeypatch.setattr(runner, "ssh", ssh)
    actions = [action for phase in installer.operation_plan(lock, "up")["phases"] for action in phase["actions"]]
    phases = {action["argv"][1] for action in actions} | {action["verify"]["argv"][1] for action in actions if "verify" in action}
    operations = sorted(phases - {"prerequisites"} | {
        "source-check", "model-adopt", "model-fetch", "model-transfer-manifest", "model-transfer-prepare",
        "model-transfer-complete", "model-reuse-receipt", "model-settled", "status"})
    assert "model-settled" in phases
    expected = [] if user == "root" else ["sudo", "-n"]
    for operation in operations:
        calls.clear()
        result = current._call(lock["site"]["ranks"][1]["host"], ["installer", operation, "1"], 30)
        assert result["returncode"] == 0, (operation, result["stderr"])
        assert calls and all(argv[:len(expected)] == expected and argv[len(expected)] == "python3" for argv in calls), (
            operation, calls)
    # The host probe reads the SSH session's address, which sudo does not keep, so it runs as the login user.
    calls.clear()
    monkeypatch.setattr(runner, "check_facts", lambda facts, row: None)
    monkeypatch.setattr(runner, "check_workloads", lambda *a, **k: None)
    monkeypatch.setattr(runner, "ssh", lambda target, argv, **kwargs: calls.append(argv) or "{}")
    current._call(lock["site"]["ranks"][1]["host"], ["installer", "prerequisites", "1"], 30)
    assert calls[0][0] == "python3"
