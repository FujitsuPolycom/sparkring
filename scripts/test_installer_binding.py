"""Stopped-container binding finalization and the create/start boundary."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

from runtime.common import compose, installer, installer_image, qwen_mesh
from runtime.common.test_installer_image import PROFILE, image_lock, ring_site
from runtime.host import node
from runtime.host.test_observation_identity import identity_files
from scripts import installer_host as host
from scripts.test_installer_host import inspection


def fixture(root):
    node_id, _ = identity_files(root)
    mountinfo = root / "proc/self/mountinfo"
    mountinfo.parent.mkdir(parents=True)
    mountinfo.write_text(f"1 0 0:1 / {root.as_posix()} rw - ext4 /dev/fixture rw\n")
    lock = installer.make_lock(PROFILE, ring_site(), "1" * 40, "2" * 64, image_runtime=image_lock())
    row = {**lock["site"]["ranks"][0], "node_id": node_id}
    info = {"Id": "a" * 64, "Image": lock["selection"]["image_id"], "State": {"Running": False},
            "Config": {"Labels": {compose.LABEL: lock["id"], "io.sparkring.rank": "0"}}}
    return lock, row, info


def test_binding_is_finalized_from_actual_id_while_stopped(tmp_path):
    lock, row, info = fixture(tmp_path)
    path = host.local_binding_path(lock, row, root=tmp_path)
    installer.write(path, {})
    expected = host.check_runtime_binding(lock, row, info, root=tmp_path, finalize=True)
    assert set(expected) == {"schema", "deployment_id", "node_id", "container_id", "image_id", "rank"}
    assert expected["container_id"] == info["Id"]
    assert host.read_runtime_binding(path) == expected
    before = path.stat().st_mtime_ns
    info["State"]["Running"] = True
    assert host.check_runtime_binding(lock, row, info, root=tmp_path, finalize=True) == expected
    assert path.stat().st_mtime_ns == before
    path.write_text("{}")
    with pytest.raises(ValueError, match="stopped"):
        host.check_runtime_binding(lock, row, info, root=tmp_path, finalize=True)
    assert path.read_text() == "{}"


@pytest.mark.parametrize("change", ["node", "container", "image", "deployment", "rank"])
def test_binding_rejects_foreign_identity_before_writing(tmp_path, change):
    lock, row, info = fixture(tmp_path)
    if change == "node":
        row["node_id"] = str(uuid.UUID(int=99))
    elif change == "container":
        info["Id"] = "not-a-docker-id"
    elif change == "image":
        info["Image"] = "sha256:" + "0" * 64
    elif change == "deployment":
        info["Config"]["Labels"][compose.LABEL] = "0" * 64
    else:
        info["Config"]["Labels"]["io.sparkring.rank"] = "1"
    with pytest.raises(ValueError):
        host.check_runtime_binding(lock, row, info, root=tmp_path, finalize=True)
    assert not (tmp_path / "srv").exists()


@pytest.mark.parametrize("filesystem", ["nfs4", "cifs", "fuse.sshfs", "unknown"])
def test_remote_or_unknown_binding_mount_is_rejected_before_source_inspection(tmp_path, monkeypatch, filesystem):
    lock, row, _ = fixture(tmp_path)
    source = tmp_path / installer_image.binding_path(lock, row).lstrip("/")
    table = tmp_path / "proc/self/mountinfo"
    table.write_text(table.read_text() + f"2 1 0:2 / {source.parent.as_posix()} rw - {filesystem} remote rw\n")
    monkeypatch.setattr(Path, "is_symlink", lambda _: pytest.fail("Inspected source before rejecting remote mount"))
    with pytest.raises(ValueError, match="local filesystem"):
        host.local_binding_path(lock, row, root=tmp_path)


def test_same_mountpoint_with_remote_overmount_is_rejected(tmp_path):
    lock, row, _ = fixture(tmp_path)
    table = tmp_path / "proc/self/mountinfo"
    table.write_text(table.read_text() + f"2 1 0:2 / {tmp_path.as_posix()} rw - cifs remote rw\n")
    with pytest.raises(ValueError, match="local filesystem"):
        host.local_binding_path(lock, row, root=tmp_path)


def test_binding_read_is_bounded(tmp_path):
    path = tmp_path / "oversize.json"
    path.write_bytes(b" " * 16385)
    with pytest.raises(ValueError, match="small regular"):
        host.read_runtime_binding(path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO and symlink behavior")
def test_non_regular_binding_does_not_block_or_follow_symlinks(tmp_path):
    lock, row, _ = fixture(tmp_path)
    source = host.local_binding_path(lock, row, root=tmp_path)
    source.parent.mkdir(parents=True)
    os.mkfifo(source)
    with pytest.raises(ValueError, match="regular"):
        host.read_runtime_binding(source)
    source.unlink()
    source.symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError, match="symlink"):
        host.local_binding_path(lock, row, root=tmp_path)


@pytest.mark.skipif(os.name != "posix", reason="Host lifecycle uses Linux paths")
def test_create_finalizes_binding_before_start_and_missing_binding_blocks_start(tmp_path, monkeypatch):
    raw = ring_site()
    raw["workspace"] = str(tmp_path / "cluster")
    lock = installer.make_lock(PROFILE, raw, "1" * 40, "2" * 64, image_runtime=image_lock())
    row = lock["site"]["ranks"][0]
    installer.write(Path(raw["workspace"]) / ".installer-owner.json", {"deployment": lock["id"]})
    spec = installer.specifications(lock, only_rank=0)[0]
    info, image = inspection(spec)
    info["Id"] = "a" * 64
    binding_path = Path(installer_image.binding_path(lock, row))
    current = []
    events = []
    monkeypatch.setattr(node, "observation_identity", lambda **kw: {"node_id": str(uuid.UUID(int=1)), "boot_id": None, "identity_errors": {}})
    monkeypatch.setattr(host, "local_binding_path", lambda *a, **kw: binding_path)
    monkeypatch.setattr(host, "container", lambda _: current[0] if current else None)
    monkeypatch.setattr(host, "image_info", lambda _: image)
    monkeypatch.setattr(host, "admit_image", lambda _: {})
    monkeypatch.setattr(host, "verify_model", lambda *a, **kw: None)
    monkeypatch.setattr(host, "require_idle", lambda: None)
    monkeypatch.setattr(host.qwen_flash_next, "verify_model_paths", lambda *a: None)
    monkeypatch.setattr(qwen_mesh, "check", lambda *a: None)
    monkeypatch.setattr(compose, "check_project_containers", lambda *a, **kw: None)
    monkeypatch.setattr(compose, "check_equivalence", lambda *a, **kw: None)
    def run(argv, **kwargs):
        if "create" in argv:
            assert host.read_runtime_binding(binding_path) == {}
            current.append(copy.deepcopy(info))
            events.append("create")
        elif argv[:2] == ["docker", "start"]:
            assert host.read_runtime_binding(binding_path)["container_id"] == current[0]["Id"]
            events.append("start")
        else:
            pytest.fail("Unexpected host action: " + str(argv))
        return SimpleNamespace(stdout="", returncode=0)
    monkeypatch.setattr(host, "run", run)
    host.perform("create", lock, 0)
    host.perform("created", lock, 0)
    final = host.read_runtime_binding(binding_path)
    assert final["container_id"] == info["Id"]
    binding_path.unlink()
    with pytest.raises(ValueError, match="Runtime binding differs"):
        host.perform("start", lock, 0)
    assert events == ["create"]
    binding_path.write_text(json.dumps(final))
    host.perform("start", lock, 0)
    assert events == ["create", "start"]
