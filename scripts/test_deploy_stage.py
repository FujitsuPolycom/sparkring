import io
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

import pytest

from scripts import deploy_stage as module
from scripts.test_deploy_runtime import prepared
from scripts.test_deploy_suite import configured_inventory


def test_shared_key_and_epoch_are_persistent(tmp_path):
    first = module.prepare_secrets(tmp_path / "private")
    key = (tmp_path / "private/health.key").read_bytes()
    assert len(key) == 32
    assert module.prepare_secrets(tmp_path / "private") == first
    assert (tmp_path / "private/health.key").read_bytes() == key
    (tmp_path / "private/epoch.txt").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        module.prepare_secrets(tmp_path / "private")


def test_source_archive_checks_tracked_payload_and_extracts(tmp_path):
    root = tmp_path / "source"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/deploy_stage.py").write_text("# source fixture\n")
    (root / "scripts/secret.tmp").write_text("untracked")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "add", "scripts/deploy_stage.py"], check=True
    )
    receipt = module.source_archive(root, tmp_path / "archive.tar.gz")
    assert list(receipt["files"]) == ["scripts/deploy_stage.py"]
    module.extract_source(
        tmp_path / "archive.tar.gz", tmp_path / "unpacked", receipt["files"]
    )
    assert (
        tmp_path / "unpacked/scripts/deploy_stage.py"
    ).read_text() == "# source fixture\n"
    with pytest.raises(FileExistsError):
        module.extract_source(
            tmp_path / "archive.tar.gz", tmp_path / "unpacked", receipt["files"]
        )


def test_archive_traversal_rejected(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("../escape")
        member.size = 1
        tar.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe"):
        module.extract_source(archive, tmp_path / "output", {"../escape": "bad"})
    assert not (tmp_path / "output").exists()


@pytest.mark.skipif(os.name != "posix", reason="staging controller targets Linux/WSL")
@pytest.mark.parametrize(
    "failure", [None, "ownership", "copy", "canonical", "launch", "source"]
)
def test_stage_sequence_uses_one_download_and_never_starts_model(
    tmp_path, monkeypatch, failure
):
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts/deploy_stage.py").write_text("# fixture")

    def pack(root, output):
        with tarfile.open(output, "w:gz") as archive:
            archive.add(
                source / "scripts/deploy_stage.py", arcname="scripts/deploy_stage.py"
            )
        return {
            "sha256": module.sha(output),
            "files": {
                "scripts/deploy_stage.py": module.sha(
                    source / "scripts/deploy_stage.py"
                )
            },
        }

    monkeypatch.setattr(module, "source_archive", pack)
    pub = json.loads((module.PROFILE / "public-image.json").read_text())
    pins = json.loads((module.PROFILE / "pins.json").read_text())
    files = {
        "config.json": pins["target"]["config_sha256"],
        "model.safetensors.index.json": pins["target"]["index_sha256"],
        "weights.bin": "b" * 64,
    }
    preparation = prepared()
    from scripts.deploy_suite import lifecycle_capabilities

    preparation["lifecycle_capabilities"] = lifecycle_capabilities(module.PROFILE)
    facts = configured_inventory(preparation["spec"])["hosts"]
    launch_content = {
        name: b"{}" for name in module.LAUNCH_FILES - {"fabric-plan.json"}
    }
    launch_content["fabric-plan.json"] = json.dumps(
        {
            "files": {
                name: hashlib.sha256(value).hexdigest()
                for name, value in launch_content.items()
            }
        }
    ).encode()
    expected_launch = {
        name: hashlib.sha256(value).hexdigest()
        for name, value in launch_content.items()
    }

    class Fake:
        def __init__(self):
            self.calls = []
            self.files = {"spark-r0": files.copy()}
            self.keys = []

        def remote(self, host, argv, *, input=None, timeout=0):
            self.calls.append((host, argv))
            if (
                argv[:3] == ["sudo", "-n", "python3"]
                and "def _workspace_local" in argv[-1]
            ):
                if (
                    failure == "ownership"
                    and host == "spark-r3"
                    and "'check',None" in argv[-1]
                ):
                    raise ValueError("Workspace has no matching ownership record")
                return json.dumps({"checked": True, "uid": 1000, "gid": 1000})
            if input is not None:
                self.keys.append(input)
            if "finish-host" in " ".join(argv) and "--workspace" in " ".join(argv):
                expected = dict(expected_launch)
                if failure == "canonical" and host == "spark-r2":
                    expected["rank2.env"] = "0" * 64
                return json.dumps({"canonical_launch_files": expected})
            if argv[:2] == ["python3", "-c"] and "def _collect_local" in argv[2]:
                return json.dumps(facts[host])
            if "base64.b64encode" in " ".join(argv):
                values = dict(launch_content)
                if failure == "launch" and host == "spark-r3":
                    values["rank3.env"] = b"changed"
                    record = json.loads(values["fabric-plan.json"])
                    record["files"]["rank3.env"] = hashlib.sha256(
                        values["rank3.env"]
                    ).hexdigest()
                    values["fabric-plan.json"] = json.dumps(record).encode()
                return json.dumps(
                    {
                        name: base64.b64encode(value).decode()
                        for name, value in values.items()
                    }
                )
            if argv[:2] == ["sha256sum", "/srv/sparkring/test-mesh/image.tar"]:
                return "a" * 64 + " image.tar"
            if argv[:3] == ["docker", "image", "inspect"]:
                return pub["config_image_id"]
            if argv[:2] == ["python3", "-c"] and "rows={}" in argv[2]:
                return json.dumps(self.files.get(host, {}))
            return ""

        def copy(self, source, destination):
            self.calls.append(("copy", [str(source), destination]))
            if failure == "copy":
                raise TimeoutError("copy may still be running remotely")

        def copy_verified(self, source, host, destination, digest):
            self.files.setdefault(host, {})[Path(destination).name] = digest

    fake = Fake()
    if failure == "source":
        snapshot = tmp_path / "state/source"
        (snapshot / "scripts").mkdir(parents=True)
        (snapshot / "scripts/deploy_stage.py").write_text("# fixture")
        (snapshot / "runpy.py").write_text("unreviewed shadow module")
    if failure:
        with pytest.raises((ValueError, TimeoutError)):
            module.stage(preparation, tmp_path / "state", run=fake, source_root=source)
        commands = [
            args[-1] for _, args in fake.calls if args[:3] == ["sudo", "-n", "python3"]
        ]
        assert not any("'release'" in command for command in commands)
        if failure == "ownership":
            assert not any("'acquire'" in command for command in commands)
            assert not any(host == "copy" for host, _ in fake.calls)
        else:
            assert (tmp_path / "state/remote-operation.json").is_file()
        return
    result = module.stage(preparation, tmp_path / "state", run=fake, source_root=source)
    assert len([a for _, a in fake.calls if a[:2] == ["docker", "pull"]]) == 1
    runs = [a for _, a in fake.calls if a[:2] == ["docker", "run"]]
    assert (
        len(runs) == 1
        and "--gpus" not in runs[0]
        and "snapshot_download" in runs[0][-1]
    )
    assert not any(a[:2] == ["docker", "start"] for _, a in fake.calls)
    assert len(fake.keys) == 4 and len(set(fake.keys)) == 1 and len(fake.keys[0]) == 32
    assert result["model_files"] == files
    assert result["launch_files"] == expected_launch
    assert (
        Path(result["controller_source"])
        .joinpath("scripts/deploy_stage.py")
        .read_text()
        == "# fixture"
    )
    assert all(fake.files[h["host"]] == files for h in result["spec"]["hosts"])
    assert runs[0][runs[0].index("--user") + 1] == "1000:1000"
    assert "HF_HUB_OFFLINE=0" in runs[0]
    assert "NVIDIA_VISIBLE_DEVICES=void" in runs[0]
    ownership_checks = [
        (index, host, args[-1])
        for index, (host, args) in enumerate(fake.calls)
        if args[:3] == ["sudo", "-n", "python3"] and "def _workspace_local" in args[-1]
    ]
    assert all("'check',None" in code for _, _, code in ownership_checks[:4])
    assert len({host for _, host, _ in ownership_checks[:4]}) == 4
    assert all("'acquire'" in code for _, _, code in ownership_checks[4:8])
    assert not (tmp_path / "state/remote-operation.json").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership and symlink behavior")
@pytest.mark.parametrize(
    "relative",
    [
        "models",
        "models/revision",
        "artifacts",
        "cache",
        "source",
        "image.tar",
        "preparation.json",
        "receipts",
    ],
)
def test_workspace_check_rejects_nested_symlinks_before_lock_or_write(
    tmp_path, relative
):
    workspace = tmp_path / "mesh"
    workspace.mkdir()
    (workspace / "deployment-owner.json").write_text(json.dumps({"owner": "mesh"}))
    outside = tmp_path / "outside"
    outside.mkdir()
    target = workspace / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module._workspace_local(str(workspace), "mesh", "check", base=str(tmp_path))
    assert not (workspace / ".stage-operation").exists()
    assert list(outside.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership and symlink behavior")
def test_workspace_rejects_symlinked_ancestor_and_shared_hardlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module._validate_staging_tree(linked / "uncreated")
    (real / "file").write_bytes(b"keep")
    os.link(real / "file", tmp_path / "shared")
    with pytest.raises(ValueError, match="hard links"):
        module._validate_staging_tree(real)


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership and locks")
def test_remote_workspace_lock_requires_exact_owner_and_survives_other_tokens(tmp_path):
    root = tmp_path / "mesh"

    def call(operation, token=None):
        return module._workspace_local(
            str(root), "mesh", operation, token, base=str(tmp_path)
        )

    assert call("check")["checked"]
    assert not root.exists()
    call("acquire", "a" * 32)
    with pytest.raises(ValueError, match="active or interrupted"):
        call("check")
    with pytest.raises(ValueError, match="another operation"):
        call("release", "b" * 32)
    assert (root / ".stage-operation/owner.json").is_file()
    assert call("verify", "a" * 32)["checked"]
    call("release", "a" * 32)
    assert call("check")["checked"]
    (root / "deployment-owner.json").write_text(json.dumps({"owner": "other"}))
    with pytest.raises(ValueError, match="ownership"):
        call("acquire", "c" * 32)


def test_staging_rejects_local_nested_symlinks_before_remote_work(
    tmp_path, monkeypatch
):
    state = tmp_path / "state"
    state.mkdir()
    external = tmp_path / "external"
    external.write_text("keep")
    try:
        (state / "source.tar.gz").symlink_to(external)
    except OSError:
        pytest.skip("symlink creation unavailable")
    monkeypatch.setattr(
        module, "_stage", lambda *a, **k: pytest.fail("remote work started")
    )
    with pytest.raises(ValueError, match="symlink"):
        module.stage({}, state)
    assert external.read_text() == "keep"


@pytest.mark.skipif(os.name != "posix", reason="POSIX staging controller")
def test_staging_rejects_changed_lifecycle_before_remote_work(tmp_path, monkeypatch):
    from scripts import deploy_suite

    monkeypatch.setattr(deploy_suite, "require_verified_network", lambda value: None)
    monkeypatch.setattr(
        deploy_suite, "lifecycle_capabilities", lambda profile: ["memory-startup"]
    )

    class NoRemote:
        def remote(self, *args, **kwargs):
            pytest.fail("remote work started before lifecycle check")

    with pytest.raises(ValueError, match="lifecycle"):
        module.stage({"lifecycle_capabilities": []}, tmp_path, run=NoRemote())


def test_launch_copy_rejects_traversal_or_bad_hash(tmp_path):
    with pytest.raises(ValueError, match="filename"):
        module.install_launch_copy(tmp_path / "launch", {"../secret": "YQ=="})
    data = {name: "YQ==" for name in module.LAUNCH_FILES - {"fabric-plan.json"}}
    data["fabric-plan.json"] = base64.b64encode(
        json.dumps({"files": {name: "f" * 64 for name in data}}).encode()
    ).decode()
    with pytest.raises(ValueError, match="checksum"):
        module.install_launch_copy(tmp_path / "launch", data)


def test_staging_lock_prevents_overlapping_execution(tmp_path, monkeypatch):
    calls = []

    def fake_stage(*args, **kwargs):
        calls.append(args)
        with pytest.raises(ValueError, match="active or interrupted"):
            module.stage({}, tmp_path)
        return {"prepared": True}

    monkeypatch.setattr(module, "_stage", fake_stage)
    assert module.stage({}, tmp_path) == {"prepared": True}
    assert len(calls) == 1
    assert not (tmp_path / "stage.lock").exists()
    (tmp_path / "stage.lock").write_text("interrupted process")
    with pytest.raises(ValueError, match="active or interrupted"):
        module.stage({}, tmp_path)
    assert (tmp_path / "stage.lock").read_text() == "interrupted process"


def verified_workspace(path):
    source = path / "source/runtime/glm53-spark-mtp3-mesh/image-receipt.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}")
    site = {"topology_file": "fabric.json", "container_prefix": "reviewed"}
    fabric = {"schema": "fixture"}
    document = {
        "spec": {"workspace": str(path), "site": site, "fabric": fabric},
        "source": {"files": {}},
    }
    launch = path / "launch"
    launch.mkdir()
    for name, value in (("site.json", site), ("fabric.json", fabric)):
        (path / name).write_text(json.dumps(value))
        (launch / name).write_text(json.dumps(value))
    for name in module.LAUNCH_FILES - {"site.json", "fabric.json", "fabric-plan.json"}:
        (launch / name).write_text("reviewed source\n")
    record = {
        "files": {
            name: module.sha(launch / name)
            for name in module.LAUNCH_FILES - {"fabric-plan.json"}
        },
        "site_sha256": module.sha(path / "site.json"),
        "topology_sha256": module.sha(path / "fabric.json"),
        "image_receipt_sha256": module.sha(source),
    }
    (launch / "fabric-plan.json").write_text(json.dumps(record))
    document["launch_files"] = {
        name: module.sha(launch / name) for name in module.LAUNCH_FILES
    }
    (path / "preparation.json").write_text(json.dumps(document))
    return document, record


def test_host_verification_binds_render_to_reviewed_inputs(tmp_path):
    document, record = verified_workspace(tmp_path)
    module.verify_host(tmp_path)
    # A valid older launch is not valid for a different reviewed site.
    document["spec"]["site"]["container_prefix"] = "different"
    (tmp_path / "preparation.json").write_text(json.dumps(document))
    (tmp_path / "site.json").write_text(json.dumps(document["spec"]["site"]))
    with pytest.raises(ValueError, match="different deployment inputs"):
        module.verify_host(tmp_path)
    record["site_sha256"] = module.sha(tmp_path / "site.json")
    (tmp_path / "launch/fabric-plan.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="approved preparation"):
        module.verify_host(tmp_path)


@pytest.mark.parametrize("attack", ["rewrite", "omit", "extra", "unanchored"])
def test_approved_launch_rejects_mutated_environment_and_self_consistent_manifest(
    tmp_path, attack
):
    document, record = verified_workspace(tmp_path)
    approved = (tmp_path / "preparation.json").read_bytes()
    if attack == "rewrite":
        (tmp_path / "launch/rank0.env").write_text("EXECUTE_UNREVIEWED=1\n")
        record["files"]["rank0.env"] = module.sha(tmp_path / "launch/rank0.env")
    elif attack == "omit":
        del record["files"]["rank0.env"]
    elif attack == "extra":
        (tmp_path / "launch/injected.py").write_text("unreviewed\n")
    else:
        del document["launch_files"]
        (tmp_path / "preparation.json").write_text(json.dumps(document))
    (tmp_path / "launch/fabric-plan.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="launch"):
        module.verify_host(tmp_path)
    if attack != "unanchored":
        assert (tmp_path / "preparation.json").read_bytes() == approved


def test_host_verification_rejects_source_manifest_escape(tmp_path):
    document, _ = verified_workspace(tmp_path)
    document["source"]["files"]["../outside"] = "0" * 64
    (tmp_path / "preparation.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="manifest path"):
        module.verify_host(tmp_path)


def test_render_rejects_changed_site_before_extracting_artifacts(tmp_path, monkeypatch):
    verified_workspace(tmp_path)
    (tmp_path / "site.json").write_text(json.dumps({"unreviewed": True}))
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda *a, **k: pytest.fail("Docker ran before checking render inputs"),
    )
    with pytest.raises(ValueError, match="render inputs"):
        module.finish_host(tmp_path)
