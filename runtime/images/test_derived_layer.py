"""CPU checks for derived installer layers: receipts, refusals and the lock."""

import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
from types import SimpleNamespace
import zipfile

import pytest

from runtime.common import installer_image
from runtime.images import derived_layer as layer

ROOT = Path(__file__).resolve().parents[2]
ADDED = layer.SITE + "fixture_hook.py"
REPLACED = layer.SITE + "vllm/v1/utils.py"
STATUS_MODULES = {"sparkring_runtime_status/__init__.py": b'__version__ = "{version}"\n',
                  "sparkring_runtime_status/collector.py": b"NAME = 'served'\n",
                  "sparkring_runtime_status/dashboard.html": b"<html></html>\n"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(value, **kwargs):
    return (json.dumps(value, indent=2, **kwargs) + "\n").encode()


def status_artifacts(version, *, collector=b"NAME = 'served'\n"):
    """A pure status wheel and its rooted source archive, built as setuptools lays them out."""
    modules = {name: raw.replace(b"{version}", version.encode()) for name, raw in STATUS_MODULES.items()}
    modules["sparkring_runtime_status/collector.py"] = collector
    project = (
        '[project]\nname = "sparkring-runtime-status"\nversion = "%s"\nrequires-python = ">=3.10"\n'
        'dependencies = ["fastapi>=0.115"]\n\n'
        '[project.entry-points."vllm.endpoint_plugins"]\n'
        'sparkring_status = "sparkring_runtime_status.plugin:StatusPlugin"\n\n'
        '[project.entry-points."vllm.general_plugins"]\n'
        'sparkring_status = "sparkring_runtime_status.plugin:register_worker_method"\n' % version).encode()
    distribution = f"sparkring_runtime_status-{version}.dist-info"
    files = dict(modules)
    files[distribution + "/METADATA"] = (
        "Metadata-Version: 2.4\nName: sparkring-runtime-status\nVersion: %s\n"
        "Requires-Python: >=3.10\nRequires-Dist: fastapi>=0.115\n" % version).encode()
    files[distribution + "/WHEEL"] = b"Wheel-Version: 1.0\nGenerator: setuptools\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    files[distribution + "/entry_points.txt"] = (
        b"[vllm.endpoint_plugins]\nsparkring_status = sparkring_runtime_status.plugin:StatusPlugin\n\n"
        b"[vllm.general_plugins]\nsparkring_status = sparkring_runtime_status.plugin:register_worker_method\n")
    files[distribution + "/top_level.txt"] = b"sparkring_runtime_status\n"
    rows = ["%s,sha256=%s,%d" % (name, base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode(),
                                 len(raw)) for name, raw in files.items()]
    files[distribution + "/RECORD"] = ("\n".join([*rows, distribution + "/RECORD,,"]) + "\n").encode()
    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, "w") as bundle:
        for name, raw in files.items():
            bundle.writestr(name, raw)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as bundle:
        for name, raw in {"pyproject.toml": project, **modules}.items():
            item = tarfile.TarInfo("runtime_status/" + name)
            item.size = len(raw)
            bundle.addfile(item, io.BytesIO(raw))
    return wheel.getvalue(), archive.getvalue(), files


@pytest.fixture
def parent(tmp_path):
    """A repository with two sources, a parent lock and the receipts of a status-0.3.1 parent."""
    repository = tmp_path / "repository"
    sources = {"integrations/fixture/fixture_hook.py": b"VALUE = 1\n",
               "integrations/fixture/utils.py": b"STAGED = True\n"}
    for relative, raw in sources.items():
        (repository / relative).parent.mkdir(parents=True, exist_ok=True)
        (repository / relative).write_bytes(raw)
    _, _, status = status_artifacts("0.3.1")
    python_root = {name: sha(raw) for name, raw in status.items()}
    base = {"schema": "sparkring-external-installed/v1", "composition_sha256": "c" * 64,
            "files": {REPLACED: "d" * 64, layer.SITE + "sparkring_transport.pth": "e" * 64,
                      **{layer.PYTHON_ROOT + "/" + name: value for name, value in python_root.items()}},
            "python_roots": {layer.PYTHON_ROOT: python_root}, "removed_files": [],
            "versions": {"fastapi": "0.136.3"},
            "capabilities": {"transport_profile": "tp2-rocenante-adaptive-prepared",
                             "transport_manifest_sha256": "3" * 64,
                             "runtime_status": {"distribution": "sparkring-runtime-status", "version": "0.3.1",
                                                "entry_points": layer.STATUS_ENTRY_POINTS,
                                                "plugin_name": "sparkring_status",
                                                "python_root": layer.PYTHON_ROOT,
                                                "wheel_sha256": "a" * 64, "source_archive_sha256": "b" * 64}}}
    base_raw = dump(base, sort_keys=True)
    toolchain = {"schema": "sparkring-toolchain-installed/v1", "variant": "combined",
                 "parent_receipt_sha256": sha(base_raw), "nvcc": "Cuda compilation tools, V13.4.92",
                 "nccl_version": 23203}
    toolchain_raw = dump(toolchain)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "base.json").write_bytes(base_raw)
    (receipts / "toolchain.json").write_bytes(toolchain_raw)
    lock = {
        "schema": installer_image.SCHEMA, "name": "dev-parent", "image_id": "sha256:" + "1" * 64,
        "image_reference": "ghcr.io/example/sparkring@sha256:" + "2" * 64,
        "parent_receipt_sha256": sha(base_raw), "toolchain_receipt_sha256": sha(toolchain_raw),
        "composition_sha256": "c" * 64, "transport_profile": "tp2-rocenante-adaptive-prepared",
        "transport_manifest_sha256": "3" * 64, "status_version": "0.3.1",
        "profiles": ["glm53-flash-nvfp4-spark-tp2", "mimo-v26-flash-rl-tp4"],
        "image_bytes": 1000, "download_bytes": 500,
    }
    lock_path = repository / "runtime/releases/dev-parent/installer-image.json"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(json.dumps(lock))
    record = {
        "schema": layer.SCHEMA, "id": "fixture-layer",
        "parent_lock": "runtime/releases/dev-parent/installer-image.json",
        "provenance": "/opt/sparkring/receipts/derived-fixture.json", "purpose": "Fixture layer.",
        "files": {
            ADDED: {"source": "integrations/fixture/fixture_hook.py", "sha256": sha(b"VALUE = 1\n"),
                    "inherited_sha256": None},
            REPLACED: {"source": "integrations/fixture/utils.py", "sha256": sha(b"STAGED = True\n"),
                       "inherited_sha256": "d" * 64},
        },
    }
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text(json.dumps(record))
    return SimpleNamespace(repository=repository, descriptor=descriptor, record=record, lock=lock, base=base,
                           base_raw=base_raw, toolchain_raw=toolchain_raw, receipts=receipts, tmp=tmp_path,
                           status=status)


def prepare(parent, output=None, descriptor=None, artifacts=None):
    return layer.prepare(descriptor or parent.descriptor, parent.repository, parent.receipts / "base.json",
                         parent.receipts / "toolchain.json", output or parent.tmp / "context", artifacts)


def rewrite(parent, change):
    record = json.loads(parent.descriptor.read_text())
    change(record)
    parent.descriptor.write_text(json.dumps(record))


def test_prepare_rerecords_both_receipts_and_lists_every_file(parent):
    result = prepare(parent)
    context = Path(result["context"])
    files = context / "files"
    assert (context / "Dockerfile").read_text() == "ARG PARENT_IMAGE\nFROM ${PARENT_IMAGE}\nCOPY files/ /\n"
    assert (files / ADDED.lstrip("/")).read_bytes() == b"VALUE = 1\n"
    assert (files / REPLACED.lstrip("/")).read_bytes() == b"STAGED = True\n"
    base_raw = (files / layer.BASE_RECEIPT.lstrip("/")).read_bytes()
    base = json.loads(base_raw)
    assert base_raw == dump(base, sort_keys=True)
    assert base["files"] == {**parent.base["files"], ADDED: sha(b"VALUE = 1\n"), REPLACED: sha(b"STAGED = True\n")}
    assert base["python_roots"] == parent.base["python_roots"]
    assert base["capabilities"] == parent.base["capabilities"]
    toolchain_raw = (files / layer.TOOLCHAIN_RECEIPT.lstrip("/")).read_bytes()
    assert json.loads(toolchain_raw)["parent_receipt_sha256"] == sha(base_raw)
    assert list(json.loads(toolchain_raw)) == list(json.loads(parent.toolchain_raw))
    provenance = json.loads((files / "opt/sparkring/receipts/derived-fixture.json").read_bytes())
    assert provenance["parent_receipt_sha256"] == sha(parent.base_raw)
    assert provenance["files"][REPLACED]["inherited_sha256"] == "d" * 64
    assert provenance["files"][ADDED]["inherited_sha256"] is None
    assert provenance["receipts"] == {layer.BASE_RECEIPT: sha(base_raw), layer.TOOLCHAIN_RECEIPT: sha(toolchain_raw)}
    assert "runtime_status" not in provenance
    plan = json.loads((context / "plan.json").read_text())
    assert plan["added"] == [ADDED] and plan["replaced"] == [REPLACED] and plan["removed"] == []
    assert plan["status_version"] == "0.3.1"
    assert plan["receipts"] == provenance["receipts"] and plan["parent_lock"] == parent.lock
    assert plan["payload_bytes"] == sum(path.stat().st_size for path in files.rglob("*") if path.is_file())


def test_prepare_refuses_an_existing_context(parent):
    (parent.tmp / "context").mkdir()
    with pytest.raises(ValueError, match="must not exist"):
        prepare(parent)


def test_prepare_requires_the_parent_locks_receipts(parent):
    (parent.receipts / "base.json").write_bytes(parent.base_raw + b"\n")
    with pytest.raises(ValueError, match="External-base receipt differs"):
        prepare(parent)


def test_prepare_requires_pinned_source_bytes(parent):
    (parent.repository / "integrations/fixture/fixture_hook.py").write_bytes(b"VALUE = 2\n")
    with pytest.raises(ValueError, match="content differs"):
        prepare(parent)


@pytest.mark.parametrize(("target", "inherited"), (
    (ADDED, "f" * 64),
    (REPLACED, None),
    (REPLACED, "f" * 64),
))
def test_prepare_requires_the_recorded_inherited_file(parent, target, inherited):
    rewrite(parent, lambda record: record["files"][target].update(inherited_sha256=inherited))
    with pytest.raises(ValueError, match="different inherited file"):
        prepare(parent)


@pytest.mark.parametrize(("target", "message"), (
    ("/opt/sparkring/python/sparkring_runtime_status/plugin.py", "under /usr/local"),
    (layer.SITE + "../escape.py", "normalized absolute path"),
    (layer.SITE + "b12x/preparation/session.py", "new transport manifest"),
    (layer.SITE + "sparkring_transport.pth", "Startup hooks"),
    (layer.SITE + "b12x/_native.so", "not native files"),
))
def test_descriptor_refuses_paths_outside_a_python_file_layer(parent, target, message):
    rewrite(parent, lambda record: record["files"].update({target: record["files"].pop(ADDED)}))
    with pytest.raises(ValueError, match=message):
        prepare(parent)


def test_derived_lock_binds_the_built_image_to_the_parent_contract(parent):
    plan = json.loads((Path(prepare(parent)["context"]) / "plan.json").read_text())
    lock = layer.derived_lock(plan, {"Id": "sha256:" + "9" * 64, "Size": 1234}, "dev-derived")
    assert lock["name"] == "dev-derived"
    assert lock["image_id"] == lock["image_reference"] == "sha256:" + "9" * 64
    assert lock["image_bytes"] == 1234 and lock["download_bytes"] == 500 + plan["payload_bytes"]
    assert lock["parent_receipt_sha256"] == plan["receipts"][layer.BASE_RECEIPT]
    assert lock["toolchain_receipt_sha256"] == plan["receipts"][layer.TOOLCHAIN_RECEIPT]
    for field in ("composition_sha256", "transport_manifest_sha256", "status_version", "profiles"):
        assert lock[field] == parent.lock[field]


def test_record_admits_every_profile_before_writing_the_lock(parent, monkeypatch):
    context = Path(prepare(parent)["context"])
    commands, admitted = [], []

    def run(command, text=True):
        commands.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, json.dumps([{"Id": "sha256:" + "9" * 64, "Size": 7}]))
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setattr(installer_image, "admit",
                        lambda lock, *, run, profile: admitted.append((lock["name"], profile)))
    output = parent.tmp / "lock.json"
    result = layer.record(context, "sha256:" + "9" * 64, "dev-derived", output, run=run)
    assert admitted == [("dev-derived", profile) for profile in parent.lock["profiles"]]
    assert commands[0][-2:] == ["sh", ADDED] and parent.lock["image_id"] in commands[0]
    assert json.loads(output.read_text())["image_id"] == result["image_id"] == "sha256:" + "9" * 64
    assert result["serving_qualified"] is False


def test_record_refuses_an_addition_the_parent_already_has(parent):
    context = Path(prepare(parent)["context"])

    def run(command, text=True):
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(ValueError, match="already has an added path"):
        layer.record(context, "sha256:" + "9" * 64, "dev-derived", parent.tmp / "lock.json", run=run)
    assert not (parent.tmp / "lock.json").exists()


def with_status(parent, version="0.3.2", **kwargs):
    """Add a pinned status replacement to the descriptor; return the artifact directory."""
    wheel, archive, files = status_artifacts(version, **kwargs)
    artifacts = parent.tmp / "status-artifacts"
    artifacts.mkdir(exist_ok=True)
    names = {"wheel": f"sparkring_runtime_status-{version}-py3-none-any.whl",
             "source_archive": f"sparkring-runtime-status-{version}-source.tar"}
    (artifacts / names["wheel"]).write_bytes(wheel)
    (artifacts / names["source_archive"]).write_bytes(archive)
    rewrite(parent, lambda record: record.update(runtime_status={
        "wheel": {"path": names["wheel"], "sha256": sha(wheel)},
        "source_archive": {"path": names["source_archive"], "sha256": sha(archive)}}))
    return artifacts, files, sha(wheel), sha(archive)


def image_tree(parent, root):
    """The parent's owned Python root as installed files under ``root``."""
    for name, raw in parent.status.items():
        path = root / layer.PYTHON_ROOT.lstrip("/") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)


def apply_layer(context, root):
    """Apply the layer as Docker does: the removal step, then the copied files."""
    plan = json.loads((context / "plan.json").read_text())
    for name in plan["removed"]:
        (root / name.lstrip("/")).unlink()
    for line in (context / "Dockerfile").read_text().splitlines():
        if " && rmdir -- " in line:
            for directory in line.split(" && rmdir -- ")[1].split():
                (root / directory.lstrip("/")).rmdir()
    shutil.copytree(context / "files", root, dirs_exist_ok=True)


def verify_owned_python(receipt, root):
    """external-base.py ``verify``: owned-root inventory, recorded files and removed paths."""
    owned = root / layer.PYTHON_ROOT.lstrip("/")
    inventory = {"/" + path.relative_to(root).as_posix(): sha(path.read_bytes())
                 for path in sorted(owned.rglob("*")) if path.is_file()}
    expected = {layer.PYTHON_ROOT + "/" + relative: value
                for relative, value in receipt["python_roots"].get(layer.PYTHON_ROOT, {}).items()}
    assert inventory == expected
    assert {name: value for name, value in receipt["files"].items()
            if PurePosixPath(name).is_relative_to(layer.PYTHON_ROOT)} == expected
    for name in receipt["removed_files"]:
        assert not (root / name.lstrip("/")).exists()


def test_status_replacement_passes_the_images_verify_and_the_installers_admission(parent):
    artifacts, files, wheel_sha, archive_sha = with_status(parent, collector=b"NAME = 'served_model_name'\n")
    result = prepare(parent, artifacts=artifacts)
    context = Path(result["context"])
    assert result["status_version"] == "0.3.2"
    old = f"{layer.PYTHON_ROOT}/sparkring_runtime_status-0.3.1.dist-info"
    removed = sorted(f"{old}/{name}" for name in ("METADATA", "RECORD", "WHEEL", "entry_points.txt", "top_level.txt"))
    plan = json.loads((context / "plan.json").read_text())
    assert plan["removed"] == removed and plan["status_version"] == "0.3.2"
    assert (context / "Dockerfile").read_text().splitlines()[2] == (
        "RUN rm -f -- " + " ".join(removed) + " && rmdir -- " + old)
    receipt = json.loads((context / "files" / layer.BASE_RECEIPT.lstrip("/")).read_bytes())
    assert receipt["python_roots"] == {layer.PYTHON_ROOT: {name: sha(raw) for name, raw in files.items()}}
    assert receipt["removed_files"] == removed
    status = receipt["capabilities"]["runtime_status"]
    assert (status["version"], status["wheel_sha256"], status["source_archive_sha256"]) == (
        "0.3.2", wheel_sha, archive_sha)
    assert status["entry_points"] == layer.STATUS_ENTRY_POINTS
    assert receipt["composition_sha256"] == parent.base["composition_sha256"]
    provenance = json.loads((context / "files/opt/sparkring/receipts/derived-fixture.json").read_bytes())
    assert provenance["runtime_status"]["inherited_version"] == "0.3.1"
    assert provenance["runtime_status"]["removed"] == removed

    tree = parent.tmp / "image"
    image_tree(parent, tree)
    verify_owned_python(parent.base, tree)
    apply_layer(context, tree)
    verify_owned_python(receipt, tree)
    assert not (tree / old.lstrip("/")).exists()

    receipts = {path: (context / "files" / path.lstrip("/")).read_bytes()
                for path in (layer.BASE_RECEIPT, layer.TOOLCHAIN_RECEIPT)}
    image_id = "sha256:" + "9" * 64
    lock = layer.derived_lock(plan, {"Id": image_id, "Size": 7}, "dev-derived-status032")
    assert lock["status_version"] == "0.3.2"

    def run(command, text=True):
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, json.dumps([{
                "Id": image_id, "Os": "linux", "Architecture": "arm64",
                "Config": {"Entrypoint": list(installer_image.ENTRYPOINT)}}]))
        if "/bin/cat" in command:
            return subprocess.CompletedProcess(command, 0, receipts[command[-1]])
        return subprocess.CompletedProcess(command, 0, "")

    for profile in lock["profiles"]:
        observation = installer_image.admit(lock, run=run, profile=profile)
        assert observation["parent_receipt_sha256"] == lock["parent_receipt_sha256"]


@pytest.mark.parametrize(("change", "message"), (
    (lambda record: record["runtime_status"]["wheel"].update(sha256="0" * 64), "differs from its pin"),
    (lambda record: record["runtime_status"]["source_archive"].update(sha256="0" * 64), "differs from its pin"),
))
def test_status_artifacts_must_match_their_pins(parent, change, message):
    artifacts, *_ = with_status(parent)
    rewrite(parent, change)
    with pytest.raises(ValueError, match=message):
        prepare(parent, artifacts=artifacts)


def test_status_wheel_must_carry_the_archived_source(parent):
    artifacts, *_ = with_status(parent)
    wheel, _, _ = status_artifacts("0.3.2", collector=b"NAME = 'other'\n")
    record = json.loads(parent.descriptor.read_text())
    (artifacts / record["runtime_status"]["wheel"]["path"]).write_bytes(wheel)
    rewrite(parent, lambda record: record["runtime_status"]["wheel"].update(sha256=sha(wheel)))
    with pytest.raises(ValueError, match="package differs from its pinned source archive"):
        prepare(parent, artifacts=artifacts)


@pytest.mark.parametrize(("version", "message"), (
    ("0.3.1", "must differ from the parent"),
    ("0.4.0", "0.3.x, which the installer admits"),
))
def test_status_version_must_change_within_the_installers_range(parent, version, message):
    artifacts, *_ = with_status(parent, version)
    with pytest.raises(ValueError, match=message):
        prepare(parent, artifacts=artifacts)


def test_status_replacement_requires_its_artifacts(parent):
    with_status(parent)
    with pytest.raises(ValueError, match="requires --status-artifacts"):
        prepare(parent)


def test_repository_descriptors_pin_their_current_sources():
    descriptors = sorted((ROOT / "runtime/images/compositions").glob("*/descriptor.json"))
    derived = [path for path in descriptors if json.loads(path.read_text())["schema"] == layer.SCHEMA]
    assert derived
    for path in derived:
        record = layer.descriptor(path)
        layer._parent_lock(record, ROOT)
        for row in record.get("files", {}).values():
            layer.pinned_bytes((ROOT / row["source"]).read_bytes(), row["sha256"])
