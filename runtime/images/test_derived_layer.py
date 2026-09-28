"""CPU checks for derived installer layers: receipts, refusals and the lock."""

import base64
import dataclasses
import functools
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import sys
from types import ModuleType, SimpleNamespace
import zipfile

import pytest

from runtime.common import installer_image
from runtime.images import derive_mimo_vision, derive_staging_fix, derive_tool_choice_contract, derive_tp2_hc
from runtime.images import derive_transport_peer_wait, derive_transport_window
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


# Code-defined layers: bytes computed from the parent's own files.

CODE_PARENT = "sha256:" + "a" * 64


def code_parent(tmp_path, files, capabilities=None):
    """An exported parent root whose receipts record every given file, and its lock."""
    root = tmp_path / "parent"
    for path, data in files.items():
        target = root / path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    receipt = {"schema": "sparkring-external-installed/v1", "composition_sha256": "c" * 64,
               "capabilities": {"transport_profile": "tp2-rocenante-adaptive-prepared",
                                "transport_manifest_sha256": "3" * 64,
                                "runtime_status": {"version": "0.3.1"}, **(capabilities or {})},
               "files": {path: sha(data) for path, data in files.items()}}
    base_raw = layer.canonical_json(receipt)
    toolchain_raw = layer.canonical_json({"variant": "combined", "parent_receipt_sha256": sha(base_raw)},
                                         sort_keys=False)
    for path, data in ((layer.BASE_RECEIPT, base_raw), (layer.TOOLCHAIN_RECEIPT, toolchain_raw)):
        target = root / path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    lock = {"schema": installer_image.SCHEMA, "name": "dev-parent", "image_id": CODE_PARENT,
            "image_reference": CODE_PARENT, "parent_receipt_sha256": sha(base_raw),
            "toolchain_receipt_sha256": sha(toolchain_raw), "composition_sha256": "c" * 64,
            "transport_profile": "tp2-rocenante-adaptive-prepared",
            "transport_manifest_sha256": receipt["capabilities"]["transport_manifest_sha256"],
            "status_version": "0.3.1", "profiles": ["glm53-flash-nvfp4-spark-tp2"],
            "image_bytes": 1000, "download_bytes": 500}
    return root, lock


def context_file(context, path):
    return (Path(context) / "files" / path.lstrip("/")).read_bytes()


def simple_layer(**changes):
    code = layer.Layer(name="fixture", purpose="fixture layer", provenance="/opt/sparkring/receipts/derived-code.json",
                       replace=lambda read, receipt: {"/opt/app/a.py": read("/opt/app/a.py") + b"# changed\n"})
    return dataclasses.replace(code, **changes)


def test_code_layer_rewrites_receipts_from_the_parents_files(tmp_path):
    root, lock = code_parent(tmp_path, {"/opt/app/a.py": b"a\n", "/opt/app/b.py": b"b\n"})
    result = layer.prepare_layer(simple_layer(), lock, layer.root_reader(root), tmp_path / "context")
    context = Path(result["context"])
    base_raw = context_file(context, layer.BASE_RECEIPT)
    base = json.loads(base_raw)
    assert base["files"] == {"/opt/app/a.py": sha(b"a\n# changed\n"), "/opt/app/b.py": sha(b"b\n")}
    toolchain_raw = context_file(context, layer.TOOLCHAIN_RECEIPT)
    assert json.loads(toolchain_raw)["parent_receipt_sha256"] == sha(base_raw)
    assert list(json.loads(toolchain_raw)) == ["variant", "parent_receipt_sha256"]
    provenance = json.loads(context_file(context, "/opt/sparkring/receipts/derived-code.json"))
    # Code layers keep the provenance format of the published chain: no id,
    # parent release or repository source.
    assert set(provenance) == {"schema", "purpose", "parent_image_id", "parent_receipt_sha256", "files", "receipts"}
    assert provenance["files"] == {"/opt/app/a.py": {"inherited_sha256": sha(b"a\n"), "sha256": sha(b"a\n# changed\n")}}
    assert provenance["receipts"] == result["receipts"]
    assert (context / "Dockerfile").read_text() == "ARG PARENT_IMAGE\nFROM ${PARENT_IMAGE}\nCOPY files/ /\n"
    assert not (context / "files/opt/app/b.py").exists()
    plan = json.loads((context / "plan.json").read_text())
    assert plan["replaced"] == ["/opt/app/a.py"] and plan["added"] == [] and plan["lock_fields"] == {}
    with pytest.raises(ValueError, match="must not exist"):
        layer.prepare_layer(simple_layer(), lock, layer.root_reader(root), context)


@pytest.mark.parametrize("change, message", [
    ({"replace": lambda read, receipt: {"/opt/app/unrecorded.py": b"x"}}, "not recorded"),
    ({"replace": lambda read, receipt: {}}, "replaces no file"),
    ({"pins": {"/opt/app/a.py": ("0" * 64, "1" * 64)}}, "pinned"),
    ({"pins": {"/opt/app/b.py": (sha(b"b\n"), sha(b"b\n"))}}, "were not written"),
])
def test_code_layer_refuses_unbound_replacements_without_output(tmp_path, change, message):
    root, lock = code_parent(tmp_path, {"/opt/app/a.py": b"a\n", "/opt/app/b.py": b"b\n"})
    with pytest.raises(ValueError, match=message):
        layer.prepare_layer(simple_layer(**change), lock, layer.root_reader(root), tmp_path / "context")
    assert not (tmp_path / "context").exists()


POLICY = layer.SITE + "vllm/fixture_policy.py"


def adding(path, data=b"POLICY = 1\n", pin=None):
    """A code layer that replaces /opt/app/a.py and adds ``path``, pinned unless ``pin`` is False."""
    def replace(read, receipt):
        return {"/opt/app/a.py": read("/opt/app/a.py") + b"# changed\n", path: data}
    pins = {} if pin is False else {path: pin or (None, sha(data))}
    return simple_layer(replace=replace, pins=pins)


def test_code_layer_adds_a_pinned_site_packages_file(tmp_path):
    root, lock = code_parent(tmp_path, {"/opt/app/a.py": b"a\n"})
    result = layer.prepare_layer(adding(POLICY), lock, layer.root_reader(root), tmp_path / "context")
    context = Path(result["context"])
    assert context_file(context, POLICY) == b"POLICY = 1\n"
    assert json.loads(context_file(context, layer.BASE_RECEIPT))["files"][POLICY] == sha(b"POLICY = 1\n")
    provenance = json.loads(context_file(context, "/opt/sparkring/receipts/derived-code.json"))
    assert provenance["files"][POLICY] == {"inherited_sha256": None, "sha256": sha(b"POLICY = 1\n")}
    plan = json.loads((context / "plan.json").read_text())
    assert plan["added"] == [POLICY] and plan["replaced"] == ["/opt/app/a.py"]


@pytest.mark.parametrize("path, pin, message", [
    (layer.SITE + "vllm/present.py", None, "already recorded"),
    (layer.SITE + "vllm/unpinned.py", False, "not recorded by the parent"),
    (POLICY, (None, "0" * 64), "pinned"),
    ("/opt/app/new.py", None, "normalized absolute path"),
    (layer.SITE + "vllm/_C.abi3.so", None, "not native files"),
    (layer.SITE + "sparkring_transport.pth", None, "Startup hooks"),
])
def test_code_layer_refuses_unpinned_or_unowned_additions(tmp_path, path, pin, message):
    root, lock = code_parent(tmp_path, {"/opt/app/a.py": b"a\n", layer.SITE + "vllm/present.py": b"p\n"})
    with pytest.raises(ValueError, match=message):
        layer.prepare_layer(adding(path, pin=pin), lock, layer.root_reader(root), tmp_path / "context")
    assert not (tmp_path / "context").exists()


def test_code_layer_refuses_a_parent_file_or_receipt_that_differs(tmp_path):
    root, lock = code_parent(tmp_path, {"/opt/app/a.py": b"a\n"})
    (root / "opt/app/a.py").write_bytes(b"edited after installation\n")
    with pytest.raises(ValueError, match="differs from its receipt"):
        layer.prepare_layer(simple_layer(), lock, layer.root_reader(root), tmp_path / "context")
    with pytest.raises(ValueError, match="External-base receipt differs"):
        layer.prepare_layer(simple_layer(), dict(lock, parent_receipt_sha256="0" * 64), layer.root_reader(root),
                            tmp_path / "context")


def test_readers_are_strict(tmp_path):
    with pytest.raises(ValueError, match="normalized"):
        layer.root_reader(tmp_path)("/opt/../etc/passwd")
    base, toolchain = tmp_path / "base.json", tmp_path / "toolchain.json"
    base.write_bytes(b"{}")
    toolchain.write_bytes(b"{}")
    read = layer.receipt_reader(base, toolchain)
    assert read(layer.BASE_RECEIPT) == b"{}"
    with pytest.raises(ValueError, match="reads parent files"):
        read("/opt/app/a.py")


def test_build_tags_the_parent_and_records_the_image(tmp_path, monkeypatch):
    root, lock = code_parent(tmp_path, {"/opt/app/a.py": b"a\n"})
    result = layer.prepare_layer(simple_layer(), lock, layer.root_reader(root), tmp_path / "context")
    commands, admitted = [], []
    built = "sha256:" + "b" * 64

    def run(command, text=True):
        commands.append(command)
        stdout = json.dumps([{"Id": built, "Size": 123}]) if command[1:3] == ["image", "inspect"] else ""
        return subprocess.CompletedProcess(command, 0, stdout)

    monkeypatch.setattr(installer_image, "admit",
                        lambda value, *, run, profile: admitted.append((value["name"], profile)))
    output = tmp_path / "lock.json"
    summary = layer.build(tmp_path / "context", "sparkring:derived", "dev-derived", output, run=run,
                          profiles=["mimo-v26-flash-rl-tp4", "glm53-flash-nvfp4-spark-tp2"])
    tag = "sparkring-dev/parent:" + "a" * 12
    assert commands[0] == ["docker", "tag", CODE_PARENT, tag]
    assert commands[1][:2] == ["docker", "build"] and "PARENT_IMAGE=" + tag in commands[1]
    assert admitted == [("dev-derived", "glm53-flash-nvfp4-spark-tp2"), ("dev-derived", "mimo-v26-flash-rl-tp4")]
    written = json.loads(output.read_text())
    assert written["profiles"] == ["glm53-flash-nvfp4-spark-tp2", "mimo-v26-flash-rl-tp4"]
    assert written["image_id"] == summary["image_id"] == built and written["image_bytes"] == 123
    assert written["parent_receipt_sha256"] == result["receipts"][layer.BASE_RECEIPT]


def transport_fixture(tmp_path, proxy=b"// paced proxy\n"):
    source = tmp_path / "bundle"
    files = {"LICENSE": b"license\n", "roce/api.py": b"api\n", "roce/_roce_proxy.c": proxy}
    for name, data in files.items():
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_bytes(data)
    (source / "manifest.json").write_text(json.dumps({"name": derive_transport_window.PROFILE,
                                                      "files": {name: sha(data) for name, data in files.items()}}))
    installed_files = dict(files, **{"roce/_roce_proxy.c": b"// unpaced proxy\n"})
    installed = layer.canonical_json({"name": derive_transport_window.PROFILE, "dependencies": {"b12x": "pinned"},
                                      "files": {name: sha(data) for name, data in installed_files.items()}})
    parent = {f"{derive_transport_window.BUNDLE}/{name}": data for name, data in installed_files.items()}
    parent[derive_transport_window.BUNDLE + "/manifest.json"] = installed
    capabilities = {"transport_profile": derive_transport_window.PROFILE, "transport_manifest_sha256": sha(installed)}
    return (source, *code_parent(tmp_path, parent, capabilities))


def test_transport_window_replaces_changed_bundle_files_and_names_the_manifest(tmp_path):
    source, root, lock = transport_fixture(tmp_path)
    code = dataclasses.replace(derive_transport_window.LAYER,
                               replace=functools.partial(derive_transport_window.replace, source=source))
    layer.prepare_layer(code, lock, layer.root_reader(root), tmp_path / "context")
    context = tmp_path / "context"
    plan = json.loads((context / "plan.json").read_text())
    manifest_path = derive_transport_window.BUNDLE + "/manifest.json"
    assert plan["replaced"] == [manifest_path, derive_transport_window.BUNDLE + "/roce/_roce_proxy.c"]
    manifest = json.loads(context_file(context, manifest_path))
    assert manifest["dependencies"] == {"b12x": "pinned"}
    assert manifest["files"]["roce/_roce_proxy.c"] == sha(b"// paced proxy\n")
    new_manifest = sha(context_file(context, manifest_path))
    base = json.loads(context_file(context, layer.BASE_RECEIPT))
    assert base["capabilities"]["transport_manifest_sha256"] == new_manifest
    assert plan["lock_fields"] == {"transport_manifest_sha256": new_manifest}
    assert [path.name for path in (context / "files/opt/sparkring/receipts").iterdir()] == [
        "external-base-installed.json"]
    derived = layer.derived_lock(plan, {"Id": "sha256:" + "9" * 64, "Size": 7}, "dev-window")
    assert derived["transport_manifest_sha256"] == new_manifest


def test_transport_window_refuses_unchanged_or_different_bundles(tmp_path):
    source, root, _ = transport_fixture(tmp_path, proxy=b"// unpaced proxy\n")
    read = layer.root_reader(root)
    receipt = json.loads(read(layer.BASE_RECEIPT))
    with pytest.raises(ValueError, match="already carries"):
        derive_transport_window.replace(read, receipt, source=source)
    (source / "roce/extra.py").write_bytes(b"extra\n")
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["files"]["roce/extra.py"] = sha(b"extra\n")
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different files"):
        derive_transport_window.replace(read, receipt, source=source)


def test_repository_transport_bundle_matches_its_manifest():
    manifest = json.loads((derive_transport_window.SOURCE / "manifest.json").read_text())
    assert manifest["name"] == derive_transport_window.PROFILE
    for name, expected in manifest["files"].items():
        assert sha((derive_transport_window.SOURCE / name).read_bytes()) == expected, name


def peer_wait_fixture(tmp_path, *, also_changed=(), unchanged=()):
    """A repository bundle and a parent whose installed bundle differs in the pinned files."""
    source = tmp_path / "bundle"
    names = sorted(derive_transport_peer_wait.FILES) + ["roce/api.py", "LICENSE"]
    files = {name: f"supervised {name}\n".encode() for name in names}
    installed_files = {
        name: (data if name in unchanged or (name not in derive_transport_peer_wait.FILES
                                              and name not in also_changed)
               else f"parent {name}\n".encode())
        for name, data in files.items()
    }
    for name, data in files.items():
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_bytes(data)
    (source / "manifest.json").write_text(json.dumps({"name": derive_transport_window.PROFILE,
                                                      "files": {name: sha(data) for name, data in files.items()}}))
    installed = layer.canonical_json({"name": derive_transport_window.PROFILE, "dependencies": {"b12x": "pinned"},
                                      "files": {name: sha(data) for name, data in installed_files.items()}})
    parent = {f"{derive_transport_window.BUNDLE}/{name}": data for name, data in installed_files.items()}
    parent[derive_transport_window.BUNDLE + "/manifest.json"] = installed
    capabilities = {"transport_profile": derive_transport_window.PROFILE, "transport_manifest_sha256": sha(installed)}
    return (source, *code_parent(tmp_path, parent, capabilities))


def test_transport_peer_wait_replaces_exactly_its_pinned_files(tmp_path):
    source, root, lock = peer_wait_fixture(tmp_path)
    # Fixture bytes cannot match the real pins; the pinned set is still enforced.
    code = dataclasses.replace(derive_transport_peer_wait.LAYER, pins={},
                               replace=functools.partial(derive_transport_peer_wait.replace, source=source))
    layer.prepare_layer(code, lock, layer.root_reader(root), tmp_path / "context")
    context = tmp_path / "context"
    plan = json.loads((context / "plan.json").read_text())
    bundle = derive_transport_window.BUNDLE
    assert plan["replaced"] == sorted([bundle + "/manifest.json"]
                                      + [f"{bundle}/{name}" for name in derive_transport_peer_wait.FILES])
    manifest_bytes = context_file(context, bundle + "/manifest.json")
    assert json.loads(manifest_bytes)["dependencies"] == {"b12x": "pinned"}
    assert plan["lock_fields"] == {"transport_manifest_sha256": sha(manifest_bytes)}
    provenance = json.loads(context_file(context, derive_transport_peer_wait.LAYER.provenance))
    assert provenance["purpose"] == derive_transport_peer_wait.LAYER.purpose
    assert set(provenance["files"]) == set(plan["replaced"])


@pytest.mark.parametrize("change", ["also_changed", "unchanged"])
def test_transport_peer_wait_refuses_a_different_file_set(tmp_path, change):
    extra = {"also_changed": {"also_changed": ("roce/api.py",)},
             "unchanged": {"unchanged": (sorted(derive_transport_peer_wait.FILES)[0],)}}[change]
    source, root, _ = peer_wait_fixture(tmp_path, **extra)
    read = layer.root_reader(root)
    receipt = json.loads(read(layer.BASE_RECEIPT))
    with pytest.raises(ValueError, match="other files than this layer pins"):
        derive_transport_peer_wait.replace(read, receipt, source=source)


def test_transport_peer_wait_pins_the_repository_bundle():
    manifest = json.loads((derive_transport_window.SOURCE / "manifest.json").read_text())
    pins = derive_transport_peer_wait.LAYER.pins
    assert set(pins) == {f"{derive_transport_window.BUNDLE}/{name}" for name in derive_transport_peer_wait.FILES}
    for name, (inherited, resulting) in derive_transport_peer_wait.FILES.items():
        assert resulting == manifest["files"][name], name
        assert inherited != resulting, name
    assert derive_transport_peer_wait.LAYER.update_receipt is derive_transport_window.update_receipt


def tp2_fixture(tmp_path):
    hc = "".join(old for old, _ in derive_tp2_hc.HC_SWAPS).encode()
    fusion_old = derive_tp2_hc.FEATURE_SWAPS[derive_tp2_hc.PREFILL + "qwen4_hc_fusion.py"][0]
    gemm_old = derive_tp2_hc.FEATURE_SWAPS[derive_tp2_hc.PREFILL + "qwen4_mtp_gemm.py"][0]
    features = {"qwen4-prefill/qwen4_hc_fusion.py": fusion_old.encode(),
                "qwen4-prefill/qwen4_mtp_gemm.py": gemm_old.encode(),
                "qwen4-prefill/qwen4_prefill.pth": b"import qwen4_prefill_bootstrap\n",
                "sparkring_features.py": b"# loader\n"}
    manifest = {"name": "qwen4-prefill", "files": {name.split("/", 1)[1]: sha(data) for name, data in features.items()
                                                    if name.startswith("qwen4-prefill/")}}
    features["qwen4-prefill/manifest.json"] = layer.canonical_json(manifest)
    capabilities = {"features": {"qwen4-prefill": {"manifest_sha256": sha(features["qwen4-prefill/manifest.json"]),
                                                   "supported_tp": [4],
                                                   "files": {name: sha(data) for name, data in features.items()
                                                             if name.startswith("qwen4-prefill/")}}}}
    features["capabilities.json"] = layer.canonical_json(capabilities)
    files = {derive_tp2_hc.FEATURES + name: data for name, data in features.items()}
    files[derive_tp2_hc.HC] = hc
    files[derive_tp2_hc.AUDIT] = ("raise SystemExit(" + derive_tp2_hc.AUDIT_SWAP[0] + ")\n").encode()
    modes = {"2": [{"prefill_row_ownership": "off", "projection_tp": "1"}],
             "4": [{"prefill_row_ownership": "shard", "projection_tp": "0"}]}
    return code_parent(tmp_path, files, {"hc_supported_modes": modes})


def test_tp2_hc_edits_rank_gates_and_rehashes_the_feature_bundle(tmp_path):
    root, lock = tp2_fixture(tmp_path)
    layer.prepare_layer(derive_tp2_hc.LAYER, lock, layer.root_reader(root), tmp_path / "context")
    context = tmp_path / "context"
    hc = context_file(context, derive_tp2_hc.HC).decode()
    assert all(new in hc for _, new in derive_tp2_hc.HC_SWAPS)
    assert "TP2 or TP4" in context_file(context, derive_tp2_hc.AUDIT).decode()
    prefill = derive_tp2_hc.PREFILL
    manifest_bytes = context_file(context, prefill + "manifest.json")
    manifest = json.loads(manifest_bytes)
    assert manifest["files"]["qwen4_hc_fusion.py"] == sha(context_file(context, prefill + "qwen4_hc_fusion.py"))
    feature = json.loads(context_file(context, derive_tp2_hc.FEATURES + "capabilities.json"))["features"]["qwen4-prefill"]
    assert feature["manifest_sha256"] == sha(manifest_bytes) and feature["supported_tp"] == [2, 4]
    assert feature["files"]["qwen4-prefill/qwen4_mtp_gemm.py"] == sha(context_file(context, prefill + "qwen4_mtp_gemm.py"))
    provenance = json.loads(context_file(context, derive_tp2_hc.LAYER.provenance))
    # Unchanged feature files are listed too, so the provenance covers the whole tree.
    assert provenance["files"][derive_tp2_hc.FEATURES + "sparkring_features.py"]["inherited_sha256"] == sha(b"# loader\n")
    assert provenance["purpose"] == derive_tp2_hc.LAYER.purpose
    base = json.loads(context_file(context, layer.BASE_RECEIPT))
    assert derive_tp2_hc.TP2_SHARD in base["capabilities"]["hc_supported_modes"]["2"]


def test_tp2_hc_refuses_a_parent_without_the_expected_gate(tmp_path):
    root, _ = tp2_fixture(tmp_path)
    (root / derive_tp2_hc.AUDIT.lstrip("/")).write_bytes(b"# no TP4 gate\n")
    receipt = json.loads((root / layer.BASE_RECEIPT.lstrip("/")).read_text())
    with pytest.raises(ValueError, match="Expected one occurrence"):
        derive_tp2_hc.replace(layer.root_reader(root), receipt)


def test_staging_fix_snapshots_rows_before_the_copy():
    source = "        cpu, gpu = self.cpu, self.gpu\n" + derive_staging_fix.COPY
    result = derive_staging_fix.replace(lambda path: source.encode(), {})[derive_staging_fix.UTILS].decode()
    assert "staging = torch.empty_like(cpu, pin_memory=True)" in result
    assert result.index("staging.copy_(cpu)") < result.index("return gpu.copy_(staging, non_blocking=True)")
    assert derive_staging_fix.COPY not in result
    assert derive_staging_fix.LAYER.pins == {derive_staging_fix.UTILS: (derive_staging_fix.INHERITED,
                                                                        derive_staging_fix.RESULT)}


def test_staging_fix_refuses_an_unpinned_parent_file(tmp_path):
    root, lock = code_parent(tmp_path, {derive_staging_fix.UTILS: derive_staging_fix.COPY.encode()})
    with pytest.raises(ValueError, match="pinned"):
        layer.prepare_layer(derive_staging_fix.LAYER, lock, layer.root_reader(root), tmp_path / "context")
    assert not (tmp_path / "context").exists()


def test_mimo_vision_moves_the_sinks_to_the_denominator():
    source = "            sinks=sinks,\n" + derive_mimo_vision.KEY_ZERO + "        )\n"
    result = derive_mimo_vision.replace(lambda path: source.encode(), {})[derive_mimo_vision.MODEL].decode()
    assert result == "            sinks=sinks,\n" + derive_mimo_vision.DENOMINATOR + "        )\n"
    assert derive_mimo_vision.LAYER.pins == {derive_mimo_vision.MODEL: (derive_mimo_vision.INHERITED,
                                                                        derive_mimo_vision.RESULT)}


def test_mimo_vision_refuses_an_unpinned_parent_file(tmp_path):
    root, lock = code_parent(tmp_path, {derive_mimo_vision.MODEL: derive_mimo_vision.KEY_ZERO.encode()})
    with pytest.raises(ValueError, match="pinned"):
        layer.prepare_layer(derive_mimo_vision.LAYER, lock, layer.root_reader(root), tmp_path / "context")
    assert not (tmp_path / "context").exists()


def test_command_lines_prepare_code_and_descriptor_layers(parent, tmp_path, capsys):
    root, lock = code_parent(tmp_path, {"/opt/app/a.py": b"a\n"})
    lock_path = tmp_path / "parent-lock.json"
    lock_path.write_text(json.dumps(lock))
    layer.main(simple_layer(), ["prepare", "--parent-lock", str(lock_path), "--parent-root", str(root),
                                "--output", str(tmp_path / "code-context")])
    assert json.loads(capsys.readouterr().out)["files"] == 1
    layer.main(argv=["prepare", "--descriptor", str(parent.descriptor), "--repository", str(parent.repository),
                     "--base-receipt", str(parent.receipts / "base.json"),
                     "--toolchain-receipt", str(parent.receipts / "toolchain.json"),
                     "--output", str(tmp_path / "descriptor-context")])
    assert json.loads(capsys.readouterr().out)["files"] == 2
    with pytest.raises(SystemExit):
        layer.main(argv=["prepare", "--descriptor", str(parent.descriptor), "--repository", str(parent.repository),
                         "--base-receipt", str(parent.receipts / "base.json"), "--output", str(tmp_path / "other")])
    assert not (tmp_path / "other").exists()


# The tool-choice layer: vLLM's Chat Completions serving module and the policy it installs.

TOOL = derive_tool_choice_contract
SERVING_FIXTURE = """\
class OpenAIServingChat:
    async def chat_completion_full_generator(self, request, result_generator):
        pass

    async def chat_completion_stream_generator(self, request, result_generator):
        yield None

    def _create_chat_logprobs(self, logprobs_content):
""" + TOOL.END


def test_tool_choice_layer_adds_the_policy_and_installs_it_after_the_serving_class():
    replaced = TOOL.replace(lambda path: SERVING_FIXTURE.encode(), {})
    assert set(replaced) == {TOOL.SERVING, TOOL.MODULE}
    assert replaced[TOOL.SERVING].decode() == SERVING_FIXTURE.replace(TOOL.END, TOOL.INSTALL)
    contract = (ROOT / "integrations/vllm/tool_choice_contract/contract.py").read_bytes()
    assert replaced[TOOL.MODULE] == contract.replace(b"\r\n", b"\n")
    assert TOOL.LAYER.pins == {TOOL.SERVING: (TOOL.INHERITED, TOOL.RESULT), TOOL.MODULE: (None, TOOL.MODULE_SHA256)}
    assert TOOL.MODULE.startswith(layer.SITE + "vllm/") and TOOL.LAYER.provenance.startswith(layer.RECEIPTS)


@pytest.mark.parametrize("setting, installed", [(None, False), ("0", False), ("1", True)])
def test_tool_choice_serving_module_installs_the_policy_only_when_enabled(monkeypatch, setting, installed):
    replaced = TOOL.replace(lambda path: SERVING_FIXTURE.encode(), {})
    package = "vllm.entrypoints.openai.chat_completion"
    policy = ModuleType(package + ".sparkring_tool_choice_contract")
    for name in ("vllm", "vllm.entrypoints", "vllm.entrypoints.openai", package):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, policy.__name__, policy)
    exec(compile(replaced[TOOL.MODULE], TOOL.MODULE, "exec"), policy.__dict__)
    sys.modules[package].sparkring_tool_choice_contract = policy
    if setting is None:
        monkeypatch.delenv(policy.ENVIRONMENT, raising=False)
    else:
        monkeypatch.setenv(policy.ENVIRONMENT, setting)
    namespace = {"__name__": package + ".serving"}
    exec(compile(replaced[TOOL.SERVING], TOOL.SERVING, "exec"), namespace)
    serving = namespace["OpenAIServingChat"]
    assert getattr(serving, policy.MARKER, False) is installed
    assert hasattr(serving.chat_completion_full_generator, "__wrapped__") is installed
    assert hasattr(serving.chat_completion_stream_generator, "__wrapped__") is installed


def test_tool_choice_layer_refuses_a_parent_other_than_the_pinned_serving_module(tmp_path):
    root, lock = code_parent(tmp_path, {TOOL.SERVING: SERVING_FIXTURE.encode()})
    with pytest.raises(ValueError, match="pinned"):
        layer.prepare_layer(TOOL.LAYER, lock, layer.root_reader(root), tmp_path / "context")
    assert not (tmp_path / "context").exists()
    unpinned = dataclasses.replace(TOOL.LAYER, pins={TOOL.MODULE: TOOL.LAYER.pins[TOOL.MODULE]})
    result = layer.prepare_layer(unpinned, lock, layer.root_reader(root), tmp_path / "context")
    plan = json.loads((Path(result["context"]) / "plan.json").read_text())
    assert plan["added"] == [TOOL.MODULE] and plan["replaced"] == [TOOL.SERVING]


def test_tool_choice_layer_refuses_a_parent_without_the_serving_modules_last_statement():
    with pytest.raises(ValueError, match="Expected one occurrence"):
        TOOL.replace(lambda path: b"class OpenAIServingChat:\n    pass\n", {})
