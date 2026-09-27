"""CPU checks for derived installer layers: receipts, refusals and the lock."""

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from runtime.common import installer_image
from runtime.images import derived_layer as layer

ROOT = Path(__file__).resolve().parents[2]
ADDED = layer.SITE + "fixture_hook.py"
REPLACED = layer.SITE + "vllm/v1/utils.py"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(value, **kwargs):
    return (json.dumps(value, indent=2, **kwargs) + "\n").encode()


@pytest.fixture
def parent(tmp_path):
    """A repository with two sources, a parent lock and the parent's receipts."""
    repository = tmp_path / "repository"
    sources = {"integrations/fixture/fixture_hook.py": b"VALUE = 1\n",
               "integrations/fixture/utils.py": b"STAGED = True\n"}
    for relative, raw in sources.items():
        (repository / relative).parent.mkdir(parents=True, exist_ok=True)
        (repository / relative).write_bytes(raw)
    base = {"schema": "sparkring-external-installed/v1", "composition_sha256": "c" * 64,
            "files": {REPLACED: "d" * 64, layer.SITE + "sparkring_transport.pth": "e" * 64}}
    base_raw = dump(base, sort_keys=True)
    toolchain = {"schema": "sparkring-toolchain-installed/v1", "variant": "combined",
                 "parent_receipt_sha256": sha(base_raw), "nccl_version": 23203}
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
              base_raw=base_raw, toolchain_raw=toolchain_raw, receipts=receipts, tmp=tmp_path)


def prepare(parent, output=None, descriptor=None):
    return layer.prepare(descriptor or parent.descriptor, parent.repository, parent.receipts / "base.json",
                         parent.receipts / "toolchain.json", output or parent.tmp / "context")


def rewrite(parent, change):
    record = json.loads(parent.descriptor.read_text())
    change(record)
    parent.descriptor.write_text(json.dumps(record))


def test_prepare_rerecords_both_receipts_and_lists_every_file(parent):
    result = prepare(parent)
    context = Path(result["context"])
    files = context / "files"
    assert (context / "Dockerfile").read_text() == layer.DOCKERFILE
    assert (files / ADDED.lstrip("/")).read_bytes() == b"VALUE = 1\n"
    assert (files / REPLACED.lstrip("/")).read_bytes() == b"STAGED = True\n"
    base_raw = (files / layer.BASE_RECEIPT.lstrip("/")).read_bytes()
    base = json.loads(base_raw)
    assert base_raw == dump(base, sort_keys=True)
    assert base["files"] == {**parent.base["files"], ADDED: sha(b"VALUE = 1\n"), REPLACED: sha(b"STAGED = True\n")}
    toolchain_raw = (files / layer.TOOLCHAIN_RECEIPT.lstrip("/")).read_bytes()
    assert json.loads(toolchain_raw)["parent_receipt_sha256"] == sha(base_raw)
    assert list(json.loads(toolchain_raw)) == list(json.loads(parent.toolchain_raw))
    provenance = json.loads((files / "opt/sparkring/receipts/derived-fixture.json").read_bytes())
    assert provenance["parent_receipt_sha256"] == sha(parent.base_raw)
    assert provenance["files"][REPLACED]["inherited_sha256"] == "d" * 64
    assert provenance["files"][ADDED]["inherited_sha256"] is None
    assert provenance["receipts"] == {layer.BASE_RECEIPT: sha(base_raw), layer.TOOLCHAIN_RECEIPT: sha(toolchain_raw)}
    plan = json.loads((context / "plan.json").read_text())
    assert plan["added"] == [ADDED] and plan["replaced"] == [REPLACED]
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


def test_repository_descriptors_pin_their_current_sources():
    descriptors = sorted((ROOT / "runtime/images/compositions").glob("*/descriptor.json"))
    derived = [path for path in descriptors if json.loads(path.read_text())["schema"] == layer.SCHEMA]
    assert derived
    for path in derived:
        record = layer.descriptor(path)
        layer._parent_lock(record, ROOT)
        for row in record["files"].values():
            layer.pinned_bytes((ROOT / row["source"]).read_bytes(), row["sha256"])
