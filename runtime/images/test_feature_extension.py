"""CPU-only source and inherited-inventory checks for feature compositions."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from runtime.images import feature_extension as extension


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def fixture(tmp_path):
    repository = tmp_path / "repository"
    source = repository / "integrations" / "hook.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"VALUE = 1\n")
    descriptor = tmp_path / "composition" / "descriptor.json"
    descriptor.parent.mkdir()
    (descriptor.parent / "Dockerfile").write_bytes(b"FROM example@sha256:fixture\n")
    record = {
        "schema": "sparkring-feature-extension/v1",
        "id": "fixture-features",
        "parent": {"image_id": "sha256:" + "a" * 64, "receipt_sha256": "b" * 64},
        "installer_sha256": sha(Path(extension.__file__).read_bytes()),
        "capabilities": ["qwen-prefill", "qwen-collectives"],
        "assets": {
            extension.FEATURE_ROOT + "fixture/hook.py": {
                "source": "integrations/hook.py", "sha256": sha(source.read_bytes()),
            },
            extension.SITE + "sparkring_features.pth": {
                "text": "import feature_bootstrap\n", "sha256": sha(b"import feature_bootstrap\n"),
            },
        },
    }
    descriptor.write_text(json.dumps(record), encoding="utf-8")
    return repository, descriptor, record


@pytest.mark.parametrize("provided,expected", [
    (b"a\nb\n", b"a\nb\n"),
    (b"a\r\nb\r\n", b"a\r\nb\r\n"),
    (b"a\nb\n", b"a\r\nb\r\n"),
    (b"a\r\nb\r\n", b"a\nb\n"),
])
def test_pinned_bytes_reconstruct_only_pinned_line_endings(provided, expected):
    assert extension.pinned_bytes(provided, sha(expected)) == expected


@pytest.mark.parametrize("changed", [b"a\nchanged\n", b"a\r\nchanged\r\n", b"a\nb", b"a \nb\n"])
def test_pinned_bytes_reject_content_changes(changed):
    with pytest.raises(ValueError, match="content differs"):
        extension.pinned_bytes(changed, sha(b"a\nb\n"))


@pytest.mark.parametrize("target", [
    "relative/hook.py", "/tmp/hook.py", "/opt/sparkring/features-elsewhere/hook.py",
    "/opt/sparkring/features/../bin/hook.py", "/opt/sparkring/features/sub/../../hook.py",
    "/opt/sparkring/features/sub\\hook.py", extension.SITE + "unowned.pth",
])
def test_descriptor_rejects_unowned_or_traversing_targets(fixture, target):
    _, descriptor, record = fixture
    record["assets"] = {target: {"text": "x", "sha256": sha(b"x")}}
    descriptor.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="outside its owner"):
        extension.descriptor(descriptor)


@pytest.mark.parametrize("source", ["/outside.py", "../outside.py", "source/../../outside.py", "source\\hook.py"])
def test_descriptor_rejects_uncontained_source_paths(fixture, source):
    _, descriptor, record = fixture
    record["assets"] = {extension.FEATURE_ROOT + "hook.py": {"source": source, "sha256": "a" * 64}}
    descriptor.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="contained repository path"):
        extension.descriptor(descriptor)


@pytest.mark.parametrize("fields", [{"sha256": "a" * 64}, {"source": "hook.py", "text": "x", "sha256": "a" * 64}])
def test_descriptor_requires_one_asset_source_kind(fixture, fields):
    _, descriptor, record = fixture
    record["assets"] = {extension.FEATURE_ROOT + "hook.py": fields}
    descriptor.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one source or literal text"):
        extension.descriptor(descriptor)


def test_prepare_writes_exact_pinned_assets_and_descriptor(fixture, tmp_path):
    repository, descriptor, record = fixture
    target = extension.FEATURE_ROOT + "fixture/hook.py"
    record["assets"][target]["sha256"] = sha(b"VALUE = 1\r\n")
    descriptor.write_text(json.dumps(record), encoding="utf-8")
    output = tmp_path / "context"
    result = extension.prepare(descriptor, repository, output)
    assert (output / "payload" / target.lstrip("/")).read_bytes() == b"VALUE = 1\r\n"
    assert (output / "payload" / (extension.SITE + "sparkring_features.pth").lstrip("/")).read_bytes() == b"import feature_bootstrap\n"
    assert (output / "descriptor.json").read_bytes() == descriptor.read_bytes()
    assert (output / "Dockerfile").read_bytes() == descriptor.with_name("Dockerfile").read_bytes()
    assert sha((output / "feature_extension.py").read_bytes()) == record["installer_sha256"]
    assert result == {"descriptor_sha256": sha(descriptor.read_bytes()), "assets": 2, "context": str(output.resolve())}


@pytest.mark.parametrize("mismatch", ["file", "literal", "installer"])
def test_prepare_rejects_all_source_mismatches_before_output_creation(fixture, tmp_path, mismatch):
    repository, descriptor, record = fixture
    if mismatch == "file":
        (repository / "integrations/hook.py").write_bytes(b"VALUE = 2\n")
    elif mismatch == "literal":
        record["assets"][extension.SITE + "sparkring_features.pth"]["text"] = "changed\n"
    else:
        record["installer_sha256"] = "0" * 64
    descriptor.write_text(json.dumps(record), encoding="utf-8")
    output = tmp_path / "context"
    with pytest.raises(ValueError, match="content differs"):
        extension.prepare(descriptor, repository, output)
    assert not output.exists()


def test_prepare_rejects_missing_source_before_output_creation(fixture, tmp_path):
    repository, descriptor, _ = fixture
    (repository / "integrations/hook.py").unlink()
    output = tmp_path / "context"
    with pytest.raises(FileNotFoundError):
        extension.prepare(descriptor, repository, output)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["empty_directory", "populated_directory", "file"])
def test_prepare_refuses_existing_output_without_changing_it(fixture, tmp_path, kind):
    repository, descriptor, _ = fixture
    output = tmp_path / "context"
    if kind == "file":
        output.write_bytes(b"keep")
    else:
        output.mkdir()
        if kind == "populated_directory":
            (output / "keep").write_bytes(b"keep")
    with pytest.raises(ValueError, match="must not exist"):
        extension.prepare(descriptor, repository, output)
    if kind == "file":
        assert output.read_bytes() == b"keep"
    elif kind == "populated_directory":
        assert list(output.iterdir()) == [output / "keep"]
        assert (output / "keep").read_bytes() == b"keep"
    else:
        assert not list(output.iterdir())


def parent_fixture():
    return {
        "schema": "candidate-installed/v1",
        "files": {"/opt/venv/model.py": sha(b"model"), "/opt/sparkring/native.so": sha(b"native")},
        "cache_extension": {"id": "cache64", "source": {"revision": "fixture-revision"}, "assets": ["native", "python"]},
        "native": {"nccl": {"sha256": "c" * 64}},
        "other_metadata": ["retained", {"entrypoint": "candidate-image.py"}],
    }


def test_expected_receipt_retains_complete_parent_and_adds_source_inventory(fixture):
    _, descriptor, record = fixture
    parent = parent_fixture()
    before = copy.deepcopy(parent)
    parent_raw = json.dumps(parent).encode()
    raw_descriptor = descriptor.read_bytes()
    payloads = {extension.FEATURE_ROOT + "fixture/hook.py": b"feature\n"}
    installer = b"installer\n"
    result = extension.expected_receipt(parent, record, raw_descriptor, parent_raw, payloads, installer)
    assert parent == before
    for key in parent.keys() - {"files"}:
        assert result[key] == parent[key]
    expected_files = {
        **parent["files"],
        **{path: sha(raw) for path, raw in payloads.items()},
        extension.PARENT_RECEIPT: sha(parent_raw),
        extension.DESCRIPTOR: sha(raw_descriptor),
        extension.INSTALLER: sha(installer),
    }
    assert result["files"] == expected_files
    assert result["feature_extension"] == {
        "id": record["id"], "descriptor_sha256": sha(raw_descriptor),
        "parent_image_id": record["parent"]["image_id"],
        "parent_receipt_sha256": sha(parent_raw), "capabilities": sorted(record["capabilities"]),
    }
    result["cache_extension"]["source"]["revision"] = "changed-result-only"
    result["native"]["nccl"]["sha256"] = "changed-result-only"
    result["files"]["/opt/venv/model.py"] = "changed-result-only"
    assert parent == before


def test_expected_receipt_refuses_replacing_any_inherited_payload(fixture):
    _, descriptor, record = fixture
    parent = parent_fixture()
    before = copy.deepcopy(parent)
    # Equal bytes are still an ownership collision, not permission to replace.
    with pytest.raises(ValueError, match="must not replace inherited files"):
        extension.expected_receipt(parent, record, descriptor.read_bytes(), b"parent", {"/opt/venv/model.py": b"model"}, b"installer")
    assert parent == before


@pytest.mark.parametrize("reserved", [extension.PARENT_RECEIPT, extension.DESCRIPTOR, extension.INSTALLER])
def test_expected_receipt_refuses_inherited_receipt_or_installer_collision(fixture, reserved):
    _, descriptor, record = fixture
    parent = parent_fixture()
    parent["files"][reserved] = "d" * 64
    before = copy.deepcopy(parent)
    with pytest.raises(ValueError, match="already belongs to the parent"):
        extension.expected_receipt(parent, record, descriptor.read_bytes(), b"parent", {extension.FEATURE_ROOT + "hook.py": b"hook"}, b"installer")
    assert parent == before
