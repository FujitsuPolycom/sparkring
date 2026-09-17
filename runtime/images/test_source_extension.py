"""Source composition admission, reproducibility, and inherited-image integrity."""
import copy
import json
from pathlib import Path
import shutil

import pytest

from runtime.images import source_extension as extension

pytestmark = pytest.mark.skipif(not shutil.which("git"), reason="Git applies the pinned source patch")

OLD = b"VALUE = 1\n"
NEW = b"VALUE = 2\n"
ADDED = b"ENABLED = True\n"
PATCH = b"""diff --git a/b12x/operation.py b/b12x/operation.py
index 1111111..2222222 100644
--- a/b12x/operation.py
+++ b/b12x/operation.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
diff --git a/vllm/feature.py b/vllm/feature.py
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/vllm/feature.py
@@ -0,0 +1 @@
+ENABLED = True
"""
CONTRACT = "/opt/sparkring/contracts/source-fixture.json"


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def put(root, name, raw):
    path = root / name.lstrip("/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def fixture(tmp_path):
    root, repository, output = (tmp_path / name for name in ("root", "repository", "context"))
    inventory = {
        extension.SITE + "b12x/operation.py": OLD,
        "/opt/sparkring/lib/native.so": b"native library retained",
        extension.SITE + "sparkring_features.pth": b"import existing_features\n",
        extension.SITE + "sparkring_transport.pth": b"import existing_transport\n",
        "/opt/sparkring/bin/candidate-image.py": b"# inherited entry point\n",
    }
    for path, data in inventory.items():
        put(root, path, data)
    parent = {
        "schema": "sparkring-candidate-installed/v1", "composition_id": "fixture-parent",
        "files": {path: extension.digest(data) for path, data in inventory.items()},
        "versions": {"vllm": "0.26.1", "b12x": "1.3.0"},
        "removed_authored_files": [extension.SITE + "vllm/removed.py"],
        "feature_extension": {"id": "retained-features", "capabilities": ["qwen-prefill"]},
        "cache_extension": {"id": "retained-cache", "native": {"sha256": "a" * 64}},
        "native": {"nccl": "dual-domain"}, "unknown_metadata": {"retained": True},
    }
    parent_raw = encoded(parent)
    put(root, extension.RECEIPT, parent_raw)
    put(repository, "patches/runtime.patch", PATCH)
    put(repository, "contracts/jobs.json", b'{"files":{},"purpose":"fixture"}\n')
    record = {
        "schema": "sparkring-source-extension/v1", "id": "fixture-source",
        "parent": {"image_id": "sha256:" + "b" * 64, "receipt_sha256": extension.digest(parent_raw)},
        "installer_sha256": extension.digest(Path(extension.__file__).read_bytes()),
        "provenance": {"upstream": [{"repository": "fixture", "revision": "c" * 40}]},
        "patch": {"source": "patches/runtime.patch", "sha256": extension.digest(PATCH)},
        "sources": {
            "b12x/operation.py": {"parent_sha256": extension.digest(OLD), "sha256": extension.digest(NEW)},
            "vllm/feature.py": {"parent_sha256": None, "sha256": extension.digest(ADDED)},
        },
        "integration_contracts": {CONTRACT: {"source": "contracts/jobs.json",
            "sha256": extension.digest((repository / "contracts/jobs.json").read_bytes())}},
    }
    descriptor_path = put(repository, "descriptor.json", encoded(record))
    return {"root": root, "repository": repository, "output": output, "record": record,
            "descriptor": descriptor_path, "parent": parent,
            "versions": parent["versions"].__getitem__}


def prepare(f):
    f["descriptor"].write_bytes(encoded(f["record"]))
    return extension.prepare(f["descriptor"], f["repository"], f["output"])


def install(f):
    return extension.install(f["output"], f["root"], f["versions"])


def test_build_and_verify_preserve_parent_metadata_native_libraries_and_hooks(fixture):
    f = fixture
    before = snapshot(f["root"])
    prepare(f)
    result = install(f)
    receipt = json.loads((f["root"] / extension.RECEIPT[1:]).read_bytes())
    for key in f["parent"].keys() - {"files"}:
        assert receipt[key] == f["parent"][key]
    replaced = extension.SITE + "b12x/operation.py"
    for name, data in before.items():
        if "/" + name not in {replaced, extension.RECEIPT}:
            assert (f["root"] / name).read_bytes() == data
    assert (f["root"] / replaced[1:]).read_bytes() == NEW
    assert (f["root"] / (extension.SITE + "vllm/feature.py")[1:]).read_bytes() == ADDED
    assert receipt["files"].keys() == f["parent"]["files"].keys() | {
        extension.SITE + "vllm/feature.py", CONTRACT, *extension.RESERVED}
    assert result["files_verified"] == len(receipt["files"])
    assert result["serving_qualified"] is False
    assert extension.verify(f["root"], f["versions"]) == result
    assert f["parent"]["files"][replaced] == extension.digest(OLD)


def test_prepare_context_contains_only_compact_patch_and_contract(fixture):
    f = fixture
    result = prepare(f)
    assert result["assets"] == 3
    files = snapshot(f["output"])
    assert set(files) == {"source.patch", "descriptor.json", "source_extension.py", "Dockerfile",
                          "payload/" + CONTRACT[1:]}
    assert files["source.patch"] == PATCH
    assert b"ARG PARENT_IMAGE=sha256:" in files["Dockerfile"]
    assert b'ENTRYPOINT ["/opt/venv/bin/python", "/opt/sparkring/bin/source-extension.py"]' in files["Dockerfile"]


@pytest.mark.parametrize("part", ["patch", "contract", "installer"])
def test_prepare_rejects_tampered_sources_before_creating_output(fixture, part):
    f = fixture
    if part == "installer":
        f["record"]["installer_sha256"] = "0" * 64
    else:
        source = "patches/runtime.patch" if part == "patch" else "contracts/jobs.json"
        (f["repository"] / source).write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="pinned bytes"):
        prepare(f)
    assert not f["output"].exists()


@pytest.mark.parametrize("part", ["receipt", "native", "hook", "patch", "contract", "installer",
                                 "wrong_preimage", "wrong_result", "unlisted_addition", "existing_contract",
                                 "reserved_path", "removed_addition", "source_unowned", "invalid_python",
                                 "invalid_contract", "parent_directory_file"])
def test_install_rejects_invalid_inputs_without_changing_any_parent_file(fixture, part):
    f = fixture
    if part == "wrong_preimage":
        f["record"]["sources"]["b12x/operation.py"]["parent_sha256"] = "0" * 64
    elif part == "wrong_result":
        f["record"]["sources"]["vllm/feature.py"]["sha256"] = "0" * 64
    elif part == "source_unowned":
        f["parent"]["files"].pop(extension.SITE + "b12x/operation.py")
    elif part == "removed_addition":
        f["parent"]["removed_authored_files"].append(extension.SITE + "vllm/feature.py")
    elif part == "invalid_python":
        patched = PATCH.replace(b"+ENABLED = True", b"+ENABLED = (")
        (f["repository"] / "patches/runtime.patch").write_bytes(patched)
        f["record"]["patch"]["sha256"] = extension.digest(patched)
        f["record"]["sources"]["vllm/feature.py"]["sha256"] = extension.digest(b"ENABLED = (\n")
    if part in {"source_unowned", "removed_addition"}:
        raw = encoded(f["parent"])
        put(f["root"], extension.RECEIPT, raw)
        f["record"]["parent"]["receipt_sha256"] = extension.digest(raw)
    prepare(f)
    if part in {"receipt", "native", "hook"}:
        name = {"receipt": extension.RECEIPT, "native": "/opt/sparkring/lib/native.so",
                "hook": extension.SITE + "sparkring_features.pth"}[part]
        put(f["root"], name, b"changed")
    elif part in {"patch", "contract", "installer"}:
        name = {"patch": "source.patch", "contract": "payload/" + CONTRACT[1:],
                "installer": "source_extension.py"}[part]
        (f["output"] / name).write_bytes(b"changed")
    elif part in {"unlisted_addition", "existing_contract", "reserved_path"}:
        name = {"unlisted_addition": extension.SITE + "vllm/feature.py", "existing_contract": CONTRACT,
                "reserved_path": extension.INSTALLER}[part]
        put(f["root"], name, b"preserve this existing file")
    elif part == "invalid_contract":
        raw = b"[]\n"
        (f["output"] / "payload" / CONTRACT[1:]).write_bytes(raw)
        f["record"]["integration_contracts"][CONTRACT]["sha256"] = extension.digest(raw)
        (f["output"] / "descriptor.json").write_bytes(encoded(f["record"]))
    elif part == "parent_directory_file":
        put(f["root"], "/opt/sparkring/contracts", b"not a directory")
    before = snapshot(f["root"])
    with pytest.raises((ValueError, SyntaxError)):
        install(f)
    assert snapshot(f["root"]) == before


@pytest.mark.parametrize("path", ["../vllm/escape.py", "vllm/../escape.py", "vllm/./alias.py",
                                 "vllm//alias.py", "vllm\\escape.py", "vllm/native.so",
                                 "vllm/native.py.so", "sparkcache/other.py", "/vllm/absolute.py",
                                 "vllm/__pycache__/other.py", "vllm/colon:stream.py"])
def test_descriptor_rejects_unsafe_or_non_python_replacements(fixture, path):
    record = copy.deepcopy(fixture["record"])
    record["sources"] = {path: {"sha256": "a" * 64, "parent_sha256": None}}
    with pytest.raises(ValueError):
        extension.descriptor(encoded(record))


@pytest.mark.parametrize("patch", [
    PATCH.replace(b"a/b12x/operation.py b/b12x/operation.py", b"a/b12x/other.py b/b12x/other.py"),
    PATCH.replace(b"new file mode 100644", b"new file mode 120000"),
    PATCH.replace(b"index 1111111..2222222 100644", b"old mode 100644\nnew mode 100755"),
    PATCH.replace(b"--- a/b12x/operation.py", b"--- /dev/null"),
    PATCH.replace(b"+++ b/b12x/operation.py", b"+++ /dev/null"),
    PATCH.replace(b"@@ -1 +1 @@", b"GIT binary patch"),
    PATCH + b"--- a/vllm/unlisted.py\n+++ b/vllm/unlisted.py\n@@ -0,0 +1 @@\n+ESCAPE = 1\n",
    PATCH.replace(b"@@ -1 +1 @@", b"@@ -1,2 +1 @@"),
    PATCH + PATCH,
    PATCH.split(b"diff --git a/vllm/")[0],
])
def test_patch_rejects_path_changes_modes_deletions_binary_duplicate_or_missing_files(fixture, patch):
    record = copy.deepcopy(fixture["record"])
    record["patch"]["sha256"] = extension.digest(patch)
    with pytest.raises(ValueError):
        extension.validate_patch(patch, record)


@pytest.mark.parametrize("part", ["source", "native", "patch", "installer", "metadata", "inventory", "parent"])
def test_verifier_rejects_changed_source_native_receipt_and_saved_evidence(fixture, part):
    f = fixture
    prepare(f)
    install(f)
    if part in {"metadata", "inventory"}:
        path = f["root"] / extension.RECEIPT[1:]
        receipt = json.loads(path.read_bytes())
        if part == "metadata":
            receipt["feature_extension"]["id"] = "changed"
        else:
            receipt["files"].pop("/opt/sparkring/lib/native.so")
        path.write_bytes(encoded(receipt))
    else:
        target = {"source": extension.SITE + "b12x/operation.py", "native": "/opt/sparkring/lib/native.so",
                  "patch": extension.PATCH, "installer": extension.INSTALLER,
                  "parent": extension.PARENT_RECEIPT}[part]
        put(f["root"], target, b"changed")
    with pytest.raises(ValueError):
        extension.verify(f["root"], f["versions"])


def test_versions_checked_before_installation_and_before_serving(fixture):
    f = fixture
    prepare(f)
    before = snapshot(f["root"])
    with pytest.raises(ValueError, match="distribution version mismatch"):
        extension.install(f["output"], f["root"], lambda _: "incorrect")
    assert snapshot(f["root"]) == before
    install(f)
    with pytest.raises(ValueError, match="distribution version mismatch"):
        extension.verify(f["root"], lambda _: "incorrect")


def test_second_installation_is_refused_without_changes(fixture):
    f = fixture
    prepare(f)
    install(f)
    before = snapshot(f["root"])
    with pytest.raises(ValueError, match="Parent receipt differs"):
        install(f)
    assert snapshot(f["root"]) == before


@pytest.mark.parametrize("where", ["repository_input", "package_parent", "existing_replacement", "contract_parent"])
def test_symlinks_are_rejected_before_installation(fixture, tmp_path, where):
    f = fixture
    prepare(f)
    if where == "repository_input":
        path = f["repository"] / "patches/runtime.patch"
        target = tmp_path / "outside.patch"
        target.write_bytes(PATCH)
    elif where == "existing_replacement":
        path = f["root"] / (extension.SITE + "b12x/operation.py")[1:]
        target = tmp_path / "outside.py"
        target.write_bytes(OLD)
    else:
        relative = extension.SITE + "vllm" if where == "package_parent" else "/opt/sparkring/contracts"
        path = f["root"] / relative[1:]
        target = tmp_path / "outside-directory"
        target.mkdir()
    if path.exists():
        path.unlink()
    try:
        path.symlink_to(target, target_is_directory=target.is_dir())
    except OSError:
        pytest.skip("Creating symlinks requires host permission")
    before = snapshot(f["root"])
    with pytest.raises(ValueError):
        if where == "repository_input":
            extension.prepare(f["descriptor"], f["repository"], tmp_path / "other-context")
        else:
            install(f)
    assert snapshot(f["root"]) == before


def test_duplicate_json_keys_are_rejected():
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        extension.json_object(b'{"sources":{},"sources":{}}')


def test_pinned_line_endings_reconstruct_exact_bytes():
    assert extension.pinned_bytes(PATCH.replace(b"\n", b"\r\n"), extension.digest(PATCH)) == PATCH


def test_serve_verifies_before_exec_and_preserves_arguments(monkeypatch):
    events = []
    monkeypatch.setattr(extension, "verify", lambda: events.append("verified") or {})
    monkeypatch.setattr(extension.os, "execve", lambda executable, arguments, environment:
                        events.append((executable, arguments, environment)))
    extension.main(["serve", "/models/qwen", "--tensor-parallel-size", "4"])
    assert events[0] == "verified"
    assert events[1][1] == ["/opt/venv/bin/python", "-m", "vllm.entrypoints.cli.main", "serve",
                            "/models/qwen", "--tensor-parallel-size", "4"]


def test_failed_verification_prevents_serve(monkeypatch):
    def fail():
        raise ValueError("inventory mismatch")
    monkeypatch.setattr(extension, "verify", fail)
    monkeypatch.setattr(extension.os, "execve", lambda *args: pytest.fail("exec after verification failure"))
    with pytest.raises(ValueError, match="inventory mismatch"):
        extension.main(["serve", "/models/qwen"])
