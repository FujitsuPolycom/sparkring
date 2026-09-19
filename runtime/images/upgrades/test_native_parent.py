"""Native-to-native foundation selection and preservation without Docker/GPU."""
import copy
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from . import build_native, native_install


def raw_record(kind="native"):
    return json.dumps(dict(
        schema=f"sparkring-{kind}-installed/v1",
        files={"/opt/sparkring/contracts/active.json": "a" * 64},
        versions={"torch": "2.13.0"}, removed_files=[],
        active_contracts=["/opt/sparkring/contracts/active.json"],
        boundary_runtime=None,
    ), indent=2).encode()


def installed_parent(tmp_path, monkeypatch, kind="native"):
    root = tmp_path / "runtime"
    (root / "receipts").mkdir(parents=True)
    monkeypatch.setattr(native_install, "ROOT", root)
    monkeypatch.setattr(native_install, "RECEIPT", root / "receipts/candidate-installed.json")
    monkeypatch.setattr(native_install, "NATIVE", root / "receipts/native-installed.json")
    monkeypatch.setattr(native_install.metadata, "version", lambda _: "2.13.0")
    owned = root / "payload"
    owned.write_bytes(b"preserved")
    value = json.loads(raw_record(kind))
    value["files"] = {str(owned): native_install.sha(owned)}
    value["active_contracts"] = []
    selected = native_install.NATIVE if kind == "native" else native_install.RECEIPT
    selected.write_text(json.dumps(value))
    return value, selected, owned


def test_native_contracts_are_normalized_from_owned_hashes_without_mutating_raw():
    raw = raw_record()
    value = native_install.normalize_parent_receipt(raw, "native")
    assert value["integration_contracts"] == {
        "/opt/sparkring/contracts/active.json": {"sha256": "a" * 64}}
    assert "integration_contracts" not in json.loads(raw)
    assert value["schema"] == "sparkring-native-installed/v1"


@pytest.mark.parametrize("defect", ["unowned", "duplicate", "outside", "bad-hash", "wrong-schema"])
def test_native_contract_normalization_rejects_unproven_selections(defect):
    value = json.loads(raw_record())
    if defect == "unowned":
        value["files"] = {"/unrelated": "b" * 64}
    elif defect == "duplicate":
        value["active_contracts"] *= 2
    elif defect == "outside":
        value["active_contracts"] = ["/unrelated"]
        value["files"]["/unrelated"] = "b" * 64
    elif defect == "bad-hash":
        value["files"][value["active_contracts"][0]] = "bad"
    else:
        value["schema"] = "sparkring-candidate-installed/v1"
    with pytest.raises(ValueError):
        native_install.normalize_parent_receipt(json.dumps(value).encode(), "native")


@pytest.mark.parametrize("kind", ["native", "candidate"])
def test_parent_selection_preserves_raw_identity(tmp_path, monkeypatch, kind):
    value, selected, owned = installed_parent(tmp_path, monkeypatch, kind)
    if kind == "native":
        native_install.RECEIPT.write_text("stale ancestor, deliberately not JSON")
    raw = selected.read_bytes()
    parent, original, evidence = native_install.read_parent_foundation()
    assert original == raw
    assert evidence == dict(kind=kind, path=str(selected), sha256=hashlib.sha256(raw).hexdigest())
    assert parent["files"] == value["files"]
    assert owned.read_bytes() == b"preserved"
    if kind == "native":
        assert native_install.RECEIPT.read_text().startswith("stale ancestor")


@pytest.mark.parametrize("failure", ["json", "schema", "bytes", "version", "removed", "descriptor"])
def test_present_native_failure_never_falls_back(tmp_path, monkeypatch, failure):
    value, selected, owned = installed_parent(tmp_path, monkeypatch)
    native_install.RECEIPT.write_bytes(raw_record("candidate"))
    descriptor = None
    if failure == "json":
        selected.write_text("broken")
    elif failure == "schema":
        value["schema"] = "sparkring-candidate-installed/v1"
        selected.write_text(json.dumps(value))
    elif failure == "bytes":
        owned.write_bytes(b"tampered")
    elif failure == "version":
        monkeypatch.setattr(native_install.metadata, "version", lambda _: "wrong")
    elif failure == "removed":
        value["removed_files"] = [str(owned)]
        selected.write_text(json.dumps(value))
    else:
        descriptor = dict(parent_installed_sha256=native_install.sha(native_install.RECEIPT),
                          parent_receipt=dict(kind="candidate", path=str(native_install.RECEIPT),
                                              sha256=native_install.sha(native_install.RECEIPT)))
    with pytest.raises(ValueError):
        native_install.read_parent_foundation(descriptor)


def test_native_boundary_supersedes_stale_candidate_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(native_install, "ROOT", tmp_path)
    (tmp_path / "contracts").mkdir()
    current = tmp_path / "contracts/boundary-native-current.json"
    current.write_text("{}")
    selected = dict(path=str(current), sha256=native_install.sha(current))
    parent = dict(schema="sparkring-native-installed/v1", boundary_runtime=selected,
                  files={str(current): selected["sha256"]},
                  cache_extension={"boundary_runtime": {"path": "/stale"}})
    assert native_install.selected_boundary_identity(parent) == current
    parent["boundary_runtime"] = None
    assert native_install.selected_boundary_identity(parent) is None
    parent["boundary_runtime"] = selected
    current.write_text("changed")
    with pytest.raises(ValueError):
        native_install.selected_boundary_identity(parent)


def test_native_active_binding_requires_no_source_extension():
    raw = b"{}"
    value = json.loads(raw_record())
    path = value["active_contracts"][0]
    value["files"][path] = hashlib.sha256(raw).hexdigest()
    original = copy.deepcopy(value)
    normalized = native_install.normalize_parent_receipt(json.dumps(value).encode(), "native")
    active, proof = build_native.select_parent_binding(normalized, path, raw)
    assert active == {path} and proof is None and value == original


def test_parent_receipt_archive_keeps_exact_bytes(tmp_path, monkeypatch):
    _, selected, _ = installed_parent(tmp_path, monkeypatch)
    parent, raw, evidence = native_install.read_parent_foundation()
    files, evidence = native_install.retain_parent_receipt(raw, evidence)
    archived = native_install.Path(evidence["retained_path"])
    assert archived != selected and archived.read_bytes() == raw
    assert files == {str(archived): hashlib.sha256(raw).hexdigest()}
    archived.write_text("tampered")
    with pytest.raises(ValueError):
        native_install.retain_parent_receipt(raw, evidence)


@pytest.mark.parametrize("mode", ["native", "candidate", "corrupt-native", "wrong-schema-native"])
def test_builder_selects_receipt_in_image_without_ancestor_fallback(tmp_path, monkeypatch, mode):
    paths = {kind: str(tmp_path / (kind + ".json")) for kind in ("native", "candidate")}
    native_install.Path(paths["candidate"]).write_bytes(raw_record("candidate"))
    if mode != "candidate":
        native_install.Path(paths["native"]).write_bytes(
            b"broken" if mode == "corrupt-native" else raw_record(
                "candidate" if mode == "wrong-schema-native" else "native"))
    monkeypatch.setattr(build_native, "PARENT_RECEIPTS", paths)

    def local_read(*args, **kwargs):
        assert "--read-only" in args and "--pull" in args and "never" in args
        assert args[-2] == "-c"
        return subprocess.check_output([sys.executable, "-c", args[-1]])

    monkeypatch.setattr(build_native, "docker", local_read)
    if mode.endswith("native") and mode != "native":
        with pytest.raises(ValueError):
            build_native.read_parent_image("sha256:" + "1" * 64)
    else:
        parent, raw, selected = build_native.read_parent_image("sha256:" + "1" * 64)
        assert raw == native_install.Path(paths[mode]).read_bytes()
        assert selected == dict(kind=mode, path=paths[mode], sha256=hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize("tamper", [False, True, "removed-reintroduced"])
def test_native_install_preserves_unrelated_inventory_and_parent_evidence(tmp_path, monkeypatch, tamper):
    parent, selected, unrelated = installed_parent(tmp_path, monkeypatch)
    root, site, context = native_install.ROOT, tmp_path / "site", tmp_path / "context"
    monkeypatch.setattr(native_install, "SITE", site)
    monkeypatch.setattr(native_install, "ENTRYPOINT", root / "bin/native-image.py")
    context.mkdir()
    (context / "wheels").mkdir()
    packages = {}
    for name in ("vllm", "b12x", "sparkcache"):
        path = site / name / "module.py"
        path.parent.mkdir(parents=True)
        path.write_text("before " + name)
        packages[name] = path
        parent["files"][str(path)] = native_install.sha(path)
    manifest = root / "features/capabilities.json"
    manifest.parent.mkdir()
    feature = manifest.parent / "active.py"
    feature.write_text("FEATURE = True")
    manifest.write_text(json.dumps(dict(schema="sparkring-image-capabilities/v1",
                                       features={"retained": {"files": {"active.py": native_install.sha(feature)}}})))
    for path in (manifest, feature):
        parent["files"][str(path)] = native_install.sha(path)
    removed = site / "vllm/removed.py"
    parent["removed_files"] = [str(removed)]
    parent["versions"].update(vllm="old", b12x="old")
    native_install.RECEIPT.write_text("stale historical candidate")
    # SGLang remains bound to its historical composition receipt, while the
    # native inventory is authoritative for the current serving foundation.
    isolated = tmp_path / "sglang"
    isolated.mkdir()
    monkeypatch.setattr(native_install, "SGLANG_PREFIX", isolated)
    library = isolated / "runtime.so"
    library.write_bytes(b"isolated runtime")
    directory = root / "sglang"
    directory.mkdir()
    (root / "bin").mkdir()
    wrapper = root / "bin/sglang-python"
    wrapper.write_text("isolated wrapper")
    composition = directory / "manifest.json"
    composition.write_text(json.dumps({"sglang_base": {"python_prefix": str(isolated)}}))
    sglang_receipt = directory / "installed.json"
    sglang_receipt.write_text(json.dumps(dict(
        schema="sparkring-sglang-installed/v1",
        composition_sha256=native_install.sha(composition),
        vllm_parent_receipt_sha256=native_install.sha(native_install.RECEIPT),
        files={str(wrapper): native_install.sha(wrapper)},
    )))
    for path in (library, wrapper, composition, sglang_receipt):
        parent["files"][str(path)] = native_install.sha(path)
    selected.write_text(json.dumps(parent))
    versions = dict(parent["versions"], torchvision="old", torchaudio="old")
    monkeypatch.setattr(native_install.metadata, "version", versions.__getitem__)
    wheels = {}
    for name in ("vllm", "b12x"):
        wheel = context / "wheels" / (name + ".whl")
        wheel.write_bytes(name.encode())
        wheels[name] = dict(file=wheel.name, sha256=native_install.sha(wheel), version="new")
    compiler = dict(schema="sparkring-native-wheel-result/v1", descriptor_sha256="c" * 64,
                    source_trees={"vllm": "a" * 64, "b12x": "b" * 64},
                    torch_version="2.13.0", wheels=wheels)
    raw = selected.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    descriptor = dict(schema="sparkring-native-install/v1", input_sha256="d" * 64,
                      parent_image_id="sha256:" + "1" * 64, parent_installed_sha256=digest,
                      parent_receipt=dict(kind="native", path=str(selected), sha256=digest),
                      compiler_descriptor_sha256="c" * 64, source_trees=compiler["source_trees"])
    (context / "descriptor.json").write_text(json.dumps(descriptor))
    (context / "compiler-result.json").write_text(json.dumps(compiler))
    monkeypatch.setattr(native_install, "distribution_files", lambda name: {
        str(packages[name]): native_install.sha(packages[name])})
    monkeypatch.setattr(native_install, "wheel_install_paths", lambda wheels: (
        {str(packages[name]) for name in wheels}
        | ({str(removed)} if tamper == "removed-reintroduced" else set()), []))
    monkeypatch.setattr(native_install, "audit_selected_ownership", lambda *a: ({}, {}))

    def pip(args, **kwargs):
        if "install" in args:
            for name in ("vllm", "b12x"):
                packages[name].write_text("after " + name)
                versions[name] = "new"
            if tamper:
                unrelated.write_text("unreviewed mutation")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(native_install.subprocess, "run", pip)
    if tamper == "removed-reintroduced":
        with pytest.raises(ValueError, match="reintroduces a foundation removed file"):
            native_install.install(context)
        assert selected.read_bytes() == raw
        assert packages["vllm"].read_text() == "before vllm"
        assert versions["vllm"] == "old"
    elif tamper:
        with pytest.raises(ValueError, match="Unrelated foundation bytes changed"):
            native_install.install(context)
        assert selected.read_bytes() == raw
    else:
        result = native_install.install(context)
        child = native_install.read(selected)
        assert result["features"] == ["retained"]
        assert child["active_contracts"] == [] and child["boundary_runtime"] is None
        assert child["removed_files"] == [str(removed)]
        for path in (unrelated, packages["sparkcache"], feature, manifest,
                     library, wrapper, composition, sglang_receipt):
            assert child["files"][str(path)] == parent["files"][str(path)] == native_install.sha(path)
        assert child["isolated_sglang"]["receipt_sha256"] == native_install.sha(sglang_receipt)
        evidence = child["parent_receipt"]
        assert {key: evidence[key] for key in descriptor["parent_receipt"]} == descriptor["parent_receipt"]
        assert native_install.Path(evidence["retained_path"]).read_bytes() == raw
        assert native_install.RECEIPT.read_text() == "stale historical candidate"
        assert child["versions"] == {"torch": "2.13.0", "vllm": "new", "b12x": "new"}
