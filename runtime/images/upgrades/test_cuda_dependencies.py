"""GPU-free admission and RECORD ownership for the reviewed CUDA Python tuple."""

import hashlib
import json
import zipfile

import pytest

from . import build_native, native_install as install

VERSIONS = {"cuda-python": "13.3.1", "cuda-bindings": "13.3.1", "cuda-core": "1.0.1"}


def wheel(path, name, version, extra=()):
    owner = name.replace("-", "_") + "-" + version + ".dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(owner + "/METADATA", f"Name: {name}\nVersion: {version}\n")
        archive.writestr(owner + "/RECORD", "")
        if name != "cuda-python":
            archive.writestr("cuda/" + name.split("-")[1] + "/__init__.py", "# owned\n")
        for file in extra:
            archive.writestr(file, "unexpected\n")


def policy(tmp_path):
    selected = []
    for name, version in VERSIONS.items():
        file = tmp_path / (name.replace("-", "_") + ".whl")
        wheel(file, name, version)
        selected.append(
            {
                "name": name,
                "version": version,
                "path": file.name,
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                "source_url": "https://files.pythonhosted.org/" + file.name,
            }
        )
    return {"_root": str(tmp_path), "foundation": {"runtime_dependencies": selected}}


def test_complete_reviewed_cuda_tuple_stages_exact_policy_wheels(tmp_path):
    selected = policy(tmp_path)
    context = tmp_path / "context"
    context.mkdir()
    records = build_native.prepare_runtime_dependencies(selected, context)
    assert {name: record["version"] for name, record in records.items()} == VERSIONS
    for row in selected["foundation"]["runtime_dependencies"]:
        assert install.sha(context / "wheels" / row["path"]) == row["sha256"]


@pytest.mark.parametrize("defect", ["partial", "version", "hash"])
def test_unreviewed_cuda_tuple_or_wheel_refused(tmp_path, defect):
    selected = policy(tmp_path)
    if defect == "partial":
        selected["foundation"]["runtime_dependencies"].pop()
    if defect == "version":
        selected["foundation"]["runtime_dependencies"][0]["version"] = "13.4.1"
    if defect == "hash":
        selected["foundation"]["runtime_dependencies"][0]["sha256"] = "f" * 64
    context = tmp_path / "context"
    context.mkdir()
    with pytest.raises(ValueError):
        build_native.prepare_runtime_dependencies(selected, context)


@pytest.mark.parametrize(
    "name,path",
    [
        ("cuda-python", "cuda/__init__.py"),
        ("cuda-bindings", "cuda/pathfinder/__init__.py"),
        ("cuda-core", "unrelated.py"),
    ],
)
def test_cuda_wheel_may_only_write_its_reviewed_namespace(tmp_path, name, path):
    file = tmp_path / "wheel.whl"
    wheel(file, name, VERSIONS[name], [path])
    with pytest.raises(ValueError, match="namespace"):
        install.cuda_wheel_paths(name, file, VERSIONS[name])


def test_exact_record_scope_does_not_own_cuda_pathfinder(tmp_path, monkeypatch):
    monkeypatch.setattr(install, "SITE", tmp_path)
    selected = str(tmp_path / "cuda/bindings/driver.so")
    assert install.package_owned(selected, [], [], {selected})
    assert not install.package_owned(
        tmp_path / "cuda/pathfinder/__init__.py", [], [], {selected}
    )


def test_foreign_record_collision_is_not_authorized_by_exact_path(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(install, "SITE", tmp_path)
    path = str(tmp_path / "cuda/bindings/shared.py")
    monkeypatch.setattr(
        install,
        "distribution_ownership",
        lambda paths: {str(p): [{"distribution": "cuda-pathfinder"}] for p in paths},
    )
    with pytest.raises(ValueError, match="ownership conflicts"):
        install.audit_selected_ownership(
            {"cuda-bindings": {path: "a" * 64}}, [], [], {}, [path], {path}
        )


def test_unrecorded_namespace_destination_cannot_be_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(install, "SITE", tmp_path)
    path = tmp_path / "cuda/core/custom.py"
    path.parent.mkdir(parents=True)
    path.write_text("unowned")
    monkeypatch.setattr(
        install, "distribution_ownership", lambda paths: {str(p): [] for p in paths}
    )
    with pytest.raises(ValueError, match="ownership conflicts"):
        install.audit_selected_ownership(
            {"cuda-core": {}}, [], [], {}, [str(path)], {str(path)}
        )


def test_protected_cuda_neighbors_and_versions_are_inventoried(tmp_path, monkeypatch):
    monkeypatch.setattr(install, "SITE", tmp_path)
    neighbor = tmp_path / "cuda/pathfinder/owned.py"
    neighbor.parent.mkdir(parents=True)
    neighbor.write_text("keep")
    extra = tmp_path / "cuda/unrecorded.py"
    extra.write_text("also keep")
    versions = {
        "cuda-pathfinder": "1.8.1",
        "nvidia-cutlass-dsl": "4.6.2",
        "torch": "2.13.0",
    }
    monkeypatch.setattr(install.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(
        install,
        "distribution_files",
        lambda name: {str(neighbor): install.sha(neighbor)},
    )
    preserved, files = install.cuda_protected_inputs(
        {name: {"version": version} for name, version in VERSIONS.items()}
    )
    assert preserved == versions and str(extra) in files
    neighbor.write_text("changed")
    with pytest.raises(ValueError, match="payload differs"):
        install.verify_files(files)
    versions["cuda-pathfinder"] = "1.9.0"
    with pytest.raises(ValueError, match="protected"):
        install.cuda_protected_inputs({"cuda-core": {"version": "1.0.1"}})


def test_missing_new_cuda_distribution_has_empty_old_record(monkeypatch):
    def missing(name):
        raise install.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(install, "distribution_files", missing)
    assert install.selected_distribution_files("cuda-core") == {}
    with pytest.raises(install.metadata.PackageNotFoundError):
        install.selected_distribution_files("vllm")


def test_old_cuda_record_cannot_claim_pathfinder(tmp_path, monkeypatch):
    monkeypatch.setattr(install, "SITE", tmp_path)
    with pytest.raises(ValueError, match="namespace"):
        install.cuda_record_paths(
            "cuda-bindings", {str(tmp_path / "cuda/pathfinder/owned.py"): "a" * 64}
        )


def test_policy_hashes_exact_runtime_wheel_inputs(tmp_path):
    from .demo import setup
    from .contracts import load_policy

    file, _ = setup(tmp_path / "demo")
    value = json.loads(file.read_text())
    value.setdefault("foundation", {})["runtime_dependencies"] = policy(file.parent)[
        "foundation"
    ]["runtime_dependencies"]
    file.write_text(json.dumps(value))
    loaded = load_policy(file)
    for item in value["foundation"]["runtime_dependencies"]:
        assert loaded["_inputs"][item["path"]] == item["sha256"]
    (file.parent / value["foundation"]["runtime_dependencies"][0]["path"]).write_bytes(
        b"changed"
    )
    with pytest.raises(ValueError, match="wheel differs"):
        load_policy(file)
