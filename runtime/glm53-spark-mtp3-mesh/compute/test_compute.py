import hashlib
import ast
import io
import importlib.util
import json
import tarfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(result)
    return result


apply_compute = _module("apply_compute")
prepare_compute_source = _module("prepare_compute_source")
verify_compute = _module("verify_compute")


def test_compute_json_io_declares_utf8():
    for name in ("apply_compute.py", "prepare_compute_source.py", "verify_compute.py"):
        tree = ast.parse((HERE / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("read_text", "write_text"):
                values = {arg.arg: arg.value for arg in node.keywords}
                assert ast.literal_eval(values["encoding"]) == "utf-8", (name, node.lineno)
                if node.func.attr == "write_text":
                    assert ast.literal_eval(values["newline"]) == "\n", (name, node.lineno)


@pytest.mark.parametrize("relative", ["../outside.py", "/tmp/outside.py", "vllm/../../outside.py", "b12x/not-vllm.py", "vllm\\outside.py"])
def test_verify_rejects_uncontained_override_paths(tmp_path, relative):
    lock = tmp_path / "source-lock.json"
    lock.write_text(json.dumps({"vllm": {"files": [[relative, "base", "result"]]}}), encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({
        "source_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "vllm_overrides": {relative: "result"},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe vLLM override path"):
        verify_compute.verify(tmp_path, receipt, lock)


@pytest.mark.parametrize("fault", [None, "base", "result", "archive", "extra", "duplicate"])
def test_b12x_selector_overrides_fail_closed(tmp_path: Path, fault: str | None) -> None:
    root = tmp_path / "source"
    path = root / "b12x" / "selector.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"base")
    archive = tmp_path / "selector.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        names = ["b12x/selector.py"]
        if fault == "extra":
            names.append("b12x/unexpected.py")
        for name in names:
            member = tarfile.TarInfo(name)
            member.size = len(b"result")
            output.addfile(member, io.BytesIO(b"result"))
    contract = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": [["b12x/selector.py", hashlib.sha256(b"base").hexdigest(),
                   hashlib.sha256(b"result").hexdigest()]],
    }
    if fault == "base":
        contract["files"][0][1] = "wrong"
    elif fault == "result":
        contract["files"][0][2] = "wrong"
    elif fault == "archive":
        contract["archive_sha256"] = "wrong"
    elif fault == "duplicate":
        contract["files"].append(contract["files"][0])
    if fault:
        with pytest.raises(ValueError):
            prepare_compute_source._apply_b12x_overrides(root, archive, contract)
        assert path.read_bytes() == b"base"
    else:
        prepare_compute_source._apply_b12x_overrides(root, archive, contract)
        assert path.read_bytes() == b"result"


def test_source_lock_binds_patch_routes_and_environment() -> None:
    lock = json.loads((HERE / "source-lock.json").read_text())
    patch = HERE / lock["vllm"]["patch"]
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == lock["vllm"][
        "patch_sha256"
    ]
    archive = HERE / lock["vllm"]["replacement_archive"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == lock["vllm"][
        "replacement_archive_sha256"
    ]
    files = lock["vllm"]["files"]
    assert len(files) == 24
    assert len({entry[0] for entry in files}) == len(files)
    assert all(base != result for _, base, result in files)
    assert lock["b12x"]["revision"] == "ef308bac0f3b3eb8fea63e4013afc0c2ea1c6301"
    assert len(lock["b12x"]["overrides"]["files"]) == 3
    assert lock["vllm"]["loader_donor_revision"] == "17e341b9ede04269f81fcac69a29951a0668a94a"
    assert lock["vllm"]["rng_donor_revision"] == "44e6766e3397e8fe8ed9c1fa8a8d2783bb4a2ae8"
    assert lock["environment"] == {
        "VLLM_B12X_DENSE_ACTIVATION_MODE": "auto",
        "VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH": "1",
        "VLLM_LM_HEAD_A16": "1",
        "VLLM_MTP_NVFP4_LM_HEAD": "1",
        "VLLM_MXFP8_LM_HEAD": "0",
    }


def test_package_map_includes_runtime_data(tmp_path: Path) -> None:
    package = tmp_path / "b12x"
    package.mkdir()
    (package / "module.py").write_text("value = 1\n")
    (package / "profile.json.gz").write_bytes(b"profile")
    (package / "README.md").write_text("runtime data\n")
    (package / "ignored.pyc").write_bytes(b"cache")
    files = prepare_compute_source._package_map(tmp_path, "b12x")
    assert set(files) == {
        "b12x/README.md",
        "b12x/module.py",
        "b12x/profile.json.gz",
    }


def test_verify_rejects_an_empty_b12x_map(tmp_path: Path) -> None:
    source_lock = tmp_path / "source-lock.json"
    source_lock.write_text(
        json.dumps(
            {
                "vllm": {"files": []},
                "b12x": {"package_files_sha256": "unused"},
                "environment": {},
            }
        )
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "source_lock_sha256": hashlib.sha256(
                    source_lock.read_bytes()
                ).hexdigest(),
                "vllm_overrides": {},
                "b12x_files": {},
                "environment": {},
                "target_head_quantization": False,
            }
        )
    )
    with pytest.raises(ValueError, match="no B12X package map"):
        verify_compute.verify(tmp_path, receipt, source_lock)


def test_vllm_install_fails_before_patch_when_base_hash_drifts(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    patch = prepared / "change.patch"
    patch.write_text("")
    archive = prepared / "files.tar.gz"
    with tarfile.open(archive, "w:gz"):
        pass
    site = tmp_path / "site"
    file = site / "vllm/example.py"
    file.parent.mkdir(parents=True)
    file.write_text("unexpected\n")
    lock = {
        "vllm": {
            "patch": patch.name,
            "patch_sha256": hashlib.sha256(b"").hexdigest(),
            "replacement_archive": archive.name,
            "replacement_archive_sha256": hashlib.sha256(
                archive.read_bytes()
            ).hexdigest(),
            "files": [
                [
                    "vllm/example.py",
                    hashlib.sha256(b"expected\n").hexdigest(),
                    "x",
                ]
            ],
        }
    }
    with pytest.raises(ValueError, match="vLLM base hash mismatch"):
        apply_compute._install_vllm(prepared, site, lock)


@pytest.mark.parametrize("fault", [None, "base", "result"])
def test_vllm_replacements_are_all_verified_before_install(tmp_path, fault):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    patch = prepared / "change.patch"
    patch.write_bytes(b"fixture patch")
    site = tmp_path / "site"
    (site / "vllm").mkdir(parents=True)
    entries = []
    archive = prepared / "files.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for index in range(2):
            name = f"vllm/file{index}.py"
            before = f"before{index}".encode()
            after = f"after{index}".encode()
            (site / name).write_bytes(before)
            row = [name, hashlib.sha256(before).hexdigest(), hashlib.sha256(after).hexdigest()]
            if index == 1 and fault:
                row[1 if fault == "base" else 2] = "0" * 64
            entries.append(row)
            member = tarfile.TarInfo(name)
            member.size = len(after)
            output.addfile(member, io.BytesIO(after))
    lock = {"vllm": {"patch": patch.name, "patch_sha256": apply_compute._sha256(patch),
                      "replacement_archive": archive.name,
                      "replacement_archive_sha256": apply_compute._sha256(archive), "files": entries}}
    if fault:
        with pytest.raises(ValueError, match=f"vLLM {fault} hash mismatch"):
            apply_compute._install_vllm(prepared, site, lock)
        assert [(site / row[0]).read_bytes() for row in entries] == [b"before0", b"before1"]
    else:
        result = apply_compute._install_vllm(prepared, site, lock)
        assert result == {name: digest for name, _, digest in entries}
        assert [(site / row[0]).read_bytes() for row in entries] == [b"after0", b"after1"]
