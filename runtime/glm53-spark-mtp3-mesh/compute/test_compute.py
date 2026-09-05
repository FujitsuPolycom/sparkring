import hashlib
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
    assert len(files) == 14
    assert len({entry[0] for entry in files}) == len(files)
    assert all(base != result for _, base, result in files)
    assert lock["b12x"]["revision"] == "b58f34eaf978277621efced6678e6713fd7122e4"
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
