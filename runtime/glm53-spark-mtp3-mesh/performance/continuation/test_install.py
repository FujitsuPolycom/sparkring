"""Verify exact continuation inputs and ownership updates without GPU access."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("continuation_source_install", HERE / "install.py")
INSTALL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(INSTALL)


def fixture(tmp_path):
    manifest, sources = INSTALL.package()
    ownership = json.loads((HERE.parent / "checkpoints/ownership-contract.json").read_text())
    for name, row in manifest["files"].items():
        baseline = HERE.parent / "checkpoints/payload-by-sha" / row["before_sha256"] / Path(name).name
        assert hashlib.sha256(baseline.read_bytes()).hexdigest() == row["before_sha256"]
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(baseline.read_bytes())
    return manifest, sources, ownership


def test_exact_sources_replace_only_four_verified_ownership_rows(tmp_path):
    manifest, sources, ownership = fixture(tmp_path)
    original = copy.deepcopy(ownership)
    result = INSTALL.apply(tmp_path, ownership)
    assert len(sources) == 4
    assert result["manifest_sha256"] == INSTALL.MANIFEST_SHA256
    for before, after in zip(original["files"], ownership["files"]):
        if before["path"] in sources:
            assert after["sha256"] == manifest["files"][before["path"]]["after_sha256"]
            assert (tmp_path / before["path"]).read_bytes() == sources[before["path"]]
            assert {k: v for k, v in after.items() if k != "sha256"} == {k: v for k, v in before.items() if k != "sha256"}
        else:
            assert after == before


@pytest.mark.parametrize("corrupt", ["runtime", "ownership"])
def test_all_preimages_checked_before_any_runtime_write(tmp_path, corrupt):
    manifest, sources, ownership = fixture(tmp_path)
    target = list(sources)[-1]
    if corrupt == "runtime":
        (tmp_path / target).write_bytes(b"unsupported")
    else:
        next(row for row in ownership["files"] if row["path"] == target)["sha256"] = "0" * 64
    before = {name: (tmp_path / name).read_bytes() for name in sources}
    original = copy.deepcopy(ownership)
    with pytest.raises(ValueError, match="preimage differs"):
        INSTALL.apply(tmp_path, ownership)
    assert ownership == original
    assert before == {name: (tmp_path / name).read_bytes() for name in sources}


@pytest.mark.parametrize("name", ["manifest.json", "source.tar.gz"])
def test_modified_source_package_is_rejected(tmp_path, name):
    shutil.copytree(HERE, tmp_path / "context", ignore=shutil.ignore_patterns("__pycache__"))
    path = tmp_path / "context" / name
    path.write_bytes(path.read_bytes() + b"unexpected")
    with pytest.raises(ValueError, match="differs"):
        INSTALL.package(tmp_path / "context")


def test_failed_write_verification_does_not_attest_ownership(tmp_path, monkeypatch):
    manifest, sources, ownership = fixture(tmp_path)
    original = copy.deepcopy(ownership)
    victim = tmp_path / next(iter(sources))
    write_bytes = Path.write_bytes

    def corrupt_write(path, data):
        return write_bytes(path, b"corrupt" if path == victim else data)

    monkeypatch.setattr(Path, "write_bytes", corrupt_write)
    with pytest.raises(ValueError, match="postimage differs"):
        INSTALL.apply(tmp_path, ownership)
    assert ownership == original
