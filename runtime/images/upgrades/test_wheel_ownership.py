"""Runtime dependency installation preserves foreign-owned namespace helpers."""

import csv
import base64
import hashlib
import io
import zipfile
from types import SimpleNamespace as NS

import pytest

from . import native_install as module


def wheel(path, *, mismatch=False, console=None):
    with zipfile.ZipFile(path, "w") as archive:
        files = {
            "flashinfer_python-0.6.18.post1.dist-info/METADATA": b"Name: flashinfer-python\nVersion: 0.6.18.post1\n",
            "flashinfer/runtime.py": b"VALUE=42\n",
            "flashinfer/kernel.so": b"native-unmodified",
        }
        for name in module.FLASHINFER_GLOBAL_BUILD_HELPERS:
            files[name] = (name + " source").encode()
            files["flashinfer/data/" + name] = files[name] + (
                b" drift" if mismatch else b""
            )
        if console:
            files["flashinfer_python-0.6.18.post1.dist-info/entry_points.txt"] = (
                "[console_scripts]\n" + console + " = flashinfer:main\n"
            ).encode()
        for name, data in files.items():
            archive.writestr(name, data)
        rows = [
            [name, "retained-digest", str(len(data))] for name, data in files.items()
        ]
        record = "flashinfer_python-0.6.18.post1.dist-info/RECORD"
        rows.append([record, "", ""])
        stream = io.StringIO()
        csv.writer(stream).writerows(rows)
        archive.writestr(record, stream.getvalue())


def test_helper_isolation_preserves_publisher_and_every_runtime_member(tmp_path):
    source = tmp_path / "fi.whl"
    wheel(source)
    original = source.read_bytes()
    normalized, proof = module.isolate_flashinfer_build_helpers(
        source, tmp_path / "normalized", "0.6.18.post1"
    )
    assert source.read_bytes() == original
    with zipfile.ZipFile(source) as before, zipfile.ZipFile(normalized) as after:
        assert set(before.namelist()) - set(after.namelist()) == set(
            module.FLASHINFER_GLOBAL_BUILD_HELPERS
        )
        for name in after.namelist():
            if not name.endswith("/RECORD"):
                assert before.read(name) == after.read(name)
        rows = list(
            csv.reader(
                io.StringIO(
                    after.read(
                        "flashinfer_python-0.6.18.post1.dist-info/RECORD"
                    ).decode()
                )
            )
        )
        assert not set(module.FLASHINFER_GLOBAL_BUILD_HELPERS) & {
            row[0] for row in rows
        }
    assert proof["publisher_wheel_sha256"] == hashlib.sha256(original).hexdigest()
    assert proof["installation_wheel_sha256"] == module.sha(normalized)


@pytest.mark.parametrize("defect", ["version", "duplicate"])
def test_unreviewed_or_nonduplicate_helpers_cannot_be_filtered(tmp_path, defect):
    source = tmp_path / "fi.whl"
    wheel(source, mismatch=defect == "duplicate")
    with pytest.raises(ValueError):
        module.isolate_flashinfer_build_helpers(
            source,
            tmp_path / "output",
            "0.7.0" if defect == "version" else "0.6.18.post1",
        )


def fake_owners(monkeypatch, values):
    monkeypatch.setattr(
        module,
        "distribution_ownership",
        lambda paths: {str(path): values.get(str(path), []) for path in paths},
    )


def test_owner_scan_includes_unselected_distributions_and_record_hashes(
    tmp_path, monkeypatch
):
    class Entry(str):
        hash = NS(
            mode="sha256",
            value=base64.urlsafe_b64encode(bytes.fromhex("a" * 64))
            .decode()
            .rstrip("="),
        )

    distributions = [
        NS(
            metadata={"Name": name},
            version="1",
            files=[Entry("build_backend.py")],
            locate_file=lambda entry: tmp_path / str(entry),
        )
        for name in ("flashinfer-python", "torch_c_dlpack_ext")
    ]
    monkeypatch.setattr(module.metadata, "distributions", lambda: distributions)
    path = str((tmp_path / "build_backend.py").resolve())
    owners = module.distribution_ownership([path])[path]
    assert {item["distribution"] for item in owners} == {
        "flashinfer-python",
        "torch-c-dlpack-ext",
    }
    assert {item["record_sha256"] for item in owners} == {"a" * 64}


def test_foreign_owned_helper_is_backed_up_with_preexisting_record_mismatch(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(module, "SITE", tmp_path)
    path = tmp_path / "build_backend.py"
    path.write_bytes(b"baseline helper")
    owners = [
        {"distribution": "flashinfer-python", "record_sha256": module.sha(path)},
        {"distribution": "torch-c-dlpack-ext", "record_sha256": "a" * 64},
    ]
    fake_owners(monkeypatch, {str(path): owners})
    audit, backups = module.audit_selected_ownership(
        {"flashinfer-python": {str(path): module.sha(path)}},
        ["flashinfer"],
        [],
        {str(path): module.sha(path)},
    )
    assert backups[str(path)]["bytes"] == b"baseline helper"
    assert audit["collisions"][str(path)][1]["record_matches_baseline_bytes"] is False
    path.unlink()  # Simulate pip removing a file claimed by the old FI RECORD.
    module.restore_build_helpers(backups)
    assert path.read_bytes() == b"baseline helper"


def test_unexpected_helper_write_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "SITE", tmp_path)
    path = tmp_path / "build_utils.py"
    path.write_bytes(b"unexpected new write")
    with pytest.raises(ValueError, match="unexpectedly exists"):
        module.restore_build_helpers({})
    assert path.read_bytes() == b"unexpected new write"


@pytest.mark.parametrize("name", ["pip", "python", "sglang", "unreviewed_helper.py"])
def test_unreviewed_record_extras_fail_before_install(tmp_path, monkeypatch, name):
    monkeypatch.setattr(module, "SITE", tmp_path / "site")
    path = str(tmp_path / name)
    fake_owners(monkeypatch, {})
    with pytest.raises(ValueError, match="ownership conflicts"):
        module.audit_selected_ownership(
            {"flashinfer-python": {path: "a" * 64}}, ["flashinfer"], [], {}
        )


def test_collision_with_foreign_runtime_owner_is_not_masked_by_selected_prefix(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(module, "SITE", tmp_path)
    path = str(tmp_path / "flashinfer/kernel.so")
    fake_owners(
        monkeypatch,
        {path: [{"distribution": "foreign-runtime", "record_sha256": "a" * 64}]},
    )
    with pytest.raises(ValueError, match="foreign-runtime"):
        module.audit_selected_ownership(
            {"flashinfer-python": {path: "a" * 64}}, ["flashinfer"], [], {}
        )


def test_candidate_wheel_cannot_add_a_reserved_console_script(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "SITE", tmp_path / "site")
    source = tmp_path / "fi.whl"
    wheel(source, console="python")
    normalized, _ = module.isolate_flashinfer_build_helpers(
        source, tmp_path / "output", "0.6.18.post1"
    )
    paths, roots = module.wheel_install_paths({"flashinfer-python": normalized})
    fake_owners(monkeypatch, {})
    with pytest.raises(ValueError, match="candidate_wheel_destination"):
        module.audit_selected_ownership(
            {"flashinfer-python": {}}, ["flashinfer"], roots, {}, paths
        )
