"""Image extension admission is bounded to reviewed transport and metadata owners."""

import hashlib
import io
import json
from pathlib import PurePosixPath

import pytest

from . import native_install as module
from . import build_native


@pytest.mark.parametrize(
    "path,scope",
    [
        ("/opt/sparkring/transports/sparkring_transport_selector.py", "transport"),
        (
            "/opt/sparkring/transports/tp2-rocenante-adaptive-prepared/roce/_roce_proxy.c",
            "transport",
        ),
        ("/opt/venv/lib/python3.12/site-packages/sparkring_transport.pth", "transport"),
        ("/opt/sparkring/licenses/components.md", "fresh-metadata"),
        ("/opt/sparkring/licenses/shared/shared-fixture.md", "fresh-metadata"),
        ("/opt/sparkring/releases/shared/manifest.json", "fresh-metadata"),
        ("/opt/sparkring/releases/shared/2026.09-candidate.json", "fresh-metadata"),
    ],
)
def test_reviewed_transport_and_fresh_metadata_paths_are_admitted(path, scope):
    assert module.feature_asset_scope(path) == scope


@pytest.mark.parametrize(
    "path",
    [
        "/opt/sparkring/transports/tp2-rocenante-adaptive/roce/api.py",
        "/opt/sparkring/transports/unreviewed/driver.so",
        "/opt/sparkring/bin/native-image.py",
        "/opt/venv/bin/python",
        "/opt/sglang/python.py",
        "/opt/sparkring/releases/shared/install.py",
        "/opt/sparkring/releases/shared/subdir/manifest.json",
        "/opt/sparkring/releases/shared/../oops.json",
        "/opt/sparkring/licenses/shared/../components.md",
        "/opt/sparkring/licenses/shared/subdir/components.md",
        "/opt/sparkring/licenses/shared/shared-fixture.py",
        "/opt/sparkring/licenses/shared/.md",
        "/opt/sparkring/licenses/shared/" + "a" * 97 + ".md",
        "/opt/sparkring/licenses/shared//shared-fixture.md",
    ],
)
def test_legacy_and_unowned_destinations_remain_protected(path):
    with pytest.raises(ValueError):
        module.feature_asset_scope(path)


def fixture(tmp_path, monkeypatch):
    root, site = tmp_path / "image", tmp_path / "site"
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "SITE", site)
    preimages = {}
    for name in (
        "preparation/__init__.py",
        "preparation/types.py",
        "preparation/tuning.py",
        "preparation/session.py",
        "_lib/compile_plan.py",
        "_lib/program_cache.py",
    ):
        path = site / "b12x" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        preimages[str(path)] = module.sha(path)
    bundle = root / "transports" / module.PREPARED_TRANSPORT_PROFILE
    data = b"transport source\n"
    manifest = {
        "schema": "sparkring-transport-bundle/v1",
        "name": module.PREPARED_TRANSPORT_PROFILE,
        "files": {"roce/api.py": hashlib.sha256(data).hexdigest()},
        "image_source_preimages": preimages,
    }
    payloads = {
        bundle / "manifest.json": json.dumps(manifest).encode(),
        bundle / "roce/api.py": data,
        root / "transports/sparkring_transport_selector.py": b"selector\n",
    }
    return bundle, manifest, payloads


def test_transport_proof_binds_complete_payload_and_installed_api(
    tmp_path, monkeypatch
):
    bundle, manifest, payloads = fixture(tmp_path, monkeypatch)
    proof = module.verify_transport_assets(payloads)[module.PREPARED_TRANSPORT_PROFILE]
    assert proof["image_source_preimages"] == manifest["image_source_preimages"]
    assert proof["files"] == {
        str(bundle / "roce/api.py"): manifest["files"]["roce/api.py"]
    }
    assert proof["serving_qualified"] is False


@pytest.mark.parametrize(
    "defect",
    [
        "missing_file",
        "extra_file",
        "changed_payload",
        "changed_api",
        "missing_api",
        "missing_manifest",
    ],
)
def test_transport_payload_or_api_drift_is_rejected(tmp_path, monkeypatch, defect):
    bundle, manifest, payloads = fixture(tmp_path, monkeypatch)
    if defect == "missing_file":
        del payloads[bundle / "roce/api.py"]
    elif defect == "extra_file":
        payloads[bundle / "roce/extra.py"] = b"extra"
    elif defect == "changed_payload":
        payloads[bundle / "roce/api.py"] += b"changed"
    elif defect == "changed_api":
        from pathlib import Path

        Path(next(iter(manifest["image_source_preimages"]))).write_bytes(b"changed")
    elif defect == "missing_api":
        manifest["image_source_preimages"].pop(
            next(iter(manifest["image_source_preimages"]))
        )
        payloads[bundle / "manifest.json"] = json.dumps(manifest).encode()
    elif defect == "missing_manifest":
        del payloads[bundle / "manifest.json"]
    with pytest.raises(ValueError):
        module.verify_transport_assets(payloads)


@pytest.mark.parametrize(
    "target",
    [
        "/opt/sparkring/releases/shared/manifest.json",
        "/opt/sparkring/licenses/shared/shared-fixture.md",
    ],
)
def test_release_metadata_update_cannot_overwrite_an_existing_identity(
    tmp_path, target
):
    source = tmp_path / "sources.json"
    source.write_text("{}")
    manifest = tmp_path / "features.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "sparkring-native-feature-update/v1",
                "assets": {
                    target: {
                        "source": source.name,
                        "sha256": module.sha(source),
                        "parent_sha256": "a" * 64,
                    }
                },
            }
        )
    )
    policy = {
        "_root": tmp_path,
        "foundation": {
            "feature_update": {
                "manifest": manifest.name,
                "sha256": module.sha(manifest),
            }
        },
    }
    with pytest.raises(ValueError, match="fresh"):
        build_native.prepare_feature_update(policy, tmp_path / "context")


@pytest.mark.parametrize("defect", [None, "existing", "parent-hash", "symlink"])
def test_versioned_metadata_install_retains_inherited_files(monkeypatch, defect):
    """Run the installer against a filesystem boundary with immutable old metadata."""
    files, symlinks = {}, set()

    class ImagePath(PurePosixPath):
        def is_symlink(self):
            return str(self) in symlinks

        def exists(self):
            return str(self) in files

        def is_file(self):
            return self.exists()

        def read_bytes(self):
            return files[str(self)]

        def read_text(self):
            return self.read_bytes().decode()

        def open(self, mode):
            assert mode == "rb"
            return io.BytesIO(self.read_bytes())

        def write_bytes(self, data):
            files[str(self)] = data

        def mkdir(self, **kwargs):
            pass

    monkeypatch.setattr(module, "Path", ImagePath)
    monkeypatch.setattr(module, "ROOT", ImagePath("/opt/sparkring"))
    catalog = "/opt/sparkring/features/capabilities.json"
    target = "/opt/sparkring/licenses/shared/shared-fixture.md"
    files[catalog] = json.dumps(
        {
            "schema": "sparkring-image-capabilities/v1",
            "features": {"retained": {}},
        }
    ).encode()
    files["/opt/sparkring/licenses/components.md"] = b"Inherited license index.\n"
    files["/opt/sparkring/releases/shared/shared-2026.09.3.json"] = b"{}\n"
    assets = {
        catalog: {"parent_sha256": module.sha(ImagePath(catalog))},
        target: {"parent_sha256": None},
    }
    payloads = {catalog: files[catalog], target: b"Versioned license index.\n"}
    for name, payload in payloads.items():
        assets[name].update(
            source="fixture", sha256=hashlib.sha256(payload).hexdigest()
        )
        files["/context/feature-assets" + name] = payload
    if defect == "existing":
        files[target] = b"Already published metadata.\n"
    elif defect == "parent-hash":
        assets[target]["parent_sha256"] = "a" * 64
    elif defect == "symlink":
        symlinks.add(str(ImagePath(target).parent))
    files["/context/feature-update.json"] = json.dumps(
        {
            "schema": "sparkring-native-feature-update/v1",
            "parent_capabilities_sha256": module.sha(ImagePath(catalog)),
            "assets": assets,
        }
    ).encode()
    descriptor = {
        "input_sha256": "b" * 64,
        "feature_update": {
            "file": "feature-update.json",
            "sha256": module.sha(ImagePath("/context/feature-update.json")),
        },
    }
    before = files.copy()
    if defect:
        with pytest.raises(ValueError, match="preimage|fresh|symlink"):
            module.install_feature_update(ImagePath("/context"), descriptor)
        assert files == before
    else:
        installed, receipt = module.install_feature_update(
            ImagePath("/context"), descriptor
        )
        assert installed[target] == hashlib.sha256(payloads[target]).hexdigest()
        assert receipt["serving_qualified"] is False
        assert all(files[name] == data for name, data in before.items())
