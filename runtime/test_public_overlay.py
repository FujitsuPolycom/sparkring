"""Offline contracts for the modules listed in runtime/public-overlay-files.json."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import pytest

RUNTIME = Path(__file__).resolve().parent
REPO = RUNTIME.parent
SPEC = RUNTIME / "public-overlay-files.json"


def load_script(name: str):
    path = RUNTIME / name
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


overlay = load_script("build-public-overlay.py")


def test_overlay_spec_names_existing_public_adapter_modules():
    document = json.loads(SPEC.read_text(encoding="utf-8"))
    listed = document["files"]
    assert listed
    assert len(listed) == len(set(listed))
    assert all((REPO / path).is_file() for path in listed)
    assert not any("/experiments/" in path for path in listed)
    # Exclude inherited Python startup-hook modules; the corresponding
    # sparkring_nf3_hybrid.pth hook is removed by runtime/exl3-r7/Containerfile.
    assert not any("nf3" in path.lower() for path in listed)


def test_overlay_build_is_content_addressed(tmp_path):
    output = tmp_path / "bundle"
    manifest = overlay.build(REPO, SPEC, output)
    expected = json.loads(SPEC.read_text(encoding="utf-8"))["files"]
    assert len(manifest["files"]) == len(expected)
    assert (output / overlay.MANIFEST).is_file()
    for record in manifest["files"]:
        path = output / record["path"]
        assert path.is_file()
        assert overlay.sha256_file(path) == record["sha256"]


def test_overlay_builder_rejects_unrecognised_layout(tmp_path):
    source = tmp_path / "repo" / "elsewhere" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema": overlay.SCHEMA,
                "files": ["elsewhere/module.py"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported public-overlay"):
        overlay.build(tmp_path / "repo", spec, tmp_path / "output")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


@pytest.mark.parametrize("relative", ["spark_transport/experiments/../escape.py", "spark_transport/integrations/vllm/./escape.py"])
def test_noncanonical_source_cannot_escape_output_or_leave_partial_bundle(tmp_path, relative):
    repo = tmp_path / "repo"
    source = repo / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"replacement")
    sentinel = tmp_path / "escape.py"
    sentinel.write_bytes(b"preserve")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema": overlay.SCHEMA, "files": [relative]}))
    output = tmp_path / "bundle"
    with pytest.raises(ValueError):
        overlay.build(repo, spec, output)
    assert sentinel.read_bytes() == b"preserve"
    assert not output.exists()


def test_manifest_filename_is_reserved_before_output_creation(tmp_path):
    repo = tmp_path / "repo"
    relative = "spark_transport/integrations/vllm/" + overlay.MANIFEST
    source = repo / relative
    source.parent.mkdir(parents=True)
    source.write_text("payload")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema": overlay.SCHEMA, "files": [relative]}))
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="manifest"):
        overlay.build(repo, spec, output)
    assert not output.exists()


def test_nonstring_inventory_has_a_validation_error(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema": overlay.SCHEMA, "files": [{}]}))
    with pytest.raises(ValueError, match="non-empty string"):
        overlay.build(tmp_path, spec, tmp_path / "bundle")
