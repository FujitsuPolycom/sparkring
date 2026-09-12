"""Offline contracts for the modules listed in runtime/public-overlay-files.json."""

from __future__ import annotations

import importlib.util
import hashlib
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
    assert {record["source"]: record["path"] for record in manifest["files"]} == {
        source: Path(source).name for source in expected
    }
    assert (output / overlay.MANIFEST).is_file()
    for record in manifest["files"]:
        path = output / record["path"]
        assert path.is_file()
        assert overlay.sha256_file(path) == record["sha256"]
        assert path.read_bytes() == (REPO / record["source"]).read_bytes()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_experiment_source_keeps_its_subdirectory(tmp_path):
    repo = tmp_path / "repo"
    relative = "spark_transport/experiments/example/module.py"
    source = repo / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(b"value = 3\n")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema": overlay.SCHEMA, "files": [relative]}))
    output = tmp_path / "bundle"
    manifest = overlay.build(repo, spec, output)
    assert manifest["files"] == [{
        "source": relative, "path": "example/module.py",
        "sha256": hashlib.sha256(b"value = 3\n").hexdigest(),
    }]
    assert (output / "example/module.py").read_bytes() == b"value = 3\n"


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


@pytest.mark.parametrize("filename", [overlay.MANIFEST, overlay.MANIFEST.upper(), "SparkRing-Overlay-Manifest.json"])
def test_manifest_filename_is_reserved_before_output_creation(tmp_path, filename):
    repo = tmp_path / "repo"
    relative = "spark_transport/integrations/vllm/" + filename
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


@pytest.mark.parametrize("value", [None, True, 7, [{}], "schema"])
def test_nondocument_spec_fails_with_validation_error(tmp_path, value):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="schema"):
        overlay.build(tmp_path, spec, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("files", [
    ["spark_transport/experiments/sparkring-overlay-manifest.json/module.py"],
    ["spark_transport/integrations/vllm/Module.py", "spark_transport/integrations/vllm/module.py"],
    ["spark_transport/integrations/vllm/nested", "spark_transport/experiments/nested/module.py"],
])
def test_destination_collisions_fail_before_output_creation(tmp_path, files):
    repo = tmp_path / "repo"
    for name in files:
        source = repo / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("payload")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema": overlay.SCHEMA, "files": files}))
    output = tmp_path / "bundle"
    with pytest.raises(ValueError):
        overlay.build(repo, spec, output)
    assert not output.exists()
