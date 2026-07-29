"""Offline contracts for the public overlay bundle and capability gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

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
capability = load_script("public-capability-gate.py")


def test_overlay_spec_covers_every_public_runtime_python_module():
    document = json.loads(SPEC.read_text(encoding="utf-8"))
    listed = set(document["files"])
    expected = {
        path.relative_to(REPO).as_posix()
        for path in (REPO / "spark_transport/integrations/vllm").glob("*.py")
        if not path.name.startswith("test_")
        and not path.name.startswith("probe_")
        and path.name != "tp4_numerical_audit.py"
    }
    for package in ("adaptive_mtp_controller", "q2r_phase_timing"):
        expected.update(
            path.relative_to(REPO).as_posix()
            for path in (REPO / "spark_transport/experiments" / package).glob("*.py")
            if not path.name.startswith("test_")
        )
    assert listed == expected


def test_overlay_build_is_content_addressed(tmp_path):
    output = tmp_path / "bundle"
    manifest = overlay.build(REPO, SPEC, output)
    assert len(manifest["files"]) == 30
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


def test_capability_gate_reports_missing_surface(tmp_path, monkeypatch):
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "config.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(capability, "vllm_root", lambda: package)
    monkeypatch.setattr(
        capability.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="--attention-backend\n", stderr=""
        ),
    )
    report = capability.evaluate("vllm")
    assert report["passed"] is False
    assert report["cli_flags"]["--attention-backend"] is True
    assert report["source_tokens"]["B12X_MLA_SPARSE"] is False
    assert report["failures"]


def test_public_entrypoint_has_valid_shell_syntax():
    script = (RUNTIME / "public-entrypoint.sh").read_text(encoding="utf-8")
    result = subprocess.run(
        ["bash", "-n"],
        input=script.replace("\r\n", "\n").encode("utf-8"),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")


def test_containerfile_installs_attested_overlay_and_entrypoint():
    text = (RUNTIME / "Containerfile").read_text(encoding="utf-8")
    assert "--tree public-overlay=/opt/spark-vllm" in text
    assert "--image-digest external" in text
    assert "COPY --from=vllm-build /out/public-overlay /opt/spark-vllm" in text
    assert 'ENTRYPOINT ["/opt/sparkring/public-entrypoint.sh"]' in text


def test_entrypoint_attests_before_capability_probe_can_import_vllm():
    text = (RUNTIME / "public-entrypoint.sh").read_text(encoding="utf-8")
    assert text.index("verify-runtime.py") < text.index("public-capability-gate.py")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
