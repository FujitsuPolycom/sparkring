"""A head-package upgrade must not make the retained controller unusable."""
import hashlib
import json
import subprocess

import pytest

from runtime.host import retained_source


def test_saved_status_never_constructs_a_live_runner(monkeypatch):
    from scripts import installer_runner
    monkeypatch.setattr(installer_runner, "Runner", lambda _: pytest.fail("Cached status contacted execution boundary"))
    monkeypatch.setattr(retained_source.installer, "status", lambda path: {"state": "saved", "path": path})
    assert retained_source._operation("fixture", "saved-status") == {"state": "saved", "path": "fixture"}


@pytest.fixture
def archived_program(tmp_path):
    source = tmp_path / "program"
    source.mkdir()
    for name in ("runtime", "runtime/common", "runtime/host", "scripts"):
        path = source / name
        path.mkdir(exist_ok=True)
        (path / "__init__.py").write_text("")
    (source / "runtime/common/installer.py").write_text("def apply(directory, operation, **kw): return {'program':'retained','operation':operation,'executed':True}\n")
    (source / "runtime/host/progress.py").write_text("from contextlib import nullcontext\ndef run(name): return nullcontext()\n")
    (source / "scripts/installer_runner.py").write_text("class Runner:\n def __init__(self, directory): pass\n")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()
    git("init", "--quiet")
    git("add", ".")
    git("-c", "user.name=Installer fixture", "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "Fixture")
    revision = git("rev-parse", "HEAD")
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    bundle = deployment / "source.bundle"
    git("bundle", "create", str(bundle), "HEAD")
    (deployment / "deployment.lock.json").write_text(json.dumps({"source_revision": revision,
                 "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest()}))
    return deployment


def test_new_package_replays_the_archived_controller_without_operator_checkout(archived_program, tmp_path, monkeypatch):
    monkeypatch.setattr(retained_source.distribution, "identity", lambda _: "0" * 40)
    result = retained_source.apply(archived_program, "down", cache=tmp_path / "cache")
    assert result == {"program": "retained", "operation": "down", "executed": True}


def test_changed_retained_bundle_is_refused_before_execution(archived_program, tmp_path):
    with (archived_program / "source.bundle").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="bundle differs"):
        retained_source.checkout(archived_program, tmp_path / "cache")


def test_dirty_cached_controller_is_refused(archived_program, tmp_path):
    root = retained_source.checkout(archived_program, tmp_path / "cache")
    (root / "runtime/common/installer.py").write_text("raise RuntimeError('unapproved source')\n")
    with pytest.raises(ValueError, match="modified"):
        retained_source.checkout(archived_program, tmp_path / "cache")
