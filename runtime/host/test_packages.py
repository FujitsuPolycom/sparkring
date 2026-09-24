import hashlib
import json
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace

from runtime.host import packages


def test_bundle_index_uses_safe_names_and_complete_package_hashes(tmp_path):
    old = tmp_path / "sample_1%3a1_arm64.deb"
    old.write_bytes(b"fixture-archive")
    digest = hashlib.sha256(old.read_bytes()).hexdigest()
    packages.repository_index(tmp_path, run=lambda *a, **k: SimpleNamespace(stdout="Package: sample\nVersion: 1:1\nArchitecture: arm64\n"))
    assert (tmp_path / ("sample_" + digest[:16] + ".deb")).is_file()
    index = (tmp_path / "Packages").read_text()
    assert "Version: 1:1" in index and "SHA256: " + digest in index
    assert "%3a" not in index


def test_worker_apt_uses_only_bundle_and_requests_only_sparkring(tmp_path, monkeypatch):
    files = {"app.deb": b"application", "dependency.deb": b"dependency", "Packages": b"index"}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    document = {"os": {"ID": "ubuntu", "VERSION_ID": '"24.04"'},
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    (tmp_path / "manifest.json").write_text(json.dumps(document))
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: 'ID=ubuntu\nVERSION_ID="24.04"\n' if self == Path("/etc/os-release") else original(self, *a, **k))
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(subprocess, "check_output", lambda argv, **k: "sparkring\n" if argv[2].endswith("app.deb") else "dependency\n")
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))
    packages.install(tmp_path, apply=True)
    assert len(calls) == 2
    assert calls[0][0][-1] == "update"
    command = calls[1][0]
    assert command[-2:] == ["install", str(tmp_path / "app.deb")]
    assert "--no-remove" in command and "--allow-downgrades" not in command
    assert "Dir::Etc::sourceparts=-" in command
    assert (tmp_path / "bundle.sources.list").read_text().startswith("deb [trusted=yes] file:")
