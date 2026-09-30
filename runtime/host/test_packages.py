import hashlib
import json
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace

import pytest

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


def worker_bundle(tmp_path, monkeypatch, simulation):
    """A verified bundle and fake apt/dpkg; returns the recorded commands.

    ``simulation`` is the output of apt's simulated install. Its versions are
    integers, which the fake ``dpkg --compare-versions`` compares.
    """
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

    def run(argv, **kw):
        calls.append(argv)
        if argv[:2] == ["dpkg", "--compare-versions"]:
            return SimpleNamespace(returncode=0 if int(argv[2]) < int(argv[4]) else 1)
        return SimpleNamespace(returncode=0, stdout=simulation if "--simulate" in argv else "")
    monkeypatch.setattr(subprocess, "run", run)
    return calls


def apt(calls):
    return [argv for argv in calls if argv[0] == "apt-get"]


def test_worker_apt_uses_only_bundle_and_requests_only_sparkring(tmp_path, monkeypatch):
    calls = worker_bundle(tmp_path, monkeypatch, "Inst sparkring [5] (6 local [arm64])\nInst avahi-utils (1 local [arm64])\n")
    packages.install(tmp_path, apply=True)
    update, plan, command = apt(calls)
    assert update[-1] == "update"
    assert "--simulate" in plan and plan[-2:] == ["install", str(tmp_path / "app.deb")]
    assert command[-2:] == ["install", str(tmp_path / "app.deb")]
    assert "--no-remove" in command and "--yes" in command and "--allow-downgrades" not in command
    assert "Dir::Etc::sourceparts=-" in command
    assert (tmp_path / "bundle.sources.list").read_text().startswith("deb [trusted=yes] file:")


def test_worker_installs_an_earlier_sparkring_that_node_a_selected(tmp_path, monkeypatch):
    # Package versions follow commit time, so returning to main after a newer
    # branch selects an earlier SparkRing version on every Spark.
    calls = worker_bundle(tmp_path, monkeypatch, "Inst sparkring [7] (6 local [arm64])\nInst avahi-utils [1] (2 local)\n")
    packages.install(tmp_path, apply=True)
    command = apt(calls)[-1]
    assert "--allow-downgrades" in command and "--yes" in command


def test_worker_never_downgrades_a_dependency(tmp_path, monkeypatch):
    calls = worker_bundle(tmp_path, monkeypatch, "Inst sparkring [7] (6 local [arm64])\nInst avahi-utils [3] (2 local)\n")
    with pytest.raises(ValueError, match="would downgrade avahi-utils"):
        packages.install(tmp_path, apply=True)
    assert len(apt(calls)) == 2
