import hashlib
import importlib.util
from pathlib import Path
import shutil

import pytest

spec = importlib.util.spec_from_file_location(
    "distribution", Path(__file__).with_name("distribute.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LocalCopy:
    def __init__(self, root):
        self.root = root

    def put(self, source, destination, relative, checksum):
        target = self.root / destination["host"] / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if module.digest(target) != checksum:
                raise ValueError("conflicting fixture")
            return "reused"
        shutil.copyfile(source, target)
        assert module.digest(target) == checksum
        return "copied"


def test_download_once_four_destinations_and_resume(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"checkpoint fixture")
    checksum = module.digest(source)
    manifest = {
        "schema": "sparkring-artifacts/v1",
        "artifacts": [
            {"path": "target/weights.bin", "source": str(source), "sha256": checksum}
        ],
        "destinations": [{"host": f"rank{i}", "root": "/srv/models"} for i in range(4)],
    }
    cache = tmp_path / "cache"
    transport = LocalCopy(tmp_path / "ranks")
    assert [r["outcome"] for r in module.distribute(manifest, cache, transport)] == [
        "copied"
    ] * 4
    source.unlink()
    assert [r["outcome"] for r in module.distribute(manifest, cache, transport)] == [
        "reused"
    ] * 4
    assert len(list(cache.iterdir())) == 1


def test_bad_source_never_reaches_destinations(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"bad")
    m = {
        "schema": "sparkring-artifacts/v1",
        "artifacts": [{"path": "weights", "source": str(source), "sha256": "0" * 64}],
        "destinations": [{"host": "rank0", "root": "/srv/models"}],
    }
    with pytest.raises(ValueError, match="checksum"):
        module.distribute(m, tmp_path / "cache", LocalCopy(tmp_path / "ranks"))
    assert not (tmp_path / "ranks").exists()
    assert list((tmp_path / "cache").iterdir()) == []


@pytest.mark.parametrize("name", ["../weights", "/weights", "a/../b", ".", "a\nb"])
def test_path_traversal_rejected(name, tmp_path):
    m = {
        "schema": "sparkring-artifacts/v1",
        "artifacts": [
            {
                "path": name,
                "source": str(tmp_path / "a"),
                "sha256": hashlib.sha256(b"").hexdigest(),
            }
        ],
        "destinations": [{"host": "rank0", "root": "/srv/models"}],
    }
    with pytest.raises(ValueError):
        module.validate(m)


def test_ssh_copy_checks_bytes_before_publication(monkeypatch, tmp_path):
    from types import SimpleNamespace

    calls = []
    source = tmp_path / "source"
    source.write_bytes(b"data")
    sha = module.digest(source)

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "scp":
            return SimpleNamespace(returncode=0)
        command = argv[-1]
        if command.startswith("test "):
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        if command.startswith("mktemp "):
            return SimpleNamespace(
                returncode=0, stdout="/srv/models/.lil-copy-abcdef12\n", stderr=""
            )
        if command.startswith("sha256sum "):
            return SimpleNamespace(returncode=0, stdout=sha + "  file\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert (
        module.SSHCopy().put(
            source, {"host": "rank0", "root": "/srv/models"}, "weight", sha
        )
        == "copied"
    )
    commands = [c[-1] for c in calls if c[0] == "ssh"]
    assert next(i for i, c in enumerate(commands) if c.startswith("sha256sum")) < next(
        i for i, c in enumerate(commands) if c.startswith("ln ")
    )
    assert commands[-1].startswith("rm -- /srv/models/.lil-copy-")


def test_ssh_unreachable_is_not_treated_as_missing(monkeypatch, tmp_path):
    from types import SimpleNamespace

    def failed(*args, **kwargs):
        return SimpleNamespace(returncode=255, stdout="", stderr="unreachable")

    monkeypatch.setattr(module.subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="unreachable"):
        module.SSHCopy().put(
            tmp_path / "file",
            {"host": "rank0", "root": "/srv/models"},
            "weight",
            "a" * 64,
        )
