"""The vendored libsircl snapshot and its sync; offline.

A fixture snapshot stands in for libsircl's workspace output: ``tree/``,
``FILES.sha256`` and ``MANIFEST``.
"""
import hashlib
import json
from pathlib import Path

import pytest

from scripts import sync_libsircl as sync


def snapshot(root, files=None):
    """A snapshot directory holding ``files`` (``{path: bytes}``); returns ``(directory, tree digest)``."""
    files = files or {"VERSION": b"0.6.0\n", "README.md": b"# libsircl\n", "src/api.c": b"int x;\r\n",
                      "kernels/prebuilt/sircl_links.fatbin": bytes(range(256)),
                      "tests/__pycache__/test_api.cpython-312.pyc": b"\0cache",
                      "verification/hardware/run.log": b"copied spark-0000\n"}
    directory = root / "snapshot"
    for path, data in files.items():
        target = directory / "tree" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    lines = "".join(f"{hashlib.sha256(data).hexdigest()}  ./{path}\n" for path, data in sorted(files.items()))
    (directory / "FILES.sha256").write_bytes(lines.encode())
    digest = hashlib.sha256(lines.encode()).hexdigest()
    (directory / "MANIFEST").write_text(f"libsircl snapshot {digest[:8]}\n\nTree digest (SHA-256 of FILES.sha256): "
                                        f"{digest}\n")
    return directory, digest


def test_the_vendored_copy_is_the_snapshot_its_manifest_names():
    result = sync.check()
    assert result["tree_digest"] == hashlib.sha256((sync.TARGET / sync.MANIFEST).read_bytes()).hexdigest()
    assert result["snapshot"] == result["tree_digest"][:8] and result["version"] == "0.6.0"
    record = json.loads((sync.TARGET / sync.RECORD).read_text(encoding="utf-8"))
    assert record["files"] == result["files"] and not (sync.TARGET / "verification").exists()
    # The library's build inputs and notices are vendored; run evidence stays with the snapshot.
    for path in ("Makefile", "VERSION", "LICENSE", "NOTICE", "vendor/NCCL-LICENSE.txt", "vendor/SIRCL-NOTICE",
                 "LICENSES/CUDA-NOTICE.txt", "LICENSES/rdma-core-verbs.txt", "tools/site_routes.py",
                 "tests/api_manifest.json", "kernels/prebuilt/sircl_links.fatbin"):
        assert (sync.TARGET / path).is_file(), path
    assert all(path.startswith("verification/") or "__pycache__" in path for path in record["excluded"]["paths"])


def test_a_synced_snapshot_keeps_its_bytes_and_leaves_out_caches_and_run_evidence(tmp_path):
    directory, digest = snapshot(tmp_path)
    target = tmp_path / "libsircl"
    result = sync.sync(directory, digest, target)
    assert result == {"snapshot": digest[:8], "tree_digest": digest, "version": "0.6.0", "files": 4, "excluded": 2}
    assert (target / sync.MANIFEST).read_bytes() == (directory / "FILES.sha256").read_bytes()
    assert (target / "src/api.c").read_bytes() == b"int x;\r\n"
    assert not (target / "verification").exists() and not (target / "tests").exists()
    record = json.loads((target / sync.RECORD).read_text())
    assert record["excluded"]["paths"] == ["tests/__pycache__/test_api.cpython-312.pyc",
                                           "verification/hardware/run.log"]
    # A later sync replaces every file, including one the earlier snapshot had and this one lacks.
    files = {"VERSION": b"0.6.1\n", "README.md": b"# libsircl 0.6.1\n"}
    later, later_digest = snapshot(tmp_path / "later", files)
    assert sync.sync(later, later_digest, target)["files"] == 2 and not (target / "src").exists()


@pytest.mark.parametrize("edit, message", [
    (lambda target: (target / "src/api.c").write_bytes(b"int y;\n"), "differ from the snapshot"),
    (lambda target: (target / "src/extra.c").write_bytes(b""), "files its snapshot does not vendor"),
    (lambda target: (target / "README.md").unlink(), "lacks files of its snapshot"),
    (lambda target: (target / sync.RECORD).write_text("{}"), "does not describe"),
])
def test_a_hand_edit_of_the_vendored_copy_is_refused(tmp_path, edit, message):
    directory, digest = snapshot(tmp_path)
    target = tmp_path / "libsircl"
    sync.sync(directory, digest, target)
    # The library's build directory and Python caches are not part of the copy.
    (target / "build").mkdir()
    (target / "build/libsircl.so.0.6.0").write_bytes(b"built")
    assert sync.check(target)["files"] == 4
    edit(target)
    with pytest.raises(sync.SnapshotError, match=message):
        sync.check(target)


@pytest.mark.parametrize("edit, message", [
    (lambda directory, digest: None, "not the tree digest"),
    (lambda directory, digest: (directory / "MANIFEST").write_text("Tree digest: " + "0" * 64), "does not state"),
    (lambda directory, digest: (directory / "tree/src/api.c").write_bytes(b"changed"), "differs from its manifest"),
    (lambda directory, digest: (directory / "tree/extra").write_bytes(b""), "does not list"),
    (lambda directory, digest: (directory / "tree/README.md").unlink(), "lacks"),
])
def test_a_snapshot_that_does_not_match_its_digest_is_refused_and_nothing_is_written(tmp_path, edit, message):
    directory, digest = snapshot(tmp_path)
    edit(directory, digest)
    given = "f" * 64 if message == "not the tree digest" else digest
    target = tmp_path / "libsircl"
    with pytest.raises(sync.SnapshotError, match=message):
        sync.sync(directory, given, target)
    assert not target.exists()


def test_the_manifest_refuses_unsafe_and_duplicate_paths():
    line = "a" * 64 + "  ./"
    for text, message in ((line + "../x\n", "unsafe"), (line + "x\n" + line + "x\n", "twice"),
                          (line + sync.RECORD + "\n", "reserves"), ("not a line\n", "is not")):
        with pytest.raises(sync.SnapshotError, match=message):
            sync.parse_manifest(text.encode())


def test_the_cli_reports_a_mismatch_without_a_traceback(tmp_path, capsys):
    directory, _ = snapshot(tmp_path)
    assert sync.main(["sync", str(directory), "--tree-digest", "0" * 64, "--target", str(tmp_path / "t")]) == 1
    assert "not the tree digest" in capsys.readouterr().err
    assert sync.main(["check"]) == 0 and json.loads(capsys.readouterr().out)["version"] == "0.6.0"


def test_pytest_does_not_collect_the_vendored_suites():
    conftest = Path(sync.ROOT / "spark_transport" / "conftest.py").read_text(encoding="utf-8")
    assert 'collect_ignore = ["libsircl"]' in conftest
