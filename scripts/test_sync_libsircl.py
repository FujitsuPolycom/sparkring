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
    assert all(path.startswith(("verification/", "requests/")) or "__pycache__" in path
               for path in record["excluded"]["paths"])
    # Passages naming the snapshot workspace's directories are rewritten; the record pins both digests.
    assert record["schema"] == sync.SCHEMA and set(record["rewritten"]) <= {path for path, _, _ in sync.REWRITES}
    assert result["rewritten"] == len(record["rewritten"]) > 0
    vendored = {path.relative_to(sync.TARGET).as_posix(): path.read_bytes() for path in sync.TARGET.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts}
    assert sync.workspace_references(vendored) == []
    source = json.loads((sync.TARGET / "SOURCE_SNAPSHOT.json").read_text(encoding="utf-8"))
    assert source["selected"] in source["sources"]


def test_a_synced_snapshot_keeps_its_bytes_and_leaves_out_caches_and_run_evidence(tmp_path):
    directory, digest = snapshot(tmp_path)
    target = tmp_path / "libsircl"
    result = sync.sync(directory, digest, target)
    assert result == {"snapshot": digest[:8], "tree_digest": digest, "version": "0.6.0", "files": 4, "excluded": 2,
                      "rewritten": 0}
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


# A Windows user path as the snapshot workspace's documents record one, assembled so this file holds none.
USER_PATH = b"C:" + b"\\Users\\someone\\work"


def test_a_snapshot_passage_naming_its_workspace_is_rewritten_and_pinned(tmp_path):
    readme = (b"# libsircl\n\n- The SIRCL reference this library follows is the clean-room implementation tree copied to\n"
              b"  `../sircl-current`; `SOURCE_SNAPSHOT.json` records its files' SHA-256 hashes.\n")
    source = json.dumps({"sources": {"lead": USER_PATH.decode()}, "selected": "lead"}).encode()
    files = {"VERSION": b"0.6.0\n", "README.md": readme, "requests/PO/land_PO.py": b"IMPL = 'impl'\n",
             "SOURCE_SNAPSHOT.json": source}
    directory, digest = snapshot(tmp_path, files)
    target = tmp_path / "libsircl"
    result = sync.sync(directory, digest, target)
    assert result["rewritten"] == 2 and result["files"] == 3 and not (target / "requests").exists()
    assert (target / "README.md").read_bytes() == (
        b"# libsircl\n\n- The SIRCL reference this library follows is a copy of SIRCL's clean-room implementation "
        b"tree;\n  `SOURCE_SNAPSHOT.json` records its files' SHA-256 hashes.\n")
    assert json.loads((target / "SOURCE_SNAPSHOT.json").read_bytes()) == {
        "sources": {"reference": "SIRCL's clean-room implementation tree (its local path is not vendored)"},
        "selected": "reference"}
    record = json.loads((target / sync.RECORD).read_text())
    assert record["rewritten"]["README.md"] == {
        "snapshot_sha256": hashlib.sha256(readme).hexdigest(),
        "sha256": hashlib.sha256((target / "README.md").read_bytes()).hexdigest()}
    assert record["excluded"]["paths"] == ["requests/PO/land_PO.py"]
    # A hand edit of a rewritten file, or a record whose snapshot digest is not the manifest's, is refused.
    (target / "README.md").write_bytes(readme)
    with pytest.raises(sync.SnapshotError, match="differ from the snapshot|name directories"):
        sync.check(target)
    sync.sync(directory, digest, target)
    record["rewritten"]["README.md"]["snapshot_sha256"] = "0" * 64
    (target / sync.RECORD).write_bytes(sync.encoded(record))
    with pytest.raises(sync.SnapshotError, match="does not describe"):
        sync.check(target)


@pytest.mark.parametrize("text", [b"see " + USER_PATH + b"\\notes\n", b"copy of ../sircl-current\n",
                                  b"the lead workspace's tree\n", b"/mnt/c/" + b"Users/someone/work\n"])
def test_a_snapshot_whose_text_still_names_its_workspace_is_refused_before_anything_changes(tmp_path, text):
    directory, digest = snapshot(tmp_path, {"VERSION": b"0.6.0\n", "docs/notes.md": text})
    target = tmp_path / "libsircl"
    with pytest.raises(sync.SnapshotError, match="names directories of its workspace"):
        sync.sync(directory, digest, target)
    assert not target.exists()


def test_the_sync_replaces_only_an_absent_empty_or_vendored_target(tmp_path):
    directory, digest = snapshot(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    (other / "keep.txt").write_bytes(b"not a vendored copy")
    with pytest.raises(sync.SnapshotError, match="neither empty nor a vendored libsircl copy"):
        sync.sync(directory, digest, other)
    assert [path.name for path in other.iterdir()] == ["keep.txt"]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert sync.sync(directory, digest, empty)["files"] == 4
    # A copy that the schema-v1 sync wrote is a vendored copy too.
    value = json.loads((empty / sync.RECORD).read_text())
    (empty / sync.RECORD).write_bytes(sync.encoded(dict(value, schema="sparkring-libsircl-snapshot/v1")))
    assert sync.sync(directory, digest, empty)["files"] == 4


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
