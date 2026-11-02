from __future__ import annotations

from pathlib import Path

import pytest

from deploy.deployment_contract import file_sha256, source_tree_sha256


def test_source_tree_identity_normalizes_line_endings_and_ignores_build_cache(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    document = source / "module.py"
    document.write_bytes(b"one\r\ntwo\r\n")
    windows_digest = source_tree_sha256(source)

    document.write_bytes(b"one\ntwo\n")
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "module.pyc").write_bytes(b"ignored")

    assert source_tree_sha256(source) == windows_digest
    assert len(file_sha256(document)) == 64


def test_source_tree_identity_rejects_absent_and_empty_roots(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="is missing"):
        source_tree_sha256(tmp_path / "absent")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="is empty"):
        source_tree_sha256(empty)
