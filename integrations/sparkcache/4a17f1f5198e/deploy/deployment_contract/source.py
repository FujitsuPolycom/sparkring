"""Deterministic file and deployable-source identities."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    """Hash one file without loading its complete contents into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_source_sha256(path: Path) -> str:
    """Hash source bytes with Git's canonical LF line endings."""

    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def source_tree_sha256(root: Path) -> str:
    """Hash deployable SparkCache source while excluding local build caches."""

    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"SparkCache source directory is missing: {root}")
    digest = hashlib.sha256(b"sparkcache-source-tree/v1\x00")
    count = 0
    files = (candidate for candidate in root.rglob("*") if candidate.is_file())
    for path in sorted(
        files,
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root)
        if (
            any(
                part in {"__pycache__", ".pytest_cache", "build"}
                for part in relative.parts
            )
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        if path.is_symlink():
            raise RuntimeError(f"SparkCache source contains a symlink: {relative}")
        encoded = relative.as_posix().encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        digest.update(bytes.fromhex(_normalized_source_sha256(path)))
        count += 1
    if count == 0:
        raise RuntimeError("SparkCache source tree is empty")
    return digest.hexdigest()
