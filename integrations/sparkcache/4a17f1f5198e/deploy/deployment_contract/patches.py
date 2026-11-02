"""Exact-preimage patch application for deployment overlays."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .source import file_sha256


def apply_verified_patch(
    *,
    work: Path,
    patch_source: Path,
    staged_source: Path,
    expected_preimage_sha256: str,
    expected_postimage_sha256: str,
    patch_name: str,
    role: str,
) -> None:
    """Apply one LF-normalized Git patch between exact file identities."""

    if not patch_source.is_file():
        raise RuntimeError(f"{role} patch is missing: {patch_source}")
    before = file_sha256(staged_source)
    if before != expected_preimage_sha256:
        raise RuntimeError(f"{role} patch preimage differs: {before}")
    patch = work / patch_name
    patch.write_bytes(patch_source.read_bytes().replace(b"\r\n", b"\n"))
    command = [
        "git",
        "-c",
        "core.autocrlf=false",
        "-c",
        "core.eol=lf",
        "apply",
    ]
    subprocess.run([*command, "--check", str(patch)], cwd=work, check=True)
    subprocess.run([*command, str(patch)], cwd=work, check=True)
    after = file_sha256(staged_source)
    if after != expected_postimage_sha256:
        raise RuntimeError(f"{role} patch postimage differs: {after}")
