from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from deploy.deployment_contract import apply_verified_patch


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_verified_patch_normalizes_eol_and_checks_both_identities(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    staged = work / "module.py"
    staged.parent.mkdir()
    staged.write_bytes(b"before\n")
    subprocess.run(["git", "init", "-q"], cwd=work, check=True)
    patch = tmp_path / "change.patch"
    patch.write_bytes(
        b"diff --git a/module.py b/module.py\r\n"
        b"--- a/module.py\r\n"
        b"+++ b/module.py\r\n"
        b"@@ -1 +1 @@\r\n"
        b"-before\r\n"
        b"+after\r\n"
    )

    apply_verified_patch(
        work=work,
        patch_source=patch,
        staged_source=staged,
        expected_preimage_sha256=_digest(b"before\n"),
        expected_postimage_sha256=_digest(b"after\n"),
        patch_name=".change.patch",
        role="test overlay",
    )

    assert staged.read_bytes() == b"after\n"
