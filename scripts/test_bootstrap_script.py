"""Static checks for bootstrap.sh; nothing is cloned, installed or contacted."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bootstrap.sh"


def test_bootstrap_keeps_the_managed_checkout_clean() -> None:
    """A tracked file whose mode the script changes makes the checkout dirty.

    The script refuses to update a dirty managed checkout, so any chmod of a
    tracked path breaks every later run. The launcher invokes python3
    explicitly, so no tracked file needs the executable bit.
    """

    text = SCRIPT.read_text(encoding="utf-8")
    chmod_targets = re.findall(r"^\s*chmod\s+\S+\s+(.+)$", text, flags=re.MULTILINE)
    assert chmod_targets == ['"$launcher"']
    assert 'exec python3 %q "$@"' in text
    assert "refusing to update dirty managed checkout" in text
    tracked = subprocess.run(
        ["git", "ls-files", "-s", "scripts/sparkring.py"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    assert tracked.startswith("100644 "), tracked


def test_bootstrap_parses_and_requires_explicit_inputs() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    assert "--ref requires a value" in text and "--install-dir requires a value" in text
    assert "git clone --branch \"$REF\" --single-branch" in text
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable for syntax check")
    result = subprocess.run([bash, "-n", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
