"""Static checks for bootstrap.sh; nothing is cloned, installed or contacted."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import os

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
    result = subprocess.run(
        [bash, "-n"], input=text.encode("utf-8"), capture_output=True, timeout=10
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(os.name != "posix", reason="Bash checkout update test requires POSIX paths")
def test_managed_single_branch_checkout_can_select_another_branch_and_tag(tmp_path):
    """Exercise the actual checkout block against a local source repository."""
    source = tmp_path / "source"
    checkout = tmp_path / "managed"

    def git(*args, cwd=source):
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                              text=True, check=True).stdout.strip()

    source.mkdir()
    git("init", "-b", "main")
    git("config", "user.name", "Bootstrap test")
    git("config", "user.email", "bootstrap@example.invalid")
    (source / "tracked").write_text("main\n")
    git("add", "tracked")
    git("commit", "-m", "Main fixture")
    main = git("rev-parse", "HEAD")
    git("tag", "release-test")
    git("checkout", "-b", "topic/test")
    (source / "tracked").write_text("topic\n")
    git("commit", "-am", "Topic fixture")
    topic = git("rev-parse", "HEAD")
    git("clone", "--branch", "main", "--single-branch", str(source), str(checkout))
    text = SCRIPT.read_text()
    block = text.split('if [[ -e "$INSTALL_DIR" ]]; then', 1)[1].split('mkdir -p "$BIN_DIR"', 1)[0]
    block = 'set -euo pipefail\nif [[ -e "$INSTALL_DIR" ]]; then' + block

    def update(ref):
        return subprocess.run(["bash"], input=block, text=True, capture_output=True,
                              env={**os.environ, "INSTALL_DIR": str(checkout), "REF": ref,
                                   "REPOSITORY": str(source)}, timeout=20)

    for ref, expected in (("topic/test", topic), ("release-test", main), ("main", main)):
        result = update(ref)
        assert result.returncode == 0, result.stderr
        assert git("rev-parse", "HEAD", cwd=checkout) == expected
        assert git("status", "--porcelain", cwd=checkout) == ""
    missing = update("missing-ref")
    assert missing.returncode != 0
    assert git("rev-parse", "HEAD", cwd=checkout) == main
    (checkout / "tracked").write_text("local edit\n")
    refused = update("topic/test")
    assert refused.returncode != 0
    assert "dirty managed checkout" in refused.stderr
    assert (checkout / "tracked").read_text() == "local edit\n"
