"""Static checks for install.sh; nothing is cloned, built, installed or contacted."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "install.sh"


def _bash():
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    if not bash or not Path(bash).is_file():
        pytest.skip("Bash required")
    return bash


def test_install_script_parses():
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    result = subprocess.run([_bash(), "-n"], input=text.encode("utf-8"), capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_published_command_names_the_default_ref():
    """The usage example fetches the script from the branch it clones by default."""
    text = SCRIPT.read_text(encoding="utf-8")
    ref = re.search(r'^REF="([^"]+)"$', text, flags=re.MULTILINE).group(1)
    assert f"FujitsuPolycom/sparkring/{ref}/install.sh" in text


def test_package_source_is_a_complete_clone():
    """build_deb.py embeds the commit history as a bundle; a shallow clone cannot supply it."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--depth" not in text and "--shallow" not in text
    assert 'scripts/build_deb.py" --output "$WORK/dist"' in text


def test_installer_runs_from_the_package_and_reads_the_terminal():
    text = SCRIPT.read_text(encoding="utf-8")
    runs = [line.strip() for line in text.splitlines() if "sparkring install" in line and "SUDO" in line]
    assert runs and all('/usr/bin/sparkring install "${INSTALL_ARGS[@]}"' in line for line in runs)
    assert any(line.endswith("</dev/tty") for line in runs)


def test_unrecognized_options_pass_through_to_sparkring_install():
    text = SCRIPT.read_text(encoding="utf-8")
    block = text.split("INSTALL_ARGS=()", 1)[1].split('if [[ $(uname -m)', 1)[0]
    block = block.split("while ((", 1)[1]
    program = ("set -euo pipefail\nINSTALL_ARGS=()\nusage() { :; }\nwhile ((" + block
               + 'printf "%s\\n" "$REF" "$REPOSITORY" "${INSTALL_ARGS[@]}"\n')
    result = subprocess.run(
        [_bash(), "-s", "--", "--profile", "qwen38-flash-next-tp2", "--ref", "topic", "--yes",
         "--repository", "/srv/sparkring.bundle", "--json"],
        input=program, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "topic", "/srv/sparkring.bundle", "--profile", "qwen38-flash-next-tp2", "--yes", "--json"]
