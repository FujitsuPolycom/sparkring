"""Checks for install.sh; nothing is cloned, built, installed or contacted.

The behavior tests run the script with stub git, package-build, dpkg, apt-get
and sudo commands and without a controlling terminal.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "install.sh"
BUILT = "0.1.0~dev.2+gitbbbbbbbbbbbb"

STUBS = {
    "uname": 'echo aarch64\n',
    "git": ('[ -n "${STUB_GIT_FAILS:-}" ] && { echo "fatal: repository not found"; exit 128; }\n'
            'case "$1" in\n'
            '  clone) for last; do :; done; mkdir -p "$last" ;;\n'
            '  -C) [ "$3" = rev-parse ] && echo ' + "a" * 40 + ' ;;\n'
            'esac\n'),
    "python3": 'mkdir -p "$3" && : > "$3/sparkring_${STUB_BUILT}_arm64.deb"\n',
    "dpkg-deb": 'echo "$STUB_BUILT"\n',
    "dpkg-query": '[ -n "${STUB_INSTALLED:-}" ] || exit 1\nprintf "ii %s" "$STUB_INSTALLED"\n',
    "dpkg": '[ "$2" != "$4" ] && [ "$(printf "%s\\n%s\\n" "$2" "$4" | sort -V | head -1)" = "$2" ]\n',
    "apt-get": 'echo "Reading package lists..."\necho "apt-get $*" >> "$STUB_LOG"\n',
    "sudo": ('if [ "$1" = /usr/bin/sparkring ]; then\n'
             '  shift; echo "sparkring $*" >> "$STUB_LOG"\n'
             '  echo \'{"schema": "sparkring-install-result/v1", "state": "planned"}\'\n'
             'else exec "$@"; fi\n'),
}


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


def run_script(tmp_path, *arguments, installed=None, git_fails=False):
    """Run install.sh with stub commands; returns the process and the logged apt and sparkring calls."""
    if not sys.platform.startswith("linux"):
        pytest.skip("install.sh runs on Linux")
    if os.geteuid() == 0:
        pytest.skip("As root the script runs /usr/bin/sparkring directly instead of through the sudo stub")
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for name, body in STUBS.items():
        (stubs / name).write_text("#!/bin/sh\n" + body)
        (stubs / name).chmod(0o755)
    log = tmp_path / "calls.log"
    log.touch()
    environment = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "STUB_LOG": str(log),
                   "STUB_BUILT": BUILT, "STUB_INSTALLED": installed or "", "STUB_GIT_FAILS": "1" if git_fails else ""}
    # A new session has no controlling terminal, as for an agent or a CI job.
    result = subprocess.run(["bash", str(SCRIPT), *arguments], stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, env=environment, timeout=60, start_new_session=True)
    return result, log.read_text().splitlines()


def only_document(stdout):
    lines = stdout.splitlines()
    assert len(lines) == 1, stdout
    return json.loads(lines[0])


def test_json_output_is_one_document_and_progress_goes_to_stderr(tmp_path):
    result, calls = run_script(tmp_path, "--profile", "qwen38-flash-next-tp2", "--yes", "--json")
    assert result.returncode == 0, result.stderr
    assert only_document(result.stdout)["schema"] == "sparkring-install-result/v1"
    assert "Reading package lists..." in result.stderr and "Fetching SparkRing" in result.stderr
    assert calls == [f"apt-get install --yes --allow-downgrades {calls[0].split()[-1]}",
                     "sparkring install --profile qwen38-flash-next-tp2 --yes --json"]


def test_package_install_needs_approval_before_apt(tmp_path):
    result, calls = run_script(tmp_path, "--profile", "qwen38-flash-next-tp2", "--json", installed="0.1.0~dev.1+gitaaaa")
    assert result.returncode == 3
    document = only_document(result.stdout)
    assert (document["state"], document["field"]) == ("needs_input", "approval")
    assert "Replace SparkRing 0.1.0~dev.1+gitaaaa on this Spark with " + BUILT in result.stderr
    assert calls == []


def test_downgrade_is_named_in_the_question(tmp_path):
    result, _ = run_script(tmp_path, "--profile", "qwen38-flash-next-tp2", installed="0.1.0~dev.3+gitcccc")
    assert result.returncode == 3 and result.stdout == ""
    assert f"with the earlier version {BUILT}" in result.stderr


def test_plan_installs_nothing_and_plans_with_the_matching_package(tmp_path):
    result, calls = run_script(tmp_path, "--profile", "qwen38-flash-next-tp2", "--plan", "--json", installed=BUILT)
    assert result.returncode == 0, result.stderr
    assert only_document(result.stdout)["state"] == "planned"
    assert calls == ["sparkring install --profile qwen38-flash-next-tp2 --plan --json"]


@pytest.mark.parametrize("installed", [None, "0.1.0~dev.1+gitaaaa"])
def test_plan_stops_when_the_installed_package_differs(tmp_path, installed):
    result, calls = run_script(tmp_path, "--profile", "qwen38-flash-next-tp2", "--plan", "--yes", "--json",
                               installed=installed)
    assert result.returncode == 3
    document = only_document(result.stdout)
    assert (document["state"], document["field"]) == ("needs_input", "package")
    assert document["details"] == {"built": BUILT, "installed": installed}
    assert calls == []


def test_bootstrap_failure_is_a_failed_result(tmp_path):
    result, calls = run_script(tmp_path, "--profile", "qwen38-flash-next-tp2", "--yes", "--json", git_fails=True)
    assert result.returncode == 2
    document = only_document(result.stdout)
    assert (document["state"], document["stage"]) == ("failed", "fetch")
    assert "repository not found" in result.stderr
    assert calls == []


@pytest.mark.parametrize("installed", [None, "0.1.0~dev.1+gitaaaa", BUILT])
def test_package_only_stops_before_sparkring_install(tmp_path, installed):
    result, calls = run_script(tmp_path, "--package-only", "--profile", "qwen38-flash-next-tp2", "--yes", "--json",
                               installed=installed)
    assert result.returncode == 0, result.stderr
    document = only_document(result.stdout)
    assert document["state"] == "package-installed"
    assert (document["version"], document["previous"], document["changed"]) == (BUILT, installed, installed != BUILT)
    assert document["source_revision"] == "a" * 40
    assert [call.split()[0] for call in calls] == ([] if installed == BUILT else ["apt-get"])


def test_package_only_still_needs_approval(tmp_path):
    result, calls = run_script(tmp_path, "--package-only", "--json")
    assert result.returncode == 3
    assert only_document(result.stdout)["field"] == "approval"
    assert calls == []


def test_package_only_and_plan_are_exclusive(tmp_path):
    result, calls = run_script(tmp_path, "--package-only", "--plan", "--json")
    assert result.returncode == 2
    assert only_document(result.stdout)["stage"] == "arguments"
    assert calls == []
