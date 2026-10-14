"""Checks for install.sh; nothing is cloned, built, installed or contacted.

The behavior tests run the script with stub git, package-build, dpkg, apt-get
and sudo commands and without a controlling terminal. The stub package's
payload holds a stand-in `scripts/sparkring.py`, which the system Python runs
when the script plans with the built package.
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
    "git": ('if [ -n "${STUB_REAL_GIT:-}" ]; then exec "$STUB_REAL_GIT" "$@"; fi\n'
            '[ -n "${STUB_GIT_FAILS:-}" ] && { echo "fatal: repository not found"; exit 128; }\n'
            'case "$1" in\n'
            '  clone) for last; do :; done; mkdir -p "$last" ;;\n'
            '  -C) if [ "$3" = merge-base ]; then exit "${STUB_ANCESTOR:-1}"; fi; [ "$3" = rev-parse ] && echo ' + "a" * 40 + ' ;;\n'
            'esac\n'),
    # Builds into WORK/dist and records WORK and the build's arguments.
    "python3": ('if [ "$1" = -I ]; then printf "%s\\n" "${STUB_REVISION:-}"; exit; fi\n'
                'mkdir -p "$3" && : > "$3/sparkring_${STUB_BUILT}_arm64.deb"\n'
                'printf "%s\\n" "${3%/dist}" > "$STUB_WORK"\n'
                'printf "%s\\n" "$*" > "$STUB_WORK.build"\n'),
    "dpkg-deb": ('if [ "$1" = -x ]; then mkdir -p "$3" && cp -R "$STUB_PAYLOAD/." "$3/"; exit; fi\n'
                 'echo "$STUB_BUILT"\n'),
    "dpkg-query": '[ -n "${STUB_INSTALLED:-}" ] || exit 1\nprintf "ii %s" "$STUB_INSTALLED"\n',
    "dpkg": '[ "$2" != "$4" ] && [ "$(printf "%s\\n%s\\n" "$2" "$4" | sort -V | head -1)" = "$2" ]\n',
    "apt-get": 'echo "Reading package lists..."\necho "apt-get $*" >> "$STUB_LOG"\n',
    "sudo": ('if [ "$1" = /usr/bin/sparkring ]; then\n'
             '  shift; echo "sparkring $*" >> "$STUB_LOG"\n'
             '  echo \'{"schema": "sparkring-install-result/v1", "state": "planned"}\'\n'
             'else exec "$@"; fi\n'),
}

# The payload's scripts/sparkring.py. It logs its arguments, records where it
# was imported from and the import path, then acts as STUB_PLAN says: print a
# planned or needs_input result, or fail without one.
PLANNER = '''\
import json
import os
from pathlib import Path
import sys


def main():
    with open(os.environ["STUB_LOG"], "a", encoding="utf-8") as log:
        log.write("extracted sparkring " + " ".join(sys.argv[1:]) + "\\n")
    Path(os.environ["STUB_PLANNER"]).write_text(json.dumps(
        {"root": str(Path(__file__).resolve().parents[1]), "path": sys.path, "argv": sys.argv}), encoding="utf-8")
    state = os.environ["STUB_PLAN"]
    if state == "crash":
        raise RuntimeError("the payload failed")
    print(json.dumps({"schema": "sparkring-install-result/v1", "state": state}))
    return 3 if state == "needs_input" else 0
'''


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
    # A plan with the built package reads the terminal in the same way.
    assert '"${planner[@]}" </dev/tty >"$result"' in text


def test_unrecognized_options_pass_through_to_sparkring_install():
    text = SCRIPT.read_text(encoding="utf-8")
    block = text.split("INSTALL_ARGS=()", 1)[1].split('if [[ $(uname -m)', 1)[0]
    block = block.split("while ((", 1)[1]
    program = ("set -euo pipefail\nINSTALL_ARGS=()\nusage() { :; }\nwhile ((" + block
               + 'printf "%s\\n" "$REF" "$REPOSITORY" "${INSTALL_ARGS[@]}"\n')
    result = subprocess.run(
        [_bash(), "-s", "--", "--profile", "qwen38-flash-next-tp2", "--ref", "topic", "--yes",
         "--repository", "/srv/sparkring.bundle", "--json", "--download-limit", "850Mbit", "--events",
         "/var/tmp/install-events.jsonl"],
        input=program, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "topic", "/srv/sparkring.bundle", "--profile", "qwen38-flash-next-tp2", "--yes", "--json",
        "--download-limit", "850Mbit", "--events", "/var/tmp/install-events.jsonl"]


def run_script(tmp_path, *arguments, installed=None, git_fails=False, plan="planned", revision=None, ancestor=True, real_git=False, approval=None):
    """Run install.sh with stub commands; returns the process and the logged apt and sparkring calls.

    ``plan`` is the stand-in payload's behavior (see ``PLANNER``).
    """
    if not sys.platform.startswith("linux"):
        pytest.skip("install.sh runs on Linux")
    if os.geteuid() == 0:
        pytest.skip("As root the script runs /usr/bin/sparkring directly instead of through the sudo stub")
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for name, body in STUBS.items():
        (stubs / name).write_text("#!/bin/sh\n" + body)
        (stubs / name).chmod(0o755)
    payload = tmp_path / "payload"
    (payload / "usr/lib/sparkring/scripts").mkdir(parents=True)
    (payload / "usr/lib/sparkring/scripts/sparkring.py").write_text(PLANNER, encoding="utf-8")
    log = tmp_path / "calls.log"
    log.touch()
    environment = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "STUB_LOG": str(log),
                   "STUB_BUILT": BUILT, "STUB_INSTALLED": installed or "", "STUB_GIT_FAILS": "1" if git_fails else "",
                   "STUB_WORK": str(tmp_path / "work"), "STUB_PAYLOAD": str(payload),
                   "STUB_PLANNER": str(tmp_path / "planner.json"), "STUB_PLAN": plan,
                   "STUB_REAL_GIT": shutil.which("git") if real_git else "",
                   "STUB_REVISION": (revision if revision is not None else "b" * 40), "STUB_ANCESTOR": "0" if ancestor else "1"}
    # A new session has no controlling terminal, as for an agent or a CI job.
    if approval is None:
        result = subprocess.run(["bash", str(SCRIPT), *arguments], stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, env=environment, timeout=60, start_new_session=True)
    else:
        import fcntl
        import pty
        import termios

        master, slave = pty.openpty()

        def terminal():
            os.setsid()
            fcntl.ioctl(slave, termios.TIOCSCTTY, 0)

        try:
            with subprocess.Popen(["bash", str(SCRIPT), *arguments], stdin=slave, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, env=environment, preexec_fn=terminal) as child:
                os.write(master, (approval + "\n").encode())
                stdout, stderr = child.communicate(timeout=60)
                result = subprocess.CompletedProcess(child.args, child.returncode, stdout, stderr)
        finally:
            os.close(master)
            os.close(slave)
    return result, log.read_text().splitlines()


def only_document(stdout):
    lines = stdout.splitlines()
    assert len(lines) == 1, stdout
    return json.loads(lines[0])


def work_directory(tmp_path):
    """The temporary directory the script built in."""
    work = Path((tmp_path / "work").read_text().strip())
    assert work.name.startswith("sparkring-install.")
    return work


def requires_system_python():
    if not Path("/usr/bin/python3").is_file():
        pytest.skip("Planning with the built package runs /usr/bin/python3")


def test_json_output_is_one_document_and_progress_goes_to_stderr(tmp_path):
    result, calls = run_script(tmp_path, "--profile", "qwen38-flash-next-tp2", "--yes", "--json")
    assert result.returncode == 0, result.stderr
    assert only_document(result.stdout)["schema"] == "sparkring-install-result/v1"
    assert "Reading package lists..." in result.stderr and "Fetching SparkRing" in result.stderr
    assert calls == [f"apt-get install --yes --allow-downgrades {calls[0].split()[-1]}",
                     "sparkring install --profile qwen38-flash-next-tp2 --yes --json"]


def test_the_package_build_compiles_nothing_and_takes_a_named_relay_marker(tmp_path):
    """The build ships the published relay marker; --relay-marker-binary names a copy and stays out of install."""
    assert "--relay-marker require" not in SCRIPT.read_text(encoding="utf-8")
    result, calls = run_script(tmp_path, "--relay-marker-binary", "/srv/sparkring-relay-marker", "--profile",
                               "qwen38-flash-next-tp2", "--yes", "--json")
    assert result.returncode == 0, result.stderr
    build = (tmp_path / "work.build").read_text().split()
    assert build[1:] == ["--output", build[2], "--relay-marker-binary", "/srv/sparkring-relay-marker"]
    assert calls[-1] == "sparkring install --profile qwen38-flash-next-tp2 --yes --json"
    (tmp_path / "default").mkdir()
    run_script(tmp_path / "default", "--profile", "qwen38-flash-next-tp2", "--yes")
    assert "--relay-marker" not in (tmp_path / "default/work.build").read_text()


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
    assert not (tmp_path / "planner.json").exists()
    assert not work_directory(tmp_path).exists()


@pytest.mark.parametrize("installed", ["0.1.0~dev.1+gitaaaa", "0.1.0~dev.3+gitcccc"])
def test_plan_of_another_version_runs_the_built_package_without_installing_it(tmp_path, installed):
    """An upgrade or a downgrade is planned by the extracted payload; apt and /usr/bin/sparkring do not run."""
    requires_system_python()
    result, calls = run_script(tmp_path, "--profile", "qwen38-flash-next-tp2", "--plan", "--yes", "--json", "--display-name", "Draupnir Lab",
                               installed=installed)
    assert result.returncode == 0, result.stderr
    assert only_document(result.stdout)["state"] == "planned"
    assert calls == ["extracted sparkring install --profile qwen38-flash-next-tp2 --plan --yes --json --display-name Draupnir Lab"]
    assert f"This Spark has SparkRing {installed}; planning with the built package {BUILT}" in result.stderr
    work = work_directory(tmp_path)
    planner = json.loads((tmp_path / "planner.json").read_text())
    assert planner["argv"][-2:] == ["--display-name", "Draupnir Lab"]
    root = str(work / "package/usr/lib/sparkring")
    # The payload imports itself: it heads the import path, and neither the
    # installed payload nor the working directory is on it.
    assert planner["root"] == root and planner["path"][0] == root
    assert "/usr/lib/sparkring" not in planner["path"] and "" not in planner["path"]
    assert not work.exists()


@pytest.mark.parametrize(("plan", "code", "state"), [("needs_input", 3, "needs_input"), ("crash", 2, "failed")])
def test_plan_of_another_version_keeps_one_result_and_its_exit_status(tmp_path, plan, code, state):
    requires_system_python()
    result, calls = run_script(tmp_path, "--profile", "qwen38-flash-next-tp2", "--plan", "--json",
                               installed="0.1.0~dev.1+gitaaaa", plan=plan)
    assert result.returncode == code, result.stderr
    document = only_document(result.stdout)
    assert document["state"] == state
    if plan == "crash":
        assert document["stage"] == "plan" and "the payload failed" in result.stderr
    assert calls == ["extracted sparkring install --profile qwen38-flash-next-tp2 --plan --json"]
    assert not work_directory(tmp_path).exists()


def test_plan_without_a_package_stops(tmp_path):
    """The plan reads this Spark's SparkRing node, which installing a package sets up."""
    result, calls = run_script(tmp_path, "--profile", "qwen38-flash-next-tp2", "--plan", "--yes", "--json")
    assert result.returncode == 3
    document = only_document(result.stdout)
    assert (document["state"], document["field"]) == ("needs_input", "package")
    assert document["details"] == {"built": BUILT, "installed": None}
    assert calls == []
    assert not work_directory(tmp_path).exists()


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


@pytest.mark.parametrize("revision", ["c" * 40, ""])
def test_yes_does_not_replace_divergent_or_unknown_source(tmp_path, revision):
    result, calls = run_script(tmp_path, "--yes", "--json", installed="0.1.0~dev.1+gitcccc",
                               revision=revision, ancestor=False)
    assert result.returncode == 3, result.stderr
    document = only_document(result.stdout)
    assert (document["state"], document["field"]) == ("needs_input", "approval")
    assert document["details"]["requested_revision"] == "a" * 40
    assert document["details"]["installed_revision"] == (revision or None)
    assert "--keep-package" in document["message"]
    assert "a" * 40 in result.stderr
    assert (revision or "unknown") in result.stderr
    assert calls == []


def test_yes_can_upgrade_an_ancestor_revision(tmp_path):
    result, calls = run_script(tmp_path, "--yes", "--json", installed="0.1.0~dev.1+gitbbbb",
                               revision="b" * 40, ancestor=True)
    assert result.returncode == 0, result.stderr
    assert calls[0].startswith("apt-get install ")
    assert calls[-1] == "sparkring install --yes --json"


@pytest.mark.parametrize("plan", [False, True])
def test_keep_package_runs_installed_command_without_fetch_or_build(tmp_path, plan):
    args = ["--keep-package", "--profile", "qwen38-flash-next-tp2", "--yes", "--json"]
    if plan:
        args.append("--plan")
    result, calls = run_script(tmp_path, *args, installed="0.1.0~dev.3+gitcccc", git_fails=True)
    assert result.returncode == 0, result.stderr
    assert calls == ["sparkring install --profile qwen38-flash-next-tp2 --yes --json" + (" --plan" if plan else "")]
    assert not (tmp_path / "work").exists()
    assert "Fetching SparkRing" not in result.stderr


def test_keep_package_requires_an_installed_package(tmp_path):
    result, calls = run_script(tmp_path, "--keep-package", "--yes", "--json")
    assert result.returncode == 3
    assert only_document(result.stdout)["field"] == "package"
    assert calls == []
    assert not (tmp_path / "work").exists()


def test_keep_package_and_package_only_are_exclusive(tmp_path):
    result, calls = run_script(tmp_path, "--keep-package", "--package-only", "--yes", "--json", installed=BUILT)
    assert result.returncode == 2
    assert only_document(result.stdout)["stage"] == "arguments"
    assert calls == []


@pytest.mark.parametrize("source", ["ancestor", "matching", "divergent", "missing"])
def test_revision_guard_with_real_git_history(tmp_path, source):
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()

    git("init", "--quiet", "--initial-branch=main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    (repository / "file").write_text("base")
    git("add", "file")
    git("commit", "--quiet", "-m", "base")
    base = git("rev-parse", "HEAD")
    git("switch", "--quiet", "-c", "development")
    (repository / "file").write_text("development")
    git("commit", "--quiet", "-am", "development")
    divergent = git("rev-parse", "HEAD")
    git("switch", "--quiet", "main")
    (repository / "file").write_text("published")
    git("commit", "--quiet", "-am", "published")
    requested = git("rev-parse", "HEAD")
    revision = {"ancestor": base, "matching": requested, "divergent": divergent, "missing": "d" * 40}[source]
    run = tmp_path / "run"
    run.mkdir()
    result, calls = run_script(run, "--repository", str(repository), "--yes", "--json",
                               installed="0.1.0~dev.1+gitbbbb", revision=revision, real_git=True)
    compatible = source in ("ancestor", "matching")
    assert result.returncode == (0 if compatible else 3), result.stderr
    assert any(call.startswith("apt-get ") for call in calls) == compatible
    if not compatible:
        document = only_document(result.stdout)
        assert document["details"] == {"installed_revision": revision, "requested_revision": requested}
        assert calls == []


@pytest.mark.parametrize("answer", ["yes", "no", ""])
def test_divergent_package_requires_explicit_terminal_approval(tmp_path, answer):
    result, calls = run_script(tmp_path, "--yes", "--json", installed="0.1.0~dev.1+gitcccc",
                               revision="c" * 40, ancestor=False, approval=answer)
    accepted = answer == "yes"
    assert result.returncode == (0 if accepted else 3), result.stderr
    assert "Replace the installed source? [y/N]" in result.stderr
    assert any(call.startswith("apt-get ") for call in calls) == accepted
    if not accepted:
        assert only_document(result.stdout)["state"] == "needs_input"
        assert calls == []
