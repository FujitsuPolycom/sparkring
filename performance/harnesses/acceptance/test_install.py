"""Installer command construction, result parsing and polling; no host is contacted."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest

from performance.harnesses.acceptance import install
from performance.harnesses.acceptance.runners import Result

REVISION = "0123456789abcdef0123456789abcdef01234567"
STDERR = f"""Fetching SparkRing main from https://github.com/FujitsuPolycom/sparkring.git
Source revision: {REVISION}
Building the SparkRing package
Reading package lists...
SparkRing install. Progress: /var/log/sparkring/install.log
Node 0: Wait for API readiness
Done: Node 1: Wait for API readiness (0.5s)
Done: Node 0: Wait for API readiness (545.3s)
Model ready: http://NODE_A:8000/v1
"""


def document(**fields):
    return json.dumps({"schema": install.RESULT_SCHEMA, **fields}, indent=2) + "\n"


def test_published_command_matches_the_documented_one_line_installer():
    command = install.install_command(install.parse_source("published"), "mimo-v26-flash-rl-tp2")
    assert command == ("curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/main/install.sh"
                       " | bash -s -- --profile mimo-v26-flash-rl-tp2 --yes --json")


def test_published_ref_fetches_the_script_and_source_at_that_ref():
    command = install.install_command(install.parse_source(f"published:{REVISION}"), "p-tp2")
    assert f"/sparkring/{REVISION}/install.sh" in command
    assert f"--ref {REVISION} --profile p-tp2" in command


def test_bundle_command_clones_then_installs_from_the_bundle():
    source = install.parse_source("bundle:~/work/sparkring.bundle:feature/branch")
    assert source == install.Source("bundle", "feature/branch", "~/work/sparkring.bundle")
    assert install.install_command(source, "p-tp4") == (
        'git clone -q --branch feature/branch "$HOME"/work/sparkring.bundle src && bash src/install.sh '
        '--repository "$HOME"/work/sparkring.bundle --ref feature/branch --profile p-tp4 --yes --json')


@pytest.mark.parametrize("value", ["", "bundle:relative.bundle:main", "bundle:/x.bundle:", "bundle:/x.bundle:-bad",
                                   "published:a b", "released", "published:$(reboot)"])
def test_unusable_sources_are_refused(value):
    with pytest.raises(ValueError):
        install.parse_source(value)


def test_complete_result_yields_revision_readiness_and_only_document():
    summary = install.summarize(document(state="complete", profile="p-tp2", image_id="img", nodes=2), STDERR, 0,
                                profile="p-tp2")
    assert summary["ok"] and summary["only_document"]
    assert summary["source_revision"] == REVISION
    assert summary["api_ready_seconds"] == 545.3
    assert summary["nodes"] == 2 and summary["problems"] == []
    assert "api_url" not in summary


def test_failed_result_reports_stage_and_message():
    stdout = document(state="failed", stage="fetch", message="Could not fetch SparkRing main.")
    summary = install.summarize(stdout, "Could not fetch SparkRing main.\n", 2, profile="p-tp2")
    assert not summary["ok"]
    assert summary["stage"] == "fetch"
    assert summary["problems"] == ["installer state failed: Could not fetch SparkRing main.", "exit status 2"]
    assert summary["source_revision"] is None and summary["api_ready_seconds"] is None


def test_needs_input_result_reports_the_missing_field():
    stdout = document(state="needs_input", field="checkpoint", message="Free 40 GiB on Node 1", details={})
    summary = install.summarize(stdout, STDERR, 3, profile="p-tp2")
    assert not summary["ok"] and summary["field"] == "checkpoint"
    assert summary["problems"][0].startswith("installer state needs_input")


def test_complete_state_with_another_profile_is_not_accepted():
    summary = install.summarize(document(state="complete", profile="other-tp2"), STDERR, 0, profile="p-tp2")
    assert summary["problems"] == ["installed profile other-tp2, requested p-tp2"]


def test_document_after_stray_output_is_found_but_marked():
    summary = install.summarize("warning: noise\n" + document(state="complete"), STDERR, 0, profile="p-tp2")
    assert summary["ok"] and not summary["only_document"]


def test_missing_document_is_an_error():
    with pytest.raises(install.InstallError):
        install.summarize("", STDERR, 2, profile="p-tp2")
    with pytest.raises(install.InstallError):
        install.summarize('{"schema": "other/v1"}', STDERR, 0, profile="p-tp2")


def test_phases_start_at_the_installer_progress():
    text = install.phases(STDERR)
    assert text.startswith("SparkRing install.") and "Source revision" not in text
    assert install.phases("no progress lines") == "no progress lines\n"


class FakeSsh:
    def __init__(self, replies):
        self.replies, self.scripts = list(replies), []

    def run(self, script, *, timeout):
        self.scripts.append(script)
        return self.replies.pop(0)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_wait_survives_ssh_drops_and_returns_the_exit_status():
    ssh = FakeSsh([Result(0, "running\nNode 0: Prepare pinned image\n", ""), Result(255, "", "Connection reset"),
                   Result(0, "running\nNode 0: Prepare pinned image\n", ""), Result(0, "exit 0\nModel ready\n", "")])
    clock, lines = Clock(), []
    assert install.wait(ssh, "run-1", timeout=600, interval=30, clock=clock, sleep=clock.sleep, log=lines.append) == 0
    assert clock.now == 90
    assert lines == ["install: Node 0: Prepare pinned image", "install: Node A unreachable (ssh exit 255); still polling"]


def test_wait_refuses_a_run_that_never_launched_and_times_out():
    with pytest.raises(install.InstallError, match="--redo install"):
        install.wait(FakeSsh([Result(0, "missing\n", "")]), "run-1", timeout=60, interval=30, clock=Clock(),
                     sleep=lambda s: None, log=lambda line: None)
    clock = Clock()
    with pytest.raises(install.InstallError, match="may still be running"):
        install.wait(FakeSsh([Result(0, "running\n", "")] * 5), "run-1", timeout=60, interval=30, clock=clock,
                     sleep=clock.sleep, log=lambda line: None)


def test_launch_requires_confirmation():
    with pytest.raises(install.InstallError, match="exit 17"):
        install.launch(FakeSsh([Result(17, "", "exists")]), "run-1", "true")


def test_scripts_never_remove_files_and_reject_unsafe_run_ids():
    for script in (install.launch_script("run-1", "true"), install.poll_script("run-1"),
                   install.read_script("run-1", "stderr.log")):
        assert "rm " not in script and "rm -" not in script
    with pytest.raises(ValueError):
        install.poll_script("../escape")


class LocalBash:
    """Runs the harness's remote scripts in a local Bash with HOME redirected."""

    def __init__(self, home):
        self.env = {**os.environ, "HOME": str(home)}

    def run(self, script, *, timeout):
        done = subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True, env=self.env,
                              timeout=timeout)
        return Result(done.returncode, done.stdout, done.stderr)


@pytest.mark.skipif(os.name == "nt" or not all(shutil.which(c) for c in ("bash", "setsid", "nohup")),
                    reason="needs POSIX Bash, setsid and nohup")
def test_detached_run_round_trip_in_local_bash(tmp_path):
    ssh = LocalBash(tmp_path)
    command = (f"printf '%s\\n' '{json.dumps({'schema': install.RESULT_SCHEMA, 'state': 'complete'})}'; "
               f"echo 'Source revision: {REVISION}' >&2; echo 'Done: Node 0: Wait for API readiness (12.5s)' >&2")
    install.launch(ssh, "run-1", command)
    with pytest.raises(install.InstallError, match="exit 17"):
        install.launch(ssh, "run-1", command)
    code = install.wait(ssh, "run-1", timeout=20, interval=0.2, clock=time.monotonic, sleep=time.sleep,
                        log=lambda line: None)
    stdout, stderr = install.fetch(ssh, "run-1", "stdout.json"), install.fetch(ssh, "run-1", "stderr.log")
    summary = install.summarize(stdout, stderr, code, profile="p-tp2")
    assert code == 0 and summary["ok"] and summary["api_ready_seconds"] == 12.5
    assert sorted(p.name for p in Path(tmp_path, install.REMOTE_ROOT, "run-1").iterdir()) == \
        ["command.sh", "exit_code", "stderr.log", "stdout.json"]


def test_install_command_appends_further_install_arguments():
    command = install.install_command(install.parse_source("bundle:/var/tmp/s.bundle:sync/next"), "p-tp2",
                                      ["--image-lock", "/var/tmp/lock one.json"])
    assert command.endswith("--profile p-tp2 --yes --json --image-lock '/var/tmp/lock one.json'")
