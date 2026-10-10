"""Dashboard quiet mode around a measurement: hold before the load, release after it, also on an error,
Ctrl-C or SIGTERM, and record the outcome. A fake runner stands in for the shell and SSH; one test runs
the real ``sparkring dashboard quiet`` command in process. No host is contacted."""
import contextlib
import io
import json
import os
import shlex
import signal
import subprocess
from unittest import mock

import pytest

from performance.harnesses import dashboard_quiet
from runtime.host import dashboard


@pytest.fixture(autouse=True)
def holding(monkeypatch):
    """These tests hold through fake runners; the harness-wide skip (conftest.py) is lifted."""
    monkeypatch.delenv(dashboard_quiet.SKIP, raising=False)


class Runner:
    """Answers the hold and release scripts with canned outputs and records each argv."""

    def __init__(self, hold="@service active\n" + json.dumps({"holder": "serving_ab-0-ab12cd34", "was_quiet": False}),
                 code=0, release=None, error=None):
        self.argv, self.hold, self.code, self.error = [], hold, code, error
        self.release = release if release is not None else json.dumps({"released": "serving_ab-0-ab12cd34",
                                                                       "still_quiet": False})

    def __call__(self, argv, **kwargs):
        self.argv.append(argv)
        if self.error:
            raise self.error
        output = self.release if " release " in argv[-1] else self.hold
        return subprocess.CompletedProcess(argv, self.code, output, "")


def test_a_hold_on_node_a_goes_through_the_harness_ssh_and_is_released():
    runner = Runner()
    ssh = ["ssh", "-o", "BatchMode=yes", "root@192.0.2.10"]
    record = {}
    with dashboard_quiet.held("serving_ab", ssh=ssh, where="spark0", runner=runner, record=record) as field:
        assert field["state"] == "on" and field["holder"] == "serving_ab-0-ab12cd34"
        assert record["dashboard_quiet"]["state"] == "on" and len(runner.argv) == 1
    hold, release = runner.argv
    assert hold[:4] == ssh and "sudo -n /usr/bin/sparkring dashboard quiet hold --label serving_ab --ttl 43200" in hold[4]
    assert "--pid" not in hold[4]           # another host's process cannot be checked from Node A
    assert release[4].endswith("sudo -n /usr/bin/sparkring dashboard quiet release serving_ab-0-ab12cd34 2>&1")
    assert record["dashboard_quiet"] == {"state": "on", "where": "spark0", "holder": "serving_ab-0-ab12cd34",
                                         "was_on_before": False, "service": "active", "detail": None,
                                         "released": True}


@pytest.mark.parametrize("failure", [RuntimeError("the load failed"), KeyboardInterrupt()])
def test_an_error_or_ctrl_c_releases_the_hold(failure):
    runner = Runner()
    with pytest.raises(type(failure)):
        with dashboard_quiet.held("accept_profile", runner=runner):
            raise failure
    assert len(runner.argv) == 2 and " release " in runner.argv[1][-1]


def test_sigterm_ends_the_block_through_the_release_and_the_handler_is_restored():
    runner = Runner()
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as stopped:
        with dashboard_quiet.held("measure_api", runner=runner):
            handler = signal.getsignal(signal.SIGTERM)
            handler(signal.SIGTERM, None)
    assert stopped.value.code == 128 + signal.SIGTERM
    assert len(runner.argv) == 2 and signal.getsignal(signal.SIGTERM) is before


def test_a_local_hold_names_the_measurements_process():
    runner = Runner()
    with dashboard_quiet.held("measure_api", runner=runner):
        pass
    assert runner.argv[0][:2] == ["sh", "-c"] and "--pid " in runner.argv[0][2]


@pytest.mark.parametrize("output, code, state", [
    ("SPARKRING-ABSENT", 0, "dashboard absent"),
    ("@service inactive\nusage: sparkring [-h]\nsparkring: error: argument command: invalid choice: 'dashboard'", 2,
     "dashboard absent"),
    ("@service active\nsudo: a password is required", 1, "not held"),
])
def test_the_load_runs_whatever_the_hold_found(output, code, state):
    runner = Runner(hold=output, code=code)
    ran = []
    with dashboard_quiet.held("serving_ab", ssh=["ssh", "root@192.0.2.10"], runner=runner) as field:
        ran.append(True)
    assert ran and field["state"] == state and len(runner.argv) == 1        # nothing to release
    if state == "not held":
        assert "password is required" in field["detail"]


def test_a_missing_ssh_client_is_recorded_not_raised():
    field = dashboard_quiet.hold("serving_ab", ssh=["ssh", "root@192.0.2.10"],
                                 runner=Runner(error=FileNotFoundError(2, "No such file", "ssh")))
    assert field["state"] == "not held" and "FileNotFoundError" in field["detail"]


def test_the_hold_works_against_the_dashboard_command(tmp_path):
    """The scripts' sparkring commands, run by runtime.host.dashboard.main against a temporary quiet file."""
    quiet = tmp_path / "dashboard.quiet"

    def runner(argv, **kwargs):
        script = argv[-1]
        command = shlex.split(script.split("; ")[-1].replace(" 2>&1", ""))
        assert command[:4] == ["sudo", "-n", "/usr/bin/sparkring", "dashboard"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = dashboard.main([*command[3:][1:], "--quiet-file", str(quiet)])
        return subprocess.CompletedProcess(argv, code, "@service active\n" + out.getvalue(), "")
    now = __import__("time").time()
    with dashboard_quiet.held("serving_ab", ssh=["ssh", "root@192.0.2.10"], runner=runner) as first:
        assert dashboard.quiet_state(quiet, now)["quiet"] and first["was_on_before"] is False
        # A second harness, another process: it does not inherit the first one's hold.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(dashboard_quiet.INHERITED)
            with dashboard_quiet.held("accept_profile", ssh=["ssh", "root@192.0.2.10"], runner=runner) as second:
                assert second["was_on_before"] is True
        assert dashboard.quiet_state(quiet, now)["quiet"]          # the outer measurement still holds it
    assert first["released"] and second["released"]
    assert dashboard.quiet_state(quiet, now)["quiet"] is False


def test_children_record_the_parents_hold_and_hold_nothing_themselves():
    runner = Runner()
    with dashboard_quiet.held("serving_ab", ssh=["ssh", "root@192.0.2.10"], runner=runner):
        child = Runner()
        with dashboard_quiet.held("prefill_probe", runner=child) as field:
            assert field["inherited"] and field["holder"] == "serving_ab-0-ab12cd34"
        assert child.argv == []
    assert dashboard_quiet.INHERITED not in os.environ and len(runner.argv) == 2


def test_the_skip_setting_holds_nothing(monkeypatch):
    monkeypatch.setenv(dashboard_quiet.SKIP, "skip")
    runner = Runner()
    with dashboard_quiet.held("serving_ab", runner=runner) as field:
        assert field["state"] == "not held" and "skip" in field["detail"]
    assert runner.argv == []


def test_the_entry_point_option(monkeypatch):
    import argparse
    monkeypatch.setenv(dashboard_quiet.TARGET_VARIABLE, "root@192.0.2.10")
    parser = argparse.ArgumentParser()
    dashboard_quiet.add_argument(parser)
    args = parser.parse_args([])
    assert dashboard_quiet.ssh_for(args.dashboard_ssh)[-1] == "root@192.0.2.10"
    assert dashboard_quiet.ssh_for(None) is None
