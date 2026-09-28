import json
import sys
import os
import io
import subprocess
import threading

import pytest

from runtime.host import progress
from runtime.host.terminal import Console


class TerminalBuffer(io.StringIO):
    def isatty(self):
        return True


def test_install_log_is_flushed_and_tailable_before_command_finishes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path))
    with progress.run("setup"):
        with progress.step("Node A: verify existing fabric"):
            assert "Node A: verify existing fabric" in (tmp_path / "install.log").read_text()
        print("Existing checkpoint will be reused")
    text = (tmp_path / "install.log").read_text()
    assert "Done: Node A" in text and "checkpoint will be reused" in text
    assert "logs --follow" in capsys.readouterr().out


def test_verbose_child_output_is_separate_from_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path))
    with progress.run("setup"):
        progress.command([sys.executable, "-c", "print('verbose fixture output')"], title="Prepare worker tools", check=True)
    assert "verbose fixture output" not in (tmp_path / "install.log").read_text()
    assert "verbose fixture output" in (tmp_path / "install-details.log").read_text()


def test_failed_result_never_reports_done(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path))
    with progress.run("up"):
        with progress.step("Node B: verify checkpoint") as state:
            state["failed"] = True
            progress.failure("trace details\nCheckpoint checksum differs")
    text = (tmp_path / "install.log").read_text()
    assert "Stopped: Node B" in text and "Done: Node B" not in text
    assert "trace details" not in text
    assert "trace details" in (tmp_path / "install-details.log").read_text()


def test_follow_prints_existing_line_immediately_through_an_ssh_style_pipe(tmp_path):
    (tmp_path / "install.log").write_text("Existing progress must be visible immediately\n")
    command = [sys.executable, "-c", "from runtime.host.progress import main; main(['--follow','--lines','1'])"]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             env={**os.environ, "SPARKRING_LOG_DIR": str(tmp_path)})
    ready = threading.Event()
    lines = []

    def read():
        lines.append(child.stdout.readline())
        ready.set()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        assert ready.wait(5), "Log follower buffered its initial output until a later write"
        assert lines == ["Existing progress must be visible immediately\n"]
    finally:
        child.terminate()
        child.communicate(timeout=5)
        reader.join(timeout=1)


def test_terminal_progress_animates_concurrent_steps_without_decorating_saved_log(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    terminal = TerminalBuffer()
    pipe = io.StringIO()
    monkeypatch.setattr(sys, "stderr", terminal)
    monkeypatch.setattr(sys, "stdout", pipe)
    with progress.run("up"):
        with progress.step("Node 0: Load weights"):
            with progress.step("Node 1: Load weights") as outcome:
                progress._console.draw()
                progress._console.draw()
                outcome["failed"] = True
            progress._console.draw()
    decorated = terminal.getvalue()
    assert "\033[36m" in decorated and "2 active" in decorated
    assert "\033[32m" in decorated and "\033[31m" in decorated
    saved = (tmp_path / "install.log").read_text()
    assert "Done: Node 0" in saved and "Stopped: Node 1" in saved
    assert "Done: Node 1" not in saved
    assert "\033" not in saved and "\r" not in saved and "active" not in saved
    assert "\033" not in pipe.getvalue()
    assert decorated.endswith("\n") or decorated.endswith("\r\033[2K")


@pytest.mark.parametrize("mode", ["pipe", "plain", "dumb", "no-color"])
def test_plain_outputs_never_get_terminal_control_codes(mode, monkeypatch):
    monkeypatch.setenv("TERM", "dumb" if mode == "dumb" else "xterm")
    monkeypatch.delenv("NO_COLOR", raising=False)
    if mode == "no-color":
        monkeypatch.setenv("NO_COLOR", "1")
    output = io.StringIO() if mode == "pipe" else TerminalBuffer()
    with Console(output, plain=mode == "plain") as console:
        console.begin("Busy")
        console.write("Done: Ready\n")
        console.draw()
    assert output.getvalue() == "Done: Ready\n"


def test_spinner_preserves_prompts_and_cleans_up_after_interrupt(monkeypatch):
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.delenv("NO_COLOR", raising=False)
    output = TerminalBuffer()
    with pytest.raises(KeyboardInterrupt):
        with Console(output) as console:
            console.begin("Waiting")
            console.write("Confirm? ")
            console.draw()
            assert output.getvalue() == "Confirm? "
            console.write("yes\n")
            console.draw()
            raise KeyboardInterrupt
    assert output.getvalue().endswith("\r\033[2K")
    assert not console.thread.is_alive()


def test_follower_colors_timestamped_events_and_can_be_plain(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.delenv("NO_COLOR", raising=False)
    line = "2026-09-24T13:00:00-05:00  Done: Node 0: Image verified\n"
    (tmp_path / "install.log").write_text(line)
    output = TerminalBuffer()
    monkeypatch.setattr(sys, "stdout", output)
    assert progress.main([]) == 0
    assert "\033[32m" in output.getvalue() and "Done: Node 0" in output.getvalue()
    output.seek(0)
    output.truncate()
    assert progress.main(["--plain"]) == 0
    assert output.getvalue() == line


def test_transfer_amounts_rates_and_time_left_read_plainly():
    gib = 1024 ** 3
    assert progress.size_text(121 * gib, 166 * gib) == "121 of 166 GiB"
    assert progress.size_text(gib // 2, 3 * gib) == "0.5 of 3.0 GiB"
    assert progress.size_text(5 * 1024 ** 2, 400 * 1024 ** 2) == "5 of 400 MiB"
    assert progress.rate_text(106_250_000) == "850 Mb/s"
    assert progress.rate_text(250_000_000) == "2.0 Gb/s"
    assert progress.time_left(30) == "less than a minute left"
    assert progress.time_left(414) == "about 7 min left"
    assert progress.time_left(2 * 3600 + 5 * 60) == "about 2 h 5 min left"


def test_meter_reports_the_recent_rate_and_the_time_left():
    gib = 1024 ** 3
    now, done = [0.0], [125 * gib - 30 * 106_250_000]
    meter = progress.Meter(166 * gib, lambda: done[0], clock=lambda: now[0])
    now[0], done[0] = 30.0, 125 * gib
    fields = meter()
    assert fields["detail"] == "125 of 166 GiB, 850 Mb/s, about 7 min left"
    assert fields["bytes_done"] == 125 * gib and fields["bytes_total"] == 166 * gib
    assert fields["rate_bps"] == 850_000_000 and fields["eta_s"] == 414 and fields["percent"] == 75.3
    # Without progress since the last report, only the amount is shown.
    now[0] = 60.0
    assert meter()["detail"] == "125 of 166 GiB" and "eta_s" not in meter()


def test_event_stream_records_each_step_with_stable_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(progress, "HEARTBEAT", 0.05)
    reported = threading.Event()

    def report(elapsed):
        reported.set()
        return {"detail": "loading weights (shard 3/9)", "log_quiet_s": 4}
    path = tmp_path / "events.jsonl"
    with progress.run("install", events=path):
        with progress.step("Node 0: Wait for API readiness", phase="ready", report=report):
            assert reported.wait(5)
            threading.Event().wait(0.1)
        with pytest.raises(RuntimeError):
            with progress.step("Node 1: Start model", phase="start"):
                raise RuntimeError("boom")
        progress.failure("trace\nNode 1: container exited")
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(event["schema"] == "sparkring-install-event/v1" and "time" in event for event in events)
    assert [(event["state"], event["node"], event["phase"]) for event in events if event["state"] != "working"] == [
        ("start", None, "install"), ("start", 0, "ready"), ("done", 0, "ready"), ("start", 1, "start"),
        ("failed", 1, "start"), ("error", None, None)]
    working = next(event for event in events if event["state"] == "working")
    assert working["label"] == "Node 0: Wait for API readiness" and working["elapsed_s"] >= 0
    assert working["detail"] == "loading weights (shard 3/9)" and working["log_quiet_s"] == 4
    assert events[-1]["message"] == "Node 1: container exited"
    log = (tmp_path / "logs/install.log").read_text()
    assert "Still working: Node 0: Wait for API readiness (0s) - loading weights (shard 3/9)" in log


def test_a_failing_report_leaves_the_plain_heartbeat(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(progress, "HEARTBEAT", 0.05)
    calls = threading.Event()

    def report(elapsed):
        calls.set()
        raise OSError("ssh: connection refused")
    with progress.run("install"):
        with progress.step("Node 0: Wait for API readiness", report=report):
            assert calls.wait(5)
            threading.Event().wait(0.1)
    log = (tmp_path / "install.log").read_text()
    assert any(line.endswith("Still working: Node 0: Wait for API readiness (0s)") for line in log.splitlines())
    assert "Done: Node 0: Wait for API readiness" in log
