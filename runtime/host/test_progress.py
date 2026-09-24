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
