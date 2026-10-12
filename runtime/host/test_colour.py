"""The command highlight: spans, the colour policy, and the plain-output contract.

Every message that names a command must render byte-identically to its plain
text wherever colour is off, and carry the escape codes on the command spans
alone wherever colour is on.
"""
import io
import json
import re

import pytest

from runtime.host import colour, fabric, fabric_tune, progress, recovery
from runtime.host.controller import transport_status_line
from runtime.host.terminal import Console


class Terminal(io.StringIO):
    """A stream that claims to be a terminal, as a video terminal would."""

    def isatty(self):
        return True


CODE = re.compile(r"\x1b\[1;36m|\x1b\[0m|\x00")


@pytest.fixture(autouse=True)
def reset_mode(monkeypatch):
    """Each test starts with automatic detection and no colour variables."""
    colour.set_mode(None)
    monkeypatch.delenv("SPARKRING_COLOR", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    yield
    colour.set_mode(None)


FABRIC_VALUE = {"layout": "cycle-4", "relays": "table", "document": "fabric-19818-0000",
                "verified": {"at": "2026-10-07T10:05:00Z", "result": "healthy", "report": "r.json"}}
FABRIC_PLAIN = ("Fabric: cycle-4, relay table restored at every boot; last verified healthy at "
                "2026-10-07T10:05:00Z; sudo sparkring fabric verify checks it")
TUNE_VALUE = {"state": "default"}
RECOVER_RECORD = {"enabled": False}


def test_render_without_a_terminal_is_the_plain_message():
    assert colour.render(fabric.status_line(FABRIC_VALUE), io.StringIO()) == FABRIC_PLAIN


def test_render_forced_colour_codes_the_command_spans_alone():
    coloured = colour.render(fabric.status_line(FABRIC_VALUE), Terminal())
    assert coloured == (FABRIC_PLAIN.replace("sudo sparkring fabric verify", "\x1b[1;36m"
                                             "sudo sparkring fabric verify\x1b[0m"))
    # No escape code and no mark survives outside the one command span.
    outside = CODE.sub("", coloured)
    assert outside == FABRIC_PLAIN
    assert coloured.count("\x1b[1;36m") == 1


def test_never_mode_and_a_pipe_drop_the_marks_entirely():
    marked = fabric_tune.status_line(TUNE_VALUE)
    colour.set_mode("never")
    assert colour.render(marked, Terminal()) == colour.plain(marked)
    colour.set_mode("auto")
    assert colour.render(marked, io.StringIO()) == colour.plain(marked)


def test_no_color_and_term_dumb_force_plain_on_a_terminal(monkeypatch):
    marked = fabric_tune.status_line(TUNE_VALUE)
    monkeypatch.setenv("NO_COLOR", "")
    assert colour.render(marked, Terminal()) == colour.plain(marked)
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("TERM", "dumb")
    assert colour.render(marked, Terminal()) == colour.plain(marked)


def test_sparkring_color_selects_and_the_flag_wins(monkeypatch):
    marked = recovery.status_lines(RECOVER_RECORD)[0]
    monkeypatch.setenv("SPARKRING_COLOR", "always")
    assert "\x1b[1;36m" in colour.render(marked, io.StringIO())
    monkeypatch.setenv("SPARKRING_COLOR", "never")
    assert colour.render(marked, Terminal()) == colour.plain(marked)
    colour.set_mode("always")
    assert "\x1b[1;36m" in colour.render(marked, Terminal())
    # An unrecognised variable value cannot force colour anywhere.
    monkeypatch.setenv("SPARKRING_COLOR", "maybe")
    colour.set_mode(None)
    assert colour.render(marked, io.StringIO()) == colour.plain(marked)


def test_action_marks_only_a_sparkring_command_field():
    assert colour.plain(colour.action("sparkring models")) == "sparkring models"
    assert colour.render(colour.action("sudo sparkring status --refresh"), Terminal()) == \
        "\x1b[1;36msudo sparkring status --refresh\x1b[0m"
    assert colour.action("systemctl status sparkring-agent") == "systemctl status sparkring-agent"
    assert colour.action("journalctl -u sparkring-mesh -b") == "journalctl -u sparkring-mesh -b"
    prose = "reconnect the cable of Node A's port0; it carries the admin tunnel"
    assert colour.action(prose) == prose


def test_console_writes_coloured_spans_only_on_a_terminal():
    marked = f"Follow from another terminal: {colour.cmd('sudo sparkring logs --follow')}"
    terminal = Terminal()
    with Console(terminal) as console:
        console.write(marked)
    assert "\x1b[1;36msudo sparkring logs --follow\x1b[0m" in terminal.getvalue()
    plain_stream = io.StringIO()
    with Console(plain_stream) as console:
        console.write(marked)
    assert plain_stream.getvalue() == "Follow from another terminal: sudo sparkring logs --follow"


def test_console_plain_flag_keeps_spans_plain_even_on_a_terminal():
    marked = f"Follow from another terminal: {colour.cmd('sudo sparkring logs --follow')}"
    colour.set_mode("always")
    terminal = Terminal()
    with Console(terminal, plain=True) as console:
        console.write(marked)
    assert terminal.getvalue() == "Follow from another terminal: sudo sparkring logs --follow"


def test_install_log_and_event_stream_receive_plain_text(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path))
    events = io.StringIO()
    with progress.run("setup"):
        progress._events = events
        print(f"Follow from another terminal: {colour.cmd('sudo sparkring logs --follow')}")
        progress.emit("start", f"Review {colour.cmd('sudo sparkring status')} first")
        progress.record(f"Note {colour.cmd('sudo sparkring status')} beside it")
    log = (tmp_path / "install.log").read_text()
    assert "Follow from another terminal: sudo sparkring logs --follow" in log
    assert "\x00" not in log and "\x1b[" not in log
    event = json.loads(events.getvalue())
    assert event["label"] == "Review sudo sparkring status first"


def test_transport_and_recovery_lines_match_their_plain_text():
    no_verdict = {"backend": "sircl", "group": "cycle-4", "nccl": "never"}
    assert colour.plain(transport_status_line(no_verdict)) == (
        "Transport: sircl on cycle-4 (NCCL never); no receipt check recorded yet: sudo sparkring check")
    idle = {"enabled": True, "failures": 0, "stopped": False}
    state = {"generation": 2, "operation": "down", "complete": True}
    assert [colour.plain(line) for line in recovery.status_lines(idle, state=state)] == [
        "Automatic recovery: on; idle until the next sudo sparkring up --execute or sudo sparkring install"]


def test_tune_status_line_keeps_its_plain_text():
    assert colour.plain(fabric_tune.status_line(TUNE_VALUE)) == (
        "SIRCL tuning: the default table; sudo sparkring fabric tune measures this fabric")
    coloured = colour.render(fabric_tune.status_line(TUNE_VALUE), Terminal())
    assert coloured == ("SIRCL tuning: the default table; \x1b[1;36msudo sparkring fabric tune\x1b[0m"
                        " measures this fabric")
