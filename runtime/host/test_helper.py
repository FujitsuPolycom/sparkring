"""The privileged helper: its argv allowlist, its request protocol and its audit line.

Offline: the server side runs against a socket pair or a temporary unix
socket with a stubbed child; no systemd unit, no Docker and no host are
involved.
"""
import io
import json
import socket
import threading

from runtime.host import helper
from scripts import sparkring


class Child:
    """The stubbed child process: what subprocess.run would return."""

    def __init__(self, code=0, out="started", err=""):
        self.returncode, self.stdout, self.stderr = code, out, err
        self.argvs = []

    def __call__(self, argv):
        self.argvs.append(list(argv))
        return self


def _serve_once(connection, child):
    audit = io.StringIO()
    helper.serve(connection, spawn=child, out=audit)
    return audit.getvalue()


def test_the_allowlist_runs_the_documented_shapes_only():
    helper.validate(["recover", "--auto"])
    helper.validate(["up", "glm53-flash", "--execute"])
    helper.validate(["down", "--execute"])
    for refused in (["up"], ["up", "--execute", "glm53-flash"], ["up", "glm53-flash", "--execute", "--fresh-mesh"],
                    ["recover"], ["recover", "--auto", "--json"], ["down", "-1"], ["install", "--profile", "glm53"],
                    ["status"], ["shell"], [], "up", ["up", "../../etc", "--execute"]):
        try:
            helper.validate(refused)
        except ValueError:
            continue
        raise AssertionError(f"the allowlist accepted {refused!r}")


def test_an_unreadable_request_is_refused_and_audited():
    connection_a, connection_b = socket.socketpair()
    child = Child()
    connection_b.sendall(b"this is not json\n")
    audit = _serve_once(connection_a, child)
    answer = json.loads(connection_b.recv(65536).decode())
    assert answer["type"] == "exit" and answer["code"] == 2
    assert child.argvs == []
    record = json.loads(audit.strip())
    assert record["schema"] == helper.PROTOCOL
    assert record["argv"] is None and record["exit"] == 2


def test_a_refused_request_returns_an_error_without_running_anything():
    connection_a, connection_b = socket.socketpair()
    child = Child()
    connection_b.sendall(b'{"argv": ["status"]}\n')
    audit = _serve_once(connection_a, child)
    answer = json.loads(connection_b.recv(65536).decode())
    assert answer["type"] == "exit" and answer["code"] == 2 and "the helper runs only" in answer["error"]
    assert child.argvs == []
    assert json.loads(audit.strip())["exit"] == 2


def test_a_run_request_streams_the_child_output_and_exit():
    connection_a, connection_b = socket.socketpair()
    child = Child(code=0, out="Model serves", err="note on stderr")
    connection_b.sendall(b'{"argv": ["up", "glm53-flash", "--execute"]}\n')
    audit = _serve_once(connection_a, child)
    lines = [json.loads(line) for line in connection_b.recv(65536).decode().splitlines()]
    assert child.argvs == [["up", "glm53-flash", "--execute"]]
    assert lines[0] == {"type": "output", "stream": "out", "text": "Model serves"}
    assert lines[1] == {"type": "output", "stream": "err", "text": "note on stderr"}
    assert lines[-1]["type"] == "exit" and lines[-1]["code"] == 0
    record = json.loads(audit.strip())
    assert record["exit"] == 0 and record["peer_uid"] is not None


def _start_helper_server(path, child):
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(4)
    audits = []

    def loop():
        while True:
            try:
                connection, _ = server.accept()
            except OSError:
                return
            audits.append(_serve_once(connection, child))

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return server, audits


def test_request_runs_one_validated_command_end_to_end(tmp_path, capsys):
    path = tmp_path / "helper.sock"
    child = Child(code=0, out="Up complete")
    server, audits = _start_helper_server(path, child)
    try:
        assert helper.request(["up", "glm53", "--execute"], socket_path=str(path)) == 0
    finally:
        server.close()
    out = capsys.readouterr()
    assert "Up complete" in out.out
    assert json.loads(audits[0].strip())["argv"] == ["up", "glm53", "--execute"]


def test_request_reports_a_refusal_as_an_error(tmp_path, capsys):
    path = tmp_path / "helper.sock"
    child = Child()
    server, _ = _start_helper_server(path, child)
    try:
        assert helper.request(["down", "does-not-exist"], socket_path=str(path)) == 2
    finally:
        server.close()
    assert "the helper runs only" in capsys.readouterr().err


def test_a_non_root_mutating_command_goes_through_the_opted_in_socket(monkeypatch, capsys):
    sent = []
    monkeypatch.setenv("SPARKRING_HELPER_SOCKET", "/run/sparkring/helper.sock")
    monkeypatch.setattr("runtime.host.privilege.running_as_root", lambda: False)
    monkeypatch.setattr(helper, "request", lambda argv, **kwargs: sent.append(argv) or 0)
    assert sparkring.main(["up", "glm53", "--execute"]) == 0
    assert sent == [["up", "glm53", "--execute"]]


def test_a_root_run_ignores_the_opted_in_socket(monkeypatch):
    monkeypatch.setenv("SPARKRING_HELPER_SOCKET", "/run/sparkring/helper.sock")
    monkeypatch.setattr("runtime.host.privilege.running_as_root", lambda: True)
    sent = []
    monkeypatch.setattr(helper, "request", lambda argv, **kwargs: sent.append(argv) or 0)
    monkeypatch.setattr("runtime.host.recovery.main", lambda argv: 0)
    assert sparkring.main(["recover", "--auto"]) == 0
    assert sent == []


def test_request_keeps_every_line_of_a_reply_that_arrives_in_one_read(tmp_path, capsys):
    """The last output line and the exit line can arrive together; the client must not wait for more."""
    path = tmp_path / "helper.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    held = []

    def reply():
        connection, _ = server.accept()
        held.append(connection)  # kept open: only the exit line may end the request
        helper._readline(connection)
        connection.sendall(json.dumps({"type": "output", "stream": "out", "text": "done\n"}).encode() + b"\n"
                           + json.dumps({"type": "exit", "code": 0}).encode() + b"\n")

    thread = threading.Thread(target=reply, daemon=True)
    thread.start()
    try:
        assert helper.request(["status"], socket_path=str(path), timeout=10.0) == 0
    finally:
        for connection in held:
            connection.close()
        server.close()
    assert capsys.readouterr().out == "done\n"
