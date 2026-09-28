"""The SSH, HTTP and process runners, exercised with local programs and a loopback server only."""
from __future__ import annotations

import http.server
import json
import sys
import threading

import pytest

from performance.harnesses.acceptance import runners

ECHO = "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read()); sys.stderr.write(' '.join(sys.argv[1:]))"


def test_ssh_sends_the_script_byte_exact_with_the_target_and_bash():
    runner = runners.SshRunner("user@node-a.test", ssh=(sys.executable, "-c", ECHO))
    script = "set -eu\necho \"$HOME\" é\n"
    result = runner.run(script, timeout=30)
    assert result == runners.Result(0, script, "user@node-a.test bash -s")


def test_ssh_failures_look_like_connection_failures():
    missing = runners.SshRunner("node-a.test", ssh=("/nonexistent/ssh",)).run("true", timeout=5)
    assert missing.returncode == runners.SSH_FAILURE
    slow = runners.SshRunner("node-a.test", ssh=(sys.executable, "-c", "import time; time.sleep(5)")).run("", timeout=0.5)
    assert slow.returncode == runners.SSH_FAILURE and "no result within" in slow.stderr
    with pytest.raises(ValueError):
        runners.SshRunner("-oProxyCommand=x")


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, body):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self.reply(200, json.dumps({"data": [{"id": "Model-TP2"}]}))
        elif self.path == "/moved":
            self.send_response(302)
            self.send_header("Location", "/v1/models")
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.reply(200, "not json")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.reply(400 if body.get("bad") else 200, json.dumps({"echo": body}))


@pytest.fixture
def server():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_http_client_round_trips_json_and_reports_errors(server):
    client = runners.HttpClient()
    assert client.get_json(server + "/v1/models", timeout=5) == {"data": [{"id": "Model-TP2"}]}
    assert client.post_json(server + "/v1/chat", {"x": 1}, timeout=5) == {"echo": {"x": 1}}
    with pytest.raises(runners.HttpError, match="HTTP 400"):
        client.post_json(server + "/v1/chat", {"bad": True}, timeout=5)
    with pytest.raises(runners.HttpError, match="not JSON"):
        client.get_json(server + "/other", timeout=5)
    with pytest.raises(runners.HttpError, match="HTTP 302"):
        client.get_json(server + "/moved", timeout=5)


def test_run_process_appends_output_to_the_log(tmp_path):
    log = tmp_path / "bench.log"
    assert runners.run_process([sys.executable, "-c", "print('first')"], log_path=log) == 0
    assert runners.run_process([sys.executable, "-c", "import sys; sys.exit(3)"], log_path=log) == 3
    assert runners.run_process([sys.executable, "-c", "import time; time.sleep(5)"], log_path=log, timeout=0.5) == -1
    text = log.read_text()
    assert text.startswith("first") and "No exit within 0.5 s" in text


def test_run_process_gives_a_prompt_end_of_input(tmp_path):
    """A program that asks a question gets end of input instead of waiting for an answer."""
    log = tmp_path / "prompt.log"
    ask = "try:\n    input('Upgrade? ')\nexcept EOFError:\n    print('no answer')\n"
    assert runners.run_process([sys.executable, "-c", ask], log_path=log, timeout=30) == 0
    assert "no answer" in log.read_text()
