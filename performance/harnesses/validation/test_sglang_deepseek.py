"""Exercise admission and result handling against a loopback HTTP fixture."""
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import threading

import pytest

spec = importlib.util.spec_from_file_location("sglang_deepseek", Path(__file__).with_name("sglang_deepseek.py"))
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)
KEYS = ("fixture-key-a", "fixture-key-b")


@pytest.fixture(autouse=True)
def process_umask():
    """The harness sets a private umask for its result files; later tests in this process keep their own."""
    previous = os.umask(0o022)
    os.umask(previous)
    yield
    os.umask(previous)


@pytest.fixture
def api():
    state = {"requests": [], "allow_unauthenticated": False, "bad_count": False,
             "bad_answer": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.respond()

        def do_POST(self):
            self.respond()

        def respond(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            auth = self.headers.get("Authorization")
            state["requests"].append((self.path, auth, body))
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/credential-destination")
                self.end_headers()
                return
            if self.path == "/health":
                status, result = 200, {}
            elif auth not in {"Bearer " + key for key in KEYS} and not state["allow_unauthenticated"]:
                status, result = 401, {"error": "unauthorized"}
            elif self.path == "/v1/models":
                status, result = 200, {"data": [{"id": "deepseek-v4.1-flash"}]}
            elif self.path == "/generate":
                status, result = 200, {"text": "IRIS-FIXTURE", "meta_info": {
                    "prompt_tokens": len(body["input_ids"]) - int(state["bad_count"]),
                    "completion_tokens": 4, "cached_tokens": 0}}
            else:
                text = '{"answer":42}' if "response_format" in body else ("41" if state["bad_answer"] else "42")
                status, result = 200, {"choices": [{"finish_reason": "stop", "message": {"content": text}}],
                                       "usage": {"prompt_tokens": 20, "completion_tokens": 4}}
            raw = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def run(tmp_path, url, *extra):
    keys = tmp_path / "keys"
    keys.write_text("\n".join(KEYS))
    output = tmp_path / "result.json"
    result = harness.main(["--url", url, "--key-file", str(keys), "--model-path", str(tmp_path),
                           "--output", str(output), *extra])
    raw = output.read_text()
    assert all(key not in raw for key in KEYS)
    return result, json.loads(raw)


def test_short_api_checks_require_auth_and_both_keys(tmp_path, api):
    url, state = api
    status, report = run(tmp_path, url)
    assert status == 0 and report["passed"] and report["short_outputs_equal"]
    assert len(report["tests"]) == 10
    denials = [test for test in report["tests"] if test["name"].startswith("deny_")]
    assert len(denials) == 4 and all(test["http_status"] == 401 for test in denials)
    generated = [body for path, _, body in state["requests"] if path == "/v1/chat/completions"]
    assert all(body["max_tokens"] == 32 and body["temperature"] == 0 for body in generated)
    assert any(body.get("response_format", {}).get("type") == "json_schema" for body in generated)


@pytest.mark.parametrize("failure", ["allow_unauthenticated", "bad_answer"])
def test_authentication_and_answer_failures_propagate(tmp_path, api, failure):
    url, state = api
    state[failure] = True
    status, report = run(tmp_path, url)
    assert status == 1 and not report["passed"]
    if failure == "allow_unauthenticated":
        assert "generation_skipped" in report
        assert not any(test["name"].startswith("deterministic_") for test in report["tests"])
    else:
        failed = [test for test in report["tests"] if not test["passed"]]
        assert len(failed) == 2 and all("exactly 42" in test["error"] for test in failed)


@pytest.mark.parametrize("bad_count", [False, True])
def test_exact_prompt_count_controls_long_result(tmp_path, api, monkeypatch, bad_count):
    url, state = api
    state["bad_count"] = bad_count
    monkeypatch.setattr(harness, "make_long_prompt", lambda *args: (
        [1, 2, 3], "IRIS-FIXTURE", {"input_tokens": 3, "input_ids_sha256": "fixture"}))
    status, report = run(tmp_path, url, "--long-context")
    assert status == int(bad_count)
    assert report["long_response"]["server_prompt_tokens"] == 3 - int(bad_count)
    assert report["tests"][-1]["passed"] is not bad_count
    body = next(body for path, _, body in state["requests"] if path == "/generate")
    assert body["input_ids"] == [1, 2, 3] and body["sampling_params"]["max_new_tokens"] == 32


def test_redirect_does_not_forward_bearer(api):
    url, state = api
    status, _, _ = harness.request(url, "/redirect", KEYS[0])
    assert status == 302
    assert [path for path, _, _ in state["requests"]] == ["/redirect"]


@pytest.mark.parametrize("changed", ["config.json", "tokenizer.json", "encoding.py"])
def test_local_tokenizer_identity_checked_before_import(tmp_path, monkeypatch, changed):
    (tmp_path / "encoding").mkdir()
    (tmp_path / "encoding/encoding.py").write_text("raise AssertionError('must not import')")
    identities = {"config.json": harness.MODEL_CONFIG_SHA256,
                  "tokenizer.json": harness.TOKENIZER_SHA256,
                  "encoding.py": harness.CHAT_ENCODER_SHA256}
    identities[changed] = "incorrect"
    monkeypatch.setattr(harness, "digest", lambda path: identities[Path(path).name])
    with pytest.raises(ValueError, match="differs from the selected DeepSeek checkpoint"):
        harness.make_long_prompt(tmp_path, 1024, 2048)
