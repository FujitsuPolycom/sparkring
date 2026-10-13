"""Management security and idle cost, entirely on CPU with a fake helper."""
import http.client
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from runtime.host import action_commands, dashboard, dashboard_actions, helper
from runtime.host.test_dashboard import collector, run_for

TOKEN = "cpu-fixture-" * 4
REVIEW = "a" * 64


def manager():
    calls = []
    def invoke(argv):
        helper.validate(argv)
        calls.append(argv)
        return {"code": 0, "stdout": json.dumps({"review": REVIEW, "activity": {"grade": "clear"}}), "stderr": "Plan output"}
    value = dashboard_actions.Manager(TOKEN, invoke=invoke)
    headers = {"Authorization": "Bearer " + TOKEN, "Host": "127.0.0.1", "Origin": "http://127.0.0.1",
               "X-CSRF-Token": value.csrf}
    return value, headers, calls


@pytest.mark.parametrize("action", ["install", "down", "restart", "recovery-on", "recovery-off", "logs"])
def test_each_button_is_one_helper_call_per_preview_or_confirmation(action):
    value, headers, calls = manager()
    request = {"action": action, "on": "4-5"}
    if action == "install":
        request.update(profile="qwen38-flash-next-tp2", options={"checkpoint": "csf", "context-length": "8192"})
    preview = value.dispatch("/api/manage/preview", headers, request)
    assert len(calls) == 1
    if action == "logs":
        assert "--follow" in preview["follow"]
        assert "confirmation" not in preview
        return
    assert "--plan" in calls[0]
    value.dispatch("/api/manage/confirm", headers, {"confirmation": preview["confirmation"]})
    assert len(calls) == 2 and calls[-1][-2:] == ["--review", REVIEW]
    with pytest.raises(ValueError, match="already used"):
        value.dispatch("/api/manage/confirm", headers, {"confirmation": preview["confirmation"]})
    assert len(calls) == 2


@pytest.mark.parametrize("changes", [{"Authorization": "Bearer wrong"}, {"Origin": "http://evil.example"},
                                    {"X-CSRF-Token": "wrong"}, {"Sec-Fetch-Site": "cross-site"}])
def test_auth_and_csrf_fail_before_helper(changes):
    value, headers, calls = manager()
    with pytest.raises(PermissionError):
        value.dispatch("/api/manage/preview", {**headers, **changes}, {"action": "down", "on": "4-5"})
    assert calls == []


def test_confirmation_expiry_and_argument_substitution():
    value, headers, calls = manager()
    now = [0]
    value.now = lambda: now[0]
    preview = value.dispatch("/api/manage/preview", headers, {"action": "down", "on": "4-5"})
    with pytest.raises(ValueError, match="exact preview"):
        value.dispatch("/api/manage/confirm", headers, {"confirmation": preview["confirmation"], "on": "0-3"})
    now[0] = dashboard_actions.TTL + 1
    with pytest.raises(ValueError, match="expired"):
        value.dispatch("/api/manage/confirm", headers, {"confirmation": preview["confirmation"]})
    assert len(calls) == 1


@pytest.mark.parametrize("suffix", [["--env", "/tmp/code"], ["--image-lock", "/tmp/image"],
                                    ["--events", "/etc/shadow"], ["--fresh-mesh"], [";reboot"],
                                    ["--model-path", "/etc"], ["--on", "0-3"]])
def test_helper_rejects_code_paths_and_extra_arguments(suffix):
    argv = action_commands.command({"action": "install", "on": "4-5", "profile": "profile"})
    with pytest.raises(ValueError):
        helper.validate([*argv[:-2], *suffix, *argv[-2:]])


def test_helper_rejects_malformed_names_ranges_and_duplicate_options():
    for request in ({"action": "install", "on": "4-5", "profile": "../../etc"},
                    {"action": "down", "on": "4;reboot"},
                    {"action": "install", "on": "4-5", "profile": "x", "options": {"context-length": "100000000000"}}):
        with pytest.raises(ValueError):
            action_commands.command(request)


def test_manage_mode_refuses_root_before_collecting(monkeypatch):
    monkeypatch.setattr(dashboard.os, "geteuid", lambda: 0, raising=False)
    with pytest.raises(ValueError, match="unprivileged"):
        dashboard.serve(SimpleNamespace(manage=True))


def test_idle_manage_mode_adds_no_threads_helper_calls_or_model_requests(tmp_path, monkeypatch):
    (tmp_path / "readonly").mkdir()
    (tmp_path / "manage").mkdir()
    first, api1, clock1 = collector(tmp_path / "readonly")
    second, api2, clock2 = collector(tmp_path / "manage")
    def refuse(*args, **kwargs):
        pytest.fail("management started a background thread")
    monkeypatch.setattr(threading.Thread, "start", refuse)
    value, _, calls = manager()
    dashboard.make_handler(second, *dashboard.page_bytes(), manager=value)
    run_for(first, clock1, 300, viewer=True)
    run_for(second, clock2, 300, viewer=True)
    assert len(api1.requests) == len(api2.requests)
    assert [r.full_url for r in api1.requests] == [r.full_url for r in api2.requests]
    assert calls == []
    script = Path(dashboard.PAGE).read_text(encoding="utf-8")
    actions = script.split("let manageTicket", 1)[1].split("// Hovering", 1)[0]
    assert all(word not in actions for word in ("setInterval", "setTimeout", "WebSocket"))


def test_http_management_auth_csrf_preview_and_confirmation(tmp_path):
    found, _, _ = collector(tmp_path)
    value, headers, calls = manager()
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0),
        dashboard.make_handler(found, *dashboard.page_bytes(), manager=value))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def request(method, path, body=None, extra=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request(method, path, body=json.dumps(body) if body else None,
                           headers={**headers, "Content-Type": "application/json", **(extra or {})})
        response = connection.getresponse()
        status, payload = response.status, json.loads(response.read())
        connection.close()
        return status, payload
    try:
        assert request("GET", "/api/manage", extra={"Authorization": "wrong"})[0] == 401
        assert request("GET", "/api/manage")[1]["csrf"] == value.csrf
        body = {"action": "down", "on": "4-5"}
        assert request("POST", "/api/manage/preview", body, {"X-CSRF-Token": ""})[0] == 403
        assert request("POST", "/api/manage/preview", body, {"Host": "evil.example"})[0] == 421
        status, preview = request("POST", "/api/manage/preview", body)
        assert status == 200 and len(calls) == 1
        status, _ = request("POST", "/api/manage/confirm", {"confirmation": preview["confirmation"]})
        assert status == 200 and len(calls) == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.skipif(os.name != "posix", reason="Linux SO_PEERCRED")
def test_audit_identity_is_uid_not_pid():
    import socket
    connection, peer = socket.socketpair()
    with connection, peer:
        assert helper._peer_uid(connection) == os.getuid()
