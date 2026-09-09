"""Requests cannot enter the inference scheduler before warmup completes."""

import asyncio
import importlib.util
from pathlib import Path

import pytest


def load_gate():
    spec = importlib.util.spec_from_file_location(
        "startup_admission", Path(__file__).with_name("startup_admission.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def request(gate, path, headers=(), client="192.0.2.10", method="POST"):
    messages = []

    async def receive():
        raise AssertionError("Rejected requests must not consume the request body")

    async def send(message):
        messages.append(message)

    asyncio.run(gate({"type": "http", "path": path, "method": method,
                      "headers": list(headers), "client": (client, 1234)}, receive, send))
    return messages


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions",
                                     "/v1/responses", "/v1/embeddings", "/generate"])
def test_public_requests_never_reach_scheduler_before_ready(tmp_path, monkeypatch, path):
    module = load_gate()
    monkeypatch.setenv("SPARKRING_STARTUP_TOKEN", "internal-secret")
    monkeypatch.setenv("SPARKRING_READY_PATH", str(tmp_path / "ready"))

    async def scheduler(*args):
        raise AssertionError("Request entered inference before warmup")

    gate = module.StartupAdmission(scheduler)
    for headers, client in [((), "192.0.2.10"), ((), "127.0.0.1"),
                            (((b"x-sparkring-startup-token", b"wrong"),), "127.0.0.1")]:
        messages = request(gate, path, headers, client)
        assert messages[0]["status"] == 503
        assert (b"retry-after", b"5") in messages[0]["headers"]


def test_internal_warmup_then_public_admission_and_marker_removal(tmp_path, monkeypatch):
    module = load_gate()
    ready = tmp_path / "ready"
    monkeypatch.setenv("SPARKRING_STARTUP_TOKEN", "internal-secret")
    monkeypatch.setenv("SPARKRING_READY_PATH", str(ready))
    calls = []

    async def scheduler(scope, receive, send):
        calls.append(scope)
        await send({"type": "http.response.start", "status": 200, "headers": []})

    gate = module.StartupAdmission(scheduler)
    assert request(gate, "/v1/chat/completions",
                   [(b"x-sparkring-startup-token", b"internal-secret")])[0]["status"] == 200
    assert calls[0]["headers"] == []
    for path in ("/health", "/v1/models", "/metrics"):
        assert request(gate, path, method="GET")[0]["status"] == 200
    ready.touch()
    assert request(gate, "/v1/chat/completions")[0]["status"] == 200
    ready.unlink()
    assert request(gate, "/v1/chat/completions")[0]["status"] == 503


def test_missing_internal_token_fails_closed(tmp_path, monkeypatch):
    module = load_gate()
    monkeypatch.delenv("SPARKRING_STARTUP_TOKEN", raising=False)
    monkeypatch.setenv("SPARKRING_READY_PATH", str(tmp_path / "ready"))
    with pytest.raises(RuntimeError, match="startup token"):
        module.StartupAdmission(None)


def test_websocket_cannot_bypass_admission(tmp_path, monkeypatch):
    module = load_gate()
    monkeypatch.setenv("SPARKRING_STARTUP_TOKEN", "internal-secret")
    monkeypatch.setenv("SPARKRING_READY_PATH", str(tmp_path / "ready"))
    messages = []

    async def scheduler(*args):
        raise AssertionError("Websocket reached scheduler during warmup")

    async def send(message):
        messages.append(message)

    asyncio.run(module.StartupAdmission(scheduler)(
        {"type": "websocket", "path": "/v1/realtime", "headers": []}, None, send
    ))
    assert messages == [{"type": "websocket.close", "code": 1013}]
