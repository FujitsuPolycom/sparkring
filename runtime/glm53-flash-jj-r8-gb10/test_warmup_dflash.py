from __future__ import annotations

import importlib.util
import io
import json
import threading
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


def _load_module():
    path = HERE / "warmup_dflash.py"
    spec = importlib.util.spec_from_file_location("warmup_dflash", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_warmup_exercises_each_concurrency_as_one_batch(monkeypatch) -> None:
    warmup = _load_module()
    active = 0
    peaks: dict[str, int] = {}
    lock = threading.Lock()
    release = threading.Barrier(4)

    def send(
        _endpoint,
        _model,
        nonce,
        _max_tokens,
        _timeout,
        _prompt_words,
        _credential,
        _temperature,
    ):
        nonlocal active
        concurrency = int(nonce.split("-", 1)[0][1:])
        with lock:
            active += 1
            peaks[str(concurrency)] = max(peaks.get(str(concurrency), 0), active)
        if concurrency == 4:
            release.wait(timeout=1)
        with lock:
            active -= 1

    monkeypatch.setattr(warmup, "send_warmup_request", send)

    result = warmup.run_warmup(
        "http://127.0.0.1:8015",
        "glm-5.3-flash",
        (1, 2, 4),
        16,
        2,
        (8, 24),
        None,
    )

    assert [item["concurrency"] for item in result] == [1, 2, 4, 2]
    assert [item["prompt_words"] for item in result] == [8, 8, 8, 24]
    assert peaks["4"] == 4


def test_wait_for_api_sends_bearer_credential(monkeypatch) -> None:
    warmup = _load_module()
    seen: list[dict[str, str]] = []

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def urlopen(request, timeout):
        assert timeout == 3
        seen.append(dict(request.header_items()))
        if request.get_header("Authorization") != "Bearer secret":
            raise warmup.urllib.error.HTTPError(
                request.full_url, 401, "Unauthorized", {}, None
            )
        return _Response()

    monkeypatch.setattr(warmup.urllib.request, "urlopen", urlopen)

    warmup.wait_for_api("http://127.0.0.1:8015", 5, "secret")

    assert seen == [{"Authorization": "Bearer secret"}]


def test_wait_for_api_without_credential_stays_anonymous(monkeypatch) -> None:
    warmup = _load_module()
    seen: list[dict[str, str]] = []

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def urlopen(request, timeout):
        seen.append(dict(request.header_items()))
        return _Response()

    monkeypatch.setattr(warmup.urllib.request, "urlopen", urlopen)

    warmup.wait_for_api("http://127.0.0.1:8015", 5)

    assert seen == [{}]


def test_temperature_defaults_preserve_parent_greedy_mode(monkeypatch):
    warmup = _load_module()
    monkeypatch.delenv("SPARKRING_WARMUP_TEMPERATURE", raising=False)
    assert warmup.resolve_temperature() == 0
    monkeypatch.setenv("SPARKRING_WARMUP_TEMPERATURE", "1")
    assert warmup.resolve_temperature() == 1
    assert warmup.resolve_temperature(0) == 0


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "no", "-0.1", "2.1", True])
def test_invalid_temperature_rejected(value):
    with pytest.raises(ValueError):
        _load_module().resolve_temperature(value)


def test_imported_warmup_resolves_environment_and_records_sampling(monkeypatch):
    warmup = _load_module()
    monkeypatch.setenv("SPARKRING_WARMUP_TEMPERATURE", "1")
    seen = []
    monkeypatch.setattr(warmup, "send_warmup_request", lambda *args: seen.append(args[-1]))
    results = warmup.run_warmup("http://localhost", "model", (1, 2), 16, 2, (8, 24))
    assert seen == [1.0] * 5
    assert all(item["temperature"] == 1 for item in results)


def test_invalid_environment_fails_before_requests(monkeypatch):
    warmup = _load_module()
    monkeypatch.setenv("SPARKRING_WARMUP_TEMPERATURE", "NaN")
    monkeypatch.setattr(warmup, "send_warmup_request", lambda *args: pytest.fail("Unexpected request"))
    with pytest.raises(ValueError):
        warmup.run_warmup("http://localhost", "model", (1,), 16, 2, (8,))


def test_request_uses_sampling_temperature_without_changing_thinking(monkeypatch):
    warmup = _load_module()
    monkeypatch.setenv("SPARKRING_WARMUP_TEMPERATURE", "1")
    seen = []

    def urlopen(request, timeout):
        seen.append(json.loads(request.data))
        return io.BytesIO(b'{"choices":[{"message":{"content":"ok"}}]}')

    monkeypatch.setattr(warmup.urllib.request, "urlopen", urlopen)
    warmup.send_warmup_request("http://localhost", "model", "sample", 16, 2, 8)
    assert seen[0]["temperature"] == 1
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen[0]["messages"][0]["content"].startswith("DFlash warmup sample. ")


def test_shape_batches_share_timeout_and_stop_before_an_unbudgeted_batch(monkeypatch):
    warmup = _load_module()
    clock = [0.0]
    monkeypatch.setattr(warmup.time, "monotonic", lambda: clock[0])
    observed = []

    def send(*args):
        observed.append(args[4])
        clock[0] += 3

    monkeypatch.setattr(warmup, "send_warmup_request", send)
    with pytest.raises(RuntimeError, match="deadline"):
        warmup.run_warmup("http://localhost", "model", (1,), 16, 5, (8, 24, 56))
    assert observed == [5, 2]


def test_shape_nonce_changes_between_warmup_runs(monkeypatch):
    warmup = _load_module()
    nonces = []
    timestamps = iter([10, 11])
    monkeypatch.setattr(warmup.time, "monotonic_ns", lambda: next(timestamps))
    monkeypatch.setattr(warmup, "send_warmup_request", lambda *args: nonces.append(args[2]))
    for _ in range(2):
        warmup.run_warmup("http://localhost", "model", (1,), 16, 5, (8,))
    assert nonces[0] != nonces[1]


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_timeout_fails_before_warmup_http(monkeypatch, timeout):
    warmup = _load_module()
    monkeypatch.setattr(warmup.urllib.request, "urlopen", lambda *_args, **_kwargs: pytest.fail("unexpected HTTP"))
    with pytest.raises(ValueError, match="finite and positive"):
        warmup.wait_for_api("http://localhost", timeout)


def test_api_wait_caps_each_attempt_by_remaining_budget(monkeypatch):
    warmup = _load_module()
    clock = [0.0]
    timeouts = []
    monkeypatch.setattr(warmup.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(warmup.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    def urlopen(_request, timeout):
        timeouts.append(timeout)
        clock[0] += timeout
        raise OSError("not ready")

    monkeypatch.setattr(warmup.urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match="did not become ready"):
        warmup.wait_for_api("http://localhost", 5)
    assert timeouts == [3, 1]
