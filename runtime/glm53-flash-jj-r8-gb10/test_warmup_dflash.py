from __future__ import annotations

import importlib.util
import threading
from pathlib import Path


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
