"""Offline contracts for the DeepSeek-V4 request-shape warmup sweep."""

from __future__ import annotations

import json
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import deepseek_v4_shape_warmup as warmup  # noqa: E402


def _config(
    endpoint: str = "http://127.0.0.1:9",
    *,
    max_num_seqs: int = 4,
    block_size: int = 256,
    num_speculative_tokens: int = 5,
    max_num_batched_tokens: int = 4096,
    max_model_len: int = 1048576,
    bearer: str | None = None,
) -> warmup.WarmupConfig:
    return warmup.WarmupConfig(
        endpoint=endpoint,
        model="deepseek-v4-flash-0731",
        max_num_seqs=max_num_seqs,
        block_size=block_size,
        num_speculative_tokens=num_speculative_tokens,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        bearer=bearer,
    )


class _ServerState:
    def __init__(
        self,
        *,
        bearer: str | None = None,
        completion_skeleton: int = 6,
        completion_density: int = 1,
        chat_tokens: int = 24,
        failing_label: str | None = None,
        malformed: bool = False,
    ) -> None:
        self.bearer = bearer
        self.completion_skeleton = completion_skeleton
        self.completion_density = completion_density
        self.chat_tokens = chat_tokens
        self.failing_label = failing_label
        self.malformed = malformed
        self.requests: list[dict] = []
        self.lock = threading.Lock()

    def record(self, path: str, headers: dict, body: dict | None) -> None:
        with self.lock:
            self.requests.append(
                {
                    "path": path,
                    "authorization": headers.get("Authorization"),
                    "body": body,
                }
            )

    def paths(self, path: str) -> list[dict]:
        with self.lock:
            return [item for item in self.requests if item["path"] == path]


def _handler(state: _ServerState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _send_json(self, status: int, document: dict) -> None:
            encoded = json.dumps(document).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_completion(self, chat: bool) -> None:
            if state.malformed:
                self._send_json(200, {"choices": []})
                return
            if chat:
                self._send_json(
                    200,
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": "warm"},
                            }
                        ]
                    },
                )
                return
            self._send_json(
                200, {"choices": [{"finish_reason": "length", "text": "warm"}]}
            )

        def _authorized(self) -> bool:
            if state.bearer is None:
                return True
            if self.headers.get("Authorization") == f"Bearer {state.bearer}":
                return True
            self._send_json(401, {"error": "unauthorized"})
            return False

        def do_GET(self) -> None:  # noqa: N802
            state.record(self.path, self.headers, None)
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw.decode("utf-8"))
            state.record(self.path, self.headers, body)
            if not self._authorized():
                return
            if self.path == "/tokenize":
                if "messages" in body:
                    count = state.chat_tokens
                else:
                    prompt = body["prompt"]
                    filler = prompt.split("\n", 1)[1] if "\n" in prompt else ""
                    count = (
                        state.completion_skeleton
                        + state.completion_density * filler.count("warmup")
                    )
                self._send_json(
                    200,
                    {"count": count, "max_model_len": 1048576, "tokens": []},
                )
                return
            if self.path not in ("/v1/chat/completions", "/v1/completions"):
                self._send_json(404, {"error": "not found"})
                return
            chat = self.path == "/v1/chat/completions"
            content = (
                body["messages"][-1]["content"] if chat else body["prompt"]
            )
            if state.failing_label and state.failing_label in content:
                self._send_json(500, {"error": "injected failure"})
                return
            self._send_completion(chat)

    return Handler


@contextmanager
def _fake_api(**kwargs: object) -> Iterator[tuple[str, _ServerState]]:
    state = _ServerState(**kwargs)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _closed_endpoint() -> str:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def _main_arguments(config: warmup.WarmupConfig) -> list[str]:
    return [
        "--endpoint",
        config.endpoint,
        "--model",
        config.model,
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--block-size",
        str(config.block_size),
        "--num-speculative-tokens",
        str(config.num_speculative_tokens),
        "--max-num-batched-tokens",
        str(config.max_num_batched_tokens),
        "--max-model-len",
        str(config.max_model_len),
        "--wait-seconds",
        "5",
        "--timeout-seconds",
        "20",
    ]


def test_next_power_of_two_rounds_up_to_the_bucket_boundary() -> None:
    assert [warmup.next_power_of_two(value) for value in (1, 2, 3, 7, 8, 9, 250, 251)] == [
        1,
        2,
        4,
        8,
        8,
        16,
        256,
        256,
    ]
    with pytest.raises(warmup.WarmupError):
        warmup.next_power_of_two(0)


def test_compile_bucket_caps_at_the_deployment_block_size() -> None:
    config = _config()
    assert [
        warmup.compile_bucket(tokens, config) for tokens in (2, 10, 26, 58, 122, 250, 4096)
    ] == [8, 16, 32, 64, 128, 256, 256]
    small = _config(block_size=64)
    assert warmup.compile_bucket(4096, small) == 64


@pytest.mark.parametrize(
    ("block_size", "num_speculative_tokens", "expected"),
    [
        (256, 5, ((8, 2), (16, 10), (32, 26), (64, 58), (128, 122), (256, 250))),
        (64, 3, ((8, 4), (16, 12), (32, 28), (64, 60))),
        (100, 5, ((8, 2), (16, 10), (32, 26), (64, 58), (100, 94))),
    ],
)
def test_bucket_targets_land_exactly_on_each_compile_bucket(
    block_size: int, num_speculative_tokens: int, expected: tuple
) -> None:
    config = _config(
        block_size=block_size, num_speculative_tokens=num_speculative_tokens
    )
    targets = warmup.bucket_targets(config)
    assert targets == expected
    for bucket, tokens in targets:
        assert warmup.compile_bucket(tokens, config) == bucket


def test_concurrency_ladder_reaches_the_sequence_limit_once() -> None:
    assert warmup.concurrency_ladder(32) == (1, 2, 4, 8, 16, 32)
    assert warmup.concurrency_ladder(6) == (1, 2, 4, 6)
    assert warmup.concurrency_ladder(1) == (1,)


def test_sweep_is_deterministic_and_covers_the_required_arms() -> None:
    config = _config(max_num_seqs=6)
    arms, skipped = warmup.build_sweep(config)
    assert skipped == ()
    assert warmup.build_sweep(config) == (arms, skipped)
    labels = [arm.label for arm in arms]
    assert labels[:4] == ["concurrency-1", "concurrency-2", "concurrency-4", "concurrency-6"]
    assert labels[4:6] == ["prefill-medium", "prefill-multi-chunk"]
    assert labels[6] == "thinking-off"
    assert labels[7:] == [
        "tokens-2-bucket-8",
        "tokens-10-bucket-16",
        "tokens-26-bucket-32",
        "tokens-58-bucket-64",
        "tokens-122-bucket-128",
        "tokens-250-bucket-256",
    ]
    by_label = {arm.label: arm for arm in arms}
    assert by_label["prefill-medium"].prompt_tokens == 2048
    assert by_label["prefill-multi-chunk"].prompt_tokens == 8192 + 4096
    assert by_label["prefill-multi-chunk"].prompt_tokens > 8192
    assert by_label["thinking-off"].thinking_off is True
    assert [by_label[label].batch for label in labels[:4]] == [1, 2, 4, 6]
    assert all(
        by_label[label].batch == 1 for label in labels[4:7]
    )
    assert [by_label[label].prompt_tokens for label in labels[7:]] == [
        2,
        10,
        26,
        58,
        122,
        250,
    ]
    assert [by_label[label].compile_bucket for label in labels[7:]] == [
        8,
        16,
        32,
        64,
        128,
        256,
    ]


def test_sweep_skips_arms_longer_than_the_model_limit() -> None:
    arms, skipped = warmup.build_sweep(_config(max_model_len=9000))
    labels = [arm.label for arm in arms]
    assert "prefill-multi-chunk" not in labels
    assert "prefill-medium" in labels
    assert any("prefill-multi-chunk" in item for item in skipped)


def test_each_request_carries_a_distinct_leading_nonce() -> None:
    with _fake_api() as (endpoint, state):
        config = _config(endpoint)
        arms, skipped = warmup.build_sweep(config)
        summary = warmup.run_warmup(
            config,
            arms,
            wait_seconds=5,
            timeout_seconds=20,
            skipped=skipped,
        )
    assert summary["status"] == "complete"
    contents = []
    for path in ("/v1/chat/completions", "/v1/completions"):
        for item in state.paths(path):
            body = item["body"]
            contents.append(
                body["messages"][-1]["content"]
                if path.endswith("chat/completions")
                else body["prompt"]
            )
    assert len(contents) == sum(arm["completed"] for arm in summary["arms"])
    nonces = [content.split()[0] for content in contents]
    assert len(set(nonces)) == len(nonces)
    labels = {arm.label for arm in arms}
    for nonce in nonces:
        remainder = nonce.split("-", 1)[1]
        assert any(remainder.startswith(label + "-") for label in labels)


def test_tokenize_verifies_the_completion_prompt_before_firing() -> None:
    with _fake_api() as (endpoint, state):
        config = _config(endpoint)
        arms, skipped = warmup.build_sweep(config)
        run = warmup.run_warmup(
            config,
            arms,
            wait_seconds=5,
            timeout_seconds=20,
            skipped=skipped,
        )
    assert run["status"] == "complete"
    by_label = {arm.label: arm for arm in arms}
    tokenized = {
        item["body"]["prompt"]: item
        for item in state.paths("/tokenize")
        if "prompt" in item["body"]
    }
    fired = [item["body"]["prompt"] for item in state.paths("/v1/completions")]
    assert fired
    for prompt in fired:
        assert prompt in tokenized
        nonce = prompt.split()[0]
        label = nonce.split("-", 1)[1].rsplit("-", 1)[0]
        arm = by_label[label]
        filler = prompt.split("\n", 1)[1]
        measured = 6 + filler.count("warmup")
        assert measured == arm.prompt_tokens


def test_unreachable_low_bucket_is_reported_as_skipped() -> None:
    with _fake_api() as (endpoint, state):
        config = _config(endpoint)
        arms, skipped = warmup.build_sweep(config)
        run = warmup.run_warmup(
            config,
            arms,
            wait_seconds=5,
            timeout_seconds=20,
            skipped=skipped,
        )
    assert run["status"] == "complete"
    assert any("tokens-2-bucket-8" in item for item in run["skipped"])
    fired = [item["body"]["prompt"].split()[0] for item in state.paths("/v1/completions")]
    assert not any("tokens-2-bucket-8" in nonce for nonce in fired)


def test_token_counts_that_cannot_be_hit_fail_their_arms() -> None:
    with _fake_api(completion_density=3) as (endpoint, _state):
        config = _config(endpoint)
        arms, skipped = warmup.build_sweep(config)
        run = warmup.run_warmup(
            config,
            arms,
            wait_seconds=5,
            timeout_seconds=20,
            skipped=skipped,
        )
    assert run["status"] == "partial"
    failed_arms = {failure["arm"] for failure in run["failures"]}
    assert "tokens-10-bucket-16" in failed_arms
    assert "prefill-medium" in failed_arms
    assert any(arm["label"] == "thinking-off" for arm in run["arms"])


def test_a_failed_request_is_recorded_and_later_arms_still_run() -> None:
    with _fake_api(failing_label="concurrency-2") as (endpoint, _state):
        config = _config(endpoint)
        arms, skipped = warmup.build_sweep(config)
        run = warmup.run_warmup(
            config,
            arms,
            wait_seconds=5,
            timeout_seconds=20,
            skipped=skipped,
        )
    assert run["status"] == "partial"
    assert {failure["arm"] for failure in run["failures"]} == {"concurrency-2"}
    assert [arm["label"] for arm in run["arms"]][-1] == "tokens-250-bucket-256"


def test_malformed_completion_is_a_failure() -> None:
    with _fake_api(malformed=True) as (endpoint, _state):
        config = _config(endpoint)
        arms, skipped = warmup.build_sweep(config)
        run = warmup.run_warmup(
            config,
            arms,
            wait_seconds=5,
            timeout_seconds=20,
            skipped=skipped,
        )
    assert run["status"] == "failed"
    assert run["failures"]


def test_bearer_from_the_key_file_reaches_every_api_request(tmp_path: Path) -> None:
    key_file = tmp_path / "api-keys"
    key_file.write_text("\n  indented\nsecret-key\n", encoding="utf-8")
    bearer = warmup.read_bearer(str(key_file))
    assert bearer == "secret-key"
    with _fake_api(bearer=bearer) as (endpoint, state):
        config = _config(endpoint, bearer=bearer)
        arms, skipped = warmup.build_sweep(config)
        run = warmup.run_warmup(
            config,
            arms,
            wait_seconds=5,
            timeout_seconds=20,
            skipped=skipped,
        )
    assert run["status"] == "complete"
    assert run["authenticated"] is True
    protected = [
        item
        for item in state.requests
        if item["path"] in ("/tokenize", "/v1/chat/completions", "/v1/completions")
    ]
    assert protected
    assert {item["authorization"] for item in protected} == {f"Bearer {bearer}"}
    health = state.paths("/health")
    assert health and all(item["authorization"] is None for item in health)


def test_an_unconfigured_bearer_sends_no_authorization_header() -> None:
    with _fake_api() as (endpoint, state):
        config = _config(endpoint)
        arms, skipped = warmup.build_sweep(config)
        warmup.run_warmup(
            config,
            arms,
            wait_seconds=5,
            timeout_seconds=20,
            skipped=skipped,
        )
    assert all(item["authorization"] is None for item in state.requests)


def test_read_bearer_rejects_a_file_without_a_key(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.write_text("\n \n", encoding="utf-8")
    with pytest.raises(warmup.WarmupError, match="no non-empty line"):
        warmup.read_bearer(str(empty))
    with pytest.raises(warmup.WarmupError, match="cannot read"):
        warmup.read_bearer(str(tmp_path / "missing"))


def test_readiness_timeout_is_reported_without_raising() -> None:
    config = _config(_closed_endpoint())
    arms, skipped = warmup.build_sweep(config)
    run = warmup.run_warmup(
        config,
        arms,
        wait_seconds=0.2,
        timeout_seconds=1,
        skipped=skipped,
    )
    assert run["status"] == "failed"
    assert run["failures"][0]["arm"] == "readiness"
    assert run["arms"] == []


def test_main_reports_a_complete_sweep_and_exits_zero(capsys) -> None:
    with _fake_api() as (endpoint, _state):
        config = _config(endpoint)
        code = warmup.main(_main_arguments(config))
        captured = capsys.readouterr()
    assert code == 0, captured.err
    summary = json.loads(captured.out)["shape_warmup"]
    assert summary["status"] == "complete"
    assert summary["max_tokens"] == 6
    assert summary["block_size"] == 256
    assert "skipped: tokens-2-bucket-8" in captured.err


def test_main_exits_nonzero_when_any_arm_fails(capsys) -> None:
    with _fake_api(completion_density=3) as (endpoint, _state):
        config = _config(endpoint)
        code = warmup.main(_main_arguments(config))
        captured = capsys.readouterr()
    assert code == 1
    summary = json.loads(captured.out)["shape_warmup"]
    assert summary["status"] == "partial"
    assert "shape warmup: warning:" in captured.err


def test_main_fails_before_network_access_without_a_readable_key_file(
    tmp_path: Path, capsys
) -> None:
    config = _config()
    code = warmup.main(
        [*_main_arguments(config), "--api-key-file", str(tmp_path / "missing")]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "cannot read the API key file" in captured.err
    assert captured.out == ""
