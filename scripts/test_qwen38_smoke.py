"""Offline integration tests for the bounded Qwen API smoke harness."""

from __future__ import annotations

import json
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import qwen38_smoke as smoke  # noqa: E402


class _ServerState:
    def __init__(
        self,
        failing_gate: str | None = None,
        model_max_model_len: int = 262144,
    ) -> None:
        self.failing_gate = failing_gate
        self.model_max_model_len = model_max_model_len
        self.response_number = 0
        self.arithmetic_requests = 0


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

        def _chat(self, message: dict, finish_reason: str = "stop") -> None:
            state.response_number += 1
            self._send_json(
                200,
                {
                    "id": f"chatcmpl-dynamic-{state.response_number}",
                    "created": 1_800_000_000 + state.response_number,
                    "model": "server-side-name",
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": finish_reason,
                        }
                    ],
                },
            )

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path == "/v1/models":
                model_id = "wrong-model" if state.failing_gate == "models" else "qwen38"
                max_model_len = (
                    131072
                    if state.failing_gate == "model_length"
                    else state.model_max_model_len
                )
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {"id": model_id, "max_model_len": max_model_len}
                        ],
                    },
                )
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._send_json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            if (
                request.get("model") != "qwen38"
                or request.get("temperature") != 0
                or request.get("seed") != smoke.SMOKE_SEED
                or request.get("chat_template_kwargs") != {"enable_thinking": False}
            ):
                self._send_json(400, {"error": "sampling contract mismatch"})
                return
            user_content = request["messages"][-1]["content"]

            if request.get("tools"):
                function = request["tools"][0]["function"]
                if (
                    function.get("name") != "multiply"
                    or function.get("parameters", {}).get("required") != ["a", "b"]
                    or function.get("parameters", {}).get("additionalProperties") is not False
                    or request.get("tool_choice", {}).get("function", {}).get("name")
                    != "multiply"
                ):
                    self._send_json(400, {"error": "tool contract mismatch"})
                    return
                self._chat(
                    {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "tool selected",
                        "tool_calls": [
                            {
                                "id": f"call-dynamic-{state.response_number}",
                                "type": "function",
                                "function": {
                                    "name": "multiply",
                                    "arguments": '{"b": 7, "a": 6}',
                                },
                            }
                        ],
                    },
                    finish_reason="tool_calls",
                )
                return

            if isinstance(user_content, list):
                image_url = user_content[1].get("image_url", {}).get("url", "")
                if not image_url.startswith("data:image/png;base64,"):
                    self._send_json(400, {"error": "vision contract mismatch"})
                    return
                marker = "WRONG_MARKER" if state.failing_gate == "vision" else "VISION_OK"
                self._chat(
                    {
                        "role": "assistant",
                        "content": marker,
                        "reasoning_content": None,
                    }
                )
                return

            if "17 * 23" in user_content:
                state.arithmetic_requests += 1
                reasoning = "checked multiplication"
                if state.failing_gate == "arithmetic_stability" and state.arithmetic_requests == 2:
                    reasoning = "different reasoning"
                self._chat(
                    {
                        "role": "assistant",
                        "content": "391",
                        "reasoning_content": reasoning,
                    }
                )
                return

            if user_content.endswith("Return exactly PREFIX_OK."):
                if "record-255:" not in user_content:
                    self._send_json(400, {"error": "prefix workload too short"})
                    return
                self._chat(
                    {
                        "role": "assistant",
                        "content": "PREFIX_OK",
                        "reasoning_content": None,
                    }
                )
                return

            if user_content.endswith("Return exactly 13."):
                self._chat({"role": "assistant", "content": "13"})
                return

            if user_content.endswith("Return exactly 17."):
                answer = "13" if state.failing_gate == "divergence" else "17"
                self._chat({"role": "assistant", "content": answer})
                return

            self._send_json(400, {"error": "unknown bounded workload"})

    return Handler


@contextmanager
def _fake_server(
    failing_gate: str | None = None,
    model_max_model_len: int = 262144,
) -> Iterator[str]:
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _handler(_ServerState(failing_gate, model_max_model_len)),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_smoke_passes_all_bounded_gates_without_leaking_response_metadata() -> None:
    with _fake_server() as endpoint:
        result = smoke.run_smoke(endpoint=endpoint, model="qwen38", timeout=2)

    assert result["status"] == "pass"
    assert set(result["gates"]) == {
        "health",
        "models",
        "arithmetic_repeat",
        "tool_call",
        "vision",
        "native_prefix_replay",
        "shared_prefix_divergence",
    }
    assert all(gate["status"] == "pass" for gate in result["gates"].values())
    assert result["gates"]["tool_call"]["observed"] == {
        "function": "multiply",
        "arguments": {"a": 6, "b": 7},
    }
    assert result["gates"]["models"]["observed"]["max_model_len"] == 262144
    assert result["sampling"]["enable_thinking"] is False
    serialized = json.dumps(result, sort_keys=True)
    assert endpoint not in serialized
    assert "chatcmpl-dynamic" not in serialized
    assert "call-dynamic" not in serialized
    assert "elapsed" not in serialized
    assert "duration" not in serialized


def test_cli_returns_nonzero_and_machine_readable_json_when_one_gate_fails(
    tmp_path: Path,
) -> None:
    output = tmp_path / "smoke.json"
    with _fake_server("vision") as endpoint:
        exit_code = smoke.main(
            [
                "--endpoint",
                endpoint,
                "--model",
                "qwen38",
                "--timeout",
                "2",
                "--output",
                str(output),
            ]
        )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert result["status"] == "fail"
    assert result["gates"]["vision"] == {
        "status": "fail",
        "detail": "vision request did not return the expected marker",
    }
    assert result["gates"]["health"]["status"] == "pass"
    assert result["gates"]["shared_prefix_divergence"]["status"] == "pass"


def test_smoke_rejects_a_drifted_model_length() -> None:
    with _fake_server("model_length") as endpoint:
        result = smoke.run_smoke(endpoint=endpoint, model="qwen38", timeout=2)

    assert result["status"] == "fail"
    assert result["gates"]["models"] == {
        "status": "fail",
        "detail": "requested model does not report the expected token limit (262144)",
    }


def test_smoke_accepts_a_profile_specific_one_million_token_limit() -> None:
    with _fake_server(model_max_model_len=1_000_000) as endpoint:
        result = smoke.run_smoke(
            endpoint=endpoint,
            model="qwen38",
            timeout=2,
            expected_max_model_len=1_000_000,
        )

    assert result["status"] == "pass"
    assert result["gates"]["models"]["observed"]["max_model_len"] == 1_000_000


def test_arithmetic_gate_ignores_response_ids_but_rejects_message_drift() -> None:
    with _fake_server("arithmetic_stability") as endpoint:
        result = smoke.run_smoke(endpoint=endpoint, model="qwen38", timeout=2)

    assert result["status"] == "fail"
    assert result["gates"]["arithmetic_repeat"] == {
        "status": "fail",
        "detail": "repeated arithmetic stable message fields differ",
    }
    assert result["gates"]["tool_call"]["status"] == "pass"
