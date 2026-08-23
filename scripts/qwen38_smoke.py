#!/usr/bin/env python3
"""Run bounded, sanitized API checks against the Qwen four-Spark profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


SCHEMA = "sparkring-qwen38-api-smoke/v1"
SMOKE_SEED = 17_023
EXPECTED_MAX_MODEL_LEN = 262_144
VISION_MARKER = "VISION_OK"
PREFIX_MARKER = "PREFIX_OK"
TINY_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
    "x8AAusB9Y9Wl1sAAAAASUVORK5CYII="
)


class SmokeFailure(RuntimeError):
    """One bounded API gate did not satisfy its public contract."""


class _Client:
    def __init__(self, endpoint: str, timeout: float) -> None:
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise SmokeFailure("endpoint must be an absolute HTTP or HTTPS URL")
        if parsed.query or parsed.fragment:
            raise SmokeFailure("endpoint must not contain a query or fragment")
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        document: dict[str, Any] | None = None,
    ) -> tuple[int, bytes]:
        encoded = None
        headers = {"Accept": "application/json"}
        if document is not None:
            encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self._endpoint}{path}",
            data=encoded,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                return response.status, response.read()
        except HTTPError as error:
            raise SmokeFailure(f"HTTP request returned status {error.code}") from None
        except URLError as error:
            reason = type(error.reason).__name__
            raise SmokeFailure(f"HTTP transport failed ({reason})") from None
        except TimeoutError:
            raise SmokeFailure("HTTP request timed out") from None

    def get(self, path: str) -> tuple[int, bytes]:
        return self._request("GET", path)

    def get_json(self, path: str) -> tuple[int, dict[str, Any]]:
        status, payload = self.get(path)
        return status, _decode_json(payload)

    def post_json(self, path: str, document: dict[str, Any]) -> dict[str, Any]:
        _status, payload = self._request("POST", path, document)
        return _decode_json(payload)


def _decode_json(payload: bytes) -> dict[str, Any]:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SmokeFailure("API response was not valid UTF-8 JSON") from None
    if not isinstance(document, dict):
        raise SmokeFailure("API response JSON must be an object")
    return document


def _normalized_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        raise SmokeFailure("assistant tool_calls must be a list")
    normalized: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            raise SmokeFailure("assistant tool call is malformed")
        function = call["function"]
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, str):
            raise SmokeFailure("assistant tool function is malformed")
        try:
            decoded_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            raise SmokeFailure("assistant tool arguments are not valid JSON") from None
        normalized.append(
            {
                "type": call.get("type"),
                "function": {"name": name, "arguments": decoded_arguments},
            }
        )
    return normalized


def _stable_choice(document: dict[str, Any]) -> dict[str, Any]:
    choices = document.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise SmokeFailure("chat response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise SmokeFailure("chat response choice is malformed")
    message = choice["message"]
    return {
        "role": message.get("role"),
        "content": message.get("content"),
        "reasoning_content": message.get("reasoning_content"),
        "tool_calls": _normalized_tool_calls(message),
        "finish_reason": choice.get("finish_reason"),
    }


def _stable_hash(stable_choice: dict[str, Any]) -> str:
    encoded = json.dumps(
        stable_choice,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _content(stable_choice: dict[str, Any], expected: str, gate: str) -> str:
    content = stable_choice.get("content")
    if not isinstance(content, str) or content.strip() != expected:
        raise SmokeFailure(f"{gate} did not return the expected marker")
    return content.strip()


def _chat_payload(
    model: str,
    content: str | list[dict[str, Any]],
    *,
    max_tokens: int = 64,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "seed": SMOKE_SEED,
        "max_tokens": max_tokens,
    }


def _shared_prefix() -> str:
    lines = [
        "Qwen API shared-prefix smoke workload version 1.",
        "The bounded suffix values are A=13 and B=17.",
    ]
    lines.extend(
        f"record-{index:03d}: alpha beta gamma delta epsilon zeta eta theta"
        for index in range(256)
    )
    return "\n".join(lines)


def _run_gate(operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        observed = operation()
    except SmokeFailure as error:
        return {"status": "fail", "detail": str(error)}
    return {"status": "pass", "observed": observed}


def run_smoke(endpoint: str, model: str, timeout: float = 60.0) -> dict[str, Any]:
    """Run every bounded gate and return a sanitized JSON-compatible result."""

    client = _Client(endpoint, timeout)
    prefix = _shared_prefix()

    def health() -> dict[str, Any]:
        status, _payload = client.get("/health")
        return {"http_status": status}

    def models() -> dict[str, Any]:
        status, document = client.get_json("/v1/models")
        entries = document.get("data")
        if not isinstance(entries, list):
            raise SmokeFailure("model list response has no data list")
        matching = [
            entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("id") == model
        ]
        if len(matching) != 1:
            raise SmokeFailure("requested model is absent from /v1/models")
        if matching[0].get("max_model_len") != EXPECTED_MAX_MODEL_LEN:
            raise SmokeFailure(
                "requested model does not report the 262144-token limit"
            )
        return {
            "http_status": status,
            "model_present": True,
            "max_model_len": EXPECTED_MAX_MODEL_LEN,
        }

    def arithmetic_repeat() -> dict[str, Any]:
        payload = _chat_payload(
            model,
            "What is 17 * 23? Return only the integer.",
            max_tokens=64,
        )
        first = _stable_choice(client.post_json("/v1/chat/completions", payload))
        second = _stable_choice(client.post_json("/v1/chat/completions", payload))
        _content(first, "391", "arithmetic request")
        _content(second, "391", "repeated arithmetic request")
        if first != second:
            raise SmokeFailure("repeated arithmetic stable message fields differ")
        return {
            "answer": "391",
            "repetitions": 2,
            "stable_fields_sha256": _stable_hash(first),
        }

    def tool_call() -> dict[str, Any]:
        payload = _chat_payload(
            model,
            "Use the multiply tool to multiply 6 by 7. Do not answer directly.",
        )
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "multiply",
                    "description": "Multiply two integers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        payload["tool_choice"] = {
            "type": "function",
            "function": {"name": "multiply"},
        }
        stable = _stable_choice(client.post_json("/v1/chat/completions", payload))
        calls = stable["tool_calls"]
        if len(calls) != 1:
            raise SmokeFailure("tool request did not return exactly one call")
        function = calls[0]["function"]
        if function != {"name": "multiply", "arguments": {"a": 6, "b": 7}}:
            raise SmokeFailure("tool request returned the wrong function or arguments")
        return {"function": "multiply", "arguments": {"a": 6, "b": 7}}

    def vision() -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": f"Process the attached image and return exactly {VISION_MARKER}.",
            },
            {"type": "image_url", "image_url": {"url": TINY_PNG_DATA_URL}},
        ]
        payload = _chat_payload(model, content, max_tokens=32)
        stable = _stable_choice(client.post_json("/v1/chat/completions", payload))
        marker = _content(stable, VISION_MARKER, "vision request")
        return {"marker": marker}

    def native_prefix_replay() -> dict[str, Any]:
        payload = _chat_payload(
            model,
            f"{prefix}\nReturn exactly {PREFIX_MARKER}.",
            max_tokens=32,
        )
        first = _stable_choice(client.post_json("/v1/chat/completions", payload))
        second = _stable_choice(client.post_json("/v1/chat/completions", payload))
        marker = _content(first, PREFIX_MARKER, "native-prefix first request")
        _content(second, PREFIX_MARKER, "native-prefix replay request")
        if first != second:
            raise SmokeFailure("native-prefix replay stable message fields differ")
        return {
            "marker": marker,
            "identical_output": True,
            "stable_fields_sha256": _stable_hash(first),
        }

    def shared_prefix_divergence() -> dict[str, Any]:
        first_payload = _chat_payload(
            model,
            f"{prefix}\nReturn exactly 13.",
            max_tokens=32,
        )
        second_payload = _chat_payload(
            model,
            f"{prefix}\nReturn exactly 17.",
            max_tokens=32,
        )
        first = _stable_choice(client.post_json("/v1/chat/completions", first_payload))
        second = _stable_choice(client.post_json("/v1/chat/completions", second_payload))
        answer_a = _content(first, "13", "shared-prefix suffix A")
        answer_b = _content(second, "17", "shared-prefix suffix B")
        return {"answers": [answer_a, answer_b], "outputs_diverged": True}

    operations: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
        ("health", health),
        ("models", models),
        ("arithmetic_repeat", arithmetic_repeat),
        ("tool_call", tool_call),
        ("vision", vision),
        ("native_prefix_replay", native_prefix_replay),
        ("shared_prefix_divergence", shared_prefix_divergence),
    )
    gates = {name: _run_gate(operation) for name, operation in operations}
    passed = all(gate["status"] == "pass" for gate in gates.values())
    return {
        "schema": SCHEMA,
        "status": "pass" if passed else "fail",
        "model": model,
        "scope": "bounded API correctness smoke; no performance or cache-hit claim",
        "sampling": {"temperature": 0, "seed": SMOKE_SEED},
        "gates": gates,
        "limitations": [
            "No timing is collected or reported.",
            "Repeated-prefix equality does not prove a native-prefix cache hit.",
            "The result contains no endpoint, response ID, timestamp, or raw model reasoning.",
        ],
    }


def _write_result(document: dict[str, Any], output: str) -> None:
    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if output == "-":
        sys.stdout.write(encoded)
        return
    Path(output).write_text(encoded, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen38")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", default="-", help="JSON path, or - for stdout")
    arguments = parser.parse_args(argv)
    if arguments.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        document = run_smoke(arguments.endpoint, arguments.model, arguments.timeout)
    except SmokeFailure as error:
        document = {
            "schema": SCHEMA,
            "status": "fail",
            "model": arguments.model,
            "gates": {
                "configuration": {"status": "fail", "detail": str(error)}
            },
        }
    _write_result(document, arguments.output)
    return 0 if document["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
