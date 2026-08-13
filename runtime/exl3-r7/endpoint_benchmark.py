"""Run a bounded correctness and throughput check against an OpenAI API server.

The harness keeps warmup outside the measured request.  It reports client-side
prompt tokens per time-to-first-token and inter-token decode throughput using
the server's final usage counters.  The result is diagnostic evidence for one
endpoint invocation; it is not an acceptance result or a reference-lane claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "sparkring-r7-endpoint-benchmark/v1"
DEFAULT_SEED = 20260811
EXPECTED_ANSWER = re.compile(r"(?<!\d)42(?!\d)")


class BenchmarkError(RuntimeError):
    """The endpoint did not satisfy the benchmark contract."""


@dataclass(frozen=True)
class StreamResult:
    started: float
    first_token_at: float
    last_token_at: float
    finished: float
    usage: dict[str, Any]
    finish_reason: str | None
    text: str
    content_events: int


def _request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> tuple[int, Any, float]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            raw = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        raw = exc.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise BenchmarkError(f"request to {url} failed: {exc}") from exc
    elapsed = time.monotonic() - started
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        body = raw
    return status, body, elapsed


def _require_success(status: int, body: Any, operation: str) -> dict[str, Any]:
    if status != 200 or not isinstance(body, dict):
        snippet = str(body).replace("\n", " ")[:500]
        raise BenchmarkError(f"{operation} returned HTTP {status}: {snippet}")
    return body


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise BenchmarkError(f"{name} must be a positive integer, got {value!r}")
    return value


def validate_completion_logprobs(
    body: dict[str, Any], expected_completion_tokens: int
) -> dict[str, Any]:
    """Validate OpenAI completion logprobs and return bounded statistics."""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise BenchmarkError("completion response has no first choice")
    logprobs = choices[0].get("logprobs")
    if not isinstance(logprobs, dict):
        raise BenchmarkError("completion response omitted requested logprobs")
    chosen = logprobs.get("token_logprobs")
    tops = logprobs.get("top_logprobs")
    tokens = logprobs.get("tokens")
    if not all(isinstance(value, list) for value in (chosen, tops, tokens)):
        raise BenchmarkError("completion logprob arrays are malformed")
    if not (len(chosen) == len(tops) == len(tokens) == expected_completion_tokens):
        raise BenchmarkError(
            "completion logprob lengths disagree with usage: "
            f"chosen={len(chosen)}, top={len(tops)}, tokens={len(tokens)}, "
            f"expected={expected_completion_tokens}"
        )

    finite_values: list[float] = []
    for index, value in enumerate(chosen):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise BenchmarkError(f"chosen logprob {index} is not numeric: {value!r}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise BenchmarkError(f"chosen logprob {index} is nonfinite: {value!r}")
        finite_values.append(numeric)
    for index, top in enumerate(tops):
        if not isinstance(top, dict) or not top:
            raise BenchmarkError(f"top_logprobs[{index}] is not a non-empty object")
        for token, value in top.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise BenchmarkError(
                    f"top_logprobs[{index}][{token!r}] is not numeric: {value!r}"
                )
            numeric = float(value)
            if not math.isfinite(numeric):
                raise BenchmarkError(
                    f"top_logprobs[{index}][{token!r}] is nonfinite: {value!r}"
                )
            finite_values.append(numeric)
    return {
        "completion_tokens": expected_completion_tokens,
        "finite_logprob_values": len(finite_values),
        "minimum_logprob": min(finite_values),
        "maximum_logprob": max(finite_values),
        "token_sequence_sha256": hashlib.sha256(
            json.dumps(tokens, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def validate_chat_answer(body: dict[str, Any]) -> dict[str, Any]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise BenchmarkError("chat response has no first choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise BenchmarkError("chat response has no message")
    fields = {
        name: value
        for name in ("reasoning", "reasoning_content", "content")
        if isinstance((value := message.get(name)), str)
    }
    combined = "\n".join(fields.values())
    if not EXPECTED_ANSWER.search(combined):
        raise BenchmarkError(
            "semantic canary did not contain the expected integer 42; "
            f"message={json.dumps(fields, ensure_ascii=False)[:500]}"
        )
    usage = body.get("usage")
    if not isinstance(usage, dict):
        raise BenchmarkError("chat response omitted usage")
    return {
        "answer_present": True,
        "finish_reason": choices[0].get("finish_reason"),
        "prompt_tokens": _positive_int(usage.get("prompt_tokens"), "chat prompt_tokens"),
        "completion_tokens": _positive_int(
            usage.get("completion_tokens"), "chat completion_tokens"
        ),
        "response_sha256": hashlib.sha256(combined.encode("utf-8")).hexdigest(),
        "message_fields": sorted(fields),
    }


def _stream_completion(url: str, payload: dict[str, Any], timeout: float) -> StreamResult:
    body = dict(payload)
    body["stream"] = True
    body["stream_options"] = {
        "include_usage": True,
        "continuous_usage_stats": True,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    first_token_at: float | None = None
    last_token_at: float | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    text_parts: list[str] = []
    content_events = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if int(response.status) != 200:
                raise BenchmarkError(f"streaming completion returned HTTP {response.status}")
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                encoded = line[5:].strip()
                if encoded == "[DONE]":
                    break
                try:
                    event = json.loads(encoded)
                except json.JSONDecodeError as exc:
                    raise BenchmarkError(f"stream emitted malformed JSON: {encoded[:200]}") from exc
                candidate_usage = event.get("usage")
                if isinstance(candidate_usage, dict):
                    usage = candidate_usage
                choices = event.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])
                token_text = choice.get("text")
                if not isinstance(token_text, str) or not token_text:
                    continue
                now = time.monotonic()
                if first_token_at is None:
                    first_token_at = now
                last_token_at = now
                content_events += 1
                text_parts.append(token_text)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise BenchmarkError(f"streaming completion returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise BenchmarkError(f"streaming completion failed: {exc}") from exc
    finished = time.monotonic()
    if first_token_at is None or last_token_at is None:
        raise BenchmarkError("streaming completion emitted no text tokens")
    if usage is None:
        raise BenchmarkError("streaming completion omitted final usage")
    return StreamResult(
        started=started,
        first_token_at=first_token_at,
        last_token_at=last_token_at,
        finished=finished,
        usage=usage,
        finish_reason=finish_reason,
        text="".join(text_parts),
        content_events=content_events,
    )


def stream_metrics(result: StreamResult, requested_decode_tokens: int) -> dict[str, Any]:
    prompt_tokens = _positive_int(result.usage.get("prompt_tokens"), "prompt_tokens")
    completion_tokens = _positive_int(
        result.usage.get("completion_tokens"), "completion_tokens"
    )
    if completion_tokens != requested_decode_tokens:
        raise BenchmarkError(
            f"stream returned {completion_tokens} completion tokens, "
            f"expected {requested_decode_tokens} with ignore_eos=true"
        )
    ttft = result.first_token_at - result.started
    decode_window = result.last_token_at - result.first_token_at
    total = result.finished - result.started
    if ttft <= 0 or decode_window <= 0 or total <= 0:
        raise BenchmarkError(
            f"invalid timings: ttft={ttft}, decode_window={decode_window}, total={total}"
        )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "time_to_first_token_seconds": ttft,
        "client_prompt_tokens_per_ttft_second": prompt_tokens / ttft,
        "decode_window_seconds": decode_window,
        "inter_token_decode_tokens_per_second": (completion_tokens - 1) / decode_window,
        "request_wall_seconds": total,
        "end_to_end_completion_tokens_per_second": completion_tokens / total,
        "content_events": result.content_events,
        "finish_reason": result.finish_reason,
        "response_sha256": hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
    }


def _tokenize_exact_prompt(
    base_url: str,
    model: str,
    target_tokens: int,
    nonce: str,
    timeout: float,
) -> tuple[list[int], str]:
    sentence = (
        "Each numbered observation describes a stable synthetic benchmark fact: "
        "the copper ring remains connected, the measured value is recorded, and "
        "the next observation follows in numerical order. "
    )
    text = f"Unique benchmark nonce {nonce}.\n" + "".join(
        f"Observation {index}: {sentence}" for index in range(target_tokens // 12 + 32)
    )
    status, body, _ = _request_json(
        f"{base_url}/tokenize",
        payload={"model": model, "prompt": text, "add_special_tokens": False},
        timeout=timeout,
    )
    document = _require_success(status, body, "tokenize benchmark prompt")
    tokens = document.get("tokens")
    if not isinstance(tokens, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in tokens
    ):
        raise BenchmarkError("tokenize response did not contain integer token IDs")
    if len(tokens) < target_tokens:
        raise BenchmarkError(
            f"constructed prompt has only {len(tokens)} tokens, target is {target_tokens}"
        )
    selected = tokens[:target_tokens]
    digest = hashlib.sha256(
        json.dumps(selected, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return selected, digest


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.base_url.rstrip("/")
    status, models_body, _ = _request_json(
        f"{base_url}/v1/models", timeout=args.readiness_timeout
    )
    models = _require_success(status, models_body, "model discovery").get("data")
    if not isinstance(models, list) or not models or not isinstance(models[0], dict):
        raise BenchmarkError("model discovery returned no models")
    model = args.model or models[0].get("id")
    if not isinstance(model, str) or not model:
        raise BenchmarkError("model discovery did not provide a usable model ID")

    chat_payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a calculator. Reply with only the integer answer.",
            },
            {"role": "user", "content": "What is 17 + 25?"},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": args.seed,
        "max_tokens": args.chat_max_tokens,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    status, chat_body, chat_seconds = _request_json(
        f"{base_url}/v1/chat/completions",
        payload=chat_payload,
        timeout=args.timeout,
    )
    chat_document = _require_success(status, chat_body, "semantic warmup canary")
    chat_result = validate_chat_answer(chat_document)
    chat_result["wall_seconds"] = chat_seconds

    logprob_payload = {
        "model": model,
        "prompt": "Continue the even integers: 2, 4, 6, 8,",
        "max_tokens": args.logprob_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": args.seed,
        "ignore_eos": True,
        "logprobs": 5,
    }
    status, logprob_body, logprob_seconds = _request_json(
        f"{base_url}/v1/completions",
        payload=logprob_payload,
        timeout=args.timeout,
    )
    logprob_document = _require_success(status, logprob_body, "finite-logprobs warmup")
    usage = logprob_document.get("usage")
    if not isinstance(usage, dict):
        raise BenchmarkError("finite-logprobs warmup omitted usage")
    completion_tokens = _positive_int(
        usage.get("completion_tokens"), "logprob completion_tokens"
    )
    if completion_tokens != args.logprob_tokens:
        raise BenchmarkError(
            f"finite-logprobs warmup returned {completion_tokens} tokens, "
            f"expected {args.logprob_tokens}"
        )
    logprob_result = validate_completion_logprobs(logprob_document, completion_tokens)
    logprob_result["wall_seconds"] = logprob_seconds

    nonce = args.nonce or uuid.uuid4().hex
    shape_warmup_ids, shape_warmup_sha256 = _tokenize_exact_prompt(
        base_url,
        model,
        args.prompt_tokens,
        f"{nonce}-shape-warmup",
        args.timeout,
    )
    shape_warmup_payload = {
        "model": model,
        "prompt": shape_warmup_ids,
        "add_special_tokens": False,
        "max_tokens": args.shape_warmup_decode_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": args.seed,
        "ignore_eos": True,
    }
    status, shape_warmup_body, shape_warmup_seconds = _request_json(
        f"{base_url}/v1/completions",
        payload=shape_warmup_payload,
        timeout=args.timeout,
    )
    shape_warmup_document = _require_success(
        status, shape_warmup_body, "exact-shape prefill warmup"
    )
    shape_warmup_usage = shape_warmup_document.get("usage")
    if not isinstance(shape_warmup_usage, dict):
        raise BenchmarkError("exact-shape prefill warmup omitted usage")
    shape_warmup_prompt_tokens = _positive_int(
        shape_warmup_usage.get("prompt_tokens"), "shape warmup prompt_tokens"
    )
    shape_warmup_completion_tokens = _positive_int(
        shape_warmup_usage.get("completion_tokens"), "shape warmup completion_tokens"
    )
    if shape_warmup_prompt_tokens != args.prompt_tokens:
        raise BenchmarkError(
            f"exact-shape warmup reported {shape_warmup_prompt_tokens} prompt tokens, "
            f"expected {args.prompt_tokens}"
        )
    if shape_warmup_completion_tokens != args.shape_warmup_decode_tokens:
        raise BenchmarkError(
            "exact-shape warmup returned "
            f"{shape_warmup_completion_tokens} completion tokens, expected "
            f"{args.shape_warmup_decode_tokens}"
        )

    prompt_ids, prompt_sha256 = _tokenize_exact_prompt(
        base_url,
        model,
        args.prompt_tokens,
        f"{nonce}-measurement",
        args.timeout,
    )
    measured_payload = {
        "model": model,
        "prompt": prompt_ids,
        "add_special_tokens": False,
        "max_tokens": args.decode_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": args.seed,
        "ignore_eos": True,
    }
    stream = _stream_completion(
        f"{base_url}/v1/completions", measured_payload, args.timeout
    )
    measured = stream_metrics(stream, args.decode_tokens)
    if measured["prompt_tokens"] != args.prompt_tokens:
        raise BenchmarkError(
            f"measured usage reported {measured['prompt_tokens']} prompt tokens, "
            f"expected the exact token-ID prompt length {args.prompt_tokens}"
        )

    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "lane": "public-functional",
            "maturity": "diagnostic",
            "statement": (
                "One warm endpoint invocation. Rates are client-side diagnostics, "
                "not acceptance or reference-lane results."
            ),
        },
        "endpoint": base_url,
        "model": model,
        "seed": args.seed,
        "warmup": {
            "semantic_chat": chat_result,
            "finite_completion_logprobs": logprob_result,
            "exact_shape_prefill": {
                "prompt_tokens": shape_warmup_prompt_tokens,
                "completion_tokens": shape_warmup_completion_tokens,
                "wall_seconds": shape_warmup_seconds,
                "prompt_token_ids_sha256": shape_warmup_sha256,
            },
        },
        "measurement": {
            "requested_prompt_tokens": args.prompt_tokens,
            "requested_decode_tokens": args.decode_tokens,
            "unique_prompt_nonce": nonce,
            "prompt_token_ids_sha256": prompt_sha256,
            **measured,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--logprob-tokens", type=int, default=16)
    parser.add_argument("--chat-max-tokens", type=int, default=128)
    parser.add_argument("--shape-warmup-decode-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--nonce")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--readiness-timeout", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in (
        "prompt_tokens",
        "decode_tokens",
        "logprob_tokens",
        "chat_max_tokens",
        "shape_warmup_decode_tokens",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
    except BenchmarkError as exc:
        print(f"endpoint-benchmark: FAIL: {exc}")
        return 2
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
