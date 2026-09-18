"""Measure client TTFT for exact-length Qwen prompts with zero cache credit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
import uuid

import httpx

COUNTS = (8192, 65536, 131072)
REPETITIONS = 3


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def prepare_fixture(client, model, count, remaining, sample):
    """Adjust identical archive text until the server returns the exact token IDs."""
    nonce = uuid.uuid4().hex
    expected = "CACHE-" + uuid.uuid4().hex[:10].upper()
    repeats, padding = max(1, count // 4), 0
    attempts = sample.setdefault("tokenization_attempts", [])
    for _ in range(16):
        require(remaining() > 0, "Fixture preparation budget exceeded")
        text = (nonce + "\nVerification key: " + expected + ".\n"
                + " ordinary archive record." * repeats + " a" * padding
                + "\nReturn only the verification key from the beginning.")
        body = dict(model=model, messages=[dict(role="user", content=text)],
                    temperature=0, seed=779386, max_tokens=128,
                    chat_template_kwargs={"enable_thinking": False})
        payload = {key: body[key] for key in ("model", "messages", "chat_template_kwargs")}
        payload["add_generation_prompt"] = True
        attempt = {"request": payload, "started_at": utc()}
        attempts.append(attempt)
        response = client.post("/tokenize", json=payload, timeout=min(30, remaining()))
        attempt.update(http_status=response.status_code, raw_response=response.text)
        response.raise_for_status()
        tokens = response.json().get("tokens")
        require(isinstance(tokens, list) and tokens
                and all(type(token) is int and token >= 0 for token in tokens),
                "Tokenizer returned no exact token IDs")
        delta = count - len(tokens)
        if delta == 0:
            return dict(count=count, expected=expected, request=body,
                        request_sha256=checksum(body), token_ids_sha256=checksum(tokens),
                        prefix=tokens[:64])
        if abs(delta) > 16:
            repeats, padding = max(1, repeats + delta // 4), 0
        else:
            padding = max(0, padding + delta)
            if delta < 0 and padding == 0:
                repeats = max(1, repeats - 1)
    raise ValueError("Could not build exact-length bounded chat fixture")


def validate_usage(usage, count):
    require(type(usage.get("prompt_tokens")) is int and usage["prompt_tokens"] == count,
            "Exact token count differs from response")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    require(type(cached) is int and cached == 0,
            "Cold prefill requires explicit zero cached-token credit")
    require(type(usage.get("completion_tokens")) is int and usage["completion_tokens"] > 0,
            "Missing completion usage")


def measure(client, model, report, checkpoint=lambda: None):
    """Run serial 8K/64K/128K triples; retain partial stream evidence on errors."""
    for repetition in range(REPETITIONS):
        for count in COUNTS:
            deadline = time.monotonic() + 300
            sample = dict(repetition=repetition, target_tokens=count, passed=False)
            report["samples"].append(sample)
            checkpoint()
            prepared = prepare_fixture(client, model, count,
                                       lambda: deadline - time.monotonic(), sample)
            body = json.loads(json.dumps(prepared["request"]))
            body.update(max_tokens=1, stream=True, stream_options={"include_usage": True})
            sample.update(fixture=prepared, streamed_request=body, started_at=utc(),
                          events=[], raw_stream_lines=[], usage=None)
            checkpoint()
            started = time.perf_counter()
            first_token = None
            try:
                with client.stream("POST", "/v1/chat/completions", json=body, timeout=300) as response:
                    sample["http_status"] = response.status_code
                    if response.is_error:
                        response.read()
                        sample["raw_response"] = response.text
                    response.raise_for_status()
                    for line in response.iter_lines():
                        sample["raw_stream_lines"].append(line)
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        event = json.loads(line[6:])
                        sample["events"].append(event)
                        require(not event.get("error"), "Endpoint reported a streaming error")
                        if event.get("usage"):
                            sample["usage"] = event["usage"]
                        for choice in event.get("choices", []):
                            delta = choice.get("delta") or {}
                            if first_token is None and (delta.get("content") or delta.get("reasoning_content")):
                                first_token = time.perf_counter() - started
            finally:
                sample["elapsed_seconds"] = time.perf_counter() - started
                if first_token is not None:
                    sample["ttft_seconds"] = first_token
            require(first_token is not None and first_token > 0, "No streamed token received")
            validate_usage(sample["usage"] or {}, count)
            sample.update(tok_per_sec=count / first_token, passed=True)
            checkpoint()


def run(client, model, output, base_url):
    """Own one output path and persist partial evidence before propagating failure."""
    output = Path(output)
    report = dict(schema="sparkring-exact-cold-prefill/v1", status="running",
                  base_url=base_url, model=model, counts=list(COUNTS), repetitions=REPETITIONS,
                  started_at=utc(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  methodology="Serial exact chat-token count divided by client first-token latency; zero cache credit required.",
                  samples=[])
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    def save():
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        measure(client, model, report, save)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["ended_at"] = utc()
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Server root URL, without /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Absent JSON output file")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    require(args.run, "Explicit --run required; this sends nine inference requests")
    endpoint = urlsplit(args.base_url)
    require(endpoint.scheme in ("http", "https") and endpoint.hostname
            and not endpoint.username and not endpoint.password and not endpoint.query
            and not endpoint.fragment and endpoint.path in ("", "/"),
            "Use an HTTP(S) server root without credentials, query or path")
    headers = {}
    if os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
    with httpx.Client(base_url=args.base_url.rstrip("/"), headers=headers, trust_env=False,
                      follow_redirects=False, timeout=300) as client:
        run(client, args.model, args.output, args.base_url.rstrip("/"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
