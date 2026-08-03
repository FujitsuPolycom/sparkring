#!/usr/bin/env python3
"""Small, non-destructive readiness/correctness/performance gate for EXL3."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_PROMPT = (
    "Write a Python function named stable_unique that preserves input order, "
    "then explain its time complexity in one sentence."
)


def request_json(
    url: str,
    *,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"request failed for {url}: {error}") from error
    if not isinstance(result, dict):
        raise RuntimeError(f"expected JSON object from {url}")
    return result


def require_http_ok(url: str, *, timeout: float) -> None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"request failed for {url}: {error}") from error


def completion_payload(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 20260731,
        "max_tokens": max_tokens,
    }


@dataclass(frozen=True)
class Completion:
    text: str
    tokens: int


def one_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> Completion:
    result = request_json(
        f"{base_url.rstrip('/')}/v1/completions",
        timeout=timeout,
        payload=completion_payload(model, prompt, max_tokens),
    )
    choices = result.get("choices")
    usage = result.get("usage")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("completion response must contain exactly one choice")
    text = choices[0].get("text") if isinstance(choices[0], dict) else None
    tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if not isinstance(text, str) or not text:
        raise RuntimeError("completion response text is empty")
    if not isinstance(tokens, int) or tokens <= 0:
        raise RuntimeError("completion response has no positive completion token count")
    return Completion(text=text, tokens=tokens)


def performance_cell(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    concurrency: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                one_completion,
                base_url,
                model,
                f"{prompt}\nRequest lane: {index}.",
                max_tokens,
                timeout,
            )
            for index in range(concurrency)
        ]
        completions = [future.result() for future in futures]
    elapsed = time.perf_counter() - started
    tokens = sum(item.tokens for item in completions)
    return {
        "concurrency": concurrency,
        "completion_tokens": tokens,
        "elapsed_seconds": elapsed,
        "aggregate_tokens_per_second": tokens / elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--min-c1-tps", type=float, required=True)
    parser.add_argument("--min-c2-tps", type=float, required=True)
    parser.add_argument("--min-c8-tps", type=float, required=True)
    args = parser.parse_args(argv)

    base = args.base_url.rstrip("/")
    require_http_ok(f"{base}/health", timeout=args.timeout_seconds)
    models = request_json(f"{base}/v1/models", timeout=args.timeout_seconds)
    model_ids = {
        entry.get("id")
        for entry in models.get("data", [])
        if isinstance(entry, dict)
    }
    if args.model not in model_ids:
        raise RuntimeError(f"served model {args.model!r} not present in {sorted(model_ids)!r}")

    first = one_completion(
        base, args.model, args.prompt, args.max_tokens, args.timeout_seconds
    )
    second = one_completion(
        base, args.model, args.prompt, args.max_tokens, args.timeout_seconds
    )
    if first != second:
        raise RuntimeError("two greedy fixed-seed completions were not identical")

    cells = [
        performance_cell(
            base,
            args.model,
            args.prompt,
            args.max_tokens,
            args.timeout_seconds,
            concurrency,
        )
        for concurrency in (1, 2, 8)
    ]
    thresholds = {
        1: args.min_c1_tps,
        2: args.min_c2_tps,
        8: args.min_c8_tps,
    }
    failures = [
        f"C{cell['concurrency']} {cell['aggregate_tokens_per_second']:.2f} < {thresholds[cell['concurrency']]:.2f} tok/s"
        for cell in cells
        if cell["aggregate_tokens_per_second"] < thresholds[cell["concurrency"]]
    ]
    require_http_ok(f"{base}/health", timeout=args.timeout_seconds)
    report = {
        "schema": "sparkring-exl3-live-gate/v1",
        "status": "pass" if not failures else "fail",
        "model": args.model,
        "deterministic_output_sha256": hashlib.sha256(
            first.text.encode("utf-8")
        ).hexdigest(),
        "deterministic_completion_tokens": first.tokens,
        "cells": cells,
        "thresholds": {f"C{key}": value for key, value in thresholds.items()},
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 5


if __name__ == "__main__":
    raise SystemExit(main())
