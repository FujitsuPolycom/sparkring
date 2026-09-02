#!/usr/bin/env python3
"""Compile DFlash request-batch shapes before a service is declared ready."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
import time
import urllib.error
import urllib.request


def wait_for_api(endpoint: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = endpoint.rstrip("/") + "/v1/models"
    while True:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("vLLM API did not become ready for DFlash warmup")
        time.sleep(1)


def send_warmup_request(
    endpoint: str,
    model: str,
    nonce: str,
    max_tokens: int,
    timeout_seconds: float,
    prompt_words: int,
    api_key: str | None = None,
) -> None:
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "warmup " * prompt_words
                    + f"\nDFlash warmup {nonce}. Reply briefly."
                ),
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        result = json.load(response)
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("DFlash warmup response has no completion")


def run_warmup(
    endpoint: str,
    model: str,
    concurrencies: tuple[int, ...],
    max_tokens: int,
    timeout_seconds: float,
    shape_words: tuple[int, ...],
    api_key: str | None = None,
) -> tuple[dict[str, float | int], ...]:
    results: list[dict[str, float | int]] = []

    def run_batch(concurrency: int, prompt_words: int) -> None:
        barrier = threading.Barrier(concurrency)

        def run(index: int) -> None:
            barrier.wait(timeout=timeout_seconds)
            send_warmup_request(
                endpoint,
                model,
                f"c{concurrency}-{index}",
                max_tokens,
                timeout_seconds,
                prompt_words,
                api_key,
            )

        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency
        ) as executor:
            tuple(executor.map(run, range(concurrency)))
        results.append(
            {
                "concurrency": concurrency,
                "prompt_words": prompt_words,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )

    for concurrency in concurrencies:
        run_batch(concurrency, shape_words[0])
    for prompt_words in shape_words[1:]:
        run_batch(min(2, max(concurrencies)), prompt_words)
    return tuple(results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--concurrencies", default="1,2,4,8,16")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--shape-words", default="8,24,56,120,248")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--api-key")
    args = parser.parse_args()
    try:
        concurrencies = tuple(
            int(value) for value in args.concurrencies.split(",")
        )
        shape_words = tuple(int(value) for value in args.shape_words.split(","))
    except ValueError as error:
        raise SystemExit("warmup concurrencies must be comma-separated integers") from error
    if (
        not concurrencies
        or not shape_words
        or any(value <= 0 for value in concurrencies)
        or any(value <= 0 for value in shape_words)
        or tuple(sorted(set(concurrencies))) != concurrencies
        or tuple(sorted(set(shape_words))) != shape_words
        or args.max_tokens <= 0
    ):
        raise SystemExit("warmup concurrencies and max tokens must be positive")
    wait_for_api(args.endpoint, args.timeout_seconds)
    result = run_warmup(
        args.endpoint,
        args.model,
        concurrencies,
        args.max_tokens,
        args.timeout_seconds,
        shape_words,
        args.api_key,
    )
    print(json.dumps({"dflash_warmup": result}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
