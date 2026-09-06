#!/usr/bin/env python3
"""Exercise speculative request shapes and sampling before readiness."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request


def resolve_temperature(value: float | str | None = None) -> float:
    """Resolve a finite warmup temperature while preserving the greedy default."""
    if value is None:
        value = os.environ.get("SPARKRING_WARMUP_TEMPERATURE", "0")
    if isinstance(value, bool):
        raise ValueError("Warmup temperature must be a finite number from 0 to 2")
    temperature = float(value)
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("Warmup temperature must be a finite number from 0 to 2")
    return temperature


def wait_for_api(
    endpoint: str,
    timeout_seconds: float,
    credential: str | None = None,
) -> None:
    """Block until ``/v1/models`` answers 200, sending the bearer key if given.

    When vLLM runs with ``--api-key`` the model list requires authentication,
    so an anonymous probe receives 401 until the deadline and readiness never
    completes. The same credential the warmup requests use is sent here.
    """

    deadline = time.monotonic() + timeout_seconds
    url = endpoint.rstrip("/") + "/v1/models"
    headers = {"Authorization": f"Bearer {credential}"} if credential else {}
    while True:
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=3) as response:
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
    credential: str | None = None,
    temperature: float | None = None,
) -> None:
    temperature = resolve_temperature(temperature)
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
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if credential:
        headers["Authorization"] = f"Bearer {credential}"
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
    credential: str | None = None,
    temperature: float | None = None,
) -> tuple[dict[str, float | int], ...]:
    temperature = resolve_temperature(temperature)
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
                credential,
                temperature,
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
                "temperature": temperature,
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
    parser.add_argument(
        "--concurrencies",
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16",
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--shape-words", default="8,24,56,120,248")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--api-key")
    parser.add_argument("--temperature", type=resolve_temperature, default=None)
    args = parser.parse_args()
    try:
        temperature = resolve_temperature(args.temperature)
    except ValueError as error:
        raise SystemExit(str(error)) from error
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
    wait_for_api(args.endpoint, args.timeout_seconds, args.api_key)
    result = run_warmup(
        args.endpoint,
        args.model,
        concurrencies,
        args.max_tokens,
        args.timeout_seconds,
        shape_words,
        args.api_key,
        temperature,
    )
    print(json.dumps({"dflash_warmup": result}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
