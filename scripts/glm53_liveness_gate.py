#!/usr/bin/env python3
"""Exercise concurrent GLM requests and require idle KV ownership to recover."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
import urllib.request
import uuid
from pathlib import Path


def _metric_sum(text: str, name: str, *, required: bool = True) -> float:
    matches = re.findall(
        rf"(?m)^{re.escape(name)}(?:\{{[^\n]*\}})?\s+([0-9.eE+-]+)$",
        text,
    )
    if not matches and required:
        raise RuntimeError(f"metrics response does not contain {name}")
    return sum(float(value) for value in matches)


def parse_metrics(text: str) -> dict[str, float]:
    return {
        "running": _metric_sum(text, "vllm:num_requests_running"),
        "waiting": _metric_sum(text, "vllm:num_requests_waiting"),
        "kv_usage": _metric_sum(text, "vllm:kv_cache_usage_perc"),
        "capture_delayed": _metric_sum(
            text,
            "vllm:sparkcache_capture_delayed_requests",
            required=False,
        ),
        "capture_pages": _metric_sum(
            text,
            "vllm:sparkcache_capture_retained_manager_pages",
            required=False,
        ),
        "capture_uncertain": _metric_sum(
            text,
            "vllm:sparkcache_capture_ownership_uncertain_ranks",
            required=False,
        ),
    }


def chat_payload(
    model: str,
    nonce: str,
    prompt_words: int,
    max_tokens: int,
) -> dict[str, object]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"{nonce} " + "cache " * prompt_words + "Reply OK.",
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def idle_satisfied(
    observed: dict[str, float],
    baseline: dict[str, float],
    *,
    kv_tolerance: float,
) -> bool:
    return (
        observed["running"] == 0
        and observed["waiting"] == 0
        and observed["capture_delayed"] == 0
        and observed["capture_pages"] == 0
        and observed["capture_uncertain"] == 0
        and observed["kv_usage"] <= baseline["kv_usage"] + kv_tolerance
    )


class Client:
    def __init__(self, endpoint: str, credential: str | None) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.credential = credential

    def _request(
        self,
        path: str,
        *,
        payload: dict[str, object] | None = None,
        timeout: float = 600,
    ) -> bytes:
        headers = {}
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode()
        if self.credential:
            headers["Authorization"] = f"Bearer {self.credential}"
        request = urllib.request.Request(
            f"{self.endpoint}{path}",
            data=data,
            headers=headers,
            method="POST" if data is not None else "GET",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    def metrics(self) -> dict[str, float]:
        return parse_metrics(self._request("/metrics", timeout=10).decode())

    def chat(self, payload: dict[str, object]) -> float:
        started = time.monotonic()
        self._request("/v1/chat/completions", payload=payload)
        return time.monotonic() - started


def _credential(path: Path | None) -> str | None:
    if path is None:
        return None
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value]
    if not values:
        raise RuntimeError("API key file does not contain a key")
    return values[0]


def run(args: argparse.Namespace) -> dict[str, object]:
    client = Client(args.endpoint, _credential(args.api_key_file))
    baseline = client.metrics()
    started = time.monotonic()
    cycles = []
    while len(cycles) < args.cycles or (
        args.duration_seconds > 0
        and time.monotonic() - started < args.duration_seconds
    ):
        cycle = len(cycles) + 1
        nonce = f"sparkring-liveness-{uuid.uuid4().hex}"
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as executor:
            latencies = list(
                executor.map(
                    lambda index: client.chat(
                        chat_payload(
                            args.model,
                            f"{nonce}-{index}",
                            args.prompt_words,
                            args.max_tokens,
                        )
                    ),
                    range(args.concurrency),
                )
            )
        deadline = time.monotonic() + args.drain_timeout_seconds
        while True:
            observed = client.metrics()
            if idle_satisfied(
                observed,
                baseline,
                kv_tolerance=args.kv_tolerance,
            ):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "scheduler did not return to its idle KV baseline: "
                    + json.dumps(observed, sort_keys=True)
                )
            time.sleep(0.5)
        receipt = {
            "cycle": cycle,
            "request_seconds": [round(value, 3) for value in latencies],
            "idle": observed,
        }
        cycles.append(receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
    return {
        "schema": "sparkring-glm53-liveness-gate/v1",
        "status": "passed",
        "endpoint": args.endpoint,
        "model": args.model,
        "concurrency": args.concurrency,
        "prompt_words": args.prompt_words,
        "baseline": baseline,
        "cycles": cycles,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--prompt-words", type=int, default=100000)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--duration-seconds", type=float, default=0)
    parser.add_argument("--drain-timeout-seconds", type=float, default=120)
    parser.add_argument("--kv-tolerance", type=float, default=0.005)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in ("concurrency", "prompt_words", "max_tokens", "cycles"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    result = run(args)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
