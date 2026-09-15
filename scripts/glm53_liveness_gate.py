#!/usr/bin/env python3
"""Exercise concurrent GLM requests and check scheduler/KV recovery.

Use --require-capture-metrics for SparkCache ownership checks. Without it,
missing capture metrics do not establish that cache ownership was released.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import time
import urllib.request
import uuid
from pathlib import Path


def _metric_sum(text: str, name: str, *, required: bool = True) -> float:
    matches = re.findall(
        rf"(?m)^{re.escape(name)}(?:\{{[^\n]*\}})?\s+(\S+)\s*$",
        text,
    )
    if not matches and required:
        raise RuntimeError(f"metrics response does not contain {name}")
    values = [float(value) for value in matches]
    total = sum(values)
    if any(not math.isfinite(value) or value < 0 for value in values) or not math.isfinite(total):
        raise RuntimeError(f"metric {name} must contain finite nonnegative values")
    return total


def parse_metrics(text: str, *, require_capture_metrics: bool = False) -> dict[str, float]:
    return {
        "running": _metric_sum(text, "vllm:num_requests_running"),
        "waiting": _metric_sum(text, "vllm:num_requests_waiting"),
        "kv_usage": _metric_sum(text, "vllm:kv_cache_usage_perc"),
        "capture_delayed": _metric_sum(
            text,
            "vllm:sparkcache_capture_delayed_requests",
            required=require_capture_metrics,
        ),
        "capture_pages": _metric_sum(
            text,
            "vllm:sparkcache_capture_retained_manager_pages",
            required=require_capture_metrics,
        ),
        "capture_uncertain": _metric_sum(
            text,
            "vllm:sparkcache_capture_ownership_uncertain_ranks",
            required=require_capture_metrics,
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
    def __init__(self, endpoint: str, credential: str | None, *, require_capture_metrics=False) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.credential = credential
        self.require_capture_metrics = require_capture_metrics

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

    def metrics(self, timeout=10) -> dict[str, float]:
        return parse_metrics(self._request("/metrics", timeout=timeout).decode(),
                             require_capture_metrics=self.require_capture_metrics)

    def chat(self, payload: dict[str, object]) -> float:
        started = time.monotonic()
        document = json.loads(self._request("/v1/chat/completions", payload=payload))
        choices = document.get("choices") if isinstance(document, dict) else None
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise RuntimeError("Chat did not return one completed choice")
        choice = choices[0]
        usage = document.get("usage", {})
        if (not isinstance(choice.get("message"), dict)
                or choice["message"].get("role") != "assistant"
                or choice.get("finish_reason") not in ("stop", "length")
                or not isinstance(usage, dict)
                or type(usage.get("completion_tokens")) is not int
                or usage["completion_tokens"] < 1):
            raise RuntimeError("Chat lacks normal completion and output-token usage")
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
    for name, positive in (("duration_seconds", False), ("drain_timeout_seconds", True), ("kv_tolerance", False)):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    require_capture = getattr(args, "require_capture_metrics", False)
    client = Client(args.endpoint, _credential(args.api_key_file), require_capture_metrics=require_capture)
    baseline = client.metrics()
    if not idle_satisfied(baseline, baseline, kv_tolerance=args.kv_tolerance):
        raise RuntimeError("Liveness baseline must be idle before submitting requests")
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
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Idle drain deadline exceeded")
            observed = client.metrics(timeout=min(10, remaining))
            if time.monotonic() > deadline:
                raise RuntimeError("Idle drain deadline exceeded")
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
        "capture_metrics_required": require_capture,
        "scope": "scheduler and KV recovery; capture ownership requires --require-capture-metrics",
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
    parser.add_argument("--require-capture-metrics", action="store_true",
                        help="require SparkCache ownership metrics; use for cache-enabled profiles")
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
