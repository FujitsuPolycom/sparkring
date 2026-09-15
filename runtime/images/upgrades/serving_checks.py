"""Bounded text, cache-credit and benchmark evidence for serving qualification."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import re
import time
import urllib.request

from .contracts import require, sha


def chat(base, model, messages, *, max_tokens=128, timeout=180):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 41,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(4 * 1024**2 + 1)
    require(len(raw) <= 4 * 1024**2, "Serving response exceeds evidence budget")
    result = json.loads(raw)
    require(result.get("choices"), "Serving response has no completion")
    message = result["choices"][0].get("message", {})
    content = message.get("content")
    require(isinstance(content, str) and content.strip(), "Serving response is empty")
    return {
        "elapsed_seconds": time.monotonic() - started,
        "content": content,
        "usage": result.get("usage", {}),
        "finish_reason": result["choices"][0].get("finish_reason"),
        "request_sha256": sha(json.dumps(payload, sort_keys=True).encode()),
    }


def needle_fixture(seed):
    rng = random.Random(seed)
    code = "SR-" + "".join(
        rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(14)
    )
    key = "record-" + "".join(rng.choice("abcdef0123456789") for _ in range(12))
    rows = []
    for index in range(1024):
        value = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(36))
        rows.append(f"Archive item {index:04d}: {value}; status stored.\n")
    rows[517] = f"The access code for {key} is {code}.\n"
    prompt = (
        "Read the archive and answer the lookup question.\n"
        + "".join(rows)
        + f"\nWhat is the access code for {key}? Reply with the code only."
    )
    return {
        "messages": [{"role": "user", "content": prompt}],
        "expected": code,
        "sha256": sha(prompt.encode()),
    }


def cached_tokens(result):
    usage = result.get("usage", {})
    value = usage.get("prompt_tokens_details", {}).get("cached_tokens")
    require(
        type(value) is int and value >= 0,
        "Server must expose exact cached prompt-token credit",
    )
    return value


def verify_needle(result, fixture, *, minimum_prompt=4096, minimum_cached=None):
    require(
        fixture["expected"] in result["content"],
        "Needle answer differs from the requested archive value",
    )
    count = result.get("usage", {}).get("prompt_tokens")
    require(
        type(count) is int and count >= minimum_prompt,
        "Cache fixture did not reach the intended prefix size",
    )
    if minimum_cached is not None:
        require(
            cached_tokens(result) >= minimum_cached,
            "Fresh worker did not credit the persisted prefix",
        )
    return result


def benchmark_measurements(paths, *, minimum_fill=0.95):
    """Read measured throughput, retaining queue/coverage warnings as evidence.

    The benchmark's capacity_limited badge can include transient cache-restore
    queueing. Admit rows by explicit readiness, actual fill and error checks;
    never infer missing throughput from a nominal concurrency or KV pool size.
    """
    metrics = {}
    warnings = []
    records = []
    for path in paths:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
        records.append({"path": str(path), "sha256": sha(raw)})
        for context, row in value.get("prefill", {}).items():
            measured = row.get("client_tok_per_sec")
            require(
                type(measured) in (int, float)
                and math.isfinite(measured)
                and measured > 0
                and row.get("samples", 0) > 0,
                "Missing measured prefill sample",
            )
            server = row.get("server_validation", {})
            require(
                not server.get("cached_tokens", 0),
                "Prefill scout received cached-token credit",
            )
            metrics.setdefault("prefill-" + str(context), []).append(measured)
        for row in value["results"]:
            require(
                not row.get("failure_reason")
                and not row.get("timeout_reason")
                and row.get("num_errors", 0) == 0
                and not row.get("loop_detected")
                and not row.get("warmup_timed_out"),
                "Benchmark cell failed its serving/readiness checks",
            )
            concurrency = row["concurrency"]
            require(
                row.get("effective_concurrency", 0) >= minimum_fill * concurrency,
                "Benchmark cell was underfilled",
            )
            key = f"decode-{row['context_tokens']}-c{concurrency}"
            for name, field in (
                (key, "aggregate_tps"),
                (key + "-steps", "server_steps_per_s"),
            ):
                measured = row.get(field)
                require(
                    type(measured) in (int, float)
                    and math.isfinite(measured)
                    and measured > 0,
                    "Missing finite benchmark measurement: " + name,
                )
                metrics.setdefault(name, []).append(measured)
            if row.get("capacity_limited") or row.get("underfilled"):
                warnings.append(
                    {
                        "path": str(path),
                        "cell": key,
                        "effective_concurrency": row["effective_concurrency"],
                        "queue_fraction": row.get("queue_fraction"),
                        "benchmark_underfilled_badge": row.get("underfilled", False),
                        "reason": "Capacity badge retained; measured fill/readiness passed.",
                    }
                )
    require(
        metrics and all(len(values) >= 3 for values in metrics.values()),
        "Performance qualification requires at least three samples per cell",
    )
    return {"measurements": metrics, "warnings": warnings, "artifacts": records}


def non_looping_text(value):
    tokens = re.findall(r"\w+", value.lower())
    require(len(tokens) >= 1, "Text completion is empty")
    require(
        not any(
            re.search(r"(" + re.escape(word) + r"){8,}", value.lower())
            for word in set(tokens)
            if len(word) > 1
        ),
        "Text completion contains sustained exact repetition",
    )
    return value
