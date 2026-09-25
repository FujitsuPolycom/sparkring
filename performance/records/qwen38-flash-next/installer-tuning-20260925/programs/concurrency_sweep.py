#!/usr/bin/env python3
"""Measure steady-state speculative decode step time at each concurrency.

Usage: concurrency_sweep.py BASE_URL OUTPUT_JSON [CONCURRENCIES] [SHORT,LONG]
BASE_URL ends in /v1. CONCURRENCIES is a comma list (default 1-16). At
concurrency C, C identical greedy requests run together with ignore_eos, so
every request drafts and accepts alike and each step verifies (drafts + 1) * C
rows. Each concurrency runs twice, generating SHORT and LONG tokens per request
(default 256 and 768). Steps are counted from vLLM's draft counter, which
records one draft per running request per step. The steady-state step time is
(wall_long - wall_short) / (steps_long - steps_short); the difference cancels
request admission, prefill, ramp-up and drain. Also reported: the long run's
aggregate tokens/s and tokens per request step, and its whole-run step time.
"""
import json
import re
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1].rstrip("/")
OUTPUT = sys.argv[2]
LEVELS = [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else ",".join(map(str, range(1, 17)))).split(",")]
SHORT, LONG = (int(v) for v in (sys.argv[4] if len(sys.argv) > 4 else "256,768").split(","))
ROOT = BASE.removesuffix("/v1")
PROMPT = ("Write a complete Python module that implements a thread-safe LRU cache with a maximum size, "
          "per-entry time-to-live, hit and miss statistics, type hints and docstrings. Include unit tests.\n\n")
model = json.loads(urllib.request.urlopen(BASE + "/models", timeout=30).read())["data"][0]["id"]


def counters():
    text = urllib.request.urlopen(ROOT + "/metrics", timeout=30).read().decode()
    totals = {}
    for line in text.splitlines():
        match = re.match(r"(vllm:spec_decode_num_(?:accepted_tokens|drafts)_total)\{[^}]*\} ([0-9.e+]+)", line)
        if match:
            totals[match.group(1)] = totals.get(match.group(1), 0.0) + float(match.group(2))
    return totals


def complete(results, index, tokens):
    body = json.dumps({"model": model, "prompt": PROMPT, "max_tokens": tokens, "temperature": 0,
                       "ignore_eos": True}).encode()
    request = urllib.request.Request(BASE + "/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        results[index] = json.loads(response.read())["usage"]["completion_tokens"]


def run(concurrency, tokens):
    results = [0] * concurrency
    before = counters()
    start = time.perf_counter()
    threads = [threading.Thread(target=complete, args=(results, i, tokens)) for i in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - start
    after = counters()
    drafts = after["vllm:spec_decode_num_drafts_total"] - before["vllm:spec_decode_num_drafts_total"]
    accepted = (after["vllm:spec_decode_num_accepted_tokens_total"]
                - before["vllm:spec_decode_num_accepted_tokens_total"])
    return {"wall_s": wall, "steps": drafts / concurrency, "tokens": sum(results),
            "tokens_per_request_step": 1 + accepted / drafts if drafts else None}


run(1, 64)  # warm-up, discarded
rows = []
for level in LEVELS:
    short, long = run(level, SHORT), run(level, LONG)
    row = {"concurrency": level,
           "steady_step_ms": round(1000 * (long["wall_s"] - short["wall_s"]) / (long["steps"] - short["steps"]), 2),
           "whole_run_step_ms": round(1000 * long["wall_s"] / long["steps"], 2),
           "aggregate_tok_s": round(long["tokens"] / long["wall_s"], 1),
           "tokens_per_request_step": round(long["tokens_per_request_step"], 3)}
    rows.append(row)
    print(json.dumps(row), flush=True)
with open(OUTPUT, "w") as handle:
    json.dump({"model": model, "tokens_per_request": [SHORT, LONG], "rows": rows}, handle, indent=1)
