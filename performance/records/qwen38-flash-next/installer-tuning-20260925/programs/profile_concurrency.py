#!/usr/bin/env python3
"""Profile one batch of C identical greedy requests on a vLLM server started with --profiler-config.

Usage: profile_concurrency.py BASE_URL C TOKENS
BASE_URL ends in /v1. Runs one unprofiled warm-up batch, then brackets one batch
with /start_profile and /stop_profile and prints the speculative steps inside
the window (draft-counter delta / C), so trace kernel time can be divided by steps.
"""
import json
import re
import sys
import threading
import time
import urllib.request

BASE, C, TOKENS = sys.argv[1].rstrip("/"), int(sys.argv[2]), int(sys.argv[3])
ROOT = BASE.removesuffix("/v1")
PROMPT = ("Write a complete Python module that implements a thread-safe LRU cache with a maximum size, "
          "per-entry time-to-live, hit and miss statistics, type hints and docstrings. Include unit tests.\n\n")
model = json.loads(urllib.request.urlopen(BASE + "/models", timeout=30).read())["data"][0]["id"]


def post(path, body=None):
    request = urllib.request.Request(ROOT + path if not path.startswith("/v1") else ROOT + path,
                                     data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return response.read()


def drafts():
    text = urllib.request.urlopen(ROOT + "/metrics", timeout=30).read().decode()
    return sum(float(m.group(1)) for m in re.finditer(r"vllm:spec_decode_num_drafts_total\{[^}]*\} ([0-9.e+]+)", text))


def batch():
    body = {"model": model, "prompt": PROMPT, "max_tokens": TOKENS, "temperature": 0, "ignore_eos": True}
    threads = [threading.Thread(target=post, args=("/v1/completions", body)) for _ in range(C)]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return time.perf_counter() - start


batch()
post("/start_profile")
before = drafts()
wall = batch()
steps = (drafts() - before) / C
post("/stop_profile")
print(json.dumps({"concurrency": C, "tokens": TOKENS, "wall_s": round(wall, 3), "steps": steps,
                  "step_ms": round(1000 * wall / steps, 2) if steps else None}))
