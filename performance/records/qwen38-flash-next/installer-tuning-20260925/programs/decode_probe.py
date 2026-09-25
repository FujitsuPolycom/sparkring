"""Decode rate and draft acceptance per prompt type, from streaming timing and /metrics.

Usage: decode_probe.py BASE_URL [TOKENS] [RUNS] [TEMPERATURE]
TEMPERATURE defaults to 0 (greedy); a positive value samples with the
checkpoint's other generation defaults (top-k, top-p).
For each prompt, reports end-to-end tokens/s (including time to first token), the
steady per-token rate after the first token, and speculative-decoding acceptance
(accepted/draft tokens and per-position acceptance) from the server counters.
"""
import json
import re
import statistics
import sys
import time
import urllib.request

base = sys.argv[1].rstrip("/")
tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 512
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
temperature = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
root = base.rsplit("/v1", 1)[0]
model = json.loads(urllib.request.urlopen(base + "/models", timeout=30).read())["data"][0]["id"]
PROMPTS = {
    "prose": "Write a detailed 700-word story about a lighthouse keeper who repairs a clockwork bird.",
    "code": "Write a complete Python module implementing a thread-safe LRU cache class with get, put, delete, "
            "resize and stats methods, full type hints and docstrings, followed by pytest unit tests.",
    "json": "Produce a JSON array of 40 fictional employees; each object has id, name, department, title, "
            "salary, start_date and a list of three skills. Output only the JSON.",
}


def counters():
    text = urllib.request.urlopen(root + "/metrics", timeout=30).read().decode()
    values = {}
    for line in text.splitlines():
        match = re.match(r'(vllm:spec_decode_num_(?:accepted_tokens|draft_tokens|drafts)(?:_per_pos)?_total)\{([^}]*)\} ([0-9.e+]+)', line)
        if match:
            name, labels, value = match.groups()
            position = re.search(r'position="(\d+)"', labels)
            key = name + (":" + position.group(1) if position else "")
            values[key] = values.get(key, 0.0) + float(value)
    return values


def run(prompt):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
            "temperature": temperature, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    start, first, produced = time.time(), None, 0
    with urllib.request.urlopen(request, timeout=1800) as response:
        for line in response:
            if not line.startswith(b"data:") or b"[DONE]" in line:
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                produced = chunk["usage"]["completion_tokens"]
            if first is None and chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                first = time.time()
    end = time.time()
    return produced / (end - start), (produced - 1) / (end - first), end - start


print("model", model)
for label, prompt in PROMPTS.items():
    before = counters()
    results = [run(prompt) for _ in range(runs)]
    after = counters()
    delta = {key: after.get(key, 0) - before.get(key, 0) for key in after}
    drafts = delta.get("vllm:spec_decode_num_drafts_total", 0)
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    draft_tokens = delta.get("vllm:spec_decode_num_draft_tokens_total", 0)
    positions = [round(delta[key] / drafts, 2) for key in sorted(k for k in delta if "per_pos" in k)] if drafts else []
    e2e = statistics.median(r[0] for r in results)
    steady = statistics.median(r[1] for r in results)
    mean_len = (1 + accepted / drafts) if drafts else None
    print(f"{label:6s} end-to-end {e2e:6.2f} tok/s | after first token {steady:6.2f} tok/s | "
          f"acceptance {accepted / draft_tokens if draft_tokens else 0:.2f} | per position {positions} | "
          f"tokens per step {mean_len:.2f}" if mean_len else f"{label}: no speculative counters")
