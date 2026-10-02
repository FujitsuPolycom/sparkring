#!/usr/bin/env python3
"""Measure MTP draft acceptance from vLLM's /metrics counters over a fixed set of seeded requests.

The probe reads the server's speculative-decoding counters, sends every
request of a fixed list (24 prompts, each with two seeds) at temperature 1.0
with a bounded number of requests in flight, reads the counters again and
reports the differences:

- acceptance rate: accepted draft tokens / proposed draft tokens;
- accept length: 1 + accepted draft tokens / drafts, the tokens each
  verification step emits on average;
- per-position acceptance: accepted tokens at draft position i / drafts.

Only this probe's requests may run while it measures, because the counters
are server-wide. Standard library only; run it on any host that reaches the
API, for example:

    python3 spec_accept_probe.py --base http://NODE_A_ADDRESS:8000 --out probe.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.request

PROMPTS = (
    "Explain how a hash map handles collisions, with a short Python example.",
    "Write a haiku sequence of four poems about autumn rain in a city.",
    "What is 17 * 23 + 144 / 12? Show your steps.",
    "Summarize the causes of the French Revolution in five bullet points.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Describe the water cycle to a ten-year-old.",
    "Translate into French: 'The library opens at nine and closes at six on weekdays.'",
    "Give three arguments for and three against remote work.",
    "Write a SQL query that finds the five customers with the highest total order value.",
    "Tell a short story about a robot that learns to paint.",
    "Explain the difference between TCP and UDP.",
    "A train leaves at 14:10 and arrives at 17:45. How long is the journey in minutes?",
    "Write a bash one-liner that counts lines in all .py files under the current directory.",
    "What are the main differences between mitosis and meiosis?",
    "Draft a polite email asking a colleague to review a pull request by Friday.",
    "Explain what a closure is in JavaScript with an example.",
    "List the planets of the solar system with one fact about each.",
    "Write a limerick about a cat who loves coffee.",
    "How does public-key cryptography let two strangers communicate securely?",
    "Write a Rust function that reverses a string slice and returns a String.",
    "Compare the climates of Norway and Spain in one paragraph.",
    "Solve for x: 3x + 7 = 2x - 5. Explain each step.",
    "Describe a recipe for a simple tomato pasta sauce.",
    "Explain gradient descent and the role of the learning rate.",
)
SEEDS = (1001, 2002)
COUNTERS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}
PER_POSITION = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9.eE+-]+)$')


def read_counters(base):
    """Sum each speculative-decoding counter over its label sets; positions are kept apart."""
    text = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    totals = {key: 0.0 for key in COUNTERS}
    positions = {}
    for line in text.splitlines():
        match = SAMPLE.match(line.strip())
        if not match:
            continue
        name, labels, value = match.group(1), match.group(2) or "", float(match.group(3))
        for key, metric in COUNTERS.items():
            if name == metric:
                totals[key] += value
        if name == PER_POSITION:
            position = re.search(r'position="(\d+)"', labels)
            if position:
                index = int(position.group(1))
                positions[index] = positions.get(index, 0.0) + value
    return totals, positions


def request(base, model, prompt, seed, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 1.0,
            "seed": seed, "max_tokens": max_tokens}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    response = json.loads(urllib.request.urlopen(req, timeout=600).read())
    choice = response["choices"][0]
    message = choice["message"]
    text = (message.get("reasoning") or message.get("reasoning_content") or "") + (message.get("content") or "")
    return {"seed": seed, "prompt": prompt, "finish_reason": choice["finish_reason"],
            "completion_tokens": response["usage"]["completion_tokens"],
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "chars": len(text)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="server root, without /v1")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4-QAD-TP2")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--label", default="", help="free text stored with the result")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    jobs = [(prompt, seed) for seed in SEEDS for prompt in PROMPTS]
    before, before_pos = read_counters(args.base)
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(lambda job: request(args.base, args.model, job[0], job[1], args.max_tokens), jobs))
    elapsed = time.time() - started
    after, after_pos = read_counters(args.base)
    delta = {key: after[key] - before[key] for key in COUNTERS}
    drafts = delta["drafts"]
    summary = {
        "label": args.label, "requests": len(results), "concurrency": args.concurrency,
        "max_tokens": args.max_tokens, "temperature": 1.0, "elapsed_s": round(elapsed, 1),
        "completion_tokens": sum(row["completion_tokens"] for row in results),
        "counters": delta,
        "acceptance_rate": delta["accepted_tokens"] / delta["draft_tokens"] if delta["draft_tokens"] else None,
        "accept_length": 1 + delta["accepted_tokens"] / drafts if drafts else None,
        "per_position_acceptance": {str(index): (after_pos[index] - before_pos.get(index, 0.0)) / drafts
                                    for index in sorted(after_pos)} if drafts else {},
        "finish_reasons": {reason: sum(row["finish_reason"] == reason for row in results)
                           for reason in sorted({row["finish_reason"] for row in results})},
        "responses": results,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key != "responses"}, indent=2))


if __name__ == "__main__":
    main()
