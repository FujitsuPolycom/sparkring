"""Sequential decode passes over 24 fixed prompts against an OpenAI-compatible vLLM server.

Each pass sends one chat request at a time (512 output tokens, at least 256),
saves the generated token IDs, and records the change in vLLM's speculative
decoding counters on /metrics over the pass. API_BASE selects the server (default http://127.0.0.1:8000).

Modes:
  greedy  OUT  -- sequential temperature-0 requests, token IDs saved per prompt
  sample  OUT  -- sequential temperature-1.0 seeded requests, /metrics spec-decode deltas saved
  compare A B  -- per-prompt token equality between two greedy result files
  metrics OUT  -- snapshot of the spec-decode counters
"""
import json
import os
import re
import sys
import time
import urllib.request

BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
MODEL = "GLM-5.3-Flash-NVFP4-Spark-TP2"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reserve_table",
            "description": "Reserve a restaurant table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "restaurant": {"type": "string"},
                    "party_size": {"type": "integer"},
                    "time": {"type": "string", "description": "ISO 8601 local time"},
                },
                "required": ["restaurant", "party_size", "time"],
            },
        },
    },
]

PROMPTS = [
    ("code-python", "Write a Python function that merges overlapping intervals. Include a docstring, type hints and three pytest tests."),
    ("code-rust", "Implement a thread-safe LRU cache in Rust with get and put, and explain the locking choice."),
    ("code-sql", "Tables: orders(id, customer_id, total, created_at) and customers(id, name, country). Write a SQL query for the top 5 countries by revenue in 2025 and explain each clause."),
    ("code-bash", "Write a bash script that lists the 10 largest files under a directory given as an argument, excluding any .git directories, with human-readable sizes."),
    ("code-js-bug", "Explain why this JavaScript prints 3 three times and show two fixes:\nfor (var i = 0; i < 3; i++) { setTimeout(() => console.log(i), 0); }"),
    ("code-c", "Write a C function that reverses a singly linked list in place, plus a small main() that demonstrates it."),
    ("code-go", "Explain the difference between a mutex and a semaphore, with a short Go example of each."),
    ("math-train", "A train leaves city A at 3:15 pm travelling at 80 km/h. Another leaves city B, 340 km away, at 4:00 pm travelling toward A at 100 km/h. At what time do they meet, and how far from A?"),
    ("math-cubic", "Solve x^3 - 6x^2 + 11x - 6 = 0 and verify each root."),
    ("math-proof", "Prove that the square root of 2 is irrational."),
    ("math-integral", "Compute the integral of x*e^x from 0 to 1 step by step."),
    ("math-dice", "Two fair six-sided dice are rolled. What is the probability that the sum is prime? Show the counting."),
    ("math-perm", "How many distinct arrangements of the letters of MISSISSIPPI are there? Explain."),
    ("prose-story", "Write a short story about a lighthouse keeper who finds a message in a bottle."),
    ("prose-email", "Write a polite professional email declining a Thursday meeting and proposing two alternative times."),
    ("prose-tcp", "Explain how TCP congestion control works to a new engineer, covering slow start, congestion avoidance and fast recovery."),
    ("prose-history", "Summarize the main causes of the French Revolution in three paragraphs."),
    ("prose-kids", "Describe the water cycle for a ten-year-old."),
    ("prose-haiku", "Write five haiku about autumn in a large city."),
    ("prose-translate", "Translate into French and Spanish: 'The library opens at nine, but the reading room stays closed until the renovation is finished next spring.'"),
    ("struct-json", "Return a JSON object describing a fictional science-fiction novel with fields title, author, year, genres (array), and a 100-word synopsis."),
    ("struct-table", "Compare quicksort and mergesort in a Markdown table covering time complexity, space, stability and typical use."),
    ("tool-weather", "What's the weather in Paris and in Tokyo right now? Use celsius."),
    ("tool-reserve", "Book a table for 4 at Luigi's tomorrow at 7 pm; tomorrow is 2026-10-01."),
]


def post(path, body, timeout=600):
    request = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def spec_metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as response:
        text = response.read().decode()
    totals = {}
    for line in text.splitlines():
        if not line.startswith("vllm:spec_decode") and not line.startswith("vllm:generation_tokens_total"):
            continue
        match = re.match(r"^([a-zA-Z_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if not match:
            continue
        name, labels, value = match.groups()
        position = re.search(r'position="(\d+)"', labels or "")
        key = name + (f"[{position.group(1)}]" if position else "")
        totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def request_body(name, prompt, temperature, seed=None, max_tokens=512, min_tokens=256):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "return_token_ids": True,
        "stream": False,
    }
    if name.startswith("tool-"):
        body["tools"] = TOOLS
        body["tool_choice"] = "auto"
    if seed is not None:
        body["seed"] = seed
    return body


def run(mode, out):
    temperature = 0.0 if mode == "greedy" else 1.0
    before = spec_metrics()
    results = []
    started = time.time()
    for number, (name, prompt) in enumerate(PROMPTS):
        body = request_body(name, prompt, temperature, seed=None if mode == "greedy" else 1000 + number)
        t0 = time.time()
        response = post("/v1/chat/completions", body)
        elapsed = time.time() - t0
        choice = response["choices"][0]
        message = choice.get("message", {})
        results.append({
            "name": name,
            "token_ids": choice.get("token_ids"),
            "finish_reason": choice.get("finish_reason"),
            "content": message.get("content"),
            "reasoning": message.get("reasoning_content") or message.get("reasoning"),
            "tool_calls": message.get("tool_calls"),
            "usage": response.get("usage"),
            "seconds": elapsed,
        })
        tokens = len(choice.get("token_ids") or [])
        print(f"{name:16s} tokens={tokens:4d} finish={choice.get('finish_reason')} {elapsed:6.1f}s", flush=True)
    after = spec_metrics()
    delta = {key: after.get(key, 0.0) - before.get(key, 0.0) for key in after}
    document = {"mode": mode, "temperature": temperature, "seconds": time.time() - started,
                "spec_before": before, "spec_after": after, "spec_delta": delta, "results": results}
    write_atomic(out, document)
    summarize_spec(delta)


def write_atomic(path, document):
    temporary = path + ".partial"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=1)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def summarize_spec(delta):
    drafts = delta.get("vllm:spec_decode_num_drafts_total", 0.0)
    draft_tokens = delta.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    if drafts:
        print(f"drafts={drafts:.0f} draft_tokens={draft_tokens:.0f} accepted={accepted:.0f} "
              f"acceptance_rate={accepted / draft_tokens:.4f} mean_acceptance_length={1 + accepted / drafts:.4f}")
        positions = sorted(k for k in delta if k.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total["))
        print("per-position accepted/drafts: " + ", ".join(
            f"{k.split('[')[1].rstrip(']')}={delta[k] / drafts:.4f}" for k in positions))


def compare(path_a, path_b):
    a = json.load(open(path_a, encoding="utf-8"))["results"]
    b = json.load(open(path_b, encoding="utf-8"))["results"]
    identical = 0
    for left, right in zip(a, b):
        ta, tb = left["token_ids"] or [], right["token_ids"] or []
        prefix = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        same = ta == tb
        identical += same
        print(f"{left['name']:16s} {'identical' if same else 'DIFFERS'} len={len(ta)}/{len(tb)} "
              f"first_divergence={'-' if same else prefix}")
    print(f"identical {identical}/{len(a)}")


if __name__ == "__main__":
    command = sys.argv[1]
    if command in ("greedy", "sample"):
        run(command, sys.argv[2])
    elif command == "compare":
        compare(sys.argv[2], sys.argv[3])
    elif command == "metrics":
        write_atomic(sys.argv[2], spec_metrics())
    else:
        raise SystemExit(__doc__)
