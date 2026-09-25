"""Repeatable TP2 benchmark: prefill TTFT, greedy decode rate and greedy output fingerprints.

Usage: bench.py BASE_URL LABEL OUTPUT_JSON
Prefill prompts carry a random prefix, so prefix caching cannot reuse earlier work;
each size runs twice and the second run is reported. Decode uses temperature 0 on a
fixed prompt so every configuration drafts and verifies the same text.
"""
import json
import random
import statistics
import string
import sys
import time
import urllib.request

base, label, output = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
model = json.loads(urllib.request.urlopen(base + "/models", timeout=30).read())["data"][0]["id"]
root = base.rsplit("/v1", 1)[0]
words = ("spark ring fabric model token cache kernel layer expert tensor rank node cable image "
         "weight shard prefix decode prefill memory stream copy verify install profile").split()


def post(path, body, timeout=1800):
    request = urllib.request.Request(path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=timeout)


def count(text):
    return json.loads(post(root + "/tokenize", {"model": model, "prompt": text}).read())["count"]


def ttft(size, seed):
    rng = random.Random(seed)
    nonce = "".join(random.choice(string.ascii_letters) for _ in range(24))
    text = nonce + " " + " ".join(rng.choice(words) for _ in range(int(size * 0.95)))
    tokens = count(text)
    start = time.time()
    with post(base + "/completions", {"model": model, "prompt": text, "max_tokens": 1, "temperature": 0, "stream": True}) as r:
        for line in r:
            if line.startswith(b"data:") and b"[DONE]" not in line:
                return tokens, time.time() - start


def chat(prompt, max_tokens, **extra):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}, **extra}
    start = time.time()
    reply = json.loads(post(base + "/chat/completions", body).read())
    return reply, time.time() - start


result = {"label": label, "model": model, "prefill": {}, "decode": [], "fingerprints": {}}
for size in (4096, 16384, 65536):
    runs = [ttft(size, size + i) for i in range(2)]
    tokens, seconds = runs[-1]
    result["prefill"][str(size)] = {"tokens": tokens, "seconds": round(seconds, 3), "tok_s": round(tokens / seconds)}
    print(f"[{label}] prefill {tokens} tokens: {seconds:.2f}s = {tokens / seconds:.0f} tok/s (first run {runs[0][1]:.2f}s)", flush=True)
story = "Write a detailed 700-word story about a lighthouse keeper who repairs a clockwork bird."
for _ in range(3):
    reply, seconds = chat(story, 512, ignore_eos=True)
    produced = reply["usage"]["completion_tokens"]
    result["decode"].append(round(produced / seconds, 2))
print(f"[{label}] decode (greedy, 512 tokens x3): {result['decode']} tok/s, median {statistics.median(result['decode'])}", flush=True)
for name, prompt in (("count", "Count from 1 to 30 separated by commas. Output only the numbers."),
                     ("math", "What is 17*23? Answer with just the number."),
                     ("code", "Write a Python function is_prime(n) and nothing else."),
                     ("story", story)):
    reply, _ = chat(prompt, 96)
    result["fingerprints"][name] = reply["choices"][0]["message"]["content"]
json.dump(result, open(output, "w"), indent=1)
print(f"[{label}] fingerprints saved: " + ", ".join(f"{k}={v[:40]!r}" for k, v in result["fingerprints"].items()), flush=True)
