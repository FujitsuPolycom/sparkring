"""Verify a SparkRing API: model list, status view, three prompts and decode rate."""
import json
import sys
import time
import urllib.request

base = sys.argv[1].rstrip("/")
extra = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}


def call(path, body=None, timeout=600):
    request = urllib.request.Request(base + path, data=json.dumps(body).encode() if body else None,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


models = call("/models")["data"]
name = models[0]["id"]
print("model:", name, "max_model_len:", models[0].get("max_model_len"))
root = base.rsplit("/v1", 1)[0]
try:
    with urllib.request.urlopen(root + "/v1/sparkring/status/view", timeout=30) as response:
        view = json.loads(response.read())
    print("status view keys:", sorted(view)[:12], "image_runtime" in json.dumps(view))
except Exception as error:  # noqa: BLE001
    print("status view error:", error)

prompts = [
    ("count", "Count from 1 to 20 separated by commas. Output only the numbers.", 200),
    ("math", "What is 17*23? Answer with just the number.", 200),
    ("code", "Write a Python function is_prime(n) and nothing else.", 400),
]
for label, prompt, limit in prompts:
    body = {"model": name, "messages": [{"role": "user", "content": prompt}], "max_tokens": limit,
            "temperature": 0, **extra}
    start = time.time()
    out = call("/chat/completions", body)
    elapsed = time.time() - start
    message = out["choices"][0]["message"]
    text = (message.get("content") or "").strip()
    print(f"[{label}] {out['usage']['completion_tokens']} tok in {elapsed:.1f}s: {text[:160]!r}")

body = {"model": name, "messages": [{"role": "user", "content": "Write a 600-word story about a lighthouse keeper."}],
        "max_tokens": 1024, "temperature": 0.7, "ignore_eos": True, **extra}
start = time.time()
out = call("/chat/completions", body)
elapsed = time.time() - start
tokens = out["usage"]["completion_tokens"]
print(f"decode: {tokens} tokens in {elapsed:.1f}s = {tokens / elapsed:.1f} tok/s (end-to-end, single stream)")
