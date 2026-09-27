"""Measure prefill throughput: unique-prefix prompts, one output token, streaming TTFT."""
import json
import random
import string
import sys
import time
import urllib.request

base = sys.argv[1].rstrip("/")
sizes = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [4096, 16384, 65536]
model = json.loads(urllib.request.urlopen(base + "/models", timeout=30).read())["data"][0]["id"]
words = ("spark ring fabric model token cache kernel layer expert tensor rank node cable image "
         "weight shard prefix decode prefill memory stream copy verify install profile").split()


def tokens(text):
    body = json.dumps({"model": model, "prompt": text}).encode()
    request = urllib.request.Request(base.rsplit("/v1", 1)[0] + "/tokenize", data=body,
                                     headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(request, timeout=120).read())["count"]


for size in sizes:
    rng = random.Random(size)
    nonce = "".join(random.choice(string.ascii_letters) for _ in range(24))
    text = nonce + " " + " ".join(rng.choice(words) for _ in range(int(size * 0.95)))
    count = tokens(text)
    body = {"model": model, "prompt": text, "max_tokens": 1, "temperature": 0, "stream": True}
    request = urllib.request.Request(base + "/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(request, timeout=1800) as response:
        for line in response:
            if line.startswith(b"data:") and b"[DONE]" not in line:
                first = time.time() - start
                break
    print(f"prompt {count:>7} tokens: TTFT {first:6.2f}s = {count / first:7.0f} tok/s prefill", flush=True)
