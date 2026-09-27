"""Separate engine warm-up from Engram row reuse in prefill time.

Sends ~16K-token prompts of random words: text A, new text B, text A again
(each with a fresh leading nonce, so the prefix cache cannot match), then new
text C. A and B cost the same unless the first request pays warm-up; A again
is faster than B only if the rows its n-grams read are cached somewhere.
"""
import json
import random
import string
import sys
import time
import urllib.request

base = sys.argv[1].rstrip("/")
size = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
model = json.loads(urllib.request.urlopen(base + "/models", timeout=30).read())["data"][0]["id"]
words = ("spark ring fabric model token cache kernel layer expert tensor rank node cable image "
         "weight shard prefix decode prefill memory stream copy verify install profile").split()


def run(label, seed):
    rng = random.Random(seed)
    nonce = "".join(random.choice(string.ascii_letters) for _ in range(24))
    text = nonce + " " + " ".join(rng.choice(words) for _ in range(int(size * 0.95)))
    body = {"model": model, "prompt": text, "max_tokens": 1, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.time()
    first = None
    usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if line.startswith(b"data:") and b"[DONE]" not in line:
                d = json.loads(line[5:])
                if first is None and d.get("choices"):
                    first = time.time() - start
                if d.get("usage"):
                    usage = d["usage"]
    n = usage["prompt_tokens"]
    print(f"{label}: {n} tokens TTFT {first:.2f}s = {n / first:.0f} tok/s", flush=True)


seed = int(time.time())
run("text A", seed)
run("text B", seed + 1)
run("text A again", seed)
run("text C", seed + 2)
