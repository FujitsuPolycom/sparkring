"""Time to first token across prompt sizes: random-prefix prompts, best of RUNS per size."""
import json
import random
import string
import sys
import time
import urllib.request

base = sys.argv[1].rstrip("/")
sizes = [int(value) for value in sys.argv[2].split(",")]
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
model = json.loads(urllib.request.urlopen(base + "/models", timeout=30).read())["data"][0]["id"]
root = base.rsplit("/v1", 1)[0]
words = "spark ring fabric model token cache kernel layer expert tensor rank node cable image".split()


def post(url, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=1800)


for size in sizes:
    times = []
    for run in range(runs):
        nonce = "".join(random.choice(string.ascii_letters) for _ in range(24))
        rng = random.Random(size * 100 + run)
        text = nonce + " " + " ".join(rng.choice(words) for _ in range(int(size * 0.95)))
        tokens = json.loads(post(root + "/tokenize", {"model": model, "prompt": text}).read())["count"]
        start = time.time()
        with post(base + "/completions", {"model": model, "prompt": text, "max_tokens": 1, "temperature": 0, "stream": True}) as r:
            for line in r:
                if line.startswith(b"data:") and b"[DONE]" not in line:
                    break
        times.append(time.time() - start)
    best = min(times)
    print(f"{tokens:6d} tokens: best {best:6.3f}s ({tokens / best:6.0f} tok/s) runs {[round(t, 3) for t in times]}", flush=True)
