"""Send one ~20K-token prompt twice and report time to first token and cached prompt tokens."""
import json
import random
import string
import sys
import time
import urllib.request

base = sys.argv[1].rstrip("/")
model = json.loads(urllib.request.urlopen(base + "/models", timeout=30).read())["data"][0]["id"]
rng = random.Random(7)
nonce = "".join(random.choice(string.ascii_letters) for _ in range(24))
text = nonce + " " + " ".join(rng.choice("alpha beta gamma delta ring spark fabric token cache".split()) for _ in range(20000))
for attempt in (1, 2):
    body = {"model": model, "messages": [{"role": "user", "content": text + "\nReply with OK."}], "max_tokens": 4,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.time()
    first = None
    usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.startswith(b"data:") and b"[DONE]" not in line:
                d = json.loads(line[5:])
                if first is None and d.get("choices"):
                    first = time.time() - start
                if d.get("usage"):
                    usage = d["usage"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    print(f"attempt {attempt}: TTFT {first:.2f}s prompt {usage['prompt_tokens']} cached {cached}")
