"""Measures prefix-cache reuse: each prompt is sent twice with identical text; prints the cached tokens of the repeat.

Usage: cache_probe.py [LABEL]
Environment: API_URL = chat completions URL (default http://192.0.2.10:8015/v1/chat/completions),
             MODEL = served model name (default GLM-5.3-Flash-NVFP4-Spark-TP4).
"""
import json
import os
import random
import sys
import urllib.request

URL = os.environ.get("API_URL", "http://192.0.2.10:8015/v1/chat/completions")
MODEL = os.environ.get("MODEL", "GLM-5.3-Flash-NVFP4-Spark-TP4")
WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split()


def send(text):
    body = json.dumps({"model": MODEL, "max_tokens": 4, "temperature": 0,
                       "chat_template_kwargs": {"reasoning_effort": "low"},
                       "messages": [{"role": "user", "content": text}]}).encode()
    request = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    usage = json.load(urllib.request.urlopen(request, timeout=600))["usage"]
    return usage["prompt_tokens"], (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)


for words in (1500, 6000, 15000, 24500):
    rng = random.Random(words)
    text = f"[probe-{sys.argv[1] if len(sys.argv) > 1 else 'x'}-{words}] " + " ".join(rng.choice(WORDS) for _ in range(words))
    send(text)
    prompt, cached = send(text)
    print(f"prompt {prompt:6d} tokens: cached {cached:6d} ({100 * cached / prompt:.1f}%)")
