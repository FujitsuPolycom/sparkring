"""Greedy fingerprints of four fixed prompts, token by token.

python -m performance.harnesses.serving_ab.fingerprint BASE_URL OUTPUT_JSON     (BASE_URL ends in /v1)

The prompts and request body are those of
performance/records/qwen38-flash-next/decode-ab-20260925/bench.py (temperature 0, 96 tokens,
``enable_thinking`` false), with seed 0 and the returned token strings, so two starts compare token by
token. A model whose chat template ignores ``enable_thinking`` returns reasoning text, which is kept.
"""
import json
import sys
import urllib.request

STORY = "Write a detailed 700-word story about a lighthouse keeper who repairs a clockwork bird."
PROMPTS = (("count", "Count from 1 to 30 separated by commas. Output only the numbers."),
           ("math", "What is 17*23? Answer with just the number."),
           ("code", "Write a Python function is_prime(n) and nothing else."),
           ("story", STORY))


def main(base: str, output: str) -> None:
    base = base.rstrip("/")
    model = json.loads(urllib.request.urlopen(base + "/models", timeout=30).read())["data"][0]["id"]
    result = {"model": model, "fingerprints": {}}
    for name, prompt in PROMPTS:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 96,
                "temperature": 0, "seed": 0, "chat_template_kwargs": {"enable_thinking": False},
                "logprobs": True, "top_logprobs": 0}
        request = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        reply = json.loads(urllib.request.urlopen(request, timeout=600).read())
        choice = reply["choices"][0]
        message = choice["message"]
        result["fingerprints"][name] = {
            "content": message.get("content"),
            "reasoning": message.get("reasoning_content") or message.get("reasoning"),
            "tokens": [item["token"] for item in ((choice.get("logprobs") or {}).get("content") or [])],
            "completion_tokens": reply["usage"]["completion_tokens"]}
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
