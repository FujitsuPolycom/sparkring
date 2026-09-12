"""Record bounded native-MTP model requests and cache evidence without host changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import urllib.request
import urllib.error
from pathlib import Path


class _NoCredentialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        raise urllib.error.HTTPError(request.full_url, code,
                                     "Authenticated requests do not follow redirects", headers, response)


def read_api_key(path):
    """Use the first nonempty key from the operator's private newline-separated file."""
    keys = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not keys or any(any(character.isspace() or ord(character) < 33 or ord(character) > 126
                           for character in key) for key in keys):
        raise ValueError("API key file requires nonempty printable ASCII keys without whitespace")
    return keys[0]


def fetch(url, payload=None, *, api_key=None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(url, data=data, headers=headers)
    open_request = (urllib.request.urlopen if api_key is None else
                    urllib.request.build_opener(_NoCredentialRedirect()).open)
    with open_request(request, timeout=120) as response:
        return response.read().decode()


def cache_metrics(text):
    result = {}
    for line in text.splitlines():
        if line.startswith("#") or not any(word in line for word in ("prefix", "external", "kv_transfer")):
            continue
        match = re.fullmatch(r'(.+?)\s+([-+0-9.eE]+)(?:\s+\d+)?', line)
        if match:
            try:
                value = float(match[2])
            except ValueError:
                continue
            if math.isfinite(value):
                result[match[1]] = value
    return result


def temperature_value(text):
    value = float(text)
    if not math.isfinite(value) or not 0 <= value <= 2:
        raise ValueError("Temperature must be finite and between 0 and 2")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-file", type=Path, help="Private newline-separated API keys; use the first nonempty key")
    parser.add_argument("--temperature", type=temperature_value, default=1.0)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--expected-text", default="SPARKCACHE_GLM53_OK")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("semantic", "persistent"), required=True)
    parser.add_argument("--phase", choices=("before-restart", "after-restart", "semantic"), default="semantic")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--execute-authorized", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output directory must be absent")
    if not 1 <= args.max_tokens <= 2048:
        raise SystemExit("Output budget must be between 1 and 2048 tokens")
    suffix = f"Respond with exactly {args.expected_text} and no other text."
    prompt = suffix if args.kind == "semantic" else "benchmark " * 8192 + "\n" + suffix
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    payload = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
               "temperature": args.temperature, "seed": 9046500, "max_tokens": args.max_tokens,
               "chat_template_kwargs": {"enable_thinking": True}}
    if args.kind == "persistent" and args.phase == "semantic":
        raise SystemExit("Persistent requests require an explicit restart phase")
    if args.phase == "after-restart":
        if args.reference is None:
            raise SystemExit("After-restart requests require the before-restart receipt directory")
        if json.loads((args.reference / "request.json").read_text()) != payload:
            raise SystemExit("Persistent request does not match the before-restart request")
    if not args.execute_authorized:
        print(json.dumps({"url": args.endpoint + "/v1/chat/completions", "kind": args.kind,
                          "temperature": args.temperature, "max_tokens": args.max_tokens,
                          "expected_text": args.expected_text,
                          "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                          "request_characters": len(prompt), "executed": False}, indent=2))
        return
    authentication = {} if args.api_key_file is None else {"api_key": read_api_key(args.api_key_file)}
    args.output.mkdir(parents=True)
    base = args.endpoint.rstrip("/")
    metrics_before = fetch(base + "/metrics", **authentication)
    (args.output / "metrics-before.txt").write_text(metrics_before, encoding="utf-8")
    (args.output / "request.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    started = time.monotonic()
    response = json.loads(fetch(base + "/v1/chat/completions", payload, **authentication))
    elapsed = time.monotonic() - started
    (args.output / "response.json").write_text(json.dumps(response, indent=2), encoding="utf-8")
    metrics_after = fetch(base + "/metrics", **authentication)
    (args.output / "metrics-after.txt").write_text(metrics_after, encoding="utf-8")
    choice = response["choices"][0]
    content = choice["message"].get("content")
    passed = choice.get("finish_reason") == "stop" and content == args.expected_text
    before, after = cache_metrics(metrics_before), cache_metrics(metrics_after)
    receipt = {"kind": args.kind, "phase": args.phase, "temperature": args.temperature,
               "expected_text": args.expected_text,
               "semantic_passed": passed, "semantic_status": "passed" if passed else "inconclusive",
               "elapsed_seconds": elapsed,
               "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
               "cache_metric_deltas": {key: value - before[key] for key, value in after.items() if key in before},
               "finish_reason": choice.get("finish_reason"), "usage": response.get("usage"),
               "content": content, "persistent_restore_proven": False,
               "limitation": "Require per-rank external restore logs and external-hit metric deltas; latency alone is insufficient. One stochastic canary does not establish model or cache correctness; a mismatch needs repeated controls."}
    if args.reference:
        reference = json.loads((args.reference / "response.json").read_text())
        receipt["visible_content_matches_reference"] = reference["choices"][0]["message"].get("content") == content
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
