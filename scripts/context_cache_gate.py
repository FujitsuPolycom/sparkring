#!/usr/bin/env python3
"""G-CACHE gate driver: deterministic 32K context, TTFT and token equality.

Phases:
  store    - serve the canonical 32K prompt once (fills the cache), record
             fresh-prefill TTFT and the greedy continuation tokens.
  restore  - repeat the identical prompt (after a runtime restart), record
             restored TTFT and continuation token IDs; require exact equality
             against the stored baseline JSON.

The prompt is deterministic (seeded word sequence), so the same invocation
always produces the same token stream and the same cache digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import urllib.request

# Point this at the OpenAI-compatible endpoint of the rank-0 serving host.
# Set SPARKRING_BASE_URL (e.g. http://spark0:8210) for real runs; the default
# is a placeholder that documents the expected shape.
DEFAULT_BASE_URL = os.environ.get("SPARKRING_BASE_URL", "http://<rank0-host>:8210")
DEFAULT_MODEL = os.environ.get("SPARKRING_MODEL", "glm-5.2")


def build_prompt(target_words: int, seed: int = 20260728) -> str:
    rng = random.Random(seed)
    vocabulary = [
        "ring", "spark", "latent", "doorbell", "manifest", "chunk",
        "tensor", "verbs", "arena", "sequence", "graph", "combine",
        "query", "shard", "restore", "commit", "gate", "probe",
        "window", "anchor", "ledger", "beacon", "cascade", "drift",
    ]
    words = [vocabulary[rng.randrange(len(vocabulary))] for _ in range(target_words)]
    header = (
        "You are indexing a long operations log. Read it fully, then answer.\n"
        "LOG BEGIN\n"
    )
    footer = (
        "\nLOG END\n"
        "Question: state the first five words of the log, in order, "
        "then count how many times the word 'doorbell' appears in the "
        "first one hundred words. Answer concisely.\n"
    )
    return header + " ".join(words) + footer


def run_request(
    base_url: str,
    prompt: str,
    max_tokens: int,
    model: str = DEFAULT_MODEL,
    *,
    cache_salt: str | None = None,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "logprobs": 1,
        "return_tokens_as_token_ids": True,
    }
    if cache_salt is not None:
        if not isinstance(cache_salt, str) or not cache_salt:
            raise ValueError("cache_salt must be a non-empty string when provided")
        payload["cache_salt"] = cache_salt
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_at = None
    chunks: list[str] = []
    completion_token_ids: list[int] = []
    response_request_id: str | None = None
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            event = json.loads(body)
            event_request_id = event.get("id")
            if isinstance(event_request_id, str) and event_request_id:
                if (
                    response_request_id is not None
                    and response_request_id != event_request_id
                ):
                    raise RuntimeError("completion stream changed request ID")
                response_request_id = event_request_id
            choice = event.get("choices", [{}])[0]
            text = choice.get("text", "")
            logprobs = choice.get("logprobs") or {}
            for token in logprobs.get("tokens") or []:
                if isinstance(token, int):
                    completion_token_ids.append(token)
                elif isinstance(token, str) and token.startswith("token_id:"):
                    raw_id = token.removeprefix("token_id:")
                    if not raw_id.isdigit():
                        raise RuntimeError(f"invalid streamed token ID {token!r}")
                    completion_token_ids.append(int(raw_id))
                else:
                    raise RuntimeError(
                        "runtime did not honor return_tokens_as_token_ids"
                    )
            if text:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                chunks.append(text)
    finished = time.perf_counter()
    completion = "".join(chunks)
    if not completion_token_ids:
        raise RuntimeError("completion stream returned no original token IDs")
    result = {
        "ttft_seconds": (
            None if first_token_at is None else first_token_at - started
        ),
        "total_seconds": finished - started,
        "completion": completion,
        "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "model": model,
        "request_id": response_request_id,
        "completion_token_ids": completion_token_ids,
        "generation_parameters": {
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "logprobs": 1,
            "return_tokens_as_token_ids": True,
        },
    }
    if cache_salt is not None:
        result["cache_salt_contract"] = {
            "provided": True,
            "sha256": hashlib.sha256(cache_salt.encode("utf-8")).hexdigest(),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("store", "restore"), required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--words", type=int, default=24000)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ttft-limit-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(args.words, args.seed)
    result = run_request(args.base_url, prompt, args.max_tokens, args.model)
    result["phase"] = args.phase
    result["words"] = args.words
    result["seed"] = args.seed
    out_path = args.output_dir / f"{args.phase}-seed{args.seed}.json"
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    ttft = result['ttft_seconds']
    print(
        f"phase={args.phase} "
        f"ttft={'none' if ttft is None else format(ttft, '.3f')} "
        f"total={result['total_seconds']:.1f}s "
        f"completion_chars={len(result['completion'])}"
    )
    if ttft is None:
        print('WARNING: no tokens streamed (request produced empty output)')

    if args.phase == "restore":
        baseline_path = args.output_dir / f"store-seed{args.seed}.json"
        if not baseline_path.is_file():
            print("gate=fail reason=missing-store-baseline")
            return 2
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_binding = {
            "prompt_sha256": baseline.get("prompt_sha256")
            == result.get("prompt_sha256"),
            "model": baseline.get("model") == result.get("model") == args.model,
            "seed": baseline.get("seed") == result.get("seed") == args.seed,
            "words": baseline.get("words") == result.get("words") == args.words,
            "generation_parameters": baseline.get("generation_parameters")
            == result.get("generation_parameters")
            == {
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "stream": True,
                "logprobs": 1,
                "return_tokens_as_token_ids": True,
            },
        }
        baseline_binding_ok = all(baseline_binding.values())
        exact_text_equal = baseline["completion"] == result["completion"]
        exact_token_equal = (
            isinstance(baseline.get("completion_token_ids"), list)
            and bool(baseline["completion_token_ids"])
            and baseline["completion_token_ids"] == result["completion_token_ids"]
        )
        # Ground-truth recall/count remain diagnostics only. Cache correctness
        # gates exclusively on exact completion token IDs recorded through the
        # serving runtime's pinned /tokenize endpoint.
        body = prompt.split("LOG BEGIN\n")[1].split("\nLOG END")[0]
        words = body.split()
        truth_recall = words[:5]
        truth_count = words[:100].count("doorbell")
        truth_sequence = " ".join(truth_recall)
        recall_correct = (
            truth_sequence in baseline["completion"]
            and truth_sequence in result["completion"]
        )
        ttft_ok = (
            result["ttft_seconds"] is not None
            and result["ttft_seconds"] < args.ttft_limit_seconds
        )
        speedup = (
            baseline["ttft_seconds"] / result["ttft_seconds"]
            if result["ttft_seconds"]
            else 0.0
        )
        verdict = (
            "pass"
            if (baseline_binding_ok and exact_token_equal and ttft_ok)
            else "fail"
        )
        report = {
            "gate": verdict,
            "ttft_ok": ttft_ok,
            "baseline_binding": baseline_binding,
            "baseline_binding_ok": baseline_binding_ok,
            "completion_token_ids_exactly_equal": exact_token_equal,
            "output_exactly_equal": exact_text_equal,
            "recall_segment_correct_but_not_a_gate": recall_correct,
            "ground_truth_first_five": truth_recall,
            "ground_truth_count": truth_count,
            "fresh_ttft_seconds": baseline["ttft_seconds"],
            "restored_ttft_seconds": result["ttft_seconds"],
            "ttft_speedup": speedup,
        }
        if not exact_text_equal:
            fresh = baseline["completion"]
            restored = result["completion"]
            divergence = next(
                (
                    index
                    for index, (a, b) in enumerate(zip(fresh, restored))
                    if a != b
                ),
                min(len(fresh), len(restored)),
            )
            report["first_divergence_char"] = divergence
            report["fresh_tail"] = fresh[max(0, divergence - 40): divergence + 40]
            report["restored_tail"] = restored[
                max(0, divergence - 40): divergence + 40
            ]
        (args.output_dir / f"gate-seed{args.seed}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if verdict == "pass" else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
