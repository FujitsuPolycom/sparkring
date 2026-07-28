#!/usr/bin/env python3
"""G-CACHE stress driver: many entries, mixed hit/miss concurrency.

Round 1 (fill): store N distinct deterministic contexts sequentially.
Round 2 (churn): re-request every stored context (expected restores) plus
an equal number of novel contexts (expected misses) with bounded
concurrency, several sweeps. Reports per-request TTFT split by expected
hit/miss and hard-fails on any request error.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from pathlib import Path

from context_cache_gate import DEFAULT_BASE_URL, build_prompt, run_request


def one(base_url: str, seed: int, words: int, max_tokens: int) -> dict:
    prompt = build_prompt(words, seed)
    started = time.perf_counter()
    result = run_request(base_url, prompt, max_tokens)
    result["seed"] = seed
    result["words"] = words
    result["wall"] = time.perf_counter() - started
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--entries", type=int, default=6)
    parser.add_argument("--sweeps", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--base-words", type=int, default=6000)
    parser.add_argument("--words-step", type=int, default=3000)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stored_seeds = [31000 + index for index in range(args.entries)]
    novel_base = 62000
    report: dict = {"fill": [], "sweeps": []}

    print(f"fill: storing {args.entries} contexts sequentially")
    for index, seed in enumerate(stored_seeds):
        words = args.base_words + index * args.words_step
        result = one(args.base_url, seed, words, args.max_tokens)
        print(
            f"  fill seed={seed} words={words}"
            f" ttft={result['ttft_seconds']:.2f}s"
        )
        report["fill"].append(result)

    for sweep in range(args.sweeps):
        jobs = []
        for index, seed in enumerate(stored_seeds):
            words = args.base_words + index * args.words_step
            jobs.append(("hit", seed, words))
        for index in range(args.entries):
            seed = novel_base + sweep * args.entries + index
            words = args.base_words + index * args.words_step
            jobs.append(("miss", seed, words))
        results = []
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
            futures = {
                pool.submit(
                    one, args.base_url, seed, words, args.max_tokens
                ): kind
                for kind, seed, words in jobs
            }
            for future in concurrent.futures.as_completed(futures):
                kind = futures[future]
                result = future.result()
                result["expected"] = kind
                results.append(result)
        hits = [r["ttft_seconds"] for r in results if r["expected"] == "hit"]
        misses = [
            r["ttft_seconds"] for r in results if r["expected"] == "miss"
        ]
        summary = {
            "sweep": sweep,
            "hit_ttft_median": statistics.median(hits),
            "hit_ttft_max": max(hits),
            "miss_ttft_median": statistics.median(misses),
            "requests": len(results),
            "failures": sum(
                1 for r in results if r["ttft_seconds"] is None
            ),
        }
        print(json.dumps(summary, sort_keys=True))
        report["sweeps"].append({"summary": summary, "results": results})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    any_failures = any(
        sweep["summary"]["failures"] for sweep in report["sweeps"]
    )
    print("stress=", "fail" if any_failures else "pass")
    return 2 if any_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
