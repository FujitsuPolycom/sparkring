"""Bounded C1/C8 comparison with disjoint cold prefixes and a warm decode pass."""

import concurrent.futures
import json
import threading
import time
import urllib.request
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--base-url", required=True, help="OpenAI-compatible base URL ending in /v1"
)
parser.add_argument("--model", required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--trials", type=int, default=2)
parser.add_argument("--fixture-id", default="", help="use the same fresh identifier for matched arms to avoid earlier prefix-cache entries")
args = parser.parse_args()
if not args.label.replace("-", "").isalnum() or args.trials < 1:
    parser.error("Use an alphanumeric label and positive trial count")
ROOT = args.output
ROOT.mkdir(parents=True, exist_ok=True)
label = args.label
if (ROOT / (label + ".json")).exists():
    parser.error("Output label already exists")


def request(index, barrier, trial):
    # Fixed text per trial/rank, identical across arms; disjoint leading content.
    prefix = f"Observatory station {index}, record series {trial}. "
    if args.fixture_id:
        prefix = f"{args.fixture_id}/{trial}/{index}. " + prefix
    prompt = (
        prefix
        + (
            "Record the wind speed, cloud coverage and air temperature each hour. "
            * 500
        )
        + "\nWrite a detailed numbered operating checklist for this observatory."
    )
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 256,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    barrier.wait()
    start = time.monotonic()
    first = None
    usage = None
    parts = []
    with urllib.request.urlopen(req, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            row = json.loads(line[6:])
            usage = row.get("usage") or usage
            for c in row.get("choices", []):
                content = c.get("delta", {}).get("content")
                if content:
                    if first is None:
                        first = time.monotonic()
                    parts.append(content)
    end = time.monotonic()
    assert first and usage and usage["completion_tokens"] > 0 and parts
    return {
        "index": index,
        "start": start,
        "first": first,
        "end": end,
        "ttft": first - start,
        "usage": usage,
        "text": "".join(parts),
    }


results = []
for trial in range(args.trials):
    for c in [1, 8]:
        for phase in ["cold", "warm"]:
            barrier = threading.Barrier(c)
            with concurrent.futures.ThreadPoolExecutor(c) as pool:
                rows = list(
                    pool.map(lambda i: request(i, barrier, f"{trial}-{c}"), range(c))
                )
            wall = max(r["end"] for r in rows) - min(r["start"] for r in rows)
            decode = max(r["end"] for r in rows) - min(r["first"] for r in rows)
            tokens = sum(r["usage"]["completion_tokens"] for r in rows)
            summary = {
                "fixture_id": args.fixture_id,
                "trial": trial,
                "c": c,
                "phase": phase,
                "aggregate_wall_tps": tokens / wall,
                "aggregate_decode_window_tps": (tokens - c) / decode,
                "mean_ttft": sum(r["ttft"] for r in rows) / c,
            }
            results.append({"summary": summary, "requests": rows})
            (ROOT / (label + ".json")).write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(summary), flush=True)
