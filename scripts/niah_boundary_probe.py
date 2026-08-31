#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deep-context NIAH boundary probe for SparkRing-style GLM lanes.

Finds the largest prompt depth an OpenAI-compatible endpoint serves cleanly.
One request at a time: builds a synthetic 4 chars/token log archive with an
8-digit verification needle at a configurable depth fraction, issues it, and
classifies the outcome:

  PASS           needle retrieved, request completed (finish=stop)
  NEEDLE-MISS    prompt processed but needle not retrievable
  TIMEOUT        wall deadline hit first

The request runs in a background thread while the main thread samples the
engine's Prometheus counter (`vllm:prompt_tokens_total` on <api>/metrics).
Probe an IDLE lane: under concurrent load the global counter advances from
unrelated traffic and the counter signal degrades to the wall deadline.

Example:
    DSPARK_API_KEY is read from the environment by default (--api-key to override)
    python3 niah_boundary_probe.py --api http://<rank-0>:8015 \
        --model GLM-5.3-Flash-Ring --depth 400000 --frac 0.5 --deadline 300

Multiple boundary walks can be driven by hashing same-seed depths, e.g.:
    for d in 100000 200000 300000 350000 375000 400000 450000; do
        python3 niah_boundary_probe.py --api ... --model ... --depth $d; done

Exit codes: 0 pass, 2 needle miss, 4 timeout, 5 request error.
"""
import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

NOUNS = ("router switch daemon ledger cipher packet kernel socket buffer "
         "index shard replica cursor token lease quorum").split()
VERBS = ("reconciles validates rotates flushes replays throttles mirrors "
         "audits caches evicts signs verifies").split()
ADJS = ("stale nightly regional encrypted ephemeral durable inbound signed "
        "pending archived").split()

ARGS = None


def build_archive(rng, target_tokens):
    def sentence(i):
        return (f"Record {i}: the {rng.choice(ADJS)} {rng.choice(NOUNS)} "
                f"{rng.choice(VERBS)} the {rng.choice(ADJS)} {rng.choice(NOUNS)}.")
    lines, size, n = [], 0, 0
    while size < target_tokens * 4:
        s = sentence(n)
        lines.append(s)
        size += len(s) + 1
        n += 1
    code = f"{rng.randint(10000000, 99999999)}"
    needle = ("\n\nIMPORTANT SECRET VERIFICATION CODE: " + code + "\n"
              "Remember this code, it will be requested later.\n\n")
    at = max(1, int(len(lines) * ARGS.frac))
    body = "\n".join(lines[:at]) + needle + "\n".join(lines[at:])
    prompt = ("The following is a long log archive. Read it carefully.\n\n" + body +
              "\n\nWhat is the IMPORTANT SECRET VERIFICATION CODE from the log? "
              "Respond with ONLY the 8-digit code and nothing else.")
    return prompt, code


def prompt_counter(metrics_url):
    try:
        for line in urllib.request.urlopen(metrics_url, timeout=8).read().decode().split("\n"):
            if line.startswith("vllm:prompt_tokens_total{"):
                return int(float(line.split()[-1]))
    except Exception:
        return None
    return None


def main():
    rng = random.Random(ARGS.seed)
    prompt, code = build_archive(rng, ARGS.depth)
    headers = {"Content-Type": "application/json"}
    if ARGS.api_key:
        headers["Authorization"] = "Bearer " + ARGS.api_key
    metrics = ARGS.api.rstrip("/") + "/metrics"

    # background thread for the request; main thread runs the watchdog
    box = {}

    def request():
        try:
            payload = {
                "model": ARGS.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 256,
            }
            req = urllib.request.Request(
                ARGS.api.rstrip("/") + "/v1/chat/completions",
                data=json.dumps(payload).encode(), headers=headers)
            # generous inner timeout; the watchdog is the effective limiter
            box["data"] = json.load(urllib.request.urlopen(req, timeout=ARGS.deadline + 300))
        except urllib.error.HTTPError as e:
            box["http_error"] = e.code
        except Exception as e:
            box["request_error"] = str(e)

    before = prompt_counter(metrics)
    print(f"prompt_counter_before={before}", flush=True)
    t0 = time.time()
    thread = threading.Thread(target=request, daemon=True)
    thread.start()
    last = before
    stuck = 0
    while time.time() - t0 < ARGS.deadline:
        thread.join(timeout=20)
        if not thread.is_alive():
            break
        cur = prompt_counter(metrics)
        print(f"t={time.time()-t0:.0f}s counter={cur}", flush=True)
        if cur is None or cur == last:
            stuck += 1
        else:
            stuck, last = 0, cur
        if ARGS.stall_ticks and stuck >= ARGS.stall_ticks:
            print("STALL: prompt counter frozen while a request is admitted "
                  "(deep prefill wedge signature)", flush=True)
            sys.exit(3)

    thread.join(timeout=60)
    elapsed = time.time() - t0
    if "http_error" in box:
        print(f"HTTPError {box['http_error']} after {elapsed:.1f}s "
              "(admission rejection is the expected behavior above the cap)",
              flush=True)
        sys.exit(5)
    if "request_error" in box:
        print(f"request error: {box['request_error']}", flush=True)
        sys.exit(5)
    if "data" not in box:
        print(f"TIMEOUT after {elapsed:.1f}s (deadline={ARGS.deadline}s)", flush=True)
        sys.exit(4)
    data = box["data"]
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    finish = (data.get("choices") or [{}])[0].get("finish_reason")
    usage = data.get("usage", {})
    present = code in text
    print(f"elapsed={elapsed:.1f}s prompt_tokens={usage.get('prompt_tokens', '?')} "
          f"completion={usage.get('completion_tokens', '?')} "
          f"finish={finish} needle_present={present} code={code}", flush=True)

    if present and finish == "stop":
        print("verdict=PASS", flush=True)
        sys.exit(0)
    print("verdict=NEEDLE-MISS", flush=True)
    sys.exit(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--api", required=True, help="OpenAI-compatible base URL of the rank-0 head")
    ap.add_argument("--model", required=True)
    ap.add_argument("--depth", type=int, required=True, help="target prompt tokens")
    ap.add_argument("--frac", type=float, default=0.5, help="needle position fraction 0..1")
    ap.add_argument("--deadline", type=int, default=900, help="wall deadline seconds")
    ap.add_argument("--stall-ticks", type=int, default=4,
                    help="counter-frozen ticks (polls run at the 20s join cadence) before STALL exit; 0 disables")
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--api-key", default=os.environ.get("DSPARK_API_KEY", ""),
                    help="bearer key (default $DSPARK_API_KEY)")
    ARGS = ap.parse_args()
    main()
