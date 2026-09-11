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
  STALL          opt-in watchdog: global prompt counter unchanged

The request runs in a background thread. By default the wall deadline is
the authoritative limiter; the prompt-counter watchdog is opt-in via
--stall-ticks N: it samples the engine's Prometheus counter
(`vllm:prompt_tokens_total` on <api>/metrics) from the main thread and
reports an unchanged counter after N successful polls. It does not establish
request admission or diagnose a prefill failure. Use an idle endpoint: under
concurrent load the global counter advances from unrelated traffic, and
this vLLM runtime may leave the counter unchanged until a healthy long
prefill completes, so watchdog-off is the default.

--depth is an approximate prompt-token target: the 4 chars/token archive
construction is tokenizer-dependent, and the measured prompt size is the
final `usage.prompt_tokens` value reported with the verdict.

Example:
    DSPARK_API_KEY is read from the environment by default (--api-key to override)
    python3 niah_boundary_probe.py --api http://<rank-0>:8015 \
        --model GLM-5.3-Flash-Ring --depth 400000 --frac 0.5 --deadline 300

Reuse --seed for reproducible filler lines across depths. The inserted needle
and its position vary with depth; complete prompts are not prefix extensions:
    for d in 100000 200000 300000 350000 375000 400000 450000; do
        python3 niah_boundary_probe.py --api ... --model ... --depth $d; done

Exit codes: 0 pass, 2 needle miss, 3 stall (opt-in watchdog), 4 timeout, 5 request error.
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


def prompt_counter(metrics_url, timeout=8):
    try:
        for line in urllib.request.urlopen(metrics_url, timeout=timeout).read().decode().split("\n"):
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

    # background thread for the request; main thread samples the counter (watchdog is opt-in)
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
            # generous inner timeout; the wall deadline is the effective limiter by default
            box["data"] = json.load(urllib.request.urlopen(req, timeout=ARGS.deadline + 300))
        except urllib.error.HTTPError as e:
            box["http_error"] = e.code
        except Exception as e:
            box["request_error"] = str(e)
        finally:
            box["completed_at"] = time.monotonic()

    before = prompt_counter(metrics)
    print(f"prompt_counter_before={before}", flush=True)
    t0 = time.monotonic()
    thread = threading.Thread(target=request, daemon=True)
    thread.start()
    last = before
    stuck = 0
    deadline = t0 + ARGS.deadline
    while thread.is_alive():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        thread.join(timeout=min(20, remaining))
        if not thread.is_alive():
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if not ARGS.stall_ticks:
            continue
        cur = prompt_counter(metrics, timeout=min(8, remaining))
        print(f"t={time.monotonic()-t0:.0f}s counter={cur}", flush=True)
        if cur is None:
            stuck = 0
        elif last is not None and cur == last:
            stuck += 1
        else:
            stuck = 0
        last = cur
        if not thread.is_alive():
            break
        if stuck >= ARGS.stall_ticks:
            print("STALL: global prompt counter unchanged during the request; "
                  "this does not prove admission or a prefill failure", flush=True)
            sys.exit(3)

    elapsed = time.monotonic() - t0
    if box.get("completed_at", float("inf")) > deadline:
        print(f"TIMEOUT after {elapsed:.1f}s (deadline={ARGS.deadline}s)", flush=True)
        sys.exit(4)
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
    ap.add_argument("--depth", type=int, required=True,
                    help="approximate prompt-token target (tokenizer-dependent; the measured size is the response's usage.prompt_tokens)")
    ap.add_argument("--frac", type=float, default=0.5, help="needle position fraction 0..1")
    ap.add_argument("--deadline", type=int, default=900, help="wall deadline seconds")
    ap.add_argument("--stall-ticks", type=int, default=0,
                    help="opt-in stall watchdog: exit 3 after this many prompt-counter polls at the 20s cadence show no advance; 0 (default) disables it and the wall deadline is authoritative")
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--api-key", default=os.environ.get("DSPARK_API_KEY", ""),
                    help="bearer key (default $DSPARK_API_KEY)")
    ARGS = ap.parse_args()
    if ARGS.depth <= 0 or ARGS.deadline <= 0 or ARGS.stall_ticks < 0 or not 0 <= ARGS.frac <= 1:
        ap.error("depth/deadline must be positive, stall-ticks nonnegative and frac in 0..1")
    main()
