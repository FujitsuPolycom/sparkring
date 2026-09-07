"""Bounded semantic and cold-prefill checks, gated on completed API warmup."""

import os
import argparse
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request
import uuid

CONTAINER_PREFIX = "sparkring-model"
BASE = os.environ.get("BENCH_API_BASE", "http://127.0.0.1:8000")
MODEL = os.environ.get("BENCH_MODEL", "glm-5.3-flash-spark")


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return r.read().decode()


def post(path, body):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def ready():
    status = subprocess.check_output(
        [
            "ssh",
            os.environ.get("BENCH_RANK0_SSH", "rank0"),
            "docker inspect --format '{{.State.Health.Status}}' "
            + CONTAINER_PREFIX
            + "-r0",
        ],
        text=True,
    ).strip()
    if status != "healthy":
        return False
    try:
        metrics = get("/metrics")
    except (OSError, TimeoutError):
        # Docker health may become ready before the API socket is reachable.
        # Never treat that interval as a benchmark sample.
        return False
    values = {
        key: [
            float(line.rsplit(" ", 1)[-1])
            for line in metrics.splitlines()
            if line.startswith("vllm:" + key + "{")
            or line.startswith("vllm:" + key + " ")
        ]
        for key in ("num_requests_running", "num_requests_waiting")
    }
    return all(rows and sum(rows) == 0 for rows in values.values())


def calibrate(tokens, fact):
    prefix = (
        "Test "
        + uuid.uuid4().hex
        + ". The project code is "
        + fact
        + ". Remember it.\n"
    )
    suffix = "\nWhat is the project code? Reply with just the code."
    words = tokens
    for _ in range(12):
        text = (
            prefix
            + " ".join(
                (["alpha", "beta", "gamma", "delta"] * ((words + 3) // 4))[:words]
            )
            + suffix
        )
        msg = [{"role": "user", "content": text}]
        count = post(
            "/tokenize",
            {"model": MODEL, "messages": msg, "add_generation_prompt": True},
        )["count"]
        if count == tokens:
            return msg
        words += tokens - count
    raise ValueError("Exact prompt calibration failed")


def semantic(tokens, fact):
    messages = calibrate(tokens, fact)
    rows = []
    for phase in ("fresh", "repeated", "extended"):
        msg = (
            messages
            if phase != "extended"
            else [
                {
                    "role": "user",
                    "content": messages[0]["content"]
                    + "\nFinal instruction: give exactly the project code.",
                }
            ]
        )
        started = time.monotonic()
        response = post(
            "/v1/chat/completions",
            {
                "model": MODEL,
                "messages": msg,
                "max_tokens": 384,
                "temperature": 0,
                "top_p": 1,
            },
        )
        choice = response["choices"][0]
        answer = (choice["message"].get("content") or "").strip()
        usage = response["usage"]
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        row = {
            "tokens": tokens,
            "phase": phase,
            "seconds": time.monotonic() - started,
            "response": response,
            "answer_ok": answer == fact and choice["finish_reason"] == "stop",
            "cache_ok": cached == 0 if phase == "fresh" else cached > 0,
        }
        row["pass"] = row["answer_ok"] and row["cache_ok"]
        if phase == "fresh":
            row["pass"] = row["pass"] and usage["prompt_tokens"] == tokens
        rows.append(row)
        print(
            json.dumps(
                {k: v for k, v in row.items() if k != "response"}
                | {"answer": answer, "cached_tokens": cached}
            ),
            flush=True,
        )
        yield row
        if not row["pass"]:
            raise RuntimeError("Semantic/cache check failed; inspect response")


def prefill(tokens):
    msg = calibrate(tokens, "STONE-7482")
    body = {
        "model": MODEL,
        "messages": msg,
        "max_tokens": 1,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = None
    usage = None
    with urllib.request.urlopen(request, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if raw == b"[DONE]":
                break
            chunk = json.loads(raw)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if first is None and any(
                    delta.get(k) for k in ("content", "reasoning", "reasoning_content")
                ):
                    first = time.perf_counter() - started
    if first is None or usage is None:
        raise RuntimeError("No streamed token or final usage")
    assert usage["prompt_tokens"] == tokens
    assert usage.get("prompt_tokens_details", {}).get("cached_tokens", 0) == 0
    return {
        "tokens": tokens,
        "ttft_seconds": first,
        "tokens_per_second": tokens / first,
        "usage": usage,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["semantic", "prefill"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sizes", default="16384,10240,32768,65536,8192")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--container-prefix", required=True)
    args = p.parse_args()
    assert not args.output.exists()
    global CONTAINER_PREFIX
    CONTAINER_PREFIX = args.container_prefix
    if not re.fullmatch(r"sparkring-[a-z0-9-]+", CONTAINER_PREFIX):
        raise ValueError("Unexpected container prefix")
    print("Timing clock: " + str(time.get_clock_info("perf_counter")), flush=True)
    deadline = time.monotonic() + 1200
    announced = False
    while not ready():
        if time.monotonic() > deadline:
            raise TimeoutError("Warmup/idle readiness deadline")
        if not announced:
            print(
                "Waiting for completed warmup and zero running/waiting requests.",
                flush=True,
            )
            announced = True
        time.sleep(5)
    print("Warmup complete; server idle. Starting controlled checks.", flush=True)
    rows = []

    def save():
        args.output.write_text(
            json.dumps({"phase": args.phase, "rows": rows}, indent=2), encoding="utf-8"
        )

    sizes = list(map(int, args.sizes.split(",")))
    if args.phase == "semantic":
        for i, tokens in enumerate(sizes):
            for row in semantic(tokens, "RIVER-" + str(5938 + i)):
                rows.append(row)
                save()
    else:
        for phase in ("warm", "measured"):
            for sample in range(1 if phase == "warm" else args.samples):
                for tokens in sizes:
                    if not ready():
                        raise RuntimeError(
                            "Concurrent activity detected before timing sample"
                        )
                    row = prefill(tokens) | {"phase": phase, "sample": sample}
                    rows.append(row)
                    save()
                    print(
                        json.dumps({k: v for k, v in row.items() if k != "usage"}),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
