"""Shared idle-service semantic and cold-prefill checks.

The two CLI entrypoints retain their recorded clock and container-selection
contracts. Clock selection preserves those interfaces; it is not an A/B claim.
No environment validation or network work occurs at import.
"""

import argparse
import os
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request
import uuid
import math

from performance.harnesses.validation.prefill_probe import events


class Harness:
    def __init__(self, *, require_prefix=False, clock_name="monotonic"):
        self.require_prefix = require_prefix
        self.clock_name = clock_name
        self.clock = getattr(time, clock_name)
        self.base = os.environ.get("BENCH_API_BASE", "http://127.0.0.1:8000")
        self.model = os.environ.get("BENCH_MODEL", "glm-5.3-flash-spark")
        self.container = (
            "sparkring-model-r0"
            if require_prefix
            else os.environ.get("BENCH_RANK0_CONTAINER", "sparkring-model-r0")
        )

    def validate_container(self, container=None):
        value = self.container if container is None else container
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError(
                "Container name must use ASCII letters, digits, dot, underscore or hyphen"
            )

    def cached_prompt_tokens(self, usage):
        """Missing accounting cannot establish cold work or successful cache reuse."""
        details = usage.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        prompt = usage.get("prompt_tokens")
        if type(prompt) is not int or prompt <= 0:
            raise ValueError("Usage must provide a positive integer prompt_tokens")
        if type(cached) is not int or not 0 <= cached <= prompt:
            raise ValueError("Usage must provide a nonnegative integer cached_tokens")
        return cached

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return r.read().decode()

    def post(self, path, body):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.load(r)

    def ready(self):
        self.validate_container()
        status = subprocess.check_output(
            [
                "ssh",
                os.environ.get("BENCH_RANK0_SSH", "rank0"),
                "docker inspect --format '{{.State.Health.Status}}' " + self.container,
            ],
            text=True,
        ).strip()
        if status != "healthy":
            return False
        try:
            metrics = self.get("/metrics")
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
        return all(
            rows and all(math.isfinite(value) and value == 0 for value in rows)
            for rows in values.values()
        )

    def calibrate(self, tokens, fact):
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
            count = self.post(
                "/tokenize",
                {"model": self.model, "messages": msg, "add_generation_prompt": True},
            )["count"]
            if count == tokens:
                return msg
            words += tokens - count
        raise ValueError("Exact prompt calibration failed")

    def semantic(self, tokens, fact):
        messages = self.calibrate(tokens, fact)
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
            response = self.post(
                "/v1/chat/completions",
                {
                    "model": self.model,
                    "messages": msg,
                    "max_tokens": 384,
                    "temperature": 0,
                    "top_p": 1,
                },
            )
            choices = response.get("choices")
            if (
                not isinstance(choices, list)
                or len(choices) != 1
                or not isinstance(choices[0], dict)
            ):
                raise ValueError("Completion must contain exactly one choice")
            choice = choices[0]
            answer = (choice["message"].get("content") or "").strip()
            usage = response["usage"]
            cached = self.cached_prompt_tokens(usage)
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

    def prefill(self, tokens):
        msg = self.calibrate(tokens, "STONE-7482")
        body = {
            "model": self.model,
            "messages": msg,
            "max_tokens": 1,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        request = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = self.clock()
        first = None
        usage = None
        finish = None
        with urllib.request.urlopen(request, timeout=180) as response:
            for chunk in events(response):
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    finish = choice.get("finish_reason") or finish
                    delta = choice.get("delta", {})
                    if first is None and any(
                        delta.get(k)
                        for k in ("content", "reasoning", "reasoning_content")
                    ):
                        first = self.clock() - started
        if (
            first is None
            or not math.isfinite(first)
            or first <= 0
            or not isinstance(usage, dict)
        ):
            raise RuntimeError("No streamed token or final usage")
        if (
            type(usage.get("prompt_tokens")) is not int
            or usage["prompt_tokens"] != tokens
        ):
            raise ValueError(
                "Reported prompt token count differs from calibrated request"
            )
        if self.cached_prompt_tokens(usage) != 0:
            raise ValueError("Cold-prefill timing requires cached_tokens=0")
        if (
            finish not in ("stop", "length")
            or type(usage.get("completion_tokens")) is not int
            or usage["completion_tokens"] < 1
        ):
            raise ValueError("Prefill stream lacks normal completion and output usage")
        return {
            "tokens": tokens,
            "ttft_seconds": first,
            "tokens_per_second": tokens / first,
            "usage": usage,
        }

    def main(self):
        p = argparse.ArgumentParser()
        p.add_argument("phase", choices=["semantic", "prefill"])
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--sizes", default="16384,10240,32768,65536,8192")
        p.add_argument("--samples", type=int, default=3)
        if self.require_prefix:
            p.add_argument("--container-prefix", required=True)
        args = p.parse_args()
        try:
            sizes = list(map(int, args.sizes.split(",")))
        except ValueError:
            p.error("Sizes must be comma-separated positive integers")
        if args.samples < 1 or not sizes or any(value <= 0 for value in sizes):
            p.error("Samples and sizes must be positive")
        if args.output.exists() or args.output.is_symlink():
            p.error("Output receipt must not already exist")
        if self.require_prefix:
            if not re.fullmatch(r"sparkring-[a-z0-9-]+", args.container_prefix):
                p.error("Container prefix must match sparkring-[a-z0-9-]+")
            container = args.container_prefix + "-r0"
        else:
            container = self.container
        try:
            self.validate_container(container)
        except ValueError as error:
            p.error(str(error))
        self.container = container
        if self.require_prefix:
            print(
                "Timing clock: " + str(time.get_clock_info(self.clock_name)), flush=True
            )
        with args.output.open("x", encoding="utf-8") as output:
            output.write(json.dumps({"phase": args.phase, "rows": []}) + "\n")
        deadline = time.monotonic() + 1200
        announced = False
        while not self.ready():
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
                json.dumps({"phase": args.phase, "rows": rows}, indent=2),
                encoding="utf-8",
            )

        if args.phase == "semantic":
            for i, tokens in enumerate(sizes):
                for row in self.semantic(tokens, "RIVER-" + str(5938 + i)):
                    rows.append(row)
                    save()
        else:
            for phase in ("warm", "measured"):
                for sample in range(1 if phase == "warm" else args.samples):
                    for tokens in sizes:
                        if not self.ready():
                            raise RuntimeError(
                                "Concurrent activity detected before timing sample"
                            )
                        row = self.prefill(tokens) | {"phase": phase, "sample": sample}
                        rows.append(row)
                        save()
                        print(
                            json.dumps({k: v for k, v in row.items() if k != "usage"}),
                            flush=True,
                        )
