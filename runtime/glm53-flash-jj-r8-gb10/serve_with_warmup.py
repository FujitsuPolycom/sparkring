#!/usr/bin/env python3
"""Run vLLM and expose Docker readiness only after DFlash warmup."""

from __future__ import annotations

import json
import concurrent.futures
import os
import signal
import secrets
import subprocess
import sys
import time
import threading
import urllib.request
from pathlib import Path

import warmup_dflash
import scheduler_liveness


READY_PATH = Path("/tmp/sparkring-engine-ready")

# Explicit neutral filters avoid checkpoint generation defaults selecting an
# unintended sampler path. A seed also exercises the non-FlashInfer fallback.
SAMPLING_CASES = (
    ("unfiltered", 1.0, -1, 1.0, None),
    ("temperature", 0.7, -1, 1.0, None),
    ("top-k", 1.0, 40, 1.0, None),
    ("top-p", 1.0, -1, 0.9, None),
    ("top-k-top-p", 1.0, 40, 0.9, None),
    ("seeded-top-k-top-p", 0.7, 40, 0.9, 0),
)


def _sampling_stream_result(response, deadline: float, max_tokens: int) -> dict:
    """Require one finished choice, final usage and a complete SSE terminator."""
    finish_reason = None
    usage = None
    request_id = None
    done = False
    data_lines = []
    event_type = ""
    byte_count = 0
    while True:
        warmup_dflash.remaining_seconds(deadline)
        raw = response.readline(65537)
        warmup_dflash.remaining_seconds(deadline)
        byte_count += len(raw)
        if len(raw) > 65536 or byte_count > 4 * 1024 * 1024:
            raise RuntimeError("Sampling warmup stream exceeds its response bound")
        if not raw:
            if data_lines or not done:
                raise RuntimeError("Sampling warmup stream ended before [DONE]")
            return {**usage, "finish_reason": finish_reason}
        try:
            line = raw.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError as error:
            raise RuntimeError("Sampling warmup stream is not UTF-8") from error
        if line.startswith(":"):
            continue
        if line:
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "data":
                data_lines.append(value)
            elif field == "event":
                event_type = value
            continue
        if event_type == "error":
            raise RuntimeError("Sampling warmup stream reported an error event")
        event_type = ""
        if not data_lines:
            continue
        payload = "\n".join(data_lines)
        data_lines = []
        if done:
            raise RuntimeError("Sampling warmup stream contains data after [DONE]")
        if payload == "[DONE]":
            if finish_reason is None or usage is None:
                raise RuntimeError("Sampling warmup stream has no completed choice and usage")
            done = True
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError as error:
            raise RuntimeError("Sampling warmup stream contains invalid JSON") from error
        if not isinstance(chunk, dict) or "error" in chunk:
            raise RuntimeError("Sampling warmup stream reported an error or invalid chunk")
        if (
            chunk.get("object") != "chat.completion.chunk"
            or not isinstance(chunk.get("id"), str) or not chunk["id"]
        ):
            raise RuntimeError("Sampling warmup stream has invalid completion identity")
        if request_id is None:
            request_id = chunk["id"]
        elif request_id != chunk["id"]:
            raise RuntimeError("Sampling warmup stream changed completion identity")
        choices = chunk.get("choices")
        if choices == []:
            if finish_reason is None or usage is not None:
                raise RuntimeError("Sampling warmup stream has misplaced or duplicate usage")
            usage = chunk.get("usage")
            if not isinstance(usage, dict):
                raise RuntimeError("Sampling warmup stream has no final usage")
            completion_tokens = usage.get("completion_tokens")
            if type(completion_tokens) is not int or completion_tokens <= 0:
                raise RuntimeError("Sampling warmup generated no tokens")
            prompt_tokens, total_tokens = usage.get("prompt_tokens"), usage.get("total_tokens")
            if (
                type(prompt_tokens) is not int or prompt_tokens <= 0
                or type(total_tokens) is not int
                or total_tokens != prompt_tokens + completion_tokens
                or completion_tokens > max_tokens
            ):
                raise RuntimeError("Sampling warmup stream has inconsistent token usage")
            usage = {key: usage[key] for key in (
                "prompt_tokens", "completion_tokens", "total_tokens")}
            continue
        if (
            finish_reason is not None or usage is not None
            or not isinstance(choices, list) or len(choices) != 1
            or not isinstance(choices[0], dict)
            or type(choices[0].get("index")) is not int or choices[0]["index"] != 0
            or not isinstance(choices[0].get("delta"), dict)
            or chunk.get("usage") is not None
        ):
            raise RuntimeError("Sampling warmup stream does not contain one ordered choice")
        reason = choices[0].get("finish_reason")
        if reason is not None:
            if reason not in ("stop", "length"):
                raise RuntimeError("Sampling warmup stream has an unsuccessful finish reason")
            finish_reason = reason


def warmup_sampling(
    endpoint: str,
    model: str,
    max_tokens: int,
    timeout_seconds: float,
    credential: str | None,
    concurrencies: tuple[int, ...] = (1,),
) -> dict[str, object]:
    """Complete the explicit sampler request recipe before declaring readiness."""
    deadline = warmup_dflash.make_deadline(timeout_seconds)
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("SPARKRING_STARTUP_TOKEN")
    if token:
        headers["X-Sparkring-Startup-Token"] = token
    if credential:
        headers["Authorization"] = f"Bearer {credential}"
    started = time.monotonic()
    def send_case(case, token_limit, *, peer=None, barrier=None):
        name, temperature, top_k, top_p, seed = case
        label = name if peer is None else f"{name} peer={peer}"
        body = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": f"Sampling warmup {time.monotonic_ns()} {label}. Reply briefly.",
            }],
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": 0.0,
            "n": 1,
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": False},
            "max_tokens": token_limit,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        if seed is not None:
            body["seed"] = seed
        if peer is not None:
            # A fixed decode span prevents EOS from ending a paired request
            # before the other request can enter the serving scheduler.
            body.update(ignore_eos=True, min_tokens=token_limit)
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(), headers=headers,
        )
        if barrier is not None:
            barrier.wait(timeout=warmup_dflash.remaining_seconds(deadline))
        case_started = time.perf_counter()
        with urllib.request.urlopen(
            request, timeout=warmup_dflash.remaining_seconds(deadline)
        ) as response:
            if response.status != 200 or response.headers.get_content_type() != "text/event-stream":
                raise RuntimeError(f"Sampling warmup did not return an SSE response: {name}")
            result = _sampling_stream_result(response, deadline, token_limit)
        case_finished = time.perf_counter()
        if peer is not None and (
            result["completion_tokens"] != token_limit or result["finish_reason"] != "length"
        ):
            raise RuntimeError("Concurrent sampling warmup did not complete its fixed token span")
        warmup_dflash.remaining_seconds(deadline)
        return {
            "name": name, "temperature": temperature, "top_k": top_k,
            "top_p": top_p, "seed": seed, **result,
            "elapsed_seconds": round(case_finished - case_started, 3),
        }, case_started, case_finished

    cases = [send_case(case, max_tokens)[0] for case in SAMPLING_CASES]
    concurrent_cases = []
    if any(concurrency >= 2 for concurrency in concurrencies):
        for case in SAMPLING_CASES:
            if case[0] not in ("top-k", "top-p", "top-k-top-p"):
                continue
            barrier = threading.Barrier(2)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(send_case, case, 128, peer=index, barrier=barrier)
                           for index in range(2)]
                results = [future.result(timeout=warmup_dflash.remaining_seconds(deadline))
                           for future in futures]
            overlap = min(result[2] for result in results) - max(result[1] for result in results)
            if overlap <= 0:
                raise RuntimeError("Concurrent sampling warmup HTTP requests did not overlap")
            concurrent_cases.append({
                "name": case[0], "concurrency": 2, "max_tokens": 128,
                "min_tokens": 128, "ignore_eos": True,
                "http_overlap_seconds": round(overlap, 6),
                "requests": [result[0] for result in results],
            })
    warmup_dflash.remaining_seconds(deadline)
    return {
        "schema": "sparkring-sampler-warmup/v1",
        "coverage": "request-recipe-complete" if concurrent_cases else "limited-c1-only",
        "jit_coverage_verified": False,
        "enable_thinking": True,
        "stream": True,
        "cases": cases,
        "concurrent_cases": concurrent_cases,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _positive_csv(value: str, name: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise RuntimeError(f"{name} must contain comma-separated integers") from error
    if not result or any(item <= 0 for item in result):
        raise RuntimeError(f"{name} values must be positive")
    return result


def complete_readiness(
    *,
    rank: int,
    endpoint: str,
    model: str,
    warmup_enabled: bool,
    concurrencies: tuple[int, ...],
    shape_words: tuple[int, ...],
    max_tokens: int,
    timeout_seconds: float,
    credential: str | None,
    ready_path: Path = READY_PATH,
) -> None:
    """Create the readiness marker after the rank's required work completes."""

    ready_path.unlink(missing_ok=True)
    if rank == 0:
        deadline = warmup_dflash.make_deadline(timeout_seconds)
        warmup_dflash.wait_for_api(
            endpoint, warmup_dflash.remaining_seconds(deadline), credential
        )
        result = ()
        if warmup_enabled:
            result = warmup_dflash.run_warmup(
                endpoint,
                model,
                concurrencies,
                max_tokens,
                warmup_dflash.remaining_seconds(deadline),
                shape_words,
                credential,
            )
            sampling = warmup_sampling(
                endpoint, model, max_tokens,
                warmup_dflash.remaining_seconds(deadline), credential, concurrencies
            )
            print(json.dumps({"sampling_warmup": sampling}, separators=(",", ":")))
        print(json.dumps({"dflash_warmup": result}, separators=(",", ":")))
        warmup_dflash.remaining_seconds(deadline)
    ready_path.touch()


def start_rank_liveness(
    *,
    rank: int,
    endpoint: str,
    credential: str | None,
):
    """Start the rank-zero scheduler monitor when the profile enables it."""

    if rank != 0 or os.environ.get("SPARKRING_LIVENESS_ENABLED", "1") != "1":
        return None
    return scheduler_liveness.start_liveness_service(
        metrics_url=f"{endpoint}/metrics",
        port=int(os.environ.get("SPARKRING_LIVENESS_PORT", "8016")),
        blocked_timeout_seconds=float(
            os.environ.get("SPARKRING_LIVENESS_BLOCKED_SECONDS", "60")
        ),
        output_timeout_seconds=float(
            os.environ.get("SPARKRING_LIVENESS_OUTPUT_SECONDS", "300")
        ),
        idle_kv_warn_seconds=float(
            os.environ.get("SPARKRING_IDLE_KV_WARN_SECONDS", "330")
        ),
        stale_sample_seconds=float(
            os.environ.get("SPARKRING_LIVENESS_STALE_SECONDS", "15")
        ),
        sample_interval_seconds=float(
os.environ.get("SPARKRING_LIVENESS_SAMPLE_SECONDS", "10")
        ),
        credential=credential,
    )


def main() -> int:
    READY_PATH.unlink(missing_ok=True)
    # Rotate the bypass on every process start, before the public listener exists.
    os.environ["SPARKRING_STARTUP_TOKEN"] = secrets.token_urlsafe(32)
    os.environ["SPARKRING_READY_PATH"] = str(READY_PATH)
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(Path(__file__).resolve().parent), child_env.get("PYTHONPATH")
    )))
    child = subprocess.Popen([
        "vllm", "serve", *sys.argv[1:],
        "--middleware", "startup_admission.StartupAdmission",
    ], env=child_env)
    liveness_service = None

    def forward(signum, _frame):
        if child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    try:
        rank = int(os.environ.get("SPARKRING_NODE_RANK", "0"))
        timeout_seconds = float(
            os.environ.get("DFLASH_WARMUP_TIMEOUT_SECONDS", "600")
        )
        endpoint = f"http://127.0.0.1:{os.environ.get('PORT', '8015')}"
        credential = os.environ.get("SPARKRING_WARMUP_API_KEY") or None
        complete_readiness(
            rank=rank,
            endpoint=endpoint,
            model=os.environ.get("SERVED_MODEL_NAME", "glm-5.3-flash"),
            warmup_enabled=os.environ.get("DFLASH_WARMUP", "0") == "1",
            concurrencies=_positive_csv(
                os.environ.get(
                    "DFLASH_WARMUP_CONCURRENCIES",
                    "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16",
                ),
                "DFLASH_WARMUP_CONCURRENCIES",
            ),
            shape_words=_positive_csv(
                os.environ.get("DFLASH_WARMUP_SHAPE_WORDS", "8,24,56,120,248"),
                "DFLASH_WARMUP_SHAPE_WORDS",
            ),
            max_tokens=int(os.environ.get("DFLASH_WARMUP_MAX_TOKENS", "16")),
            timeout_seconds=timeout_seconds,
            credential=credential,
        )
        liveness_service = start_rank_liveness(
            rank=rank,
            endpoint=endpoint,
            credential=credential,
        )
        return child.wait()
    except BaseException:
        READY_PATH.unlink(missing_ok=True)
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        raise
    finally:
        READY_PATH.unlink(missing_ok=True)
        if liveness_service is not None:
            liveness_service.close()


if __name__ == "__main__":
    raise SystemExit(main())
