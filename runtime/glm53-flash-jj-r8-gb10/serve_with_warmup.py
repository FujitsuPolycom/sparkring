#!/usr/bin/env python3
"""Run vLLM and expose Docker readiness only after DFlash warmup."""

from __future__ import annotations

import json
import os
import signal
import secrets
import subprocess
import sys
import time
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


def warmup_sampling(
    endpoint: str,
    model: str,
    max_tokens: int,
    timeout_seconds: float,
    credential: str | None,
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
    cases = []
    for name, temperature, top_k, top_p, seed in SAMPLING_CASES:
        body = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": f"Sampling warmup {time.monotonic_ns()} {name}. Reply briefly.",
            }],
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": 0.0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        if seed is not None:
            body["seed"] = seed
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(), headers=headers,
        )
        case_started = time.monotonic()
        with urllib.request.urlopen(
            request, timeout=warmup_dflash.remaining_seconds(deadline)
        ) as response:
            result = json.load(response)
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Sampling warmup response has no completion: {name}")
        usage = result.get("usage")
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if type(completion_tokens) is not int or completion_tokens <= 0:
            raise RuntimeError(f"Sampling warmup generated no tokens: {name}")
        warmup_dflash.remaining_seconds(deadline)
        cases.append({
            "name": name, "temperature": temperature, "top_k": top_k,
            "top_p": top_p, "seed": seed, "completion_tokens": completion_tokens,
            "elapsed_seconds": round(time.monotonic() - case_started, 3),
        })
    return {
        "schema": "sparkring-sampler-warmup/v1",
        "coverage": "request-recipe-complete",
        "jit_coverage_verified": False,
        "enable_thinking": True,
        "cases": cases,
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
                warmup_dflash.remaining_seconds(deadline), credential
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
