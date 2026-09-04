#!/usr/bin/env python3
"""Run vLLM and expose Docker readiness only after DFlash warmup."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import warmup_dflash
import scheduler_liveness


READY_PATH = Path("/tmp/sparkring-engine-ready")


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
        warmup_dflash.wait_for_api(endpoint, timeout_seconds, credential)
        result = ()
        if warmup_enabled:
            result = warmup_dflash.run_warmup(
                endpoint,
                model,
                concurrencies,
                max_tokens,
                timeout_seconds,
                shape_words,
                credential,
            )
        print(json.dumps({"dflash_warmup": result}, separators=(",", ":")))
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
        idle_kv_warn_seconds=float(
            os.environ.get("SPARKRING_IDLE_KV_WARN_SECONDS", "330")
        ),
        stale_sample_seconds=float(
            os.environ.get("SPARKRING_LIVENESS_STALE_SECONDS", "15")
        ),
        sample_interval_seconds=float(
            os.environ.get("SPARKRING_LIVENESS_SAMPLE_SECONDS", "2")
        ),
        credential=credential,
    )


def main() -> int:
    READY_PATH.unlink(missing_ok=True)
    child = subprocess.Popen(["vllm", "serve", *sys.argv[1:]])
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
                os.environ.get("DFLASH_WARMUP_CONCURRENCIES", "1,2,4,8,16"),
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
        if liveness_service is not None:
            liveness_service.close()


if __name__ == "__main__":
    raise SystemExit(main())
