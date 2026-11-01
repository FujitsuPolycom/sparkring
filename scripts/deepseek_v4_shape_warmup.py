#!/usr/bin/env python3
"""Exercise the DeepSeek-V4 request shapes that the engine compiles at first use.

A shape first reached by client traffic compiles while every rank waits in a
collective, and the distributed watchdog can terminate the waiting rank. This
program walks the shape buckets a DeepSeek-V4-Flash deployment can schedule
after its API reports ready and before the launcher hands the endpoint to
clients.

The bucket rule is provisional. Measurements on an earlier DeepSeek-V4-Flash
serving image found the DSpark input-preparation kernel keyed by
``min(block_size, next_pow2(scheduled_tokens + num_speculative_tokens + 1))``,
not by request concurrency; those measurements used a different serving image
than this repository pins. The sweep covers that rule, and the calibration
procedure in ``docs/operations/deepseek-0731.md`` establishes the actual keys
on a deployment before it enables the phase by default.

Every request starts with a unique nonce so prefix caching cannot serve a
cached prefill in place of the shape being warmed. Chat requests carry the
nonce in the user message. Prefill and token-ladder requests use the
completion endpoint, whose prompt has no chat-template overhead, so their
prompt length can land exactly on a compile bucket.

The phase is non-fatal: failed requests are recorded, the remaining arms still
run, and the exit status reports whether any arm failed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# Recorded pair serving completed requests through 8,193 input tokens. A
# multi-chunk arm longer than this boundary exercises the second prefill chunk
# at the deployed `--max-num-batched-tokens`.
MULTI_CHUNK_BOUNDARY_TOKENS = 8192

THINKING_OFF = {"enable_thinking": False}
CHAT_KIND = "chat"
COMPLETION_KIND = "completion"


class WarmupError(RuntimeError):
    """A warmup step failed; the caller records it and continues."""


class UnreachableTarget(WarmupError):
    """The nonce alone is longer than an arm's exact prompt target."""


@dataclass(frozen=True)
class WarmupConfig:
    """Deployment values the sweep must match, all taken from the rank env."""

    endpoint: str
    model: str
    max_num_seqs: int
    block_size: int
    num_speculative_tokens: int
    max_num_batched_tokens: int
    max_model_len: int
    bearer: str | None = None

    @property
    def max_tokens(self) -> int:
        """One draft round: the speculative depth plus its verification token."""
        return self.num_speculative_tokens + 1

    @property
    def max_prompt_tokens(self) -> int:
        return self.max_model_len - self.max_tokens


@dataclass(frozen=True)
class WarmupArm:
    """One batch of requests at one endpoint and, for completions, one target."""

    label: str
    batch: int
    kind: str
    prompt_tokens: int | None = None
    compile_bucket: int | None = None
    thinking_off: bool = False


def _positive(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WarmupError(f"{name} must be a positive integer: {value!r}")
    return value


def next_power_of_two(value: int) -> int:
    _positive(value, "value")
    return 1 << (value - 1).bit_length()


def compile_bucket(prompt_tokens: int, config: WarmupConfig) -> int:
    """Return the provisional DSpark input-preparation compile bucket."""
    _positive(prompt_tokens, "prompt_tokens")
    return min(
        config.block_size,
        next_power_of_two(prompt_tokens + config.num_speculative_tokens + 1),
    )


def concurrency_ladder(max_num_seqs: int) -> tuple[int, ...]:
    """Return powers of two up to the sequence limit, then the limit itself."""
    _positive(max_num_seqs, "max_num_seqs")
    rungs: list[int] = []
    rung = 1
    while rung < max_num_seqs:
        rungs.append(rung)
        rung *= 2
    rungs.append(max_num_seqs)
    return tuple(rungs)


def bucket_targets(config: WarmupConfig) -> tuple[tuple[int, int], ...]:
    """Return (compile bucket, exact prompt tokens) through the block size."""
    buckets = []
    bucket = 1
    while bucket < config.block_size:
        buckets.append(bucket)
        bucket *= 2
    buckets.append(config.block_size)
    targets = []
    for bucket in buckets:
        tokens = bucket - (config.num_speculative_tokens + 1)
        if tokens < 1:
            continue
        if compile_bucket(tokens, config) == bucket:
            targets.append((bucket, tokens))
    return tuple(targets)


def build_sweep(
    config: WarmupConfig,
) -> tuple[tuple[WarmupArm, ...], tuple[str, ...]]:
    """Return the deterministic sweep and the labels the model length excludes."""
    _positive(config.max_model_len, "max_model_len")
    limit = config.max_prompt_tokens
    if limit < 1:
        raise WarmupError(
            "max_model_len leaves no room for a warmup prompt: "
            f"{config.max_model_len} tokens with {config.max_tokens} generated"
        )
    ladder = bucket_targets(config)
    if not ladder:
        raise WarmupError(
            "block_size and speculative depth produce no warmup bucket: "
            f"block_size={config.block_size} "
            f"num_speculative_tokens={config.num_speculative_tokens}"
        )
    arms: list[WarmupArm] = []
    skipped: list[str] = []
    for concurrency in concurrency_ladder(config.max_num_seqs):
        arms.append(
            WarmupArm(
                label=f"concurrency-{concurrency}",
                batch=concurrency,
                kind=CHAT_KIND,
            )
        )
    for label, target in (
        ("prefill-medium", config.max_num_batched_tokens // 2),
        (
            "prefill-multi-chunk",
            MULTI_CHUNK_BOUNDARY_TOKENS + config.max_num_batched_tokens,
        ),
    ):
        if target > limit or target < 1:
            skipped.append(f"{label}: {target} prompt tokens exceed max_model_len")
            continue
        arms.append(
            WarmupArm(
                label=label,
                batch=1,
                kind=COMPLETION_KIND,
                prompt_tokens=target,
                compile_bucket=compile_bucket(target, config),
            )
        )
    arms.append(
        WarmupArm(label="thinking-off", batch=1, kind=CHAT_KIND, thinking_off=True)
    )
    for bucket, tokens in ladder:
        label = f"tokens-{tokens}-bucket-{bucket}"
        if tokens > limit:
            skipped.append(f"{label}: exceeds max_model_len")
            continue
        arms.append(
            WarmupArm(
                label=label,
                batch=1,
                kind=COMPLETION_KIND,
                prompt_tokens=tokens,
                compile_bucket=bucket,
            )
        )
    return tuple(arms), tuple(skipped)


def read_bearer(path: str) -> str:
    """Return the first line that does not start with whitespace.

    The cycle controller's readiness probe reads the same file with this rule,
    so one key file serves both the probe and the warmup.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise WarmupError(f"cannot read the API key file {path}: {error}") from error
    for line in text.splitlines():
        if line and not line[0].isspace():
            return line.strip()
    raise WarmupError(f"the API key file has no non-empty line: {path}")


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise WarmupError("the warmup deadline expired")
    return remaining


def _post_json(
    endpoint: str,
    path: str,
    payload: dict,
    *,
    bearer: str | None,
    timeout: float,
) -> dict:
    headers = {"Content-Type": "application/json"}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    request = urllib.request.Request(
        endpoint.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        document = json.load(response)
    if not isinstance(document, dict):
        raise WarmupError(f"{path} returned no JSON object")
    return document


def wait_for_health(endpoint: str, timeout_seconds: float) -> None:
    """Block until the API's /health answers 200 or the budget expires."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise WarmupError("the readiness budget must be finite and positive")
    deadline = time.monotonic() + timeout_seconds
    url = endpoint.rstrip("/") + "/health"
    while True:
        try:
            with urllib.request.urlopen(
                url, timeout=min(5.0, max(0.1, deadline - time.monotonic()))
            ) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WarmupError(
                f"/health did not answer 200 within {timeout_seconds:g} seconds"
            )
        time.sleep(min(1.0, remaining))


def tokenize_chat(
    config: WarmupConfig,
    messages: list[dict],
    *,
    thinking_off: bool,
    timeout: float,
) -> int:
    """Return the rendered chat prompt length that the engine will schedule."""
    payload: dict = {
        "model": config.model,
        "messages": messages,
        "add_generation_prompt": True,
    }
    if thinking_off:
        payload["chat_template_kwargs"] = dict(THINKING_OFF)
    document = _post_json(
        config.endpoint, "/tokenize", payload, bearer=config.bearer, timeout=timeout
    )
    return _token_count(document)


def tokenize_completion(config: WarmupConfig, prompt: str, *, timeout: float) -> int:
    """Return the token length of a raw completion prompt."""
    document = _post_json(
        config.endpoint,
        "/tokenize",
        {"model": config.model, "prompt": prompt},
        bearer=config.bearer,
        timeout=timeout,
    )
    return _token_count(document)


def _token_count(document: dict) -> int:
    count = document.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise WarmupError("the tokenize response has no positive token count")
    return count


def chat_content(nonce: str) -> str:
    return f"{nonce} Reply with one word."


def _completion_prompt(nonce: str, repeats: int) -> str:
    # The nonce leads the prompt so no token shares a cached prefix block with
    # another request. No instruction follows: completion prompts need none,
    # and the shorter skeleton keeps the low buckets reachable.
    return f"{nonce}\n" + "warmup " * repeats


def resolve_completion_prompt(
    config: WarmupConfig,
    arm: WarmupArm,
    nonce: str,
    *,
    timeout: float,
) -> str:
    """Return a prompt whose token count is exactly the arm's target.

    Filler repetitions are searched with /tokenize; a length the nonce cannot
    reach is reported separately, and a length no filler count produces fails
    the arm instead of approximating the bucket.
    """
    target = arm.prompt_tokens
    if target is None:
        raise WarmupError(f"{arm.label} has no completion target")

    def measure(repeats: int) -> int:
        return tokenize_completion(
            config, _completion_prompt(nonce, repeats), timeout=timeout
        )

    skeleton = measure(0)
    if skeleton == target:
        return _completion_prompt(nonce, 0)
    if skeleton > target:
        raise UnreachableTarget(
            f"the nonce prompt alone uses {skeleton} tokens; target {target} "
            "cannot be reached"
        )
    low, high = 1, max(target, 1)
    while low <= high:
        middle = (low + high) // 2
        tokens = measure(middle)
        if tokens == target:
            return _completion_prompt(nonce, middle)
        if tokens < target:
            low = middle + 1
        else:
            high = middle - 1
    raise WarmupError(
        f"no filler length tokenizes to exactly {target} tokens; "
        "the arm cannot cover its compile bucket"
    )


def send_chat_request(
    config: WarmupConfig,
    content: str,
    *,
    thinking_off: bool,
    timeout: float,
) -> None:
    payload: dict = {
        "model": config.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": config.max_tokens,
    }
    if thinking_off:
        payload["chat_template_kwargs"] = dict(THINKING_OFF)
    document = _post_json(
        config.endpoint,
        "/v1/chat/completions",
        payload,
        bearer=config.bearer,
        timeout=timeout,
    )
    choices = document.get("choices")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or document.get("error") is not None
        or not isinstance(choices[0], dict)
    ):
        raise WarmupError("the warmup response has no single completion")
    choice = choices[0]
    message = choice.get("message")
    if (
        choice.get("finish_reason") not in ("stop", "length")
        or not isinstance(message, dict)
        or not any(
            isinstance(message.get(key), str)
            for key in ("content", "reasoning", "reasoning_content")
        )
    ):
        raise WarmupError("the warmup response did not complete generation")


def send_completion_request(
    config: WarmupConfig,
    prompt: str,
    *,
    timeout: float,
) -> None:
    document = _post_json(
        config.endpoint,
        "/v1/completions",
        {
            "model": config.model,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": config.max_tokens,
        },
        bearer=config.bearer,
        timeout=timeout,
    )
    choices = document.get("choices")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or document.get("error") is not None
        or not isinstance(choices[0], dict)
    ):
        raise WarmupError("the warmup response has no single completion")
    choice = choices[0]
    if choice.get("finish_reason") not in ("stop", "length") or not isinstance(
        choice.get("text"), str
    ):
        raise WarmupError("the warmup response did not complete generation")


def make_nonce(run_id: str, arm: WarmupArm, index: int) -> str:
    return f"{run_id}-{arm.label}-{index:02d}"


def _arm_prompt(
    config: WarmupConfig,
    arm: WarmupArm,
    nonce: str,
    deadline: float,
) -> tuple[str, int]:
    """Return the request content and its verified prompt token count."""
    timeout = _remaining(deadline)
    if arm.kind == COMPLETION_KIND:
        prompt = resolve_completion_prompt(config, arm, nonce, timeout=timeout)
        return prompt, arm.prompt_tokens or 0
    content = chat_content(nonce)
    measured = tokenize_chat(
        config,
        [{"role": "user", "content": content}],
        thinking_off=arm.thinking_off,
        timeout=timeout,
    )
    return content, measured


def _run_arm(
    config: WarmupConfig,
    arm: WarmupArm,
    run_id: str,
    deadline: float,
) -> tuple[dict, list[dict], list[str]]:
    failures: list[dict] = []
    skipped: list[str] = []
    prompts: list[tuple[int, str, int]] = []
    for index in range(arm.batch):
        nonce = make_nonce(run_id, arm, index)
        try:
            content, measured = _arm_prompt(config, arm, nonce, deadline)
        except UnreachableTarget as error:
            skipped.append(f"{arm.label}: {error}")
            break
        except Exception as error:  # One failed request never aborts the sweep.
            failures.append(
                {
                    "arm": arm.label,
                    "request": index,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        prompts.append((index, content, measured))
    started = time.perf_counter()
    completed = 0
    if prompts:
        barrier = threading.Barrier(len(prompts))

        def fire(item: tuple[int, str, int]) -> str | None:
            _, content, _ = item
            try:
                barrier.wait(timeout=_remaining(deadline))
                if arm.kind == COMPLETION_KIND:
                    send_completion_request(config, content, timeout=_remaining(deadline))
                else:
                    send_chat_request(
                        config,
                        content,
                        thinking_off=arm.thinking_off,
                        timeout=_remaining(deadline),
                    )
                return None
            except Exception as error:  # Reported as a failed request.
                return f"{type(error).__name__}: {error}"

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(prompts)
        ) as executor:
            outcomes = list(executor.map(fire, prompts))
        for (index, _, _), error in zip(prompts, outcomes):
            if error is None:
                completed += 1
            else:
                failures.append({"arm": arm.label, "request": index, "error": error})
    measured_tokens = sorted({item[2] for item in prompts})
    buckets = sorted({compile_bucket(tokens, config) for tokens in measured_tokens})
    return (
        {
            "label": arm.label,
            "kind": arm.kind,
            "batch": arm.batch,
            "prompt_tokens": arm.prompt_tokens,
            "measured_tokens": measured_tokens,
            "compile_buckets": buckets,
            "thinking_off": arm.thinking_off,
            "completed": completed,
            "failed": len(failures),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
        failures,
        skipped,
    )


def run_warmup(
    config: WarmupConfig,
    arms: Sequence[WarmupArm],
    *,
    wait_seconds: float,
    timeout_seconds: float,
    skipped: Sequence[str] = (),
) -> dict:
    """Run every arm, collecting failures instead of aborting."""
    if not arms:
        raise WarmupError("the sweep has no arms")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise WarmupError("the sweep budget must be finite and positive")
    summary = {
        "model": config.model,
        "max_num_seqs": config.max_num_seqs,
        "block_size": config.block_size,
        "num_speculative_tokens": config.num_speculative_tokens,
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "max_model_len": config.max_model_len,
        "max_tokens": config.max_tokens,
        "authenticated": config.bearer is not None,
        "arms": [],
        "failures": [],
        "skipped": list(skipped),
        "status": "failed",
    }
    try:
        wait_for_health(config.endpoint, wait_seconds)
    except Exception as error:  # A readiness failure is reported, not raised.
        summary["failures"].append(
            {
                "arm": "readiness",
                "request": 0,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        return summary
    deadline = time.monotonic() + timeout_seconds
    run_id = f"s{secrets.token_hex(4)}"
    for arm in arms:
        try:
            result, failures, arm_skips = _run_arm(config, arm, run_id, deadline)
        except Exception as error:  # An arm failure is recorded.
            summary["failures"].append(
                {
                    "arm": arm.label,
                    "request": 0,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        summary["arms"].append(result)
        summary["failures"].extend(failures)
        summary["skipped"].extend(arm_skips)
    completed = sum(arm["completed"] for arm in summary["arms"])
    if completed and not summary["failures"]:
        summary["status"] = "complete"
    elif completed:
        summary["status"] = "partial"
    return summary


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer: {text}")
    return value


def _positive_seconds(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive seconds: {text}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-num-seqs", type=_positive_int, required=True)
    parser.add_argument("--block-size", type=_positive_int, required=True)
    parser.add_argument("--num-speculative-tokens", type=_positive_int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=_positive_int, required=True)
    parser.add_argument("--max-model-len", type=_positive_int, required=True)
    parser.add_argument("--api-key-file")
    parser.add_argument("--wait-seconds", type=_positive_seconds, default=900.0)
    parser.add_argument("--timeout-seconds", type=_positive_seconds, default=600.0)
    args = parser.parse_args(argv)
    try:
        bearer = read_bearer(args.api_key_file) if args.api_key_file else None
        config = WarmupConfig(
            endpoint=args.endpoint,
            model=args.model,
            max_num_seqs=args.max_num_seqs,
            block_size=args.block_size,
            num_speculative_tokens=args.num_speculative_tokens,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=args.max_model_len,
            bearer=bearer,
        )
        arms, skipped = build_sweep(config)
    except (WarmupError, ValueError, OSError) as error:
        print(f"shape warmup: warning: {error}", file=sys.stderr)
        return 1
    summary = run_warmup(
        config,
        arms,
        wait_seconds=args.wait_seconds,
        timeout_seconds=args.timeout_seconds,
        skipped=skipped,
    )
    print(json.dumps({"shape_warmup": summary}, separators=(",", ":")))
    for skip in summary["skipped"]:
        print(f"shape warmup: skipped: {skip}", file=sys.stderr)
    for failure in summary["failures"]:
        print(
            "shape warmup: warning: "
            f"{failure['arm']} request {failure['request']}: {failure['error']}",
            file=sys.stderr,
        )
    if summary["status"] != "complete":
        print(
            f"shape warmup: warning: sweep {summary['status']}; "
            "serving continues without a complete warmup",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
