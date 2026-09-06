"""Measure bounded growing conversations and idle probes; never change server cache state."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from needle_hunt import base_url  # noqa: E402
from prefill_probe import build_text, events, has_token, request, tokens  # noqa: E402

SCHEMA = "sparkring-conversation-soak/v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fixture(identity, chars):
    text = build_text(identity, chars)
    return text.replace("Read the notes and reply OK.",
                        "Analyze these maintenance notes in detail and recommend next actions.", 1)


def count_tokens(config, messages):
    payload = {"model": config["model"], "messages": messages, "add_generation_prompt": True,
               "chat_template_kwargs": config["chat_template_kwargs"]}
    with request(config["endpoint"] + "/tokenize", payload, config["api_key"], config["timeout"]) as response:
        count = json.load(response).get("count")
    if type(count) is not int or count <= 0:
        raise ValueError("Tokenizer must return a positive integer count")
    return count


def calibrated_user(config, history, identity, target_total, image=None):
    """Calibrate the added user message while preserving every preceding message byte."""
    previous = count_tokens(config, history) if history else 0
    chars = max(256, (target_total - previous) * 5)
    for _ in range(10):
        text = fixture(identity, chars)
        content = text if image is None else [
            {"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": image}}]
        messages = history + [{"role": "user", "content": content}]
        count = count_tokens(config, messages)
        if abs(count - target_total) <= max(8, (target_total - previous) * 0.01):
            return messages, count
        chars = max(128, min(64 * 1024 * 1024, round(chars * max(1, target_total - previous)
                                                   / max(1, count - previous))))
    raise ValueError("Token calibration did not converge")


def stream_turn(config, messages, count, max_tokens, identity, phase, *, request_id=None, clock=time.perf_counter):
    if count + max_tokens > config["context_limit"]:
        raise ValueError("Tokenized prompt plus output exceeds the context limit")
    request_id = request_id or "soak-" + uuid.uuid4().hex
    payload = {"model": config["model"], "messages": messages, "max_tokens": max_tokens,
               "stream": True, "stream_options": {"include_usage": True},
               "temperature": config["temperature"], "seed": config["seed"], "top_p": 1,
               "chat_template_kwargs": config["chat_template_kwargs"]}
    effort = config["probe_reasoning_effort"] if phase != "soak" else config["reasoning_effort"]
    if effort:
        payload["reasoning_effort"] = effort
    chunks, content, reasoning, usage, response_id, finish = [], [], [], None, None, None
    started_unix, started = time.time(), clock()
    with request(config["endpoint"] + "/v1/chat/completions", payload, config["api_key"],
                 config["timeout"], request_id=request_id) as response:
        header_id = response.headers.get("X-Request-ID") if hasattr(response, "headers") else None
        for event in events(response):
            now = clock()
            if event.get("id"):
                response_id = event["id"]
            if has_token(event):
                chunks.append(now - started)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content") or delta.get("reasoning"):
                    reasoning.append(delta.get("reasoning_content") or delta["reasoning"])
                finish = choice.get("finish_reason") or finish
    elapsed = clock() - started
    if not chunks or not isinstance(usage, dict) or type(usage.get("prompt_tokens")) is not int or usage["prompt_tokens"] <= 0:
        raise ValueError("Missing first delta or authoritative prompt usage")
    completion = usage.get("completion_tokens")
    if type(completion) is not int or completion < 1 or finish not in ("stop", "length"):
        raise ValueError("Missing output usage or normal finish reason")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is not None and (type(cached) is not int or not 0 <= cached <= usage["prompt_tokens"]):
        raise ValueError("Invalid reported cached-token count")
    assistant = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        assistant["reasoning_content"] = "".join(reasoning)
    span = chunks[-1] - chunks[0]
    record = {"type": "turn", "phase": phase, "identity": identity, "valid": True,
              "request_id": request_id, "server_request_id": header_id, "response_id": response_id,
              "started_unix": started_unix, "prompt_sha256": digest(messages),
              "assistant": assistant, "tokenized_prompt_tokens": count, "usage": usage,
              "cached_tokens_reported": cached, "cached_fraction_reported": cached / usage["prompt_tokens"]
              if cached is not None and usage["prompt_tokens"] > 0 else None,
              "ttft_seconds": chunks[0], "elapsed_seconds": elapsed,
              "content_delta_offsets_seconds": chunks, "decode_span_seconds": span,
              "decode_tokens_per_second_estimate": (completion - 1) / span if span > 0 else None,
              "finish_reason": finish, "output_budget_exhausted": finish == "length"}
    return record, assistant


def median(values):
    available = [value for value in values if value is not None]
    return statistics.median(available) if available else None


def summarize(records):
    good = [r for r in records if r.get("type") == "turn" and r.get("valid")]
    continuation = [r for r in good if r.get("continuation")]
    known = [r for r in continuation if r["cached_fraction_reported"] is not None]
    low = [r for r in known if r["cached_fraction_reported"] < 0.5]
    high = [r for r in known if r["cached_fraction_reported"] >= 0.5]
    probes = {phase: {"samples": len(rows),
                       "cached_token_counts_reported": [r["cached_tokens_reported"] for r in rows],
                       "median_ttft_seconds": median(r["ttft_seconds"] for r in rows),
                       "median_decode_tokens_per_second_estimate": median(r["decode_tokens_per_second_estimate"] for r in rows)}
              for phase in ("before", "after")
              for rows in [[r for r in good if r["phase"] == phase]]}
    return {"type": "summary", "valid_turns": len(good),
            "soak_turns": sum(r["phase"] == "soak" for r in good),
            "errors": sum(r.get("type") == "error" for r in records),
            "continuations": len(continuation), "continuations_with_cache_usage": len(known),
            "continuations_below_half_cached": len(low),
            "fraction_below_half_cached": len(low) / len(known) if known else None,
            "low_cached_median_latency_seconds": median(r["elapsed_seconds"] for r in low),
            "high_cached_median_latency_seconds": median(r["elapsed_seconds"] for r in high),
            "probes": probes,
            "cache_interpretation": "server-reported cached tokens; no inference of local hit, external restore, or recompute"}


def execute(config, output):
    lock, stop = threading.Lock(), threading.Event()
    records, consumed = [], 0
    image = config.get("image_data")
    safe_config = {key: value for key, value in config.items() if key not in ("api_key", "image_data")}
    with Path(output).open("x", encoding="utf-8") as stream:
        def emit(record):
            with lock:
                records.append(record)
                line = json.dumps(record)
                stream.write(line + "\n")
                stream.flush()
                print(line, flush=True)

        emit({"type": "start", "schema": SCHEMA, "config": safe_config,
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "http_helper_sha256": hashlib.sha256(Path(__file__).with_name("prefill_probe.py").read_bytes()).hexdigest(),
              "started_unix": time.time()})

        def probe(phase, index):
            identity = f"{config['seed']}-probe-{phase}-{index}"
            request_id = "soak-" + uuid.uuid4().hex
            try:
                messages, count = calibrated_user(config, [], identity, config["probe_tokens"])
                record, _ = stream_turn(config, messages, count, config["probe_output_tokens"], identity,
                                        phase, request_id=request_id)
                emit(record)
            except Exception as error:
                emit({"type": "error", "phase": phase, "identity": identity,
                      "request_id": request_id, "error": type(error).__name__})
                stop.set()

        def worker(agent, deadline):
            nonlocal consumed
            history, prior_count, conversation, turn = [], None, 0, 0
            for _ in range(config["max_turns_per_agent"]):
                if stop.is_set() or time.monotonic() >= deadline:
                    break
                if prior_count is not None and prior_count + config["tail_tokens"] + config["max_tokens"] > config["reset_tokens"]:
                    history, prior_count, turn = [], None, 0
                    conversation += 1
                identity = f"{config['seed']}-agent-{agent}-conversation-{conversation}-turn-{turn}"
                request_id = "soak-" + uuid.uuid4().hex
                try:
                    initial = config["start_tokens"][agent % len(config["start_tokens"])]
                    target = initial if prior_count is None else prior_count + config["tail_tokens"]
                    picture = image if image and turn > 0 and turn % config["image_every"] == 0 else None
                    messages, count = calibrated_user(config, history, identity, target, picture)
                    if count + config["max_tokens"] > config["context_limit"]:
                        raise ValueError("Context limit exceeded")
                    with lock:
                        if stop.is_set() or time.monotonic() >= deadline or consumed + count > config["max_soak_prompt_tokens"]:
                            break
                        consumed += count
                    record, assistant = stream_turn(config, messages, count, config["max_tokens"], identity,
                                                    "soak", request_id=request_id)
                    record.update(agent=agent, conversation=conversation, turn=turn,
                                  prompt_token_growth=count - prior_count if prior_count is not None else None,
                                  continuation=prior_count is not None and 0 < count - prior_count < 10000,
                                  image_added=picture is not None)
                    emit(record)
                    history, prior_count, turn = messages + [assistant], count, turn + 1
                except Exception as error:
                    emit({"type": "error", "phase": "soak", "identity": identity,
                          "request_id": request_id, "error": type(error).__name__})
                    stop.set()
                    break

        try:
            for index in range(config["probe_repeats"]):
                probe("before", index)
                if stop.is_set():
                    break
            deadline = time.monotonic() + config["duration_seconds"]
            with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
                list(pool.map(lambda agent: worker(agent, deadline), range(config["concurrency"])))
            if not stop.is_set():
                for index in range(config["probe_repeats"]):
                    probe("after", index)
                    if stop.is_set():
                        break
        except Exception as error:
            emit({"type": "error", "phase": "probe", "error": type(error).__name__})
        summary = summarize(records)
        summary["soak_prompt_tokens_admitted"] = consumed
        summary["success"] = not summary["errors"] and summary["soak_turns"] > 0
        summary["completed_unix"] = time.time()
        emit(summary)
    return 0 if summary["success"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", required=True, help="Evidence label only; never changes server configuration")
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--start-tokens", default="75000,106000,128000,150000")
    parser.add_argument("--tail-tokens", type=tokens, default=2048)
    parser.add_argument("--reset-tokens", type=tokens, default=300000)
    parser.add_argument("--context-limit", type=tokens, required=True)
    parser.add_argument("--max-tokens", type=tokens, default=512)
    parser.add_argument("--duration-seconds", type=float, default=300)
    parser.add_argument("--max-turns-per-agent", type=int, default=20)
    parser.add_argument("--max-soak-prompt-tokens", type=int, default=10000000)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--probe-reasoning-effort", default="low")
    parser.add_argument("--probe-tokens", type=tokens, default=5300)
    parser.add_argument("--probe-output-tokens", type=tokens, default=300)
    parser.add_argument("--probe-repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900, help="HTTP inactivity timeout, not a wall-clock deadline")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--chat-template-kwargs", default="{}")
    parser.add_argument("--metadata", default="{}", help="JSON evidence identities; do not include credentials")
    parser.add_argument("--image", type=Path, help="Optional local PNG or JPEG, at most 4 MiB")
    parser.add_argument("--image-every", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", action="store_true", help="Print bounds without tokenization or HTTP requests")
    parser.add_argument("--analyze", type=Path, help="Summarize an existing JSONL receipt without HTTP")
    args = parser.parse_args()
    if args.analyze:
        print(json.dumps(summarize([json.loads(line) for line in args.analyze.read_text(encoding="utf-8").splitlines()])))
        return 0
    config = vars(args).copy()
    for name in ("output", "plan", "analyze", "image"):
        config.pop(name)
    try:
        config["start_tokens"] = [tokens(value) for value in args.start_tokens.split(",")]
        config["chat_template_kwargs"] = json.loads(args.chat_template_kwargs)
        config["metadata"] = json.loads(args.metadata)
        if not isinstance(config["chat_template_kwargs"], dict) or not isinstance(config["metadata"], dict):
            raise ValueError("Template options and metadata must be JSON objects")
        if not 1 <= args.concurrency <= 32 or not 1 <= args.max_turns_per_agent <= 10000 or not 0 <= args.probe_repeats <= 20 or args.image_every < 1:
            raise ValueError("Invalid concurrency, repetition, or image frequency")
        if not math.isfinite(args.duration_seconds) or not 0 < args.duration_seconds <= 14400:
            raise ValueError("Duration must be positive and at most 4 hours")
        if not 1 <= args.max_soak_prompt_tokens <= 1000000000:
            raise ValueError("Soak token budget must be between 1 and 1000000000")
        if not math.isfinite(args.timeout) or not 0 < args.timeout <= 3600 or not 0 <= args.temperature <= 2:
            raise ValueError("Invalid timeout or temperature")
        if max(config["start_tokens"]) + args.max_tokens > args.reset_tokens or args.reset_tokens > args.context_limit:
            raise ValueError("Starting prompts plus output must fit reset threshold and context limit")
        if args.probe_tokens + args.probe_output_tokens > args.context_limit:
            raise ValueError("Probe prompt plus output must fit context limit")
        if args.endpoint:
            config["endpoint"] = base_url(args.endpoint)
        if args.image:
            if args.image.stat().st_size > 4 * 1024 * 1024:
                raise ValueError("Image exceeds 4 MiB")
            raw = args.image.read_bytes()
            mime = "image/png" if raw.startswith(b"\x89PNG\r\n\x1a\n") else "image/jpeg" if raw.startswith(b"\xff\xd8\xff") else None
            if mime is None:
                raise ValueError("Image must be PNG or JPEG")
            config["image_sha256"] = hashlib.sha256(raw).hexdigest()
            image_data = "data:" + mime + ";base64," + base64.b64encode(raw).decode()
        else:
            image_data = None
    except (ValueError, OSError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))
    if args.plan:
        print(json.dumps({"schema": SCHEMA, "config": config,
                          "max_chat_requests": args.concurrency * args.max_turns_per_agent + 2 * args.probe_repeats,
                          "max_output_tokens": args.concurrency * args.max_turns_per_agent * args.max_tokens
                          + 2 * args.probe_repeats * args.probe_output_tokens,
                          "duration_scope": "soak admission window; probes, calibration and in-flight requests may extend elapsed time",
                          "status": "implemented; no hardware qualification"}))
        return 0
    if not args.endpoint or not args.output:
        parser.error("Execution requires --endpoint and --output")
    config["api_key"] = os.environ.get(args.api_key_env, "")
    config["image_data"] = image_data
    return execute(config, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
