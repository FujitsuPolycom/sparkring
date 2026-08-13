"""Qualify fixed-depth MTP2 against a matching MTP0 endpoint.

The HTTP mode captures or compares deterministic greedy outputs and verifies
that speculative-decoding counters behave as configured.  The transport mode
checks four before/after SparkRing graph-status snapshots and rejects stock
capture for every query width required by eight-way fixed-MTP2 decode.

This tool sends inference requests but does not start, stop, or modify a model
service.  Capture the MTP0 baseline and MTP2 candidate from the same checkpoint,
runtime, parallelism, cache, graph, and transport configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "sparkring-r7-fixed-mtp2-qualification/v1"
TRANSPORT_SCHEMA = "sparkring-r7-mtp2-transport-audit/v1"
SEED = 20260811
PROMPT_TOKENS = 512
OUTPUT_LENGTHS = (128, 256)
REPEATS = 3
MAX_CONCURRENT_SEQUENCES = 8
FIXED_MTP_DEPTH = 2
MAX_QUERY_ROWS = MAX_CONCURRENT_SEQUENCES * (FIXED_MTP_DEPTH + 1)
# The target graph captures model forward only. Vocabulary collectives belong
# to the speculator's full prefill and decode graphs, each captured once for
# C1 through C8. Query-width coverage is enforced separately by rejecting
# stock vocabulary capture signatures for Q1 through Q24.
MIN_VOCABULARY_CAPTURED_NODES = 2 * MAX_CONCURRENT_SEQUENCES
EXPECTED_ANSWER = re.compile(r"(?<!\d)42(?!\d)")
SPEC_COUNTERS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)
POSITION_COUNTER = "vllm:spec_decode_num_accepted_tokens_per_pos_total"


class QualificationError(RuntimeError):
    """A required correctness, speculation, or transport invariant failed."""


def _request(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> tuple[int, str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise QualificationError(f"request to {url} failed: {exc}") from exc


def _json_request(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> dict[str, Any]:
    status, raw = _request(url, payload=payload, timeout=timeout)
    if status != 200:
        raise QualificationError(
            f"request to {url} returned HTTP {status}: {raw[:500]}"
        )
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise QualificationError(f"request to {url} returned non-JSON data") from exc
    if not isinstance(body, dict):
        raise QualificationError(f"request to {url} returned a non-object JSON value")
    return body


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise QualificationError(f"{name} must be a positive integer, got {value!r}")
    return value


def discover_model(base_url: str, expected_model: str | None, timeout: float) -> str:
    health_status, _ = _request(f"{base_url}/health", timeout=timeout)
    if health_status != 200:
        raise QualificationError(f"health returned HTTP {health_status}")
    body = _json_request(f"{base_url}/v1/models", timeout=timeout)
    models = body.get("data")
    if not isinstance(models, list) or not models or not isinstance(models[0], dict):
        raise QualificationError("model discovery returned no models")
    served = models[0].get("id")
    if not isinstance(served, str) or not served:
        raise QualificationError("model discovery returned an invalid model ID")
    if expected_model is not None and served != expected_model:
        raise QualificationError(
            f"served model {served!r} does not match expected {expected_model!r}"
        )
    return served


def _metric_labels(raw: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    if not raw:
        return labels
    for item in re.findall(r'(\w+)="((?:\\.|[^"\\])*)"', raw):
        labels[item[0]] = bytes(item[1], "utf-8").decode("unicode_escape")
    return labels


def parse_spec_metrics(text: str, expected_model: str | None = None) -> dict[str, Any]:
    totals = {name: 0.0 for name in SPEC_COUNTERS}
    positions: dict[int, float] = {}
    seen: set[str] = set()
    pattern = re.compile(
        r"^(vllm:spec_decode_[a-z0-9_]+_total)(?:\{([^}]*)\})?\s+"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
    )
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if match is None:
            continue
        name, labels_raw, value_raw = match.groups()
        labels = _metric_labels(labels_raw or "")
        if expected_model is not None and (
            labels.get("model_name") != expected_model
            or labels.get("engine") != "0"
        ):
            continue
        value = float(value_raw)
        if not math.isfinite(value) or value < 0:
            raise QualificationError(f"invalid speculative metric {name}={value_raw}")
        seen.add(name)
        if name == POSITION_COUNTER:
            if "position" not in labels:
                raise QualificationError(f"{POSITION_COUNTER} omitted position label")
            position = int(labels["position"])
            positions[position] = positions.get(position, 0.0) + value
        elif name in totals:
            totals[name] += value
    return {"totals": totals, "positions": positions, "seen": sorted(seen)}


def metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    totals = {
        name: after["totals"].get(name, 0.0) - before["totals"].get(name, 0.0)
        for name in SPEC_COUNTERS
    }
    all_positions = set(before["positions"]) | set(after["positions"])
    positions = {
        position: after["positions"].get(position, 0.0)
        - before["positions"].get(position, 0.0)
        for position in sorted(all_positions)
    }
    if any(value < 0 for value in (*totals.values(), *positions.values())):
        raise QualificationError("speculative counters decreased during qualification")
    return {
        "totals": totals,
        "positions": positions,
        "position_keys": sorted(after["positions"]),
    }


def validate_mtp2_metrics(delta: dict[str, Any]) -> dict[str, Any]:
    drafts = delta["totals"][SPEC_COUNTERS[0]]
    drafted = delta["totals"][SPEC_COUNTERS[1]]
    accepted = delta["totals"][SPEC_COUNTERS[2]]
    pos0 = delta["positions"].get(0, 0.0)
    pos1 = delta["positions"].get(1, 0.0)
    if delta.get("position_keys") != [0, 1]:
        raise QualificationError(
            "fixed MTP2 must expose exactly zero-based draft positions 0 and 1"
        )
    if drafts <= 0 or drafted <= 0 or accepted <= 0 or pos0 <= 0 or pos1 <= 0:
        raise QualificationError(
            "MTP2 counters must advance for drafts, drafted tokens, accepted tokens, "
            "and both draft positions"
        )
    if drafted > 2 * drafts or accepted > drafted:
        raise QualificationError(
            f"invalid MTP2 totals: drafts={drafts}, drafted={drafted}, accepted={accepted}"
        )
    if pos1 > pos0 or pos0 > drafts or not math.isclose(accepted, pos0 + pos1):
        raise QualificationError(
            f"invalid MTP2 position counters: accepted={accepted}, pos0={pos0}, pos1={pos1}"
        )
    return {
        "drafts": int(drafts),
        "draft_tokens": int(drafted),
        "accepted_tokens": int(accepted),
        "accepted_tokens_per_position": [int(pos0), int(pos1)],
        "draft_acceptance_rate": accepted / drafted,
        "mean_acceptance_length_including_bonus": 1 + accepted / drafts,
    }


def validate_mtp0_metrics(delta: dict[str, Any]) -> None:
    if any(value != 0 for value in delta["totals"].values()) or any(
        value != 0 for value in delta["positions"].values()
    ):
        raise QualificationError("MTP0 baseline unexpectedly advanced speculation counters")


def semantic_canary(base_url: str, model: str, timeout: float) -> dict[str, Any]:
    body = _json_request(
        f"{base_url}/v1/chat/completions",
        payload={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a calculator. Reply with only the integer answer.",
                },
                {"role": "user", "content": "What is 17 + 25?"},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": SEED,
            "max_tokens": 128,
            "chat_template_kwargs": {"reasoning_effort": "low"},
        },
        timeout=timeout,
    )
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise QualificationError("semantic canary returned no choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise QualificationError("semantic canary returned no message")
    fields = [
        value
        for name in ("reasoning", "reasoning_content", "content")
        if isinstance((value := message.get(name)), str)
    ]
    combined = "\n".join(fields)
    if not EXPECTED_ANSWER.search(combined):
        raise QualificationError("semantic canary did not contain the integer 42")
    return {"answer_present": True, "response_sha256": hashlib.sha256(combined.encode()).hexdigest()}


def finite_logprob_canary(base_url: str, model: str, timeout: float) -> dict[str, Any]:
    expected = 16
    body = _json_request(
        f"{base_url}/v1/completions",
        payload={
            "model": model,
            "prompt": "Continue the even integers: 2, 4, 6, 8,",
            "max_tokens": expected,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": SEED,
            "ignore_eos": True,
            "logprobs": 5,
        },
        timeout=timeout,
    )
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise QualificationError("logprob canary returned no choice")
    logprobs = choices[0].get("logprobs")
    if not isinstance(logprobs, dict):
        raise QualificationError("logprob canary omitted logprobs")
    chosen = logprobs.get("token_logprobs")
    tops = logprobs.get("top_logprobs")
    tokens = logprobs.get("tokens")
    if not all(isinstance(value, list) for value in (chosen, tops, tokens)):
        raise QualificationError("logprob arrays are malformed")
    if not (len(chosen) == len(tops) == len(tokens) == expected):
        raise QualificationError("logprob arrays do not contain exactly 16 tokens")
    finite: list[float] = []
    for value in chosen:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise QualificationError("chosen logprob is nonfinite")
        finite.append(float(value))
    for top in tops:
        if not isinstance(top, dict) or not top:
            raise QualificationError("top-logprob entry is empty")
        for value in top.values():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise QualificationError("top logprob is nonfinite")
            finite.append(float(value))
    return {
        "completion_tokens": expected,
        "finite_logprob_values": len(finite),
        "token_sequence_sha256": _hash_json(tokens),
        "minimum_logprob": min(finite),
        "maximum_logprob": max(finite),
    }


def exact_prompt(base_url: str, model: str, timeout: float) -> tuple[list[int], str]:
    sentence = (
        "Each observation is part of a deterministic model-equivalence test. "
        "Preserve the numbered order and continue with a concise technical analysis. "
    )
    text = "Fixed SparkRing MTP equivalence corpus v1.\n" + "".join(
        f"Observation {index}: {sentence}" for index in range(128)
    )
    body = _json_request(
        f"{base_url}/tokenize",
        payload={"model": model, "prompt": text, "add_special_tokens": False},
        timeout=timeout,
    )
    tokens = body.get("tokens")
    if not isinstance(tokens, list) or not all(
        isinstance(token, int) and not isinstance(token, bool) for token in tokens
    ):
        raise QualificationError("tokenize did not return integer token IDs")
    if len(tokens) < PROMPT_TOKENS:
        raise QualificationError("equivalence prompt is shorter than 512 tokens")
    selected = tokens[:PROMPT_TOKENS]
    return selected, _hash_json(selected)


def greedy_equivalence_canaries(
    base_url: str,
    model: str,
    timeout: float,
) -> dict[str, Any]:
    prompt, prompt_hash = exact_prompt(base_url, model, timeout)
    results: dict[str, Any] = {}
    for output_tokens in OUTPUT_LENGTHS:
        runs: list[dict[str, Any]] = []
        for repeat in range(REPEATS):
            started = time.monotonic()
            body = _json_request(
                f"{base_url}/v1/completions",
                payload={
                    "model": model,
                    "prompt": prompt,
                    "add_special_tokens": False,
                    "max_tokens": output_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": SEED,
                    "ignore_eos": True,
                },
                timeout=timeout,
            )
            elapsed = time.monotonic() - started
            choices = body.get("choices")
            usage = body.get("usage")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise QualificationError(f"{output_tokens}-token greedy run returned no choice")
            if not isinstance(usage, dict):
                raise QualificationError(f"{output_tokens}-token greedy run omitted usage")
            text = choices[0].get("text")
            if not isinstance(text, str):
                raise QualificationError(f"{output_tokens}-token greedy run omitted text")
            prompt_count = _positive_int(usage.get("prompt_tokens"), "prompt_tokens")
            completion_count = _positive_int(
                usage.get("completion_tokens"), "completion_tokens"
            )
            if prompt_count != PROMPT_TOKENS or completion_count != output_tokens:
                raise QualificationError(
                    f"greedy run usage mismatch: prompt={prompt_count}, "
                    f"completion={completion_count}"
                )
            runs.append(
                {
                    "repeat": repeat,
                    "completion_tokens": completion_count,
                    "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "finish_reason": choices[0].get("finish_reason"),
                    "wall_seconds": elapsed,
                }
            )
        hashes = {run["response_sha256"] for run in runs}
        if len(hashes) != 1:
            raise QualificationError(
                f"{output_tokens}-token greedy outputs are not repeatable: {sorted(hashes)}"
            )
        results[str(output_tokens)] = {
            "reference_response_sha256": runs[0]["response_sha256"],
            "runs": runs,
        }
    return {"prompt_token_ids_sha256": prompt_hash, "lengths": results}


def compare_greedy(candidate: dict[str, Any], baseline: dict[str, Any]) -> None:
    baseline_canaries = baseline.get("greedy_equivalence")
    if not isinstance(baseline_canaries, dict):
        raise QualificationError("baseline omitted greedy-equivalence evidence")
    if candidate["prompt_token_ids_sha256"] != baseline_canaries.get(
        "prompt_token_ids_sha256"
    ):
        raise QualificationError("candidate and baseline prompt token IDs differ")
    baseline_lengths = baseline_canaries.get("lengths")
    if not isinstance(baseline_lengths, dict):
        raise QualificationError("baseline omitted greedy output lengths")
    for length in OUTPUT_LENGTHS:
        key = str(length)
        candidate_hash = candidate["lengths"][key]["reference_response_sha256"]
        baseline_entry = baseline_lengths.get(key)
        if not isinstance(baseline_entry, dict):
            raise QualificationError(f"baseline omitted {length}-token output")
        if candidate_hash != baseline_entry.get("reference_response_sha256"):
            raise QualificationError(
                f"MTP2 {length}-token greedy output differs from MTP0: "
                f"candidate={candidate_hash}, "
                f"baseline={baseline_entry.get('reference_response_sha256')}"
            )


def run_http(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.base_url.rstrip("/")
    model = discover_model(base_url, args.model, args.timeout)
    metrics_before_status, metrics_before_raw = _request(
        f"{base_url}/metrics", timeout=args.timeout
    )
    if metrics_before_status != 200:
        raise QualificationError(
            f"metrics-before returned HTTP {metrics_before_status}"
        )
    metrics_before = parse_spec_metrics(metrics_before_raw, model)
    semantic = semantic_canary(base_url, model, args.timeout)
    logprobs = finite_logprob_canary(base_url, model, args.timeout)
    greedy = greedy_equivalence_canaries(base_url, model, args.timeout)
    metrics_after_status, metrics_after_raw = _request(
        f"{base_url}/metrics", timeout=args.timeout
    )
    if metrics_after_status != 200:
        raise QualificationError(f"metrics-after returned HTTP {metrics_after_status}")
    metrics_after = parse_spec_metrics(metrics_after_raw, model)
    delta = metric_delta(metrics_before, metrics_after)

    baseline_sha256: str | None = None
    speculation: dict[str, Any]
    if args.mode == "capture-mtp0":
        validate_mtp0_metrics(delta)
        speculation = {"enabled": False, "counter_delta": delta}
    else:
        baseline_path = Path(args.baseline)
        baseline_bytes = baseline_path.read_bytes()
        baseline_sha256 = hashlib.sha256(baseline_bytes).hexdigest()
        baseline = json.loads(baseline_bytes)
        if not isinstance(baseline, dict) or baseline.get("status") != "pass":
            raise QualificationError("MTP0 baseline is not a passing qualification artifact")
        if baseline.get("model") != model:
            raise QualificationError("candidate and MTP0 baseline model IDs differ")
        compare_greedy(greedy, baseline)
        speculation = {
            "enabled": True,
            "counter_delta": delta,
            **validate_mtp2_metrics(delta),
        }

    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "mode": args.mode,
        "scope": {
            "lane": "public-functional",
            "maturity": "diagnostic",
            "statement": (
                "Bounded endpoint equivalence and speculative-counter evidence. "
                "Performance and transport are qualified by separate artifacts."
            ),
        },
        "endpoint": base_url,
        "model": model,
        "seed": SEED,
        "baseline_sha256": baseline_sha256,
        "semantic_chat": semantic,
        "finite_completion_logprobs": logprobs,
        "greedy_equivalence": greedy,
        "speculation": speculation,
    }


def _rank_session(snapshot: dict[str, Any], family: str, rank: int) -> dict[str, Any]:
    family_snapshot = snapshot.get(family)
    if not isinstance(family_snapshot, dict):
        raise QualificationError(f"rank {rank} status omitted {family}")
    sessions = family_snapshot.get("sessions")
    if not isinstance(sessions, dict):
        raise QualificationError(f"rank {rank} {family} status omitted sessions")
    session = sessions.get(str(rank), sessions.get(rank))
    if not isinstance(session, dict):
        raise QualificationError(f"rank {rank} {family} status omitted its local session")
    return session


def _load_status(path: str, expected_rank: int) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 3:
        raise QualificationError(f"{path} is not a graph-status v3 snapshot")
    if payload.get("rank") != expected_rank:
        raise QualificationError(
            f"{path} reports rank {payload.get('rank')}, expected {expected_rank}"
        )
    snapshot = payload.get("snapshot")
    if not isinstance(snapshot, dict):
        raise QualificationError(f"{path} omitted snapshot")
    return payload


def _validate_transport_session(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    family: str,
    rank: int,
    minimum_captured_nodes: int,
) -> dict[str, Any]:
    required_true = (
        "capture_configured",
        "polling_enabled",
        "host_native_atomics",
        "submit_affinity_verified",
        "progress_affinity_verified",
    )
    for name in required_true:
        if after.get(name) is not True:
            raise QualificationError(f"rank {rank} {family} did not prove {name}")
    captured = after.get("captured_nodes")
    if not isinstance(captured, int) or captured < minimum_captured_nodes:
        raise QualificationError(
            f"rank {rank} {family} captured only {captured!r} nodes; "
            f"at least {minimum_captured_nodes} are required"
        )
    for name in ("published_sequence", "consumed_sequence", "completed_sequence"):
        if not isinstance(after.get(name), int):
            raise QualificationError(f"rank {rank} {family} omitted {name}")
    published = after["published_sequence"]
    if not (
        published > before.get("published_sequence", -1)
        and published == after["consumed_sequence"] == after["completed_sequence"]
    ):
        raise QualificationError(f"rank {rank} {family} replay did not advance and catch up")
    if after.get("overflow_sequence") != 0 or after.get("fatal") is True:
        raise QualificationError(f"rank {rank} {family} reported overflow or fatal state")
    return {
        "captured_nodes": captured,
        "minimum_captured_nodes": minimum_captured_nodes,
        "published_before": before.get("published_sequence"),
        "published_after": published,
        "published_delta": published - before.get("published_sequence", 0),
    }


def _forbidden_stock_signatures(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    stock = snapshot.get("stock_collectives")
    if not isinstance(stock, dict):
        raise QualificationError("transport status omitted stock-collective audit")
    dropped = stock.get("signature_dropped_calls", {})
    if not isinstance(dropped, dict):
        raise QualificationError("stock-signature dropped-call audit is malformed")
    for phase in ("capture", "eager"):
        if dropped.get(phase, 0) != 0:
            raise QualificationError(
                f"{phase} stock-signature audit dropped calls"
            )
    signatures = stock.get("signatures", {})
    if not isinstance(signatures, dict):
        raise QualificationError("stock-signature audit is malformed")
    forbidden: list[dict[str, Any]] = []
    for phase in ("capture", "eager"):
        phase_signatures = signatures.get(phase, [])
        if not isinstance(phase_signatures, list):
            raise QualificationError(
                f"{phase} stock-signature audit is malformed"
            )
        for signature in phase_signatures:
            if not isinstance(signature, dict):
                continue
            shape = signature.get("shape")
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or not isinstance(shape[0], int)
            ):
                continue
            q = shape[0]
            if not 1 <= q <= MAX_QUERY_ROWS:
                continue
            if shape in ([q, 6144], [q, 38720]):
                forbidden.append({"phase": phase, **signature})
    return forbidden


def run_transport(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.before_status) != 4 or len(args.after_status) != 4:
        raise QualificationError("transport audit requires four before and four after files")
    ranks: list[dict[str, Any]] = []
    for rank, (before_path, after_path) in enumerate(
        zip(args.before_status, args.after_status, strict=True)
    ):
        before_payload = _load_status(before_path, rank)
        after_payload = _load_status(after_path, rank)
        if before_payload.get("pid") != after_payload.get("pid"):
            raise QualificationError(f"rank {rank} worker PID changed during transport audit")
        before_end = before_payload.get("snapshot_end_unix_ns")
        after_start = after_payload.get("snapshot_start_unix_ns")
        if not isinstance(before_end, int) or not isinstance(after_start, int):
            raise QualificationError(f"rank {rank} status omitted collection intervals")
        if after_start <= before_end:
            raise QualificationError(f"rank {rank} after-status does not follow before-status")
        before_snapshot = before_payload["snapshot"]
        after_snapshot = after_payload["snapshot"]
        all_reduce = _validate_transport_session(
            _rank_session(before_snapshot, "all_reduce", rank),
            _rank_session(after_snapshot, "all_reduce", rank),
            family="all_reduce",
            rank=rank,
            minimum_captured_nodes=MAX_QUERY_ROWS,
        )
        vocabulary = _validate_transport_session(
            _rank_session(before_snapshot, "vocabulary", rank),
            _rank_session(after_snapshot, "vocabulary", rank),
            family="vocabulary",
            rank=rank,
            minimum_captured_nodes=MIN_VOCABULARY_CAPTURED_NODES,
        )
        forbidden = _forbidden_stock_signatures(after_snapshot)
        if forbidden:
            raise QualificationError(
                f"rank {rank} used stock collectives for required "
                f"Q1-Q{MAX_QUERY_ROWS} signatures: "
                f"{forbidden[:3]}"
            )
        ranks.append(
            {
                "rank": rank,
                "all_reduce": all_reduce,
                "vocabulary": vocabulary,
                "required_query_rows": list(range(1, MAX_QUERY_ROWS + 1)),
                "audited_stock_phases": ["capture", "eager"],
                "required_stock_capture_signatures": 0,
                "required_stock_eager_signatures": 0,
            }
        )
    return {
        "schema": TRANSPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "fixed_mtp_depth": FIXED_MTP_DEPTH,
        "maximum_sequences": MAX_CONCURRENT_SEQUENCES,
        "maximum_query_rows": MAX_QUERY_ROWS,
        "ranks": ranks,
    }


def _write_report(path: str, report: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("capture-mtp0", "qualify-mtp2"):
        command = subparsers.add_parser(mode)
        command.add_argument("--base-url", default="http://127.0.0.1:8000")
        command.add_argument("--model")
        command.add_argument("--timeout", type=float, default=1800.0)
        command.add_argument("--output", required=True)
        if mode == "qualify-mtp2":
            command.add_argument("--baseline", required=True)
    transport = subparsers.add_parser("audit-transport")
    transport.add_argument("--before-status", action="append", required=True)
    transport.add_argument("--after-status", action="append", required=True)
    transport.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_transport(args) if args.mode == "audit-transport" else run_http(args)
    except Exception as exc:
        report = {
            "schema": TRANSPORT_SCHEMA if args.mode == "audit-transport" else SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "fail",
            "mode": args.mode,
            "failures": [f"{type(exc).__name__}: {exc}"],
        }
        _write_report(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    _write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
