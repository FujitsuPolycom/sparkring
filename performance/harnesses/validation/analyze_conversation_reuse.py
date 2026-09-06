"""Join saved conversation receipts to rank-local SparkCache reuse traces; no network access."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

TRACE_SCHEMA = "sparkcache-reuse-trace/v1"
MARKER = "spark-context-cache-reuse:"


def parse_trace(line):
    """Accept Docker log prefixes and JSON log envelopes without interpreting prose."""
    raw = line.split(MARKER, 1)[1].lstrip() if MARKER in line and not line.lstrip().startswith("{") else line.lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict):
        return None
    if value.get("schema") == TRACE_SCHEMA:
        return value
    for field in ("log", "message"):
        if isinstance(value.get(field), str) and MARKER in value[field]:
            return parse_trace(value[field])
    return None


def aliases(record):
    values = set()
    for name in ("request_id", "response_id", "server_request_id"):
        value = record.get(name)
        if isinstance(value, str) and value:
            values.add(value)
            values.add(value.removeprefix("chatcmpl-") if value.startswith("chatcmpl-") else "chatcmpl-" + value)
    return values


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def median(values):
    valid = [value for value in values if finite_number(value)]
    return statistics.median(valid) if valid else None


def latest_completions(traces, expected):
    candidates = defaultdict(list)
    for trace in traces:
        rank = trace.get("rank")
        if (trace.get("event") == "worker_restore_completed" and trace.get("role") == "worker"
                and type(rank) is int and rank in expected and type(trace.get("time_ns")) is int):
            candidates[rank].append(trace)
    latest, conflicts = {}, []
    for rank, rows in candidates.items():
        newest = max(row["time_ns"] for row in rows)
        current = [row for row in rows if row["time_ns"] == newest]
        identities = {json.dumps(row, sort_keys=True) for row in current}
        if len(identities) > 1:
            conflicts.append(rank)
        else:
            latest[rank] = current[0]
    return latest, conflicts


def classify(turn, traces, expected, *, ambiguous=False):
    latest, conflicts = latest_completions(traces, expected)
    verified = len(latest) == len(expected) and not conflicts
    identities = set()
    for row in latest.values():
        span = row.get("requested_span_tokens")
        if type(span) is not int or not isinstance(row.get("digest"), str):
            verified = False
            continue
        verified &= (row.get("outcome") == "verified" and span > 0
                     and type(row.get("verified_span_tokens")) is int
                     and row.get("verified_span_tokens") == span
                     and bool(row["digest"]))
        identities.add((row.get("digest"), span, row.get("request_id")))
    verified = bool(verified and len(identities) == 1 and turn.get("valid") and not ambiguous)
    attached = [row for row in traces if row.get("event") == "gpu_lease_attached"
                and row.get("role") == "scheduler" and type(row.get("lease_span_tokens")) is int
                and row["lease_span_tokens"] > 0]
    cached = turn.get("cached_tokens_reported")
    usage = turn.get("usage") or {}
    prompt = usage.get("prompt_tokens")
    positive_report = type(cached) is int and type(prompt) is int and 0 < cached <= prompt
    if ambiguous or not turn.get("valid"):
        source = "unknown"
    elif verified:
        source = "verified_all_rank_external_restore"
    elif attached:
        source = "gpu_lease_attached"
    elif positive_report:
        source = "reported_cached"
    else:
        source = "unknown"
    return {"type": "turn_analysis", "request_id": turn.get("request_id"),
            "response_id": turn.get("response_id"), "identity": turn.get("identity"),
            "phase": turn.get("phase"), "continuation": turn.get("continuation", False),
            "elapsed_seconds": turn.get("elapsed_seconds"), "ttft_seconds": turn.get("ttft_seconds"),
            "source": source, "trace_join_ambiguous": ambiguous,
            "cached_tokens_reported": cached, "prompt_tokens_reported": prompt,
            "all_rank_worker_verification": verified,
            "verified_span_tokens": next(iter(identities))[1] if verified else None,
            "gpu_lease_attachment_observed": bool(attached),
            "lease_span_tokens_observed": max((row["lease_span_tokens"] for row in attached), default=None),
            "offer_count": sum(row.get("event") == "external_restore_offer" for row in traces),
            "expected_ranks": sorted(expected), "completed_ranks": sorted(latest),
            "missing_ranks": sorted(expected - latest.keys()), "conflicting_ranks": sorted(conflicts),
            "worker_completions_by_rank": {str(rank): row for rank, row in sorted(latest.items())},
            "trace_events": traces,
            "reported_cache_interpretation": "API cached tokens only; positive counts do not distinguish local reuse from external restore, and zero counts do not exclude GPU lease reuse"}


def analyze(receipts, traces, expected_ranks):
    expected = set(expected_ranks)
    if not expected or any(type(rank) is not int or rank < 0 for rank in expected):
        raise ValueError("Expected physical ranks must be explicit nonnegative integers")
    turns = [row for row in receipts if row.get("type") == "turn"]
    owners, response_owners, joined, ambiguous = defaultdict(set), defaultdict(set), defaultdict(list), set()
    for index, turn in enumerate(turns):
        for alias in aliases(turn):
            owners[alias].add(index)
        response_id = turn.get("response_id")
        if isinstance(response_id, str) and response_id:
            response_owners[response_id].add(index)
    seen, unmatched, duplicated = set(), 0, 0
    for trace in traces:
        encoded = json.dumps(trace, sort_keys=True)
        if encoded in seen:
            duplicated += 1
            continue
        seen.add(encoded)
        request_id = trace.get("request_id")
        matches = set(owners.get(request_id, set())) if isinstance(request_id, str) else set()
        if isinstance(request_id, str):
            # vLLM appends an eight-hex engine nonce to the observed API response
            # ID. Only that exact suffix grammar is accepted; never prefix-match.
            match = re.fullmatch(r"(.+)-[0-9a-fA-F]{8}", request_id)
            if match:
                matches.update(response_owners.get(match.group(1), set()))
        if len(matches) == 1:
            joined[next(iter(matches))].append(trace)
        elif matches:
            ambiguous.update(matches)
        else:
            unmatched += 1
    rows = [classify(turn, joined[index], expected, ambiguous=index in ambiguous)
            for index, turn in enumerate(turns)]
    by_source = {}
    by_phase_and_source = {}
    for source in sorted({row["source"] for row in rows}):
        population = [row for row in rows if row["source"] == source]
        by_source[source] = {"turns": len(population),
                             "median_latency_seconds": median(row["elapsed_seconds"] for row in population),
                             "median_ttft_seconds": median(row["ttft_seconds"] for row in population)}
    for phase in sorted({row.get("phase") or "unknown" for row in rows}):
        by_phase_and_source[phase] = {}
        for source in by_source:
            population = [row for row in rows if (row.get("phase") or "unknown") == phase and row["source"] == source]
            if population:
                by_phase_and_source[phase][source] = {
                    "turns": len(population), "continuations": sum(row["continuation"] for row in population),
                    "median_latency_seconds": median(row["elapsed_seconds"] for row in population),
                    "median_ttft_seconds": median(row["ttft_seconds"] for row in population)}
    timings = {}
    for rank in sorted(expected):
        population = [trace for row in rows for trace in row["trace_events"]
                      if trace.get("role") == "worker" and trace.get("event") == "worker_restore_completed"
                      and type(trace.get("rank")) is int and trace["rank"] == rank]
        phases = sorted({phase for row in population for phase in (row.get("phase_ms") or {})})
        timings[str(rank)] = {"completions": len(population),
                             "outcomes": dict(Counter(row.get("outcome", "unknown") for row in population)),
                             **{f"median_{field}": median(row.get(field) for row in population)
                                for field in ("queue_wait_ms", "service_ms", "end_to_end_ms")},
                             "median_phase_ms": {phase: median((row.get("phase_ms") or {}).get(phase) for row in population)
                                                 for phase in phases}}
    return {"schema": "sparkring-conversation-reuse-analysis/v1", "trace_schema": TRACE_SCHEMA,
            "expected_ranks": sorted(expected), "turns": rows,
            "summary": {"turns": len(rows), "by_source": by_source, "by_phase_and_source": by_phase_and_source,
                        "per_rank_restore_timings": timings,
                        "unmatched_trace_events": unmatched, "duplicate_trace_events": duplicated,
                        "ambiguous_turns": len(ambiguous)},
            "limits": ["No recomputation or local-miss inference from zero or missing API cache counts",
                       "Offers do not prove an executed restore; every expected physical rank must verify the same span and digest",
                       "Worker verification plus successful client completion does not expose vLLM's internal receive-aggregation decision",
                       "An observed lease attachment proves attachment, not its full subsequent lifetime",
                       "Rank token counts are never summed; source classes may coexist and the strongest observed evidence is selected"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--log", type=Path, action="append", required=True,
                        help="Saved rank log; repeat for every physical rank")
    parser.add_argument("--expected-ranks", required=True, help="Physical ranks, for example 0,1,2,3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        ranks = [int(value) for value in args.expected_ranks.split(",")]
        if len(set(ranks)) != len(ranks):
            raise ValueError("Expected ranks must be unique")
        receipts = [json.loads(line) for line in args.receipt.read_text(encoding="utf-8").splitlines() if line.strip()]
        traces, malformed, sources = [], 0, []
        for path in args.log:
            sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            with path.open(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    record = parse_trace(line)
                    if record:
                        traces.append(record)
                    elif MARKER in line:
                        malformed += 1
        result = analyze(receipts, traces, ranks)
        result["inputs"] = {"receipt": str(args.receipt),
                            "receipt_sha256": hashlib.sha256(args.receipt.read_bytes()).hexdigest(), "logs": sources}
        result["summary"]["malformed_trace_lines"] = malformed
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
