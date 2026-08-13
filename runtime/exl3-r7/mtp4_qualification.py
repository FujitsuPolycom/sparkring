"""Qualify fixed-depth MTP4 against a matching MTP0 endpoint.

The HTTP mode captures or compares deterministic greedy outputs, rejects
non-finite log probabilities, and verifies that every fixed-MTP4 acceptance
counter advances coherently. The transport mode checks four before/after
SparkRing graph-status snapshots and rejects stock capture or eager fallback
for every query width required by eight-way fixed-MTP4 decode.

This tool sends inference requests but does not start, stop, or modify a model
service. Capture the MTP0 baseline and MTP4 candidate from the same checkpoint,
runtime, parallelism, cache, graph, and transport configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_common_module() -> Any:
    """Load the adjacent fixed-MTP3 harness as the shared HTTP implementation."""

    path = Path(__file__).with_name("mtp3_qualification.py")
    spec = importlib.util.spec_from_file_location(
        "sparkring_r7_mtp_qualification_common", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared qualification helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_common = _load_common_module()

SCHEMA = "sparkring-r7-fixed-mtp4-qualification/v1"
TRANSPORT_SCHEMA = "sparkring-r7-mtp4-transport-audit/v1"
FIXED_MTP_DEPTH = 4
MAX_CONCURRENT_SEQUENCES = 8
MAX_QUERY_ROWS = MAX_CONCURRENT_SEQUENCES * (FIXED_MTP_DEPTH + 1)
# Draft prefill and the reusable single-step draft-decode graph are each
# captured once for C1 through C8. Increasing fixed depth replays the latter;
# it does not create another vocabulary graph family.
MIN_VOCABULARY_CAPTURED_NODES = 2 * MAX_CONCURRENT_SEQUENCES
_REQUIRED_STOCK_CAPTURE_COUNTS = (
    "dcp_combine:original",
    "dcp_lse_all_gather:ineligible_signature",
    "dcp_query_all_gather:ineligible_signature",
    "dcp_owner_topk_all_gather:graph_capture_unsupported",
)
_REQUIRED_STOCK_EAGER_COUNTS = (
    "dcp_combine:original",
    "dcp_lse_all_gather:ineligible_signature",
    "dcp_query_all_gather:ineligible_signature",
    "dcp_owner_topk_all_gather:ineligible_signature",
)

QualificationError = _common.QualificationError
SPEC_COUNTERS = _common.SPEC_COUNTERS
POSITION_COUNTER = _common.POSITION_COUNTER
SEED = _common.SEED
PROMPT_TOKENS = _common.PROMPT_TOKENS
OUTPUT_LENGTHS = _common.OUTPUT_LENGTHS
REPEATS = _common.REPEATS

# Expose the shared, depth-independent primitives for callers and focused
# tests. Depth-specific validation remains local and fail-closed below.
_request = _common._request
_json_request = _common._json_request
parse_spec_metrics = _common.parse_spec_metrics
metric_delta = _common.metric_delta
discover_model = _common.discover_model
semantic_canary = _common.semantic_canary
finite_logprob_canary = _common.finite_logprob_canary
greedy_equivalence_canaries = _common.greedy_equivalence_canaries
validate_mtp0_metrics = _common.validate_mtp0_metrics
_rank_session = _common._rank_session
_load_status = _common._load_status
_validate_transport_session = _common._validate_transport_session


def validate_mtp4_metrics(delta: dict[str, Any]) -> dict[str, Any]:
    """Require coherent drafts and acceptance at all four fixed positions."""

    drafts = delta["totals"][SPEC_COUNTERS[0]]
    drafted = delta["totals"][SPEC_COUNTERS[1]]
    accepted = delta["totals"][SPEC_COUNTERS[2]]
    positions = [delta["positions"].get(position, 0.0) for position in range(4)]
    if delta.get("position_keys") != [0, 1, 2, 3]:
        raise QualificationError(
            "fixed MTP4 must expose exactly zero-based draft positions 0, 1, 2, "
            "and 3"
        )
    if drafts <= 0 or drafted <= 0 or accepted <= 0 or any(
        value <= 0 for value in positions
    ):
        raise QualificationError(
            "MTP4 counters must advance for drafts, drafted tokens, accepted "
            "tokens, and all four draft positions"
        )
    if drafted > FIXED_MTP_DEPTH * drafts or accepted > drafted:
        raise QualificationError(
            f"invalid MTP4 totals: drafts={drafts}, drafted={drafted}, "
            f"accepted={accepted}"
        )
    if (
        positions[3] > positions[2]
        or positions[2] > positions[1]
        or positions[1] > positions[0]
        or positions[0] > drafts
        or not math.isclose(accepted, sum(positions))
    ):
        raise QualificationError(
            f"invalid MTP4 position counters: accepted={accepted}, "
            f"positions={positions}"
        )
    return {
        "drafts": int(drafts),
        "draft_tokens": int(drafted),
        "accepted_tokens": int(accepted),
        "acceptance_rate": accepted / drafted,
        "accepted_tokens_per_position": [int(value) for value in positions],
    }


def compare_greedy(candidate: dict[str, Any], baseline: dict[str, Any]) -> None:
    """Require all repeated candidate outputs to match the sealed MTP0 hashes."""

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
                f"MTP4 {length}-token greedy output differs from MTP0: "
                f"candidate={candidate_hash}, "
                f"baseline={baseline_entry.get('reference_response_sha256')}"
            )


def run_http(args: argparse.Namespace) -> dict[str, Any]:
    """Run bounded semantic, finite-logprob, hash, and counter gates."""

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
            raise QualificationError(
                "MTP0 baseline is not a passing qualification artifact"
            )
        if baseline.get("model") != model:
            raise QualificationError("candidate and MTP0 baseline model IDs differ")
        compare_greedy(greedy, baseline)
        speculation = {
            "enabled": True,
            "counter_delta": delta,
            **validate_mtp4_metrics(delta),
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


def _forbidden_stock_signatures(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Return stock TP collective signatures inside fixed-MTP4's Q1-Q40 range."""

    stock = snapshot.get("stock_collectives")
    if not isinstance(stock, dict):
        raise QualificationError("transport status omitted stock-collective audit")
    dropped = stock.get("signature_dropped_calls", {})
    if not isinstance(dropped, dict):
        raise QualificationError("stock-signature dropped-call audit is malformed")
    for phase in ("capture", "eager"):
        if dropped.get(phase, 0) != 0:
            raise QualificationError(f"{phase} stock-signature audit dropped calls")
    signatures = stock.get("signatures", {})
    if not isinstance(signatures, dict):
        raise QualificationError("stock-signature audit is malformed")
    forbidden: list[dict[str, Any]] = []
    for phase in ("capture", "eager"):
        phase_signatures = signatures.get(phase, [])
        if not isinstance(phase_signatures, list):
            raise QualificationError(f"{phase} stock-signature audit is malformed")
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
            if 1 <= q <= MAX_QUERY_ROWS and shape in ([q, 6144], [q, 38720]):
                forbidden.append({"phase": phase, **signature})
    return forbidden


def _positive_stock_count(
    counts: dict[str, Any], name: str, *, rank: int, phase: str
) -> int:
    value = counts.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise QualificationError(
            f"rank {rank} stock {phase} audit did not prove {name}"
        )
    return value


def _validate_stock_dcp_indexer(
    before: dict[str, Any], after: dict[str, Any], *, rank: int
) -> dict[str, Any]:
    """Prove the inherited DCP and sparse-indexer paths remain stock NCCL."""

    dcp = after.get("dcp")
    if not isinstance(dcp, dict):
        raise QualificationError(f"rank {rank} status omitted DCP diagnostics")
    family_selection = dcp.get("family_selection")
    expected_selection = {
        "mode": "stock",
        "query_enabled": False,
        "combine_enabled": False,
    }
    if family_selection != expected_selection or dcp.get("sessions") != {}:
        raise QualificationError(
            f"rank {rank} DCP must remain stock with no custom sessions"
        )
    indexer = after.get("indexer")
    if not isinstance(indexer, dict) or indexer.get("sessions") != {}:
        raise QualificationError(
            f"rank {rank} indexer must remain stock with no custom sessions"
        )

    before_stock = before.get("stock_collectives")
    after_stock = after.get("stock_collectives")
    if not isinstance(before_stock, dict) or not isinstance(after_stock, dict):
        raise QualificationError(f"rank {rank} omitted stock-collective counters")
    after_capture = after_stock.get("capture")
    before_eager = before_stock.get("eager")
    after_eager = after_stock.get("eager")
    if not all(isinstance(value, dict) for value in (after_capture, before_eager, after_eager)):
        raise QualificationError(
            f"rank {rank} stock DCP/indexer phase counters are malformed"
        )

    capture_counts = {
        name: _positive_stock_count(after_capture, name, rank=rank, phase="capture")
        for name in _REQUIRED_STOCK_CAPTURE_COUNTS
    }
    eager_deltas: dict[str, int] = {}
    for name in _REQUIRED_STOCK_EAGER_COUNTS:
        before_value = before_eager.get(name, 0)
        after_value = after_eager.get(name)
        if (
            not isinstance(before_value, int)
            or isinstance(before_value, bool)
            or before_value < 0
            or not isinstance(after_value, int)
            or isinstance(after_value, bool)
            or after_value <= before_value
        ):
            raise QualificationError(
                f"rank {rank} stock eager audit did not advance {name}"
            )
        eager_deltas[name] = after_value - before_value
    return {
        "mode": "stock",
        "custom_dcp_sessions": 0,
        "custom_indexer_sessions": 0,
        "capture_counts": capture_counts,
        "eager_deltas": eager_deltas,
    }


def run_transport(args: argparse.Namespace) -> dict[str, Any]:
    """Audit all-rank TP all-reduce/vocabulary native replay through Q40."""

    if len(args.before_status) != 4 or len(args.after_status) != 4:
        raise QualificationError(
            "transport audit requires four before and four after files"
        )
    ranks: list[dict[str, Any]] = []
    for rank, (before_path, after_path) in enumerate(
        zip(args.before_status, args.after_status, strict=True)
    ):
        before_payload = _load_status(before_path, rank)
        after_payload = _load_status(after_path, rank)
        if before_payload.get("pid") != after_payload.get("pid"):
            raise QualificationError(
                f"rank {rank} worker PID changed during transport audit"
            )
        before_end = before_payload.get("snapshot_end_unix_ns")
        after_start = after_payload.get("snapshot_start_unix_ns")
        if not isinstance(before_end, int) or not isinstance(after_start, int):
            raise QualificationError(f"rank {rank} status omitted collection intervals")
        if after_start <= before_end:
            raise QualificationError(
                f"rank {rank} after-status does not follow before-status"
            )
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
                f"Q1-Q{MAX_QUERY_ROWS} signatures: {forbidden[:3]}"
            )
        dcp_indexer = _validate_stock_dcp_indexer(
            before_snapshot, after_snapshot, rank=rank
        )
        ranks.append(
            {
                "rank": rank,
                "all_reduce": all_reduce,
                "vocabulary": vocabulary,
                "dcp_indexer": dcp_indexer,
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
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("capture-mtp0", "qualify-mtp4"):
        command = subparsers.add_parser(mode)
        command.add_argument("--base-url", default="http://127.0.0.1:8000")
        command.add_argument("--model")
        command.add_argument("--timeout", type=float, default=1800.0)
        command.add_argument("--output", required=True)
        if mode == "qualify-mtp4":
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
