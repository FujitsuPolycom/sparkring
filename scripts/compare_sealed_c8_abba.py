#!/usr/bin/env python3
"""Fail-closed offline comparison of four sealed 16K/C8 benchmark arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from llm_decode_prompt_manifest import (
    HARNESS_NAME,
    HARNESS_VERSION,
    SAFE_MODEL_IDENTIFIER,
    load_summary,
)

SCHEMA = "sparkring-sealed-c8-abba-comparison/v1"
PROMPT_SUMMARY_SCHEMA = "sparkring-llm-decode-prompt-manifest-summary/v1"
PROMPT_MANIFEST_SCHEMA = "sparkring-llm-decode-prompt-manifest/v1"
HEX64 = re.compile(r"[0-9a-f]{64}")
REQUESTED_CONTEXT_TOKENS = 16384
REQUESTED_CONCURRENCY = 8
REQUESTED_DURATION_SECONDS = 25.0
MEASUREMENT_WINDOW_TOLERANCE_SECONDS = 1.0
AGGREGATE_REL_TOLERANCE = 1e-6


class EvidenceError(ValueError):
    pass


MATCHED_METADATA = (
    "version",
    "engine",
    "model",
    "decode_mode",
    "duration_per_test",
    "max_tokens",
    "temperature",
    "reasoning_effort",
    "ignore_eos",
    "dcp_size",
    "unique_context_percent",
    "shared_context_percent",
    "decode_warmup_seconds",
    "cell_warmup_timeout_seconds",
    "context_lengths",
    "concurrency_levels",
    "skip_prefill",
    "token_targeting",
)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> Any:
    raise EvidenceError(f"non-finite JSON number {value!r} is unsupported")


def _load(path: Path) -> tuple[dict[str, Any], str]:
    try:
        encoded = path.read_bytes()
        document = json.loads(
            encoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, json.JSONDecodeError, EvidenceError) as error:
        raise EvidenceError(f"cannot read benchmark evidence: {error}") from error
    if not isinstance(document, dict):
        raise EvidenceError("benchmark evidence must be a JSON object")
    return document, hashlib.sha256(encoded).hexdigest()


def validate_arm(document: dict[str, Any], logical_label: str) -> dict[str, Any]:
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise EvidenceError(f"{logical_label}: metadata must be an object")
    missing = [key for key in MATCHED_METADATA if key not in metadata]
    if missing:
        raise EvidenceError(f"{logical_label}: missing metadata fields: {', '.join(missing)}")
    if metadata["version"] != HARNESS_VERSION:
        raise EvidenceError(
            f"{logical_label}: version must be pinned llm_decode_bench.py {HARNESS_VERSION}"
        )
    if metadata["engine"] != "vllm":
        raise EvidenceError(f"{logical_label}: engine must be vllm")
    model = metadata["model"]
    if (
        not isinstance(model, str)
        or SAFE_MODEL_IDENTIFIER.fullmatch(model) is None
        or model in {".", ".."}
        or "/." in model
    ):
        raise EvidenceError(
            f"{logical_label}: model must be a safe served-model identifier, not a path"
        )
    if metadata["decode_mode"] != "duration":
        raise EvidenceError(f"{logical_label}: decode_mode must be duration")
    if metadata["context_lengths"] != [REQUESTED_CONTEXT_TOKENS]:
        raise EvidenceError(
            f"{logical_label}: context_lengths must be exactly [{REQUESTED_CONTEXT_TOKENS}]"
        )
    if metadata["concurrency_levels"] != [REQUESTED_CONCURRENCY]:
        raise EvidenceError(
            f"{logical_label}: concurrency_levels must be exactly [{REQUESTED_CONCURRENCY}]"
        )
    if metadata["skip_prefill"] is not True:
        raise EvidenceError(f"{logical_label}: skip_prefill must be true")
    if not _is_number(metadata["temperature"]) or float(metadata["temperature"]) != 0.0:
        raise EvidenceError(f"{logical_label}: temperature must be exactly zero")
    if (
        not _is_number(metadata["duration_per_test"])
        or float(metadata["duration_per_test"]) != REQUESTED_DURATION_SECONDS
    ):
        raise EvidenceError(
            f"{logical_label}: duration_per_test must be exactly {REQUESTED_DURATION_SECONDS:g}s"
        )
    if not isinstance(metadata["max_tokens"], int) or isinstance(metadata["max_tokens"], bool):
        raise EvidenceError(f"{logical_label}: max_tokens must be an integer")
    if metadata["max_tokens"] < 256:
        raise EvidenceError(f"{logical_label}: max_tokens is not sustained")
    if metadata["token_targeting"] not in {"estimate", "exact"}:
        raise EvidenceError(f"{logical_label}: token_targeting is invalid")
    for key, expected in (
        ("request_count", 0),
        ("run_burst", False),
        ("prefill_only", False),
        ("standalone_prefill", False),
    ):
        if metadata.get(key) != expected:
            raise EvidenceError(f"{logical_label}: metadata.{key} must be {expected!r}")
    if document.get("prefill") not in ({}, None):
        raise EvidenceError(f"{logical_label}: skip-prefill artifact contains prefill results")
    if document.get("burst_results") not in ([], None):
        raise EvidenceError(f"{logical_label}: artifact contains an unapproved burst result")

    prompt = metadata.get("prompt_manifest")
    if not isinstance(prompt, dict):
        raise EvidenceError(f"{logical_label}: prompt_manifest summary is missing")
    prompt_fields = {
        "schema",
        "manifest_schema",
        "mode",
        "content_sha256",
        "payload_sequence_sha256",
        "call_count",
        "consumed_call_count",
        "consumed_all",
    }
    if set(prompt) != prompt_fields:
        raise EvidenceError(
            f"{logical_label}: prompt_manifest contains missing or private/unknown fields"
        )
    required_prompt = {
        "schema": PROMPT_SUMMARY_SCHEMA,
        "manifest_schema": PROMPT_MANIFEST_SCHEMA,
        "mode": "import",
        "consumed_all": True,
    }
    for key, expected in required_prompt.items():
        if prompt.get(key) != expected:
            raise EvidenceError(f"{logical_label}: prompt_manifest.{key} must be {expected!r}")
    call_count = prompt.get("call_count")
    consumed = prompt.get("consumed_call_count")
    if (
        not isinstance(call_count, int)
        or isinstance(call_count, bool)
        or call_count < 1
        or not isinstance(consumed, int)
        or isinstance(consumed, bool)
        or consumed != call_count
    ):
        raise EvidenceError(f"{logical_label}: prompt manifest was not completely consumed")
    for key in ("content_sha256", "payload_sequence_sha256"):
        if not isinstance(prompt.get(key), str) or HEX64.fullmatch(prompt[key]) is None:
            raise EvidenceError(f"{logical_label}: prompt_manifest.{key} is invalid")

    results = document.get("results")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise EvidenceError(f"{logical_label}: results must contain exactly one C8 cell")
    cell = results[0]
    if (
        cell.get("context_tokens") != REQUESTED_CONTEXT_TOKENS
        or cell.get("concurrency") != REQUESTED_CONCURRENCY
    ):
        raise EvidenceError(f"{logical_label}: result is not the exact 16K/C8 cell")
    required_false = ("underfilled", "warmup_timed_out", "capacity_limited")
    if any(cell.get(key) is not False for key in required_false):
        raise EvidenceError(f"{logical_label}: result has an invalid readiness/capacity flag")
    if cell.get("benchmark_mode") != "duration":
        raise EvidenceError(f"{logical_label}: result benchmark_mode must be duration")
    num_errors = cell.get("num_errors")
    effective_concurrency = cell.get("effective_concurrency")
    if (
        not isinstance(num_errors, int)
        or isinstance(num_errors, bool)
        or num_errors != 0
        or not _is_number(effective_concurrency)
        or float(effective_concurrency) != float(REQUESTED_CONCURRENCY)
    ):
        raise EvidenceError(f"{logical_label}: result did not sustain all eight streams")
    throughput = cell.get("aggregate_tps")
    measurement_seconds = cell.get("measurement_seconds")
    if not _is_number(throughput) or throughput <= 0:
        raise EvidenceError(f"{logical_label}: aggregate_tps must be finite and positive")
    if (
        not _is_number(measurement_seconds)
        or abs(float(measurement_seconds) - REQUESTED_DURATION_SECONDS)
        > MEASUREMENT_WINDOW_TOLERANCE_SECONDS
    ):
        raise EvidenceError(
            f"{logical_label}: measurement_seconds is inconsistent with the requested 25s window"
        )
    if cell.get("aggregate_source") != "openai_continuous_usage":
        raise EvidenceError(
            f"{logical_label}: aggregate_source must be openai_continuous_usage"
        )
    output_tokens = cell.get("client_output_tokens")
    total_tokens = cell.get("total_tokens")
    if (
        not isinstance(output_tokens, int)
        or isinstance(output_tokens, bool)
        or output_tokens <= 0
    ):
        raise EvidenceError(
            f"{logical_label}: client_output_tokens must be a positive integer"
        )
    if total_tokens != output_tokens:
        raise EvidenceError(
            f"{logical_label}: total_tokens must equal client_output_tokens for continuous usage"
        )
    derived_tps = output_tokens / float(measurement_seconds)
    if not math.isclose(
        float(throughput),
        derived_tps,
        rel_tol=AGGREGATE_REL_TOLERANCE,
        abs_tol=AGGREGATE_REL_TOLERANCE,
    ):
        raise EvidenceError(
            f"{logical_label}: aggregate_tps is inconsistent with output tokens / measurement_seconds"
        )
    return {
        "logical_label": logical_label,
        "metadata": {key: metadata[key] for key in MATCHED_METADATA},
        "prompt": prompt,
        "aggregate_tps": float(throughput),
        "measurement_seconds": float(measurement_seconds),
        "client_output_tokens": output_tokens,
    }


def _expected_call_descriptors(workload: dict[str, Any]) -> list[dict[str, Any]]:
    context_tokens = workload["contexts"][0]
    descriptors: list[dict[str, Any]] = []
    if float(workload["decode_warmup_seconds"]) > 0:
        descriptors.extend(
            [
                {
                    "phase": "decode-warmup",
                    "request_kind": "prefix-scout",
                    "context_tokens": context_tokens,
                    "concurrency": 1,
                    "benchmark_mode": "request-count",
                },
                {
                    "phase": "decode-warmup",
                    "request_kind": "decode-stream-template",
                    "context_tokens": context_tokens,
                    "concurrency": 1,
                    "benchmark_mode": "duration",
                },
            ]
        )
    descriptors.extend(
        [
            {
                "phase": "decode-measured",
                "request_kind": "prefix-scout",
                "context_tokens": context_tokens,
                "concurrency": 1,
                "benchmark_mode": "request-count",
            },
            {
                "phase": "decode-measured",
                "request_kind": "decode-stream-template",
                "context_tokens": context_tokens,
                "concurrency": REQUESTED_CONCURRENCY,
                "benchmark_mode": "duration",
            },
        ]
    )
    return descriptors


def _validate_authoritative_manifest_summary(summary: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(summary, dict):
        raise EvidenceError("sealed manifest summary must be an object")
    if summary.get("schema") != PROMPT_SUMMARY_SCHEMA:
        raise EvidenceError("sealed manifest summary schema is invalid")
    if summary.get("manifest_schema") != PROMPT_MANIFEST_SCHEMA:
        raise EvidenceError("sealed manifest schema identity is invalid")
    workload = summary.get("workload")
    descriptors = summary.get("ordered_call_descriptors")
    if not isinstance(workload, dict) or not isinstance(descriptors, list):
        raise EvidenceError("sealed manifest lacks its semantic workload/call projection")
    if workload.get("harness") != {
        "name": HARNESS_NAME,
        "version": HARNESS_VERSION,
    }:
        raise EvidenceError(
            f"sealed manifest harness must be {HARNESS_NAME} v{HARNESS_VERSION}"
        )
    exact_requirements = {
        "contexts": [REQUESTED_CONTEXT_TOKENS],
        "concurrencies": [REQUESTED_CONCURRENCY],
        "duration_seconds": REQUESTED_DURATION_SECONDS,
        "temperature": 0.0,
        "ignore_eos": True,
        "unique_context_percent": 100.0,
        "dcp_size": 4,
    }
    for key, expected in exact_requirements.items():
        if workload.get(key) != expected:
            raise EvidenceError(
                f"sealed manifest workload.{key} must be {expected!r}"
            )
    if not isinstance(workload.get("max_tokens"), int) or isinstance(
        workload.get("max_tokens"), bool
    ) or workload["max_tokens"] < 256:
        raise EvidenceError("sealed manifest workload.max_tokens is not sustained")
    if workload.get("reasoning_effort") is not None:
        raise EvidenceError("sealed manifest workload.reasoning_effort must be null")
    expected_calls = _expected_call_descriptors(workload)
    if descriptors != expected_calls:
        raise EvidenceError(
            "sealed manifest ordered call descriptors are not the exact sustained 16K/C8 sequence"
        )
    if summary.get("call_count") != len(expected_calls):
        raise EvidenceError("sealed manifest call_count does not match its call sequence")
    return workload


def compare_abba(
    documents: Sequence[dict[str, Any]],
    artifact_hashes: Sequence[str],
    *,
    manifest_summary: dict[str, Any],
    manifest_artifact_sha256: str,
) -> dict[str, Any]:
    if len(documents) != 4 or len(artifact_hashes) != 4:
        raise EvidenceError("sealed ABBA comparison requires exactly four artifacts")
    if any(
        not isinstance(artifact_hash, str)
        or HEX64.fullmatch(artifact_hash) is None
        for artifact_hash in artifact_hashes
    ):
        raise EvidenceError("benchmark artifact SHA-256 is invalid")
    workload = _validate_authoritative_manifest_summary(manifest_summary)
    logical_labels = ("A-open", "B-first", "B-second", "A-close")
    arms = [validate_arm(document, label) for document, label in zip(documents, logical_labels)]
    baseline_metadata = arms[0]["metadata"]
    for arm in arms[1:]:
        if _canonical(arm["metadata"]) != _canonical(baseline_metadata):
            raise EvidenceError(
                f"{arm['logical_label']}: workload metadata differs from A-open"
            )
    expected_metadata = {
        "version": workload["harness"]["version"],
        "engine": "vllm",
        "model": workload["model"],
        "decode_mode": "duration",
        "duration_per_test": workload["duration_seconds"],
        "max_tokens": workload["max_tokens"],
        "temperature": workload["temperature"],
        "reasoning_effort": workload["reasoning_effort"],
        "ignore_eos": workload["ignore_eos"],
        "dcp_size": workload["dcp_size"],
        "unique_context_percent": workload["unique_context_percent"],
        "shared_context_percent": 100.0 - workload["unique_context_percent"],
        "decode_warmup_seconds": workload["decode_warmup_seconds"],
        "cell_warmup_timeout_seconds": workload[
            "cell_warmup_timeout_seconds"
        ],
        "context_lengths": workload["contexts"],
        "concurrency_levels": workload["concurrencies"],
        "skip_prefill": True,
        "token_targeting": workload["token_targeting"],
    }
    if _canonical(baseline_metadata) != _canonical(expected_metadata):
        differing = sorted(
            key
            for key in MATCHED_METADATA
            if baseline_metadata.get(key) != expected_metadata.get(key)
        )
        raise EvidenceError(
            "A-open: benchmark metadata does not match the authoritative sealed "
            f"manifest workload ({', '.join(differing)})"
        )
    commitment = arms[0]["prompt"]["content_sha256"]
    payload_commitment = arms[0]["prompt"]["payload_sequence_sha256"]
    call_count = arms[0]["prompt"]["call_count"]
    for arm in arms[1:]:
        prompt = arm["prompt"]
        if (
            prompt["content_sha256"] != commitment
            or prompt["payload_sequence_sha256"] != payload_commitment
            or prompt["call_count"] != call_count
        ):
            raise EvidenceError(
                f"{arm['logical_label']}: exact prompt/payload commitment differs from A-open"
            )
    if HEX64.fullmatch(manifest_artifact_sha256) is None:
        raise EvidenceError("sealed manifest artifact SHA-256 is invalid")
    authoritative = {
        "content_sha256": manifest_summary.get("content_sha256"),
        "payload_sequence_sha256": manifest_summary.get(
            "payload_sequence_sha256"
        ),
        "call_count": manifest_summary.get("call_count"),
    }
    observed = {
        "content_sha256": commitment,
        "payload_sequence_sha256": payload_commitment,
        "call_count": call_count,
    }
    if authoritative != observed:
        raise EvidenceError(
            "benchmark prompt commitments do not match the supplied sealed manifest"
        )

    a_values = [arms[0]["aggregate_tps"], arms[3]["aggregate_tps"]]
    b_values = [arms[1]["aggregate_tps"], arms[2]["aggregate_tps"]]
    a_mean = mean(a_values)
    b_mean = mean(b_values)
    return {
        "schema": SCHEMA,
        "passed": True,
        "order": list(logical_labels),
        "exact_prompt_replay": {
            "passed": True,
            "manifest_artifact_sha256": manifest_artifact_sha256,
            "content_sha256": commitment,
            "payload_sequence_sha256": payload_commitment,
            "call_count": call_count,
            "workload": workload,
            "ordered_call_descriptors": manifest_summary[
                "ordered_call_descriptors"
            ],
        },
        "workload": baseline_metadata,
        "arms": [
            {
                "logical_label": arm["logical_label"],
                "artifact_sha256": artifact_hash,
                "aggregate_tps": arm["aggregate_tps"],
                "measurement_seconds": arm["measurement_seconds"],
                "client_output_tokens": arm["client_output_tokens"],
            }
            for arm, artifact_hash in zip(arms, artifact_hashes)
        ],
        "comparison": {
            "a_mean_aggregate_tps": a_mean,
            "b_mean_aggregate_tps": b_mean,
            "b_vs_a_percent": ((b_mean / a_mean) - 1.0) * 100.0,
            "a_close_vs_open_percent": ((a_values[1] / a_values[0]) - 1.0) * 100.0,
            "b_second_vs_first_percent": ((b_values[1] / b_values[0]) - 1.0) * 100.0,
            "interpretation": (
                "descriptive sealed ABBA comparison; two observations per arm do not establish significance"
            ),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-open", type=Path, required=True)
    parser.add_argument("--b-first", type=Path, required=True)
    parser.add_argument("--b-second", type=Path, required=True)
    parser.add_argument("--a-close", type=Path, required=True)
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        required=True,
        help="the exact complete manifest imported by all four measured arms",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = (args.a_open, args.b_first, args.b_second, args.a_close)
    try:
        loaded = [_load(path) for path in paths]
        manifest_encoded = args.prompt_manifest.read_bytes()
        manifest_summary = load_summary(args.prompt_manifest)
        report = compare_abba(
            [item[0] for item in loaded],
            [item[1] for item in loaded],
            manifest_summary=manifest_summary,
            manifest_artifact_sha256=hashlib.sha256(manifest_encoded).hexdigest(),
        )
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as output:
                output.write(encoded)
        else:
            print(encoded, end="")
    except (EvidenceError, FileExistsError, OSError, ValueError) as error:
        print(f"sealed ABBA comparison failed: {error}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
