#!/usr/bin/env python3
"""Reduce and compare two private EXL3 correctness reports for attribution.

This is an offline evidence reducer. Its private inputs may carry the exact
contacted API origin; its closed output excludes that origin, hostnames, paths,
prompt text, and completion text. It compares retokenized completion token IDs
exactly and, when present, separately compares the persisted OpenAI
``top_logprobs`` observations. The latter are truncated observations and are
never represented as full-vocabulary KL divergence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from exl3_attribution_cache_contract import (
    ARM_MTP_TOKENS,
    LIVE_ARM_RECEIPT_SCHEMA,
    cache_salt_for_arm,
    expected_layout,
)


LEGACY_INPUT_SCHEMA = "sparkring-exl3-correctness-report/v1"
LEGACY_INPUT_SCHEMA_V2 = "sparkring-exl3-correctness-report/v2"
LEGACY_INPUT_SCHEMA_V3 = "sparkring-exl3-correctness-report/v3"
INPUT_SCHEMA = "sparkring-exl3-correctness-report/v4"
OUTPUT_SCHEMA = "sparkring-exl3-attribution-comparison/v1"
EXIT_OK = 0
EXIT_INVALID = 3
SHA256_RE = re.compile(r"[0-9a-f]{64}")
LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}")
CASE_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
PHYSICAL_BLOCK_TOKENS_PER_DCP_RANK = 64
DCP_DEGREE = 4
DCP_GLOBAL_APC_ALIGNMENT_TOKENS = 256
LMCACHE_CHUNK_TOKENS = 512
METRIC_COUNTERS = {
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
}
METRIC_SOURCES = {"local_compute", "local_cache_hit", "external_kv_transfer"}
MAX_TOKEN_ID = 2**31 - 1
MAX_TOKEN_COUNT = 2**31 - 1
MAX_SEED = 2**63 - 1
MAX_COMPLETION_TOKENS = 10_000_000
MIN_LOGPROB = -1_000_000.0


class ComparisonError(ValueError):
    """An input is invalid or the pair is not comparable."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ComparisonError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(message)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _token_ids_sha256(token_ids: list[int]) -> str:
    encoded = json.dumps(token_ids, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(encoded.encode("utf-8"))


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        document = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except ComparisonError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"cannot read report {path}: {exc}") from exc
    _require(isinstance(document, dict), f"report {path} must be a JSON object")
    return document, _sha256_bytes(raw)


def _validate_logprobs(value: Any, prefix: str, requested: int) -> dict[str, Any] | None:
    if value is None:
        _require(requested == 0, f"{prefix} omitted requested completion_logprobs")
        return None
    _require(requested > 0, f"{prefix} persisted completion_logprobs when none were requested")
    _require(isinstance(value, dict), f"{prefix}.completion_logprobs must be an object")
    required = {"tokens", "token_logprobs", "top_logprobs"}
    _require(required <= set(value), f"{prefix}.completion_logprobs lacks {sorted(required - set(value))}")
    tokens = value["tokens"]
    chosen = value["token_logprobs"]
    tops = value["top_logprobs"]
    _require(isinstance(tokens, list), f"{prefix}.tokens must be a list")
    _require(isinstance(chosen, list), f"{prefix}.token_logprobs must be a list")
    _require(isinstance(tops, list), f"{prefix}.top_logprobs must be a list")
    _require(
        len(tokens) == len(chosen) == len(tops),
        f"{prefix} completion_logprobs arrays must have equal lengths",
    )
    for position, (token, logprob, top) in enumerate(zip(tokens, chosen, tops)):
        item = f"{prefix}.completion_logprobs[{position}]"
        _require(isinstance(token, str) and len(token) <= 8192, f"{item} chosen token must be a bounded string")
        _require(logprob is None or _is_number(logprob), f"{item} chosen logprob must be finite or null")
        _require(logprob is None or MIN_LOGPROB <= float(logprob) <= 1e-9, f"{item} chosen logprob is outside the safe range")
        _require(isinstance(top, dict), f"{item} top_logprobs must be an object")
        _require(bool(top), f"{item} top_logprobs must be non-empty")
        probability_sum = 0.0
        for candidate, candidate_logprob in top.items():
            _require(isinstance(candidate, str) and len(candidate) <= 8192, f"{item} candidate token must be a bounded string")
            _require(_is_number(candidate_logprob), f"{item} candidate logprob must be finite")
            candidate_logprob = float(candidate_logprob)
            _require(MIN_LOGPROB <= candidate_logprob <= 1e-9, f"{item} candidate logprob is outside the safe range")
            probability_sum += math.exp(candidate_logprob)
        _require(probability_sum <= 1.000001, f"{item} persisted top-token probability mass exceeds 1")
    return value


def _validate_report(
    document: dict[str, Any], source: str, *, allow_legacy_v1: bool = False
) -> dict[str, Any]:
    schema = document.get("schema")
    legacy_schemas = (
        LEGACY_INPUT_SCHEMA,
        LEGACY_INPUT_SCHEMA_V2,
        LEGACY_INPUT_SCHEMA_V3,
    )
    _require(
        schema in (*legacy_schemas, INPUT_SCHEMA),
        f"{source}.schema must be {LEGACY_INPUT_SCHEMA}, "
        f"{LEGACY_INPUT_SCHEMA_V2}, {LEGACY_INPUT_SCHEMA_V3}, or {INPUT_SCHEMA}",
    )
    current_format = schema == INPUT_SCHEMA
    _require(
        current_format or allow_legacy_v1,
        f"{source} legacy v1/v2 input requires explicit --allow-legacy-v1",
    )
    model = document.get("model")
    run_label = document.get("run_label")
    config_sha = document.get("config_sha256")
    repetitions = document.get("repetitions")
    top_logprobs = document.get("top_logprobs")
    attribution_arm = document.get("attribution_arm") if current_format else None
    cache_salt = document.get("cache_salt") if current_format else None
    _require(isinstance(model, str) and MODEL_RE.fullmatch(model) is not None, f"{source}.model must be a sanitized model identifier")
    _require(isinstance(run_label, str) and LABEL_RE.fullmatch(run_label) is not None, f"{source}.run_label is not a sanitized label")
    _require(isinstance(config_sha, str) and SHA256_RE.fullmatch(config_sha) is not None, f"{source}.config_sha256 must be lowercase SHA-256")
    _require(_is_int(repetitions) and repetitions >= 2, f"{source}.repetitions must be >= 2")
    _require(_is_int(top_logprobs) and 0 <= top_logprobs <= 20, f"{source}.top_logprobs must be in [0, 20]")
    if current_format:
        _require(
            attribution_arm in ARM_MTP_TOKENS,
            f"{source}.attribution_arm is unsupported",
        )
        _require(
            cache_salt == cache_salt_for_arm(attribution_arm),
            f"{source}.cache_salt does not bind its attribution layout",
        )
        live_receipt = document.get("live_arm_receipt")
        receipt_keys = {
            "schema", "status", "arm", "canonical_profile_id",
            "diagnostic_profile_id", "canonical_profile_file_sha256",
            "image_id", "model_repository", "model_revision", "layout",
            "cache_salt", "ranks", "attestation_scope", "artifact_sha256",
        }
        _require(
            isinstance(live_receipt, dict) and set(live_receipt) == receipt_keys,
            f"{source}.live_arm_receipt is missing or has unsupported fields",
        )
        _require(
            live_receipt["schema"] == LIVE_ARM_RECEIPT_SCHEMA
            and live_receipt["status"] == "live-arm-attested",
            f"{source}.live_arm_receipt is not an attested receipt",
        )
        _require(
            live_receipt["arm"] == attribution_arm
            and live_receipt["cache_salt"] == cache_salt
            and live_receipt["layout"] == expected_layout(attribution_arm),
            f"{source}.live_arm_receipt does not bind arm/layout/cache salt",
        )
        for field in ("canonical_profile_file_sha256", "artifact_sha256"):
            _require(
                isinstance(live_receipt[field], str)
                and SHA256_RE.fullmatch(live_receipt[field]) is not None,
                f"{source}.live_arm_receipt.{field} must be SHA-256",
            )
        _require(
            isinstance(live_receipt["image_id"], str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", live_receipt["image_id"])
            is not None,
            f"{source}.live_arm_receipt.image_id is invalid",
        )
        _require(
            isinstance(live_receipt["model_revision"], str)
            and re.fullmatch(r"[0-9a-f]{40}", live_receipt["model_revision"])
            is not None,
            f"{source}.live_arm_receipt.model_revision is invalid",
        )
        _require(
            isinstance(live_receipt["canonical_profile_id"], str)
            and live_receipt["diagnostic_profile_id"]
            == f'{live_receipt["canonical_profile_id"]}-diag-{attribution_arm}',
            f"{source}.live_arm_receipt profile identity is invalid",
        )
        _require(
            isinstance(live_receipt["model_repository"], str)
            and bool(live_receipt["model_repository"]),
            f"{source}.live_arm_receipt model repository is invalid",
        )
        ranks = live_receipt["ranks"]
        _require(
            isinstance(ranks, list) and len(ranks) == 4,
            f"{source}.live_arm_receipt must attest four ordered ranks",
        )
        rank_keys = {
            "rank", "status", "container_name", "labels", "image_id",
            "container_id", "started_at",
            "explicit_environment_sha256", "config_cmd_sha256",
        }
        expected_labels = {
            "org.sparkring.managed": "true",
            "org.sparkring.exl3-profile": live_receipt[
                "diagnostic_profile_id"
            ],
            "org.sparkring.component": "engine",
            "org.sparkring.exl3-attribution": attribution_arm,
        }
        for rank, rank_receipt in enumerate(ranks):
            _require(
                isinstance(rank_receipt, dict)
                and set(rank_receipt) == rank_keys,
                f"{source}.live_arm_receipt rank fields are unsupported",
            )
            _require(
                rank_receipt["rank"] == rank
                and rank_receipt["status"] == "attested",
                f"{source}.live_arm_receipt ranks are not ordered attestations",
            )
            _require(
                isinstance(rank_receipt["container_name"], str)
                and rank_receipt["container_name"].endswith(
                    f"-diag-{attribution_arm}-r{rank}"
                ),
                f"{source}.live_arm_receipt container identity is invalid",
            )
            _require(
                rank_receipt["labels"] == expected_labels
                and rank_receipt["image_id"] == live_receipt["image_id"],
                f"{source}.live_arm_receipt rank identity does not bind report",
            )
            for field in (
                "explicit_environment_sha256", "config_cmd_sha256"
            ):
                _require(
                    isinstance(rank_receipt[field], str)
                    and SHA256_RE.fullmatch(rank_receipt[field]) is not None,
                    f"{source}.live_arm_receipt rank {field} is invalid",
                )
            _require(
                isinstance(rank_receipt["container_id"], str)
                and re.fullmatch(r"[0-9a-f]{64}", rank_receipt["container_id"])
                is not None,
                f"{source}.live_arm_receipt rank container ID is invalid",
            )
            _require(
                isinstance(rank_receipt["started_at"], str)
                and re.fullmatch(
                    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z",
                    rank_receipt["started_at"],
                )
                is not None,
                f"{source}.live_arm_receipt rank StartedAt is invalid",
            )
        _require(
            isinstance(live_receipt["attestation_scope"], str)
            and bool(live_receipt["attestation_scope"]),
            f"{source}.live_arm_receipt attestation scope is invalid",
        )
        live_receipt_sha256 = live_receipt["artifact_sha256"]
    else:
        live_receipt_sha256 = None
    cases = document.get("cases")
    _require(isinstance(cases, list) and cases, f"{source}.cases must be non-empty")
    seen: set[str] = set()
    normalized_cases: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        prefix = f"{source}.cases[{case_index}]"
        _require(isinstance(case, dict), f"{prefix} must be an object")
        case_id = case.get("id")
        _require(isinstance(case_id, str) and CASE_RE.fullmatch(case_id) is not None, f"{prefix}.id must be lowercase kebab-case")
        _require(case_id not in seen, f"{source} has duplicate case id {case_id!r}")
        seen.add(case_id)
        _require(isinstance(case.get("prompt_sha256"), str) and SHA256_RE.fullmatch(case["prompt_sha256"]) is not None, f"{prefix}.prompt_sha256 must be lowercase SHA-256")
        prompt_token_ids_sha256 = case.get("prompt_token_ids_sha256")
        _require(prompt_token_ids_sha256 is None or (isinstance(prompt_token_ids_sha256, str) and SHA256_RE.fullmatch(prompt_token_ids_sha256) is not None), f"{prefix}.prompt_token_ids_sha256 must be lowercase SHA-256 when present")
        prompt_token_count = case.get("prompt_token_count")
        _require(prompt_token_count is None or (_is_int(prompt_token_count) and 0 <= prompt_token_count <= MAX_TOKEN_COUNT), f"{prefix}.prompt_token_count must be a bounded nonnegative integer when present")
        if current_format:
            _require(prompt_token_ids_sha256 is not None, f"{prefix}.prompt_token_ids_sha256 is required by current format")
            _require(prompt_token_count is not None, f"{prefix}.prompt_token_count is required by current format")
            _require(isinstance(case.get("ignore_eos"), bool), f"{prefix}.ignore_eos must be boolean")
            _require(isinstance(case.get("cache_attribution"), bool), f"{prefix}.cache_attribution must be boolean")
            boundaries = case.get("cache_boundaries")
            required_boundaries = {
                "physical_block_tokens_per_dcp_rank",
                "dcp_degree",
                "dcp_global_apc_alignment_tokens",
                "lmcache_chunk_tokens",
                "minimum_prompt_tokens",
                "reusable_prompt_tokens",
                "reusable_dcp_global_apc_units",
                "reusable_lmcache_chunks",
                "has_reusable_dcp_global_apc_unit",
                "has_reusable_lmcache_chunk",
                "qualifies_for_cache_attribution",
            }
            _require(isinstance(boundaries, dict), f"{prefix}.cache_boundaries must be an object")
            _require(set(boundaries) == required_boundaries, f"{prefix}.cache_boundaries keys must be {sorted(required_boundaries)}")
            _require(boundaries["physical_block_tokens_per_dcp_rank"] == PHYSICAL_BLOCK_TOKENS_PER_DCP_RANK, f"{prefix}.cache_boundaries physical block must be canonical {PHYSICAL_BLOCK_TOKENS_PER_DCP_RANK}")
            _require(boundaries["dcp_degree"] == DCP_DEGREE, f"{prefix}.cache_boundaries DCP degree must be canonical {DCP_DEGREE}")
            _require(boundaries["dcp_global_apc_alignment_tokens"] == DCP_GLOBAL_APC_ALIGNMENT_TOKENS, f"{prefix}.cache_boundaries global APC alignment must be canonical {DCP_GLOBAL_APC_ALIGNMENT_TOKENS}")
            _require(boundaries["lmcache_chunk_tokens"] == LMCACHE_CHUNK_TOKENS, f"{prefix}.cache_boundaries.lmcache_chunk_tokens must be canonical {LMCACHE_CHUNK_TOKENS}")
            minimum = boundaries["minimum_prompt_tokens"]
            _require(minimum is None or (_is_int(minimum) and LMCACHE_CHUNK_TOKENS <= minimum <= MAX_TOKEN_COUNT), f"{prefix}.cache_boundaries.minimum_prompt_tokens must be null or a bounded integer >= {LMCACHE_CHUNK_TOKENS}")
            reusable = max(prompt_token_count - 1, 0)
            for field in ("reusable_prompt_tokens", "reusable_dcp_global_apc_units", "reusable_lmcache_chunks"):
                _require(_is_int(boundaries[field]) and boundaries[field] >= 0, f"{prefix}.cache_boundaries.{field} must be nonnegative integer")
            _require(boundaries["reusable_prompt_tokens"] == reusable, f"{prefix}.cache_boundaries must use prompt_tokens-1 reusable semantics")
            _require(boundaries["reusable_dcp_global_apc_units"] == reusable // DCP_GLOBAL_APC_ALIGNMENT_TOKENS, f"{prefix}.cache_boundaries APC reusable-unit count is inconsistent")
            _require(boundaries["reusable_lmcache_chunks"] == reusable // LMCACHE_CHUNK_TOKENS, f"{prefix}.cache_boundaries LMCache reusable-unit count is inconsistent")
            for field in ("has_reusable_dcp_global_apc_unit", "has_reusable_lmcache_chunk", "qualifies_for_cache_attribution"):
                _require(isinstance(boundaries[field], bool), f"{prefix}.cache_boundaries.{field} must be boolean")
            _require(boundaries["has_reusable_dcp_global_apc_unit"] == (reusable >= DCP_GLOBAL_APC_ALIGNMENT_TOKENS), f"{prefix}.cache_boundaries APC reusable-unit flag disagrees with prompt count")
            _require(boundaries["has_reusable_lmcache_chunk"] == (reusable >= LMCACHE_CHUNK_TOKENS), f"{prefix}.cache_boundaries LMCache reusable-unit flag disagrees with prompt count")
            expected_qualifies = bool(
                case["cache_attribution"]
                and minimum is not None
                and reusable >= DCP_GLOBAL_APC_ALIGNMENT_TOKENS
                and reusable >= LMCACHE_CHUNK_TOKENS
                and prompt_token_count >= minimum
            )
            _require(boundaries["qualifies_for_cache_attribution"] == expected_qualifies, f"{prefix}.cache_boundaries qualification is inconsistent")
            probe = case.get("cache_metric_probe")
            if case["cache_attribution"]:
                _require(isinstance(probe, dict), f"{prefix}.cache_metric_probe must bind a validated artifact")
                required_probe = {
                    "artifact_sha256", "schema", "run_label", "model",
                    "attribution_arm", "cache_salt",
                    "live_arm_receipt_sha256", "live_arm_receipt",
                    "probe_prompt_sha256", "probe_prompt_token_count",
                    "probe_prompt_token_ids_sha256",
                    "geometry", "native_prefix_caching_enabled",
                    "aggregate_counter_deltas",
                    "aggregate_prompt_tokens_by_source",
                    "observed_cache_layers", "observation_count",
                    "evidence_classification",
                }
                _require(set(probe) == required_probe, f"{prefix}.cache_metric_probe keys are unsupported")
                for field in ("artifact_sha256", "probe_prompt_token_ids_sha256"):
                    _require(isinstance(probe[field], str) and SHA256_RE.fullmatch(probe[field]) is not None, f"{prefix}.cache_metric_probe.{field} must be SHA-256")
                _require(probe["probe_prompt_sha256"] is None or (isinstance(probe["probe_prompt_sha256"], str) and SHA256_RE.fullmatch(probe["probe_prompt_sha256"]) is not None), f"{prefix}.cache_metric_probe.probe_prompt_sha256 must be SHA-256 or null")
                _require(probe["schema"] == "sparkring-exl3-cache-metric-probe/v2", f"{prefix}.cache_metric_probe schema is unsupported")
                _require(probe["model"] == model, f"{prefix}.cache_metric_probe model does not bind report")
                _require(probe["attribution_arm"] == attribution_arm, f"{prefix}.cache_metric_probe attribution arm does not bind report")
                _require(probe["cache_salt"] == cache_salt, f"{prefix}.cache_metric_probe cache salt does not bind report")
                _require(isinstance(probe["live_arm_receipt_sha256"], str) and SHA256_RE.fullmatch(probe["live_arm_receipt_sha256"]) is not None, f"{prefix}.cache_metric_probe live-arm receipt digest is invalid")
                _require(probe["live_arm_receipt_sha256"] == live_receipt_sha256, f"{prefix}.cache_metric_probe live-arm receipt does not bind report")
                _require(probe["live_arm_receipt"] == live_receipt, f"{prefix}.cache_metric_probe embedded live-arm receipt does not bind report")
                _require(_is_int(probe["probe_prompt_token_count"]) and probe["probe_prompt_token_count"] > 0, f"{prefix}.cache_metric_probe probe prompt count must be positive")
                _require(probe["probe_prompt_sha256"] == case["prompt_sha256"], f"{prefix}.cache_metric_probe prompt text digest does not bind case")
                _require(probe["probe_prompt_token_ids_sha256"] == prompt_token_ids_sha256, f"{prefix}.cache_metric_probe prompt token digest does not bind case")
                _require(probe["probe_prompt_token_count"] == prompt_token_count, f"{prefix}.cache_metric_probe prompt token count does not bind case")
                _require(probe["geometry"] == {
                    "physical_block_tokens_per_dcp_rank": PHYSICAL_BLOCK_TOKENS_PER_DCP_RANK,
                    "dcp_degree": DCP_DEGREE,
                    "dcp_global_apc_alignment_tokens": DCP_GLOBAL_APC_ALIGNMENT_TOKENS,
                    "lmcache_chunk_tokens": LMCACHE_CHUNK_TOKENS,
                    "physical_blocks_per_dcp_global_apc_unit_per_rank": 1,
                    "dcp_global_apc_units_per_lmcache_chunk": 2,
                    "physical_blocks_per_lmcache_chunk_per_rank": 2,
                }, f"{prefix}.cache_metric_probe geometry is unsupported")
                _require(isinstance(probe["native_prefix_caching_enabled"], bool), f"{prefix}.cache_metric_probe native prefix state must be boolean")
                aggregate_counters = probe["aggregate_counter_deltas"]
                aggregate_sources = probe["aggregate_prompt_tokens_by_source"]
                _require(isinstance(aggregate_counters, dict) and set(aggregate_counters) == METRIC_COUNTERS, f"{prefix}.cache_metric_probe counter family is incomplete")
                _require(isinstance(aggregate_sources, dict) and set(aggregate_sources) == METRIC_SOURCES, f"{prefix}.cache_metric_probe prompt source family is incomplete")
                _require(all(_is_number(value) and float(value) >= 0 for value in [*aggregate_counters.values(), *aggregate_sources.values()]), f"{prefix}.cache_metric_probe aggregate metrics must be finite nonnegative numbers")
                if not probe["native_prefix_caching_enabled"]:
                    _require(
                        aggregate_counters["vllm:prefix_cache_queries_total"] == 0
                        and aggregate_counters["vllm:prefix_cache_hits_total"] == 0
                        and aggregate_sources["local_cache_hit"] == 0,
                        f"{prefix}.cache_metric_probe reports native cache activity while native prefix caching is disabled",
                    )
                expected_layers = [
                    layer
                    for layer, observed in (
                        ("native-apc", aggregate_sources["local_cache_hit"] > 0),
                        (
                            "external-kv-transfer",
                            aggregate_sources["external_kv_transfer"] > 0,
                        ),
                    )
                    if observed
                ]
                _require(probe["observed_cache_layers"] == expected_layers, f"{prefix}.cache_metric_probe observed cache layers are inconsistent")
                _require(_is_int(probe["observation_count"]) and probe["observation_count"] >= 2, f"{prefix}.cache_metric_probe observation_count must be >= 2")
                _require(isinstance(probe["run_label"], str) and LABEL_RE.fullmatch(probe["run_label"]) is not None, f"{prefix}.cache_metric_probe run_label is invalid")
                _require(probe["evidence_classification"] == "request-interval-correlated-prometheus-counter-delta", f"{prefix}.cache_metric_probe evidence classification is unsupported")
            else:
                _require(probe is None, f"{prefix}.cache_metric_probe must be null for a non-attribution case")
            cache_scope = case.get("cache_evidence_scope")
            _require(isinstance(cache_scope, dict), f"{prefix}.cache_evidence_scope must be an object")
            required_scope = {
                "boundary_geometry_bound",
                "request_ids_bound",
                "request_count",
                "request_correlated_hit_evidence_count",
                "request_correlated_store_evidence_count",
                "causal_cache_claim",
                "statement",
            }
            _require(set(cache_scope) == required_scope, f"{prefix}.cache_evidence_scope keys must be {sorted(required_scope)}")
        for field in ("seed", "max_tokens"):
            upper = MAX_SEED if field == "seed" else MAX_COMPLETION_TOKENS
            _require(_is_int(case.get(field)) and 0 <= case[field] <= upper, f"{prefix}.{field} must be a bounded nonnegative integer")
        _require(case.get("token_id_source") == "retokenized-completion-text", f"{prefix}.token_id_source is unsupported")
        _require(case.get("requested_top_logprobs") == top_logprobs, f"{prefix}.requested_top_logprobs disagrees with report")
        observations = case.get("observations")
        _require(isinstance(observations, list), f"{prefix}.observations must be a list")
        _require(len(observations) == repetitions, f"{prefix} must contain exactly {repetitions} observations")
        normalized_observations = []
        request_id_count = 0
        hit_evidence_count = 0
        for observation_index, observation in enumerate(observations):
            op = f"{prefix}.observations[{observation_index}]"
            _require(isinstance(observation, dict), f"{op} must be an object")
            _require(observation.get("repetition") == observation_index + 1, f"{op}.repetition must be {observation_index + 1}")
            token_ids = observation.get("token_ids")
            _require(isinstance(token_ids, list) and token_ids, f"{op}.token_ids must be non-empty")
            _require(len(token_ids) <= MAX_COMPLETION_TOKENS, f"{op}.token_ids is unreasonably large")
            _require(all(_is_int(token_id) and 0 <= token_id <= MAX_TOKEN_ID for token_id in token_ids), f"{op}.token_ids must be bounded nonnegative integers")
            token_hash = observation.get("token_ids_sha256")
            _require(isinstance(token_hash, str) and SHA256_RE.fullmatch(token_hash) is not None, f"{op}.token_ids_sha256 must be lowercase SHA-256")
            _require(token_hash == _token_ids_sha256(token_ids), f"{op}.token_ids_sha256 does not bind token_ids")
            logprobs = _validate_logprobs(observation.get("completion_logprobs"), op, top_logprobs)
            if current_format:
                request_evidence = observation.get("request_evidence")
                _require(isinstance(request_evidence, dict), f"{op}.request_evidence must be an object")
                request_hash = request_evidence.get("response_id_sha256")
                _require(request_hash is None or (isinstance(request_hash, str) and SHA256_RE.fullmatch(request_hash) is not None), f"{op}.request_evidence.response_id_sha256 must be SHA-256 or null")
                for field in ("usage_prompt_tokens", "usage_cached_prompt_tokens"):
                    value = request_evidence.get(field)
                    _require(value is None or (_is_int(value) and 0 <= value <= MAX_TOKEN_COUNT), f"{op}.request_evidence.{field} must be bounded nonnegative or null")
                usage_prompt = request_evidence.get("usage_prompt_tokens")
                usage_cached = request_evidence.get("usage_cached_prompt_tokens")
                _require(usage_prompt is None or usage_prompt == prompt_token_count, f"{op}.request_evidence usage prompt count does not bind the tokenized prompt")
                _require(usage_cached is None or (usage_prompt is not None and usage_cached <= usage_prompt), f"{op}.request_evidence cached tokens exceed prompt tokens")
                expected_hit_source = (
                    "openai-usage.prompt_tokens_details.cached_tokens"
                    if usage_cached is not None and usage_cached > 0
                    else None
                )
                _require(request_evidence.get("hit_evidence_source") == expected_hit_source, f"{op}.request_evidence hit source is inconsistent")
                _require(request_evidence.get("store_evidence_source") is None, f"{op}.request_evidence cannot claim unavailable store evidence")
                request_id_count += request_hash is not None
                hit_evidence_count += usage_cached is not None and usage_cached > 0
            normalized_observations.append({"token_ids": token_ids, "token_ids_sha256": token_hash, "completion_logprobs": logprobs})
        if current_format:
            for field in (
                "request_ids_bound",
                "request_count",
                "request_correlated_hit_evidence_count",
                "request_correlated_store_evidence_count",
            ):
                _require(_is_int(cache_scope[field]) and cache_scope[field] >= 0, f"{prefix}.cache_evidence_scope.{field} must be a nonnegative integer")
            _require(isinstance(cache_scope["boundary_geometry_bound"], bool), f"{prefix}.cache_evidence_scope.boundary_geometry_bound must be boolean")
            _require(cache_scope["boundary_geometry_bound"] == case["cache_attribution"], f"{prefix}.cache_evidence_scope boundary binding is inconsistent")
            _require(cache_scope["request_count"] == repetitions, f"{prefix}.cache_evidence_scope request_count is inconsistent")
            _require(cache_scope["request_ids_bound"] == request_id_count, f"{prefix}.cache_evidence_scope request ID count is inconsistent")
            _require(cache_scope["request_correlated_hit_evidence_count"] == hit_evidence_count, f"{prefix}.cache_evidence_scope hit count is inconsistent")
            _require(cache_scope["request_correlated_store_evidence_count"] == 0, f"{prefix}.cache_evidence_scope store count is unsupported")
            _require(isinstance(cache_scope["causal_cache_claim"], str) and cache_scope["causal_cache_claim"], f"{prefix}.cache_evidence_scope causal claim must be non-empty")
            _require(isinstance(cache_scope["statement"], str) and cache_scope["statement"], f"{prefix}.cache_evidence_scope statement must be non-empty")
        alignment = {
            field: case.get(field)
            for field in (
                "prompt_sha256",
                "seed",
                "max_tokens",
                "token_id_source",
                "requested_top_logprobs",
            )
        }
        if current_format:
            alignment.update(
                {
                    "ignore_eos": case["ignore_eos"],
                    "cache_attribution": case["cache_attribution"],
                    "cache_boundaries": case["cache_boundaries"],
                    "cache_evidence_contract": {
                "boundary_geometry_bound": cache_scope["boundary_geometry_bound"],
                "causal_cache_claim": cache_scope["causal_cache_claim"],
                "request_correlated_store_evidence_count": cache_scope[
                    "request_correlated_store_evidence_count"
                ],
                    },
                }
            )
        normalized_cases.append(
            {
                "id": case_id,
                "geometry_evidence_scope": (
                    "validated-metric-probe-and-cache-namespace-bound-v4"
                    if current_format and case["cache_attribution"]
                    else (
                        "non-cache-v4-cache-namespace-bound-no-cache-geometry-claim"
                        if current_format
                        else "unbound-legacy-token-output-only"
                    )
                ),
                "alignment": alignment,
                "optional_alignment": {
                    "prompt_token_ids_sha256": prompt_token_ids_sha256,
                    "prompt_token_count": prompt_token_count,
                },
                "observations": normalized_observations,
                "cache_metric_probe": case.get("cache_metric_probe") if current_format else None,
            }
        )
    return {
        "model": model,
        "schema": schema,
        "current_format": current_format,
        "run_label": run_label,
        "config_sha256": config_sha,
        "repetitions": repetitions,
        "top_logprobs": top_logprobs,
        "attribution_arm": attribution_arm,
        "cache_salt": cache_salt,
        "live_arm_receipt_sha256": live_receipt_sha256,
        "cases": normalized_cases,
    }


def _sequence_comparison(left: dict[str, Any], right: dict[str, Any], repetition: int) -> dict[str, Any]:
    left_ids = left["token_ids"]
    right_ids = right["token_ids"]
    prefix = 0
    for left_id, right_id in zip(left_ids, right_ids):
        if left_id != right_id:
            break
        prefix += 1
    exact = left_ids == right_ids
    return {
        "repetition": repetition,
        "exact_match": exact,
        "first_divergence_index": None if exact else prefix,
        "prefix_match_length": prefix,
        "left_sequence_length": len(left_ids),
        "right_sequence_length": len(right_ids),
        "left_token_ids_sha256": left["token_ids_sha256"],
        "right_token_ids_sha256": right["token_ids_sha256"],
    }


def _conditional_common_support_symmetric_kl(left: dict[str, float], right: dict[str, float], common: list[str]) -> float | None:
    if not common:
        return None
    left_log_norm = _logsumexp([float(left[token]) for token in common])
    right_log_norm = _logsumexp([float(right[token]) for token in common])
    left_kl = 0.0
    right_kl = 0.0
    for token in common:
        left_log = float(left[token]) - left_log_norm
        right_log = float(right[token]) - right_log_norm
        left_probability = math.exp(left_log)
        right_probability = math.exp(right_log)
        left_kl += left_probability * (left_log - right_log)
        right_kl += right_probability * (right_log - left_log)
    # Clamp only sub-ulp cancellation around the mathematical lower bound zero.
    result = 0.5 * (left_kl + right_kl)
    return 0.0 if -1e-15 < result < 0.0 else result


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _logprob_comparison(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    *,
    retokenized_shared_prefix: int,
) -> dict[str, Any]:
    if left is None and right is None:
        return {"available": False, "reason": "top-logprobs-not-persisted"}
    _require(left is not None and right is not None, "aligned observations disagree on logprob availability")
    left_tokens = left["tokens"]
    right_tokens = right["tokens"]
    emitted_shared_prefix = 0
    for left_token, right_token in zip(left_tokens, right_tokens):
        if left_token != right_token:
            break
        emitted_shared_prefix += 1
    position_count = min(
        len(left_tokens),
        len(right_tokens),
        emitted_shared_prefix,
        retokenized_shared_prefix,
    )
    positions = []
    for index in range(position_count):
        left_top = left["top_logprobs"][index]
        right_top = right["top_logprobs"][index]
        common = sorted(set(left_top) & set(right_top))
        left_chosen_lp = left["token_logprobs"][index]
        right_chosen_lp = right["token_logprobs"][index]
        positions.append(
            {
                "position": index,
                "chosen_token_match": left_tokens[index] == right_tokens[index],
                "left_chosen_token_logprob": left_chosen_lp,
                "right_chosen_token_logprob": right_chosen_lp,
                "chosen_token_logprob_delta_left_minus_right": (
                    None
                    if (
                        left_tokens[index] != right_tokens[index]
                        or left_chosen_lp is None
                        or right_chosen_lp is None
                    )
                    else float(left_chosen_lp) - float(right_chosen_lp)
                ),
                "persisted_top_token_count_left": len(left_top),
                "persisted_top_token_count_right": len(right_top),
                "common_persisted_token_count": len(common),
                "common_token_mass_left": sum(math.exp(float(left_top[token])) for token in common),
                "common_token_mass_right": sum(math.exp(float(right_top[token])) for token in common),
                "conditional_common_support_symmetric_kl": _conditional_common_support_symmetric_kl(left_top, right_top, common),
            }
        )
    return {
        "available": True,
        "alignment": "same-context positions only: shared emitted-token prefix, conservatively bounded by the retokenized shared prefix",
        "left_position_count": len(left_tokens),
        "right_position_count": len(right_tokens),
        "aligned_position_count": position_count,
        "different_context_or_unproven_position_count_left": len(left_tokens) - position_count,
        "different_context_or_unproven_position_count_right": len(right_tokens) - position_count,
        "later_position_policy": "omitted-different-context-or-unproven",
        "positions": positions,
    }


def _within_arm(case: dict[str, Any]) -> dict[str, Any]:
    observations = case["observations"]
    reference = observations[0]
    comparisons = [
        _sequence_comparison(reference, observation, index + 1)
        for index, observation in enumerate(observations)
    ]
    return {
        "stable_across_repetitions": len({item["token_ids_sha256"] for item in observations}) == 1,
        "distinct_sequence_count": len({item["token_ids_sha256"] for item in observations}),
        "repetitions_vs_repetition_1": comparisons,
    }


def compare_reports(left: dict[str, Any], right: dict[str, Any], *, left_sha256: str, right_sha256: str, allow_legacy_v1: bool = False) -> dict[str, Any]:
    left = _validate_report(left, "left", allow_legacy_v1=allow_legacy_v1)
    right = _validate_report(right, "right", allow_legacy_v1=allow_legacy_v1)
    _require(
        left["current_format"] == right["current_format"],
        "current v4 and legacy v1/v2/v3 reports cannot be mixed",
    )
    for field in ("model", "config_sha256", "repetitions", "top_logprobs"):
        _require(left[field] == right[field], f"left/right {field} values do not align")
    _require([case["id"] for case in left["cases"]] == [case["id"] for case in right["cases"]], "left/right ordered case IDs do not align")
    results = []
    total_pairs = 0
    exact_pairs = 0
    for left_case, right_case in zip(left["cases"], right["cases"]):
        _require(left_case["alignment"] == right_case["alignment"], f"case {left_case['id']!r} metadata does not align")
        _require(
            left_case["geometry_evidence_scope"]
            == right_case["geometry_evidence_scope"],
            f"case {left_case['id']!r} cache geometry evidence scope does not align",
        )
        optional_alignment = {}
        for field in ("prompt_token_ids_sha256", "prompt_token_count"):
            left_value = left_case["optional_alignment"][field]
            right_value = right_case["optional_alignment"][field]
            if left_value is not None and right_value is not None:
                _require(left_value == right_value, f"case {left_case['id']!r} {field} does not align")
            optional_alignment[field] = {
                "left_present": left_value is not None,
                "right_present": right_value is not None,
                "compared_when_present_in_both": left_value is not None and right_value is not None,
            }
        between = []
        for repetition, (left_observation, right_observation) in enumerate(zip(left_case["observations"], right_case["observations"]), 1):
            sequence = _sequence_comparison(left_observation, right_observation, repetition)
            total_pairs += 1
            exact_pairs += int(sequence["exact_match"])
            sequence["completion_logprobs"] = _logprob_comparison(
                left_observation["completion_logprobs"],
                right_observation["completion_logprobs"],
                retokenized_shared_prefix=sequence["prefix_match_length"],
            )
            between.append(sequence)
        results.append(
            {
                "case_id": left_case["id"],
                "cache_geometry_evidence": {
                    "scope": left_case["geometry_evidence_scope"],
                    "left_metric_probe_artifact_sha256": (
                        left_case["cache_metric_probe"]["artifact_sha256"]
                        if left_case["cache_metric_probe"] is not None
                        else None
                    ),
                    "right_metric_probe_artifact_sha256": (
                        right_case["cache_metric_probe"]["artifact_sha256"]
                        if right_case["cache_metric_probe"] is not None
                        else None
                    ),
                },
                "optional_alignment_evidence": optional_alignment,
                "within_arm": {
                    "left": _within_arm(left_case),
                    "right": _within_arm(right_case),
                },
                "between_arms_by_repetition": between,
            }
        )
    case_geometry_scopes = {
        case["cache_geometry_evidence"]["scope"] for case in results
    }
    aggregate_geometry_scope = (
        next(iter(case_geometry_scopes))
        if len(case_geometry_scopes) == 1
        else "case-specific-see-cases"
    )
    return {
        "schema": OUTPUT_SCHEMA,
        "comparison_status": "exact-match" if exact_pairs == total_pairs else "functional-divergence-observed",
        "sources": {
            "left": {
                "run_label": left["run_label"],
                "artifact_sha256": left_sha256,
                "attribution_arm": left["attribution_arm"],
                "cache_salt": left["cache_salt"],
                "live_arm_receipt_sha256": left["live_arm_receipt_sha256"],
            },
            "right": {
                "run_label": right["run_label"],
                "artifact_sha256": right_sha256,
                "attribution_arm": right["attribution_arm"],
                "cache_salt": right["cache_salt"],
                "live_arm_receipt_sha256": right["live_arm_receipt_sha256"],
            },
        },
        "alignment": {
            "input_schemas": {"left": left["schema"], "right": right["schema"]},
            "current_format_alignment_complete": left["current_format"],
            "cache_geometry_evidence_scope": aggregate_geometry_scope,
            "model": left["model"],
            "config_sha256": left["config_sha256"],
            "repetitions": left["repetitions"],
            "requested_top_logprobs": left["top_logprobs"],
            "case_ids": [case["id"] for case in left["cases"]],
        },
        "distribution_evidence_scope": {
            "is_full_vocabulary_kld": False,
            "statement": "Metrics use only same-context positions inside the shared emitted-token prefix and persisted truncated OpenAI top_logprobs; later different-context or unproven positions are omitted. conditional_common_support_symmetric_kl is the symmetric KL between distributions renormalized solely over tokens present in both persisted top sets; it is not full-vocabulary KL and does not bound it.",
        },
        "legacy_evidence_scope": (
            None
            if left["current_format"]
            else {
                "cache_geometry_bound": False,
                "scope": "token-output-only",
                "statement": (
                    "Legacy correctness report v1/v2/v3 cache geometry or request "
                    "namespace is explicitly "
                    "unbound and is not used for cache-attribution claims."
                ),
            }
        ),
        "summary": {
            "between_arm_repetition_pair_count": total_pairs,
            "exact_match_pair_count": exact_pairs,
            "divergent_pair_count": total_pairs - exact_pairs,
        },
        "cases": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="left correctness report")
    parser.add_argument("--right", required=True, help="right correctness report")
    parser.add_argument("--output", help="exclusive-create sanitized comparison JSON")
    parser.add_argument(
        "--allow-legacy-v1",
        action="store_true",
        help=(
            "explicitly permit legacy v1/v2/v3 reports as token-output-only evidence; "
            "cache geometry remains unbound"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        output = Path(args.output) if args.output else None
        if output is not None and output.exists():
            raise ComparisonError(f"--output already exists: {output}")
        left, left_sha = _load_json(Path(args.left))
        right, right_sha = _load_json(Path(args.right))
        comparison = compare_reports(
            left,
            right,
            left_sha256=left_sha,
            right_sha256=right_sha,
            allow_legacy_v1=args.allow_legacy_v1,
        )
        rendered = json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if output is not None:
            try:
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(rendered)
            except FileExistsError as exc:
                raise ComparisonError(f"--output already exists: {output}") from exc
            except OSError as exc:
                raise ComparisonError(f"cannot write --output {output}: {exc}") from exc
        print(rendered, end="")
        return EXIT_OK
    except ComparisonError as exc:
        print(f"exl3-attribution-compare: CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
