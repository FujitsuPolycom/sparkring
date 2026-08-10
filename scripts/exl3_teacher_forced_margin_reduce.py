#!/usr/bin/env python3
"""Reduce a private EXL3 teacher-forced report to a public-safe receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from acceptance_gate import canonical_json
import exl3_attribution_cache_contract as cache_contract
import exl3_teacher_forced_margin_probe as probe


OUTPUT_SCHEMA = "sparkring-exl3-teacher-forced-margin-public-receipt/v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
CASE_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
EXPECTED_TOP_KEYS = {
    "autoregressive_discovery",
    "case_id",
    "config_sha256",
    "contacted_base_url",
    "diagnostic_classification",
    "evidence_policy",
    "forced_context_token_ids",
    "forced_context_token_ids_sha256",
    "lane",
    "limitations",
    "maturity",
    "model",
    "prompt_token_count",
    "prompt_token_ids",
    "prompt_token_ids_sha256",
    "request_contract",
    "runtime_identity",
    "schema",
    "status",
    "teacher_forced_positions",
}
POSITION_KEYS = {
    "absolute_prompt_index",
    "context_token_ids_sha256",
    "forced_token_id",
    "generated_index",
    "observations",
    "summary",
}
OBSERVATION_KEYS = {
    "candidate_pair_margin",
    "forced_token_id",
    "forced_token_rank",
    "forced_token_value",
    "repetition",
    "returned_probability_mass",
    "returned_topk_sha256",
    "returned_values",
    "top1_token_id",
    "top1_top2_margin",
    "top1_value",
    "top2_token_id",
    "top2_value",
}
SUMMARY_KEYS = {
    "candidate_margin_observation_count",
    "candidate_margin_sign_changes",
    "candidate_token_ids",
    "classification",
    "distinct_top1_count",
    "distribution_metric_scope",
    "max_abs_common_token_value_delta",
    "maximum_conditional_common_support_symmetric_kl",
    "minimum_pairwise_topk_jaccard",
    "top1_token_ids",
    "top1_top2_margin",
}
CLASSIFICATIONS = {
    "same-context-forward-ranking-nondeterminism",
    "same-context-top1-nondeterminism",
    "teacher-forced-top1-and-returned-values-stable",
    "teacher-forced-top1-stable-returned-values-vary",
}


class ReduceError(ValueError):
    pass


def _duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReduceError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise ReduceError(f"non-finite JSON number {value!r} is unsupported")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReduceError(message)


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    _require(set(value) == expected, f"{label} keys do not match the v1 contract")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str) -> float:
    _require(type(value) in {int, float}, f"{label} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ReduceError(f"{label} cannot be represented as a finite float") from exc
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _sha256(value: Any, label: str) -> str:
    _require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None, f"{label} must be lowercase SHA-256")
    return value


def _token_ids(value: Any, label: str) -> list[int]:
    _require(isinstance(value, list) and value, f"{label} must be a non-empty array")
    _require(all(type(item) is int and item >= 0 for item in value), f"{label} must contain token IDs")
    return value


def _hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validate_hash(value: Any, expected: Any, label: str) -> None:
    _require(_sha256(value, label) == _hash_json(expected), f"{label} does not bind its value")


def _validate_request_contract(value: Any) -> dict[str, Any]:
    expected_keys = {
        "discovery_max_tokens",
        "discovery_repetitions",
        "focus_generated_index",
        "logprobs_mode",
        "logprobs_mode_provenance",
        "prompt_logprobs",
        "server_max_logprobs",
        "teacher_forced_repetitions",
        "window_after",
        "window_before",
    }
    value = _exact_keys(value, expected_keys, "request_contract")
    for key in expected_keys - {"focus_generated_index", "logprobs_mode", "logprobs_mode_provenance"}:
        _integer(value[key], f"request_contract.{key}", minimum=1 if "window" not in key else 0)
    focus = value["focus_generated_index"]
    _require(focus is None or (type(focus) is int and focus >= 0), "request_contract.focus_generated_index is invalid")
    _require(value["logprobs_mode"] == "raw_logprobs", "only raw_logprobs reports are supported")
    _require(
        value["logprobs_mode_provenance"]
        == "pinned-vllm-668275901b55230f4a70841a9aac1c0be22ef8d3-default",
        "unexpected raw-logprob provenance",
    )
    _require(value["prompt_logprobs"] == value["server_max_logprobs"], "prompt/server logprob widths differ")
    return value


def _validate_runtime(value: Any) -> dict[str, Any]:
    value = _exact_keys(
        value,
        {
            "attribution_arm",
            "live_arm_receipt",
            "live_arm_revalidation",
            "lmcache_attached",
            "mtp_tokens",
            "native_prefix_caching_enabled",
            "runtime_source_pins",
        },
        "runtime_identity",
    )
    _require(value["attribution_arm"] == probe.REQUIRED_ARM, "unexpected attribution arm")
    _require(value["mtp_tokens"] == 0, "teacher-forced public receipt requires MTP0")
    _require(value["native_prefix_caching_enabled"] is False, "teacher-forced public receipt requires APC off")
    _require(value["lmcache_attached"] is False, "teacher-forced public receipt requires LMCache detached")

    live = _exact_keys(
        value["live_arm_receipt"],
        {
            "schema",
            "status",
            "arm",
            "canonical_profile_id",
            "diagnostic_profile_id",
            "canonical_profile_file_sha256",
            "image_id",
            "model_repository",
            "model_revision",
            "layout",
            "cache_salt",
            "ranks",
            "attestation_scope",
            "artifact_sha256",
        },
        "live_arm_receipt",
    )
    _require(live.get("schema") == "sparkring-exl3-live-arm-receipt/v2", "unexpected live-arm receipt schema")
    _require(live.get("status") == "live-arm-attested", "live arm was not attested")
    _require(live.get("arm") == probe.REQUIRED_ARM, "live-arm receipt arm mismatch")
    _sha256(live.get("artifact_sha256"), "live-arm artifact_sha256")
    embedded_receipt = {
        key: item for key, item in live.items() if key != "artifact_sha256"
    }
    embedded_receipt_bytes = (
        json.dumps(embedded_receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _require(
        hashlib.sha256(embedded_receipt_bytes).hexdigest()
        == live["artifact_sha256"],
        "embedded live-arm receipt does not match its artifact SHA-256",
    )
    _sha256(live.get("canonical_profile_file_sha256"), "live-arm canonical profile SHA-256")
    _require(
        live.get("canonical_profile_id") == "glm52-exl3-tr3-3.25bpw-lmcache-cs512",
        "unexpected canonical profile ID",
    )
    _require(
        live.get("diagnostic_profile_id")
        == f"{live['canonical_profile_id']}-diag-{probe.REQUIRED_ARM}",
        "unexpected diagnostic profile ID",
    )
    image_id = live.get("image_id")
    _require(isinstance(image_id, str) and image_id.startswith("sha256:") and SHA256_RE.fullmatch(image_id[7:]) is not None, "live-arm image_id is invalid")
    _require(live.get("model_repository") == "willfalco/GLM-5.2-EXL3-TR3-3.25bpw", "unexpected model repository")
    model_revision = live.get("model_revision")
    _require(isinstance(model_revision, str) and COMMIT_RE.fullmatch(model_revision) is not None, "model revision is invalid")
    layout = live.get("layout")
    _require(
        layout == cache_contract.expected_layout(probe.REQUIRED_ARM),
        "live-arm layout does not match arm A",
    )
    _require(
        live.get("cache_salt") == cache_contract.cache_salt_for_arm(probe.REQUIRED_ARM),
        "live-arm cache salt does not match arm A",
    )
    ranks = live.get("ranks")
    _require(isinstance(ranks, list) and len(ranks) == 4, "live-arm receipt must attest four ranks")
    rank_keys = {
        "rank",
        "status",
        "container_name",
        "labels",
        "image_id",
        "container_id",
        "started_at",
        "explicit_environment_sha256",
        "config_cmd_sha256",
    }
    suffix = f"-diag-{probe.REQUIRED_ARM}-r0"
    first_name = ranks[0].get("container_name") if isinstance(ranks[0], dict) else None
    _require(isinstance(first_name, str) and first_name.endswith(suffix), "rank-0 diagnostic container name is invalid")
    canonical_container_name = first_name[: -len(suffix)]
    _require(bool(canonical_container_name), "canonical container name is empty")
    expected_labels = {
        "org.sparkring.managed": "true",
        "org.sparkring.exl3-profile": live["diagnostic_profile_id"],
        "org.sparkring.component": "engine",
        "org.sparkring.exl3-attribution": probe.REQUIRED_ARM,
    }
    for rank, item in enumerate(ranks):
        item = _exact_keys(item, rank_keys, f"live_arm_receipt.ranks[{rank}]")
        _require(item["rank"] == rank and item["status"] == "attested", "live-arm receipt ranks are not ordered attestations")
        _require(
            item["container_name"]
            == f"{canonical_container_name}-diag-{probe.REQUIRED_ARM}-r{rank}",
            "live-arm receipt container name mismatch",
        )
        _require(item["labels"] == expected_labels, "live-arm receipt labels mismatch")
        _require(item["image_id"] == image_id, "live-arm receipt rank image mismatch")
        _require(
            isinstance(item["container_id"], str)
            and cache_contract.CONTAINER_ID_RE.fullmatch(item["container_id"]) is not None,
            "live-arm receipt container ID is invalid",
        )
        _require(
            isinstance(item["started_at"], str)
            and cache_contract.DOCKER_STARTED_AT_RE.fullmatch(item["started_at"]) is not None,
            "live-arm receipt StartedAt is invalid",
        )
        _sha256(item["explicit_environment_sha256"], "live-arm environment SHA-256")
        _sha256(item["config_cmd_sha256"], "live-arm Config.Cmd SHA-256")
    expected_scope = cache_contract.build_live_arm_receipt(
        arm_id=probe.REQUIRED_ARM,
        canonical_profile_id=live["canonical_profile_id"],
        canonical_profile_file_sha256=live["canonical_profile_file_sha256"],
        image_id=image_id,
        model_repository=live["model_repository"],
        model_revision=model_revision,
        canonical_container_name=canonical_container_name,
        explicit_environment_sha256=[item["explicit_environment_sha256"] for item in ranks],
        config_cmd_sha256=[item["config_cmd_sha256"] for item in ranks],
        observed_runtime_instances=[
            {"container_id": item["container_id"], "started_at": item["started_at"]}
            for item in ranks
        ],
    )["attestation_scope"]
    _require(live["attestation_scope"] == expected_scope, "live-arm attestation scope mismatch")

    revalidation = _exact_keys(
        value["live_arm_revalidation"],
        {"status", "rank_count", "runtime_instances"},
        "live_arm_revalidation",
    )
    _require(revalidation.get("status") == "live-arm-re-attested" and revalidation.get("rank_count") == 4, "live arm was not re-attested on four ranks")
    instances = revalidation.get("runtime_instances")
    _require(isinstance(instances, list) and len(instances) == 4, "live-arm runtime instance count is invalid")
    for rank, instance in enumerate(instances):
        instance = _exact_keys(instance, {"container_id", "started_at"}, f"live_arm_revalidation.runtime_instances[{rank}]")
        _require(
            instance
            == {
                "container_id": ranks[rank]["container_id"],
                "started_at": ranks[rank]["started_at"],
            },
            "live-arm revalidation runtime identity mismatch",
        )

    pins = _exact_keys(
        value["runtime_source_pins"],
        {"evidence_scope", "exllamav3_commit", "model_repository", "model_revision", "pins_file_sha256", "sparkinfer_commit", "vllm_commit"},
        "runtime_source_pins",
    )
    _require(pins["evidence_scope"] == "declared-canonical-pins-bound-to-receipt-model-not-live-binary-introspection", "unexpected source-pin evidence scope")
    _require(pins["model_repository"] == live["model_repository"] and pins["model_revision"] == model_revision, "source pins disagree with live receipt")
    for key in ("exllamav3_commit", "sparkinfer_commit", "vllm_commit"):
        _require(isinstance(pins[key], str) and COMMIT_RE.fullmatch(pins[key]) is not None, f"{key} is invalid")
    _require(
        pins["vllm_commit"] == probe.PINNED_API_VLLM_COMMIT,
        "raw-logprob provenance does not match the runtime vLLM pin",
    )
    _sha256(pins["pins_file_sha256"], "pins_file_sha256")
    try:
        expected_pins = probe._runtime_source_pins(live)
    except (OSError, ValueError, probe.ConfigError) as exc:
        raise ReduceError(f"cannot bind source pins to the public pins file: {exc}") from exc
    _require(
        pins == expected_pins,
        "runtime source pins do not match the hash-bound public pins file",
    )
    return value


def _validate_position(
    value: Any,
    *,
    prompt_ids: list[int],
    reference_ids: list[int],
    candidates: list[int],
    repetitions: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _exact_keys(value, POSITION_KEYS, "teacher_forced_position")
    generated_index = _integer(value["generated_index"], "generated_index")
    _require(generated_index < len(reference_ids), "generated_index exceeds the reference completion")
    _require(value["absolute_prompt_index"] == len(prompt_ids) + generated_index, "absolute_prompt_index is inconsistent")
    _require(value["forced_token_id"] == reference_ids[generated_index], "forced token differs from reference completion")
    _validate_hash(
        value["context_token_ids_sha256"],
        [*prompt_ids, *reference_ids[:generated_index]],
        "context_token_ids_sha256",
    )
    observations = value["observations"]
    _require(isinstance(observations, list) and len(observations) == repetitions, "teacher-forced repetition count is inconsistent")
    for expected_repetition, observation in enumerate(observations, 1):
        observation = _exact_keys(observation, OBSERVATION_KEYS, "teacher_forced_observation")
        _require(observation["repetition"] == expected_repetition, "teacher-forced repetition sequence is invalid")
        _require(observation["forced_token_id"] == value["forced_token_id"], "observation forced token mismatch")
        for key in ("forced_token_id", "forced_token_rank", "top1_token_id", "top2_token_id"):
            _integer(observation[key], f"observation.{key}")
        for key in ("forced_token_value", "returned_probability_mass", "top1_top2_margin", "top1_value", "top2_value"):
            _number(observation[key], f"observation.{key}")
        if observation["candidate_pair_margin"] is not None:
            _number(observation["candidate_pair_margin"], "observation.candidate_pair_margin")
        returned = observation["returned_values"]
        _require(isinstance(returned, dict) and returned, "returned_values must be a non-empty object")
        parsed_returned: dict[int, dict[str, Any]] = {}
        ranks: list[int] = []
        for token_id, record in returned.items():
            _require(
                isinstance(token_id, str)
                and token_id.isdigit()
                and str(int(token_id)) == token_id,
                "returned_values token keys must be canonical nonnegative integers",
            )
            record = _exact_keys(record, {"rank", "value"}, "returned_values record")
            rank = _integer(record["rank"], "returned_values rank", minimum=1)
            value_number = _number(record["value"], "returned_values value")
            _require(
                value_number <= 1e-6,
                "returned_values contains a positive value incompatible with raw logprobs",
            )
            parsed_returned[int(token_id)] = {"rank": rank, "value": value_number}
            ranks.append(rank)
        _require(sorted(ranks) == list(range(1, len(ranks) + 1)), "returned_values ranks must be unique and contiguous")
        _require(len(parsed_returned) >= 2, "returned_values must contain at least rank 1 and rank 2")
        by_rank = {record["rank"]: (token_id, record["value"]) for token_id, record in parsed_returned.items()}
        top1_id, top1_value = by_rank[1]
        top2_id, top2_value = by_rank[2]
        forced = parsed_returned.get(observation["forced_token_id"])
        _require(forced is not None, "forced token is absent from returned_values")
        _require(
            observation["forced_token_rank"] == forced["rank"]
            and observation["forced_token_value"] == forced["value"],
            "forced-token rank/value is not bound to returned_values",
        )
        _require(
            observation["top1_token_id"] == top1_id
            and observation["top1_value"] == top1_value
            and observation["top2_token_id"] == top2_id
            and observation["top2_value"] == top2_value,
            "top-1/top-2 fields are not bound to returned_values ranks",
        )
        _require(
            observation["top1_top2_margin"] == top1_value - top2_value,
            "top-1/top-2 margin does not recompute",
        )
        expected_candidate_margin = None
        if len(candidates) == 2 and all(token_id in parsed_returned for token_id in candidates):
            expected_candidate_margin = (
                parsed_returned[candidates[0]]["value"]
                - parsed_returned[candidates[1]]["value"]
            )
        _require(
            observation["candidate_pair_margin"] == expected_candidate_margin,
            "candidate-pair margin does not recompute",
        )
        expected_mass = sum(math.exp(record["value"]) for record in parsed_returned.values())
        _require(
            math.isclose(
                observation["returned_probability_mass"],
                expected_mass,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ),
            "returned probability mass does not recompute",
        )
        _require(
            observation["returned_probability_mass"] <= 1.00001,
            "returned probability mass violates the raw-logprob contract",
        )
        _validate_hash(observation["returned_topk_sha256"], parsed_returned, "returned_topk_sha256")

    position_candidates = candidates if candidates else []
    # Candidate margins were computed only for the discovery focus.  Their
    # presence is an unambiguous, non-secret marker for that one position.
    if not any(item["candidate_pair_margin"] is not None for item in observations):
        position_candidates = []
    recomputed = probe._position_summary(observations, position_candidates)
    supplied = _exact_keys(value["summary"], SUMMARY_KEYS, "teacher_forced_position.summary")
    _require(canonical_json(recomputed) == canonical_json(supplied), "teacher-forced summary does not recompute from observations")
    _require(supplied["classification"] in CLASSIFICATIONS, "unknown position classification")
    top1_counts = sorted(Counter(item["top1_token_id"] for item in observations).values(), reverse=True)
    public = {
        "generated_index": generated_index,
        "observation_count": len(observations),
        "classification": supplied["classification"],
        "distinct_top1_count": supplied["distinct_top1_count"],
        "top1_multiplicities": top1_counts,
        "top1_top2_margin": supplied["top1_top2_margin"],
        "candidate_margin_observation_count": supplied["candidate_margin_observation_count"],
        "candidate_margin_sign_changes": supplied["candidate_margin_sign_changes"],
        "minimum_pairwise_topk_jaccard": supplied["minimum_pairwise_topk_jaccard"],
        "max_abs_common_token_value_delta": supplied["max_abs_common_token_value_delta"],
        "maximum_conditional_common_support_symmetric_kl": supplied["maximum_conditional_common_support_symmetric_kl"],
        "distribution_metric_scope": supplied["distribution_metric_scope"],
    }
    return value, public


def reduce_document(document: Any, raw_sha256: str) -> dict[str, Any]:
    document = _exact_keys(document, EXPECTED_TOP_KEYS, "raw report")
    _require(document["schema"] == probe.REPORT_SCHEMA, "unexpected raw report schema")
    _require(document["status"] == "completed", "raw report is not completed")
    _require(document["lane"] == "public-functional", "unexpected evidence lane")
    _require(document["maturity"] == "diagnostic-observation-not-acceptance", "unexpected evidence maturity")
    _require(document["model"] == "glm-5.2-exl3-tr3-3.25bpw", "unexpected model")
    _require(isinstance(document["case_id"], str) and CASE_RE.fullmatch(document["case_id"]) is not None, "case_id is not publishable")
    _sha256(document["config_sha256"], "config_sha256")
    _sha256(raw_sha256, "raw report SHA-256")

    prompt_ids = _token_ids(document["prompt_token_ids"], "prompt_token_ids")
    _require(document["prompt_token_count"] == len(prompt_ids), "prompt_token_count is inconsistent")
    _validate_hash(document["prompt_token_ids_sha256"], prompt_ids, "prompt_token_ids_sha256")
    forced_context = _token_ids(document["forced_context_token_ids"], "forced_context_token_ids")
    _validate_hash(document["forced_context_token_ids_sha256"], forced_context, "forced_context_token_ids_sha256")
    request = _validate_request_contract(document["request_contract"])
    runtime = _validate_runtime(document["runtime_identity"])

    discovery = _exact_keys(
        document["autoregressive_discovery"],
        {"candidate_token_ids_at_focus", "earliest_divergence_index", "focus_generated_index", "observations", "reference_token_ids", "reference_token_ids_sha256"},
        "autoregressive_discovery",
    )
    reference_ids = _token_ids(discovery["reference_token_ids"], "reference_token_ids")
    _validate_hash(discovery["reference_token_ids_sha256"], reference_ids, "reference_token_ids_sha256")
    earliest = _integer(discovery["earliest_divergence_index"], "earliest_divergence_index")
    focus = _integer(discovery["focus_generated_index"], "focus_generated_index")
    candidates = _token_ids(discovery["candidate_token_ids_at_focus"], "candidate_token_ids_at_focus")
    _require(len(set(candidates)) == len(candidates) >= 2, "focus candidate set is invalid")
    discovery_observations = discovery["observations"]
    _require(isinstance(discovery_observations, list) and len(discovery_observations) == request["discovery_repetitions"], "discovery repetition count is inconsistent")
    sequence_hashes: list[str] = []
    sequences: list[list[int]] = []
    for expected_repetition, observation in enumerate(discovery_observations, 1):
        observation = _exact_keys(observation, {"repetition", "token_ids", "token_ids_sha256"}, "discovery observation")
        _require(observation["repetition"] == expected_repetition, "discovery repetition sequence is invalid")
        token_ids = _token_ids(observation["token_ids"], "discovery token_ids")
        _validate_hash(observation["token_ids_sha256"], token_ids, "discovery token_ids_sha256")
        sequences.append(token_ids)
        sequence_hashes.append(observation["token_ids_sha256"])
    _require(probe._first_divergence(sequences) == earliest, "earliest divergence does not recompute")
    _require(reference_ids == min(sequences), "reference completion is not the producer-selected discovery sequence")
    requested_focus = request["focus_generated_index"]
    _require(
        focus == (earliest if requested_focus is None else requested_focus),
        "focus generated index does not match the request contract",
    )
    _require(focus < len(reference_ids), "focus exceeds the reference completion")
    _require(all(focus < len(sequence) for sequence in sequences), "a discovery sequence does not reach the focus index")
    _require(
        candidates == sorted({sequence[focus] for sequence in sequences}),
        "focus candidate tokens do not match the discovery observations",
    )

    raw_positions = document["teacher_forced_positions"]
    _require(isinstance(raw_positions, list) and raw_positions, "teacher_forced_positions must be non-empty")
    validated_positions: list[dict[str, Any]] = []
    public_positions: list[dict[str, Any]] = []
    for position in raw_positions:
        validated, public = _validate_position(
            position,
            prompt_ids=prompt_ids,
            reference_ids=reference_ids,
            candidates=candidates if position.get("generated_index") == focus else [],
            repetitions=request["teacher_forced_repetitions"],
        )
        validated_positions.append(validated)
        public_positions.append(public)
    indices = [item["generated_index"] for item in validated_positions]
    _require(indices == sorted(set(indices)), "teacher-forced positions must be unique and sorted")
    expected_start = max(0, focus - request["window_before"])
    expected_end = min(len(reference_ids) - 1, focus + request["window_after"])
    _require(
        indices == list(range(expected_start, expected_end + 1)),
        "teacher-forced positions do not cover the requested focus window",
    )
    _require(
        forced_context == [*prompt_ids, *reference_ids[: max(indices) + 1]],
        "forced context differs from the prompt plus probed reference prefix",
    )
    corrected = probe._diagnostic_classification(validated_positions, earliest)
    raw_classification = document["diagnostic_classification"]
    if raw_classification != corrected:
        _require(
            raw_classification
            == "autoregressive-divergence-without-teacher-forced-top1-flip"
            and corrected
            in {
                "cache-not-required-forward-ranking-nondeterminism-observed",
                "cache-not-required-top1-nondeterminism-observed",
            },
            "raw diagnostic classification is unrelated to the one known stale-classifier case",
        )

    policy = _exact_keys(document["evidence_policy"], {"full_vocabulary_kld_claimed", "raw_report_is_private", "reason"}, "evidence_policy")
    _require(policy["full_vocabulary_kld_claimed"] is False and policy["raw_report_is_private"] is True, "raw report evidence policy is unsafe")
    limitations = _exact_keys(document["limitations"], {"full_vocabulary_kld_available", "raw_logits_returned", "statement"}, "limitations")
    _require(limitations["full_vocabulary_kld_available"] is False and limitations["raw_logits_returned"] is False, "unsupported full-logit/KLD claim")

    live = runtime["live_arm_receipt"]
    pins = runtime["runtime_source_pins"]
    multiplicities = sorted(Counter(sequence_hashes).values(), reverse=True)
    return {
        "schema": OUTPUT_SCHEMA,
        "source_schema": probe.REPORT_SCHEMA,
        "private_raw_artifact_sha256": raw_sha256,
        "status": "completed",
        "lane": "public-functional",
        "maturity": "live-validated",
        "evidence_scope": "diagnostic observation; not acceptance",
        "hardware": "four directly cabled DGX Sparks",
        "model": document["model"],
        "case_id": document["case_id"],
        "config_sha256": document["config_sha256"],
        "runtime_arm": {
            "id": probe.REQUIRED_ARM,
            "rank_count": 4,
            "dcp_size": 4,
            "mtp_tokens": 0,
            "native_prefix_caching_enabled": False,
            "lmcache_attached": False,
            "live_arm_receipt_sha256": live["artifact_sha256"],
            "runtime_re_attested_immediately_before_requests": True,
        },
        "public_source_pins": {
            "image_id": live["image_id"],
            "model_repository": live["model_repository"],
            "model_revision": live["model_revision"],
            "vllm_commit": pins["vllm_commit"],
            "sparkinfer_commit": pins["sparkinfer_commit"],
            "exllamav3_commit": pins["exllamav3_commit"],
            "pins_file_sha256": pins["pins_file_sha256"],
            "evidence_scope": pins["evidence_scope"],
        },
        "request_contract": {
            "discovery_repetitions": request["discovery_repetitions"],
            "discovery_max_tokens": request["discovery_max_tokens"],
            "teacher_forced_repetitions_per_position": request["teacher_forced_repetitions"],
            "returned_logprob_width": request["prompt_logprobs"],
            "logprobs_mode": request["logprobs_mode"],
        },
        "autoregressive_discovery": {
            "repetitions": len(discovery_observations),
            "unique_token_sequence_count": len(multiplicities),
            "token_sequence_multiplicities": multiplicities,
            "earliest_divergence_generated_index": earliest,
            "focus_generated_index": focus,
            "focus_candidate_count": len(candidates),
        },
        "teacher_forced": {
            "positions_probed": len(public_positions),
            "generated_index_minimum": min(indices),
            "generated_index_maximum": max(indices),
            "positions": public_positions,
        },
        "diagnostic": {
            "raw_report_classification": raw_classification,
            "recomputed_classification": corrected,
            "raw_report_classifier_was_stale": raw_classification != corrected,
            "conclusion": "same-context top-1 nondeterminism was observed with MTP disabled, native APC disabled, and LMCache detached; cache reuse was not required for this observation",
            "non_conclusion": "this diagnostic does not identify the responsible kernel, graph, attention path, or collective",
        },
        "limitations": {
            "raw_logits_returned": False,
            "full_vocabulary_kld_available": False,
            "reported_symmetric_kl_scope": "truncated returned common support, renormalized; not full-vocabulary KLD",
            "teacher_forcing_removes_context_drift_only": True,
            "acceptance_claimed": False,
        },
        "privacy": {
            "raw_report_publishable": False,
            "prompt_omitted": True,
            "token_ids_omitted": True,
            "outputs_omitted": True,
            "site_and_runtime_instance_identifiers_omitted": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        raw = Path(args.input).read_bytes()
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicates,
            parse_constant=_reject_nonfinite,
        )
        receipt = reduce_document(document, hashlib.sha256(raw).hexdigest())
        rendered = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with output_path.open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(rendered)
            except FileExistsError as exc:
                raise ReduceError(f"--output already exists: {output_path}") from exc
        sys.stdout.write(rendered)
        return 0
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ReduceError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
