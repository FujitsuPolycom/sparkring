#!/usr/bin/env python3
"""Locate EXL3 divergence and replay the same token context teacher-forced."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from acceptance_gate import UrllibHttpClient, canonical_json, sha256_hex
import exl3_attribution_launcher as attribution
from exl3_attribution_cache_contract import (
    cache_salt_for_arm,
    validate_live_arm_receipt,
)
from exl3_correctness_gate import (
    ConfigError as CorrectnessConfigError,
    MODEL_RE,
    case_prompt,
    inspectable_base_url,
    load_cases,
)
from sparkring_site import SiteConfigError, load_site


PLAN_SCHEMA = "sparkring-exl3-teacher-forced-margin-plan/v1"
REPORT_SCHEMA = "sparkring-exl3-teacher-forced-margin-report/v1"
REQUIRED_ARM = "a-mtp0-apc0-lmcache0"
PINNED_API_VLLM_COMMIT = "668275901b55230f4a70841a9aac1c0be22ef8d3"
EXIT_OK = 0
EXIT_REQUEST_FAILURE = 2
EXIT_CONFIG_ERROR = 3


class ConfigError(ValueError):
    pass


class RequestFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class StrictJsonFailure:
    reason: str


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number {value!r}")


class StrictUrllibHttpClient(UrllibHttpClient):
    """Real HTTP client that preserves rejection evidence for unsafe JSON."""

    def _send(self, request: Any, timeout: float) -> tuple[int, Any]:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", "replace")
                status = int(getattr(response, "status", 0) or 0)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace") if exc.fp else ""
            status = int(exc.code)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return 0, str(exc)
        try:
            return status, json.loads(
                raw,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            return status, StrictJsonFailure(str(exc))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _token_ids(value: Any, label: str, *, nonempty: bool = True) -> list[int]:
    if (
        not isinstance(value, list)
        or (nonempty and not value)
        or any(not _is_int(token_id) or token_id < 0 for token_id in value)
    ):
        raise RequestFailure(f"{label} must be a list of nonnegative token IDs")
    return list(value)


def _one_choice(body: Any, label: str) -> dict[str, Any]:
    choices = body.get("choices") if isinstance(body, dict) else None
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
    ):
        raise RequestFailure(f"{label} did not return exactly one choice")
    return choices[0]


def _first_divergence(sequences: list[list[int]]) -> int | None:
    for index in range(max(map(len, sequences))):
        values = {
            sequence[index] if index < len(sequence) else None
            for sequence in sequences
        }
        if len(values) > 1:
            return index
    return None


def _parse_position(value: Any, label: str) -> dict[int, dict[str, Any]]:
    if not isinstance(value, dict) or not value:
        raise RequestFailure(f"{label} must be a non-empty token-logprob mapping")
    result: dict[int, dict[str, Any]] = {}
    seen_ranks: set[int] = set()
    for raw_token_id, raw_record in value.items():
        try:
            token_id = int(raw_token_id)
        except (TypeError, ValueError) as exc:
            raise RequestFailure(f"{label} contains a nonnumeric token ID") from exc
        if str(token_id) != str(raw_token_id) or token_id < 0 or token_id in result:
            raise RequestFailure(f"{label} contains an invalid or duplicate token ID")
        if not isinstance(raw_record, dict):
            raise RequestFailure(f"{label}[{token_id}] must be an object")
        logprob = raw_record.get("logprob")
        rank = raw_record.get("rank")
        if (
            isinstance(logprob, bool)
            or not isinstance(logprob, (int, float))
            or not math.isfinite(float(logprob))
            or not _is_int(rank)
            or rank < 1
            or rank in seen_ranks
        ):
            raise RequestFailure(f"{label}[{token_id}] has invalid value or rank")
        seen_ranks.add(rank)
        result[token_id] = {"value": float(logprob), "rank": rank}
    if 1 not in seen_ranks:
        raise RequestFailure(f"{label} does not contain rank 1")
    return result


def _has_sign_flip(values: list[float]) -> bool:
    return bool(values) and min(values) < 0 < max(values)


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _conditional_symmetric_kl(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]
) -> float | None:
    common = sorted(set(left) & set(right))
    if not common:
        return None
    left_norm = _logsumexp([left[token_id]["value"] for token_id in common])
    right_norm = _logsumexp([right[token_id]["value"] for token_id in common])
    left_kl = 0.0
    right_kl = 0.0
    for token_id in common:
        left_logprob = left[token_id]["value"] - left_norm
        right_logprob = right[token_id]["value"] - right_norm
        left_kl += math.exp(left_logprob) * (left_logprob - right_logprob)
        right_kl += math.exp(right_logprob) * (right_logprob - left_logprob)
    return (left_kl + right_kl) / 2.0


def _position_summary(
    observations: list[dict[str, Any]], candidate_token_ids: list[int]
) -> dict[str, Any]:
    top1_ids = {item["top1_token_id"] for item in observations}
    margins = [item["top1_top2_margin"] for item in observations]
    candidate_margins = [
        item["candidate_pair_margin"]
        for item in observations
        if item["candidate_pair_margin"] is not None
    ]
    jaccards: list[float] = []
    value_deltas: list[float] = []
    symmetric_kls: list[float] = []
    for left, right in itertools.combinations(observations, 2):
        left_values = left["returned_values"]
        right_values = right["returned_values"]
        left_tokens = set(left_values)
        right_tokens = set(right_values)
        union = left_tokens | right_tokens
        common = left_tokens & right_tokens
        jaccards.append(len(common) / len(union))
        value_deltas.extend(
            abs(left_values[token_id]["value"] - right_values[token_id]["value"])
            for token_id in common
        )
        symmetric_kl = _conditional_symmetric_kl(left_values, right_values)
        if symmetric_kl is not None:
            symmetric_kls.append(symmetric_kl)
    max_value_delta = max(value_deltas) if value_deltas else None
    candidate_margin_sign_changes = _has_sign_flip(candidate_margins)
    if len(top1_ids) > 1 and candidate_margin_sign_changes:
        classification = "same-context-forward-ranking-nondeterminism"
    elif len(top1_ids) > 1:
        classification = "same-context-top1-nondeterminism"
    elif max_value_delta == 0:
        classification = "teacher-forced-top1-and-returned-values-stable"
    else:
        classification = "teacher-forced-top1-stable-returned-values-vary"
    return {
        "distinct_top1_count": len(top1_ids),
        "top1_token_ids": sorted(top1_ids),
        "top1_top2_margin": {
            "minimum": min(margins),
            "median": statistics.median(margins),
            "maximum": max(margins),
        },
        "candidate_token_ids": candidate_token_ids,
        "candidate_margin_observation_count": len(candidate_margins),
        "candidate_margin_sign_changes": candidate_margin_sign_changes,
        "minimum_pairwise_topk_jaccard": min(jaccards) if jaccards else None,
        "max_abs_common_token_value_delta": max_value_delta,
        "maximum_conditional_common_support_symmetric_kl": (
            max(symmetric_kls) if symmetric_kls else None
        ),
        "distribution_metric_scope": (
            "truncated-returned-common-support-renormalized-not-full-vocabulary-kld"
        ),
        "classification": classification,
    }


def _diagnostic_classification(
    positions: list[dict[str, Any]], observed_divergence: int | None
) -> str:
    classifications = {
        position["summary"]["classification"] for position in positions
    }
    if "same-context-forward-ranking-nondeterminism" in classifications:
        return "cache-not-required-forward-ranking-nondeterminism-observed"
    if "same-context-top1-nondeterminism" in classifications:
        return "cache-not-required-top1-nondeterminism-observed"
    if observed_divergence is not None:
        return "autoregressive-divergence-without-teacher-forced-top1-flip"
    return "divergence-not-reproduced-at-probed-context"


def run_probe(
    http: Any,
    *,
    base_url: str,
    model: str,
    prompt: str,
    case_id: str,
    seed: int,
    discovery_repetitions: int,
    discovery_max_tokens: int,
    focus_generated_index: int | None,
    window_before: int,
    window_after: int,
    teacher_forced_repetitions: int,
    top_logprobs: int,
    timeout: float,
    cache_salt: str,
    ignore_eos: bool = True,
) -> dict[str, Any]:
    token_status, token_body = http.post_json(
        f"{base_url}/tokenize",
        {"model": model, "prompt": prompt, "add_special_tokens": True},
        timeout=timeout,
    )
    if token_status != 200:
        raise RequestFailure(f"case {case_id} /tokenize returned HTTP {token_status}")
    prompt_token_ids = _token_ids(
        token_body.get("tokens") if isinstance(token_body, dict) else None,
        f"case {case_id} prompt tokens",
    )

    discovery: list[dict[str, Any]] = []
    for repetition in range(1, discovery_repetitions + 1):
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": seed,
            "max_tokens": discovery_max_tokens,
            "stream": False,
            "cache_salt": cache_salt,
            "logprobs": top_logprobs,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
        }
        if ignore_eos:
            payload["ignore_eos"] = True
        status, body = http.post_json(
            f"{base_url}/v1/completions", payload, timeout=timeout
        )
        if status != 200:
            raise RequestFailure(
                f"case {case_id} discovery repetition {repetition} returned HTTP {status}"
            )
        choice = _one_choice(body, f"case {case_id} discovery repetition {repetition}")
        token_ids = _token_ids(
            choice.get("token_ids"),
            f"case {case_id} discovery repetition {repetition} token_ids",
        )
        if not isinstance(choice.get("logprobs"), dict):
            raise RequestFailure(
                f"case {case_id} discovery repetition {repetition} omitted logprobs"
            )
        discovery.append(
            {
                "repetition": repetition,
                "token_ids": token_ids,
                "token_ids_sha256": sha256_hex(
                    canonical_json(token_ids).encode("utf-8")
                ),
            }
        )

    sequences = [item["token_ids"] for item in discovery]
    observed_divergence = _first_divergence(sequences)
    focus = observed_divergence if focus_generated_index is None else focus_generated_index
    if focus is None:
        raise RequestFailure(
            "discovery repetitions did not diverge; supply --focus-generated-index "
            "to probe a previously established position"
        )
    reference_ids = min(sequences)
    start = max(0, focus - window_before)
    end = focus + window_after
    if end >= len(reference_ids):
        raise RequestFailure(
            f"teacher-forced window ends at {end}, beyond reference length {len(reference_ids)}"
        )
    candidate_ids = sorted(
        {
            sequence[focus]
            for sequence in sequences
            if focus < len(sequence)
        }
    )
    combined_prompt = [*prompt_token_ids, *reference_ids[: end + 1]]
    forced_context_sha256 = sha256_hex(
        canonical_json(combined_prompt).encode("utf-8")
    )
    position_observations: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(start, end + 1)
    }
    for repetition in range(1, teacher_forced_repetitions + 1):
        payload = {
            "model": model,
            "prompt": combined_prompt,
            "add_special_tokens": False,
            "echo": True,
            "max_tokens": 0,
            "prompt_logprobs": top_logprobs,
            "return_token_ids": True,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": seed,
            "stream": False,
            "cache_salt": cache_salt,
        }
        status, body = http.post_json(
            f"{base_url}/v1/completions", payload, timeout=timeout
        )
        if status != 200:
            raise RequestFailure(
                f"case {case_id} teacher-forced repetition {repetition} returned HTTP {status}"
            )
        choice = _one_choice(
            body, f"case {case_id} teacher-forced repetition {repetition}"
        )
        returned_ids = _token_ids(
            choice.get("prompt_token_ids"),
            f"case {case_id} teacher-forced prompt_token_ids",
        )
        if returned_ids != combined_prompt:
            raise RequestFailure("teacher-forced prompt_token_ids do not match request")
        prompt_logprobs = choice.get("prompt_logprobs")
        if not isinstance(prompt_logprobs, list) or len(prompt_logprobs) != len(
            combined_prompt
        ):
            raise RequestFailure("teacher-forced prompt_logprobs length is invalid")
        if prompt_logprobs[0] is not None:
            raise RequestFailure("teacher-forced first prompt_logprobs entry must be null")
        for generated_index in range(start, end + 1):
            absolute_index = len(prompt_token_ids) + generated_index
            values = _parse_position(
                prompt_logprobs[absolute_index],
                f"teacher-forced repetition {repetition} position {absolute_index}",
            )
            if any(record["value"] > 1e-6 for record in values.values()):
                raise RequestFailure(
                    f"teacher-forced position {absolute_index} violates raw-logprob contract"
                )
            forced_token_id = reference_ids[generated_index]
            if forced_token_id not in values:
                raise RequestFailure(
                    f"teacher-forced target token {forced_token_id} is absent at position {absolute_index}"
                )
            ranked = sorted(values.items(), key=lambda item: item[1]["rank"])
            if len(ranked) < 2 or ranked[0][1]["rank"] != 1 or ranked[1][1]["rank"] != 2:
                raise RequestFailure(
                    f"teacher-forced position {absolute_index} lacks ranks 1 and 2"
                )
            top1_id, top1 = ranked[0]
            top2_id, top2 = ranked[1]
            relevant_candidates = candidate_ids if generated_index == focus else []
            candidate_margin = None
            if len(relevant_candidates) == 2 and all(
                token_id in values for token_id in relevant_candidates
            ):
                candidate_margin = (
                    values[relevant_candidates[0]]["value"]
                    - values[relevant_candidates[1]]["value"]
                )
            returned_mass = sum(
                math.exp(record["value"]) for record in values.values()
            )
            if returned_mass > 1.00001:
                raise RequestFailure(
                    f"teacher-forced position {absolute_index} violates raw-logprob mass contract"
                )
            position_observations[generated_index].append(
                {
                    "repetition": repetition,
                    "forced_token_id": forced_token_id,
                    "forced_token_rank": values[forced_token_id]["rank"],
                    "forced_token_value": values[forced_token_id]["value"],
                    "top1_token_id": top1_id,
                    "top1_value": top1["value"],
                    "top2_token_id": top2_id,
                    "top2_value": top2["value"],
                    "top1_top2_margin": top1["value"] - top2["value"],
                    "candidate_pair_margin": candidate_margin,
                    "returned_probability_mass": returned_mass,
                    "returned_values": {
                        str(token_id): record
                        for token_id, record in sorted(values.items())
                    },
                    "returned_topk_sha256": sha256_hex(
                        canonical_json(values).encode("utf-8")
                    ),
                }
            )

    positions = []
    for generated_index, observations in position_observations.items():
        position_candidates = candidate_ids if generated_index == focus else []
        positions.append(
            {
                "generated_index": generated_index,
                "absolute_prompt_index": len(prompt_token_ids) + generated_index,
                "context_token_ids_sha256": sha256_hex(
                    canonical_json(
                        [*prompt_token_ids, *reference_ids[:generated_index]]
                    ).encode("utf-8")
                ),
                "forced_token_id": reference_ids[generated_index],
                "observations": observations,
                "summary": _position_summary(observations, position_candidates),
            }
        )
    diagnostic_classification = _diagnostic_classification(
        positions, observed_divergence
    )
    return {
        "case_id": case_id,
        "prompt_token_count": len(prompt_token_ids),
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_ids_sha256": sha256_hex(
            canonical_json(prompt_token_ids).encode("utf-8")
        ),
        "autoregressive_discovery": {
            "observations": discovery,
            "earliest_divergence_index": observed_divergence,
            "focus_generated_index": focus,
            "reference_token_ids": reference_ids,
            "reference_token_ids_sha256": sha256_hex(
                canonical_json(reference_ids).encode("utf-8")
            ),
            "candidate_token_ids_at_focus": candidate_ids,
        },
        "forced_context_token_ids": combined_prompt,
        "forced_context_token_ids_sha256": forced_context_sha256,
        "teacher_forced_positions": positions,
        "diagnostic_classification": diagnostic_classification,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--activation-receipt", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--attribution-arm", required=True, choices=(REQUIRED_ARM,)
    )
    parser.add_argument("--discovery-repetitions", type=int, default=20)
    parser.add_argument("--discovery-max-tokens", type=int)
    parser.add_argument("--focus-generated-index", type=int)
    parser.add_argument("--window-before", type=int, default=8)
    parser.add_argument("--window-after", type=int, default=8)
    parser.add_argument("--teacher-forced-repetitions", type=int, default=20)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--output")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("command", choices=("plan", "run"))
    return parser


def _selected_case(config_path: Path, case_id: str) -> tuple[dict[str, Any], str]:
    cases, config_sha256 = load_cases(config_path)
    matches = [case for case in cases if case["id"] == case_id]
    if len(matches) != 1:
        raise ConfigError(f"--case-id {case_id!r} is not present exactly once")
    return matches[0], config_sha256


def _profile_logprobs_contract(
    profile: Any, requested_top_logprobs: int
) -> dict[str, Any]:
    args = profile.extra_vllm_args
    if not isinstance(args, list) or any(not isinstance(value, str) for value in args):
        raise ConfigError("profile extra_vllm_args must be a string list")

    def option_values(name: str) -> list[str]:
        values: list[str] = []
        for index, token in enumerate(args):
            if token == name:
                if index + 1 >= len(args):
                    raise ConfigError(f"profile {name} has no value")
                values.append(args[index + 1])
            elif token.startswith(f"{name}="):
                values.append(token.split("=", 1)[1])
        if len(values) > 1:
            raise ConfigError(f"profile repeats {name}")
        return values

    modes = option_values("--logprobs-mode")
    mode = modes[0] if modes else "raw_logprobs"
    if mode != "raw_logprobs":
        raise ConfigError(
            "teacher-forced margin evidence requires --logprobs-mode raw_logprobs"
        )
    maximum_values = option_values("--max-logprobs")
    if maximum_values:
        try:
            maximum = int(maximum_values[0])
        except ValueError as exc:
            raise ConfigError("profile --max-logprobs must be an integer") from exc
    else:
        maximum = 20
    if maximum != -1 and maximum < requested_top_logprobs:
        raise ConfigError(
            f"profile max-logprobs {maximum} is below requested {requested_top_logprobs}"
        )
    return {
        "logprobs_mode": mode,
        "logprobs_mode_provenance": (
            "explicit-profile-argument"
            if modes
            else "pinned-vllm-668275901b55230f4a70841a9aac1c0be22ef8d3-default"
        ),
        "server_max_logprobs": maximum,
    }


def _runtime_source_pins(live_arm_receipt: dict[str, Any]) -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "runtime" / "exl3" / "pins.json"
    try:
        raw = path.read_bytes()
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ConfigError(f"cannot bind canonical EXL3 pins: {exc}") from exc
    try:
        model = document["model"]
        sources = document["sources"]
        vllm_commit = sources["vllm_exl3_port"]["commit"]
        sparkinfer_commit = sources["sparkinfer"]["commit"]
        exllamav3_commit = sources["exllamav3"]["commit"]
    except (KeyError, TypeError) as exc:
        raise ConfigError("canonical EXL3 pins lack required source identities") from exc
    if document.get("schema") != "sparkring-public-exl3-pins/v1":
        raise ConfigError("canonical EXL3 pins schema is unsupported")
    if vllm_commit != PINNED_API_VLLM_COMMIT:
        raise ConfigError(
            "pinned vLLM commit changed; prompt-logprob API contract requires review"
        )
    if (
        model.get("repository") != live_arm_receipt["model_repository"]
        or model.get("revision") != live_arm_receipt["model_revision"]
    ):
        raise ConfigError("live receipt model does not match canonical EXL3 pins")
    return {
        "pins_file_sha256": sha256_hex(raw),
        "vllm_commit": vllm_commit,
        "sparkinfer_commit": sparkinfer_commit,
        "exllamav3_commit": exllamav3_commit,
        "model_repository": model["repository"],
        "model_revision": model["revision"],
        "evidence_scope": (
            "declared-canonical-pins-bound-to-receipt-model-not-live-binary-introspection"
        ),
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    http: Any | None = None,
) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.discovery_repetitions < 2:
            raise ConfigError("--discovery-repetitions must be at least 2")
        if args.teacher_forced_repetitions < 2:
            raise ConfigError("--teacher-forced-repetitions must be at least 2")
        if not 2 <= args.top_logprobs <= 20:
            raise ConfigError("--top-logprobs must be between 2 and 20")
        if args.window_before < 0 or args.window_after < 0:
            raise ConfigError("teacher-forced window sizes must be nonnegative")
        if args.focus_generated_index is not None and args.focus_generated_index < 0:
            raise ConfigError("--focus-generated-index must be nonnegative")
        if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
            raise ConfigError("--timeout-seconds must be a finite positive number")
        if MODEL_RE.fullmatch(args.model) is None:
            raise ConfigError("--model must be a sanitized model identifier")
        base_url = inspectable_base_url(args.base_url)
        case, config_sha256 = _selected_case(Path(args.config), args.case_id)
        discovery_max_tokens = (
            case["max_tokens"]
            if args.discovery_max_tokens is None
            else args.discovery_max_tokens
        )
        if discovery_max_tokens <= 0:
            raise ConfigError("--discovery-max-tokens must be positive")
        if (
            args.focus_generated_index is not None
            and args.focus_generated_index + args.window_after
            >= discovery_max_tokens
        ):
            raise ConfigError(
                "teacher-forced focus/window must fit inside discovery max tokens"
            )
        output_path = Path(args.output) if args.output else None
        if output_path is not None and output_path.exists():
            raise ConfigError(f"--output already exists: {output_path}")
        try:
            live_arm_receipt = validate_live_arm_receipt(
                Path(args.activation_receipt),
                Path(args.profile),
                args.attribution_arm,
            )
            site = load_site(args.site)
            canonical_profile = attribution.exl3.load_profile(Path(args.profile))
            logprobs_contract = _profile_logprobs_contract(
                canonical_profile, args.top_logprobs
            )
            runtime_source_pins = _runtime_source_pins(live_arm_receipt)
            revalidation_actions = attribution.live_arm_revalidation_actions(
                site,
                canonical_profile,
                args.attribution_arm,
                live_arm_receipt,
            )
        except (
            OSError,
            json.JSONDecodeError,
            SiteConfigError,
            attribution.exl3.ProfileError,
            ValueError,
        ) as exc:
            raise ConfigError(f"cannot bind live attribution arm: {exc}") from exc
        cache_salt = cache_salt_for_arm(args.attribution_arm)
        plan = {
            "schema": PLAN_SCHEMA,
            "mutates_remote": False,
            "contacts_remote_when_executed": True,
            "remote_safety_class": "READ-ONLY REMOTE",
            "execute_requested": args.execute,
            "contacted_base_url": base_url,
            "contacted_http_targets": [
                f"{base_url}/health",
                f"{base_url}/tokenize",
                f"{base_url}/v1/completions",
            ],
            "model": args.model,
            "case_id": args.case_id,
            "config_sha256": config_sha256,
            "attribution_arm": args.attribution_arm,
            "cache_salt": cache_salt,
            "live_arm_receipt": live_arm_receipt,
            "runtime_source_pins": runtime_source_pins,
            "live_arm_revalidation_before_first_http": {
                "required": True,
                "actions": attribution.render(revalidation_actions),
            },
            "request_contract": {
                "discovery_repetitions": args.discovery_repetitions,
                "discovery_max_tokens": discovery_max_tokens,
                "focus_generated_index": args.focus_generated_index,
                "window_before": args.window_before,
                "window_after": args.window_after,
                "teacher_forced_repetitions": args.teacher_forced_repetitions,
                "prompt_logprobs": args.top_logprobs,
                **logprobs_contract,
            },
            "evidence_policy": {
                "raw_plan_and_report_are_private": True,
                "reason": (
                    "the plan contains reviewed SSH targets/API origin and the report "
                    "contains reversible exact token ID arrays"
                ),
                "full_vocabulary_kld_claimed": False,
            },
        }
        if args.command == "plan" or not args.execute:
            print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
            return EXIT_OK
        if output_path is None:
            raise ConfigError("executed run requires --output for private evidence")
        try:
            live_revalidation = attribution.revalidate_live_arm(
                site,
                canonical_profile,
                args.attribution_arm,
                live_arm_receipt,
                timeout=max(1, int(args.timeout_seconds)),
            )
        except attribution.exl3.ProfileError as exc:
            raise RequestFailure(f"live-arm re-attestation failed: {exc}") from exc
        client = http if http is not None else StrictUrllibHttpClient()
        status, _health_body = client.get_json(
            f"{base_url}/health", timeout=args.timeout_seconds
        )
        # vLLM's /health endpoint is status-only and normally returns an empty
        # body, so strict JSON parsing is intentionally not part of this check.
        if status != 200:
            raise RequestFailure(f"/health returned HTTP {status}")
        probe_result = run_probe(
            client,
            base_url=base_url,
            model=args.model,
            prompt=case_prompt(case, f"case {args.case_id}"),
            case_id=args.case_id,
            seed=case["seed"],
            discovery_repetitions=args.discovery_repetitions,
            discovery_max_tokens=discovery_max_tokens,
            focus_generated_index=args.focus_generated_index,
            window_before=args.window_before,
            window_after=args.window_after,
            teacher_forced_repetitions=args.teacher_forced_repetitions,
            top_logprobs=args.top_logprobs,
            timeout=args.timeout_seconds,
            cache_salt=cache_salt,
            ignore_eos=bool(case.get("ignore_eos", False)),
        )
        report = {
            "schema": REPORT_SCHEMA,
            "status": "completed",
            "lane": "public-functional",
            "maturity": "diagnostic-observation-not-acceptance",
            "contacted_base_url": base_url,
            "model": args.model,
            "case_id": args.case_id,
            "config_sha256": config_sha256,
            "runtime_identity": {
                "attribution_arm": args.attribution_arm,
                "mtp_tokens": 0,
                "native_prefix_caching_enabled": False,
                "lmcache_attached": False,
                "live_arm_receipt": live_arm_receipt,
                "live_arm_revalidation": live_revalidation,
                "runtime_source_pins": runtime_source_pins,
            },
            "request_contract": plan["request_contract"],
            **probe_result,
            "limitations": {
                "full_vocabulary_kld_available": False,
                "raw_logits_returned": False,
                "statement": (
                    "The pinned service returns raw top-k prompt logprobs. Pairwise "
                    "raw-logprob differences equal raw-logit margins, but truncated "
                    "common-support metrics are not full-vocabulary KLD. Teacher "
                    "forcing removes context drift; it does not identify the kernel, "
                    "graph, attention, or collective that produced changing logits."
                ),
            },
            "evidence_policy": {
                "raw_report_is_private": True,
                "reason": (
                    "contacted_base_url is site-specific and exact token ID arrays "
                    "are reversible"
                ),
                "full_vocabulary_kld_claimed": False,
            },
        }
        rendered = json.dumps(
            report, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(rendered)
        except FileExistsError as exc:
            raise ConfigError(f"--output already exists: {output_path}") from exc
        except OSError as exc:
            raise ConfigError(f"cannot write --output {output_path}: {exc}") from exc
        print(rendered, end="")
        return EXIT_OK
    except (ConfigError, CorrectnessConfigError, RequestFailure) as exc:
        is_config_error = isinstance(exc, (ConfigError, CorrectnessConfigError))
        kind = "CONFIG ERROR" if is_config_error else "FAIL"
        print(f"exl3-teacher-forced-margin-probe: {kind}: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR if is_config_error else EXIT_REQUEST_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
