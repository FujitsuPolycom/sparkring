#!/usr/bin/env python3
"""Multi-case deterministic correctness gate for the default EXL3 profile.

Executed runs perform a four-rank READ-ONLY REMOTE re-attestation of the
runtime-unique activation receipt before their first HTTP request. Plans and raw
reports are private because they disclose the reviewed SSH targets or API
origin; ``exl3_attribution_compare.py`` emits the sanitized comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit

from acceptance_gate import UrllibHttpClient, canonical_json, sha256_hex
import exl3_attribution_launcher as attribution
from exl3_attribution_cache_contract import (
    ARM_MTP_TOKENS,
    cache_salt_for_arm,
    validate_live_arm_receipt,
)
from sparkring_site import SiteConfigError, load_site


SCHEMA = "sparkring-exl3-correctness-cases/v1"
REPORT_SCHEMA = "sparkring-exl3-correctness-report/v4"
METRIC_PROBE_SCHEMA = "sparkring-exl3-cache-metric-probe/v2"
EXIT_OK = 0
EXIT_FUNCTIONAL_FAIL = 2
EXIT_CONFIG_ERROR = 3
EXIT_BASELINE_RECORDED = 4
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
MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}")
RUN_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
DNS_LABEL_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")


class ConfigError(ValueError):
    pass


class RequestFailure(RuntimeError):
    pass


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> Any:
    raise ConfigError(f"non-finite JSON number {value!r} is unsupported")


def _is_finite_nonnegative_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 0
    return isinstance(value, float) and math.isfinite(value) and value >= 0


def load_metric_probe(path: Path) -> dict[str, Any]:
    """Load and validate the bounded public fields of a cache metric probe."""
    try:
        raw = path.read_bytes()
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ConfigError) as exc:
        raise ConfigError(f"cannot read cache metric probe {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != METRIC_PROBE_SCHEMA:
        raise ConfigError(f"cache metric probe schema must be {METRIC_PROBE_SCHEMA}")
    model = document.get("model")
    run_label = document.get("run_label")
    attribution_arm = document.get("attribution_arm")
    cache_salt = document.get("cache_salt")
    live_arm_receipt = document.get("live_arm_receipt")
    prompt_sha = document.get("prompt_sha256")
    prompt_ids_sha = document.get("prompt_token_ids_sha256")
    prompt_count = document.get("prompt_token_count")
    if not isinstance(model, str) or MODEL_RE.fullmatch(model) is None:
        raise ConfigError("cache metric probe model is not a sanitized identifier")
    if not isinstance(run_label, str) or RUN_LABEL_RE.fullmatch(run_label) is None:
        raise ConfigError("cache metric probe run_label is not a sanitized identifier")
    if attribution_arm not in ARM_MTP_TOKENS:
        raise ConfigError("cache metric probe attribution_arm is unsupported")
    if cache_salt != cache_salt_for_arm(attribution_arm):
        raise ConfigError(
            "cache metric probe cache_salt does not bind its attribution layout"
        )
    if not isinstance(live_arm_receipt, dict):
        raise ConfigError("cache metric probe lacks a live-arm receipt binding")
    receipt_keys = {
        "schema", "status", "arm", "canonical_profile_id",
        "diagnostic_profile_id", "canonical_profile_file_sha256",
        "image_id", "model_repository", "model_revision", "layout",
        "cache_salt", "ranks", "attestation_scope", "artifact_sha256",
    }
    if set(live_arm_receipt) != receipt_keys:
        raise ConfigError("cache metric probe live-arm receipt fields are unsupported")
    live_receipt_sha = live_arm_receipt.get("artifact_sha256")
    if (
        not isinstance(live_receipt_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", live_receipt_sha) is None
        or live_arm_receipt.get("arm") != attribution_arm
        or live_arm_receipt.get("cache_salt") != cache_salt
        or live_arm_receipt.get("status") != "live-arm-attested"
    ):
        raise ConfigError("cache metric probe live-arm receipt binding is invalid")
    if prompt_sha is not None and (
        not isinstance(prompt_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", prompt_sha) is None
    ):
        raise ConfigError("cache metric probe prompt_sha256 must be lowercase SHA-256")
    for label, value in (("prompt_token_ids_sha256", prompt_ids_sha),):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ConfigError(f"cache metric probe {label} must be lowercase SHA-256")
    if not _is_int(prompt_count) or prompt_count < 1:
        raise ConfigError("cache metric probe prompt_token_count must be positive")
    expected_geometry = {
        "physical_block_tokens_per_dcp_rank": PHYSICAL_BLOCK_TOKENS_PER_DCP_RANK,
        "dcp_degree": DCP_DEGREE,
        "dcp_global_apc_alignment_tokens": DCP_GLOBAL_APC_ALIGNMENT_TOKENS,
        "lmcache_chunk_tokens": LMCACHE_CHUNK_TOKENS,
        "physical_blocks_per_dcp_global_apc_unit_per_rank": 1,
        "dcp_global_apc_units_per_lmcache_chunk": 2,
        "physical_blocks_per_lmcache_chunk_per_rank": 2,
    }
    observed_geometry = document.get("geometry")
    if observed_geometry != expected_geometry:
        raise ConfigError(
            "cache metric probe geometry does not bind physical=64, DCP4/global=256, "
            "and LMCache chunk=512 axes"
        )
    initial = document.get("initial_snapshot")
    cache_config = initial.get("cache_config") if isinstance(initial, dict) else None
    if not isinstance(cache_config, dict) or (
        cache_config.get("physical_block_tokens_per_dcp_rank")
        != PHYSICAL_BLOCK_TOKENS_PER_DCP_RANK
    ):
        raise ConfigError("cache metric probe initial snapshot does not bind live block size")
    native_prefix_value = cache_config.get("enable_prefix_caching")
    if native_prefix_value not in ("True", "False"):
        raise ConfigError(
            "cache metric probe did not bind the native prefix-caching state"
        )
    native_prefix_enabled = native_prefix_value == "True"
    for field in ("kv_cache_size_tokens", "num_gpu_blocks"):
        if not _is_int(cache_config.get(field)) or cache_config[field] < 1:
            raise ConfigError(f"cache metric probe initial {field} must be positive")
    initial_counters = initial.get("counters")
    initial_sources = initial.get("prompt_tokens_by_source")
    if not isinstance(initial_counters, dict) or set(initial_counters) != METRIC_COUNTERS:
        raise ConfigError("cache metric probe initial counter family is incomplete")
    if not isinstance(initial_sources, dict) or set(initial_sources) != METRIC_SOURCES:
        raise ConfigError("cache metric probe initial prompt source family is incomplete")
    initial_values = list(initial_counters.values()) + list(initial_sources.values())
    if any(not _is_finite_nonnegative_number(value) for value in initial_values):
        raise ConfigError(
            "cache metric probe initial counters must be finite nonnegative numbers"
        )
    observations = document.get("observations")
    if not isinstance(observations, list) or len(observations) < 2:
        raise ConfigError("cache metric probe must contain at least two observations")
    aggregate_counters = {name: 0.0 for name in METRIC_COUNTERS}
    aggregate_sources = {name: 0.0 for name in METRIC_SOURCES}
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict) or observation.get("repetition") != index + 1:
            raise ConfigError("cache metric probe observations are not ordered repetitions")
        delta = observation.get("metric_interval_delta")
        counters = delta.get("counters") if isinstance(delta, dict) else None
        sources = delta.get("prompt_tokens_by_source") if isinstance(delta, dict) else None
        if not isinstance(counters, dict) or not isinstance(sources, dict):
            raise ConfigError("cache metric probe observation lacks interval counter deltas")
        if set(counters) != METRIC_COUNTERS or set(sources) != METRIC_SOURCES:
            raise ConfigError("cache metric probe interval counter families are incomplete")
        interval_ns = delta.get("interval_ns")
        if not _is_int(interval_ns) or interval_ns <= 0:
            raise ConfigError("cache metric probe interval_ns must be a positive integer")
        values = list(counters.values()) + list(sources.values())
        if any(not _is_finite_nonnegative_number(value) for value in values):
            raise ConfigError(
                "cache metric probe counter deltas must be finite nonnegative numbers"
            )
        for name, value in counters.items():
            aggregate_counters[name] += float(value)
        for name, value in sources.items():
            aggregate_sources[name] += float(value)
    if not native_prefix_enabled and any(
        aggregate_counters[name] != 0.0
        for name in (
            "vllm:prefix_cache_queries_total",
            "vllm:prefix_cache_hits_total",
        )
    ):
        raise ConfigError(
            "cache metric probe reports native prefix counters while native prefix caching is disabled"
        )
    if not native_prefix_enabled and aggregate_sources["local_cache_hit"] != 0.0:
        raise ConfigError(
            "cache metric probe reports local cache-hit tokens while native prefix caching is disabled"
        )
    scope = document.get("evidence_scope")
    if not isinstance(scope, dict) or scope.get("classification") != (
        "request-interval-correlated-prometheus-counter-delta"
    ):
        raise ConfigError("cache metric probe evidence scope is unsupported")
    return {
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "schema": METRIC_PROBE_SCHEMA,
        "run_label": run_label,
        "model": model,
        "attribution_arm": attribution_arm,
        "cache_salt": cache_salt,
        "live_arm_receipt_sha256": live_receipt_sha,
        "live_arm_receipt": live_arm_receipt,
        "probe_prompt_sha256": prompt_sha,
        "probe_prompt_token_count": prompt_count,
        "probe_prompt_token_ids_sha256": prompt_ids_sha,
        "geometry": expected_geometry,
        "native_prefix_caching_enabled": native_prefix_enabled,
        "aggregate_counter_deltas": aggregate_counters,
        "aggregate_prompt_tokens_by_source": aggregate_sources,
        "observed_cache_layers": [
            layer
            for layer, observed in (
                ("native-apc", aggregate_sources["local_cache_hit"] > 0.0),
                (
                    "external-kv-transfer",
                    aggregate_sources["external_kv_transfer"] > 0.0,
                ),
            )
            if observed
        ],
        "observation_count": len(observations),
        "evidence_classification": scope["classification"],
    }


def load_metric_probe_mappings(
    values: Sequence[str], cases: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Load one closed metric-probe mapping for every cache-attribution case."""
    case_cache_policy = {
        case["id"]: bool(case.get("cache_attribution", False)) for case in cases
    }
    mappings: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, str) or "=" not in value:
            raise ConfigError(
                "--cache-metric-probe must use CASE_ID=PATH syntax"
            )
        case_id, path_text = value.split("=", 1)
        if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", case_id) is None:
            raise ConfigError(
                "--cache-metric-probe CASE_ID must be lowercase kebab-case"
            )
        if not path_text:
            raise ConfigError(
                f"--cache-metric-probe {case_id} path must be non-empty"
            )
        if case_id in mappings:
            raise ConfigError(
                f"duplicate --cache-metric-probe mapping for case {case_id}"
            )
        if case_id not in case_cache_policy:
            raise ConfigError(
                f"--cache-metric-probe maps unknown case {case_id}"
            )
        if not case_cache_policy[case_id]:
            raise ConfigError(
                f"--cache-metric-probe maps non-cache-attribution case {case_id}"
            )
        mappings[case_id] = load_metric_probe(Path(path_text))
    expected = {
        case_id for case_id, cache_attribution in case_cache_policy.items()
        if cache_attribution
    }
    missing = sorted(expected - set(mappings))
    if missing:
        raise ConfigError(
            "missing --cache-metric-probe mappings for cache-attribution cases: "
            + ", ".join(missing)
        )
    return mappings


def inspectable_base_url(value: str) -> str:
    """Validate and normalize the exact API origin contacted by the gate."""
    if (
        not isinstance(value, str)
        or not value
        or any(ord(character) <= 32 or ord(character) == 127 for character in value)
    ):
        raise ConfigError(
            "--base-url must not contain whitespace or control characters"
        )
    try:
        parsed = urlsplit(value)
        port = parsed.port
        host = parsed.hostname
    except ValueError as exc:
        raise ConfigError(f"--base-url is invalid: {exc}") from exc
    if (
        parsed.scheme not in ("http", "https")
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ConfigError(
            "--base-url must be an HTTP(S) origin without credentials, path, query, or fragment"
        )
    if (
        "%" in parsed.netloc
        or "\\" in parsed.netloc
        or not host.isascii()
        or len(host) > 253
    ):
        raise ConfigError(
            "--base-url authority must be an unescaped ASCII IP address or DNS name"
        )
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if any(DNS_LABEL_RE.fullmatch(label) is None for label in host.split(".")):
            raise ConfigError("--base-url hostname syntax is invalid")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host if port is None else f"{host}:{port}"
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


def case_prompt(case: dict[str, Any], prefix: str) -> str:
    prompt = case.get("prompt")
    generator = case.get("prompt_generator")
    if (prompt is None) == (generator is None):
        raise ConfigError(
            f"{prefix} must contain exactly one of prompt or prompt_generator"
        )
    if prompt is not None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ConfigError(f"{prefix}.prompt must be non-empty")
        return prompt
    required = {"kind", "fragment", "repetitions", "suffix"}
    if not isinstance(generator, dict) or set(generator) != required:
        raise ConfigError(
            f"{prefix}.prompt_generator keys must be {sorted(required)}"
        )
    if generator.get("kind") != "repeated-prefix-v1":
        raise ConfigError(
            f"{prefix}.prompt_generator.kind must be repeated-prefix-v1"
        )
    fragment = generator.get("fragment")
    suffix = generator.get("suffix")
    repetitions = generator.get("repetitions")
    if not isinstance(fragment, str) or not fragment.strip():
        raise ConfigError(f"{prefix}.prompt_generator.fragment must be non-empty")
    if not isinstance(suffix, str) or not suffix.strip():
        raise ConfigError(f"{prefix}.prompt_generator.suffix must be non-empty")
    if not isinstance(repetitions, int) or not 1 <= repetitions <= 4096:
        raise ConfigError(
            f"{prefix}.prompt_generator.repetitions must be in [1, 4096]"
        )
    return fragment * repetitions + suffix


def load_cases(path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read correctness cases {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ConfigError(f"correctness config schema must be {SCHEMA}")
    allowed_document_keys = {"schema", "cases", "cache_metric_probe_case_ids"}
    if not set(document) <= allowed_document_keys:
        raise ConfigError("correctness config contains unsupported top-level fields")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ConfigError("correctness config must contain at least one case")
    seen: set[str] = set()
    for index, case in enumerate(cases):
        prefix = f"cases[{index}]"
        if not isinstance(case, dict):
            raise ConfigError(f"{prefix} must be an object")
        identifier = case.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(
            r"[a-z0-9]+(?:-[a-z0-9]+)*", identifier
        ):
            raise ConfigError(f"{prefix}.id must be lowercase kebab-case")
        if identifier in seen:
            raise ConfigError(f"duplicate correctness case id {identifier!r}")
        seen.add(identifier)
        case_prompt(case, prefix)
        cache_attribution = case.get("cache_attribution", False)
        if not isinstance(cache_attribution, bool):
            raise ConfigError(f"{prefix}.cache_attribution must be boolean")
        minimum_prompt_tokens = case.get("minimum_prompt_tokens")
        if cache_attribution:
            if (
                not isinstance(minimum_prompt_tokens, int)
                or minimum_prompt_tokens < LMCACHE_CHUNK_TOKENS
            ):
                raise ConfigError(
                    f"{prefix}.minimum_prompt_tokens must be an integer >= "
                    f"{LMCACHE_CHUNK_TOKENS} for cache attribution"
                )
        elif minimum_prompt_tokens is not None:
            raise ConfigError(
                f"{prefix}.minimum_prompt_tokens requires cache_attribution=true"
            )
        for field in ("seed", "max_tokens"):
            if not isinstance(case.get(field), int) or case[field] <= 0:
                raise ConfigError(f"{prefix}.{field} must be a positive integer")
        ignore_eos = case.get("ignore_eos", False)
        if not isinstance(ignore_eos, bool):
            raise ConfigError(f"{prefix}.ignore_eos must be boolean")
        expected = case.get("expected_token_ids_sha256")
        if expected is not None and (
            not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
        ):
            raise ConfigError(
                f"{prefix}.expected_token_ids_sha256 must be null or SHA-256"
            )
    declared_probe_cases = document.get("cache_metric_probe_case_ids")
    expected_probe_cases = [
        case["id"] for case in cases if case.get("cache_attribution", False)
    ]
    if declared_probe_cases is not None and declared_probe_cases != expected_probe_cases:
        raise ConfigError(
            "cache_metric_probe_case_ids must exactly match ordered "
            "cache_attribution case IDs"
        )
    return cases, hashlib.sha256(raw).hexdigest()


def one_case(
    http: Any,
    *,
    base_url: str,
    model: str,
    case: dict[str, Any],
    repetitions: int,
    timeout: float,
    top_logprobs: int,
    metric_probe: dict[str, Any] | None,
    cache_salt: str,
) -> dict[str, Any]:
    prompt = case_prompt(case, f"case {case['id']}")
    prompt_status, prompt_body = http.post_json(
        f"{base_url}/tokenize",
        {"model": model, "prompt": prompt, "add_special_tokens": True},
        timeout=timeout,
    )
    prompt_token_ids = (
        prompt_body.get("tokens") if isinstance(prompt_body, dict) else None
    )
    if (
        prompt_status != 200
        or not isinstance(prompt_token_ids, list)
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in prompt_token_ids
        )
    ):
        raise RequestFailure(
            f"case {case['id']} could not measure prompt token count"
        )
    prompt_token_count = len(prompt_token_ids)
    prompt_token_ids_sha256 = sha256_hex(
        canonical_json(prompt_token_ids).encode("utf-8")
    )
    minimum_prompt_tokens = case.get("minimum_prompt_tokens")
    cache_attribution = bool(case.get("cache_attribution", False))
    if cache_attribution:
        if metric_probe is None:
            raise RequestFailure(
                f"case {case['id']} has no validated cache metric probe binding"
            )
        if metric_probe["model"] != model:
            raise RequestFailure(f"case {case['id']} metric probe model does not match")
        expected_probe_identity = {
            "probe_prompt_sha256": sha256_hex(prompt.encode("utf-8")),
            "probe_prompt_token_ids_sha256": prompt_token_ids_sha256,
            "probe_prompt_token_count": prompt_token_count,
        }
        for field, expected_value in expected_probe_identity.items():
            if metric_probe[field] != expected_value:
                raise RequestFailure(
                    f"case {case['id']} metric probe {field} does not bind this prompt"
                )
    reusable_prompt_tokens = max(prompt_token_count - 1, 0)
    boundary_report = {
        "physical_block_tokens_per_dcp_rank": PHYSICAL_BLOCK_TOKENS_PER_DCP_RANK,
        "dcp_degree": DCP_DEGREE,
        "dcp_global_apc_alignment_tokens": DCP_GLOBAL_APC_ALIGNMENT_TOKENS,
        "lmcache_chunk_tokens": LMCACHE_CHUNK_TOKENS,
        "minimum_prompt_tokens": minimum_prompt_tokens,
        "reusable_prompt_tokens": reusable_prompt_tokens,
        "reusable_dcp_global_apc_units": (
            reusable_prompt_tokens // DCP_GLOBAL_APC_ALIGNMENT_TOKENS
        ),
        "reusable_lmcache_chunks": reusable_prompt_tokens // LMCACHE_CHUNK_TOKENS,
        "has_reusable_dcp_global_apc_unit": (
            reusable_prompt_tokens >= DCP_GLOBAL_APC_ALIGNMENT_TOKENS
        ),
        "has_reusable_lmcache_chunk": reusable_prompt_tokens >= LMCACHE_CHUNK_TOKENS,
        "qualifies_for_cache_attribution": (
            cache_attribution
            and reusable_prompt_tokens >= DCP_GLOBAL_APC_ALIGNMENT_TOKENS
            and reusable_prompt_tokens >= LMCACHE_CHUNK_TOKENS
            and prompt_token_count >= minimum_prompt_tokens
        ),
    }
    if cache_attribution and not boundary_report["qualifies_for_cache_attribution"]:
        return {
            "id": case["id"],
            "status": "fail",
            "prompt_sha256": sha256_hex(prompt.encode("utf-8")),
            "prompt_token_count": prompt_token_count,
            "prompt_token_ids_sha256": prompt_token_ids_sha256,
            "cache_attribution": True,
            "cache_boundaries": boundary_report,
            "cache_metric_probe": metric_probe,
            "cache_evidence_scope": {
                "boundary_geometry_bound": True,
                "request_ids_bound": 0,
                "request_count": 0,
                "request_correlated_hit_evidence_count": 0,
                "request_correlated_store_evidence_count": 0,
                "causal_cache_claim": "not-claimed-boundary-failed",
                "statement": "No completion ran, so no hit or store is attributed.",
            },
            "seed": case["seed"],
            "max_tokens": case["max_tokens"],
            "ignore_eos": bool(case.get("ignore_eos", False)),
            "requested_top_logprobs": top_logprobs,
            "token_id_source": "retokenized-completion-text",
            "distinct_output_count": 0,
            "divergences_from_repetition_1": [],
            "sequence_boundary": "No completion ran because the prompt was below the declared cache-attribution boundary.",
            "expected_token_ids_sha256": case.get("expected_token_ids_sha256"),
            "observations": [],
            "failure": (
                f"prompt has {prompt_token_count} tokens; cache attribution "
                f"requires at least {minimum_prompt_tokens} prompt tokens and at least "
                f"257/513 prompt tokens for one reusable DCP-global APC/LMCache unit"
            ),
        }
    observations = []
    for repetition in range(repetitions):
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": case["seed"],
            "max_tokens": case["max_tokens"],
            "stream": False,
            "cache_salt": cache_salt,
        }
        if case.get("ignore_eos", False):
            payload["ignore_eos"] = True
        if top_logprobs:
            payload["logprobs"] = top_logprobs
        status, body = http.post_json(
            f"{base_url}/v1/completions", payload, timeout=timeout
        )
        choices = body.get("choices") if isinstance(body, dict) else None
        choice = choices[0] if isinstance(choices, list) and len(choices) == 1 else None
        text = (
            choice.get("text")
            if isinstance(choices, list)
            and len(choices) == 1
            and isinstance(choice, dict)
            else None
        )
        if status != 200 or not isinstance(text, str) or not text:
            raise RequestFailure(
                f"case {case['id']} completion failed with HTTP {status}"
            )
        completion_logprobs = choice.get("logprobs") if isinstance(choice, dict) else None
        if top_logprobs and not isinstance(completion_logprobs, dict):
            raise RequestFailure(
                f"case {case['id']} requested logprobs but the response omitted them"
            )
        response_id = body.get("id") if isinstance(body, dict) else None
        response_id_sha256 = (
            sha256_hex(response_id.encode("utf-8"))
            if isinstance(response_id, str) and response_id
            else None
        )
        usage = body.get("usage") if isinstance(body, dict) else None
        usage_prompt_tokens = (
            usage.get("prompt_tokens") if isinstance(usage, dict) else None
        )
        prompt_details = (
            usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
        )
        cached_tokens = (
            prompt_details.get("cached_tokens")
            if isinstance(prompt_details, dict)
            else None
        )
        if usage_prompt_tokens is not None and (
            not isinstance(usage_prompt_tokens, int)
            or isinstance(usage_prompt_tokens, bool)
            or usage_prompt_tokens < 0
        ):
            raise RequestFailure(
                f"case {case['id']} returned invalid usage.prompt_tokens"
            )
        if (
            usage_prompt_tokens is not None
            and usage_prompt_tokens != prompt_token_count
        ):
            raise RequestFailure(
                f"case {case['id']} usage.prompt_tokens does not match /tokenize"
            )
        if cached_tokens is not None and (
            not isinstance(cached_tokens, int)
            or isinstance(cached_tokens, bool)
            or cached_tokens < 0
            or usage_prompt_tokens is None
            or cached_tokens > usage_prompt_tokens
        ):
            raise RequestFailure(
                f"case {case['id']} returned invalid cached-token metadata"
            )
        token_status, token_body = http.post_json(
            f"{base_url}/tokenize",
            {"model": model, "prompt": text, "add_special_tokens": False},
            timeout=timeout,
        )
        token_ids = token_body.get("tokens") if isinstance(token_body, dict) else None
        if (
            token_status != 200
            or not isinstance(token_ids, list)
            or not token_ids
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in token_ids
            )
        ):
            raise RequestFailure(
                f"case {case['id']} could not recover output token ids"
            )
        observations.append(
            {
                "repetition": repetition + 1,
                "run_position": (
                    "first-in-run" if repetition == 0 else "subsequent-in-run"
                ),
                "token_ids": token_ids,
                "token_ids_sha256": sha256_hex(
                    canonical_json(token_ids).encode("utf-8")
                ),
                "text_sha256": sha256_hex(text.encode("utf-8")),
                "completion_logprobs": completion_logprobs,
                "request_evidence": {
                    "response_id_sha256": response_id_sha256,
                    "usage_prompt_tokens": usage_prompt_tokens,
                    "usage_cached_prompt_tokens": cached_tokens,
                    "hit_evidence_source": (
                        "openai-usage.prompt_tokens_details.cached_tokens"
                        if cached_tokens is not None and cached_tokens > 0
                        else None
                    ),
                    "store_evidence_source": None,
                },
            }
        )
    hashes = {item["token_ids_sha256"] for item in observations}
    reference_ids = observations[0]["token_ids"]
    divergences = []
    for observation in observations[1:]:
        observed_ids = observation["token_ids"]
        first_divergence = next(
            (
                index
                for index, (expected_id, observed_id) in enumerate(
                    zip(reference_ids, observed_ids)
                )
                if expected_id != observed_id
            ),
            min(len(reference_ids), len(observed_ids)),
        )
        divergences.append(
            {
                "repetition": observation["repetition"],
                "matches_repetition_1": observed_ids == reference_ids,
                "first_divergence_index": (
                    None if observed_ids == reference_ids else first_divergence
                ),
                "reference_token_count": len(reference_ids),
                "observed_token_count": len(observed_ids),
            }
        )
    expected = case.get("expected_token_ids_sha256")
    status = "pass"
    failure = None
    if len(hashes) != 1:
        status = "fail"
        failure = "repetitions diverged"
    elif expected is None:
        status = "baseline-recorded"
    elif next(iter(hashes)) != expected:
        status = "fail"
        failure = f"observed {next(iter(hashes))} != expected {expected}"
    request_ids_bound = sum(
        item["request_evidence"]["response_id_sha256"] is not None
        for item in observations
    )
    hit_evidence_bound = sum(
        (
            item["request_evidence"]["usage_cached_prompt_tokens"] is not None
            and item["request_evidence"]["usage_cached_prompt_tokens"] > 0
        )
        for item in observations
    )
    return {
        "id": case["id"],
        "status": status,
        "prompt_sha256": sha256_hex(prompt.encode("utf-8")),
        "prompt_token_count": prompt_token_count,
        "prompt_token_ids_sha256": prompt_token_ids_sha256,
        "cache_attribution": cache_attribution,
        "cache_boundaries": boundary_report,
        "cache_metric_probe": metric_probe,
        "cache_evidence_scope": {
            "boundary_geometry_bound": cache_attribution,
            "request_ids_bound": request_ids_bound,
            "request_count": len(observations),
            "request_correlated_hit_evidence_count": hit_evidence_bound,
            "request_correlated_store_evidence_count": 0,
            "causal_cache_claim": (
                "not-claimed-store-evidence-unavailable"
                if cache_attribution
                else "not-applicable"
            ),
            "statement": (
                "Prompt geometry is bound, but boundary crossing alone does not prove "
                "a cache hit or store. Hit evidence is counted only when the response "
                "contains request-correlated cached-token usage; this API response "
                "does not expose store events."
            ),
        },
        "seed": case["seed"],
        "max_tokens": case["max_tokens"],
        "ignore_eos": bool(case.get("ignore_eos", False)),
        "requested_top_logprobs": top_logprobs,
        "token_id_source": "retokenized-completion-text",
        "distinct_output_count": len(hashes),
        "divergences_from_repetition_1": divergences,
        "sequence_boundary": (
            "The first observation is only first-in-run, not necessarily cold; "
            "earlier process traffic may already have populated native APC or "
            "an external cache. Subsequent observations reuse the identical "
            "prompt without an intervening cache reset."
        ),
        "expected_token_ids_sha256": expected,
        "observations": observations,
        "failure": failure,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--site",
        required=True,
        help="site file used for READ-ONLY REMOTE four-rank live-arm re-attestation",
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--attribution-arm",
        required=True,
        choices=tuple(ARM_MTP_TOKENS),
        help=(
            "bind every cache-writing request to the layout-derived attribution "
            "namespace emitted by exl3_attribution_launcher.py"
        ),
    )
    parser.add_argument(
        "--activation-receipt",
        required=True,
        help="public-safe live-arm receipt emitted by the attribution launcher",
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="exact canonical profile whose digest/identity the receipt must bind",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--run-label",
        default="unlabelled",
        help="stable arm/run identity recorded in the plan and report",
    )
    parser.add_argument(
        "--output",
        help=(
            "write the run report as UTF-8 JSON using exclusive creation; "
            "an existing path is never overwritten"
        ),
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=0,
        help="request and persist this many completion top-logprobs (0 disables)",
    )
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--cache-metric-probe",
        action="append",
        default=[],
        help=(
            "CASE_ID=PATH mapping to a validated exl3_cache_metric_probe JSON; "
            "repeat exactly once for every cache_attribution case"
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("command", choices=("plan", "run"))
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    http: Any | None = None,
) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.repetitions < 2:
            raise ConfigError("--repetitions must be at least 2")
        if not 0 <= args.top_logprobs <= 20:
            raise ConfigError("--top-logprobs must be between 0 and 20")
        if MODEL_RE.fullmatch(args.model) is None:
            raise ConfigError("--model must be a sanitized model identifier")
        if RUN_LABEL_RE.fullmatch(args.run_label) is None:
            raise ConfigError(
                "--run-label must be a bounded public-safe identifier"
            )
        try:
            live_arm_receipt = validate_live_arm_receipt(
                Path(args.activation_receipt),
                Path(args.profile),
                args.attribution_arm,
            )
        except ValueError as exc:
            raise ConfigError(f"invalid --activation-receipt: {exc}") from exc
        try:
            site = load_site(args.site)
            canonical_profile = attribution.exl3.load_profile(Path(args.profile))
            revalidation_actions = attribution.live_arm_revalidation_actions(
                site,
                canonical_profile,
                args.attribution_arm,
                live_arm_receipt,
            )
        except (OSError, json.JSONDecodeError, SiteConfigError, attribution.exl3.ProfileError) as exc:
            raise ConfigError(f"cannot compose live-arm re-attestation: {exc}") from exc
        base = inspectable_base_url(args.base_url)
        output_path = Path(args.output) if args.output else None
        if output_path is not None and output_path.exists():
            raise ConfigError(f"--output already exists: {output_path}")
        cases, config_sha256 = load_cases(Path(args.config))
        cache_cases = [case for case in cases if case.get("cache_attribution", False)]
        metric_probes = load_metric_probe_mappings(args.cache_metric_probe, cases)
        cache_salt = cache_salt_for_arm(args.attribution_arm)
        for case in cache_cases:
            case_id = case["id"]
            metric_probe = metric_probes[case_id]
            if (
                metric_probe["attribution_arm"] != args.attribution_arm
                or metric_probe["cache_salt"] != cache_salt
                or metric_probe["live_arm_receipt_sha256"]
                != live_arm_receipt["artifact_sha256"]
                or metric_probe["live_arm_receipt"] != live_arm_receipt
            ):
                raise ConfigError(
                    f"cache metric probe for case {case_id} attribution arm/cache "
                    "salt/live-arm receipt does not match this run"
                )
            expected_prompt_sha256 = sha256_hex(
                case_prompt(case, f"case {case_id}").encode("utf-8")
            )
            if metric_probe["model"] != args.model:
                raise ConfigError(
                    f"cache metric probe for case {case_id} model does not match"
                )
            if metric_probe["probe_prompt_sha256"] != expected_prompt_sha256:
                raise ConfigError(
                    f"cache metric probe for case {case_id} prompt text does not match"
                )
        plan = {
            "schema": "sparkring-exl3-correctness-plan/v1",
            "mutates_remote": False,
            "contacts_remote_when_executed": True,
            "remote_safety_class": "READ-ONLY REMOTE",
            "live_arm_revalidation_before_first_http": {
                "required": True,
                "actions": attribution.render(revalidation_actions),
            },
            "execute_requested": args.execute,
            "contacted_base_url": base,
            "contacted_http_targets": [
                f"{base}/health",
                f"{base}/tokenize",
                f"{base}/v1/completions",
            ],
            "model": args.model,
            "run_label": args.run_label,
            "attribution_arm": args.attribution_arm,
            "cache_salt": cache_salt,
            "live_arm_receipt": live_arm_receipt,
            "case_ids": [case["id"] for case in cases],
            "repetitions": args.repetitions,
            "top_logprobs": args.top_logprobs,
            "config_sha256": config_sha256,
            "cache_metric_probe_artifacts": [
                {
                    "case_id": case["id"],
                    "artifact_sha256": metric_probes[case["id"]]["artifact_sha256"],
                    "geometry": metric_probes[case["id"]]["geometry"],
                    "native_prefix_caching_enabled": metric_probes[case["id"]][
                        "native_prefix_caching_enabled"
                    ],
                    "observed_cache_layers": metric_probes[case["id"]][
                        "observed_cache_layers"
                    ],
                }
                for case in cache_cases
            ],
            "evidence_policy": {
                "raw_plan_and_report_are_private": True,
                "reason": (
                    "the plan carries SSH targets and the report carries the exact "
                    "contacted base URL"
                ),
                "publishable_comparator": "scripts/exl3_attribution_compare.py",
            },
        }
        if args.command == "plan" or not args.execute:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return EXIT_OK
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
        client = http if http is not None else UrllibHttpClient()
        status, _ = client.get_json(f"{base}/health", timeout=args.timeout_seconds)
        if status != 200:
            raise RequestFailure(f"/health returned {status}")
        results = [
            one_case(
                client,
                base_url=base,
                model=args.model,
                case=case,
                repetitions=args.repetitions,
                timeout=args.timeout_seconds,
                top_logprobs=args.top_logprobs,
                metric_probe=metric_probes.get(case["id"]),
                cache_salt=cache_salt,
            )
            for case in cases
        ]
        failures = [item["id"] for item in results if item["status"] == "fail"]
        baselines = [
            item["id"] for item in results if item["status"] == "baseline-recorded"
        ]
        report_status = (
            "fail" if failures else "baseline-recorded" if baselines else "pass"
        )
        report = {
            "schema": REPORT_SCHEMA,
            "status": report_status,
            "contacted_base_url": base,
            "model": args.model,
            "run_label": args.run_label,
            "attribution_arm": args.attribution_arm,
            "cache_salt": cache_salt,
            "live_arm_receipt": live_arm_receipt,
            "live_arm_revalidation": live_revalidation,
            "config_sha256": config_sha256,
            "repetitions": args.repetitions,
            "top_logprobs": args.top_logprobs,
            "cases": results,
            "failures": failures,
            "baselines": baselines,
            "evidence_policy": {
                "raw_report_is_private": True,
                "reason": "contacted_base_url contains a site endpoint",
                "publishable_comparator": "scripts/exl3_attribution_compare.py",
            },
        }
        rendered = json.dumps(
            report, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        if output_path is not None:
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(rendered)
            except FileExistsError as exc:
                raise ConfigError(f"--output already exists: {output_path}") from exc
            except OSError as exc:
                raise ConfigError(f"cannot write --output {output_path}: {exc}") from exc
        print(rendered, end="")
        if failures:
            return EXIT_FUNCTIONAL_FAIL
        if baselines:
            return EXIT_BASELINE_RECORDED
        return EXIT_OK
    except (ConfigError, RequestFailure) as exc:
        kind = "CONFIG ERROR" if isinstance(exc, ConfigError) else "FAIL"
        print(f"exl3-correctness-gate: {kind}: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR if isinstance(exc, ConfigError) else EXIT_FUNCTIONAL_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
