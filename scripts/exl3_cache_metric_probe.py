#!/usr/bin/env python3
"""Capture bounded native/external prefix-cache metric deltas for one prompt.

This probe is deliberately separate from the correctness gate.  A Prometheus
counter delta is interval-correlated evidence, not an OpenAI request field and
not a request-ID-level cache receipt.  The report preserves that distinction.
Executed runs first perform a four-rank READ-ONLY REMOTE re-attestation of the
runtime-unique activation receipt; stale or replaced containers abort before
the first HTTP request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit

import exl3_attribution_launcher as attribution
from exl3_attribution_cache_contract import (
    ARM_MTP_TOKENS,
    cache_salt_for_arm,
    validate_live_arm_receipt,
)
from sparkring_site import SiteConfigError, load_site


SCHEMA = "sparkring-exl3-cache-metric-probe/v2"
MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}")
RUN_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
SAMPLE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
)
LABEL_RE = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"(?:,|$)')

COUNTERS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
)
SOURCES = ("local_compute", "local_cache_hit", "external_kv_transfer")


class ProbeError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def inspectable_base_url(value: str) -> str:
    """Return an inspectable HTTP(S) origin and reject hidden URL state."""
    try:
        parsed = urlsplit(value.rstrip("/"))
        port = parsed.port
    except ValueError as exc:
        raise ProbeError(f"--base-url is invalid: {exc}") from exc
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ProbeError(
            "--base-url must be an HTTP(S) origin without credentials, path, query, or fragment"
        )
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host if port is None else f"{host}:{port}"
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


def _decode_label(value: str) -> str:
    return bytes(value, "utf-8").decode("unicode_escape")


def parse_samples(text: str) -> list[tuple[str, dict[str, str], float]]:
    samples = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_RE.fullmatch(line)
        if match is None:
            continue
        labels_text = match.group("labels") or ""
        labels: dict[str, str] = {}
        cursor = 0
        while cursor < len(labels_text):
            label_match = LABEL_RE.match(labels_text, cursor)
            if label_match is None:
                raise ProbeError(f"malformed Prometheus labels for {match.group('name')}")
            labels[label_match.group(1)] = _decode_label(label_match.group(2))
            cursor = label_match.end()
        samples.append((match.group("name"), labels, float(match.group("value"))))
    return samples


def _one_sample(
    samples: list[tuple[str, dict[str, str], float]],
    name: str,
    *,
    source: str | None = None,
) -> tuple[dict[str, str], float]:
    matches = [
        (labels, value)
        for metric_name, labels, value in samples
        if metric_name == name and (source is None or labels.get("source") == source)
    ]
    if len(matches) != 1:
        raise ProbeError(f"expected one {name} sample, found {len(matches)}")
    return matches[0]


def metric_snapshot(text: str) -> dict[str, Any]:
    samples = parse_samples(text)
    counters = {name: _one_sample(samples, name)[1] for name in COUNTERS}
    sources = {
        source: _one_sample(samples, "vllm:prompt_tokens_by_source_total", source=source)[1]
        for source in SOURCES
    }
    config_labels, config_value = _one_sample(samples, "vllm:cache_config_info")
    values = list(counters.values()) + list(sources.values())
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ProbeError("cache metrics must be finite nonnegative numbers")
    if not math.isfinite(config_value) or config_value != 1.0:
        raise ProbeError("vllm:cache_config_info must equal 1")
    try:
        block_size = int(config_labels["block_size"])
    except (KeyError, ValueError) as exc:
        raise ProbeError("cache_config_info.block_size is missing or invalid") from exc
    try:
        kv_cache_size_tokens = int(config_labels["kv_cache_size_tokens"])
        num_gpu_blocks = int(config_labels["num_gpu_blocks"])
    except (KeyError, ValueError) as exc:
        raise ProbeError("cache_config_info capacity labels are missing or invalid") from exc
    if block_size < 1 or kv_cache_size_tokens < 1 or num_gpu_blocks < 1:
        raise ProbeError("cache_config_info geometry and capacity must be positive")
    return {
        "captured_unix_ns": time.time_ns(),
        "counters": counters,
        "prompt_tokens_by_source": sources,
        "cache_config": {
            "physical_block_tokens_per_dcp_rank": block_size,
            "enable_prefix_caching": config_labels.get("enable_prefix_caching"),
            "kv_cache_size_tokens": kv_cache_size_tokens,
            "num_gpu_blocks": num_gpu_blocks,
        },
    }


def snapshot_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    if before["cache_config"] != after["cache_config"]:
        raise ProbeError("cache configuration changed during the request interval")
    delta: dict[str, Any] = {
        "interval_ns": after["captured_unix_ns"] - before["captured_unix_ns"],
        "counters": {},
        "prompt_tokens_by_source": {},
    }
    for family in ("counters", "prompt_tokens_by_source"):
        for key, value in before[family].items():
            change = after[family][key] - value
            if not isinstance(change, (int, float)) or isinstance(change, bool):
                raise ProbeError(f"counter {family}.{key} is not numeric")
            if not math.isfinite(float(change)):
                raise ProbeError(f"counter {family}.{key} delta is non-finite")
            if change < 0:
                raise ProbeError(f"counter {family}.{key} decreased")
            delta[family][key] = change
    return delta


class HttpClient:
    def _request(self, request: urllib.request.Request, timeout: float) -> tuple[int, bytes]:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()
        except urllib.error.URLError as error:
            raise ProbeError(f"HTTP request failed: {type(error.reason).__name__}") from error

    def get_text(self, url: str, timeout: float) -> tuple[int, str]:
        status, body = self._request(urllib.request.Request(url, method="GET"), timeout)
        return status, body.decode("utf-8")

    def post_json(self, url: str, payload: dict[str, Any], timeout: float) -> tuple[int, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        status, body = self._request(request, timeout)
        try:
            return status, json.loads(body)
        except json.JSONDecodeError as exc:
            raise ProbeError(f"HTTP {status} returned non-JSON response") from exc


def _get_metrics(client: Any, base_url: str, timeout: float) -> dict[str, Any]:
    status, text = client.get_text(f"{base_url}/metrics", timeout)
    if status != 200:
        raise ProbeError(f"/metrics returned HTTP {status}")
    return metric_snapshot(text)


def run_probe(
    client: Any,
    *,
    base_url: str,
    model: str,
    prompt: str,
    repetitions: int,
    max_tokens: int,
    seed: int,
    timeout: float,
    dcp_degree: int,
    expected_logical_block_tokens: int,
    lmcache_chunk_tokens: int,
    cache_salt: str,
) -> dict[str, Any]:
    token_status, token_body = client.post_json(
        f"{base_url}/tokenize",
        {"model": model, "prompt": prompt, "add_special_tokens": True},
        timeout,
    )
    prompt_ids = token_body.get("tokens") if isinstance(token_body, dict) else None
    if (
        token_status != 200
        or not isinstance(prompt_ids, list)
        or not prompt_ids
        or any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            for token_id in prompt_ids
        )
    ):
        raise ProbeError("could not tokenize probe prompt")

    initial = _get_metrics(client, base_url, timeout)
    physical = initial["cache_config"]["physical_block_tokens_per_dcp_rank"]
    logical = physical * dcp_degree
    if logical != expected_logical_block_tokens:
        raise ProbeError(
            f"live logical block is {physical} * {dcp_degree} = {logical}, "
            f"expected {expected_logical_block_tokens}"
        )
    if lmcache_chunk_tokens % logical:
        raise ProbeError(
            f"LMCache chunk {lmcache_chunk_tokens} is not divisible by logical block {logical}"
        )

    observations = []
    previous = initial
    for repetition in range(1, repetitions + 1):
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": seed,
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "stream": False,
            "cache_salt": cache_salt,
        }
        status, body = client.post_json(f"{base_url}/v1/completions", payload, timeout)
        choices = body.get("choices") if isinstance(body, dict) else None
        choice = choices[0] if isinstance(choices, list) and len(choices) == 1 else None
        text = choice.get("text") if isinstance(choice, dict) else None
        if status != 200 or not isinstance(text, str):
            raise ProbeError(f"completion repetition {repetition} returned HTTP {status}")
        after = _get_metrics(client, base_url, timeout)
        usage = body.get("usage") if isinstance(body, dict) else None
        details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
        usage_prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        cached_tokens = details.get("cached_tokens") if isinstance(details, dict) else None
        if usage_prompt_tokens is not None and (
            not isinstance(usage_prompt_tokens, int)
            or isinstance(usage_prompt_tokens, bool)
            or usage_prompt_tokens < 0
        ):
            raise ProbeError("completion usage.prompt_tokens must be a nonnegative integer")
        if usage_prompt_tokens is not None and usage_prompt_tokens != len(prompt_ids):
            raise ProbeError("completion usage.prompt_tokens does not match /tokenize")
        if cached_tokens is not None and (
            not isinstance(cached_tokens, int)
            or isinstance(cached_tokens, bool)
            or cached_tokens < 0
        ):
            raise ProbeError(
                "completion usage.prompt_tokens_details.cached_tokens must be a nonnegative integer"
            )
        if cached_tokens is not None and (
            usage_prompt_tokens is None or cached_tokens > usage_prompt_tokens
        ):
            raise ProbeError("completion cached tokens do not bind usage.prompt_tokens")
        response_id = body.get("id") if isinstance(body, dict) else None
        observations.append(
            {
                "repetition": repetition,
                "response_id_sha256": (
                    sha256_hex(response_id.encode("utf-8"))
                    if isinstance(response_id, str) and response_id
                    else None
                ),
                "completion_text_sha256": sha256_hex(text.encode("utf-8")),
                "usage_prompt_tokens": usage_prompt_tokens,
                "usage_cached_prompt_tokens": cached_tokens,
                "metric_interval_delta": snapshot_delta(previous, after),
            }
        )
        previous = after

    return {
        "prompt_sha256": sha256_hex(prompt.encode("utf-8")),
        "prompt_token_count": len(prompt_ids),
        "prompt_token_ids_sha256": sha256_hex(canonical_json(prompt_ids).encode("utf-8")),
        "geometry": {
            "physical_block_tokens_per_dcp_rank": physical,
            "dcp_degree": dcp_degree,
            "dcp_global_apc_alignment_tokens": logical,
            "lmcache_chunk_tokens": lmcache_chunk_tokens,
            "physical_blocks_per_dcp_global_apc_unit_per_rank": 1,
            "dcp_global_apc_units_per_lmcache_chunk": lmcache_chunk_tokens // logical,
            "physical_blocks_per_lmcache_chunk_per_rank": lmcache_chunk_tokens // logical,
        },
        "initial_snapshot": initial,
        "observations": observations,
        "evidence_scope": {
            "classification": "request-interval-correlated-prometheus-counter-delta",
            "request_id_level_cache_receipt": False,
            "concurrent_traffic_excluded_by_probe": False,
            "statement": (
                "Each delta brackets one completion request, but Prometheus counters are "
                "process-global and cannot exclude unrelated concurrent requests."
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site",
        required=True,
        help="site file used for READ-ONLY REMOTE four-rank live-arm re-attestation",
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--attribution-arm", required=True, choices=tuple(ARM_MTP_TOKENS)
    )
    parser.add_argument("--activation-receipt", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-fragment", required=True)
    parser.add_argument("--prompt-repetitions", type=int, default=64)
    parser.add_argument("--prompt-suffix", required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--dcp-degree", type=int, default=4)
    parser.add_argument("--expected-logical-block-tokens", type=int, default=256)
    parser.add_argument("--lmcache-chunk-tokens", type=int, default=512)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None, *, client: Any | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if MODEL_RE.fullmatch(args.model) is None:
            raise ProbeError("--model must be a sanitized identifier")
        if RUN_LABEL_RE.fullmatch(args.run_label) is None:
            raise ProbeError("--run-label must be a bounded public-safe identifier")
        if args.repetitions < 2 or args.prompt_repetitions < 1 or args.max_tokens < 1:
            raise ProbeError("repetitions and token counts must be positive; repetitions >= 2")
        if args.dcp_degree < 1 or args.expected_logical_block_tokens < 1:
            raise ProbeError("DCP and block geometry must be positive")
        try:
            live_arm_receipt = validate_live_arm_receipt(
                Path(args.activation_receipt),
                Path(args.profile),
                args.attribution_arm,
            )
        except ValueError as exc:
            raise ProbeError(f"invalid --activation-receipt: {exc}") from exc
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
            raise ProbeError(f"cannot compose live-arm re-attestation: {exc}") from exc
        base_url = inspectable_base_url(args.base_url)
        cache_salt = cache_salt_for_arm(args.attribution_arm)
        output = Path(args.output)
        if output.exists():
            raise ProbeError(f"output already exists: {output}")
        plan = {
            "schema": "sparkring-exl3-cache-metric-probe-plan/v1",
            "contacted_base_url": base_url,
            "model": args.model,
            "run_label": args.run_label,
            "attribution_arm": args.attribution_arm,
            "cache_salt": cache_salt,
            "live_arm_receipt": live_arm_receipt,
            "repetitions": args.repetitions,
            "execute_requested": args.execute,
            "safety_class": "READ-ONLY REMOTE",
            "live_arm_revalidation_before_first_http": {
                "required": True,
                "actions": attribution.render(revalidation_actions),
            },
            "transient_serving_state_effect": "fills native/external prefix caches",
            "evidence_policy": {
                "raw_plan_and_report_are_private": True,
                "reason": (
                    "the plan carries SSH targets and the report carries the exact "
                    "contacted base URL"
                ),
            },
        }
        if not args.execute:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        prompt = args.prompt_fragment * args.prompt_repetitions + args.prompt_suffix
        try:
            live_revalidation = attribution.revalidate_live_arm(
                site,
                canonical_profile,
                args.attribution_arm,
                live_arm_receipt,
                timeout=max(1, int(args.timeout_seconds)),
            )
        except attribution.exl3.ProfileError as exc:
            raise ProbeError(f"live-arm re-attestation failed: {exc}") from exc
        result = run_probe(
            client or HttpClient(),
            base_url=base_url,
            model=args.model,
            prompt=prompt,
            repetitions=args.repetitions,
            max_tokens=args.max_tokens,
            seed=args.seed,
            timeout=args.timeout_seconds,
            dcp_degree=args.dcp_degree,
            expected_logical_block_tokens=args.expected_logical_block_tokens,
            lmcache_chunk_tokens=args.lmcache_chunk_tokens,
            cache_salt=cache_salt,
        )
        report = {
            "schema": SCHEMA,
            "run_label": args.run_label,
            "model": args.model,
            "attribution_arm": args.attribution_arm,
            "cache_salt": cache_salt,
            "live_arm_receipt": live_arm_receipt,
            "live_arm_revalidation": live_revalidation,
            "contacted_base_url": base_url,
            "evidence_policy": {
                "raw_report_is_private": True,
                "reason": "contacted_base_url contains a site endpoint",
            },
            **result,
        }
        rendered = json.dumps(
            report, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
        print(rendered, end="")
        return 0
    except (OSError, ProbeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
