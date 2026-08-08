#!/usr/bin/env python3
"""Multi-case deterministic correctness gate for the default EXL3 profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from acceptance_gate import UrllibHttpClient, canonical_json, sha256_hex


SCHEMA = "sparkring-exl3-correctness-cases/v1"
REPORT_SCHEMA = "sparkring-exl3-correctness-report/v1"
EXIT_OK = 0
EXIT_FUNCTIONAL_FAIL = 2
EXIT_CONFIG_ERROR = 3
EXIT_BASELINE_RECORDED = 4


class ConfigError(ValueError):
    pass


class RequestFailure(RuntimeError):
    pass


def load_cases(path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read correctness cases {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ConfigError(f"correctness config schema must be {SCHEMA}")
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
        if not isinstance(case.get("prompt"), str) or not case["prompt"].strip():
            raise ConfigError(f"{prefix}.prompt must be non-empty")
        for field in ("seed", "max_tokens"):
            if not isinstance(case.get(field), int) or case[field] <= 0:
                raise ConfigError(f"{prefix}.{field} must be a positive integer")
        expected = case.get("expected_token_ids_sha256")
        if expected is not None and (
            not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
        ):
            raise ConfigError(
                f"{prefix}.expected_token_ids_sha256 must be null or SHA-256"
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
) -> dict[str, Any]:
    observations = []
    for _ in range(repetitions):
        payload = {
            "model": model,
            "prompt": case["prompt"],
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": case["seed"],
            "max_tokens": case["max_tokens"],
            "stream": False,
        }
        status, body = http.post_json(
            f"{base_url}/v1/completions", payload, timeout=timeout
        )
        choices = body.get("choices") if isinstance(body, dict) else None
        text = (
            choices[0].get("text")
            if isinstance(choices, list)
            and len(choices) == 1
            and isinstance(choices[0], dict)
            else None
        )
        if status != 200 or not isinstance(text, str) or not text:
            raise RequestFailure(
                f"case {case['id']} completion failed with HTTP {status}"
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
            or any(not isinstance(value, int) for value in token_ids)
        ):
            raise RequestFailure(
                f"case {case['id']} could not recover output token ids"
            )
        observations.append(
            {
                "token_ids": token_ids,
                "token_ids_sha256": sha256_hex(
                    canonical_json(token_ids).encode("utf-8")
                ),
                "text_sha256": sha256_hex(text.encode("utf-8")),
            }
        )
    hashes = {item["token_ids_sha256"] for item in observations}
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
    return {
        "id": case["id"],
        "status": status,
        "prompt_sha256": sha256_hex(case["prompt"].encode("utf-8")),
        "seed": case["seed"],
        "max_tokens": case["max_tokens"],
        "expected_token_ids_sha256": expected,
        "observations": observations,
        "failure": failure,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
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
        cases, config_sha256 = load_cases(Path(args.config))
        plan = {
            "schema": "sparkring-exl3-correctness-plan/v1",
            "mutates_remote": False,
            "execute_requested": args.execute,
            "model": args.model,
            "case_ids": [case["id"] for case in cases],
            "repetitions": args.repetitions,
            "config_sha256": config_sha256,
        }
        if args.command == "plan" or not args.execute:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return EXIT_OK
        client = http if http is not None else UrllibHttpClient()
        base = args.base_url.rstrip("/")
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
            "model": args.model,
            "config_sha256": config_sha256,
            "repetitions": args.repetitions,
            "cases": results,
            "failures": failures,
            "baselines": baselines,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
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
