#!/usr/bin/env python3
"""Dry-run-first LMCache persistence-boundary gate for the EXL3 default."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from acceptance_gate import SubprocessExecutor, UrllibHttpClient
import sparkring_exl3_lmcache_launcher as lmcache_launcher


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/sparkring_exl3_lmcache_launcher.py"
CONFIRMATION = "RUN-EXL3-CACHE-BOUNDARY-ALL-FOUR"
LAUNCHER_CONFIRMATION = "START-EXL3-LMCACHE-CS512-ALL-FOUR"
SCHEMA = "sparkring-exl3-cache-acceptance/v1"
EXIT_OK = 0
EXIT_FUNCTIONAL_FAIL = 2
EXIT_CONFIG_ERROR = 3

PREFIX_FRAGMENT = (
    "SparkRing cache acceptance uses a stable, unique prefix to distinguish "
    "cold prefill from server-resident reuse across controlled restarts. "
)


class ConfigError(ValueError):
    """The operator supplied an unsafe or incomplete gate configuration."""


class GateFailure(RuntimeError):
    """A live functional assertion failed."""


def launcher_argv(args: argparse.Namespace, command: str) -> list[str]:
    argv = [
        sys.executable,
        str(LAUNCHER),
        "--site",
        args.site,
        "--profile",
        args.profile,
        "--execute",
    ]
    if args.experimental_memory_profile is not None:
        argv += [
            "--experimental-memory-profile",
            args.experimental_memory_profile,
        ]
    if command in ("restart-engines", "restart-stack"):
        argv += ["--confirmation", LAUNCHER_CONFIRMATION]
    return argv + [command]


def json_objects(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    objects: list[Any] = []
    cursor = 0
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            break
        try:
            value, cursor = decoder.raw_decode(text, cursor)
        except json.JSONDecodeError:
            cursor += 1
            continue
        objects.append(value)
    return objects


def parse_launcher_status(text: str) -> dict[str, dict[str, Any]]:
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GateFailure(f"launcher status was not JSON: {exc}") from exc
    ranks = document.get("server_health") if isinstance(document, dict) else None
    if not isinstance(ranks, dict) or set(ranks) != {"0", "1", "2", "3"}:
        raise GateFailure("launcher status did not contain four LMCache ranks")
    observed: dict[str, dict[str, Any]] = {}
    for rank, result in sorted(ranks.items()):
        if not isinstance(result, dict) or result.get("exit_code") != 0:
            raise GateFailure(f"rank {rank} LMCache status command failed")
        candidates = [value for value in json_objects(str(result.get("stdout", ""))) if isinstance(value, dict)]
        status = next(
            (value for value in reversed(candidates) if "storage_manager" in value),
            None,
        )
        if not isinstance(status, dict) or status.get("is_healthy") is not True:
            raise GateFailure(f"rank {rank} LMCache server is not healthy")
        storage = status.get("storage_manager")
        l1 = storage.get("l1_manager") if isinstance(storage, dict) else None
        if not isinstance(l1, dict) or l1.get("is_healthy") is not True:
            raise GateFailure(f"rank {rank} LMCache L1 is not healthy")
        registered = status.get("registered_gpu_ids")
        if not isinstance(registered, list) or len(registered) != 1:
            raise GateFailure(f"rank {rank} has no unique registered GPU context")
        observed[rank] = {
            "objects": int(l1.get("total_object_count", -1)),
            "bytes": int(l1.get("memory_used_bytes", -1)),
            "write_locked": int(l1.get("write_locked_count", -1)),
            "read_locked": int(l1.get("read_locked_count", -1)),
            "temporary": int(l1.get("temporary_count", -1)),
            "registered_gpu_contexts": len(registered),
        }
        if any(
            observed[rank][key] != 0
            for key in ("write_locked", "read_locked", "temporary")
        ):
            raise GateFailure(f"rank {rank} has locked or temporary cache objects")
    return observed


def run_launcher(
    args: argparse.Namespace,
    executor: Any,
    command: str,
    *,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]] | None]:
    result = executor.run(launcher_argv(args, command), timeout=timeout)
    record = {
        "argv": result.argv,
        "exit_code": result.exit_code,
        "duration_seconds": result.duration_seconds,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    if result.exit_code != 0:
        raise GateFailure(
            f"launcher {command} exited {result.exit_code}: "
            f"{result.stderr.strip()[:400]}"
        )
    snapshot = parse_launcher_status(result.stdout) if command == "status" else None
    return record, snapshot


def require_api(args: argparse.Namespace, http: Any) -> None:
    base = args.base_url.rstrip("/")
    status, _ = http.get_json(f"{base}/health", timeout=args.timeout_seconds)
    if status != 200:
        raise GateFailure(f"/health returned {status}")
    status, models = http.get_json(
        f"{base}/v1/models", timeout=args.timeout_seconds
    )
    ids = {
        entry.get("id")
        for entry in (models.get("data", []) if isinstance(models, dict) else [])
        if isinstance(entry, dict)
    }
    if status != 200 or args.model not in ids:
        raise GateFailure(f"served model {args.model!r} is absent from {sorted(ids)!r}")


def probe_prompt(args: argparse.Namespace) -> str:
    prefix = PREFIX_FRAGMENT * args.prefix_repetitions
    return (
        f"Cache acceptance probe id: {args.probe_id}.\n{prefix}\n"
        "Return exactly one short sentence confirming that the prefix was read."
    )


def sample(args: argparse.Namespace, http: Any, label: str) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "prompt": probe_prompt(args),
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 20260807,
        "max_tokens": args.max_tokens,
    }
    result = http.stream_completion(
        f"{args.base_url.rstrip('/')}/v1/completions",
        payload,
        timeout=args.timeout_seconds,
    )
    if result.error or result.ttft_seconds is None or result.tokens <= 0:
        raise GateFailure(f"{label} request failed: {result.error or 'no tokens'}")
    return {
        "label": label,
        "ttft_seconds": result.ttft_seconds,
        "total_seconds": result.total_seconds,
        "completion_tokens": result.tokens,
        "token_count_source": result.token_count_source,
        "text_sha256": hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
    }


def require_ratio(
    samples: dict[str, dict[str, Any]],
    numerator: str,
    denominator: str,
    maximum: float,
    failures: list[str],
) -> None:
    ratio = samples[numerator]["ttft_seconds"] / samples[denominator]["ttft_seconds"]
    samples[numerator][f"ratio_vs_{denominator}"] = ratio
    if ratio > maximum:
        failures.append(
            f"{numerator} TTFT ratio {ratio:.3f} exceeds {maximum:.3f} vs {denominator}"
        )


def execute_gate(args: argparse.Namespace, executor: Any, http: Any) -> dict[str, Any]:
    require_api(args, http)
    commands: list[dict[str, Any]] = []
    snapshots: dict[str, dict[str, dict[str, Any]]] = {}
    samples: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    record, snapshots["initial"] = run_launcher(
        args, executor, "status", timeout=180
    )
    commands.append(record)
    samples["cold"] = sample(args, http, "cold")
    samples["warm"] = sample(args, http, "warm")
    record, snapshots["after_warm"] = run_launcher(
        args, executor, "status", timeout=180
    )
    commands.append(record)
    if any(value["objects"] <= 0 for value in snapshots["after_warm"].values()):
        failures.append("one or more LMCache ranks stored no objects after warm probe")
    require_ratio(samples, "warm", "cold", args.max_warm_ratio, failures)

    record, _ = run_launcher(
        args, executor, "restart-engines", timeout=args.restart_timeout_seconds
    )
    commands.append(record)
    record, snapshots["after_engine_restart"] = run_launcher(
        args, executor, "status", timeout=180
    )
    commands.append(record)
    samples["after_engine_restart"] = sample(
        args, http, "after_engine_restart"
    )
    require_ratio(
        samples,
        "after_engine_restart",
        "cold",
        args.max_engine_restart_ratio,
        failures,
    )
    if any(
        snapshots["after_engine_restart"][rank]["objects"] <= 0
        for rank in snapshots["after_engine_restart"]
    ):
        failures.append("LMCache objects did not survive the engine-only restart")

    record, _ = run_launcher(
        args, executor, "restart-stack", timeout=args.restart_timeout_seconds
    )
    commands.append(record)
    record, snapshots["after_server_restart"] = run_launcher(
        args, executor, "status", timeout=180
    )
    commands.append(record)
    if any(
        value["objects"] != 0
        for value in snapshots["after_server_restart"].values()
    ):
        failures.append(
            "LMCache L1 was not empty after server restart; persistence boundary is ambiguous"
        )
    samples["after_server_restart_cold"] = sample(
        args, http, "after_server_restart_cold"
    )
    samples["after_server_restart_warm"] = sample(
        args, http, "after_server_restart_warm"
    )
    require_ratio(
        samples,
        "after_server_restart_warm",
        "after_server_restart_cold",
        args.max_warm_ratio,
        failures,
    )
    record, snapshots["final"] = run_launcher(
        args, executor, "status", timeout=180
    )
    commands.append(record)
    if any(value["objects"] <= 0 for value in snapshots["final"].values()):
        failures.append("LMCache did not repopulate on every rank after server restart")

    hashes = {entry["text_sha256"] for entry in samples.values()}
    if len(hashes) != 1:
        failures.append("fixed-seed probe output changed across cache/restart phases")
    require_api(args, http)
    return {
        "schema": SCHEMA,
        "status": "pass" if not failures else "fail",
        "probe_id": args.probe_id,
        "model": args.model,
        "experimental_memory_profile": args.experimental_memory_profile,
        "prompt_sha256": hashlib.sha256(
            probe_prompt(args).encode("utf-8")
        ).hexdigest(),
        "samples": samples,
        "cache_snapshots": snapshots,
        "commands": commands,
        "thresholds": {
            "max_warm_ratio": args.max_warm_ratio,
            "max_engine_restart_ratio": args.max_engine_restart_ratio,
        },
        "failures": failures,
        "evidence_scope": (
            "LMCache server-resident reuse across engine restart and volatile "
            "L1 reset across LMCache-server restart; not NVMe durability"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--probe-id", required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prefix-repetitions", type=int, default=2048)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--restart-timeout-seconds", type=float, default=9000.0)
    parser.add_argument("--max-warm-ratio", type=float, default=0.75)
    parser.add_argument("--max-engine-restart-ratio", type=float, default=0.75)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument(
        "--experimental-memory-profile",
        choices=tuple(lmcache_launcher.EXPERIMENTAL_MEMORY_PROFILES),
        default=None,
        help="pass a named non-canonical memory preset through every launcher phase",
    )
    parser.add_argument("command", choices=("plan", "run"))
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for field in ("max_tokens", "prefix_repetitions"):
        if getattr(args, field) <= 0:
            raise ConfigError(f"--{field.replace('_', '-')} must be positive")
    for field in ("max_warm_ratio", "max_engine_restart_ratio"):
        value = getattr(args, field)
        if not 0 < value <= 1:
            raise ConfigError(f"--{field.replace('_', '-')} must be in (0, 1]")
    if args.command == "run" and args.execute and args.confirmation != CONFIRMATION:
        raise ConfigError(f"execute requires --confirmation {CONFIRMATION}")


def plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "sparkring-exl3-cache-acceptance-plan/v1",
        "mutates_remote": True,
        "execute_requested": args.execute,
        "probe_id": args.probe_id,
        "experimental_memory_profile": args.experimental_memory_profile,
        "prompt_sha256": hashlib.sha256(
            probe_prompt(args).encode("utf-8")
        ).hexdigest(),
        "phases": [
            "initial health and cache snapshot",
            "unique-prefix cold/warm probe",
            "label-guarded engine-only restart and reuse probe",
            "label-guarded engine plus LMCache-server restart",
            "volatile-L1 reset and repopulation probe",
            "post-run health/resource snapshot",
        ],
        "commands": {
            command: launcher_argv(args, command)
            for command in ("status", "restart-engines", "restart-stack")
        },
        "evidence_boundary": "server-resident LMCache L1; no durable L2 configured",
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    executor: Any | None = None,
    http: Any | None = None,
) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        validate_args(args)
        if args.command == "plan" or not args.execute:
            print(json.dumps(plan(args), indent=2, sort_keys=True))
            return EXIT_OK
        report = execute_gate(
            args,
            executor if executor is not None else SubprocessExecutor(),
            http if http is not None else UrllibHttpClient(),
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return EXIT_OK if report["status"] == "pass" else EXIT_FUNCTIONAL_FAIL
    except ConfigError as exc:
        print(f"exl3-cache-acceptance: CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except GateFailure as exc:
        print(
            json.dumps(
                {"schema": SCHEMA, "status": "fail", "failures": [str(exc)]},
                indent=2,
                sort_keys=True,
            )
        )
        return EXIT_FUNCTIONAL_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
