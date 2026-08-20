#!/usr/bin/env python3
"""Offline decision gate for external LMCache key-value reuse.

An external key-value tier is functional for a checkpoint only when a request
served by an engine that has been destroyed and recreated completes, reproduces
the store-phase output hash, is not attributable to the engine's own prefix
cache, and records a non-zero external hit count, with no transfer failures in
either log.

Two instrument properties make partial evidence misleading, and this gate
rejects both shapes explicitly:

- The external hit counter counts lookup matches, not completed transfers. A
  configuration that fails every transfer can report a hit count close to its
  query count while recomputing every token and never returning a response, so
  a non-zero hit count alongside an incomplete request is a failure.
- A replay inside a live engine process is served by that engine's own prefix
  cache. Such a replay is fast and byte-identical while the external counter
  reads zero, so the gate requires proof that the engine process was recreated
  and that native prefix caching was disabled.

The gate reads a collected evidence document and decides. It is fail-closed:
a missing or malformed required field is a configuration error, never a pass.

Each checkpoint declares whether the external tier is load-bearing for it.
``external_reuse_policy`` is ``required`` when serving depends on the tier and
``optional`` when serving proceeds without it. An optional tier that does not
qualify withholds reuse and nothing else: the verdict is still ``fail``, but
``serving_blocked`` is false and the process exits ``4`` rather than ``2``, so
a caller never reads a serving break out of an absent accelerator and never
suppresses a real one in order to tolerate it.

Safety class: OFFLINE. Reads a local evidence file and contacts nothing. The
``plan`` command prints the live collection procedure, whose own steps are
STOPS SERVING and require separate authorization.

Exit codes: ``0`` pass, ``2`` a required capability failed, ``3`` configuration
error, ``4`` an optional capability did not qualify.

Usage::

    python scripts/lmcache_external_reuse_gate.py plan
    python scripts/lmcache_external_reuse_gate.py evaluate --evidence PATH
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "sparkring-lmcache-external-reuse-gate/v1"
EVIDENCE_SCHEMA = "sparkring-lmcache-external-reuse-evidence/v1"
PLAN_SCHEMA = "sparkring-lmcache-external-reuse-plan/v1"

EXIT_OK = 0
EXIT_FAIL = 2
EXIT_CONFIG_ERROR = 3
# An optional capability that did not qualify is not a serving failure. It gets
# its own exit code so a caller never has to read a serving break out of it, and
# never has to suppress a real one to tolerate it.
EXIT_OPTIONAL_UNAVAILABLE = 4

# Whether serving this checkpoint depends on the external tier. "required" means
# a failed gate blocks serving; "optional" means serving proceeds without the
# tier and the failure withholds only the reuse capability.
POLICY_REQUIRED = "required"
POLICY_OPTIONAL = "optional"
POLICIES = (POLICY_REQUIRED, POLICY_OPTIONAL)

# Every pointer the verdict depends on. Absence is a configuration error rather
# than a failure, so that an incomplete collection cannot read as a refutation.
REQUIRED_POINTERS = (
    "/checkpoint",
    "/package_version",
    "/external_reuse_policy",
    "/store_phase/completed",
    "/store_phase/prompt_sha256",
    "/store_phase/output_sha256",
    "/teardown/engine_containers_removed",
    "/teardown/cache_servers_removed",
    "/teardown/memory_tier_cleared",
    "/teardown/filesystem_tier_retained",
    "/replay_phase/completed",
    "/replay_phase/prompt_sha256",
    "/replay_phase/output_sha256",
    "/replay_phase/engine_process_recreated",
    "/replay_phase/native_prefix_caching_enabled",
    "/replay_phase/native_prefix_cache_hits",
    "/replay_phase/external_prefix_cache_hits",
    "/logs/server_size_mismatch_count",
    "/logs/engine_retrieve_failed_count",
)


class ConfigError(Exception):
    """The evidence document cannot be read or is missing a required field."""


class Condition:
    """One named gate condition and the evidence that decides it."""

    def __init__(self, name: str, passed: bool, statement: str, detail: Any):
        self.name = name
        self.passed = passed
        self.statement = statement
        self.detail = detail

    def render(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "statement": self.statement,
            "detail": self.detail,
        }


def resolve(document: Any, pointer: str) -> Any:
    """Return the value at a JSON pointer, raising ConfigError when absent."""
    value = document
    for token in pointer.strip("/").split("/"):
        if not isinstance(value, dict) or token not in value:
            raise ConfigError(f"evidence is missing required field {pointer}")
        value = value[token]
    if value is None:
        raise ConfigError(f"evidence field {pointer} is null")
    return value


def _require_bool(document: Any, pointer: str) -> bool:
    value = resolve(document, pointer)
    if not isinstance(value, bool):
        raise ConfigError(f"evidence field {pointer} must be a boolean")
    return value


def _require_count(document: Any, pointer: str) -> int:
    value = resolve(document, pointer)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"evidence field {pointer} must be an integer")
    if value < 0:
        raise ConfigError(f"evidence field {pointer} must not be negative")
    return value


def _require_text(document: Any, pointer: str) -> str:
    value = resolve(document, pointer)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"evidence field {pointer} must be a non-empty string")
    return value


def load_evidence(path: Path) -> dict[str, Any]:
    """Load and shape-check an evidence document."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read evidence {path}: {exc}") from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"evidence {path} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ConfigError(f"evidence {path} must be a JSON object")
    if document.get("schema") != EVIDENCE_SCHEMA:
        raise ConfigError(
            f"evidence {path} declares schema {document.get('schema')!r}; "
            f"expected {EVIDENCE_SCHEMA!r}"
        )
    for pointer in REQUIRED_POINTERS:
        resolve(document, pointer)
    return document


def evaluate(document: dict[str, Any]) -> dict[str, Any]:
    """Decide the gate for one collected evidence document."""
    store_prompt = _require_text(document, "/store_phase/prompt_sha256")
    store_output = _require_text(document, "/store_phase/output_sha256")
    replay_prompt = _require_text(document, "/replay_phase/prompt_sha256")
    replay_output = _require_text(document, "/replay_phase/output_sha256")

    store_completed = _require_bool(document, "/store_phase/completed")
    replay_completed = _require_bool(document, "/replay_phase/completed")
    recreated = _require_bool(document, "/replay_phase/engine_process_recreated")
    native_enabled = _require_bool(
        document, "/replay_phase/native_prefix_caching_enabled"
    )
    native_hits = _require_count(document, "/replay_phase/native_prefix_cache_hits")
    external_hits = _require_count(
        document, "/replay_phase/external_prefix_cache_hits"
    )
    size_mismatch = _require_count(document, "/logs/server_size_mismatch_count")
    retrieve_failed = _require_count(document, "/logs/engine_retrieve_failed_count")

    teardown = {
        pointer.rsplit("/", 1)[-1]: _require_bool(document, pointer)
        for pointer in REQUIRED_POINTERS
        if pointer.startswith("/teardown/")
    }

    conditions = [
        Condition(
            "store_phase_completed",
            store_completed,
            "The store-phase request completed and produced a recorded hash.",
            {"completed": store_completed, "output_sha256": store_output},
        ),
        Condition(
            "deployment_destroyed_and_recreated",
            all(teardown.values()) and recreated,
            "Engine containers, cache servers, and the memory tier were removed, "
            "the filesystem tier was retained, and the engine process was recreated.",
            {"teardown": teardown, "engine_process_recreated": recreated},
        ),
        Condition(
            "replay_request_completed",
            replay_completed,
            "The replayed request completed rather than stalling in the queue.",
            {"completed": replay_completed},
        ),
        Condition(
            "replayed_prompt_is_identical",
            store_prompt == replay_prompt,
            "The replay used the identical prompt recorded in the store phase.",
            {"store_phase": store_prompt, "replay_phase": replay_prompt},
        ),
        Condition(
            "output_hash_matches_store_phase",
            store_output == replay_output,
            "The replay output hash matches the store-phase hash for that prompt.",
            {"store_phase": store_output, "replay_phase": replay_output},
        ),
        Condition(
            "native_prefix_cache_cannot_account_for_result",
            (not native_enabled) and native_hits == 0,
            "Native prefix caching was disabled and its hit counter read zero, so "
            "the engine's own cache cannot account for the result.",
            {
                "native_prefix_caching_enabled": native_enabled,
                "native_prefix_cache_hits": native_hits,
            },
        ),
        Condition(
            "external_tier_recorded_hits",
            external_hits > 0,
            "The external hit counter is non-zero.",
            {"external_prefix_cache_hits": external_hits},
        ),
        Condition(
            "no_transfer_size_mismatch",
            size_mismatch == 0,
            "The cache server log records no size mismatch.",
            {"server_size_mismatch_count": size_mismatch},
        ),
        Condition(
            "no_retrieve_failures",
            retrieve_failed == 0,
            "The engine log records no retrieve failure.",
            {"engine_retrieve_failed_count": retrieve_failed},
        ),
    ]

    # A non-zero external hit count alongside an incomplete request is the
    # counter's known failure shape, not a partial success.
    misleading_hits = external_hits > 0 and not replay_completed
    conditions.append(
        Condition(
            "hit_counter_not_misreporting_a_stall",
            not misleading_hits,
            "A non-zero external hit count is not being read as success for a "
            "request that never completed.",
            {
                "external_prefix_cache_hits": external_hits,
                "replay_completed": replay_completed,
            },
        )
    )

    policy = _require_text(document, "/external_reuse_policy")
    if policy not in POLICIES:
        raise ConfigError(
            f"evidence field /external_reuse_policy must be one of "
            f"{list(POLICIES)}; got {policy!r}"
        )

    failed = [condition.name for condition in conditions if not condition.passed]
    verdict = "fail" if failed else "pass"

    # Serving is blocked only when the tier is load-bearing for this checkpoint.
    # An optional tier that does not qualify withholds reuse and nothing else.
    serving_blocked = bool(failed) and policy == POLICY_REQUIRED

    return {
        "schema": SCHEMA,
        "checkpoint": _require_text(document, "/checkpoint"),
        "package_version": _require_text(document, "/package_version"),
        "external_reuse_policy": policy,
        "conditions": [condition.render() for condition in conditions],
        "failed_conditions": failed,
        "verdict": verdict,
        "serving_blocked": serving_blocked,
        "serving_disposition": (
            "Serving this checkpoint without the external tier is unaffected by "
            "this result and is the supported configuration."
            if policy == POLICY_OPTIONAL
            else "Serving this checkpoint depends on the external tier."
        ),
        "claim_on_pass": (
            "External key-value reuse is functional for this checkpoint on the "
            "measured deployment. This is not a persistence, performance, or "
            "acceptance result."
        ),
    }


def exit_code_for(report: dict[str, Any]) -> int:
    """Map a report to a process exit code, separating optional from blocking."""
    if report["verdict"] == "pass":
        return EXIT_OK
    if report["external_reuse_policy"] == POLICY_OPTIONAL:
        return EXIT_OPTIONAL_UNAVAILABLE
    return EXIT_FAIL


def build_plan() -> dict[str, Any]:
    """Return the live collection procedure for one checkpoint."""
    return {
        "schema": PLAN_SCHEMA,
        "purpose": (
            "Collect the evidence this gate evaluates for one checkpoint on one "
            "four-rank deployment."
        ),
        "preconditions": [
            {
                "statement": (
                    "The cache package is identical on every rank and passes "
                    "runtime/exl3/verify_lmcache_group_tracking.py."
                ),
                "reason": (
                    "A mixed deployment registers successfully on some ranks and "
                    "fails on others, which presents as a stalled request rather "
                    "than an error."
                ),
            },
            {
                "statement": "Each cache server runs inside its rank's engine container.",
                "reason": (
                    "A server in a separate container cannot import the engine's "
                    "device memory; registration fails with "
                    "cudaErrorInvalidResourceHandle."
                ),
            },
            {
                "statement": (
                    "The registration deadline exceeds the observed registration "
                    "time for the largest reservation in use."
                ),
                "reason": (
                    "Registration that outruns lmcache.mp.mq_timeout aborts before "
                    "the tier is usable."
                ),
            },
            {
                "statement": "The engine runs with --no-enable-prefix-caching.",
                "reason": (
                    "Without it the engine's own cache serves the replay and no "
                    "measurement attributes a result to the external tier."
                ),
            },
        ],
        "steps": [
            {
                "id": "record-registration",
                "safety_class": "READ-ONLY REMOTE",
                "statement": (
                    "Record the kernel group inventory the server logs at "
                    "registration, before configuring chunk size."
                ),
            },
            {
                "id": "store",
                "safety_class": "STOPS SERVING",
                "statement": (
                    "Serve a fixed prompt and record its prompt and output hashes."
                ),
            },
            {
                "id": "destroy",
                "safety_class": "STOPS SERVING",
                "statement": (
                    "Remove the engine containers, the cache servers, and the "
                    "memory tier, retaining only the filesystem tier."
                ),
            },
            {
                "id": "recreate",
                "safety_class": "STOPS SERVING",
                "statement": "Recreate the deployment from the same pinned inputs.",
            },
            {
                "id": "replay",
                "safety_class": "STOPS SERVING",
                "statement": (
                    "Replay the identical prompt and record completion, output "
                    "hash, both prefix-cache counters, and both log counts."
                ),
            },
        ],
        "counted_log_lines": {
            "server_size_mismatch_count": "Size mismatch",
            "engine_retrieve_failed_count": "LMCache retrieve failed",
        },
        "evidence_schema": EVIDENCE_SCHEMA,
        "required_fields": list(REQUIRED_POINTERS),
        "external_reuse_policy": {
            "values": list(POLICIES),
            "meaning": {
                POLICY_REQUIRED: (
                    "serving this checkpoint depends on the external tier; a "
                    "failed gate blocks serving and exits 2"
                ),
                POLICY_OPTIONAL: (
                    "serving this checkpoint proceeds without the external tier; "
                    "a failed gate withholds reuse only and exits 4"
                ),
            },
            "note": (
                "Declare the policy before collecting, not after reading the "
                "result, so an unqualified capability cannot be reclassified as "
                "optional to make a failure disappear."
            ),
        },
        "authorization": (
            "Every step above other than record-registration stops serving and "
            "requires explicit authorization for the named hosts."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "evaluate"))
    parser.add_argument(
        "--evidence",
        default=None,
        help="path to a collected evidence document (required by evaluate)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "plan":
        print(json.dumps(build_plan(), indent=2, sort_keys=True))
        return EXIT_OK

    if args.evidence is None:
        print(
            "lmcache-external-reuse: CONFIG ERROR: evaluate requires --evidence",
            file=sys.stderr,
        )
        return EXIT_CONFIG_ERROR

    try:
        report = evaluate(load_evidence(Path(args.evidence)))
    except ConfigError as exc:
        print(f"lmcache-external-reuse: CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code_for(report)


if __name__ == "__main__":
    raise SystemExit(main())
