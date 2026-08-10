#!/usr/bin/env python3
"""P3 sabotage matrix: damage, request, verdict — machine-readable.

Pass criteria per mode (goal G-CACHE P3):
  engine_survived  - API healthy after the request
  no_wrong_output  - request completes with the correct recall; every request
                     error is classified fail-closed and fails the mode
  self_healed      - the damaged rank's entry was removed
  no_partial       - retirement follows the connector's scheduler/worker contract
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from context_cache_gate import DEFAULT_BASE_URL, build_prompt, run_request  # noqa: E402

# Ring SSH targets, one per rank. Set SPARKRING_TARGETS to a comma-separated
# user@host list (rank 0 first), as the sibling runner scripts do; the default
# below is a placeholder list so the module stays importable without it.
_DEFAULT_TARGETS = "user@spark0,user@spark1,user@spark2,user@spark3"
HOSTS = dict(
    enumerate(
        target.strip()
        for target in os.environ.get(
            "SPARKRING_TARGETS", _DEFAULT_TARGETS
        ).split(",")
    )
)
# Path to cache_manifest.py (the fail-closed engine) as deployed on each host.
# Override with SPARKRING_ENGINE for real runs.
ENGINE = os.environ.get(
    "SPARKRING_ENGINE",
    "<engine-path>/persistent_context_cache/cache_manifest.py",
)
# Name of the running vLLM serving container on each host.
CONTAINER = os.environ.get("SPARKRING_CONTAINER", "<container>")
CACHE_ROOT = os.environ.get("SPARKRING_CACHE_ROOT", "<cache-root>")
VERIFY_SCRIPT = os.environ.get(
    "SPARKRING_VERIFY_SCRIPT", "/tmp/context_cache_verify_store.py"
)
CONFIRMATION = "CORRUPT-EXL3-SPARKCACHE-RANKS-1-2-3"
SABOTAGE_SCRIPT = "/tmp/context_cache_sabotage.py"
_SCRIPT_DIR = Path(__file__).resolve().parent
SABOTAGE_SHA256 = hashlib.sha256(
    (_SCRIPT_DIR / "context_cache_sabotage.py").read_bytes()
).hexdigest()
VERIFY_SHA256 = hashlib.sha256(
    (_SCRIPT_DIR / "context_cache_verify_store.py").read_bytes()
).hexdigest()
_EVENT_RE = re.compile(
    r"spark-context-cache-event/v1 event="
    r"(?P<event>worker_invalidated|scheduler_retired|store_committed) "
    r"rank=(?P<rank>[0-9]+) digest=(?P<digest>[0-9a-f]{64}) "
    r"request_id=(?P<request_id>\S+)"
)
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")
_CHILD_REQUEST_ID_SUFFIX = r"-0-[0-9a-f]{8}"
MATRIX_MODES = (
    ("bitflip", 2),
    ("truncate", 1),
    ("fingerprint", 3),
)
CACHE_SALT_SCHEMA = "sparkring-context-cache-p3-apc-isolation/v1"
CACHE_SALT_ROLES = (
    "target-seed",
    "sentinel-seed",
    "pre-sabotage-probe",
    "corruption-trigger",
    "recovery",
)
_ACTIVE_RUN: tuple[dict, Path, int, str, int] | None = None


def derive_cache_salt(run_nonce: str, mode: str, request_role: str) -> str:
    """Derive an auditable APC namespace without changing prompt tokens."""

    if not run_nonce:
        raise ValueError("run nonce must be non-empty")
    if mode not in {candidate for candidate, _rank in MATRIX_MODES}:
        raise ValueError(f"unsupported matrix mode {mode!r}")
    if request_role not in CACHE_SALT_ROLES:
        raise ValueError(f"unsupported cache-salt request role {request_role!r}")
    material = "\0".join(
        (CACHE_SALT_SCHEMA, run_nonce, mode, request_role)
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def cache_salt_contract() -> dict:
    """Describe the APC-isolation namespace without exposing run salt values."""

    return {
        "schema": CACHE_SALT_SCHEMA,
        "request_field": "cache_salt",
        "derivation": (
            "sha256(UTF-8 NUL-joined schema, private run nonce, mode, request role)"
        ),
        "request_roles": list(CACHE_SALT_ROLES),
        "prompt_tokens_unchanged": True,
        "sparkcache_digest_unchanged": True,
        "raw_values_recorded": False,
        "receipt": "sha256 of each transmitted cache_salt, keyed by mode and role",
    }


def cache_salts_for_mode(run_nonce: str, mode: str) -> dict[str, str]:
    """Return one deterministic, role-specific APC namespace per request."""

    return {
        role: derive_cache_salt(run_nonce, mode, role)
        for role in CACHE_SALT_ROLES
    }


def cache_salt_receipts(salts: dict[str, str]) -> dict[str, dict[str, str]]:
    """Hash transmitted salt values again so evidence never contains them raw."""

    if set(salts) != set(CACHE_SALT_ROLES):
        raise ValueError("cache salts must cover every request role exactly once")
    return {
        role: {
            "sha256": hashlib.sha256(salts[role].encode("utf-8")).hexdigest()
        }
        for role in CACHE_SALT_ROLES
    }


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number {value!r} is unsupported")


def strict_json_loads(value: str, label: str) -> object:
    try:
        return json.loads(
            value,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} returned invalid JSON: {error}") from error


def validate_rank_targets(hosts: dict[int, str]) -> None:
    """Require one distinct, explicit SSH authority for each DCP4 rank."""

    if set(hosts) != {0, 1, 2, 3}:
        raise ValueError("SPARKRING_TARGETS must define exactly ranks 0,1,2,3")
    targets = [hosts[rank] for rank in range(4)]
    if any(not isinstance(target, str) or not target.strip() for target in targets):
        raise ValueError("SPARKRING_TARGETS contains an empty rank target")
    if len(set(targets)) != 4:
        raise ValueError("SPARKRING_TARGETS must use four distinct rank targets")


def _encoded_report(report: dict) -> bytes:
    return (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _fsync_parent_directory(path: Path) -> bool:
    """Flush a directory entry where the controller OS exposes that primitive."""

    if os.name == "nt":
        return False
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path.parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def create_evidence_exclusive(path: Path, report: dict) -> None:
    """Reserve the evidence name before any remote work can mutate the run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(_encoded_report(report))
        output.flush()
        os.fsync(output.fileno())
    _fsync_parent_directory(path)


def checkpoint_evidence_atomic(path: Path, report: dict) -> None:
    """Replace a run-owned report atomically, retaining the prior checkpoint."""

    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(_encoded_report(report))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_parent_directory(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def record_mode_checkpoint(
    report: dict,
    entry: dict,
    output: Path,
    mode_index: int,
) -> bool:
    """Atomically record one mode and quarantine the matrix on any failure."""

    entry.setdefault("passed", False)
    report["modes"].append(entry)
    report["completed_mode_count"] = len(report["modes"])
    passed = entry.get("passed") is True
    if passed:
        report["execution_state"] = (
            "completed" if mode_index == len(MATRIX_MODES) - 1 else "running"
        )
        report["remaining_modes"] = [
            mode for mode, _rank in MATRIX_MODES[mode_index + 1 :]
        ]
        report["passed"] = report["execution_state"] == "completed"
    else:
        report["execution_state"] = "quarantined_after_failure"
        report["sabotage_halted"] = True
        report["terminal_failure"] = {
            "mode": entry.get("mode"),
            "rank": entry.get("rank"),
            "error": entry.get("error", "mode-gates-failed"),
        }
        report["remaining_modes"] = [
            mode for mode, _rank in MATRIX_MODES[mode_index + 1 :]
        ]
        report["passed"] = False
    checkpoint_evidence_atomic(output, report)
    return passed


def record_mutation_checkpoint(
    report: dict,
    output: Path,
    *,
    phase: str,
    mode: str,
    rank: int,
    digest: str,
    sabotage_applied: bool,
    recovery_required: bool,
) -> None:
    """Persist the exact actionable mutation state around irreversible sabotage."""

    report["active_mutation"] = {
        "phase": phase,
        "mode": mode,
        "rank": rank,
        "digest": digest,
        "sabotage_applied": sabotage_applied,
        "recovery_required": recovery_required,
    }
    checkpoint_evidence_atomic(output, report)


def ssh(rank: int, command: str) -> str:
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", HOSTS[rank], command],
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"rank {rank} SSH command failed rc={result.returncode}: "
            f"{result.stderr.strip()[-400:]}"
        )
    return result.stdout.strip()


def manifest_counts(cache_root: str) -> dict[int, int]:
    counts = {}
    manifests = f"{cache_root.rstrip('/')}/manifests"
    for rank in HOSTS:
        out = ssh(
            rank,
            "find "
            f"{shlex.quote(manifests)} -mindepth 2 -maxdepth 2 "
            "-type f -name '*.json' 2>/dev/null | wc -l",
        )
        counts[rank] = int(out or 0)
    return counts


def verify_stores(cache_root: str, engine: str, verify_script: str) -> dict[int, dict]:
    reports = {}
    for rank in HOSTS:
        output = ssh(
            rank,
            shlex.join(
                [
                    "python3",
                    "-B",
                    verify_script,
                    "--store",
                    cache_root,
                    "--engine",
                    engine,
                ]
            ),
        )
        try:
            parsed = strict_json_loads(output, f"rank {rank} store verifier")
            if not isinstance(parsed, dict):
                raise ValueError("store verifier report must be a JSON object")
            reports[rank] = parsed
        except ValueError:
            reports[rank] = {"error": "invalid-verifier-json", "raw": output[-200:]}
    return reports


def attest_remote_tools(verify_script: str) -> dict[int, dict[str, str]]:
    results = {}
    for rank in HOSTS:
        command = (
            f"test \"$(sha256sum {shlex.quote(SABOTAGE_SCRIPT)} | awk '{{print $1}}')\" = {SABOTAGE_SHA256}"
            f" && test \"$(sha256sum {shlex.quote(verify_script)} | awk '{{print $1}}')\" = {VERIFY_SHA256}"
        )
        output = ssh(rank, command)
        results[rank] = {"stdout": output, "status": "attested"}
    return results


def store_report_healthy(report: dict) -> bool:
    entries = report.get("entries")
    return (
        isinstance(entries, list)
        and bool(entries)
        and all(
            entry.get("lookup") == "hit" and entry.get("restore") == "ok"
            for entry in entries
        )
    )


def digest_entry_healthy(report: dict, digest: str) -> bool:
    entries = report.get("entries")
    if not isinstance(entries, list):
        return False
    matches = [entry for entry in entries if entry.get("digest") == digest]
    return (
        len(matches) == 1
        and matches[0].get("lookup") == "hit"
        and matches[0].get("restore") == "ok"
    )


def digest_absent(report: dict, digest: str) -> bool:
    entries = report.get("entries")
    return isinstance(entries, list) and all(
        entry.get("digest") != digest for entry in entries
    )


def digest_corrupt_present(report: dict, digest: str) -> bool:
    entries = report.get("entries")
    if not isinstance(entries, list):
        return False
    matches = [entry for entry in entries if entry.get("digest") == digest]
    return len(matches) == 1 and not (
        matches[0].get("lookup") == "hit"
        and matches[0].get("restore") == "ok"
    )


def unrelated_entries_unchanged(before: dict, after: dict, digest: str) -> bool:
    """Prove every unrelated entry retained the same verified payload metadata.

    Store paths are deliberately excluded because the verifier reports the
    host-local root separately.  The entry records themselves contain the
    identity, shard, committed-token, position, and record-size evidence that
    must remain byte-for-byte equal across withdrawal.
    """
    before_entries = before.get("entries")
    after_entries = after.get("entries")
    if not isinstance(before_entries, list) or not isinstance(after_entries, list):
        return False
    before_other = [entry for entry in before_entries if entry.get("digest") != digest]
    after_other = [entry for entry in after_entries if entry.get("digest") != digest]
    return (
        all(
            entry.get("lookup") == "hit" and entry.get("restore") == "ok"
            for entry in before_other + after_other
        )
        and before_other == after_other
    )


def retirement_contract_met(
    reports: dict[int, dict], damaged_rank: int, digest: str
) -> bool:
    """Match SparkCache's documented cluster retirement semantics.

    A worker load failure retires the damaged worker's copy and the scheduler's
    rank-0 admission.  Non-scheduler workers that restored successfully retain
    their copies until the subsequent clean republish converges all ranks.
    """
    if damaged_rank == 0 or damaged_rank not in HOSTS:
        return False
    return all(
        digest_absent(reports.get(rank, {}), digest)
        if rank in {0, damaged_rank}
        else digest_entry_healthy(reports.get(rank, {}), digest)
        for rank in HOSTS
    )


def digest_healthy_on_all_ranks(reports: dict[int, dict], digest: str) -> bool:
    return all(digest_entry_healthy(reports.get(rank, {}), digest) for rank in HOSTS)


def common_new_healthy_digest(
    before: dict[int, dict], after: dict[int, dict], excluded: set[str] | None = None
) -> str | None:
    excluded = excluded or set()
    old = {
        entry.get("digest")
        for report in before.values()
        for entry in report.get("entries", [])
        if isinstance(entry, dict) and isinstance(entry.get("digest"), str)
    }
    common: set[str] | None = None
    for rank in HOSTS:
        entries = after.get(rank, {}).get("entries")
        if not isinstance(entries, list):
            return None
        healthy = {
            entry.get("digest")
            for entry in entries
            if isinstance(entry, dict)
            and entry.get("lookup") == "hit"
            and entry.get("restore") == "ok"
            and isinstance(entry.get("digest"), str)
        }
        common = healthy if common is None else common & healthy
    candidates = {
        digest
        for digest in (common or set()) - old - excluded
        if all(digest_entry_healthy(after.get(rank, {}), digest) for rank in HOSTS)
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def parse_connector_events(logs: dict[int, str]) -> list[dict]:
    events = []
    for log_rank, log in logs.items():
        for sequence, line in enumerate(log.splitlines()):
            match = _EVENT_RE.search(line)
            if match is None:
                continue
            event = match.groupdict()
            event["rank"] = int(event["rank"])
            event["log_rank"] = log_rank
            event["sequence"] = sequence
            events.append(event)
    return events


def resolve_connector_request_id(
    events: list[dict],
    public_request_id: str | None,
    expected_ranks: set[int],
    event_names: set[str],
) -> dict:
    """Strictly bind a public SSE ID to one connector request-ID envelope.

    Connectors may retain the public ID verbatim or append the documented vLLM
    child suffix ``-0-<eight lowercase hex characters>``. Every expected rank
    must use one shared identity. Ambiguous children, malformed suffixes,
    prefix collisions, and rank/log-rank disagreement fail closed.
    """

    base = {
        "passed": False,
        "public_request_id": public_request_id,
        "resolved_request_id": None,
        "internal_child_request_id": None,
        "expected_ranks": sorted(expected_ranks),
        "child_suffix_grammar": _CHILD_REQUEST_ID_SUFFIX,
    }
    if (
        not isinstance(public_request_id, str)
        or _REQUEST_ID_RE.fullmatch(public_request_id) is None
        or not expected_ranks
        or not event_names
    ):
        return {**base, "reason": "missing-or-malformed-public-request-id"}

    child_pattern = re.compile(
        rf"{re.escape(public_request_id)}{_CHILD_REQUEST_ID_SUFFIX}"
    )
    candidates = []
    prefix_collisions = []
    for event in events:
        if event.get("event") not in event_names:
            continue
        event_request_id = event.get("request_id")
        if not isinstance(event_request_id, str):
            continue
        if event_request_id == public_request_id or child_pattern.fullmatch(
            event_request_id
        ):
            candidates.append(event)
        elif event_request_id.startswith(public_request_id):
            prefix_collisions.append(event)

    if prefix_collisions:
        return {
            **base,
            "reason": "malformed-child-request-id-or-prefix-collision",
            "malformed_events": prefix_collisions,
        }
    spoofed = [
        event
        for event in candidates
        if event.get("rank") != event.get("log_rank")
    ]
    if spoofed:
        return {
            **base,
            "reason": "request-rank-log-mismatch",
            "malformed_events": spoofed,
        }
    identities = {event["request_id"] for event in candidates}
    if len(identities) != 1:
        return {
            **base,
            "reason": (
                "no-request-id-candidate"
                if not identities
                else "multiple-request-id-candidates"
            ),
            "candidate_request_ids": sorted(identities),
        }
    resolved_request_id = next(iter(identities))
    missing_ranks = [
        rank
        for rank in sorted(expected_ranks)
        if not any(
            event.get("rank") == event.get("log_rank") == rank
            and event.get("request_id") == resolved_request_id
            for event in candidates
        )
    ]
    if missing_ranks:
        return {
            **base,
            "reason": "request-id-not-shared-by-all-expected-ranks",
            "candidate_request_ids": [resolved_request_id],
            "missing_ranks": missing_ranks,
        }
    return {
        **base,
        "passed": True,
        "reason": None,
        "resolved_request_id": resolved_request_id,
        "internal_child_request_id": (
            None
            if resolved_request_id == public_request_id
            else resolved_request_id
        ),
        "resolution": (
            "exact-public-id"
            if resolved_request_id == public_request_id
            else "shared-internal-child-id"
        ),
    }


def committed_request_binding(
    events: list[dict], request_id: str | None
) -> dict:
    """Resolve one request identity and its one all-rank committed digest."""

    binding = resolve_connector_request_id(
        events,
        request_id,
        set(HOSTS),
        {"store_committed"},
    )
    if not binding["passed"]:
        return {**binding, "digest": None}
    resolved_request_id = binding["resolved_request_id"]
    per_rank: dict[int, list[str]] = {}
    for rank in HOSTS:
        per_rank[rank] = [
            event["digest"]
            for event in events
            if event["event"] == "store_committed"
            and event["rank"] == event["log_rank"] == rank
            and event["request_id"] == resolved_request_id
        ]
        if len(per_rank[rank]) != 1:
            return {
                **binding,
                "passed": False,
                "reason": "request-commit-events-not-unique",
                "digest": None,
            }
    digests = {rank_digests[0] for rank_digests in per_rank.values()}
    if len(digests) != 1:
        return {
            **binding,
            "passed": False,
            "reason": "request-commit-digests-disagree",
            "digest": None,
        }
    return {**binding, "digest": next(iter(digests))}


def committed_digest_for_request(events: list[dict], request_id: str | None) -> str | None:
    """Return the one strictly request-bound digest committed on every rank."""

    binding = committed_request_binding(events, request_id)
    return binding["digest"] if binding["passed"] else None


def read_connector_events_since(
    container_pattern: str, since: str
) -> tuple[dict[int, str], list[dict]]:
    logs = {}
    for rank in HOSTS:
        container = rank_container(container_pattern, rank)
        logs[rank] = ssh(
            rank,
            f"docker logs --timestamps --since {shlex.quote(since)} "
            f"{shlex.quote(container)} 2>&1 | "
            "grep -a 'spark-context-cache-event/v1' | tail -200",
        )
    return logs, parse_connector_events(logs)


def retirement_event_proof(
    events: list[dict], damaged_rank: int, digest: str,
    expected_request_id: str | None,
) -> dict:
    if (
        not isinstance(expected_request_id, str)
        or _REQUEST_ID_RE.fullmatch(expected_request_id) is None
    ):
        return {
            "passed": False,
            "request_ids": [],
            "reason": "missing-or-malformed-trigger-request-id",
        }
    binding = resolve_connector_request_id(
        events,
        expected_request_id,
        {0, damaged_rank},
        {"worker_invalidated", "scheduler_retired", "store_committed"},
    )
    if not binding["passed"]:
        reason = binding["reason"]
        if reason == "request-rank-log-mismatch":
            reason = "trigger-request-rank-log-mismatch"
        else:
            reason = "trigger-request-id-resolution-failed"
        return {
            "passed": False,
            "request_ids": [expected_request_id],
            "reason": reason,
            "public_request_id": expected_request_id,
            "resolved_request_id": binding.get("resolved_request_id"),
            "internal_child_request_id": binding.get(
                "internal_child_request_id"
            ),
            "request_id_binding": binding,
            **(
                {"malformed_events": binding["malformed_events"]}
                if "malformed_events" in binding
                else {}
            ),
        }
    request_id = binding["resolved_request_id"]
    worker = [
        event
        for event in events
        if event["event"] == "worker_invalidated"
        and event["rank"] == event["log_rank"] == damaged_rank
        and event["digest"] == digest
        and event["request_id"] == request_id
    ]
    scheduler = [
        event
        for event in events
        if event["event"] == "scheduler_retired"
        and event["rank"] == event["log_rank"] == 0
        and event["digest"] == digest
        and event["request_id"] == request_id
    ]
    if len(worker) != 1 or len(scheduler) != 1:
        return {
            "passed": False,
            "request_ids": [request_id],
            "reason": "trigger-request-retirement-events-not-unique",
            "public_request_id": expected_request_id,
            "resolved_request_id": request_id,
            "internal_child_request_id": binding["internal_child_request_id"],
            "request_id_binding": binding,
        }

    def invalidation_then_commit(rank: int, invalidation_event: str) -> bool:
        invalidations = [
            event
            for event in events
            if event["log_rank"] == rank
            and event["rank"] == rank
            and event["event"] == invalidation_event
            and event["digest"] == digest
            and event["request_id"] == request_id
        ]
        commits = [
            event
            for event in events
            if event["log_rank"] == rank
            and event["rank"] == rank
            and event["event"] == "store_committed"
            and event["digest"] == digest
            and event["request_id"] == request_id
        ]
        return len(invalidations) == 1 and len(commits) == 1 and (
            invalidations[0]["sequence"] < commits[0]["sequence"]
        )

    rank0_recovered = invalidation_then_commit(0, "scheduler_retired")
    damaged_recovered = invalidation_then_commit(
        damaged_rank, "worker_invalidated"
    )
    return {
        "passed": rank0_recovered and damaged_recovered,
        "request_ids": [request_id],
        "public_request_id": expected_request_id,
        "resolved_request_id": request_id,
        "internal_child_request_id": binding["internal_child_request_id"],
        "request_id_binding": binding,
        "rank0_retired_then_committed": rank0_recovered,
        "damaged_worker_invalidated_then_committed": damaged_recovered,
    }


def parse_sabotage_report(output: str, expected_mode: str) -> dict:
    try:
        report = strict_json_loads(output, "sabotage helper")
    except ValueError as error:
        raise RuntimeError("sabotage returned invalid JSON") from error
    if not isinstance(report, dict) or report.get("mode") != expected_mode:
        raise RuntimeError("sabotage JSON has the wrong mode")
    expected_fields = {
        "mode",
        "digest",
        "manifest",
        "damaged",
        "post_damage_lookup",
        "post_damage_probe",
    }
    if expected_mode in {"bitflip", "truncate"}:
        expected_fields.add("chunk")
    if set(report) != expected_fields:
        raise RuntimeError("sabotage JSON fields are unsupported")
    digest = report.get("digest")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RuntimeError("sabotage JSON has no exact context digest")
    if not isinstance(report.get("damaged"), str) or not report["damaged"]:
        raise RuntimeError("sabotage JSON does not prove damage")
    return report


def rank_container(pattern: str, rank: int) -> str:
    if "{rank}" in pattern:
        return pattern.format(rank=rank)
    return pattern


def mode_pass(entry: dict) -> bool:
    return bool(
        entry.get("engine_survived")
        and entry.get("no_wrong_output")
        and entry.get("self_healed")
        and entry.get("others_intact")
        and entry.get("recovery_ok")
        and entry.get("request_error") is None
    )


def healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=8) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def wait_healthy(base_url: str, seconds: int = 120) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if healthy(base_url):
            return True
        time.sleep(5)
    return False


def fire(
    base_url: str,
    words: int,
    seed: int,
    model: str,
    run_nonce: str,
    *,
    cache_salt: str | None = None,
    max_tokens: int = 48,
) -> dict:
    # Put the nonce before the long body so it is inside the aligned cached
    # prefix, not in the trailing partial chunk omitted from the digest.
    prompt = f"SparkRing P3 run nonce: {run_nonce}\n" + build_prompt(words, seed)
    try:
        if cache_salt is None:
            result = run_request(base_url, prompt, max_tokens, model)
        else:
            result = run_request(
                base_url,
                prompt,
                max_tokens,
                model,
                cache_salt=cache_salt,
            )
        result["error"] = None
        return result
    except Exception as error:  # noqa: BLE001
        return {"error": repr(error), "completion": "", "ttft_seconds": None}


def _main() -> int:
    global _ACTIVE_RUN

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--words", type=int, default=24000)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--model", default=os.environ.get("SPARKRING_MODEL", "glm-5.2"))
    parser.add_argument("--cache-root", default=CACHE_ROOT)
    parser.add_argument("--engine", default=ENGINE)
    parser.add_argument("--container-pattern", default=CONTAINER)
    parser.add_argument("--verify-script", default=VERIFY_SCRIPT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.cache_root.startswith("<") or not args.cache_root.startswith("/"):
        parser.error("--cache-root must be the absolute deployed cache root")
    if args.engine.startswith("<") or not args.engine.startswith("/"):
        parser.error("--engine must be the absolute deployed cache_manifest.py path")
    if args.container_pattern.startswith("<"):
        parser.error("--container-pattern must name the deployed container(s)")
    try:
        validate_rank_targets(HOSTS)
    except ValueError as error:
        parser.error(str(error))

    plan = {
        "schema": "sparkring-context-cache-corruption-plan/v1",
        "sensitivity": "private-do-not-publish",
        "dry_run": not args.execute,
        "mutates_remote": True,
        "confirmation_required": CONFIRMATION,
        "max_tokens": args.max_tokens,
        "cache_salt_contract": cache_salt_contract(),
        "cache_root": args.cache_root,
        "engine": args.engine,
        "container_pattern": args.container_pattern,
        "verify_script": args.verify_script,
        "sabotage_script": SABOTAGE_SCRIPT,
        "sabotage_sha256": SABOTAGE_SHA256,
        "verify_sha256": VERIFY_SHA256,
        "target_binding": (
            "each mode uses a fresh prompt nonce and binds the SSE request ID to "
            "one common all-rank store_committed digest; sabotage receives that "
            "exact verified digest"
        ),
        "sentinel_binding": (
            "each mode creates one distinct all-rank healthy sentinel that must "
            "remain byte-for-byte unchanged"
        ),
        "withdrawal_proof": (
            "full-digest/request-ID worker_invalidated and scheduler_retired events "
            "must precede same-request store_committed events"
        ),
        "rank_targets": {str(rank): HOSTS[rank] for rank in range(4)},
        "modes": [
            {"mode": mode, "rank": rank} for mode, rank in MATRIX_MODES
        ],
        "warning": "sudo sabotage irreversibly modifies selected cache objects/manifests",
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.confirmation != CONFIRMATION:
        parser.error(f"--execute requires --confirmation {CONFIRMATION}")

    report = {
        "schema": "sparkring-context-cache-corruption-evidence/v1",
        "execution_state": "initializing",
        "sabotage_halted": False,
        "passed": False,
        "evidence_scope": {
            "classification": "private-raw-live-corruption-evidence",
            "publishable": False,
            "contains_private_rank_targets_paths_and_logs": True,
            "power_loss_durability": (
                "file-fsync-and-atomic-replace; parent-directory-fsync only where supported"
            ),
        },
        "active_mutation": None,
        "rank_targets": {str(rank): HOSTS[rank] for rank in range(4)},
        "planned_modes": [
            {"mode": mode, "rank": rank} for mode, rank in MATRIX_MODES
        ],
        "remaining_modes": [mode for mode, _rank in MATRIX_MODES],
        "completed_mode_count": 0,
        "seed_policy": (
            "target=base+mode_index*10000; sentinel=target+1; fresh run nonce "
            "is included inside each aligned cached prefix"
        ),
        "max_tokens": args.max_tokens,
        "cache_salt_contract": cache_salt_contract(),
        "cache_salt_receipts": {},
        "run_nonce": secrets.token_hex(16),
        "modes": [],
    }
    try:
        create_evidence_exclusive(args.output, report)
    except FileExistsError:
        parser.error(f"refusing to overwrite existing evidence: {args.output}")
    _ACTIVE_RUN = (report, args.output, -1, "setup", -1)
    try:
        report["remote_tool_attestation"] = attest_remote_tools(args.verify_script)
    except Exception as error:  # noqa: BLE001
        report["execution_state"] = "quarantined_after_setup_failure"
        report["sabotage_halted"] = True
        report["setup_failure"] = {
            "error": "remote-tool-attestation-failed",
            "detail": repr(error),
        }
        checkpoint_evidence_atomic(args.output, report)
        print("P3 matrix: FAIL (remote tool attestation)", flush=True)
        return 2
    report["execution_state"] = "running"
    checkpoint_evidence_atomic(args.output, report)

    for mode_index, (mode, rank) in enumerate(MATRIX_MODES):
        _ACTIVE_RUN = (report, args.output, mode_index, mode, rank)
        print(f"=== {mode} on rank {rank} ===", flush=True)
        if not wait_healthy(args.base_url):
            record_mode_checkpoint(
                report,
                {"mode": mode, "rank": rank, "error": "api-not-healthy"},
                args.output,
                mode_index,
            )
            break

        target_seed = args.seed + mode_index * 10000
        sentinel_seed = target_seed + 1
        target_nonce = f"{report['run_nonce']}-{mode}-target"
        sentinel_nonce = f"{report['run_nonce']}-{mode}-sentinel"
        mode_cache_salts = cache_salts_for_mode(report["run_nonce"], mode)
        mode_cache_salt_receipts = cache_salt_receipts(mode_cache_salts)
        report["cache_salt_receipts"][mode] = mode_cache_salt_receipts
        checkpoint_evidence_atomic(args.output, report)
        initial_verification = verify_stores(
            args.cache_root, args.engine, args.verify_script
        )
        target_event_start = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z")
        seed_result = fire(
            args.base_url,
            args.words,
            target_seed,
            args.model,
            target_nonce,
            cache_salt=mode_cache_salts["target-seed"],
            max_tokens=args.max_tokens,
        )
        target_digest = None
        target_verification = initial_verification
        target_seed_logs: dict[int, str] = {}
        target_seed_events: list[dict] = []
        target_request_binding: dict = {
            "passed": False,
            "public_request_id": seed_result.get("request_id"),
            "resolved_request_id": None,
            "internal_child_request_id": None,
        }
        for _ in range(12):
            time.sleep(2)
            target_verification = verify_stores(
                args.cache_root, args.engine, args.verify_script
            )
            target_seed_logs, target_seed_events = read_connector_events_since(
                args.container_pattern, target_event_start
            )
            target_request_binding = committed_request_binding(
                target_seed_events, seed_result.get("request_id")
            )
            candidate = (
                target_request_binding["digest"]
                if target_request_binding["passed"]
                else None
            )
            if candidate is not None and digest_healthy_on_all_ranks(
                target_verification, candidate
            ):
                target_digest = candidate
                break
        if seed_result.get("error") is not None or target_digest is None:
            record_mode_checkpoint(
                report,
                {
                    "mode": mode,
                    "rank": rank,
                    "error": "target-request-has-no-request-bound-shared-commit",
                    "seed": target_seed,
                    "nonce": target_nonce,
                    "request_id": seed_result.get("request_id"),
                    "request_id_binding": target_request_binding,
                    "request_error": seed_result.get("error"),
                    "before": initial_verification,
                    "after": target_verification,
                    "connector_events": target_seed_events,
                    "rank_logs": target_seed_logs,
                },
                args.output,
                mode_index,
            )
            break

        sentinel_event_start = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z")
        sentinel_result = fire(
            args.base_url,
            args.words,
            sentinel_seed,
            args.model,
            sentinel_nonce,
            cache_salt=mode_cache_salts["sentinel-seed"],
            max_tokens=args.max_tokens,
        )
        sentinel_digest = None
        before_verification = target_verification
        sentinel_seed_logs: dict[int, str] = {}
        sentinel_seed_events: list[dict] = []
        sentinel_request_binding: dict = {
            "passed": False,
            "public_request_id": sentinel_result.get("request_id"),
            "resolved_request_id": None,
            "internal_child_request_id": None,
        }
        for _ in range(12):
            time.sleep(2)
            before_verification = verify_stores(
                args.cache_root, args.engine, args.verify_script
            )
            sentinel_seed_logs, sentinel_seed_events = read_connector_events_since(
                args.container_pattern, sentinel_event_start
            )
            sentinel_request_binding = committed_request_binding(
                sentinel_seed_events, sentinel_result.get("request_id")
            )
            candidate = (
                sentinel_request_binding["digest"]
                if sentinel_request_binding["passed"]
                else None
            )
            if (
                candidate is not None
                and candidate != target_digest
                and digest_healthy_on_all_ranks(before_verification, candidate)
            ):
                sentinel_digest = candidate
                break
        if (
            sentinel_result.get("error") is not None
            or sentinel_digest is None
            or sentinel_digest == target_digest
        ):
            record_mode_checkpoint(
                report,
                {
                    "mode": mode,
                    "rank": rank,
                    "error": "sentinel-request-has-no-distinct-request-bound-shared-commit",
                    "seed": sentinel_seed,
                    "nonce": sentinel_nonce,
                    "request_id": sentinel_result.get("request_id"),
                    "request_id_binding": sentinel_request_binding,
                    "request_error": sentinel_result.get("error"),
                    "before": target_verification,
                    "after": before_verification,
                    "connector_events": sentinel_seed_events,
                    "rank_logs": sentinel_seed_logs,
                },
                args.output,
                mode_index,
            )
            break

        if not all(
            store_report_healthy(before_verification.get(r, {}))
            and digest_entry_healthy(before_verification.get(r, {}), target_digest)
            and digest_entry_healthy(before_verification.get(r, {}), sentinel_digest)
            for r in HOSTS
        ):
            record_mode_checkpoint(
                report,
                {
                    "mode": mode,
                    "rank": rank,
                    "error": "pre-sabotage-store-verification-failed",
                    "verification": before_verification,
                },
                args.output,
                mode_index,
            )
            break
        # and prove it actually restores before we damage it
        probe = fire(
            args.base_url,
            args.words,
            target_seed,
            args.model,
            target_nonce,
            cache_salt=mode_cache_salts["pre-sabotage-probe"],
            max_tokens=args.max_tokens,
        )
        if (
            seed_result is None
            or seed_result.get("error") is not None
            or probe.get("error") is not None
            or not seed_result.get("completion_token_ids")
            or seed_result.get("completion_token_ids")
            != probe.get("completion_token_ids")
        ):
            record_mode_checkpoint(
                report,
                {
                    "mode": mode,
                    "rank": rank,
                    "error": "pre-sabotage-exact-token-probe-failed",
                    "seed_error": None if seed_result is None else seed_result.get("error"),
                    "probe_error": probe.get("error"),
                },
                args.output,
                mode_index,
            )
            break
        seeded_restore_ttft = probe.get("ttft_seconds")
        record_mutation_checkpoint(
            report,
            args.output,
            phase="sabotage-command-armed",
            mode=mode,
            rank=rank,
            digest=target_digest,
            sabotage_applied=False,
            recovery_required=True,
        )
        damage_output = ssh(
            rank,
            shlex.join(
                [
                    "sudo",
                    "-n",
                    "python3",
                    SABOTAGE_SCRIPT,
                    "--store",
                    args.cache_root,
                    "--engine",
                    args.engine,
                    "--mode",
                    mode,
                    "--digest",
                    target_digest,
                ]
            ),
        )
        try:
            damage_report = parse_sabotage_report(damage_output, mode)
        except RuntimeError as error:
            record_mode_checkpoint(
                report,
                {
                    "mode": mode,
                    "rank": rank,
                    "passed": False,
                    "error": "sabotage-did-not-apply",
                    "detail": str(error),
                    "tool_output": damage_output[-200:],
                },
                args.output,
                mode_index,
            )
            print(f"  SABOTAGE FAILED TO APPLY: {damage_output[-200:]}", flush=True)
            break
        damaged_digest = damage_report["digest"]
        if damaged_digest != target_digest or not digest_healthy_on_all_ranks(
            before_verification, damaged_digest
        ):
            record_mode_checkpoint(
                report,
                {
                    "mode": mode,
                    "rank": rank,
                    "passed": False,
                    "error": "sabotaged-digest-was-not-preverified",
                    "digest": damaged_digest,
                },
                args.output,
                mode_index,
            )
            break
        record_mutation_checkpoint(
            report,
            args.output,
            phase="sabotage-applied",
            mode=mode,
            rank=rank,
            digest=damaged_digest,
            sabotage_applied=True,
            recovery_required=True,
        )
        # A plain ManifestStore lookup reports corruption but does not retire
        # the entry. Prove the exact damaged digest remains present and corrupt
        # before asking the running connector to load it.
        corrupt_verification = verify_stores(
            args.cache_root, args.engine, args.verify_script
        )
        corrupt_present = digest_corrupt_present(
            corrupt_verification.get(rank, {}), damaged_digest
        )
        corruption_isolated = corrupt_present and all(
            digest_entry_healthy(corrupt_verification.get(r, {}), damaged_digest)
            for r in HOSTS
            if r != rank
        )
        sentinel_healthy_after_corruption = digest_healthy_on_all_ranks(
            corrupt_verification, sentinel_digest
        )
        if not corruption_isolated or not sentinel_healthy_after_corruption:
            record_mode_checkpoint(
                report,
                {
                    "mode": mode,
                    "rank": rank,
                    "passed": False,
                    "error": "damaged-digest-not-observed-corrupt",
                    "digest": damaged_digest,
                    "corrupt_verification": corrupt_verification,
                },
                args.output,
                mode_index,
            )
            break
        # This identical request is the bounded connector lookup/load trigger.
        # Only the running connector owns invalidation; the verifier does not
        # claim or perform withdrawal.
        event_window_start = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z")
        result = fire(
            args.base_url,
            args.words,
            target_seed,
            args.model,
            target_nonce,
            cache_salt=mode_cache_salts["corruption-trigger"],
            max_tokens=args.max_tokens,
        )
        time.sleep(1)
        after_damage = manifest_counts(args.cache_root)
        withdrawal_verification = verify_stores(
            args.cache_root, args.engine, args.verify_script
        )
        withdrawn = digest_absent(
            withdrawal_verification.get(rank, {}), damaged_digest
        )
        retirement_snapshot_observed = retirement_contract_met(
            withdrawal_verification, rank, damaged_digest
        )
        # The triggering recompute may republish before this snapshot. The
        # subsequent identical request proves the converged entry restores.
        recovery = fire(
            args.base_url,
            args.words,
            target_seed,
            args.model,
            target_nonce,
            cache_salt=mode_cache_salts["recovery"],
            max_tokens=args.max_tokens,
        )
        time.sleep(5)
        after = manifest_counts(args.cache_root)
        final_verification = verify_stores(
            args.cache_root, args.engine, args.verify_script
        )
        # Digest- and request-bound connector events prove the transient
        # retirement even when recompute republishes before manifest sampling.
        logs = {}
        for r in HOSTS:
            container = rank_container(args.container_pattern, r)
            logs[r] = ssh(
                r,
                f"docker logs --timestamps --since {shlex.quote(event_window_start)} "
                f"{shlex.quote(container)} 2>&1 | "
                "grep -a 'spark-context-cache-event/v1' | tail -200",
            )
        connector_events = parse_connector_events(logs)
        retirement_events = retirement_event_proof(
            connector_events,
            rank,
            damaged_digest,
            result.get("request_id"),
        )
        engine_survived = healthy(args.base_url)
        completion = result.get("completion", "")
        request_error = result.get("error")
        expected_token_ids = probe.get("completion_token_ids")
        no_wrong_output = (
            request_error is None
            and bool(expected_token_ids)
            and result.get("completion_token_ids") == expected_token_ids
        )
        unrelated_intact = all(
            all(
                unrelated_entries_unchanged(
                    before_verification.get(r, {}), snapshot.get(r, {}), damaged_digest
                )
                for snapshot in (
                    corrupt_verification,
                    withdrawal_verification,
                    final_verification,
                )
            )
            for r in HOSTS
        )
        sentinel_intact = all(
            digest_healthy_on_all_ranks(snapshot, sentinel_digest)
            for snapshot in (
                before_verification,
                corrupt_verification,
                withdrawal_verification,
                final_verification,
            )
        )
        unaffected_workers_intact = all(
            digest_entry_healthy(
                withdrawal_verification.get(r, {}), damaged_digest
            )
            for r in HOSTS
            if r not in {0, rank}
        )
        final_digest_healthy = digest_healthy_on_all_ranks(
            final_verification, damaged_digest
        )
        final_stores_healthy = all(
            store_report_healthy(final_verification.get(r, {})) for r in HOSTS
        )
        record_mutation_checkpoint(
            report,
            args.output,
            phase=(
                "recovery-verified"
                if final_digest_healthy and final_stores_healthy
                else "recovery-still-required"
            ),
            mode=mode,
            rank=rank,
            digest=damaged_digest,
            sabotage_applied=True,
            recovery_required=not (
                final_digest_healthy and final_stores_healthy
            ),
        )
        others_intact = (
            unrelated_intact
            and sentinel_intact
            and unaffected_workers_intact
            and final_digest_healthy
            and final_stores_healthy
        )
        entry = {
            "mode": mode,
            "rank": rank,
            "damage": damage_report,
            "damaged_context_digest": damaged_digest,
            "target_seed": target_seed,
            "target_nonce": target_nonce,
            "target_seed_request_id": seed_result.get("request_id"),
            "target_seed_resolved_request_id": target_request_binding[
                "resolved_request_id"
            ],
            "target_seed_internal_child_request_id": target_request_binding[
                "internal_child_request_id"
            ],
            "target_seed_request_id_binding": target_request_binding,
            "target_seed_connector_events": target_seed_events,
            "cache_salt_receipts": mode_cache_salt_receipts,
            "sentinel_seed": sentinel_seed,
            "sentinel_nonce": sentinel_nonce,
            "sentinel_seed_request_id": sentinel_result.get("request_id"),
            "sentinel_seed_resolved_request_id": sentinel_request_binding[
                "resolved_request_id"
            ],
            "sentinel_seed_internal_child_request_id": sentinel_request_binding[
                "internal_child_request_id"
            ],
            "sentinel_seed_request_id_binding": sentinel_request_binding,
            "sentinel_seed_connector_events": sentinel_seed_events,
            "sentinel_context_digest": sentinel_digest,
            "initial_store_verification": initial_verification,
            "pre_sabotage_store_verification": before_verification,
            "post_sabotage_corrupt_store_verification": corrupt_verification,
            "damaged_digest_corrupt_present": corrupt_present,
            "withdrawal_store_verification": withdrawal_verification,
            "damaged_digest_absent_after_withdrawal": withdrawn,
            "scheduler_digest_absent_after_withdrawal": digest_absent(
                withdrawal_verification.get(0, {}), damaged_digest
            ),
            "retirement_snapshot_observed": retirement_snapshot_observed,
            "cluster_retirement_contract_met": retirement_events["passed"],
            "retirement_event_proof": retirement_events,
            "corruption_trigger_public_request_id": result.get("request_id"),
            "corruption_trigger_resolved_request_id": retirement_events.get(
                "resolved_request_id"
            ),
            "corruption_trigger_internal_child_request_id": retirement_events.get(
                "internal_child_request_id"
            ),
            "connector_events": connector_events,
            "unrelated_entries_intact": unrelated_intact,
            "sentinel_entries_intact": sentinel_intact,
            "unaffected_non_scheduler_workers_intact": unaffected_workers_intact,
            "after_damage": after_damage,
            "post_damage_request_store_verification": withdrawal_verification,
            "after": after,
            "final_recovery_store_verification": final_verification,
            "engine_survived": engine_survived,
            "no_wrong_output": no_wrong_output,
            "self_healed": (
                corrupt_present
                and corruption_isolated
                and retirement_events["passed"]
                and final_digest_healthy
            ),
            "damaged_digest_healthy_all_ranks_after_recovery": final_digest_healthy,
            "final_stores_healthy": final_stores_healthy,
            "others_intact": others_intact,
            "request_error": request_error,
            "request_outcome": (
                "correct_completion"
                if request_error is None and no_wrong_output
                else "error_fail_closed"
                if request_error is not None
                else "wrong_output"
            ),
            "ttft_seconds": result.get("ttft_seconds"),
            "completion_head": completion[:80],
            "rank_logs": logs,
            "seeded_restore_ttft": seeded_restore_ttft,
        }
        entry["recovery_ok"] = (
            recovery.get("error") is None
            and bool(expected_token_ids)
            and recovery.get("completion_token_ids") == expected_token_ids
        )
        entry["recovery_ttft"] = recovery.get("ttft_seconds")
        entry["passed"] = mode_pass(entry)
        print(json.dumps(entry, indent=1, sort_keys=True), flush=True)
        if not record_mode_checkpoint(
            report, entry, args.output, mode_index
        ):
            break
        _ACTIVE_RUN = None

    report["passed"] = bool(
        report.get("execution_state") == "completed"
        and len(report["modes"]) == len(MATRIX_MODES)
        and all(m.get("passed") is True for m in report["modes"])
    )
    checkpoint_evidence_atomic(args.output, report)
    print("P3 matrix:", "PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 2


def main() -> int:
    """Run the matrix and preserve a terminal checkpoint on unexpected errors."""

    global _ACTIVE_RUN

    try:
        return _main()
    except BaseException as error:  # noqa: BLE001
        active = _ACTIVE_RUN
        if active is None:
            raise
        report, output, mode_index, mode, rank = active
        if report.get("execution_state") in {"initializing", "running"}:
            if mode_index >= 0:
                record_mode_checkpoint(
                    report,
                    {
                        "mode": mode,
                        "rank": rank,
                        "error": "unexpected-mode-exception",
                        "detail": repr(error),
                    },
                    output,
                    mode_index,
                )
            else:
                report["execution_state"] = "quarantined_after_setup_failure"
                report["sabotage_halted"] = True
                report["setup_failure"] = {
                    "error": "unexpected-setup-exception",
                    "detail": repr(error),
                }
                checkpoint_evidence_atomic(output, report)
        print(f"P3 matrix: FAIL ({error!r})", flush=True)
        if not isinstance(error, Exception):
            raise
        return 2
    finally:
        _ACTIVE_RUN = None


if __name__ == "__main__":
    raise SystemExit(main())
