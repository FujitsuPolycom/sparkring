#!/usr/bin/env python3
"""Reduce private EXL3 launcher evidence to a publishable sanitized receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence


RAW_PLAN_SCHEMA = "sparkring-exl3-attribution-plan/v1"
OUTPUT_SCHEMA = "sparkring-exl3-attribution-launch-receipt/v1"
RANK_RE = re.compile(r"[0-3]")
COMMANDS = {"activate", "restart-arm", "rollback", "status", "transition"}
ARMS = {
    "a-mtp0-apc0-lmcache0": (0, False, False),
    "b-mtp2-apc0-lmcache0": (2, False, False),
    "c-mtp2-apc1-lmcache0": (2, True, False),
    "d-mtp2-apc1-lmcache1": (2, True, True),
    "e-mtp0-apc0-lmcache1": (0, False, True),
    "f-mtp2-apc0-lmcache1": (2, False, True),
}
PROFILE_IDS = {"glm52-exl3-tr3-3.25bpw-lmcache-cs512"}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PHASES = {
    "server_health",
    "canonical_engine_exclusive",
    "no_other_diagnostic",
    "prepare_page_cache_reclaim_entrypoint",
    "remove_canonical_engines",
    "start_diagnostic",
    "diagnostic_ready",
    "diagnostic_live_arm_attestation",
    "remove_diagnostic",
    "start_canonical",
    "canonical_ready",
    "canonical_engine_exclusive_after_restore",
    "source_server_health",
    "source_diagnostic_ready",
    "source_diagnostic_exclusive",
    "remove_source_diagnostic",
    "post_source_removal_server_health",
    "start_target_diagnostic",
    "target_diagnostic_ready",
    "target_server_health",
    "target_live_arm_attestation",
    "remove_target_diagnostic",
    "rollback_remove_canonical_engines",
    "rollback_server_health_before_restore",
    "rollback_server_health_after_restore",
    "isolate_target_lmcache_remove_servers",
    "isolate_target_lmcache_start_servers",
    "isolate_target_lmcache_server_health",
    "rollback_reset_lmcache_remove_servers",
    "rollback_reset_lmcache_start_servers",
    "rollback_reset_lmcache_server_health",
    "emit_execution_report",
}
EXCEPTION_TYPES = {
    "AssertionError",
    "CalledProcessError",
    "ConnectionError",
    "OSError",
    "ProfileError",
    "RuntimeError",
    "TimeoutError",
    "ValueError",
}
RAW_DOCUMENT_KEYS = {
    "plan",
    "results",
    "original_exception",
    "automatic_rollback",
}
RAW_PLAN_KEYS = {
    "schema",
    "command",
    "mutates_remote",
    "startup_memory_hygiene",
    "arm",
    "from_arm",
    "functional_settings",
    "canonical_attestation",
    "diagnostic_identity",
    "source_diagnostic_identity",
    "server_policy",
    "lmcache_l1_isolation",
    "live_arm_receipt_contract",
    "sequence",
    "phases",
    "rollback_phases",
    "automatic_failure_action",
    "evidence_policy",
}
CANONICAL_ATTESTATION_KEYS = {
    "profile_id",
    "profile_file_sha256",
    "image_id",
    "model_revision",
}
CURRENT_CACHE_GEOMETRY = {
    "expected_engine_block_rows_per_dcp_rank": 64,
    "expected_dcp_size": 4,
    "expected_global_apc_alignment_tokens": 256,
    "expected_lmcache_chunk_tokens_global": 512,
    "runtime_attestation_required": True,
    "recipe_predecessor_chunk_size_is_geometry_evidence": False,
}
LEGACY_CACHE_GEOMETRY = {
    "native_apc_block_tokens": 256,
    "lmcache_chunk_tokens": 512,
}
LEGACY_DISABLED_MEMORY_HYGIENE = {
    "post_verification_host_page_cache_reclaim": False,
    "requires_passwordless_sudo": False,
    "safety_class": "no-additional-mutation",
}
LEGACY_ENABLED_MEMORY_HYGIENE = {
    "post_verification_host_page_cache_reclaim": True,
    "requires_passwordless_sudo": True,
    "safety_class": "MUTATES HOST",
}
DISABLED_MEMORY_HYGIENE = {
    **LEGACY_DISABLED_MEMORY_HYGIENE,
    "boundaries": [],
    "inner_model_verification": "unchanged-image-entrypoint",
    "outer_verified_entrypoint_sha256": None,
    "preflight_before_engine_removal": False,
}
ENABLED_MEMORY_HYGIENE = {
    **LEGACY_ENABLED_MEMORY_HYGIENE,
    "boundaries": [
        "after the sole full model verification and before docker run/vLLM",
    ],
    "inner_model_verification": "skipped-by-sha256-attested-entrypoint-after-outer-pass",
    "outer_verified_entrypoint_sha256": "0" * 64,
    "preflight_before_engine_removal": True,
}


class ReduceError(ValueError):
    pass


def _duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReduceError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> Any:
    raise ReduceError(f"non-finite JSON number {value!r} is unsupported")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReduceError(message)


def _hash_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_exact(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int equivalence."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _json_exact(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_exact(a, b) for a, b in zip(left, right)
        )
    return left == right


def _current_enabled_memory_hygiene(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != set(ENABLED_MEMORY_HYGIENE):
        return False
    digest = value.get("outer_verified_entrypoint_sha256")
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        return False
    normalized = dict(value)
    normalized["outer_verified_entrypoint_sha256"] = "0" * 64
    return _json_exact(normalized, ENABLED_MEMORY_HYGIENE)


def _expected_settings(arm: str) -> dict[str, Any]:
    mtp, apc, cache = ARMS[arm]
    return {
        "mtp_tokens": mtp,
        "native_prefix_cache": apc,
        "lmcache_connector": cache,
        "cache_boundary_geometry": dict(CURRENT_CACHE_GEOMETRY),
    }


def _raw_settings_contract(value: Any, arm: str) -> str:
    """Validate known historical settings shapes and name their contract.

    Launcher reports kept the same plan schema while the cache-geometry
    attestation evolved.  Accept only the three exact shapes actually emitted;
    normalize all of them to the current public settings object.
    """
    _require(isinstance(value, dict), "raw plan functional_settings must be an object")
    expected = _expected_settings(arm)
    base = {key: expected[key] for key in ("mtp_tokens", "native_prefix_cache", "lmcache_connector")}
    if _json_exact(value, base):
        return "legacy-v1-no-cache-geometry"
    if _json_exact(value, {**base, "cache_boundary_geometry": LEGACY_CACHE_GEOMETRY}):
        return "legacy-v2-cache-geometry"
    if _json_exact(value, expected):
        return "current-cache-geometry"
    raise ReduceError("raw plan functional settings are unsupported or inconsistent")


def _memory_hygiene(value: Any) -> dict[str, Any]:
    # Historical v1/v2 reports predate this field and therefore mean the
    # additional reclaim operation was not requested.
    if value is None:
        return dict(DISABLED_MEMORY_HYGIENE)
    if _json_exact(value, DISABLED_MEMORY_HYGIENE) or _json_exact(
        value, LEGACY_DISABLED_MEMORY_HYGIENE
    ):
        return dict(DISABLED_MEMORY_HYGIENE)
    if _current_enabled_memory_hygiene(value):
        return dict(value)
    if _json_exact(value, LEGACY_ENABLED_MEMORY_HYGIENE):
        return dict(LEGACY_ENABLED_MEMORY_HYGIENE)
    raise ReduceError("raw plan startup_memory_hygiene is unsupported")


def _raw_exception(value: Any, prefix: str) -> dict[str, str] | None:
    if value is None:
        return None
    _require(isinstance(value, dict), f"{prefix} must be an object or null")
    _require(set(value) == {"phase", "type", "message"}, f"{prefix} fields are unsupported")
    _require(value["phase"] in PHASES, f"{prefix} phase is unsupported")
    _require(value["type"] in EXCEPTION_TYPES, f"{prefix} type is unsupported")
    _require(isinstance(value["message"], str), f"{prefix} message must be text")
    return {
        "phase": value["phase"],
        "type": value["type"],
        "message_sha256": _hash_text(value["message"]),
    }


def _phase_receipts(value: Any, prefix: str) -> list[dict[str, Any]]:
    _require(isinstance(value, dict), f"{prefix} must be an object")
    receipts = []
    for phase, rank_results in value.items():
        if phase == "execution_exception":
            _raw_exception(rank_results, f"{prefix}.execution_exception")
            continue
        if phase == "automatic_rollback_failed":
            _phase_receipts(rank_results, f"{prefix}.automatic_rollback_failed")
            continue
        if phase == "rollback_exception":
            _raw_exception(rank_results, f"{prefix}.rollback_exception")
            continue
        if phase == "rollback_exceptions":
            _require(
                isinstance(rank_results, list),
                f"{prefix}.rollback_exceptions must be a list",
            )
            for index, item in enumerate(rank_results):
                _raw_exception(item, f"{prefix}.rollback_exceptions[{index}]")
            continue
        _require(phase in PHASES, f"{prefix} has unsupported phase")
        _require(isinstance(rank_results, dict), f"{prefix}.{phase} must be an object")
        marker_keys = set(rank_results)
        if marker_keys in ({"malformed_executor_result"}, {"executor_exception"}):
            marker = next(iter(marker_keys))
            _require(rank_results[marker] is True, f"{prefix}.{phase} marker must be true")
            receipts.append(
                {
                    "phase": phase,
                    "all_ranks_zero": False,
                    "executor_result_status": marker.replace("_", "-"),
                    "ranks": [],
                }
            )
            continue
        ranks = []
        for rank, result in rank_results.items():
            rank_text = str(rank)
            _require(RANK_RE.fullmatch(rank_text) is not None, f"{prefix}.{phase} has invalid rank")
            _require(isinstance(result, dict), f"{prefix}.{phase}.{rank} must be an object")
            _require(
                set(result) == {"exit_code", "stdout", "stderr"},
                f"{prefix}.{phase}.{rank} fields are unsupported",
            )
            exit_code = result.get("exit_code")
            _require(
                isinstance(exit_code, int)
                and not isinstance(exit_code, bool)
                and 0 <= exit_code <= 255,
                f"{prefix}.{phase}.{rank}.exit_code must be in [0, 255]",
            )
            _require(
                isinstance(result["stdout"], str) and isinstance(result["stderr"], str),
                f"{prefix}.{phase}.{rank} output fields must be text",
            )
            ranks.append({"rank": int(rank_text), "exit_code": exit_code})
        ranks.sort(key=lambda item: item["rank"])
        receipts.append(
            {
                "phase": phase,
                "all_ranks_zero": len(ranks) == 4 and all(item["exit_code"] == 0 for item in ranks),
                "executor_result_status": "rank-results",
                "ranks": ranks,
            }
        )
    return receipts


def _validate_exception_receipt(value: Any, prefix: str) -> None:
    if value is None:
        return
    _require(isinstance(value, dict), f"{prefix} must be an object or null")
    _require(
        set(value) == {"phase", "type", "message_sha256"},
        f"{prefix} fields are unsupported",
    )
    _require(value["phase"] in PHASES, f"{prefix}.phase is unsupported")
    _require(value["type"] in EXCEPTION_TYPES, f"{prefix}.type is unsupported")
    _require(
        isinstance(value["message_sha256"], str)
        and SHA256_RE.fullmatch(value["message_sha256"]) is not None,
        f"{prefix}.message_sha256 must be lowercase SHA-256",
    )


def _validate_phase_receipts(value: Any, prefix: str) -> None:
    _require(isinstance(value, list), f"{prefix} must be a list")
    for index, receipt in enumerate(value):
        item = f"{prefix}[{index}]"
        _require(isinstance(receipt, dict), f"{item} must be an object")
        _require(
            set(receipt)
            == {"phase", "all_ranks_zero", "executor_result_status", "ranks"},
            f"{item} fields are unsupported",
        )
        _require(receipt["phase"] in PHASES, f"{item}.phase is unsupported")
        _require(
            isinstance(receipt["all_ranks_zero"], bool),
            f"{item}.all_ranks_zero must be boolean",
        )
        _require(
            receipt["executor_result_status"]
            in {"rank-results", "executor-exception", "malformed-executor-result"},
            f"{item}.executor_result_status is unsupported",
        )
        ranks = receipt["ranks"]
        _require(isinstance(ranks, list), f"{item}.ranks must be a list")
        for rank_index, rank in enumerate(ranks):
            rank_item = f"{item}.ranks[{rank_index}]"
            _require(
                isinstance(rank, dict) and set(rank) == {"rank", "exit_code"},
                f"{rank_item} fields are unsupported",
            )
            _require(
                isinstance(rank["rank"], int)
                and not isinstance(rank["rank"], bool)
                and 0 <= rank["rank"] <= 3,
                f"{rank_item}.rank must be in [0, 3]",
            )
            _require(
                isinstance(rank["exit_code"], int)
                and not isinstance(rank["exit_code"], bool)
                and 0 <= rank["exit_code"] <= 255,
                f"{rank_item}.exit_code must be in [0, 255]",
            )


def _validate_publishable_receipt(receipt: dict[str, Any]) -> None:
    """Audit the complete public schema after all private input is discarded."""
    expected_keys = {
        "schema", "private_raw_artifact_sha256", "command", "arm", "from_arm",
        "raw_functional_settings_contract", "functional_settings",
        "startup_memory_hygiene", "canonical_profile_id", "phase_outcomes",
        "original_exception", "automatic_rollback_attempted",
        "automatic_rollback_exceptions", "automatic_rollback_outcomes", "scope",
    }
    _require(set(receipt) == expected_keys, "publishable receipt fields are unsupported")
    _require(receipt["schema"] == OUTPUT_SCHEMA, "publishable receipt schema drifted")
    _require(
        isinstance(receipt["private_raw_artifact_sha256"], str)
        and SHA256_RE.fullmatch(receipt["private_raw_artifact_sha256"]) is not None,
        "private raw artifact digest must be lowercase SHA-256",
    )
    _require(receipt["command"] in COMMANDS, "publishable command is unsupported")
    _require(receipt["arm"] in ARMS, "publishable arm is unsupported")
    _require(
        receipt["from_arm"] is None or receipt["from_arm"] in ARMS,
        "publishable source arm is unsupported",
    )
    expected_settings = _expected_settings(receipt["arm"])
    _require(
        receipt["raw_functional_settings_contract"]
        in {
            "legacy-v1-no-cache-geometry",
            "legacy-v2-cache-geometry",
            "current-cache-geometry",
        },
        "publishable raw settings contract is unsupported",
    )
    _require(
        _json_exact(receipt["functional_settings"], expected_settings),
        "publishable functional settings are unsupported",
    )
    _require(
        _json_exact(receipt["startup_memory_hygiene"], DISABLED_MEMORY_HYGIENE)
        or _current_enabled_memory_hygiene(receipt["startup_memory_hygiene"])
        or _json_exact(
            receipt["startup_memory_hygiene"], LEGACY_ENABLED_MEMORY_HYGIENE
        ),
        "publishable startup memory hygiene is unsupported",
    )
    _require(receipt["canonical_profile_id"] in PROFILE_IDS, "publishable profile ID is unsupported")
    _validate_phase_receipts(receipt["phase_outcomes"], "phase_outcomes")
    _validate_exception_receipt(receipt["original_exception"], "original_exception")
    _require(isinstance(receipt["automatic_rollback_attempted"], bool), "rollback attempted must be boolean")
    rollback_exceptions = receipt["automatic_rollback_exceptions"]
    _require(isinstance(rollback_exceptions, list), "rollback exceptions must be a list")
    for index, item in enumerate(rollback_exceptions):
        _validate_exception_receipt(item, f"automatic_rollback_exceptions[{index}]")
    _validate_phase_receipts(receipt["automatic_rollback_outcomes"], "automatic_rollback_outcomes")
    _require(
        receipt["scope"]
        == "Exit-code and transaction receipt only; remote execution details and exception text are intentionally omitted.",
        "publishable receipt scope drifted",
    )


def reduce_document(document: dict[str, Any], raw_sha256: str) -> dict[str, Any]:
    _require(isinstance(document, dict), "raw evidence must be an object")
    _require(
        {"plan", "results"} <= set(document) <= RAW_DOCUMENT_KEYS,
        "raw evidence top-level fields are unsupported",
    )
    plan = document.get("plan")
    _require(isinstance(plan, dict), "raw evidence.plan must be an object")
    _require(set(plan) <= RAW_PLAN_KEYS, "raw evidence.plan fields are unsupported")
    _require(plan.get("schema") == RAW_PLAN_SCHEMA, f"raw plan schema must be {RAW_PLAN_SCHEMA}")
    command = plan.get("command")
    arm = plan.get("arm")
    from_arm = plan.get("from_arm")
    _require(command in COMMANDS, "raw plan command is unsupported")
    _require(arm in ARMS, "raw plan arm is unsupported")
    _require(from_arm is None or from_arm in ARMS, "raw plan from_arm is unsupported")
    _require(
        (command in {"transition", "restart-arm"}) == (from_arm is not None),
        "raw plan command/source-arm contract is inconsistent",
    )
    if command == "restart-arm":
        _require(from_arm == arm, "raw restart-arm must use the same source and target arm")
    elif command == "transition":
        _require(from_arm != arm, "raw transition must use distinct source and target arms")
    settings_contract = _raw_settings_contract(plan.get("functional_settings"), arm)
    expected_settings = _expected_settings(arm)
    memory_hygiene = _memory_hygiene(plan.get("startup_memory_hygiene"))
    canonical_attestation = plan.get("canonical_attestation")
    _require(
        isinstance(canonical_attestation, dict),
        "raw plan canonical_attestation must be an object",
    )
    _require(
        set(canonical_attestation) == CANONICAL_ATTESTATION_KEYS,
        "raw plan canonical_attestation fields are unsupported",
    )
    profile_id = canonical_attestation.get("profile_id")
    _require(profile_id in PROFILE_IDS, "raw plan canonical profile ID is unsupported")
    exception_receipt = _raw_exception(document.get("original_exception"), "original_exception")
    rollback = document.get("automatic_rollback")
    rollback_exceptions = []
    if isinstance(rollback, dict):
        raw_exceptions = rollback.get("rollback_exceptions", [])
        _require(isinstance(raw_exceptions, list), "rollback_exceptions must be a list")
        for item in raw_exceptions:
            rollback_exceptions.append(_raw_exception(item, "rollback exception"))
    receipt = {
        "schema": OUTPUT_SCHEMA,
        "private_raw_artifact_sha256": raw_sha256,
        "command": command,
        "arm": arm,
        "from_arm": from_arm,
        "raw_functional_settings_contract": settings_contract,
        "functional_settings": expected_settings,
        "startup_memory_hygiene": memory_hygiene,
        "canonical_profile_id": profile_id,
        "phase_outcomes": _phase_receipts(document.get("results"), "results"),
        "original_exception": exception_receipt,
        "automatic_rollback_attempted": rollback is not None,
        "automatic_rollback_exceptions": rollback_exceptions,
        "automatic_rollback_outcomes": (
            _phase_receipts(rollback, "automatic_rollback") if rollback is not None else []
        ),
        "scope": "Exit-code and transaction receipt only; remote execution details and exception text are intentionally omitted.",
    }
    _validate_publishable_receipt(receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    try:
        args = parser.parse_args(argv)
        raw = Path(args.input).read_bytes()
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicates,
            parse_constant=_reject_nonfinite_constant,
        )
        reduced = reduce_document(document, hashlib.sha256(raw).hexdigest())
        rendered = json.dumps(reduced, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(rendered)
        print(rendered, end="")
        return 0
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ReduceError) as exc:
        print(f"exl3-attribution-reduce: ERROR: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
