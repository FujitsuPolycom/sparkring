#!/usr/bin/env python3
"""Reduce private SparkCache P3 evidence to a closed publishable receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence


RAW_SCHEMA = "sparkring-context-cache-corruption-evidence/v1"
OUTPUT_SCHEMA = "sparkring-context-cache-corruption-receipt/v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
MODE_ORDER = (("bitflip", 2), ("truncate", 1), ("fingerprint", 3))
TERMINAL_STATES = {
    "completed",
    "quarantined_after_failure",
    "quarantined_after_setup_failure",
}
MODE_ERRORS = {
    "api-not-healthy",
    "target-request-has-no-request-bound-shared-commit",
    "sentinel-request-has-no-distinct-request-bound-shared-commit",
    "pre-sabotage-store-verification-failed",
    "pre-sabotage-exact-token-probe-failed",
    "sabotage-did-not-apply",
    "sabotaged-digest-was-not-preverified",
    "damaged-digest-not-observed-corrupt",
    "unexpected-mode-exception",
    "mode-gates-failed",
}
SETUP_ERRORS = {
    "remote-tool-attestation-failed",
    "unexpected-setup-exception",
}
REQUEST_OUTCOMES = {"correct_completion", "error_fail_closed", "wrong_output"}
MUTATION_PHASES = {
    "sabotage-command-armed",
    "sabotage-applied",
    "recovery-verified",
    "recovery-still-required",
}
MUTATION_FLAGS = {
    "sabotage-command-armed": (False, True),
    "sabotage-applied": (True, True),
    "recovery-still-required": (True, True),
    "recovery-verified": (True, False),
}
BOOL_FIELDS = (
    "engine_survived",
    "no_wrong_output",
    "self_healed",
    "others_intact",
    "recovery_ok",
    "cluster_retirement_contract_met",
    "damaged_digest_healthy_all_ranks_after_recovery",
    "final_stores_healthy",
)


class ReduceError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReduceError(message)


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReduceError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise ReduceError(f"non-finite JSON number {value!r} is unsupported")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _optional_sha(value: Any, prefix: str) -> str | None:
    if value is None:
        return None
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{prefix} must be null or lowercase SHA-256",
    )
    return value


def _optional_bool(value: Any, prefix: str) -> bool | None:
    _require(value is None or isinstance(value, bool), f"{prefix} must be boolean or null")
    return value


def _mode_receipt(value: Any, index: int) -> dict[str, Any]:
    _require(isinstance(value, dict), f"modes[{index}] must be an object")
    expected_mode, expected_rank = MODE_ORDER[index]
    _require(value.get("mode") == expected_mode, f"modes[{index}].mode is out of order")
    _require(value.get("rank") == expected_rank, f"modes[{index}].rank is not canonical")
    passed = value.get("passed", False)
    _require(isinstance(passed, bool), f"modes[{index}].passed must be boolean")
    error = value.get("error")
    _require(error is None or error in MODE_ERRORS, f"modes[{index}].error is unsupported")
    outcome = value.get("request_outcome")
    _require(
        outcome is None or outcome in REQUEST_OUTCOMES,
        f"modes[{index}].request_outcome is unsupported",
    )
    digest = value.get("damaged_context_digest", value.get("digest"))
    receipt = {
        "mode": expected_mode,
        "rank": expected_rank,
        "passed": passed,
        "error": error,
        "damaged_context_digest": _optional_sha(
            digest, f"modes[{index}].damaged_context_digest"
        ),
        "request_outcome": outcome,
        **{
            field: _optional_bool(value.get(field), f"modes[{index}].{field}")
            for field in BOOL_FIELDS
        },
    }
    if passed:
        _require(error is None, f"modes[{index}] passed with an error")
        _require(
            "request_error" in value and value["request_error"] is None,
            f"modes[{index}] passed with a request error or missing request_error proof",
        )
        _require(
            receipt["damaged_context_digest"] is not None,
            f"modes[{index}] passed without a damaged digest",
        )
        _require(
            outcome == "correct_completion",
            f"modes[{index}] passed without an exact correct completion",
        )
        _require(
            all(receipt[field] is True for field in BOOL_FIELDS),
            f"modes[{index}] passed without every healing/isolation proof",
        )
    return receipt


def _mutation_receipt(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    _require(isinstance(value, dict), "active_mutation must be an object or null")
    _require(
        set(value)
        == {
            "phase",
            "mode",
            "rank",
            "digest",
            "sabotage_applied",
            "recovery_required",
        },
        "active_mutation fields are unsupported",
    )
    _require(value["phase"] in MUTATION_PHASES, "active_mutation.phase is unsupported")
    pair = (value["mode"], value["rank"])
    _require(pair in MODE_ORDER, "active_mutation mode/rank is unsupported")
    _require(isinstance(value["sabotage_applied"], bool), "active_mutation sabotage flag must be boolean")
    _require(isinstance(value["recovery_required"], bool), "active_mutation recovery flag must be boolean")
    _require(
        (value["sabotage_applied"], value["recovery_required"])
        == MUTATION_FLAGS[value["phase"]],
        "active_mutation flags disagree with its phase",
    )
    return {
        **value,
        "digest": _optional_sha(value["digest"], "active_mutation.digest"),
    }


def reduce_document(document: Any, raw_sha256: str) -> dict[str, Any]:
    _require(isinstance(document, dict), "raw evidence must be a JSON object")
    _require(
        isinstance(raw_sha256, str) and SHA256_RE.fullmatch(raw_sha256) is not None,
        "raw artifact digest must be lowercase SHA-256",
    )
    _require(document.get("schema") == RAW_SCHEMA, f"raw schema must be {RAW_SCHEMA}")
    evidence_scope = document.get("evidence_scope")
    _require(
        evidence_scope
        == {
            "classification": "private-raw-live-corruption-evidence",
            "publishable": False,
            "contains_private_rank_targets_paths_and_logs": True,
            "power_loss_durability": (
                "file-fsync-and-atomic-replace; parent-directory-fsync only where supported"
            ),
        },
        "raw evidence scope is missing or unsupported",
    )
    rank_targets = document.get("rank_targets")
    _require(
        isinstance(rank_targets, dict) and set(rank_targets) == {"0", "1", "2", "3"},
        "raw rank targets must cover exactly ranks 0..3",
    )
    _require(
        all(isinstance(value, str) and value for value in rank_targets.values())
        and len(set(rank_targets.values())) == 4,
        "raw rank targets must be four distinct non-empty strings",
    )
    _require(
        document.get("planned_modes")
        == [{"mode": mode, "rank": rank} for mode, rank in MODE_ORDER],
        "raw planned modes are not canonical",
    )
    state = document.get("execution_state")
    _require(state in TERMINAL_STATES, "raw evidence is not terminal")
    passed = document.get("passed")
    _require(isinstance(passed, bool), "raw passed must be boolean")
    _require(
        document.get("sabotage_halted") is (not passed),
        "raw sabotage_halted disagrees with terminal result",
    )
    modes = document.get("modes")
    _require(isinstance(modes, list), "raw modes must be a list")
    _require(len(modes) <= len(MODE_ORDER), "raw evidence has too many modes")
    normalized_modes = [_mode_receipt(value, index) for index, value in enumerate(modes)]
    completed_count = document.get("completed_mode_count")
    _require(
        _is_int(completed_count) and completed_count == len(modes),
        "completed_mode_count disagrees with modes",
    )
    if passed:
        _require(state == "completed", "passed evidence must be completed")
        _require(len(normalized_modes) == len(MODE_ORDER), "passed evidence must contain all modes")
        _require(all(item["passed"] for item in normalized_modes), "passed evidence contains a failed mode")
    else:
        _require(state != "completed", "completed evidence cannot be failed")

    setup_failure = document.get("setup_failure")
    setup_error = None
    if setup_failure is not None:
        _require(isinstance(setup_failure, dict), "setup_failure must be an object")
        setup_error = setup_failure.get("error")
        _require(setup_error in SETUP_ERRORS, "setup_failure.error is unsupported")
    active_mutation = _mutation_receipt(document.get("active_mutation"))
    remaining_modes = document.get("remaining_modes")
    terminal_failure = document.get("terminal_failure")
    if state == "quarantined_after_setup_failure":
        _require(not passed, "setup failure cannot pass")
        _require(not normalized_modes, "setup failure cannot contain mode results")
        _require(active_mutation is None, "setup failure cannot contain a cache mutation")
        _require(setup_error is not None, "setup failure must contain a supported setup error")
        _require(
            remaining_modes == [mode for mode, _rank in MODE_ORDER],
            "setup failure remaining_modes must retain the complete canonical matrix",
        )
        _require(
            terminal_failure is None,
            "setup failure cannot contain a terminal mode failure",
        )
    elif state == "quarantined_after_failure":
        _require(not passed, "mode failure cannot pass")
        _require(setup_error is None, "mode failure cannot contain a setup error")
        _require(bool(normalized_modes), "mode failure must identify the failed mode")
        _require(
            normalized_modes[-1]["passed"] is False,
            "mode failure must end with a failed mode",
        )
        _require(
            all(item["passed"] is True for item in normalized_modes[:-1]),
            "mode failure cannot follow an earlier failed mode",
        )
        _require(
            remaining_modes
            == [mode for mode, _rank in MODE_ORDER[len(normalized_modes) :]],
            "mode failure remaining_modes disagrees with completed modes",
        )
        expected_error = normalized_modes[-1]["error"] or "mode-gates-failed"
        _require(
            terminal_failure
            == {
                "mode": normalized_modes[-1]["mode"],
                "rank": normalized_modes[-1]["rank"],
                "error": expected_error,
            },
            "terminal_failure disagrees with the failed mode",
        )
    else:
        _require(setup_error is None, "completed evidence cannot contain a setup error")
        _require(remaining_modes == [], "completed evidence must have no remaining modes")
        _require(
            terminal_failure is None,
            "completed evidence cannot contain a terminal failure",
        )
    if passed:
        _require(
            active_mutation is not None
            and active_mutation["phase"] == "recovery-verified"
            and active_mutation["mode"] == MODE_ORDER[-1][0]
            and active_mutation["rank"] == MODE_ORDER[-1][1]
            and active_mutation["digest"]
            == normalized_modes[-1]["damaged_context_digest"]
            and active_mutation["sabotage_applied"] is True
            and active_mutation["recovery_required"] is False,
            "passed evidence must prove the final mutation reconciled",
        )
    receipt = {
        "schema": OUTPUT_SCHEMA,
        "private_raw_artifact_sha256": raw_sha256,
        "execution_state": state,
        "passed": passed,
        "completed_mode_count": completed_count,
        "setup_error": setup_error,
        "active_mutation": active_mutation,
        "modes": normalized_modes,
        "scope": (
            "Sanitized transaction and gate receipt only; SSH targets, host paths, "
            "request content, full verifier records, and container logs are omitted."
        ),
    }
    # Prove encoding cannot silently admit NaN/Inf before returning it.
    json.dumps(receipt, allow_nan=False)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_nonfinite,
        )
        receipt = reduce_document(document, hashlib.sha256(raw).hexdigest())
        rendered = json.dumps(
            receipt, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(rendered)
        print(rendered, end="")
        return 0
    except (
        FileExistsError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ReduceError,
        ValueError,
    ) as error:
        print(f"context-cache-p3-reduce: ERROR: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
