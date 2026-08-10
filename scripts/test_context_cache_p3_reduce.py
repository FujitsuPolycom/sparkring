from __future__ import annotations

import hashlib
import json

import context_cache_p3_reduce as reduce
import pytest


def raw_report(*, passed: bool = True) -> dict:
    modes = []
    for mode, rank in reduce.MODE_ORDER:
        modes.append(
            {
                "mode": mode,
                "rank": rank,
                "passed": passed,
                "damage": {"manifest": "/private/cache/manifest.json"},
                "damaged_context_digest": (str(rank) * 64),
                "request_outcome": "correct_completion" if passed else "wrong_output",
                "request_error": None,
                **{field: passed for field in reduce.BOOL_FIELDS},
                "rank_logs": {"0": "private-host /srv/cache"},
            }
        )
        if not passed:
            modes[-1]["error"] = "mode-gates-failed"
            break
    return {
        "schema": reduce.RAW_SCHEMA,
        "evidence_scope": {
            "classification": "private-raw-live-corruption-evidence",
            "publishable": False,
            "contains_private_rank_targets_paths_and_logs": True,
            "power_loss_durability": (
                "file-fsync-and-atomic-replace; parent-directory-fsync only where supported"
            ),
        },
        "execution_state": "completed" if passed else "quarantined_after_failure",
        "passed": passed,
        "sabotage_halted": not passed,
        "completed_mode_count": len(modes),
        "remaining_modes": (
            [] if passed else [mode for mode, _rank in reduce.MODE_ORDER[1:]]
        ),
        "terminal_failure": (
            None
            if passed
            else {
                "mode": modes[-1]["mode"],
                "rank": modes[-1]["rank"],
                "error": modes[-1]["error"],
            }
        ),
        "active_mutation": {
            "phase": "recovery-verified" if passed else "recovery-still-required",
            "mode": modes[-1]["mode"],
            "rank": modes[-1]["rank"],
            "digest": modes[-1]["damaged_context_digest"],
            "sabotage_applied": True,
            "recovery_required": not passed,
        },
        "rank_targets": {
            "0": "private-rank0",
            "1": "private-rank1",
            "2": "private-rank2",
            "3": "private-rank3",
        },
        "planned_modes": [
            {"mode": mode, "rank": rank} for mode, rank in reduce.MODE_ORDER
        ],
        "modes": modes,
    }


def test_reducer_binds_raw_and_omits_private_values(tmp_path, capsys):
    document = raw_report()
    raw = json.dumps(document).encode()
    source = tmp_path / "private.json"
    source.write_bytes(raw)
    output = tmp_path / "public.json"
    assert reduce.main(["--input", str(source), "--output", str(output)]) == 0
    rendered = capsys.readouterr().out
    assert output.read_text(encoding="utf-8") == rendered
    assert "private-host" not in rendered
    assert "/private/" not in rendered
    receipt = json.loads(rendered)
    assert receipt["private_raw_artifact_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["passed"] is True
    assert len(receipt["modes"]) == 3


def test_reducer_preserves_failed_recovery_state_without_private_detail():
    receipt = reduce.reduce_document(raw_report(passed=False), "a" * 64)
    assert receipt["passed"] is False
    assert receipt["active_mutation"]["recovery_required"] is True
    assert receipt["modes"][0]["error"] == "mode-gates-failed"


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda value: value["modes"][-1].pop("damaged_context_digest"),
            "damaged digest",
        ),
        (
            lambda value: value["modes"][-1].update(request_outcome=None),
            "correct completion",
        ),
        (
            lambda value: value["modes"][-1].update(
                request_error="HTTP 500 private/path"
            ),
            "request error",
        ),
        (
            lambda value: value["modes"][-1].pop("request_error"),
            "missing request_error proof",
        ),
        (
            lambda value: value["modes"][-1].update(self_healed=None),
            "healing/isolation",
        ),
        (
            lambda value: value["active_mutation"].update(sabotage_applied=False),
            "active_mutation flags disagree with its phase",
        ),
        (
            lambda value: value["active_mutation"].update(phase="sabotage-command-armed"),
            "active_mutation flags disagree with its phase",
        ),
        (
            lambda value: value["active_mutation"].update(digest="e" * 64),
            "final mutation reconciled",
        ),
    ],
)
def test_pass_requires_complete_corruption_and_recovery_proof(mutation, match):
    document = raw_report()
    mutation(document)
    with pytest.raises(reduce.ReduceError, match=match):
        reduce.reduce_document(document, "a" * 64)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda value: value["active_mutation"].update(
                phase="recovery-still-required", recovery_required=False
            ),
            "flags disagree",
        ),
        (
            lambda value: value["active_mutation"].update(
                phase="recovery-still-required", sabotage_applied=False
            ),
            "flags disagree",
        ),
        (
            lambda value: value.update(
                execution_state="quarantined_after_setup_failure",
                setup_failure={"error": "remote-tool-attestation-failed"},
                completed_mode_count=0,
                modes=[],
            ),
            "setup failure cannot contain a cache mutation",
        ),
        (
            lambda value: value.update(
                execution_state="quarantined_after_failure",
                setup_failure={"error": "remote-tool-attestation-failed"},
            ),
            "mode failure cannot contain a setup error",
        ),
    ],
)
def test_failure_receipt_phase_and_terminal_state_are_consistent(mutation, match):
    document = raw_report(passed=False)
    mutation(document)
    with pytest.raises(reduce.ReduceError, match=match):
        reduce.reduce_document(document, "a" * 64)


def test_setup_failure_requires_no_mode_or_mutation():
    document = raw_report(passed=False)
    document.update(
        execution_state="quarantined_after_setup_failure",
        completed_mode_count=0,
        modes=[],
        active_mutation=None,
        remaining_modes=[mode for mode, _rank in reduce.MODE_ORDER],
        terminal_failure=None,
        setup_failure={"error": "remote-tool-attestation-failed", "detail": "private"},
    )
    receipt = reduce.reduce_document(document, "a" * 64)
    assert receipt["setup_error"] == "remote-tool-attestation-failed"
    assert receipt["modes"] == []
    assert receipt["active_mutation"] is None


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda value: value.update(remaining_modes=["bitflip"]),
            "no remaining modes",
        ),
        (
            lambda value: value.update(
                terminal_failure={
                    "mode": "fingerprint",
                    "rank": 3,
                    "error": "mode-gates-failed",
                }
            ),
            "cannot contain a terminal failure",
        ),
    ],
)
def test_completed_receipt_rejects_terminal_state_drift(mutation, match):
    document = raw_report()
    mutation(document)
    with pytest.raises(reduce.ReduceError, match=match):
        reduce.reduce_document(document, "a" * 64)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.update(execution_state="running"), "not terminal"),
        (lambda value: value["modes"][0].update(mode="private/path"), "out of order"),
        (lambda value: value["modes"][0].update(rank=3), "canonical"),
        (lambda value: value["modes"][0].update(error="private failure"), "unsupported"),
        (lambda value: value["modes"][0].update(engine_survived=1), "boolean"),
        (lambda value: value["modes"][0].update(damaged_context_digest="short"), "SHA-256"),
        (lambda value: value.update(completed_mode_count=0), "disagrees"),
        (lambda value: value.update(sabotage_halted=False), "sabotage_halted"),
        (lambda value: value["rank_targets"].update({"3": "private-rank2"}), "distinct"),
        (lambda value: value["evidence_scope"].update(publishable=True), "scope"),
        (lambda value: value["planned_modes"].reverse(), "planned modes"),
    ],
)
def test_reducer_rejects_false_or_private_passthrough(mutation, match):
    document = raw_report(passed=False)
    mutation(document)
    with pytest.raises(reduce.ReduceError, match=match):
        reduce.reduce_document(document, "a" * 64)


@pytest.mark.parametrize("payload", ['{"schema":"x","schema":"y"}', '{"forged":NaN}'])
def test_cli_rejects_duplicate_or_nonfinite_json(tmp_path, capsys, payload):
    source = tmp_path / "private.json"
    source.write_text(payload, encoding="utf-8")
    assert reduce.main(["--input", str(source)]) == 3
    assert "ERROR" in capsys.readouterr().err


def test_output_is_exclusive(tmp_path, capsys):
    source = tmp_path / "private.json"
    source.write_text(json.dumps(raw_report()), encoding="utf-8")
    output = tmp_path / "public.json"
    output.write_text("keep", encoding="utf-8")
    assert reduce.main(["--input", str(source), "--output", str(output)]) == 3
    assert output.read_text(encoding="utf-8") == "keep"
    capsys.readouterr()
