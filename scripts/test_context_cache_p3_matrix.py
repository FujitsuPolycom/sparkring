import hashlib
import sys
import json
import inspect
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import context_cache_p3_matrix as matrix  # noqa: E402
import context_cache_sabotage as sabotage  # noqa: E402


def test_rank_targets_require_exact_distinct_dcp4_authorities():
    matrix.validate_rank_targets({0: "r0", 1: "r1", 2: "r2", 3: "r3"})
    for invalid in (
        {0: "r0", 1: "r1", 2: "r2"},
        {0: "r0", 1: "r1", 2: "r2", 4: "r4"},
        {0: "r0", 1: "r1", 2: "r2", 3: ""},
        {0: "r0", 1: "r1", 2: "r2", 3: "r2"},
    ):
        with pytest.raises(ValueError):
            matrix.validate_rank_targets(invalid)


def test_evidence_name_is_exclusive_and_checkpoints_are_atomic(
    monkeypatch, tmp_path
):
    output = tmp_path / "p3.json"
    initial = {"execution_state": "initializing", "modes": []}
    matrix.create_evidence_exclusive(output, initial)
    with pytest.raises(FileExistsError):
        matrix.create_evidence_exclusive(output, initial)

    replacements = []
    real_replace = matrix.os.replace

    def capture_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(matrix.os, "replace", capture_replace)
    updated = {"execution_state": "running", "modes": []}
    matrix.checkpoint_evidence_atomic(output, updated)
    assert json.loads(output.read_text(encoding="utf-8")) == updated
    assert len(replacements) == 1
    assert replacements[0][1] == output
    assert replacements[0][0].parent == output.parent
    assert not replacements[0][0].exists()


def test_failed_mode_checkpoint_terminally_quarantines_remaining_sabotage(tmp_path):
    output = tmp_path / "p3.json"
    report = {
        "execution_state": "running",
        "sabotage_halted": False,
        "modes": [],
    }
    matrix.create_evidence_exclusive(output, report)
    assert matrix.record_mode_checkpoint(
        report,
        {"mode": "bitflip", "rank": 2, "error": "api-not-healthy"},
        output,
        0,
    ) is False
    checkpoint = json.loads(output.read_text(encoding="utf-8"))
    assert checkpoint["execution_state"] == "quarantined_after_failure"
    assert checkpoint["sabotage_halted"] is True
    assert checkpoint["remaining_modes"] == ["truncate", "fingerprint"]
    assert checkpoint["terminal_failure"] == {
        "mode": "bitflip",
        "rank": 2,
        "error": "api-not-healthy",
    }
    assert len(checkpoint["modes"]) == 1


def test_rank_container_supports_rank_specific_candidate_names():
    assert (
        matrix.rank_container("glm52-sparkring-exl3-sparkcache-v51-r{rank}", 3)
        == "glm52-sparkring-exl3-sparkcache-v51-r3"
    )
    assert matrix.rank_container("common-name", 2) == "common-name"


def test_ssh_nonzero_is_fail_closed(monkeypatch):
    class Result:
        returncode = 23
        stdout = ""
        stderr = "denied"

    monkeypatch.setattr(matrix.subprocess, "run", lambda *_args, **_kwargs: Result())
    monkeypatch.setattr(matrix, "HOSTS", {0: "rank0"})
    with pytest.raises(RuntimeError, match="rc=23"):
        matrix.ssh(0, "true")


def test_manifest_counts_uses_explicit_quoted_cache_root(monkeypatch):
    monkeypatch.setattr(matrix, "HOSTS", {0: "r0", 1: "r1"})
    commands = []

    def fake_ssh(rank, command):
        commands.append((rank, command))
        return "3"

    monkeypatch.setattr(matrix, "ssh", fake_ssh)
    assert matrix.manifest_counts("/var/lib/cache with spaces") == {0: 3, 1: 3}
    assert len(commands) == 2
    assert all("'/var/lib/cache with spaces/manifests'" in item[1] for item in commands)
    assert all("$HOME" not in item[1] for item in commands)


def test_store_verifier_disables_bytecode_writes_in_attested_staging(monkeypatch):
    monkeypatch.setattr(matrix, "HOSTS", {0: "r0"})
    commands = []

    def fake_ssh(rank, command):
        commands.append((rank, command))
        return '{"entries": []}'

    monkeypatch.setattr(matrix, "ssh", fake_ssh)
    assert matrix.verify_stores(
        "/var/lib/sparkring/cache",
        "/opt/sparkring/staging/cache_manifest.py",
        "/tmp/context_cache_verify_store.py",
    ) == {0: {"entries": []}}
    assert commands == [
        (
            0,
            "python3 -B /tmp/context_cache_verify_store.py --store "
            "/var/lib/sparkring/cache --engine "
            "/opt/sparkring/staging/cache_manifest.py",
        )
    ]


def test_store_report_health_requires_every_payload_restore():
    assert matrix.store_report_healthy({"entries": []}) is False
    assert matrix.store_report_healthy(
        {"entries": [{"lookup": "hit", "restore": "ok"}]}
    ) is True
    assert matrix.store_report_healthy(
        {"entries": [{"lookup": "hit", "restore": "failed"}]}
    ) is False
    assert matrix.store_report_healthy({"error": "bad"}) is False


def test_digest_proof_is_exact_and_requires_one_healthy_match():
    digest = "a" * 64
    healthy = {
        "entries": [{"digest": digest, "lookup": "hit", "restore": "ok"}]
    }
    assert matrix.digest_entry_healthy(healthy, digest) is True
    assert matrix.digest_absent(healthy, digest) is False
    assert matrix.digest_absent({"entries": []}, digest) is True
    assert matrix.digest_entry_healthy({"entries": []}, digest) is False
    assert matrix.digest_entry_healthy(
        {"entries": healthy["entries"] * 2}, digest
    ) is False
    corrupt = {
        "entries": [{"digest": digest, "lookup": "corrupt", "restore": "failed"}]
    }
    assert matrix.digest_corrupt_present(corrupt, digest) is True
    assert matrix.digest_corrupt_present(healthy, digest) is False
    assert matrix.digest_corrupt_present({"entries": []}, digest) is False
    assert matrix.digest_corrupt_present(
        {"entries": corrupt["entries"] * 2}, digest
    ) is False


def _verified_entry(digest, rank=0):
    return {
        "digest": digest,
        "identity_key": f"identity-{rank}",
        "dcp_shard_rank": rank,
        "committed_tokens": 1024,
        "lookup": "hit",
        "restore": "ok",
        "position_count": 1024,
    }


def test_unrelated_entries_must_remain_identical_and_healthy():
    damaged = "a" * 64
    unrelated = "b" * 64
    before = {"entries": [_verified_entry(damaged), _verified_entry(unrelated)]}
    after = {"entries": [_verified_entry(unrelated)]}
    assert matrix.unrelated_entries_unchanged(before, after, damaged) is True

    changed = {"entries": [dict(_verified_entry(unrelated), committed_tokens=768)]}
    assert matrix.unrelated_entries_unchanged(before, changed, damaged) is False
    corrupt = {"entries": [dict(_verified_entry(unrelated), restore="failed")]}
    assert matrix.unrelated_entries_unchanged(before, corrupt, damaged) is False


def test_retirement_contract_withdraws_scheduler_and_damaged_worker(monkeypatch):
    digest = "c" * 64
    monkeypatch.setattr(matrix, "HOSTS", {0: "r0", 1: "r1", 2: "r2", 3: "r3"})
    reports = {
        0: {"entries": []},
        1: {"entries": [_verified_entry(digest, 1)]},
        2: {"entries": []},
        3: {"entries": [_verified_entry(digest, 3)]},
    }
    assert matrix.retirement_contract_met(reports, 2, digest) is True

    reports[0] = {"entries": [_verified_entry(digest, 0)]}
    assert matrix.retirement_contract_met(reports, 2, digest) is False
    reports[0] = {"entries": []}
    reports[1] = {"entries": []}
    assert matrix.retirement_contract_met(reports, 2, digest) is False
    assert matrix.retirement_contract_met(reports, 0, digest) is False


def test_recovery_requires_exact_digest_healthy_on_every_rank(monkeypatch):
    digest = "d" * 64
    monkeypatch.setattr(matrix, "HOSTS", {0: "r0", 1: "r1", 2: "r2", 3: "r3"})
    reports = {
        rank: {"entries": [_verified_entry(digest, rank)]} for rank in matrix.HOSTS
    }
    assert matrix.digest_healthy_on_all_ranks(reports, digest) is True
    reports[3] = {"entries": []}
    assert matrix.digest_healthy_on_all_ranks(reports, digest) is False


def test_new_digest_must_be_unique_healthy_shared_and_not_preexisting(monkeypatch):
    monkeypatch.setattr(matrix, "HOSTS", {0: "r0", 1: "r1", 2: "r2", 3: "r3"})
    old = "a" * 64
    new = "b" * 64
    before = {
        rank: {"entries": [_verified_entry(old, rank)]} for rank in matrix.HOSTS
    }
    after = {
        rank: {"entries": [_verified_entry(old, rank), _verified_entry(new, rank)]}
        for rank in matrix.HOSTS
    }
    assert matrix.common_new_healthy_digest(before, after) == new
    assert matrix.common_new_healthy_digest(before, after, {new}) is None
    after[3]["entries"][1]["restore"] = "failed"
    assert matrix.common_new_healthy_digest(before, after) is None


def test_seed_digest_is_bound_to_exact_request_and_all_rank_commit_events(monkeypatch):
    monkeypatch.setattr(matrix, "HOSTS", {0: "r0", 1: "r1", 2: "r2", 3: "r3"})
    target = "b" * 64
    unrelated = "c" * 64
    events = []
    for rank in matrix.HOSTS:
        events.extend(
            [
                {
                    "event": "store_committed",
                    "rank": rank,
                    "log_rank": rank,
                    "digest": unrelated,
                    "request_id": "concurrent-request",
                },
                {
                    "event": "store_committed",
                    "rank": rank,
                    "log_rank": rank,
                    "digest": target,
                    "request_id": "cmpl-target",
                },
            ]
        )
    assert matrix.committed_digest_for_request(events, "cmpl-target") == target
    assert matrix.committed_digest_for_request(events, "missing") is None
    assert matrix.committed_digest_for_request(events, None) is None

    events[-1]["digest"] = "d" * 64
    assert matrix.committed_digest_for_request(events, "cmpl-target") is None


def test_seed_digest_rejects_duplicate_rank_event_even_when_digest_matches(monkeypatch):
    monkeypatch.setattr(matrix, "HOSTS", {0: "r0", 1: "r1"})
    digest = "e" * 64
    events = [
        {
            "event": "store_committed",
            "rank": rank,
            "log_rank": rank,
            "digest": digest,
            "request_id": "cmpl-target",
        }
        for rank in matrix.HOSTS
    ]
    events.append(dict(events[0]))
    assert matrix.committed_digest_for_request(events, "cmpl-target") is None


def test_seed_digest_accepts_one_shared_internal_child_request_id(monkeypatch):
    monkeypatch.setattr(matrix, "HOSTS", {0: "r0", 1: "r1", 2: "r2", 3: "r3"})
    public = "cmpl-a6c9ec16baa5aee1"
    child = f"{public}-0-b11c9acd"
    digest = "a" * 64
    events = [
        {
            "event": "store_committed",
            "rank": rank,
            "log_rank": rank,
            "digest": digest,
            "request_id": child,
        }
        for rank in matrix.HOSTS
    ]

    binding = matrix.committed_request_binding(events, public)
    assert binding["passed"] is True
    assert binding["public_request_id"] == public
    assert binding["resolved_request_id"] == child
    assert binding["internal_child_request_id"] == child
    assert binding["digest"] == digest
    assert matrix.committed_digest_for_request(events, public) == digest


@pytest.mark.parametrize(
    "rank_request_ids",
    [
        # Two otherwise valid children are ambiguous.
        [
            "cmpl-public-0-aaaaaaaa",
            "cmpl-public-0-aaaaaaaa",
            "cmpl-public-0-bbbbbbbb",
            "cmpl-public-0-bbbbbbbb",
        ],
        # A public-ID prefix with a malformed child suffix is a collision.
        [
            "cmpl-public-0-aaaaaaaa",
            "cmpl-public-0-aaaaaaaa",
            "cmpl-public-0-nothex00",
            "cmpl-public-0-aaaaaaaa",
        ],
        # A longer ID sharing the public prefix cannot be treated as unrelated.
        [
            "cmpl-public-0-aaaaaaaa",
            "cmpl-public-0-aaaaaaaa",
            "cmpl-public-collision",
            "cmpl-public-0-aaaaaaaa",
        ],
    ],
)
def test_seed_digest_rejects_ambiguous_or_malformed_child_ids(
    monkeypatch, rank_request_ids
):
    monkeypatch.setattr(matrix, "HOSTS", {0: "r0", 1: "r1", 2: "r2", 3: "r3"})
    events = [
        {
            "event": "store_committed",
            "rank": rank,
            "log_rank": rank,
            "digest": "b" * 64,
            "request_id": rank_request_ids[rank],
        }
        for rank in matrix.HOSTS
    ]
    assert matrix.committed_digest_for_request(events, "cmpl-public") is None


def test_seed_digest_rejects_child_request_rank_log_spoof(monkeypatch):
    monkeypatch.setattr(matrix, "HOSTS", {0: "r0", 1: "r1", 2: "r2", 3: "r3"})
    public = "cmpl-public"
    child = f"{public}-0-deadbeef"
    events = [
        {
            "event": "store_committed",
            "rank": rank,
            "log_rank": rank,
            "digest": "c" * 64,
            "request_id": child,
        }
        for rank in matrix.HOSTS
    ]
    events.append(dict(events[0], rank=2))
    assert matrix.committed_digest_for_request(events, public) is None


def test_fire_places_nonce_inside_prompt_and_preserves_response_request_id(monkeypatch):
    captured = {}

    def fake_run(_base_url, prompt, max_tokens, model):
        captured.update(prompt=prompt, max_tokens=max_tokens, model=model)
        return {"request_id": "cmpl-exact", "completion_token_ids": [1]}

    monkeypatch.setattr(matrix, "run_request", fake_run)
    result = matrix.fire("http://rank0", 8, 7, "glm", "fresh-nonce")
    assert captured["prompt"].startswith("SparkRing P3 run nonce: fresh-nonce\n")
    assert captured["max_tokens"] == 48
    assert result["request_id"] == "cmpl-exact"


def test_cache_salt_derivation_is_deterministic_and_distinct_per_request_role():
    roles = (
        "target-seed",
        "sentinel-seed",
        "pre-sabotage-probe",
        "corruption-trigger",
        "recovery",
    )
    salts = {
        role: matrix.derive_cache_salt("run-nonce", "bitflip", role)
        for role in roles
    }

    assert len(set(salts.values())) == len(roles)
    assert all(len(salt) == 64 for salt in salts.values())
    assert salts["corruption-trigger"] == matrix.derive_cache_salt(
        "run-nonce", "bitflip", "corruption-trigger"
    )


def test_repeated_target_prompt_uses_distinct_cache_salts_without_prompt_drift(
    monkeypatch,
):
    calls = []

    def fake_run(_base_url, prompt, max_tokens, model, *, cache_salt=None):
        calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "model": model,
                "cache_salt": cache_salt,
            }
        )
        return {"request_id": f"cmpl-{len(calls)}", "completion_token_ids": [1]}

    monkeypatch.setattr(matrix, "run_request", fake_run)
    target_roles = (
        "target-seed",
        "pre-sabotage-probe",
        "corruption-trigger",
        "recovery",
    )
    salts = matrix.cache_salts_for_mode("run-nonce", "bitflip")
    for role in target_roles:
        matrix.fire(
            "http://rank0",
            8,
            7,
            "glm",
            "same-prompt-nonce",
            cache_salt=salts[role],
        )

    assert len({call["prompt"] for call in calls}) == 1
    assert [call["cache_salt"] for call in calls] == [
        salts[role] for role in target_roles
    ]
    assert len({call["cache_salt"] for call in calls}) == len(target_roles)
    # In particular, the post-damage trigger cannot be served from the APC
    # namespace populated by either pre-damage target request.
    assert salts["corruption-trigger"] not in {
        salts["target-seed"],
        salts["pre-sabotage-probe"],
    }


def test_cache_salt_receipts_are_complete_and_do_not_record_raw_values():
    salts = matrix.cache_salts_for_mode("private-run-nonce", "truncate")
    receipts = matrix.cache_salt_receipts(salts)

    assert list(receipts) == list(matrix.CACHE_SALT_ROLES)
    for role, salt in salts.items():
        assert receipts[role] == {
            "sha256": hashlib.sha256(salt.encode("utf-8")).hexdigest()
        }
    encoded = json.dumps(receipts, sort_keys=True)
    assert all(salt not in encoded for salt in salts.values())


def test_retirement_events_bind_full_digest_rank_request_and_republish_order():
    digest = "c" * 64
    request = "req-123"
    logs = {
        0: "\n".join(
            [
                "spark-context-cache-event/v1 event=scheduler_retired "
                f"rank=0 digest={digest} request_id={request}",
                "spark-context-cache-event/v1 event=store_committed "
                f"rank=0 digest={digest} request_id={request}",
            ]
        ),
        2: "\n".join(
            [
                "spark-context-cache-event/v1 event=worker_invalidated "
                f"rank=2 digest={digest} request_id={request}",
                "spark-context-cache-event/v1 event=store_committed "
                f"rank=2 digest={digest} request_id={request}",
            ]
        ),
    }
    events = matrix.parse_connector_events(logs)
    proof = matrix.retirement_event_proof(events, 2, digest, request)
    assert proof["passed"] is True
    assert proof["request_ids"] == [request]

    reversed_logs = dict(logs)
    reversed_logs[2] = "\n".join(reversed(logs[2].splitlines()))
    assert matrix.retirement_event_proof(
        matrix.parse_connector_events(reversed_logs), 2, digest, request
    )["passed"] is False


def test_retirement_events_resolve_one_shared_internal_child_id():
    digest = "9" * 64
    public = "cmpl-trigger"
    child = f"{public}-0-b11c9acd"
    logs = {
        0: "\n".join(
            [
                "spark-context-cache-event/v1 event=scheduler_retired "
                f"rank=0 digest={digest} request_id={child}",
                "spark-context-cache-event/v1 event=store_committed "
                f"rank=0 digest={digest} request_id={child}",
            ]
        ),
        2: "\n".join(
            [
                "spark-context-cache-event/v1 event=worker_invalidated "
                f"rank=2 digest={digest} request_id={child}",
                "spark-context-cache-event/v1 event=store_committed "
                f"rank=2 digest={digest} request_id={child}",
            ]
        ),
    }

    proof = matrix.retirement_event_proof(
        matrix.parse_connector_events(logs), 2, digest, public
    )
    assert proof["passed"] is True
    assert proof["public_request_id"] == public
    assert proof["resolved_request_id"] == child
    assert proof["internal_child_request_id"] == child
    assert proof["request_ids"] == [child]


def test_retirement_events_reject_different_internal_children_across_ranks():
    digest = "8" * 64
    public = "cmpl-trigger"
    rank0_child = f"{public}-0-aaaaaaaa"
    rank2_child = f"{public}-0-bbbbbbbb"
    logs = {
        0: "\n".join(
            [
                "spark-context-cache-event/v1 event=scheduler_retired "
                f"rank=0 digest={digest} request_id={rank0_child}",
                "spark-context-cache-event/v1 event=store_committed "
                f"rank=0 digest={digest} request_id={rank0_child}",
            ]
        ),
        2: "\n".join(
            [
                "spark-context-cache-event/v1 event=worker_invalidated "
                f"rank=2 digest={digest} request_id={rank2_child}",
                "spark-context-cache-event/v1 event=store_committed "
                f"rank=2 digest={digest} request_id={rank2_child}",
            ]
        ),
    }
    proof = matrix.retirement_event_proof(
        matrix.parse_connector_events(logs), 2, digest, public
    )
    assert proof["passed"] is False
    assert proof["reason"] == "trigger-request-id-resolution-failed"


def test_retirement_events_reject_stale_or_cross_request_pairing():
    digest = "d" * 64
    logs = {
        0: (
            "spark-context-cache-event/v1 event=scheduler_retired "
            f"rank=0 digest={digest} request_id=req-a\n"
            "spark-context-cache-event/v1 event=store_committed "
            f"rank=0 digest={digest} request_id=req-a"
        ),
        2: (
            "spark-context-cache-event/v1 event=worker_invalidated "
            f"rank=2 digest={digest} request_id=req-b\n"
            "spark-context-cache-event/v1 event=store_committed "
            f"rank=2 digest={digest} request_id=req-b"
        ),
    }
    assert matrix.retirement_event_proof(
        matrix.parse_connector_events(logs), 2, digest, "req-a"
    )["passed"] is False


def test_retirement_events_cannot_substitute_later_recovery_request():
    digest = "e" * 64
    trigger = "cmpl-trigger"
    recovery = "cmpl-recovery"
    logs = {
        0: "\n".join(
            [
                "spark-context-cache-event/v1 event=scheduler_retired "
                f"rank=0 digest={digest} request_id={recovery}",
                "spark-context-cache-event/v1 event=store_committed "
                f"rank=0 digest={digest} request_id={recovery}",
            ]
        ),
        2: "\n".join(
            [
                "spark-context-cache-event/v1 event=worker_invalidated "
                f"rank=2 digest={digest} request_id={recovery}",
                "spark-context-cache-event/v1 event=store_committed "
                f"rank=2 digest={digest} request_id={recovery}",
            ]
        ),
    }
    proof = matrix.retirement_event_proof(
        matrix.parse_connector_events(logs), 2, digest, trigger
    )
    assert proof["passed"] is False
    assert proof["request_ids"] == [trigger]


@pytest.mark.parametrize("request_id", [None, "", "contains whitespace", "!bad"])
def test_retirement_events_reject_missing_or_malformed_trigger_id(request_id):
    proof = matrix.retirement_event_proof([], 2, "f" * 64, request_id)
    assert proof == {
        "passed": False,
        "request_ids": [],
        "reason": "missing-or-malformed-trigger-request-id",
    }


def test_retirement_events_reject_extra_same_request_rank_spoof():
    digest = "a" * 64
    request = "cmpl-trigger"
    logs = {
        0: "\n".join(
            [
                "spark-context-cache-event/v1 event=scheduler_retired "
                f"rank=0 digest={digest} request_id={request}",
                "spark-context-cache-event/v1 event=store_committed "
                f"rank=0 digest={digest} request_id={request}",
                # Rank 2 is falsely declared by the rank-0 log. This must
                # poison the entire trigger evidence envelope.
                "spark-context-cache-event/v1 event=worker_invalidated "
                f"rank=2 digest={digest} request_id={request}",
            ]
        ),
        2: "\n".join(
            [
                "spark-context-cache-event/v1 event=worker_invalidated "
                f"rank=2 digest={digest} request_id={request}",
                "spark-context-cache-event/v1 event=store_committed "
                f"rank=2 digest={digest} request_id={request}",
            ]
        ),
    }
    proof = matrix.retirement_event_proof(
        matrix.parse_connector_events(logs), 2, digest, request
    )
    assert proof["passed"] is False
    assert proof["reason"] == "trigger-request-rank-log-mismatch"
    assert len(proof["malformed_events"]) == 1


def test_sabotage_report_requires_json_mode_damage_and_sha256_digest():
    digest = "b" * 64
    valid = {
        "mode": "bitflip",
        "digest": digest,
        "manifest": "/private/cache/manifest.json",
        "chunk": "/private/cache/chunk.spcc",
        "damaged": "bit flipped",
        "post_damage_lookup": "corrupt",
        "post_damage_probe": "hit",
    }
    report = matrix.parse_sabotage_report(
        json.dumps(valid),
        "bitflip",
    )
    assert report["digest"] == digest
    for invalid in (
        "damaged but not json",
        json.dumps({**valid, "mode": "truncate"}),
        json.dumps({**valid, "digest": "prefix"}),
        json.dumps({key: value for key, value in valid.items() if key != "damaged"}),
        json.dumps({**valid, "unexpected": "field"}),
        (
            '{"mode":"wrong","mode":"bitflip","digest":"'
            + digest
            + '","manifest":"/m","chunk":"/c","damaged":"yes",'
            '"post_damage_lookup":"corrupt","post_damage_probe":"hit"}'
        ),
        json.dumps(valid)[:-1] + ',"forged":NaN}',
    ):
        with pytest.raises(RuntimeError):
            matrix.parse_sabotage_report(invalid, "bitflip")


def test_report_encoding_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        matrix._encoded_report({"forged": float("nan")})


def test_applied_sabotage_checkpoint_is_immediately_actionable(tmp_path):
    output = tmp_path / "report.json"
    report = {"active_mutation": None}
    matrix.create_evidence_exclusive(output, report)
    matrix.record_mutation_checkpoint(
        report,
        output,
        phase="sabotage-applied",
        mode="bitflip",
        rank=2,
        digest="d" * 64,
        sabotage_applied=True,
        recovery_required=True,
    )
    checkpoint = json.loads(output.read_text(encoding="utf-8"))
    assert checkpoint["active_mutation"] == {
        "phase": "sabotage-applied",
        "mode": "bitflip",
        "rank": 2,
        "digest": "d" * 64,
        "sabotage_applied": True,
        "recovery_required": True,
    }


def test_sabotage_checkpoint_precedes_first_post_damage_verification():
    source = inspect.getsource(matrix._main)
    armed = source.index('phase="sabotage-command-armed"')
    command = source.index("damage_output = ssh", armed)
    applied = source.index('phase="sabotage-applied"')
    corrupt = source.index("corrupt_verification = verify_stores", applied)
    assert armed < command < applied < corrupt


def test_sabotage_targets_only_the_explicit_digest(monkeypatch, tmp_path, capsys):
    class Identity:
        def __init__(self, **values):
            self.values = values

    class Store:
        def __init__(self, _root):
            pass

        def lookup(self, _identity, _digest, verify_chunks=True):
            reason = "corrupt" if verify_chunks else "hit"
            return types.SimpleNamespace(reason=reason)

    monkeypatch.setattr(
        sabotage,
        "load_engine",
        lambda _path: types.SimpleNamespace(CacheIdentity=Identity, ManifestStore=Store),
    )
    first = "a" * 64
    target = "b" * 64
    manifests = tmp_path / "manifests" / "identity"
    manifests.mkdir(parents=True)
    for digest in (first, target):
        (manifests / f"{digest}.json").write_text(
            json.dumps(
                {
                    "identity": {"quantization_layout": "expected"},
                    "context_digest": digest,
                    "chunks": [],
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "context_cache_sabotage.py",
            "--store",
            str(tmp_path),
            "--engine",
            str(tmp_path / "engine.py"),
            "--mode",
            "fingerprint",
            "--digest",
            target,
        ],
    )
    assert sabotage.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["digest"] == target
    assert json.loads((manifests / f"{first}.json").read_text())["identity"][
        "quantization_layout"
    ] == "expected"
    assert json.loads((manifests / f"{target}.json").read_text())["identity"][
        "quantization_layout"
    ] == "mismatched-layout-v0"


def test_p3_orders_corrupt_presence_connector_withdrawal_then_recovery():
    source = inspect.getsource(matrix._main)
    corrupt = source.index("corrupt_verification = verify_stores")
    trigger = source.index("result = fire", corrupt)
    absent = source.index("withdrawn = digest_absent", trigger)
    recovery = source.index("recovery = fire", absent)
    final = source.index("final_verification = verify_stores", recovery)
    assert corrupt < trigger < absent < recovery < final
    assert "sabotage helper's verified lookup must retire" not in source
    assert '"--digest"' in source


def test_mode_pass_requires_healing_isolation_and_no_request_error():
    good = {
        "engine_survived": True,
        "no_wrong_output": True,
        "self_healed": True,
        "others_intact": True,
        "recovery_ok": True,
        "request_error": None,
    }
    assert matrix.mode_pass(good) is True
    for field in ("self_healed", "others_intact", "no_wrong_output"):
        bad = dict(good, **{field: False})
        assert matrix.mode_pass(bad) is False
    assert matrix.mode_pass(dict(good, request_error="HTTP 500")) is False


def test_default_is_dry_run_and_never_calls_ssh(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(matrix, "ssh", lambda *_args: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "context_cache_p3_matrix.py",
            "--cache-root",
            "/srv/cache",
            "--engine",
            "/srv/cache_manifest.py",
            "--container-pattern",
            "candidate-r{rank}",
            "--output",
            str(tmp_path / "report.json"),
        ],
    )
    assert matrix.main() == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert plan["sensitivity"] == "private-do-not-publish"
    assert plan["confirmation_required"] == matrix.CONFIRMATION
    assert len(plan["sabotage_sha256"]) == 64
    assert len(plan["verify_sha256"]) == 64
    assert plan["max_tokens"] == 48
    assert plan["cache_salt_contract"] == matrix.cache_salt_contract()
    assert plan["cache_salt_contract"]["request_roles"] == list(
        matrix.CACHE_SALT_ROLES
    )
    assert plan["cache_salt_contract"]["raw_values_recorded"] is False
    assert "run_nonce" not in plan
    assert not (tmp_path / "report.json").exists()


def test_execute_requires_exact_confirmation(monkeypatch, tmp_path):
    monkeypatch.setattr(matrix, "ssh", lambda *_args: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "context_cache_p3_matrix.py",
            "--cache-root",
            "/srv/cache",
            "--engine",
            "/srv/cache_manifest.py",
            "--container-pattern",
            "candidate-r{rank}",
            "--output",
            str(tmp_path / "report.json"),
            "--execute",
            "--confirmation",
            "WRONG",
        ],
    )
    with pytest.raises(SystemExit):
        matrix.main()


def test_execute_stops_after_first_failed_mode_and_checkpoints(monkeypatch, tmp_path):
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        matrix, "HOSTS", {0: "rank0", 1: "rank1", 2: "rank2", 3: "rank3"}
    )
    monkeypatch.setattr(
        matrix,
        "attest_remote_tools",
        lambda _verify: {rank: {"status": "attested"} for rank in range(4)},
    )
    health_calls = []
    monkeypatch.setattr(
        matrix,
        "wait_healthy",
        lambda _url: health_calls.append(True) or False,
    )
    monkeypatch.setattr(
        matrix,
        "fire",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("later sabotage setup must not run")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "context_cache_p3_matrix.py",
            "--cache-root",
            "/srv/cache",
            "--engine",
            "/srv/cache_manifest.py",
            "--container-pattern",
            "candidate-r{rank}",
            "--output",
            str(output),
            "--execute",
            "--confirmation",
            matrix.CONFIRMATION,
        ],
    )
    assert matrix.main() == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert health_calls == [True]
    assert [entry["mode"] for entry in report["modes"]] == ["bitflip"]
    assert report["execution_state"] == "quarantined_after_failure"
    assert report["remaining_modes"] == ["truncate", "fingerprint"]
    assert report["passed"] is False


def test_main_has_no_continue_path_to_a_later_sabotage_mode():
    assert "continue" not in inspect.getsource(matrix._main)


def test_unexpected_mode_exception_is_checkpointed_and_stops_matrix(
    monkeypatch, tmp_path
):
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        matrix, "HOSTS", {0: "rank0", 1: "rank1", 2: "rank2", 3: "rank3"}
    )
    monkeypatch.setattr(
        matrix,
        "attest_remote_tools",
        lambda _verify: {rank: {"status": "attested"} for rank in range(4)},
    )
    monkeypatch.setattr(matrix, "wait_healthy", lambda _url: True)
    verification_calls = []

    def fail_verification(*_args):
        verification_calls.append(True)
        raise RuntimeError("rank 2 verifier disconnected")

    monkeypatch.setattr(matrix, "verify_stores", fail_verification)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "context_cache_p3_matrix.py",
            "--cache-root",
            "/srv/cache",
            "--engine",
            "/srv/cache_manifest.py",
            "--container-pattern",
            "candidate-r{rank}",
            "--output",
            str(output),
            "--execute",
            "--confirmation",
            matrix.CONFIRMATION,
        ],
    )
    assert matrix.main() == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert verification_calls == [True]
    assert len(report["modes"]) == 1
    assert report["modes"][0]["mode"] == "bitflip"
    assert report["modes"][0]["error"] == "unexpected-mode-exception"
    assert report["execution_state"] == "quarantined_after_failure"
    assert report["remaining_modes"] == ["truncate", "fingerprint"]
    assert report["cache_salt_contract"] == matrix.cache_salt_contract()
    salts = matrix.cache_salts_for_mode(report["run_nonce"], "bitflip")
    assert report["cache_salt_receipts"]["bitflip"] == matrix.cache_salt_receipts(
        salts
    )
    encoded_receipts = json.dumps(report["cache_salt_receipts"], sort_keys=True)
    assert all(salt not in encoded_receipts for salt in salts.values())


def test_keyboard_interrupt_is_checkpointed_before_propagation(monkeypatch, tmp_path):
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        matrix, "HOSTS", {0: "rank0", 1: "rank1", 2: "rank2", 3: "rank3"}
    )
    monkeypatch.setattr(
        matrix,
        "attest_remote_tools",
        lambda _verify: {rank: {"status": "attested"} for rank in range(4)},
    )
    monkeypatch.setattr(matrix, "wait_healthy", lambda _url: True)
    monkeypatch.setattr(
        matrix,
        "verify_stores",
        lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "context_cache_p3_matrix.py",
            "--cache-root",
            "/srv/cache",
            "--engine",
            "/srv/cache_manifest.py",
            "--container-pattern",
            "candidate-r{rank}",
            "--output",
            str(output),
            "--execute",
            "--confirmation",
            matrix.CONFIRMATION,
        ],
    )
    with pytest.raises(KeyboardInterrupt):
        matrix.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["execution_state"] == "quarantined_after_failure"
    assert report["sabotage_halted"] is True
    assert report["modes"][0]["error"] == "unexpected-mode-exception"
    assert report["remaining_modes"] == ["truncate", "fingerprint"]
