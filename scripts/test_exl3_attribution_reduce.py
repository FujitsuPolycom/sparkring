from __future__ import annotations

import hashlib
import json

import exl3_attribution_reduce as reduce
import pytest


def raw_report():
    ranks = {
        str(rank): {
            "exit_code": 0,
            "stdout": f"private-host-{rank}",
            "stderr": "private diagnostic",
        }
        for rank in range(4)
    }
    return {
        "plan": {
            "schema": reduce.RAW_PLAN_SCHEMA,
            "command": "transition",
            "arm": "b-mtp2-apc0-lmcache0",
            "from_arm": "a-mtp0-apc0-lmcache0",
            "functional_settings": {
                "mtp_tokens": 2,
                "native_prefix_cache": False,
                "lmcache_connector": False,
                "cache_boundary_geometry": {
                    "expected_engine_block_rows_per_dcp_rank": 64,
                    "expected_dcp_size": 4,
                    "expected_global_apc_alignment_tokens": 256,
                    "expected_lmcache_chunk_tokens_global": 512,
                    "runtime_attestation_required": True,
                    "recipe_predecessor_chunk_size_is_geometry_evidence": False,
                },
            },
            "canonical_attestation": {
                "profile_id": "glm52-exl3-tr3-3.25bpw-lmcache-cs512",
                "profile_file_sha256": "a" * 64,
                "image_id": "sha256:" + "b" * 64,
                "model_revision": "c" * 40,
            },
            "phases": [{"ssh_target": "private-host"}],
        },
        "results": {"remove_source_diagnostic": ranks},
        "original_exception": {
            "phase": "start_target_diagnostic",
            "type": "RuntimeError",
            "message": "private-host exploded",
        },
        "automatic_rollback": {"start_canonical": ranks},
    }


def test_reducer_binds_raw_and_omits_private_fields(tmp_path, capsys):
    source = tmp_path / "private.json"
    raw = json.dumps(raw_report()).encode()
    source.write_bytes(raw)
    output = tmp_path / "public.json"
    assert reduce.main(["--input", str(source), "--output", str(output)]) == 0
    rendered = capsys.readouterr().out
    assert output.read_text(encoding="utf-8") == rendered
    assert "private-host" not in rendered
    assert "stdout" not in rendered
    receipt = json.loads(rendered)
    assert receipt["private_raw_artifact_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["phase_outcomes"][0]["all_ranks_zero"] is True
    assert receipt["automatic_rollback_attempted"] is True
    assert receipt["raw_functional_settings_contract"] == "current-cache-geometry"
    assert receipt["startup_memory_hygiene"] == reduce.DISABLED_MEMORY_HYGIENE
    assert receipt["original_exception"]["message_sha256"] == hashlib.sha256(
        b"private-host exploded"
    ).hexdigest()


def test_reducer_rejects_non_four_rank_or_duplicate_input(tmp_path, capsys):
    document = raw_report()
    del document["results"]["remove_source_diagnostic"]["3"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    # A partial phase is retained as failed evidence, not promoted as success.
    assert reduce.main(["--input", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["phase_outcomes"][0]["all_ranks_zero"] is False
    path.write_text('{"plan":{},"plan":{}}', encoding="utf-8")
    assert reduce.main(["--input", str(path)]) == 3
    assert "duplicate JSON key" in capsys.readouterr().err


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_reducer_rejects_nonfinite_json_constants(tmp_path, capsys, constant):
    path = tmp_path / "bad.json"
    path.write_text(
        '{"plan": {}, "results": {}, "forged": ' + constant + "}",
        encoding="utf-8",
    )
    assert reduce.main(["--input", str(path)]) == 3
    assert "non-finite JSON number" in capsys.readouterr().err


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["plan"].__setitem__("command", "https://private/run"),
        lambda value: value["plan"].__setitem__("arm", "../../private"),
        lambda value: value["results"].__setitem__("/private/phase", {}),
        lambda value: value["original_exception"].__setitem__(
            "type", "PrivateCustomException"
        ),
        lambda value: value["plan"]["functional_settings"].__setitem__(
            "caller_note", "private prompt"
        ),
        lambda value: value["plan"]["functional_settings"].__setitem__(
            "native_prefix_cache", 0
        ),
        lambda value: value["plan"]["canonical_attestation"].__setitem__(
            "profile_id", "private-profile-/srv/models"
        ),
        lambda value: value["results"]["remove_source_diagnostic"]["0"].__setitem__(
            "exit_code", "private exit detail"
        ),
        lambda value: value["plan"].__setitem__("private_note", "/srv/private"),
        lambda value: value["original_exception"].__setitem__(
            "private_host", "rank0.private"
        ),
        lambda value: value.__setitem__("private_endpoint", "http://rank0:8000"),
    ],
)
def test_reducer_rejects_non_allowlisted_passthrough_values(
    tmp_path, capsys, mutate
):
    document = raw_report()
    mutate(document)
    path = tmp_path / "private.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert reduce.main(["--input", str(path)]) == 3
    assert "ERROR" in capsys.readouterr().err


def test_reducer_preserves_all_sanitized_rollback_failures(tmp_path, capsys):
    document = raw_report()
    document["automatic_rollback"] = {
        "remove_source_diagnostic": {"executor_exception": True},
        "start_canonical": {"malformed_executor_result": True},
        "canonical_engine_exclusive_after_restore": {
            str(rank): {"exit_code": 0, "stdout": "", "stderr": ""}
            for rank in range(4)
        },
        "rollback_exceptions": [
            {
                "phase": "remove_source_diagnostic",
                "type": "RuntimeError",
                "message": "private host cleanup failed",
            },
            {
                "phase": "start_canonical",
                "type": "ValueError",
                "message": "private malformed result",
            },
        ],
    }
    path = tmp_path / "private.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert reduce.main(["--input", str(path)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert len(receipt["automatic_rollback_exceptions"]) == 2
    assert {item["executor_result_status"] for item in receipt["automatic_rollback_outcomes"]} == {
        "executor-exception",
        "malformed-executor-result",
        "rank-results",
    }
    assert "private host" not in json.dumps(receipt)


def test_reducer_public_schema_audit_rejects_arbitrary_copied_strings():
    document = raw_report()
    raw = json.dumps(document).encode()
    receipt = reduce.reduce_document(document, hashlib.sha256(raw).hexdigest())
    receipt["scope"] = "private host and /srv/model"
    with pytest.raises(reduce.ReduceError, match="scope drifted"):
        reduce._validate_publishable_receipt(receipt)


@pytest.mark.parametrize(
    ("geometry", "contract"),
    [
        (None, "legacy-v1-no-cache-geometry"),
        (reduce.LEGACY_CACHE_GEOMETRY, "legacy-v2-cache-geometry"),
        (reduce.CURRENT_CACHE_GEOMETRY, "current-cache-geometry"),
    ],
)
def test_reducer_accepts_and_normalizes_historical_settings_contracts(
    geometry, contract
):
    document = raw_report()
    settings = document["plan"]["functional_settings"]
    if geometry is None:
        settings.pop("cache_boundary_geometry")
    else:
        settings["cache_boundary_geometry"] = dict(geometry)
    receipt = reduce.reduce_document(document, "a" * 64)
    assert receipt["raw_functional_settings_contract"] == contract
    assert receipt["functional_settings"] == reduce._expected_settings(
        "b-mtp2-apc0-lmcache0"
    )


def test_reducer_accepts_restart_arm_and_retains_enabled_memory_hygiene():
    document = raw_report()
    document["plan"].update(
        {
            "command": "restart-arm",
            "arm": "b-mtp2-apc0-lmcache0",
            "from_arm": "b-mtp2-apc0-lmcache0",
            "startup_memory_hygiene": dict(reduce.ENABLED_MEMORY_HYGIENE),
        }
    )
    receipt = reduce.reduce_document(document, "a" * 64)
    assert receipt["command"] == "restart-arm"
    assert receipt["startup_memory_hygiene"] == reduce.ENABLED_MEMORY_HYGIENE


def test_reducer_accepts_current_wrapper_preparation_phase_and_runtime_hash():
    document = raw_report()
    current = dict(reduce.ENABLED_MEMORY_HYGIENE)
    current["outer_verified_entrypoint_sha256"] = "d" * 64
    document["plan"]["startup_memory_hygiene"] = current
    document["results"]["prepare_page_cache_reclaim_entrypoint"] = document[
        "results"
    ]["remove_source_diagnostic"]
    receipt = reduce.reduce_document(document, "a" * 64)
    assert receipt["startup_memory_hygiene"] == current
    phases = {item["phase"] for item in receipt["phase_outcomes"]}
    assert "prepare_page_cache_reclaim_entrypoint" in phases


@pytest.mark.parametrize(
    ("legacy", "normalized"),
    [
        (reduce.LEGACY_DISABLED_MEMORY_HYGIENE, reduce.DISABLED_MEMORY_HYGIENE),
        (
            reduce.LEGACY_ENABLED_MEMORY_HYGIENE,
            reduce.LEGACY_ENABLED_MEMORY_HYGIENE,
        ),
    ],
)
def test_reducer_normalizes_exact_legacy_memory_hygiene(legacy, normalized):
    document = raw_report()
    document["plan"]["startup_memory_hygiene"] = dict(legacy)
    receipt = reduce.reduce_document(document, "a" * 64)
    assert receipt["startup_memory_hygiene"] == normalized


def test_reducer_rejects_inconsistent_or_arbitrary_memory_hygiene():
    document = raw_report()
    document["plan"]["startup_memory_hygiene"] = {
        "post_verification_host_page_cache_reclaim": True,
        "requires_passwordless_sudo": False,
        "safety_class": "private-host-action",
    }
    with pytest.raises(reduce.ReduceError, match="startup_memory_hygiene"):
        reduce.reduce_document(document, "a" * 64)
