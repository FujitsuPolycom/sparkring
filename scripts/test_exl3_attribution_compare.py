from __future__ import annotations

import hashlib
import json

import exl3_attribution_compare as compare
import pytest
from exl3_attribution_cache_contract import (
    build_live_arm_receipt,
    cache_salt_for_arm,
)


ATTRIBUTION_ARM = "d-mtp2-apc1-lmcache1"
CACHE_SALT = cache_salt_for_arm(ATTRIBUTION_ARM)


def token_hash(token_ids):
    return hashlib.sha256(
        json.dumps(token_ids, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def logprobs(*, second=" beta", second_lp=-1.2, alternate_lp=-2.0):
    return {
        "tokens": [" alpha", second],
        "token_logprobs": [-0.1, second_lp],
        "top_logprobs": [
            {" alpha": -0.1, " gamma": -2.5},
            {second: second_lp, " delta": alternate_lp},
        ],
        "text_offset": [0, 6],
    }


def report(run_label, sequences, *, logs=True):
    observations = []
    for repetition, token_ids in enumerate(sequences, 1):
        observations.append(
            {
                "repetition": repetition,
                "run_position": "first-in-run" if repetition == 1 else "subsequent-in-run",
                "token_ids": token_ids,
                "token_ids_sha256": token_hash(token_ids),
                "text_sha256": "a" * 64,
                "completion_logprobs": logprobs() if logs else None,
                "request_evidence": {
                    "response_id_sha256": "e" * 64,
                    "usage_prompt_tokens": 1024,
                    "usage_cached_prompt_tokens": None,
                    "hit_evidence_source": None,
                    "store_evidence_source": None,
                },
            }
        )
    live_receipt = build_live_arm_receipt(
        arm_id=ATTRIBUTION_ARM,
        canonical_profile_id="glm52-exl3-tr3-3.25bpw-lmcache-cs512",
        canonical_profile_file_sha256="9" * 64,
        image_id="sha256:" + "a" * 64,
        model_repository="willfalco/GLM-5.2-EXL3-TR3-3.25bpw",
        model_revision="b" * 40,
        canonical_container_name="glm52-sparkring-exl3-lmcache-cs512",
        explicit_environment_sha256=[
            hashlib.sha256(f"env-{rank}".encode()).hexdigest()
            for rank in range(4)
        ],
        config_cmd_sha256=[
            hashlib.sha256(f"cmd-{rank}".encode()).hexdigest()
            for rank in range(4)
        ],
        observed_runtime_instances=[
            {
                "container_id": f"{rank + 1:064x}",
                "started_at": f"2026-08-10T03:2{rank}:00.123456789Z",
            }
            for rank in range(4)
        ],
    )
    live_receipt["artifact_sha256"] = "8" * 64
    return {
        "schema": compare.INPUT_SCHEMA,
        "status": "pass",
        "model": "glm-5.2-exl3-tr3-3.25bpw",
        "run_label": run_label,
        "attribution_arm": ATTRIBUTION_ARM,
        "cache_salt": CACHE_SALT,
        "live_arm_receipt": live_receipt,
        "config_sha256": "b" * 64,
        "repetitions": len(sequences),
        "top_logprobs": 2 if logs else 0,
        "cases": [
            {
                "id": "focused-case",
                "status": "pass",
                "prompt_sha256": "c" * 64,
                "prompt_token_ids_sha256": "d" * 64,
                "prompt_token_count": 1024,
                "ignore_eos": True,
                "cache_attribution": True,
                "cache_boundaries": {
                    "physical_block_tokens_per_dcp_rank": 64,
                    "dcp_degree": 4,
                    "dcp_global_apc_alignment_tokens": 256,
                    "lmcache_chunk_tokens": 512,
                    "minimum_prompt_tokens": 1024,
                    "reusable_prompt_tokens": 1023,
                    "reusable_dcp_global_apc_units": 3,
                    "reusable_lmcache_chunks": 1,
                    "has_reusable_dcp_global_apc_unit": True,
                    "has_reusable_lmcache_chunk": True,
                    "qualifies_for_cache_attribution": True,
                },
                "cache_metric_probe": {
                    "artifact_sha256": "f" * 64,
                    "schema": "sparkring-exl3-cache-metric-probe/v2",
                    "run_label": "test-probe",
                    "model": "glm-5.2-exl3-tr3-3.25bpw",
                    "attribution_arm": ATTRIBUTION_ARM,
                    "cache_salt": CACHE_SALT,
                    "live_arm_receipt_sha256": live_receipt[
                        "artifact_sha256"
                    ],
                    "live_arm_receipt": live_receipt,
                    "probe_prompt_sha256": "c" * 64,
                    "probe_prompt_token_count": 1024,
                    "probe_prompt_token_ids_sha256": "d" * 64,
                    "geometry": {
                        "physical_block_tokens_per_dcp_rank": 64,
                        "dcp_degree": 4,
                        "dcp_global_apc_alignment_tokens": 256,
                        "lmcache_chunk_tokens": 512,
                        "physical_blocks_per_dcp_global_apc_unit_per_rank": 1,
                        "dcp_global_apc_units_per_lmcache_chunk": 2,
                        "physical_blocks_per_lmcache_chunk_per_rank": 2,
                    },
                    "native_prefix_caching_enabled": True,
                    "aggregate_counter_deltas": {
                        "vllm:prefix_cache_queries_total": 2.0,
                        "vllm:prefix_cache_hits_total": 0.0,
                        "vllm:external_prefix_cache_queries_total": 2.0,
                        "vllm:external_prefix_cache_hits_total": 0.0,
                    },
                    "aggregate_prompt_tokens_by_source": {
                        "local_compute": 2048.0,
                        "local_cache_hit": 0.0,
                        "external_kv_transfer": 0.0,
                    },
                    "observed_cache_layers": [],
                    "observation_count": 2,
                    "evidence_classification": "request-interval-correlated-prometheus-counter-delta",
                },
                "cache_evidence_scope": {
                    "boundary_geometry_bound": True,
                    "request_ids_bound": 2,
                    "request_count": 2,
                    "request_correlated_hit_evidence_count": 0,
                    "request_correlated_store_evidence_count": 0,
                    "causal_cache_claim": "not-claimed-store-evidence-unavailable",
                    "statement": "No store evidence is available.",
                },
                "seed": 20260809,
                "max_tokens": 128,
                "requested_top_logprobs": 2 if logs else 0,
                "token_id_source": "retokenized-completion-text",
                "observations": observations,
            }
        ],
    }


def write_report(tmp_path, name, document):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def run_pair(tmp_path, capsys, left, right, *, output=None, allow_legacy=False):
    left_path = write_report(tmp_path, "left.json", left)
    right_path = write_report(tmp_path, "right.json", right)
    args = ["--left", str(left_path), "--right", str(right_path)]
    if output is not None:
        args += ["--output", str(output)]
    if allow_legacy:
        args.append("--allow-legacy-v1")
    status = compare.main(args)
    captured = capsys.readouterr()
    return status, captured


def test_exact_and_divergent_sequences_are_reported_without_failure(tmp_path, capsys):
    left = report("arm-a", [[10, 20, 30], [10, 20, 31]])
    right = report("arm-b", [[10, 20, 30], [10, 99]])
    status, captured = run_pair(tmp_path, capsys, left, right)
    assert status == compare.EXIT_OK
    result = json.loads(captured.out)
    assert result["comparison_status"] == "functional-divergence-observed"
    assert result["summary"] == {
        "between_arm_repetition_pair_count": 2,
        "exact_match_pair_count": 1,
        "divergent_pair_count": 1,
    }
    case = result["cases"][0]
    assert case["within_arm"]["left"]["stable_across_repetitions"] is False
    assert case["within_arm"]["right"]["distinct_sequence_count"] == 2
    divergent = case["between_arms_by_repetition"][1]
    assert divergent["exact_match"] is False
    assert divergent["first_divergence_index"] == 1
    assert divergent["prefix_match_length"] == 1
    assert divergent["left_sequence_length"] == 3
    assert divergent["right_sequence_length"] == 2
    assert divergent["left_token_ids_sha256"] == token_hash([10, 20, 31])
    assert divergent["right_token_ids_sha256"] == token_hash([10, 99])


def test_logprob_metrics_are_truncated_positional_and_sanitized(tmp_path, capsys):
    left = report("arm-a", [[10, 20], [10, 20]])
    right = report("arm-b", [[10, 20], [10, 20]])
    right_logs = right["cases"][0]["observations"][0]["completion_logprobs"]
    right_logs["tokens"][1] = " omega"
    right_logs["token_logprobs"][1] = -1.5
    right_logs["top_logprobs"][1] = {" beta": -1.4, " delta": -1.7}
    status, captured = run_pair(tmp_path, capsys, left, right)
    assert status == 0
    result = json.loads(captured.out)
    scope = result["distribution_evidence_scope"]
    assert scope["is_full_vocabulary_kld"] is False
    assert "not full-vocabulary KL" in scope["statement"]
    evidence = result["cases"][0]["between_arms_by_repetition"][0]["completion_logprobs"]
    assert evidence["alignment"].startswith("same-context positions only")
    assert evidence["aligned_position_count"] == 1
    assert evidence["later_position_policy"] == "omitted-different-context-or-unproven"
    matching_position = evidence["positions"][0]
    assert matching_position["chosen_token_logprob_delta_left_minus_right"] == pytest.approx(0.0)
    rendered = captured.out
    assert " omega" not in rendered
    assert " beta" not in rendered
    assert "http://" not in rendered
    assert "prompt_sha256" not in rendered
    assert "text_sha256" not in rendered


def test_no_logprobs_is_explicitly_unavailable(tmp_path, capsys):
    status, captured = run_pair(
        tmp_path,
        capsys,
        report("arm-a", [[1], [1]], logs=False),
        report("arm-b", [[1], [1]], logs=False),
    )
    assert status == 0
    evidence = json.loads(captured.out)["cases"][0]["between_arms_by_repetition"][0]["completion_logprobs"]
    assert evidence == {
        "available": False,
        "reason": "top-logprobs-not-persisted",
    }


def test_logprobs_after_retokenized_divergence_are_omitted_even_if_strings_match(
    tmp_path, capsys
):
    left = report("arm-a", [[10, 20], [10, 20]])
    right = report("arm-b", [[99, 20], [99, 20]])
    status, captured = run_pair(tmp_path, capsys, left, right)
    assert status == 0
    evidence = json.loads(captured.out)["cases"][0]["between_arms_by_repetition"][0][
        "completion_logprobs"
    ]
    assert evidence["aligned_position_count"] == 0
    assert evidence["positions"] == []
    assert evidence["different_context_or_unproven_position_count_left"] == 2


def test_older_v1_optional_prompt_token_metadata_is_accepted_but_disclosed(
    tmp_path, capsys
):
    left = report("arm-new", [[1], [1]], logs=False)
    right = report("arm-old", [[1], [1]], logs=False)
    left["schema"] = compare.LEGACY_INPUT_SCHEMA
    right["schema"] = compare.LEGACY_INPUT_SCHEMA
    del right["cases"][0]["prompt_token_count"]
    del right["cases"][0]["prompt_token_ids_sha256"]
    status, captured = run_pair(tmp_path, capsys, left, right)
    assert status == compare.EXIT_INVALID
    assert "requires explicit --allow-legacy-v1" in captured.err
    status, captured = run_pair(
        tmp_path / "explicit", capsys, left, right, allow_legacy=True
    )
    assert status == 0
    result = json.loads(captured.out)
    assert result["alignment"]["current_format_alignment_complete"] is False
    optional = result["cases"][0]["optional_alignment_evidence"]
    assert optional["prompt_token_count"] == {
        "left_present": True,
        "right_present": False,
        "compared_when_present_in_both": False,
    }


def test_legacy_v1_v2_pair_is_explicitly_token_only_and_geometry_unbound(
    tmp_path, capsys
):
    left = report("arm-v1", [[1], [1]], logs=False)
    right = report("arm-v2", [[1], [1]], logs=False)
    left["schema"] = compare.LEGACY_INPUT_SCHEMA
    right["schema"] = compare.LEGACY_INPUT_SCHEMA_V2
    status, captured = run_pair(
        tmp_path, capsys, left, right, allow_legacy=True
    )
    assert status == 0
    result = json.loads(captured.out)
    assert result["alignment"]["cache_geometry_evidence_scope"] == (
        "unbound-legacy-token-output-only"
    )
    assert result["legacy_evidence_scope"]["cache_geometry_bound"] is False
    assert result["legacy_evidence_scope"]["scope"] == "token-output-only"


def test_non_cache_v4_case_does_not_claim_metric_probe_bound_geometry(
    tmp_path, capsys
):
    left = report("arm-a", [[1], [1]], logs=False)
    right = report("arm-b", [[1], [1]], logs=False)
    for document in (left, right):
        case = document["cases"][0]
        case["cache_attribution"] = False
        case["cache_metric_probe"] = None
        case["cache_boundaries"]["minimum_prompt_tokens"] = None
        case["cache_boundaries"]["qualifies_for_cache_attribution"] = False
        case["cache_evidence_scope"]["boundary_geometry_bound"] = False
    status, captured = run_pair(tmp_path, capsys, left, right)
    assert status == compare.EXIT_OK
    result = json.loads(captured.out)
    expected = "non-cache-v4-cache-namespace-bound-no-cache-geometry-claim"
    assert result["alignment"]["cache_geometry_evidence_scope"] == expected
    assert result["cases"][0]["cache_geometry_evidence"]["scope"] == expected
    assert result["cases"][0]["cache_geometry_evidence"][
        "left_metric_probe_artifact_sha256"
    ] is None


def test_mixed_v4_report_uses_case_specific_geometry_scope(tmp_path, capsys):
    left = report("arm-a", [[1], [1]], logs=False)
    right = report("arm-b", [[1], [1]], logs=False)
    for document in (left, right):
        non_cache = json.loads(json.dumps(document["cases"][0]))
        non_cache["id"] = "non-cache-case"
        non_cache["cache_attribution"] = False
        non_cache["cache_metric_probe"] = None
        non_cache["cache_boundaries"]["minimum_prompt_tokens"] = None
        non_cache["cache_boundaries"]["qualifies_for_cache_attribution"] = False
        non_cache["cache_evidence_scope"]["boundary_geometry_bound"] = False
        document["cases"].append(non_cache)
    status, captured = run_pair(tmp_path, capsys, left, right)
    assert status == compare.EXIT_OK
    result = json.loads(captured.out)
    assert result["alignment"]["cache_geometry_evidence_scope"] == (
        "case-specific-see-cases"
    )
    assert [case["cache_geometry_evidence"]["scope"] for case in result["cases"]] == [
        "validated-metric-probe-and-cache-namespace-bound-v4",
        "non-cache-v4-cache-namespace-bound-no-cache-geometry-claim",
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_token_count", None),
        ("prompt_token_ids_sha256", None),
        ("ignore_eos", None),
        ("cache_attribution", None),
        ("cache_boundaries", None),
        ("cache_evidence_scope", None),
        ("cache_metric_probe", None),
    ],
)
def test_current_format_requires_complete_cache_alignment_metadata(
    tmp_path, capsys, field, value
):
    left = report("arm-a", [[1], [1]], logs=False)
    right = report("arm-b", [[1], [1]], logs=False)
    right["cases"][0][field] = value
    status, captured = run_pair(tmp_path, capsys, left, right)
    assert status == compare.EXIT_INVALID
    assert "CONFIG ERROR" in captured.err


def test_cache_geometry_and_ignore_eos_must_align(tmp_path, capsys):
    left = report("arm-a", [[1], [1]], logs=False)
    right = report("arm-b", [[1], [1]], logs=False)
    right["cases"][0]["ignore_eos"] = False
    status, captured = run_pair(tmp_path / "ignore", capsys, left, right)
    assert status == compare.EXIT_INVALID
    assert "metadata does not align" in captured.err


def test_v4_metric_probe_geometry_and_artifact_binding_fail_closed(tmp_path, capsys):
    left = report("arm-a", [[1], [1]], logs=False)
    right = report("arm-b", [[1], [1]], logs=False)
    right["cases"][0]["cache_metric_probe"]["geometry"][
        "physical_block_tokens_per_dcp_rank"
    ] = 256
    status, captured = run_pair(tmp_path / "geometry", capsys, left, right)
    assert status == compare.EXIT_INVALID
    assert "probe geometry is unsupported" in captured.err

    right = report("arm-b", [[1], [1]], logs=False)
    right["cases"][0]["cache_metric_probe"]["artifact_sha256"] = "not-a-hash"
    status, captured = run_pair(tmp_path / "hash", capsys, left, right)
    assert status == compare.EXIT_INVALID
    assert "must be SHA-256" in captured.err


@pytest.mark.parametrize(
    "mutation",
    ("extra-private-field", "missing-live-receipt", "inconsistent-cache-layers"),
)
def test_v4_metric_probe_normalized_schema_is_closed(
    tmp_path, capsys, mutation
):
    left = report("arm-a", [[1], [1]], logs=False)
    right = report("arm-b", [[1], [1]], logs=False)
    probe = right["cases"][0]["cache_metric_probe"]
    if mutation == "extra-private-field":
        probe["contacted_base_url"] = "https://private.invalid"
    elif mutation == "missing-live-receipt":
        del probe["live_arm_receipt"]
    else:
        probe["observed_cache_layers"] = ["external-kv-transfer"]
    status, captured = run_pair(tmp_path, capsys, left, right)
    assert status == compare.EXIT_INVALID
    assert "CONFIG ERROR" in captured.err


def test_v4_report_and_metric_probe_cache_namespace_binding_fail_closed(
    tmp_path, capsys
):
    left = report("arm-a", [[1], [1]], logs=False)
    right = report("arm-b", [[1], [1]], logs=False)
    right["cache_salt"] = cache_salt_for_arm("e-mtp0-apc0-lmcache1")
    status, captured = run_pair(tmp_path / "report", capsys, left, right)
    assert status == compare.EXIT_INVALID
    assert "cache_salt does not bind" in captured.err

    right = report("arm-b", [[1], [1]], logs=False)
    right["cases"][0]["cache_metric_probe"]["cache_salt"] = (
        cache_salt_for_arm("e-mtp0-apc0-lmcache1")
    )
    status, captured = run_pair(tmp_path / "probe", capsys, left, right)
    assert status == compare.EXIT_INVALID
    assert "cache salt does not bind report" in captured.err


def test_cache_evidence_counts_cannot_be_forged_with_booleans(tmp_path, capsys):
    left = report("arm-a", [[1], [1]], logs=False)
    right = report("arm-b", [[1], [1]], logs=False)
    right["cases"][0]["cache_evidence_scope"]["request_ids_bound"] = True
    status, captured = run_pair(tmp_path, capsys, left, right)
    assert status == compare.EXIT_INVALID
    assert "nonnegative integer" in captured.err
    right = report("arm-b", [[1], [1]], logs=False)
    right["cases"][0]["cache_boundaries"]["lmcache_chunk_tokens"] = 256
    status, captured = run_pair(tmp_path / "geometry", capsys, left, right)
    assert status == compare.EXIT_INVALID
    assert "must be canonical 512" in captured.err


def test_output_records_source_hashes_and_exclusive_creation(tmp_path, capsys):
    left = report("arm-a", [[1], [1]], logs=False)
    right = report("arm-b", [[1], [1]], logs=False)
    output = tmp_path / "evidence" / "comparison.json"
    status, captured = run_pair(tmp_path, capsys, left, right, output=output)
    assert status == 0
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored == json.loads(captured.out)
    assert stored["sources"]["left"]["run_label"] == "arm-a"
    assert stored["sources"]["left"]["artifact_sha256"] == hashlib.sha256(
        (tmp_path / "left.json").read_bytes()
    ).hexdigest()
    assert stored["sources"]["left"]["live_arm_receipt_sha256"] == "8" * 64
    assert stored["sources"]["right"]["live_arm_receipt_sha256"] == "8" * 64
    assert compare.main(
        [
            "--left",
            str(tmp_path / "left.json"),
            "--right",
            str(tmp_path / "right.json"),
            "--output",
            str(output),
        ]
    ) == compare.EXIT_INVALID
    assert "already exists" in capsys.readouterr().err


def test_alignment_mismatches_fail_closed(tmp_path, capsys):
    fields = [
        ("model", "another-model"),
        ("config_sha256", "e" * 64),
        ("repetitions", 3),
        ("top_logprobs", 1),
    ]
    for index, (field, value) in enumerate(fields):
        left = report(f"arm-a-{index}", [[1], [1]])
        right = report(f"arm-b-{index}", [[1], [1]])
        right[field] = value
        left_path = write_report(tmp_path, f"left-{index}.json", left)
        right_path = write_report(tmp_path, f"right-{index}.json", right)
        assert compare.main(["--left", str(left_path), "--right", str(right_path)]) == compare.EXIT_INVALID
        assert "CONFIG ERROR" in capsys.readouterr().err


def test_case_order_repetition_and_metadata_mismatches_fail_closed(tmp_path, capsys):
    mutations = []
    extra = report("unused", [[1], [1]])["cases"][0]
    extra["id"] = "second-case"
    mutations.append(lambda right: right["cases"].append(extra))
    mutations.append(lambda right: right["cases"][0].__setitem__("seed", 7))
    mutations.append(lambda right: right["cases"][0]["observations"][1].__setitem__("repetition", 1))
    for index, mutate in enumerate(mutations):
        left = report(f"arm-a-{index}", [[1], [1]])
        right = report(f"arm-b-{index}", [[1], [1]])
        mutate(right)
        status, captured = run_pair(tmp_path / str(index), capsys, left, right)
        assert status == compare.EXIT_INVALID
        assert "CONFIG ERROR" in captured.err


def test_forged_token_hash_and_malformed_logprobs_fail_closed(tmp_path, capsys):
    mutations = [
        lambda right: right["cases"][0]["observations"][0].__setitem__("token_ids_sha256", "f" * 64),
        lambda right: right["cases"][0]["observations"][0]["completion_logprobs"]["tokens"].append("extra"),
        lambda right: right["cases"][0]["observations"][0]["completion_logprobs"]["top_logprobs"][0].__setitem__("bad", 1.0),
    ]
    for index, mutate in enumerate(mutations):
        left = report(f"arm-a-{index}", [[1], [1]])
        right = report(f"arm-b-{index}", [[1], [1]])
        mutate(right)
        status, captured = run_pair(tmp_path / str(index), capsys, left, right)
        assert status == compare.EXIT_INVALID
        assert "CONFIG ERROR" in captured.err


def test_duplicate_json_keys_are_rejected(tmp_path, capsys):
    valid = report("arm-a", [[1], [1]], logs=False)
    left = tmp_path / "left.json"
    left.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")
    right = write_report(tmp_path, "right.json", valid)
    assert compare.main(["--left", str(left), "--right", str(right)]) == compare.EXIT_INVALID
    assert "duplicate JSON object key" in capsys.readouterr().err


def test_current_cache_qualification_and_positive_hit_evidence_are_recomputed(
    tmp_path, capsys
):
    left = report("arm-a", [[1], [1]], logs=False)
    right = report("arm-b", [[1], [1]], logs=False)
    right["cases"][0]["cache_boundaries"]["qualifies_for_cache_attribution"] = False
    status, captured = run_pair(tmp_path / "qualifier", capsys, left, right)
    assert status == compare.EXIT_INVALID
    assert "qualification is inconsistent" in captured.err

    right = report("arm-b", [[1], [1]], logs=False)
    evidence = right["cases"][0]["observations"][0]["request_evidence"]
    evidence["usage_cached_prompt_tokens"] = 0
    evidence["hit_evidence_source"] = (
        "openai-usage.prompt_tokens_details.cached_tokens"
    )
    right["cases"][0]["cache_evidence_scope"][
        "request_correlated_hit_evidence_count"
    ] = 1
    status, captured = run_pair(tmp_path / "zero-hit", capsys, left, right)
    assert status == compare.EXIT_INVALID
    assert "hit source is inconsistent" in captured.err

    right = report("arm-b", [[1], [1]], logs=False)
    evidence = right["cases"][0]["observations"][0]["request_evidence"]
    evidence["usage_cached_prompt_tokens"] = 0
    evidence["hit_evidence_source"] = None
    right["cases"][0]["cache_evidence_scope"][
        "request_correlated_hit_evidence_count"
    ] = 0
    status, captured = run_pair(tmp_path / "zero-metadata-only", capsys, left, right)
    assert status == compare.EXIT_OK
    comparison = json.loads(captured.out)
    assert comparison["comparison_status"] == "exact-match"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(model="/private/model"), "sanitized model identifier"),
        (
            lambda value: value["cases"][0]["observations"][0][
                "completion_logprobs"
            ]["token_logprobs"].__setitem__(0, float("inf")),
            "finite or null",
        ),
        (
            lambda value: value["cases"][0]["observations"][0][
                "token_ids"
            ].__setitem__(0, 2**31),
            "bounded nonnegative integers",
        ),
    ],
)
def test_current_reports_reject_unsanitized_or_unbounded_values(
    tmp_path, capsys, mutation, message
):
    left = report("arm-a", [[1], [1]])
    right = report("arm-b", [[1], [1]])
    mutation(right)
    status, captured = run_pair(tmp_path, capsys, left, right)
    assert status == compare.EXIT_INVALID
    assert message in captured.err
