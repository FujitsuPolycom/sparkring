from __future__ import annotations

import copy
import hashlib
import json
import math

from acceptance_gate import canonical_json
import exl3_attribution_cache_contract as cache_contract
import exl3_teacher_forced_margin_probe as probe
import exl3_teacher_forced_margin_reduce as reduce
import pytest


def _digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _observation(repetition, top1, top2, first, second):
    values = {
        "31": {"rank": 1 if top1 == 31 else 2, "value": first},
        "32": {"rank": 1 if top1 == 32 else 2, "value": second},
    }
    return {
        "candidate_pair_margin": values["31"]["value"] - values["32"]["value"],
        "forced_token_id": 31,
        "forced_token_rank": values["31"]["rank"],
        "forced_token_value": values["31"]["value"],
        "repetition": repetition,
        "returned_probability_mass": math.exp(first) + math.exp(second),
        "returned_topk_sha256": _digest({31: values["31"], 32: values["32"]}),
        "returned_values": values,
        "top1_token_id": top1,
        "top1_top2_margin": values[str(top1)]["value"] - values[str(top2)]["value"],
        "top1_value": values[str(top1)]["value"],
        "top2_token_id": top2,
        "top2_value": values[str(top2)]["value"],
    }


def raw_report():
    prompt_ids = [1, 2]
    reference = [10, 31, 40]
    sequences = [[10, 31, 40], [10, 32, 40]]
    observations = [
        _observation(1, 31, 32, -0.6, -0.8),
        _observation(2, 32, 31, -0.8, -0.6),
    ]
    summary = probe._position_summary(observations, [31, 32])
    runtime_instances = [
        {
            "container_id": f"{rank + 1:064x}",
            "started_at": f"2026-08-10T00:00:0{rank}.000000000Z",
        }
        for rank in range(4)
    ]
    live = cache_contract.build_live_arm_receipt(
        arm_id=probe.REQUIRED_ARM,
        canonical_profile_id="glm52-exl3-tr3-3.25bpw-lmcache-cs512",
        canonical_profile_file_sha256="9" * 64,
        image_id="sha256:" + "b" * 64,
        model_repository="willfalco/GLM-5.2-EXL3-TR3-3.25bpw",
        model_revision="d7d79c2d14599dfce7a5d12b85f7ad73f40e623d",
        canonical_container_name="glm52-sparkring-exl3-lmcache-cs512",
        explicit_environment_sha256=[f"{rank + 3:x}" * 64 for rank in range(4)],
        config_cmd_sha256=[f"{rank + 7:x}" * 64 for rank in range(4)],
        observed_runtime_instances=runtime_instances,
    )
    live["artifact_sha256"] = hashlib.sha256(
        (json.dumps(live, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "autoregressive_discovery": {
            "candidate_token_ids_at_focus": [31, 32],
            "earliest_divergence_index": 1,
            "focus_generated_index": 1,
            "observations": [
                {
                    "repetition": index,
                    "token_ids": token_ids,
                    "token_ids_sha256": _digest(token_ids),
                }
                for index, token_ids in enumerate(sequences, 1)
            ],
            "reference_token_ids": reference,
            "reference_token_ids_sha256": _digest(reference),
        },
        "case_id": "code-stable-unique",
        "config_sha256": "d" * 64,
        "contacted_base_url": "http://private-rank-0:8000",
        # Deliberately model the stale aggregate classifier found in the live
        # report; the reducer must recompute from every position.
        "diagnostic_classification": "autoregressive-divergence-without-teacher-forced-top1-flip",
        "evidence_policy": {
            "full_vocabulary_kld_claimed": False,
            "raw_report_is_private": True,
            "reason": "private token arrays",
        },
        "forced_context_token_ids": [*prompt_ids, *reference[:2]],
        "forced_context_token_ids_sha256": _digest([*prompt_ids, *reference[:2]]),
        "lane": "public-functional",
        "limitations": {
            "full_vocabulary_kld_available": False,
            "raw_logits_returned": False,
            "statement": "top-k only",
        },
        "maturity": "diagnostic-observation-not-acceptance",
        "model": "glm-5.2-exl3-tr3-3.25bpw",
        "prompt_token_count": len(prompt_ids),
        "prompt_token_ids": prompt_ids,
        "prompt_token_ids_sha256": _digest(prompt_ids),
        "request_contract": {
            "discovery_max_tokens": 3,
            "discovery_repetitions": 2,
            "focus_generated_index": None,
            "logprobs_mode": "raw_logprobs",
            "logprobs_mode_provenance": "pinned-vllm-668275901b55230f4a70841a9aac1c0be22ef8d3-default",
            "prompt_logprobs": 2,
            "server_max_logprobs": 2,
            "teacher_forced_repetitions": 2,
            "window_after": 0,
            "window_before": 0,
        },
        "runtime_identity": {
            "attribution_arm": probe.REQUIRED_ARM,
            "live_arm_receipt": live,
            "live_arm_revalidation": {
                "status": "live-arm-re-attested",
                "rank_count": 4,
                "runtime_instances": runtime_instances,
            },
            "lmcache_attached": False,
            "mtp_tokens": 0,
            "native_prefix_caching_enabled": False,
            "runtime_source_pins": {
                "evidence_scope": "declared-canonical-pins-bound-to-receipt-model-not-live-binary-introspection",
                "exllamav3_commit": "d21d9b3182e746199093b77b49a708635c1d1b5d",
                "model_repository": live["model_repository"],
                "model_revision": live["model_revision"],
                "pins_file_sha256": "f462110e39488fcd2d600938f9303b17e7f08c9e742a00373a79e0dca82edf91",
                "sparkinfer_commit": "018de520e40f6bf9bd0b11c5da5517ef3364a985",
                "vllm_commit": probe.PINNED_API_VLLM_COMMIT,
            },
        },
        "schema": probe.REPORT_SCHEMA,
        "status": "completed",
        "teacher_forced_positions": [
            {
                "absolute_prompt_index": len(prompt_ids) + 1,
                "context_token_ids_sha256": _digest([*prompt_ids, reference[0]]),
                "forced_token_id": reference[1],
                "generated_index": 1,
                "observations": observations,
                "summary": summary,
            }
        ],
    }


def test_reducer_recomputes_stale_classification_and_omits_private_values():
    document = raw_report()
    receipt = reduce.reduce_document(document, "3" * 64)
    assert receipt["maturity"] == "live-validated"
    assert receipt["evidence_scope"] == "diagnostic observation; not acceptance"
    assert receipt["hardware"] == "four directly cabled DGX Sparks"
    assert receipt["diagnostic"] == {
        "raw_report_classification": "autoregressive-divergence-without-teacher-forced-top1-flip",
        "recomputed_classification": "cache-not-required-forward-ranking-nondeterminism-observed",
        "raw_report_classifier_was_stale": True,
        "conclusion": "same-context top-1 nondeterminism was observed with MTP disabled, native APC disabled, and LMCache detached; cache reuse was not required for this observation",
        "non_conclusion": "this diagnostic does not identify the responsible kernel, graph, attention path, or collective",
    }
    assert receipt["autoregressive_discovery"]["token_sequence_multiplicities"] == [1, 1]
    assert receipt["teacher_forced"]["positions"][0]["top1_multiplicities"] == [1, 1]
    rendered = json.dumps(receipt)
    for forbidden in ("private-rank", "contacted_base_url", "prompt_token_ids", "token_ids_sha256", "top1_token_id", "container_id"):
        assert forbidden not in rendered


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("private_extra", "secret"),
        lambda value: value["autoregressive_discovery"]["observations"][0].__setitem__("token_ids", [99]),
        lambda value: value["teacher_forced_positions"][0]["summary"].__setitem__("distinct_top1_count", 1),
        lambda value: value["runtime_identity"].__setitem__("mtp_tokens", 2),
        lambda value: value["runtime_identity"]["runtime_source_pins"].__setitem__("sparkinfer_commit", "f" * 40),
        lambda value: value.__setitem__("diagnostic_classification", "divergence-not-reproduced-at-probed-context"),
        lambda value: value["evidence_policy"].__setitem__("raw_report_is_private", False),
    ],
)
def test_reducer_rejects_schema_hash_summary_and_arm_tampering(mutate):
    document = raw_report()
    mutate(document)
    with pytest.raises(reduce.ReduceError):
        reduce.reduce_document(document, "3" * 64)


def test_reducer_binds_claimed_top1_and_margins_to_hashed_returned_values():
    document = raw_report()
    observation = document["teacher_forced_positions"][0]["observations"][1]
    observation["top1_token_id"] = 31
    observation["top2_token_id"] = 32
    observation["top1_value"] = observation["returned_values"]["31"]["value"]
    observation["top2_value"] = observation["returned_values"]["32"]["value"]
    observation["top1_top2_margin"] = (
        observation["top1_value"] - observation["top2_value"]
    )
    document["teacher_forced_positions"][0]["summary"] = probe._position_summary(
        document["teacher_forced_positions"][0]["observations"], [31, 32]
    )
    with pytest.raises(reduce.ReduceError, match="returned_values ranks"):
        reduce.reduce_document(document, "3" * 64)


def test_reducer_rejects_impossible_returned_probability_mass():
    document = raw_report()
    observation = document["teacher_forced_positions"][0]["observations"][0]
    observation["returned_values"]["31"]["value"] = -0.1
    observation["returned_values"]["32"]["value"] = -0.1
    observation["forced_token_value"] = -0.1
    observation["top1_value"] = -0.1
    observation["top2_value"] = -0.1
    observation["top1_top2_margin"] = 0.0
    observation["candidate_pair_margin"] = 0.0
    observation["returned_probability_mass"] = 2 * math.exp(-0.1)
    observation["returned_topk_sha256"] = _digest(
        {int(key): value for key, value in observation["returned_values"].items()}
    )
    document["teacher_forced_positions"][0]["summary"] = probe._position_summary(
        document["teacher_forced_positions"][0]["observations"], [31, 32]
    )
    with pytest.raises(reduce.ReduceError, match="probability mass violates"):
        reduce.reduce_document(document, "3" * 64)


def test_reducer_binds_reference_to_the_producer_selected_discovery_sequence():
    document = raw_report()
    alternate = document["autoregressive_discovery"]["observations"][1][
        "token_ids"
    ]
    document["autoregressive_discovery"]["reference_token_ids"] = alternate
    document["autoregressive_discovery"]["reference_token_ids_sha256"] = _digest(
        alternate
    )
    with pytest.raises(reduce.ReduceError, match="producer-selected"):
        reduce.reduce_document(document, "3" * 64)


def test_reducer_binds_focus_and_candidates_to_discovery():
    document = raw_report()
    document["autoregressive_discovery"]["focus_generated_index"] = 0
    with pytest.raises(reduce.ReduceError, match="focus generated index"):
        reduce.reduce_document(document, "3" * 64)

    document = raw_report()
    document["autoregressive_discovery"]["candidate_token_ids_at_focus"] = [31, 33]
    with pytest.raises(reduce.ReduceError, match="candidate tokens"):
        reduce.reduce_document(document, "3" * 64)


def test_reducer_requires_complete_requested_teacher_forced_window():
    document = raw_report()
    document["request_contract"]["window_before"] = 1
    with pytest.raises(reduce.ReduceError, match="requested focus window"):
        reduce.reduce_document(document, "3" * 64)


def test_cli_rejects_duplicate_and_nonfinite_json(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    assert reduce.main(["--input", str(path)]) == 3
    assert "duplicate JSON key" in capsys.readouterr().err
    path.write_text('{"value": NaN}', encoding="utf-8")
    assert reduce.main(["--input", str(path)]) == 3
    assert "non-finite JSON number" in capsys.readouterr().err


def test_cli_rejects_unrepresentable_numeric_observation(tmp_path, capsys):
    document = raw_report()
    document["teacher_forced_positions"][0]["observations"][0][
        "top1_value"
    ] = 10**1000
    path = tmp_path / "overflow.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert reduce.main(["--input", str(path)]) == 3
    assert "cannot be represented as a finite float" in capsys.readouterr().err


def test_cli_binds_raw_bytes_and_round_trips(tmp_path, capsys):
    source = tmp_path / "private.json"
    raw = json.dumps(raw_report()).encode("utf-8")
    source.write_bytes(raw)
    output = tmp_path / "public.json"
    assert reduce.main(["--input", str(source), "--output", str(output)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["private_raw_artifact_sha256"] == hashlib.sha256(raw).hexdigest()
    assert json.loads(output.read_text(encoding="utf-8")) == receipt
    assert reduce.main(["--input", str(source), "--output", str(output)]) == 3
    assert "already exists" in capsys.readouterr().err


def test_fixture_is_not_mutated_by_reduction():
    document = raw_report()
    original = copy.deepcopy(document)
    reduce.reduce_document(document, "3" * 64)
    assert document == original
