import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
ASSET_ROOT = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location(
    "r33_profile_verifier", HERE / "verify_profile.py"
)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def image_receipt():
    contract = verifier.load_contract()
    return {
        "schema": "sparkring-r33-image-receipt/v1",
        "checks_passed": True,
        "platform": "linux/arm64",
        "image_id": "sha256:" + "a" * 64,
        "image_reference": "ghcr.io/fujitsupolycom/sparkring@sha256:" + "b" * 64,
        "artifact_lock_sha256": contract["image"]["artifact_lock_sha256"],
        "sources": contract["image"]["required_sources"],
        "component_receipts": {
            name: "c" * 64 for name in contract["image"]["required_receipts"]
        },
        "source_lock_sha256": "d" * 64,
        "nccl_version": "2.31.2",
        "source_locks_match": True,
        "source_lock_receipts_match": True,
        "installed_payload_bytes_match": True,
        "package_checks_passed": True,
    }


def activation(name):
    contract, profile = verifier.profile(name)
    ranks = []
    for rank in range(profile["node_count"]):
        item = {
            "rank": rank,
            "image_id": "sha256:" + "a" * 64,
            "nccl_version": "2.31.2",
            "nccl_host_domains": ["primary", "secondary"],
            "captured_graph_sizes": profile["cudagraph_capture_sizes"],
            "instanttensor_allocations": 1,
            "mtp_draft_tokens": 8,
            "continuation_coalesced_groups": 0 if name == "tp2-dcp1" else 1,
            "mhc_sharded_prefill_calls": 1,
            "mhc_owner_rows": [2048],
        }
        if name.startswith("tp2-"):
            item["mhc_prefill_rows"] = 4096
        if profile.get("load_format") == "b12x":
            item.pop("instanttensor_allocations")
            item["managed_b12x_allocations"] = 1
        item[
            "rocenante_collectives" if name.startswith("tp2-") else "sircl_collectives"
        ] = 1
        ranks.append(item)
    result = {
        "schema": "sparkring-r33-activation-receipt/v1",
        "checks_passed": True,
        "profile": name,
        "profile_contract_sha256": hashlib.sha256(
            verifier.CONTRACT_PATH.read_bytes()
        ).hexdigest(),
        "image": image_receipt(),
        "ranks": ranks,
        "serving": {
            "max_model_len": 1048576,
            "kv_capacity_tokens": 100000,
            "prefill_decode_passed": True,
            "correctness_passed": True,
        },
        "sparkcache": {"enabled": profile["sparkcache"]},
    }
    if name.startswith("tp4"):
        result["long_prefill_sample_tokens"] = {
            "bounded": True,
            "prompt_tokens": 32768,
            "completed_requests": 2,
            "timeouts": 0,
            "fatal_engine_errors": 0,
        }
    if profile["sparkcache"]:
        result["sparkcache"].update(
            capture_jobs_completed=1,
            restores_completed=1,
            recoveries_after_fault=1,
            payload_correctness_passed=True,
        )
    if profile.get("required_capabilities"):
        capability = {
            "schema": "sparkring-r33-runtime-capabilities/v1",
            "profile": name,
            "sources": {
                k: contract["image"]["required_sources"][k]
                for k in (
                    "vllm_integrated_tree",
                    "vllm_tp2_continuation_port_commit",
                    "b12x_tree",
                    "sparkcache_tree",
                )
            },
            "evidence_kind": "source-component-tests",
            "live_qualification": "pending",
            "checks": {k: "implemented" for k in profile["required_capabilities"]},
            "evidence_sha256": {k: "e" * 64 for k in profile["required_capabilities"]},
        }
        digest = hashlib.sha256(
            (
                json.dumps(capability, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
        ).hexdigest()
        result["image"]["runtime_capabilities"] = {
            "document": capability,
            "sha256": digest,
        }
        native = contract["sparkcache_native"]
        result["image"]["verification"] = {
            "checked_files": {
                "/opt/sparkring/profile-contract/" + profile["capability_file"]: digest,
                native["placement_path"]: native["placement_sha256"],
                native["snapshot_path"]: native["snapshot_sha256"],
            }
        }
    return result


@pytest.mark.parametrize(
    "name", ["tp2-dcp1", "tp2-dcp1-sparkcache", "tp4-dcp1", "tp4-dcp1-sparkcache"]
)
def test_candidate_templates_match_contract_and_pinned_inputs(name):
    assert verifier.validate_template(name, ASSET_ROOT)["checks_passed"] is True


def test_tp2_uses_one_dac_across_two_host_domains_and_one_dcp_rank():
    contract, profile = verifier.profile("tp2-dcp1")
    values = verifier.parse_template(HERE / profile["template"])
    assert profile["physical_dacs_per_rank"] == 1
    assert profile["host_domains"] == "dual"
    assert profile["decode_context_parallel_size"] == 1
    assert values["NCCL_IB_HCA"] == "=rocep1s0f0,roceP2p1s0f0"
    assert values["B12X_ROCE_PEER_HCA_MAP"] == "<peer-rank>=0/2"
    assert contract["model"]["max_model_len"] == 1048576
    assert contract["model"]["loader"] == {
        "load_format": "instanttensor",
        "instanttensor_commit": "49b4010afc1cae0441e71fe0b0bffc24fa05e932",
        "backend_selection": "automatic",
        "required_environment": {},
    }
    assert values["LOAD_FORMAT"] == "instanttensor"


def test_tp4_profiles_share_mesh_graphs_and_dcp1_baseline():
    _, baseline = verifier.profile("tp4-dcp1")
    _, cached = verifier.profile("tp4-dcp1-sparkcache")
    assert (
        baseline["decode_context_parallel_size"]
        == cached["decode_context_parallel_size"]
        == 1
    )
    assert (
        baseline["cudagraph_capture_sizes"]
        == cached["cudagraph_capture_sizes"]
        == list(range(4, 65, 4))
    )
    assert baseline["mesh_pins_sha256"] == cached["mesh_pins_sha256"]
    assert baseline["sparkcache"] is False and cached["sparkcache"] is True
    baseline_values = verifier.parse_template(HERE / baseline["template"])
    assert baseline_values["LOAD_FORMAT"] == "instanttensor"
    assert baseline_values["VLLM_GLM53_MHC_PREFILL_DIAGNOSTICS"] == "1"


def test_image_selection_has_no_implicit_candidate_identity():
    values = verifier.parse_template(HERE / "image.env.example")
    assert not verifier.IMAGE_ID.fullmatch(values["SPARKRING_R33_IMAGE_ID"])
    assert not verifier.REGISTRY_DIGEST.fullmatch(values["SPARKRING_R33_IMAGE_REF"])
    with pytest.raises(ValueError, match="exact qualified R33"):
        verifier.validate_image_receipt({})


def test_local_config_identity_can_be_qualified_before_publication():
    document = image_receipt()
    document["image_reference"] = document["image_id"]
    assert (
        verifier.validate_image_receipt(document)["image_reference"]
        == document["image_id"]
    )


@pytest.mark.parametrize("name", ["tp2-dcp1", "tp4-dcp1", "tp4-dcp1-sparkcache"])
def test_activation_receipt_requires_runtime_counters(name):
    document = activation(name)
    document["ranks"][0]["instanttensor_allocations"] = 0
    with pytest.raises(ValueError, match="instanttensor_allocations"):
        verifier.validate_activation(document)


@pytest.mark.parametrize("name", ["tp2-dcp1", "tp4-dcp1", "tp4-dcp1-sparkcache"])
def test_complete_activation_receipt_passes(name):
    assert verifier.validate_activation(activation(name)) == {
        "profile": name,
        "ranks": 2 if name == "tp2-dcp1" else 4,
        "checks_passed": True,
    }


def test_tp4_long_prefill_timeout_fails_qualification():
    document = activation("tp4-dcp1")
    document["long_prefill_sample_tokens"]["timeouts"] = 1
    with pytest.raises(ValueError, match="sample_tokens"):
        verifier.validate_activation(document)


def test_tp2_cache_activation_requires_managed_loader_and_coalescing_evidence():
    document = activation("tp2-dcp1-sparkcache")
    assert verifier.validate_activation(document)["ranks"] == 2
    document["ranks"][0]["managed_b12x_allocations"] = 0
    document["ranks"][0]["instanttensor_allocations"] = 1
    with pytest.raises(ValueError, match="managed_b12x_allocations"):
        verifier.validate_activation(document)
    document["ranks"][0]["managed_b12x_allocations"] = 1
    document["ranks"][0]["continuation_coalesced_groups"] = 0
    with pytest.raises(ValueError, match="continuation_coalesced_groups"):
        verifier.validate_activation(document)


@pytest.mark.parametrize("rows", [4096, 8192])
def test_tp2_accepts_supported_mhc_ceilings_with_coalescing_disabled(rows):
    document = activation("tp2-dcp1")
    for rank in document["ranks"]:
        rank["mhc_prefill_rows"] = rows
        rank["mhc_owner_rows"] = [rows // 2]
    assert verifier.validate_activation(document)["checks_passed"] is True


@pytest.mark.parametrize("count", [None, False, -1, 1])
def test_tp2_rejects_missing_or_executed_coalescing(count):
    document = activation("tp2-dcp1")
    document["ranks"][0]["continuation_coalesced_groups"] = count
    with pytest.raises(ValueError, match="coalescing is disabled"):
        verifier.validate_activation(document)


@pytest.mark.parametrize("rows", [None, 2048, 16384])
def test_tp2_rejects_missing_or_unsupported_mhc_ceiling(rows):
    document = activation("tp2-dcp1")
    document["ranks"][0]["mhc_prefill_rows"] = rows
    with pytest.raises(ValueError, match="mhc_prefill_rows"):
        verifier.validate_activation(document)


def test_tp2_rejects_tp4_owner_count_for_8192_rows():
    document = activation("tp2-dcp1")
    document["ranks"][0]["mhc_prefill_rows"] = 8192
    with pytest.raises(ValueError, match="owner rows must match"):
        verifier.validate_activation(document)


def test_tp2_rejects_inconsistent_mhc_ceilings_across_ranks():
    document = activation("tp2-dcp1")
    document["ranks"][0].update(mhc_prefill_rows=8192, mhc_owner_rows=[4096])
    with pytest.raises(ValueError, match="same mHC prefill row ceiling"):
        verifier.validate_activation(document)


@pytest.mark.parametrize("name", ["tp4-dcp1", "tp4-dcp1-sparkcache"])
@pytest.mark.parametrize(
    "field,value,match",
    [
        ("continuation_coalesced_groups", 0, "continuation_coalesced_groups"),
        ("mhc_owner_rows", [4096], "2,048-row"),
    ],
)
def test_tp4_retains_coalescing_and_mhc_execution_requirements(
    name, field, value, match
):
    document = activation(name)
    document["ranks"][0][field] = value
    with pytest.raises(ValueError, match=match):
        verifier.validate_activation(document)


def test_sparkcache_profile_requires_capture_restore_and_fault_recovery():
    document = activation("tp4-dcp1-sparkcache")
    document["sparkcache"]["recoveries_after_fault"] = 0
    with pytest.raises(ValueError, match="recoveries_after_fault"):
        verifier.validate_activation(document)


def test_environment_flags_alone_are_not_activation_evidence():
    values = verifier.parse_template(HERE / "tp4-dcp1.env.example")
    with pytest.raises(ValueError):
        verifier.validate_activation({"profile": "tp4-dcp1", "environment": values})
