from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_exl3_r7_mtp2 as mtp2  # noqa: E402


EXPECTED_SPECULATIVE_CONFIG = {
    "model": "/models/glm52-exl3-r7-3.5bpw",
    "method": "mtp",
    "num_speculative_tokens": 2,
    "draft_tensor_parallel_size": 4,
    "quantization": "exl3",
    "moe_backend": "b12x",
    "attention_backend": "B12X_MLA_SPARSE",
    "use_local_argmax_reduction": False,
    "draft_sample_method": "greedy",
}


def stock_profile() -> dict:
    quantization = {
        "linear": {"weight": "mxfp8"},
        "shared_experts": {"weight": "mxfp8"},
        "ignore": ["model.layers.78.eh_proj", "lm_head"],
    }
    compilation = {
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes": list(range(1, 33)),
    }
    return {
        "schema": "sparkring-runtime-profile/v1",
        "profile_id": "glm52-exl3-r7-3.5bpw",
        "model_family": "exl3-r7",
        "model_container_path": "/models/glm52-exl3-r7-3.5bpw",
        "identity": copy.deepcopy(mtp2._EXPECTED_IDENTITY),
        "environment": {
            "LD_PRELOAD": (
                "/usr/local/cuda/compat/libcuda.so.1:"
                "/opt/sparkring/nccl/libnccl.so.2"
            ),
            "NCCL_ALGO": "Ring",
            "NCCL_CROSS_NIC": "1",
            "NCCL_IB_DISABLE": "0",
            "NCCL_MAX_NCHANNELS": "4",
            "NCCL_MIN_NCHANNELS": "4",
            "NCCL_NET": "IB",
            "ONLINE_QUANT": "exl3-b6",
            "TORCH_USE_RTLD_GLOBAL": "1",
            "VLLM_EXL3_ONLINE_TRELLIS_BITS": "6",
            "VLLM_NCCL_SO_PATH": "/opt/sparkring/nccl/libnccl.so.2",
            "VLLM_SPARK_MAX_QUERY_ROWS": "8",
            "VLLM_SPARK_MTP_TOKENS": "0",
            "VLLM_SPARK_SHARED_CAPTURE_STREAM": "1",
            "VLLM_SPARK_TP4_ALLGATHER_MODE": "custom",
            "VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM": "0",
            "VLLM_SPARK_TP4_MODE": "custom",
            "VLLM_SPARK_TP4_VOCAB_MODE": "custom",
        },
        "extra_vllm_args": [
            "--quantization",
            "exl3",
            "--quantization-config",
            json.dumps(quantization),
            "--moe-backend",
            "b12x",
            "--dcp-comm-backend",
            "ag_rs",
            "--dcp-kv-cache-interleave-size",
            "1",
            "--kv-cache-dtype",
            "fp8_ds_mla",
            "--compilation-config",
            json.dumps(compilation),
            "--max-num-batched-tokens",
            "2048",
        ],
        "extra_volumes": [],
        "extra_labels": {"org.sparkring.candidate": "exl3-r7-3.5bpw"},
        "attestation_hook": [
            "/bin/sh",
            "-c",
            "printf '%s\\n' 'base  /base' | sha256sum --check --strict -",
        ],
    }


def test_candidate_changes_only_exact_fixed_mtp2_contract() -> None:
    stock = stock_profile()
    candidate = mtp2.derive_candidate(stock)

    expected = copy.deepcopy(stock)
    expected["profile_id"] += "-fixed-mtp2"
    expected["environment"].update(mtp2.MTP_ENVIRONMENT)
    expected["extra_labels"].update(
        {
            "org.sparkring.r7.online-k6-scope": "target-only",
            "org.sparkring.r7.target-weight-contract": (
                "checkpoint-exl3-routed+online-k6-eligible-bf16"
            ),
            "org.sparkring.r7.draft-weight-contract": (
                "checkpoint-exl3-routed+producer-bf16-nonexpert"
            ),
            "org.sparkring.r7.capture-stream-contract": (
                "process-device-shared-target+draft"
            ),
        }
    )
    expected["extra_volumes"].append(
        {
            "host": mtp2.SHARED_CAPTURE_HOST_PATH,
            "container": mtp2.SHARED_CAPTURE_CONTAINER_PATH,
            "mode": "ro",
        }
    )
    expected["attestation_hook"][2] = expected["attestation_hook"][2].replace(
        " | sha256sum --check --strict -",
        (
            f" '{mtp2.SHARED_CAPTURE_SHA256}  "
            f"{mtp2.SHARED_CAPTURE_CONTAINER_PATH}'"
            " | sha256sum --check --strict -"
        ),
    )
    spec = EXPECTED_SPECULATIVE_CONFIG
    index = expected["extra_vllm_args"].index("--max-num-batched-tokens")
    expected["extra_vllm_args"][index:index] = [
        "--speculative-config",
        json.dumps(spec, separators=(",", ":")),
    ]
    assert candidate == expected
    assert candidate["environment"]["VLLM_EXL3_ONLINE_TRELLIS_BITS"] == "6"
    assert candidate["environment"]["VLLM_SPARK_MAX_QUERY_ROWS"] == "24"
    assert candidate["environment"]["VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM"] == "0"
    spec_index = candidate["extra_vllm_args"].index("--speculative-config") + 1
    assert json.loads(candidate["extra_vllm_args"][spec_index]) == (
        EXPECTED_SPECULATIVE_CONFIG
    )
    assert "quantization_config" not in EXPECTED_SPECULATIVE_CONFIG
    assert "kv_cache_dtype" not in EXPECTED_SPECULATIVE_CONFIG
    assert (
        candidate["extra_vllm_args"][
            candidate["extra_vllm_args"].index("--kv-cache-dtype") + 1
        ]
        == "fp8_ds_mla"
    )
    assert candidate["extra_labels"] == {
        "org.sparkring.candidate": "exl3-r7-3.5bpw",
        "org.sparkring.r7.online-k6-scope": "target-only",
        "org.sparkring.r7.target-weight-contract": (
            "checkpoint-exl3-routed+online-k6-eligible-bf16"
        ),
        "org.sparkring.r7.draft-weight-contract": (
            "checkpoint-exl3-routed+producer-bf16-nonexpert"
        ),
        "org.sparkring.r7.capture-stream-contract": (
            "process-device-shared-target+draft"
        ),
    }
    assert candidate["extra_volumes"][-1] == {
        "host": mtp2.SHARED_CAPTURE_HOST_PATH,
        "container": mtp2.SHARED_CAPTURE_CONTAINER_PATH,
        "mode": "ro",
    }
    assert (
        f"{mtp2.SHARED_CAPTURE_SHA256}  {mtp2.SHARED_CAPTURE_CONTAINER_PATH}"
        in candidate["attestation_hook"][2]
    )
    mtp2.validate_candidate(stock, candidate)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("environment", "VLLM_SPARK_TP4_DCP_MODE"), "custom", "must not enable"),
        (("environment", "VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM"), "1", "requires"),
        (("environment", "VLLM_SPARK_MTP_TOKENS"), "2", "requires"),
        (("environment", "NCCL_NET"), "Socket", "requires"),
    ],
)
def test_stock_control_rejects_transport_or_mtp_drift(
    path: tuple[str, str], value: str, match: str
) -> None:
    stock = stock_profile()
    stock[path[0]][path[1]] = value
    with pytest.raises(mtp2.ContractError, match=match):
        mtp2.derive_candidate(stock)


def test_candidate_validation_rejects_eager_or_adaptive_drift() -> None:
    stock = stock_profile()
    candidate = mtp2.derive_candidate(stock)
    candidate["extra_vllm_args"].append("--enforce-eager")
    with pytest.raises(mtp2.ContractError, match="exact fixed-MTP2"):
        mtp2.validate_candidate(stock, candidate)


def test_candidate_rejects_missing_or_wrong_shared_stream_overlay() -> None:
    stock = stock_profile()
    candidate = mtp2.derive_candidate(stock)
    candidate["extra_volumes"] = [
        volume
        for volume in candidate["extra_volumes"]
        if volume["container"] != mtp2.SHARED_CAPTURE_CONTAINER_PATH
    ]
    with pytest.raises(mtp2.ContractError, match="shared capture stream overlay"):
        mtp2.validate_candidate(stock, candidate)

    candidate = mtp2.derive_candidate(stock)
    candidate["attestation_hook"][2] = candidate["attestation_hook"][2].replace(
        mtp2.SHARED_CAPTURE_SHA256,
        "0" * 64,
    )
    with pytest.raises(mtp2.ContractError, match="shared capture stream SHA"):
        mtp2.validate_candidate(stock, candidate)

    candidate = mtp2.derive_candidate(stock)
    candidate["environment"]["SPARK_GLM52_MTP_INDEX_REUSE"] = "1"
    with pytest.raises(mtp2.ContractError, match="exact fixed-MTP2"):
        mtp2.validate_candidate(stock, candidate)


def test_draft_rejects_online_quantization_config_or_k6_scope_drift() -> None:
    candidate = mtp2.derive_candidate(stock_profile())
    spec_index = candidate["extra_vllm_args"].index("--speculative-config") + 1
    spec = json.loads(candidate["extra_vllm_args"][spec_index])
    spec["quantization_config"] = {"linear": {"weight": "mxfp8"}}
    candidate["extra_vllm_args"][spec_index] = json.dumps(spec)
    with pytest.raises(mtp2.ContractError, match="without quantization_config"):
        mtp2.validate_weight_scope(candidate)

    candidate = mtp2.derive_candidate(stock_profile())
    candidate["extra_labels"]["org.sparkring.r7.online-k6-scope"] = "target+draft"
    with pytest.raises(mtp2.ContractError, match="online-k6-scope=target-only"):
        mtp2.validate_weight_scope(candidate)


def test_draft_rejects_explicit_kv_dtype_cache_reconstruction() -> None:
    candidate = mtp2.derive_candidate(stock_profile())
    spec_index = candidate["extra_vllm_args"].index("--speculative-config") + 1
    spec = json.loads(candidate["extra_vllm_args"][spec_index])
    spec["kv_cache_dtype"] = "fp8_ds_mla"
    candidate["extra_vllm_args"][spec_index] = json.dumps(spec)
    with pytest.raises(mtp2.ContractError, match="inherit target fp8_ds_mla"):
        mtp2.validate_weight_scope(candidate)

def test_site_derivative_changes_only_mtp_declaration() -> None:
    stock = (
        "serving:\n"
        "  tensor_parallel_size: 4\n"
        "  decode_context_parallel_size: 4\n"
        '  mtp_mode: "off"\n'
        "  mtp_tokens: 0\n"
        "  kv_cache_bytes_per_rank: 9000000000\n"
        "  max_num_seqs: 8\n"
    )
    candidate = mtp2.derive_site_text(stock)
    assert candidate == (
        "serving:\n"
        "  tensor_parallel_size: 4\n"
        "  decode_context_parallel_size: 4\n"
        '  mtp_mode: "static"\n'
        "  mtp_tokens: 2\n"
        "  kv_cache_bytes_per_rank: 9000000000\n"
        "  max_num_seqs: 8\n"
    )


def test_stock_control_rejects_custom_dcp_backend_overlay() -> None:
    stock = stock_profile()
    stock["extra_volumes"] = [
        {
            "host": "/var/tmp/backend.py",
            "container": "/opt/spark-vllm/spark_tp4_dcp_backend.py",
            "mode": "ro",
        }
    ]
    with pytest.raises(mtp2.ContractError, match="custom DCP/indexer/verifier"):
        mtp2.derive_candidate(stock)
