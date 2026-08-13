from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_exl3_r7_mtp2 as mtp2  # noqa: E402
import prepare_exl3_r7_mtp3 as mtp3  # noqa: E402


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


def mtp2_profile() -> tuple[dict, dict]:
    stock = stock_profile()
    return stock, mtp2.derive_candidate(stock)


def speculative_config(profile: dict) -> dict:
    values = mtp3._option_values(profile["extra_vllm_args"], "--speculative-config")
    assert len(values) == 1
    return json.loads(values[0])


def test_candidate_changes_only_fixed_depth_and_required_query_capacity() -> None:
    stock, control = mtp2_profile()
    candidate = mtp3.derive_candidate(stock, control)

    expected = copy.deepcopy(control)
    expected["profile_id"] = expected["profile_id"].removesuffix(
        "-fixed-mtp2"
    ) + "-fixed-mtp3"
    expected["environment"].update(
        {
            "VLLM_SPARK_MAX_QUERY_ROWS": "32",
            "VLLM_SPARK_MTP_MODE_ID": "fixed-mtp3",
            "VLLM_SPARK_MTP_TOKENS": "3",
        }
    )
    spec_index = expected["extra_vllm_args"].index("--speculative-config") + 1
    spec = json.loads(expected["extra_vllm_args"][spec_index])
    spec["num_speculative_tokens"] = 3
    expected["extra_vllm_args"][spec_index] = json.dumps(
        spec, separators=(",", ":")
    )

    assert candidate == expected
    assert speculative_config(candidate)["num_speculative_tokens"] == 3
    assert candidate["environment"]["VLLM_SPARK_MAX_QUERY_ROWS"] == "32"
    assert candidate["environment"]["VLLM_EXL3_ONLINE_TRELLIS_BITS"] == "6"
    assert candidate["environment"]["VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM"] == "0"
    assert candidate["extra_volumes"] == control["extra_volumes"]
    assert candidate["attestation_hook"] == control["attestation_hook"]
    mtp3.validate_candidate(stock, control, candidate)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("VLLM_SPARK_MAX_QUERY_ROWS", "24", "query-row ceiling 32"),
        ("VLLM_SPARK_MTP_MODE_ID", "fixed-mtp2", "mode fixed-mtp3"),
        ("VLLM_SPARK_MTP_TOKENS", "2", "depth 3"),
        ("VLLM_EXL3_ONLINE_TRELLIS_BITS", "5", "target online K6"),
        ("VLLM_SPARK_SHARED_CAPTURE_STREAM", "0", "shared capture stream"),
    ],
)
def test_candidate_rejects_environment_contract_drift(
    field: str, value: str, match: str
) -> None:
    stock, control = mtp2_profile()
    candidate = mtp3.derive_candidate(stock, control)
    candidate["environment"][field] = value
    with pytest.raises(mtp3.ContractError, match=match):
        mtp3.validate_candidate(stock, control, candidate)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"num_speculative_tokens": 2}, "depth 3"),
        ({"kv_cache_dtype": "fp8_ds_mla"}, "inherit target fp8_ds_mla"),
        ({"quantization_config": {"linear": {"weight": "mxfp8"}}}, "target-only"),
    ],
)
def test_speculative_parser_rejects_depth_or_weight_scope_drift(
    mutation: dict, match: str
) -> None:
    stock, control = mtp2_profile()
    candidate = mtp3.derive_candidate(stock, control)
    spec_index = candidate["extra_vllm_args"].index("--speculative-config") + 1
    spec = json.loads(candidate["extra_vllm_args"][spec_index])
    spec.update(mutation)
    candidate["extra_vllm_args"][spec_index] = json.dumps(spec)
    with pytest.raises(mtp3.ContractError, match=match):
        mtp3.validate_candidate(stock, control, candidate)


def test_speculative_parser_rejects_invalid_or_duplicate_json() -> None:
    stock, control = mtp2_profile()
    candidate = mtp3.derive_candidate(stock, control)
    spec_index = candidate["extra_vllm_args"].index("--speculative-config") + 1
    candidate["extra_vllm_args"][spec_index] = "{"
    with pytest.raises(mtp3.ContractError, match="valid JSON"):
        mtp3.validate_candidate(stock, control, candidate)

    candidate = mtp3.derive_candidate(stock, control)
    candidate["extra_vllm_args"].extend(
        ["--speculative-config", json.dumps(mtp3.expected_spec(candidate))]
    )
    with pytest.raises(mtp3.ContractError, match="exactly one speculative config"):
        mtp3.validate_candidate(stock, control, candidate)


def test_candidate_rejects_shared_stream_or_graph_coverage_drift() -> None:
    stock, control = mtp2_profile()
    candidate = mtp3.derive_candidate(stock, control)
    candidate["environment"]["VLLM_SPARK_SHARED_CAPTURE_STREAM"] = "0"
    with pytest.raises(mtp3.ContractError, match="shared capture stream"):
        mtp3.validate_candidate(stock, control, candidate)

    candidate = mtp3.derive_candidate(stock, control)
    compilation_index = candidate["extra_vllm_args"].index(
        "--compilation-config"
    ) + 1
    compilation = json.loads(candidate["extra_vllm_args"][compilation_index])
    compilation["cudagraph_capture_sizes"].remove(32)
    candidate["extra_vllm_args"][compilation_index] = json.dumps(compilation)
    with pytest.raises(mtp3.ContractError, match="capture Q32"):
        mtp3.validate_candidate(stock, control, candidate)


def test_site_derivative_changes_only_static_depth() -> None:
    site = (
        "serving:\n"
        "  tensor_parallel_size: 4\n"
        "  decode_context_parallel_size: 4\n"
        '  mtp_mode: "static"\n'
        "  mtp_tokens: 2\n"
        "  kv_cache_bytes_per_rank: 9000000000\n"
        "  max_num_seqs: 8\n"
    )
    assert mtp3.derive_site_text(site) == site.replace(
        "  mtp_tokens: 2", "  mtp_tokens: 3"
    )


def test_cli_emits_distinct_candidate_and_byte_identical_mtp2_rollback(
    tmp_path: Path,
) -> None:
    stock, control = mtp2_profile()
    stock_path = tmp_path / "stock.json"
    control_path = tmp_path / "mtp2.json"
    site_path = tmp_path / "mtp2.yaml"
    candidate_path = tmp_path / "mtp3.json"
    candidate_site_path = tmp_path / "mtp3.yaml"
    rollback_path = tmp_path / "mtp3-rollback.json"
    stock_path.write_text(json.dumps(stock, indent=2) + "\n", encoding="utf-8")
    control_bytes = (json.dumps(control, indent=2) + "\n").encode()
    control_path.write_bytes(control_bytes)
    site_path.write_text(
        "serving:\n"
        "  tensor_parallel_size: 4\n"
        "  decode_context_parallel_size: 4\n"
        '  mtp_mode: "static"\n'
        "  mtp_tokens: 2\n"
        "  kv_cache_bytes_per_rank: 9000000000\n"
        "  max_num_seqs: 8\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_exl3_r7_mtp3.py"),
            "--stock-dcp4-profile",
            str(stock_path),
            "--mtp2-profile",
            str(control_path),
            "--mtp2-site",
            str(site_path),
            "--candidate-profile",
            str(candidate_path),
            "--candidate-site",
            str(candidate_site_path),
            "--rollback-profile",
            str(rollback_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert control_path.read_bytes() == control_bytes
    assert rollback_path.read_bytes() == control_bytes
    assert json.loads(candidate_path.read_text())["profile_id"].endswith(
        "-fixed-mtp3"
    )
    assert "  mtp_tokens: 3" in candidate_site_path.read_text()


def test_cli_refuses_to_overwrite_mtp2_input(tmp_path: Path) -> None:
    stock, control = mtp2_profile()
    stock_path = tmp_path / "stock.json"
    control_path = tmp_path / "mtp2.json"
    site_path = tmp_path / "mtp2.yaml"
    stock_path.write_text(json.dumps(stock), encoding="utf-8")
    control_path.write_text(json.dumps(control), encoding="utf-8")
    site_path.write_text(
        "serving:\n"
        "  tensor_parallel_size: 4\n"
        "  decode_context_parallel_size: 4\n"
        '  mtp_mode: "static"\n'
        "  mtp_tokens: 2\n"
        "  kv_cache_bytes_per_rank: 9000000000\n"
        "  max_num_seqs: 8\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_exl3_r7_mtp3.py"),
            "--stock-dcp4-profile",
            str(stock_path),
            "--mtp2-profile",
            str(control_path),
            "--mtp2-site",
            str(site_path),
            "--candidate-profile",
            str(control_path),
            "--candidate-site",
            str(tmp_path / "candidate.yaml"),
            "--rollback-profile",
            str(tmp_path / "rollback.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "must not overwrite an input" in result.stderr
