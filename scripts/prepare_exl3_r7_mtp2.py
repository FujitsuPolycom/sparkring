#!/usr/bin/env python3
"""Prepare and validate the isolated fixed-MTP2 R7 experiment profiles."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path


class ContractError(ValueError):
    """The fixed-MTP2 experiment is not an exact stock-DCP4 derivative."""


SPECULATIVE_CONFIG = {
    "method": "mtp",
    "num_speculative_tokens": 2,
    "draft_tensor_parallel_size": 4,
    "quantization": "exl3",
    "moe_backend": "b12x",
    "attention_backend": "B12X_MLA_SPARSE",
    "use_local_argmax_reduction": False,
    "draft_sample_method": "greedy",
}

MTP_ENVIRONMENT = {
    "SPARK_ADAPTIVE_MTP_CONTROL": "0",
    "SPARK_GLM52_MTP_INDEX_REUSE": "0",
    "VLLM_ADAPTIVE_SPEC_DEPTHS": None,
    "VLLM_SPARK_MAX_QUERY_ROWS": "24",
    "VLLM_SPARK_MTP_ADAPTIVE_WINDOW": "0",
    "VLLM_SPARK_MTP_MODE_ID": "fixed-mtp2",
    "VLLM_SPARK_MTP_TOKENS": "2",
    "VLLM_SPARK_TRUE_ADAPTIVE_DRAFT": "0",
}

ONLINE_K6_SCOPE_LABEL = "target-only"
TARGET_WEIGHT_CONTRACT_LABEL = (
    "checkpoint-exl3-routed+online-k6-eligible-bf16"
)
DRAFT_WEIGHT_CONTRACT_LABEL = (
    "checkpoint-exl3-routed+producer-bf16-nonexpert"
)
SHARED_CAPTURE_CONTRACT_LABEL = "process-device-shared-target+draft"
SHARED_CAPTURE_CONTAINER_PATH = (
    "/opt/venv/lib/python3.12/site-packages/vllm/distributed/parallel_state.py"
)
SHARED_CAPTURE_SHA256 = (
    "b087e93463e9a2d9bede71d3a6e4d696c8f2657449e8dc1119b38613d5750e4e"
)

_FORBIDDEN_CUSTOM_DCP_ENV = {
    "VLLM_SPARK_TP4_DCP_MODE",
    "VLLM_SPARK_TP4_DCP_QUERY_ENABLED",
    "VLLM_SPARK_TP4_DCP_COMBINE_ENABLED",
    "VLLM_SPARK_TP4_DCP_GRAPH_CUSTOM",
}

_FORBIDDEN_CUSTOM_OVERLAY_TARGETS = {
    "/opt/spark-vllm/spark_tp4_allgather_backend.py",
    "/opt/spark-vllm/spark_tp4_dcp_backend.py",
    "/opt/sparkring-r7/verify_runtime.py",
}

_EXPECTED_IDENTITY = {
    "model_repository": "brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78",
    "model_revision": "9ab9579774cc432df91567a36f6e9e863e0d4c9f",
    "model_config_sha256": (
        "fabb73eb513ec64f3a365da396b38de8d55b3930edfb11baeecbf34ecafa6126"
    ),
    "model_index_sha256": (
        "9fd852f69ed64442e31dce1cbc5fe7acd0a76bfb848e945d272fe98d00d0c9cd"
    ),
}


def _option_values(arguments: list[str], option: str) -> list[str]:
    values: list[str] = []
    for index, value in enumerate(arguments):
        if value == option:
            if index + 1 >= len(arguments):
                raise ContractError(f"{option} has no value")
            values.append(arguments[index + 1])
        elif value.startswith(option + "="):
            values.append(value.split("=", 1)[1])
    return values


def _require_option(arguments: list[str], option: str, expected: str) -> None:
    if _option_values(arguments, option) != [expected]:
        raise ContractError(f"stock DCP4 control requires {option}={expected}")


def validate_stock_control(profile: dict) -> None:
    """Reject a baseline that is not the corrected stock-DCP4 R7 control."""

    if profile.get("schema") != "sparkring-runtime-profile/v1":
        raise ContractError("stock control has the wrong profile schema")
    if profile.get("model_family") != "exl3-r7":
        raise ContractError("stock control is not an EXL3 R7 profile")
    if profile.get("model_container_path") != "/models/glm52-exl3-r7-3.5bpw":
        raise ContractError("stock control has the wrong R7 checkpoint mount")
    if profile.get("identity") != _EXPECTED_IDENTITY:
        raise ContractError("stock control has the wrong attested R7 checkpoint")
    environment = profile.get("environment")
    arguments = profile.get("extra_vllm_args")
    if not isinstance(environment, dict) or not isinstance(arguments, list):
        raise ContractError("stock control has malformed environment or arguments")
    expected_environment = {
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
    }
    for name, expected in expected_environment.items():
        if environment.get(name) != expected:
            raise ContractError(f"stock DCP4 control requires {name}={expected}")
    for name in _FORBIDDEN_CUSTOM_DCP_ENV:
        if environment.get(name) not in (None, "", "0"):
            raise ContractError(f"stock DCP4 control must not enable {name}")
    for volume in profile.get("extra_volumes", []):
        if volume.get("container") == SHARED_CAPTURE_CONTAINER_PATH:
            raise ContractError(
                "stock DCP4 control must not mount the MTP shared capture stream overlay"
            )
        if volume.get("container") in _FORBIDDEN_CUSTOM_OVERLAY_TARGETS:
            raise ContractError(
                "stock DCP4 control must not mount a custom DCP/indexer/verifier overlay"
            )
    if _option_values(arguments, "--speculative-config"):
        raise ContractError("stock DCP4 control must be MTP-off")
    if "--enforce-eager" in arguments:
        raise ContractError("stock DCP4 control must use CUDA graphs")
    _require_option(arguments, "--dcp-comm-backend", "ag_rs")
    _require_option(arguments, "--dcp-kv-cache-interleave-size", "1")
    _require_option(arguments, "--kv-cache-dtype", "fp8_ds_mla")
    _require_option(arguments, "--quantization", "exl3")
    _require_option(arguments, "--moe-backend", "b12x")
    compilation_values = _option_values(arguments, "--compilation-config")
    if len(compilation_values) != 1:
        raise ContractError("stock DCP4 control requires one compilation config")
    compilation = json.loads(compilation_values[0])
    if compilation.get("cudagraph_mode") != "FULL_AND_PIECEWISE":
        raise ContractError("stock DCP4 control requires FULL_AND_PIECEWISE graphs")
    capture_sizes = compilation.get("cudagraph_capture_sizes")
    if not isinstance(capture_sizes, list) or 24 not in capture_sizes:
        raise ContractError("stock DCP4 control must capture the MTP2 Q24 ceiling")
    quant_values = _option_values(arguments, "--quantization-config")
    if len(quant_values) != 1:
        raise ContractError("stock DCP4 control requires one online quantization config")
    quantization = json.loads(quant_values[0])
    if "model.layers.78.eh_proj" not in quantization.get("ignore", []):
        raise ContractError("MTP layer 78 eh_proj must remain BF16")
    if "lm_head" not in quantization.get("ignore", []):
        raise ContractError("checkpoint MTP shared head must remain unquantized")


def derive_candidate(stock: dict) -> dict:
    """Return the exact fixed-MTP2 derivative of a stock-DCP4 profile."""

    validate_stock_control(stock)
    candidate = copy.deepcopy(stock)
    candidate["profile_id"] = f"{stock['profile_id']}-fixed-mtp2"
    candidate["environment"].update(MTP_ENVIRONMENT)
    candidate["extra_labels"].update(
        {
            "org.sparkring.r7.online-k6-scope": ONLINE_K6_SCOPE_LABEL,
            "org.sparkring.r7.target-weight-contract": (
                TARGET_WEIGHT_CONTRACT_LABEL
            ),
            "org.sparkring.r7.draft-weight-contract": DRAFT_WEIGHT_CONTRACT_LABEL,
            "org.sparkring.r7.capture-stream-contract": (
                SHARED_CAPTURE_CONTRACT_LABEL
            ),
        }
    )
    hook = candidate.get("attestation_hook")
    if not isinstance(hook, list) or len(hook) != 3 or not isinstance(hook[2], str):
        raise ContractError("stock DCP4 control has no shell attestation hook")
    attestation_line = f"{SHARED_CAPTURE_SHA256}  {SHARED_CAPTURE_CONTAINER_PATH}"
    marker = " | sha256sum --check --strict -"
    if hook[2].count(marker) != 1 or SHARED_CAPTURE_CONTAINER_PATH in hook[2]:
        raise ContractError(
            "stock DCP4 attestation cannot accept the MTP shared stream overlay"
        )
    hook[2] = hook[2].replace(marker, f" '{attestation_line}'{marker}", 1)
    spec = {"model": stock["model_container_path"], **SPECULATIVE_CONFIG}
    arguments = candidate["extra_vllm_args"]
    insert_at = arguments.index("--max-num-batched-tokens")
    arguments[insert_at:insert_at] = [
        "--speculative-config",
        json.dumps(spec, separators=(",", ":")),
    ]
    validate_weight_scope(candidate)
    validate_capture_stream_contract(candidate)
    return candidate


def validate_weight_scope(profile: dict) -> None:
    """Prove that online K6 is target-only and draft weights stay producer-owned."""

    environment = profile["environment"]
    arguments = profile["extra_vllm_args"]
    labels = profile.get("extra_labels", {})
    expected_labels = {
        "org.sparkring.r7.online-k6-scope": ONLINE_K6_SCOPE_LABEL,
        "org.sparkring.r7.target-weight-contract": TARGET_WEIGHT_CONTRACT_LABEL,
        "org.sparkring.r7.draft-weight-contract": DRAFT_WEIGHT_CONTRACT_LABEL,
        "org.sparkring.r7.capture-stream-contract": (
            SHARED_CAPTURE_CONTRACT_LABEL
        ),
    }
    for name, expected in expected_labels.items():
        if labels.get(name) != expected:
            raise ContractError(f"fixed-MTP2 weight contract requires {name}={expected}")
    if environment.get("ONLINE_QUANT") != "exl3-b6":
        raise ContractError("target online quantization must remain exl3-b6")
    if environment.get("VLLM_EXL3_ONLINE_TRELLIS_BITS") != "6":
        raise ContractError("target online Trellis encoding must remain K6")
    quantization_values = _option_values(arguments, "--quantization-config")
    if len(quantization_values) != 1:
        raise ContractError("target must have exactly one online quantization config")
    spec_values = _option_values(arguments, "--speculative-config")
    if len(spec_values) != 1:
        raise ContractError("fixed-MTP2 requires exactly one speculative config")
    spec = json.loads(spec_values[0])
    if "kv_cache_dtype" in spec:
        raise ContractError(
            "draft must inherit target fp8_ds_mla without reconstructing "
            "CacheConfig with an explicit 16-token block size"
        )
    expected_spec = {"model": profile["model_container_path"], **SPECULATIVE_CONFIG}
    if spec != expected_spec:
        raise ContractError(
            "draft must use exact checkpoint EXL3/BF16 contract without "
            "quantization_config or online-K6 overrides"
        )


def validate_capture_stream_contract(profile: dict) -> None:
    """Require the target and draft managers to share one dedicated stream."""

    if profile["environment"].get("VLLM_SPARK_SHARED_CAPTURE_STREAM") != "1":
        raise ContractError("fixed-MTP2 requires the shared capture stream gate")
    if profile.get("extra_labels", {}).get(
        "org.sparkring.r7.capture-stream-contract"
    ) != SHARED_CAPTURE_CONTRACT_LABEL:
        raise ContractError("fixed-MTP2 has the wrong shared capture stream label")
    mounted = [
        volume
        for volume in profile.get("extra_volumes", [])
        if volume.get("container") == SHARED_CAPTURE_CONTAINER_PATH
    ]
    if mounted:
        raise ContractError(
            "fixed-MTP2 requires the shared capture stream implementation baked into the image"
        )
    hook = profile.get("attestation_hook")
    if not isinstance(hook, list) or len(hook) != 3 or not isinstance(hook[2], str):
        raise ContractError("fixed-MTP2 has no shell attestation hook")
    expected_line = f"{SHARED_CAPTURE_SHA256}  {SHARED_CAPTURE_CONTAINER_PATH}"
    if hook[2].count(expected_line) != 1:
        raise ContractError("fixed-MTP2 has the wrong shared capture stream SHA")


def validate_candidate(stock: dict, candidate: dict) -> None:
    """Fail closed unless candidate is exactly the expected MTP2 derivative."""

    validate_weight_scope(candidate)
    validate_capture_stream_contract(candidate)
    expected = derive_candidate(stock)
    if candidate != expected:
        raise ContractError(
            "candidate differs from the exact fixed-MTP2 stock-DCP4 derivative"
        )


def derive_site_text(stock_site: str) -> str:
    """Change only the site-level MTP declaration from off/0 to static/2."""

    required = (
        "  tensor_parallel_size: 4",
        "  decode_context_parallel_size: 4",
        "  kv_cache_bytes_per_rank: 9000000000",
        "  max_num_seqs: 8",
    )
    for line in required:
        if stock_site.count(line) != 1:
            raise ContractError(f"stock site requires exactly one {line.strip()}")
    mode = '  mtp_mode: "off"'
    tokens = "  mtp_tokens: 0"
    if stock_site.count(mode) != 1 or stock_site.count(tokens) != 1:
        raise ContractError("stock site must declare exactly mtp_mode off and mtp_tokens 0")
    return stock_site.replace(mode, '  mtp_mode: "static"').replace(
        tokens, "  mtp_tokens: 2"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-profile", type=Path, required=True)
    parser.add_argument("--stock-site", type=Path, required=True)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--candidate-site", type=Path, required=True)
    parser.add_argument("--rollback-profile", type=Path, required=True)
    args = parser.parse_args()
    try:
        stock_bytes = args.stock_profile.read_bytes()
        stock = json.loads(stock_bytes)
        candidate = derive_candidate(stock)
        validate_candidate(stock, candidate)
        stock_site = args.stock_site.read_text(encoding="utf-8")
        candidate_site = derive_site_text(stock_site)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    for output in (
        args.candidate_profile,
        args.candidate_site,
        args.rollback_profile,
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_profile.write_text(
        json.dumps(candidate, indent=2) + "\n", encoding="utf-8"
    )
    args.candidate_site.write_text(candidate_site, encoding="utf-8")
    shutil.copyfile(args.stock_profile, args.rollback_profile)
    if args.rollback_profile.read_bytes() != stock_bytes:
        parser.error("rollback profile is not byte-identical to stock control")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
