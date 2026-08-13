#!/usr/bin/env python3
"""Prepare and validate an isolated fixed-MTP3 derivative of qualified MTP2."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import prepare_exl3_r7_mtp2 as mtp2


class ContractError(ValueError):
    """The fixed-MTP3 profile is not the exact qualified-MTP2 derivative."""


MTP2_SUFFIX = "-fixed-mtp2"
MTP3_SUFFIX = "-fixed-mtp3"
MTP3_DEPTH = 3
MTP3_MAX_QUERY_ROWS = 32


def _option_values(arguments: list[str], option: str) -> list[str]:
    """Return all values for one command-line option."""

    try:
        return mtp2._option_values(arguments, option)
    except mtp2.ContractError as exc:
        raise ContractError(str(exc)) from exc


def _require_option(arguments: list[str], option: str, expected: str) -> None:
    if _option_values(arguments, option) != [expected]:
        raise ContractError(f"fixed-MTP3 requires {option}={expected}")


def _load_single_json_option(
    arguments: list[str], option: str, *, role: str
) -> dict:
    values = _option_values(arguments, option)
    if len(values) != 1:
        raise ContractError(f"fixed-MTP3 requires exactly one {role}")
    try:
        value = json.loads(values[0])
    except json.JSONDecodeError as exc:
        raise ContractError(f"fixed-MTP3 {role} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ContractError(f"fixed-MTP3 {role} must be a JSON object")
    return value


def validate_mtp2_control(stock: dict, control: dict) -> None:
    """Require the byte source to be the exact qualified fixed-MTP2 profile."""

    try:
        mtp2.validate_candidate(stock, control)
    except (KeyError, TypeError, mtp2.ContractError) as exc:
        raise ContractError(f"MTP2 control is not exact: {exc}") from exc
    profile_id = control.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id.endswith(MTP2_SUFFIX):
        raise ContractError("MTP2 control profile_id must end in -fixed-mtp2")


def expected_spec(profile: dict) -> dict:
    """Return the exact fixed-depth-three draft configuration."""

    config = copy.deepcopy(mtp2.SPECULATIVE_CONFIG)
    config["num_speculative_tokens"] = MTP3_DEPTH
    return {"model": profile["model_container_path"], **config}


def validate_mtp3_contract(profile: dict) -> None:
    """Fail closed on depth, query capacity, weights, graphs, or stream drift."""

    profile_id = profile.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id.endswith(MTP3_SUFFIX):
        raise ContractError("fixed-MTP3 profile_id must end in -fixed-mtp3")
    environment = profile.get("environment")
    arguments = profile.get("extra_vllm_args")
    if not isinstance(environment, dict) or not isinstance(arguments, list):
        raise ContractError("fixed-MTP3 has malformed environment or arguments")

    expected_environment = {
        "SPARK_ADAPTIVE_MTP_CONTROL": "0",
        "SPARK_GLM52_MTP_INDEX_REUSE": "0",
        "VLLM_SPARK_MAX_QUERY_ROWS": str(MTP3_MAX_QUERY_ROWS),
        "VLLM_SPARK_MTP_ADAPTIVE_WINDOW": "0",
        "VLLM_SPARK_MTP_MODE_ID": "fixed-mtp3",
        "VLLM_SPARK_MTP_TOKENS": str(MTP3_DEPTH),
        "VLLM_SPARK_TRUE_ADAPTIVE_DRAFT": "0",
        "VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM": "0",
    }
    for name, expected in expected_environment.items():
        actual = environment.get(name)
        if actual != expected:
            if name == "VLLM_SPARK_MAX_QUERY_ROWS":
                raise ContractError("fixed-MTP3 requires query-row ceiling 32")
            if name == "VLLM_SPARK_MTP_MODE_ID":
                raise ContractError("fixed-MTP3 requires mode fixed-mtp3")
            if name == "VLLM_SPARK_MTP_TOKENS":
                raise ContractError("fixed-MTP3 requires speculative depth 3")
            raise ContractError(f"fixed-MTP3 requires {name}={expected}")
    if environment.get("VLLM_ADAPTIVE_SPEC_DEPTHS") is not None:
        raise ContractError("fixed-MTP3 must not enable adaptive spec depths")
    if environment.get("ONLINE_QUANT") != "exl3-b6" or environment.get(
        "VLLM_EXL3_ONLINE_TRELLIS_BITS"
    ) != "6":
        raise ContractError("fixed-MTP3 must preserve target online K6")
    if environment.get("VLLM_SPARK_SHARED_CAPTURE_STREAM") != "1":
        raise ContractError("fixed-MTP3 requires the shared capture stream")

    labels = profile.get("extra_labels", {})
    expected_labels = {
        "org.sparkring.r7.online-k6-scope": mtp2.ONLINE_K6_SCOPE_LABEL,
        "org.sparkring.r7.target-weight-contract": (
            mtp2.TARGET_WEIGHT_CONTRACT_LABEL
        ),
        "org.sparkring.r7.draft-weight-contract": (
            mtp2.DRAFT_WEIGHT_CONTRACT_LABEL
        ),
        "org.sparkring.r7.capture-stream-contract": (
            mtp2.SHARED_CAPTURE_CONTRACT_LABEL
        ),
    }
    for name, expected in expected_labels.items():
        if labels.get(name) != expected:
            raise ContractError(f"fixed-MTP3 requires {name}={expected}")
    try:
        mtp2.validate_capture_stream_contract(profile)
    except (KeyError, TypeError, mtp2.ContractError) as exc:
        raise ContractError(f"fixed-MTP3 shared capture stream contract: {exc}") from exc

    spec = _load_single_json_option(
        arguments,
        "--speculative-config",
        role="speculative config",
    )
    if spec.get("num_speculative_tokens") != MTP3_DEPTH:
        raise ContractError("fixed-MTP3 requires speculative depth 3")
    if "kv_cache_dtype" in spec:
        raise ContractError(
            "fixed-MTP3 draft must inherit target fp8_ds_mla without "
            "reconstructing CacheConfig"
        )
    if "quantization_config" in spec:
        raise ContractError(
            "fixed-MTP3 online K6 scope is target-only; draft must not have "
            "quantization_config"
        )
    if spec != expected_spec(profile):
        raise ContractError(
            "fixed-MTP3 requires the exact checkpoint EXL3/BF16 draft contract"
        )

    _require_option(arguments, "--kv-cache-dtype", "fp8_ds_mla")
    _require_option(arguments, "--quantization", "exl3")
    _require_option(arguments, "--moe-backend", "b12x")
    _require_option(arguments, "--dcp-comm-backend", "ag_rs")
    _require_option(arguments, "--dcp-kv-cache-interleave-size", "1")
    if "--enforce-eager" in arguments:
        raise ContractError("fixed-MTP3 requires CUDA graphs")
    compilation = _load_single_json_option(
        arguments,
        "--compilation-config",
        role="compilation config",
    )
    if compilation.get("cudagraph_mode") != "FULL_AND_PIECEWISE":
        raise ContractError("fixed-MTP3 requires FULL_AND_PIECEWISE graphs")
    capture_sizes = compilation.get("cudagraph_capture_sizes")
    if not isinstance(capture_sizes, list) or MTP3_MAX_QUERY_ROWS not in capture_sizes:
        raise ContractError("fixed-MTP3 compilation config must capture Q32")


def derive_candidate(stock: dict, mtp2_control: dict) -> dict:
    """Return the exact fixed-depth-three derivative of qualified MTP2."""

    validate_mtp2_control(stock, mtp2_control)
    candidate = copy.deepcopy(mtp2_control)
    profile_id = candidate["profile_id"]
    candidate["profile_id"] = (
        profile_id.removesuffix(MTP2_SUFFIX) + MTP3_SUFFIX
    )
    candidate["environment"].update(
        {
            "VLLM_SPARK_MAX_QUERY_ROWS": str(MTP3_MAX_QUERY_ROWS),
            "VLLM_SPARK_MTP_MODE_ID": "fixed-mtp3",
            "VLLM_SPARK_MTP_TOKENS": str(MTP3_DEPTH),
        }
    )
    arguments = candidate["extra_vllm_args"]
    values = _option_values(arguments, "--speculative-config")
    if len(values) != 1:
        raise ContractError("MTP2 control requires exactly one speculative config")
    try:
        spec = json.loads(values[0])
    except json.JSONDecodeError as exc:
        raise ContractError("MTP2 speculative config must be valid JSON") from exc
    spec["num_speculative_tokens"] = MTP3_DEPTH
    spec_index = arguments.index("--speculative-config") + 1
    arguments[spec_index] = json.dumps(spec, separators=(",", ":"))
    validate_mtp3_contract(candidate)
    return candidate


def validate_candidate(stock: dict, mtp2_control: dict, candidate: dict) -> None:
    """Require exactly the MTP2-to-MTP3 semantic delta."""

    validate_mtp2_control(stock, mtp2_control)
    validate_mtp3_contract(candidate)
    expected = derive_candidate(stock, mtp2_control)
    if candidate != expected:
        raise ContractError(
            "candidate differs from the exact fixed-MTP3 derivative of qualified MTP2"
        )


def derive_site_text(mtp2_site: str) -> str:
    """Change only the site-level static depth from two to three."""

    required = (
        "  tensor_parallel_size: 4",
        "  decode_context_parallel_size: 4",
        '  mtp_mode: "static"',
        "  kv_cache_bytes_per_rank: 9000000000",
        "  max_num_seqs: 8",
    )
    for line in required:
        if mtp2_site.count(line) != 1:
            raise ContractError(f"MTP2 site requires exactly one {line.strip()}")
    tokens = "  mtp_tokens: 2"
    if mtp2_site.count(tokens) != 1:
        raise ContractError("MTP2 site must declare exactly mtp_tokens 2")
    return mtp2_site.replace(tokens, "  mtp_tokens: 3")


def _reject_path_collisions(inputs: list[Path], outputs: list[Path]) -> None:
    input_paths = {path.resolve() for path in inputs}
    output_paths = [path.resolve() for path in outputs]
    if any(path in input_paths for path in output_paths):
        raise ContractError("MTP3 outputs must not overwrite an input")
    if len(set(output_paths)) != len(output_paths):
        raise ContractError("MTP3 output paths must be distinct")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-dcp4-profile", type=Path, required=True)
    parser.add_argument("--mtp2-profile", type=Path, required=True)
    parser.add_argument("--mtp2-site", type=Path, required=True)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--candidate-site", type=Path, required=True)
    parser.add_argument("--rollback-profile", type=Path, required=True)
    args = parser.parse_args()
    inputs = [args.stock_dcp4_profile, args.mtp2_profile, args.mtp2_site]
    outputs = [
        args.candidate_profile,
        args.candidate_site,
        args.rollback_profile,
    ]
    try:
        _reject_path_collisions(inputs, outputs)
        stock = json.loads(args.stock_dcp4_profile.read_bytes())
        mtp2_bytes = args.mtp2_profile.read_bytes()
        mtp2_control = json.loads(mtp2_bytes)
        candidate = derive_candidate(stock, mtp2_control)
        validate_candidate(stock, mtp2_control, candidate)
        candidate_site = derive_site_text(
            args.mtp2_site.read_text(encoding="utf-8")
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_profile.write_text(
        json.dumps(candidate, indent=2) + "\n", encoding="utf-8"
    )
    args.candidate_site.write_text(candidate_site, encoding="utf-8")
    shutil.copyfile(args.mtp2_profile, args.rollback_profile)
    if args.rollback_profile.read_bytes() != mtp2_bytes:
        parser.error("rollback profile is not byte-identical to qualified MTP2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
