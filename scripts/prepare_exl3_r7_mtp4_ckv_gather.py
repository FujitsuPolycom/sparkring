#!/usr/bin/env python3
"""Prepare a bounded full-CKV prefill-gather arm from the live MTP4 profile."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
from pathlib import Path


class ContractError(ValueError):
    """The source or derived profile violates the sealed experiment contract."""


SOURCE_PROFILE_ID = (
    "glm52-exl3-r7-3.5bpw-fixed-mtp4-nvfp4-rope8-ctx256k-b4096"
)
CANDIDATE_PROFILE_ID = f"{SOURCE_PROFILE_ID}-ckv-gather"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

CKV_LABEL = "org.sparkring.r7.ckv-prefill-contract"
CKV_LABEL_VALUE = "dcp4-full-ckv-prefetch1-max262144"
CKV_GATHER_MAX_TOKENS = 262_144
CKV_PREFETCH_DEPTH = 1
CKV_RECORD_BYTES = 368
DCP_WORLD_SIZE = 4
MAX_NUM_SEQS = 8
DCP_KV_INTERLEAVE_SIZE = 1
KV_BLOCK_SIZE = 64
CKV_EXECUTION_LANES = 2  # One target and one speculative-runner lane.
KV_CACHE_BYTES_PER_RANK = 9_250_000_000
REPORTED_KV_CAPACITY_TOKENS = 1_156_864


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


CKV_LOCAL_CAPACITY_TOKENS = (
    _ceil_div(
        _ceil_div(CKV_GATHER_MAX_TOKENS, DCP_WORLD_SIZE)
        + MAX_NUM_SEQS * DCP_KV_INTERLEAVE_SIZE,
        KV_BLOCK_SIZE,
    )
    * KV_BLOCK_SIZE
)
CKV_WORKSPACE_BYTES_PER_LANE = (
    1 + (CKV_PREFETCH_DEPTH + 1) * DCP_WORLD_SIZE
) * CKV_LOCAL_CAPACITY_TOKENS * CKV_RECORD_BYTES
CKV_WORKSPACE_POOL_BYTES_PER_RANK = (
    CKV_WORKSPACE_BYTES_PER_LANE * CKV_EXECUTION_LANES
)
CKV_WORKSPACE_POOL_MIB_PER_RANK = CKV_WORKSPACE_POOL_BYTES_PER_RANK / (1024**2)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _expected_sha256(value: str, role: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ContractError(f"expected {role} SHA-256 must be 64 lowercase hex")
    return value


def _arg_value(args: list[str], name: str) -> str:
    if args.count(name) != 1:
        raise ContractError(f"source profile must declare exactly one {name}")
    index = args.index(name)
    if index + 1 >= len(args):
        raise ContractError(f"source profile has no value after {name}")
    return args[index + 1]


def validate_source_profile(profile: dict) -> None:
    """Require the exact live MTP4/NVFP4/DCP4 behavior before deriving an arm."""

    if profile.get("profile_id") != SOURCE_PROFILE_ID:
        raise ContractError("source profile_id is not the live b4096 profile")
    environment = profile.get("environment")
    args = profile.get("extra_vllm_args")
    labels = profile.get("extra_labels")
    if not isinstance(environment, dict) or not isinstance(args, list):
        raise ContractError("source profile environment/args are malformed")
    if not isinstance(labels, dict):
        raise ContractError("source profile labels are malformed")

    expected_environment = {
        "ONLINE_QUANT": "exl3-b6",
        "VLLM_EXL3_ONLINE_TRELLIS_BITS": "6",
        "VLLM_EXL3_PREFILL_CAPACITY": "4096",
        "VLLM_SPARK_MAX_QUERY_ROWS": "40",
        "VLLM_SPARK_MTP_TOKENS": "4",
        "VLLM_USE_B12X_DCP_A2A": "1",
        "KV_FP8_ROPE": "1",
        "VLLM_NVFP4_MLA_DYNAMIC_SCALE": "1",
    }
    descriptions = {
        "VLLM_EXL3_PREFILL_CAPACITY": "prefill capacity",
        "KV_FP8_ROPE": "FP8-RoPE contract",
        "VLLM_NVFP4_MLA_DYNAMIC_SCALE": "dynamic NVFP4 contract",
    }
    for key, expected in expected_environment.items():
        if environment.get(key) != expected:
            description = descriptions.get(key, key)
            raise ContractError(f"source {description} must be {expected}")

    for key in (
        "VLLM_B12X_MLA_CKV_GATHER",
        "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS",
        "VLLM_B12X_MLA_CKV_PREFETCH_DEPTH",
        "VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB",
    ):
        if key in environment:
            raise ContractError(f"source profile already declares {key}")

    if _arg_value(args, "--max-num-batched-tokens") != "4096":
        raise ContractError("source --max-num-batched-tokens must be 4096")
    if _arg_value(args, "--kv-cache-dtype") != "nvfp4_ds_mla":
        raise ContractError("source KV-cache dtype must be nvfp4_ds_mla")
    speculative = json.loads(_arg_value(args, "--speculative-config"))
    if speculative.get("num_speculative_tokens") != 4:
        raise ContractError("source speculative depth must be fixed-MTP4")
    if labels.get("org.sparkring.r7.kv-contract") != (
        "nvfp4-dynamic-per-token+fp8-rope-368-byte"
    ):
        raise ContractError("source 368-byte NVFP4 KV label is missing")
    if CKV_LABEL in labels:
        raise ContractError("source profile already carries a CKV-prefill label")


def derive_candidate(source: dict) -> dict:
    """Enable bounded full-CKV prefill gather without changing serving geometry."""

    validate_source_profile(source)
    candidate = copy.deepcopy(source)
    candidate["profile_id"] = CANDIDATE_PROFILE_ID
    candidate["environment"]["VLLM_B12X_MLA_CKV_GATHER"] = "1"
    candidate["environment"]["VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS"] = str(
        CKV_GATHER_MAX_TOKENS
    )
    candidate["extra_labels"][CKV_LABEL] = CKV_LABEL_VALUE
    return candidate


def validate_source_site(site_text: str) -> None:
    expected_lines = (
        "  decode_context_parallel_size: 4",
        "  max_model_len: 262144",
        "  kv_cache_bytes_per_rank: 9250000000",
        "  max_num_seqs: 8",
    )
    for line in expected_lines:
        if site_text.count(line) != 1:
            raise ContractError(f"source site must declare exactly one {line.strip()}")


def _reject_path_collisions(inputs: list[Path], outputs: list[Path]) -> None:
    input_paths = {path.resolve() for path in inputs}
    output_paths = [path.resolve() for path in outputs]
    if any(path in input_paths for path in output_paths):
        raise ContractError("CKV-gather outputs must not overwrite an input")
    if len(set(output_paths)) != len(output_paths):
        raise ContractError("CKV-gather output paths must be distinct")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-profile", type=Path, required=True)
    parser.add_argument("--source-site", type=Path, required=True)
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--expected-site-sha256", required=True)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--candidate-site", type=Path, required=True)
    parser.add_argument("--rollback-profile", type=Path, required=True)
    parser.add_argument("--rollback-site", type=Path, required=True)
    args = parser.parse_args()
    inputs = [args.source_profile, args.source_site]
    outputs = [
        args.candidate_profile,
        args.candidate_site,
        args.rollback_profile,
        args.rollback_site,
    ]

    try:
        _reject_path_collisions(inputs, outputs)
        source_profile_bytes = args.source_profile.read_bytes()
        source_site_bytes = args.source_site.read_bytes()
        source_profile_sha256 = _sha256_bytes(source_profile_bytes)
        source_site_sha256 = _sha256_bytes(source_site_bytes)
        if source_profile_sha256 != _expected_sha256(
            args.expected_profile_sha256, "profile"
        ):
            raise ContractError("source profile SHA-256 mismatch")
        if source_site_sha256 != _expected_sha256(
            args.expected_site_sha256, "site"
        ):
            raise ContractError("source site SHA-256 mismatch")
        source_profile = json.loads(source_profile_bytes)
        source_site = source_site_bytes.decode("utf-8")
        validate_source_site(source_site)
        candidate_profile = derive_candidate(source_profile)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ContractError) as exc:
        parser.error(str(exc))

    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_profile.write_text(
        json.dumps(candidate_profile, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    shutil.copyfile(args.source_site, args.candidate_site)
    shutil.copyfile(args.source_profile, args.rollback_profile)
    shutil.copyfile(args.source_site, args.rollback_site)

    if args.candidate_site.read_bytes() != source_site_bytes:
        parser.error("candidate site is not byte-identical to the live site")
    if args.rollback_profile.read_bytes() != source_profile_bytes:
        parser.error("rollback profile is not byte-identical to the live profile")
    if args.rollback_site.read_bytes() != source_site_bytes:
        parser.error("rollback site is not byte-identical to the live site")

    print(
        json.dumps(
            {
                "candidate_profile_sha256": _sha256(args.candidate_profile),
                "candidate_site_sha256": _sha256(args.candidate_site),
                "ckv_gather_max_tokens": CKV_GATHER_MAX_TOKENS,
                "ckv_prefetch_depth": CKV_PREFETCH_DEPTH,
                "ckv_record_bytes": CKV_RECORD_BYTES,
                "ckv_workspace_bytes_per_lane": CKV_WORKSPACE_BYTES_PER_LANE,
                "ckv_workspace_pool_bytes_per_rank": (
                    CKV_WORKSPACE_POOL_BYTES_PER_RANK
                ),
                "kv_cache_bytes_per_rank": KV_CACHE_BYTES_PER_RANK,
                "reported_kv_capacity_tokens": REPORTED_KV_CAPACITY_TOKENS,
                "rollback_profile_sha256": _sha256(args.rollback_profile),
                "rollback_site_sha256": _sha256(args.rollback_site),
                "source_profile_sha256": source_profile_sha256,
                "source_site_sha256": source_site_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
