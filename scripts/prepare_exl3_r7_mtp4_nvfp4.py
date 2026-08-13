#!/usr/bin/env python3
"""Derive the 262K dynamic-NVFP4 R7 profile from fixed-MTP4 KV9.25.

The transformation changes only the model limit, prefill capacity, batched
token ceiling, and KV representation required by the operator-accepted
368-byte MLA record.  Input hashes are supplied by the preceding generated
stage rather than embedded operator-local receipts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
from pathlib import Path

import prepare_exl3_r7_mtp4 as mtp4


class ContractError(ValueError):
    """The source or derived profile violates the dynamic-NVFP4 contract."""


SOURCE_MODEL_LEN = 65_536
CANDIDATE_MODEL_LEN = 262_144
SOURCE_BATCHED_TOKENS = 2_048
CANDIDATE_BATCHED_TOKENS = 4_096
KV_CACHE_DTYPE = "nvfp4_ds_mla"
KV_CONTRACT_LABEL = "nvfp4-dynamic-per-token+fp8-rope-368-byte"
REPORTED_KV_CAPACITY_TOKENS = 1_156_864
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _option_values(arguments: list[str], option: str) -> list[str]:
    values: list[str] = []
    for index, value in enumerate(arguments):
        if value == option:
            if index + 1 >= len(arguments):
                raise ContractError(f"source profile has no value after {option}")
            values.append(arguments[index + 1])
    return values


def _replace_option(arguments: list[str], option: str, expected: str, value: str) -> None:
    if _option_values(arguments, option) != [expected]:
        raise ContractError(f"source profile requires exactly {option}={expected}")
    arguments[arguments.index(option) + 1] = value


def validate_source_profile(profile: dict) -> None:
    """Require the complete fixed-MTP4 contract before changing KV format."""

    try:
        mtp4.validate_mtp4_contract(profile)
    except (KeyError, TypeError, mtp4.ContractError) as exc:
        raise ContractError(f"source fixed-MTP4 profile is not qualified: {exc}") from exc
    environment = profile.get("environment")
    arguments = profile.get("extra_vllm_args")
    labels = profile.get("extra_labels")
    if not isinstance(environment, dict) or not isinstance(arguments, list):
        raise ContractError("source environment or arguments are malformed")
    if not isinstance(labels, dict):
        raise ContractError("source labels are malformed")
    if environment.get("VLLM_EXL3_PREFILL_CAPACITY") != str(SOURCE_BATCHED_TOKENS):
        raise ContractError("source EXL3 prefill capacity must be 2048")
    if _option_values(arguments, "--max-num-batched-tokens") != [
        str(SOURCE_BATCHED_TOKENS)
    ]:
        raise ContractError("source batched-token ceiling must be 2048")
    if _option_values(arguments, "--kv-cache-dtype") != ["fp8_ds_mla"]:
        raise ContractError("source KV representation must be fp8_ds_mla")
    for key in ("KV_FP8_ROPE", "VLLM_NVFP4_MLA_DYNAMIC_SCALE"):
        if key in environment:
            raise ContractError(f"source profile already declares {key}")
    if "org.sparkring.r7.kv-contract" in labels:
        raise ContractError("source profile already carries a KV contract label")


def derive_candidate(source: dict) -> dict:
    """Return the exact 262K dynamic-NVFP4 derivative."""

    validate_source_profile(source)
    candidate = copy.deepcopy(source)
    candidate["profile_id"] = f"{source['profile_id']}-nvfp4-rope8-ctx256k-b4096"
    candidate["environment"].update(
        {
            "KV_FP8_ROPE": "1",
            "VLLM_EXL3_PREFILL_CAPACITY": str(CANDIDATE_BATCHED_TOKENS),
            "VLLM_NVFP4_MLA_DYNAMIC_SCALE": "1",
        }
    )
    arguments = candidate["extra_vllm_args"]
    _replace_option(
        arguments,
        "--max-num-batched-tokens",
        str(SOURCE_BATCHED_TOKENS),
        str(CANDIDATE_BATCHED_TOKENS),
    )
    _replace_option(arguments, "--kv-cache-dtype", "fp8_ds_mla", KV_CACHE_DTYPE)
    candidate["extra_labels"]["org.sparkring.r7.kv-contract"] = KV_CONTRACT_LABEL
    validate_candidate(source, candidate)
    return candidate


def validate_candidate(source: dict, candidate: dict) -> None:
    """Require exactly the declared dynamic-NVFP4 semantic delta."""

    validate_source_profile(source)
    expected = copy.deepcopy(source)
    expected["profile_id"] = f"{source['profile_id']}-nvfp4-rope8-ctx256k-b4096"
    expected["environment"].update(
        {
            "KV_FP8_ROPE": "1",
            "VLLM_EXL3_PREFILL_CAPACITY": str(CANDIDATE_BATCHED_TOKENS),
            "VLLM_NVFP4_MLA_DYNAMIC_SCALE": "1",
        }
    )
    _replace_option(
        expected["extra_vllm_args"],
        "--max-num-batched-tokens",
        str(SOURCE_BATCHED_TOKENS),
        str(CANDIDATE_BATCHED_TOKENS),
    )
    _replace_option(
        expected["extra_vllm_args"],
        "--kv-cache-dtype",
        "fp8_ds_mla",
        KV_CACHE_DTYPE,
    )
    expected["extra_labels"]["org.sparkring.r7.kv-contract"] = KV_CONTRACT_LABEL
    if candidate != expected:
        raise ContractError("candidate differs outside the dynamic-NVFP4 allowlist")


def derive_site_text(source: str) -> str:
    """Change only the site-level maximum model length."""

    source_line = f"  max_model_len: {SOURCE_MODEL_LEN}"
    candidate_line = f"  max_model_len: {CANDIDATE_MODEL_LEN}"
    required = (
        "  tensor_parallel_size: 4",
        "  decode_context_parallel_size: 4",
        '  mtp_mode: "static"',
        "  mtp_tokens: 4",
        "  kv_cache_bytes_per_rank: 9250000000",
        "  max_num_seqs: 8",
    )
    for line in required:
        if source.count(line) != 1:
            raise ContractError(f"source site must declare exactly one {line.strip()}")
    if source.count(source_line) != 1 or candidate_line in source:
        raise ContractError("source site must declare exactly max_model_len 65536")
    candidate = source.replace(source_line, candidate_line, 1)
    if candidate.replace(candidate_line, source_line, 1) != source:
        raise ContractError("candidate site changed beyond maximum model length")
    return candidate


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _expected_sha256(value: str, role: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ContractError(f"expected {role} SHA-256 must be 64 lowercase hex")
    return value


def _reject_path_collisions(inputs: list[Path], outputs: list[Path]) -> None:
    input_paths = {path.resolve() for path in inputs}
    output_paths = [path.resolve() for path in outputs]
    if any(path in input_paths for path in output_paths):
        raise ContractError("dynamic-NVFP4 outputs must not overwrite an input")
    if len(set(output_paths)) != len(output_paths):
        raise ContractError("dynamic-NVFP4 output paths must be distinct")


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
        profile_bytes = args.source_profile.read_bytes()
        site_bytes = args.source_site.read_bytes()
        profile_sha = _sha256_bytes(profile_bytes)
        site_sha = _sha256_bytes(site_bytes)
        if profile_sha != _expected_sha256(args.expected_profile_sha256, "profile"):
            raise ContractError("source profile SHA-256 mismatch")
        if site_sha != _expected_sha256(args.expected_site_sha256, "site"):
            raise ContractError("source site SHA-256 mismatch")
        source = json.loads(profile_bytes)
        candidate = derive_candidate(source)
        candidate_site = derive_site_text(site_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ContractError) as exc:
        parser.error(str(exc))

    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_profile.write_text(
        json.dumps(candidate, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    args.candidate_site.write_text(candidate_site, encoding="utf-8", newline="")
    shutil.copyfile(args.source_profile, args.rollback_profile)
    shutil.copyfile(args.source_site, args.rollback_site)
    if args.rollback_profile.read_bytes() != profile_bytes:
        parser.error("rollback profile is not byte-identical to the source")
    if args.rollback_site.read_bytes() != site_bytes:
        parser.error("rollback site is not byte-identical to the source")

    print(
        json.dumps(
            {
                "candidate_profile_sha256": _sha256_bytes(
                    args.candidate_profile.read_bytes()
                ),
                "candidate_site_sha256": _sha256_bytes(args.candidate_site.read_bytes()),
                "kv_cache_dtype": KV_CACHE_DTYPE,
                "max_model_len": CANDIDATE_MODEL_LEN,
                "max_num_batched_tokens": CANDIDATE_BATCHED_TOKENS,
                "reported_kv_capacity_tokens": REPORTED_KV_CAPACITY_TOKENS,
                "rollback_profile_sha256": profile_sha,
                "rollback_site_sha256": site_sha,
                "source_profile_sha256": profile_sha,
                "source_site_sha256": site_sha,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
