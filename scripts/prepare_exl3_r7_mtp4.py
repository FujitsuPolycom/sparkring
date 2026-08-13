#!/usr/bin/env python3
"""Prepare the fixed-MTP4 KV9.25 derivative of qualified fixed-MTP3."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
from pathlib import Path

import prepare_exl3_r7_mtp3 as mtp3


class ContractError(ValueError):
    """The fixed-MTP4 arm is not the exact qualified-MTP3 derivative."""


MTP3_SUFFIX = "-fixed-mtp3"
MTP4_SUFFIX = "-fixed-mtp4"
MTP3_DEPTH = 3
MTP4_DEPTH = 4
MTP3_MAX_QUERY_ROWS = 32
MTP4_MAX_QUERY_ROWS = 40
KV925_BYTES_PER_RANK = 9_250_000_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _option_values(arguments: list[str], option: str) -> list[str]:
    try:
        return mtp3._option_values(arguments, option)
    except mtp3.ContractError as exc:
        raise ContractError(str(exc)) from exc


def _require_option(arguments: list[str], option: str, expected: str) -> None:
    if _option_values(arguments, option) != [expected]:
        raise ContractError(f"fixed-MTP4 requires {option}={expected}")


def _load_single_json_option(
    arguments: list[str], option: str, *, role: str
) -> dict:
    values = _option_values(arguments, option)
    if len(values) != 1:
        raise ContractError(f"fixed-MTP4 requires exactly one {role}")
    try:
        value = json.loads(values[0])
    except json.JSONDecodeError as exc:
        raise ContractError(f"fixed-MTP4 {role} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ContractError(f"fixed-MTP4 {role} must be a JSON object")
    return value


def _replace_option(arguments: list[str], option: str, value: str) -> None:
    if _option_values(arguments, option) == []:
        raise ContractError(f"qualified fixed-MTP3 requires exactly one {option}")
    if len(_option_values(arguments, option)) != 1:
        raise ContractError(f"qualified fixed-MTP3 requires exactly one {option}")
    arguments[arguments.index(option) + 1] = value


def _compact_json(value: dict) -> str:
    return json.dumps(value, separators=(",", ":"))


def validate_mtp3_source(profile: dict) -> None:
    """Require the exact fixed-MTP3 capture geometry used by KV9.25."""

    try:
        mtp3.validate_mtp3_contract(profile)
    except (KeyError, TypeError, mtp3.ContractError) as exc:
        raise ContractError(f"fixed-MTP3 source is not qualified: {exc}") from exc

    arguments = profile.get("extra_vllm_args")
    if not isinstance(arguments, list):
        raise ContractError("fixed-MTP3 source has malformed arguments")
    compilation = _load_single_json_option(
        arguments,
        "--compilation-config",
        role="source compilation config",
    )
    if compilation.get("cudagraph_capture_sizes") != list(
        range(1, MTP3_MAX_QUERY_ROWS + 1)
    ):
        raise ContractError("fixed-MTP3 source must capture Q1 through Q32")
    if _option_values(arguments, "--max-cudagraph-capture-size") != ["32"]:
        raise ContractError(
            "fixed-MTP3 source requires capture-size ceiling 32"
        )


def _derive_candidate_unvalidated(source: dict) -> dict:
    candidate = copy.deepcopy(source)
    candidate["profile_id"] = (
        candidate["profile_id"].removesuffix(MTP3_SUFFIX) + MTP4_SUFFIX
    )
    candidate["environment"].update(
        {
            "VLLM_SPARK_MAX_QUERY_ROWS": str(MTP4_MAX_QUERY_ROWS),
            "VLLM_SPARK_MTP_MODE_ID": "fixed-mtp4",
            "VLLM_SPARK_MTP_TOKENS": str(MTP4_DEPTH),
        }
    )
    arguments = candidate["extra_vllm_args"]

    spec = _load_single_json_option(
        arguments,
        "--speculative-config",
        role="speculative config",
    )
    spec["num_speculative_tokens"] = MTP4_DEPTH
    _replace_option(arguments, "--speculative-config", _compact_json(spec))

    compilation = _load_single_json_option(
        arguments,
        "--compilation-config",
        role="compilation config",
    )
    compilation["cudagraph_capture_sizes"] = list(
        range(1, MTP4_MAX_QUERY_ROWS + 1)
    )
    _replace_option(arguments, "--compilation-config", _compact_json(compilation))
    _replace_option(
        arguments,
        "--max-cudagraph-capture-size",
        str(MTP4_MAX_QUERY_ROWS),
    )
    return candidate


def _mtp3_view(profile: dict) -> dict:
    """Reverse only the allowed MTP4 delta for inherited-contract checks."""

    view = copy.deepcopy(profile)
    profile_id = view.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id.endswith(MTP4_SUFFIX):
        raise ContractError("fixed-MTP4 profile_id must end in -fixed-mtp4")
    view["profile_id"] = profile_id.removesuffix(MTP4_SUFFIX) + MTP3_SUFFIX
    environment = view.get("environment")
    arguments = view.get("extra_vllm_args")
    if not isinstance(environment, dict) or not isinstance(arguments, list):
        raise ContractError("fixed-MTP4 has malformed environment or arguments")
    environment.update(
        {
            "VLLM_SPARK_MAX_QUERY_ROWS": str(MTP3_MAX_QUERY_ROWS),
            "VLLM_SPARK_MTP_MODE_ID": "fixed-mtp3",
            "VLLM_SPARK_MTP_TOKENS": str(MTP3_DEPTH),
        }
    )
    spec = _load_single_json_option(
        arguments,
        "--speculative-config",
        role="speculative config",
    )
    spec["num_speculative_tokens"] = MTP3_DEPTH
    _replace_option(arguments, "--speculative-config", _compact_json(spec))
    compilation = _load_single_json_option(
        arguments,
        "--compilation-config",
        role="compilation config",
    )
    compilation["cudagraph_capture_sizes"] = list(
        range(1, MTP3_MAX_QUERY_ROWS + 1)
    )
    _replace_option(arguments, "--compilation-config", _compact_json(compilation))
    _replace_option(
        arguments,
        "--max-cudagraph-capture-size",
        str(MTP3_MAX_QUERY_ROWS),
    )
    return view


def validate_mtp4_contract(profile: dict) -> None:
    """Fail closed on depth, Q40 graph coverage, or inherited-profile drift."""

    profile_id = profile.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id.endswith(MTP4_SUFFIX):
        raise ContractError("fixed-MTP4 profile_id must end in -fixed-mtp4")
    environment = profile.get("environment")
    arguments = profile.get("extra_vllm_args")
    if not isinstance(environment, dict) or not isinstance(arguments, list):
        raise ContractError("fixed-MTP4 has malformed environment or arguments")

    if environment.get("VLLM_SPARK_MAX_QUERY_ROWS") != "40":
        raise ContractError("fixed-MTP4 requires query-row ceiling 40")
    if environment.get("VLLM_SPARK_MTP_MODE_ID") != "fixed-mtp4":
        raise ContractError("fixed-MTP4 requires mode fixed-mtp4")
    if environment.get("VLLM_SPARK_MTP_TOKENS") != "4":
        raise ContractError("fixed-MTP4 requires speculative depth 4")

    spec = _load_single_json_option(
        arguments,
        "--speculative-config",
        role="speculative config",
    )
    if spec.get("num_speculative_tokens") != MTP4_DEPTH:
        raise ContractError("fixed-MTP4 requires speculative depth 4")
    compilation = _load_single_json_option(
        arguments,
        "--compilation-config",
        role="compilation config",
    )
    if compilation.get("cudagraph_capture_sizes") != list(
        range(1, MTP4_MAX_QUERY_ROWS + 1)
    ):
        raise ContractError("fixed-MTP4 must capture Q1 through Q40")
    if _option_values(arguments, "--max-cudagraph-capture-size") != ["40"]:
        raise ContractError("fixed-MTP4 requires capture-size ceiling 40")

    # Reversing only the declared MTP4 delta must yield a fully valid MTP3
    # source. This preserves target-only K6, fp8_ds_mla, DCP/indexer choices,
    # shared-stream mounts, labels, attestation hashes, and every other field.
    validate_mtp3_source(_mtp3_view(profile))


def derive_candidate(source: dict) -> dict:
    """Return the exact fixed-depth-four derivative of qualified MTP3."""

    validate_mtp3_source(source)
    candidate = _derive_candidate_unvalidated(source)
    validate_mtp4_contract(candidate)
    return candidate


def validate_candidate(source: dict, candidate: dict) -> None:
    """Require exactly the declared MTP3-to-MTP4 semantic delta."""

    validate_mtp3_source(source)
    validate_mtp4_contract(candidate)
    if candidate != _derive_candidate_unvalidated(source):
        raise ContractError(
            "candidate differs from the exact fixed-MTP4 derivative of "
            "qualified fixed-MTP3"
        )


def derive_site_text(source: str) -> str:
    """Change only site-level static depth three to four at KV9.25."""

    required = (
        "  tensor_parallel_size: 4",
        "  decode_context_parallel_size: 4",
        '  mtp_mode: "static"',
        f"  kv_cache_bytes_per_rank: {KV925_BYTES_PER_RANK}",
        "  max_num_seqs: 8",
    )
    for line in required:
        if source.count(line) != 1:
            raise ContractError(
                f"fixed-MTP3 KV9.25 site requires exactly one {line.strip()}"
            )
    source_tokens = f"  mtp_tokens: {MTP3_DEPTH}"
    candidate_tokens = f"  mtp_tokens: {MTP4_DEPTH}"
    if source.count(source_tokens) != 1:
        raise ContractError(
            "fixed-MTP3 KV9.25 site must declare exactly mtp_tokens 3"
        )
    if candidate_tokens in source:
        raise ContractError("fixed-MTP3 KV9.25 site already declares MTP4")
    candidate = source.replace(source_tokens, candidate_tokens)
    if candidate.replace(candidate_tokens, source_tokens) != source:
        raise ContractError("MTP4 site contains drift beyond static depth")
    return candidate


def _reject_path_collisions(inputs: list[Path], outputs: list[Path]) -> None:
    input_paths = {path.resolve() for path in inputs}
    output_paths = [path.resolve() for path in outputs]
    if any(path in input_paths for path in output_paths):
        raise ContractError("MTP4 outputs must not overwrite an input")
    if len(set(output_paths)) != len(output_paths):
        raise ContractError("MTP4 output paths must be distinct")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes_like_source(value: dict, source: bytes) -> bytes:
    """Serialize canonical JSON while preserving source newline bytes."""

    source_text = source.decode("utf-8")
    crlf_count = source_text.count("\r\n")
    bare_lf_count = source_text.replace("\r\n", "").count("\n")
    if crlf_count and bare_lf_count:
        raise ContractError("fixed-MTP3 profile must not mix newline styles")
    newline = "\r\n" if crlf_count else "\n"
    if not source_text.endswith(newline):
        raise ContractError("fixed-MTP3 profile must end with one newline")
    if source_text.endswith(newline + newline):
        raise ContractError("fixed-MTP3 profile must end with one newline")
    rendered = json.dumps(value, indent=2) + "\n"
    if newline == "\r\n":
        rendered = rendered.replace("\n", "\r\n")
    return rendered.encode("utf-8")


def _expected_sha256(value: str, role: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ContractError(f"expected {role} SHA-256 must be 64 lowercase hex")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mtp3-profile", type=Path, required=True)
    parser.add_argument("--mtp3-site", type=Path, required=True)
    parser.add_argument("--expected-mtp3-profile-sha256", required=True)
    parser.add_argument("--expected-mtp3-site-sha256", required=True)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--candidate-site", type=Path, required=True)
    parser.add_argument("--rollback-profile", type=Path, required=True)
    parser.add_argument("--rollback-site", type=Path, required=True)
    args = parser.parse_args()
    inputs = [args.mtp3_profile, args.mtp3_site]
    outputs = [
        args.candidate_profile,
        args.candidate_site,
        args.rollback_profile,
        args.rollback_site,
    ]
    try:
        _reject_path_collisions(inputs, outputs)
        expected_profile_sha = _expected_sha256(
            args.expected_mtp3_profile_sha256,
            "profile",
        )
        expected_site_sha = _expected_sha256(
            args.expected_mtp3_site_sha256,
            "site",
        )
        source_profile_bytes = args.mtp3_profile.read_bytes()
        source_site_bytes = args.mtp3_site.read_bytes()
        source_profile_sha = _sha256_bytes(source_profile_bytes)
        source_site_sha = _sha256_bytes(source_site_bytes)
        if source_profile_sha != expected_profile_sha:
            raise ContractError(
                "fixed-MTP3 profile SHA-256 mismatch: "
                f"expected {expected_profile_sha}, got {source_profile_sha}"
            )
        if source_site_sha != expected_site_sha:
            raise ContractError(
                "fixed-MTP3 site SHA-256 mismatch: "
                f"expected {expected_site_sha}, got {source_site_sha}"
            )
        source_profile = json.loads(source_profile_bytes)
        source_site = source_site_bytes.decode("utf-8")
        if _json_bytes_like_source(source_profile, source_profile_bytes) != (
            source_profile_bytes
        ):
            raise ContractError(
                "fixed-MTP3 profile is not canonical indent-2 JSON; refusing "
                "to reformat bytes while deriving MTP4"
            )
        candidate_profile = derive_candidate(source_profile)
        validate_candidate(source_profile, candidate_profile)
        candidate_site = derive_site_text(source_site)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ContractError,
    ) as exc:
        parser.error(str(exc))

    candidate_profile_bytes = _json_bytes_like_source(
        candidate_profile,
        source_profile_bytes,
    )
    candidate_site_bytes = candidate_site.encode("utf-8")
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_profile.write_bytes(candidate_profile_bytes)
    args.candidate_site.write_bytes(candidate_site_bytes)
    shutil.copyfile(args.mtp3_profile, args.rollback_profile)
    shutil.copyfile(args.mtp3_site, args.rollback_site)

    if args.rollback_profile.read_bytes() != source_profile_bytes:
        parser.error("rollback profile is not byte-identical to fixed-MTP3 KV9.25")
    if args.rollback_site.read_bytes() != source_site_bytes:
        parser.error("rollback site is not byte-identical to fixed-MTP3 KV9.25")
    try:
        validate_candidate(
            source_profile,
            json.loads(args.candidate_profile.read_bytes()),
        )
    except (json.JSONDecodeError, ContractError) as exc:
        parser.error(f"written candidate validation failed: {exc}")

    print(
        json.dumps(
            {
                "candidate_profile_sha256": _sha256_bytes(
                    args.candidate_profile.read_bytes()
                ),
                "candidate_site_sha256": _sha256_bytes(
                    args.candidate_site.read_bytes()
                ),
                "rollback_profile_sha256": _sha256_bytes(
                    args.rollback_profile.read_bytes()
                ),
                "rollback_site_sha256": _sha256_bytes(
                    args.rollback_site.read_bytes()
                ),
                "source_profile_sha256": source_profile_sha,
                "source_site_sha256": source_site_sha,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
