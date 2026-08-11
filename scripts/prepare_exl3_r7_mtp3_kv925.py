#!/usr/bin/env python3
"""Prepare the one-variable 9.25-GB KV-cache arm for fixed-MTP3."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import prepare_exl3_r7_mtp3 as mtp3


class ContractError(ValueError):
    """The KV-cache arm is not an exact derivative of qualified fixed-MTP3."""


QUALIFIED_KV_BYTES = 9_000_000_000
CANDIDATE_KV_BYTES = 9_250_000_000
KV_POOL_PAGE_BYTES = 3_502_592
DCP_GLOBAL_TOKENS_PER_BLOCK = 256
EXPECTED_CAPACITY_TOKENS = (
    CANDIDATE_KV_BYTES // KV_POOL_PAGE_BYTES * DCP_GLOBAL_TOKENS_PER_BLOCK
)


def derive_candidate_site(qualified_site: str) -> str:
    """Change only the explicit per-rank KV-cache reservation."""

    qualified_line = f"  kv_cache_bytes_per_rank: {QUALIFIED_KV_BYTES}"
    candidate_line = f"  kv_cache_bytes_per_rank: {CANDIDATE_KV_BYTES}"
    if qualified_site.count(qualified_line) != 1:
        raise ContractError(
            "qualified fixed-MTP3 site must declare exactly "
            f"kv_cache_bytes_per_rank={QUALIFIED_KV_BYTES}"
        )
    if candidate_line in qualified_site:
        raise ContractError("qualified fixed-MTP3 site already declares the candidate value")
    candidate_site = qualified_site.replace(qualified_line, candidate_line)
    if candidate_site == qualified_site:
        raise ContractError("candidate site did not change")
    if candidate_site.replace(candidate_line, qualified_line) != qualified_site:
        raise ContractError("candidate site contains semantic drift beyond KV-cache bytes")
    return candidate_site


def validate_profile(profile: dict) -> None:
    """Require the launch profile to retain the qualified fixed-MTP3 contract."""

    try:
        mtp3.validate_mtp3_contract(profile)
    except (KeyError, TypeError, mtp3.ContractError) as exc:
        raise ContractError(f"fixed-MTP3 launch profile is not qualified: {exc}") from exc


def _reject_path_collisions(inputs: list[Path], outputs: list[Path]) -> None:
    input_paths = {path.resolve() for path in inputs}
    output_paths = [path.resolve() for path in outputs]
    if any(path in input_paths for path in output_paths):
        raise ContractError("KV-cache arm outputs must not overwrite an input")
    if len(set(output_paths)) != len(output_paths):
        raise ContractError("KV-cache arm output paths must be distinct")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualified-profile", type=Path, required=True)
    parser.add_argument("--qualified-site", type=Path, required=True)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--candidate-site", type=Path, required=True)
    parser.add_argument("--rollback-profile", type=Path, required=True)
    parser.add_argument("--rollback-site", type=Path, required=True)
    args = parser.parse_args()
    inputs = [args.qualified_profile, args.qualified_site]
    outputs = [
        args.candidate_profile,
        args.candidate_site,
        args.rollback_profile,
        args.rollback_site,
    ]
    try:
        _reject_path_collisions(inputs, outputs)
        qualified_profile_bytes = args.qualified_profile.read_bytes()
        qualified_profile = json.loads(qualified_profile_bytes)
        validate_profile(qualified_profile)
        qualified_site_bytes = args.qualified_site.read_bytes()
        qualified_site = qualified_site_bytes.decode("utf-8")
        candidate_site = derive_candidate_site(qualified_site)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ContractError) as exc:
        parser.error(str(exc))

    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.qualified_profile, args.candidate_profile)
    # Write bytes so the candidate preserves the qualified site's line endings.
    args.candidate_site.write_bytes(candidate_site.encode("utf-8"))
    shutil.copyfile(args.qualified_profile, args.rollback_profile)
    shutil.copyfile(args.qualified_site, args.rollback_site)

    if args.candidate_profile.read_bytes() != qualified_profile_bytes:
        parser.error("candidate launch profile is not byte-identical to fixed-MTP3")
    if args.rollback_profile.read_bytes() != qualified_profile_bytes:
        parser.error("rollback launch profile is not byte-identical to fixed-MTP3")
    if args.rollback_site.read_bytes() != qualified_site_bytes:
        parser.error("rollback site is not byte-identical to qualified 9-GB site")

    print(
        json.dumps(
            {
                "candidate_kv_cache_bytes_per_rank": CANDIDATE_KV_BYTES,
                "predicted_dcp_global_token_capacity": EXPECTED_CAPACITY_TOKENS,
                "candidate_profile_sha256": _sha256(args.candidate_profile),
                "candidate_site_sha256": _sha256(args.candidate_site),
                "rollback_profile_sha256": _sha256(args.rollback_profile),
                "rollback_site_sha256": _sha256(args.rollback_site),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
