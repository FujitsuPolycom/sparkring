#!/usr/bin/env python3
"""Resolve one source-built GLM-5.3 e10536a profile and matching site."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from sparkcache_terminology import (
    SparkCacheTerminologyError,
    canonicalize_profile_connector_arguments,
    resolve_string_alias,
)


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SHA256_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
CUDA_PLACEMENT_PLACEHOLDER = "REPLACE_WITH_CUDA_PLACEMENT_LIBRARY_SHA256"
LEGACY_PLACEMENT_PLACEHOLDER = "REPLACE_WITH_NATIVE_LIBRARY_SHA256"
IMAGE_PLACEHOLDER = "REPLACE_WITH_E10536A_SPARKCACHE_IMAGE"
PARENT_PLACEHOLDER = "REPLACE_WITH_E10536A_RUNTIME_IMAGE"
SPARKCACHE_COMMIT = "eb3690c1aac2b9e86be8d513799dbb64afa53f25"
SPARKCACHE_SOURCE_SHA256 = (
    "34108fb22ba95b457bf4b357407b176dcbf3a6db6227227b21ecee045502a16f"
)
LEASE_CONTRACT_SHA256 = (
    "70cd4e923d049da96bcfa4a5b460e2ff5f7460881d5cfd0621607080fd70f68f"
)


class ResolveError(ValueError):
    """A source-built image identity or profile template is incomplete."""


def _replace_cuda_placement(value: Any, digest: str) -> Any:
    if isinstance(value, str):
        return value.replace(CUDA_PLACEMENT_PLACEHOLDER, digest).replace(
            LEGACY_PLACEMENT_PLACEHOLDER, digest
        )
    if isinstance(value, list):
        return [_replace_cuda_placement(item, digest) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_cuda_placement(item, digest)
            for key, item in value.items()
        }
    return value


def resolve(
    profile: dict[str, Any],
    site: dict[str, Any],
    *,
    image: str,
    image_id: str,
    parent_image: str,
    parent_image_id: str,
    cuda_placement_library_sha256: str | None = None,
    native_library_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        profile = canonicalize_profile_connector_arguments(profile)
        cuda_placement_library_sha256 = resolve_string_alias(
            cuda_placement_library_sha256,
            native_library_sha256,
            canonical_name="cuda_placement_library_sha256",
            legacy_name="native_library_sha256",
        )
    except SparkCacheTerminologyError as error:
        raise ResolveError(str(error)) from error
    if not image or IMAGE_PLACEHOLDER in image:
        raise ResolveError("SparkCache image reference is unresolved")
    if not parent_image or PARENT_PLACEHOLDER in parent_image:
        raise ResolveError("runtime parent image reference is unresolved")
    if SHA256_ID.fullmatch(image_id) is None:
        raise ResolveError("SparkCache image ID must be sha256 plus 64 lowercase hex")
    if SHA256_ID.fullmatch(parent_image_id) is None:
        raise ResolveError("runtime parent image ID must be sha256 plus 64 lowercase hex")
    if SHA256.fullmatch(cuda_placement_library_sha256) is None:
        raise ResolveError(
            "SparkCache CUDA placement library SHA-256 must be 64 lowercase hex"
        )
    identity = profile.get("identity", {})
    if identity.get("sparkcache_source_revision") != SPARKCACHE_COMMIT:
        raise ResolveError("profile does not name the integrated SparkCache commit")
    if identity.get("sparkcache_source_sha256") != SPARKCACHE_SOURCE_SHA256:
        raise ResolveError("profile does not name the integrated SparkCache source")
    attestation = " ".join(str(value) for value in profile.get("attestation_hook", []))
    if SPARKCACHE_SOURCE_SHA256 not in attestation:
        raise ResolveError("profile does not attest the integrated SparkCache source")
    if LEASE_CONTRACT_SHA256 not in attestation:
        raise ResolveError("profile does not attest the e10536a lease contract")

    profile = _replace_cuda_placement(profile, cuda_placement_library_sha256)
    profile["image"] = image
    profile["image_id"] = image_id
    labels = profile["required_image_labels"]
    labels["org.opencontainers.image.base.name"] = parent_image
    labels["org.sparkcache.parent-image-id"] = parent_image_id

    runtime = site["runtime"]
    runtime["container_image"] = image
    runtime["container_image_digest"] = image_id
    if site["serving"]["kv_cache_bytes_per_rank"] != 20 * 1024**3:
        raise ResolveError("e10536a site must reserve 20 GiB of FP8 KV per rank")
    return profile, site


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-template", type=Path, required=True)
    parser.add_argument("--site-template", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--parent-image", required=True)
    parser.add_argument("--parent-image-id", required=True)
    parser.add_argument("--cuda-placement-library-sha256")
    parser.add_argument("--native-library-sha256")
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument("--site-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        profile = json.loads(args.profile_template.read_text(encoding="utf-8"))
        site = yaml.safe_load(args.site_template.read_text(encoding="utf-8"))
        profile, site = resolve(
            profile,
            site,
            image=args.image,
            image_id=args.image_id,
            parent_image=args.parent_image,
            parent_image_id=args.parent_image_id,
            cuda_placement_library_sha256=(
                args.cuda_placement_library_sha256
            ),
            native_library_sha256=args.native_library_sha256,
        )
    except (OSError, KeyError, json.JSONDecodeError, ResolveError) as exc:
        parser.error(str(exc))
    args.profile_output.write_text(
        json.dumps(profile, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    args.site_output.write_text(
        yaml.safe_dump(site, sort_keys=False), encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
