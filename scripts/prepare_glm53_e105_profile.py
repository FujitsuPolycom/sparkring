#!/usr/bin/env python3
"""Resolve one source-built GLM-5.3 e10536a profile and matching site."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SHA256_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
NATIVE_PLACEHOLDER = "REPLACE_WITH_NATIVE_LIBRARY_SHA256"
IMAGE_PLACEHOLDER = "REPLACE_WITH_E10536A_SPARKCACHE_IMAGE"
PARENT_PLACEHOLDER = "REPLACE_WITH_E10536A_RUNTIME_IMAGE"


class ResolveError(ValueError):
    """A source-built image identity or profile template is incomplete."""


def _replace_native(value: Any, digest: str) -> Any:
    if isinstance(value, str):
        return value.replace(NATIVE_PLACEHOLDER, digest)
    if isinstance(value, list):
        return [_replace_native(item, digest) for item in value]
    if isinstance(value, dict):
        return {key: _replace_native(item, digest) for key, item in value.items()}
    return value


def resolve(
    profile: dict[str, Any],
    site: dict[str, Any],
    *,
    image: str,
    image_id: str,
    parent_image: str,
    parent_image_id: str,
    native_library_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not image or IMAGE_PLACEHOLDER in image:
        raise ResolveError("SparkCache image reference is unresolved")
    if not parent_image or PARENT_PLACEHOLDER in parent_image:
        raise ResolveError("runtime parent image reference is unresolved")
    if SHA256_ID.fullmatch(image_id) is None:
        raise ResolveError("SparkCache image ID must be sha256 plus 64 lowercase hex")
    if SHA256_ID.fullmatch(parent_image_id) is None:
        raise ResolveError("runtime parent image ID must be sha256 plus 64 lowercase hex")
    if SHA256.fullmatch(native_library_sha256) is None:
        raise ResolveError("native library SHA-256 must be 64 lowercase hex")

    profile = _replace_native(profile, native_library_sha256)
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
    parser.add_argument("--native-library-sha256", required=True)
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
