#!/usr/bin/env python3
"""Resolve the GLM-5.3 live-tensor B12X KDA SparkCache profile and site."""

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
IMAGE_PLACEHOLDER = "REPLACE_WITH_B12X_KDA_ADAPTIVE_MTP_SPARKCACHE_IMAGE"
PARENT_PLACEHOLDER = "REPLACE_WITH_B12X_KDA_ADAPTIVE_MTP_RUNTIME_IMAGE"
SPARKCACHE_COMMIT = "20838ace3ebda570ca039cb7f1976c29da554b39"
SPARKCACHE_SOURCE_SHA256 = (
    "4998b24f4f504aeeb9bf92769ec720e282f546e6726d89fdfd06c4efa8d17c10"
)
LEASE_CONTRACT_SHA256 = (
    "6defde9551cbb586fd09bb2d3020495531b6573397875a767eaae1dbad126024"
)
VLLM_COMMIT = "0b67266a0f37d6146a8403fb8482403c62f412d5"
B12X_COMMIT = "b1d541f9e71a35f030d45fae437630fff7507c2a"
B12X_TREE = "c69cdec1c59a08e8e0e549f930fa8abcfb5134ae"
B12X_KDA_CONTRACT = "sparkring-vllm-b12x-live-tensor-kda/v1"
MTP_CACHE_IDENTITY_SHA256 = (
    "2761488b43742f849e2cb7ac4977dc28d771faf5afef1951044327948d6ad71e"
)


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


def _argument(profile: dict[str, Any], option: str) -> str:
    arguments = profile.get("extra_vllm_args", [])
    if arguments.count(option) != 1:
        raise ResolveError(f"profile must contain exactly one {option}")
    return str(arguments[arguments.index(option) + 1])


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
    identity = profile.get("identity", {})
    if identity.get("sparkcache_source_revision") != SPARKCACHE_COMMIT:
        raise ResolveError("profile does not name the integrated SparkCache commit")
    if identity.get("sparkcache_source_sha256") != SPARKCACHE_SOURCE_SHA256:
        raise ResolveError("profile does not name the integrated SparkCache source")
    if identity.get("vllm_revision") != VLLM_COMMIT:
        raise ResolveError("profile does not name the live-tensor B12X KDA runtime")
    if identity.get("b12x_revision") != B12X_COMMIT:
        raise ResolveError("profile does not name the compatible B12X revision")
    if identity.get("b12x_tree") != B12X_TREE:
        raise ResolveError("profile does not name the compatible B12X source tree")
    if identity.get("mtp_cache_identity_sha256") != MTP_CACHE_IDENTITY_SHA256:
        raise ResolveError("profile does not use the runtime-bound MTP identity")
    expected_identity = {
        "mtp_maximum_tokens": "5",
        "mtp_adaptive": "true",
        "mtp_adaptive_initial_tokens": "3",
        "mtp_adaptive_window": "32",
        "mtp_cache_identity_schema": "glm53-embedded-mtp-vllm-b12x-runtime-v1",
        "weight_loader": "fastsafetensors",
        "weight_loader_queue_size": "1",
        "weight_loader_tp_nogds": "true",
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise ResolveError(f"profile identity {key} must be {expected}")
    if _argument(profile, "--load-format") != "fastsafetensors":
        raise ResolveError("profile must select the fastsafetensors loader")
    speculative = json.loads(_argument(profile, "--speculative-config"))
    if speculative.get("num_speculative_tokens") != 5:
        raise ResolveError("adaptive MTP maximum depth must be five")
    if speculative.get("adaptive_speculative_tokens_initial") != 3:
        raise ResolveError("adaptive MTP initial depth must be three")
    if speculative.get("adaptive_speculative_tokens_window") != 32:
        raise ResolveError("adaptive MTP observation window must be 32")
    if profile.get("environment", {}).get("VLLM_FASTSAFETENSORS_QUEUE_SIZE") != "1":
        raise ResolveError("fastsafetensors queue size must be one")
    required_labels = profile.get("required_image_labels", {})
    expected_b12x_labels = {
        "org.jovian.b12x.commit": B12X_COMMIT,
        "org.jovian.b12x.tree": B12X_TREE,
        "org.jovian.b12x.kda-contract": B12X_KDA_CONTRACT,
    }
    for name, expected in expected_b12x_labels.items():
        if required_labels.get(name) != expected:
            raise ResolveError(f"profile image label {name} must be {expected}")
    attestation = " ".join(str(value) for value in profile.get("attestation_hook", []))
    if SPARKCACHE_SOURCE_SHA256 not in attestation:
        raise ResolveError("profile does not attest the integrated SparkCache source")
    if LEASE_CONTRACT_SHA256 not in attestation:
        raise ResolveError("profile does not attest the live-tensor KDA lease contract")

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
        raise ResolveError("GLM-5.3 site must reserve 20 GiB of FP8 KV per rank")
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
