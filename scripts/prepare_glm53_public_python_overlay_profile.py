#!/usr/bin/env python3
"""Resolve the GLM-5.3 public-base Python-overlay profile and TP4 site."""

from __future__ import annotations

import argparse
import hashlib
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
IMAGE_PLACEHOLDER = "REPLACE_WITH_PUBLIC_PYTHON_OVERLAY_SPARKCACHE_IMAGE"
CUDA_PLACEMENT_PLACEHOLDER = "REPLACE_WITH_CUDA_PLACEMENT_LIBRARY_SHA256"
LEGACY_PLACEMENT_PLACEHOLDER = "REPLACE_WITH_NATIVE_LIBRARY_SHA256"
CUDA_PLACEMENT_LABEL = "org.sparkcache.cuda-placement-library-sha256"
LEGACY_PLACEMENT_LABEL = "org.sparkcache.native-library-sha256"
PLACEHOLDERS = {
    CUDA_PLACEMENT_PLACEHOLDER: "cuda_placement_library_sha256",
    "REPLACE_WITH_VLLM_NATIVE_ELF_MANIFEST_SHA256": "native_elf_manifest_sha256",
    "REPLACE_WITH_VLLM_NATIVE_DISPATCH_MANIFEST_SHA256": (
        "native_dispatch_manifest_sha256"
    ),
    "REPLACE_WITH_SOURCE_RECEIPT_SHA256": "source_receipt_sha256",
}
PUBLIC_BASE = (
    "ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:"
    "864adfe68f458223e186a19844ac80c7adc7365e5db1f25e109b85fc19850dcd"
)
PUBLIC_BASE_ID = "sha256:7e8c0ebcb2001efb4cdab0ec9d20d53972e62db3688230044e22e61ffb1d35d5"
VLLM_NATIVE_COMMIT = "da4d7be6c97434f6942292ed8abbf4b32dc44355"
VLLM_PYTHON_COMMIT = "0b67266a0f37d6146a8403fb8482403c62f412d5"
VLLM_PYTHON_TREE = "ba9484ccb33aa56e90ff2f447f15ca9b9da97639"
B12X_COMMIT = "b1d541f9e71a35f030d45fae437630fff7507c2a"
B12X_TREE = "c69cdec1c59a08e8e0e549f930fa8abcfb5134ae"
OVERLAY_MANIFEST_SHA256 = (
    "e5e528288b173399611a4930fecc4182b7208bc1564881d52ca5d2c5c4ae0f6a"
)
DFLASH_LOADER_PATCH_SHA256 = (
    "39b567013ee7aed79f63200ed460129587933dc77fb430decdf19f78178de279"
)
DFLASH_LOADER_POSTIMAGE_SHA256 = (
    "98acbae2b3bb4482d83f9637c163ce7c92707ccdf6561b7e431f23337f151cf4"
)
RECURRENT_BOUNDARY_PATCH_SHA256 = (
    "5a6561a5bbab990dcd03bfd6a485ea26c3b5a578c2fd61b76305767b16dbfba0"
)
SPARKCACHE_COMMIT = "08e297769a796da2668ea58d0ed5c0d9b588565b"
SPARKCACHE_TREE = "18497db629a204d761f2514824a4c18408a40184"
SPARKCACHE_SOURCE_SHA256 = (
    "88633ef676b4dfe258a6fa9b788ddeb22cad68349d0cae0c503ee404d1724f7b"
)
LEASE_CONTRACT_SHA256 = (
    "45d7a92b38b836a4f829f02df85e339cfeea860e1080e4663a8340af6c125125"
)
TARGET_IDENTITY = "a35e6bf2875c1875609b8deaec404c07c6cc80259e4222fc0b51e649498bd6b9"
MTP_CACHE_IDENTITY_SHA256 = (
    "2e06d909ce5bb71c0c0e3e8be74a70e3b41d92ba4c30196cfb0957fb812acef6"
)


class ResolveError(ValueError):
    """The composed runtime identity or profile template is incomplete."""


def composed_mtp_identity() -> str:
    fields = (
        "glm53-embedded-mtp-composed-runtime-v1",
        TARGET_IDENTITY,
        VLLM_PYTHON_COMMIT,
        VLLM_NATIVE_COMMIT,
        B12X_COMMIT,
        "5",
        "adaptive:3:32",
    )
    return hashlib.sha256("\0".join(fields).encode()).hexdigest()


def _replace(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for placeholder, replacement in replacements.items():
            value = value.replace(placeholder, replacement)
        return value
    if isinstance(value, list):
        return [_replace(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace(item, replacements) for key, item in value.items()}
    return value


def _argument(profile: dict[str, Any], option: str) -> str:
    arguments = profile.get("extra_vllm_args", [])
    if arguments.count(option) != 1:
        raise ResolveError(f"profile must contain exactly one {option}")
    return str(arguments[arguments.index(option) + 1])


def _require_sha256(value: str, label: str) -> None:
    if SHA256.fullmatch(value) is None:
        raise ResolveError(f"{label} must contain 64 lowercase hexadecimal characters")


def resolve(
    profile: dict[str, Any],
    site: dict[str, Any],
    *,
    image: str,
    image_id: str,
    cuda_placement_library_sha256: str | None = None,
    native_elf_manifest_sha256: str,
    native_dispatch_manifest_sha256: str,
    source_receipt_sha256: str,
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
    profile = _replace(
        profile,
        {LEGACY_PLACEMENT_PLACEHOLDER: CUDA_PLACEMENT_PLACEHOLDER},
    )
    labels = dict(profile.get("required_image_labels", {}))
    if (
        CUDA_PLACEMENT_LABEL in labels
        and LEGACY_PLACEMENT_LABEL in labels
        and labels[CUDA_PLACEMENT_LABEL] != labels[LEGACY_PLACEMENT_LABEL]
    ):
        raise ResolveError(
            f"profile image labels {CUDA_PLACEMENT_LABEL} and compatibility "
            f"alias {LEGACY_PLACEMENT_LABEL} have conflicting values"
        )
    if LEGACY_PLACEMENT_LABEL in labels and CUDA_PLACEMENT_LABEL not in labels:
        labels[CUDA_PLACEMENT_LABEL] = labels[LEGACY_PLACEMENT_LABEL]
    labels.pop(LEGACY_PLACEMENT_LABEL, None)
    profile["required_image_labels"] = labels
    if not image or IMAGE_PLACEHOLDER in image:
        raise ResolveError("Python-overlay image reference is unresolved")
    if SHA256_ID.fullmatch(image_id) is None:
        raise ResolveError("Python-overlay image ID must be sha256 plus 64 lowercase hex")
    supplied = {
        "cuda_placement_library_sha256": cuda_placement_library_sha256,
        "native_elf_manifest_sha256": native_elf_manifest_sha256,
        "native_dispatch_manifest_sha256": native_dispatch_manifest_sha256,
        "source_receipt_sha256": source_receipt_sha256,
    }
    for name, value in supplied.items():
        _require_sha256(value, name.replace("_", " "))

    identity = profile.get("identity", {})
    expected_identity = {
        "vllm_native_revision": VLLM_NATIVE_COMMIT,
        "vllm_python_revision": VLLM_PYTHON_COMMIT,
        "vllm_python_tree": VLLM_PYTHON_TREE,
        "b12x_revision": B12X_COMMIT,
        "b12x_tree": B12X_TREE,
        "vllm_python_overlay_manifest_sha256": OVERLAY_MANIFEST_SHA256,
        "sparkcache_source_revision": SPARKCACHE_COMMIT,
        "sparkcache_source_tree": SPARKCACHE_TREE,
        "sparkcache_source_sha256": SPARKCACHE_SOURCE_SHA256,
        "mtp_cache_identity_schema": "glm53-embedded-mtp-composed-runtime-v1",
        "mtp_cache_identity_sha256": MTP_CACHE_IDENTITY_SHA256,
        "mtp_maximum_tokens": "5",
        "mtp_adaptive": "true",
        "mtp_adaptive_initial_tokens": "3",
        "mtp_adaptive_window": "32",
        "weight_loader": "fastsafetensors",
        "weight_loader_queue_size": "1",
        "weight_loader_tp_nogds": "true",
        "sparkcache_publication_schema": "tail-cow-v1",
        "sparkcache_effective_publication_schema": "page-tail-cow-v1",
    }
    for name, expected in expected_identity.items():
        if identity.get(name) != expected:
            raise ResolveError(f"profile identity {name} must be {expected}")
    if composed_mtp_identity() != MTP_CACHE_IDENTITY_SHA256:
        raise ResolveError("composed MTP identity constant is inconsistent")
    if _argument(profile, "--load-format") != "fastsafetensors":
        raise ResolveError("profile must select the fastsafetensors loader")
    speculative = json.loads(_argument(profile, "--speculative-config"))
    expected_speculative = {
        "num_speculative_tokens": 5,
        "adaptive_speculative_tokens_initial": 3,
        "adaptive_speculative_tokens_window": 32,
    }
    for name, expected in expected_speculative.items():
        if speculative.get(name) != expected:
            raise ResolveError(f"speculative configuration {name} must be {expected}")
    if profile.get("environment", {}).get("VLLM_FASTSAFETENSORS_QUEUE_SIZE") != "1":
        raise ResolveError("fastsafetensors queue size must be one")
    transfer = json.loads(_argument(profile, "--kv-transfer-config"))
    transfer_extra = transfer.get("kv_connector_extra_config", {})
    if transfer_extra.get("spark_cache_publication_schema") != "tail-cow-v1":
        raise ResolveError("opaque page-tail publication requires tail-cow-v1")
    canonical_cuda_keys = {
        "spark_cache_cuda_restore",
        "spark_cache_cuda_placement_library",
        "spark_cache_cuda_placement_library_sha256",
        "spark_cache_cuda_placement_arena_bytes",
        "spark_cache_cuda_restore_io_workers",
    }
    missing_cuda_keys = canonical_cuda_keys.difference(transfer_extra)
    if missing_cuda_keys:
        raise ResolveError(
            "profile omits canonical SparkCache CUDA keys: "
            + ", ".join(sorted(missing_cuda_keys))
        )
    if any(name.startswith("spark_cache_native_") for name in transfer_extra):
        raise ResolveError("profile may not use legacy SparkCache native keys")

    labels = profile.get("required_image_labels", {})
    fixed_labels = {
        "org.jovian.vllm.commit": VLLM_NATIVE_COMMIT,
        "org.sparkring.vllm.native.commit": VLLM_NATIVE_COMMIT,
        "org.sparkring.vllm.python.commit": VLLM_PYTHON_COMMIT,
        "org.sparkring.vllm.python.tree": VLLM_PYTHON_TREE,
        "org.sparkring.vllm.python-overlay-manifest-sha256": OVERLAY_MANIFEST_SHA256,
        "org.sparkring.vllm.dflash-draft-loader-patch-sha256": (
            DFLASH_LOADER_PATCH_SHA256
        ),
        "org.sparkring.vllm.dflash-draft-loader-postimage-sha256": (
            DFLASH_LOADER_POSTIMAGE_SHA256
        ),
        "org.sparkring.vllm.recurrent-boundary-patch-sha256": (
            RECURRENT_BOUNDARY_PATCH_SHA256
        ),
        "org.jovian.b12x.commit": B12X_COMMIT,
        "org.sparkring.b12x.tree": B12X_TREE,
        "org.opencontainers.image.base.name": PUBLIC_BASE,
        "org.sparkring.base.image-id": PUBLIC_BASE_ID,
        "org.sparkcache.source-revision": SPARKCACHE_COMMIT,
        "org.sparkcache.source-tree": SPARKCACHE_TREE,
        "org.sparkcache.source-sha256": SPARKCACHE_SOURCE_SHA256,
        "org.sparkcache.vllm-contract-sha256": LEASE_CONTRACT_SHA256,
    }
    for name, expected in fixed_labels.items():
        if labels.get(name) != expected:
            raise ResolveError(f"profile image label {name} must be {expected}")

    attestation = " ".join(str(item) for item in profile.get("attestation_hook", []))
    required_attestations = (
        OVERLAY_MANIFEST_SHA256,
        SPARKCACHE_SOURCE_SHA256,
        LEASE_CONTRACT_SHA256,
        '"tail-cow-v1"',
        "/opt/sparkring/runtime/python-overlay/sparkcache-source-tree.sha256",
        *PLACEHOLDERS,
    )
    for required in required_attestations:
        if required not in attestation:
            raise ResolveError(f"profile attestation omits {required}")
    if "source_tree_sha256(" in attestation:
        raise ResolveError(
            "profile attestation must use the clean SparkCache source receipt"
        )

    replacements = {
        placeholder: supplied[name] for placeholder, name in PLACEHOLDERS.items()
    }
    profile = _replace(profile, replacements)
    profile["image"] = image
    profile["image_id"] = image_id
    site["site"]["name"] = "glm53-flash-public-python-overlay-four-rank-cycle"
    site["site"]["description"] = (
        "Four DGX Spark systems serving vLLM 0b67266 Python over retained "
        "da4d7be native extensions with B12X b1d541f and SparkCache tail publication."
    )
    site["runtime"]["container_image"] = image
    site["runtime"]["container_image_digest"] = image_id
    site["paths"]["jit_cache_dir"] = (
        "/var/lib/sparkring/glm53-public-python-overlay/jit"
    )
    site["paths"]["context_cache_dir"] = (
        "/var/lib/sparkring/glm53-public-python-overlay/context"
    )
    site["paths"]["evidence_dir"] = (
        "./evidence/glm53-public-python-overlay"
    )
    if site["serving"]["kv_cache_bytes_per_rank"] != 20 * 1024**3:
        raise ResolveError("GLM-5.3 site must reserve 20 GiB of FP8 KV per rank")
    if "REPLACE_WITH" in json.dumps(profile):
        raise ResolveError("resolved profile still contains an artifact placeholder")
    return profile, site


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-template", type=Path, required=True)
    parser.add_argument("--site-template", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--cuda-placement-library-sha256")
    parser.add_argument("--native-library-sha256")
    parser.add_argument("--native-elf-manifest-sha256", required=True)
    parser.add_argument("--native-dispatch-manifest-sha256", required=True)
    parser.add_argument("--source-receipt-sha256", required=True)
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
            cuda_placement_library_sha256=(
                args.cuda_placement_library_sha256
            ),
            native_elf_manifest_sha256=args.native_elf_manifest_sha256,
            native_dispatch_manifest_sha256=args.native_dispatch_manifest_sha256,
            source_receipt_sha256=args.source_receipt_sha256,
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
