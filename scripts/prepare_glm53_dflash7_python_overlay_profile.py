#!/usr/bin/env python3
"""Resolve an exact GLM-5.3 DFlash7 Python-overlay profile and TP4 site."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SHA256_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
IMAGE_PLACEHOLDER = "REPLACE_WITH_DFLASH7_PYTHON_OVERLAY_SPARKCACHE_IMAGE"
PLACEHOLDERS = {
    "REPLACE_WITH_CUDA_PLACEMENT_LIBRARY_SHA256": "cuda_placement_library_sha256",
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
SPARKCACHE_COMMIT = "5d571018de5b63a9a90e5c11e6d6e86bbff4a957"
SPARKCACHE_TREE = "e864ed9ad64f771188fdb59aa9738e348134d636"
SPARKCACHE_SOURCE_SHA256 = (
    "f7c0565521fddeff7085e4cc08043cb8d1e2bde33abc67f83b8608a162d05b88"
)
LEASE_CONTRACT_SHA256 = (
    "6defde9551cbb586fd09bb2d3020495531b6573397875a767eaae1dbad126024"
)
TARGET_IDENTITY = "a35e6bf2875c1875609b8deaec404c07c6cc80259e4222fc0b51e649498bd6b9"
DFLASH_CONFIG_SHA256 = (
    "c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573"
)
DFLASH_WEIGHTS_SHA256 = (
    "b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b"
)


class ResolveError(ValueError):
    """The DFlash7 image identity or runtime profile is incomplete."""


def _argument(profile: dict[str, Any], option: str) -> str:
    arguments = profile.get("extra_vllm_args", [])
    if arguments.count(option) != 1:
        raise ResolveError(f"profile must contain exactly one {option}")
    return str(arguments[arguments.index(option) + 1])


def _replace(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace(item, replacements) for key, item in value.items()}
    return value


def _require_sha256(value: str, label: str) -> None:
    if SHA256.fullmatch(value) is None:
        raise ResolveError(f"{label} must contain 64 lowercase hexadecimal characters")


def resolve(
    profile: dict[str, Any],
    site: dict[str, Any],
    *,
    image: str,
    image_id: str,
    cuda_placement_library_sha256: str,
    native_elf_manifest_sha256: str,
    native_dispatch_manifest_sha256: str,
    source_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not image or IMAGE_PLACEHOLDER in image:
        raise ResolveError("DFlash7 Python-overlay image reference is unresolved")
    if SHA256_ID.fullmatch(image_id) is None:
        raise ResolveError("DFlash7 image ID must be sha256 plus 64 lowercase hex")
    supplied = {
        "cuda_placement_library_sha256": cuda_placement_library_sha256,
        "native_elf_manifest_sha256": native_elf_manifest_sha256,
        "native_dispatch_manifest_sha256": native_dispatch_manifest_sha256,
        "source_receipt_sha256": source_receipt_sha256,
    }
    for name, value in supplied.items():
        _require_sha256(value, name.replace("_", " "))

    identity = profile.get("identity", {})
    fixed_identity = {
        "speculator": "external_dflash",
        "draft_weights_sha256": DFLASH_WEIGHTS_SHA256,
        "draft_config_sha256": DFLASH_CONFIG_SHA256,
        "draft_speculative_tokens": "7",
        "draft_tensor_parallel_size": "4",
        "kv_cache_dtype": "fp8",
        "vllm_block_size": "256",
        "max_num_seqs": "32",
        "sparkcache_publication_schema": "tail-cow-v1",
        "sparkcache_effective_publication_schema": "page-tail-cow-v1",
        "sparkcache_source_revision": SPARKCACHE_COMMIT,
        "sparkcache_source_tree": SPARKCACHE_TREE,
        "sparkcache_source_sha256": SPARKCACHE_SOURCE_SHA256,
        "vllm_native_revision": VLLM_NATIVE_COMMIT,
        "vllm_python_revision": VLLM_PYTHON_COMMIT,
        "vllm_python_tree": VLLM_PYTHON_TREE,
        "b12x_revision": B12X_COMMIT,
        "b12x_tree": B12X_TREE,
        "vllm_python_overlay_manifest_sha256": OVERLAY_MANIFEST_SHA256,
    }
    for name, expected in fixed_identity.items():
        if identity.get(name) != expected:
            raise ResolveError(f"profile identity {name} must be {expected}")

    speculative = json.loads(_argument(profile, "--speculative-config"))
    loader = _argument(profile, "--load-format")
    if loader != identity.get("target_weight_loader"):
        raise ResolveError("target loader argument and profile identity differ")
    if loader == "fastsafetensors":
        if profile.get("environment", {}).get("VLLM_FASTSAFETENSORS_QUEUE_SIZE") != "1":
            raise ResolveError("fastsafetensors queue size must be one")
        if identity.get("dflash_peak_gpu_memory_status") != "implemented":
            raise ResolveError("separated fastsafetensors DFlash status must be implemented")
        if speculative.get("draft_load_config") != {"load_format": "safetensors"}:
            raise ResolveError(
                "fastsafetensors target requires draft_load_config safetensors"
            )
    elif loader == "safetensors":
        if identity.get("dflash_peak_gpu_memory_status") != "implemented":
            raise ResolveError("safetensors DFlash loader status must be implemented")
    else:
        raise ResolveError("DFlash7 target loader must be safetensors or fastsafetensors")

    expected_speculative = {
        "method": "dflash",
        "model": "/dflash-draft",
        "num_speculative_tokens": 7,
        "draft_tensor_parallel_size": 4,
    }
    for name, expected in expected_speculative.items():
        if speculative.get(name) != expected:
            raise ResolveError(f"DFlash configuration {name} must be {expected}")

    transfer = json.loads(_argument(profile, "--kv-transfer-config"))
    extra = transfer.get("kv_connector_extra_config", {})
    cuda_contract = {
        "spark_cache_publication_schema": "tail-cow-v1",
        "spark_cache_draft_checkpoint_sha256": DFLASH_WEIGHTS_SHA256,
        "spark_cache_cuda_restore": True,
        "spark_cache_cuda_placement_library": (
            "/opt/sparkcache-src/sparkcache/native/build-cuda/"
            "libspark_cache_placement.so"
        ),
        "spark_cache_cuda_placement_library_sha256": (
            "REPLACE_WITH_CUDA_PLACEMENT_LIBRARY_SHA256"
        ),
        "spark_cache_cuda_placement_arena_bytes": 256 * 1024**2,
        "spark_cache_cuda_restore_io_workers": 8,
        "spark_cache_load_threads": 2,
    }
    for name, expected in cuda_contract.items():
        if extra.get(name) != expected:
            raise ResolveError(f"SparkCache CUDA restore setting {name} must be {expected}")

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
        "org.jovian.b12x.commit": B12X_COMMIT,
        "org.sparkring.b12x.tree": B12X_TREE,
        "org.opencontainers.image.base.name": PUBLIC_BASE,
        "org.sparkring.base.image-id": PUBLIC_BASE_ID,
        "org.sparkcache.deployment-profile": "glm53-flash-dflash7-python-overlay",
        "org.sparkcache.source-revision": SPARKCACHE_COMMIT,
        "org.sparkcache.source-tree": SPARKCACHE_TREE,
        "org.sparkcache.source-sha256": SPARKCACHE_SOURCE_SHA256,
        "org.sparkcache.vllm-contract-sha256": LEASE_CONTRACT_SHA256,
    }
    for name, expected in fixed_labels.items():
        if labels.get(name) != expected:
            raise ResolveError(f"profile image label {name} must be {expected}")

    attestation = " ".join(str(item) for item in profile.get("attestation_hook", []))
    for required in (
        DFLASH_CONFIG_SHA256,
        DFLASH_WEIGHTS_SHA256,
        SPARKCACHE_SOURCE_SHA256,
        LEASE_CONTRACT_SHA256,
        "/opt/sparkring/runtime/python-overlay/sparkcache-source-tree.sha256",
        *PLACEHOLDERS,
    ):
        if required not in attestation:
            raise ResolveError(f"profile attestation omits {required}")
    if "adaptive" in " ".join(str(value) for value in labels.values()).lower():
        raise ResolveError("DFlash7 image labels may not claim adaptive MTP")

    replacements = {
        placeholder: supplied[name] for placeholder, name in PLACEHOLDERS.items()
    }
    profile = _replace(profile, replacements)
    profile["image"] = image
    profile["image_id"] = image_id
    site["site"]["name"] = "glm53-flash-dflash7-python-overlay-four-rank-cycle"
    site["site"]["description"] = (
        "Four DGX Spark systems serving external DFlash7 through the exact "
        "vLLM Python-overlay and SparkCache CUDA restore contracts."
    )
    site["runtime"]["container_image"] = image
    site["runtime"]["container_image_digest"] = image_id
    site["paths"]["jit_cache_dir"] = "/var/lib/sparkring/glm53-dflash7-python-overlay/jit"
    site["paths"]["context_cache_dir"] = (
        "/var/lib/sparkring/glm53-dflash7-python-overlay/context"
    )
    site["paths"]["evidence_dir"] = "./evidence/glm53-dflash7-python-overlay"
    serving = site["serving"]
    if serving["tensor_parallel_size"] != 4:
        raise ResolveError("DFlash7 site tensor parallel size must be four")
    if serving["decode_context_parallel_size"] != 1:
        raise ResolveError("DFlash7 site DCP size must be one")
    if serving["max_num_seqs"] != 32:
        raise ResolveError("DFlash7 site max_num_seqs must be 32")
    if "REPLACE_WITH" in json.dumps(profile):
        raise ResolveError("resolved DFlash7 profile still contains a placeholder")
    return profile, site


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-template", type=Path, required=True)
    parser.add_argument("--site-template", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--cuda-placement-library-sha256", required=True)
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
            cuda_placement_library_sha256=args.cuda_placement_library_sha256,
            native_elf_manifest_sha256=args.native_elf_manifest_sha256,
            native_dispatch_manifest_sha256=args.native_dispatch_manifest_sha256,
            source_receipt_sha256=args.source_receipt_sha256,
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
