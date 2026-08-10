#!/usr/bin/env python3
"""Build an offline EXL3/DCP4 SparkCache candidate profile.

This module performs no SSH, container, or service operations.  It derives an
experimental profile from the receipt-gated public EXL3+LMCache profile so the
model, image, TP4/DCP4, MTP2, graph, and memory contracts remain unchanged.
Only the cache composition changes: LMCache is not injected, SparkCache's
KV-Connector-V1 configuration is added, and the connector staging import path
is declared.

The output is a *candidate input* for a confirmation-gated launcher.  The
ordinary EXL3 launcher does not add the two required bind mounts, so this tool
also emits their exact host/container contract and deliberately marks direct
execution unsupported.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import unicodedata
from pathlib import Path, PurePosixPath

import checkpoint_manifest_generator as checkpoint_manifest


SCHEMA = "sparkring-exl3-sparkcache-candidate/v1"
PROFILE_ID = "glm52-exl3-tr3-3.25bpw-sparkcache-v51"
CONTAINER_NAME = "glm52-sparkring-exl3-sparkcache-v51"
STAGING_DESTINATION = "/opt/sparkcache-staging"
CACHE_DESTINATION = "/cache/context"
HEX64 = re.compile(r"[0-9a-f]{64}")


class SparkCacheProfileError(ValueError):
    """Raised when the source profile cannot be transformed safely."""


def _absolute_posix(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise SparkCacheProfileError(f"{label} must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if not path.is_absolute() or value == "/" or any(part in (".", "..") for part in path.parts):
        raise SparkCacheProfileError(
            f"{label} must be an absolute non-root POSIX path without dot segments"
        )
    return value


def _option_positions(arguments: list[str], option: str) -> list[int]:
    return [index for index, value in enumerate(arguments) if value == option]


def validate_checkpoint_receipt(receipt: dict) -> str:
    """Validate a canonical v2 full-inventory receipt and return its identity.

    Identity semantics are exactly ``checkpoint_manifest_generator`` v2:
    SHA-256 over the domain ``sparkcache-checkpoint-manifest-v2``, a NUL byte,
    and canonical JSON of the complete normalized file inventory, excluding
    only the display root name and the identity field itself.
    """
    if not isinstance(receipt, dict):
        raise SparkCacheProfileError("checkpoint receipt must be an object")
    required = {
        "manifest_version",
        "path_normalization",
        "artifact_root_name",
        "file_count",
        "files",
        "checkpoint_identity_sha256",
    }
    if set(receipt) != required:
        raise SparkCacheProfileError("checkpoint receipt keys are not canonical v2")
    if receipt["manifest_version"] != checkpoint_manifest.MANIFEST_VERSION:
        raise SparkCacheProfileError("checkpoint receipt manifest version is wrong")
    if receipt["path_normalization"] != "POSIX separators + Unicode NFC":
        raise SparkCacheProfileError("checkpoint receipt path normalization is wrong")
    if not isinstance(receipt["artifact_root_name"], str):
        raise SparkCacheProfileError("checkpoint receipt display root name is invalid")
    files = receipt["files"]
    if (
        not isinstance(files, list)
        or not files
        or isinstance(receipt["file_count"], bool)
        or not isinstance(receipt["file_count"], int)
        or receipt["file_count"] != len(files)
    ):
        raise SparkCacheProfileError("checkpoint receipt inventory is empty/incomplete")
    previous = None
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {
            "rel_path",
            "byte_size",
            "content_sha256",
        }:
            raise SparkCacheProfileError("checkpoint receipt file entry is malformed")
        rel = entry["rel_path"]
        if (
            not isinstance(rel, str)
            or not rel
            or unicodedata.normalize("NFC", rel) != rel
            or PurePosixPath(rel).as_posix() != rel
            or PurePosixPath(rel).is_absolute()
            or any(part in (".", "..") for part in PurePosixPath(rel).parts)
            or previous is not None
            and rel <= previous
        ):
            raise SparkCacheProfileError("checkpoint receipt paths are not strictly canonical")
        if (
            isinstance(entry["byte_size"], bool)
            or not isinstance(entry["byte_size"], int)
            or entry["byte_size"] < 0
        ):
            raise SparkCacheProfileError("checkpoint receipt byte size is invalid")
        if HEX64.fullmatch(entry["content_sha256"]) is None:
            raise SparkCacheProfileError("checkpoint receipt file hash is invalid")
        previous = rel
    identity = receipt["checkpoint_identity_sha256"]
    if HEX64.fullmatch(identity) is None:
        raise SparkCacheProfileError("checkpoint receipt identity is invalid")
    if checkpoint_manifest.compute_identity(receipt) != identity:
        raise SparkCacheProfileError("checkpoint receipt identity does not match inventory")
    return identity


def _require_source_contract(document: dict) -> None:
    required = {
        "schema",
        "profile_id",
        "container_name",
        "model_repository",
        "model_revision",
        "image_id",
        "environment",
        "extra_vllm_args",
    }
    missing = sorted(required - set(document))
    if missing:
        raise SparkCacheProfileError(f"source profile is missing {missing}")
    if document["profile_id"] != "glm52-exl3-tr3-3.25bpw-lmcache-cs512":
        raise SparkCacheProfileError(
            "source must be the canonical EXL3 3.25-bpw LMCache CS512 profile"
        )
    if document["model_repository"] != "willfalco/GLM-5.2-EXL3-TR3-3.25bpw":
        raise SparkCacheProfileError("source uses the wrong EXL3 model")
    environment = document["environment"]
    expected = {
        "VLLM_SPARK_DCP_SIZE": "4",
        "VLLM_SPARK_MTP_MODE_ID": "fixed-mtp2",
        "VLLM_SPARK_MTP_TOKENS": "2",
        "VLLM_SPARK_KV_CACHE_DTYPE": "nvfp4_ds_mla",
        "VLLM_SPARK_MAX_MODEL_LEN": "524288",
    }
    drift = {
        key: {"expected": value, "observed": environment.get(key)}
        for key, value in expected.items()
        if environment.get(key) != value
    }
    if drift:
        raise SparkCacheProfileError(f"source EXL3 contract drift: {drift}")
    arguments = document["extra_vllm_args"]
    for option in ("--kv-transfer-config", "--disable-hybrid-kv-cache-manager"):
        if _option_positions(arguments, option):
            raise SparkCacheProfileError(f"source already contains {option}")


def build_candidate(
    source: dict,
    *,
    checkpoint_receipt: dict,
    connector_bundle_identity: str,
    connector_staging_host: str,
    cache_root_host: str,
) -> dict:
    """Return a fail-closed EXL3/DCP4 SparkCache candidate document."""
    _require_source_contract(source)
    target_checkpoint = validate_checkpoint_receipt(checkpoint_receipt)
    if HEX64.fullmatch(connector_bundle_identity) is None:
        raise SparkCacheProfileError(
            "connector_bundle_identity must be a 64-character lowercase SHA-256"
        )
    connector_staging_host = _absolute_posix(
        connector_staging_host, "connector_staging_host"
    )
    cache_root_host = _absolute_posix(cache_root_host, "cache_root_host")

    profile = copy.deepcopy(source)
    profile["profile_id"] = PROFILE_ID
    profile["container_name"] = CONTAINER_NAME
    environment = profile["environment"]
    # The legacy variable has no connector authority.  None uses the public
    # launcher's explicit-unset contract and prevents a misleading value.
    environment["SPARK_CONTEXT_CACHE_ENABLE"] = None
    environment["SPARKRING_EXL3_CONTEXT_PROFILE"] = (
        "native-512k-kv4.5gb-sparkcache-v51"
    )
    environment["PYTHONPATH"] = (
        f"{STAGING_DESTINATION}:{STAGING_DESTINATION}/sparkcache:/opt/spark-vllm"
    )
    # Match the canonical EXL3+LMCache engine envelope exactly.  The canonical
    # composition overrides the raw profile's expandable-segments setting.
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

    extra = {
        "spark_cache_root": CACHE_DESTINATION,
        "spark_cache_target_checkpoint_sha256": target_checkpoint,
        # EXL3 MTP is colocated in the target model/cache pool.  Supplying a
        # second draft digest would misdescribe this serving configuration.
        "spark_cache_draft_policy": "colocated_target",
        "spark_cache_store": True,
        "spark_cache_restore": True,
        # First gate uses the simpler end-of-prefill snapshot path.  Streaming
        # and native direct placement are promoted only after equivalence.
        "spark_cache_streaming_snapshots": False,
        "spark_cache_native_restore": False,
        "spark_cache_min_span_tokens": 1024,
        "spark_cache_max_span_tokens": 524288,
    }
    kv_transfer = {
        "kv_connector": "SparkContextCacheConnector",
        "kv_connector_module_path": "spark_context_cache_connector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": extra,
    }
    profile["extra_vllm_args"].extend(
        [
            "--kv-transfer-config",
            json.dumps(kv_transfer, separators=(",", ":"), sort_keys=True),
            "--disable-hybrid-kv-cache-manager",
        ]
    )

    return {
        "schema": SCHEMA,
        "lane": "public-functional",
        "maturity": "offline-validated",
        "configuration_status": "candidate",
        "hardware": "four directly cabled NVIDIA DGX Sparks / GB10",
        "execution_supported": False,
        "source_profile": {
            "profile_id": source["profile_id"],
            "image_id": source["image_id"],
            "model_revision": source["model_revision"],
        },
        "checkpoint_receipt": copy.deepcopy(checkpoint_receipt),
        "connector_bundle_identity_sha256": connector_bundle_identity,
        "required_mounts": [
            {
                "source": connector_staging_host,
                "destination": STAGING_DESTINATION,
                "read_only": True,
            },
            {
                "source": cache_root_host,
                "destination": CACHE_DESTINATION,
                "read_only": False,
            },
        ],
        "profile": profile,
        "unchanged_contract": {
            "topology": "TP4/DCP4",
            "mtp": "fixed MTP2",
            "max_model_len": 524288,
            "kv_cache_memory_bytes_per_rank": 4500000000,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 4096,
        },
        "promotion_order": [
            "end-of-prefill snapshot plus Python restore equivalence",
            "corruption withdrawal and clean recompute",
            "engine/full-stack restart durability and rollback",
            "matched repeated C8 interference gate",
            "native restore, then streaming snapshots as separate candidates",
        ],
        "blockers_before_execution": [
            "use the confirmation-gated SparkCache launcher; direct profile execution is unsupported",
            "never hash the EXL3 checkpoint while a model engine is live; cutover must stop all owned engines and pass the all-rank quiescence barrier first",
            "connector bundle identity must be re-hashed on every rank",
            "deployed image must attest both public SparkCache vLLM patches",
            "baseline EXL3+LMCache rollback actions must be generated and inspected",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--connector-bundle-identity", required=True)
    parser.add_argument("--connector-staging-host", required=True)
    parser.add_argument("--cache-root-host", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        source = json.loads(args.profile.read_text(encoding="utf-8"))
        checkpoint_receipt = json.loads(
            args.checkpoint_receipt.read_text(encoding="utf-8")
        )
        candidate = build_candidate(
            source,
            checkpoint_receipt=checkpoint_receipt,
            connector_bundle_identity=args.connector_bundle_identity,
            connector_staging_host=args.connector_staging_host,
            cache_root_host=args.cache_root_host,
        )
    except (OSError, json.JSONDecodeError, SparkCacheProfileError) as error:
        parser.error(str(error))
    rendered = json.dumps(candidate, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        try:
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(rendered)
        except OSError as error:
            parser.error(f"cannot create exclusive output {args.output}: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
