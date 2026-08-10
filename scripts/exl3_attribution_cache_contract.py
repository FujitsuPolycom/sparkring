#!/usr/bin/env python3
"""Public-safe cache namespace contract for EXL3 attribution requests."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


ARM_MTP_TOKENS = {
    "a-mtp0-apc0-lmcache0": 0,
    "b-mtp2-apc0-lmcache0": 2,
    "c-mtp2-apc1-lmcache0": 2,
    "d-mtp2-apc1-lmcache1": 2,
    "e-mtp0-apc0-lmcache1": 0,
    "f-mtp2-apc0-lmcache1": 2,
}
LAYOUT_SCHEMA = "sparkring-exl3-lmcache-layout/v1"
SALT_PREFIX = "sr-exl3-layout-v1-"
LIVE_ARM_RECEIPT_SCHEMA = "sparkring-exl3-live-arm-receipt/v2"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}")
HEX40_RE = re.compile(r"[0-9a-f]{40}")
CONTAINER_ID_RE = re.compile(r"[0-9a-f]{64}")
DOCKER_STARTED_AT_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z"
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def expected_layout(arm_id: str) -> dict[str, Any]:
    try:
        mtp_tokens = ARM_MTP_TOKENS[arm_id]
    except KeyError as exc:
        raise ValueError(f"unknown EXL3 attribution arm {arm_id!r}") from exc
    return {
        "schema": LAYOUT_SCHEMA,
        "mtp_tokens": mtp_tokens,
        "dcp_size": 4,
        "kv_cache_memory_bytes_per_rank": 4_500_000_000,
        "max_model_len": 524_288,
        "lmcache_chunk_tokens_global": 512,
    }


def cache_salt_for_layout(layout: dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(layout).encode("ascii")).hexdigest()
    # 51 public-safe characters, below LMCache's 128-character hard limit.
    return f"{SALT_PREFIX}{digest[:32]}"


def cache_salt_for_arm(arm_id: str) -> str:
    return cache_salt_for_layout(expected_layout(arm_id))


def _duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicates,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, raw


def build_live_arm_receipt(
    *,
    arm_id: str,
    canonical_profile_id: str,
    canonical_profile_file_sha256: str,
    image_id: str,
    model_repository: str,
    model_revision: str,
    canonical_container_name: str,
    explicit_environment_sha256: list[str],
    config_cmd_sha256: list[str],
    observed_runtime_instances: list[dict[str, str]],
) -> dict[str, Any]:
    if (
        len(explicit_environment_sha256) != 4
        or len(config_cmd_sha256) != 4
        or len(observed_runtime_instances) != 4
    ):
        raise ValueError("live-arm receipt requires four rank digests")
    if any(
        not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None
        for digest in explicit_environment_sha256 + config_cmd_sha256
    ):
        raise ValueError("live-arm receipt rank digests must be SHA-256")
    for rank, instance in enumerate(observed_runtime_instances):
        if not isinstance(instance, dict) or set(instance) != {
            "container_id",
            "started_at",
        }:
            raise ValueError(
                f"live-arm receipt rank {rank} runtime identity fields are unsupported"
            )
        if (
            not isinstance(instance["container_id"], str)
            or CONTAINER_ID_RE.fullmatch(instance["container_id"]) is None
        ):
            raise ValueError(f"live-arm receipt rank {rank} container ID is invalid")
        if (
            not isinstance(instance["started_at"], str)
            or DOCKER_STARTED_AT_RE.fullmatch(instance["started_at"]) is None
        ):
            raise ValueError(f"live-arm receipt rank {rank} StartedAt is invalid")
    layout = expected_layout(arm_id)
    diagnostic_profile_id = f"{canonical_profile_id}-diag-{arm_id}"
    diagnostic_container_name = f"{canonical_container_name}-diag-{arm_id}"
    return {
        "schema": LIVE_ARM_RECEIPT_SCHEMA,
        "status": "live-arm-attested",
        "arm": arm_id,
        "canonical_profile_id": canonical_profile_id,
        "diagnostic_profile_id": diagnostic_profile_id,
        "canonical_profile_file_sha256": canonical_profile_file_sha256,
        "image_id": image_id,
        "model_repository": model_repository,
        "model_revision": model_revision,
        "layout": layout,
        "cache_salt": cache_salt_for_layout(layout),
        "ranks": [
            {
                "rank": rank,
                "status": "attested",
                "container_name": f"{diagnostic_container_name}-r{rank}",
                "labels": {
                    "org.sparkring.managed": "true",
                    "org.sparkring.exl3-profile": diagnostic_profile_id,
                    "org.sparkring.component": "engine",
                    "org.sparkring.exl3-attribution": arm_id,
                },
                "image_id": image_id,
                "container_id": observed_runtime_instances[rank]["container_id"],
                "started_at": observed_runtime_instances[rank]["started_at"],
                "explicit_environment_sha256": explicit_environment_sha256[
                    rank
                ],
                "config_cmd_sha256": config_cmd_sha256[rank],
            }
            for rank in range(4)
        ],
        "attestation_scope": (
            "Exact container name, managed/profile/component/arm labels, image ID, "
            "runtime-unique Docker container ID and StartedAt, generated Config.Cmd, "
            "every explicitly generated environment key, running/not-OOM state, and zero "
            "restart count were checked on all four ranks after readiness. Request probes "
            "must re-attest the same runtime-unique identities immediately before their "
            "first HTTP request."
        ),
    }


def validate_live_arm_receipt(
    receipt_path: Path,
    profile_path: Path,
    expected_arm_id: str,
) -> dict[str, Any]:
    """Validate a public receipt against the exact local canonical profile."""
    if expected_arm_id not in ARM_MTP_TOKENS:
        raise ValueError("expected attribution arm is unsupported")
    receipt, receipt_raw = _read_json(receipt_path, "live-arm receipt")
    profile, profile_raw = _read_json(profile_path, "canonical profile")
    expected_keys = {
        "schema",
        "status",
        "arm",
        "canonical_profile_id",
        "diagnostic_profile_id",
        "canonical_profile_file_sha256",
        "image_id",
        "model_repository",
        "model_revision",
        "layout",
        "cache_salt",
        "ranks",
        "attestation_scope",
    }
    if set(receipt) != expected_keys:
        raise ValueError("live-arm receipt fields are unsupported")
    if receipt["schema"] != LIVE_ARM_RECEIPT_SCHEMA:
        raise ValueError("live-arm receipt schema is unsupported")
    if receipt["status"] != "live-arm-attested":
        raise ValueError("live-arm receipt status is not attested")
    if receipt["arm"] != expected_arm_id:
        raise ValueError("live-arm receipt arm does not match requested arm")
    required_profile = {
        "profile_id",
        "image_id",
        "model_repository",
        "model_revision",
        "container_name",
        "environment",
    }
    if not required_profile <= set(profile):
        raise ValueError("canonical profile lacks live-arm identity fields")
    profile_sha = hashlib.sha256(profile_raw).hexdigest()
    if receipt["canonical_profile_file_sha256"] != profile_sha:
        raise ValueError("live-arm receipt canonical profile digest does not match")
    profile_id = profile["profile_id"]
    if receipt["canonical_profile_id"] != profile_id:
        raise ValueError("live-arm receipt canonical profile ID does not match")
    if receipt["diagnostic_profile_id"] != f"{profile_id}-diag-{expected_arm_id}":
        raise ValueError("live-arm receipt diagnostic profile ID does not match")
    image_id = profile["image_id"]
    if not isinstance(image_id, str) or IMAGE_ID_RE.fullmatch(image_id) is None:
        raise ValueError("canonical profile image ID is invalid")
    if receipt["image_id"] != image_id:
        raise ValueError("live-arm receipt image ID does not match canonical profile")
    model_repository = profile["model_repository"]
    if (
        not isinstance(model_repository, str)
        or not model_repository
        or receipt["model_repository"] != model_repository
    ):
        raise ValueError("live-arm receipt model repository does not match")
    model_revision = profile["model_revision"]
    if (
        not isinstance(model_revision, str)
        or HEX40_RE.fullmatch(model_revision) is None
        or receipt["model_revision"] != model_revision
    ):
        raise ValueError("live-arm receipt model revision does not match")
    canonical_container_name = profile["container_name"]
    if (
        not isinstance(canonical_container_name, str)
        or not canonical_container_name
    ):
        raise ValueError("canonical profile container name is invalid")
    layout = expected_layout(expected_arm_id)
    if receipt["layout"] != layout:
        raise ValueError("live-arm receipt layout does not match attribution arm")
    environment = profile["environment"]
    if not isinstance(environment, dict):
        raise ValueError("canonical profile environment is invalid")
    for key, expected in (
        (
            "VLLM_SPARK_KV_CACHE_MEMORY_BYTES",
            str(layout["kv_cache_memory_bytes_per_rank"]),
        ),
        ("VLLM_SPARK_MAX_MODEL_LEN", str(layout["max_model_len"])),
    ):
        if environment.get(key) != expected:
            raise ValueError(f"canonical profile {key} does not match live-arm layout")
    if receipt["cache_salt"] != cache_salt_for_layout(layout):
        raise ValueError("live-arm receipt cache salt does not match layout")
    ranks = receipt["ranks"]
    if not isinstance(ranks, list) or len(ranks) != 4:
        raise ValueError("live-arm receipt does not attest exactly four ranks")
    rank_keys = {
        "rank", "status", "container_name", "labels", "image_id",
        "container_id", "started_at",
        "explicit_environment_sha256", "config_cmd_sha256",
    }
    diagnostic_container_name = f"{canonical_container_name}-diag-{expected_arm_id}"
    expected_labels = {
        "org.sparkring.managed": "true",
        "org.sparkring.exl3-profile": f"{profile_id}-diag-{expected_arm_id}",
        "org.sparkring.component": "engine",
        "org.sparkring.exl3-attribution": expected_arm_id,
    }
    for rank, item in enumerate(ranks):
        if not isinstance(item, dict) or set(item) != rank_keys:
            raise ValueError("live-arm receipt rank fields are unsupported")
        if item["rank"] != rank or item["status"] != "attested":
            raise ValueError("live-arm receipt ranks are not ordered attestations")
        if item["container_name"] != f"{diagnostic_container_name}-r{rank}":
            raise ValueError("live-arm receipt container name does not match")
        if item["labels"] != expected_labels:
            raise ValueError("live-arm receipt labels do not match")
        if item["image_id"] != image_id:
            raise ValueError("live-arm receipt rank image does not match")
        if (
            not isinstance(item["container_id"], str)
            or CONTAINER_ID_RE.fullmatch(item["container_id"]) is None
        ):
            raise ValueError("live-arm receipt rank container ID is invalid")
        if (
            not isinstance(item["started_at"], str)
            or DOCKER_STARTED_AT_RE.fullmatch(item["started_at"]) is None
        ):
            raise ValueError("live-arm receipt rank StartedAt is invalid")
        for field in ("explicit_environment_sha256", "config_cmd_sha256"):
            if (
                not isinstance(item[field], str)
                or SHA256_RE.fullmatch(item[field]) is None
            ):
                raise ValueError(f"live-arm receipt rank {field} is invalid")
    expected_scope = build_live_arm_receipt(
        arm_id=expected_arm_id,
        canonical_profile_id=profile_id,
        canonical_profile_file_sha256=profile_sha,
        image_id=image_id,
        model_repository=model_repository,
        model_revision=model_revision,
        canonical_container_name=canonical_container_name,
        explicit_environment_sha256=[
            item["explicit_environment_sha256"] for item in ranks
        ],
        config_cmd_sha256=[item["config_cmd_sha256"] for item in ranks],
        observed_runtime_instances=[
            {
                "container_id": item["container_id"],
                "started_at": item["started_at"],
            }
            for item in ranks
        ],
    )["attestation_scope"]
    if receipt["attestation_scope"] != expected_scope:
        raise ValueError("live-arm receipt attestation scope is unsupported")
    if not isinstance(receipt["canonical_profile_file_sha256"], str) or (
        SHA256_RE.fullmatch(receipt["canonical_profile_file_sha256"]) is None
    ):
        raise ValueError("live-arm receipt profile digest is invalid")
    return {
        **receipt,
        "artifact_sha256": hashlib.sha256(receipt_raw).hexdigest(),
    }
