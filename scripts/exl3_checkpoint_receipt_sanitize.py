#!/usr/bin/env python3
"""Create a closed, path-free public summary of a private EXL3 receipt run.

The receipt lifecycle's plan, raw checkpoint receipt, and raw execution
evidence are private operator artifacts.  This offline-only tool accepts only a
successful, restored lifecycle result and emits a small schema-closed summary.
It never copies phase output, errors, SSH targets, commands, paths, timestamps,
model filenames, or display-root strings from the private inputs.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import exl3_checkpoint_receipt_lifecycle as lifecycle  # noqa: E402
import exl3_sparkcache_config as sparkcache_config  # noqa: E402
import sparkring_exl3_lmcache_launcher as lmcache  # noqa: E402


SCHEMA = "sparkring-exl3-checkpoint-receipt-public/v1"
SENSITIVITY = "public-sanitized"
HEX32 = re.compile(r"^[0-9a-f]{32}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


class SanitizeError(RuntimeError):
    """The private inputs cannot authorize a public summary."""


def _require_exact_keys(value: object, expected: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise SanitizeError(f"{label} keys are not schema-closed")
    return value


def _require_hex(value: object, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise SanitizeError(f"{label} is invalid")
    return value


def build_public_receipt(
    receipt: object,
    evidence: object,
    *,
    receipt_payload: bytes,
    evidence_payload: bytes,
) -> dict:
    """Whitelist a successful private transaction into the public v1 schema."""
    try:
        identity = sparkcache_config.validate_checkpoint_receipt(receipt)
    except sparkcache_config.SparkCacheProfileError as error:
        raise SanitizeError(f"private checkpoint receipt is invalid: {error}") from error
    evidence = _require_exact_keys(
        evidence,
        {
            "schema",
            "sensitivity",
            "run_id",
            "lane",
            "maturity",
            "execution_state",
            "started_at",
            "completed_at",
            "phases",
            "receipt",
            "restoration",
            "passed",
            "profile_id",
            "image_id",
            "generator_sha256",
            "required_ranks",
        },
        "private evidence",
    )
    if evidence["schema"] != lifecycle.EVIDENCE_SCHEMA:
        raise SanitizeError("private evidence schema is wrong")
    if evidence["sensitivity"] != lifecycle.PRIVATE_SENSITIVITY:
        raise SanitizeError("private evidence sensitivity is wrong")
    if evidence["lane"] != "public-functional":
        raise SanitizeError("private evidence lane is wrong")
    if evidence["profile_id"] != lmcache.PROFILE_ID:
        raise SanitizeError("private evidence profile is not canonical")
    if evidence["required_ranks"] != [0, 1, 2, 3]:
        raise SanitizeError("private evidence does not bind exact ranks 0..3")
    if evidence["passed"] is not True or evidence["execution_state"] != "completed-restored":
        raise SanitizeError("only a completed-restored successful run is publishable")
    restoration = evidence["restoration"]
    if not isinstance(restoration, dict) or restoration.get("passed") is not True:
        raise SanitizeError("canonical restoration is not proven")
    _require_hex(evidence["run_id"], HEX32, "private run id")
    image_id = _require_hex(evidence["image_id"], IMAGE_ID, "image id")
    generator_sha256 = _require_hex(
        evidence["generator_sha256"], HEX64, "generator SHA-256"
    )

    receipt_summary = _require_exact_keys(
        evidence["receipt"],
        {
            "schema",
            "checkpoint_identity_sha256",
            "receipt_sha256",
            "file_count",
        },
        "private evidence receipt summary",
    )
    receipt_sha256 = lifecycle.sha256_bytes(receipt_payload)
    if receipt_summary != {
        "schema": "sparkcache-checkpoint-manifest-v2",
        "checkpoint_identity_sha256": identity,
        "receipt_sha256": receipt_sha256,
        "file_count": receipt["file_count"],
    }:
        raise SanitizeError("private receipt bytes do not match execution evidence")

    public = {
        "schema": SCHEMA,
        "sensitivity": SENSITIVITY,
        "lane": "public-functional",
        "maturity": "live-evidence-not-acceptance",
        "hardware": "four directly cabled NVIDIA DGX Sparks / GB10",
        "profile_id": lmcache.PROFILE_ID,
        "required_ranks": [0, 1, 2, 3],
        "image_id": image_id,
        "generator_sha256": generator_sha256,
        "checkpoint": {
            "schema": "sparkcache-checkpoint-manifest-v2-summary",
            "checkpoint_identity_sha256": identity,
            "file_count": receipt["file_count"],
            "canonical_receipt_sha256": lifecycle.sha256_bytes(
                lifecycle.canonical_bytes(receipt)
            ),
            "receipt_file_sha256": receipt_sha256,
        },
        "transaction": {
            "execution_state": "completed-restored",
            "restoration_passed": True,
        },
        "private_source_commitments": {
            "receipt_sha256": receipt_sha256,
            "evidence_sha256": lifecycle.sha256_bytes(evidence_payload),
        },
        "publication_scope": (
            "Sanitized receipt-transaction evidence only; not correctness, "
            "performance, persistence validation, or acceptance."
        ),
    }
    validate_public_receipt(public)
    return public


def validate_public_receipt(document: object) -> None:
    document = _require_exact_keys(
        document,
        {
            "schema",
            "sensitivity",
            "lane",
            "maturity",
            "hardware",
            "profile_id",
            "required_ranks",
            "image_id",
            "generator_sha256",
            "checkpoint",
            "transaction",
            "private_source_commitments",
            "publication_scope",
        },
        "public receipt",
    )
    fixed = {
        "schema": SCHEMA,
        "sensitivity": SENSITIVITY,
        "lane": "public-functional",
        "maturity": "live-evidence-not-acceptance",
        "hardware": "four directly cabled NVIDIA DGX Sparks / GB10",
        "profile_id": lmcache.PROFILE_ID,
        "required_ranks": [0, 1, 2, 3],
        "publication_scope": (
            "Sanitized receipt-transaction evidence only; not correctness, "
            "performance, persistence validation, or acceptance."
        ),
    }
    for key, expected in fixed.items():
        if document[key] != expected:
            raise SanitizeError(f"public receipt {key} is wrong")
    _require_hex(document["image_id"], IMAGE_ID, "public image id")
    _require_hex(document["generator_sha256"], HEX64, "public generator hash")
    checkpoint = _require_exact_keys(
        document["checkpoint"],
        {
            "schema",
            "checkpoint_identity_sha256",
            "file_count",
            "canonical_receipt_sha256",
            "receipt_file_sha256",
        },
        "public checkpoint summary",
    )
    if checkpoint["schema"] != "sparkcache-checkpoint-manifest-v2-summary":
        raise SanitizeError("public checkpoint summary schema is wrong")
    for key in (
        "checkpoint_identity_sha256",
        "canonical_receipt_sha256",
        "receipt_file_sha256",
    ):
        _require_hex(checkpoint[key], HEX64, f"public checkpoint {key}")
    if (
        isinstance(checkpoint["file_count"], bool)
        or not isinstance(checkpoint["file_count"], int)
        or checkpoint["file_count"] < 1
    ):
        raise SanitizeError("public checkpoint file_count is invalid")
    transaction = _require_exact_keys(
        document["transaction"],
        {"execution_state", "restoration_passed"},
        "public transaction",
    )
    if transaction != {
        "execution_state": "completed-restored",
        "restoration_passed": True,
    }:
        raise SanitizeError("public transaction is not completed-restored")
    commitments = _require_exact_keys(
        document["private_source_commitments"],
        {"receipt_sha256", "evidence_sha256"},
        "public source commitments",
    )
    for key, value in commitments.items():
        _require_hex(value, HEX64, f"public source commitment {key}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        receipt_payload = args.receipt.read_bytes()
        evidence_payload = args.evidence.read_bytes()
        receipt = lifecycle.strict_json_loads(receipt_payload, "private receipt")
        evidence = lifecycle.strict_json_loads(evidence_payload, "private evidence")
        public = build_public_receipt(
            receipt,
            evidence,
            receipt_payload=receipt_payload,
            evidence_payload=evidence_payload,
        )
        payload = lifecycle.encoded_pretty(public)
        with args.output.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except (
        OSError,
        lifecycle.LifecycleError,
        SanitizeError,
    ) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
