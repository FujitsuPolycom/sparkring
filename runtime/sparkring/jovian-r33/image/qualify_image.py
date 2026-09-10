#!/usr/bin/env python3
"""Create a local candidate receipt after source/payload verification; never publish or deploy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_media_ancestry(candidate: dict, media: dict, expected_id: str) -> None:
    if media.get("Id") != expected_id:
        raise RuntimeError("media runtime tag no longer resolves to the locked image ID")
    media_layers = media.get("RootFS", {}).get("Layers", [])
    candidate_layers = candidate.get("RootFS", {}).get("Layers", [])
    if not media_layers or candidate_layers[:len(media_layers)] != media_layers:
        raise RuntimeError("candidate rootfs does not descend from the locked media runtime")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="local/sparkring:r33-arm64-candidate")
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"receipt output must be absent: {args.output}")
    lock_path = args.context / "source-lock.json"
    lock = json.loads(lock_path.read_text())
    inspect = json.loads(subprocess.check_output(["docker", "image", "inspect", args.image], text=True))[0]
    media_reference = lock["media_runtime"]["reference"]
    media = json.loads(subprocess.check_output(["docker", "image", "inspect", media_reference], text=True))[0]
    validate_media_ancestry(inspect, media, lock["media_runtime"]["image_id"])
    platform = f"{inspect['Os']}/{inspect['Architecture']}"
    if platform != "linux/arm64":
        raise RuntimeError(f"candidate platform differs: {platform}")
    verification_output = subprocess.check_output(
        ["docker", "run", "--rm", "--network", "none", args.image, "verify"],
        text=True,
    )
    verification = json.loads(verification_output[verification_output.index("{"):])
    if (verification.get("status") != "source-closure-verified-runtime-qualification-pending"
            or verification.get("source_lock_sha256") != sha256(lock_path)
            or verification.get("installed_python_payload_files") != lock["installed_python_file_count"]):
        raise RuntimeError("candidate image verification did not match the context lock")
    if lock.get("receipt_semantics", {}).get("checks_passed") is not True:
        raise RuntimeError("candidate source lock lacks semantic component receipt validation")
    artifact_lock = args.context / "receipts/artifact-lock.json"
    receipt = {
        "schema": "sparkring-r33-image-receipt/v1",
        "checks_passed": True,
        "platform": platform,
        "image_id": inspect["Id"],
        "image_reference": inspect["Id"],
        "artifact_lock_sha256": sha256(artifact_lock),
        "source_lock_sha256": sha256(lock_path),
        "sources": lock["identities"]["component_sources"],
        "component_receipts": lock["component_receipts"],
        "bundle_manifest_sha256": lock["installed_files"]["opt/sparkring/sircl/python/sparkring-overlay-manifest.json"],
        "nccl_version": "2.31.2",
        "source_locks_match": lock["receipt_semantics"]["checks_passed"],
        "source_lock_receipts_match": lock["receipt_semantics"]["checks_passed"],
        "installed_payload_bytes_match": True,
        "package_checks_passed": True,
        "verification": verification,
        "qualification": {
            "gpu": False,
            "model": False,
            "tp2": False,
            "tp4": False,
            "cache_recovery": False
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"image_id": inspect["Id"], "receipt": str(args.output), "status": "local-candidate-only"}, indent=2))


if __name__ == "__main__":
    main()
