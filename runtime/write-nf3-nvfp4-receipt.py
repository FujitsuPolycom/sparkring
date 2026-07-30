#!/usr/bin/env python3
"""Write a canonical host receipt for the final NVFP4/FP8-RoPE image."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_receipt(
    *,
    image: str,
    image_id: str,
    nf3_image_id: str,
    mla_image_id: str,
    source_commit: str,
    verifier_report: Path,
) -> dict[str, object]:
    for label, value in (
        ("final image ID", image_id),
        ("NF3 parent image ID", nf3_image_id),
        ("MLA parent image ID", mla_image_id),
    ):
        if not IMAGE_ID_RE.fullmatch(value):
            raise ValueError(f"{label} must be an immutable sha256 image ID")
    if not COMMIT_RE.fullmatch(source_commit):
        raise ValueError("SparkRing source commit must be 40 lowercase hex")
    if not image or any(character.isspace() for character in image):
        raise ValueError("image reference must be nonempty and whitespace-free")

    verifier = json.loads(verifier_report.read_text(encoding="utf-8"))
    if verifier.get("schema") != (
        "sparkring-nf3-nvfp4-rope8-verification/v1"
    ):
        raise ValueError("unexpected NVFP4 verifier-report schema")
    if verifier.get("passed") is not True:
        raise ValueError("NVFP4 verifier report did not pass")

    return {
        "schema": "sparkring-nf3-nvfp4-runtime-receipt/v1",
        "profile": "nvfp4-rope8",
        "image": image,
        "image_id": image_id,
        "parents": {
            "nf3_image_id": nf3_image_id,
            "mla_image_id": mla_image_id,
        },
        "sparkring_source_commit": source_commit,
        "verifier_report": {
            "schema": verifier["schema"],
            "sha256": _sha256(verifier_report),
            "passed": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--nf3-image-id", required=True)
    parser.add_argument("--mla-image-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--verifier-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    receipt = build_receipt(
        image=args.image,
        image_id=args.image_id,
        nf3_image_id=args.nf3_image_id,
        mla_image_id=args.mla_image_id,
        source_commit=args.source_commit,
        verifier_report=args.verifier_report,
    )
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(args.output)
    print(f"RECEIPT={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
