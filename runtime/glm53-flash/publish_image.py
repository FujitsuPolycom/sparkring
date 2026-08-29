#!/usr/bin/env python3
"""Publish a verified GLM-5.3 runtime image with its SPDX SBOM."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


RECEIPT_SCHEMA = "sparkring-glm53-runtime-publication/v1"
DESTINATION = re.compile(
    r"ghcr\.io/fujitsupolycom/sparkring-glm53-runtime:[a-z0-9._-]+\Z"
)


class PublishError(RuntimeError):
    """Raised when runtime publication cannot prove its immutable inputs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublishError(f"invalid {description}: {path}") from exc
    if not isinstance(value, dict):
        raise PublishError(f"{description} must contain one JSON object")
    return value


def run(argv: Iterable[str]) -> str:
    arguments = list(argv)
    completed = subprocess.run(
        arguments,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PublishError(f"command failed ({' '.join(arguments)}): {detail}")
    return completed.stdout.strip()


def validate_destination(destination: str) -> None:
    if DESTINATION.fullmatch(destination) is None:
        raise PublishError("destination must use the SparkRing GLM-5.3 GHCR repository")
    if destination.endswith(":latest"):
        raise PublishError("the moving latest tag is not a publication identity")


def publish(
    *,
    image: str,
    destination: str,
    build_receipt_path: Path,
    sbom_path: Path,
) -> dict[str, Any]:
    validate_destination(destination)
    build_receipt = load_json(build_receipt_path, "build receipt")
    if build_receipt.get("schema") != "sparkring-glm53-runtime-image/v1":
        raise PublishError("build receipt schema is not a GLM-5.3 runtime image receipt")
    image_id = build_receipt.get("image_id")
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise PublishError("build receipt does not contain an immutable image ID")
    sbom = load_json(sbom_path, "SPDX SBOM")
    if not str(sbom.get("spdxVersion", "")).startswith("SPDX-"):
        raise PublishError("SBOM does not declare an SPDX version")
    observed = run(("docker", "image", "inspect", "--format", "{{.Id}}", image))
    if observed != image_id:
        raise PublishError(
            f"local image drift: receipt names {image_id}, Docker reports {observed}"
        )
    run(("docker", "tag", image_id, destination))
    run(("docker", "push", destination))
    repo_digests = sorted(
        json.loads(
            run(
                (
                    "docker",
                    "image",
                    "inspect",
                    "--format",
                    "{{json .RepoDigests}}",
                    destination,
                )
            )
        )
    )
    repository = destination.rsplit(":", 1)[0]
    matching = [item for item in repo_digests if item.startswith(repository + "@sha256:")]
    if len(matching) != 1:
        raise PublishError(f"registry digest resolution is ambiguous: {matching}")
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "implemented",
        "image_id": image_id,
        "destination_tag": destination,
        "registry_digest": matching[0],
        "build_receipt_sha256": sha256_file(build_receipt_path),
        "sbom": {
            "format": "SPDX JSON",
            "sha256": sha256_file(sbom_path),
        },
        "limitation": (
            "The image is implemented but unqualified. Four-rank TP4/DCP1 "
            "qualification must name registry_digest before status changes."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--build-receipt", required=True, type=Path)
    parser.add_argument("--sbom", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        receipt = publish(
            image=args.image,
            destination=args.destination,
            build_receipt_path=args.build_receipt.resolve(),
            sbom_path=args.sbom.resolve(),
        )
    except (OSError, json.JSONDecodeError, PublishError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
