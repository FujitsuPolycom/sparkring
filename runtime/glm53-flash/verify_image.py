#!/usr/bin/env python3
"""Verify a GLM-5.3 ARM64 runtime image and write its immutable receipt."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_PINS = HERE / "pins.json"
PINS_SCHEMA = "sparkring-glm53-flash-runtime-lock/v1"
RECEIPT_SCHEMA = "sparkring-glm53-runtime-image/v1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SHA256_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")


class VerifyError(RuntimeError):
    """Raised when an image differs from the GLM-5.3 runtime contract."""


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
        raise VerifyError(f"command failed ({' '.join(arguments)}): {detail}")
    return completed.stdout.strip()


def load_pins(path: Path = DEFAULT_PINS) -> dict[str, Any]:
    pins = json.loads(path.read_text(encoding="utf-8"))
    if pins.get("schema") != PINS_SCHEMA:
        raise VerifyError(f"unsupported pins schema: {pins.get('schema')!r}")
    return pins


def expected_labels(pins: dict[str, Any]) -> dict[str, str]:
    build = pins["public_image_build"]
    sources = build["sources"]
    return {
        "org.opencontainers.image.source": "https://github.com/FujitsuPolycom/sparkring",
        "org.opencontainers.image.licenses": (
            "LicenseRef-NVIDIA-Deep-Learning-Container AND Apache-2.0 AND BSD-3-Clause"
        ),
        "org.jovian.architecture": "linux-arm64-sm121",
        "org.jovian.vllm.commit": sources["vllm"]["commit"],
        "org.jovian.b12x.commit": sources["b12x"]["commit"],
        "org.sparkring.nccl.commit": sources["nccl"]["commit"],
        "org.sparkring.nccl.patched-tree": sources["nccl"]["patched_tree"],
        "org.sparkring.nccl.patch-sha256": sources["nccl"]["patches"][0]["sha256"],
    }


def validate_inspection(document: dict[str, Any], pins: dict[str, Any]) -> None:
    image_id = str(document.get("Id", ""))
    if SHA256_ID.fullmatch(image_id) is None:
        raise VerifyError(f"invalid image ID: {image_id!r}")
    if document.get("Architecture") != "arm64" or document.get("Os") != "linux":
        raise VerifyError(
            "runtime image must use linux/arm64, got "
            f"{document.get('Os')}/{document.get('Architecture')}"
        )
    labels = document.get("Config", {}).get("Labels") or {}
    for name, expected in expected_labels(pins).items():
        observed = labels.get(name)
        if observed != expected:
            raise VerifyError(
                f"image label {name} drift: expected {expected!r}, got {observed!r}"
            )


def runtime_probe(engine: str, image: str) -> dict[str, Any]:
    program = """
import hashlib
import importlib.metadata
import json
import pathlib
import platform

nccl = pathlib.Path('/opt/sparkring/nccl/libnccl.so.2.30.7')
document = {
    'python': platform.python_version(),
    'vllm': importlib.metadata.version('vllm'),
    'b12x': importlib.metadata.version('b12x'),
    'instanttensor': importlib.metadata.version('instanttensor'),
    'nccl_sha256': hashlib.sha256(nccl.read_bytes()).hexdigest(),
    'nccl_bytes': nccl.stat().st_size,
    'source_receipt_present': pathlib.Path('/opt/sparkring/runtime/source-receipt.json').is_file(),
    'pins_present': pathlib.Path('/opt/sparkring/runtime/pins.json').is_file(),
}
print(json.dumps(document, sort_keys=True))
""".strip()
    output = run(
        (
            engine,
            "run",
            "--rm",
            "--entrypoint",
            "python3",
            image,
            "-c",
            program,
        )
    )
    try:
        probe = json.loads(output)
    except json.JSONDecodeError as exc:
        raise VerifyError(f"runtime probe did not return JSON: {output!r}") from exc
    if SHA256.fullmatch(str(probe.get("nccl_sha256", ""))) is None:
        raise VerifyError("runtime probe returned an invalid NCCL SHA-256")
    if not probe.get("source_receipt_present") or not probe.get("pins_present"):
        raise VerifyError("runtime image omits its source receipt or pins")
    return probe


def verify_image(engine: str, image: str, pins_path: Path) -> dict[str, Any]:
    pins = load_pins(pins_path)
    inspection_raw = run((engine, "image", "inspect", image))
    inspections = json.loads(inspection_raw)
    if not isinstance(inspections, list) or len(inspections) != 1:
        raise VerifyError("container engine returned an unexpected inspection document")
    inspection = inspections[0]
    validate_inspection(inspection, pins)
    probe = runtime_probe(engine, image)
    expected_nccl = pins["public_image_build"]["outputs"]["nccl_library_sha256"]
    if expected_nccl is not None and probe["nccl_sha256"] != expected_nccl:
        raise VerifyError(
            "NCCL library drift: expected "
            f"{expected_nccl}, got {probe['nccl_sha256']}"
        )
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "implemented",
        "image": image,
        "image_id": inspection["Id"],
        "repo_digests": sorted(inspection.get("RepoDigests") or []),
        "platform": "linux/arm64",
        "size_bytes": inspection.get("Size"),
        "labels": dict(sorted((inspection.get("Config", {}).get("Labels") or {}).items())),
        "runtime_probe": probe,
        "limitation": (
            "This receipt verifies construction and imports. Four-rank TP4/DCP1 "
            "serving remains unqualified until a live receipt names this image digest."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--engine", default="docker")
    parser.add_argument("--pins", type=Path, default=DEFAULT_PINS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        receipt = verify_image(args.engine, args.image, args.pins.resolve())
    except (OSError, KeyError, json.JSONDecodeError, VerifyError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
