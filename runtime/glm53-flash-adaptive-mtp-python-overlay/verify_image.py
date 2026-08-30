#!/usr/bin/env python3
"""Verify the public-base GLM-5.3 Python-overlay image."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
PINS = HERE / "pins.json"
PINS_SCHEMA = "sparkring-glm53-public-python-overlay/v1"
RECEIPT_SCHEMA = "sparkring-glm53-public-python-overlay-image/v1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SHA256_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class VerifyError(RuntimeError):
    """An image differs from the public Python-overlay contract."""


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


def load_pins(path: Path = PINS) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != PINS_SCHEMA:
        raise VerifyError(f"{path} does not use schema {PINS_SCHEMA}")
    return value


def inspect_image(engine: str, image: str) -> dict[str, Any]:
    values = json.loads(run((engine, "image", "inspect", image)))
    if not isinstance(values, list) or len(values) != 1:
        raise VerifyError("container engine returned an unexpected image inspection")
    return values[0]


def validate_platform(document: dict[str, Any]) -> None:
    if document.get("Architecture") != "arm64" or document.get("Os") != "linux":
        raise VerifyError(
            "image platform mismatch: expected linux/arm64, got "
            f"{document.get('Os')}/{document.get('Architecture')}"
        )
    if SHA256_ID.fullmatch(str(document.get("Id", ""))) is None:
        raise VerifyError(f"image ID is not an immutable SHA-256: {document.get('Id')!r}")


def validate_labels(document: dict[str, Any], expected: dict[str, str]) -> None:
    labels = document.get("Config", {}).get("Labels") or {}
    if not isinstance(labels, dict):
        raise VerifyError("image labels are not a JSON object")
    for name, value in expected.items():
        observed = labels.get(name)
        if observed != value:
            raise VerifyError(
                f"image label {name} mismatch: expected {value!r}, got {observed!r}"
            )


def validate_base(document: dict[str, Any], pins: dict[str, Any]) -> None:
    validate_platform(document)
    base = pins["public_base"]
    if document["Id"] != base["image_id"]:
        raise VerifyError(
            f"public base image ID mismatch: expected {base['image_id']}, "
            f"got {document['Id']}"
        )
    validate_labels(document, base["labels"])


def expected_output_labels(pins: dict[str, Any]) -> dict[str, str]:
    return {
        "org.opencontainers.image.base.name": pins["public_base"]["reference"],
        "org.sparkring.base.image-id": pins["public_base"]["image_id"],
        "org.sparkring.vllm.native.commit": pins["vllm"]["native_commit"],
        "org.sparkring.vllm.python.commit": pins["vllm"]["python_commit"],
        "org.sparkring.vllm.python.tree": pins["vllm"]["python_tree"],
        "org.sparkring.vllm.python-overlay-manifest-sha256": pins["vllm"][
            "overlay_manifest_sha256"
        ],
        "org.jovian.vllm.commit": pins["vllm"]["native_commit"],
        "org.jovian.b12x.commit": pins["b12x"]["commit"],
        "org.sparkring.b12x.tree": pins["b12x"]["tree"],
        "org.sparkring.nccl.commit": pins["public_base"]["labels"][
            "org.sparkring.nccl.commit"
        ],
        "org.sparkcache.source-revision": pins["sparkcache"]["commit"],
        "org.sparkcache.source-sha256": pins["sparkcache"]["source_tree_sha256"],
        "org.sparkcache.vllm-contract-sha256": pins["sparkcache"]["contract"][
            "sha256"
        ],
        "org.sparkcache.deployment-profile": "glm53-flash-adaptive-mtp-python-overlay",
    }


def validate_output(document: dict[str, Any], pins: dict[str, Any]) -> None:
    validate_platform(document)
    validate_labels(document, expected_output_labels(pins))
    labels = document.get("Config", {}).get("Labels") or {}
    revision = labels.get("org.opencontainers.image.revision")
    if not isinstance(revision, str) or GIT_COMMIT.fullmatch(revision) is None:
        raise VerifyError("SparkRing image revision must contain 40 hexadecimal characters")
    receipt = labels.get("org.sparkring.source-receipt-sha256")
    if not isinstance(receipt, str) or SHA256.fullmatch(receipt) is None:
        raise VerifyError("source receipt label must contain a lowercase SHA-256")


def runtime_contract_probe(engine: str, image: str) -> dict[str, Any]:
    root = "/opt/sparkring/runtime/python-overlay"
    output = run(
        (
            engine,
            "run",
            "--rm",
            "--entrypoint",
            "python3",
            image,
            f"{root}/overlay_contract.py",
            "--pins",
            f"{root}/pins.json",
            "--manifest",
            f"{root}/vllm-python-overlay.json",
            "verify-composed",
            "--root",
            "/usr/local/lib/python3.12/dist-packages",
            "--site-root",
            "/usr/local/lib/python3.12/dist-packages",
            "--console-script",
            "/usr/local/bin/vllm",
            "--base-record",
            f"{root}/retained-native.json",
        )
    )
    try:
        return json.loads(output.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise VerifyError(f"runtime contract probe did not return JSON: {output!r}") from exc


def artifact_probe(engine: str, image: str) -> dict[str, Any]:
    program = r"""
import hashlib
import json
import pathlib

root = pathlib.Path('/opt/sparkring/runtime/python-overlay')
native = pathlib.Path('/opt/sparkcache-src/sparkcache/native/build-cuda/libspark_cache_placement.so')
contract = pathlib.Path('/opt/sparkcache-src/sparkcache/runtime_patches/vllm-kv-block-lease-contract-glm53-b12x-kda-adaptive-mtp.json')
wheel_receipt = (root / 'b12x-wheel.sha256').read_text(encoding='utf-8').split()
print(json.dumps({
    'b12x_wheel_sha256': wheel_receipt[0],
    'b12x_wheel': pathlib.Path(wheel_receipt[1]).name,
    'sparkcache_native_sha256': hashlib.sha256(native.read_bytes()).hexdigest(),
    'sparkcache_contract_sha256': hashlib.sha256(contract.read_bytes()).hexdigest(),
    'source_receipt_sha256': hashlib.sha256((root / 'source-receipt.json').read_bytes()).hexdigest(),
}, sort_keys=True))
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
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise VerifyError(f"artifact probe did not return JSON: {output!r}") from exc


def verify_image(engine: str, image: str, pins_path: Path = PINS) -> dict[str, Any]:
    pins = load_pins(pins_path)
    inspection = inspect_image(engine, image)
    validate_output(inspection, pins)
    runtime = runtime_contract_probe(engine, image)
    artifacts = artifact_probe(engine, image)
    if runtime.get("vllm_python_files_verified") != 31:
        raise VerifyError("runtime did not verify all 31 vLLM Python overlay files")
    if artifacts["sparkcache_contract_sha256"] != pins["sparkcache"]["contract"]["sha256"]:
        raise VerifyError("installed SparkCache lease contract differs from its pin")
    for name in (
        "b12x_wheel_sha256",
        "sparkcache_native_sha256",
        "source_receipt_sha256",
    ):
        if SHA256.fullmatch(str(artifacts.get(name, ""))) is None:
            raise VerifyError(f"artifact probe returned an invalid {name}")
    labels = inspection.get("Config", {}).get("Labels") or {}
    dynamic_labels = {
        "org.sparkring.vllm.native-elf-manifest-sha256": runtime[
            "native_elf_manifest_sha256"
        ],
        "org.sparkring.vllm.native-dispatch-manifest-sha256": runtime[
            "native_dispatch_manifest_sha256"
        ],
        "org.sparkcache.native-library-sha256": artifacts[
            "sparkcache_native_sha256"
        ],
        "org.sparkring.source-receipt-sha256": artifacts[
            "source_receipt_sha256"
        ],
    }
    for name, expected in dynamic_labels.items():
        if labels.get(name) != expected:
            raise VerifyError(
                f"image label {name} mismatch: expected {expected}, got {labels.get(name)}"
            )
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "implemented",
        "image": image,
        "image_id": inspection["Id"],
        "repo_digests": sorted(inspection.get("RepoDigests") or []),
        "platform": "linux/arm64",
        "labels": dict(sorted((inspection.get("Config", {}).get("Labels") or {}).items())),
        "runtime_contract": runtime,
        "artifacts": artifacts,
        "limitation": (
            "Construction, Python imports, retained native bytes, and source contracts "
            "are verified. Four-rank model loading and serving are not qualified."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default="docker")
    parser.add_argument("--pins", type=Path, default=PINS)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--base-image")
    group.add_argument("--image")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        pins = load_pins(args.pins.resolve())
        if args.base_image:
            document = inspect_image(args.engine, args.base_image)
            validate_base(document, pins)
            result: dict[str, Any] = {
                "status": "qualified",
                "base_image": args.base_image,
                "image_id": document["Id"],
            }
        else:
            result = verify_image(args.engine, args.image, args.pins.resolve())
    except (OSError, KeyError, json.JSONDecodeError, VerifyError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
