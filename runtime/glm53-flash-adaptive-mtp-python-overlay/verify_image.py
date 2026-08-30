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
    labels = {
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
        "org.sparkcache.source-tree": pins["sparkcache"]["tree"],
        "org.sparkcache.source-sha256": pins["sparkcache"]["source_tree_sha256"],
        "org.sparkcache.vllm-contract-sha256": pins["sparkcache"]["contract"][
            "sha256"
        ],
        "org.sparkcache.deployment-profile": "glm53-flash-adaptive-mtp-python-overlay",
    }
    runtime_patches = pins["vllm"].get("runtime_patches", ())
    if len(runtime_patches) != 1:
        raise VerifyError("runtime contract requires one DFlash loader patch")
    patch = runtime_patches[0]
    labels["org.sparkring.vllm.dflash-draft-loader-patch-sha256"] = patch[
        "sha256"
    ]
    labels["org.sparkring.vllm.dflash-draft-loader-postimage-sha256"] = patch[
        "postimage_sha256"
    ]
    composed = pins["vllm"].get("composed_runtime_patches", ())
    if len(composed) != 1:
        raise VerifyError("runtime contract requires one recurrent-boundary patch")
    labels["org.sparkring.vllm.recurrent-boundary-patch-sha256"] = composed[0][
        "sha256"
    ]
    cleanup = pins.get("runtime_cleanup", {}).get("deep_ep")
    if cleanup is not None:
        labels["org.sparkring.runtime.removed-deep-ep-distribution"] = (
            f"{cleanup['distribution']}=={cleanup['version']}"
        )
        labels["org.sparkring.runtime.deep-ep-removal-receipt-sha256"] = cleanup[
            "receipt_sha256"
        ]
    return labels


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
import importlib.metadata
import importlib.util
import json
import pathlib

root = pathlib.Path('/opt/sparkring/runtime/python-overlay')
cuda_placement = pathlib.Path('/opt/sparkcache-src/sparkcache/native/build-cuda/libspark_cache_placement.so')
contract = pathlib.Path('/opt/sparkcache-src/sparkcache/runtime_patches/vllm-kv-block-lease-contract-glm53-b12x-kda-adaptive-mtp.json')
sparkcache_source_receipt = root / 'sparkcache-source-tree.sha256'
wheel_receipt = (root / 'b12x-wheel.sha256').read_text(encoding='utf-8').split()
deep_ep_receipt = root / 'deep-ep-removal-receipt.json'
try:
    importlib.metadata.distribution('deep_ep')
    deep_ep_distribution_present = True
except importlib.metadata.PackageNotFoundError:
    deep_ep_distribution_present = False
print(json.dumps({
    'b12x_wheel_sha256': wheel_receipt[0],
    'b12x_wheel': pathlib.Path(wheel_receipt[1]).name,
    'sparkcache_cuda_placement_sha256': hashlib.sha256(cuda_placement.read_bytes()).hexdigest(),
    'sparkcache_contract_sha256': hashlib.sha256(contract.read_bytes()).hexdigest(),
    'sparkcache_source_tree_sha256': sparkcache_source_receipt.read_text(encoding='utf-8').strip(),
    'source_receipt_sha256': hashlib.sha256((root / 'source-receipt.json').read_bytes()).hexdigest(),
    'deep_ep_receipt': json.loads(deep_ep_receipt.read_text(encoding='utf-8')) if deep_ep_receipt.is_file() else None,
    'deep_ep_receipt_sha256': hashlib.sha256(deep_ep_receipt.read_bytes()).hexdigest() if deep_ep_receipt.is_file() else None,
    'deep_ep_module_present': importlib.util.find_spec('deep_ep') is not None,
    'deep_ep_owners': importlib.metadata.packages_distributions().get('deep_ep') or [],
    'deep_ep_distribution_present': deep_ep_distribution_present,
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


def verify_runtime_cleanup(artifacts: dict[str, Any], pins: dict[str, Any]) -> None:
    cleanup = pins.get("runtime_cleanup", {}).get("deep_ep")
    if cleanup is None:
        return
    expected_removal_receipt = {
        "schema": "sparkring-python-distribution-removal/v1",
        "status": "implemented",
        "module": cleanup["module"],
        "distribution": cleanup["distribution"],
        "version": cleanup["version"],
        "postcondition": "module-absent",
        "reason": cleanup["reason"],
    }
    if artifacts.get("deep_ep_receipt") != expected_removal_receipt:
        raise VerifyError("installed DeepEP removal receipt differs from its pin")
    if artifacts.get("deep_ep_receipt_sha256") != cleanup["receipt_sha256"]:
        raise VerifyError("installed DeepEP removal receipt checksum differs")
    if artifacts.get("deep_ep_module_present") is not False:
        raise VerifyError("deep_ep remains importable in the composed image")
    if artifacts.get("deep_ep_owners") != []:
        raise VerifyError("deep_ep still has installed distribution owners")
    if artifacts.get("deep_ep_distribution_present") is not False:
        raise VerifyError("the attested deep_ep distribution remains installed")


def verify_runtime_patch_report(runtime: dict[str, Any], pins: dict[str, Any]) -> None:
    """Require the exact DFlash member and exact recurrent postimage set."""

    records = runtime.get("vllm_runtime_patches")
    if not isinstance(records, list) or any(
        not isinstance(record, dict)
        or set(record) != {"path", "sha256"}
        or not isinstance(record["path"], str)
        or not isinstance(record["sha256"], str)
        for record in records
    ):
        raise VerifyError("runtime patch verification report is malformed")
    observed = {record["path"]: record["sha256"] for record in records}
    if len(observed) != len(records):
        raise VerifyError("runtime patch verification report contains duplicate paths")

    expected_dflash = {
        record["target"]: record["postimage_sha256"]
        for record in pins["vllm"].get("runtime_patches", ())
    }
    observed_dflash = {
        path: observed[path] for path in expected_dflash if path in observed
    }
    if observed_dflash != expected_dflash:
        raise VerifyError("runtime did not verify the exact DFlash draft-loader patch")

    expected_recurrent = {
        target["path"]: target["postimage_sha256"]
        for record in pins["vllm"].get("composed_runtime_patches", ())
        for target in record["targets"]
    }
    observed_recurrent = {
        path: observed[path] for path in expected_recurrent if path in observed
    }
    if observed_recurrent != expected_recurrent:
        raise VerifyError("runtime did not verify the exact recurrent postimage set")
    if set(observed) != set(expected_dflash) | set(expected_recurrent):
        raise VerifyError("runtime patch verification report contains unexpected paths")


def verify_image(engine: str, image: str, pins_path: Path = PINS) -> dict[str, Any]:
    pins = load_pins(pins_path)
    inspection = inspect_image(engine, image)
    validate_output(inspection, pins)
    runtime = runtime_contract_probe(engine, image)
    artifacts = artifact_probe(engine, image)
    if runtime.get("vllm_python_files_verified") != 31:
        raise VerifyError("runtime did not verify all 31 vLLM Python overlay files")
    verify_runtime_patch_report(runtime, pins)
    if artifacts["sparkcache_contract_sha256"] != pins["sparkcache"]["contract"]["sha256"]:
        raise VerifyError("installed SparkCache lease contract differs from its pin")
    if artifacts["sparkcache_source_tree_sha256"] != pins["sparkcache"][
        "source_tree_sha256"
    ]:
        raise VerifyError("clean SparkCache source receipt differs from its pin")
    verify_runtime_cleanup(artifacts, pins)
    for name in (
        "b12x_wheel_sha256",
        "sparkcache_cuda_placement_sha256",
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
        "org.sparkcache.cuda-placement-library-sha256": artifacts[
            "sparkcache_cuda_placement_sha256"
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
