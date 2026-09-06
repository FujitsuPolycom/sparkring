#!/usr/bin/env python3
"""Prepare or build a private ARM64 image containing the pinned mesh bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = "sparkring-overlay-manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def safe_relative(relative: str) -> PurePosixPath:
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe manifest path: {relative!r}")
    return path


def verify_bundle(bundle: Path, expected: str) -> list[dict]:
    manifest = bundle / MANIFEST
    if sha256(manifest) != expected:
        raise ValueError("Mesh bundle manifest differs from the profile pin")
    records = read_json(manifest).get("files", [])
    if not isinstance(records, list) or not records:
        raise ValueError("Mesh bundle manifest contains no files")
    names = set()
    for record in records:
        relative = safe_relative(record["path"])
        if relative.as_posix() in names:
            raise ValueError(f"Duplicate mesh bundle file: {relative}")
        names.add(relative.as_posix())
        candidate = bundle.joinpath(*relative.parts)
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(bundle.resolve()):
            raise ValueError(f"Mesh bundle path escapes its root: {relative}")
        if not candidate.is_file() or sha256(candidate) != record["sha256"]:
            raise ValueError(f"Mesh bundle file differs from its manifest: {relative}")
    observed = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file()}
    if observed != names | {MANIFEST}:
        raise ValueError(f"Mesh bundle contains unmanifested files: {sorted(observed - names - {MANIFEST})}")
    return records


def prepare(bundle: Path, context: Path) -> dict:
    """Copy only content-verified inputs into a directory that does not exist."""
    if context.exists():
        raise ValueError(f"Build context already exists: {context}")
    profile = read_json(HERE / "pins.json")
    base_path = (HERE / profile["image_pins"]).resolve()
    base = read_json(base_path)
    records = verify_bundle(bundle, profile["canonical_bundle_manifest_sha256"])
    marker = (HERE / profile["marker"]["source"]).resolve()
    if sha256(marker) != profile["marker"]["source_sha256"]:
        raise ValueError("RDMA transmit marker source differs from the profile pin")
    files = {
        "Dockerfile": HERE / "Dockerfile",
        "verify_mesh_image.py": HERE / "verify_mesh_image.py",
        "marker.c": marker,
        "warmup_dflash.py": HERE.parent / "glm53-flash-jj-r8-gb10/warmup_dflash.py",
        "receipts/profile-pins.json": HERE / "pins.json",
        "receipts/parent-pins.json": base_path,
        "receipts/build_image.py": HERE / "build_image.py",
        "receipts/build-Dockerfile": HERE / "Dockerfile",
        "receipts/sparkring/LICENSE": ROOT / "LICENSE",
        "receipts/sparkring/NOTICE": ROOT / "NOTICE",
        "receipts/sparkring/THIRD_PARTY_NOTICES.md": ROOT / "THIRD_PARTY_NOTICES.md",
        "receipts/b12x-roce-LICENSE": ROOT / "third_party/b12x_roce/LICENSE",
        "receipts/b12x-roce-provenance.json": ROOT / "third_party/b12x_roce/provenance.json",
        f"bundle/{MANIFEST}": bundle / MANIFEST,
    }
    files.update({f"bundle/{record['path']}": bundle / record["path"] for record in records})
    for relative, source in files.items():
        destination = context / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    receipt = {
        "schema": "sparkring-mtp3-mesh-image-source/v1",
        "status": "research-only",
        "parent": base["operator_image"],
        "bundle_manifest_sha256": profile["canonical_bundle_manifest_sha256"],
        "files": {relative: sha256(context / relative) for relative in sorted(files)},
        "readiness_warmup": {"helper_path": "/opt/sparkring/bin/warmup_dflash.py",
                             "helper_sha256": sha256(context / "warmup_dflash.py"),
                             "temperature_environment": "SPARKRING_WARMUP_TEMPERATURE",
                             "default_temperature": 1.0},
        "scope": "Embedded transport bundle, compiled host marker, and readiness warmup helper with temperature one; target weights are not image contents.",
    }
    write_json(context / "receipts/source-receipt.json", receipt)
    return receipt


def run(argv: list[str]) -> str:
    completed = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(argv)}\n{completed.stderr}\n{completed.stdout}")
    return completed.stdout


def validate_parent(document: dict, expected: dict) -> None:
    if document.get("Id") != expected["image_id"]:
        raise ValueError("Parent image ID differs from the source receipt")
    if document.get("Architecture") != "arm64" or document.get("Os") != "linux":
        raise ValueError("Parent image must use linux/arm64")


def validate_context(context: Path) -> dict:
    receipt = read_json(context / "receipts/source-receipt.json")
    if receipt.get("schema") != "sparkring-mtp3-mesh-image-source/v1":
        raise ValueError("Image source receipt uses an unsupported schema")
    expected_files = set(receipt["files"]) | {"receipts/source-receipt.json"}
    observed_files = {path.relative_to(context).as_posix() for path in context.rglob("*") if path.is_file()}
    if observed_files != expected_files:
        raise ValueError("Build context contains missing or unmanifested files")
    for relative, expected in receipt["files"].items():
        item = context.joinpath(*safe_relative(relative).parts)
        if item.is_symlink() or not item.resolve().is_relative_to(context.resolve()):
            raise ValueError(f"Image source path escapes context: {relative}")
        if sha256(item) != expected:
            raise ValueError(f"Image source differs from its receipt: {relative}")
    profile = read_json(context / "receipts/profile-pins.json")
    verify_bundle(context / "bundle", profile["canonical_bundle_manifest_sha256"])
    return receipt


def build(context: Path, image: str, receipt_path: Path, engine: str, pull: bool) -> dict:
    source = validate_context(context)
    parent = source["parent"]
    if pull:
        subprocess.run([engine, "pull", "--platform", "linux/arm64", parent["reference"]], check=True)
    inspected = json.loads(run([engine, "image", "inspect", parent["image_id"]]))[0]
    validate_parent(inspected, parent)
    source_sha = sha256(context / "receipts/source-receipt.json")
    argv = [engine, "build", "--platform", "linux/arm64", "--network", "none", "--pull=false"]
    for name, value in {
        "PARENT_IMAGE": parent["reference"],
        "PARENT_IMAGE_ID": parent["image_id"],
        "BUNDLE_MANIFEST_SHA256": source["bundle_manifest_sha256"],
        "SOURCE_RECEIPT_SHA256": source_sha,
    }.items():
        argv.extend(["--build-arg", f"{name}={value}"])
    argv.extend(["--file", str(context / "Dockerfile"), "--tag", image, str(context)])
    subprocess.run(argv, check=True)
    command = [sys.executable, str(context / "verify_mesh_image.py"), "--image", image,
               "--engine", engine, "--expected-parent", parent["image_id"],
               "--expected-source", source_sha, "--expected-bundle", source["bundle_manifest_sha256"],
               "--output", str(receipt_path)]
    subprocess.run(command, check=True)
    return read_json(receipt_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="OFFLINE: construct a content-verified build directory")
    prepare_parser.add_argument("--bundle", type=Path, required=True)
    prepare_parser.add_argument("--context", type=Path, required=True)
    build_parser = sub.add_parser("build", help="MUTATES HOST: build and CPU-check an image; no GPU or fabric access")
    build_parser.add_argument("--context", type=Path, required=True)
    build_parser.add_argument("--image", required=True)
    build_parser.add_argument("--receipt", type=Path, required=True)
    build_parser.add_argument("--engine", default="docker")
    build_parser.add_argument("--pull-parent", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.bundle.resolve(), args.context.resolve())
        print(json.dumps({"context": str(args.context.resolve()), "bundle_manifest_sha256": result["bundle_manifest_sha256"]}, indent=2))
    else:
        build(args.context.resolve(), args.image, args.receipt.resolve(), args.engine, args.pull_parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
