#!/usr/bin/env python3
"""Verify transport sources or prepare an offline source tree and overlay bundle."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

ROOT = Path(__file__).resolve().parent
MANIFEST = "sparkring-overlay-manifest.json"
LIBRARY = "libspark_transport_capi.so"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_name(name: str) -> str:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
        raise ValueError("Archive or manifest contains an unsafe path")
    return name


def verified_sources(root: Path = ROOT) -> tuple[dict, dict[str, bytes], dict[str, bytes]]:
    receipt = json.loads((root / "source-manifest.json").read_bytes())
    archive = (root / "native-source.tar.gz").read_bytes()
    if digest(archive) != receipt["native_source_archive_sha256"]:
        raise ValueError("Native source archive digest differs")
    sources = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as stream:
        for member in stream:
            name = checked_name(member.name)
            if not member.isfile() or name in sources:
                raise ValueError("Archive contains duplicate or non-regular entries")
            sources[name] = stream.extractfile(member).read()
    if {n: digest(b) for n, b in sources.items()} != receipt["source_files"]:
        raise ValueError("Native source inventory differs")
    bundle_root = root / "bundle-source"
    bundle = {}
    for path in bundle_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Bundle source contains a symbolic link")
        if path.is_file():
            bundle[checked_name(path.relative_to(bundle_root).as_posix())] = path.read_bytes()
    if {n: digest(b) for n, b in bundle.items()} != receipt["bundle_files"]:
        raise ValueError("Bundle source inventory differs")
    if digest(bundle[MANIFEST]) != receipt["reference_bundle_manifest_sha256"]:
        raise ValueError("Reference bundle manifest digest differs")
    document = json.loads(bundle[MANIFEST])
    listed = {row["path"]: row["sha256"] for row in document["files"]}
    actual = {n: digest(b) for n, b in bundle.items() if n != MANIFEST}
    actual[LIBRARY] = receipt["reference_native_sha256"]
    if len(listed) != len(document["files"]) or listed != actual:
        raise ValueError("Reference bundle content differs from its manifest")
    return receipt, sources, bundle


def write_tree(destination: Path, files: dict[str, bytes]) -> None:
    if destination.exists():
        raise ValueError("Output directory must not exist")
    destination.mkdir(parents=True)
    for name, data in sorted(files.items()):
        path = destination / checked_name(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def prepare_bundle(library: Path, expected_sha256: str, destination: Path,
                   root: Path = ROOT) -> dict:
    receipt, _, bundle = verified_sources(root)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Native library digest must be a full lowercase SHA256")
    data = library.read_bytes()
    if digest(data) != expected_sha256:
        raise ValueError("Native library digest differs")
    bundle[LIBRARY] = data
    matched = expected_sha256 == receipt["reference_native_sha256"]
    if not matched:
        document = json.loads(bundle[MANIFEST])
        for row in document["files"]:
            if row["path"] == LIBRARY:
                row["sha256"] = expected_sha256
        document["status"] = "research-only"
        bundle[MANIFEST] = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    write_tree(destination, bundle)
    return {"status": "implemented" if matched else "research-only",
            "reference_artifact_match": matched,
            "native_sha256": expected_sha256,
            "bundle_manifest_sha256": digest(bundle[MANIFEST]),
            "hardware_qualification_performed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    extract = commands.add_parser("extract")
    extract.add_argument("--output", type=Path, required=True)
    bundle = commands.add_parser("bundle")
    bundle.add_argument("--native-library", type=Path, required=True)
    bundle.add_argument("--native-sha256", required=True)
    bundle.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt, sources, _ = verified_sources()
    if args.command == "extract":
        write_tree(args.output, sources)
    elif args.command == "bundle":
        print(json.dumps(prepare_bundle(args.native_library, args.native_sha256, args.output), sort_keys=True))
        return
    print(json.dumps({"verified": True, "source_archive_sha256": receipt["native_source_archive_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
