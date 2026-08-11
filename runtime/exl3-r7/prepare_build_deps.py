#!/usr/bin/env python3
"""Prepare immutable local CUTLASS and Triton-kernel sources for R7 builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path


SCHEMA = "sparkring-r7-build-deps/v1"
BUNDLED_LICENSE_NAME = "UPSTREAM_LICENSE.txt"
SOURCES = {
    "cutlass": {
        "repository": "https://github.com/NVIDIA/cutlass.git",
        "commit": "da5e086dab31d63815acafdac9a9c5893b1c69e2",
        "subdirectory": ".",
        "license_path": "LICENSE.txt",
    },
    "triton_kernels": {
        "repository": "https://github.com/triton-lang/triton.git",
        "commit": "0add68262ab0a2e33b84524346cb27cbb2787356",
        "subdirectory": "python/triton_kernels/triton_kernels",
        "license_path": "LICENSE",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(path: Path) -> list[dict]:
    rows = []
    for item in sorted(path.rglob("*")):
        if item.is_file() and ".git" not in item.relative_to(path).parts:
            rows.append(
                {
                    "path": item.relative_to(path).as_posix(),
                    "size": item.stat().st_size,
                    "sha256": sha256(item),
                }
            )
    return rows


def verify(output: Path) -> dict:
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != SCHEMA:
        raise RuntimeError("wrong R7 build-dependency receipt schema")
    if receipt.get("sources") != SOURCES:
        raise RuntimeError("R7 build-dependency source pins do not match")
    for name in SOURCES:
        if not (output / name / BUNDLED_LICENSE_NAME).is_file():
            raise RuntimeError(f"R7 build-dependency license is missing: {name}")
        observed = inventory(output / name)
        if observed != receipt.get("inventories", {}).get(name):
            raise RuntimeError(f"R7 build-dependency inventory mismatch: {name}")
    return receipt


def run(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def remove_tree(path: Path) -> None:
    """Remove a private staging tree, including read-only Git pack files."""

    def make_writable(function, target, _error):
        os.chmod(target, stat.S_IWRITE)
        function(target)

    shutil.rmtree(path, onexc=make_writable)


def checkout(source: dict, destination: Path) -> None:
    run("git", "clone", "--filter=blob:none", "--no-checkout", source["repository"], str(destination))
    run("git", "-C", str(destination), "config", "core.autocrlf", "false")
    run("git", "-C", str(destination), "fetch", "--depth", "1", "origin", source["commit"])
    run("git", "-C", str(destination), "checkout", "--detach", source["commit"])
    if run("git", "-C", str(destination), "rev-parse", "HEAD") != source["commit"]:
        raise RuntimeError(f"checkout did not resolve exact commit {source['commit']}")


def prepare(output: Path) -> dict:
    try:
        return verify(output)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        pass
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        inventories = {}
        for name, source in SOURCES.items():
            clone = staging / f".{name}-checkout"
            checkout(source, clone)
            selected = clone / source["subdirectory"]
            if not selected.is_dir():
                raise RuntimeError(f"missing expected source directory for {name}: {source['subdirectory']}")
            shutil.copytree(selected, staging / name, ignore=shutil.ignore_patterns(".git"))
            license_path = clone / source["license_path"]
            if not license_path.is_file():
                raise RuntimeError(
                    f"missing expected upstream license for {name}: "
                    f"{source['license_path']}"
                )
            shutil.copy2(license_path, staging / name / BUNDLED_LICENSE_NAME)
            inventories[name] = inventory(staging / name)
            remove_tree(clone)
        receipt = {"schema": SCHEMA, "sources": SOURCES, "inventories": inventories}
        (staging / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verify(staging)
        if output.exists():
            replaced = output.with_name(f".{output.name}.replaced-{os.getpid()}")
            os.replace(output, replaced)
            os.replace(staging, output)
            remove_tree(replaced)
        else:
            os.replace(staging, output)
        return verify(output)
    finally:
        if staging.exists():
            remove_tree(staging)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    receipt = verify(args.output.resolve()) if args.verify else prepare(args.output.resolve())
    print(json.dumps({"status": "pass", "schema": receipt["schema"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
