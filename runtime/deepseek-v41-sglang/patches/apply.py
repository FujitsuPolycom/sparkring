#!/usr/bin/env python3
"""Install pinned SGLang overlays after checking every affected source byte."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def source_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError(f"Invalid overlay path: {relative}")
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()) or target.is_symlink():
        raise ValueError(f"Overlay path leaves the source tree: {relative}")
    return target


def verify_files(root: Path, files: list[dict], field: str) -> None:
    for item in files:
        path = source_path(root, item["path"])
        actual = sha256(path)
        if (path.exists() and not path.is_file()) or actual != item[field]:
            raise ValueError(
                f"SGLang source mismatch for {item['path']}: "
                f"expected {item[field]}, found {actual}"
            )


def install(source_root: Path, patch_root: Path = HERE, *, check: bool = False) -> dict:
    source_root = source_root.resolve(strict=True)
    manifest_path = patch_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported SGLang overlay manifest schema")
    files = manifest["files"]
    if len({item["path"] for item in files}) != len(files):
        raise ValueError("Duplicate SGLang source paths in overlay manifest")
    for patch in manifest["patches"]:
        path = source_path(patch_root, patch["path"])
        if sha256(path) != patch["sha256"]:
            raise ValueError(f"SGLang patch hash mismatch: {patch['path']}")
    already_installed = all(
        sha256(source_path(source_root, item["path"])) == item["output_sha256"]
        for item in files
    )
    if check or already_installed:
        verify_files(source_root, files, "output_sha256")
    else:
        verify_files(source_root, files, "input_sha256")
        # Validate both overlays in a small staging tree before changing any
        # source file. An unexpected source or patch fails without partial edits.
        with tempfile.TemporaryDirectory(prefix="sglang-overlay-") as directory:
            stage = Path(directory)
            for item in files:
                if item["input_sha256"] is not None:
                    target = source_path(stage, item["path"])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_root / item["path"], target)
            for index, patch in enumerate(manifest["patches"]):
                subprocess.run(
                    ["git", "-c", "core.autocrlf=false", "apply",
                     str((patch_root / patch["path"]).resolve())],
                    cwd=stage, check=True, capture_output=True, text=True,
                )
                verify_files(stage, files, "verify_overlay_sha256" if index == 0 else "output_sha256")
            for item in files:
                target = source_path(source_root, item["path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(stage / item["path"], target)
        verify_files(source_root, files, "output_sha256")
    return {
        "schema_version": 1,
        "source_revision": manifest["source_revision"],
        "manifest_sha256": sha256(manifest_path),
        "patches": manifest["patches"],
        "files": [{"path": item["path"], "sha256": item["output_sha256"]} for item in files],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Verify installed overlay bytes without writing")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    receipt = install(args.source_root, check=args.check)
    rendered = json.dumps(receipt, indent=2) + "\n"
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
