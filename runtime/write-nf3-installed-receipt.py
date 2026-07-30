#!/usr/bin/env python3
"""Regenerate the NF3 installed-file receipt after a composed image overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_file(path: Path) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )


def build_installed_receipt(
    *,
    parent_receipt: Path,
    site_packages: Path,
    sparkring_root: Path,
    profile: str,
) -> dict[str, object]:
    parent_bytes = parent_receipt.read_bytes()
    parent = json.loads(parent_bytes)
    if parent.get("schema") != "sparkring-nf3-bootstrap-input/v1":
        raise ValueError("unexpected parent NF3 receipt schema")
    parent_files = parent.get("files")
    if not isinstance(parent_files, dict) or not parent_files:
        raise ValueError("parent NF3 receipt has no file inventory")
    if profile != "nvfp4-rope8":
        raise ValueError(f"unsupported composed NF3 profile: {profile}")

    installed_roots = {
        "b12x/": site_packages / "b12x",
        "overlay/": site_packages,
        "sparkring/": sparkring_root,
    }
    installed_files: dict[str, str] = {}
    for relative in sorted(parent_files):
        prefix = next(
            (
                candidate
                for candidate in installed_roots
                if relative.startswith(candidate)
            ),
            None,
        )
        if prefix is None:
            raise ValueError(f"unsupported parent receipt path: {relative}")
        path = installed_roots[prefix] / relative.removeprefix(prefix)
        if not _stable_file(path):
            raise ValueError(f"installed receipt file is missing/unstable: {path}")
        installed_files[relative] = _sha256(path)

    # The MLA composition can add files that were absent from the NF3 parent's
    # B12X source tree. Inventory the complete stable B12X installation so the
    # final receipt covers additions as well as replacements.
    b12x_root = installed_roots["b12x/"]
    for path in sorted(b12x_root.rglob("*")):
        if _stable_file(path):
            relative = "b12x/" + path.relative_to(b12x_root).as_posix()
            installed_files[relative] = _sha256(path)

    receipt = dict(parent)
    receipt.update(
        {
            "schema": "sparkring-nf3-bootstrap-input/v1",
            "profile": profile,
            "parent_input_receipt_sha256": hashlib.sha256(
                parent_bytes
            ).hexdigest(),
            "files": dict(sorted(installed_files.items())),
        }
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-receipt", required=True, type=Path)
    parser.add_argument(
        "--site-packages",
        type=Path,
        default=Path("/opt/venv/lib/python3.12/site-packages"),
    )
    parser.add_argument(
        "--sparkring-root",
        type=Path,
        default=Path("/opt/spark-vllm"),
    )
    parser.add_argument("--profile", default="nvfp4-rope8")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = build_installed_receipt(
        parent_receipt=args.parent_receipt,
        site_packages=args.site_packages,
        sparkring_root=args.sparkring_root,
        profile=args.profile,
    )
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
