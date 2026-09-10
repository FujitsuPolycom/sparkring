#!/usr/bin/env python3
"""Capture complete current Git and recursive-submodule identities after native builds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


SOURCES = {
    "vllm": ("sources/vllm-sparkring", "ae89131442359dc332d9c46009be3c1f8cdee0b4"),
    "b12x": ("sources/b12x", "d95137245253d5c145e4b0700d677ce13b87ebea"),
    "flashinfer": ("sources/flashinfer", "803c4664f4771ddc418f20a57f752469a237a825"),
    "lmcache": ("sources/lmcache", "29bc5a2efde737c436b04499eb62cd1776cebeec"),
    "instanttensor": ("sources/instanttensor", "49b4010afc1cae0441e71fe0b0bffc24fa05e932"),
    "xgrammar": ("sources/xgrammar", "2ea71da4ccb997a06928c9fb69b99f330da56697"),
    "sparkcache": ("sources/sparkcache", "f220230a5a85b94af8a296187241b6aacc3ed724"),
    "torchvision": ("sources/vision", "8fb87713a24951e639c494b0f2a8a81b5f8e33a6"),
    "torchaudio": ("sources/audio", "34c52a67e8941bbd8e6adaca0eb0b9eabec11d78"),
    "torch": ("sources/pytorch", "cf30153c4c131c8164ee7798e5022d810682e2cb"),
    "cutlass": ("sources/cutlass", "e6233cbac5d7c7a865c19c91cd684ceece19513c"),
}


def git(path: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(path), *arguments])


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", type=Path, default=Path("/var/tmp/sparkring-r33-20260910"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {}
    for name, (relative, expected_head) in SOURCES.items():
        path = args.build_root / relative
        head = git(path, "rev-parse", "HEAD").decode().strip()
        if head != expected_head:
            raise RuntimeError(f"{name} HEAD differs: {head}")
        tree = git(path, "write-tree").decode().strip()
        status = git(path, "status", "--porcelain=v1", "--untracked-files=all")
        submodules = git(path, "submodule", "status", "--recursive")
        cached = git(path, "diff", "--cached", "--binary")
        unstaged = git(path, "diff", "--binary")
        records[name] = {
            "path": relative,
            "head": head,
            "index_tree": tree,
            "status_sha256": hash_bytes(status),
            "status_lines": len(status.splitlines()),
            "cached_diff_sha256": hash_bytes(cached),
            "unstaged_diff_sha256": hash_bytes(unstaged),
            "recursive_submodules_sha256": hash_bytes(submodules),
            "recursive_submodules": submodules.decode().splitlines(),
        }
    result = {
        "schema": "sparkring-r33-post-build-source-identities/v1",
        "status": "captured-after-component-builds-before-image-assembly",
        "sources": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sources": len(records)}, indent=2))


if __name__ == "__main__":
    main()
