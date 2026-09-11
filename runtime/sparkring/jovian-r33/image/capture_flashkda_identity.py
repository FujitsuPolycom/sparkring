#!/usr/bin/env python3
"""Capture the patched FlashKDA source result used by the vLLM native build."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


EXPECTED_HEAD = "3b225bf26bb8e218928a1fe14751cb48cf31d11b"
EXPECTED_BASE_TREE = "e8cf226562e56d0817462de76a614d32a83409ef"
EXPECTED_RESULT_TREE = "3668743d4c392da0270b6c36cbfeb3545571a613"
EXPECTED_DIFF_SHA256 = "a21c2a2ee49e17c4f356a1b40ef2d30ac8f826f14d9f054eb6622e94ee634796"
EXPECTED_CHANGED = [
    "README.md", "csrc/flash_kda.cpp", "csrc/flash_kda.h", "csrc/fwd.h",
    "csrc/smxx/fwd_kernel2.cuh", "csrc/smxx/fwd_launch.cu", "csrc/torch_api.cpp",
    "flash_kda/__init__.py", "tests/test_vsplit.py",
]


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-c", f"safe.directory={root}", "-C", str(root), *args])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    head = git(source, "rev-parse", "HEAD").decode().strip()
    base_tree = git(source, "rev-parse", "HEAD^{tree}").decode().strip()
    diff = git(source, "diff", "--binary")
    changed = git(source, "diff", "--name-only").decode().splitlines()
    if (head != EXPECTED_HEAD or base_tree != EXPECTED_BASE_TREE
            or hashlib.sha256(diff).hexdigest() != EXPECTED_DIFF_SHA256
            or changed != EXPECTED_CHANGED):
        raise RuntimeError("FlashKDA materialized source differs from the locked native build input")
    result = {
        "schema": "sparkring-r33-flashkda-source-identity/v1",
        "status": "patched-source-materialization-verified",
        "head": head,
        "base_tree": base_tree,
        "result_tree": EXPECTED_RESULT_TREE,
        "diff_sha256": EXPECTED_DIFF_SHA256,
        "changed_files": changed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
