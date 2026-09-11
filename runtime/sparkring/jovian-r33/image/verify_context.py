#!/usr/bin/env python3
"""Verify every byte in a finalized candidate build context."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, required=True)
    args = parser.parse_args()
    context = args.context.resolve()
    lock_path = context / "source-lock.json"
    lock = json.loads(lock_path.read_text())
    if lock.get("schema") != "sparkring-r33-candidate-source-lock/v1":
        raise RuntimeError("unsupported source lock")
    actual = {
        path.relative_to(context).as_posix(): digest(path)
        for path in context.rglob("*")
        if path.is_file() and path != lock_path
    }
    if actual != lock["context_files"]:
        bad = sorted(name for name in set(actual) | set(lock["context_files"]) if actual.get(name) != lock["context_files"].get(name))
        raise RuntimeError(f"context closure differs: {bad[:20]}")
    if lock["inputs"]["nccl-2.31.2-sparkring-routing"]["sha256"] != lock["identities"]["nccl_sha256"]:
        raise RuntimeError("NCCL input and identity differ")
    if lock["inputs"]["sircl"]["sha256"] != lock["identities"]["sircl_sha256"]:
        raise RuntimeError("SIRCL input and identity differ")
    print(json.dumps({
        "schema": "sparkring-r33-candidate-context-verification/v1",
        "status": "verified",
        "source_lock_sha256": digest(lock_path),
        "context_files": len(actual),
        "wheelhouse_files": len(lock["wheelhouse"]),
        "python_closure_distributions": len(lock["python_closure"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
