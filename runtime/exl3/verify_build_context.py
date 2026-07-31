#!/usr/bin/env python3
"""Verify every byte in a generated EXL3 build context."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, required=True)
    args = parser.parse_args()
    root = args.context.resolve()
    manifest = json.loads((root / "context-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "sparkring-public-exl3-build-context/v1":
        raise RuntimeError("wrong EXL3 build-context schema")
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("EXL3 build-context file map is empty")
    observed = {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "context-manifest.json"
    }
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        changed = sorted(name for name in set(expected) & set(observed) if expected[name] != observed[name])
        raise RuntimeError(f"EXL3 build context drift: missing={missing}, extra={extra}, changed={changed}")
    print(json.dumps({"schema": manifest["schema"], "profile_id": manifest["profile_id"], "status": "pass"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
