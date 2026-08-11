#!/usr/bin/env python3
"""Make the composed r34 vLLM build preserve its requested SM121a target."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = (
    'set(CUDA_SUPPORTED_ARCHS '
    '"7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0")'
)
NEW = (
    'set(CUDA_SUPPORTED_ARCHS '
    '"7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0a;12.1a")'
)


def patch(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if source.count(OLD) != 1:
        raise RuntimeError(
            "r34 CUDA_SUPPORTED_ARCHS preimage is absent or ambiguous; "
            "refusing an unverified SM121 build patch"
        )
    updated = source.replace(OLD, NEW)
    if updated.count(NEW) != 1 or OLD in updated:
        raise RuntimeError("SM121 CUDA_SUPPORTED_ARCHS postimage verification failed")
    path.write_text(updated, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmake_lists", type=Path)
    args = parser.parse_args()
    patch(args.cmake_lists)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
