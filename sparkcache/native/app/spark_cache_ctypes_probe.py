#!/usr/bin/env python3
"""Smoke-test the exported shared-library ABI from the canonical binding."""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path


NATIVE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NATIVE_ROOT / "python"))

from spark_cache_native import (  # noqa: E402
    AbiInfo,
    ArenaView,
    NativePlacementError,
    PlacementConfig,
    PlacementHandle,
    load_library,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("library", type=Path)
    arguments = parser.parse_args()
    library, info = load_library(arguments.library)
    print(
        "ctypes ABI PASS:"
        f" abi={info.abi_version}"
        f" cudart={info.cudart_version}"
        f" caps=0x{info.capability_flags:x}"
        f" abi_info={ctypes.sizeof(AbiInfo)}"
        f" arena_view={ctypes.sizeof(ArenaView)}"
    )

    invalid = PlacementConfig(abi_version=0xFFFFFFFF)
    try:
        PlacementHandle(library, invalid)
    except NativePlacementError as error:
        if "ABI" not in str(error):
            raise
        print(f"create-error accessor PASS: {error}")
    else:
        raise AssertionError("invalid ABI unexpectedly created a handle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
