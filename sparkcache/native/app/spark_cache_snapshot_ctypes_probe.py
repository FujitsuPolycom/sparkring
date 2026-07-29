#!/usr/bin/env python3
"""Attest and smoke-test the exported SparkCache snapshot ABI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


NATIVE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NATIVE_ROOT / "python"))

from spark_cache_snapshot_native import load_library  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("library", type=Path)
    parser.add_argument("--sha256", required=True)
    arguments = parser.parse_args()

    library, info = load_library(
        arguments.library,
        expected_sha256=arguments.sha256,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "abi_version": int(info.abi_version),
                "cudart_version": int(info.cudart_version),
                "min_slots": int(info.min_slots),
                "max_slots": int(info.max_slots),
                "max_record_kinds": int(info.max_record_kinds),
                "capability_flags": int(info.capability_flags),
            },
            sort_keys=True,
        )
    )
    owner = getattr(library, "_spark_cache_snapshot_attested_file", None)
    if owner is None or owner.closed:
        raise RuntimeError("attested library descriptor is not retained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
