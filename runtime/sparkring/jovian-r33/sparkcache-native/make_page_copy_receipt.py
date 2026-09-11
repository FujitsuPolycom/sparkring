#!/usr/bin/env python3
"""Create a separate receipt for the test-only page-copy ceiling correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    out = arguments.out
    modes: dict[str, object] = {}
    maximum = 0
    for mode in (1, 2, 3):
        rows = [json.loads(line) for line in (out / f"mode-{mode}.jsonl").read_text().splitlines()]
        if len(rows) != 6 or not all(row.get("byte_equal") is True for row in rows):
            raise RuntimeError(f"mode {mode} did not pass every fixture")
        maximum = max(maximum, *(int(row["explicit_allocation_bytes"]) for row in rows))
        modes[str(mode)] = {
            "status": "pass",
            "fixtures": len(rows),
            "max_explicit_allocation_bytes": max(
                int(row["explicit_allocation_bytes"]) for row in rows
            ),
            "log": f"mode-{mode}.jsonl",
            "cuda_memory_before": json.loads(
                (out / f"mode-{mode}-cuda-before.json").read_text()
            ),
            "cuda_memory_after": json.loads(
                (out / f"mode-{mode}-cuda-after.json").read_text()
            ),
        }
    receipt = {
        "schema": "sparkring-r33-sparkcache-page-copy-test/v1",
        "status": "qualified-test-only-harness",
        "source": {
            "commit": "f220230a5a85b94af8a296187241b6aacc3ed724",
            "tree": "86ef46de45dd0f4ed776b86109a30df6f83db557",
        },
        "library": {
            "filename": "../libspark_cache_placement.so",
            "sha256_before_and_after": "d89c9fdae8dc99ae3f7a151cc3dd9e92fdc8fd0b994069fc263027fd4d056c93",
        },
        "test_only_patch": {
            "filename": "page-copy-benchmark-1g-ceiling-64mib-slabs.patch",
            "sha256": digest(out / "page-copy-benchmark-1g-ceiling-64mib-slabs.patch"),
            "change": (
                "benchmark ceiling 512 MiB to 1024 MiB; represent the 257 MiB "
                "grid-stride payload as five spans that reuse the ABI-valid 64 MiB arena"
            ),
            "library_sources_changed": False,
        },
        "ceiling_math": {
            "formula": "2*64MiB arenas + 2*pool_bytes + 8MiB readback + staged_mode*2*64MiB",
            "maximum_observed_bytes": maximum,
            "maximum_observed_mib": maximum / (1024 * 1024),
            "test_ceiling_bytes": 1024 * 1024 * 1024,
            "sufficient": maximum < 1024 * 1024 * 1024,
        },
        "modes": modes,
        "result": "18 of 18 fixture/mode combinations passed byte equality",
        "limits": [
            "The patch changes only the test harness and is not part of either delivered library.",
            "This is a single-host GB10 microbenchmark, not distributed TP4 serving evidence.",
        ],
    }
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (out / "SHA256SUMS").open("w", encoding="utf-8") as sums:
        for path in sorted(out.iterdir()):
            if path.is_file() and path.name not in {"SHA256SUMS"}:
                sums.write(f"{digest(path)}  {path.name}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
