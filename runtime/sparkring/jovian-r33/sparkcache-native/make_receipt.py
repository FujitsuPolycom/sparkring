#!/usr/bin/env python3
"""Write the content-addressed SparkCache native build receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
from pathlib import Path


def output(*arguments: str) -> str:
    return subprocess.check_output(arguments, text=True).strip()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def elf_facts(path: Path, header: Path, dynamic: Path, cuobjdump: Path) -> dict[str, object]:
    header_text = header.read_text(encoding="utf-8")
    dynamic_text = dynamic.read_text(encoding="utf-8")
    cuda_text = cuobjdump.read_text(encoding="utf-8")
    machine = re.search(r"^\s*Machine:\s*(.+)$", header_text, re.MULTILINE)
    needed = re.findall(r"Shared library: \[(.+?)\]", dynamic_text)
    sm_targets = sorted(set(re.findall(r"sm_[0-9]+", cuda_text)))
    if machine is None or "AArch64" not in machine.group(1):
        raise RuntimeError(f"{path.name} is not AArch64")
    if "sm_121" not in sm_targets:
        raise RuntimeError(f"{path.name} does not contain sm_121 device code")
    return {
        "filename": path.name,
        "sha256": digest(path),
        "size_bytes": path.stat().st_size,
        "elf_machine": machine.group(1).strip(),
        "needed": needed,
        "sm_targets": sm_targets,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    source = arguments.source
    out = arguments.out
    placement = elf_facts(
        out / "libspark_cache_placement.so",
        out / "placement-readelf-header.txt",
        out / "placement-readelf-dynamic.txt",
        out / "placement-cuobjdump.txt",
    )
    snapshot = elf_facts(
        out / "libspark_cache_snapshot.so",
        out / "snapshot-readelf-header.txt",
        out / "snapshot-readelf-dynamic.txt",
        out / "snapshot-cuobjdump.txt",
    )
    receipt = {
        "schema": "sparkring-r33-sparkcache-native-build/v1",
        "status": "qualified-native-build",
        "qualification_scope": (
            "single-host GB10 build, CPU contracts, public ctypes ABI, "
            "bounded GPU probes, hybrid page-copy C API, and compact snapshot matrix"
        ),
        "source": {
            "repository": "https://github.com/FujitsuPolycom/sparkcache.git",
            "commit": output("git", "-C", str(source), "rev-parse", "HEAD"),
            "tree": output("git", "-C", str(source), "rev-parse", "HEAD^{tree}"),
            "status_porcelain": output("git", "-C", str(source), "status", "--porcelain"),
            "archive": {
                "filename": "sparkcache-f220230a-source.tar.gz",
                "sha256": digest(out / "sparkcache-f220230a-source.tar.gz"),
            },
        },
        "build": {
            "foundation_reference": "local/sparkring:r33-arm64-foundation",
            "foundation_image_id": os.environ["SPARKCACHE_FOUNDATION_IMAGE_ID"],
            "host_arch": platform.machine(),
            "cuda_version": output("nvcc", "--version").splitlines()[-1],
            "cuda_architectures": ["121"],
            "build_type": "Release",
            "parallel_jobs": 2,
            "container_limits": {"cpus": 4, "memory": "12g", "shm_size": "2g"},
        },
        "artifacts": {"placement": placement, "snapshot": snapshot},
        "destinations": {
            "placement": "/opt/sparkring/sparkcache/lib/libspark_cache_placement.so",
            "snapshot": "/opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so",
        },
        "tests": {
            "ctest": "pass",
            "native_python": "pass",
            "placement_ctypes": "pass",
            "snapshot_ctypes_attested": "pass",
            "placement_gpu_probe": "pass",
            "hybrid_page_gpu_probe": "pass",
            "snapshot_gpu_probe": "pass",
            "page_copy_gpu_modes_1_2_3": (
                "not_exercised: exact f220230a benchmark self-rejects its "
                "257 MiB fixture because calculated explicit allocation is "
                "at least 650 MiB while its hard ceiling is 512 MiB"
            ),
            "hybrid_page_c_api_byte_correctness": "pass",
            "snapshot_compact_matrix": "pass",
        },
        "logs": sorted(path.name for path in out.glob("*.log")),
        "limits": [
            "No distributed serving or model-level capture/restore was exercised.",
            "The snapshot matrix used compact rank-0 geometry, not full GLM geometry.",
            "The build does not qualify TP4 topology, cache persistence, or failure recovery.",
            "The exact-source page-copy benchmark rejects its first fixture before CUDA copy; "
            "the smaller exact-source hybrid-page C API probe passed instead.",
        ],
    }
    if receipt["source"]["status_porcelain"]:
        raise RuntimeError("source checkout is dirty")
    destination = out / "build-receipt.json"
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (out / "SHA256SUMS").open("a", encoding="utf-8") as sums:
        sums.write(f"{digest(destination)}  {destination.name}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
