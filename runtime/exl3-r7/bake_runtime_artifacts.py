#!/usr/bin/env python3
"""Apply hash-bound compatibility edits to public R7 runtime inputs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


class ArtifactError(RuntimeError):
    """A runtime input did not match the published compatibility contract."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_exact(
    path: Path,
    *,
    input_sha256: str,
    output_sha256: str,
    replacements: tuple[tuple[bytes, bytes], ...],
    expected_counts: tuple[int, ...] | None = None,
) -> None:
    observed = sha256(path)
    if observed != input_sha256:
        raise ArtifactError(
            f"{path}: input SHA-256 mismatch: expected {input_sha256}, got {observed}"
        )
    content = path.read_bytes()
    counts = expected_counts or (1,) * len(replacements)
    if len(counts) != len(replacements):
        raise ArtifactError(f"{path}: replacement-count contract is invalid")
    for (before, after), expected_count in zip(replacements, counts, strict=True):
        count = content.count(before)
        if count != expected_count:
            raise ArtifactError(
                f"{path}: expected {expected_count} compatibility edit preimages, "
                f"found {count}"
            )
        content = content.replace(before, after)
    path.write_bytes(content)
    observed = sha256(path)
    if observed != output_sha256:
        raise ArtifactError(
            f"{path}: output SHA-256 mismatch: expected {output_sha256}, got {observed}"
        )


def bake_vllm(root: Path) -> None:
    weight_utils = root / "vllm/model_executor/model_loader/weight_utils.py"
    replace_exact(
        weight_utils,
        input_sha256="7194c83a3043c76a54f7b8ebf32a3728084583e737e73c6e34ffcdfb25aa92c4",
        output_sha256="da5e6c3429293870d0de611183818fa57c0e9e0ad896784bc739c8a812343102",
        replacements=((
            b"process_group = world_group.device_group if world_group.world_size > 1 else None",
            b"process_group = None",
        ),),
    )
    expected = {
        root / "vllm/model_executor/layers/quantization/exl3.py":
            "8e0051faf9b8bac9eefd6f38a5f0133a30bca4c0b5ab41962537e2f13cf968f4",
        root / "vllm/v1/worker/gpu/cudagraph_utils.py":
            "ef03d64297ed2d1a5161847b48a435bf8ae5feda7a5b81b668d00ae9a1d65a2a",
    }
    for path, required_hash in expected.items():
        observed = sha256(path)
        if observed != required_hash:
            raise ArtifactError(
                f"{path}: SHA-256 mismatch: expected {required_hash}, got {observed}"
            )


def bake_quack(root: Path) -> None:
    replace_exact(
        root / "layout_utils.py",
        input_sha256="0772f590e47c0cb00341a4df7fe111a9b4558d3ce3eb4010db9c07389fce95b5",
        output_sha256="3199dc3f55f346183e3d284f6da98f4394eaf14f28b7616d147e6e49ec896194",
        replacements=(
            (b"thr_mma: cute.core.ThrMma", b"thr_mma: cute.ThrMma"),
            (b"thr_copy: cute.core.ThrCopy", b"thr_copy: cute.ThrCopy"),
        ),
        expected_counts=(2, 2),
    )
    replace_exact(
        root / "copy_utils.py",
        input_sha256="80947161e20f1bad6c8671164f628faa0c8d2818ad7dc9824da908e197c6325a",
        output_sha256="2ce88b0d7ee9afe025e52c02fcb32e772a429f1ee626b59546ab8b61d7a37929",
        replacements=((b"thr_copy: cute.core.ThrCopy", b"thr_copy: cute.ThrCopy"),),
        expected_counts=(2,),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("vllm", "quack"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.component == "vllm":
        bake_vllm(args.root.resolve())
    else:
        bake_quack(args.root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
