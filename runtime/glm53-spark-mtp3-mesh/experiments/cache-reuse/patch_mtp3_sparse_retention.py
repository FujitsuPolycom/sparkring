#!/usr/bin/env python3
"""Pair speculative replay checkpoint materialization with sparse retention.

Attested for the installed scheduler after lease accounting/local preference.
Serving qualification targets native MTP3 with 512-token hash/Mamba pages and
DCP4; no GPU or model execution is performed by this source transformer.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


SCHEDULER_BEFORE = "0df01bf90bbe6ab1e6bc127ca7b15f6286ace944b430c994539d3d81049f2b4a"
SCHEDULER_AFTER = "5c0bd785d2d17dce39cdf867a55b5327487070d315851f3c67d9c2cf7a1d4c49"
MANAGER_BEFORE = "10846c4994e7860deab8b42c8bcd3315ddc96d14a478d4012c398418cc17a04c"
MANAGER_AFTER = "d2e35b012e0cf45ab3771f545c35ca48f2a5858549c574a352975607369124e2"
SCHEDULER_TRANSFORMS = (
    (b'''        if self.use_eagle:
            last_cache_position = max(last_cache_position - block_size, 0)
''', b'''        if self.use_eagle:
            # Lookup excludes the final prompt token, then drops its proof
            # block. The corresponding recurrent state needs its own stop.
            last_cache_position = max(
                (request.num_tokens - 1) // block_size * block_size - block_size, 0
            )
'''),
    (b'''        if use_internal_checkpoint:
            last_cache_position = 0
''', b'''        if use_internal_checkpoint and not self.use_eagle:
            # The internal checkpoint covers the final aligned state only;
            # speculative replay may require the preceding state as well.
            last_cache_position = 0
'''),
)
MANAGER_TRANSFORMS = (
    (b'''            if start_block <= boundary_block < end_block:
                mask[boundary_block - start_block] = True

        return mask

    def remove_skipped_blocks(
''', b'''            if start_block <= boundary_block < end_block:
                mask[boundary_block - start_block] = True

            if use_eagle:
                # Retain the predecessor when materialized, alongside the
                # scheduler-aligned fallback. Unmaterialized shared-junction
                # slots remain null and cache_full_blocks skips them.
                predecessor_block = boundary_tokens // block_size - 2
                if start_block <= predecessor_block < end_block:
                    mask[predecessor_block - start_block] = True

        return mask

    def remove_skipped_blocks(
'''),
)


def transform(source, expected_before, expected_after, transforms):
    digest = hashlib.sha256(source).hexdigest()
    if digest == expected_after:
        return source
    if digest != expected_before:
        raise RuntimeError(f"unsupported MTP3 checkpoint preimage: {digest}")
    for before, after in transforms:
        if source.count(before) != 1:
            raise RuntimeError("MTP3 checkpoint source anchor differs")
        source = source.replace(before, after, 1)
    if hashlib.sha256(source).hexdigest() != expected_after:
        raise RuntimeError("MTP3 checkpoint postimage differs")
    ast.parse(source.decode("utf-8"))
    return source


def apply_patch(root: Path, *, check_only=False):
    files = (
        (root / "v1/core/sched/scheduler.py", SCHEDULER_BEFORE, SCHEDULER_AFTER, SCHEDULER_TRANSFORMS),
        (root / "v1/core/single_type_kv_cache_manager.py", MANAGER_BEFORE, MANAGER_AFTER, MANAGER_TRANSFORMS),
    )
    prepared = []
    for path, before, after, transforms in files:
        original = path.read_bytes()
        patched = transform(original, before, after, transforms)
        prepared.append((path, original, patched))
    # Validate both preimages before changing either source file.
    if not check_only:
        for path, original, patched in prepared:
            if original != patched:
                path.write_bytes(patched)
    return {str(path.relative_to(root)): dict(before_sha256=hashlib.sha256(original).hexdigest(),
             after_sha256=hashlib.sha256(patched).hexdigest(), changed=original != patched and not check_only)
            for path, original, patched in prepared}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vllm_root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(apply_patch(args.vllm_root, check_only=args.check), sort_keys=True))
