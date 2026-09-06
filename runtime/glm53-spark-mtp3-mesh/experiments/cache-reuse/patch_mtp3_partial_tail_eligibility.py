#!/usr/bin/env python3
"""Avoid recurrent partial-tail stops created only by DCP attention geometry."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


BEFORE_SHA256 = "5c0bd785d2d17dce39cdf867a55b5327487070d315851f3c67d9c2cf7a1d4c49"
AFTER_SHA256 = "75efa57e7ff5a77c76714b85e2e4d8e1d7f456d9a9eec6c67ebb11ca382942f9"
BEFORE = b'''        # A finer prefix_match_unit is configured: a mamba partial tail entry
        # can only be registered by a step ending exactly at the prompt's last
        # hash boundary, so the split adds that stop.
        self.mamba_partial_cache_hit = (
            self.need_mamba_block_aligned_split
            and self.hash_block_size < self.block_size
            and self.kv_cache_manager.coordinator.enable_partial_hash_hits
        )
'''
AFTER = b'''        # An interior recurrent-page hash needs an explicit tail stop.
        # DCP can enlarge attention's scheduling unit without making the
        # recurrent page larger than a hash, so inspect Mamba specs directly.
        self.mamba_partial_cache_hit = (
            self.need_mamba_block_aligned_split
            and any(
                isinstance(group.kv_cache_spec, MambaSpec)
                and self.hash_block_size < group.kv_cache_spec.block_size
                for group in kv_cache_config.kv_cache_groups
            )
            and self.kv_cache_manager.coordinator.enable_partial_hash_hits
        )
'''


def apply_patch(path: Path, *, check_only=False):
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest == AFTER_SHA256:
        return {"status": "already_patched", "source_sha256": digest, "changed": False}
    if digest != BEFORE_SHA256:
        raise RuntimeError(f"unsupported checkpoint-patched MTP3 scheduler: {digest}")
    if source.count(BEFORE) != 1:
        raise RuntimeError("MTP3 partial-tail eligibility anchor differs")
    patched = source.replace(BEFORE, AFTER, 1)
    if hashlib.sha256(patched).hexdigest() != AFTER_SHA256:
        raise RuntimeError("MTP3 partial-tail eligibility postimage differs")
    ast.parse(patched.decode("utf-8"))
    if not check_only:
        path.write_bytes(patched)
    return {"status": "checked" if check_only else "patched", "changed": not check_only,
            "before_sha256": digest, "after_sha256": AFTER_SHA256}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(apply_patch(args.path, check_only=args.check), sort_keys=True))
