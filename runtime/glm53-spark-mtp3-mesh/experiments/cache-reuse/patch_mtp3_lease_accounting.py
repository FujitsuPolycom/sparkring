#!/usr/bin/env python3
"""Account for resident GPU lease reuse in the pinned MTP3 prefill statistics."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


BEFORE_SHA256 = "122d9722b48f8cec267d2136a18c184b8901a5a0a46187eab7030d6227f1983d"
AFTER_SHA256 = "3ec6f357fdc770212528ca3f5dfd48234186f972ec5657a514e8b841df61e48b"
_BEFORE = b'''                            request.num_computed_tokens = min(
                                attached_tokens, request.num_tokens - 1
                            )
                            attached = getattr(
'''
_AFTER = b'''                            request.num_computed_tokens = min(
                                attached_tokens, request.num_tokens - 1
                            )
                            # Lease attachment skips hash lookup but still reuses
                            # resident GPU state. Include it in prefill/API totals
                            # without reporting a fresh external KV transfer.
                            if request.prefill_stats and request.num_preemptions <= 0:
                                request.prefill_stats.set(
                                    num_prompt_tokens=request.num_prompt_tokens,
                                    num_local_cached_tokens=request.num_computed_tokens,
                                    num_external_cached_tokens=0,
                                )
                            attached = getattr(
'''


def apply_patch(path: Path, *, check_only: bool = False) -> dict:
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest == AFTER_SHA256:
        return {"status": "already_patched", "source_sha256": digest, "changed": False}
    if digest != BEFORE_SHA256:
        raise RuntimeError(f"unsupported MTP3 scheduler preimage: {digest}")
    if source.count(_BEFORE) != 1:
        raise RuntimeError("MTP3 lease attachment anchor differs")
    patched = source.replace(_BEFORE, _AFTER, 1)
    if hashlib.sha256(patched).hexdigest() != AFTER_SHA256:
        raise RuntimeError("MTP3 lease accounting postimage differs")
    ast.parse(patched.decode("utf-8"), filename=str(path))
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
