#!/usr/bin/env python3
"""Prefer a strictly longer converged GPU prefix over a shorter shared lease."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


BEFORE_SHA256 = "3ec6f357fdc770212528ca3f5dfd48234186f972ec5657a514e8b841df61e48b"
AFTER_SHA256 = "0df01bf90bbe6ab1e6bc127ca7b15f6286ace944b430c994539d3d81049f2b4a"
TRANSFORMS = (
    (b"                did_prefix_cache_lookup = False\n",
     b"                did_prefix_cache_lookup = False\n                local_lease_alternative = None\n"),
    (b'''                    candidate = get_lease(request) if get_lease is not None else None
                    if candidate is not None:
                        lease_key, lease_tokens = candidate
''', b'''                    candidate = get_lease(request) if get_lease is not None else None
                    if candidate is not None:
                        lease_key, lease_tokens = candidate
                        if 0 < lease_tokens <= request.num_tokens:
                            # This lookup reconciles every KV group and applies
                            # speculative backoff before any blocks are adopted.
                            alternative = self.kv_cache_manager.get_computed_blocks(request)
                            if alternative[1] > min(lease_tokens, request.num_tokens - 1):
                                local_lease_alternative = (*alternative, False)
                                candidate = None
                                # Release this request's follower binding only;
                                # the verified lease and its other users remain.
                                rejected = getattr(
                                    self.connector, "shared_prefix_lease_rejected", None
                                )
                                if rejected is not None:
                                    rejected(request_id, lease_key)
                    if candidate is not None:
                        lease_key, lease_tokens = candidate
'''),
    (b'''                    ) = self._get_local_prefix_cache_hit(request)
''', b'''                    ) = (
                        local_lease_alternative
                        if local_lease_alternative is not None
                        else self._get_local_prefix_cache_hit(request)
                    )
'''),
)


def apply_patch(path: Path, *, check_only: bool = False) -> dict:
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest == AFTER_SHA256:
        return {"status": "already_patched", "source_sha256": digest, "changed": False}
    if digest != BEFORE_SHA256:
        raise RuntimeError(f"unsupported accounted MTP3 scheduler preimage: {digest}")
    patched = source
    for before, after in TRANSFORMS:
        if patched.count(before) != 1:
            raise RuntimeError("MTP3 local/lease selection anchor differs")
        patched = patched.replace(before, after, 1)
    if hashlib.sha256(patched).hexdigest() != AFTER_SHA256:
        raise RuntimeError("MTP3 local/lease selection postimage differs")
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
