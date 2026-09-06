#!/usr/bin/env python3
"""Patch the attested MTP3 B12X histogram barrier, preserving its source bytes."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


BEFORE_SHA256 = "893fbcade135b7e1d146b8fb6530cde0650be515f69bf9a17ced0a9c61a141e2"
AFTER_SHA256 = "b43a4a2802c7dfc4a049bbb5751fc7e7688b05cb06d4ece716ab7a1d91d23d2a"
_BEFORE = '''    """Grid barrier over the group's CTAs on the arrival counter; returns the next phase."""
    arrival_ptr = _fused_state_ptr(state, group_id, Int32(_FUSED_STATE_ARRIVAL))
'''.replace("\n", "\r\n").encode()
_AFTER = '''    """Grid barrier over the group's CTAs on the arrival counter; returns the next phase."""
    # Every publishing warp must finish before the leader releases this CTA's
    # arrival; otherwise peers can scan partial histograms and diverge in rounds.
    cute.arch.sync_threads()
    arrival_ptr = _fused_state_ptr(state, group_id, Int32(_FUSED_STATE_ARRIVAL))
'''.replace("\n", "\r\n").encode()
_OLD_CACHE = b'"attention.indexer.fused_indexer", 1, cache_key, labels=labels'
_NEW_CACHE = b'"attention.indexer.fused_indexer", 2, cache_key, labels=labels'


def apply_patch(path: Path, *, check_only: bool = False) -> dict:
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest == AFTER_SHA256:
        return {"status": "already_patched", "source_sha256": digest, "changed": False}
    if digest != BEFORE_SHA256:
        raise RuntimeError(f"unsupported MTP3 B12X source preimage: {digest}")
    if source.count(_BEFORE) != 1 or source.count(_OLD_CACHE) != 1:
        raise RuntimeError("attested barrier or compile revision anchor differs")
    patched = source.replace(_BEFORE, _AFTER, 1).replace(_OLD_CACHE, _NEW_CACHE, 1)
    if hashlib.sha256(patched).hexdigest() != AFTER_SHA256:
        raise RuntimeError("MTP3 B12X transform differs from its expected postimage")
    ast.parse(patched.decode("utf-8"), filename=str(path))
    if not check_only:
        path.write_bytes(patched)
    return {"status": "checked" if check_only else "patched", "before_sha256": digest,
            "after_sha256": AFTER_SHA256, "changed": not check_only, "compile_revision": 2}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--check", action="store_true", help="validate the transform without writing")
    args = parser.parse_args()
    print(json.dumps(apply_patch(args.path, check_only=args.check), sort_keys=True))
