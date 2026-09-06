#!/usr/bin/env python3
"""Apply the GB10-tested histogram publication barrier to pinned B12X source."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


BEFORE_SHA256 = "d3ec6274e142a4e7d1062ea6d2d99b97db0a02e92bb976c6570ae990b836b18d"
AFTER_SHA256 = "49f6fd916fd1ccf94311ee99427551edbd0dc3a5de23aeeb426418370f76f66d"
_BEFORE = '''    """Grid barrier over the group's CTAs on the arrival counter; returns the next phase."""
    arrival_ptr = _fused_state_ptr(state, group_id, Int32(_FUSED_STATE_ARRIVAL))
'''
_AFTER = '''    """Grid barrier over the group's CTAs on the arrival counter; returns the next phase."""
    # Every publishing warp must finish before the leader releases this CTA's
    # arrival; otherwise peers can scan partial histograms and diverge in rounds.
    cute.arch.sync_threads()
    arrival_ptr = _fused_state_ptr(state, group_id, Int32(_FUSED_STATE_ARRIVAL))
'''
_OLD_CACHE = '"attention.indexer.fused_indexer", 1, cache_key, labels=labels'
_NEW_CACHE = '"attention.indexer.fused_indexer", 2, cache_key, labels=labels'


def apply_patch(path: Path) -> None:
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest == AFTER_SHA256:
        return
    if digest != BEFORE_SHA256:
        raise RuntimeError(f"unsupported B12X fused indexer source: {digest}")
    text = source.decode("utf-8")
    if text.count(_BEFORE) != 1 or text.count(_OLD_CACHE) != 1:
        raise RuntimeError("pinned B12X barrier or compile revision differs")
    patched = text.replace(_BEFORE, _AFTER, 1).replace(_OLD_CACHE, _NEW_CACHE, 1).encode("utf-8")
    if hashlib.sha256(patched).hexdigest() != AFTER_SHA256:
        raise RuntimeError("B12X barrier transform differs from the GPU-tested source")
    path.write_bytes(patched)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    apply_patch(parser.parse_args().path)
