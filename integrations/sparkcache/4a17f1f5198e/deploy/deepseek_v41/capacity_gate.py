"""Small live-NVMe check for SparkCache quota and eviction behavior.

Model-neutral; the DeepSeek-V4.1 deployment runs it before and after
publication to prove quota and LRU eviction behave on the target NVMe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sparkcache.spark_context_cache_store import (
    CacheIdentity,
    CapacityPolicy,
    ContextChunk,
    ManifestStore,
    StateRecord,
)

_MIB = 1024 * 1024


def _identity() -> CacheIdentity:
    return CacheIdentity(
        target_checkpoint="1" * 64,
        draft_checkpoint="2" * 64,
        quantization_layout="capacity-gate-v1",
        rope_layout="capacity-gate-rope-v1",
        tp_degree=1,
        dcp_degree=1,
    )


def _chunk(label: int) -> ContextChunk:
    return ContextChunk(
        logical_start=0,
        logical_end=256,
        records={
            StateRecord.TARGET_CKV: bytes((label,)) * (2 * _MIB),
            StateRecord.SPARSE_INDEXER: b"indexer",
            StateRecord.MTP_DRAFT_KV: b"draft",
            StateRecord.BOUNDARY_HIDDEN: b"hidden",
            StateRecord.LOGICAL_POSITIONS: b"positions",
        },
    )


def run(root: Path) -> dict[str, Any]:
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"capacity check root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    store = ManifestStore(root)
    identity = _identity()
    digests = [hashlib.sha256(f"gate-{index}".encode()).hexdigest() for index in range(3)]
    for index, digest in enumerate(digests):
        store.commit(
            identity=identity,
            context_digest=digest,
            chunks=[_chunk(index + 1)],
            span_tokens=256,
        )
        manifest = root / "manifests" / identity.storage_key / f"{digest}.json"
        mtime_ns = (index + 1) * 10**9
        os.utime(manifest, ns=(mtime_ns, mtime_ns))

    open_transaction = store.begin_context(
        identity=identity,
        context_digest=hashlib.sha256(b"gate-open").hexdigest(),
        span_tokens=256,
    )
    open_transaction.append_chunk(_chunk(9))
    busy = store.maintain(
        CapacityPolicy(max_bytes=5 * _MIB, low_watermark_bytes=3 * _MIB)
    )
    if not busy.skipped_busy:
        raise RuntimeError("capacity maintenance did not respect the open transaction")
    open_transaction.abort()

    report = store.maintain(
        CapacityPolicy(max_bytes=5 * _MIB, low_watermark_bytes=3 * _MIB),
        now_ns=10 * 10**9,
    )
    hits = [store.lookup(identity, digest, verify_chunks=False).is_hit for digest in digests]
    if hits != [False, False, True]:
        raise RuntimeError(f"capacity check selected the wrong LRU survivors: {hits}")
    if not report.capacity_satisfied or report.bytes_after > 5 * _MIB:
        raise RuntimeError("capacity check did not satisfy the high watermark")
    return {
        "schema": "sparkcache-capacity-gate/v1",
        "bytes_before": report.bytes_before,
        "bytes_after": report.bytes_after,
        "bytes_reclaimed": report.bytes_reclaimed,
        "manifests_evicted": report.manifests_evicted,
        "chunks_deleted": report.chunks_deleted,
        "orphan_chunks_deleted": report.orphan_chunks_deleted,
        "survivors": sum(hits),
        "capacity_satisfied": report.capacity_satisfied,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
