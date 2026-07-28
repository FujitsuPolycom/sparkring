#!/usr/bin/env python3
"""Host-side store verification for one rank (stdlib only).

Walks every manifest in a rank-local context-cache store, re-verifies it
through the fail-closed engine (lookup + restore re-hash every chunk),
unpacks the logical_positions records, and emits a JSON coverage report
for cross-rank reconciliation.

Run on a Spark host:
  python3 context_cache_verify_store.py \
      --store ~/sparkring-context-cache/store \
      --engine <engine-path>/persistent_context_cache/cache_manifest.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
from pathlib import Path


def load_engine(path: Path):
    spec = importlib.util.spec_from_file_location("cache_manifest", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cache_manifest"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    args = parser.parse_args()

    engine = load_engine(args.engine)
    store = engine.ManifestStore(args.store)
    report = {"store": str(args.store), "entries": []}
    manifest_files = sorted(args.store.glob("manifests/*/*.json"))
    for manifest_path in manifest_files:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity_wire = raw["identity"]
        identity = engine.CacheIdentity(**identity_wire)
        digest = raw["context_digest"]
        lookup = store.lookup(identity, digest)
        entry = {
            "digest": digest,
            "identity_key": identity.storage_key,
            "dcp_shard_rank": identity_wire.get("dcp_shard_rank"),
            "committed_tokens": raw.get("committed_tokens"),
            "lookup": lookup.reason,
        }
        if lookup.is_hit:
            chunks = store.restore(lookup)
            if chunks is None:
                entry["restore"] = "failed"
            else:
                positions = []
                for chunk in chunks:
                    payload = chunk.records[
                        engine.StateRecord.LOGICAL_POSITIONS
                    ]
                    positions.extend(
                        struct.unpack_from("<I", payload, offset)[0]
                        for offset in range(0, len(payload), 4)
                    )
                entry["restore"] = "ok"
                entry["position_count"] = len(positions)
                entry["position_min"] = min(positions)
                entry["position_max"] = max(positions)
                entry["positions_sha1k"] = positions[:8]
                entry["record_bytes"] = {
                    kind.value: sum(
                        len(chunk.records[kind]) for chunk in chunks
                        if kind in chunk.records
                    )
                    for kind in engine.StateRecord
                }
        report["entries"].append(entry)
    report["entry_count"] = len(report["entries"])
    json.dump(report, sys.stdout, indent=1, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
