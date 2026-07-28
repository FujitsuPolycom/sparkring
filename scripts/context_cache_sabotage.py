#!/usr/bin/env python3
"""P3 sabotage driver (runs on a Spark host).

Damages one rank's persisted entry in a chosen way, then reports the store
state so the caller can drive a request and observe fail-closed behavior.

Modes:
  bitflip   - flip one bit inside a middle chunk payload
  truncate  - cut a chunk file in half
  fingerprint - rewrite the manifest's identity to a mismatching layout
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
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
    parser.add_argument(
        "--mode",
        choices=("bitflip", "truncate", "fingerprint"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    engine = load_engine(args.engine)
    manifests = sorted(args.store.glob("manifests/*/*.json"))
    if not manifests:
        print(json.dumps({"error": "no manifests in store"}))
        return 2
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = engine.CacheIdentity(**manifest["identity"])
    digest = manifest["context_digest"]
    rng = random.Random(args.seed)
    report = {
        "mode": args.mode,
        "digest": digest,
        "manifest": str(manifest_path),
    }

    if args.mode == "fingerprint":
        manifest["identity"]["quantization_layout"] = "mismatched-layout-v0"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        report["damaged"] = "manifest identity rewritten"
    else:
        descriptors = manifest["chunks"]
        target = descriptors[len(descriptors) // 2]
        chunk_path = args.store / "chunks" / f"{target['sha256']}.spcc"
        payload = bytearray(chunk_path.read_bytes())
        if args.mode == "bitflip":
            index = rng.randrange(len(payload))
            payload[index] ^= 1 << rng.randrange(8)
            chunk_path.write_bytes(bytes(payload))
            report["damaged"] = f"bit flipped at byte {index}"
        else:
            chunk_path.write_bytes(bytes(payload[: len(payload) // 2]))
            report["damaged"] = "chunk truncated to half length"
        report["chunk"] = str(chunk_path)

    lookup = engine.ManifestStore(args.store).lookup(identity, digest)
    report["post_damage_lookup"] = lookup.reason
    probe = engine.ManifestStore(args.store).lookup(
        identity, digest, verify_chunks=False
    )
    report["post_damage_probe"] = probe.reason
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
