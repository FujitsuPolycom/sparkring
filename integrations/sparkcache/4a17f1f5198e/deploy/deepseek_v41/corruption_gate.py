"""Copy one SparkCache entry and prove chunk corruption is rejected.

Model-neutral; the DeepSeek-V4.1 deployment runs it against a disposable
root to prove CRC and digest verification reject damaged chunks and that
invalidation returns the entry to a clean miss.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sparkcache.spark_context_cache_store import CacheIdentity, ManifestStore


def _identity(document: dict[str, Any]) -> CacheIdentity:
    wire = dict(document)
    if "record_schema" in wire:
        wire["record_schema"] = tuple(wire["record_schema"])
    return CacheIdentity(**wire)


def run(source_root: Path, test_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    test_root = test_root.resolve()
    if not source_root.is_dir():
        raise RuntimeError(f"source cache root is absent: {source_root}")
    if test_root.exists():
        raise RuntimeError(f"refusing to overwrite corruption root: {test_root}")
    if source_root == test_root or source_root in test_root.parents:
        raise RuntimeError("corruption root must not contain the source root")
    if test_root in source_root.parents:
        raise RuntimeError("corruption root must not be inside the source root")

    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for path in source_root.glob("manifests/*/*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        candidates.append((int(document["committed_tokens"]), path, document))
    if not candidates:
        raise RuntimeError("source cache contains no manifests")
    committed_tokens, manifest_path, manifest = min(
        candidates,
        key=lambda item: (item[0], item[1].as_posix()),
    )
    context_digest = str(manifest["context_digest"])
    identity = _identity(dict(manifest["identity"]))
    descriptors = list(manifest["chunks"])
    if not descriptors:
        raise RuntimeError("selected manifest contains no chunks")

    copied_bytes = 0
    for descriptor in descriptors:
        digest = str(descriptor["sha256"])
        source = source_root / "chunks" / f"{digest}.spcc"
        destination = test_root / "chunks" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_bytes += destination.stat().st_size
    copied_manifest = test_root / manifest_path.relative_to(source_root)
    copied_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, copied_manifest)

    store = ManifestStore(test_root)
    healthy = store.lookup(identity, context_digest, verify_chunks=True)
    if not healthy.is_hit:
        raise RuntimeError(f"copied entry did not verify: {healthy.reason}")

    damaged_digest = str(descriptors[0]["sha256"])
    damaged = test_root / "chunks" / f"{damaged_digest}.spcc"
    with damaged.open("r+b") as stream:
        offset = max(0, damaged.stat().st_size // 2)
        stream.seek(offset)
        original = stream.read(1)
        if not original:
            raise RuntimeError("selected chunk is empty")
        stream.seek(offset)
        stream.write(bytes((original[0] ^ 0x40,)))
        stream.flush()

    corrupted = store.lookup(identity, context_digest, verify_chunks=True)
    if corrupted.is_hit or corrupted.reason != "corrupt":
        raise RuntimeError("damaged entry was not rejected as corrupt")
    invalidated = store.invalidate(
        identity,
        context_digest,
        verify_chunk_payloads=True,
    )
    if not invalidated:
        raise RuntimeError("damaged entry could not be invalidated")
    after = store.lookup(identity, context_digest, verify_chunks=True)
    if after.is_hit or after.reason != "absent":
        raise RuntimeError("invalidated entry did not become an absent miss")
    if damaged.exists():
        raise RuntimeError("invalidated corrupt chunk was not removed")
    return {
        "schema": "sparkcache-corruption-gate/v1",
        "context_digest": context_digest,
        "committed_tokens": committed_tokens,
        "chunks_copied": len(descriptors),
        "bytes_copied": copied_bytes,
        "healthy_before_corruption": True,
        "corruption_reason": corrupted.reason,
        "invalidated": invalidated,
        "absent_after_invalidation": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--test-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.source_root, args.test_root), sort_keys=True))


if __name__ == "__main__":
    main()
