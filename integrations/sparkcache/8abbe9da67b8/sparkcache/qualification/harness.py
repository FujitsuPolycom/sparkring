"""Local page-tail qualification harness.

Drives the production ``sparkcache.persistent_context_cache`` commit and
restore paths through the byte-exact fixture and records a machine-readable
receipt. The harness never imports torch or vLLM and never executes model
code; it exercises the exact storage, identity, and integrity contracts that
the live runner measures again on four GPU ranks.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sparkcache.persistent_context_cache.cache_manifest import (
    CacheIdentity,
    ManifestStore,
    PageDeltaDepthExceeded,
)
from sparkcache.qualification.fixture import (
    PageTailFixture,
    build_fixture,
    snapshot_digest,
)
from sparkcache.spark_context_cache_codec import (
    context_prefix_digest,
)


PAGE_SNAPSHOT_SCHEMA = "sparkcache-page-snapshot-manifest/v2"
PAGE_DELTA_SCHEMA = "sparkcache-page-delta-manifest/v2"


class HarnessError(RuntimeError):
    """The qualification path produced a receipt-defining inconsistency."""


@dataclass(frozen=True)
class _Tokens:
    """Digest and identity inputs for one qualification boundary."""

    identity: CacheIdentity
    salt: str
    prefix: tuple[int, ...]
    base_digest: str
    result_digest: str


def _cache_identity(
    fixture: PageTailFixture,
    dcp_shard_rank: int,
) -> CacheIdentity:
    return CacheIdentity(
        target_checkpoint=fixture.target_checkpoint,
        draft_checkpoint=fixture.draft_checkpoint,
        quantization_layout=fixture.quantization_layout,
        rope_layout=fixture.rope_layout,
        tp_degree=4,
        dcp_degree=4,
        dcp_shard_rank=dcp_shard_rank,
        tp_shard_rank=dcp_shard_rank,
        boundary_hidden_policy="live_forward",
        draft_kv_policy="separate",
        chunk_tokens=fixture.tokens_per_page,
        record_schema=("target_ckv", "logical_positions"),
        publication_schema="page-tail-cow-v1",
    )

def _elapsed_ms(start_ns: int) -> float:
    return round((time.perf_counter_ns() - start_ns) / 1_000_000, 3)
def _commit_base(
    store: ManifestStore,
    fixture: PageTailFixture,
    identity: CacheIdentity,
    context_digest: str,
    base_page_count: int,
    base_snapshot: bytes,
) -> tuple[str, float, int]:
    """Publish the complete-snapshot base in the page-tail namespace.

    The connector publishes every baseless hybrid snapshot through
    ``commit_page_snapshot`` (spark_context_cache_connector.py); the row
    chunk path does not apply to opaque page payloads. ``commit_page_snapshot``
    accepts the ``page-tail-cow-v1`` publication schema, so the base shares
    the delta identity namespace by construction.
    """

    started = time.perf_counter_ns()
    receipt = store.commit_page_snapshot(
        identity=identity,
        context_digest=context_digest,
        span_tokens=base_page_count * fixture.tokens_per_page,
        snapshot=base_snapshot,
    )
    return receipt.manifest_digest, _elapsed_ms(started), len(base_snapshot)


def _manifest_of(
    store: ManifestStore,
    identity: CacheIdentity,
    context_digest: str,
) -> dict[str, Any]:
    path = store._manifest_path(identity, context_digest)
    return dict(json.loads(path.read_bytes()))


def _restore_and_compare(
    store: ManifestStore,
    fixture: PageTailFixture,
    identity: CacheIdentity,
    result_digest: str,
    result_tokens: int,
    result_page_count: int,
    expected_snapshot: bytes,
) -> float:
    """Authenticate and restore one result, requiring byte-exact equality."""

    started = time.perf_counter_ns()
    lookup = store.lookup(identity, result_digest)
    if not lookup.is_hit:
        raise HarnessError(f"result lookup missed: {lookup.reason}")
    restored = store.restore_page_snapshot(
        lookup,
        layout=fixture.layout,
        result_block_counts=(result_page_count, 1),
        result_boundary_tokens=result_tokens,
    )
    elapsed_ms = _elapsed_ms(started)
    if bytes(restored) != expected_snapshot:
        raise HarnessError("restored page snapshot differs from the exact result")
    return elapsed_ms


def _extend(
    store: ManifestStore,
    fixture: PageTailFixture,
    tokens: _Tokens,
    base_page_count: int,
    result_page_count: int,
    base_snapshot_bytes: int,
) -> dict[str, Any]:
    """Publish one bounded-depth page delta and restore it exactly."""

    result_tokens = result_page_count * fixture.tokens_per_page
    base_boundary_tokens = base_page_count * fixture.tokens_per_page
    result_snapshot = fixture.snapshot_bytes(result_page_count)
    started = time.perf_counter_ns()
    try:
        receipt = store.commit_page_extension(
            identity=tokens.identity,
            base_context_digest=tokens.base_digest,
            token_ids=tokens.prefix,
            identity_salt=tokens.salt,
            layout=fixture.layout,
            base_block_counts=(base_page_count, 1),
            result_block_counts=(result_page_count, 1),
            base_boundary_tokens=base_boundary_tokens,
            result_boundary_tokens=result_tokens,
            result_snapshot=result_snapshot,
        )
        compacted = False
    except PageDeltaDepthExceeded:
        receipt = store.commit_page_snapshot(
            identity=tokens.identity,
            context_digest=tokens.result_digest,
            span_tokens=result_tokens,
            snapshot=result_snapshot,
        )
        compacted = True
    commit_ms = _elapsed_ms(started)

    manifest = _manifest_of(store, tokens.identity, tokens.result_digest)
    schema = manifest["schema"]
    if compacted != (schema == PAGE_SNAPSHOT_SCHEMA):
        raise HarnessError("publication root kind disagrees with the commit path")
    restore_ms = _restore_and_compare(
        store,
        fixture,
        tokens.identity,
        tokens.result_digest,
        result_tokens,
        result_page_count,
        result_snapshot,
    )
    return {
        "context_digest": tokens.result_digest,
        "manifest_digest": receipt.manifest_digest,
        "manifest_schema": schema,
        "compacted": compacted,
        "commit_ms": commit_ms,
        "restore_ms": restore_ms,
        "base_context_digest": tokens.base_digest,
        "stored_base_context_digest": manifest.get(
            "base_context_digest", tokens.base_digest
        ),
        "flattened": manifest.get("base_context_digest") not in (
            None,
            tokens.base_digest,
        ),
        "base_committed_tokens": base_boundary_tokens,
        "committed_tokens": result_tokens,
        "base_snapshot_bytes": base_snapshot_bytes,
        "delta_encoded_bytes": manifest.get("delta_encoded_bytes", 0),
        "delta_objects": len(manifest.get("delta_objects", ())),
        "result_snapshot_bytes": len(result_snapshot),
        "result_snapshot_sha256": snapshot_digest(result_snapshot),
    }


def run_page_tail_qualification(
    root: Path,
    fixture: PageTailFixture | None = None,
    *,
    dcp_shard_rank: int = 0,
    base_page_count: int = 4,
    steps: tuple[int, ...] = (2, 2),
    corrupt_result: bool = True,
) -> dict[str, Any]:
    """Run the complete local qualification cohort and return its receipt.

    The cohort publishes the byte-exact base once, extends it once per
    ``steps`` entry, compares every restored snapshot byte-for-byte against
    the independently encoded expected image, verifies restart restoration
    from a fresh ``ManifestStore`` over the same on-disk root, and finally
    corrupts one stored result object to confirm clean rejection. All
    measurements are wall-clock times of the production commit/restore calls
    on the calling thread.
    """

    fixture = fixture or build_fixture()
    root = Path(root)
    store = ManifestStore(root)
    identity = _cache_identity(fixture, dcp_shard_rank)
    tokens_per_page = fixture.tokens_per_page
    base_tokens = base_page_count * tokens_per_page
    final_tokens = (base_page_count + sum(steps)) * tokens_per_page
    result_token_values = tuple(range(final_tokens))
    base_digest = context_prefix_digest(
        result_token_values, fixture.identity_salt, token_count=base_tokens
    )

    base_snapshot = fixture.snapshot_bytes(base_page_count)
    base_commit_digest, base_commit_ms, base_snapshot_bytes = _commit_base(
        store,
        fixture,
        identity,
        base_digest,
        base_page_count,
        base_snapshot,
    )

    deltas: list[dict[str, Any]] = []
    prior_digest = base_digest
    prior_page_count = base_page_count
    for growth in steps:
        current_page_count = prior_page_count + growth
        boundary = current_page_count * tokens_per_page
        step_tokens = _Tokens(
            identity=identity,
            salt=fixture.identity_salt,
            prefix=result_token_values,
            base_digest=prior_digest,
            result_digest=context_prefix_digest(
                result_token_values, fixture.identity_salt, token_count=boundary
            ),
        )
        record = _extend(
            store,
            fixture,
            step_tokens,
            prior_page_count,
            current_page_count,
            base_snapshot_bytes,
        )
        deltas.append(record)
        prior_digest = record["context_digest"]
        prior_page_count = current_page_count

    final = deltas[-1]
    result_tokens = final["committed_tokens"]
    expected_snapshot = fixture.snapshot_bytes(result_tokens // tokens_per_page)
    if final["result_snapshot_sha256"] != snapshot_digest(expected_snapshot):
        raise HarnessError("result snapshot digest differs from the expected image")

    restarted = ManifestStore(root)
    restart_restore_ms = _restore_and_compare(
        restarted,
        fixture,
        identity,
        final["context_digest"],
        result_tokens,
        result_tokens // tokens_per_page,
        expected_snapshot,
    )

    corruption = (
        _corrupt_and_verify(store, identity, final["context_digest"])
        if corrupt_result
        else None
    )

    return {
        "schema": "sparkcache-page-tail-qualification/v1",
        "runner": "local",
        "identity": {
            "publication_schema": identity.publication_schema,
            "storage_key": identity.storage_key,
            "tp_degree": identity.tp_degree,
            "dcp_degree": identity.dcp_degree,
            "dcp_shard_rank": identity.dcp_shard_rank,
            "record_schema": list(identity.record_schema),
            "chunk_tokens": identity.chunk_tokens,
        },
        "fixture": {
            "identity_salt": fixture.identity_salt,
            "layout_sha256": fixture.layout.digest,
            "tokens_per_page": tokens_per_page,
        },
        "base": {
            "context_digest": base_digest,
            "manifest_digest": base_commit_digest,
            "commit_ms": base_commit_ms,
            "snapshot_bytes": base_snapshot_bytes,
            "committed_tokens": base_tokens,
        },
        "deltas": deltas,
        "result": {
            "context_digest": final["context_digest"],
            "committed_tokens": result_tokens,
            "expected_snapshot_sha256": snapshot_digest(expected_snapshot),
            "restart_restore_ms": restart_restore_ms,
        },
        "unique_publication_bytes": _unique_publication_bytes(root),
        "logical_snapshot_bytes": sum(
            step["result_snapshot_bytes"] for step in deltas
        ),
        "corruption": corruption,
        "validation": {
            "byte_exact_restores": True,
            "complete_snapshot_equal": True,
            "corruption_rejected": corruption is not None and corruption["rejected"],
        },
    }


def _corrupt_and_verify(
    store: ManifestStore,
    identity: CacheIdentity,
    result_digest: str,
) -> dict[str, Any]:
    """Flip one byte in a stored result object and require clean rejection."""

    manifest = _manifest_of(store, identity, result_digest)
    descriptors = manifest.get("delta_objects") or manifest.get("snapshot_objects")
    if not descriptors:
        raise HarnessError("result manifest has no object descriptors")
    last = descriptors[-1]
    object_path = store.root / "chunks" / f"{last['sha256']}.spcc"
    original = object_path.read_bytes()
    corrupted = bytearray(original)
    corrupted[len(corrupted) // 2] ^= 0x01
    object_path.write_bytes(bytes(corrupted))
    try:
        lookup = store.lookup(identity, result_digest)
        rejected = not lookup.is_hit and lookup.reason == "corrupt"
    finally:
        object_path.write_bytes(original)
    if lookup.is_hit or not rejected:
        raise HarnessError("corrupted result was not rejected as corrupt")
    return {
        "object": object_path.name,
        "byte_offset": len(original) // 2,
        "rejected": True,
        "lookup_reason": "corrupt",
    }


def _unique_publication_bytes(root: Path) -> int:
    """Allocated on-disk bytes across stored objects and manifests."""

    total = 0
    for name in ("chunks", "manifests"):
        directory = root / name
        if directory.is_dir():
            for path in directory.rglob("*"):
                if path.is_file():
                    metadata = path.stat()
                    blocks = getattr(metadata, "st_blocks", 0)
                    total += int(blocks) * 512 if blocks else metadata.st_size
    return total
