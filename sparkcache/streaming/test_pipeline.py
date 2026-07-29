from __future__ import annotations

from sparkcache.persistent_context_cache.cache_manifest import (
    CacheIdentity,
    ContextChunk,
    ManifestStore,
    StateRecord,
)
from sparkcache.streaming import (
    BlockTableRangeMapper,
    StreamingSnapshotCoordinator,
)


def _identity() -> CacheIdentity:
    return CacheIdentity(
        target_checkpoint="1" * 64,
        draft_checkpoint="2" * 64,
        quantization_layout="nvfp4-ds-mla-v1",
        rope_layout="glm52-bf16-rope-v1",
        tp_degree=4,
        dcp_degree=4,
        dcp_shard_rank=0,
        chunk_tokens=256,
    )


def _chunk(start: int) -> ContextChunk:
    marker = start.to_bytes(4, "little")
    return ContextChunk(
        logical_start=start,
        logical_end=start + 256,
        records={record: record.value.encode() + marker for record in StateRecord},
    )


def test_planner_batches_feed_incremental_manifest_transaction(tmp_path) -> None:
    store = ManifestStore(tmp_path)
    digest = "a" * 64
    transaction = store.begin_context(
        identity=_identity(),
        context_digest=digest,
        span_tokens=1024,
    )
    planner = StreamingSnapshotCoordinator(
        chunks_per_batch=2,
        max_inflight_batches=2,
    )
    assert planner.begin("request", digest, 1024)
    block_map = BlockTableRangeMapper(tuple(range(16)), 64)

    for completed in (512, 1024):
        offer = planner.offer_completed(
            "request",
            completed,
            block_map,
        )
        assert not offer.aborted
        assert len(offer.batches) == 1
        batch = offer.batches[0]
        for chunk_index in batch.chunk_indexes:
            transaction.append_chunk(_chunk(chunk_index * 256))
        planner.complete_batch("request", batch.batch_index)

    transaction.commit_manifest()
    planner.commit("request")
    lookup = store.lookup(_identity(), digest)
    assert lookup.is_hit, lookup.reason
    restored = store.restore(lookup)
    assert restored is not None
    assert [chunk.logical_start for chunk in restored] == [0, 256, 512, 768]


def test_backpressure_abort_never_publishes_partial_manifest(tmp_path) -> None:
    store = ManifestStore(tmp_path)
    digest = "b" * 64
    transaction = store.begin_context(
        identity=_identity(),
        context_digest=digest,
        span_tokens=1024,
    )
    planner = StreamingSnapshotCoordinator(
        chunks_per_batch=1,
        max_inflight_batches=1,
    )
    assert planner.begin("request", digest, 1024)

    overflow = planner.offer_completed(
        "request",
        1024,
        BlockTableRangeMapper(tuple(range(16)), 64),
    )
    assert overflow.aborted
    transaction.abort()

    assert not store.lookup(_identity(), digest).is_hit
    assert not tuple((tmp_path / "manifests").rglob("*.json"))
