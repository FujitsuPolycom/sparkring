from __future__ import annotations

import pytest

from sparkcache.streaming.planner import (
    BlockTableRangeMapper,
    SnapshotState,
    StreamingSnapshotCoordinator,
)


def mapper(
    block_ids: tuple[int, ...],
    logical_tokens_per_block: int = 1024,
) -> BlockTableRangeMapper:
    return BlockTableRangeMapper(block_ids, logical_tokens_per_block)


def test_chunked_prefill_emits_bounded_macrobatches_and_commits() -> None:
    coordinator = StreamingSnapshotCoordinator(
        chunks_per_batch=16,
        max_inflight_batches=3,
    )
    assert coordinator.begin("request", "a" * 64, 8192)

    first = coordinator.offer_completed(
        "request", 4096, mapper(tuple(range(10, 18)))
    ).batches
    assert len(first) == 1
    assert first[0].logical_start == 0
    assert first[0].logical_end == 4096
    assert first[0].chunk_count == 16
    assert not coordinator.complete_batch("request", first[0].batch_index)

    second = coordinator.offer_completed(
        "request", 8192, mapper(tuple(range(10, 18)))
    ).batches
    assert len(second) == 1
    assert second[0].logical_start == 4096
    assert second[0].logical_end == 8192
    assert first[0].block_ids == (10, 11, 12, 13)
    assert second[0].block_ids == (14, 15, 16, 17)
    assert coordinator.complete_batch("request", second[0].batch_index)
    assert coordinator.state("request") is SnapshotState.READY_TO_COMMIT

    coordinator.commit("request")
    assert coordinator.state("request") is None
    assert coordinator.counters["committed"] == 1


def test_duplicate_digest_is_single_flight() -> None:
    coordinator = StreamingSnapshotCoordinator()
    assert coordinator.begin("owner", "b" * 64, 4096)
    assert not coordinator.begin("duplicate", "b" * 64, 4096)
    assert coordinator.counters["duplicate_suppressed"] == 1


def test_ring_backpressure_aborts_cache_without_returning_partial_work() -> None:
    coordinator = StreamingSnapshotCoordinator(
        chunks_per_batch=4,
        max_inflight_batches=2,
    )
    assert coordinator.begin("request", "c" * 64, 4096)

    offer = coordinator.offer_completed(
        "request", 4096, mapper((1, 2, 3, 4))
    )
    assert offer.batches == ()
    assert offer.aborted
    assert offer.release_batches == ()
    assert coordinator.state("request") is None
    assert coordinator.inflight_batches == 0
    assert coordinator.counters["aborted_backpressure"] == 1


def test_explicit_abort_returns_batches_whose_leases_must_be_released() -> None:
    coordinator = StreamingSnapshotCoordinator()
    assert coordinator.begin("request", "d" * 64, 8192)
    batches = coordinator.offer_completed(
        "request", 4096, mapper(tuple(range(8)))
    ).batches
    assert len(batches) == 1

    pending = coordinator.abort("request")

    assert pending == batches
    assert coordinator.state("request") is None
    assert coordinator.inflight_batches == 0


def test_completed_tokens_must_be_monotonic() -> None:
    coordinator = StreamingSnapshotCoordinator()
    assert coordinator.begin("request", "e" * 64, 8192)
    batch = coordinator.offer_completed(
        "request", 4096, mapper(tuple(range(8)))
    ).batches[0]
    coordinator.complete_batch("request", batch.batch_index)

    with pytest.raises(ValueError, match="monotonic"):
        coordinator.offer_completed(
            "request", 2048, mapper(tuple(range(8)))
        )


def test_partial_chunk_waits_for_a_complete_256_token_record() -> None:
    coordinator = StreamingSnapshotCoordinator()
    assert coordinator.begin("request", "f" * 64, 1024)

    block_map = mapper((1, 2, 3, 4), logical_tokens_per_block=256)
    assert coordinator.offer_completed("request", 255, block_map).batches == ()
    emitted = coordinator.offer_completed("request", 256, block_map).batches

    assert len(emitted) == 1
    assert emitted[0].logical_start == 0
    assert emitted[0].logical_end == 256


def test_backpressure_returns_prior_batches_that_need_lease_release() -> None:
    coordinator = StreamingSnapshotCoordinator(
        chunks_per_batch=4,
        max_inflight_batches=1,
    )
    assert coordinator.begin("request", "1" * 64, 2048)
    first = coordinator.offer_completed(
        "request", 1024, mapper((1, 2))
    ).batches
    assert len(first) == 1

    overflow = coordinator.offer_completed(
        "request", 2048, mapper((1, 2))
    )

    assert overflow.aborted
    assert overflow.release_batches == first
    assert coordinator.state("request") is None


def test_disjoint_macrobatches_can_hold_simultaneous_block_leases() -> None:
    from sparkcache.streaming.block_lease import BlockLeaseRegistry, LeaseCapacity

    coordinator = StreamingSnapshotCoordinator(
        chunks_per_batch=16,
        max_inflight_batches=3,
    )
    assert coordinator.begin("request", "2" * 64, 8192)
    batches = coordinator.offer_completed(
        "request",
        8192,
        mapper(tuple(range(20, 28))),
    ).batches

    assert [batch.block_ids for batch in batches] == [
        (20, 21, 22, 23),
        (24, 25, 26, 27),
    ]
    leases = BlockLeaseRegistry(LeaseCapacity(2, 8))
    first = leases.try_reserve("request", batches[0].block_ids)
    second = leases.try_reserve("request", batches[1].block_ids)
    assert first is not None
    assert second is not None
    assert leases.active_leases == 2


def test_mapping_failure_aborts_cache_without_raising() -> None:
    coordinator = StreamingSnapshotCoordinator()
    assert coordinator.begin("request", "3" * 64, 4096)
    too_short = mapper((1,), logical_tokens_per_block=1024)

    offer = coordinator.offer_completed("request", 4096, too_short)

    assert offer.aborted
    assert offer.batches == ()
    assert coordinator.state("request") is None
    assert coordinator.counters["aborted_mapping"] == 1
