from __future__ import annotations

from dataclasses import dataclass

import pytest

from sparkcache.streaming.block_lease import (
    BlockLeaseRegistry,
    LeaseCapacity,
    LeaseFenceError,
)
from sparkcache.streaming.preemption import (
    PreemptionDrainAdapter,
    PreemptionMetadataError,
    StreamingPreemptionMetadata,
)


class OrderedFence:
    def __init__(self, events: list[str], *, fail: bool = False) -> None:
        self.events = events
        self.fail = fail

    def query(self) -> bool:
        return False

    def synchronize(self) -> None:
        self.events.append("fence-synchronized")
        if self.fail:
            raise RuntimeError("CUDA event failed")


def registry() -> BlockLeaseRegistry:
    return BlockLeaseRegistry(LeaseCapacity(4, 16))


def test_armed_gather_is_synchronized_before_publication_abandon() -> None:
    events: list[str] = []
    leases = registry()
    handle = leases.try_reserve("req", [4, 5])
    assert handle is not None
    handle.submit(lambda: OrderedFence(events))
    adapter = PreemptionDrainAdapter(
        leases, lambda request_id: events.append(f"abandon-{request_id}")
    )

    released = adapter.handle_preemptions(StreamingPreemptionMetadata(("req",)))

    assert released == 1
    assert events == ["fence-synchronized", "abandon-req"]
    assert not leases.has_pending()


def test_reserved_work_cancels_without_synchronization() -> None:
    events: list[str] = []
    leases = registry()
    assert leases.try_reserve("req", [4]) is not None
    adapter = PreemptionDrainAdapter(
        leases, lambda request_id: events.append(f"abandon-{request_id}")
    )

    assert adapter.handle_preemptions(StreamingPreemptionMetadata(("req",))) == 1
    assert events == ["abandon-req"]
    assert not leases.has_pending()


def test_only_preempted_request_is_drained() -> None:
    leases = registry()
    first = leases.try_reserve("first", [1])
    second = leases.try_reserve("second", [2])
    assert first is not None and second is not None
    first.submit(lambda: OrderedFence([]))
    second.submit(lambda: OrderedFence([]))
    abandoned: list[str] = []
    adapter = PreemptionDrainAdapter(leases, abandoned.append)

    assert adapter.handle_preemptions(
        StreamingPreemptionMetadata(("first",))
    ) == 1
    assert abandoned == ["first"]
    assert not leases.has_pending("first")
    assert leases.has_pending("second")


def test_unknown_fence_state_is_fatal_and_blocks_remain_leased() -> None:
    events: list[str] = []
    leases = registry()
    handle = leases.try_reserve("req", [7])
    assert handle is not None
    handle.submit(lambda: OrderedFence(events, fail=True))
    abandoned: list[str] = []
    adapter = PreemptionDrainAdapter(leases, abandoned.append)

    with pytest.raises(LeaseFenceError, match="cannot safely abort"):
        adapter.handle_preemptions(StreamingPreemptionMetadata(("req",)))

    assert events == ["fence-synchronized"]
    assert abandoned == []
    assert leases.has_pending("req")
    assert leases.leased_blocks == 1


@dataclass
class MissingPreemptionField:
    plans: list[object]


@pytest.mark.parametrize(
    "metadata,message",
    [
        (MissingPreemptionField([]), "lacks"),
        (
            type("BadMetadata", (), {"preempted_request_ids": {"req"}})(),
            "deterministic tuple",
        ),
        (
            type("BadMetadata", (), {"preempted_request_ids": ("z", "a")})(),
            "sorted",
        ),
        (
            type("BadMetadata", (), {"preempted_request_ids": ("a", "a")})(),
            "unique",
        ),
    ],
)
def test_malformed_metadata_fails_closed(metadata: object, message: str) -> None:
    adapter = PreemptionDrainAdapter(registry(), lambda _: None)
    with pytest.raises(PreemptionMetadataError, match=message):
        adapter.handle_preemptions(metadata)
