from __future__ import annotations

import threading
import time

import pytest

from sparkcache.streaming.block_lease import (
    BlockLeaseRegistry,
    LeaseCapacity,
    LeaseFenceError,
    LeaseStateError,
)


class FakeFence:
    def __init__(
        self,
        *,
        done: bool = False,
        query_error: bool = False,
        sync_error: bool = False,
    ) -> None:
        self.done = done
        self.query_error = query_error
        self.sync_error = sync_error
        self.sync_calls = 0

    def query(self) -> bool:
        if self.query_error:
            raise RuntimeError("query failed")
        return self.done

    def synchronize(self) -> None:
        self.sync_calls += 1
        if self.sync_error:
            raise RuntimeError("sync failed")
        self.done = True


def registry(*, leases: int = 2, blocks: int = 8) -> BlockLeaseRegistry:
    return BlockLeaseRegistry(LeaseCapacity(leases, blocks))


def test_completed_fence_releases_blocks_and_finished_request() -> None:
    leases = registry()
    handle = leases.try_reserve("req", [3, 4])
    assert handle is not None
    fence = FakeFence()
    handle.submit(lambda: fence)

    assert leases.take_finished({"req"}) == set()
    assert leases.active_leases == 1
    fence.done = True
    assert leases.take_finished({"req"}) == {"req"}
    assert leases.active_leases == 0
    assert leases.leased_blocks == 0


def test_admission_is_fail_open_at_both_capacity_limits() -> None:
    leases = registry(leases=1, blocks=2)
    assert leases.try_reserve("first", [1, 2]) is not None
    assert leases.try_reserve("lease-full", [3]) is None
    assert leases.counters["rejected_lease_capacity"] == 1

    block_limited = registry(leases=4, blocks=2)
    assert block_limited.try_reserve("first", [1, 2]) is not None
    assert block_limited.try_reserve("block-full", [3]) is None
    assert block_limited.counters["rejected_block_capacity"] == 1


def test_overlapping_blocks_are_rejected_without_disturbing_owner() -> None:
    leases = registry()
    first = leases.try_reserve("first", [1, 2])
    assert first is not None
    assert leases.try_reserve("second", [2, 3]) is None
    assert leases.leased_blocks == 2
    assert leases.counters["rejected_overlap"] == 1


def test_cancel_before_submit_releases_but_armed_cancel_is_forbidden() -> None:
    leases = registry()
    unsubmitted = leases.try_reserve("unsubmitted", [1])
    assert unsubmitted is not None
    unsubmitted.cancel_before_submit()
    assert not leases.has_pending()

    submitted = leases.try_reserve("submitted", [2])
    assert submitted is not None
    submitted.submit(FakeFence)
    with pytest.raises(LeaseStateError, match="must be drained"):
        submitted.cancel_before_submit()
    assert leases.has_pending("submitted")


def test_abort_synchronizes_armed_and_releases_reserved_leases() -> None:
    leases = registry()
    armed = leases.try_reserve("req", [1, 2])
    reserved = leases.try_reserve("req", [3])
    assert armed is not None and reserved is not None
    fence = FakeFence()
    armed.submit(lambda: fence)

    assert leases.abort_request("req") == 2
    assert fence.sync_calls == 1
    assert not leases.has_pending()
    with pytest.raises(LeaseStateError, match="no longer active"):
        reserved.submit(FakeFence)


def test_failed_query_or_synchronize_never_releases_blocks() -> None:
    leases = registry()
    handle = leases.try_reserve("req", [1])
    assert handle is not None
    fence = FakeFence(query_error=True, sync_error=True)
    handle.submit(lambda: fence)

    with pytest.raises(LeaseFenceError, match="status unknown"):
        leases.poll()
    assert leases.leased_blocks == 1

    with pytest.raises(LeaseFenceError, match="cannot safely abort"):
        leases.abort_request("req")
    assert leases.leased_blocks == 1


def test_abort_all_drains_every_request() -> None:
    leases = registry(leases=4)
    one = leases.try_reserve("one", [1])
    two = leases.try_reserve("two", [2])
    assert one is not None and two is not None
    one.submit(FakeFence)
    two.submit(lambda: FakeFence(done=True))

    assert leases.abort_all() == 2
    assert not leases.has_pending()


@pytest.mark.parametrize(
    "request_id,block_ids,message",
    [
        ("", [1], "request_id"),
        ("req", [], "block_ids"),
        ("req", [1, 1], "unique"),
        ("req", [-1], "non-negative"),
    ],
)
def test_invalid_reservations_are_programmer_errors(
    request_id: str, block_ids: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        registry().try_reserve(request_id, block_ids)


def test_unknown_delayed_request_is_immediately_releasable() -> None:
    leases = registry()
    assert leases.take_finished({"abandoned-before-submit"}) == {
        "abandoned-before-submit"
    }


def test_preemption_cannot_release_during_submit_to_fence_transition() -> None:
    leases = registry()
    handle = leases.try_reserve("req", [1, 2])
    assert handle is not None
    submit_entered = threading.Event()
    allow_fence_publication = threading.Event()
    preemption_started = threading.Event()
    events: list[str] = []
    fence = FakeFence()

    def submitter() -> FakeFence:
        events.append("cuda-submitted")
        submit_entered.set()
        assert allow_fence_publication.wait(timeout=5)
        events.append("fence-published")
        return fence

    submit_thread = threading.Thread(target=lambda: handle.submit(submitter))
    submit_thread.start()
    assert submit_entered.wait(timeout=5)

    def preempt() -> None:
        preemption_started.set()
        leases.abort_request("req")
        events.append("preemption-returned")

    preempt_thread = threading.Thread(target=preempt)
    preempt_thread.start()
    assert preemption_started.wait(timeout=5)
    time.sleep(0.02)
    assert preempt_thread.is_alive()

    allow_fence_publication.set()
    submit_thread.join(timeout=5)
    preempt_thread.join(timeout=5)

    assert not submit_thread.is_alive()
    assert not preempt_thread.is_alive()
    assert fence.sync_calls == 1
    assert events == [
        "cuda-submitted",
        "fence-published",
        "preemption-returned",
    ]
    assert leases.leased_blocks == 0


def test_submitter_failure_retains_blocks_as_unknown_and_is_fatal() -> None:
    leases = registry()
    handle = leases.try_reserve("req", [7])
    assert handle is not None

    def failed_after_possible_submit() -> FakeFence:
        raise RuntimeError("enqueue may have happened")

    with pytest.raises(LeaseFenceError, match="submission status unknown"):
        handle.submit(failed_after_possible_submit)

    assert leases.leased_blocks == 1
    with pytest.raises(LeaseFenceError, match="unknown submission"):
        leases.abort_request("req")
    assert leases.leased_blocks == 1
