from __future__ import annotations

import random
import types
from pathlib import Path

from sparkcache.streaming.block_lease import (
    BlockLeaseRegistry,
    LeaseCapacity,
)
from sparkcache.streaming.factory import (
    ProductionStreamingSettings,
    WorkerStreamingSnapshotAdapter,
)


class _Fence:
    def __init__(self) -> None:
        self.done = False

    def query(self) -> bool:
        return self.done

    def synchronize(self) -> None:
        self.done = True


class _PreemptingRuntime:
    def __init__(self, leases: BlockLeaseRegistry) -> None:
        self._leases = leases
        self.preempted: list[str] = []

    def preempt(self, request_id: str) -> bool:
        self.preempted.append(request_id)
        self._leases.abort_request(request_id)
        return True


def _adapter() -> WorkerStreamingSnapshotAdapter:
    adapter = WorkerStreamingSnapshotAdapter(
        types.SimpleNamespace(),
        settings=ProductionStreamingSettings(
            native_library_path=Path("/opt/spark/lib/libspcc_snapshot.so"),
            native_library_sha256="a" * 64,
        ),
    )
    leases = BlockLeaseRegistry(
        LeaseCapacity(max_active_leases=64, max_leased_blocks=64)
    )
    adapter._runtime = _PreemptingRuntime(leases)
    adapter._leases = leases
    adapter._bound = True
    # The tests exercise only finished-ID ownership. The production poll also
    # drains runtime publication state, which is orthogonal to this contract.
    adapter.poll = leases.poll
    return adapter


def _arm(
    adapter: WorkerStreamingSnapshotAdapter,
    request_id: str,
    block_id: int,
) -> _Fence:
    adapter._seen_requests.add(request_id)
    fence = _Fence()
    lease = adapter._leases.try_reserve(request_id, (block_id,))
    assert lease is not None
    lease.submit(lambda: fence)
    return fence


def _assert_request_clean(
    adapter: WorkerStreamingSnapshotAdapter,
    request_id: str,
) -> None:
    assert request_id not in adapter._seen_requests
    assert request_id not in adapter._pending_finished_requests
    assert request_id not in adapter._admitted_requests
    assert request_id not in adapter._suppressed_requests
    assert request_id not in adapter._contexts
    assert request_id not in adapter._last_watermark_at
    assert not adapter._leases.has_pending(request_id)


def test_finished_ids_are_filtered_deduplicated_and_cleaned_exactly_once() -> None:
    adapter = _adapter()
    adapter._seen_requests.update({"seen-a", "seen-b", "aborted-no-lease"})
    adapter._suppressed_requests.add("aborted-no-lease")
    adapter._contexts["aborted-no-lease"] = ("b" * 64, 4096, 1.0)
    adapter._last_watermark_at["aborted-no-lease"] = 2.0

    batches = (
        {"tiny-unseen", "seen-b"},
        {"seen-a", "seen-b"},
        {"aborted-no-lease", "seen-a", "tiny-unseen"},
        {"aborted-no-lease", "seen-b"},
    )
    completions = [adapter.take_finished(batch) for batch in batches]

    assert completions == [
        {"seen-b"},
        {"seen-a"},
        {"aborted-no-lease"},
        set(),
    ]
    assert "tiny-unseen" not in set().union(*completions)
    for request_id in ("seen-a", "seen-b", "aborted-no-lease"):
        _assert_request_clean(adapter, request_id)


def test_active_lease_survives_repeated_finished_polls_then_completes_once() -> None:
    adapter = _adapter()
    fence = _arm(adapter, "leased", 7)

    assert adapter.take_finished({"leased"}) == set()
    assert adapter.take_finished(set()) == set()
    assert adapter.take_finished({"leased"}) == set()
    assert adapter._pending_finished_requests == {"leased"}

    fence.done = True

    assert adapter.take_finished(set()) == {"leased"}
    assert adapter.take_finished({"leased"}) == set()
    assert adapter.take_finished(set()) == set()
    _assert_request_clean(adapter, "leased")


def test_preemption_retires_stale_finished_generation_without_echoing_it() -> None:
    adapter = _adapter()
    _arm(adapter, "resumed", 11)
    adapter._contexts["resumed"] = ("c" * 64, 4096, 1.0)
    adapter._last_watermark_at["resumed"] = 2.0
    adapter._admitted_requests.add("resumed")

    assert adapter.take_finished({"resumed"}) == set()
    assert adapter._pending_finished_requests == {"resumed"}

    adapter.handle_preemptions(
        types.SimpleNamespace(preempted_request_ids=("resumed",))
    )

    # Scheduler-side preemption already retired this delayed-free generation.
    # Worker-side completion must not later echo its stale ID.
    assert adapter.take_finished(set()) == set()
    assert adapter.take_finished({"resumed"}) == set()
    _assert_request_clean(adapter, "resumed")

    # The same public request ID may begin a new generation after resume.
    resumed_fence = _arm(adapter, "resumed", 12)
    assert adapter.take_finished({"resumed"}) == set()
    resumed_fence.done = True
    assert adapter.take_finished(set()) == {"resumed"}
    _assert_request_clean(adapter, "resumed")


def test_seeded_finished_id_fuzz_preserves_ownership_and_exactly_once() -> None:
    rng = random.Random(0x5A17CA)
    adapter = _adapter()
    seen = [f"seen-{index:02d}" for index in range(32)]
    unseen = [f"tiny-{index:02d}" for index in range(16)]
    adapter._seen_requests.update(seen)

    fences: dict[str, _Fence] = {}
    release_tick: dict[str, int] = {}
    for index, request_id in enumerate(seen):
        if index % 3 == 0:
            fences[request_id] = _arm(adapter, request_id, index)
            release_tick[request_id] = rng.randrange(2, 8)

    shuffled = seen.copy()
    rng.shuffle(shuffled)
    first_batches = [
        set(shuffled[offset : offset + 7])
        for offset in range(0, len(shuffled), 7)
    ]
    emitted: set[str] = set()
    for tick in range(10):
        for request_id, fence in fences.items():
            if release_tick[request_id] == tick:
                fence.done = True
        supplied = first_batches[tick] if tick < len(first_batches) else set()
        # Re-present random old/new IDs in a different batch order. vLLM may
        # duplicate finished IDs across adjacent polls, and unknown tiny IDs
        # are never part of this adapter's ownership domain.
        supplied.update(rng.sample(seen + unseen, 9))
        ready = adapter.take_finished(supplied)
        assert ready <= set(seen)
        assert ready.isdisjoint(emitted)
        emitted.update(ready)

    for fence in fences.values():
        fence.done = True
    for _ in range(3):
        ready = adapter.take_finished(set())
        assert ready.isdisjoint(emitted)
        emitted.update(ready)

    assert emitted == set(seen)
    assert not adapter._pending_finished_requests
    assert not adapter._seen_requests
    assert adapter._leases.active_leases == 0
    for request_id in seen:
        _assert_request_clean(adapter, request_id)
