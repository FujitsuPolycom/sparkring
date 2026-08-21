from __future__ import annotations

from dataclasses import dataclass

import pytest

from .phase_timing import (
    PhaseDescriptor,
    PhaseKind,
    PhaseTimingCollector,
    SnapshotMismatch,
    snapshot_delta,
)


@dataclass
class FakeEvent:
    elapsed_ms: float
    ready: bool = True
    records: int = 0
    queries: int = 0
    elapsed_calls: int = 0

    def record(self, stream: object) -> None:
        assert stream is not None
        self.records += 1

    def query(self) -> bool:
        self.queries += 1
        return self.ready

    def elapsed_time(self, end: "FakeEvent") -> float:
        assert end is not self
        self.elapsed_calls += 1
        return self.elapsed_ms


class EventFactory:
    def __init__(self, durations: list[float]) -> None:
        self._durations = iter(durations)
        self.events: list[FakeEvent] = []

    def __call__(self) -> FakeEvent:
        event = FakeEvent(next(self._durations))
        self.events.append(event)
        return event


class FakeNvtx:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def range_push(self, message: str) -> None:
        self.calls.append(f"push:{message}")

    def range_pop(self) -> None:
        self.calls.append("pop")


TARGET = PhaseDescriptor(PhaseKind.TARGET_FULL_GRAPH, "Q6")
DRAFT = PhaseDescriptor(PhaseKind.DRAFT_MULTISTEP_GRAPH, "Q1")


def collector(
    factory: EventFactory,
    capacity: int = 2,
    nvtx: FakeNvtx | None = None,
) -> PhaseTimingCollector:
    return PhaseTimingCollector(
        event_factory=factory,
        capacity=capacity,
        descriptors=(TARGET, DRAFT),
        nvtx=nvtx,
    )


def test_preallocates_every_event_and_unarmed_path_is_inert() -> None:
    factory = EventFactory([1.0] * 4)
    timing = collector(factory)
    assert len(factory.events) == 4

    assert timing.measure(TARGET, object(), lambda: 17) == 17
    assert len(factory.events) == 4
    assert all(event.records == 0 for event in factory.events)
    assert timing.snapshot()["reserved"] == 0


def test_descriptor_registry_can_finalize_after_manager_startup() -> None:
    factory = EventFactory([1.0, 0.0])
    timing = PhaseTimingCollector(
        event_factory=factory,
        capacity=1,
        descriptors=(),
    )
    with pytest.raises(RuntimeError, match="descriptor"):
        timing.arm("too-early")
    timing.register_descriptors((TARGET,))
    timing.arm("ready")
    with pytest.raises(RuntimeError, match="before an epoch"):
        timing.register_descriptors((DRAFT,))


def test_hot_path_only_records_and_balances_nvtx() -> None:
    factory = EventFactory([3.5, 9.0, 1.0, 1.0])
    nvtx = FakeNvtx()
    timing = collector(factory, nvtx=nvtx)
    timing.arm("request-1")

    assert timing.measure(TARGET, object(), lambda: "ok") == "ok"
    assert sum(event.records for event in factory.events) == 2
    assert sum(event.queries for event in factory.events) == 0
    assert sum(event.elapsed_calls for event in factory.events) == 0
    assert nvtx.calls == [f"push:{TARGET.key}", "pop"]


def test_nonblocking_drain_retains_not_ready_samples() -> None:
    factory = EventFactory([4.25, 0.0])
    timing = PhaseTimingCollector(
        event_factory=factory,
        capacity=1,
        descriptors=(TARGET,),
    )
    timing.arm("request-2")
    timing.measure(TARGET, object(), lambda: None)
    factory.events[1].ready = False

    first = timing.drain()
    assert first.completed == 0
    assert first.still_pending == 1
    factory.events[1].ready = True
    second = timing.drain()
    assert second.completed == 1
    snapshot = timing.snapshot()
    assert snapshot["descriptors"][TARGET.key]["total_ms"] == 4.25
    assert snapshot["samples"] == [
        {
            "sequence": 0,
            "descriptor": TARGET.key,
            "duration_ms": 4.25,
        }
    ]


def test_raw_samples_retain_reservation_order_across_partial_drains() -> None:
    factory = EventFactory([2.0, 0.0, 3.0, 0.0])
    timing = collector(factory)
    timing.arm("ordered")
    timing.measure(TARGET, object(), lambda: None)
    timing.measure(DRAFT, object(), lambda: None)
    factory.events[1].ready = False

    assert timing.drain().completed == 1
    assert timing.snapshot()["samples"] == [
        {
            "sequence": 1,
            "descriptor": DRAFT.key,
            "duration_ms": 3.0,
        }
    ]

    factory.events[1].ready = True
    assert timing.drain().completed == 1
    assert timing.snapshot()["samples"] == [
        {
            "sequence": 0,
            "descriptor": TARGET.key,
            "duration_ms": 2.0,
        },
        {
            "sequence": 1,
            "descriptor": DRAFT.key,
            "duration_ms": 3.0,
        },
    ]


def test_capacity_and_unregistered_drops_are_explicit() -> None:
    factory = EventFactory([1.0, 0.0])
    timing = PhaseTimingCollector(
        event_factory=factory,
        capacity=1,
        descriptors=(TARGET,),
    )
    unknown = PhaseDescriptor(PhaseKind.COLLECTIVE, "stock-all-reduce")
    timing.arm("request-3")

    timing.measure(unknown, object(), lambda: None)
    timing.measure(TARGET, object(), lambda: None)
    timing.measure(TARGET, object(), lambda: None)
    snapshot = timing.snapshot()
    assert snapshot["reserved"] == 1
    assert snapshot["dropped"] == {
        "capacity": 1,
        "unregistered_descriptor": 1,
    }


def test_operation_exception_propagates_and_end_event_records() -> None:
    factory = EventFactory([2.0, 0.0])
    timing = PhaseTimingCollector(
        event_factory=factory,
        capacity=1,
        descriptors=(TARGET,),
    )
    timing.arm("request-4")

    with pytest.raises(ZeroDivisionError):
        timing.measure(TARGET, object(), lambda: 1 / 0)
    assert factory.events[1].records == 1
    assert timing.snapshot()["pending"] == 1


def test_snapshot_delta_is_additive_and_epoch_checked() -> None:
    factory = EventFactory([2.5, 0.0, 5.0, 0.0])
    timing = collector(factory)
    timing.arm("request-5")
    before = timing.snapshot()
    timing.measure(TARGET, object(), lambda: None)
    timing.drain()
    after = timing.snapshot()

    delta = snapshot_delta(before, after)
    assert delta["reserved"] == 1
    assert delta["completed"] == 1
    assert delta["descriptors"][TARGET.key] == {
        "count": 1,
        "total_ms": 2.5,
    }
    changed = dict(after)
    changed["epoch"] = "other"
    with pytest.raises(SnapshotMismatch, match="epoch"):
        snapshot_delta(before, changed)


def test_reset_requires_disarm_and_a_fully_drained_epoch() -> None:
    factory = EventFactory([1.0, 0.0])
    timing = PhaseTimingCollector(
        event_factory=factory,
        capacity=1,
        descriptors=(TARGET,),
    )
    timing.arm("request-6")
    timing.measure(TARGET, object(), lambda: None)
    with pytest.raises(RuntimeError, match="disarm"):
        timing.reset()
    timing.disarm()
    with pytest.raises(RuntimeError, match="pending"):
        timing.reset()
    timing.drain()
    timing.reset()
    snapshot = timing.snapshot()
    assert snapshot["reserved"] == 0
    assert snapshot["samples"] == []


@pytest.mark.parametrize("capacity", [0, 65_537])
def test_capacity_is_bounded(capacity: int) -> None:
    with pytest.raises(ValueError, match="capacity"):
        PhaseTimingCollector(
            event_factory=lambda: FakeEvent(1.0),
            capacity=capacity,
            descriptors=(TARGET,),
        )
