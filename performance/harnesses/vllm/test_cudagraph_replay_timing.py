from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import cudagraph_replay_timing as timing


@dataclass
class FakeEvent:
    elapsed_ms: float
    ready: bool = True
    recorded_stream: object | None = None

    def record(self, stream: object) -> None:
        self.recorded_stream = stream

    def query(self) -> bool:
        return self.ready

    def elapsed_time(self, other: "FakeEvent") -> float:
        del other
        return self.elapsed_ms


class EventFactory:
    def __init__(self, durations: list[float]) -> None:
        self._durations = iter(durations)
        self.events: list[FakeEvent] = []

    def __call__(self) -> FakeEvent:
        event = FakeEvent(next(self._durations))
        self.events.append(event)
        return event


def test_unarmed_collector_does_not_allocate_events(tmp_path: Path) -> None:
    factory = EventFactory([1.0, 1.0])
    collector = timing.ReplayTimingCollector(
        event_factory=factory,
        arm_path=tmp_path / "absent.arm",
        sample_limit=2,
    )

    assert collector.measure("Q5", object(), lambda: 17) == 17
    assert factory.events == []
    assert collector.snapshot()["reserved"] == 0


def test_completed_events_are_aggregated_by_descriptor(
    tmp_path: Path,
) -> None:
    arm = tmp_path / "timing.arm"
    arm.touch()
    factory = EventFactory([2.0, 2.0, 4.0, 4.0])
    collector = timing.ReplayTimingCollector(
        event_factory=factory,
        arm_path=arm,
        sample_limit=4,
    )
    stream = object()

    collector.measure("Q5", stream, lambda: None)
    collector.measure("Q5", stream, lambda: None)
    snapshot = collector.snapshot()

    assert snapshot["completed"] == 2
    assert snapshot["pending"] == 0
    assert snapshot["total_completed_ms"] == 6.0
    assert snapshot["descriptors"]["Q5"] == {
        "count": 2,
        "total_ms": 6.0,
        "mean_ms": 3.0,
        "p50_ms": 2.0,
        "p90_ms": 4.0,
        "max_ms": 4.0,
    }
    assert all(event.recorded_stream is stream for event in factory.events)


def test_snapshot_never_synchronizes_pending_event(tmp_path: Path) -> None:
    arm = tmp_path / "timing.arm"
    arm.touch()
    factory = EventFactory([3.0, 3.0])
    collector = timing.ReplayTimingCollector(
        event_factory=factory,
        arm_path=arm,
        sample_limit=1,
    )
    collector.measure("Q1", object(), lambda: None)
    factory.events[-1].ready = False

    assert collector.snapshot()["pending"] == 1
    factory.events[-1].ready = True
    assert collector.snapshot()["completed"] == 1


def test_sample_limit_drops_without_changing_operation(tmp_path: Path) -> None:
    arm = tmp_path / "timing.arm"
    arm.touch()
    factory = EventFactory([1.0, 1.0])
    collector = timing.ReplayTimingCollector(
        event_factory=factory,
        arm_path=arm,
        sample_limit=1,
    )

    assert collector.measure("Q1", object(), lambda: "first") == "first"
    assert collector.measure("Q1", object(), lambda: "second") == "second"
    snapshot = collector.snapshot()
    assert snapshot["reserved"] == 1
    assert snapshot["dropped"] == 1


def test_descriptor_key_separates_q1_and_q5() -> None:
    full = SimpleNamespace(name="FULL")
    q1 = SimpleNamespace(
        cg_mode=full,
        num_tokens=1,
        num_reqs=1,
        num_active_loras=0,
    )
    q5 = SimpleNamespace(
        cg_mode=full,
        num_tokens=5,
        num_reqs=1,
        num_active_loras=0,
    )

    assert timing._descriptor_key(q1) != timing._descriptor_key(q5)


def test_descriptor_key_supports_optional_glm53_fields() -> None:
    descriptor = SimpleNamespace(
        cg_mode=SimpleNamespace(name="FULL"),
        num_tokens=8,
        num_reqs=1,
        uniform_token_count=None,
        max_query_len=8,
        num_active_loras=0,
    )

    assert timing._descriptor_key(descriptor) == (
        "mode=FULL,num_tokens=8,num_reqs=1,"
        "uniform_token_count=none,max_query_len=8,num_active_loras=0"
    )


@pytest.mark.parametrize("sample_limit", [0, 65537])
def test_collector_rejects_unbounded_sample_counts(
    tmp_path: Path,
    sample_limit: int,
) -> None:
    with pytest.raises(ValueError, match="sample_limit"):
        timing.ReplayTimingCollector(
            event_factory=lambda: FakeEvent(1.0),
            arm_path=tmp_path / "arm",
            sample_limit=sample_limit,
        )
