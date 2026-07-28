"""Fixed-capacity, stream-ordered timing for the Q-2R phase census.

The serving path records preallocated CUDA events. It never queries an event,
computes elapsed time, synchronizes a stream/device, or grows a result
container. Completion polling and duration aggregation are explicit work for
``drain()`` after timed execution.

This module deliberately has no torch or vLLM import. Production code supplies
the event factory, CUDA stream, and optional NVTX implementation.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, TypeVar

_MAX_CAPACITY = 65_536
_MAX_DESCRIPTORS = 256
_MAX_NAME_CHARS = 160
_SCHEMA = "sparkring-q2r-phase-timing/v1"

T = TypeVar("T")


class Event(Protocol):
    """Smallest CUDA-event surface required by the collector."""

    def record(self, stream: Any) -> None: ...

    def query(self) -> bool: ...

    def elapsed_time(self, end_event: Any) -> float: ...


class Nvtx(Protocol):
    """Optional NVTX surface; disabled by default."""

    def range_push(self, message: str) -> Any: ...

    def range_pop(self) -> Any: ...


class PhaseKind(str, Enum):
    STEP_ENVELOPE = "step_envelope"
    TARGET_FULL_GRAPH = "target_full_graph"
    DRAFT_MULTISTEP_GRAPH = "draft_multistep_graph"
    DRAFT_GENERATION = "draft_generation"
    OTHER_GRAPH = "other_graph"
    EAGER_TRANSITION = "eager_transition"
    COLLECTIVE = "collective"


@dataclass(frozen=True, order=True)
class PhaseDescriptor:
    kind: PhaseKind
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PhaseKind):
            raise ValueError("kind must be a PhaseKind")
        if (
            not isinstance(self.name, str)
            or not self.name
            or len(self.name) > _MAX_NAME_CHARS
        ):
            raise ValueError(
                f"name must contain 1..{_MAX_NAME_CHARS} characters"
            )

    @property
    def key(self) -> str:
        return f"{self.kind.value}:{self.name}"


@dataclass
class _Slot:
    start: Event
    end: Event
    descriptor_index: int = -1
    pending: bool = False


@dataclass
class _Aggregate:
    count: int = 0
    total_ms: float = 0.0
    minimum_ms: float = math.inf
    maximum_ms: float = -math.inf

    def add(self, duration_ms: float) -> None:
        self.count += 1
        self.total_ms += duration_ms
        self.minimum_ms = min(self.minimum_ms, duration_ms)
        self.maximum_ms = max(self.maximum_ms, duration_ms)

    def snapshot(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "count": 0,
                "total_ms": 0.0,
                "mean_ms": None,
                "min_ms": None,
                "max_ms": None,
            }
        return {
            "count": self.count,
            "total_ms": round(self.total_ms, 6),
            "mean_ms": round(self.total_ms / self.count, 6),
            "min_ms": round(self.minimum_ms, 6),
            "max_ms": round(self.maximum_ms, 6),
        }


@dataclass(frozen=True)
class DrainResult:
    examined: int
    completed: int
    still_pending: int
    errors: int


class SnapshotMismatch(ValueError):
    """Raised when two snapshots cannot form one trustworthy delta."""


class PhaseTimingCollector:
    """One fixed-capacity timing epoch.

    Event pairs and descriptor aggregates are allocated in ``__init__``.
    ``measure`` performs only: bounded slot reservation, two event records,
    the wrapped operation, and optional NVTX push/pop. It does not recycle
    slots during an epoch. This makes overflow explicit and rules out an ABA
    race between a status/drain thread and a serving thread.
    """

    def __init__(
        self,
        *,
        event_factory: Callable[[], Event],
        capacity: int,
        descriptors: tuple[PhaseDescriptor, ...],
        nvtx: Nvtx | None = None,
    ) -> None:
        if (
            not isinstance(capacity, int)
            or isinstance(capacity, bool)
            or not 1 <= capacity <= _MAX_CAPACITY
        ):
            raise ValueError(f"capacity must be in [1, {_MAX_CAPACITY}]")
        if len(descriptors) > _MAX_DESCRIPTORS:
            raise ValueError(
                f"descriptors must contain 0..{_MAX_DESCRIPTORS} entries"
            )
        if len(set(descriptors)) != len(descriptors):
            raise ValueError("descriptors must be unique")

        # All event allocation happens before the recorder can be armed.
        self._slots = [
            _Slot(start=event_factory(), end=event_factory())
            for _ in range(capacity)
        ]
        # Filled only by drain(), never by the serving hot path. Keeping one
        # preallocated cell per event preserves exact reservation order even
        # when later events become ready before earlier events.
        self._sample_durations: list[float | None] = [None] * capacity
        self._capacity = capacity
        self._descriptors = descriptors
        self._descriptor_indexes = {
            descriptor: index
            for index, descriptor in enumerate(self._descriptors)
        }
        self._aggregates = [_Aggregate() for _ in descriptors]
        self._nvtx = nvtx
        self._lock = threading.Lock()
        self._armed = False
        self._epoch = ""
        self._next_slot = 0
        self._pending_count = 0
        self._completed = 0
        self._dropped_capacity = 0
        self._dropped_unregistered = 0
        self._record_errors = 0
        self._drain_errors = 0
        self._nvtx_errors = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def register_descriptors(
        self, descriptors: tuple[PhaseDescriptor, ...]
    ) -> None:
        """Extend the fixed registry before an epoch starts.

        This supports graph managers whose stable startup IDs are known only
        after construction. It is forbidden once armed or once any slot in
        the current epoch has been reserved.
        """
        if not descriptors or len(set(descriptors)) != len(descriptors):
            raise ValueError("new descriptors must be nonempty and unique")
        with self._lock:
            if self._armed or self._next_slot or self._completed:
                raise RuntimeError(
                    "descriptors can only be registered before an epoch"
                )
            if any(
                descriptor in self._descriptor_indexes
                for descriptor in descriptors
            ):
                raise ValueError("descriptor is already registered")
            combined = self._descriptors + descriptors
            if len(combined) > _MAX_DESCRIPTORS:
                raise ValueError(
                    f"descriptor count exceeds {_MAX_DESCRIPTORS}"
                )
            first_index = len(self._descriptors)
            self._descriptors = combined
            for offset, descriptor in enumerate(descriptors):
                self._descriptor_indexes[descriptor] = first_index + offset
                self._aggregates.append(_Aggregate())

    def arm(self, epoch: str) -> None:
        if (
            not isinstance(epoch, str)
            or not epoch
            or len(epoch) > _MAX_NAME_CHARS
        ):
            raise ValueError(
                f"epoch must contain 1..{_MAX_NAME_CHARS} characters"
            )
        with self._lock:
            if self._armed:
                raise RuntimeError("collector is already armed")
            if not self._descriptors:
                raise RuntimeError("at least one descriptor is required")
            if self._next_slot or self._pending_count or self._completed:
                raise RuntimeError("reset is required before a new epoch")
            self._epoch = epoch
            self._armed = True

    def disarm(self) -> None:
        with self._lock:
            self._armed = False

    def measure(
        self,
        descriptor: PhaseDescriptor,
        stream: Any,
        operation: Callable[[], T],
    ) -> T:
        """Measure one operation without querying or synchronizing CUDA."""
        slot: _Slot | None = None
        with self._lock:
            if self._armed:
                descriptor_index = self._descriptor_indexes.get(descriptor)
                if descriptor_index is None:
                    self._dropped_unregistered += 1
                elif self._next_slot >= self._capacity:
                    self._dropped_capacity += 1
                else:
                    slot = self._slots[self._next_slot]
                    self._next_slot += 1
                    slot.descriptor_index = descriptor_index
        if slot is None:
            return operation()

        try:
            slot.start.record(stream)
        except Exception:
            with self._lock:
                self._record_errors += 1
            return operation()

        nvtx_pushed = False
        if self._nvtx is not None:
            try:
                self._nvtx.range_push(descriptor.key)
                nvtx_pushed = True
            except Exception:
                with self._lock:
                    self._nvtx_errors += 1

        try:
            return operation()
        finally:
            if nvtx_pushed:
                try:
                    assert self._nvtx is not None
                    self._nvtx.range_pop()
                except Exception:
                    with self._lock:
                        self._nvtx_errors += 1
            try:
                slot.end.record(stream)
            except Exception:
                with self._lock:
                    self._record_errors += 1
            else:
                with self._lock:
                    slot.pending = True
                    self._pending_count += 1

    def drain(self) -> DrainResult:
        """Poll and aggregate ready events without synchronizing.

        Call this only from a low-rate reporter or after measured execution.
        ``query`` can return false; such samples remain pending. This method
        never calls ``synchronize`` on an event, stream, or device.
        """
        completed_now = 0
        errors_now = 0
        with self._lock:
            reserved = self._next_slot

        completed_durations: list[tuple[int, int, float]] = []
        examined = 0
        for sequence, slot in enumerate(self._slots[:reserved]):
            if not slot.pending:
                continue
            examined += 1
            try:
                if not slot.end.query():
                    continue
                duration_ms = float(slot.start.elapsed_time(slot.end))
                if not math.isfinite(duration_ms) or duration_ms < 0:
                    raise ValueError("invalid CUDA event duration")
            except Exception:
                errors_now += 1
                slot.pending = False
                continue
            completed_durations.append(
                (sequence, slot.descriptor_index, duration_ms)
            )
            slot.pending = False
            completed_now += 1

        with self._lock:
            for sequence, descriptor_index, duration_ms in completed_durations:
                self._aggregates[descriptor_index].add(duration_ms)
                self._sample_durations[sequence] = duration_ms
            self._completed += completed_now
            self._drain_errors += errors_now
            self._pending_count -= completed_now + errors_now
            still_pending = self._pending_count
        return DrainResult(
            examined=examined,
            completed=completed_now,
            still_pending=still_pending,
            errors=errors_now,
        )

    def snapshot(self) -> dict[str, Any]:
        """Copy counters only; deliberately does not poll CUDA events."""
        with self._lock:
            descriptor_metrics = {
                descriptor.key: self._aggregates[index].snapshot()
                for index, descriptor in enumerate(self._descriptors)
            }
            samples = [
                {
                    "sequence": sequence,
                    "descriptor": self._descriptors[
                        self._slots[sequence].descriptor_index
                    ].key,
                    "duration_ms": round(duration_ms, 6),
                }
                for sequence, duration_ms in enumerate(
                    self._sample_durations[: self._next_slot]
                )
                if duration_ms is not None
            ]
            return {
                "schema": _SCHEMA,
                "epoch": self._epoch,
                "armed": self._armed,
                "capacity": self._capacity,
                "reserved": self._next_slot,
                "completed": self._completed,
                "pending": self._pending_count,
                "dropped": {
                    "capacity": self._dropped_capacity,
                    "unregistered_descriptor": self._dropped_unregistered,
                },
                "errors": {
                    "record": self._record_errors,
                    "drain": self._drain_errors,
                    "nvtx": self._nvtx_errors,
                },
                "descriptors": descriptor_metrics,
                "samples": samples,
            }

    def reset(self) -> None:
        """Clear a disarmed, fully drained epoch for event-pair reuse."""
        with self._lock:
            if self._armed:
                raise RuntimeError("disarm before reset")
            if self._pending_count:
                raise RuntimeError("all pending events must drain before reset")
            for slot in self._slots:
                slot.descriptor_index = -1
                slot.pending = False
            for index in range(self._capacity):
                self._sample_durations[index] = None
            self._aggregates = [
                _Aggregate() for _descriptor in self._descriptors
            ]
            self._epoch = ""
            self._next_slot = 0
            self._pending_count = 0
            self._completed = 0
            self._dropped_capacity = 0
            self._dropped_unregistered = 0
            self._record_errors = 0
            self._drain_errors = 0
            self._nvtx_errors = 0


def _counter_delta(
    before: Mapping[str, int], after: Mapping[str, int]
) -> dict[str, int]:
    keys = set(before) | set(after)
    result = {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in sorted(keys)
    }
    if any(value < 0 for value in result.values()):
        raise SnapshotMismatch("snapshot counters moved backwards")
    return result


def snapshot_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Build an exact additive delta from two snapshots in one epoch."""
    for field in ("schema", "epoch", "capacity"):
        if before.get(field) != after.get(field):
            raise SnapshotMismatch(f"snapshot {field} differs")
    counters = {}
    for field in ("reserved", "completed", "pending"):
        counters[field] = int(after[field]) - int(before[field])
    if counters["reserved"] < 0 or counters["completed"] < 0:
        raise SnapshotMismatch("snapshot counters moved backwards")

    descriptors: dict[str, dict[str, float | int]] = {}
    before_descriptors = before.get("descriptors", {})
    after_descriptors = after.get("descriptors", {})
    if set(before_descriptors) != set(after_descriptors):
        raise SnapshotMismatch("snapshot descriptors differ")
    for key in sorted(before_descriptors):
        old = before_descriptors[key]
        new = after_descriptors[key]
        count = int(new["count"]) - int(old["count"])
        total_ms = float(new["total_ms"]) - float(old["total_ms"])
        if count < 0 or total_ms < -1e-6:
            raise SnapshotMismatch("descriptor metrics moved backwards")
        descriptors[key] = {
            "count": count,
            "total_ms": round(max(0.0, total_ms), 6),
        }
    return {
        "schema": before["schema"],
        "epoch": before["epoch"],
        "capacity": before["capacity"],
        **counters,
        "dropped": _counter_delta(
            before.get("dropped", {}), after.get("dropped", {})
        ),
        "errors": _counter_delta(
            before.get("errors", {}), after.get("errors", {})
        ),
        "descriptors": descriptors,
    }
