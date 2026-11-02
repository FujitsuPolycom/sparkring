"""Bounded in-flight sharing for authenticated opaque-page base snapshots.

The coordinator retains one immutable base byte buffer only for a declared
request cohort. It does not discover cache entries, interpret page manifests,
apply private deltas, place GPU state, or retain data after every registered
member has acquired or abandoned the result.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field


class PageBaseReadError(ValueError):
    """A shared base read could not produce authenticated bytes."""


class PageBaseReadCancelled(PageBaseReadError):
    """The request left its registered base-read cohort."""


@dataclass(frozen=True)
class PageBaseReadEvidence:
    """Authenticated metadata that makes one base snapshot shareable."""

    identity_storage_key: str
    base_context_digest: str
    base_root_sha256: str
    base_root_kind: str
    layout_sha256: str
    base_block_counts: tuple[int, ...]
    base_boundary_tokens: int
    base_encoded_bytes: int


@dataclass(frozen=True)
class PageBaseReadFlightKey:
    """Process-local ownership plus authenticated persistent-base evidence."""

    worker_generation: str
    storage_mode: str
    evidence: PageBaseReadEvidence


@dataclass(frozen=True)
class PageBaseReadFlightSnapshot:
    """Bounded coordinator state for diagnostics and GPU-free tests."""

    active_flights: int
    registered_members: int
    retained_bytes: int
    counters: dict[str, int]


@dataclass(frozen=True)
class PageBaseCohortRegistration:
    """Members accepted for one base plus the request queued as reader."""

    member_ids: tuple[str, ...] = ()
    leader_request_id: str | None = None
    flight_state: str = "none"


@dataclass(frozen=True)
class PageBaseReadResult:
    """One immutable authenticated base representation and its byte charge.

    SparkCache CUDA page restore retains independently authenticated macro objects
    instead of materializing one contiguous snapshot.  The coordinator needs
    only their aggregate authenticated size to enforce the same flight bounds.
    """

    value: object
    encoded_bytes: int


@dataclass
class _Flight:
    key: PageBaseReadFlightKey
    members: set[str]
    state: str = "registered"
    leader_request_id: str | None = None
    designated_leader_request_id: str | None = None
    result: bytes | PageBaseReadResult | None = None
    error: str | None = None
    cancelled: set[str] = field(default_factory=set)
    participant_count: int = 0
    shared_acquisitions: int = 0
    read_started_ns: int = 0
    read_elapsed_ns: int = 0
    cancellation_count: int = 0


class PageBaseReadFlights:
    """Share one pending base read across one pre-registered request cohort."""

    def __init__(
        self,
        *,
        max_flights: int = 2,
        max_members: int = 16,
        max_bytes_per_flight: int = 1024**3,
        max_bytes_total: int = 2 * 1024**3,
    ) -> None:
        if (
            max_flights <= 0
            or max_members <= 1
            or max_bytes_per_flight <= 0
            or max_bytes_total < max_bytes_per_flight
        ):
            raise ValueError("page-base flight bounds must be positive")
        self._max_flights = max_flights
        self._max_members = max_members
        self._max_bytes_per_flight = max_bytes_per_flight
        self._max_bytes_total = max_bytes_total
        self._condition = threading.Condition()
        self._flights: dict[PageBaseReadFlightKey, _Flight] = {}
        self._request_keys: dict[str, PageBaseReadFlightKey] = {}
        self._closed = False
        self._counters: Counter[str] = Counter()
        self._summaries: list[dict[str, object]] = []

    def register_cohort(
        self,
        key: PageBaseReadFlightKey,
        request_ids: Iterable[str],
    ) -> PageBaseCohortRegistration:
        """Create a provisional reader flight or join a retained cohort."""

        ordered = tuple(dict.fromkeys(request_ids))
        if not ordered or any(not item for item in ordered):
            return PageBaseCohortRegistration()
        with self._condition:
            if self._closed:
                return PageBaseCohortRegistration()
            flight = self._flights.get(key)
            if flight is not None and flight.state == "error":
                self._counters["registration_after_rejection"] += len(ordered)
                return PageBaseCohortRegistration()
            if flight is None:
                declared_bytes = key.evidence.base_encoded_bytes
                if type(declared_bytes) is not int or declared_bytes <= 0:
                    self._counters["invalid_declared_byte_bypasses"] += len(ordered)
                    return PageBaseCohortRegistration()
                if declared_bytes > self._max_bytes_per_flight:
                    self._counters["per_flight_byte_limit_bypasses"] += len(ordered)
                    return PageBaseCohortRegistration()
                # A reader may assemble into bytearray before publication as
                # immutable bytes. Reserve both representations so the peak,
                # not only the retained result, stays within the total bound.
                reserved = sum(
                    2 * item.key.evidence.base_encoded_bytes
                    for item in self._flights.values()
                )
                if reserved + 2 * declared_bytes > self._max_bytes_total:
                    self._counters["total_byte_limit_bypasses"] += len(ordered)
                    return PageBaseCohortRegistration()
                if len(self._flights) >= self._max_flights:
                    self._counters["flight_limit_bypasses"] += len(ordered)
                    return PageBaseCohortRegistration()
                flight = _Flight(key=key, members=set())
                self._flights[key] = flight
                created = True
            else:
                created = False
            selected: list[str] = []
            for request_id in ordered:
                if request_id in self._request_keys:
                    continue
                if flight.participant_count >= self._max_members:
                    self._counters["member_limit_bypasses"] += 1
                    continue
                flight.members.add(request_id)
                self._request_keys[request_id] = key
                selected.append(request_id)
                flight.participant_count += 1
            if not selected:
                if created:
                    self._flights.pop(key, None)
                return PageBaseCohortRegistration()
            if flight.designated_leader_request_id is None:
                flight.designated_leader_request_id = next(
                    item for item in ordered if item in flight.members
                )
            if selected:
                self._counters["members_registered"] += len(selected)
                if flight.state in {"reading", "ready"}:
                    self._counters["late_members_registered"] += len(selected)
                self._counters["cohorts_registered"] += int(
                    len(flight.members) == len(selected)
                )
            return PageBaseCohortRegistration(
                member_ids=tuple(selected),
                leader_request_id=flight.designated_leader_request_id,
                flight_state=flight.state,
            )

    def registered_key(self, request_id: str) -> PageBaseReadFlightKey | None:
        with self._condition:
            return self._request_keys.get(request_id)

    def flight_state(self, key: PageBaseReadFlightKey) -> str | None:
        """Return process-local state without exposing retained base bytes."""

        with self._condition:
            flight = self._flights.get(key)
            return flight.state if flight is not None else None

    def resolve(
        self,
        request_id: str,
        key: PageBaseReadFlightKey,
        reader: Callable[[], bytes | bytearray | PageBaseReadResult],
        *,
        allow_independent: bool = True,
    ) -> bytes | PageBaseReadResult | None:
        """Resolve an admitted base, optionally reading unadmitted bases.

        Callers with a selective restore path set ``allow_independent=False``
        and receive ``None`` when admission is absent or evidence differs.
        This decision stays under the coordinator lock so an unadmitted read
        cannot allocate a complete base outside its memory reservation.
        """

        with self._condition:
            if self._closed:
                self._finish_locked(request_id)
                raise PageBaseReadCancelled("page-base coordinator is closed")
            registered_key = self._request_keys.get(request_id)
            flight = self._flights.get(key)
            if registered_key != key or flight is None or request_id not in flight.members:
                if registered_key is not None:
                    self._finish_locked(request_id)
                    self._counters["evidence_mismatch_bypasses"] += 1
                self._counters[
                    "independent_reads" if allow_independent else "selective_read_bypasses"
                ] += 1
                flight = None
                leader = False
            elif request_id in flight.cancelled or self._closed:
                self._finish_locked(request_id)
                raise PageBaseReadCancelled(
                    "page-base request left the registered cohort"
                )
            elif flight.state == "registered":
                flight.state = "reading"
                flight.leader_request_id = request_id
                flight.read_started_ns = time.perf_counter_ns()
                self._counters["base_reads_started"] += 1
                leader = True
            else:
                self._counters["followers_waited"] += 1
                leader = False

        if flight is None:
            return reader() if allow_independent else None
        if leader:
            try:
                readable = reader()
                if isinstance(readable, PageBaseReadResult):
                    if (
                        type(readable.encoded_bytes) is not int
                        or readable.encoded_bytes <= 0
                    ):
                        raise TypeError(
                            "page-base reader result has an invalid byte charge"
                        )
                    readable_bytes = readable.encoded_bytes
                    result = readable
                elif isinstance(readable, (bytes, bytearray)):
                    readable_bytes = len(readable)
                    result = bytes(readable)
                else:
                    raise TypeError(
                        "page-base reader must return bytes or an authenticated result"
                    )
                if readable_bytes != key.evidence.base_encoded_bytes:
                    raise PageBaseReadError(
                        "page-base reader length differs from authenticated geometry"
                    )
            except Exception as error:
                with self._condition:
                    if self._flights.get(key) is flight:
                        flight.read_elapsed_ns = (
                            time.perf_counter_ns() - flight.read_started_ns
                        )
                        flight.error = f"{type(error).__name__}: {error}"
                        flight.state = "error"
                        self._counters["base_reads_rejected"] += 1
                        self._condition.notify_all()
                    cancelled = request_id in flight.cancelled or self._closed
                    error_message = flight.error or "page-base read rejected"
                    self._finish_locked(request_id)
                if cancelled:
                    raise PageBaseReadCancelled(
                        "page-base leader left the registered cohort"
                    ) from error
                raise PageBaseReadError(error_message) from error
            with self._condition:
                if self._flights.get(key) is flight:
                    flight.read_elapsed_ns = (
                        time.perf_counter_ns() - flight.read_started_ns
                    )
                    flight.result = result
                    flight.state = "ready"
                    self._counters["base_reads_completed"] += 1
                    self._condition.notify_all()
                cancelled = request_id in flight.cancelled or self._closed
                shared = flight.result
                self._finish_locked(request_id)
            if cancelled:
                raise PageBaseReadCancelled(
                    "page-base leader left the registered cohort"
                )
            if shared is None:
                raise PageBaseReadError("page-base result was released before use")
            return shared

        with self._condition:
            self._condition.wait_for(
                lambda: (
                    flight.state in {"ready", "error"}
                    or request_id in flight.cancelled
                    or self._closed
                )
            )
            if request_id in flight.cancelled or self._closed:
                self._finish_locked(request_id)
                raise PageBaseReadCancelled(
                    "page-base follower left the registered cohort"
                )
            if flight.state == "error":
                error = flight.error or "page-base read rejected"
                self._finish_locked(request_id)
                raise PageBaseReadError(error)
            result = flight.result
            self._counters["shared_results_acquired"] += 1
            flight.shared_acquisitions += 1
            self._finish_locked(request_id)
        if result is None:
            raise PageBaseReadError("page-base result was released before use")
        return result

    def cancel(self, request_id: str) -> bool:
        """Wake one member without cancelling its leader or other followers."""

        with self._condition:
            key = self._request_keys.get(request_id)
            flight = self._flights.get(key) if key is not None else None
            if flight is None:
                return False
            if request_id not in flight.cancelled:
                flight.cancelled.add(request_id)
                flight.cancellation_count += 1
                self._counters["members_cancelled"] += 1
            self._condition.notify_all()
            return True

    def finish(self, request_id: str) -> None:
        """Remove a member whose load ended before or after base resolution."""

        with self._condition:
            self._finish_locked(request_id)

    def close(self) -> None:
        """Cancel members, release retained bytes, and reject later reads."""

        with self._condition:
            self._closed = True
            for flight in self._flights.values():
                newly_cancelled = flight.members - flight.cancelled
                flight.cancelled.update(flight.members)
                flight.cancellation_count += len(newly_cancelled)
                self._counters["members_cancelled"] += len(newly_cancelled)
                if flight.state != "error":
                    flight.state = "closed"
            for request_id in tuple(self._request_keys):
                self._finish_locked(request_id)
            self._condition.notify_all()

    def snapshot(self) -> PageBaseReadFlightSnapshot:
        with self._condition:
            return PageBaseReadFlightSnapshot(
                active_flights=len(self._flights),
                registered_members=len(self._request_keys),
                retained_bytes=sum(
                    (
                        flight.result.encoded_bytes
                        if isinstance(flight.result, PageBaseReadResult)
                        else len(flight.result)
                    )
                    for flight in self._flights.values()
                    if flight.result is not None
                ),
                counters=dict(self._counters),
            )

    def take_summaries(self) -> tuple[dict[str, object], ...]:
        """Return and clear one summary per released request cohort."""

        with self._condition:
            summaries = tuple(self._summaries)
            self._summaries.clear()
            return summaries

    def _finish_locked(self, request_id: str) -> None:
        key = self._request_keys.pop(request_id, None)
        if key is None:
            return
        flight = self._flights.get(key)
        if flight is None:
            return
        flight.members.discard(request_id)
        flight.cancelled.discard(request_id)
        if flight.members:
            return
        self._flights.pop(key, None)
        evidence = flight.key.evidence
        physical_reads = int(flight.read_started_ns > 0)
        if flight.state == "ready":
            outcome = "verified"
        elif flight.state == "error":
            outcome = "recompute"
        else:
            outcome = "cancelled"
        self._summaries.append(
            {
                "schema": "sparkcache-page-base-restore-flight/v1",
                "base_context_digest": evidence.base_context_digest,
                "base_root_sha256": evidence.base_root_sha256,
                "participants": flight.participant_count,
                "physical_base_reads": physical_reads,
                "base_bytes": (
                    (
                        flight.result.encoded_bytes
                        if isinstance(flight.result, PageBaseReadResult)
                        else len(flight.result)
                    )
                    if flight.result is not None
                    else (evidence.base_encoded_bytes if physical_reads else 0)
                ),
                "base_read_ms": round(flight.read_elapsed_ns / 1_000_000, 3),
                "avoided_base_reads": flight.shared_acquisitions,
                "cancelled_members": flight.cancellation_count,
                "outcome": outcome,
                "worker_generation": flight.key.worker_generation,
                "storage_mode": flight.key.storage_mode,
            }
        )
        flight.result = None
        flight.error = None
        self._counters["flights_released"] += 1
