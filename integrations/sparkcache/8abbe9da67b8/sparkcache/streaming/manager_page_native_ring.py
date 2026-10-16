"""Owned Python interface for asynchronous CUDA manager-page capture.

Importing this module does not load CUDA. The attested ctypes backend is
constructed only by :meth:`NativeManagerPageRing.from_attested`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from os import PathLike
from typing import Protocol, Sequence

from .manager_page_capture import (
    ManagerPageCapturePlan,
    ManagerPageSource,
    plan_manager_page_capture,
)
from .native_ring import (
    NativeRingConfig,
    NativeSnapshotRingStateError,
    NativeSnapshotRingStatusError,
    NativeStatus,
    RawReadyView,
    RawTicket,
)


class ManagerPageRingBackend(Protocol):
    def create(self, config: NativeRingConfig) -> int: ...

    def configure_page_sources(
        self,
        sources: tuple[ManagerPageSource, ...],
        group_count: int,
    ) -> int: ...

    def submit_pages(
        self,
        *,
        context_sequence: int,
        logical_start: int,
        groups: tuple[tuple[int, ...], ...],
        producer_stream: int,
        used_bytes: int,
    ) -> tuple[int, RawTicket | None]: ...

    def poll(self, ticket: RawTicket) -> tuple[int, RawReadyView | None]: ...

    def claim(self, ticket: RawTicket) -> tuple[int, RawReadyView | None]: ...

    def release(self, ticket: RawTicket) -> int: ...

    def abandon_context(self, context_sequence: int) -> int: ...

    def drain_context(self, context_sequence: int) -> int: ...

    def shutdown(self) -> int: ...

    def destroy(self) -> None: ...

    def status_text(self, status: int) -> str: ...


@dataclass(frozen=True, slots=True)
class ManagerPageTicket:
    generation: int
    slot_index: int
    context_sequence: int
    logical_start: int
    plan: ManagerPageCapturePlan
    _owner: object


class ManagerPageView:
    """Read-only ring bytes valid until release or context drain."""

    __slots__ = ("ticket", "_payload", "_valid")

    def __init__(self, ticket: ManagerPageTicket, payload: memoryview) -> None:
        self.ticket = ticket
        self._payload = payload
        self._valid = True

    @property
    def payload(self) -> memoryview:
        if not self._valid:
            raise NativeSnapshotRingStateError(
                "manager-page capture view is no longer valid"
            )
        return self._payload

    def _invalidate(self) -> None:
        if self._valid:
            self._valid = False
            self._payload.release()


@dataclass(slots=True)
class _OwnedPageTicket:
    public: ManagerPageTicket
    raw: RawTicket
    ready_view: ManagerPageView | None = None
    claimed_view: ManagerPageView | None = None


class NativeManagerPageRing:
    """Generation-checked owner of one manager-page capture handle."""

    def __init__(
        self,
        config: NativeRingConfig,
        *,
        backend: ManagerPageRingBackend,
    ) -> None:
        self.config = config
        self._backend = backend
        self._lock = threading.RLock()
        self._owner = object()
        self._sources: tuple[ManagerPageSource, ...] | None = None
        self._group_count = 0
        self._tickets: dict[tuple[int, int], _OwnedPageTicket] = {}
        self._closed = False
        status = int(backend.create(config))
        if status != NativeStatus.OK:
            self._raise_status("create", status)

    @classmethod
    def from_attested(
        cls,
        config: NativeRingConfig,
        *,
        library_path: str | PathLike[str],
        expected_sha256: str,
    ) -> "NativeManagerPageRing":
        from .manager_page_native_ring_ctypes import CtypesManagerPageRingBackend

        return cls(
            config,
            backend=CtypesManagerPageRingBackend(
                library_path,
                expected_sha256=expected_sha256,
            ),
        )

    @property
    def active_ticket_count(self) -> int:
        with self._lock:
            return len(self._tickets)

    def configure_sources(
        self,
        sources: Sequence[ManagerPageSource],
        *,
        group_count: int,
    ) -> None:
        inventory = tuple(sources)
        if not isinstance(group_count, int) or isinstance(group_count, bool):
            raise ValueError("group_count must be an integer")
        if group_count <= 0:
            raise ValueError("group_count must be positive")
        # A one-page request reaches every inventory validation rule without
        # allocating or touching a live tensor.
        plan_manager_page_capture(
            inventory,
            tuple((0,) for _ in range(group_count)),
            slot_bytes=self.config.slot_bytes,
        )
        with self._lock:
            self._require_open()
            if self._sources is not None:
                if inventory == self._sources and group_count == self._group_count:
                    return
                raise NativeSnapshotRingStateError(
                    "manager-page sources are already configured"
                )
            status = int(
                self._backend.configure_page_sources(inventory, group_count)
            )
            if status != NativeStatus.OK:
                self._raise_status("configure_page_sources", status)
            self._sources = inventory
            self._group_count = group_count

    def submit(
        self,
        *,
        context_sequence: int,
        logical_start: int,
        physical_pages_by_group: Sequence[Sequence[int]],
        producer_stream: int,
    ) -> ManagerPageTicket | None:
        if context_sequence <= 0:
            raise ValueError("context_sequence must be positive")
        if logical_start < 0:
            raise ValueError("logical_start must be nonnegative")
        if (
            not isinstance(producer_stream, int)
            or isinstance(producer_stream, bool)
            or not 0 <= producer_stream <= 0xFFFFFFFFFFFFFFFF
        ):
            raise ValueError("producer_stream must be a nonnegative handle")
        groups = tuple(tuple(group) for group in physical_pages_by_group)
        with self._lock:
            self._require_open()
            if self._sources is None:
                raise NativeSnapshotRingStateError(
                    "manager-page sources must be configured before submit"
                )
            if len(groups) != self._group_count:
                raise ValueError("request group count differs from source inventory")
            plan = plan_manager_page_capture(
                self._sources,
                groups,
                slot_bytes=self.config.slot_bytes,
            )
            status, raw = self._backend.submit_pages(
                context_sequence=context_sequence,
                logical_start=logical_start,
                groups=groups,
                producer_stream=producer_stream,
                used_bytes=plan.used_bytes,
            )
            status = int(status)
            if status in (NativeStatus.WOULD_BLOCK, NativeStatus.DROPPED):
                if raw is not None:
                    raise NativeSnapshotRingStateError(
                        "dropped manager-page submission returned a ticket"
                    )
                return None
            if status != NativeStatus.OK or raw is None:
                self._raise_status("submit_pages", status)
            key = (raw.slot_index, raw.generation)
            if (
                raw.generation <= 0
                or not 0 <= raw.slot_index < self.config.slot_count
                or key in self._tickets
            ):
                drain_status = int(
                    self._backend.drain_context(context_sequence)
                )
                if drain_status != NativeStatus.OK:
                    self._raise_status("drain_invalid_ticket", drain_status)
                raise NativeSnapshotRingStateError(
                    "manager-page submission returned an invalid ticket"
                )
            public = ManagerPageTicket(
                generation=raw.generation,
                slot_index=raw.slot_index,
                context_sequence=context_sequence,
                logical_start=logical_start,
                plan=plan,
                _owner=self._owner,
            )
            self._tickets[key] = _OwnedPageTicket(public, raw)
            return public

    def poll(self, ticket: ManagerPageTicket) -> ManagerPageView | None:
        with self._lock:
            owned = self._require_ticket(ticket)
            if owned.ready_view is not None:
                return owned.ready_view
            status, raw = self._backend.poll(owned.raw)
            if int(status) == NativeStatus.NOT_READY:
                return None
            if int(status) != NativeStatus.OK or raw is None:
                self._raise_status("poll", int(status))
            owned.ready_view = self._validated_view(ticket, raw, expected_state=2)
            return owned.ready_view

    def claim(self, ticket: ManagerPageTicket) -> ManagerPageView:
        with self._lock:
            owned = self._require_ticket(ticket)
            if owned.claimed_view is not None:
                return owned.claimed_view
            status, raw = self._backend.claim(owned.raw)
            if int(status) != NativeStatus.OK or raw is None:
                self._raise_status("claim", int(status))
            if owned.ready_view is not None:
                owned.ready_view._invalidate()
                owned.ready_view = None
            owned.claimed_view = self._validated_view(
                ticket, raw, expected_state=3
            )
            return owned.claimed_view

    def release(self, ticket: ManagerPageTicket) -> None:
        with self._lock:
            owned = self._require_ticket(ticket)
            status = int(self._backend.release(owned.raw))
            if status != NativeStatus.OK:
                self._raise_status("release", status)
            self._retire(owned)

    def drain_context(self, context_sequence: int) -> None:
        with self._lock:
            self._require_open()
            status = int(self._backend.drain_context(context_sequence))
            if status != NativeStatus.OK:
                self._raise_status("drain_context", status)
            for owned in tuple(self._tickets.values()):
                if owned.public.context_sequence == context_sequence:
                    if owned.claimed_view is not None:
                        release_status = int(self._backend.release(owned.raw))
                        if release_status != NativeStatus.OK:
                            self._raise_status(
                                "release_drained_writer", release_status
                            )
                    self._retire(owned)

    def abandon_context(self, context_sequence: int) -> None:
        with self._lock:
            self._require_open()
            status = int(self._backend.abandon_context(context_sequence))
            if status != NativeStatus.OK:
                self._raise_status("abandon_context", status)

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            status = int(self._backend.shutdown())
            if status != NativeStatus.OK:
                self._raise_status("shutdown", status)
            for owned in tuple(self._tickets.values()):
                self._retire(owned)
            self._backend.destroy()
            self._closed = True

    def _validated_view(
        self,
        ticket: ManagerPageTicket,
        raw: RawReadyView,
        *,
        expected_state: int,
    ) -> ManagerPageView:
        if (
            raw.context_sequence != ticket.context_sequence
            or raw.logical_start != ticket.logical_start
            or raw.generation != ticket.generation
            or raw.slot_index != ticket.slot_index
            or raw.capacity_bytes != self.config.slot_bytes
            or raw.used_bytes != ticket.plan.used_bytes
            or raw.row_count != sum(ticket.plan.group_page_counts)
            or raw.record_mask != 0
            or raw.state != expected_state
        ):
            raise NativeSnapshotRingStateError(
                "manager-page ready view disagrees with its submission"
            )
        payload = raw.payload.toreadonly()
        if (
            payload.format != "B"
            or payload.itemsize != 1
            or payload.ndim != 1
            or not payload.contiguous
            or payload.nbytes != raw.used_bytes
        ):
            payload.release()
            raise NativeSnapshotRingStateError(
                "manager-page ready payload is not contiguous bytes"
            )
        return ManagerPageView(ticket, payload)

    def _require_ticket(self, ticket: ManagerPageTicket) -> _OwnedPageTicket:
        self._require_open()
        if not isinstance(ticket, ManagerPageTicket) or ticket._owner is not self._owner:
            raise NativeSnapshotRingStateError(
                "manager-page ticket does not belong to this ring"
            )
        owned = self._tickets.get((ticket.slot_index, ticket.generation))
        if owned is None or owned.public is not ticket:
            raise NativeSnapshotRingStateError(
                "manager-page ticket is stale or released"
            )
        return owned

    def _retire(self, owned: _OwnedPageTicket) -> None:
        if owned.ready_view is not None:
            owned.ready_view._invalidate()
        if owned.claimed_view is not None:
            owned.claimed_view._invalidate()
        self._tickets.pop((owned.raw.slot_index, owned.raw.generation), None)

    def _require_open(self) -> None:
        if self._closed:
            raise NativeSnapshotRingStateError(
                "manager-page capture ring is shut down"
            )

    def _raise_status(self, operation: str, status: int) -> None:
        try:
            detail = self._backend.status_text(status)
        except Exception as error:  # noqa: BLE001
            detail = f"status text unavailable: {error}"
        raise NativeSnapshotRingStatusError(operation, status, detail)


__all__ = [
    "ManagerPageRingBackend",
    "ManagerPageTicket",
    "ManagerPageView",
    "NativeManagerPageRing",
]
