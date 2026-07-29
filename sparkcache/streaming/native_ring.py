"""High-level, explicitly constructed owner for the native snapshot ring.

Importing this module is CPU-only. The ctypes binding is imported and its
attested shared library is loaded only by :meth:`NativeSnapshotRing.from_attested`.
Tests and offline orchestration can inject a backend without importing CUDA.
"""

from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass
from enum import Enum, IntEnum
from os import PathLike
from typing import Protocol, Sequence


MAX_RECORD_KINDS = 4
SLOT_STATE_READY = 2
SLOT_STATE_WRITING = 3


class NativeStatus(IntEnum):
    OK = 0
    INVALID_ARGUMENT = 1
    INVALID_STATE = 2
    CUDA_ERROR = 3
    WOULD_BLOCK = 4
    NOT_READY = 5
    DROPPED = 6


class NativeSnapshotRingError(RuntimeError):
    """Base class for high-level ownership failures."""


class NativeSnapshotRingStateError(NativeSnapshotRingError):
    """The requested operation violates local ownership state."""


class NativeSnapshotRingStatusError(NativeSnapshotRingError):
    """The native ABI returned an unexpected status."""

    def __init__(self, operation: str, status: int, detail: str) -> None:
        super().__init__(
            f"snapshot ring {operation} failed: status={status} ({detail})"
        )
        self.operation = operation
        self.status = status
        self.detail = detail


@dataclass(frozen=True, slots=True)
class NativeRingConfig:
    arena_mode: int
    slot_bytes: int
    slot_count: int
    max_sources: int
    max_rows: int
    device_ordinal: int

    def __post_init__(self) -> None:
        if self.arena_mode not in (1, 2):
            raise ValueError("arena_mode must be mapped-host (1) or managed (2)")
        if self.slot_count not in (2, 3):
            raise ValueError("slot_count must be two or three")
        for name in ("slot_bytes", "max_sources", "max_rows"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_sources > 0xFFFFFFFF or self.max_rows > 0xFFFFFFFF:
            raise ValueError("max_sources and max_rows must fit uint32")
        if not -(1 << 31) <= self.device_ordinal < (1 << 31):
            raise ValueError("device_ordinal must fit int32")


@dataclass(frozen=True, slots=True)
class SnapshotSourceSpec:
    source_base: int
    source_rows: int
    source_row_stride_bytes: int
    bytes_per_token: int
    record_kind: int
    source_layer_ordinal: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_base, int)
            or isinstance(self.source_base, bool)
            or self.source_base <= 0
            or self.source_base > 0xFFFFFFFFFFFFFFFF
        ):
            raise ValueError("source_base must be a nonzero uint64")
        if (
            not isinstance(self.source_rows, int)
            or isinstance(self.source_rows, bool)
            or self.source_rows <= 0
            or self.source_rows > 0xFFFFFFFFFFFFFFFF
        ):
            raise ValueError("source_rows must be a nonzero uint64")
        for name in (
            "source_row_stride_bytes",
            "bytes_per_token",
            "record_kind",
            "source_layer_ordinal",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 0 <= value <= 0xFFFFFFFF
            ):
                raise ValueError(f"{name} must fit uint32")
        if self.bytes_per_token == 0:
            raise ValueError("bytes_per_token must be nonzero")
        if self.source_row_stride_bytes < self.bytes_per_token:
            raise ValueError("source row stride is smaller than one token")
        if self.record_kind >= MAX_RECORD_KINDS:
            raise ValueError("record_kind is outside the snapshot ABI")


@dataclass(frozen=True, slots=True)
class RawTicket:
    generation: int
    slot_index: int


@dataclass(frozen=True, slots=True)
class RawReadyView:
    payload: memoryview
    capacity_bytes: int
    used_bytes: int
    context_sequence: int
    logical_start: int
    generation: int
    row_count: int
    slot_index: int
    record_mask: int
    state: int
    record_offsets: tuple[int, ...]
    record_lengths: tuple[int, ...]


class SnapshotRingBackend(Protocol):
    """Handle-owning backend used by the high-level state machine."""

    def create(self, config: NativeRingConfig) -> int: ...

    def configure_sources(self, sources: tuple[SnapshotSourceSpec, ...]) -> int: ...

    def submit(
        self,
        *,
        context_sequence: int,
        logical_start: int,
        row_count: int,
        physical_slots: tuple[int, ...],
        producer_stream: int,
    ) -> tuple[int, RawTicket | None]: ...

    def poll(self, ticket: RawTicket) -> tuple[int, RawReadyView | None]: ...

    def claim(self, ticket: RawTicket) -> tuple[int, RawReadyView | None]: ...

    def release(self, ticket: RawTicket) -> int: ...

    def abandon_context(self, context_sequence: int) -> int: ...

    def shutdown(self) -> int: ...

    def destroy(self) -> None: ...

    def status_text(self, status: int) -> str: ...


class CtypesSnapshotRingBackend:
    """Thin adapter from the high-level protocol to the checksum-pinned ABI."""

    def __init__(
        self,
        library_path: str | PathLike[str],
        *,
        expected_sha256: str,
    ) -> None:
        # This is the sole lazy import/load edge. Merely importing the owner
        # module, or constructing it with a fake backend, never enters here.
        abi = importlib.import_module(
            "sparkcache.native.python.spark_cache_snapshot_native"
        )
        library, _info = abi.load_library(
            library_path,
            expected_sha256=expected_sha256,
        )
        self._abi = abi
        self._library = library
        self._handle = None

    def create(self, config: NativeRingConfig) -> int:
        import ctypes

        if self._handle is not None:
            raise NativeSnapshotRingStateError("native backend is already created")
        raw_config = self._abi.SnapshotConfig(
            abi_version=self._abi.ABI_VERSION,
            arena_mode=config.arena_mode,
            slot_bytes=config.slot_bytes,
            slot_count=config.slot_count,
            max_sources=config.max_sources,
            max_rows=config.max_rows,
            device_ordinal=config.device_ordinal,
            flags=0,
        )
        handle = ctypes.c_void_p()
        result = int(
            self._library.spark_cache_snapshot_create(
                ctypes.byref(raw_config),
                ctypes.byref(handle),
            )
        )
        if result == NativeStatus.OK:
            if not handle.value:
                raise NativeSnapshotRingStateError(
                    "native create returned OK without a handle"
                )
            self._handle = handle
        return result

    def configure_sources(self, sources: tuple[SnapshotSourceSpec, ...]) -> int:
        import ctypes

        values = [
            self._abi.SnapshotSource(
                source_base=source.source_base,
                source_rows=source.source_rows,
                source_row_stride_bytes=source.source_row_stride_bytes,
                bytes_per_token=source.bytes_per_token,
                record_kind=source.record_kind,
                source_layer_ordinal=source.source_layer_ordinal,
            )
            for source in sources
        ]
        array = (self._abi.SnapshotSource * len(values))(*values)
        return int(
            self._library.spark_cache_snapshot_configure_sources(
                self._require_handle(),
                ctypes.cast(array, ctypes.POINTER(self._abi.SnapshotSource)),
                len(array),
            )
        )

    def submit(
        self,
        *,
        context_sequence: int,
        logical_start: int,
        row_count: int,
        physical_slots: tuple[int, ...],
        producer_stream: int,
    ) -> tuple[int, RawTicket | None]:
        import ctypes

        submission = self._abi.SnapshotSubmission(
            context_sequence=context_sequence,
            logical_start=logical_start,
            row_count=row_count,
            flags=0,
        )
        slots = self._abi.slot_array(physical_slots)
        ticket = self._abi.SnapshotTicket()
        result = int(
            self._library.spark_cache_snapshot_try_submit(
                self._require_handle(),
                ctypes.byref(submission),
                slots,
                producer_stream,
                ctypes.byref(ticket),
            )
        )
        if result != NativeStatus.OK:
            return result, None
        return result, RawTicket(int(ticket.generation), int(ticket.slot_index))

    def poll(self, ticket: RawTicket) -> tuple[int, RawReadyView | None]:
        return self._read_view("poll", ticket)

    def claim(self, ticket: RawTicket) -> tuple[int, RawReadyView | None]:
        return self._read_view("claim", ticket)

    def release(self, ticket: RawTicket) -> int:
        import ctypes

        raw = self._raw_ticket(ticket)
        return int(
            self._library.spark_cache_snapshot_release(
                self._require_handle(),
                ctypes.byref(raw),
            )
        )

    def abandon_context(self, context_sequence: int) -> int:
        return int(
            self._library.spark_cache_snapshot_abandon_context(
                self._require_handle(),
                context_sequence,
            )
        )

    def shutdown(self) -> int:
        return int(self._library.spark_cache_snapshot_shutdown(self._require_handle()))

    def destroy(self) -> None:
        if self._handle is None:
            return
        self._library.spark_cache_snapshot_destroy(self._handle)
        self._handle = None

    def status_text(self, status: int) -> str:
        value = self._library.spark_cache_snapshot_status_string(status)
        if value is None:
            return "unknown"
        return value.decode("utf-8", errors="replace")

    def _read_view(
        self,
        operation: str,
        ticket: RawTicket,
    ) -> tuple[int, RawReadyView | None]:
        import ctypes

        raw_ticket = self._raw_ticket(ticket)
        raw_view = self._abi.SnapshotReadyView()
        function = getattr(
            self._library,
            f"spark_cache_snapshot_{operation}",
        )
        result = int(
            function(
                self._require_handle(),
                ctypes.byref(raw_ticket),
                ctypes.byref(raw_view),
            )
        )
        if result != NativeStatus.OK:
            return result, None
        return result, RawReadyView(
            payload=self._abi.ready_memoryview(raw_view),
            capacity_bytes=int(raw_view.capacity_bytes),
            used_bytes=int(raw_view.used_bytes),
            context_sequence=int(raw_view.context_sequence),
            logical_start=int(raw_view.logical_start),
            generation=int(raw_view.generation),
            row_count=int(raw_view.row_count),
            slot_index=int(raw_view.slot_index),
            record_mask=int(raw_view.record_mask),
            state=int(raw_view.state),
            record_offsets=tuple(int(value) for value in raw_view.record_offset_bytes),
            record_lengths=tuple(int(value) for value in raw_view.record_length_bytes),
        )

    def _raw_ticket(self, ticket: RawTicket):
        return self._abi.SnapshotTicket(
            generation=ticket.generation,
            slot_index=ticket.slot_index,
        )

    def _require_handle(self):
        if self._handle is None:
            raise NativeSnapshotRingStateError("native backend has no live handle")
        return self._handle


@dataclass(frozen=True, slots=True)
class SnapshotTicket:
    generation: int
    slot_index: int
    context_sequence: int
    logical_start: int
    row_count: int
    _owner: object


class SnapshotView:
    """Borrowed, read-only bytes valid only until ticket release/transition."""

    __slots__ = (
        "ticket",
        "capacity_bytes",
        "used_bytes",
        "record_mask",
        "record_offsets",
        "record_lengths",
        "native_state",
        "_payload",
        "_valid",
    )

    def __init__(
        self,
        *,
        ticket: SnapshotTicket,
        raw: RawReadyView,
        payload: memoryview,
    ) -> None:
        self.ticket = ticket
        self.capacity_bytes = raw.capacity_bytes
        self.used_bytes = raw.used_bytes
        self.record_mask = raw.record_mask
        self.record_offsets = raw.record_offsets
        self.record_lengths = raw.record_lengths
        self.native_state = raw.state
        self._payload = payload
        self._valid = True

    @property
    def payload(self) -> memoryview:
        if not self._valid:
            raise NativeSnapshotRingStateError("snapshot view is no longer valid")
        return self._payload

    def record(self, record_kind: int) -> memoryview:
        if not 0 <= record_kind < MAX_RECORD_KINDS:
            raise ValueError("record_kind is outside the snapshot ABI")
        if not self.record_mask & (1 << record_kind):
            raise KeyError(record_kind)
        start = self.record_offsets[record_kind]
        end = start + self.record_lengths[record_kind]
        return self.payload[start:end]

    def _invalidate(self) -> None:
        if not self._valid:
            return
        self._valid = False
        self._payload.release()


class _TicketPhase(Enum):
    SUBMITTED = "submitted"
    READY = "ready"
    CLAIMED = "claimed"
    ABANDONED_SUBMITTED = "abandoned_submitted"
    ABANDONED_CLAIMED = "abandoned_claimed"


@dataclass(slots=True)
class _OwnedTicket:
    public: SnapshotTicket
    raw: RawTicket
    phase: _TicketPhase = _TicketPhase.SUBMITTED
    view: SnapshotView | None = None


class NativeSnapshotRing:
    """Generation-checked owner of one configured native snapshot handle."""

    def __init__(
        self,
        config: NativeRingConfig,
        *,
        backend: SnapshotRingBackend,
    ) -> None:
        self.config = config
        self._backend = backend
        self._lock = threading.RLock()
        self._owner_token = object()
        self._sources: tuple[SnapshotSourceSpec, ...] | None = None
        self._expected_record_mask = 0
        self._tickets: dict[tuple[int, int], _OwnedTicket] = {}
        self._closed = False
        self._poisoned: NativeSnapshotRingError | None = None
        result = int(self._backend.create(config))
        if result != NativeStatus.OK:
            self._raise_status("create", result)

    @classmethod
    def from_attested(
        cls,
        config: NativeRingConfig,
        *,
        library_path: str | PathLike[str],
        expected_sha256: str,
    ) -> "NativeSnapshotRing":
        """Explicit native load/allocation edge for production callers."""

        backend = CtypesSnapshotRingBackend(
            library_path,
            expected_sha256=expected_sha256,
        )
        return cls(config, backend=backend)

    @property
    def configured_sources(self) -> tuple[SnapshotSourceSpec, ...] | None:
        return self._sources

    @property
    def active_ticket_count(self) -> int:
        with self._lock:
            return len(self._tickets)

    def configure_sources(
        self,
        sources: Sequence[SnapshotSourceSpec],
    ) -> None:
        inventory = tuple(sources)
        self._validate_source_inventory(inventory)
        with self._lock:
            self._require_usable()
            if self._sources is not None:
                if inventory == self._sources:
                    return
                raise NativeSnapshotRingStateError(
                    "snapshot source inventory is already configured"
                )
            result = int(self._backend.configure_sources(inventory))
            if result != NativeStatus.OK:
                self._raise_status("configure_sources", result)
            self._sources = inventory
            self._expected_record_mask = sum(
                1 << record_kind
                for record_kind in {source.record_kind for source in inventory}
            )

    def submit(
        self,
        *,
        context_sequence: int,
        logical_start: int,
        physical_slots: Sequence[int],
        producer_stream: int,
    ) -> SnapshotTicket | None:
        slots = tuple(physical_slots)
        self._validate_submission(
            context_sequence=context_sequence,
            logical_start=logical_start,
            physical_slots=slots,
            producer_stream=producer_stream,
        )
        with self._lock:
            self._require_usable()
            if self._sources is None:
                raise NativeSnapshotRingStateError(
                    "snapshot sources must be configured before submit"
                )
            status, raw = self._backend.submit(
                context_sequence=context_sequence,
                logical_start=logical_start,
                row_count=len(slots),
                physical_slots=slots,
                producer_stream=producer_stream,
            )
            status = int(status)
            if status in (NativeStatus.WOULD_BLOCK, NativeStatus.DROPPED):
                if raw is not None:
                    self._poison(
                        NativeSnapshotRingStateError(
                            "native submit returned a ticket for a dropped submission"
                        )
                    )
                return None
            if status != NativeStatus.OK:
                self._raise_status("submit", status)
            if raw is None:
                self._poison(
                    NativeSnapshotRingStateError(
                        "native submit returned OK without a ticket"
                    )
                )
            assert raw is not None
            if raw.generation <= 0 or not 0 <= raw.slot_index < self.config.slot_count:
                self._poison(
                    NativeSnapshotRingStateError(
                        "native submit returned an invalid generation ticket"
                    )
                )
            key = (raw.slot_index, raw.generation)
            if key in self._tickets:
                self._poison(
                    NativeSnapshotRingStateError(
                        "native submit reused an active generation ticket"
                    )
                )
            public = SnapshotTicket(
                generation=raw.generation,
                slot_index=raw.slot_index,
                context_sequence=context_sequence,
                logical_start=logical_start,
                row_count=len(slots),
                _owner=self._owner_token,
            )
            self._tickets[key] = _OwnedTicket(public=public, raw=raw)
            return public

    def poll(self, ticket: SnapshotTicket) -> SnapshotView | None:
        with self._lock:
            owned = self._require_ticket(ticket)
            if owned.phase is _TicketPhase.READY:
                assert owned.view is not None
                return owned.view
            if owned.phase is _TicketPhase.CLAIMED:
                raise NativeSnapshotRingStateError(
                    "claimed snapshot tickets are not pollable"
                )
            status, raw_view = self._backend.poll(owned.raw)
            status = int(status)
            if status == NativeStatus.NOT_READY:
                return None
            if status == NativeStatus.DROPPED:
                if owned.phase is _TicketPhase.ABANDONED_SUBMITTED:
                    self._retire(owned)
                    return None
                self._poison(
                    NativeSnapshotRingStateError(
                        "native dropped a locally active generation ticket"
                    )
                )
            if status != NativeStatus.OK:
                self._raise_status("poll", status)
            if owned.phase is _TicketPhase.ABANDONED_SUBMITTED:
                self._poison(
                    NativeSnapshotRingStateError(
                        "abandoned native ticket unexpectedly became READY"
                    )
                )
            view = self._validated_view(
                owned,
                raw_view,
                expected_state=SLOT_STATE_READY,
            )
            owned.phase = _TicketPhase.READY
            owned.view = view
            return view

    def claim(self, ticket: SnapshotTicket) -> SnapshotView | None:
        with self._lock:
            owned = self._require_ticket(ticket)
            if owned.phase in (
                _TicketPhase.ABANDONED_SUBMITTED,
                _TicketPhase.ABANDONED_CLAIMED,
            ):
                raise NativeSnapshotRingStateError(
                    "abandoned snapshot tickets cannot be claimed"
                )
            if owned.phase is _TicketPhase.CLAIMED:
                assert owned.view is not None
                return owned.view
            status, raw_view = self._backend.claim(owned.raw)
            status = int(status)
            if status == NativeStatus.NOT_READY:
                return None
            if status == NativeStatus.DROPPED:
                self._poison(
                    NativeSnapshotRingStateError(
                        "native dropped a locally active generation ticket"
                    )
                )
            if status != NativeStatus.OK:
                self._raise_status("claim", status)
            view = self._validated_view(
                owned,
                raw_view,
                expected_state=SLOT_STATE_WRITING,
            )
            if owned.view is not None:
                owned.view._invalidate()
            owned.phase = _TicketPhase.CLAIMED
            owned.view = view
            return view

    def release(self, ticket: SnapshotTicket) -> None:
        with self._lock:
            self._require_usable()
            owned = self._require_ticket(ticket)
            if owned.phase not in (
                _TicketPhase.READY,
                _TicketPhase.CLAIMED,
                _TicketPhase.ABANDONED_CLAIMED,
            ):
                raise NativeSnapshotRingStateError(
                    "only READY or claimed snapshot tickets can be released"
                )
            result = int(self._backend.release(owned.raw))
            if result != NativeStatus.OK:
                self._raise_status("release", result)
            self._retire(owned)

    def abandon(self, context_sequence: int) -> None:
        if (
            not isinstance(context_sequence, int)
            or isinstance(context_sequence, bool)
            or context_sequence <= 0
            or context_sequence > 0xFFFFFFFFFFFFFFFF
        ):
            raise ValueError("context_sequence must be a nonzero uint64")
        with self._lock:
            self._require_usable()
            result = int(self._backend.abandon_context(context_sequence))
            if result != NativeStatus.OK:
                self._raise_status("abandon_context", result)
            for owned in tuple(self._tickets.values()):
                if owned.public.context_sequence != context_sequence:
                    continue
                if owned.phase is _TicketPhase.SUBMITTED:
                    owned.phase = _TicketPhase.ABANDONED_SUBMITTED
                elif owned.phase is _TicketPhase.READY:
                    # Native abandon frees unclaimed READY slots immediately.
                    self._retire(owned)
                elif owned.phase is _TicketPhase.CLAIMED:
                    owned.phase = _TicketPhase.ABANDONED_CLAIMED

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            writing = [
                owned
                for owned in self._tickets.values()
                if owned.phase in (_TicketPhase.CLAIMED, _TicketPhase.ABANDONED_CLAIMED)
            ]
            if writing:
                raise NativeSnapshotRingStateError(
                    "release every claimed snapshot view before shutdown"
                )
            result = int(self._backend.shutdown())
            if result != NativeStatus.OK:
                self._raise_status("shutdown", result)
            for owned in tuple(self._tickets.values()):
                self._retire(owned)
            self._backend.destroy()
            self._closed = True

    def _validate_source_inventory(
        self,
        inventory: tuple[SnapshotSourceSpec, ...],
    ) -> None:
        if not inventory:
            raise ValueError("snapshot source inventory must not be empty")
        if len(inventory) > self.config.max_sources:
            raise ValueError("snapshot source inventory exceeds max_sources")
        next_ordinal = [0] * MAX_RECORD_KINDS
        widths: dict[int, int] = {}
        for source in inventory:
            if source.source_layer_ordinal != next_ordinal[source.record_kind]:
                raise ValueError(
                    "source layer ordinals must be dense and ordered per record kind"
                )
            next_ordinal[source.record_kind] += 1
            previous = widths.setdefault(
                source.record_kind,
                source.bytes_per_token,
            )
            if previous != source.bytes_per_token:
                raise ValueError(
                    "sources of one record kind must share bytes_per_token"
                )

    def _validate_submission(
        self,
        *,
        context_sequence: int,
        logical_start: int,
        physical_slots: tuple[int, ...],
        producer_stream: int,
    ) -> None:
        for name, value, allow_zero in (
            ("context_sequence", context_sequence, False),
            ("logical_start", logical_start, True),
            ("producer_stream", producer_stream, True),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < (0 if allow_zero else 1)
                or value > 0xFFFFFFFFFFFFFFFF
            ):
                qualifier = "a uint64" if allow_zero else "a nonzero uint64"
                raise ValueError(f"{name} must be {qualifier}")
        if not physical_slots or len(physical_slots) > self.config.max_rows:
            raise ValueError("physical_slots must contain 1..max_rows entries")
        if any(
            not isinstance(slot, int)
            or isinstance(slot, bool)
            or not 0 <= slot <= 0xFFFFFFFF
            for slot in physical_slots
        ):
            raise ValueError("physical slots must be uint32 values")

    def _validated_view(
        self,
        owned: _OwnedTicket,
        raw: RawReadyView | None,
        *,
        expected_state: int,
    ) -> SnapshotView:
        if raw is None:
            self._poison(
                NativeSnapshotRingStateError("native READY status omitted its view")
            )
        assert raw is not None
        public = owned.public
        if (
            raw.generation != public.generation
            or raw.slot_index != public.slot_index
            or raw.context_sequence != public.context_sequence
            or raw.logical_start != public.logical_start
            or raw.row_count != public.row_count
        ):
            self._poison(
                NativeSnapshotRingStateError(
                    "native READY view disagrees with its generation ticket"
                )
            )
        if raw.state != expected_state:
            self._poison(
                NativeSnapshotRingStateError(
                    "native READY view has an unexpected ownership state"
                )
            )
        if (
            raw.capacity_bytes != self.config.slot_bytes
            or raw.used_bytes <= 0
            or raw.used_bytes > raw.capacity_bytes
            or raw.record_mask != self._expected_record_mask
            or len(raw.record_offsets) != MAX_RECORD_KINDS
            or len(raw.record_lengths) != MAX_RECORD_KINDS
        ):
            self._poison(
                NativeSnapshotRingStateError(
                    "native READY view has invalid capacity or record metadata"
                )
            )
        expected_offsets, expected_lengths, expected_used = self._layout_for_rows(
            public.row_count
        )
        if (
            raw.record_offsets != expected_offsets
            or raw.record_lengths != expected_lengths
            or raw.used_bytes != expected_used
        ):
            self._poison(
                NativeSnapshotRingStateError(
                    "native READY view disagrees with the configured source layout"
                )
            )
        try:
            payload = raw.payload.toreadonly()
        except (TypeError, ValueError) as error:
            self._poison(
                NativeSnapshotRingStateError(
                    f"native READY payload is not a usable buffer: {error}"
                )
            )
        if (
            payload.format != "B"
            or payload.itemsize != 1
            or payload.ndim != 1
            or not payload.contiguous
            or payload.nbytes != raw.used_bytes
        ):
            payload.release()
            self._poison(
                NativeSnapshotRingStateError(
                    "native READY payload is not a contiguous used_bytes byte view"
                )
            )
        return SnapshotView(ticket=public, raw=raw, payload=payload)

    def _layout_for_rows(
        self,
        row_count: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        assert self._sources is not None
        layer_counts = [0] * MAX_RECORD_KINDS
        widths = [0] * MAX_RECORD_KINDS
        for source in self._sources:
            layer_counts[source.record_kind] += 1
            widths[source.record_kind] = source.bytes_per_token
        offsets = [0] * MAX_RECORD_KINDS
        lengths = [0] * MAX_RECORD_KINDS
        cursor = 0
        for kind in range(MAX_RECORD_KINDS):
            if not layer_counts[kind]:
                continue
            cursor = (cursor + 63) // 64 * 64
            offsets[kind] = cursor
            lengths[kind] = row_count * widths[kind] * layer_counts[kind]
            cursor += lengths[kind]
        if cursor > self.config.slot_bytes:
            self._poison(
                NativeSnapshotRingStateError(
                    "configured sources exceed the arena slot for this submission"
                )
            )
        return tuple(offsets), tuple(lengths), cursor

    def _require_ticket(self, ticket: SnapshotTicket) -> _OwnedTicket:
        self._require_usable()
        if (
            not isinstance(ticket, SnapshotTicket)
            or ticket._owner is not self._owner_token
        ):
            raise NativeSnapshotRingStateError(
                "snapshot ticket does not belong to this ring"
            )
        owned = self._tickets.get((ticket.slot_index, ticket.generation))
        if owned is None or owned.public is not ticket:
            raise NativeSnapshotRingStateError(
                "snapshot ticket is stale or already released"
            )
        return owned

    def _require_usable(self) -> None:
        if self._closed:
            raise NativeSnapshotRingStateError("snapshot ring is shut down")
        if self._poisoned is not None:
            raise NativeSnapshotRingStateError(
                f"snapshot ring is poisoned: {self._poisoned}"
            ) from self._poisoned

    def _raise_status(self, operation: str, status: int) -> None:
        try:
            detail = self._backend.status_text(status)
        except Exception as error:  # noqa: BLE001 - retain original status
            detail = f"status text unavailable: {error}"
        self._poison(NativeSnapshotRingStatusError(operation, status, detail))

    def _poison(self, error: NativeSnapshotRingError) -> None:
        self._poisoned = error
        raise error

    def _retire(self, owned: _OwnedTicket) -> None:
        if owned.view is not None:
            owned.view._invalidate()
            owned.view = None
        self._tickets.pop((owned.raw.slot_index, owned.raw.generation), None)
