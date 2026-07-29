from __future__ import annotations

import ctypes
import dataclasses
from collections import deque

import pytest

from sparkcache.streaming import native_ring
from sparkcache.streaming.native_ring import (
    NativeRingConfig,
    NativeSnapshotRing,
    NativeSnapshotRingStateError,
    NativeSnapshotRingStatusError,
    NativeStatus,
    RawReadyView,
    RawTicket,
    SnapshotSourceSpec,
)


class FakeBackend:
    """GPU-free ABI double with the native ring's ownership states."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.config: NativeRingConfig | None = None
        self.sources: tuple[SnapshotSourceSpec, ...] = ()
        self.next_generation = 1
        self.next_slot = 0
        self.entries: dict[tuple[int, int], dict] = {}
        self.submit_statuses: deque[int] = deque()
        self.poll_statuses: deque[int] = deque()
        self.claim_statuses: deque[int] = deque()
        self.configure_status = NativeStatus.OK
        self.shutdown_status = NativeStatus.OK
        self.corrupt_generation = False
        self.destroyed = False

    def create(self, config: NativeRingConfig) -> int:
        self.calls.append(("create", config))
        self.config = config
        return NativeStatus.OK

    def configure_sources(self, sources: tuple[SnapshotSourceSpec, ...]) -> int:
        self.calls.append(("configure", sources))
        if self.configure_status == NativeStatus.OK:
            self.sources = sources
        return self.configure_status

    def submit(
        self,
        *,
        context_sequence: int,
        logical_start: int,
        row_count: int,
        physical_slots: tuple[int, ...],
        producer_stream: int,
    ) -> tuple[int, RawTicket | None]:
        self.calls.append(
            (
                "submit",
                context_sequence,
                logical_start,
                row_count,
                physical_slots,
                producer_stream,
            )
        )
        status = (
            self.submit_statuses.popleft() if self.submit_statuses else NativeStatus.OK
        )
        if status != NativeStatus.OK:
            return status, None
        assert self.config is not None
        ticket = RawTicket(self.next_generation, self.next_slot)
        self.next_generation += 1
        self.next_slot = (self.next_slot + 1) % self.config.slot_count
        offsets, lengths, used = self._layout(row_count)
        self.entries[(ticket.slot_index, ticket.generation)] = {
            "ticket": ticket,
            "context_sequence": context_sequence,
            "logical_start": logical_start,
            "row_count": row_count,
            "state": "filling",
            "buffer": bytearray(range(used)),
            "offsets": offsets,
            "lengths": lengths,
            "used": used,
        }
        return status, ticket

    def poll(self, ticket: RawTicket) -> tuple[int, RawReadyView | None]:
        self.calls.append(("poll", ticket))
        if self.poll_statuses:
            status = self.poll_statuses.popleft()
            if status != NativeStatus.OK:
                if status == NativeStatus.DROPPED:
                    self.entries.pop((ticket.slot_index, ticket.generation), None)
                return status, None
        entry = self.entries[(ticket.slot_index, ticket.generation)]
        if entry["state"] in ("filling", "abandoned"):
            return NativeStatus.NOT_READY, None
        return NativeStatus.OK, self._view(entry, state=2)

    def claim(self, ticket: RawTicket) -> tuple[int, RawReadyView | None]:
        self.calls.append(("claim", ticket))
        if self.claim_statuses:
            status = self.claim_statuses.popleft()
            if status != NativeStatus.OK:
                return status, None
        entry = self.entries[(ticket.slot_index, ticket.generation)]
        if entry["state"] == "filling":
            return NativeStatus.NOT_READY, None
        if entry["state"] == "abandoned":
            self.entries.pop((ticket.slot_index, ticket.generation))
            return NativeStatus.DROPPED, None
        entry["state"] = "claimed"
        return NativeStatus.OK, self._view(entry, state=3)

    def release(self, ticket: RawTicket) -> int:
        self.calls.append(("release", ticket))
        entry = self.entries.get((ticket.slot_index, ticket.generation))
        if entry is None or entry["state"] not in ("ready", "claimed"):
            return NativeStatus.DROPPED
        del self.entries[(ticket.slot_index, ticket.generation)]
        return NativeStatus.OK

    def abandon_context(self, context_sequence: int) -> int:
        self.calls.append(("abandon", context_sequence))
        for key, entry in tuple(self.entries.items()):
            if entry["context_sequence"] != context_sequence:
                continue
            if entry["state"] == "ready":
                del self.entries[key]
            elif entry["state"] == "filling":
                entry["state"] = "abandoned"
            # A claimed writer retains ownership until release.
        return NativeStatus.OK

    def shutdown(self) -> int:
        self.calls.append(("shutdown",))
        if self.shutdown_status != NativeStatus.OK:
            return self.shutdown_status
        if any(entry["state"] == "claimed" for entry in self.entries.values()):
            return NativeStatus.INVALID_STATE
        self.entries.clear()
        return NativeStatus.OK

    def destroy(self) -> None:
        self.calls.append(("destroy",))
        self.destroyed = True

    def status_text(self, status: int) -> str:
        return NativeStatus(status).name.lower()

    def ready(self, ticket) -> bytearray:
        entry = self.entries[(ticket.slot_index, ticket.generation)]
        entry["state"] = "ready"
        return entry["buffer"]

    def _layout(self, rows: int) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        counts = [0, 0, 0, 0]
        widths = [0, 0, 0, 0]
        for source in self.sources:
            counts[source.record_kind] += 1
            widths[source.record_kind] = source.bytes_per_token
        offsets = [0, 0, 0, 0]
        lengths = [0, 0, 0, 0]
        cursor = 0
        for kind in range(4):
            if not counts[kind]:
                continue
            cursor = (cursor + 63) // 64 * 64
            offsets[kind] = cursor
            lengths[kind] = rows * counts[kind] * widths[kind]
            cursor += lengths[kind]
        return tuple(offsets), tuple(lengths), cursor

    def _view(self, entry: dict, *, state: int) -> RawReadyView:
        ticket = entry["ticket"]
        generation = (
            ticket.generation + 1 if self.corrupt_generation else ticket.generation
        )
        mask = 0
        for source in self.sources:
            mask |= 1 << source.record_kind
        assert self.config is not None
        return RawReadyView(
            payload=memoryview(entry["buffer"]),
            capacity_bytes=self.config.slot_bytes,
            used_bytes=entry["used"],
            context_sequence=entry["context_sequence"],
            logical_start=entry["logical_start"],
            generation=generation,
            row_count=entry["row_count"],
            slot_index=ticket.slot_index,
            record_mask=mask,
            state=state,
            record_offsets=entry["offsets"],
            record_lengths=entry["lengths"],
        )


def config() -> NativeRingConfig:
    return NativeRingConfig(
        arena_mode=1,
        slot_bytes=128,
        slot_count=2,
        max_sources=4,
        max_rows=8,
        device_ordinal=0,
    )


def sources() -> tuple[SnapshotSourceSpec, ...]:
    return (
        SnapshotSourceSpec(0x1000, 64, 2, 2, 0, 0),
        SnapshotSourceSpec(0x2000, 64, 2, 2, 0, 1),
        SnapshotSourceSpec(0x3000, 64, 3, 3, 1, 0),
    )


def configured_ring() -> tuple[NativeSnapshotRing, FakeBackend]:
    backend = FakeBackend()
    ring = NativeSnapshotRing(config(), backend=backend)
    ring.configure_sources(sources())
    return ring, backend


def test_fake_construction_does_not_import_or_load_ctypes_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        native_ring.importlib,
        "import_module",
        lambda _name: pytest.fail("native binding imported"),
    )
    backend = FakeBackend()
    ring = NativeSnapshotRing(config(), backend=backend)
    ring.configure_sources(sources())
    ring.shutdown()
    assert backend.destroyed


def test_attested_constructor_drives_existing_ctypes_binding_without_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sparkcache.native.python import spark_cache_snapshot_native as abi

    class FakeLibrary:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.buffer = (ctypes.c_ubyte * 70)(*range(70))

        def spark_cache_snapshot_create(self, config_pointer, handle_pointer):
            self.calls.append("create")
            assert config_pointer._obj.slot_count == 2
            handle_pointer._obj.value = 0x1234
            return NativeStatus.OK

        def spark_cache_snapshot_configure_sources(
            self, _handle, source_pointer, source_count
        ):
            self.calls.append("configure")
            assert source_count == 3
            assert [source_pointer[index].record_kind for index in range(3)] == [
                0,
                0,
                1,
            ]
            return NativeStatus.OK

        def spark_cache_snapshot_try_submit(
            self,
            _handle,
            submission_pointer,
            physical_slots,
            producer_stream,
            ticket_pointer,
        ):
            self.calls.append("submit")
            assert submission_pointer._obj.context_sequence == 21
            assert list(physical_slots) == [9, 3]
            assert producer_stream == 77
            ticket_pointer._obj.generation = 5
            ticket_pointer._obj.slot_index = 1
            return NativeStatus.OK

        def spark_cache_snapshot_poll(self, _handle, ticket_pointer, view_pointer):
            self.calls.append("poll")
            self._fill_view(ticket_pointer, view_pointer, state=2)
            return NativeStatus.OK

        def spark_cache_snapshot_claim(self, _handle, ticket_pointer, view_pointer):
            self.calls.append("claim")
            self._fill_view(ticket_pointer, view_pointer, state=3)
            return NativeStatus.OK

        def spark_cache_snapshot_release(self, _handle, _ticket_pointer):
            self.calls.append("release")
            return NativeStatus.OK

        def spark_cache_snapshot_abandon_context(self, _handle, _context):
            self.calls.append("abandon")
            return NativeStatus.OK

        def spark_cache_snapshot_shutdown(self, _handle):
            self.calls.append("shutdown")
            return NativeStatus.OK

        def spark_cache_snapshot_destroy(self, _handle):
            self.calls.append("destroy")

        @staticmethod
        def spark_cache_snapshot_status_string(_status):
            return b"fake"

        def _fill_view(self, ticket_pointer, view_pointer, *, state: int) -> None:
            view = view_pointer._obj
            view.host_address = ctypes.addressof(self.buffer)
            view.capacity_bytes = 128
            view.used_bytes = 70
            view.context_sequence = 21
            view.logical_start = 256
            view.generation = ticket_pointer._obj.generation
            view.row_count = 2
            view.slot_index = ticket_pointer._obj.slot_index
            view.record_mask = 0b11
            view.state = state
            view.record_offset_bytes[0] = 0
            view.record_offset_bytes[1] = 64
            view.record_length_bytes[0] = 8
            view.record_length_bytes[1] = 6

    library = FakeLibrary()
    loads: list[tuple[object, str]] = []

    def load_library(path, *, expected_sha256):
        loads.append((path, expected_sha256))
        return library, object()

    monkeypatch.setattr(abi, "load_library", load_library)
    monkeypatch.setattr(
        native_ring.importlib,
        "import_module",
        lambda name: (
            abi
            if name == "sparkcache.native.python.spark_cache_snapshot_native"
            else pytest.fail(f"unexpected import: {name}")
        ),
    )

    ring = NativeSnapshotRing.from_attested(
        config(),
        library_path="/attested/snapshot.so",
        expected_sha256="a" * 64,
    )
    ring.configure_sources(sources())
    ticket = ring.submit(
        context_sequence=21,
        logical_start=256,
        physical_slots=(9, 3),
        producer_stream=77,
    )
    assert ticket is not None
    ready = ring.poll(ticket)
    assert ready is not None
    assert bytes(ready.record(1)) == bytes(range(64, 70))
    assert ring.claim(ticket) is not None
    ring.release(ticket)
    ring.abandon(999)
    ring.shutdown()

    assert loads == [("/attested/snapshot.so", "a" * 64)]
    assert library.calls == [
        "create",
        "configure",
        "submit",
        "poll",
        "claim",
        "release",
        "abandon",
        "shutdown",
        "destroy",
    ]


def test_source_inventory_is_configured_exactly_once() -> None:
    ring, backend = configured_ring()
    ring.configure_sources(sources())
    assert [call[0] for call in backend.calls].count("configure") == 1

    changed = sources()[:-1]
    with pytest.raises(NativeSnapshotRingStateError, match="already configured"):
        ring.configure_sources(changed)
    ring.shutdown()


def test_source_inventory_rejects_sparse_ordinals_and_mixed_widths() -> None:
    ring = NativeSnapshotRing(config(), backend=FakeBackend())
    sparse = (SnapshotSourceSpec(0x1000, 64, 2, 2, 0, 1),)
    with pytest.raises(ValueError, match="dense and ordered"):
        ring.configure_sources(sparse)
    mixed = (
        SnapshotSourceSpec(0x1000, 64, 2, 2, 0, 0),
        SnapshotSourceSpec(0x2000, 64, 3, 3, 0, 1),
    )
    with pytest.raises(ValueError, match="share bytes_per_token"):
        ring.configure_sources(mixed)
    ring.shutdown()


def test_submit_requires_configuration_and_handles_bounded_backpressure() -> None:
    backend = FakeBackend()
    ring = NativeSnapshotRing(config(), backend=backend)
    with pytest.raises(NativeSnapshotRingStateError, match="configured"):
        ring.submit(
            context_sequence=1,
            logical_start=0,
            physical_slots=(3, 4),
            producer_stream=0,
        )
    ring.configure_sources(sources())
    backend.submit_statuses.extend((NativeStatus.WOULD_BLOCK, NativeStatus.DROPPED))
    for context in (1, 2):
        assert (
            ring.submit(
                context_sequence=context,
                logical_start=0,
                physical_slots=(3, 4),
                producer_stream=99,
            )
            is None
        )
    assert ring.active_ticket_count == 0
    ring.shutdown()


def test_ready_view_is_zero_copy_read_only_and_generation_owned() -> None:
    ring, backend = configured_ring()
    ticket = ring.submit(
        context_sequence=7,
        logical_start=256,
        physical_slots=(9, 3),
        producer_stream=123,
    )
    assert ticket is not None
    assert ring.poll(ticket) is None

    backing = backend.ready(ticket)
    ready = ring.poll(ticket)
    assert ready is not None
    assert ready.payload.readonly
    assert ready.record(0).nbytes == 8
    assert ready.record(1).nbytes == 6
    backing[0] = 211
    assert ready.payload[0] == 211

    forged = dataclasses.replace(ticket)
    with pytest.raises(NativeSnapshotRingStateError, match="stale"):
        ring.poll(forged)

    claimed = ring.claim(ticket)
    assert claimed is not None
    with pytest.raises(NativeSnapshotRingStateError, match="no longer valid"):
        _ = ready.payload
    assert claimed.payload[0] == 211
    ring.release(ticket)
    with pytest.raises(NativeSnapshotRingStateError, match="no longer valid"):
        _ = claimed.payload
    assert ring.active_ticket_count == 0
    ring.shutdown()


def test_generation_mismatch_poisoning_blocks_further_use_but_allows_shutdown() -> None:
    ring, backend = configured_ring()
    ticket = ring.submit(
        context_sequence=8,
        logical_start=0,
        physical_slots=(1, 2),
        producer_stream=0,
    )
    assert ticket is not None
    backend.ready(ticket)
    backend.corrupt_generation = True
    with pytest.raises(
        NativeSnapshotRingStateError,
        match="generation ticket",
    ):
        ring.poll(ticket)
    with pytest.raises(NativeSnapshotRingStateError, match="poisoned"):
        ring.submit(
            context_sequence=9,
            logical_start=0,
            physical_slots=(1, 2),
            producer_stream=0,
        )
    ring.shutdown()
    assert backend.destroyed


def test_unexpected_native_drop_of_active_ticket_is_generation_failure() -> None:
    ring, backend = configured_ring()
    ticket = ring.submit(
        context_sequence=9,
        logical_start=0,
        physical_slots=(1, 2),
        producer_stream=0,
    )
    assert ticket is not None
    backend.poll_statuses.append(NativeStatus.DROPPED)
    with pytest.raises(NativeSnapshotRingStateError, match="locally active"):
        ring.poll(ticket)
    ring.shutdown()


def test_every_unexpected_native_status_is_fail_closed() -> None:
    ring, backend = configured_ring()
    ticket = ring.submit(
        context_sequence=10,
        logical_start=0,
        physical_slots=(1, 2),
        producer_stream=0,
    )
    assert ticket is not None
    backend.poll_statuses.append(NativeStatus.CUDA_ERROR)
    with pytest.raises(NativeSnapshotRingStatusError) as raised:
        ring.poll(ticket)
    assert raised.value.status == NativeStatus.CUDA_ERROR
    with pytest.raises(NativeSnapshotRingStateError, match="poisoned"):
        ring.poll(ticket)
    ring.shutdown()


def test_abandon_retires_ready_drains_filling_and_preserves_claimed_writer() -> None:
    ring, backend = configured_ring()
    ready_ticket = ring.submit(
        context_sequence=11,
        logical_start=0,
        physical_slots=(1, 2),
        producer_stream=0,
    )
    assert ready_ticket is not None
    backend.ready(ready_ticket)
    assert ring.poll(ready_ticket) is not None
    ring.abandon(11)
    with pytest.raises(NativeSnapshotRingStateError, match="stale"):
        ring.poll(ready_ticket)

    filling = ring.submit(
        context_sequence=12,
        logical_start=0,
        physical_slots=(1, 2),
        producer_stream=0,
    )
    assert filling is not None
    ring.abandon(12)
    backend.poll_statuses.append(NativeStatus.DROPPED)
    assert ring.poll(filling) is None

    claimed_ticket = ring.submit(
        context_sequence=13,
        logical_start=0,
        physical_slots=(1, 2),
        producer_stream=0,
    )
    assert claimed_ticket is not None
    backend.ready(claimed_ticket)
    assert ring.claim(claimed_ticket) is not None
    ring.abandon(13)
    with pytest.raises(NativeSnapshotRingStateError, match="cannot be claimed"):
        ring.claim(claimed_ticket)
    ring.release(claimed_ticket)
    ring.shutdown()


def test_shutdown_requires_claimed_views_released_and_is_idempotent() -> None:
    ring, backend = configured_ring()
    ticket = ring.submit(
        context_sequence=14,
        logical_start=0,
        physical_slots=(1, 2),
        producer_stream=0,
    )
    assert ticket is not None
    backend.ready(ticket)
    assert ring.claim(ticket) is not None
    with pytest.raises(NativeSnapshotRingStateError, match="release every claimed"):
        ring.shutdown()
    assert ("shutdown",) not in backend.calls
    ring.release(ticket)
    ring.shutdown()
    ring.shutdown()
    assert [call[0] for call in backend.calls].count("shutdown") == 1
    assert [call[0] for call in backend.calls].count("destroy") == 1
