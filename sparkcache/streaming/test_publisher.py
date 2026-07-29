from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path

import pytest

import spark_context_cache_store as connector_store
from sparkcache.persistent_context_cache.cache_manifest import (
    CacheIdentity,
    ContextChunk,
    ManifestStore,
    StateRecord,
    _encode_chunk,
)
from sparkcache.spark_context_cache_codec import (
    LayerPlan,
    pack_positions,
    pack_record,
    unpack_positions,
)
from sparkcache.streaming.block_lease import BlockLeaseRegistry, LeaseCapacity
from sparkcache.streaming.native_ring import (
    NativeRingConfig,
    NativeSnapshotRing,
    NativeStatus,
    RawReadyView,
    RawTicket,
    SnapshotSourceSpec,
)
from sparkcache.streaming.planner import (
    SnapshotBatch,
    StreamingSnapshotCoordinator,
)
from sparkcache.streaming.publisher import (
    GLM52_INDEXER_BYTES_PER_TOKEN,
    GLM52_INDEXER_LAYERS,
    GLM52_LAYER_ORDER,
    GLM52_MACRO_PAYLOAD_BYTES,
    GLM52_MACRO_ROWS,
    GLM52_RING_DEPTH,
    GLM52_SLOT_BYTES,
    GLM52_SOURCE_COUNT,
    GLM52_TARGET_BYTES_PER_TOKEN,
    GLM52_TARGET_LAYERS,
    Glm52ReadyViewTranslator,
    JournalState,
    ManifestSnapshotJournalWriter,
    SnapshotTranslationError,
    SnapshotWriterBackpressure,
)
from sparkcache.streaming.runtime import (
    SnapshotJournalTransaction,
    SnapshotJournalWriter,
    StreamingSnapshotRuntime,
    StreamingSnapshotRuntimeConfig,
    WriterCompletion,
)


class FakeGlmRingBackend:
    """Immediately-ready, CPU-owned implementation of the native ABI."""

    def __init__(self) -> None:
        self.config: NativeRingConfig | None = None
        self.sources: tuple[SnapshotSourceSpec, ...] = ()
        self.entries: dict[tuple[int, int], dict] = {}
        self.next_generation = 1
        self.releases = 0
        self.abandons = 0

    def create(self, config: NativeRingConfig) -> int:
        self.config = config
        return NativeStatus.OK

    def configure_sources(self, sources: tuple[SnapshotSourceSpec, ...]) -> int:
        self.sources = sources
        return NativeStatus.OK

    def submit(
        self,
        *,
        context_sequence: int,
        logical_start: int,
        row_count: int,
        physical_slots: tuple[int, ...],
        producer_stream: int,
    ) -> tuple[int, RawTicket | None]:
        del physical_slots, producer_stream
        assert self.config is not None
        ticket = RawTicket(
            generation=self.next_generation,
            slot_index=(self.next_generation - 1) % self.config.slot_count,
        )
        self.next_generation += 1
        target = _record_bytes(0, row_count, 0, row_count)
        indexer = _record_bytes(1, row_count, 0, row_count)
        indexer_offset = (len(target) + 63) // 64 * 64
        used = indexer_offset + len(indexer)
        payload = bytearray([0xA5]) * used
        payload[: len(target)] = target
        payload[indexer_offset:] = indexer
        self.entries[(ticket.slot_index, ticket.generation)] = {
            "ticket": ticket,
            "context_sequence": context_sequence,
            "logical_start": logical_start,
            "row_count": row_count,
            "payload": payload,
            "indexer_offset": indexer_offset,
            "target_bytes": len(target),
            "indexer_bytes": len(indexer),
            "state": "ready",
        }
        return NativeStatus.OK, ticket

    def poll(self, ticket: RawTicket) -> tuple[int, RawReadyView | None]:
        entry = self.entries.get((ticket.slot_index, ticket.generation))
        if entry is None or entry["state"] == "abandoned":
            self.entries.pop((ticket.slot_index, ticket.generation), None)
            return NativeStatus.DROPPED, None
        if entry["state"] == "claimed":
            return NativeStatus.INVALID_STATE, None
        return NativeStatus.OK, self._view(entry, state=2)

    def claim(self, ticket: RawTicket) -> tuple[int, RawReadyView | None]:
        entry = self.entries.get((ticket.slot_index, ticket.generation))
        if entry is None or entry["state"] == "abandoned":
            self.entries.pop((ticket.slot_index, ticket.generation), None)
            return NativeStatus.DROPPED, None
        entry["state"] = "claimed"
        return NativeStatus.OK, self._view(entry, state=3)

    def release(self, ticket: RawTicket) -> int:
        entry = self.entries.get((ticket.slot_index, ticket.generation))
        if entry is None or entry["state"] != "claimed":
            return NativeStatus.DROPPED
        del self.entries[(ticket.slot_index, ticket.generation)]
        self.releases += 1
        return NativeStatus.OK

    def abandon_context(self, context_sequence: int) -> int:
        self.abandons += 1
        for key, entry in tuple(self.entries.items()):
            if entry["context_sequence"] != context_sequence:
                continue
            if entry["state"] == "ready":
                del self.entries[key]
            elif entry["state"] != "claimed":
                entry["state"] = "abandoned"
        return NativeStatus.OK

    def shutdown(self) -> int:
        if any(entry["state"] == "claimed" for entry in self.entries.values()):
            return NativeStatus.INVALID_STATE
        self.entries.clear()
        return NativeStatus.OK

    def destroy(self) -> None:
        return

    @staticmethod
    def status_text(status: int) -> str:
        return NativeStatus(status).name.lower()

    def _view(self, entry: dict, *, state: int) -> RawReadyView:
        ticket = entry["ticket"]
        assert self.config is not None
        return RawReadyView(
            payload=memoryview(entry["payload"]),
            capacity_bytes=self.config.slot_bytes,
            used_bytes=len(entry["payload"]),
            context_sequence=entry["context_sequence"],
            logical_start=entry["logical_start"],
            generation=ticket.generation,
            row_count=entry["row_count"],
            slot_index=ticket.slot_index,
            record_mask=0b011,
            state=state,
            record_offsets=(0, entry["indexer_offset"], 0, 0),
            record_lengths=(
                entry["target_bytes"],
                entry["indexer_bytes"],
                0,
                0,
            ),
        )


def _byte_value(record_kind: int, layer: int, row: int) -> int:
    return (record_kind * 97 + layer * 13 + row * 17) & 0xFF


def _record_bytes(
    record_kind: int,
    batch_rows: int,
    row_start: int,
    row_count: int,
) -> bytes:
    if record_kind == 0:
        layers = GLM52_TARGET_LAYERS
        width = GLM52_TARGET_BYTES_PER_TOKEN
    else:
        layers = GLM52_INDEXER_LAYERS
        width = GLM52_INDEXER_BYTES_PER_TOKEN
    del batch_rows
    return b"".join(
        bytes([_byte_value(record_kind, layer, row)]) * width
        for layer in range(layers)
        for row in range(row_start, row_start + row_count)
    )


def _reference_chunk(
    *,
    logical_start: int,
    dcp_rank: int,
    batch_row_start: int,
) -> ContextChunk:
    plans = tuple(
        sorted(
            (
                LayerPlan(
                    layer.name,
                    ("target_ckv" if layer.record_kind == 0 else "sparse_indexer"),
                    layer.bytes_per_token,
                )
                for layer in GLM52_LAYER_ORDER
            ),
            key=lambda plan: plan.name,
        )
    )
    layer_rows = {
        layer.name: b"".join(
            bytes([_byte_value(layer.record_kind, layer.source_ordinal, row)])
            * layer.bytes_per_token
            for row in range(batch_row_start, batch_row_start + 64)
        )
        for layer in GLM52_LAYER_ORDER
    }
    positions = tuple(range(logical_start + dcp_rank, logical_start + 256, 4))
    return ContextChunk(
        logical_start=logical_start,
        logical_end=logical_start + 256,
        records={
            StateRecord.LOGICAL_POSITIONS: pack_positions(positions),
            StateRecord.TARGET_CKV: pack_record(
                plans,
                "target_ckv",
                layer_rows,
                64,
            ),
            StateRecord.SPARSE_INDEXER: pack_record(
                plans,
                "sparse_indexer",
                layer_rows,
                64,
            ),
        },
    )


def _sources() -> tuple[SnapshotSourceSpec, ...]:
    return tuple(
        SnapshotSourceSpec(
            source_base=0x100000 + index * 0x10000,
            source_rows=4096,
            source_row_stride_bytes=layer.bytes_per_token,
            bytes_per_token=layer.bytes_per_token,
            record_kind=layer.record_kind,
            source_layer_ordinal=layer.source_ordinal,
        )
        for index, layer in enumerate(GLM52_LAYER_ORDER)
    )


def _ring() -> tuple[NativeSnapshotRing, FakeGlmRingBackend]:
    backend = FakeGlmRingBackend()
    ring = NativeSnapshotRing(
        NativeRingConfig(
            arena_mode=1,
            slot_bytes=GLM52_SLOT_BYTES,
            slot_count=GLM52_RING_DEPTH,
            max_sources=GLM52_SOURCE_COUNT,
            max_rows=GLM52_MACRO_ROWS,
            device_ordinal=0,
        ),
        backend=backend,
    )
    ring.configure_sources(_sources())
    return ring, backend


def _identity(rank: int = 0) -> CacheIdentity:
    return CacheIdentity(
        target_checkpoint="1" * 64,
        draft_checkpoint="1" * 64,
        quantization_layout="nvfp4_ds_mla-per-token-v1",
        rope_layout="glm52-rope-v1",
        tp_degree=4,
        dcp_degree=4,
        dcp_shard_rank=rank,
        chunk_tokens=256,
        boundary_hidden_policy="live_forward",
        draft_kv_policy="colocated_target",
    )


def _submit(
    ring: NativeSnapshotRing,
    *,
    context_sequence: int,
    logical_start: int,
    rows: int,
):
    ticket = ring.submit(
        context_sequence=context_sequence,
        logical_start=logical_start,
        physical_slots=tuple(range(rows)),
        producer_stream=0,
    )
    assert ticket is not None
    return ticket


def _batch(
    *,
    request_id: str,
    context_digest: str,
    batch_index: int,
    logical_start: int,
    logical_end: int,
) -> SnapshotBatch:
    return SnapshotBatch(
        request_id=request_id,
        context_digest=context_digest,
        batch_index=batch_index,
        logical_start=logical_start,
        logical_end=logical_end,
        chunk_tokens=256,
        block_ids=(),
    )


def test_glm52_profile_constants_lock_standalone_winner() -> None:
    assert len(GLM52_LAYER_ORDER) == 101
    assert [layer.source_ordinal for layer in GLM52_LAYER_ORDER[:79]] == list(range(79))
    assert [layer.source_ordinal for layer in GLM52_LAYER_ORDER[79:]] == list(range(22))
    assert GLM52_MACRO_PAYLOAD_BYTES == 32_743_424


def test_macro_ready_view_translates_byte_exact_chunks_without_mtp_record() -> None:
    ring, _backend = _ring()
    ticket = _submit(
        ring,
        context_sequence=1,
        logical_start=0,
        rows=GLM52_MACRO_ROWS,
    )
    view = ring.claim(ticket)
    assert view is not None
    assert view.used_bytes == GLM52_MACRO_PAYLOAD_BYTES
    translator = Glm52ReadyViewTranslator.for_ring(ring, dcp_rank=2)
    first = None
    last = None
    count = 0
    for count, chunk in enumerate(translator.iter_chunks(view), start=1):
        assert _encode_chunk(chunk) == _encode_chunk(
            _reference_chunk(
                logical_start=(count - 1) * 256,
                dcp_rank=2,
                batch_row_start=(count - 1) * 64,
            )
        )
        if first is None:
            first = chunk
        last = chunk

    assert count == 16
    assert first is not None
    assert last is not None
    assert first.records[StateRecord.TARGET_CKV] == _record_bytes(0, 1024, 0, 64)
    assert first.records[StateRecord.SPARSE_INDEXER] == _record_bytes(1, 1024, 0, 64)
    assert unpack_positions(first.records[StateRecord.LOGICAL_POSITIONS]) == tuple(
        range(2, 256, 4)
    )
    assert last.records[StateRecord.TARGET_CKV] == _record_bytes(0, 1024, 960, 64)
    assert unpack_positions(last.records[StateRecord.LOGICAL_POSITIONS]) == tuple(
        range(3842, 4096, 4)
    )
    assert StateRecord.MTP_DRAFT_KV not in first.records
    ring.release(ticket)
    ring.shutdown()


def test_journal_writer_commits_only_full_expected_context(
    tmp_path: Path,
) -> None:
    ring, backend = _ring()
    store = ManifestStore(tmp_path)
    digest = "a" * 64
    writer = ManifestSnapshotJournalWriter(
        store=store,
        identity=_identity(),
        translator=Glm52ReadyViewTranslator.for_ring(ring, dcp_rank=0),
    )
    assert isinstance(writer, SnapshotJournalWriter)
    transaction = writer.begin_context(
        request_id="request-a",
        context_digest=digest,
        span_tokens=512,
    )
    assert isinstance(transaction, SnapshotJournalTransaction)

    first = _submit(ring, context_sequence=2, logical_start=0, rows=64)
    first_view = ring.claim(first)
    assert first_view is not None
    first_completion = transaction.submit_ready(
        _batch(
            request_id="request-a",
            context_digest=digest,
            batch_index=0,
            logical_start=0,
            logical_end=256,
        ),
        first_view,
    )
    assert isinstance(first_completion, WriterCompletion)
    first_completion.synchronize()
    first_completion.result()
    assert first_completion.query()
    assert backend.releases == 0
    ring.release(first)
    assert backend.releases == 1
    assert not store.lookup(_identity(), digest).is_hit
    with pytest.raises(RuntimeError, match="exact full-span coverage"):
        transaction.commit_manifest()

    second = _submit(ring, context_sequence=2, logical_start=256, rows=64)
    second_view = ring.claim(second)
    assert second_view is not None
    second_completion = transaction.submit_ready(
        _batch(
            request_id="request-a",
            context_digest=digest,
            batch_index=1,
            logical_start=256,
            logical_end=512,
        ),
        second_view,
    )
    second_completion.synchronize()
    second_completion.result()
    assert backend.releases == 1
    ring.release(second)
    receipt = transaction.commit_manifest()

    assert receipt.committed_tokens == 512
    assert transaction.state is JournalState.COMMITTED
    assert transaction.appended_tokens == 512
    lookup = store.lookup(_identity(), digest)
    assert lookup.is_hit, lookup.reason
    restored = store.restore(lookup)
    assert restored is not None
    assert [chunk.logical_start for chunk in restored] == [0, 256]
    assert restored[1].records[StateRecord.TARGET_CKV] == _record_bytes(0, 64, 0, 64)
    assert backend.releases == 2
    writer.shutdown()
    ring.shutdown()


def test_final_batch_timing_covers_writer_and_manifest_boundaries(
    tmp_path: Path,
) -> None:
    class TimingTrace:
        def __init__(self) -> None:
            self.registrations: list[tuple[str, int, int]] = []
            self.stages: list[str] = []

        def register_final(
            self,
            request_id: str,
            batch_index: int,
            span_tokens: int,
        ) -> None:
            self.registrations.append((request_id, batch_index, span_tokens))

        def mark(
            self,
            _request_id: str,
            _batch_index: int,
            stage: str,
            *,
            at_ns: int | None = None,
        ) -> None:
            del at_ns
            self.stages.append(stage)

    timing = TimingTrace()
    ring, _backend = _ring()
    writer = ManifestSnapshotJournalWriter(
        store=ManifestStore(tmp_path),
        identity=_identity(),
        translator=Glm52ReadyViewTranslator.for_ring(ring, dcp_rank=0),
        timing_trace=timing,
    )
    digest = "f" * 64
    transaction = writer.begin_context(
        request_id="timed-request",
        context_digest=digest,
        span_tokens=256,
    )
    ticket = _submit(ring, context_sequence=17, logical_start=0, rows=64)
    view = ring.claim(ticket)
    assert view is not None

    completion = transaction.submit_ready(
        _batch(
            request_id="timed-request",
            context_digest=digest,
            batch_index=0,
            logical_start=0,
            logical_end=256,
        ),
        view,
    )
    completion.synchronize()
    completion.result()
    ring.release(ticket)
    transaction.commit_manifest()

    assert timing.registrations == [("timed-request", 0, 256)]
    assert timing.stages == [
        "writer_enqueued",
        "writer_start",
        "writer_end",
        "manifest_publish_begin",
        "manifest_publish_end",
    ]
    writer.shutdown()
    ring.shutdown()


class _BlockingTranslator:
    def __init__(self, delegate: Glm52ReadyViewTranslator) -> None:
        self.delegate = delegate
        self.dcp_degree = delegate.dcp_degree
        self.dcp_rank = delegate.dcp_rank
        self.started = threading.Event()
        self.resume = threading.Event()

    def batch_logical_tokens(self, view):
        return self.delegate.batch_logical_tokens(view)

    def iter_chunks(self, view):
        self.started.set()
        if not self.resume.wait(5):
            raise TimeoutError("test translator remained blocked")
        yield from self.delegate.iter_chunks(view)


def test_writer_backpressure_aborts_without_manifest_or_waiting(
    tmp_path: Path,
) -> None:
    ring, backend = _ring()
    translator = _BlockingTranslator(
        Glm52ReadyViewTranslator.for_ring(ring, dcp_rank=0)
    )
    store = ManifestStore(tmp_path)
    digest = "b" * 64
    writer = ManifestSnapshotJournalWriter(
        store=store,
        identity=_identity(),
        max_pending_batches=1,
        translator=translator,
    )
    transaction = writer.begin_context(
        request_id="request-b",
        context_digest=digest,
        span_tokens=512,
    )
    first = _submit(ring, context_sequence=3, logical_start=0, rows=64)
    first_view = ring.claim(first)
    assert first_view is not None
    first_completion = transaction.submit_ready(
        _batch(
            request_id="request-b",
            context_digest=digest,
            batch_index=0,
            logical_start=0,
            logical_end=256,
        ),
        first_view,
    )
    assert translator.started.wait(5)
    second = _submit(ring, context_sequence=3, logical_start=256, rows=64)
    second_view = ring.claim(second)
    assert second_view is not None

    with pytest.raises(SnapshotWriterBackpressure, match="capacity exhausted"):
        transaction.submit_ready(
            _batch(
                request_id="request-b",
                context_digest=digest,
                batch_index=1,
                logical_start=256,
                logical_end=512,
            ),
            second_view,
        )
    assert transaction.state is JournalState.ABORTED
    assert backend.releases == 0
    # submit_ready raised before retaining second_view, so runtime may release
    # this ticket immediately even though the first writer is still blocked.
    ring.release(second)
    translator.resume.set()
    first_completion.synchronize()
    first_completion.result()
    assert backend.releases == 1
    ring.release(first)
    assert backend.releases == 2
    assert not store.lookup(_identity(), digest).is_hit
    writer.shutdown()
    ring.shutdown()


class _FailingTranslator:
    def __init__(self, delegate: Glm52ReadyViewTranslator) -> None:
        self.delegate = delegate
        self.dcp_degree = delegate.dcp_degree
        self.dcp_rank = delegate.dcp_rank

    def batch_logical_tokens(self, view):
        return self.delegate.batch_logical_tokens(view)

    @staticmethod
    def iter_chunks(_view):
        raise RuntimeError("synthetic canonical encoding failure")
        yield  # pragma: no cover


class _BlockingManifestTransaction:
    def __init__(self) -> None:
        self.append_started = threading.Event()
        self.resume_append = threading.Event()
        self.abort_called = threading.Event()
        self._lock = threading.RLock()

    def append_chunk(self, _chunk) -> None:
        with self._lock:
            self.append_started.set()
            if not self.resume_append.wait(5):
                raise TimeoutError("test manifest append remained blocked")

    @staticmethod
    def commit_manifest():
        raise AssertionError("aborted test transaction must not commit")

    def abort(self) -> None:
        with self._lock:
            self.abort_called.set()


class _BlockingManifestStore:
    def __init__(self, transaction: _BlockingManifestTransaction) -> None:
        self.transaction = transaction

    def begin_context(self, **_kwargs):
        return self.transaction


def test_abort_returns_without_waiting_for_inflight_manifest_io() -> None:
    ring, backend = _ring()
    manifest_transaction = _BlockingManifestTransaction()
    writer = ManifestSnapshotJournalWriter(
        store=_BlockingManifestStore(manifest_transaction),
        identity=_identity(),
        translator=Glm52ReadyViewTranslator.for_ring(ring, dcp_rank=0),
    )
    transaction = writer.begin_context(
        request_id="request-cancel",
        context_digest="f" * 64,
        span_tokens=256,
    )
    ticket = _submit(ring, context_sequence=6, logical_start=0, rows=64)
    view = ring.claim(ticket)
    assert view is not None
    completion = transaction.submit_ready(
        _batch(
            request_id="request-cancel",
            context_digest="f" * 64,
            batch_index=0,
            logical_start=0,
            logical_end=256,
        ),
        view,
    )
    assert manifest_transaction.append_started.wait(5)

    abort_returned = threading.Event()
    abort_errors: list[BaseException] = []

    def abort_journal() -> None:
        try:
            transaction.abort()
        except BaseException as error:
            abort_errors.append(error)
        finally:
            abort_returned.set()

    abort_thread = threading.Thread(target=abort_journal)
    abort_thread.start()
    returned_before_io = abort_returned.wait(0.5)
    abort_called_before_io = manifest_transaction.abort_called.is_set()

    manifest_transaction.resume_append.set()
    abort_thread.join(5)
    completion.synchronize()
    completion.result()

    assert returned_before_io
    assert not abort_errors
    assert not abort_called_before_io
    assert manifest_transaction.abort_called.is_set()
    assert transaction.state is JournalState.ABORTED
    assert backend.releases == 0
    ring.release(ticket)
    writer.shutdown()
    ring.shutdown()


def test_translation_error_aborts_but_runtime_still_owns_claimed_ticket(
    tmp_path: Path,
) -> None:
    ring, backend = _ring()
    store = ManifestStore(tmp_path)
    digest = "c" * 64
    writer = ManifestSnapshotJournalWriter(
        store=store,
        identity=_identity(),
        translator=_FailingTranslator(
            Glm52ReadyViewTranslator.for_ring(ring, dcp_rank=0)
        ),
    )
    transaction = writer.begin_context(
        request_id="request-c",
        context_digest=digest,
        span_tokens=256,
    )
    ticket = _submit(ring, context_sequence=4, logical_start=0, rows=64)
    view = ring.claim(ticket)
    assert view is not None
    completion = transaction.submit_ready(
        _batch(
            request_id="request-c",
            context_digest=digest,
            batch_index=0,
            logical_start=0,
            logical_end=256,
        ),
        view,
    )
    completion.synchronize()
    with pytest.raises(RuntimeError, match="synthetic canonical encoding failure"):
        completion.result()
    assert transaction.state is JournalState.ABORTED
    assert not store.lookup(_identity(), digest).is_hit
    # Failure completion transfers no ring ownership back implicitly.
    assert backend.releases == 0
    ring.release(ticket)
    assert backend.releases == 1
    writer.shutdown()
    ring.shutdown()


def test_completion_is_hidden_until_transaction_bookkeeping_finishes(
    tmp_path: Path,
) -> None:
    ring, _backend = _ring()
    writer = ManifestSnapshotJournalWriter(
        store=ManifestStore(tmp_path),
        identity=_identity(),
        translator=Glm52ReadyViewTranslator.for_ring(ring, dcp_rank=0),
        max_pending_batches=1,
    )
    transaction = writer.begin_context(
        request_id="request-boundary",
        context_digest="7" * 64,
        span_tokens=256,
    )
    bookkeeping_started = threading.Event()
    resume_bookkeeping = threading.Event()
    original_completion_done = transaction._completion_done

    def block_completion_bookkeeping(completion) -> None:
        bookkeeping_started.set()
        if not resume_bookkeeping.wait(5):
            raise TimeoutError("test completion bookkeeping remained blocked")
        original_completion_done(completion)

    transaction._completion_done = block_completion_bookkeeping
    ticket = _submit(ring, context_sequence=7, logical_start=0, rows=64)
    view = ring.claim(ticket)
    assert view is not None
    completion = transaction.submit_ready(
        _batch(
            request_id="request-boundary",
            context_digest="7" * 64,
            batch_index=0,
            logical_start=0,
            logical_end=256,
        ),
        view,
    )
    assert bookkeeping_started.wait(5)
    visible_before_bookkeeping = completion.query()

    second_transaction = writer.begin_context(
        request_id="request-boundary-2",
        context_digest="8" * 64,
        span_tokens=256,
    )
    second_ticket = _submit(
        ring,
        context_sequence=8,
        logical_start=0,
        rows=64,
    )
    second_view = ring.claim(second_ticket)
    assert second_view is not None
    with pytest.raises(SnapshotWriterBackpressure, match="capacity exhausted"):
        second_transaction.submit_ready(
            _batch(
                request_id="request-boundary-2",
                context_digest="8" * 64,
                batch_index=0,
                logical_start=0,
                logical_end=256,
            ),
            second_view,
        )
    assert second_transaction.state is JournalState.ABORTED
    ring.release(second_ticket)

    resume_bookkeeping.set()
    completion.synchronize()
    completion.result()
    ring.release(ticket)
    receipt = transaction.commit_manifest()

    assert not visible_before_bookkeeping
    assert receipt.committed_tokens == 256
    writer.shutdown()
    ring.shutdown()


def test_out_of_order_submission_aborts_before_retaining_view(
    tmp_path: Path,
) -> None:
    ring, backend = _ring()
    store = ManifestStore(tmp_path)
    digest = "d" * 64
    writer = ManifestSnapshotJournalWriter(
        store=store,
        identity=_identity(),
        translator=Glm52ReadyViewTranslator.for_ring(ring, dcp_rank=0),
    )
    transaction = writer.begin_context(
        request_id="request-d",
        context_digest=digest,
        span_tokens=512,
    )
    ticket = _submit(ring, context_sequence=5, logical_start=256, rows=64)
    view = ring.claim(ticket)
    assert view is not None
    with pytest.raises(SnapshotTranslationError, match="disagrees"):
        transaction.submit_ready(
            _batch(
                request_id="request-d",
                context_digest=digest,
                batch_index=1,
                logical_start=256,
                logical_end=512,
            ),
            view,
        )

    assert transaction.state is JournalState.ABORTED
    assert backend.releases == 0
    ring.release(ticket)
    assert not tuple(tmp_path.rglob("*.json"))
    writer.shutdown()
    ring.shutdown()


def test_runtime_and_manifest_writer_compose_with_runtime_owned_ring(
    tmp_path: Path,
) -> None:
    ring, backend = _ring()
    store = ManifestStore(tmp_path)
    digest = "9" * 64
    writer = ManifestSnapshotJournalWriter(
        store=store,
        identity=_identity(),
        translator=Glm52ReadyViewTranslator.for_ring(ring, dcp_rank=0),
    )
    runtime = StreamingSnapshotRuntime(
        StreamingSnapshotRuntimeConfig(
            enabled=True,
            block_size=16,
            dcp_degree=4,
            dcp_rank=0,
            gather_abort_timeout_seconds=1,
            poll_sleep_seconds=0,
        ),
        planner=StreamingSnapshotCoordinator(
            chunk_tokens=256,
            chunks_per_batch=1,
            max_inflight_batches=2,
        ),
        leases=BlockLeaseRegistry(
            LeaseCapacity(
                max_active_leases=2,
                max_leased_blocks=8,
            )
        ),
        ring=ring,
        writer=writer,
    )

    assert runtime.begin_context(
        request_id="request-runtime",
        context_digest=digest,
        span_tokens=512,
    )
    offer = runtime.accept_completed_prefill(
        request_id="request-runtime",
        completed_tokens=512,
        block_table=tuple(range(100, 108)),
        producer_stream=77,
    )
    assert offer.submitted_batches == 2

    deadline = time.monotonic() + 5
    while runtime.active_contexts and time.monotonic() < deadline:
        runtime.poll()
        time.sleep(0.001)

    assert runtime.active_contexts == 0
    assert backend.releases == 2
    assert store.lookup(_identity(), digest).is_hit
    assert runtime.take_committed() == {digest}
    assert runtime.take_committed() == set()
    runtime.shutdown()
    writer.shutdown()


def test_runtime_publisher_uses_connector_store_abi_without_class_identity_leak(
    tmp_path: Path,
) -> None:
    """Production loads the store through the flat staged-module shim.

    The publisher imports the same ABI as a package. Keep this cross-module
    composition locked down so duplicate Python class/Enum identities cannot
    make an otherwise valid live journal fail only after the first gather.
    """

    ring, backend = _ring()
    identity = connector_store.CacheIdentity(**dataclasses.asdict(_identity()))
    store = connector_store.ManifestStore(tmp_path)
    digest = "8" * 64
    writer = ManifestSnapshotJournalWriter(
        store=store,
        identity=identity,
        translator=Glm52ReadyViewTranslator.for_ring(ring, dcp_rank=0),
    )
    runtime = StreamingSnapshotRuntime(
        StreamingSnapshotRuntimeConfig(
            enabled=True,
            block_size=16,
            dcp_degree=4,
            dcp_rank=0,
            gather_abort_timeout_seconds=1,
            poll_sleep_seconds=0,
        ),
        planner=StreamingSnapshotCoordinator(
            chunk_tokens=256,
            chunks_per_batch=1,
            max_inflight_batches=2,
        ),
        leases=BlockLeaseRegistry(
            LeaseCapacity(max_active_leases=2, max_leased_blocks=8)
        ),
        ring=ring,
        writer=writer,
    )
    assert runtime.begin_context(
        request_id="flat-store-runtime",
        context_digest=digest,
        span_tokens=256,
    )
    assert (
        runtime.accept_completed_prefill(
            request_id="flat-store-runtime",
            completed_tokens=256,
            block_table=tuple(range(100, 104)),
            producer_stream=77,
        ).submitted_batches
        == 1
    )

    deadline = time.monotonic() + 5
    while runtime.active_contexts and time.monotonic() < deadline:
        runtime.poll()
        time.sleep(0.001)

    assert runtime.active_contexts == 0
    assert backend.releases == 1
    assert store.lookup(identity, digest).is_hit
    assert runtime.take_committed() == {digest}
    runtime.shutdown()
    writer.shutdown()


def test_writer_thread_start_failure_occurs_before_view_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ring, backend = _ring()

    def fail_thread_start(_thread) -> None:
        raise RuntimeError("synthetic writer thread start failure")

    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", fail_thread_start)
        with pytest.raises(RuntimeError, match="synthetic writer thread start"):
            ManifestSnapshotJournalWriter(
                store=ManifestStore(tmp_path),
                identity=_identity(),
                translator=Glm52ReadyViewTranslator.for_ring(
                    ring,
                    dcp_rank=0,
                ),
            )

    assert not backend.entries
    writer = ManifestSnapshotJournalWriter(
        store=ManifestStore(tmp_path),
        identity=_identity(),
        translator=Glm52ReadyViewTranslator.for_ring(ring, dcp_rank=0),
    )
    writer.shutdown()
    ring.shutdown()


def test_wrong_identity_policy_is_rejected_before_writer_start(
    tmp_path: Path,
) -> None:
    ring, _backend = _ring()
    wrong = dataclasses.replace(_identity(), draft_kv_policy="separate")
    with pytest.raises(ValueError, match="colocated MTP"):
        ManifestSnapshotJournalWriter(
            store=ManifestStore(tmp_path),
            identity=wrong,
            translator=Glm52ReadyViewTranslator.for_ring(ring, dcp_rank=0),
        )
    ring.shutdown()
