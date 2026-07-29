"""Bounded READY-view translation and manifest-last publication.

This module is intentionally not wired into the connector feature gate. The
runtime exclusively owns ring tickets and lends each claimed, immutable view
to this writer. Canonical chunk bytes are copied one chunk at a time, durable
append completes, and the returned completion then permits the runtime to
release its ticket. The manifest is published only after exact full-span
coverage.
"""

from __future__ import annotations

import concurrent.futures
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterator, Protocol, Sequence

from sparkcache.persistent_context_cache.cache_manifest import (
    CacheIdentity,
    CommitReceipt,
    ContextChunk,
    ManifestStore,
    ManifestTransaction,
    StateRecord,
)
from sparkcache.spark_context_cache_codec import CHUNK_TOKENS, pack_positions

from .native_ring import NativeSnapshotRing, SnapshotSourceSpec, SnapshotView
from .planner import SnapshotBatch


GLM52_ARENA_MODE_MAPPED_HOST = 1
GLM52_RING_DEPTH = 2
GLM52_SLOT_BYTES = 64 * 1024 * 1024
GLM52_MACRO_ROWS = 1024
GLM52_DCP_DEGREE = 4
GLM52_ROWS_PER_CHUNK = CHUNK_TOKENS // GLM52_DCP_DEGREE
GLM52_TARGET_LAYERS = 79
GLM52_TARGET_BYTES_PER_TOKEN = 368
GLM52_INDEXER_LAYERS = 22
GLM52_INDEXER_BYTES_PER_TOKEN = 132
GLM52_SOURCE_COUNT = GLM52_TARGET_LAYERS + GLM52_INDEXER_LAYERS
GLM52_MACRO_PAYLOAD_BYTES = GLM52_MACRO_ROWS * (
    GLM52_TARGET_LAYERS * GLM52_TARGET_BYTES_PER_TOKEN
    + GLM52_INDEXER_LAYERS * GLM52_INDEXER_BYTES_PER_TOKEN
)

_TARGET_KIND = 0
_INDEXER_KIND = 1


@dataclass(frozen=True, slots=True)
class Glm52LayerOrder:
    name: str
    record_kind: int
    source_ordinal: int
    bytes_per_token: int


# Frozen native and canonical order. The 79 target sources include colocated
# MTP state; there is deliberately no separate native or persisted MTP record.
GLM52_LAYER_ORDER = tuple(
    Glm52LayerOrder(
        name=f"model.layers.{ordinal:02d}.mla",
        record_kind=_TARGET_KIND,
        source_ordinal=ordinal,
        bytes_per_token=GLM52_TARGET_BYTES_PER_TOKEN,
    )
    for ordinal in range(GLM52_TARGET_LAYERS)
) + tuple(
    Glm52LayerOrder(
        name=f"model.layers.{ordinal:02d}.indexer",
        record_kind=_INDEXER_KIND,
        source_ordinal=ordinal,
        bytes_per_token=GLM52_INDEXER_BYTES_PER_TOKEN,
    )
    for ordinal in range(GLM52_INDEXER_LAYERS)
)


class SnapshotTranslationError(ValueError):
    """A claimed READY view cannot represent canonical SparkCache chunks."""


class JournalState(str, Enum):
    OPEN = "open"
    COMMITTED = "committed"
    ABORTED = "aborted"


class SnapshotWriterBackpressure(RuntimeError):
    """The bounded writer could not accept ownership without waiting."""


class ReadyViewTranslator(Protocol):
    dcp_degree: int
    dcp_rank: int

    def batch_logical_tokens(self, view: SnapshotView) -> int: ...

    def iter_chunks(self, view: SnapshotView) -> Iterator[ContextChunk]: ...


class Glm52ReadyViewTranslator:
    """Translate the proven DCP4 GLM-5.2 ring layout one chunk at a time."""

    def __init__(
        self,
        sources: Sequence[SnapshotSourceSpec],
        *,
        dcp_rank: int,
        context_chunk_factory: Callable[..., ContextChunk] = ContextChunk,
        state_record_type: type[StateRecord] = StateRecord,
        pack_positions_function: Callable[[Sequence[int]], bytes] = pack_positions,
    ) -> None:
        if not 0 <= dcp_rank < GLM52_DCP_DEGREE:
            raise ValueError("dcp_rank must be in [0, 4)")
        self.dcp_degree = GLM52_DCP_DEGREE
        self.dcp_rank = dcp_rank
        self._sources = tuple(sources)
        self._context_chunk_factory = context_chunk_factory
        self._state_record_type = state_record_type
        self._pack_positions = pack_positions_function
        self._validate_source_order()

    @classmethod
    def for_ring(
        cls,
        ring: NativeSnapshotRing,
        *,
        dcp_rank: int,
        context_chunk_factory: Callable[..., ContextChunk] = ContextChunk,
        state_record_type: type[StateRecord] = StateRecord,
        pack_positions_function: Callable[[Sequence[int]], bytes] = pack_positions,
    ) -> "Glm52ReadyViewTranslator":
        config = ring.config
        if (
            config.arena_mode != GLM52_ARENA_MODE_MAPPED_HOST
            or config.slot_count != GLM52_RING_DEPTH
            or config.slot_bytes != GLM52_SLOT_BYTES
            or config.max_sources < GLM52_SOURCE_COUNT
            or config.max_rows < GLM52_MACRO_ROWS
        ):
            raise SnapshotTranslationError(
                "ring config is not the proven mapped/depth-2/64-MiB GLM profile"
            )
        sources = ring.configured_sources
        if sources is None:
            raise SnapshotTranslationError(
                "ring source inventory must be configured before translation"
            )
        return cls(
            sources,
            dcp_rank=dcp_rank,
            context_chunk_factory=context_chunk_factory,
            state_record_type=state_record_type,
            pack_positions_function=pack_positions_function,
        )

    def batch_logical_tokens(self, view: SnapshotView) -> int:
        rows = view.ticket.row_count
        if rows <= 0 or rows % GLM52_ROWS_PER_CHUNK:
            raise SnapshotTranslationError(
                "READY row count must cover complete 256-token chunks"
            )
        return rows * self.dcp_degree

    def iter_chunks(self, view: SnapshotView) -> Iterator[ContextChunk]:
        logical_tokens = self.batch_logical_tokens(view)
        logical_start = view.ticket.logical_start
        if logical_start % CHUNK_TOKENS:
            raise SnapshotTranslationError(
                "READY logical_start must be 256-token aligned"
            )
        logical_end = logical_start + logical_tokens
        if logical_end > 0x100000000:
            raise SnapshotTranslationError(
                "logical positions exceed the uint32 storage ABI"
            )

        rows = view.ticket.row_count
        expected_target = rows * GLM52_TARGET_LAYERS * GLM52_TARGET_BYTES_PER_TOKEN
        expected_indexer = rows * GLM52_INDEXER_LAYERS * GLM52_INDEXER_BYTES_PER_TOKEN
        if (
            view.record_mask != 0b011
            or view.record_lengths[_TARGET_KIND] != expected_target
            or view.record_lengths[_INDEXER_KIND] != expected_indexer
            or view.record_lengths[2:] != (0, 0)
        ):
            raise SnapshotTranslationError(
                "READY records do not match the colocated-MTP GLM layout"
            )

        for batch_row_start in range(0, rows, GLM52_ROWS_PER_CHUNK):
            chunk_start = logical_start + batch_row_start * self.dcp_degree
            chunk_end = chunk_start + CHUNK_TOKENS
            positions = tuple(
                range(
                    chunk_start + self.dcp_rank,
                    chunk_end,
                    self.dcp_degree,
                )
            )
            state_record = self._state_record_type
            records = {
                state_record.LOGICAL_POSITIONS: self._pack_positions(positions),
                state_record.TARGET_CKV: self._copy_chunk_record(
                    view,
                    record_kind=_TARGET_KIND,
                    layer_count=GLM52_TARGET_LAYERS,
                    width=GLM52_TARGET_BYTES_PER_TOKEN,
                    batch_rows=rows,
                    batch_row_start=batch_row_start,
                ),
                state_record.SPARSE_INDEXER: self._copy_chunk_record(
                    view,
                    record_kind=_INDEXER_KIND,
                    layer_count=GLM52_INDEXER_LAYERS,
                    width=GLM52_INDEXER_BYTES_PER_TOKEN,
                    batch_rows=rows,
                    batch_row_start=batch_row_start,
                ),
            }
            yield self._context_chunk_factory(
                logical_start=chunk_start,
                logical_end=chunk_end,
                records=records,
            )

    def _validate_source_order(self) -> None:
        if len(self._sources) != len(GLM52_LAYER_ORDER):
            raise SnapshotTranslationError(
                "GLM source inventory must contain 79 target and 22 indexer layers"
            )
        for source, expected in zip(self._sources, GLM52_LAYER_ORDER, strict=True):
            if (
                source.record_kind != expected.record_kind
                or source.source_layer_ordinal != expected.source_ordinal
                or source.bytes_per_token != expected.bytes_per_token
            ):
                raise SnapshotTranslationError(
                    f"source inventory disagrees at {expected.name}"
                )

    @staticmethod
    def _copy_chunk_record(
        view: SnapshotView,
        *,
        record_kind: int,
        layer_count: int,
        width: int,
        batch_rows: int,
        batch_row_start: int,
    ) -> bytes:
        record = view.record(record_kind)
        layer_bytes = batch_rows * width
        chunk_layer_bytes = GLM52_ROWS_PER_CHUNK * width
        parts: list[memoryview] = []
        try:
            for layer in range(layer_count):
                start = layer * layer_bytes + batch_row_start * width
                parts.append(record[start : start + chunk_layer_bytes])
            # This is the one required ownership transition: ContextChunk's
            # immutable bytes are copied while the ring ticket is WRITING.
            return b"".join(parts)
        finally:
            for part in parts:
                part.release()
            record.release()


class ManifestWriterCompletion:
    """Runtime-compatible completion for one borrowed READY view."""

    def __init__(self) -> None:
        self._future: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._before_visible: Callable[[], None] | None = None

    def query(self) -> bool:
        return self._future.done()

    def synchronize(self) -> None:
        concurrent.futures.wait((self._future,))

    def result(self) -> None:
        if not self._future.done():
            raise RuntimeError("writer completion is not ready")
        self._future.result()

    def _set_before_visible(
        self,
        callback: Callable[[], None],
    ) -> None:
        self._before_visible = callback

    def _resolve(
        self,
        error: BaseException | None,
        release_capacity: Callable[[], None],
    ) -> None:
        callback = self._before_visible
        self._before_visible = None
        if callback is not None:
            try:
                callback()
            except BaseException as callback_error:
                if error is None:
                    error = callback_error
        try:
            release_capacity()
        except BaseException as capacity_error:
            if error is None:
                error = capacity_error
        if error is None:
            self._future.set_result(None)
        else:
            self._future.set_exception(error)


class ManifestSnapshotJournalWriter:
    """Shared one-thread writer with a strict global outstanding-work bound."""

    def __init__(
        self,
        *,
        store: ManifestStore,
        identity: CacheIdentity,
        translator: ReadyViewTranslator,
        max_pending_batches: int = GLM52_RING_DEPTH,
        timing_trace: Any | None = None,
    ) -> None:
        if not 1 <= max_pending_batches <= GLM52_RING_DEPTH:
            raise ValueError("max_pending_batches must be one or two")
        if (
            identity.chunk_tokens != CHUNK_TOKENS
            or identity.dcp_degree != GLM52_DCP_DEGREE
            or not 0 <= identity.dcp_shard_rank < identity.dcp_degree
            or identity.boundary_hidden_policy != "live_forward"
            or identity.draft_kv_policy != "colocated_target"
        ):
            raise ValueError(
                "writer requires DCP4, 256-token chunks, live boundary "
                "state, and colocated MTP"
            )
        if (
            translator.dcp_degree != identity.dcp_degree
            or translator.dcp_rank != identity.dcp_shard_rank
        ):
            raise ValueError("translator DCP identity disagrees with manifest identity")
        self.store = store
        self.identity = identity
        self.translator = translator
        self.max_pending_batches = max_pending_batches
        self._timing_trace = timing_trace
        self._capacity = threading.BoundedSemaphore(max_pending_batches)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="spark-cache-streaming-writer",
        )
        try:
            # ThreadPoolExecutor queues a work item before it tries to start its
            # first worker. Prestarting without a borrowed view makes all later
            # submit failures atomic: with the sole worker alive, enqueue
            # cannot fail during thread creation after retaining a view.
            self._executor.submit(lambda: None).result()
        except BaseException:
            self._executor.shutdown(wait=False, cancel_futures=True)
            raise
        self._transactions: dict[str, ManifestSnapshotJournalTransaction] = {}
        self._closed = False
        self._lock = threading.RLock()

    def begin_context(
        self,
        *,
        request_id: str,
        context_digest: str,
        span_tokens: int,
    ) -> "ManifestSnapshotJournalTransaction":
        if not request_id:
            raise ValueError("request_id must be nonempty")
        if span_tokens <= 0 or span_tokens % CHUNK_TOKENS:
            raise ValueError("span_tokens must be a positive 256-token multiple")
        with self._lock:
            if self._closed:
                raise RuntimeError("snapshot journal writer is shut down")
            if request_id in self._transactions:
                raise ValueError("request_id already owns a snapshot journal")
            transaction = ManifestSnapshotJournalTransaction(
                owner=self,
                request_id=request_id,
                context_digest=context_digest,
                span_tokens=span_tokens,
                transaction=self.store.begin_context(
                    identity=self.identity,
                    context_digest=context_digest,
                    span_tokens=span_tokens,
                ),
            )
            self._transactions[request_id] = transaction
            return transaction

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            transactions = tuple(self._transactions.values())
        for transaction in transactions:
            transaction.abort()
        # Every queued task may still hold a runtime-owned ring view. Waiting is
        # mandatory so callers cannot mistake writer shutdown for permission to
        # destroy the ring while a borrowed view is still being read.
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _submit(
        self,
        function: Callable[[], None],
        completion: ManifestWriterCompletion,
    ) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("snapshot journal writer is shut down")
            if not self._capacity.acquire(blocking=False):
                raise SnapshotWriterBackpressure(
                    "streaming journal writer capacity exhausted"
                )
            try:
                self._executor.submit(
                    self._run,
                    function,
                    completion,
                )
            except BaseException:
                self._capacity.release()
                raise

    def _run(
        self,
        function: Callable[[], None],
        completion: ManifestWriterCompletion,
    ) -> None:
        error: BaseException | None = None
        try:
            function()
        except BaseException as caught:
            error = caught

        # Transaction bookkeeping and the capacity credit both complete before
        # readiness becomes observable. Until then runtime still owns the
        # borrowed view, so releasing the credit earlier would exceed the
        # configured maximum number of outstanding views.
        completion._resolve(error, self._capacity.release)

    def _retire(
        self,
        request_id: str,
        transaction: "ManifestSnapshotJournalTransaction",
    ) -> None:
        with self._lock:
            if self._transactions.get(request_id) is transaction:
                del self._transactions[request_id]

    def _register_final_timing(
        self,
        request_id: str,
        batch_index: int,
        span_tokens: int,
    ) -> None:
        trace = self._timing_trace
        if trace is None:
            return
        try:
            trace.register_final(request_id, batch_index, span_tokens)
        except Exception:
            return

    def _mark_timing(
        self,
        request_id: str,
        batch_index: int,
        stage: str,
    ) -> None:
        trace = self._timing_trace
        if trace is None:
            return
        try:
            trace.mark(request_id, batch_index, stage)
        except Exception:
            return


class ManifestSnapshotJournalTransaction:
    """Runtime-facing transaction that borrows but never releases ring views."""

    def __init__(
        self,
        *,
        owner: ManifestSnapshotJournalWriter,
        request_id: str,
        context_digest: str,
        span_tokens: int,
        transaction: ManifestTransaction,
    ) -> None:
        self._owner = owner
        self.request_id = request_id
        self.context_digest = context_digest
        self.span_tokens = span_tokens
        self._transaction = transaction
        self._state = JournalState.OPEN
        self._next_batch_index = 0
        self._next_submitted = 0
        self._appended_tokens = 0
        self._pending: set[ManifestWriterCompletion] = set()
        self._receipt: CommitReceipt | None = None
        self._error: BaseException | None = None
        self._abort_finalized = False
        self._lock = threading.RLock()

    @property
    def state(self) -> JournalState:
        with self._lock:
            return self._state

    @property
    def appended_tokens(self) -> int:
        with self._lock:
            return self._appended_tokens

    def submit_ready(
        self,
        batch: SnapshotBatch,
        view: SnapshotView,
    ) -> ManifestWriterCompletion:
        """Borrow ``view`` atomically or raise before retaining it."""

        with self._lock:
            self._require_open()
            completion: ManifestWriterCompletion | None = None
            try:
                self._validate_submission(batch, view)
                completion = ManifestWriterCompletion()
                self._pending.add(completion)
                completion._set_before_visible(
                    lambda: self._completion_done(completion)
                )
                self._next_batch_index += 1
                self._next_submitted = batch.logical_end
                if batch.logical_end == self.span_tokens:
                    self._owner._register_final_timing(
                        self.request_id,
                        batch.batch_index,
                        self.span_tokens,
                    )
                    self._owner._mark_timing(
                        self.request_id,
                        batch.batch_index,
                        "writer_enqueued",
                    )

                # This is deliberately the final potentially failing action.
                # If enqueue raises, no worker can retain or inspect ``view``.
                # If it returns, ``completion`` is already fully registered.
                self._owner._submit(
                    lambda: self._append_batch(batch, view),
                    completion,
                )
            except BaseException as error:
                if completion is not None:
                    self._pending.discard(completion)
                self._abort_locked(error)
                raise
            assert completion is not None
            return completion

    def commit_manifest(self) -> CommitReceipt:
        with self._lock:
            if self._state is JournalState.COMMITTED:
                assert self._receipt is not None
                return self._receipt
            self._require_open()
            if self._pending:
                raise RuntimeError("cannot commit while writer completions are pending")
            if (
                self._next_submitted != self.span_tokens
                or self._appended_tokens != self.span_tokens
            ):
                raise RuntimeError("cannot commit before exact full-span coverage")
            final_batch_index = self._next_batch_index - 1
            self._owner._mark_timing(
                self.request_id,
                final_batch_index,
                "manifest_publish_begin",
            )
            try:
                receipt = self._transaction.commit_manifest()
            finally:
                self._owner._mark_timing(
                    self.request_id,
                    final_batch_index,
                    "manifest_publish_end",
                )
            self._receipt = receipt
            self._state = JournalState.COMMITTED
        self._owner._retire(self.request_id, self)
        return receipt

    def abort(self) -> None:
        with self._lock:
            if self._state is JournalState.COMMITTED:
                return
            if self._state is JournalState.ABORTED:
                return
            self._abort_locked(RuntimeError("snapshot journal aborted"))

    def _append_batch(self, batch: SnapshotBatch, view: SnapshotView) -> None:
        final_batch = batch.logical_end == self.span_tokens
        if final_batch:
            self._owner._mark_timing(
                self.request_id,
                batch.batch_index,
                "writer_start",
            )
        try:
            with self._lock:
                if self._state is JournalState.ABORTED:
                    return
            expected = batch.logical_start
            for chunk in self._owner.translator.iter_chunks(view):
                with self._lock:
                    if self._state is JournalState.ABORTED:
                        return
                    if (
                        chunk.logical_start != expected
                        or chunk.logical_end > batch.logical_end
                    ):
                        raise SnapshotTranslationError(
                            "translator emitted noncontiguous batch chunks"
                        )
                self._transaction.append_chunk(chunk)
                expected = chunk.logical_end
                with self._lock:
                    if self._state is JournalState.OPEN:
                        self._appended_tokens = expected
            if expected != batch.logical_end:
                raise SnapshotTranslationError(
                    "translator did not cover the complete snapshot batch"
                )
        except BaseException as error:
            with self._lock:
                if self._state is JournalState.OPEN:
                    self._abort_locked(error)
            raise
        finally:
            if final_batch:
                self._owner._mark_timing(
                    self.request_id,
                    batch.batch_index,
                    "writer_end",
                )

    def _validate_submission(
        self,
        batch: SnapshotBatch,
        view: SnapshotView,
    ) -> None:
        if (
            batch.request_id != self.request_id
            or batch.context_digest != self.context_digest
            or batch.batch_index != self._next_batch_index
            or batch.chunk_tokens != CHUNK_TOKENS
            or batch.logical_start != self._next_submitted
            or batch.logical_end <= batch.logical_start
            or batch.logical_end > self.span_tokens
            or view.ticket.logical_start != batch.logical_start
        ):
            raise SnapshotTranslationError(
                "snapshot batch disagrees with journal transaction"
            )
        logical_tokens = self._owner.translator.batch_logical_tokens(view)
        if batch.logical_start + logical_tokens != batch.logical_end:
            raise SnapshotTranslationError(
                "READY row count disagrees with snapshot batch range"
            )

    def _completion_done(self, completion: ManifestWriterCompletion) -> None:
        with self._lock:
            self._pending.discard(completion)
            if self._state is JournalState.ABORTED and not self._pending:
                self._finalize_abort_locked()

    def _abort_locked(self, error: BaseException) -> None:
        self._state = JournalState.ABORTED
        self._error = error
        # ManifestTransaction.abort() takes the same lock as append_chunk(),
        # which may include slow encoding and durable I/O. Runtime cancellation
        # must remain nonblocking, so descriptor cleanup waits for the last
        # writer completion whenever an append is in flight.
        if not self._pending:
            self._finalize_abort_locked()

    def _finalize_abort_locked(self) -> None:
        if self._abort_finalized:
            return
        assert self._state is JournalState.ABORTED
        assert not self._pending
        try:
            self._transaction.abort()
        finally:
            self._abort_finalized = True
            self._owner._retire(self.request_id, self)

    def _require_open(self) -> None:
        if self._state is not JournalState.OPEN:
            raise RuntimeError(f"snapshot journal is {self._state.value}")
