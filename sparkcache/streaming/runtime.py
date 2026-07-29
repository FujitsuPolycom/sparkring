"""Worker-side orchestration for opportunistic streaming snapshots.

This module is CPU-importable and disabled by default.  It owns no connector
hook and loads no native library.  Production code must explicitly construct
it with an already configured ring, bounded lease registry, planner, and
transactional writer.

The runtime deliberately separates two ownership intervals:

* a :class:`BlockLeaseRegistry` lease protects source KV blocks until the
  native gather reports GPU completion;
* a claimed native-ring view remains owned until the injected writer has
  finished consuming the staged bytes.

Thus cancellation can release model KV as soon as the gather completes
without waiting for hashing or NVMe.  The invisible journal is aborted, while
already claimed ring slots drain in later calls to :meth:`poll`.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from .block_lease import (
    BlockLeaseRegistry,
    CompletionFence,
    LeaseFenceError,
)
from .planner import (
    BlockTableRangeMapper,
    SnapshotBatch,
    StreamingSnapshotCoordinator,
)

__all__ = [
    "RuntimeOffer",
    "SnapshotJournalTransaction",
    "SnapshotJournalWriter",
    "SnapshotRing",
    "StreamingSnapshotFatalError",
    "StreamingSnapshotRuntime",
    "StreamingSnapshotRuntimeConfig",
    "StreamingSnapshotRuntimeError",
    "WriterCompletion",
]


class StreamingSnapshotRuntimeError(RuntimeError):
    """Base class for worker-side streaming-snapshot failures."""


class StreamingSnapshotFatalError(StreamingSnapshotRuntimeError):
    """Ownership status is unknown; the worker must fail closed."""


@runtime_checkable
class SnapshotRing(Protocol):
    """Subset of :class:`NativeSnapshotRing` used by this coordinator."""

    def submit(
        self,
        *,
        context_sequence: int,
        logical_start: int,
        physical_slots: Sequence[int],
        producer_stream: int,
    ) -> Any | None: ...

    def poll(self, ticket: Any) -> Any | None: ...

    def claim(self, ticket: Any) -> Any | None: ...

    def release(self, ticket: Any) -> None: ...

    def abandon(self, context_sequence: int) -> None: ...

    def shutdown(self) -> None: ...


@runtime_checkable
class WriterCompletion(Protocol):
    """Completion for one writer-owned claimed ring view.

    ``query()`` becomes true at the ownership-transfer boundary: the writer
    will never again read the borrowed ``SnapshotView``. At that point
    ``result()`` must return immediately (or immediately report the append
    failure), because the worker poll path must never wait for disk.

    The boundary may mean the chunk is already durable, or that its bytes now
    live in immutable writer-owned storage and the transaction guarantees
    durability before ``commit_manifest()``. It may never mean "a background
    thread still reads the ring view."
    """

    def query(self) -> bool:
        """Return true only when the writer no longer reads the view."""

    def synchronize(self) -> None:
        """Wait through the same ownership-transfer boundary as ``query``."""

    def result(self) -> None:
        """Nonblocking success/failure result, valid only after completion."""


@runtime_checkable
class SnapshotJournalTransaction(Protocol):
    """Invisible context journal owned by one runtime session.

    Runtime calls ``submit_ready`` in strictly increasing ``batch_index``
    order. It must either return a completion that owns the borrowed view or
    raise before retaining/using that view. An implementation that can raise
    after handing the view to another thread needs its own atomic submission
    wrapper, analogous to ``LeaseHandle.submit``.

    The publisher owns no ring ticket and must never call ring
    submit/poll/claim/release/abandon. Runtime retains that ownership and
    invalidates the borrowed view immediately after the completion boundary.
    """

    def submit_ready(
        self,
        batch: SnapshotBatch,
        view: Any,
    ) -> WriterCompletion: ...

    def commit_manifest(self) -> Any: ...

    def abort(self) -> None:
        """Permanently suppress visibility; it need not wait for I/O."""
        ...


@runtime_checkable
class SnapshotJournalWriter(Protocol):
    """Factory for context-scoped transactional journals."""

    def begin_context(
        self,
        *,
        request_id: str,
        context_digest: str,
        span_tokens: int,
    ) -> SnapshotJournalTransaction: ...


@dataclass(frozen=True, slots=True)
class StreamingSnapshotRuntimeConfig:
    """Static worker geometry and bounded synchronization policy."""

    enabled: bool = False
    block_size: int = 1
    dcp_degree: int = 1
    dcp_rank: int = 0
    gather_abort_timeout_seconds: float = 30.0
    poll_sleep_seconds: float = 0.0005

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.dcp_degree <= 0:
            raise ValueError("dcp_degree must be positive")
        if not 0 <= self.dcp_rank < self.dcp_degree:
            raise ValueError("dcp_rank must be in [0, dcp_degree)")
        if self.gather_abort_timeout_seconds <= 0:
            raise ValueError("gather_abort_timeout_seconds must be positive")
        if self.poll_sleep_seconds < 0:
            raise ValueError("poll_sleep_seconds must be non-negative")


@dataclass(frozen=True, slots=True)
class RuntimeOffer:
    """Result of one completed-prefill watermark."""

    submitted_batches: int = 0
    active: bool = False
    aborted: bool = False
    reason: str | None = None


@dataclass(slots=True)
class _RingFence:
    ring: SnapshotRing
    ticket: Any
    timeout_seconds: float
    sleep_seconds: float
    ready: bool = False
    ready_at_ns: int | None = None

    def query(self) -> bool:
        if self.ready:
            return True
        self.ready = self.ring.poll(self.ticket) is not None
        if self.ready:
            self.ready_at_ns = time.monotonic_ns()
        return self.ready

    def synchronize(self) -> None:
        deadline = time.monotonic() + self.timeout_seconds
        while not self.query():
            if time.monotonic() >= deadline:
                raise TimeoutError("snapshot gather did not complete before abort")
            time.sleep(self.sleep_seconds)


@dataclass(frozen=True, slots=True)
class _ImmediateFence:
    """Known-no-submission fence used for native WOULD_BLOCK/DROPPED."""

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None


@dataclass(slots=True)
class _Inflight:
    batch: SnapshotBatch
    ticket: Any
    fence: _RingFence
    writer_completion: WriterCompletion | None = None


@dataclass(slots=True)
class _Context:
    request_id: str
    context_digest: str
    span_tokens: int
    context_sequence: int
    transaction: SnapshotJournalTransaction
    inflight: dict[int, _Inflight] = field(default_factory=dict)
    aborted: bool = False
    abort_reason: str | None = None


class StreamingSnapshotRuntime:
    """Fail-open cache state machine for one vLLM worker rank.

    Ordinary capacity, mapping, native-ring backpressure, cancellation, and
    writer-result errors abort only cache publication.  Fence/native ownership
    errors poison the runtime and raise :class:`StreamingSnapshotFatalError`;
    continuing could allow vLLM to recycle a block still read by CUDA.
    """

    def __init__(
        self,
        config: StreamingSnapshotRuntimeConfig = StreamingSnapshotRuntimeConfig(),
        *,
        planner: StreamingSnapshotCoordinator | None = None,
        leases: BlockLeaseRegistry | None = None,
        ring: SnapshotRing | None = None,
        writer: SnapshotJournalWriter | None = None,
        timing_trace: Any | None = None,
    ) -> None:
        self.config = config
        self._planner = planner
        self._leases = leases
        self._ring = ring
        self._writer = writer
        self._timing_trace = timing_trace
        if config.enabled:
            missing = [
                name
                for name, dependency in (
                    ("planner", planner),
                    ("leases", leases),
                    ("ring", ring),
                    ("writer", writer),
                )
                if dependency is None
            ]
            if missing:
                raise ValueError(
                    "enabled streaming runtime is missing: " + ", ".join(missing)
                )
            assert planner is not None
            if planner.chunk_tokens != 256:
                raise ValueError(
                    "streaming runtime requires 256-token journal chunks"
                )
        self.counters: Counter[str] = Counter()
        self._contexts: dict[str, _Context] = {}
        self._next_context_sequence = 1
        self._committed_digests: set[str] = set()
        self._aborted_contexts: dict[str, str] = {}
        self._poisoned: BaseException | None = None
        self._closed = False
        self._lock = threading.RLock()

    @property
    def active_contexts(self) -> int:
        with self._lock:
            return len(self._contexts)

    @property
    def needs_progress(self) -> bool:
        """Whether non-blocking work or a terminal handoff remains pending."""

        if not self.config.enabled:
            return False
        with self._lock:
            return (
                any(context.inflight for context in self._contexts.values())
                or bool(self._committed_digests)
                or bool(self._aborted_contexts)
            )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def begin_context(
        self,
        *,
        request_id: str,
        context_digest: str,
        span_tokens: int,
    ) -> bool:
        """Admit an invisible journal without touching CUDA or disk payloads."""

        if not self.config.enabled:
            self.counters["disabled_begin"] += 1
            return False
        with self._lock:
            self._require_usable()
            existing = self._contexts.get(request_id)
            if existing is not None:
                if (
                    existing.context_digest != context_digest
                    or existing.span_tokens != span_tokens
                ):
                    raise ValueError(
                        "request_id already owns a different runtime context"
                    )
                return not existing.aborted
            assert self._planner is not None
            if not self._planner.begin(
                request_id,
                context_digest,
                span_tokens,
            ):
                self.counters["begin_suppressed"] += 1
                return False
            try:
                assert self._writer is not None
                transaction = self._writer.begin_context(
                    request_id=request_id,
                    context_digest=context_digest,
                    span_tokens=span_tokens,
                )
            except Exception:
                self._planner.abort(request_id)
                self.counters["writer_begin_failed"] += 1
                return False
            if not isinstance(transaction, SnapshotJournalTransaction):
                self._planner.abort(request_id)
                self.counters["writer_begin_failed"] += 1
                return False
            sequence = self._next_context_sequence
            self._next_context_sequence += 1
            self._contexts[request_id] = _Context(
                request_id=request_id,
                context_digest=context_digest,
                span_tokens=span_tokens,
                context_sequence=sequence,
                transaction=transaction,
            )
            self.counters["contexts_begun"] += 1
            return True

    def accept_completed_prefill(
        self,
        *,
        request_id: str,
        completed_tokens: int,
        block_table: Sequence[int],
        producer_stream: int,
    ) -> RuntimeOffer:
        """Map and submit every newly completed 256-token range.

        This method never waits for ring or lease capacity.  Any expected
        pressure aborts the optional cache transaction and returns immediately.
        """

        watermark_observed_at_ns = time.monotonic_ns()
        if not self.config.enabled:
            self.counters["disabled_watermark"] += 1
            return RuntimeOffer(reason="disabled")
        with self._lock:
            self._require_usable()
            context = self._contexts.get(request_id)
            if context is None:
                # A writer/manifest failure may have retired this optional
                # context during an earlier poll while connector metadata for
                # a later step was already in flight. Treat it as a cache miss.
                self.counters["unknown_context_offers"] += 1
                return RuntimeOffer(aborted=True, reason="unknown_context")
            if context.aborted:
                return RuntimeOffer(aborted=True, reason=context.abort_reason)
            if (
                not isinstance(producer_stream, int)
                or isinstance(producer_stream, bool)
                or not 0 <= producer_stream <= 0xFFFFFFFFFFFFFFFF
            ):
                raise ValueError("producer_stream must be a uint64")
            try:
                table = tuple(block_table)
                mapper = BlockTableRangeMapper(
                    block_ids=table,
                    logical_tokens_per_block=(
                        self.config.block_size * self.config.dcp_degree
                    ),
                )
            except (TypeError, ValueError):
                self._abort_context(context, "invalid_block_table")
                return RuntimeOffer(aborted=True, reason="invalid_block_table")

            assert self._planner is not None
            try:
                offer = self._planner.offer_completed(
                    request_id,
                    completed_tokens,
                    mapper,
                )
            except (TypeError, ValueError):
                # Scheduler metadata that regresses or changes shape is not an
                # ownership ambiguity: no new gather was submitted. Abandon
                # this optional cache publication instead of failing serving.
                self._abort_context(context, "invalid_completed_watermark")
                return RuntimeOffer(
                    aborted=True,
                    reason="invalid_completed_watermark",
                )
            if offer.aborted:
                self._abort_context(context, "planner_backpressure_or_mapping")
                return RuntimeOffer(
                    aborted=True,
                    reason="planner_backpressure_or_mapping",
                )

            submitted = 0
            for batch in offer.batches:
                if batch.logical_end == context.span_tokens:
                    self._register_final_timing(context, batch)
                    self._mark_timing(
                        context,
                        batch,
                        "final_watermark",
                        at_ns=watermark_observed_at_ns,
                    )
                if not self._submit_batch(
                    context,
                    batch,
                    table,
                    producer_stream=producer_stream,
                ):
                    return RuntimeOffer(
                        submitted_batches=submitted,
                        aborted=True,
                        reason=context.abort_reason,
                    )
                submitted += 1
            self.counters["batches_submitted"] += submitted
            return RuntimeOffer(
                submitted_batches=submitted,
                active=True,
            )

    def poll(self) -> int:
        """Advance gathers, writers, and manifest publication without waiting."""

        if not self.config.enabled:
            return 0
        with self._lock:
            self._require_usable()
            assert self._leases is not None
            try:
                self._leases.poll()
            except LeaseFenceError as error:
                self._poison(error)

            completed = 0
            for context in tuple(self._contexts.values()):
                if context.aborted:
                    completed += self._drain_aborted(context)
                else:
                    completed += self._advance_active(context)
            return completed

    def take_committed(self) -> set[str]:
        """Drain manifests that crossed their visibility point successfully.

        The connector uses this handoff to advertise the digest in worker
        stats. A digest is inserted only after both the writer's manifest
        commit and the planner commit succeed; partial or aborted journals are
        therefore never surfaced as restorable entries.
        """

        if not self.config.enabled:
            return set()
        with self._lock:
            self._require_usable()
            committed, self._committed_digests = self._committed_digests, set()
            return committed

    def take_aborted(self) -> dict[str, str]:
        """Drain cache-only terminal aborts discovered inside the runtime.

        Writer and manifest failures can surface asynchronously from ``poll``.
        The connector adapter must learn about them so a later scheduler offer
        for the same request becomes an ignored cache miss rather than an
        ``unknown context`` error that interrupts serving.
        """

        if not self.config.enabled:
            return {}
        with self._lock:
            self._require_usable()
            aborted, self._aborted_contexts = self._aborted_contexts, {}
            return aborted

    def cancel(self, request_id: str, *, reason: str = "cancelled") -> bool:
        """Abort publication for cancellation or scheduler preemption."""

        if not self.config.enabled:
            return False
        with self._lock:
            self._require_usable()
            context = self._contexts.get(request_id)
            if context is None:
                return False
            self._abort_context(context, reason)
            return True

    def preempt(self, request_id: str) -> bool:
        """Named cancellation edge for worker-side preemption metadata."""

        return self.cancel(request_id, reason="preempted")

    def shutdown(self) -> None:
        """Abort every journal, drain borrowed views, then close the ring."""

        if not self.config.enabled:
            self._closed = True
            return
        with self._lock:
            if self._closed:
                return
            self._require_not_poisoned()
            for context in tuple(self._contexts.values()):
                self._abort_context(context, "shutdown")
            for context in tuple(self._contexts.values()):
                for inflight in tuple(context.inflight.values()):
                    completion = inflight.writer_completion
                    if completion is None:
                        continue
                    try:
                        completion.synchronize()
                    except Exception as error:
                        self._poison(error)
                    self._finish_aborted_writer(context, inflight)
                self._retire_if_drained(context)
            assert self._leases is not None
            try:
                self._leases.abort_all()
            except LeaseFenceError as error:
                self._poison(error)
            assert self._ring is not None
            try:
                self._ring.shutdown()
            except Exception as error:
                self._poison(error)
            self._closed = True
            self.counters["shutdown"] += 1

    def _submit_batch(
        self,
        context: _Context,
        batch: SnapshotBatch,
        block_table: tuple[int, ...],
        *,
        producer_stream: int,
    ) -> bool:
        final_batch = batch.logical_end == context.span_tokens
        assert self._leases is not None
        handle = self._leases.try_reserve(
            context.request_id,
            batch.block_ids,
        )
        if handle is None:
            self.counters["lease_backpressure"] += 1
            self._abort_context(context, "lease_backpressure")
            return False
        try:
            physical_slots = self._physical_slots(batch, block_table)
        except (IndexError, TypeError, ValueError):
            handle.cancel_before_submit()
            self._abort_context(context, "physical_slot_mapping")
            return False

        box: dict[str, Any] = {}

        def submitter() -> CompletionFence:
            assert self._ring is not None
            if final_batch:
                self._mark_timing(context, batch, "ring_submit_begin")
            ticket = self._ring.submit(
                context_sequence=context.context_sequence,
                logical_start=batch.logical_start,
                physical_slots=physical_slots,
                producer_stream=producer_stream,
            )
            if final_batch:
                self._mark_timing(context, batch, "ring_submit_end")
            box["ticket"] = ticket
            if ticket is None:
                fence: CompletionFence = _ImmediateFence()
            else:
                fence = _RingFence(
                    self._ring,
                    ticket,
                    self.config.gather_abort_timeout_seconds,
                    self.config.poll_sleep_seconds,
                )
            box["fence"] = fence
            return fence

        try:
            handle.submit(submitter)
        except LeaseFenceError as error:
            self._poison(error)
        ticket = box["ticket"]
        if ticket is None:
            # Native WOULD_BLOCK/DROPPED is known not to have launched work.
            # The immediate fence lets abort release the source lease safely.
            self.counters["native_backpressure"] += 1
            self._abort_context(context, "native_backpressure")
            return False
        fence = box["fence"]
        assert isinstance(fence, _RingFence)
        context.inflight[batch.batch_index] = _Inflight(
            batch=batch,
            ticket=ticket,
            fence=fence,
        )
        return True

    def _advance_active(self, context: _Context) -> int:
        completed = 0
        for inflight in tuple(
            sorted(
                context.inflight.values(),
                key=lambda candidate: candidate.batch.batch_index,
            )
        ):
            final_batch = inflight.batch.logical_end == context.span_tokens
            if inflight.writer_completion is None:
                if not inflight.fence.ready:
                    # Journal chunks are contiguous. Never hand batch N+1 to
                    # the writer before batch N, even if polling happened to
                    # observe a later ring slot first.
                    break
                if final_batch:
                    self._mark_timing(
                        context,
                        inflight.batch,
                        "fence_ready_observed",
                        at_ns=inflight.fence.ready_at_ns,
                    )
                assert self._ring is not None
                try:
                    if final_batch:
                        self._mark_timing(
                            context,
                            inflight.batch,
                            "claim_begin",
                        )
                    view = self._ring.claim(inflight.ticket)
                    if final_batch:
                        self._mark_timing(
                            context,
                            inflight.batch,
                            "claim_end",
                        )
                except Exception as error:
                    self._poison(error)
                if view is None:
                    self._poison(
                        StreamingSnapshotRuntimeError(
                            "GPU-complete snapshot could not be claimed"
                        )
                    )
                try:
                    completion = context.transaction.submit_ready(
                        inflight.batch,
                        view,
                    )
                except Exception:
                    # SnapshotJournalTransaction guarantees that an exception
                    # means it did not retain the borrowed view.
                    try:
                        self._ring.release(inflight.ticket)
                    except Exception as release_error:
                        self._poison(release_error)
                    context.inflight.pop(inflight.batch.batch_index, None)
                    self.counters["writer_submit_failed"] += 1
                    self._abort_context(context, "writer_submit_failed")
                    return completed
                if not isinstance(completion, WriterCompletion):
                    self._poison(
                        TypeError(
                            "submit_ready must return a WriterCompletion"
                        )
                    )
                inflight.writer_completion = completion

            completion = inflight.writer_completion
            assert completion is not None
            try:
                writer_done = completion.query()
            except Exception as error:
                self._poison(error)
            if not writer_done:
                break
            if final_batch:
                self._mark_timing(
                    context,
                    inflight.batch,
                    "writer_completion_polled",
                )
            try:
                completion.result()
            except Exception:
                assert self._ring is not None
                try:
                    self._ring.release(inflight.ticket)
                except Exception as release_error:
                    self._poison(release_error)
                context.inflight.pop(inflight.batch.batch_index, None)
                self.counters["writer_result_failed"] += 1
                self._abort_context(context, "writer_result_failed")
                return completed

            assert self._ring is not None
            try:
                self._ring.release(inflight.ticket)
            except Exception as error:
                self._poison(error)
            context.inflight.pop(inflight.batch.batch_index, None)
            assert self._planner is not None
            ready_to_commit = self._planner.complete_batch(
                context.request_id,
                inflight.batch.batch_index,
            )
            completed += 1
            self.counters["batches_written"] += 1
            if ready_to_commit:
                self._commit_context(context, inflight.batch)
                break
        return completed

    def _commit_context(self, context: _Context, final_batch: SnapshotBatch) -> None:
        if context.inflight:
            self._poison(
                StreamingSnapshotRuntimeError(
                    "planner reached commit while native batches remain"
                )
            )
        try:
            context.transaction.commit_manifest()
        except Exception:
            self.counters["manifest_commit_failed"] += 1
            self._abort_context(context, "manifest_commit_failed")
            return
        assert self._planner is not None
        try:
            self._planner.commit(context.request_id)
        except Exception as error:
            # The manifest may now be visible, so internal state disagreement
            # is not a fail-open cache miss.
            self._poison(error)
        self._committed_digests.add(context.context_digest)
        self._contexts.pop(context.request_id, None)
        self.counters["contexts_committed"] += 1
        self._mark_timing(context, final_batch, "runtime_commit")

    def _register_final_timing(
        self,
        context: _Context,
        batch: SnapshotBatch,
    ) -> None:
        trace = self._timing_trace
        if trace is None:
            return
        try:
            trace.register_final(
                context.request_id,
                batch.batch_index,
                context.span_tokens,
            )
        except Exception:
            self.counters["timing_trace_failed"] += 1

    def _mark_timing(
        self,
        context: _Context,
        batch: SnapshotBatch,
        stage: str,
        *,
        at_ns: int | None = None,
    ) -> None:
        trace = self._timing_trace
        if trace is None:
            return
        try:
            trace.mark(
                context.request_id,
                batch.batch_index,
                stage,
                at_ns=at_ns,
            )
        except Exception:
            self.counters["timing_trace_failed"] += 1

    def _abort_context(self, context: _Context, reason: str) -> None:
        if context.aborted:
            return
        context.aborted = True
        context.abort_reason = reason
        self._aborted_contexts[context.request_id] = reason
        # Every submitted batch still owned by this context becomes
        # non-publishable at the terminal abort edge, including a claimed
        # view whose writer must drain before its ring slot can be released.
        self.counters["batches_abandoned"] += len(context.inflight)
        try:
            context.transaction.abort()
        except Exception:
            self.counters["journal_abort_failed"] += 1

        assert self._planner is not None
        self._planner.abort(context.request_id)
        assert self._leases is not None
        try:
            self._leases.abort_request(context.request_id)
        except LeaseFenceError as error:
            self._poison(error)
        assert self._ring is not None
        try:
            self._ring.abandon(context.context_sequence)
        except Exception as error:
            self._poison(error)

        # Unclaimed tickets were retired by ring.abandon().  Claimed tickets
        # remain borrowed until their writer completions finish.
        context.inflight = {
            index: inflight
            for index, inflight in context.inflight.items()
            if inflight.writer_completion is not None
        }
        self.counters[f"aborted_{reason}"] += 1
        self.counters["contexts_aborted"] += 1
        self._retire_if_drained(context)

    def _drain_aborted(self, context: _Context) -> int:
        completed = 0
        for inflight in tuple(context.inflight.values()):
            completion = inflight.writer_completion
            assert completion is not None
            try:
                done = completion.query()
            except Exception as error:
                self._poison(error)
            if not done:
                continue
            self._finish_aborted_writer(context, inflight)
            completed += 1
        self._retire_if_drained(context)
        return completed

    def _finish_aborted_writer(
        self,
        context: _Context,
        inflight: _Inflight,
    ) -> None:
        completion = inflight.writer_completion
        assert completion is not None
        try:
            completion.result()
        except Exception:
            self.counters["aborted_writer_result_failed"] += 1
        assert self._ring is not None
        try:
            self._ring.release(inflight.ticket)
        except Exception as error:
            self._poison(error)
        context.inflight.pop(inflight.batch.batch_index, None)
        self.counters["aborted_writers_drained"] += 1

    def _retire_if_drained(self, context: _Context) -> None:
        if context.aborted and not context.inflight:
            self._contexts.pop(context.request_id, None)

    def _physical_slots(
        self,
        batch: SnapshotBatch,
        block_table: tuple[int, ...],
    ) -> tuple[int, ...]:
        rank = self.config.dcp_rank
        degree = self.config.dcp_degree
        first = batch.logical_start + (
            (rank - batch.logical_start) % degree
        )
        slots: list[int] = []
        for position in range(first, batch.logical_end, degree):
            ordinal = position // degree
            table_index = ordinal // self.config.block_size
            block_id = block_table[table_index]
            if (
                not isinstance(block_id, int)
                or isinstance(block_id, bool)
                or block_id < 0
            ):
                raise ValueError("block table contains an invalid block ID")
            slots.append(
                block_id * self.config.block_size
                + ordinal % self.config.block_size
            )
        if not slots:
            raise ValueError("batch maps to no DCP-local physical slots")
        return tuple(slots)

    def _require_context(self, request_id: str) -> _Context:
        try:
            return self._contexts[request_id]
        except KeyError as error:
            raise ValueError("unknown streaming snapshot request") from error

    def _require_not_poisoned(self) -> None:
        if self._poisoned is not None:
            raise StreamingSnapshotFatalError(
                "streaming snapshot runtime is poisoned"
            ) from self._poisoned

    def _require_usable(self) -> None:
        self._require_not_poisoned()
        if self._closed:
            raise StreamingSnapshotRuntimeError(
                "streaming snapshot runtime is closed"
            )

    def _poison(self, error: BaseException) -> None:
        self._poisoned = error
        self.counters["fatal_errors"] += 1
        raise StreamingSnapshotFatalError(
            "streaming snapshot ownership status is unknown"
        ) from error
