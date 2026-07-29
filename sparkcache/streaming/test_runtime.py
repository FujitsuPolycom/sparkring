from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from sparkcache.streaming.block_lease import (
    BlockLeaseRegistry,
    LeaseCapacity,
)
from sparkcache.streaming.planner import StreamingSnapshotCoordinator
from sparkcache.streaming.runtime import (
    SnapshotJournalTransaction,
    SnapshotJournalWriter,
    StreamingSnapshotFatalError,
    StreamingSnapshotRuntime,
    StreamingSnapshotRuntimeConfig,
    WriterCompletion,
)


@dataclass(frozen=True)
class FakeTicket:
    value: int


@dataclass(frozen=True)
class FakeView:
    marker: int


class FakeRing:
    def __init__(self, *, capacity: int | None = None) -> None:
        self.capacity = capacity
        self.next_ticket = 1
        self.entries: dict[FakeTicket, dict[str, Any]] = {}
        self.submissions: list[dict[str, Any]] = []
        self.releases: list[FakeTicket] = []
        self.abandoned: list[int] = []
        self.drop_next = False
        self.raise_on_submit = False
        self.raise_on_poll = False
        self.closed = False

    def submit(self, **submission: Any) -> FakeTicket | None:
        self.submissions.append(dict(submission))
        if self.raise_on_submit:
            raise RuntimeError("unknown launch status")
        if self.capacity is not None and len(self.entries) >= self.capacity:
            return None
        if self.drop_next:
            self.drop_next = False
            return None
        ticket = FakeTicket(self.next_ticket)
        self.next_ticket += 1
        self.entries[ticket] = {
            "context_sequence": submission["context_sequence"],
            "ready": False,
            "phase": "submitted",
            "view": FakeView(ticket.value),
        }
        return ticket

    def make_all_ready(self) -> None:
        for entry in self.entries.values():
            entry["ready"] = True

    def make_ready(self, ticket: FakeTicket) -> None:
        self.entries[ticket]["ready"] = True

    def poll(self, ticket: FakeTicket) -> FakeView | None:
        if self.raise_on_poll:
            raise RuntimeError("completion query failed")
        entry = self.entries[ticket]
        if not entry["ready"]:
            return None
        entry["phase"] = "ready"
        return entry["view"]

    def claim(self, ticket: FakeTicket) -> FakeView | None:
        entry = self.entries[ticket]
        if not entry["ready"]:
            return None
        entry["phase"] = "claimed"
        return entry["view"]

    def release(self, ticket: FakeTicket) -> None:
        entry = self.entries[ticket]
        assert entry["phase"] in ("ready", "claimed", "abandoned_claimed")
        self.releases.append(ticket)
        del self.entries[ticket]

    def abandon(self, context_sequence: int) -> None:
        self.abandoned.append(context_sequence)
        for ticket, entry in tuple(self.entries.items()):
            if entry["context_sequence"] != context_sequence:
                continue
            if entry["phase"] == "claimed":
                entry["phase"] = "abandoned_claimed"
            else:
                del self.entries[ticket]

    def shutdown(self) -> None:
        assert not self.entries
        self.closed = True


class FakeWriterCompletion:
    def __init__(
        self,
        transaction: "FakeTransaction",
        batch_index: int,
        *,
        done: bool,
        error: BaseException | None,
    ) -> None:
        self.transaction = transaction
        self.batch_index = batch_index
        self.done = done
        self.error = error
        self.reported = False
        self.synchronize_calls = 0

    def query(self) -> bool:
        return self.done

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        self.done = True

    def result(self) -> None:
        if not self.done:
            raise RuntimeError("result requested before completion")
        if self.error is not None:
            raise self.error
        if not self.reported:
            self.transaction.durable_batches.add(self.batch_index)
            self.reported = True


class FakeTransaction:
    def __init__(
        self,
        request_id: str,
        span_tokens: int,
        writer: "FakeWriter",
    ) -> None:
        self.request_id = request_id
        self.span_tokens = span_tokens
        self.writer = writer
        self.submitted_batches: list[Any] = []
        self.completions: list[FakeWriterCompletion] = []
        self.durable_batches: set[int] = set()
        self.committed = False
        self.aborted = False

    def submit_ready(
        self,
        batch: Any,
        view: Any,
    ) -> FakeWriterCompletion:
        assert view is not None
        if self.writer.raise_on_submit:
            raise RuntimeError("writer rejected view before retaining it")
        self.submitted_batches.append(batch)
        completion = FakeWriterCompletion(
            self,
            batch.batch_index,
            done=self.writer.auto_complete,
            error=self.writer.completion_error,
        )
        self.completions.append(completion)
        return completion

    def commit_manifest(self) -> None:
        if self.writer.commit_error:
            raise RuntimeError("commit failed")
        assert not self.aborted
        assert {
            batch.batch_index for batch in self.submitted_batches
        } == self.durable_batches
        covered = sum(
            batch.logical_end - batch.logical_start
            for batch in self.submitted_batches
        )
        assert covered == self.span_tokens
        self.committed = True

    def abort(self) -> None:
        assert not self.committed
        self.aborted = True


class FakeWriter:
    def __init__(self, *, auto_complete: bool = True) -> None:
        self.auto_complete = auto_complete
        self.raise_on_begin = False
        self.raise_on_submit = False
        self.completion_error: BaseException | None = None
        self.commit_error = False
        self.transactions: dict[str, FakeTransaction] = {}

    def begin_context(
        self,
        *,
        request_id: str,
        context_digest: str,
        span_tokens: int,
    ) -> FakeTransaction:
        assert context_digest
        if self.raise_on_begin:
            raise RuntimeError("begin failed")
        transaction = FakeTransaction(request_id, span_tokens, self)
        self.transactions[request_id] = transaction
        return transaction


class FakeTimingTrace:
    def __init__(self) -> None:
        self.registrations: list[tuple[str, int, int]] = []
        self.stages: list[tuple[str, int, str, int | None]] = []

    def register_final(
        self,
        request_id: str,
        batch_index: int,
        span_tokens: int,
    ) -> None:
        self.registrations.append((request_id, batch_index, span_tokens))

    def mark(
        self,
        request_id: str,
        batch_index: int,
        stage: str,
        *,
        at_ns: int | None = None,
    ) -> None:
        self.stages.append((request_id, batch_index, stage, at_ns))


def make_runtime(
    *,
    writer: FakeWriter | None = None,
    max_inflight: int = 2,
    max_leases: int = 2,
    ring_capacity: int | None = None,
    timing_trace: FakeTimingTrace | None = None,
) -> tuple[
    StreamingSnapshotRuntime,
    FakeRing,
    FakeWriter,
    BlockLeaseRegistry,
]:
    ring = FakeRing(capacity=ring_capacity)
    writer = writer or FakeWriter()
    leases = BlockLeaseRegistry(
        LeaseCapacity(
            max_active_leases=max_leases,
            max_leased_blocks=128,
        )
    )
    runtime = StreamingSnapshotRuntime(
        StreamingSnapshotRuntimeConfig(
            enabled=True,
            block_size=4,
            dcp_degree=2,
            dcp_rank=0,
            gather_abort_timeout_seconds=0.1,
            poll_sleep_seconds=0,
        ),
        planner=StreamingSnapshotCoordinator(
            chunk_tokens=256,
            chunks_per_batch=1,
            max_inflight_batches=max_inflight,
        ),
        leases=leases,
        ring=ring,
        writer=writer,
        timing_trace=timing_trace,
    )
    return runtime, ring, writer, leases


def begin(
    runtime: StreamingSnapshotRuntime,
    *,
    request_id: str = "req",
    span_tokens: int = 512,
    context_digest: str = "a" * 64,
) -> None:
    assert runtime.begin_context(
        request_id=request_id,
        context_digest=context_digest,
        span_tokens=span_tokens,
    )


def block_table(count: int = 64) -> tuple[int, ...]:
    return tuple(range(100, 100 + count))


def test_default_runtime_is_disabled_and_touches_no_dependencies() -> None:
    runtime = StreamingSnapshotRuntime()

    assert not runtime.begin_context(
        request_id="req",
        context_digest="a" * 64,
        span_tokens=256,
    )
    result = runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=256,
        block_table=(1,),
        producer_stream=0,
    )

    assert not result.active
    assert not result.aborted
    assert result.reason == "disabled"


def test_writer_fakes_implement_the_stable_publisher_protocol() -> None:
    writer = FakeWriter()
    transaction = writer.begin_context(
        request_id="req",
        context_digest="a" * 64,
        span_tokens=256,
    )
    completion = FakeWriterCompletion(
        transaction,
        0,
        done=True,
        error=None,
    )

    assert isinstance(writer, SnapshotJournalWriter)
    assert isinstance(transaction, SnapshotJournalTransaction)
    assert isinstance(completion, WriterCompletion)


def test_two_watermarks_write_all_chunks_before_manifest_commit() -> None:
    writer = FakeWriter(auto_complete=False)
    runtime, ring, writer, leases = make_runtime(writer=writer)
    begin(runtime)

    first = runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=256,
        block_table=block_table(),
        producer_stream=77,
    )
    assert first.submitted_batches == 1
    assert leases.active_leases == 1
    assert ring.submissions[0]["physical_slots"][:5] == (
        400,
        401,
        402,
        403,
        404,
    )

    ring.make_all_ready()
    runtime.poll()
    transaction = writer.transactions["req"]
    assert leases.active_leases == 0
    assert len(ring.entries) == 1
    assert not transaction.committed

    transaction.completions[0].done = True
    assert runtime.poll() == 1
    assert not transaction.committed
    assert runtime.active_contexts == 1

    second = runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=512,
        block_table=block_table(),
        producer_stream=77,
    )
    assert second.submitted_batches == 1
    ring.make_all_ready()
    runtime.poll()
    transaction.completions[1].done = True

    assert runtime.poll() == 1
    assert transaction.committed
    assert runtime.active_contexts == 0
    assert leases.active_leases == 0
    assert not ring.entries
    assert runtime.take_committed() == {"a" * 64}
    assert runtime.take_committed() == set()


def test_final_batch_timing_marks_native_and_poll_boundaries() -> None:
    timing = FakeTimingTrace()
    runtime, ring, _writer, _leases = make_runtime(timing_trace=timing)
    begin(runtime, span_tokens=256)

    result = runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=256,
        block_table=block_table(),
        producer_stream=77,
    )
    assert result.submitted_batches == 1
    ring.make_all_ready()
    assert runtime.poll() == 1

    assert timing.registrations == [("req", 0, 256)]
    assert [stage for _request, _batch, stage, _at_ns in timing.stages] == [
        "final_watermark",
        "ring_submit_begin",
        "ring_submit_end",
        "fence_ready_observed",
        "claim_begin",
        "claim_end",
        "writer_completion_polled",
        "runtime_commit",
    ]
    assert next(
        at_ns
        for _request, _batch, stage, at_ns in timing.stages
        if stage == "fence_ready_observed"
    ) is not None


def test_later_ready_slot_never_reaches_writer_before_earlier_batch() -> None:
    runtime, ring, writer, _leases = make_runtime()
    begin(runtime)
    result = runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=512,
        block_table=block_table(),
        producer_stream=77,
    )
    assert result.submitted_batches == 2
    first_ticket, second_ticket = tuple(ring.entries)
    ring.make_ready(second_ticket)

    runtime.poll()

    transaction = writer.transactions["req"]
    assert not transaction.submitted_batches
    assert ring.entries[second_ticket]["phase"] == "ready"

    ring.make_ready(first_ticket)
    assert runtime.poll() == 2
    assert [
        batch.batch_index for batch in transaction.submitted_batches
    ] == [0, 1]
    assert transaction.committed


def test_mapping_failure_aborts_without_native_submission() -> None:
    runtime, ring, writer, leases = make_runtime()
    begin(runtime)

    result = runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=256,
        block_table=(1,),
        producer_stream=7,
    )

    assert result.aborted
    assert result.reason == "planner_backpressure_or_mapping"
    assert writer.transactions["req"].aborted
    assert runtime.active_contexts == 0
    assert leases.active_leases == 0
    assert not ring.submissions


def test_native_backpressure_aborts_without_waiting_or_publication() -> None:
    runtime, ring, writer, leases = make_runtime()
    begin(runtime)
    ring.drop_next = True

    result = runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=256,
        block_table=block_table(),
        producer_stream=7,
    )

    assert result.aborted
    assert result.reason == "native_backpressure"
    assert writer.transactions["req"].aborted
    assert leases.active_leases == 0
    assert runtime.active_contexts == 0


def test_full_depth_two_ring_aborts_only_new_cache_without_blocking_owners() -> None:
    writer = FakeWriter(auto_complete=False)
    runtime, ring, writer, leases = make_runtime(
        writer=writer,
        max_inflight=3,
        max_leases=3,
        ring_capacity=2,
    )
    for index in range(3):
        begin(
            runtime,
            request_id=f"req-{index}",
            span_tokens=256,
            context_digest=str(index) * 64,
        )

    for index in range(2):
        offered = runtime.accept_completed_prefill(
            request_id=f"req-{index}",
            completed_tokens=256,
            block_table=tuple(range(100 + index * 64, 164 + index * 64)),
            producer_stream=7,
        )
        assert offered.submitted_batches == 1

    rejected = runtime.accept_completed_prefill(
        request_id="req-2",
        completed_tokens=256,
        block_table=tuple(range(300, 364)),
        producer_stream=7,
    )

    assert rejected.aborted
    assert rejected.reason == "native_backpressure"
    assert writer.transactions["req-2"].aborted
    assert not writer.transactions["req-2"].committed
    assert runtime.take_committed() == set()
    assert leases.active_leases == 2
    assert len(ring.entries) == 2
    assert runtime.active_contexts == 2

    ring.make_all_ready()
    runtime.poll()
    for request_id in ("req-0", "req-1"):
        writer.transactions[request_id].completions[0].done = True
    assert runtime.poll() == 2

    assert runtime.take_committed() == {"0" * 64, "1" * 64}
    assert leases.active_leases == 0
    assert not ring.entries


def test_preemption_drains_gather_before_releasing_source_lease() -> None:
    runtime, ring, writer, leases = make_runtime()
    begin(runtime, span_tokens=256)
    runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=256,
        block_table=block_table(64),
        producer_stream=7,
    )
    assert leases.active_leases == 1
    ring.make_all_ready()

    assert runtime.preempt("req")

    assert leases.active_leases == 0
    assert writer.transactions["req"].aborted
    assert runtime.active_contexts == 0
    assert not ring.entries
    assert ring.abandoned


def test_preemption_does_not_wait_for_writer_after_gpu_completion() -> None:
    writer = FakeWriter(auto_complete=False)
    runtime, ring, writer, leases = make_runtime(writer=writer)
    begin(runtime, span_tokens=256)
    runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=256,
        block_table=block_table(64),
        producer_stream=7,
    )
    ring.make_all_ready()
    runtime.poll()
    transaction = writer.transactions["req"]

    assert leases.active_leases == 0
    assert runtime.preempt("req")
    assert transaction.aborted
    assert runtime.active_contexts == 1
    assert len(ring.entries) == 1
    assert not ring.releases

    transaction.completions[0].done = True
    assert runtime.poll() == 1
    assert runtime.active_contexts == 0
    assert not ring.entries
    assert ring.releases
    assert not transaction.committed
    assert runtime.take_committed() == set()


def test_cancellation_of_slow_writer_is_immediate_and_never_publishes() -> None:
    writer = FakeWriter(auto_complete=False)
    runtime, ring, writer, leases = make_runtime(writer=writer)
    begin(runtime, span_tokens=256)
    runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=256,
        block_table=block_table(64),
        producer_stream=7,
    )
    ring.make_all_ready()
    runtime.poll()
    transaction = writer.transactions["req"]
    completion = transaction.completions[0]

    assert runtime.cancel("req", reason="client_disconnected")

    assert completion.synchronize_calls == 0
    assert transaction.aborted
    assert not transaction.committed
    assert runtime.take_committed() == set()
    assert runtime.take_aborted() == {"req": "client_disconnected"}
    assert leases.active_leases == 0
    assert runtime.active_contexts == 1

    completion.done = True
    assert runtime.poll() == 1
    assert runtime.active_contexts == 0
    assert not ring.entries
    assert runtime.take_committed() == set()


def test_slow_writer_never_blocks_poll_or_publishes_early() -> None:
    writer = FakeWriter(auto_complete=False)
    runtime, ring, writer, leases = make_runtime(
        writer=writer,
        ring_capacity=2,
    )
    begin(runtime, span_tokens=256)
    runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=256,
        block_table=block_table(32),
        producer_stream=7,
    )
    ring.make_all_ready()

    assert runtime.poll() == 0

    completion = writer.transactions["req"].completions[0]
    assert completion.synchronize_calls == 0
    assert leases.active_leases == 0
    assert runtime.active_contexts == 1
    assert len(ring.entries) == 1
    assert runtime.take_committed() == set()

    assert runtime.poll() == 0
    assert completion.synchronize_calls == 0
    assert runtime.take_committed() == set()

    completion.done = True
    assert runtime.poll() == 1
    assert writer.transactions["req"].committed
    assert runtime.take_committed() == {"a" * 64}
    assert not ring.entries


def test_runtime_needs_progress_through_terminal_handoff() -> None:
    runtime, ring, writer, _leases = make_runtime()
    assert runtime.needs_progress is False

    begin(runtime, span_tokens=256)
    assert runtime.needs_progress is False

    runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=256,
        block_table=block_table(32),
        producer_stream=7,
    )
    assert runtime.needs_progress is True

    ring.make_all_ready()
    assert runtime.poll() == 1
    assert writer.transactions["req"].committed
    assert runtime.needs_progress is True

    assert runtime.take_committed() == {"a" * 64}
    assert runtime.needs_progress is False


def test_writer_result_error_aborts_and_releases_claimed_slot() -> None:
    writer = FakeWriter()
    writer.completion_error = OSError("disk write failed")
    runtime, ring, writer, leases = make_runtime(writer=writer)
    begin(runtime, span_tokens=256)
    runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=256,
        block_table=block_table(32),
        producer_stream=7,
    )
    ring.make_all_ready()

    runtime.poll()

    assert writer.transactions["req"].aborted
    assert not writer.transactions["req"].committed
    assert runtime.active_contexts == 0
    assert leases.active_leases == 0
    assert not ring.entries
    assert ring.releases
    assert runtime.counters["writer_result_failed"] == 1
    assert runtime.take_aborted() == {"req": "writer_result_failed"}
    assert runtime.take_aborted() == {}

    later = runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=256,
        block_table=block_table(32),
        producer_stream=7,
    )
    assert later.aborted
    assert later.reason == "unknown_context"


def test_writer_submit_exception_is_fail_open_and_leaves_no_ownership() -> None:
    writer = FakeWriter()
    writer.raise_on_submit = True
    runtime, ring, writer, leases = make_runtime(writer=writer)
    begin(runtime, span_tokens=256)
    runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=256,
        block_table=block_table(32),
        producer_stream=7,
    )
    ring.make_all_ready()

    assert runtime.poll() == 0

    transaction = writer.transactions["req"]
    assert transaction.aborted
    assert not transaction.committed
    assert runtime.take_committed() == set()
    assert runtime.take_aborted() == {"req": "writer_submit_failed"}
    assert leases.active_leases == 0
    assert runtime.active_contexts == 0
    assert not ring.entries
    assert len(ring.releases) == 1


def test_manifest_commit_error_keeps_partial_journal_invisible() -> None:
    writer = FakeWriter()
    writer.commit_error = True
    runtime, ring, writer, _leases = make_runtime(writer=writer)
    begin(runtime, span_tokens=256)
    runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=256,
        block_table=block_table(32),
        producer_stream=7,
    )
    ring.make_all_ready()

    runtime.poll()

    transaction = writer.transactions["req"]
    assert transaction.aborted
    assert not transaction.committed
    assert runtime.active_contexts == 0
    assert runtime.take_committed() == set()


def test_each_batch_uses_its_current_producer_stream() -> None:
    runtime, ring, writer, leases = make_runtime()
    begin(runtime, span_tokens=512)
    first = runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=256,
        block_table=block_table(64),
        producer_stream=7,
    )
    assert first.submitted_batches == 1
    ring.make_all_ready()
    assert runtime.poll() == 1

    second = runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=512,
        block_table=block_table(64),
        producer_stream=8,
    )
    assert second.submitted_batches == 1
    ring.make_all_ready()
    assert runtime.poll() == 1

    assert [item["producer_stream"] for item in ring.submissions] == [7, 8]
    assert writer.transactions["req"].committed
    assert leases.active_leases == 0
    assert runtime.active_contexts == 0


def test_regressed_completed_watermark_aborts_cache_without_failing_serving() -> None:
    runtime, ring, writer, leases = make_runtime()
    begin(runtime, span_tokens=512)
    first = runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=256,
        block_table=block_table(64),
        producer_stream=7,
    )
    assert first.submitted_batches == 1
    ring.make_all_ready()
    assert runtime.poll() == 1

    regressed = runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=128,
        block_table=block_table(64),
        producer_stream=7,
    )

    assert regressed.aborted
    assert regressed.reason == "invalid_completed_watermark"
    assert writer.transactions["req"].aborted
    assert runtime.active_contexts == 0
    assert leases.active_leases == 0
    assert runtime.take_committed() == set()


def test_unknown_native_submission_poisoning_is_fatal_and_retains_lease() -> None:
    runtime, ring, _writer, leases = make_runtime()
    begin(runtime, span_tokens=256)
    ring.raise_on_submit = True

    with pytest.raises(StreamingSnapshotFatalError):
        runtime.accept_completed_prefill(
            request_id="req",
            completed_tokens=256,
            block_table=block_table(32),
            producer_stream=7,
        )

    assert leases.active_leases == 1
    with pytest.raises(StreamingSnapshotFatalError):
        runtime.poll()


def test_fence_query_failure_is_fatal_and_retains_source_lease() -> None:
    runtime, ring, writer, leases = make_runtime()
    begin(runtime, span_tokens=256)
    runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=256,
        block_table=block_table(32),
        producer_stream=7,
    )
    ring.raise_on_poll = True

    with pytest.raises(StreamingSnapshotFatalError):
        runtime.poll()

    assert leases.active_leases == 1
    assert len(ring.entries) == 1
    assert not writer.transactions["req"].committed
    with pytest.raises(StreamingSnapshotFatalError):
        runtime.take_committed()


def test_fence_synchronize_failure_during_preemption_retains_source_lease() -> None:
    runtime, ring, writer, leases = make_runtime()
    begin(runtime, span_tokens=256)
    runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=256,
        block_table=block_table(32),
        producer_stream=7,
    )
    ring.raise_on_poll = True

    with pytest.raises(StreamingSnapshotFatalError):
        runtime.preempt("req")

    transaction = writer.transactions["req"]
    assert transaction.aborted
    assert not transaction.committed
    assert leases.active_leases == 1
    assert len(ring.entries) == 1
    with pytest.raises(StreamingSnapshotFatalError):
        runtime.poll()


def test_shutdown_aborts_and_synchronizes_pending_writer() -> None:
    writer = FakeWriter(auto_complete=False)
    runtime, ring, writer, leases = make_runtime(writer=writer)
    begin(runtime, span_tokens=256)
    runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=256,
        block_table=block_table(32),
        producer_stream=7,
    )
    ring.make_all_ready()
    runtime.poll()

    runtime.shutdown()

    assert writer.transactions["req"].aborted
    assert leases.active_leases == 0
    assert runtime.active_contexts == 0
    assert ring.closed


def test_shutdown_drains_gpu_complete_unclaimed_ticket_without_publication() -> None:
    runtime, ring, writer, leases = make_runtime()
    begin(runtime, span_tokens=256)
    runtime.accept_completed_prefill(
        request_id="req",
        completed_tokens=256,
        block_table=block_table(32),
        producer_stream=7,
    )
    ring.make_all_ready()

    runtime.shutdown()

    transaction = writer.transactions["req"]
    assert transaction.aborted
    assert not transaction.committed
    assert leases.active_leases == 0
    assert runtime.active_contexts == 0
    assert not ring.entries
    assert ring.closed
