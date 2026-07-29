"""GPU-free state machine for opportunistic SparkCache streaming snapshots.

The production connector will feed this coordinator with monotonically
completed chunked-prefill spans. It emits bounded macro-batches whose source
blocks must be leased only until a native gather completion event fires.

The coordinator never asks inference to wait. If the bounded snapshot ring
cannot accept every newly completed token, the cache transaction is aborted
and serving continues without publishing a manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol


class LogicalBlockMapper(Protocol):
    """Resolve one completed logical token range to exact physical blocks."""

    def blocks_for_range(
        self,
        logical_start: int,
        logical_end: int,
    ) -> tuple[int, ...]: ...


@dataclass(frozen=True, slots=True)
class BlockTableRangeMapper:
    """Uniform logical-range adapter for a vLLM physical block table.

    For DCP, ``logical_tokens_per_block`` is normally
    ``block_size * dcp_degree`` because one rank owns one out of every
    ``dcp_degree`` logical tokens.
    """

    block_ids: tuple[int, ...]
    logical_tokens_per_block: int

    def __post_init__(self) -> None:
        if self.logical_tokens_per_block <= 0:
            raise ValueError("logical_tokens_per_block must be positive")
        if not self.block_ids:
            raise ValueError("block_ids must not be empty")
        if any(
            not isinstance(block_id, int) or block_id < 0
            for block_id in self.block_ids
        ):
            raise ValueError("block_ids must contain non-negative integers")

    def blocks_for_range(
        self,
        logical_start: int,
        logical_end: int,
    ) -> tuple[int, ...]:
        if logical_start < 0 or logical_end <= logical_start:
            raise ValueError("logical range must be positive and ordered")
        first = logical_start // self.logical_tokens_per_block
        last = (
            logical_end + self.logical_tokens_per_block - 1
        ) // self.logical_tokens_per_block
        if last > len(self.block_ids):
            raise ValueError("logical range exceeds the physical block table")
        blocks = self.block_ids[first:last]
        if not blocks:
            raise ValueError("logical range maps to no physical blocks")
        return blocks


class SnapshotState(Enum):
    ACTIVE = auto()
    READY_TO_COMMIT = auto()
    ABORTED = auto()
    COMMITTED = auto()


@dataclass(frozen=True)
class SnapshotBatch:
    request_id: str
    context_digest: str
    batch_index: int
    logical_start: int
    logical_end: int
    chunk_tokens: int
    block_ids: tuple[int, ...]

    @property
    def chunk_count(self) -> int:
        return (self.logical_end - self.logical_start) // self.chunk_tokens

    @property
    def chunk_indexes(self) -> range:
        return range(
            self.logical_start // self.chunk_tokens,
            self.logical_end // self.chunk_tokens,
        )


@dataclass(frozen=True)
class SnapshotOffer:
    batches: tuple[SnapshotBatch, ...] = ()
    aborted: bool = False
    release_batches: tuple[SnapshotBatch, ...] = ()


@dataclass
class _Session:
    request_id: str
    context_digest: str
    span_tokens: int
    state: SnapshotState = SnapshotState.ACTIVE
    next_token: int = 0
    next_batch_index: int = 0
    pending: dict[int, SnapshotBatch] = field(default_factory=dict)


class StreamingSnapshotCoordinator:
    """Bounded, duplicate-safe planner for write-behind prefill snapshots."""

    def __init__(
        self,
        *,
        chunk_tokens: int = 256,
        chunks_per_batch: int = 16,
        max_inflight_batches: int = 3,
        max_sessions: int = 8,
    ) -> None:
        for name, value in (
            ("chunk_tokens", chunk_tokens),
            ("chunks_per_batch", chunks_per_batch),
            ("max_inflight_batches", max_inflight_batches),
            ("max_sessions", max_sessions),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.chunk_tokens = chunk_tokens
        self.chunks_per_batch = chunks_per_batch
        self.max_inflight_batches = max_inflight_batches
        self.max_sessions = max_sessions
        self._sessions: dict[str, _Session] = {}
        self._digest_owner: dict[str, str] = {}
        self.counters = {
            "admitted": 0,
            "duplicate_suppressed": 0,
            "capacity_suppressed": 0,
            "aborted_backpressure": 0,
            "aborted_mapping": 0,
            "aborted_explicit": 0,
            "batches_emitted": 0,
            "batches_completed": 0,
            "committed": 0,
        }

    @property
    def inflight_batches(self) -> int:
        return sum(len(session.pending) for session in self._sessions.values())

    def begin(
        self,
        request_id: str,
        context_digest: str,
        span_tokens: int,
    ) -> bool:
        if not request_id or not context_digest:
            raise ValueError("request_id and context_digest are required")
        if span_tokens <= 0 or span_tokens % self.chunk_tokens:
            raise ValueError("span_tokens must be a positive chunk multiple")
        existing = self._sessions.get(request_id)
        if existing is not None:
            if (
                existing.context_digest != context_digest
                or existing.span_tokens != span_tokens
            ):
                raise ValueError("request_id already owns a different snapshot")
            return existing.state in (
                SnapshotState.ACTIVE,
                SnapshotState.READY_TO_COMMIT,
            )
        if context_digest in self._digest_owner:
            self.counters["duplicate_suppressed"] += 1
            return False
        if len(self._sessions) >= self.max_sessions:
            self.counters["capacity_suppressed"] += 1
            return False
        self._sessions[request_id] = _Session(
            request_id=request_id,
            context_digest=context_digest,
            span_tokens=span_tokens,
        )
        self._digest_owner[context_digest] = request_id
        self.counters["admitted"] += 1
        return True

    def offer_completed(
        self,
        request_id: str,
        completed_tokens: int,
        block_mapper: LogicalBlockMapper,
    ) -> SnapshotOffer:
        session = self._require_session(request_id)
        if session.state is not SnapshotState.ACTIVE:
            return SnapshotOffer()
        if completed_tokens < session.next_token:
            raise ValueError("completed_tokens must be monotonic")
        complete_end = min(completed_tokens, session.span_tokens)
        complete_end -= complete_end % self.chunk_tokens
        if complete_end <= session.next_token:
            return SnapshotOffer()

        batch_tokens = self.chunk_tokens * self.chunks_per_batch
        required = (
            complete_end - session.next_token + batch_tokens - 1
        ) // batch_tokens
        if self.inflight_batches + required > self.max_inflight_batches:
            release_batches = tuple(session.pending.values())
            self.counters["aborted_backpressure"] += 1
            self._abort(session)
            return SnapshotOffer(
                aborted=True,
                release_batches=release_batches,
            )

        mapped: list[tuple[int, int, tuple[int, ...]]] = []
        cursor = session.next_token
        try:
            while cursor < complete_end:
                logical_end = min(cursor + batch_tokens, complete_end)
                blocks = tuple(
                    block_mapper.blocks_for_range(cursor, logical_end)
                )
                if (
                    not blocks
                    or len(set(blocks)) != len(blocks)
                    or any(
                        not isinstance(block_id, int) or block_id < 0
                        for block_id in blocks
                    )
                ):
                    raise ValueError("invalid physical block mapping")
                mapped.append((cursor, logical_end, blocks))
                cursor = logical_end
        except Exception:  # Mapping failure abandons caching, never serving.
            release_batches = tuple(session.pending.values())
            self.counters["aborted_mapping"] += 1
            self._abort(session)
            return SnapshotOffer(
                aborted=True,
                release_batches=release_batches,
            )

        emitted: list[SnapshotBatch] = []
        for logical_start, logical_end, blocks in mapped:
            batch = SnapshotBatch(
                request_id=request_id,
                context_digest=session.context_digest,
                batch_index=session.next_batch_index,
                logical_start=logical_start,
                logical_end=logical_end,
                chunk_tokens=self.chunk_tokens,
                block_ids=blocks,
            )
            session.pending[batch.batch_index] = batch
            session.next_batch_index += 1
            emitted.append(batch)
        session.next_token = complete_end
        self.counters["batches_emitted"] += len(emitted)
        return SnapshotOffer(batches=tuple(emitted))

    def complete_batch(self, request_id: str, batch_index: int) -> bool:
        session = self._require_session(request_id)
        if batch_index not in session.pending:
            raise ValueError("unknown or already completed snapshot batch")
        del session.pending[batch_index]
        self.counters["batches_completed"] += 1
        if session.next_token == session.span_tokens and not session.pending:
            session.state = SnapshotState.READY_TO_COMMIT
            return True
        return False

    def abort(self, request_id: str) -> tuple[SnapshotBatch, ...]:
        session = self._sessions.get(request_id)
        if session is None:
            return ()
        pending = tuple(session.pending.values())
        self.counters["aborted_explicit"] += 1
        self._abort(session)
        return pending

    def commit(self, request_id: str) -> None:
        session = self._require_session(request_id)
        if session.state is not SnapshotState.READY_TO_COMMIT:
            raise ValueError("snapshot is not ready to commit")
        session.state = SnapshotState.COMMITTED
        self.counters["committed"] += 1
        self._retire(session)

    def state(self, request_id: str) -> SnapshotState | None:
        session = self._sessions.get(request_id)
        return None if session is None else session.state

    def _require_session(self, request_id: str) -> _Session:
        try:
            return self._sessions[request_id]
        except KeyError as error:
            raise ValueError("unknown snapshot request") from error

    def _abort(self, session: _Session) -> None:
        session.state = SnapshotState.ABORTED
        session.pending.clear()
        self._retire(session)

    def _retire(self, session: _Session) -> None:
        self._sessions.pop(session.request_id, None)
        if self._digest_owner.get(session.context_digest) == session.request_id:
            self._digest_owner.pop(session.context_digest, None)
