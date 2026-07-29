"""Bounded sender/receiver state machines for SparkCache buddy replication."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import time
from typing import Callable

from .protocol import (
    Frame,
    FrameType,
    ProtocolError,
    decode_commit_record,
    sha256,
)


@dataclass(frozen=True)
class CommittedReplica:
    transaction_id: str
    generation: int
    context_digest: str
    identity_digest: str
    manifest_digest: str
    chunk_digests: tuple[str, ...]


@dataclass(frozen=True)
class ReplicaChunk:
    transaction_id: str
    generation: int
    context_digest: str
    identity_digest: str
    chunk_index: int
    chunk_digest: str
    payload: bytes


@dataclass
class _ReceiveTransaction:
    transaction_id: str
    generation: int
    context_digest: str
    identity_digest: str
    last_activity: float
    chunks: dict[int, tuple[str, bytes | None]] = field(default_factory=dict)
    sequences: dict[int, str] = field(default_factory=dict)

    @property
    def staged_bytes(self) -> int:
        return sum(
            len(payload)
            for _, payload in self.chunks.values()
            if payload is not None
        )

    @property
    def staged_chunks(self) -> int:
        return sum(payload is not None for _, payload in self.chunks.values())


CommitCallback = Callable[[CommittedReplica, tuple[bytes, ...], bytes], None]
ChunkCallback = Callable[[ReplicaChunk], None]
StreamCommitCallback = Callable[[CommittedReplica, bytes], None]


class BuddyReceiver:
    """Fail-closed remote receiver with bounded volatile staging.

    A successful COMMIT callback is the remote publication boundary. Protocol
    failures discard only the replica transaction; they cannot affect the
    sender's local SparkCache entry or inference request.
    """

    def __init__(
        self,
        *,
        max_active_transactions: int = 4,
        max_staged_bytes: int = 64 << 20,
        max_staged_chunks: int = 64,
        max_context_chunks: int = 4096,
        recent_commits: int = 256,
        recent_generations: int = 512,
        transaction_ttl_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        on_commit: CommitCallback | None = None,
        on_chunk: ChunkCallback | None = None,
        on_stream_commit: StreamCommitCallback | None = None,
    ) -> None:
        for name, value in (
            ("max_active_transactions", max_active_transactions),
            ("max_staged_bytes", max_staged_bytes),
            ("max_staged_chunks", max_staged_chunks),
            ("max_context_chunks", max_context_chunks),
            ("recent_commits", recent_commits),
            ("recent_generations", recent_generations),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.max_active_transactions = max_active_transactions
        self.max_staged_bytes = max_staged_bytes
        self.max_staged_chunks = max_staged_chunks
        self.max_context_chunks = max_context_chunks
        self.recent_commits = recent_commits
        self.recent_generations = recent_generations
        if transaction_ttl_seconds <= 0:
            raise ValueError("transaction_ttl_seconds must be positive")
        self.transaction_ttl_seconds = transaction_ttl_seconds
        self._clock = clock
        self.on_commit = on_commit
        self.on_chunk = on_chunk
        self.on_stream_commit = on_stream_commit
        self._active: dict[tuple[str, int], _ReceiveTransaction] = {}
        self._latest_generation: OrderedDict[str, int] = OrderedDict()
        self._committed: OrderedDict[tuple[str, int], CommittedReplica] = OrderedDict()
        self._response_sequence = 0

    @property
    def staged_bytes(self) -> int:
        return sum(transaction.staged_bytes for transaction in self._active.values())

    @property
    def staged_chunks(self) -> int:
        return sum(
            transaction.staged_chunks for transaction in self._active.values()
        )

    @property
    def active_transactions(self) -> int:
        return len(self._active)

    @property
    def committed(self) -> tuple[CommittedReplica, ...]:
        return tuple(self._committed.values())

    def _next_sequence(self) -> int:
        sequence = self._response_sequence
        self._response_sequence += 1
        return sequence

    def _ack(self, frame: Frame, status: str) -> Frame:
        return Frame(
            FrameType.ACK,
            {
                "transaction_id": frame.header["transaction_id"],
                "generation": frame.header["generation"],
                "sequence": self._next_sequence(),
                "ack_sequence": frame.header["sequence"],
                "status": status,
            },
        )

    def _credit(self, frame: Frame) -> Frame:
        return Frame(
            FrameType.CREDIT,
            {
                "transaction_id": frame.header["transaction_id"],
                "generation": frame.header["generation"],
                "sequence": self._next_sequence(),
                "available_bytes": max(
                    0,
                    self.max_staged_bytes - self.staged_bytes,
                ),
                "available_frames": max(
                    0,
                    self.max_staged_chunks - self.staged_chunks,
                ),
            },
        )

    def _responses(self, frame: Frame, status: str) -> tuple[Frame, Frame]:
        return self._ack(frame, status), self._credit(frame)

    def _discard(self, key: tuple[str, int]) -> None:
        self._active.pop(key, None)

    def _check_generation(self, frame: Frame) -> bool:
        transaction_id = frame.header["transaction_id"]
        generation = frame.header["generation"]
        latest = self._latest_generation.get(transaction_id)
        return latest is None or generation >= latest

    def _remember_generation(self, transaction_id: str, generation: int) -> None:
        self._latest_generation[transaction_id] = generation
        self._latest_generation.move_to_end(transaction_id)
        while len(self._latest_generation) > self.recent_generations:
            removable = next(
                (
                    candidate
                    for candidate in self._latest_generation
                    if not any(key[0] == candidate for key in self._active)
                ),
                None,
            )
            if removable is None:
                break
            self._latest_generation.pop(removable)

    def receive(self, frame: Frame) -> tuple[Frame, Frame]:
        self.expire_inactive()
        if frame.frame_type not in {
            FrameType.BEGIN,
            FrameType.PUT_CHUNK,
            FrameType.COMMIT,
            FrameType.ABORT,
        }:
            raise ProtocolError("receiver accepts only transaction frames")
        if not self._check_generation(frame):
            return self._responses(frame, "rejected")
        handlers = {
            FrameType.BEGIN: self._begin,
            FrameType.PUT_CHUNK: self._put_chunk,
            FrameType.COMMIT: self._commit,
            FrameType.ABORT: self._abort,
        }
        status = handlers[frame.frame_type](frame)
        return self._responses(frame, status)

    def expire_inactive(self, now: float | None = None) -> int:
        """Discard abandoned remote transactions and release their credits."""

        current = self._clock() if now is None else now
        expired = tuple(
            key
            for key, transaction in self._active.items()
            if current - transaction.last_activity >= self.transaction_ttl_seconds
        )
        for key in expired:
            self._discard(key)
        return len(expired)

    def _begin(self, frame: Frame) -> str:
        transaction_id = frame.header["transaction_id"]
        generation = frame.header["generation"]
        key = (transaction_id, generation)
        committed = self._committed.get(key)
        if committed is not None:
            if (
                committed.context_digest == frame.header["context_digest"]
                and committed.identity_digest == frame.header["identity_digest"]
            ):
                return "committed"
            return "rejected"
        existing = self._active.get(key)
        if existing is not None:
            if (
                existing.context_digest == frame.header["context_digest"]
                and existing.identity_digest == frame.header["identity_digest"]
            ):
                existing.last_activity = self._clock()
                return "duplicate"
            self._discard(key)
            return "aborted"
        latest = self._latest_generation.get(transaction_id)
        if latest is not None and generation > latest:
            self._discard((transaction_id, latest))
        if len(self._active) >= self.max_active_transactions:
            return "rejected"
        self._active[key] = _ReceiveTransaction(
            transaction_id=transaction_id,
            generation=generation,
            context_digest=frame.header["context_digest"],
            identity_digest=frame.header["identity_digest"],
            last_activity=self._clock(),
            sequences={frame.header["sequence"]: sha256(frame.encode())},
        )
        self._remember_generation(transaction_id, generation)
        return "ok"

    def _put_chunk(self, frame: Frame) -> str:
        key = (frame.header["transaction_id"], frame.header["generation"])
        transaction = self._active.get(key)
        if transaction is None:
            return "rejected"
        sequence = frame.header["sequence"]
        fingerprint = sha256(frame.encode())
        seen = transaction.sequences.get(sequence)
        if seen is not None:
            if seen == fingerprint:
                transaction.last_activity = self._clock()
                return "duplicate"
            self._discard(key)
            return "aborted"
        digest = frame.header["chunk_digest"]
        if sha256(frame.payload) != digest:
            self._discard(key)
            return "aborted"
        index = frame.header["chunk_index"]
        existing = transaction.chunks.get(index)
        if existing is not None:
            if existing[0] == digest and (
                existing[1] is None or existing[1] == frame.payload
            ):
                transaction.last_activity = self._clock()
                return "duplicate"
            self._discard(key)
            return "aborted"
        if len(transaction.chunks) + 1 > self.max_context_chunks:
            self._discard(key)
            return "aborted"
        staged_payload: bytes | None = frame.payload
        if self.on_chunk is not None:
            try:
                self.on_chunk(
                    ReplicaChunk(
                        transaction_id=transaction.transaction_id,
                        generation=transaction.generation,
                        context_digest=transaction.context_digest,
                        identity_digest=transaction.identity_digest,
                        chunk_index=index,
                        chunk_digest=digest,
                        payload=frame.payload,
                    )
                )
            except Exception:
                self._discard(key)
                return "aborted"
            staged_payload = None
        if (
            self.staged_bytes + (len(frame.payload) if staged_payload is not None else 0)
            > self.max_staged_bytes
            or self.staged_chunks + (staged_payload is not None)
            > self.max_staged_chunks
        ):
            self._discard(key)
            return "aborted"
        transaction.chunks[index] = (digest, staged_payload)
        transaction.sequences[sequence] = fingerprint
        transaction.last_activity = self._clock()
        return "ok"

    def _commit(self, frame: Frame) -> str:
        key = (frame.header["transaction_id"], frame.header["generation"])
        prior = self._committed.get(key)
        if prior is not None:
            if prior.manifest_digest == frame.header["manifest_digest"]:
                return "committed"
            return "rejected"
        transaction = self._active.get(key)
        if transaction is None:
            return "rejected"
        if sha256(frame.payload) != frame.header["manifest_digest"]:
            self._discard(key)
            return "aborted"
        try:
            record = decode_commit_record(frame.payload)
        except ProtocolError:
            self._discard(key)
            return "aborted"
        chunk_count = frame.header["chunk_count"]
        expected_indices = set(range(chunk_count))
        if (
            record["transaction_id"] != transaction.transaction_id
            or record["generation"] != transaction.generation
            or record["context_digest"] != transaction.context_digest
            or len(record["chunk_digests"]) != chunk_count
            or set(transaction.chunks) != expected_indices
        ):
            self._discard(key)
            return "aborted"
        ordered = tuple(transaction.chunks[index] for index in range(chunk_count))
        digests = tuple(digest for digest, _ in ordered)
        if digests != tuple(record["chunk_digests"]):
            self._discard(key)
            return "aborted"
        committed = CommittedReplica(
            transaction_id=transaction.transaction_id,
            generation=transaction.generation,
            context_digest=transaction.context_digest,
            identity_digest=transaction.identity_digest,
            manifest_digest=frame.header["manifest_digest"],
            chunk_digests=digests,
        )
        try:
            if self.on_chunk is not None:
                if self.on_stream_commit is not None:
                    self.on_stream_commit(committed, frame.payload)
            elif self.on_commit is not None:
                self.on_commit(
                    committed,
                    tuple(
                        payload
                        for _, payload in ordered
                        if payload is not None
                    ),
                    frame.payload,
                )
        except Exception:
            self._discard(key)
            return "aborted"
        self._discard(key)
        self._committed[key] = committed
        self._committed.move_to_end(key)
        while len(self._committed) > self.recent_commits:
            self._committed.popitem(last=False)
        return "committed"

    def _abort(self, frame: Frame) -> str:
        key = (frame.header["transaction_id"], frame.header["generation"])
        if key in self._active:
            self._discard(key)
            return "aborted"
        return "duplicate"


class BuddySender:
    """Credit-limited sender window that fails open to local-only caching."""

    def __init__(
        self,
        *,
        max_inflight_frames: int = 32,
        max_inflight_bytes: int = 64 << 20,
        recent_local_only: int = 1024,
    ) -> None:
        if (
            max_inflight_frames <= 0
            or max_inflight_bytes <= 0
            or recent_local_only <= 0
        ):
            raise ValueError("sender bounds must be positive")
        self.max_inflight_frames = max_inflight_frames
        self.max_inflight_bytes = max_inflight_bytes
        self.recent_local_only = recent_local_only
        self._available_bytes = 0
        self._available_frames = 0
        self._last_credit_sequence = -1
        self._inflight: OrderedDict[tuple[str, int, int], bytes] = OrderedDict()
        self._local_only: OrderedDict[tuple[str, int], None] = OrderedDict()
        self._best_effort_aborts: OrderedDict[tuple[str, int], bytes] = (
            OrderedDict()
        )

    @property
    def inflight_bytes(self) -> int:
        return sum(len(encoded) for encoded in self._inflight.values())

    @property
    def inflight_frames(self) -> int:
        return len(self._inflight)

    def is_local_only(
        self,
        transaction_id: str,
        generation: int | None = None,
    ) -> bool:
        if generation is None:
            return any(key[0] == transaction_id for key in self._local_only)
        return (transaction_id, generation) in self._local_only

    def apply_credit(self, frame: Frame) -> None:
        if frame.frame_type is not FrameType.CREDIT:
            raise ProtocolError("expected CREDIT frame")
        # Credits are absolute receiver availability. Replayed CREDIT frames
        # therefore cannot inflate the window. Older absolute snapshots are
        # ignored as well, or reordering could resurrect spent capacity.
        sequence = frame.header["sequence"]
        if sequence <= self._last_credit_sequence:
            return
        self._last_credit_sequence = sequence
        self._available_bytes = frame.header["available_bytes"]
        self._available_frames = frame.header["available_frames"]

    def _mark_local_only(self, transaction_id: str, generation: int) -> None:
        key = (transaction_id, generation)
        self._local_only[key] = None
        self._local_only.move_to_end(key)
        while len(self._local_only) > self.recent_local_only:
            self._local_only.popitem(last=False)
        self._inflight = OrderedDict(
            (frame_key, encoded)
            for frame_key, encoded in self._inflight.items()
            if frame_key[:2] != key
        )
        self._best_effort_aborts[key] = Frame(
            FrameType.ABORT,
            {
                "transaction_id": transaction_id,
                "generation": generation,
                # Reserved sender-generated control sequence. ABORT is
                # idempotent and receiver cleanup does not depend on ordering.
                "sequence": (1 << 63) - 1,
                "reason": "sender degraded to local-only",
            },
        ).encode()
        self._best_effort_aborts.move_to_end(key)

    def offer(self, frame: Frame) -> bytes | None:
        if frame.frame_type not in {
            FrameType.BEGIN,
            FrameType.PUT_CHUNK,
            FrameType.COMMIT,
            FrameType.ABORT,
        }:
            raise ProtocolError("sender accepts only transaction frames")
        transaction_id = frame.header["transaction_id"]
        generation = frame.header["generation"]
        if (
            (transaction_id, generation) in self._local_only
            and frame.frame_type is not FrameType.ABORT
        ):
            return None
        encoded = frame.encode()
        sequence = frame.header["sequence"]
        frame_key = (transaction_id, generation, sequence)
        existing = self._inflight.get(frame_key)
        if existing is not None:
            if existing == encoded:
                return existing
            self._mark_local_only(transaction_id, generation)
            return None
        if (
            self.inflight_frames + 1 > self.max_inflight_frames
            or self.inflight_bytes + len(encoded) > self.max_inflight_bytes
        ):
            self._mark_local_only(transaction_id, generation)
            return None
        if frame.frame_type is FrameType.PUT_CHUNK:
            if (
                self._available_frames < 1
                or self._available_bytes < len(frame.payload)
            ):
                self._mark_local_only(transaction_id, generation)
                return None
            self._available_frames -= 1
            self._available_bytes -= len(frame.payload)
        self._inflight[frame_key] = encoded
        return encoded

    def acknowledge(self, frame: Frame) -> None:
        if frame.frame_type is not FrameType.ACK:
            raise ProtocolError("expected ACK frame")
        frame_key = (
            frame.header["transaction_id"],
            frame.header["generation"],
            frame.header["ack_sequence"],
        )
        if frame_key not in self._inflight:
            return
        transaction_id, generation, _ = frame_key
        self._inflight.pop(frame_key)
        if frame.header["status"] in {"rejected", "aborted"}:
            self._mark_local_only(transaction_id, generation)

    def retransmit(self) -> tuple[bytes, ...]:
        """Return exact unacknowledged wire frames without consuming credits."""

        return tuple(self._inflight.values())

    def take_best_effort_aborts(self) -> tuple[bytes, ...]:
        """Return receiver-cleanup frames without delaying local publication."""

        frames = tuple(self._best_effort_aborts.values())
        self._best_effort_aborts.clear()
        return frames
