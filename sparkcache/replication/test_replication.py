from __future__ import annotations

import unittest

from sparkcache.replication.protocol import (
    MAX_FRAME_BYTES,
    PREFIX,
    Frame,
    FrameStreamDecoder,
    FrameType,
    ProtocolError,
    encode_commit_record,
    sha256,
)
from sparkcache.replication.state import BuddyReceiver, BuddySender


TX = "1" * 32
CONTEXT = "2" * 64
IDENTITY = "3" * 64


def _begin(*, generation: int = 1, sequence: int = 0) -> Frame:
    return Frame(
        FrameType.BEGIN,
        {
            "transaction_id": TX,
            "generation": generation,
            "sequence": sequence,
            "context_digest": CONTEXT,
            "identity_digest": IDENTITY,
        },
    )


def _put(
    payload: bytes,
    *,
    index: int = 0,
    generation: int = 1,
    sequence: int = 1,
) -> Frame:
    return Frame(
        FrameType.PUT_CHUNK,
        {
            "transaction_id": TX,
            "generation": generation,
            "sequence": sequence,
            "chunk_index": index,
            "chunk_digest": sha256(payload),
        },
        payload,
    )


def _commit(
    payloads: tuple[bytes, ...],
    *,
    generation: int = 1,
    sequence: int = 10,
) -> Frame:
    record = encode_commit_record(
        transaction_id=TX,
        generation=generation,
        context_digest=CONTEXT,
        chunk_digests=[sha256(payload) for payload in payloads],
    )
    return Frame(
        FrameType.COMMIT,
        {
            "transaction_id": TX,
            "generation": generation,
            "sequence": sequence,
            "manifest_digest": sha256(record),
            "chunk_count": len(payloads),
        },
        record,
    )


class CodecTests(unittest.TestCase):
    def test_fragmented_and_coalesced_frames_round_trip(self) -> None:
        encoded = _begin().encode() + _put(b"chunk").encode()
        decoder = FrameStreamDecoder()
        decoded = []
        for offset in range(0, len(encoded), 3):
            decoded.extend(decoder.feed(encoded[offset : offset + 3]))
        self.assertEqual(decoded, [_begin(), _put(b"chunk")])
        self.assertEqual(decoder.buffered_bytes, 0)

    def test_noncanonical_or_oversized_input_fails_closed(self) -> None:
        canonical = _begin().encode()
        prefix = canonical[: PREFIX.size]
        header = canonical[PREFIX.size :]
        noncanonical = prefix + header.replace(b",", b", ", 1)
        magic, version, frame_type, flags, _, payload = PREFIX.unpack_from(prefix)
        noncanonical = (
            PREFIX.pack(
                magic,
                version,
                frame_type,
                flags,
                len(noncanonical) - PREFIX.size,
                payload,
            )
            + noncanonical[PREFIX.size :]
        )
        with self.assertRaisesRegex(ProtocolError, "not canonical"):
            FrameStreamDecoder().feed(noncanonical)

        decoder = FrameStreamDecoder(
            max_frame_bytes=MAX_FRAME_BYTES,
            max_buffer_bytes=MAX_FRAME_BYTES,
        )
        with self.assertRaisesRegex(ProtocolError, "buffer limit"):
            decoder.feed(b"x" * (MAX_FRAME_BYTES + 1))


class ReceiverTests(unittest.TestCase):
    def test_manifest_last_commit_and_credit_release(self) -> None:
        callbacks = []
        receiver = BuddyReceiver(
            max_staged_bytes=32,
            max_staged_chunks=2,
            on_commit=lambda *items: callbacks.append(items),
        )
        begin_ack, _ = receiver.receive(_begin())
        self.assertEqual(begin_ack.header["status"], "ok")
        payloads = (b"first", b"second")
        receiver.receive(_put(payloads[0], index=0, sequence=1))
        receiver.receive(_put(payloads[1], index=1, sequence=2))
        self.assertEqual(receiver.committed, ())
        self.assertEqual(receiver.staged_bytes, len(b"firstsecond"))

        commit_ack, credit = receiver.receive(_commit(payloads))
        self.assertEqual(commit_ack.header["status"], "committed")
        self.assertEqual(len(callbacks), 1)
        self.assertEqual(callbacks[0][1], payloads)
        self.assertEqual(receiver.staged_bytes, 0)
        self.assertEqual(credit.header["available_bytes"], 32)

    def test_duplicate_frames_are_idempotent(self) -> None:
        callbacks = []
        receiver = BuddyReceiver(on_commit=lambda *items: callbacks.append(items))
        self.assertEqual(receiver.receive(_begin())[0].header["status"], "ok")
        self.assertEqual(receiver.receive(_begin())[0].header["status"], "duplicate")
        put = _put(b"same")
        self.assertEqual(receiver.receive(put)[0].header["status"], "ok")
        staged = receiver.staged_bytes
        self.assertEqual(receiver.receive(put)[0].header["status"], "duplicate")
        self.assertEqual(receiver.staged_bytes, staged)
        commit = _commit((b"same",))
        self.assertEqual(receiver.receive(commit)[0].header["status"], "committed")
        self.assertEqual(receiver.receive(commit)[0].header["status"], "committed")
        self.assertEqual(len(callbacks), 1)

    def test_bad_digest_or_conflicting_index_aborts_only_replica(self) -> None:
        receiver = BuddyReceiver()
        receiver.receive(_begin())
        bad = Frame(
            FrameType.PUT_CHUNK,
            {
                **dict(_put(b"good").header),
                "chunk_digest": sha256(b"not-the-payload"),
            },
            b"good",
        )
        self.assertEqual(receiver.receive(bad)[0].header["status"], "aborted")
        self.assertEqual(receiver.active_transactions, 0)

        receiver.receive(_begin(generation=2))
        receiver.receive(_put(b"first", generation=2))
        conflict = _put(b"other", generation=2, sequence=2)
        self.assertEqual(receiver.receive(conflict)[0].header["status"], "aborted")
        self.assertEqual(receiver.active_transactions, 0)

    def test_reused_sequence_with_different_frame_aborts_transaction(self) -> None:
        receiver = BuddyReceiver()
        receiver.receive(_begin())
        receiver.receive(_put(b"first", index=0, sequence=1))
        reused = _put(b"second", index=1, sequence=1)
        self.assertEqual(receiver.receive(reused)[0].header["status"], "aborted")
        self.assertEqual(receiver.active_transactions, 0)

    def test_stale_generation_and_capacity_are_bounded(self) -> None:
        receiver = BuddyReceiver(max_staged_bytes=4, max_staged_chunks=1)
        receiver.receive(_begin(generation=2))
        stale = _put(b"x", generation=1)
        self.assertEqual(receiver.receive(stale)[0].header["status"], "rejected")
        too_large = _put(b"12345", generation=2)
        self.assertEqual(receiver.receive(too_large)[0].header["status"], "aborted")
        self.assertEqual(receiver.staged_bytes, 0)

    def test_abort_is_idempotent(self) -> None:
        receiver = BuddyReceiver()
        receiver.receive(_begin())
        abort = Frame(
            FrameType.ABORT,
            {
                "transaction_id": TX,
                "generation": 1,
                "sequence": 5,
                "reason": "local queue full",
            },
        )
        self.assertEqual(receiver.receive(abort)[0].header["status"], "aborted")
        self.assertEqual(receiver.receive(abort)[0].header["status"], "duplicate")

    def test_inactive_transaction_expires_and_releases_staging(self) -> None:
        now = [10.0]
        receiver = BuddyReceiver(
            transaction_ttl_seconds=5.0,
            clock=lambda: now[0],
        )
        receiver.receive(_begin())
        receiver.receive(_put(b"staged"))
        self.assertEqual(receiver.active_transactions, 1)
        self.assertEqual(receiver.staged_bytes, len(b"staged"))

        now[0] = 15.0

        self.assertEqual(receiver.expire_inactive(), 1)
        self.assertEqual(receiver.active_transactions, 0)
        self.assertEqual(receiver.staged_bytes, 0)

    def test_streaming_receiver_persists_393k_chunk_count_without_ram_staging(
        self,
    ) -> None:
        persisted_indexes: list[int] = []
        commits = []
        receiver = BuddyReceiver(
            max_staged_bytes=1,
            max_staged_chunks=1,
            max_context_chunks=2000,
            on_chunk=lambda chunk: persisted_indexes.append(chunk.chunk_index),
            on_stream_commit=lambda *items: commits.append(items),
        )
        receiver.receive(_begin())
        payloads = tuple(f"chunk-{index}".encode() for index in range(1535))
        for index, payload in enumerate(payloads):
            ack, _ = receiver.receive(
                _put(payload, index=index, sequence=index + 1)
            )
            self.assertEqual(ack.header["status"], "ok")

        self.assertEqual(receiver.staged_bytes, 0)
        self.assertEqual(receiver.staged_chunks, 0)
        self.assertEqual(receiver.active_transactions, 1)
        self.assertEqual(persisted_indexes, list(range(1535)))

        ack, _ = receiver.receive(_commit(payloads, sequence=1536))

        self.assertEqual(ack.header["status"], "committed")
        self.assertEqual(receiver.active_transactions, 0)
        self.assertEqual(len(commits), 1)
        self.assertEqual(len(commits[0][0].chunk_digests), 1535)


class SenderTests(unittest.TestCase):
    def test_sequences_are_scoped_to_transaction_and_generation(self) -> None:
        sender = BuddySender(max_inflight_frames=4, max_inflight_bytes=4096)
        other_tx = "4" * 32
        first = _begin(sequence=0)
        second = Frame(
            FrameType.BEGIN,
            {
                "transaction_id": other_tx,
                "generation": 1,
                "sequence": 0,
                "context_digest": "5" * 64,
                "identity_digest": IDENTITY,
            },
        )

        self.assertIsNotNone(sender.offer(first))
        self.assertIsNotNone(sender.offer(second))
        self.assertEqual(sender.inflight_frames, 2)
        self.assertFalse(sender.is_local_only(other_tx))

    def test_credit_is_absolute_and_retransmit_is_exact(self) -> None:
        sender = BuddySender(max_inflight_frames=4, max_inflight_bytes=4096)
        credit = Frame(
            FrameType.CREDIT,
            {
                "transaction_id": TX,
                "generation": 1,
                "sequence": 100,
                "available_bytes": 8,
                "available_frames": 1,
            },
        )
        sender.apply_credit(credit)
        sender.apply_credit(credit)
        wire = sender.offer(_put(b"12345678"))
        self.assertIsNotNone(wire)
        self.assertEqual(sender.retransmit(), (wire,))

        no_second_credit = _put(b"x", index=1, sequence=2)
        self.assertIsNone(sender.offer(no_second_credit))
        self.assertTrue(sender.is_local_only(TX))
        self.assertEqual(sender.inflight_frames, 0)

    def test_out_of_order_credit_and_wrong_transaction_ack_are_ignored(self) -> None:
        sender = BuddySender(max_inflight_frames=4, max_inflight_bytes=4096)
        sender.apply_credit(
            Frame(
                FrameType.CREDIT,
                {
                    "transaction_id": TX,
                    "generation": 1,
                    "sequence": 101,
                    "available_bytes": 0,
                    "available_frames": 0,
                },
            )
        )
        sender.apply_credit(
            Frame(
                FrameType.CREDIT,
                {
                    "transaction_id": TX,
                    "generation": 1,
                    "sequence": 100,
                    "available_bytes": 1024,
                    "available_frames": 10,
                },
            )
        )
        self.assertIsNone(sender.offer(_put(b"x")))

        second = BuddySender(max_inflight_frames=4, max_inflight_bytes=4096)
        self.assertIsNotNone(second.offer(_begin()))
        wrong_ack = Frame(
            FrameType.ACK,
            {
                "transaction_id": "9" * 32,
                "generation": 1,
                "sequence": 102,
                "ack_sequence": 0,
                "status": "ok",
            },
        )
        second.acknowledge(wrong_ack)
        self.assertEqual(second.inflight_frames, 1)

    def test_ack_releases_window_and_remote_failure_stays_local_only(self) -> None:
        sender = BuddySender(max_inflight_frames=1, max_inflight_bytes=4096)
        wire = sender.offer(_begin())
        self.assertIsNotNone(wire)
        self.assertEqual(sender.inflight_frames, 1)
        ack = Frame(
            FrameType.ACK,
            {
                "transaction_id": TX,
                "generation": 1,
                "sequence": 99,
                "ack_sequence": 0,
                "status": "rejected",
            },
        )
        sender.acknowledge(ack)
        self.assertEqual(sender.inflight_frames, 0)
        self.assertTrue(sender.is_local_only(TX))
        self.assertIsNone(sender.offer(_begin(sequence=1)))

    def test_local_only_transition_emits_best_effort_abort(self) -> None:
        receiver = BuddyReceiver()
        sender = BuddySender(max_inflight_frames=4, max_inflight_bytes=4096)
        begin = _begin()
        self.assertIsNotNone(sender.offer(begin))
        receiver.receive(begin)
        self.assertEqual(receiver.active_transactions, 1)

        # No receiver credit has been installed, so this PUT degrades only the
        # replica and queues a cleanup control frame.
        self.assertIsNone(sender.offer(_put(b"payload")))
        encoded_aborts = sender.take_best_effort_aborts()
        self.assertEqual(len(encoded_aborts), 1)
        self.assertEqual(sender.take_best_effort_aborts(), ())
        abort = FrameStreamDecoder().feed(encoded_aborts[0])
        self.assertEqual(len(abort), 1)
        self.assertIs(abort[0].frame_type, FrameType.ABORT)

        receiver.receive(abort[0])
        self.assertEqual(receiver.active_transactions, 0)


if __name__ == "__main__":
    unittest.main()
