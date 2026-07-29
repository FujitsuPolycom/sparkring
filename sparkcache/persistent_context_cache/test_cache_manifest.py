from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import tempfile
import threading
import unittest
import weakref
from pathlib import Path
from unittest import mock

import cache_manifest
from cache_manifest import (
    CacheIdentity,
    CommitConflict,
    ContextChunk,
    IncompleteEntry,
    ManifestStore,
    StateRecord,
)


def _identity() -> CacheIdentity:
    return CacheIdentity(
        target_checkpoint="1" * 64,
        draft_checkpoint="2" * 64,
        quantization_layout="nvfp4-ds-mla-v1",
        rope_layout="glm52-bf16-rope-v1",
        tp_degree=4,
        dcp_degree=1,
        chunk_tokens=256,
    )


def _chunk(start: int = 0, end: int = 256) -> ContextChunk:
    return ContextChunk(
        logical_start=start,
        logical_end=end,
        records={
            StateRecord.TARGET_CKV: b"target-ckv-and-scales",
            StateRecord.SPARSE_INDEXER: b"sparse-indexer",
            StateRecord.MTP_DRAFT_KV: b"mtp-draft-kv",
            StateRecord.BOUNDARY_HIDDEN: b"boundary-hidden",
            StateRecord.LOGICAL_POSITIONS: b"logical-positions",
        },
    )


class ManifestStoreTests(unittest.TestCase):
    def test_streaming_macro_batch_uses_one_chunk_directory_barrier(self) -> None:
        context_digest = hashlib.sha256(b"streamed-macro-batch").hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunk_directory = root / "chunks"
            chunk_directory.mkdir()
            store = ManifestStore(root)
            transaction = store.begin_context(
                identity=_identity(),
                context_digest=context_digest,
                span_tokens=768,
            )
            barriers: list[Path] = []

            with mock.patch.object(
                cache_manifest,
                "_fsync_directory",
                side_effect=lambda path: barriers.append(Path(path)),
            ):
                receipts = transaction.append_chunks(
                    (
                        _chunk(0, 256),
                        _chunk(256, 512),
                        _chunk(512, 768),
                    )
                )

            self.assertEqual(len(receipts), 3)
            self.assertEqual(barriers, [chunk_directory])

            transaction.commit_manifest()
            lookup = store.lookup(_identity(), context_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(
                store.restore(lookup),
                (
                    _chunk(0, 256),
                    _chunk(256, 512),
                    _chunk(512, 768),
                ),
            )

    def test_streaming_macro_batch_retry_is_idempotent(self) -> None:
        context_digest = hashlib.sha256(b"macro-batch-retry").hexdigest()
        chunks = (_chunk(0, 256), _chunk(256, 512), _chunk(512, 768))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            transaction = store.begin_context(
                identity=_identity(),
                context_digest=context_digest,
                span_tokens=768,
            )

            first = transaction.append_chunks(chunks)
            second = transaction.append_chunks(chunks)
            self.assertEqual(second, first)

            transaction.commit_manifest()
            self.assertEqual(transaction.append_chunks(chunks), first)
            self.assertEqual(len(tuple((root / "chunks").glob("*.spcc"))), 3)
            lookup = store.lookup(_identity(), context_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(store.restore(lookup), chunks)

    def test_streaming_macro_batch_conflict_fails_before_partial_append(self) -> None:
        context_digest = hashlib.sha256(b"macro-batch-conflict").hexdigest()
        first = _chunk(0, 256)
        second = _chunk(256, 512)
        replacement_records = dict(second.records)
        replacement_records[StateRecord.TARGET_CKV] = b"different-target"
        conflicting_second = ContextChunk(256, 512, replacement_records)

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            transaction = store.begin_context(
                identity=_identity(),
                context_digest=context_digest,
                span_tokens=512,
            )
            transaction.append_chunks((first, second))

            with self.assertRaisesRegex(CommitConflict, "logical range"):
                transaction.append_chunks((first, conflicting_second))

            transaction.commit_manifest()
            lookup = store.lookup(_identity(), context_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(store.restore(lookup), (first, second))

    def test_streaming_transaction_keeps_partial_publication_invisible(self) -> None:
        context_digest = hashlib.sha256(b"partial-stream").hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            transaction = store.begin(
                identity=_identity(),
                context_digest=context_digest,
            )
            transaction.append_chunk(_chunk())

            self.assertEqual(len(tuple((root / "chunks").glob("*.spcc"))), 1)
            self.assertFalse(store.lookup(_identity(), context_digest).is_hit)

            transaction.abort()
            self.assertFalse(store.lookup(_identity(), context_digest).is_hit)

    def test_streaming_transaction_commits_without_retaining_chunk_payloads(
        self,
    ) -> None:
        context_digest = hashlib.sha256(b"streamed-two-chunks").hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            transaction = store.begin_context(
                identity=_identity(),
                context_digest=context_digest,
            )
            first = _chunk(0, 256)
            first_reference = weakref.ref(first)
            first_receipt = transaction.append_chunk(first)
            del first
            gc.collect()

            self.assertIsNone(first_reference())
            self.assertEqual(first_receipt.logical_start, 0)
            self.assertEqual(first_receipt.logical_end, 256)

            transaction.append_chunk(_chunk(256, 384))
            receipt = transaction.commit_manifest()
            self.assertEqual(receipt.committed_tokens, 384)

            lookup = store.lookup(_identity(), context_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(
                [(chunk.logical_start, chunk.logical_end) for chunk in store.restore(lookup)],
                [(0, 256), (256, 384)],
            )

    def test_streaming_duplicate_publication_is_idempotent(self) -> None:
        context_digest = hashlib.sha256(b"idempotent-stream").hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            receipts = []
            for _ in range(2):
                transaction = store.begin_context(
                    identity=_identity(),
                    context_digest=context_digest,
                )
                transaction.append_chunk(_chunk())
                receipts.append(transaction.commit_manifest())

            self.assertEqual(receipts[0], receipts[1])
            self.assertEqual(
                len(tuple((Path(directory) / "chunks").glob("*.spcc"))),
                1,
            )
            self.assertEqual(
                len(tuple((Path(directory) / "manifests").rglob("*.json"))),
                1,
            )

    def test_streaming_concurrent_identical_append_is_serialized_and_idempotent(
        self,
    ) -> None:
        context_digest = hashlib.sha256(b"concurrent-identical-append").hexdigest()
        entered_publish = threading.Event()
        release_publish = threading.Event()
        second_started = threading.Event()
        original_publish = cache_manifest._publish_immutable_batch
        chunk_publish_calls = 0
        receipts = []
        errors = []

        def blocking_publish(objects: tuple[tuple[Path, bytes], ...]) -> None:
            nonlocal chunk_publish_calls
            if objects:
                chunk_publish_calls += 1
                entered_publish.set()
                if not release_publish.wait(timeout=5):
                    raise TimeoutError("test did not release chunk publication")
            original_publish(objects)

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            transaction = store.begin_context(
                identity=_identity(),
                context_digest=context_digest,
                span_tokens=256,
            )

            def append(*, announce: bool = False) -> None:
                if announce:
                    second_started.set()
                try:
                    receipts.append(transaction.append_chunk(_chunk()))
                except BaseException as error:  # pragma: no cover - assertion data
                    errors.append(error)

            with mock.patch.object(
                cache_manifest,
                "_publish_immutable_batch",
                blocking_publish,
            ):
                first = threading.Thread(target=append)
                second = threading.Thread(
                    target=append,
                    kwargs={"announce": True},
                )
                first.start()
                self.assertTrue(entered_publish.wait(timeout=5))
                second.start()
                self.assertTrue(second_started.wait(timeout=5))
                release_publish.set()
                first.join(timeout=5)
                second.join(timeout=5)

            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(len(receipts), 2)
            self.assertEqual(receipts[0], receipts[1])
            self.assertEqual(chunk_publish_calls, 1)

            committed = transaction.commit_manifest()
            self.assertEqual(transaction.commit_manifest(), committed)
            self.assertEqual(transaction.append_chunk(_chunk()), receipts[0])
            lookup = store.lookup(_identity(), context_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(store.restore(lookup), (_chunk(),))

    def test_streaming_conflicting_retry_for_same_range_fails_closed(self) -> None:
        context_digest = hashlib.sha256(b"conflicting-range-retry").hexdigest()
        replacement_records = dict(_chunk().records)
        replacement_records[StateRecord.TARGET_CKV] = b"different-target"
        replacement = ContextChunk(0, 256, replacement_records)

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            transaction = store.begin_context(
                identity=_identity(),
                context_digest=context_digest,
                span_tokens=256,
            )
            transaction.append_chunk(_chunk())
            with self.assertRaisesRegex(
                CommitConflict,
                "logical range",
            ):
                transaction.append_chunk(replacement)

            transaction.commit_manifest()
            lookup = store.lookup(_identity(), context_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(store.restore(lookup), (_chunk(),))

    def test_streaming_conflict_cannot_replace_committed_manifest(self) -> None:
        context_digest = hashlib.sha256(b"conflicting-stream").hexdigest()
        replacement_records = dict(_chunk().records)
        replacement_records[StateRecord.TARGET_CKV] = b"different-target"
        replacement = ContextChunk(
            logical_start=0,
            logical_end=256,
            records=replacement_records,
        )

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            original = store.begin_context(
                identity=_identity(),
                context_digest=context_digest,
            )
            original.append_chunk(_chunk())
            original_receipt = original.commit_manifest()

            conflict = store.begin_context(
                identity=_identity(),
                context_digest=context_digest,
            )
            conflict.append_chunk(replacement)
            with self.assertRaises(CommitConflict):
                conflict.commit_manifest()
            conflict.abort()

            lookup = store.lookup(_identity(), context_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(lookup.manifest_digest, original_receipt.manifest_digest)
            self.assertEqual(store.restore(lookup), (_chunk(),))

    def test_streaming_transaction_rejects_invalid_lifecycle_and_geometry(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)

            empty = store.begin_context(
                identity=_identity(),
                context_digest="a" * 64,
            )
            with self.assertRaisesRegex(ValueError, "at least one context chunk"):
                empty.commit_manifest()
            empty.abort()
            with self.assertRaisesRegex(RuntimeError, "aborted"):
                empty.append_chunk(_chunk())

            partial = store.begin_context(
                identity=_identity(),
                context_digest="b" * 64,
            )
            partial.append_chunk(_chunk(0, 128))
            with self.assertRaisesRegex(
                ValueError,
                "only the final context chunk may be partial",
            ):
                partial.append_chunk(_chunk(128, 256))
            partial.abort()

            committed = store.begin_context(
                identity=_identity(),
                context_digest="c" * 64,
            )
            committed.append_chunk(_chunk())
            committed_receipt = committed.commit_manifest()
            self.assertEqual(committed.commit_manifest(), committed_receipt)
            with self.assertRaisesRegex(RuntimeError, "committed"):
                committed.abort()

    def test_declared_span_prevents_partial_manifest_visibility(self) -> None:
        context_digest = hashlib.sha256(b"declared-span").hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            transaction = store.begin_context(
                identity=_identity(),
                context_digest=context_digest,
                span_tokens=512,
            )
            transaction.append_chunk(_chunk(0, 256))

            with self.assertRaisesRegex(IncompleteEntry, "declared context span"):
                transaction.commit_manifest()
            self.assertFalse(store.lookup(_identity(), context_digest).is_hit)

            transaction.append_chunk(_chunk(256, 512))
            receipt = transaction.commit_manifest()
            self.assertEqual(receipt.committed_tokens, 512)
            self.assertTrue(store.lookup(_identity(), context_digest).is_hit)

    def test_checkpoint_identity_requires_content_digests(self) -> None:
        for field, value in (
            ("target_checkpoint", "mutable/model/path"),
            ("draft_checkpoint", "latest"),
            ("target_checkpoint", "A" * 64),
            ("draft_checkpoint", None),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    f"{field} must be a 64-character lowercase SHA-256",
                ):
                    dataclasses.replace(_identity(), **{field: value})

    def test_successful_commit_durably_publishes_chunks_before_manifest(
        self,
    ) -> None:
        events: list[tuple[str, Path]] = []
        original_link = cache_manifest.os.link

        def recording_link(source: Path, destination: Path) -> None:
            original_link(source, destination)
            events.append(("link", Path(destination)))

        def recording_directory_fsync(path: Path) -> None:
            events.append(("fsync-directory", Path(path)))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            with (
                mock.patch.object(cache_manifest.os, "link", recording_link),
                mock.patch.object(
                    cache_manifest,
                    "_fsync_directory",
                    recording_directory_fsync,
                ),
            ):
                store.commit(
                    identity=_identity(),
                    context_digest="0" * 64,
                    chunks=(_chunk(),),
                )

        chunk_link = next(
            index
            for index, event in enumerate(events)
            if event[0] == "link" and event[1].suffix == ".spcc"
        )
        manifest_link = next(
            index
            for index, event in enumerate(events)
            if event[0] == "link" and event[1].suffix == ".json"
        )
        chunk_fsync = next(
            index
            for index, event in enumerate(events)
            if index > chunk_link
            and event == ("fsync-directory", events[chunk_link][1].parent)
        )
        manifest_fsync = next(
            index
            for index, event in enumerate(events)
            if index > manifest_link
            and event == ("fsync-directory", events[manifest_link][1].parent)
        )
        self.assertLess(chunk_link, chunk_fsync)
        self.assertLess(chunk_fsync, manifest_link)
        self.assertLess(manifest_link, manifest_fsync)

    def test_manifest_directory_fsync_failure_fails_the_commit(self) -> None:
        identity = _identity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "chunks").mkdir()
            manifest_directory = root / "manifests" / identity.storage_key
            manifest_directory.mkdir(parents=True)
            store = ManifestStore(root)

            def fail_manifest_barrier(path: Path) -> None:
                if Path(path) == manifest_directory:
                    raise OSError("simulated directory fsync failure")

            with (
                mock.patch.object(
                    cache_manifest,
                    "_fsync_directory",
                    fail_manifest_barrier,
                ),
                self.assertRaisesRegex(OSError, "directory fsync failure"),
            ):
                store.commit(
                    identity=identity,
                    context_digest="0" * 64,
                    chunks=(_chunk(),),
                )

    def test_complete_entry_round_trips_byte_exactly(self) -> None:
        context_digest = hashlib.sha256(b"tokens-0-through-255").hexdigest()
        expected = _chunk()

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(Path(directory))
            receipt = store.commit(
                identity=_identity(),
                context_digest=context_digest,
                chunks=[expected],
            )

            lookup = store.lookup(_identity(), context_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(lookup.manifest_digest, receipt.manifest_digest)
            self.assertEqual(store.restore(lookup), (expected,))

    def test_manifest_probe_then_restore_reads_each_chunk_once(self) -> None:
        """The optimized load path probes metadata, then verifies once."""
        context_digest = hashlib.sha256(b"single-pass-restore").hexdigest()
        expected = _chunk()

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(Path(directory))
            store.commit(
                identity=_identity(),
                context_digest=context_digest,
                chunks=[expected],
            )
            chunk_path = next((Path(directory) / "chunks").glob("*.spcc"))
            original_read_bytes = Path.read_bytes
            chunk_reads = 0

            def counted_read_bytes(path: Path) -> bytes:
                nonlocal chunk_reads
                if path == chunk_path:
                    chunk_reads += 1
                return original_read_bytes(path)

            with mock.patch.object(Path, "read_bytes", counted_read_bytes):
                lookup = store.lookup(_identity(), context_digest, verify_chunks=False)
                self.assertTrue(lookup.is_hit, lookup.reason)
                self.assertEqual(store.restore(lookup), (expected,))

            self.assertEqual(chunk_reads, 1)

    def test_restore_hashes_the_authenticated_chunk_only_once(self) -> None:
        """The outer descriptor hash makes per-record rehashing redundant."""
        context_digest = hashlib.sha256(b"single-hash-restore").hexdigest()
        expected = _chunk()

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(Path(directory))
            store.commit(
                identity=_identity(),
                context_digest=context_digest,
                chunks=[expected],
            )
            lookup = store.lookup(_identity(), context_digest, verify_chunks=False)
            original_sha256 = cache_manifest._sha256
            hashed_lengths: list[int] = []

            def counted_sha256(
                value: bytes | bytearray | memoryview,
            ) -> str:
                hashed_lengths.append(len(value))
                return original_sha256(value)

            with mock.patch.object(cache_manifest, "_sha256", counted_sha256):
                self.assertEqual(store.restore(lookup), (expected,))

            chunk_bytes = (
                next((Path(directory) / "chunks").glob("*.spcc")).stat().st_size
            )
            self.assertEqual(hashed_lengths, [chunk_bytes])

    def test_standalone_decoder_still_checks_record_digests(self) -> None:
        encoded = bytearray(cache_manifest._encode_chunk(_chunk()))
        prefix = cache_manifest._CHUNK_PREFIX
        _, _, header_length = prefix.unpack_from(encoded)
        payload_start = prefix.size + header_length
        encoded[payload_start] ^= 0x01

        with self.assertRaises(cache_manifest.CacheFormatError):
            cache_manifest._decode_chunk(bytes(encoded))

    def test_new_encoder_is_byte_identical_to_frozen_v1_encoder(self) -> None:
        """Changing assembly mechanics must not change the on-disk ABI."""
        chunk = _chunk()
        payload = bytearray()
        descriptors = []
        for kind in sorted(chunk.records, key=lambda item: item.value):
            value = chunk.records[kind]
            offset = len(payload)
            payload.extend(value)
            descriptors.append(
                {
                    "kind": kind.value,
                    "offset": offset,
                    "length": len(value),
                    "sha256": hashlib.sha256(value).hexdigest(),
                }
            )
        header = cache_manifest._canonical_json(
            {
                "format_abi": cache_manifest.FORMAT_ABI,
                "logical_start": chunk.logical_start,
                "logical_end": chunk.logical_end,
                "records": descriptors,
            }
        )
        legacy = (
            cache_manifest._CHUNK_PREFIX.pack(
                cache_manifest._CHUNK_MAGIC,
                cache_manifest.FORMAT_ABI,
                len(header),
            )
            + header
            + payload
        )
        encoded = cache_manifest._encode_chunk(chunk)

        self.assertEqual(encoded, legacy)
        self.assertEqual(cache_manifest._decode_chunk(legacy), chunk)

    def test_duplicate_writer_cannot_replace_a_committed_context(self) -> None:
        context_digest = hashlib.sha256(b"same-logical-context").hexdigest()
        original = _chunk()
        replacement = ContextChunk(
            logical_start=0,
            logical_end=256,
            records={
                **original.records,
                StateRecord.BOUNDARY_HIDDEN: b"different-boundary",
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(Path(directory))
            store.commit(
                identity=_identity(),
                context_digest=context_digest,
                chunks=[original],
            )

            with self.assertRaises(CommitConflict):
                store.commit(
                    identity=_identity(),
                    context_digest=context_digest,
                    chunks=[replacement],
                )

            lookup = store.lookup(_identity(), context_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(store.restore(lookup), (original,))

    def test_missing_mtp_state_is_rejected_before_publication(self) -> None:
        records = dict(_chunk().records)
        del records[StateRecord.MTP_DRAFT_KV]
        incomplete = ContextChunk(
            logical_start=0,
            logical_end=256,
            records=records,
        )

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            with self.assertRaises(IncompleteEntry):
                store.commit(
                    identity=_identity(),
                    context_digest="c" * 64,
                    chunks=(incomplete,),
                )

    def test_live_forward_policy_permits_absent_boundary_hidden(self) -> None:
        records = dict(_chunk().records)
        del records[StateRecord.BOUNDARY_HIDDEN]
        chunk = ContextChunk(logical_start=0, logical_end=256, records=records)
        identity = dataclasses.replace(
            _identity(),
            boundary_hidden_policy="live_forward",
            dcp_degree=4,
            dcp_shard_rank=2,
        )

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            receipt = store.commit(
                identity=identity, context_digest="d" * 64, chunks=(chunk,)
            )
            self.assertEqual(receipt.committed_tokens, 256)
            lookup = store.lookup(identity, "d" * 64)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(store.restore(lookup), (chunk,))
            strict_lookup = store.lookup(
                dataclasses.replace(identity, boundary_hidden_policy="persisted"),
                "d" * 64,
            )
            self.assertFalse(strict_lookup.is_hit)

    def test_shard_rank_isolates_entries_between_ranks(self) -> None:
        chunk = _chunk()
        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            rank1 = dataclasses.replace(_identity(), dcp_degree=4, dcp_shard_rank=1)
            store.commit(identity=rank1, context_digest="e" * 64, chunks=(chunk,))
            rank2 = dataclasses.replace(rank1, dcp_shard_rank=2)
            self.assertFalse(store.lookup(rank2, "e" * 64).is_hit)
            self.assertTrue(store.lookup(rank1, "e" * 64).is_hit)

    def test_invalidate_purges_corrupt_chunks_so_republish_succeeds(
        self,
    ) -> None:
        chunk = _chunk()
        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            identity = _identity()
            digest = "f" * 64
            store.commit(identity=identity, context_digest=digest, chunks=(chunk,))
            chunk_file = sorted((Path(directory) / "chunks").glob("*.spcc"))[0]
            payload = bytearray(chunk_file.read_bytes())
            payload[len(payload) // 2] ^= 0x40
            chunk_file.write_bytes(bytes(payload))
            self.assertFalse(store.lookup(identity, digest).is_hit)

            self.assertTrue(store.invalidate(identity, digest))
            self.assertFalse(chunk_file.exists())

            # republication of the identical content must now succeed
            store.commit(identity=identity, context_digest=digest, chunks=(chunk,))
            restored = store.lookup(identity, digest)
            self.assertTrue(restored.is_hit, restored.reason)
            self.assertEqual(store.restore(restored), (chunk,))

    def test_invalidate_keeps_healthy_shared_chunks(self) -> None:
        chunk = _chunk()
        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            identity = _identity()
            store.commit(identity=identity, context_digest="a" * 64, chunks=(chunk,))
            chunk_file = sorted((Path(directory) / "chunks").glob("*.spcc"))[0]
            self.assertTrue(store.invalidate(identity, "a" * 64))
            self.assertTrue(chunk_file.exists())

    def test_context_chunk_is_an_immutable_snapshot(self) -> None:
        chunk = _chunk()

        with self.assertRaises(TypeError):
            chunk.records[StateRecord.BOUNDARY_HIDDEN] = b"mutated"  # type: ignore[index]

    def test_empty_required_record_is_not_treated_as_complete(self) -> None:
        records = dict(_chunk().records)
        records[StateRecord.MTP_DRAFT_KV] = b""

        with self.assertRaises(IncompleteEntry):
            ContextChunk(
                logical_start=0,
                logical_end=256,
                records=records,
            )

    def test_only_the_final_chunk_may_be_partial(self) -> None:
        context_digest = hashlib.sha256(b"two-short-chunks").hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(Path(directory))

            with self.assertRaises(ValueError):
                store.commit(
                    identity=_identity(),
                    context_digest=context_digest,
                    chunks=[_chunk(0, 128), _chunk(128, 256)],
                )

    def test_corrupted_chunk_becomes_a_clean_cache_miss(self) -> None:
        context_digest = hashlib.sha256(b"corruption-probe").hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            store.commit(
                identity=_identity(),
                context_digest=context_digest,
                chunks=[_chunk()],
            )
            chunk_path = next((root / "chunks").glob("*.spcc"))
            encoded = bytearray(chunk_path.read_bytes())
            encoded[-1] ^= 0x01
            chunk_path.write_bytes(encoded)

            lookup = store.lookup(_identity(), context_digest)
            self.assertFalse(lookup.is_hit)
            self.assertEqual(lookup.reason, "corrupt")
            with self.assertRaises(ValueError):
                store.restore(lookup)

    def test_corruption_after_lookup_cannot_be_restored(self) -> None:
        context_digest = hashlib.sha256(b"post-lookup-corruption").hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            store.commit(
                identity=_identity(),
                context_digest=context_digest,
                chunks=[_chunk()],
            )
            lookup = store.lookup(_identity(), context_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)

            chunk_path = next((root / "chunks").glob("*.spcc"))
            with chunk_path.open("ab") as stream:
                stream.write(b"CORRUPT_TRAILER")

            self.assertIsNone(store.restore(lookup))

    def test_restore_rejects_chunk_range_that_disagrees_with_manifest(
        self,
    ) -> None:
        context_digest = hashlib.sha256(b"range-mismatch").hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            store.commit(
                identity=_identity(),
                context_digest=context_digest,
                chunks=[_chunk()],
            )
            mismatched = cache_manifest._encode_chunk(_chunk(256, 512))
            mismatched_digest = hashlib.sha256(mismatched).hexdigest()
            (root / "chunks" / f"{mismatched_digest}.spcc").write_bytes(mismatched)
            manifest_path = next((root / "manifests").rglob("*.json"))
            manifest = json.loads(manifest_path.read_bytes())
            manifest["chunks"][0]["sha256"] = mismatched_digest
            manifest["chunks"][0]["bytes"] = len(mismatched)
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                encoding="ascii",
            )

            lookup = store.lookup(
                _identity(),
                context_digest,
                verify_chunks=False,
                verify_chunk_metadata=True,
            )
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertIsNone(store.restore(lookup))

    def test_chunk_decoder_rejects_checksum_valid_trailing_bytes(self) -> None:
        context_digest = hashlib.sha256(b"claimed-trailer").hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            store.commit(
                identity=_identity(),
                context_digest=context_digest,
                chunks=[_chunk()],
            )
            chunk_path = next((root / "chunks").glob("*.spcc"))
            encoded = chunk_path.read_bytes() + b"CLAIMED_BUT_UNPARSED"
            digest = hashlib.sha256(encoded).hexdigest()
            (root / "chunks" / f"{digest}.spcc").write_bytes(encoded)

            manifest_path = next((root / "manifests").rglob("*.json"))
            manifest = json.loads(manifest_path.read_bytes())
            manifest["chunks"][0]["sha256"] = digest
            manifest["chunks"][0]["bytes"] = len(encoded)
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="ascii",
            )

            lookup = store.lookup(_identity(), context_digest)
            self.assertFalse(lookup.is_hit)
            self.assertEqual(lookup.reason, "corrupt")


if __name__ == "__main__":
    unittest.main()
