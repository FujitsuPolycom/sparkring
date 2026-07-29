from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
import unittest
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
        target_checkpoint="target-sha256",
        draft_checkpoint="draft-sha256",
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
