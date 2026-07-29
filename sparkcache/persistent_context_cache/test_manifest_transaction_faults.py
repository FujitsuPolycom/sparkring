from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from sparkcache.persistent_context_cache import cache_manifest
from sparkcache.persistent_context_cache.cache_manifest import (
    CacheIdentity,
    ContextChunk,
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


def _chunk(index: int) -> ContextChunk:
    start = index * 256
    end = start + 256
    suffix = index.to_bytes(2, "little")
    return ContextChunk(
        logical_start=start,
        logical_end=end,
        records={
            StateRecord.TARGET_CKV: b"target" + suffix,
            StateRecord.SPARSE_INDEXER: b"indexer" + suffix,
            StateRecord.MTP_DRAFT_KV: b"draft" + suffix,
            StateRecord.BOUNDARY_HIDDEN: b"hidden" + suffix,
            StateRecord.LOGICAL_POSITIONS: b"positions" + suffix,
        },
    )


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def test_crash_after_any_append_point_stays_invisible_and_retry_succeeds() -> None:
    """Only the final manifest is a visibility point, including after restart."""

    expected = tuple(_chunk(index) for index in range(3))
    for appended_count in range(4):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = _digest(f"append-crash-{appended_count}")
            transaction = ManifestStore(root).begin_context(
                identity=_identity(),
                context_digest=digest,
                span_tokens=768,
            )
            for chunk in expected[:appended_count]:
                transaction.append_chunk(chunk)

            # Simulate process loss: do not call abort on the old in-memory
            # transaction. A fresh store must not discover begin-only, middle,
            # or fully-appended-but-uncommitted state.
            restarted = ManifestStore(root)
            lookup = restarted.lookup(_identity(), digest)
            assert not lookup.is_hit
            assert lookup.reason == "absent"
            assert not tuple((root / "manifests").rglob(f"{digest}.json"))

            retry = restarted.begin_context(
                identity=_identity(),
                context_digest=digest,
                span_tokens=768,
            )
            for chunk in expected:
                retry.append_chunk(chunk)
            retry.commit_manifest()

            committed = restarted.lookup(_identity(), digest)
            assert committed.is_hit, committed.reason
            assert restarted.restore(committed) == expected


def test_crash_temporary_manifest_is_never_a_lookup_hit() -> None:
    """A power-loss remnant at the temporary path is not a held context."""

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        identity = _identity()
        digest = _digest("temporary-manifest")
        manifest_directory = root / "manifests" / identity.storage_key
        manifest_directory.mkdir(parents=True)
        temporary = manifest_directory / f".{digest}.json.writing-crash"
        temporary.write_bytes(b'{"truncated":')

        lookup = ManifestStore(root).lookup(identity, digest)

        assert not lookup.is_hit
        assert lookup.reason == "absent"
        assert temporary.exists()


def test_commit_fault_before_link_is_invisible_and_retryable() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        identity = _identity()
        digest = _digest("commit-before-link")
        store = ManifestStore(root)
        transaction = store.begin_context(
            identity=identity,
            context_digest=digest,
            span_tokens=256,
        )
        transaction.append_chunk(_chunk(0))
        original_link = cache_manifest.os.link

        def fail_manifest_link(source: Path, destination: Path) -> None:
            if Path(destination).suffix == ".json":
                raise OSError("simulated crash before manifest link")
            original_link(source, destination)

        with (
            mock.patch.object(cache_manifest.os, "link", fail_manifest_link),
            mock.patch.object(
                cache_manifest.uuid,
                "uuid4",
                return_value=mock.Mock(hex="deterministic"),
            ),
        ):
            try:
                transaction.commit_manifest()
            except OSError as error:
                assert "before manifest link" in str(error)
            else:  # pragma: no cover - proves the injected edge was reached
                raise AssertionError("manifest link fault did not fire")

        lookup = store.lookup(identity, digest)
        assert not lookup.is_hit
        assert lookup.reason == "absent"
        assert not tuple(root.rglob("*.writing-*"))

        retry = store.begin_context(
            identity=identity,
            context_digest=digest,
            span_tokens=256,
        )
        retry.append_chunk(_chunk(0))
        retry.commit_manifest()
        committed = store.lookup(identity, digest)
        assert committed.is_hit, committed.reason
        assert store.restore(committed) == (_chunk(0),)


def test_commit_fault_after_link_exposes_only_complete_entry_and_retries() -> None:
    """An uncertain durability barrier may be a hit, but never a partial hit."""

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        identity = _identity()
        digest = _digest("commit-after-link")
        store = ManifestStore(root)
        transaction = store.begin_context(
            identity=identity,
            context_digest=digest,
            span_tokens=256,
        )
        transaction.append_chunk(_chunk(0))
        manifest_directory = root / "manifests" / identity.storage_key
        manifest_directory.mkdir(parents=True)

        def fail_manifest_barrier(path: Path) -> None:
            if Path(path) == manifest_directory:
                raise OSError("simulated crash after manifest link")

        with mock.patch.object(
            cache_manifest,
            "_fsync_directory",
            fail_manifest_barrier,
        ):
            try:
                transaction.commit_manifest()
            except OSError as error:
                assert "after manifest link" in str(error)
            else:  # pragma: no cover - proves the injected edge was reached
                raise AssertionError("manifest durability fault did not fire")

        uncertain = store.lookup(identity, digest)
        assert uncertain.is_hit, uncertain.reason
        assert store.restore(uncertain) == (_chunk(0),)

        receipt = transaction.commit_manifest()
        committed = store.lookup(identity, digest)
        assert committed.is_hit, committed.reason
        assert committed.manifest_digest == receipt.manifest_digest
        assert store.restore(committed) == (_chunk(0),)


def test_macro_batch_link_fault_stays_invisible_and_retries_exactly() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        identity = _identity()
        digest = _digest("macro-batch-link-fault")
        chunks = tuple(_chunk(index) for index in range(3))
        store = ManifestStore(root)
        transaction = store.begin_context(
            identity=identity,
            context_digest=digest,
            span_tokens=768,
        )
        original_link = cache_manifest.os.link
        chunk_links = 0

        def fail_second_chunk_link(source: Path, destination: Path) -> None:
            nonlocal chunk_links
            if Path(destination).suffix == ".spcc":
                chunk_links += 1
                if chunk_links == 2:
                    raise OSError("simulated macro-batch link failure")
            original_link(source, destination)

        with (
            mock.patch.object(cache_manifest.os, "link", fail_second_chunk_link),
            pytest.raises(OSError, match="macro-batch link failure"),
        ):
            transaction.append_chunks(chunks)

        assert not store.lookup(identity, digest).is_hit
        assert not tuple((root / "manifests").rglob(f"{digest}.json"))

        transaction.append_chunks(chunks)
        transaction.commit_manifest()
        committed = store.lookup(identity, digest)
        assert committed.is_hit, committed.reason
        assert store.restore(committed) == chunks


def test_macro_batch_directory_barrier_fault_stays_invisible_and_retries() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        chunk_directory = root / "chunks"
        chunk_directory.mkdir()
        identity = _identity()
        digest = _digest("macro-batch-directory-fault")
        chunks = tuple(_chunk(index) for index in range(3))
        store = ManifestStore(root)
        transaction = store.begin_context(
            identity=identity,
            context_digest=digest,
            span_tokens=768,
        )

        def fail_chunk_barrier(path: Path) -> None:
            if Path(path) == chunk_directory:
                raise OSError("simulated macro-batch directory barrier failure")

        with (
            mock.patch.object(
                cache_manifest,
                "_fsync_directory",
                fail_chunk_barrier,
            ),
            pytest.raises(OSError, match="directory barrier failure"),
        ):
            transaction.append_chunks(chunks)

        assert len(tuple(chunk_directory.glob("*.spcc"))) == 3
        assert not store.lookup(identity, digest).is_hit
        assert not tuple((root / "manifests").rglob(f"{digest}.json"))

        transaction.append_chunks(chunks)
        transaction.commit_manifest()
        committed = store.lookup(identity, digest)
        assert committed.is_hit, committed.reason
        assert store.restore(committed) == chunks
