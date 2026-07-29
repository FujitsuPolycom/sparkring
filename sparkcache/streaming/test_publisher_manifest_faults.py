from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from sparkcache.persistent_context_cache import cache_manifest
from sparkcache.persistent_context_cache.cache_manifest import ManifestStore
from sparkcache.streaming.publisher import (
    Glm52ReadyViewTranslator,
    JournalState,
    ManifestSnapshotJournalWriter,
)
from sparkcache.streaming.test_publisher import (
    _batch,
    _identity,
    _ring,
    _submit,
)


def test_real_publisher_commit_fault_never_advertises_partial_context(
    tmp_path: Path,
) -> None:
    """Exercise the async journal writer and real manifest store together."""

    ring, _backend = _ring()
    identity = _identity()
    digest = "d" * 64
    store = ManifestStore(tmp_path)
    writer = ManifestSnapshotJournalWriter(
        store=store,
        identity=identity,
        translator=Glm52ReadyViewTranslator.for_ring(ring, dcp_rank=0),
    )
    transaction = writer.begin_context(
        request_id="publisher-fault",
        context_digest=digest,
        span_tokens=256,
    )
    ticket = _submit(
        ring,
        context_sequence=1,
        logical_start=0,
        rows=64,
    )
    view = ring.claim(ticket)
    assert view is not None
    completion = transaction.submit_ready(
        _batch(
            request_id="publisher-fault",
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

    assert transaction.state is JournalState.OPEN
    assert transaction.appended_tokens == 256
    assert not store.lookup(identity, digest).is_hit
    original_link = cache_manifest.os.link

    def fail_manifest_link(source: Path, destination: Path) -> None:
        if Path(destination).suffix == ".json":
            raise OSError("simulated manifest publication failure")
        original_link(source, destination)

    with (
        mock.patch.object(cache_manifest.os, "link", fail_manifest_link),
        pytest.raises(OSError, match="manifest publication failure"),
    ):
        transaction.commit_manifest()

    assert transaction.state is JournalState.OPEN
    assert not store.lookup(identity, digest).is_hit

    receipt = transaction.commit_manifest()
    assert receipt.committed_tokens == 256
    assert transaction.state is JournalState.COMMITTED
    lookup = store.lookup(identity, digest)
    assert lookup.is_hit, lookup.reason
    restored = store.restore(lookup)
    assert restored is not None
    assert len(restored) == 1

    writer.shutdown()
    ring.shutdown()
