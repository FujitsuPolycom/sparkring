from __future__ import annotations

import ctypes
import hashlib
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from spark_context_cache_native_placement import RestoreState
from spark_context_cache_native_restore import (
    NativeRestoreError,
    execute_native_restore,
    plan_native_restore,
)
import spark_cache_native as native


def _encoded_chunk(payload: bytes, header: bytes = b"{}") -> bytes:
    return struct.pack("<8sII", b"SPCKV001", 1, len(header)) + header + payload


def _lookup_for(root: Path, chunks: list[tuple[bytes, int, int]]):
    descriptors = []
    for encoded, logical_start, logical_end in chunks:
        digest = hashlib.sha256(encoded).hexdigest()
        (root / "chunks").mkdir(parents=True, exist_ok=True)
        (root / "chunks" / f"{digest}.spcc").write_bytes(encoded)
        descriptors.append(
            {
                "sha256": digest,
                "bytes": len(encoded),
                "logical_start": logical_start,
                "logical_end": logical_end,
            }
        )
    return SimpleNamespace(
        is_hit=True,
        _manifest={
            "committed_tokens": chunks[-1][2],
            "chunks": descriptors,
        },
    )


class _FakeTransaction:
    def __init__(self, request_id: str, slots: tuple[int, ...], arena_bytes: int):
        self.request_id = request_id
        self.slots = slots
        self.state = RestoreState.PARKED
        self.can_resume = False
        self.needs_recompute = False
        self.submissions = []
        self._owners = [
            (ctypes.c_ubyte * arena_bytes)(),
            (ctypes.c_ubyte * arena_bytes)(),
        ]

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        if self.state is RestoreState.PARKED:
            self.state = RestoreState.ABORTED
            self.needs_recompute = True

    def acquire_arena(self, arena_index: int):
        owner = self._owners[arena_index]
        return native.ArenaView(
            ctypes.addressof(owner),
            0xB000 + arena_index,
            len(owner),
            arena_index,
            native.ARENA_MAPPED_HOST,
            1,
            0,
        )

    def parse_verified_chunk(self, **kwargs):
        descriptor = native.ChunkDescriptor()
        descriptor.arena_offset_bytes = kwargs["arena_offset_bytes"]
        descriptor.encoded_bytes = kwargs["encoded_bytes"]
        prefix = ctypes.string_at(
            kwargs["arena"].host_address + kwargs["arena_offset_bytes"], 16
        )
        _, _, header_bytes = struct.unpack("<8sII", prefix)
        descriptor.payload_offset_bytes = 16 + header_bytes
        descriptor.first_slot_index = kwargs["first_slot_index"]
        descriptor.row_count = kwargs["encoded_bytes"] and 2
        return descriptor

    def submit_direct_slab(self, **kwargs):
        self.submissions.append(kwargs)

    def finish(self):
        self.state = RestoreState.FINISHED
        self.can_resume = True
        stats = native.PlacementStats()
        stats.source_bytes = sum(item["arena_used_bytes"] for item in self.submissions)
        stats.restored_rows = len(self.slots)
        stats.slot_uploads = 1
        stats.destination_table_uploads = 1
        stats.slabs_submitted = len(self.submissions)
        stats.scatter_kernel_launches = len(self.submissions)
        return stats


class _FakeAdapter:
    def __init__(self, arena_bytes: int):
        self.arena_bytes = arena_bytes
        self.transactions: list[_FakeTransaction] = []

    def begin_parked_restore(self, request_id: str, slots):
        transaction = _FakeTransaction(request_id, tuple(slots), self.arena_bytes)
        self.transactions.append(transaction)
        return transaction


def test_plan_builds_bounded_payload_aligned_slabs(tmp_path):
    first = _encoded_chunk(b"a" * 80, header=b"x" * 13)
    second = _encoded_chunk(b"b" * 80, header=b"y" * 29)
    lookup = _lookup_for(tmp_path, [(first, 0, 8), (second, 8, 16)])

    slabs = plan_native_restore(
        lookup,
        cache_root=tmp_path,
        expected_span_tokens=16,
        dcp_degree=4,
        arena_bytes=180,
        payload_alignment=64,
    )

    assert len(slabs) == 2
    assert all(slab.arena_used_bytes <= 180 for slab in slabs)
    for slab in slabs:
        for chunk in slab.chunks:
            assert (chunk.arena_offset_bytes + chunk.payload_offset_bytes) % 64 == 0


def test_success_reads_directly_and_releases_only_after_finish(tmp_path):
    encoded = _encoded_chunk(b"authenticated payload" * 4)
    lookup = _lookup_for(tmp_path, [(encoded, 0, 8)])
    adapter = _FakeAdapter(arena_bytes=512)

    result = execute_native_restore(
        adapter=adapter,
        request_id="restore-1",
        lookup=lookup,
        cache_root=tmp_path,
        slots=(9, 3),
        expected_span_tokens=8,
        dcp_degree=4,
        dcp_rank=1,
        arena_bytes=512,
        required_data_record_mask=0b11,
        io_workers=2,
    )

    transaction = adapter.transactions[0]
    assert transaction.state is RestoreState.FINISHED
    assert transaction.can_resume is True
    assert transaction.needs_recompute is False
    assert len(transaction.submissions) == 1
    slab = transaction.submissions[0]
    assert slab["chunks"][0].encoded_bytes == len(encoded)
    assert result.placement_stats.restored_rows == 2
    assert result.verified_chunks == 1
    assert result.verified_encoded_bytes == len(encoded)


def test_outer_hash_corruption_aborts_without_releasing_request(tmp_path):
    encoded = _encoded_chunk(b"before")
    lookup = _lookup_for(tmp_path, [(encoded, 0, 8)])
    descriptor = lookup._manifest["chunks"][0]
    path = tmp_path / "chunks" / f"{descriptor['sha256']}.spcc"
    path.write_bytes(_encoded_chunk(b"after!"))
    adapter = _FakeAdapter(arena_bytes=512)

    with pytest.raises(NativeRestoreError, match="SHA-256 mismatch"):
        execute_native_restore(
            adapter=adapter,
            request_id="restore-corrupt",
            lookup=lookup,
            cache_root=tmp_path,
            slots=(0, 1),
            expected_span_tokens=8,
            dcp_degree=4,
            dcp_rank=0,
            arena_bytes=512,
            required_data_record_mask=0b11,
        )

    transaction = adapter.transactions[0]
    assert transaction.state is RestoreState.ABORTED
    assert transaction.can_resume is False
    assert transaction.needs_recompute is True
    assert transaction.submissions == []


def test_manifest_length_change_aborts_before_native_submit(tmp_path):
    encoded = _encoded_chunk(b"unchanged")
    lookup = _lookup_for(tmp_path, [(encoded, 0, 8)])
    descriptor = lookup._manifest["chunks"][0]
    path = tmp_path / "chunks" / f"{descriptor['sha256']}.spcc"
    path.write_bytes(encoded + b"x")
    adapter = _FakeAdapter(arena_bytes=512)

    with pytest.raises(NativeRestoreError, match="length changed"):
        execute_native_restore(
            adapter=adapter,
            request_id="restore-length",
            lookup=lookup,
            cache_root=tmp_path,
            slots=(0, 1),
            expected_span_tokens=8,
            dcp_degree=4,
            dcp_rank=0,
            arena_bytes=512,
            required_data_record_mask=0b11,
        )

    # Length validation happens while planning, before the native transaction
    # owns any destination memory.
    assert adapter.transactions == []
