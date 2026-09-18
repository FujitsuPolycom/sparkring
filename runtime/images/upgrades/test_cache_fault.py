"""Synthetic corruption preserves recovery and cannot target a serving cache."""

import pytest
import json

from . import cache_fault


def setup(tmp_path):
    (tmp_path / cache_fault.MARKER).write_text("owned-trial")
    chunk = tmp_path / "persistent/chunks/fixture.spcc"
    chunk.parent.mkdir(parents=True)
    chunk.write_bytes(bytes(range(256)))
    return chunk


def test_fault_requires_owned_root_and_stopped_workers(tmp_path):
    setup(tmp_path)
    with pytest.raises(ValueError, match="Stop both"):
        cache_fault.corrupt(tmp_path, "owned-trial", "persistent")
    with pytest.raises(ValueError, match="not owned"):
        cache_fault.corrupt(tmp_path, "different", "persistent", workers_stopped=True)


def test_corrupted_chunk_has_verified_original_backup(tmp_path):
    chunk = setup(tmp_path)
    before = chunk.read_bytes()
    rows = cache_fault.corrupt(
        tmp_path, "owned-trial", "persistent", workers_stopped=True
    )
    assert chunk.read_bytes() != before and len(rows) == 1
    assert (tmp_path / rows[0]["backup"]).read_bytes() == before
    cache_fault.repair(tmp_path, "owned-trial", workers_stopped=True)
    assert chunk.read_bytes() == before


def test_recomputation_replacement_is_not_overwritten(tmp_path):
    chunk = setup(tmp_path)
    cache_fault.corrupt(tmp_path, "owned-trial", "persistent", workers_stopped=True)
    chunk.write_bytes(b"recomputed" * 32)
    result = cache_fault.repair(tmp_path, "owned-trial", workers_stopped=True)
    assert result[0]["result"] == "replacement preserved"
    assert chunk.read_bytes() == b"recomputed" * 32


def test_persistent_path_cannot_escape_the_owned_root(tmp_path):
    setup(tmp_path)
    with pytest.raises(ValueError, match="contained"):
        cache_fault.inventory(tmp_path, "owned-trial", "../another-cache")


def test_repair_handles_interruption_before_corrupt_hash_was_journaled(tmp_path):
    chunk = setup(tmp_path)
    original = chunk.read_bytes()
    cache_fault.corrupt(tmp_path, "owned-trial", "persistent", workers_stopped=True)
    journal = tmp_path / "fault-backup/journal.json"
    value = json.loads(journal.read_text())
    del value["chunks"][0]["corrupt_sha256"]
    journal.write_text(json.dumps(value))
    cache_fault.repair(tmp_path, "owned-trial", workers_stopped=True)
    assert chunk.read_bytes() == original
