"""Offline tests for the immutable R7 checkpoint downloader."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

import download_exl3_r7 as r7


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_inventory_rejects_revision_drift():
    api = SimpleNamespace(model_info=lambda *a, **k: SimpleNamespace(sha="main", siblings=[]))
    with pytest.raises(RuntimeError, match="resolved to main"):
        r7.inventory(api)


def test_index_rejects_stale_qualified_total(tmp_path, monkeypatch):
    monkeypatch.setattr(r7, "EXPECTED_WEIGHT_COUNT", 1)
    monkeypatch.setattr(r7, "EXPECTED_SHARD_COUNT", 1)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": r7.STALE_INDEX_TOTAL_SIZE}, "weight_map": {"x": "model-sharedbf16.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="stale payload total"):
        r7.indexed_shards(tmp_path)


def test_index_accepts_only_exact_runtime_closure(tmp_path, monkeypatch):
    monkeypatch.setattr(r7, "EXPECTED_WEIGHT_COUNT", 1)
    monkeypatch.setattr(r7, "EXPECTED_SHARD_COUNT", 1)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": r7.EXPECTED_INDEX_TOTAL_SIZE}, "weight_map": {"x": "model-sharedbf16.safetensors"}}),
        encoding="utf-8",
    )
    assert r7.indexed_shards(tmp_path) == {"model-sharedbf16.safetensors"}


def test_verify_checks_every_indexed_shard(tmp_path, monkeypatch):
    metadata = b"index"
    shard = b"shared"
    (tmp_path / "model.safetensors.index.json").write_bytes(metadata)
    (tmp_path / "model-sharedbf16.safetensors").write_bytes(shard)
    monkeypatch.setattr(r7, "PINNED_FILES", {"model.safetensors.index.json": (len(metadata), _sha(metadata))})
    monkeypatch.setattr(r7, "indexed_shards", lambda _: {"model-sharedbf16.safetensors"})
    report = r7.verify(tmp_path, {"model-sharedbf16.safetensors": (len(shard), _sha(shard))})
    assert report["runtime_shard_count"] == 1
    (tmp_path / "model-sharedbf16.safetensors").write_bytes(b"damage")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        r7.verify(tmp_path, {"model-sharedbf16.safetensors": (len(shard), _sha(shard))})


def test_verify_fails_when_revision_has_no_digest_for_a_runtime_shard(tmp_path, monkeypatch):
    metadata = b"index"
    (tmp_path / "model.safetensors.index.json").write_bytes(metadata)
    monkeypatch.setattr(r7, "PINNED_FILES", {"model.safetensors.index.json": (len(metadata), _sha(metadata))})
    monkeypatch.setattr(r7, "indexed_shards", lambda _: {"missing.safetensors"})
    with pytest.raises(RuntimeError, match="lacks LFS SHA-256 metadata"):
        r7.verify(tmp_path, {})
