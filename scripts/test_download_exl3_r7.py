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


def _plant_checkpoint(root, monkeypatch, *, shards=None, config=b"cfg", index=b"idx"):
    """Create a directory the locator should accept, and pin it to those bytes."""
    monkeypatch.setitem(r7.PINNED_FILES, "config.json", (len(config), _sha(config)))
    monkeypatch.setitem(r7.PINNED_FILES, r7.INDEX_NAME, (len(index), _sha(index)))
    monkeypatch.setattr(r7, "EXPECTED_SHARD_COUNT", 3)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_bytes(config)
    (root / r7.INDEX_NAME).write_bytes(index)
    for number in range(3 if shards is None else shards):
        (root / f"shard-{number}.safetensors").write_bytes(b"")
    return root


def test_locate_finds_checkpoint_under_any_directory_name(tmp_path, monkeypatch):
    """The directory is named for its repository, not for a mount point."""
    planted = _plant_checkpoint(
        tmp_path / "home" / "someone" / "models" / "GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78",
        monkeypatch,
    )

    report = r7.locate([tmp_path])

    assert report["status"] == "pass"
    assert report["found"] == [str(planted.resolve())]


def test_locate_rejects_a_directory_whose_config_differs(tmp_path, monkeypatch):
    """A different quantization of the same model must not be accepted."""
    _plant_checkpoint(tmp_path / "models" / "other", monkeypatch)
    (tmp_path / "models" / "other" / "config.json").write_bytes(b"different")

    report = r7.locate([tmp_path])

    assert report["status"] == "absent"
    assert report["found"] == []


def test_locate_rejects_a_partial_shard_set(tmp_path, monkeypatch):
    _plant_checkpoint(tmp_path / "models" / "partial", monkeypatch, shards=2)

    assert r7.locate([tmp_path])["status"] == "absent"


def test_locate_reports_absent_for_a_missing_root(tmp_path):
    report = r7.locate([tmp_path / "nowhere"])

    assert report["status"] == "absent"
    assert report["searched_roots"] == [str(tmp_path / "nowhere")]


def test_candidate_directories_stops_at_the_depth_limit(tmp_path):
    deep = tmp_path / "a" / "b" / "c" / "d" / "e"
    deep.mkdir(parents=True)
    (deep / r7.INDEX_NAME).write_bytes(b"idx")

    assert list(r7.candidate_directories([tmp_path], max_depth=2)) == []
    assert deep in list(r7.candidate_directories([tmp_path], max_depth=9))


def test_candidate_directories_skips_dotted_directories(tmp_path):
    hidden = tmp_path / ".cache" / "model"
    hidden.mkdir(parents=True)
    (hidden / r7.INDEX_NAME).write_bytes(b"idx")

    assert list(r7.candidate_directories([tmp_path])) == []
