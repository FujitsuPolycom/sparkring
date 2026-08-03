"""Offline tests for resumable EXL3 model adoption and verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import download_exl3


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture(path: Path) -> dict:
    shard_a = b"aaa"
    shard_b = b"bbbb"
    tokenizer = b'{"model":{"type":"BPE"}}'
    index = {
        "weight_map": {
            "layer.0": "model-00001-of-00002.safetensors",
            "layer.1": "model-00002-of-00002.safetensors",
        }
    }
    config = b'{"model_type":"glm4_moe"}'
    tier = b'{"tiers":[3,4]}'
    index_bytes = json.dumps(index).encode()
    manifest = (
        f"{_sha(config)}  config.json\n"
        f"{_sha(index_bytes)}  model.safetensors.index.json\n"
        f"{_sha(shard_a)}  model-00001-of-00002.safetensors\n"
        f"{_sha(shard_b)}  model-00002-of-00002.safetensors\n"
        f"{_sha(tier)}  tier_bitmap.json\n"
        f"{_sha(tokenizer)}  tokenizer.json\n"
    ).encode()
    path.mkdir()
    (path / "config.json").write_bytes(config)
    (path / "model.safetensors.index.json").write_bytes(index_bytes)
    (path / "tier_bitmap.json").write_bytes(tier)
    (path / "MANIFEST.sha256").write_bytes(manifest)
    (path / "tokenizer.json").write_bytes(tokenizer)
    (path / "model-00001-of-00002.safetensors").write_bytes(shard_a)
    (path / "model-00002-of-00002.safetensors").write_bytes(shard_b)
    return {
        "repository": "example/model",
        "revision": "0" * 40,
        "config_sha256": _sha(config),
        "index_sha256": _sha(index_bytes),
        "tier_bitmap_sha256": _sha(tier),
        "manifest_sha256": _sha(manifest),
        "shard_count": 2,
        "weight_bytes": len(shard_a) + len(shard_b),
        "repository_bytes": 100,
    }


def test_exact_existing_model_is_adopted_without_downloading(tmp_path, monkeypatch, capsys):
    model_path = tmp_path / "model"
    model = _fixture(model_path)
    monkeypatch.setattr(download_exl3, "require_capacity", lambda *_: pytest.fail("capacity should not run"))
    download_exl3.download(model_path, model)
    assert "download skipped" in capsys.readouterr().out


def test_verifier_rejects_missing_or_changed_payload(tmp_path):
    model_path = tmp_path / "model"
    model = _fixture(model_path)
    assert download_exl3.verify(model_path, model)["status"] == "pass"
    (model_path / "tier_bitmap.json").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="tier_bitmap.json hash mismatch"):
        download_exl3.verify(model_path, model)


def test_verifier_rejects_same_size_shard_corruption(tmp_path):
    model_path = tmp_path / "model"
    model = _fixture(model_path)
    (model_path / "model-00001-of-00002.safetensors").write_bytes(b"bbb")
    with pytest.raises(RuntimeError, match="model shard hash mismatch"):
        download_exl3.verify(model_path, model)


def test_verifier_rejects_corruption_in_any_manifest_entry(tmp_path):
    model_path = tmp_path / "model"
    model = _fixture(model_path)
    (model_path / "tokenizer.json").write_bytes(b'{"model":{"type":"BAD"}}')
    with pytest.raises(RuntimeError, match="model file hash mismatch for tokenizer.json"):
        download_exl3.verify(model_path, model)


def test_verifier_rejects_unmanifested_runtime_files_but_ignores_download_cache(tmp_path):
    model_path = tmp_path / "model"
    model = _fixture(model_path)
    cache_file = model_path / ".cache" / "huggingface" / "download.lock"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"cache")
    assert download_exl3.verify(model_path, model)["status"] == "pass"

    (model_path / "unmanifested.json").write_bytes(b"{}")
    with pytest.raises(RuntimeError, match="unmanifested model files: .*unmanifested.json"):
        download_exl3.verify(model_path, model)


def test_capacity_prices_only_missing_bytes_plus_headroom(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "partial").write_bytes(b"x" * 60)
    model = {"repository_bytes": 100}

    class Usage:
        free = download_exl3.HEADROOM_BYTES + 39

    monkeypatch.setattr(download_exl3.shutil, "disk_usage", lambda _: Usage())
    with pytest.raises(RuntimeError, match="insufficient model disk space"):
        download_exl3.require_capacity(model_path, model)
    Usage.free += 1
    download_exl3.require_capacity(model_path, model)


def test_download_fetches_only_manifest_owned_files(tmp_path):
    source = tmp_path / "source"
    model = _fixture(source)
    destination = tmp_path / "destination"
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        destination.mkdir(exist_ok=True)
        for name in kwargs["allow_patterns"]:
            source_file = source / name
            if source_file.is_file():
                (destination / name).write_bytes(source_file.read_bytes())

    download_exl3.download_manifested_snapshot(
        destination, model, fake_snapshot_download
    )
    assert calls[0]["allow_patterns"] == ["MANIFEST.sha256"]
    assert calls[1]["allow_patterns"] == [
        "MANIFEST.sha256",
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "tier_bitmap.json",
        "tokenizer.json",
    ]
    assert "patch_exl3_mixk.py" not in calls[1]["allow_patterns"]
