"""Tests for the read-only public model preflight."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent / "public-model-preflight.py"
SPEC = importlib.util.spec_from_file_location("public_model_preflight", SCRIPT)
assert SPEC and SPEC.loader
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def model_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "part-1.safetensors").write_bytes(b"weights-a")
    (root / "part-2.safetensors").write_bytes(b"weights-b")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "part-1.safetensors",
                    "layer.1": "part-2.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    draft = root / "mtp-draft"
    draft.mkdir()
    (draft / "config.json").write_text("{}", encoding="utf-8")
    (draft / "draft.safetensors").write_bytes(b"draft")
    return root


def test_complete_sharded_model_passes(tmp_path):
    report = preflight.inspect_model(model_fixture(tmp_path), "mtp-draft")
    assert report["passed"] is True
    assert report["unique_shards"] == 2
    assert report["weight_bytes"] == 18


def test_missing_referenced_shard_fails(tmp_path):
    root = model_fixture(tmp_path)
    (root / "part-2.safetensors").unlink()
    with pytest.raises(preflight.ModelPreflightError, match="missing"):
        preflight.inspect_model(root, "mtp-draft")


def test_weight_map_cannot_escape_model_root(tmp_path):
    root = model_fixture(tmp_path)
    index = root / "model.safetensors.index.json"
    index.write_text(
        json.dumps({"weight_map": {"layer.0": "../escape.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(preflight.ModelPreflightError, match="escapes"):
        preflight.inspect_model(root, "mtp-draft")


def test_missing_draft_fails(tmp_path):
    root = model_fixture(tmp_path)
    for path in (root / "mtp-draft").iterdir():
        path.unlink()
    (root / "mtp-draft").rmdir()
    with pytest.raises(preflight.ModelPreflightError, match="draft config"):
        preflight.inspect_model(root, "mtp-draft")


def test_external_draft_path_passes(tmp_path):
    root = model_fixture(tmp_path)
    draft = root / "mtp-draft"
    report = preflight.inspect_model(
        root,
        draft_path=draft,
    )
    assert report["draft_path"] == str(draft.resolve())


def test_external_draft_path_must_exist(tmp_path):
    root = model_fixture(tmp_path)
    with pytest.raises(preflight.ModelPreflightError, match="draft root"):
        preflight.inspect_model(
            root,
            draft_path=tmp_path / "absent-draft",
        )


def test_pinned_index_and_draft_hashes_are_verified(tmp_path):
    root = model_fixture(tmp_path)
    draft = root / "mtp-draft"
    draft_index = draft / "model.safetensors.index.json"
    draft_weight = draft / "model-mtp.safetensors"
    draft_index.write_text('{"weight_map":{}}', encoding="utf-8")
    draft_weight.write_bytes(b"mtp")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    report = preflight.inspect_model(
        root,
        draft_path=draft,
        index_sha256=digest(root / "model.safetensors.index.json"),
        draft_config_sha256=digest(draft / "config.json"),
        draft_index_sha256=digest(draft_index),
        draft_weight_sha256=digest(draft_weight),
    )
    assert report["passed"] is True

    with pytest.raises(preflight.ModelPreflightError, match="sha256 mismatch"):
        preflight.inspect_model(
            root,
            draft_path=draft,
            index_sha256="0" * 64,
        )
