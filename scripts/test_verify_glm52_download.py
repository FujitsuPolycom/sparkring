from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "verify_glm52_download.py"
SPEC = importlib.util.spec_from_file_location("verify_glm52_download", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return digest(payload)


def fixture(tmp_path: Path) -> tuple[list[str], Path, Path]:
    model = tmp_path / "model"
    draft = tmp_path / "draft"
    model_config = b"model-config"
    model_shards = {
        "model-00001-of-00002.safetensors": b"a" * 20,
        "model-00002-of-00002.safetensors": b"b" * 30,
    }
    model_index = json.dumps(
        {
            "metadata": {"total_size": 45},
            "weight_map": {
                "tensor.0": "model-00001-of-00002.safetensors",
                "tensor.1": "model-00002-of-00002.safetensors",
            },
        },
        sort_keys=True,
    ).encode()
    draft_config = b"draft-config"
    draft_weight = b"w" * 40
    draft_scales = b"s" * 10
    draft_index = json.dumps(
        {
            "metadata": {"total_size": 45},
            "weight_map": {
                "draft.weight": "model-mtp.safetensors",
                "draft.scale": "model-mtp-inputscales.safetensors",
            },
        },
        sort_keys=True,
    ).encode()

    for name, payload in model_shards.items():
        write(model / name, payload)
    pins = {
        "model_config": write(model / "config.json", model_config),
        "model_index": write(model / "model.safetensors.index.json", model_index),
        "draft_config": write(draft / "config.json", draft_config),
        "draft_index": write(draft / "model.safetensors.index.json", draft_index),
        "draft_weight": write(draft / "model-mtp.safetensors", draft_weight),
        "draft_scales": write(
            draft / "model-mtp-inputscales.safetensors", draft_scales
        ),
    }
    argv = [
        "--model-dir",
        str(model),
        "--draft-dir",
        str(draft),
        "--model-repository",
        "example/model",
        "--model-revision",
        "1" * 40,
        "--model-config-sha256",
        pins["model_config"],
        "--model-index-sha256",
        pins["model_index"],
        "--model-shards",
        "2",
        "--draft-repository",
        "example/draft",
        "--draft-revision",
        "2" * 40,
        "--draft-config-sha256",
        pins["draft_config"],
        "--draft-index-sha256",
        pins["draft_index"],
        "--draft-weight-sha256",
        pins["draft_weight"],
        "--draft-inputscales-sha256",
        pins["draft_scales"],
    ]
    return argv, model, draft


def test_adopts_complete_unmarked_payload(tmp_path: Path):
    argv, model, draft = fixture(tmp_path)
    assert verifier.main([*argv, "--adopt"]) == 0
    assert "repository=example/model" in (
        model / ".sparkring-model.txt"
    ).read_text()
    assert "inputscales_sha256=" in (
        draft / ".sparkring-model.txt"
    ).read_text()
    assert verifier.main(argv) == 0


def test_incomplete_unmarked_payload_is_resumable(tmp_path: Path):
    argv, model, _ = fixture(tmp_path)
    (model / "model-00002-of-00002.safetensors").unlink()
    assert verifier.main([*argv, "--adopt"]) == verifier.INCOMPLETE
    assert not (model / ".sparkring-model.txt").exists()


def test_marked_incomplete_payload_fails_closed(tmp_path: Path):
    argv, model, _ = fixture(tmp_path)
    assert verifier.main([*argv, "--adopt"]) == 0
    (model / "model-00002-of-00002.safetensors").unlink()
    assert verifier.main([*argv, "--adopt"]) == verifier.INTEGRITY_FAILURE


def test_hash_mismatch_fails_closed_without_markers(tmp_path: Path):
    argv, model, _ = fixture(tmp_path)
    (model / "config.json").write_bytes(b"wrong")
    assert verifier.main([*argv, "--adopt"]) == verifier.INTEGRITY_FAILURE
    assert not (model / ".sparkring-model.txt").exists()


def test_truncated_aggregate_size_fails_closed(tmp_path: Path):
    argv, model, _ = fixture(tmp_path)
    (model / "model-00001-of-00002.safetensors").write_bytes(b"a")
    assert verifier.main([*argv, "--adopt"]) == verifier.INTEGRITY_FAILURE


def test_wrong_existing_marker_is_not_replaced(tmp_path: Path):
    argv, model, _ = fixture(tmp_path)
    marker = model / ".sparkring-model.txt"
    marker.write_text("repository=wrong/model\n", encoding="utf-8")
    assert verifier.main([*argv, "--adopt"]) == verifier.INTEGRITY_FAILURE
    assert marker.read_text(encoding="utf-8") == "repository=wrong/model\n"


def test_verified_legacy_draft_marker_is_upgraded(tmp_path: Path):
    argv, _, draft = fixture(tmp_path)
    assert verifier.main([*argv, "--adopt"]) == 0
    marker = draft / ".sparkring-model.txt"
    marker.write_text(
        marker.read_text(encoding="utf-8").replace(
            next(
                line
                for line in marker.read_text(encoding="utf-8").splitlines(True)
                if line.startswith("inputscales_sha256=")
            ),
            "",
        ),
        encoding="utf-8",
    )
    assert verifier.main([*argv, "--adopt"]) == 0
    assert "inputscales_sha256=" in marker.read_text(encoding="utf-8")


def test_unsafe_index_shard_path_is_rejected(tmp_path: Path):
    root = tmp_path / "payload"
    index = root / "model.safetensors.index.json"
    index.parent.mkdir()
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {"tensor": "../escape.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(verifier.IntegrityFailure, match="unsafe shard"):
        verifier.verify_indexed_payload(root, index)
