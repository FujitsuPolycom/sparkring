"""Real filesystem verification around checkpoint transfer receipts."""
import hashlib
import io
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import installer_host as host


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    contents = {"config.json": b"{}", "model.safetensors.index.json": b'{"weight_map":{"w":"part.safetensors"}}',
                "part.safetensors": b"checkpoint data"}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}
    manifest = {"repository": "fixture/model", "revision": "a" * 40, "files": hashes,
                "sizes": {name: len(value) for name, value in contents.items()}}
    lock = {"id": "b" * 64, "backend": "compose", "selection": {"profile": "fixture", "model_repository": manifest["repository"], "model_revision": manifest["revision"]}}
    row = {"model": str(tmp_path / "model"), "cache": str(tmp_path / "cache")}
    state = tmp_path / "state"
    monkeypatch.setattr(host.installer, "checkpoint_contract", lambda _: {"config_sha256": hashes["config.json"], "index_sha256": hashes["model.safetensors.index.json"]})
    import shutil
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=1024**4))
    def call(operation, document=manifest):
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(document)))
        return host.transfer_model(operation, lock, row, state)
    return contents, row, state, call


def test_receiver_publishes_receipt_only_after_full_checksum_verification(checkpoint):
    from pathlib import Path
    contents, row, state, call = checkpoint
    call("model-transfer-prepare")
    for name, value in contents.items():
        (Path(row["model"]) / name).write_bytes(value)
    call("model-transfer-complete")
    assert json.loads((state / "model.json").read_text())["origin"] == "verified-fabric-copy"
    assert Path(row["cache"]).is_dir()
    assert call("model-present")["present"] is True


def test_interrupted_or_corrupt_copy_retains_owner_and_rejects_readiness(checkpoint):
    from pathlib import Path
    contents, row, state, call = checkpoint
    call("model-transfer-prepare")
    for name, value in contents.items():
        (Path(row["model"]) / name).write_bytes(value)
    (Path(row["model"]) / "part.safetensors").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="differs"):
        call("model-transfer-complete")
    assert not (state / "model.json").exists()
    assert call("model-present")["present"] is False
    # The same transfer can retry without clearing files or deleting receipts.
    call("model-transfer-prepare")
    (Path(row["model"]) / "part.safetensors").write_bytes(contents["part.safetensors"])
    call("model-transfer-complete")


def test_unowned_nonempty_destination_is_never_overwritten(checkpoint):
    from pathlib import Path
    _, row, state, call = checkpoint
    Path(row["model"]).mkdir()
    (Path(row["model"]) / "personal.txt").write_text("keep me")
    with pytest.raises(ValueError, match="no owned transfer"):
        call("model-transfer-prepare")
    assert not state.exists()


def test_manifest_cannot_escape_the_model_directory(checkpoint):
    _, _, state, call = checkpoint
    with pytest.raises(ValueError, match="Unsafe"):
        call("model-transfer-prepare", {"repository": "fixture/model", "revision": "a" * 40,
                                      "files": {"../escape": "b" * 64}, "sizes": {"../escape": 1}})
    assert not state.exists()
