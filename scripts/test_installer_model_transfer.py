"""Checkpoint transfers between Sparks against real files: prepare, rsync staging, completion and receipts."""
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from scripts.test_installer_adopt import (REPOSITORY, REVISION, SHARDS, changes, contents, environment, journal,
                                          plain_folder, required, state_directory, tree_state, weights, write)

linux = pytest.mark.skipif(not sys.platform.startswith("linux"),
                           reason="SparkRing checkpoint directories use Linux hard links, /proc/self/fd and flock")


def manifest(data, names=None):
    names = required(data) if names is None else names
    return {"repository": REPOSITORY, "revision": REVISION,
            "files": {name: hashlib.sha256(data[name]).hexdigest() for name in names},
            "sizes": {name: len(data[name]) for name in names}}


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    return environment(tmp_path, monkeypatch)


def deliver(prepared, data, names, changed=None):
    """Write ``names`` where rsync delivers them: the prepared ``receive/rsync`` staging directory."""
    for name in names:
        write(Path(prepared["rsync"]) / name, (changed or {}).get(name, data[name]))


@linux
def test_receiver_publishes_receipt_only_after_full_checksum_verification(checkpoint):
    data, document = checkpoint.data, manifest(checkpoint.data)
    prepared = checkpoint.call("model-transfer-prepare", document)
    assert prepared["needed"] == required(data)
    assert prepared["rsync"] == str(state_directory(checkpoint.model) / "receive/rsync")
    assert os.listdir(prepared["rsync"]) == [] and os.listdir(checkpoint.model) == []
    deliver(prepared, data, required(data))
    result = checkpoint.call("model-transfer-complete", document)
    assert result == {"ok": True, "complete": True, "missing": []}
    receipt = json.loads((checkpoint.state / "model.json").read_text())
    assert receipt["origin"] == "verified-fabric-copy" and sorted(receipt["files"]) == required(data)
    assert {entry["origin"] for entry in journal(checkpoint.model).values()} == {"rsync"}
    assert Path(checkpoint.row["cache"]).is_dir()
    assert not (state_directory(checkpoint.model) / "receive").exists()
    for name in required(data):
        assert (checkpoint.model / name).read_bytes() == data[name] and os.lstat(checkpoint.model / name).st_nlink == 1


@linux
def test_interrupted_or_corrupt_copy_retains_owner_and_rejects_readiness(checkpoint):
    data, document = checkpoint.data, manifest(checkpoint.data)
    prepared = checkpoint.call("model-transfer-prepare", document)
    deliver(prepared, data, required(data), changed={SHARDS[1]: data[SHARDS[1]].upper()})
    with pytest.raises(ValueError, match=rf"differs from its verified source \({SHARDS[1]} not placed; rsync "
                                         rf"delivered different bytes for {SHARDS[1]}\)"):
        checkpoint.call("model-transfer-complete", document)
    assert not (checkpoint.state / "model.json").exists()
    assert not (checkpoint.model / SHARDS[1]).exists() and SHARDS[1] not in journal(checkpoint.model)
    # The same transfer resumes: only the file that was not placed is needed again.
    prepared = checkpoint.call("model-transfer-prepare", document)
    assert prepared["needed"] == [SHARDS[1]] and os.listdir(prepared["rsync"]) == []
    deliver(prepared, data, [SHARDS[1]])
    assert checkpoint.call("model-transfer-complete", document)["complete"]
    assert json.loads((checkpoint.state / "model.json").read_text())["origin"] == "verified-fabric-copy"


@linux
def test_pooled_subset_is_placed_without_completing_the_directory(checkpoint):
    data = checkpoint.data
    document = manifest(data, weights(data))
    prepared = checkpoint.call("model-transfer-prepare", document)
    assert prepared["needed"] == weights(data)
    deliver(prepared, data, weights(data))
    result = checkpoint.call("model-transfer-complete", document)
    assert result["ok"] and not result["complete"]
    assert result["missing"] == sorted(set(required(data)) - set(weights(data)))
    assert not (checkpoint.state / "model.json").exists()
    assert sorted(journal(checkpoint.model)) == weights(data)


@linux
def test_unowned_nonempty_destination_is_never_overwritten(checkpoint):
    checkpoint.model.mkdir(parents=True)
    (checkpoint.model / "personal.txt").write_text("keep me")
    with pytest.raises(ValueError, match="is not empty and was not created by SparkRing"):
        checkpoint.call("model-transfer-prepare", manifest(checkpoint.data))
    assert os.listdir(checkpoint.model) == ["personal.txt"]
    assert not state_directory(checkpoint.model).exists()


@pytest.mark.parametrize("name", ["../escape", "/etc/passwd", ".cache/huggingface/x", "a\\b"])
def test_manifest_cannot_escape_the_model_directory(checkpoint, name):
    document = {"repository": REPOSITORY, "revision": REVISION, "files": {name: "b" * 64}, "sizes": {name: 1}}
    with pytest.raises(ValueError, match="Unsafe"):
        checkpoint.call("model-transfer-prepare", document)
    assert not checkpoint.model.exists() and not state_directory(checkpoint.model).exists()


def test_manifest_names_only_pinned_files_with_their_pins(checkpoint):
    data = checkpoint.data
    for document in (manifest(data, ["README.md"]),
                     {**manifest(data), "files": {**manifest(data)["files"], "config.json": "0" * 64}},
                     {**manifest(data), "revision": "d" * 40}):
        with pytest.raises(ValueError, match="not a required file with its pinned hash|identity differs"):
            checkpoint.call("model-transfer-prepare", document)
    assert not checkpoint.model.exists()


def test_transfer_never_targets_a_reused_copy(tmp_path, monkeypatch):
    data = contents()
    folder = plain_folder(tmp_path / "models/qwen", data, weights(data))
    env = environment(tmp_path, monkeypatch, reuse=True, model=folder)
    before = tree_state(folder)
    for operation in ("model-transfer-prepare", "model-transfer-complete"):
        with pytest.raises(ValueError, match=r"serves .* in place; SparkRing never writes into a copy it did not create"):
            env.call(operation, manifest(data))
    assert changes(before, tree_state(folder)) == {}
    assert not state_directory(folder).exists() and not (env.state / "model.json").exists()
