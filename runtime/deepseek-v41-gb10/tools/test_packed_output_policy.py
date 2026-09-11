"""Synthetic packed-output ownership checks; no model or GPU inputs."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_packer():
    spec = importlib.util.spec_from_file_location(
        "synthetic_engram_packer", Path(__file__).with_name("pack_engram_rows.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(module, monkeypatch, model, output):
    monkeypatch.setattr(
        module.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            model_dir=str(model),
            out_dir=str(output),
            tp=1,
            rank=0,
            balanced=True,
            contiguous=False,
            chunk_rows=2,
        ),
    )


@pytest.mark.parametrize("kind", ["binary", "manifest", "partial", "empty"])
def test_existing_destination_is_preserved_before_source_reads(
    tmp_path, monkeypatch, kind
):
    module = load_packer()
    output = tmp_path / "packed"
    output.mkdir()
    sentinel = output / kind
    if kind != "empty":
        sentinel.write_bytes(b"keep")
    arguments(module, monkeypatch, tmp_path / "absent-source", output)
    with pytest.raises(FileExistsError, match="fresh"):
        module.main()
    if kind != "empty":
        assert sentinel.read_bytes() == b"keep"


def test_fresh_destination_packs_tiny_synthetic_rows(tmp_path, monkeypatch):
    module = load_packer()
    source = tmp_path / "synthetic"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"engram_max_ngram_size": 2, "engram_n_heads": 1})
    )
    weights, scales = source / "weights.fake", source / "scales.fake"
    weights.write_bytes(bytes([1, 2, 3, 4]))
    scales.write_bytes(bytes([5, 6]))
    output = tmp_path / "fresh-checkpoint-packed"
    arguments(module, monkeypatch, source, output)
    monkeypatch.setattr(module, "head_sizes_per_layer", lambda cfg: ([1], [[2]]))
    monkeypatch.setattr(
        module,
        "tensor_loc",
        lambda root, name: (
            (str(weights), 0, (2, 2), "F8_E4M3")
            if name.endswith("weight")
            else (str(scales), 0, (2, 1), "F8_E8M0")
        ),
    )

    def pread(fd, count, offset):
        os.lseek(fd, offset, os.SEEK_SET)
        return os.read(fd, count)

    def pwrite(fd, data, offset):
        os.lseek(fd, offset, os.SEEK_SET)
        return os.write(fd, data)

    monkeypatch.setattr(module.os, "pread", pread, raising=False)
    monkeypatch.setattr(module.os, "pwrite", pwrite, raising=False)
    module.main()
    assert (output / "engram-l1-packed.bin").read_bytes() == bytes([1, 2, 5, 3, 4, 6])
    assert json.loads((output / "engram-l1-packed.bin.json").read_text())["ranges"] == [
        [0, 2]
    ]
