"""Offline tests for hash-bound R7 runtime compatibility edits."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "r7_bake_runtime_artifacts", HERE / "bake_runtime_artifacts.py"
)
assert SPEC is not None and SPEC.loader is not None
BAKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BAKER)


def test_replace_exact_is_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "input.py"
    path.write_bytes(b"before\n")
    BAKER.replace_exact(
        path,
        input_sha256=BAKER.hashlib.sha256(b"before\n").hexdigest(),
        output_sha256=BAKER.hashlib.sha256(b"after\n").hexdigest(),
        replacements=((b"before", b"after"),),
    )
    assert path.read_bytes() == b"after\n"

    with pytest.raises(BAKER.ArtifactError, match="input SHA-256 mismatch"):
        BAKER.replace_exact(
            path,
            input_sha256="0" * 64,
            output_sha256="0" * 64,
            replacements=((b"after", b"before"),),
        )


def test_replace_exact_checks_replacement_count(tmp_path: Path) -> None:
    path = tmp_path / "input.py"
    path.write_bytes(b"before before\n")
    with pytest.raises(BAKER.ArtifactError, match="expected 1.*found 2"):
        BAKER.replace_exact(
            path,
            input_sha256=BAKER.sha256(path),
            output_sha256="0" * 64,
            replacements=((b"before", b"after"),),
        )


def test_cudagraph_component_is_hash_bound() -> None:
    """The patch on disk matches its pin, and the contract names both states."""

    import hashlib

    patch = BAKER.CUDAGRAPH_PATCH
    assert patch.is_file()
    observed = hashlib.sha256(patch.read_bytes()).hexdigest()
    assert observed == BAKER.CUDAGRAPH_PATCH_SHA256
    assert BAKER.CUDAGRAPH_INPUT_SHA256 != BAKER.CUDAGRAPH_OUTPUT_SHA256
    for digest in (BAKER.CUDAGRAPH_INPUT_SHA256, BAKER.CUDAGRAPH_OUTPUT_SHA256):
        assert len(digest) == 64 and int(digest, 16) >= 0
    text = patch.read_text(encoding="utf-8")
    assert "VLLM_SPARK_SHARED_CAPTURE_STREAM" in text
    assert "vllm/v1/worker/gpu/cudagraph_utils.py" in text


def test_cudagraph_component_refuses_an_unexpected_tree(tmp_path) -> None:
    target = tmp_path / "vllm/v1/worker/gpu/cudagraph_utils.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"unrelated source\n")
    try:
        BAKER.bake_cudagraph(tmp_path)
    except BAKER.ArtifactError as error:
        assert "expected" in str(error)
    else:  # pragma: no cover - the refusal is the contract
        raise AssertionError("an unexpected tree must be refused")
