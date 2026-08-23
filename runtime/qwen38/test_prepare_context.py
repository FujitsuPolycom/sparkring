from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.qwen38 import prepare_context


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def test_pins_name_only_public_immutable_sources() -> None:
    pins = prepare_context.load_pins(HERE / "pins.json")
    assert pins["parent_image"]["reference"].endswith(
        "@sha256:5c36750138dc1447a17dafbb397674f167d3b44ce18d9160d769df114577b35d"
    )
    assert pins["companion"]["commit"] == (
        "b9e1031b80b6f3f64bfc75ae3922322f56954fd6"
    )
    assert pins["sources"]["vllm"]["commit"] == (
        "229effc810ee6b8112f661472f6aace4eb8c787d"
    )
    assert pins["sources"]["exllamav3"]["patched_tree"] == (
        "c0b055f5e651ae0c93ba9407d90e6f136aa778c3"
    )
    assert pins["sources"]["nccl"]["patched_tree"] == (
        "9e80bc2489864b4e6c6e2184af8797b07baa68f1"
    )
    assert pins["nccl"]["library_sha256"] == (
        "e69a8c240f45d10166bcd901d99db78bb63147adda66e586d8dd505c6d608b54"
    )


def test_tracked_nccl_patch_hashes_match_pins() -> None:
    pins = prepare_context.load_pins(HERE / "pins.json")
    for patch in pins["nccl"]["patches"]:
        path = ROOT / patch["path"]
        assert prepare_context.sha256_file(path) == patch["sha256"]


def test_sanitize_requirements_removes_only_local_runtime_entries() -> None:
    pins = prepare_context.load_pins(HERE / "pins.json")
    excluded = pins["python_packages"]["excluded_from_public_freeze"]
    text = "\n".join(["foo==1.0", *excluded, "bar==2.0", ""])
    assert prepare_context.sanitize_requirements(text, excluded) == (
        "foo==1.0\nbar==2.0\n"
    )


@pytest.mark.parametrize(
    "requirement",
    (
        "other @ file:///private/wheel.whl",
        "-e /unreviewed/source",
        "thing @ git+https://example.invalid/thing.git",
    ),
)
def test_sanitize_requirements_rejects_unreviewed_sources(requirement: str) -> None:
    pins = prepare_context.load_pins(HERE / "pins.json")
    excluded = pins["python_packages"]["excluded_from_public_freeze"]
    text = "\n".join([*excluded, requirement, ""])
    with pytest.raises(prepare_context.PrepareError, match="unapproved"):
        prepare_context.sanitize_requirements(text, excluded)


def test_requirements_pin_is_the_canonical_lf_file_hash() -> None:
    pins = json.loads((HERE / "pins.json").read_text(encoding="utf-8"))
    assert pins["companion"]["requirements_freeze_sha256"] == (
        "d773c781bcc1de6cf81a64f9fa6b2ab80535f77eea08c5aeb5b96c2ce4423ba8"
    )
