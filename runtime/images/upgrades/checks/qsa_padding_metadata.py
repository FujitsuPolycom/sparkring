# SPDX-License-Identifier: Apache-2.0
"""CPU checks of QSA metadata storage retained by captured GPU views."""

import __future__
import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS
from typing import cast

import pytest
import torch


@pytest.fixture
def build():
    root = Path(os.environ["SPARKRING_TEST_SOURCE_ROOT"])
    path = root / "vllm/models/qwen4_exp/nvidia/b12x_qsa.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen4ExpQSAMetadataBuilder"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "build"
    )
    namespace = {"torch": torch, "cast": cast, "Qwen4ExpQSAMetadata": NS}
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(
        compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    return namespace["build"]


def owner():
    return NS(
        _request_ids=torch.full((64,), 987, dtype=torch.int32),
        _capture_state_slot_ids=torch.arange(1, dtype=torch.int32),
        _capture_state_is_fresh=torch.ones(1, dtype=torch.bool),
        _capture_num_accepted_tokens=torch.ones(1, dtype=torch.int32),
        _capture_is_prefilling=torch.zeros(1, dtype=torch.bool),
    )


def metadata(live, actual=None):
    actual = live if actual is None else actual
    cache = []

    def mapping(buffer):
        if not cache:
            buffer[: max(live, actual)].fill_(0)
            cache.append(buffer[:actual])
        return cache[0]

    return NS(
        seq_lens=torch.tensor([live]),
        query_start_loc_cpu=torch.tensor([0, live]),
        num_actual_tokens=actual,
        max_query_len=live,
        query_start_loc=torch.tensor([0, live]),
        max_seq_len=live,
        block_table_tensor=torch.tensor([[0]]),
        slot_mapping=torch.arange(actual),
        causal=True,
        is_prefilling=torch.tensor([True]),
        token_to_req_indices=mapping,
    )


@pytest.mark.parametrize("live", [0, 1, 17, 18, 19, 25, 26, 27, 28])
def test_captured_view_marks_shorter_batch_tail_inactive(build, live):
    builder = owner()
    captured = build(builder, 0, metadata(28)).request_ids
    build(builder, 0, metadata(live))
    assert captured[:live].tolist() == [0] * live
    assert captured[live:].tolist() == [-1] * (28 - live)
    assert builder._request_ids[live:].eq(-1).all()


def test_shared_cached_mapping_survives_repeated_and_cross_builder_calls(build):
    first, second = owner(), owner()
    common = metadata(26, actual=28)
    captured = build(first, 0, common).request_ids
    again = build(first, 0, common).request_ids
    shared = build(second, 0, common).request_ids
    assert captured.data_ptr() == again.data_ptr() == shared.data_ptr()
    assert captured[:26].eq(0).all()
    assert captured[26:].eq(-1).all()
    assert first._request_ids[26:].eq(-1).all()
    assert second._request_ids[28:].eq(-1).all()
