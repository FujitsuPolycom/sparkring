"""Exercise ported PLE metadata against the allocator's checkpoint geometry."""

import ast
from dataclasses import dataclass, replace
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


@dataclass
class Metadata:
    num_reqs: int
    num_decodes: int
    num_prefills: int
    graph_inputs: object = None
    checkpoint_columns: object = None
    checkpoint_offsets: object = None
    checkpoint_slots: object = None


class ShortBuilder:
    def __init__(self, spec, names, config, device):
        self.kv_cache_spec = spec

    def build(self, prefix, common, fast=False, **kwargs):
        return Metadata(
            common.num_reqs, common.num_decodes, common.num_reqs - common.num_decodes
        )

    def update_block_table(self, metadata, table, slots):
        return metadata


class GraphInputs:
    def __init__(self, seqs, tokens, device):
        self.num_tokens = torch.zeros(1, dtype=torch.int32)
        self.query_start_loc = None

    def stage(self, metadata, starts):
        self.query_start_loc = starts


def builder(block, hash_block):
    root = Path(os.environ["SPARKRING_TEST_SOURCE_ROOT"])
    path = root / "vllm/models/qwen4_exp/nvidia/ple_attn.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "PLEAttentionMetadataBuilder"
    )
    helpers = ast.parse(
        (root / "vllm/v1/kv_cache_interface.py").read_text(encoding="utf-8")
    )
    functions = [
        n
        for n in helpers.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in {
            "get_mamba_prefill_checkpoint_position",
            "is_mamba_prefill_checkpoint_valid",
        }
    ]
    env = dict(
        torch=torch,
        replace=replace,
        PLEAttentionMetadata=Metadata,
        PLEGraphInputs=GraphInputs,
        ShortConvAttentionMetadataBuilder=ShortBuilder,
        AttentionCGSupport=NS(ALWAYS="always"),
        CommonAttentionMetadata=object,
        _B12X_NULL_STATE_SLOT=-1,
        NULL_BLOCK_ID=0,
        cdiv=lambda a, b: (a + b - 1) // b,
    )
    exec(
        compile(ast.Module(body=[*functions, cls], type_ignores=[]), str(path), "exec"),
        env,
    )
    spec = NS(
        block_size=block,
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=16,
    )
    config = NS(
        scheduler_config=NS(max_num_seqs=4, max_num_batched_tokens=8192),
        cache_config=NS(prefix_match_unit=hash_block),
        speculative_config=None,
    )
    return env["PLEAttentionMetadataBuilder"](spec, [], config, torch.device("cpu"))


@pytest.mark.parametrize(
    "start,end,block,hash_block,expected",
    [
        (5760, 8194, 1440, 1440, (1440, 4)),
        (32, 64, 16, 16, (16, 2)),
        (32, 67, 16, 16, (32, 3)),
        (33, 67, 16, 16, None),
        (64, 151, 64, 16, (80, 1)),
    ],
)
def test_metadata_selects_allocator_boundary(start, end, block, hash_block, expected):
    b = builder(block, hash_block)
    table = torch.arange(1, 33, dtype=torch.int32).reshape(1, -1)
    starts = torch.tensor([0, end - start], dtype=torch.int32)
    common = NS(
        num_reqs=1,
        num_decodes=0,
        query_start_loc=starts,
        query_start_loc_cpu=starts,
        seq_lens_cpu_upper_bound=torch.tensor([end]),
        block_table_tensor=table,
    )
    value = b.build(0, common)
    if expected is None:
        assert value.checkpoint_offsets[0].item() == 0
        assert value.checkpoint_slots[0].item() == -1
    else:
        offset, column = expected
        assert value.checkpoint_offsets[0].item() == offset
        assert value.checkpoint_columns[0].item() == column
        assert value.checkpoint_slots[0].item() == table[0, column].item()
        replacement = table + 100
        after = b.update_block_table(value, replacement, None)
        assert after.checkpoint_slots[0].item() == replacement[0, column].item()
        replacement[0, column] = 0
        assert (
            b.update_block_table(after, replacement, None).checkpoint_slots[0].item()
            == -1
        )
