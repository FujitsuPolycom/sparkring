# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay R37 checkpoint reservation and allocation using production methods.

The failure is a continuation whose checkpoint column is occupied by a private
speculative block that allocation would relocate. The observable contract is a
non-null checkpoint destination, distinct running/scratch slots, and a reservation
covering the actual allocations without changing prior worker block IDs.

Only the block pool is a test double. AST loading executes the actual Mamba
reservation, allocation, relocation, and spec-column methods without importing
unavailable R37 native/Transformers dependencies on a CPU development host.
This does not establish GPU state contents or external-cache restore correctness.
"""

import __future__
import ast
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(os.environ.get('SPARKRING_TEST_SOURCE_ROOT', str(Path(__file__).resolve().parents[1])))
MANAGER = ROOT / "vllm/v1/core/single_type_kv_cache_manager.py"


def _class_methods(path, class_name, names, bases=()):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    cls.bases = [ast.Name(id=name, ctx=ast.Load()) for name in bases]
    cls.decorator_list = []
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in cls.body} == set(names)
    return cls


@pytest.fixture(scope="module")
def classes():
    base = _class_methods(
        MANAGER,
        "SingleTypeKVCacheManager",
        (
            "_get_num_evictable_blocks",
            "_has_partial_local_hit",
        ),
    )
    manager = _class_methods(
        MANAGER,
        "MambaManager",
        (
            "_needs_internal_checkpoint",
            "get_num_blocks_to_allocate",
            "allocate_new_blocks",
            "_relocate_speculative_block",
        ),
        ("SingleTypeKVCacheManager",),
    )
    spec = _class_methods(
        ROOT / "vllm/v1/kv_cache_interface.py",
        "MambaSpec",
        ("prefill_checkpoint_indices",),
    )
    code = ast.fix_missing_locations(
        ast.Module(body=[base, spec, manager], type_ignores=[])
    )
    namespace = {"cdiv": lambda a, b: (a + b - 1) // b}
    exec(
        compile(code, str(MANAGER), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    return namespace["MambaManager"], namespace["MambaSpec"]


@dataclass
class Block:
    block_id: int
    is_null: bool = False
    block_hash: object = None
    ref_cnt: int = 1


class Pool:
    def __init__(self):
        self.next_id = 100
        self.allocations = []

    def get_new_blocks(self, count):
        self.allocations.append(count)
        blocks = [Block(index) for index in range(self.next_id, self.next_id + count)]
        self.next_id += count
        return blocks

    def is_block_writable(self, block):
        return not block.is_null and block.block_hash is None and block.ref_cnt == 1


def make_manager(classes, *, block_size=16, speculative=3, checkpoints=1):
    manager_cls, spec_cls = classes
    manager = manager_cls()
    manager.kv_cache_spec = spec_cls()
    manager.kv_cache_spec.num_prefill_checkpoint_blocks = checkpoints
    manager.kv_cache_spec.block_size = block_size
    manager.block_size = block_size
    manager.num_speculative_blocks = speculative
    manager.mamba_cache_mode = "align"
    manager.req_to_blocks = {"request": []}
    manager._null_block = Block(0, is_null=True)
    manager._planned_recurrent_checkpoints = {}
    manager._num_checkpoint_blocks = {}
    manager._partial_hit_reqs = {}
    manager._allocated_block_reqs = set()
    manager._packed_prefill_checkpoint_reqs = set()
    manager.last_state_block_idx = {}
    manager.cached_blocks_this_step = set()
    manager.block_pool = Pool()
    return manager


def advance(manager, start, end):
    arguments = dict(
        request_id="request",
        num_tokens=end + manager.num_speculative_blocks,
        new_computed_blocks=(),
        total_computed_tokens=start,
        num_local_computed_tokens=start,
        num_tokens_main_model=end,
    )
    before = dict(manager._num_checkpoint_blocks)
    admission = manager.get_num_blocks_to_allocate(
        **arguments, apply_admission_cap=True
    )
    assert manager._num_checkpoint_blocks == before
    reservation = manager.get_num_blocks_to_allocate(**arguments)
    assert admission >= reservation
    pool_start = manager.block_pool.next_id
    old_blocks = list(manager.req_to_blocks["request"])
    appended = manager.allocate_new_blocks(
        "request", end + manager.num_speculative_blocks, end
    )
    assert manager.block_pool.next_id - pool_start == reservation
    current = manager.req_to_blocks["request"]
    assert appended == current[len(old_blocks) :]
    # Workers receive only appended IDs. Existing live columns may become null
    # on the scheduler, but allocation cannot retarget an existing live column.
    for old, new in zip(old_blocks, current):
        assert new.is_null or new is old
    live = [block.block_id for block in current if not block.is_null]
    assert len(live) == len(set(live))
    final_column = (end + manager.block_size - 1) // manager.block_size - 1
    assert len(current[final_column:]) == manager.num_speculative_blocks + 1
    assert all(not block.is_null for block in current[final_column:])
    return current


@pytest.mark.parametrize("end", [7869, 8194])
def test_mtp_continuation_keeps_the_7200_checkpoint_slot(classes, end):
    manager = make_manager(classes, block_size=1440)
    first = advance(manager, 0, 5760)
    former_scratch = first[4]
    second = advance(manager, 5760, end)
    assert not second[4].is_null, (
        "7200-token checkpoint was lost during scratch relocation"
    )
    assert second[4] is former_scratch
    assert manager._num_checkpoint_blocks["request"] == 1
    assert manager.block_pool.allocations[-1] == 2


@pytest.mark.parametrize("speculative", [0, 1, 3, 5])
@pytest.mark.parametrize("end", [51, 67, 83, 99, 131])
def test_checkpoint_survives_short_and_multi_boundary_continuations(
    classes, speculative, end
):
    manager = make_manager(classes, speculative=speculative)
    advance(manager, 0, 32)
    blocks = advance(manager, 32, end)
    checkpoint_column = end // manager.block_size - 1
    assert not blocks[checkpoint_column].is_null


@pytest.mark.parametrize("speculative", [0, 1, 3])
def test_existing_retained_checkpoint_is_not_replaced_or_counted_as_scratch(
    classes, speculative
):
    manager = make_manager(classes, block_size=1440, speculative=speculative)
    first = advance(manager, 0, 7200)
    checkpoint = first[4]
    checkpoint.block_hash = "retained-prefix"
    checkpoint.ref_cnt = 2
    second = advance(manager, 7200, 8194)
    assert second[4] is checkpoint
    assert manager._num_checkpoint_blocks["request"] == 0
    assert checkpoint.block_hash == "retained-prefix" and checkpoint.ref_cnt == 2


def test_initial_prefill_checkpoint_allocation_is_unchanged(classes):
    manager = make_manager(classes, block_size=1440)
    blocks = advance(manager, 0, 7860)
    assert [i for i, block in enumerate(blocks) if not block.is_null] == [4, 5, 6, 7, 8]
    assert manager.block_pool.allocations == [5]


@pytest.mark.parametrize("block_hash,ref_cnt", [("retained-elsewhere", 1), (None, 2)])
def test_speculative_checkpoint_cannot_consume_hashed_or_shared_storage(
    classes, block_hash, ref_cnt
):
    manager = make_manager(classes)
    blocks = advance(manager, 0, 32)
    scratch = blocks[2]
    scratch.block_hash, scratch.ref_cnt = block_hash, ref_cnt
    with pytest.raises(AssertionError, match="exclusively owned"):
        advance(manager, 32, 51)


@pytest.mark.parametrize("end,checkpoints", [(64, 1), (67, 0)])
def test_aligned_endpoint_and_disabled_checkpoint_keep_existing_scratch_relocation(
    classes, end, checkpoints
):
    manager = make_manager(classes, checkpoints=checkpoints)
    advance(manager, 0, 32)
    blocks = advance(manager, 32, end)
    assert manager._num_checkpoint_blocks["request"] == 0
    assert blocks[2].is_null


def test_unaligned_continuation_does_not_request_an_internal_checkpoint(classes):
    manager = make_manager(classes)
    advance(manager, 0, 33)
    assert not manager._needs_internal_checkpoint("request", 67, 33)


def test_packed_multiple_checkpoints_keep_their_existing_allocation_path(classes):
    manager = make_manager(classes, checkpoints=3)
    advance(manager, 0, 32)
    blocks = advance(manager, 32, 67)
    assert "request" in manager._packed_prefill_checkpoint_reqs
    assert manager._num_checkpoint_blocks["request"] == 2
    assert all(not blocks[column].is_null for column in (2, 3))
