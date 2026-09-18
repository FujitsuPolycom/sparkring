# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint planning, append-only ownership and worker metadata contracts."""

from types import SimpleNamespace as NS
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

import importlib.util
import os
from pathlib import Path

_SOURCE = (
    Path(os.environ["SPARKRING_TEST_SOURCE_ROOT"])
    / "vllm/v1/core/recurrent_prefill_checkpoint.py"
)
_SPEC = importlib.util.spec_from_file_location("selected_checkpoint_math", _SOURCE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
checkpoint_metadata = _MODULE.checkpoint_metadata
checkpoint_plan_rows = _MODULE.checkpoint_plan_rows
continuation_layout = _MODULE.continuation_layout
prefill_checkpoint_plan = _MODULE.prefill_checkpoint_plan
validate_coalescing_config = _MODULE.validate_coalescing_config

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

pytestmark = pytest.mark.cpu_test


def test_fresh_and_continuation_plans_export_exact_retained_boundaries():
    assert prefill_checkpoint_plan(
        start=0,
        end=8192,
        prompt=8192,
        num_tokens=8192,
        block_size=512,
        publications=(6144, 7168),
    ) == (0, 8192, (6144, 7168))
    assert prefill_checkpoint_plan(
        start=8192,
        end=16384,
        prompt=16384,
        num_tokens=16384,
        block_size=512,
        publications=(14336, 15360),
    ) == (8192, 16384, (14336, 15360))
    assert checkpoint_metadata((8192, 16384, (14336, 15360)), 8192, 16384, 512, 2) == (
        [6144, 7168],
        [27, 29],
    )


def test_checkpoint_plan_can_carry_only_the_final_running_state():
    assert prefill_checkpoint_plan(
        start=0,
        end=8192,
        prompt=10240,
        num_tokens=10240,
        block_size=512,
        publications=(9216, 9728),
    ) == (0, 8192, ())
    assert checkpoint_metadata((0, 8192, ()), 0, 8192, 512, 4) == (
        [0, 0, 0, 0],
        [-1, -1, -1, -1],
    )


@pytest.mark.parametrize("end", [8193, 16385, 24576])
def test_unrepresentable_or_over_budget_final_spans_use_ordinary_splitting(end):
    assert (
        prefill_checkpoint_plan(
            start=8192,
            end=end,
            prompt=end,
            num_tokens=end,
            block_size=512,
            publications=(14336,),
        )
        is None
    )


def test_worker_checkpoint_rows_follow_request_order_and_reject_span_drift():
    batch = NS(
        req_ids=["b", "a"],
        num_reqs=2,
        num_computed_tokens_np=np.array([8192, 0]),
        num_scheduled_tokens=np.array([8192, 8192]),
        query_start_loc_np=np.array([0, 8192, 16384]),
    )
    plans = {"a": (0, 8192, (6144, 7168)), "b": (8192, 16384, (14336, 15360))}
    assert checkpoint_plan_rows(batch, plans, 3) == [plans["b"], plans["a"], None]
    with pytest.raises(ValueError, match="capture"):
        checkpoint_plan_rows(batch, plans, 3, for_capture=True)
    batch.num_computed_tokens_np[0] += 1
    with pytest.raises(ValueError, match="query span"):
        checkpoint_plan_rows(batch, plans, 3)


def supported_config():
    return NS(
        model_config=NS(
            hf_text_config=NS(model_type="glm5_next_text"),
            dtype=torch.bfloat16,
            enable_sleep_mode=False,
            enable_return_routed_experts=False,
        ),
        cache_config=NS(
            enable_prefix_caching=True,
            mamba_cache_mode="align",
            prefix_cache_retention_interval=0,
        ),
        parallel_config=NS(
            tensor_parallel_size=4,
            decode_context_parallel_size=4,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
        ),
        scheduler_config=NS(
            max_num_batched_tokens=8192,
            max_num_scheduled_tokens=None,
            long_prefill_token_threshold=0,
            prefill_compute_share=None,
            max_parallel_prefills=1,
            enable_chunked_prefill=True,
        ),
        use_v2_model_runner=True,
        lora_config=None,
        use_request_boundary_checkpoints=False,
        additional_config={"kda_prefill_backend": "b12x"},
        speculative_config=None,
    )


def test_coalescing_default_off_does_not_require_model_capabilities(monkeypatch):
    monkeypatch.setenv("VLLM_B12X_KDA_PREFILL_COALESCING", "0")
    assert not validate_coalescing_config(object())


def test_coalescing_diagnostic_limit_is_nonnegative(monkeypatch):
    from vllm import envs

    monkeypatch.setenv("VLLM_B12X_KDA_PREFILL_COALESCING_LOG_LIMIT", "-1")
    assert envs.VLLM_B12X_KDA_PREFILL_COALESCING_LOG_LIMIT == 0
    monkeypatch.setenv("VLLM_B12X_KDA_PREFILL_COALESCING_LOG_LIMIT", "3")
    assert envs.VLLM_B12X_KDA_PREFILL_COALESCING_LOG_LIMIT == 3


@pytest.mark.parametrize("dcp", [1, 2, 4])
def test_coalescing_accepts_context_groups_within_tp4(monkeypatch, dcp):
    monkeypatch.setenv("VLLM_B12X_KDA_PREFILL_COALESCING", "1")
    config = supported_config()
    config.parallel_config.decode_context_parallel_size = dcp
    assert validate_coalescing_config(config)


def test_coalescing_accepts_tp2_dcp1(monkeypatch):
    monkeypatch.setenv("VLLM_B12X_KDA_PREFILL_COALESCING", "1")
    config = supported_config()
    config.parallel_config.tensor_parallel_size = 2
    config.parallel_config.decode_context_parallel_size = 1
    assert validate_coalescing_config(config)


@pytest.mark.parametrize("dcp", [2, 4])
def test_coalescing_rejects_unqualified_tp2_context_groups(monkeypatch, dcp):
    monkeypatch.setenv("VLLM_B12X_KDA_PREFILL_COALESCING", "1")
    config = supported_config()
    config.parallel_config.tensor_parallel_size = 2
    config.parallel_config.decode_context_parallel_size = dcp
    with pytest.raises(ValueError, match="TP2 with DCP1"):
        validate_coalescing_config(config)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("parallel_config", "tensor_parallel_size", 3),
        ("parallel_config", "decode_context_parallel_size", 3),
        ("scheduler_config", "max_num_batched_tokens", 4096),
        ("cache_config", "prefix_cache_retention_interval", 2048),
        ("model_config", "dtype", torch.float16),
    ],
)
def test_explicit_coalescing_rejects_unsupported_configuration(
    monkeypatch, section, field, value
):
    monkeypatch.setenv("VLLM_B12X_KDA_PREFILL_COALESCING", "1")
    config = supported_config()
    assert validate_coalescing_config(config)
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError, match="requires GLM5Next"):
        validate_coalescing_config(config)


def cache_fixture(prompt, speculative=3, *, checkpoints=4, dcp=4):
    from tests.v1.core.utils import create_requests
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )

    config = KVCacheConfig(
        num_blocks=1000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attention"],
                FullAttentionSpec(
                    block_size=512, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            ),
            KVCacheGroupSpec(
                ["recurrent"],
                MambaSpec(
                    block_size=512,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=speculative,
                    num_prefill_checkpoint_blocks=checkpoints,
                ),
            ),
        ],
    )
    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        config,
        NS(
            cache_config=NS(
                block_size=512, enable_prefix_caching=True, prefix_match_unit=None
            ),
            parallel_config=NS(decode_context_parallel_size=dcp),
            kv_transfer_config=None,
        ),
    )
    assert (scheduler_block_size, hash_block_size) == (512 * dcp, 512)
    cache = KVCacheManager(
        config,
        max_model_len=131072,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        enable_caching=True,
        use_eagle=True,
        dcp_world_size=dcp,
    )
    (request,) = create_requests(1, num_tokens=prompt, block_size=512)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._kda_coalescing_enabled = True
    scheduler._kda_coalescing_exclusive = True
    scheduler._kda_coalescing_origins = {}
    scheduler.cache_config = NS(block_size=512, prefix_cache_retention_interval=0)
    scheduler.kv_cache_manager = cache
    scheduler.max_num_scheduled_tokens = 8192
    scheduler.scheduler_config = NS(long_prefill_token_threshold=0)
    scheduler.mamba_has_prefill_checkpoint_blocks = True
    scheduler.mamba_partial_cache_hit = cache.coordinator.enable_partial_hash_hits
    scheduler.hash_block_size = 512
    scheduler.drop_last_prefix_cache_block = True
    scheduler.use_eagle = True
    return cache, cache.coordinator.single_type_managers[1], scheduler, request


@pytest.mark.parametrize("prompt", [8192, 10240, 16384, 32768])
@pytest.mark.parametrize("speculative", [0, 3])
@pytest.mark.parametrize("dcp", [1, 2, 4])
def test_scheduler_and_allocator_keep_8k_chunks_and_worker_column_ownership(
    prompt, speculative, dcp
):
    from vllm.v1.request import RequestStatus

    cache, manager, scheduler, request = cache_fixture(prompt, speculative, dcp=dcp)
    worker: list[KVCacheBlock] = []
    chunks = []
    while request.num_computed_tokens < prompt:
        start = request.num_computed_tokens
        size = scheduler._mamba_block_aligned_split(request, min(8192, prompt - start))
        plan = scheduler._recurrent_checkpoint_plan(request, start, start + size)
        prefix = tuple(worker)
        result = cache.allocate_slots(
            request,
            size,
            num_lookahead_tokens=speculative,
            recurrent_checkpoint_plan=plan,
        )
        assert result is not None
        worker.extend(result.blocks[1])
        assert worker[: len(prefix)] == list(prefix)
        if plan:
            active = ([start // 512 - 1] if start else []) + [
                p // 512 - 1 for p in plan[2]
            ]
            active += list(range((start + size) // 512 - 1, len(worker)))
            assert all(
                worker[c] is manager.req_to_blocks[request.request_id][c]
                for c in active
            )
            assert len(active) == len({worker[c].block_id for c in active})
        assert manager._planned_recurrent_checkpoints == {}
        if start == 0:
            scheduler._record_coalescing_origin(request, 0, 0, 0, False)
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens += size
        chunks.append(size)
    assert chunks == [8192] * (prompt // 8192) + (
        [prompt % 8192] if prompt % 8192 else []
    )


@pytest.mark.parametrize("dcp", [1, 2, 4])
def test_continuation_admission_failure_releases_only_its_temporary_plan(
    monkeypatch, dcp
):
    cache, manager, scheduler, request = cache_fixture(16384, dcp=dcp)
    initial = scheduler._recurrent_checkpoint_plan(request, 0, 8192)
    assert (
        cache.allocate_slots(request, 8192, recurrent_checkpoint_plan=initial)
        is not None
    )
    scheduler._record_coalescing_origin(request, 0, 0, 0, False)
    request.num_computed_tokens = 8192
    plan = scheduler._recurrent_checkpoint_plan(request, 8192, 16384)
    assert plan is not None
    retained = tuple(manager.req_to_blocks[request.request_id])
    monkeypatch.setattr(cache.block_pool, "get_num_free_blocks", lambda: 0)
    assert cache.allocate_slots(request, 8192, recurrent_checkpoint_plan=plan) is None
    assert manager._planned_recurrent_checkpoints == {}
    assert tuple(manager.req_to_blocks[request.request_id]) == retained


@pytest.mark.parametrize("dcp", [1, 2, 4])
def test_cache_hit_preemption_and_shared_speculative_slots_disable_coalescing(dcp):
    cache, manager, scheduler, request = cache_fixture(16384, dcp=dcp)
    initial = scheduler._recurrent_checkpoint_plan(request, 0, 8192)
    assert (
        cache.allocate_slots(request, 8192, recurrent_checkpoint_plan=initial)
        is not None
    )
    request.num_computed_tokens = 8192
    assert scheduler._recurrent_checkpoint_plan(request, 8192, 16384) is None
    scheduler._kda_coalescing_origins[request.request_id] = request
    assert scheduler._recurrent_checkpoint_plan(request, 8192, 16384) is not None
    request.num_preemptions = 1
    assert scheduler._recurrent_checkpoint_plan(request, 8192, 16384) is None
    request.num_preemptions = 0
    manager.req_to_blocks[request.request_id][-1].ref_cnt += 1
    assert scheduler._recurrent_checkpoint_plan(request, 8192, 16384) is None
    with pytest.raises(ValueError, match="private"):
        continuation_layout(
            manager.req_to_blocks[request.request_id],
            (8192, 16384, (14336, 15360)),
            512,
            3,
        )


def test_full_prompt_admission_reserves_checkpoint_peak_for_internal_8k_chunk():
    cache, manager, scheduler, request = cache_fixture(10240)
    plan = scheduler._recurrent_checkpoint_plan(request, 0, 8192)
    assert plan == (0, 8192, (6144,))
    manager._planned_recurrent_checkpoints[request.request_id] = plan
    try:
        peak = manager.get_num_blocks_to_allocate(
            request.request_id, 10240, [], 0, 0, 10240, apply_admission_cap=True
        )
        assert peak == 9
        assert request.request_id not in manager._num_checkpoint_blocks
    finally:
        manager._planned_recurrent_checkpoints.clear()
    result = cache.allocate_slots(
        request, 8192, recurrent_checkpoint_plan=plan, full_sequence_must_fit=True
    )
    assert result is not None
    assert manager._planned_recurrent_checkpoints == {}
    assert not manager.req_to_blocks[request.request_id][11].is_null


@pytest.mark.parametrize("prompt", [8192, 16384, 32768])
@pytest.mark.parametrize("dcp", [1, 2, 4])
def test_dcp_geometry_retains_required_states_without_extra_passes(prompt, dcp):
    cache, manager, scheduler, request = cache_fixture(prompt, dcp=dcp)
    assert manager.hit_alignment_tokens == 512
    assert manager.scheduler_block_size == 512 * dcp
    offsets = {1: (1024, 512), 2: (2048, 1024, 512), 4: (4096, 2048, 1024, 512)}
    expected = tuple(prompt - offset for offset in offsets[dcp])
    assert tuple(sorted(manager._expand_reachable_boundaries([prompt - 1]))) == expected
    start = prompt - 8192
    if start:
        while request.num_computed_tokens < start:
            initial = scheduler._recurrent_checkpoint_plan(
                request, request.num_computed_tokens, request.num_computed_tokens + 8192
            )
            assert (
                cache.allocate_slots(request, 8192, recurrent_checkpoint_plan=initial)
                is not None
            )
            if request.num_computed_tokens == 0:
                scheduler._record_coalescing_origin(request, 0, 0, 0, False)
            request.num_computed_tokens += 8192
    plan = scheduler._recurrent_checkpoint_plan(request, start, prompt)
    assert plan == (start, prompt, expected)
    assert scheduler._mamba_block_aligned_split(request, 8192) == 8192
    assert (
        cache.allocate_slots(
            request, 8192, num_lookahead_tokens=3, recurrent_checkpoint_plan=plan
        )
        is not None
    )
    blocks = manager.req_to_blocks[request.request_id]
    columns = [p // 512 - 1 for p in expected] + [prompt // 512 - 1]
    assert all(not blocks[column].is_null for column in columns)
    assert len({blocks[column].block_id for column in columns}) == len(columns)


def test_two_checkpoint_capacity_keeps_safe_dcp4_fallback():
    _, manager, scheduler, request = cache_fixture(8192, checkpoints=2)
    assert manager.hit_alignment_tokens == 512
    assert scheduler._recurrent_checkpoint_plan(request, 0, 8192) is None
    assert scheduler._mamba_block_aligned_split(request, 8192) == 4096


@pytest.mark.parametrize(
    "share,lanes,accepted", [(None, 1, True), (0.5, 1, False), (None, 2, False)]
)
def test_coalescing_real_r33_scheduler_config(monkeypatch, share, lanes, accepted):
    from vllm.config.scheduler import SchedulerConfig

    monkeypatch.setenv("VLLM_B12X_KDA_PREFILL_COALESCING", "1")
    config = supported_config()
    config.scheduler_config = SchedulerConfig(
        max_model_len=1048576,
        is_encoder_decoder=False,
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        enable_chunked_prefill=True,
        prefill_compute_share=share,
        max_parallel_prefills=lanes,
    )
    if accepted:
        assert validate_coalescing_config(config)
    else:
        with pytest.raises(ValueError, match="requires"):
            validate_coalescing_config(config)
