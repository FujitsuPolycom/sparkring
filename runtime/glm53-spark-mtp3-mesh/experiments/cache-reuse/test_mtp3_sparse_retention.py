"""Installed-algorithm tests for MTP3 checkpoint production and reuse."""

import os
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest

from repro_mtp3_checkpoint_materialization import checkpoint_kernel_metadata, load_scheduler, partial_hit_eligibility
from repro_mtp3_sparse_retention import ROOT, FullAttentionSpec, MambaSpec, Pool, load_algorithms


SOURCE = Path(os.environ.get("MTP3_RETENTION_SOURCE", str(Path(__file__).parent / "mtp3-retention-patched")))


def trace(prompt, *, budget=8192, eagle=True, scheduler_source=None, manager_source=None):
    algorithms = load_algorithms(ROOT / "v1/core/kv_cache_utils.py",
        manager_source=manager_source or SOURCE / "v1/core/single_type_kv_cache_manager.py")
    Scheduler = load_scheduler(scheduler_source or SOURCE / "v1/core/sched/scheduler.py")
    scheduler = Scheduler()
    scheduler.cache_config = SimpleNamespace(block_size=512)
    scheduler.use_eagle = eagle
    scheduler.mamba_has_prefill_checkpoint_blocks = True
    scheduler.hash_block_size = 512
    scheduler.block_size = 2048
    scheduler.need_mamba_block_aligned_split = True
    scheduler.kv_cache_manager = SimpleNamespace(coordinator=SimpleNamespace(enable_partial_hash_hits=True))
    partial_hit_eligibility(scheduler,
        SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=MambaSpec(512))]),
        scheduler_source or SOURCE / "v1/core/sched/scheduler.py")
    scheduler.max_num_scheduled_tokens = 8192
    scheduler.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
    scheduler.num_prefill_lookahead = 1
    publication = (prompt - 1) // 2048 * 2048
    scheduler._recurrent_publication_boundaries = lambda request: (publication,) if publication else ()
    request = SimpleNamespace(num_prompt_tokens=prompt, num_tokens=prompt,
                              num_computed_tokens=0, shared_prefix_boundary=0)
    spec = MambaSpec(512)
    produced = set()
    retained = set()
    steps = []
    cached = 0
    checkpoint = checkpoint_kernel_metadata()
    while request.num_computed_tokens < prompt:
        start = request.num_computed_tokens
        n = scheduler._mamba_block_aligned_split(request, min(budget, prompt - start))
        n = scheduler._reserve_prefill_lookahead(request, start, n)
        assert n > 0
        end = start + n
        point = checkpoint(start, end, 512)
        if end % 512 == 0:
            produced.add(end)
        if point is not None:
            produced.add(point)
        num_full = end // 512
        mask = algorithms.MambaManager.reachable_block_mask(start_block=cached,
            end_block=num_full, alignment_tokens=2048, kv_cache_spec=spec,
            use_eagle=eagle, retention_interval=0, reachable_boundaries=(prompt - 1,))
        retained.update((cached + index + 1) * 512 for index, keep in enumerate(mask)
                        if keep and (cached + index + 1) * 512 in produced)
        cached = num_full
        request.num_computed_tokens = end
        steps.append(end)
        assert len(steps) < 100
    return algorithms, produced, retained, steps


def replay(algorithms, prompt, retained, *, eagle=True):
    target, recurrent = FullAttentionSpec(512), MambaSpec(512)
    pool = Pool(512, retained, prompt // 512 * 512)
    coordinator = algorithms.HybridKVCacheCoordinator()
    coordinator.kv_cache_config = SimpleNamespace(kv_cache_groups=(target, recurrent))
    coordinator.single_type_managers = (SimpleNamespace(block_size=2048), SimpleNamespace(block_size=512))
    group = namedtuple("SpecGroup", "spec group_ids manager_cls use_eagle")
    coordinator.attention_groups = [group(target, [0], algorithms.FullAttentionManager, eagle),
                                    group(recurrent, [1], algorithms.MambaManager, eagle)]
    coordinator.block_pool = pool
    coordinator.hash_block_size = 512
    coordinator.scheduler_block_size = 2048
    coordinator.enable_partial_hash_hits = True
    coordinator.dcp_world_size = 4
    hashes = list(range(512, prompt + 1, 512))
    return coordinator.find_longest_cache_hit(hashes, prompt - 1)[1]


@pytest.mark.parametrize("prompt", [32768, 32769, 32789, 33280, 33281, 100968])
@pytest.mark.parametrize("budget", [8192, 7680])
def test_speculative_predecessor_is_materialized_retained_and_reusable(prompt, budget):
    algorithms, produced, retained, _ = trace(prompt, budget=budget)
    required = ((prompt - 1) // 512 - 1) * 512
    assert required in produced
    assert required in retained
    assert replay(algorithms, prompt, retained) == required
    # Preserve the scheduler-aligned publication checkpoint as well.
    assert (prompt - 1) // 2048 * 2048 in retained


def test_known_successful_100k_prompt_keeps_its_existing_reuse():
    algorithms, _, retained, _ = trace(100968)
    assert replay(algorithms, 100968, retained) == 100352


def test_non_speculative_internal_checkpoint_behavior_is_unchanged():
    _, produced, retained, steps = trace(32789, eagle=False)
    assert steps == [8192, 16384, 24576, 32768, 32789]
    assert 32256 not in produced
    assert retained == {32768}


def test_dense_retention_remains_dense():
    algorithms = load_algorithms(ROOT / "v1/core/kv_cache_utils.py",
        manager_source=SOURCE / "v1/core/single_type_kv_cache_manager.py")
    assert algorithms.MambaManager.reachable_block_mask(start_block=0, end_block=64,
        alignment_tokens=2048, kv_cache_spec=MambaSpec(512), use_eagle=True,
        retention_interval=None, reachable_boundaries=(32788,)) is None


def test_retention_change_alone_cannot_create_the_missing_checkpoint():
    _, produced, retained, _ = trace(32789, scheduler_source=ROOT / "v1/core/sched/scheduler.py")
    assert 32256 not in produced
    assert 32256 not in retained


def test_materialization_change_alone_does_not_preserve_the_checkpoint():
    _, produced, retained, _ = trace(32789, manager_source=ROOT / "v1/core/single_type_kv_cache_manager.py")
    assert 32256 in produced
    assert 32256 not in retained
