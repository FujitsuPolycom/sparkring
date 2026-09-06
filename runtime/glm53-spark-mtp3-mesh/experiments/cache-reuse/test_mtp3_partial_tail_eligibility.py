"""Execute the actual recurrent partial-tail predicate and scheduling path."""

from types import SimpleNamespace

import pytest

from repro_mtp3_checkpoint_materialization import partial_hit_eligibility
from repro_mtp3_sparse_retention import MambaSpec
from test_mtp3_sparse_retention import SOURCE, trace


@pytest.mark.parametrize("hash_size,page_sizes,alignment,fine,expected", [
    (512, [512], 2048, True, False),
    (256, [512], 2048, True, True),
    (512, [1024], 2048, True, True),
    (512, [], 2048, True, False),
    (256, [512], 2048, False, False),
])
def test_predicate_uses_actual_mamba_pages(hash_size, page_sizes, alignment, fine, expected):
    scheduler = SimpleNamespace(need_mamba_block_aligned_split=True,
        hash_block_size=hash_size, block_size=alignment,
        kv_cache_manager=SimpleNamespace(coordinator=SimpleNamespace(enable_partial_hash_hits=fine)))
    config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=MambaSpec(size)) for size in page_sizes])
    assert partial_hit_eligibility(scheduler, config, SOURCE / "v1/core/sched/scheduler.py") is expected


@pytest.mark.parametrize("prompt,expected", [
    (5306, [4096, 4608, 5306]),
    (32789, [8192, 16384, 24576, 32256, 32768, 32789]),
])
def test_unneeded_stop_removed_but_publication_and_predecessor_remain(prompt, expected):
    _, _, retained, steps = trace(prompt)
    assert steps == expected
    assert (prompt - 1) // 2048 * 2048 in retained
    assert ((prompt - 1) // 512 - 1) * 512 in retained


def test_successful_100k_case_drops_only_the_unneeded_fine_tail_stop():
    _, _, retained, steps = trace(100968)
    assert 100352 in steps and 100864 not in steps
    assert steps[-1] == 100968
    assert 100352 in retained
