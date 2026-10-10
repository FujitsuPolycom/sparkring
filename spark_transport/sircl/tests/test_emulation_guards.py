"""The GPU emulation's sizing of the one shared GPU, offline: hardware queues and resident grids."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sparkring_sircl.testing import gpu_emulation as ge  # noqa: E402


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
    monkeypatch.delenv("SIRCL_LARGE_BLOCKS", raising=False)
    return monkeypatch


def test_a_multi_rank_group_asks_for_32_hardware_queues_before_the_context_exists(clean):
    clean.setattr(torch.cuda, "is_initialized", lambda: False)
    assert ge.share_hardware_queues(1) is None and "CUDA_DEVICE_MAX_CONNECTIONS" not in ge.os.environ
    assert ge.share_hardware_queues(8) == "CUDA_DEVICE_MAX_CONNECTIONS=32 (set by the emulation)"
    assert ge.os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] == "32"
    assert ge.share_hardware_queues(8) is None            # already 32
    clean.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "16")
    assert ge.share_hardware_queues(2) == "CUDA_DEVICE_MAX_CONNECTIONS=16 (set by the caller, below 32)"
    assert ge.os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] == "16"


def test_a_context_created_first_is_reported_not_changed(clean):
    clean.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.warns(RuntimeWarning, match="the ranks' streams may share hardware queues"):
        text = ge.share_hardware_queues(8)
    assert "CUDA_DEVICE_MAX_CONNECTIONS unset (8)" in text and "CUDA_DEVICE_MAX_CONNECTIONS" not in ge.os.environ
    clean.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32")
    assert ge.share_hardware_queues(8) is None


def test_chain_and_link_grids_stay_resident_at_one_block_per_multiprocessor():
    # A GB10 has 48 multiprocessors: four roles per rank leave 1 block per role for eight ranks, 2 for four, 4 for
    # two; an RTX 5090 (170) keeps 4 for eight ranks.
    assert [ge.resident_role_cap(world, 48) for world in (2, 3, 4, 8, 16)] == [4, 4, 2, 1, 1]
    assert [ge.resident_role_cap(world, 170) for world in (2, 4, 8, 16)] == [16, 8, 4, 2]
    assert ge.resident_role_cap(8, 48, roles=2) == 2


def _session(chain, **link):
    import types

    return types.SimpleNamespace(chain_blocks=chain, link_blocks=dict(link))


def test_the_sessions_own_blocks_are_lowered_unless_the_environment_sets_them():
    sessions = [_session(4, ring_reduce=4, chain_scatter=4, ring_gather=1) for _ in range(8)]
    notes = ge.cap_session_blocks(sessions, 1, environ={})
    assert notes == ["chain blocks per role 4 -> 1", "link blocks per role of chain_scatter, ring_reduce -> 1"]
    assert all(s.chain_blocks == 1 and s.link_blocks == {"ring_reduce": 1, "chain_scatter": 1, "ring_gather": 1}
               for s in sessions)
    assert ge.cap_session_blocks(sessions, 1, environ={}) == []            # already within the cap
    sessions = [_session(4, ring_reduce=4) for _ in range(2)]
    notes = ge.cap_session_blocks(sessions, 2, environ={"SIRCL_CHAIN_BLOCKS": "4", "SIRCL_REDUCE_LINK_BLOCKS": "4"})
    assert notes == [] and sessions[0].chain_blocks == 4 and sessions[0].link_blocks == {"ring_reduce": 4}
    notes = ge.cap_session_blocks(sessions, 2, environ={"SIRCL_LINK_BLOCKS": "4"})
    assert notes == ["chain blocks per role 4 -> 2"] and sessions[1].link_blocks == {"ring_reduce": 4}


def test_every_ranks_grid_stays_resident_at_one_block_per_multiprocessor(clean):
    # An RTX 5090 has 170 multiprocessors: eight ranks get 16 blocks each, four and fewer keep the default 32.
    assert [ge.resident_grid_cap(world, 170, 32) for world in (2, 3, 4, 6, 8, 16)] == [None, None, None, 16, 16, 8]
    assert ge.resident_grid_cap(8, 48, 32) == 4
    clean.setenv("SIRCL_LARGE_BLOCKS", "32")
    assert ge.resident_grid_cap(8, 170, 32) is None       # the caller's cap stands


def test_the_grid_cap_holds_while_sessions_are_constructed_and_only_then(clean):
    # Four ranks on a GB10 (48 multiprocessors): 8 blocks each while inside the block, the variable gone after.
    with ge.large_block_cap(4, 48, 32) as note:
        assert ge.os.environ["SIRCL_LARGE_BLOCKS"] == "8"
        assert note == ("SIRCL_LARGE_BLOCKS=8 (set by the emulation: 4 grids of 32 blocks exceed the 48 "
                        "multiprocessors at one block each)")
    assert "SIRCL_LARGE_BLOCKS" not in ge.os.environ
    with ge.large_block_cap(4, 170, 32) as note:             # the default fits
        assert note is None and "SIRCL_LARGE_BLOCKS" not in ge.os.environ
    clean.setenv("SIRCL_LARGE_BLOCKS", "32")
    with ge.large_block_cap(8, 48, 32) as note:              # the caller's cap stands, and stays
        assert note is None and ge.os.environ["SIRCL_LARGE_BLOCKS"] == "32"
    assert ge.os.environ["SIRCL_LARGE_BLOCKS"] == "32"


def test_the_post_order_is_read_once_for_every_rank(clean):
    clean.delenv("SIRCL_POST_ORDER", raising=False)
    assert ge.construction_post_order() == ""               # the layout's default
    clean.setenv("SIRCL_POST_ORDER", "ring-farthest")
    assert ge.construction_post_order() == "ring-farthest"


def test_role_blocks_are_lowered_for_the_world_that_shares_the_gpu(clean):
    sessions = [_session(4, ring_reduce=4, chain_gather=4) for _ in range(4)]
    note = ge.lower_role_blocks(sessions, 4, 48)
    assert note == ("chain blocks per role 4 -> 2; link blocks per role of chain_gather, ring_reduce -> 2 (set by the "
                    "emulation: 4 grids of 4 roles at the sessions' blocks exceed the 48 multiprocessors at one "
                    "block each)")
    assert all(s.chain_blocks == 2 and s.link_blocks == {"ring_reduce": 2, "chain_gather": 2} for s in sessions)
    assert ge.lower_role_blocks(sessions, 4, 48) is None
