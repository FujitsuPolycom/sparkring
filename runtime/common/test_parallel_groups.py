"""The runtime process groups on a ring of Sparks: the shapes, the aliases and the arc rule; offline.

Each expectation is the rank list the named runtime builds for those widths,
transcribed from SGLang v0.5.21 ``initialize_model_parallel`` and vLLM
v0.19.1 ``initialize_model_parallel``; the ring picture in each test says why
the listed communicators are or are not one arc.
"""
import pytest

from runtime.common import parallel_groups as groups
from runtime.common.parallel_groups import ARC, SINGLE, STRIDED, WHOLE


def ranks_of(found, name):
    return [group.ranks for group in found if group.name == name]


def problems(found, positions):
    return groups.arc_problems(found, positions)


def test_world_sizes_follow_each_runtime():
    # SGLang: DP attention divides the tensor-parallel width; plain DP multiplies it.
    assert groups.sglang_world_size(tp=4, dp=2, enable_dp_attention=True) == 4
    assert groups.sglang_world_size(tp=4, dp=2, enable_dp_attention=False) == 8
    assert groups.sglang_world_size(tp=4, dp=2, pp=2) == 16
    # vLLM: data parallel outermost, so it always multiplies.
    assert groups.vllm_world_size(tp=2, dp=2) == 4
    assert groups.vllm_world_size(tp=4, dp=2, pp=2) == 16


def test_vllm_two_by_two_breaks_the_data_parallel_group():
    # Ring 0-1-2-3-0. tp keeps cable pairs [0,1] and [2,3]; dp gets the diagonals [0,2] and [1,3].
    found = groups.vllm_groups(tp=2, dp=2)
    assert ranks_of(found, "tp") == [(0, 1), (2, 3)]
    assert ranks_of(found, "dp") == [(0, 2), (1, 3)]
    # No expert-parallel group without expert parallelism.
    assert ranks_of(found, "ep") == []
    found_problems = problems(found, [0, 1, 2, 3])
    assert len(found_problems) == 2
    assert all("dp" in problem for problem in found_problems)


def test_vllm_expert_parallel_group_is_one_stage():
    # EP spans dp x pcp x tp, i.e. the whole world at one pipeline stage.
    found = groups.vllm_groups(tp=2, dp=2, enable_expert_parallel=True)
    assert ranks_of(found, "ep") == [(0, 1, 2, 3)]
    assert groups.ring_kind(groups.Group("ep", (0, 1, 2, 3)), [0, 1, 2, 3]) == WHOLE


def test_vllm_data_parallel_at_one_tpu_is_the_whole_ring():
    found = groups.vllm_groups(tp=1, dp=2)
    assert ranks_of(found, "dp") == [(0, 1)]
    assert problems(found, [0, 1]) == []


def test_vllm_pipeline_pairs_are_diagonal_at_eight():
    # tp4 pp2 on a ring of eight: the stage pairs are positions i and i + 4, four positions apart.
    found = groups.vllm_groups(tp=4, pp=2)
    assert ranks_of(found, "pp") == [(0, 4), (1, 5), (2, 6), (3, 7)]
    found_problems = problems(found, list(range(8)))
    assert len(found_problems) == 4
    assert all("pp" in problem for problem in found_problems)


def test_the_rank_to_position_map_decides_not_the_rank_numbers():
    # Ranks 1 and 2 swapped on the cable: a group over ranks 0 and 3 spans positions 0 and 2 (strided),
    # while the group over ranks 0 and 2 spans the cable neighbours 0 and 3.
    positions = [0, 1, 3, 2]
    assert groups.ring_kind(groups.Group("diagonal", (0, 3)), positions) == STRIDED
    assert groups.ring_kind(groups.Group("cable", (0, 2)), positions) == ARC
    assert groups.ring_kind(groups.Group("cable", (1, 3)), positions) == ARC


def test_two_by_two_fits_four_sparks_under_a_cable_order():
    # The installer chooses the positions: the tensor-parallel pairs keep cables [0,1] and [2,3], and
    # ranks 2 and 3 swap places so the data-parallel pairs land on cables [0,3] and [1,2].
    found = groups.vllm_groups(tp=2, dp=2)
    placed = groups.assignment(found, 4)
    assert placed is not None
    positions = [placed[rank] for rank in range(4)]
    assert problems(found, positions) == []
    assert groups.ring_kind(groups.Group("tp", (0, 1)), positions) == ARC
    assert groups.ring_kind(groups.Group("dp", (0, 2)), positions) == ARC


def test_two_by_four_does_not_fit_eight_sparks():
    # Four ranks per tensor-parallel group are arcs only as [0-3] and [4-7]; the data-parallel pairs
    # then join 0 with 4 and so on, and only the two cables between the arcs are ring edges.
    found = groups.vllm_groups(tp=4, dp=2)
    assert groups.assignment(found, 8) is None
    assert all("dp" in problem for problem in problems(found, list(range(8))))


def test_sglang_expert_parallel_at_the_tensor_width_is_one_communicator():
    # moe_ep_size == tp_size makes _MOE_EP the tensor-parallel group itself, not a second communicator.
    found = groups.sglang_groups(tp=4, moe_ep=4)
    assert ranks_of(found, "tp") == [(0, 1, 2, 3)]
    assert ranks_of(found, "moe_ep") == [(0, 1, 2, 3)]
    # world, tp, attention_tp and moe_ep all cover the same ranks, so one rank set carries every name.
    assert ("world/tp/attention_tp/moe_ep", (0, 1, 2, 3)) in groups.rank_sets(found)
    assert problems(found, [0, 1, 2, 3]) == []
    assert groups.assignment(found, 4) is not None


def test_sglang_dp_attention_groups_are_arcs():
    # tp4 dp2 with DP attention: world 4 (the replicas divide tp), attention TP 2 per replica.
    found = groups.sglang_groups(tp=4, dp=2, enable_dp_attention=True)
    assert ranks_of(found, "tp") == [(0, 1, 2, 3)]
    assert ranks_of(found, "attention_tp") == [(0, 1), (2, 3)]
    assert problems(found, [0, 1, 2, 3]) == []


def test_sglang_expert_group_strided_by_the_expert_width():
    # tp8 ep4 with one MoE replica: moe_tp is 2, so each expert group takes every second rank.
    found = groups.sglang_groups(tp=8, moe_ep=4)
    assert ranks_of(found, "moe_ep") == [(0, 2, 4, 6), (1, 3, 5, 7)]
    assert ranks_of(found, "moe_tp") == [(0, 1), (2, 3), (4, 5), (6, 7)]
    assert groups.ring_kind(groups.Group("moe_ep", (0, 2, 4, 6)), list(range(8))) == STRIDED
    assert groups.ring_kind(groups.Group("moe_tp", (0, 1)), list(range(8))) == ARC
    found_problems = problems(found, list(range(8)))
    assert len(found_problems) == 2
    assert all("moe_ep" in problem for problem in found_problems)


def test_sglang_moe_replica_group_is_the_strided_one():
    # tp8 ep4 moe_dp2: the expert groups become [0-3] and [4-7], and the MoE replicas become the diagonals.
    found = groups.sglang_groups(tp=8, moe_ep=4, moe_dp=2)
    assert ranks_of(found, "moe_ep") == [(0, 1, 2, 3), (4, 5, 6, 7)]
    assert ranks_of(found, "moe_dp") == [(0, 4), (1, 5), (2, 6), (3, 7)]
    found_problems = problems(found, list(range(8)))
    assert len(found_problems) == 4
    assert all("moe_dp" in problem for problem in found_problems)


def test_ring_kind_reads_the_ring_both_ways_round():
    ring = groups.Group("pair", (0, 3))
    assert groups.ring_kind(ring, [0, 1, 2, 3]) == ARC          # 3-0 is a cable
    assert groups.ring_kind(groups.Group("one", (2,)), [0, 1, 2, 3]) == SINGLE
    assert groups.ring_kind(groups.Group("all", (3, 0, 1, 2)), [0, 1, 2, 3]) == WHOLE
    assert groups.ring_kind(groups.Group("span", (0, 1, 2)), [0, 1, 2, 3]) == ARC
    assert groups.ring_kind(groups.Group("diagonal", (0, 2)), [0, 1, 2, 3]) == STRIDED


def test_a_group_must_name_each_rank_once_and_positions_must_cover_the_ring():
    with pytest.raises(ValueError):
        groups.ring_kind(groups.Group("twice", (0, 0)), [0, 1])
    with pytest.raises(ValueError):
        groups.ring_kind(groups.Group("pair", (0, 1)), [0, 1, 1])


def test_widths_must_divide_the_tensor_width():
    with pytest.raises(ValueError):
        groups.sglang_groups(tp=4, moe_ep=3)
    with pytest.raises(ValueError):
        groups.sglang_groups(tp=4, dp=3, enable_dp_attention=True)
    with pytest.raises(ValueError):
        groups.vllm_groups(tp=4, dcp=3)
