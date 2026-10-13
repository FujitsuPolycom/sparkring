"""Subsets of a universal relay table must use the whole fabric's lanes."""
import pytest

from sparkring_sircl.vllm import fabric, settings
from sparkring_sircl.vllm.serve import plan


@pytest.mark.parametrize("members", [tuple(range(6)), (6, 7, 0, 1, 2, 3)])
def test_universal_routes_equal_whole_cycle(members):
    layout = fabric.Layout.ring(8)
    whole = fabric.describe_group(layout, range(8))
    subset = fabric.describe_group(layout, members, route_scope="fabric")
    assert subset.max_relays() == 3
    for rank, origin in enumerate(members):
        for peer, destination in enumerate(members):
            if rank == peer:
                continue
            for actual, expected in zip(subset.lanes(rank, peer), whole.lanes(origin, destination)):
                assert (actual.positions, actual.device(), actual.remote_device()) == (
                    expected.positions, expected.device(), expected.remote_device())
    session = plan.session_settings(subset, name="tp", groups="", scoped=True)
    assert not session.link_area and not session.chain_area and not session.problems
    assert subset.nccl_policy == fabric.NcclPolicy.NONE


def test_universal_dcp6_and_parent_membership():
    layout = fabric.Layout.ring(8)
    top = fabric.describe_group(layout, range(6), parent=range(6), route_scope="fabric")
    assert [lane.positions for lane in top.lanes(5, 0)] == [(5, 6, 7, 0)] * 2
    with pytest.raises(fabric.FabricError, match="not inside"):
        fabric.describe_group(layout, (0, 7), parent=range(6), route_scope="fabric")


def test_default_scope_and_invalid_setting():
    assert settings.route_scope({}) == "group"
    assert settings.route_scope({"SIRCL_ROUTE_SCOPE": "fabric"}) == "fabric"
    with pytest.raises(settings.SettingError, match="SIRCL_ROUTE_SCOPE"):
        settings.route_scope({"SIRCL_ROUTE_SCOPE": "typo"})
    with pytest.raises(fabric.FabricError, match="qualified limit"):
        fabric.describe_group(fabric.Layout.ring(8), range(6))


def test_adapter_and_p2p_share_universal_routes():
    pytest.importorskip('torch')
    from sparkring_sircl.vllm.adapter import AdapterConfig, GroupPlacement
    from sparkring_sircl.vllm.p2p import InstancePlan, Shape
    env = {'SIRCL_FABRIC': 'ring:8', 'SIRCL_RANK_POSITIONS': '0,1,2,3,4,5',
           'SIRCL_ROUTE_SCOPE': 'fabric', 'SIRCL_NCCL': 'never'}
    config = AdapterConfig.from_env(6, env)
    placed = GroupPlacement.of('tp:0', range(6), 0, config, environ=env)
    assert placed.topology.lanes(5, 0)[0].positions == (5, 6, 7, 0)
    # The P2P planner used to independently derive the rejected four-relay path.
    planned = InstancePlan(config.layout, config.positions, ('tp', 'dcp'), ('tp', 'dcp'),
                           Shape(world=6, tp=6, pp=1, dcp=2), env)
    assert planned.basis == 'instance'
