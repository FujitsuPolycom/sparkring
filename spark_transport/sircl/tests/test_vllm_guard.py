"""The no-NCCL-across-relays guard: construction, dispatch and the torch.distributed tripwire."""

from __future__ import annotations

import os
import sys
import types

import pytest
torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402

from sparkring_sircl.vllm import guard  # noqa: E402
from sparkring_sircl.vllm.adapter import AdapterConfig, resolver_for  # noqa: E402
from sparkring_sircl.vllm.fabric import Layout, NcclPolicy  # noqa: E402
from sparkring_sircl.vllm.guard import GuardedGroup, NcclAcrossRelayError  # noqa: E402


@pytest.fixture
def tripwire():
    guard.reset()
    guard.install_tripwire()
    yield
    guard.uninstall_tripwire()
    guard.reset()


class FakeGroup:
    """Stands for a device process group; the tripwire refuses before torch sees it."""


def path_entry() -> GuardedGroup:
    return GuardedGroup("tp:0", (0, 1, 2, 3), NcclPolicy.NONE,
                        "no cable between ranks 3-0 (positions 3-0, 2 relays)")


def ring_entry() -> GuardedGroup:
    cabled = lambda a, b: (a - b) % 8 in (1, 7)  # noqa: E731
    return GuardedGroup("tp:0", tuple(range(8)), NcclPolicy.RING, "ring", cabled=cabled)


@pytest.mark.parametrize("call", [
    lambda g: dist.all_reduce(torch.ones(4), group=g),
    lambda g: dist.all_gather_into_tensor(torch.empty(16), torch.ones(4), group=g),
    lambda g: dist.reduce_scatter_tensor(torch.empty(1), torch.ones(4), group=g),
    lambda g: dist.all_to_all_single(torch.empty(4), torch.ones(4), group=g),
    lambda g: dist.broadcast(torch.ones(4), 0, group=g),
    lambda g: dist.broadcast(torch.ones(4), 0, g),                     # positional group
    lambda g: dist.barrier(group=g),
    lambda g: dist.send(torch.ones(4), 1, group=g),
    lambda g: dist.recv(torch.ones(4), 1, g),
    lambda g: dist.broadcast_object_list([None], 0, group=g),
    lambda g: torch.distributed.distributed_c10d.all_reduce(torch.ones(4), group=g),
])
def test_tripwire_refuses_every_nccl_call_on_a_group_nccl_may_not_run(tripwire, call):
    group = FakeGroup()
    guard.register(group, path_entry())
    with pytest.raises(NcclAcrossRelayError, match="tp:0.*no cable between ranks 3-0"):
        call(group)


def test_tripwire_point_to_point_in_batches(tripwire):
    group = FakeGroup()
    guard.register(group, path_entry())
    op = types.SimpleNamespace(group=group, peer=1)
    with pytest.raises(NcclAcrossRelayError, match="batch_isend_irecv on group tp:0"):
        dist.batch_isend_irecv([op])


def test_ring_groups_allow_ring_collectives_and_cabled_pairs_only():
    entry = ring_entry()
    assert entry.allows("all_reduce") and entry.allows("broadcast")
    assert not entry.allows("all_to_all_single") and not entry.allows("gather")
    assert entry.allows("send", pair=(0, 1)) and entry.allows("recv", pair=(0, 7))
    with pytest.raises(NcclAcrossRelayError, match="between global ranks 0 and 4"):
        entry.check("send", pair=(0, 4))


def test_tripwire_passes_groups_it_does_not_know(tripwire):
    calls = []
    original = guard._ORIGINALS[("torch.distributed", "all_reduce")]

    def fake_all_reduce(tensor, op=None, group=None, async_op=False):
        calls.append(group)

    # Reinstall over a recording stand-in so the pass-through is observable.
    guard.uninstall_tripwire()
    saved = dist.all_reduce
    dist.all_reduce = fake_all_reduce
    try:
        guard.install_tripwire()
        unknown = FakeGroup()
        dist.all_reduce(torch.ones(1), group=unknown)
        assert calls == [unknown]
    finally:
        guard.uninstall_tripwire()
        dist.all_reduce = saved
        guard.install_tripwire()
    assert guard._ORIGINALS[("torch.distributed", "all_reduce")] is original


def test_install_is_idempotent_and_uninstall_restores(tripwire):
    assert guard.install_tripwire() == []
    assert getattr(dist.all_to_all_single, "_sircl_original", None) is not None
    guard.uninstall_tripwire()
    assert getattr(dist.all_to_all_single, "_sircl_original", None) is None
    assert not guard.tripwire_installed()
    guard.install_tripwire()


def test_resolver_classifies_groups_sircl_did_not_build():
    config = AdapterConfig(Layout.ring(8), tuple(range(4)), ("tp", "dcp"), "topology", "auto", None, None)
    resolve = resolver_for(config, environ={})
    world = resolve((0, 1, 2, 3))
    assert world.policy is NcclPolicy.NONE and "ranks 3-0" in world.reason
    assert resolve((1, 2)).policy is NcclPolicy.ALL
    assert resolve((0, 7)).policy is NcclPolicy.NONE          # position 7 is not on this layout
    full = AdapterConfig(Layout.ring(8), tuple(range(8)), ("tp",), "topology", "auto", None, None)
    assert resolver_for(full, environ={})(tuple(range(8))).policy is NcclPolicy.NONE
    ring_env = {"NCCL_ALGO": "Ring", "NCCL_SKIP_TREE_CONNECT": "1"}
    assert resolver_for(full, environ=ring_env)(tuple(range(8))).policy is NcclPolicy.RING


def test_pynccl_suppression_is_scoped_and_refused_after_the_env_cache(monkeypatch):
    monkeypatch.delenv("VLLM_DISABLE_PYNCCL", raising=False)
    with guard.pynccl_suppressed(True):
        assert os.environ["VLLM_DISABLE_PYNCCL"] == "1"
    assert "VLLM_DISABLE_PYNCCL" not in os.environ
    monkeypatch.setenv("VLLM_DISABLE_PYNCCL", "0")
    with guard.pynccl_suppressed(True):
        assert os.environ["VLLM_DISABLE_PYNCCL"] == "1"
    assert os.environ["VLLM_DISABLE_PYNCCL"] == "0"
    with guard.pynccl_suppressed(False):
        assert os.environ["VLLM_DISABLE_PYNCCL"] == "0"
    vllm = types.ModuleType("vllm")
    envs = types.ModuleType("vllm.envs")
    envs._is_envs_cache_enabled = lambda: True
    vllm.envs = envs
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs)
    with pytest.raises(NcclAcrossRelayError, match="cached its environment"):
        with guard.pynccl_suppressed(True):
            pass


def test_eager_nccl_connections_at_creation_are_refused():
    assert guard.environment_problems({}) == []
    assert guard.environment_problems({"NCCL_RUNTIME_CONNECT": "0"}) == []
    problems = guard.environment_problems({"SIRCL_NCCL": "topology", "VLLM_DISTRIBUTED_USE_SPLIT_GROUP": "1",
                                           "NCCL_RUNTIME_CONNECT": "0"})
    assert len(problems) == 1 and "NCCL_RUNTIME_CONNECT" in problems[0]


def test_split_group_initialization_is_refused_when_sircl_nccl_is_unset():
    """Unset SIRCL_NCCL is never, so vLLM's split-group initialization, which creates NCCL communicators over
    every rank at startup, is refused as under an explicit never."""
    problems = guard.environment_problems({"VLLM_DISTRIBUTED_USE_SPLIT_GROUP": "1"})
    assert len(problems) == 1 and "SIRCL_NCCL=never (the default when it is unset)" in problems[0]
    assert guard.environment_problems({"VLLM_DISTRIBUTED_USE_SPLIT_GROUP": "1", "SIRCL_NCCL": " "}) == problems
    assert guard.environment_problems({"VLLM_DISTRIBUTED_USE_SPLIT_GROUP": "1", "SIRCL_NCCL": "auto"}) == []


def test_ring_algorithm_detection():
    assert guard.ring_algorithm_enforced({"NCCL_ALGO": "ring", "NCCL_SKIP_TREE_CONNECT": "1"})[0]
    assert guard.ring_algorithm_enforced({"NCCL_ALGO": "allreduce:ring,allgather:ring",
                                          "NCCL_SKIP_TREE_CONNECT": "1"})[0]
    enforced, detail = guard.ring_algorithm_enforced({"NCCL_ALGO": "Ring,Tree"})
    assert not enforced and "Tree" in detail and "NCCL_SKIP_TREE_CONNECT" in detail
