"""The adapter inside a vLLM process: entry points, construction and registration order.

These tests run the vLLM-facing modules (platform, general plugin,
``SirclCudaCommunicator``) against a stand-in vLLM package written to disk
(:mod:`vllm_stub`) and SIRCL's emulated sessions, with every rank of a group as
a thread. They cover:

- the platform plugin returns SIRCL's communicator only when enabled;
- the general plugin is idempotent, refuses a bad NCCL environment, and
  installs the RoCE slot shim only against a pinned vLLM;
- a TP4 group on four Sparks of a ring of eight is built without PyNccl (no
  warm-up all-reduce), carries every collective on SIRCL, and its EP sibling
  refuses collectives;
- vLLM's own RoCE slot instance is reused when vLLM builds it;
- SparkRing's other vLLM adapters (the four-rank all-reduce and vocabulary
  adapters installed from ``sitecustomize``, the RoCEnante overlay) give the
  same outcome whether they install before or after SIRCL registers: refused
  on a group they cannot serve, composed on the four-Spark ring.
"""

from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path

import pytest
torch = pytest.importorskip("torch")

import vllm_stub  # noqa: E402
from sparkring_sircl.vllm import adapter as adapter_module  # noqa: E402
from sparkring_sircl.vllm import emulation, guard, pins, platform, plugin, shims, tp4  # noqa: E402
from sparkring_sircl.vllm.adapter import SirclDispatchError, SirclSetupError  # noqa: E402
from sparkring_sircl.vllm.emulation import emulated_groups, reference_sum, run_ranks  # noqa: E402
from sparkring_sircl.vllm.settings import SettingError  # noqa: E402
from sparkring_sircl.vllm.tp_slot import SirclRingAllReduce  # noqa: E402


@pytest.fixture
def stub(tmp_path, monkeypatch):
    vllm_stub.purge()
    root = vllm_stub.write(tmp_path / "site")
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    module = emulation.session_module("sircl_emulated_registration")
    monkeypatch.setitem(sys.modules, "sircl_emulated_registration", module)
    environ = vllm_stub.ThreadEnviron(dict(os.environ))
    monkeypatch.setattr(os, "environ", environ)
    for name in ("VLLM_ENABLE_ROCE_ALLREDUCE", "VLLM_DISABLE_PYNCCL", "VLLM_SPARK_TP4_MODE",
                 "VLLM_SPARK_TP4_VOCAB_MODE", "SIRCL_PEER_ROUTES", "NCCL_ALGO", "SIRCL_VLLM_SHIMS",
                 "SIRCL_RANK_POSITIONS", "SIRCL_TOPOLOGY", "SIRCL_LAYOUT"):
        environ.pop(name, None)
    # NCCL where the cabling allows it (SIRCL_NCCL=topology, which the adapter reads as auto); the tests of
    # SIRCL_NCCL=never and auto set their own, and the adapter's default (never) has a test of its own.
    environ.update({"SIRCL_MODE": "custom", "SIRCL_FABRIC": "ring:8", "SIRCL_NCCL": "topology",
                    "SIRCL_SESSION_MODULE": "sircl_emulated_registration",
                    "SIRCL_ALLREDUCE_CAPACITY_BYTES": str(1 << 20),
                    "SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES": str(64 << 10),
                    "SIRCL_ALLGATHER_MAX_BYTES": str(48 << 10)})
    plugin.reset_for_tests()
    guard.reset()
    shims._INSTALLED.clear()
    adapter_module.reset_regimes_for_tests()
    yield types.SimpleNamespace(root=root, sessions=module, environ=environ)
    for live in adapter_module.live_adapters():
        live.close()
    if guard.tripwire_installed():
        guard.uninstall_tripwire()
    guard.reset()
    shims._INSTALLED.clear()
    adapter_module.reset_regimes_for_tests()
    plugin.reset_for_tests()
    vllm_stub.purge()


def _pin_stub(monkeypatch, root: Path) -> None:
    files = pins.SLOT_FILES
    build = pins.VllmBuild("stub", "0", "test stand-in", pins.hashes(root, files))
    monkeypatch.setattr(pins, "SUPPORTED", (build,))


def _communicators(unique_name, global_ranks, *, per_rank_env=None, world=None, environ=None):
    from sparkring_sircl.vllm.communicator import SirclCudaCommunicator

    groups = emulated_groups(global_ranks)

    def body(rank):
        if per_rank_env is not None:
            environ.override(per_rank_env[rank])
        return SirclCudaCommunicator(groups[rank], torch.device("cpu"), object(), unique_name,
                                     list(global_ranks), world or len(global_ranks))

    return run_ranks(len(global_ranks), body)


def _events():
    return sys.modules["vllm.distributed.device_communicators.cuda_communicator"].EVENTS


def test_platform_plugin_returns_sircl_only_when_enabled(stub):
    stub.environ.pop("SIRCL_MODE")
    assert platform.activate() is None
    stub.environ["SIRCL_MODE"] = "custom"
    assert platform.activate() == "sparkring_sircl.vllm.cuda_platform.SirclCudaPlatform"
    from sparkring_sircl.vllm.communicator import SirclCudaCommunicator
    from sparkring_sircl.vllm.cuda_platform import SirclCudaPlatform
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

    assert SirclCudaPlatform.get_device_communicator_cls() == (
        "sparkring_sircl.vllm.communicator.SirclCudaCommunicator")
    assert issubclass(SirclCudaCommunicator, CudaCommunicator)


def test_general_plugin_is_idempotent_and_fails_loudly(stub, monkeypatch):
    plugin.register()
    assert guard.tripwire_installed()
    plugin.register()                                 # second call: nothing more
    plugin.reset_for_tests()
    stub.environ["VLLM_ENABLE_ROCE_ALLREDUCE"] = "1"
    with pytest.raises(shims.ShimRefused, match="roce_slot refuses to load"):
        plugin.register()                             # the stand-in vLLM is not a pinned build
    _pin_stub(monkeypatch, stub.root)
    plugin.register()
    from vllm.distributed.device_communicators import b12x_roce_all_reduce

    assert b12x_roce_all_reduce.B12xRoceAllReduce is SirclRingAllReduce
    assert shims.installed() == {"roce_slot": "stub"}
    plugin.reset_for_tests()
    stub.environ.update({"VLLM_DISTRIBUTED_USE_SPLIT_GROUP": "1", "NCCL_RUNTIME_CONNECT": "0"})
    with pytest.raises(SirclSetupError, match="NCCL_RUNTIME_CONNECT"):
        plugin.register()
    stub.environ.pop("NCCL_RUNTIME_CONNECT")
    stub.environ.pop("SIRCL_FABRIC")
    with pytest.raises(SettingError, match="SIRCL_FABRIC is not set"):
        plugin.register()


def test_general_plugin_refuses_when_the_platform_plugin_did_not_activate(stub):
    import vllm.platforms

    vllm.platforms.SKIP_OUT_OF_TREE = True
    with pytest.raises(SirclSetupError, match="did not activate"):
        plugin.register()
    assert not guard.tripwire_installed()
    vllm.platforms.SKIP_OUT_OF_TREE = False
    plugin.register()
    assert guard.tripwire_installed()


def test_tp4_path_group_has_no_pynccl_and_every_collective_stays_on_sircl(stub):
    plugin.register()
    comms = _communicators("tp:0", [0, 1, 2, 3])
    assert sorted(_events()) == [("pynccl", rank, "disabled") for rank in range(4)]
    assert "VLLM_DISABLE_PYNCCL" not in os.environ
    for comm in comms:
        assert isinstance(comm.b12x_ar_comm, SirclRingAllReduce)
        assert comm.b12x_ar_comm.check_health == adapter_module.check_all_health
    inputs = [torch.randn(300, 1024).to(torch.bfloat16) for _ in range(4)]
    logits = [torch.randn(4, 1000).to(torch.bfloat16) for _ in range(4)]
    results = run_ranks(4, lambda rank: (comms[rank].all_reduce(inputs[rank]),
                                         comms[rank].all_reduce_in_place(inputs[rank].clone()),
                                         comms[rank].all_gather(logits[rank], -1)))
    for reduced, in_place, gathered in results:
        assert torch.equal(reduced, reference_sum(inputs)) and torch.equal(in_place, reduced)
        assert torch.equal(gathered, torch.cat(logits, dim=-1))
    assert not [event for event in _events() if event[0].startswith("stock")]
    report = comms[0].sircl_report()
    assert report["nccl"] == "none" and report["pynccl"] == "skipped"
    with pytest.raises(SirclDispatchError, match="point-to-point"):
        comms[0].send(torch.ones(2), 1)
    # The default process group of this instance is a path too: the tripwire refuses NCCL on it.
    world = object()
    guard.register(world, adapter_module.resolver_for(adapter_module.AdapterConfig.from_env(4))((0, 1, 2, 3)))
    with pytest.raises(guard.NcclAcrossRelayError):
        torch.distributed.all_reduce(torch.ones(1), group=world)


def test_an_expert_parallel_group_without_a_session_to_share_refuses_collectives(stub):
    plugin.register()
    comms = _communicators("ep:0", [0, 1, 2, 3])
    assert all(comm.sircl.session is None for comm in comms)
    assert all(event[2] == "disabled" for event in _events())
    with pytest.raises(SirclDispatchError, match="no cable between ranks 3-0"):
        comms[2].all_reduce(torch.ones(8, dtype=torch.bfloat16))


def test_vllm_built_roce_slot_is_reused(stub, monkeypatch):
    _pin_stub(monkeypatch, stub.root)
    stub.environ["VLLM_ENABLE_ROCE_ALLREDUCE"] = "1"
    plugin.register()
    routes = [
        "1=rocep1s0f0/roceP2p1s0f0,2=rocep1s0f0/roceP2p1s0f0,3=rocep1s0f0/roceP2p1s0f0",
        "0=rocep1s0f1/roceP2p1s0f1,2=rocep1s0f0/roceP2p1s0f0,3=rocep1s0f0/roceP2p1s0f0",
        "0=rocep1s0f1/roceP2p1s0f1,1=rocep1s0f1/roceP2p1s0f1,3=rocep1s0f0/roceP2p1s0f0",
        "0=rocep1s0f1/roceP2p1s0f1,1=rocep1s0f1/roceP2p1s0f1,2=rocep1s0f1/roceP2p1s0f1",
    ]
    comms = _communicators("tp:0", [0, 1, 2, 3], environ=stub.environ,
                           per_rank_env=[{"SIRCL_PEER_ROUTES": text} for text in routes])
    assert len(stub.sessions.created) == 4
    for comm in comms:
        assert comm.sircl.slot is comm.b12x_ar_comm
    run_ranks(4, lambda rank: comms[rank].all_reduce(torch.ones(4, 512, dtype=torch.bfloat16)))
    assert not [event for event in _events() if event[0].startswith("stock")]


def _install_tp4_adapter(stub, *, pinned: bool, monkeypatch):
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

    module = types.ModuleType("spark_tp4_backend")
    module.__file__ = str(Path(__file__))
    module._mode = lambda: "custom"
    module._eligible = lambda communicator, tensor, mode: tuple(tensor.shape) == (1, 6144)
    monkeypatch.setitem(sys.modules, "spark_tp4_backend", module)
    original = CudaCommunicator.all_reduce

    def spark_all_reduce(self, input_):
        vllm_stub_events = sys.modules["vllm.distributed.device_communicators.cuda_communicator"].EVENTS
        vllm_stub_events.append(("four-rank all_reduce", self.unique_name))
        return input_ * 4

    spark_all_reduce._spark_tp4_backend = True
    spark_all_reduce._spark_original = original
    monkeypatch.setattr(CudaCommunicator, "all_reduce", spark_all_reduce)
    if pinned:
        monkeypatch.setattr(tp4, "PINNED_ADAPTER_SHA256", pins.file_hash(Path(__file__)))


def _install_vocab_adapter(stub, monkeypatch):
    from vllm.distributed.parallel_state import GroupCoordinator

    original = GroupCoordinator._all_gather_out_place

    def spark_vocab_all_gather(self, input_, dim):
        return original(self, input_, dim)

    spark_vocab_all_gather._spark_original = original
    monkeypatch.setattr(GroupCoordinator, "_all_gather_out_place", spark_vocab_all_gather)


def _install_rocenante_overlay(stub, monkeypatch):
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

    original = CudaCommunicator.all_reduce

    def wrapped(self, tensor):
        return original(self, tensor)

    wrapped._rocenante_virtual_diagonal = True
    monkeypatch.setattr(CudaCommunicator, "all_reduce", wrapped)


@pytest.mark.parametrize("sircl_first", [True, False])
@pytest.mark.parametrize("other,env,message", [
    ("tp4", {"VLLM_SPARK_TP4_MODE": "custom"}, "four-rank all-reduce adapter"),
    ("vocab", {"VLLM_SPARK_TP4_VOCAB_MODE": "custom"}, "vocabulary all-gather adapter"),
    ("rocenante", {}, "RoCEnante virtual-diagonal overlay"),
])
def test_other_sparkring_adapters_are_refused_on_a_path_in_either_order(stub, monkeypatch, sircl_first,
                                                                        other, env, message):
    stub.environ.update(env)
    install = {"tp4": lambda: _install_tp4_adapter(stub, pinned=True, monkeypatch=monkeypatch),
               "vocab": lambda: _install_vocab_adapter(stub, monkeypatch),
               "rocenante": lambda: _install_rocenante_overlay(stub, monkeypatch)}[other]
    if sircl_first:
        plugin.register()
        install()
    else:
        install()
        plugin.register()
    with pytest.raises(SirclSetupError, match=message):
        _communicators("tp:0", [0, 1, 2, 3])


@pytest.mark.parametrize("sircl_first", [True, False])
def test_four_rank_sessions_compose_on_the_four_spark_ring_in_either_order(stub, monkeypatch, sircl_first):
    stub.environ.update({"SIRCL_FABRIC": "ring:4", "VLLM_SPARK_TP4_MODE": "custom",
                         "NCCL_ALGO": "Ring", "NCCL_SKIP_TREE_CONNECT": "1"})
    if sircl_first:
        plugin.register()
        _install_tp4_adapter(stub, pinned=True, monkeypatch=monkeypatch)
    else:
        _install_tp4_adapter(stub, pinned=True, monkeypatch=monkeypatch)
        plugin.register()
    comms = _communicators("tp:0", [0, 1, 2, 3])
    assert all(event[2] == "warm-up all-reduce" for event in _events() if event[0] == "pynccl")
    admitted = [torch.ones(1, 6144, dtype=torch.bfloat16) for _ in range(4)]
    other = [torch.ones(8, 512, dtype=torch.bfloat16) for _ in range(4)]
    results = run_ranks(4, lambda rank: (comms[rank].all_reduce(admitted[rank]),
                                         comms[rank].all_reduce(other[rank])))
    assert all(torch.equal(result[0], admitted[0] * 4) for result in results)
    assert all(torch.equal(result[1], reference_sum(other)) for result in results)
    four_rank = [event for event in _events() if event[0] == "four-rank all_reduce"]
    assert len(four_rank) == 4
    rows = comms[0].sircl_report()["decisions"]
    assert {(row["backend"], row["method"]) for row in rows} == {("tp4", "four-rank"), ("sircl", "direct")}


def test_a_four_rank_adapter_installed_but_disabled_is_harmless_on_a_path(stub, monkeypatch):
    _install_tp4_adapter(stub, pinned=False, monkeypatch=monkeypatch)
    stub.environ["VLLM_SPARK_TP4_MODE"] = "disabled"
    plugin.register()
    comms = _communicators("tp:0", [0, 1, 2, 3])
    run_ranks(4, lambda rank: comms[rank].all_reduce(torch.ones(1, 6144, dtype=torch.bfloat16)))
    assert not [event for event in _events() if event[0] == "four-rank all_reduce"]


def test_dcp4_inside_tp8_replaces_the_nccl_combine_with_the_communicator_exchange(stub, monkeypatch):
    build = pins.VllmBuild("stub", "0", "test stand-in",
                           pins.hashes(stub.root, pins.SLOT_FILES + pins.DCP_FILES + pins.DCP_TRANSPORT_FILES))
    monkeypatch.setattr(pins, "SUPPORTED", (build,))
    stub.environ.update({"NCCL_ALGO": "Ring", "NCCL_SKIP_TREE_CONNECT": "1"})
    plugin.register()
    import vllm.distributed.parallel_state as parallel_state

    parallel_state.TP_RANKS = list(range(8))
    comms = _communicators("dcp:0", [0, 1, 2, 3], world=8)
    assert shims.installed()["dcp_all_to_all"] == "stub"
    from vllm.v1.attention.ops import dcp

    outputs = [torch.randn(5, 8, 16).to(torch.bfloat16) for _ in range(4)]

    def combine(rank):
        cp_group = types.SimpleNamespace(device_communicator=comms[rank], world_size=4,
                                         device_group=comms[rank].device_group)
        return dcp.dcp_a2a_lse_reduce(outputs[rank], torch.zeros(5, 8), cp_group)

    results = run_ranks(4, combine)
    for rank, result in enumerate(results):
        expected = sum(out[:, rank * 2:(rank + 1) * 2].float() for out in outputs).to(torch.bfloat16)
        assert torch.equal(result, expected)
    assert not [event for event in _events() if event[0].startswith("stock")]
    # vLLM's own combine calls torch.distributed.all_to_all_single on the DCP group's NCCL group:
    # the tripwire hands the call to SIRCL's carrier, which gives the same result.
    original = dcp.dcp_a2a_lse_reduce._sircl_original

    def unshimmed(rank):
        cp_group = types.SimpleNamespace(device_communicator=comms[rank], world_size=4,
                                         device_group=comms[rank].device_group)
        return original(outputs[rank], torch.zeros(5, 8), cp_group)

    for rank, result in enumerate(run_ranks(4, unshimmed)):
        expected = sum(out[:, rank * 2:(rank + 1) * 2].float() for out in outputs).to(torch.bfloat16)
        assert torch.equal(result, expected)
    rows = {(row["collective"], row["method"]) for row in comms[0].sircl.report()["decisions"]}
    assert ("torch.all_to_all_single", "equal") in rows


def test_a_tp_group_without_a_session_gets_a_hub_slot_for_capture_and_health(stub):
    stub.environ.update({"SIRCL_GROUPS": "dcp", "NCCL_ALGO": "Ring", "NCCL_SKIP_TREE_CONNECT": "1"})
    plugin.register()
    from sparkring_sircl.vllm.communicator import HubSlot

    comms = _communicators("tp:0", list(range(8)))
    assert all(isinstance(comm.b12x_ar_comm, HubSlot) for comm in comms)
    assert all(comm.sircl.session is None for comm in comms)
    assert all(event[2] == "warm-up all-reduce" for event in _events() if event[0] == "pynccl")
    with comms[0].b12x_ar_comm.capture():
        pass
    comms[0].b12x_ar_comm.check_health()
    result = comms[3].all_reduce(torch.ones(4, dtype=torch.bfloat16))
    assert torch.equal(result, torch.ones(4, dtype=torch.bfloat16))
    assert ("stock all_reduce", "tp:0") in _events()


# -- GLM-5.3-Flash mHC prefill row ownership ---------------------------------------------------


def _pin_mhc(monkeypatch, root: Path) -> None:
    build = pins.VllmBuild("stub", "0", "test stand-in", pins.hashes(root, pins.MHC_FILES))
    monkeypatch.setattr(pins, "SUPPORTED", (build,))


class _TpGroup:
    """The calling rank's tensor-parallel group coordinator as the mHC module reads it."""

    def __init__(self, communicator):
        from sparkring_sircl.vllm import groupops

        self.device_communicator = communicator
        self.world_size = communicator.world_size
        self.rank_in_group = communicator.rank_in_group
        self.vote_object = lambda value: groupops.all_gather_object(communicator.cpu_group, value)


def _prefill(comms, partials):
    """Every rank runs one eager forward with row ownership; returns (owner, outputs, PyNccl slot after)."""
    from vllm.distributed import parallel_state
    from vllm.models.glm5next.nvidia import model

    def body(rank):
        parallel_state.set_tp_group(_TpGroup(comms[rank]))
        hidden = torch.zeros(partials[0][rank].shape, dtype=torch.bfloat16)
        owner, outputs = model.prefill(hidden, [layer[rank] for layer in partials])
        return owner, outputs, comms[rank].pynccl_comm

    return run_ranks(len(comms), body)


@pytest.mark.parametrize("scatter", [False, True], ids=["composed", "session-scatter"])
def test_mhc_prefill_sharding_on_a_path_reduce_scatters_and_gathers_through_sircl(stub, monkeypatch, scatter):
    from sparkring_sircl.vllm import mhc

    module = emulation.session_module(f"sircl_emulated_mhc_{scatter}", scatter_available=scatter)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    stub.environ.update({"SIRCL_SESSION_MODULE": module.__name__, "VLLM_GLM53_MHC_PREFILL_SHARD": "1"})
    _pin_mhc(monkeypatch, stub.root)
    plugin.register()
    comms = _communicators("tp:0", [0, 1, 2, 3])
    assert shims.installed()["mhc_prefill_shard"] == "stub"
    from vllm.models.glm5next.nvidia import mhc_prefill_sharding, model

    assert model.maybe_create_mhc_prefill_ownership is mhc_prefill_sharding.maybe_create
    assert all(comm.sircl.limits.scatter_dtypes == (("bfloat16",) if scatter else ()) for comm in comms)
    partials = [[torch.randn(16, 64).to(torch.bfloat16) for _ in range(4)] for _ in range(3)]
    results = _prefill(comms, partials)
    for rank, (owner, outputs, slot) in enumerate(results):
        assert isinstance(owner.comm, mhc.SirclPrefillComm) and (owner.rs_count, owner.ag_count) == (3, 3)
        assert slot is comms[rank].pynccl_comm and slot.disabled          # restored after maybe_create
        for layer, output in enumerate(outputs):
            assert torch.equal(output, reference_sum(partials[layer]))   # exact rows, gathered back
    assert sorted(event[2] for event in _events() if event[0] == "pynccl") == ["disabled"] * 4
    assert not [event for event in _events() if event[0].startswith("stock")]
    rows = {(row["collective"], row["method"]): row["calls"] for row in comms[0].sircl.counters.snapshot()}
    assert rows[("reduce_scatter", "scatter" if scatter else "allreduce_slice")] == 3
    assert sum(calls for (collective, _), calls in rows.items() if collective == "all_gather") == 3


def test_mhc_prefill_sharding_is_refused_where_sircl_cannot_carry_it(stub, monkeypatch):
    stub.environ["VLLM_GLM53_MHC_PREFILL_SHARD"] = "1"
    plugin.register()
    with pytest.raises(SirclSetupError, match="mHC prefill sharding of tp:0.*refuses to load.*"
                                             "VLLM_GLM53_MHC_PREFILL_SHARD=0"):
        _communicators("tp:0", [0, 1, 2, 3])                 # the stand-in is no pinned build


def test_mhc_prefill_sharding_is_refused_on_a_path_group_without_a_session(stub, monkeypatch):
    _pin_mhc(monkeypatch, stub.root)
    stub.environ.update({"VLLM_GLM53_MHC_PREFILL_SHARD": "1", "SIRCL_GROUPS": "dcp"})
    plugin.register()
    with pytest.raises(SirclSetupError, match="no SIRCL session to carry them"):
        _communicators("tp:0", [0, 1, 2, 3])


def test_a_group_with_working_pynccl_keeps_vllms_own_mhc_path(stub, monkeypatch):
    from sparkring_sircl.vllm import mhc

    stub.environ["VLLM_GLM53_MHC_PREFILL_SHARD"] = "1"
    _pin_mhc(monkeypatch, stub.root)
    plugin.register()
    comms = _communicators("tp:0", [0, 1])                  # a pair: NCCL may run, PyNccl is built
    assert comms[0].pynccl_comm.available and mhc.prefill_comm(comms[0]) is None
    assert "mhc_prefill_shard" not in shims.installed()
    assert mhc.requested({"VLLM_GLM53_MHC_PREFILL_SHARD": "0"}) is False
    with pytest.raises(ValueError, match="int"):
        mhc.requested({"VLLM_GLM53_MHC_PREFILL_SHARD": "yes"})


def test_vllms_own_mhc_path_on_a_path_group_fails_on_every_rank_without_reaching_nccl(stub, monkeypatch):
    stub.environ["VLLM_GLM53_MHC_PREFILL_SHARD"] = "1"
    _pin_mhc(monkeypatch, stub.root)
    plugin.register()
    comms = _communicators("tp:0", [0, 1, 2, 3])
    from vllm.distributed import parallel_state
    from vllm.models.glm5next.nvidia import mhc_prefill_sharding

    original = mhc_prefill_sharding.maybe_create._sircl_original

    def body(rank):
        parallel_state.set_tp_group(_TpGroup(comms[rank]))
        try:
            original(None, torch.zeros(16, 64, dtype=torch.bfloat16), None)
        except RuntimeError as exc:
            return str(exc)
        return "no error"

    assert all("requires the enabled TP PyNccl communicator" in text for text in run_ranks(4, body))
    assert not [event for event in _events() if event[0].startswith("stock")]


def _pin_worker(monkeypatch, root: Path) -> None:
    build = pins.VllmBuild("stub", "0", "test stand-in", pins.hashes(root, pins.WORKER_FILES))
    monkeypatch.setattr(pins, "SUPPORTED", (build,))


def _regimes(comms):
    return [comm.sircl.session.wait_regime for comm in comms]


def test_sessions_serve_after_a_completed_step_and_return_to_startup_around_worker_start_up_work(
        stub, monkeypatch):
    from sparkring_sircl.vllm import receipt

    _pin_worker(monkeypatch, stub.root)
    plugin.register()
    comms = _communicators("tp:0", [0, 1, 2, 3])
    assert shims.installed()["worker_regimes"] == "stub"
    from vllm.v1.worker.gpu_worker import Worker

    assert _regimes(comms) == ["startup"] * 4                 # setup
    assert comms[0].sircl.report()["wait"] == "startup:600s"
    comms[0].b12x_ar_comm.check_health()                     # a step before warm-up: not armed yet
    assert _regimes(comms) == ["startup"] * 4 and not adapter_module.serving_armed()
    seen = []

    def warm_up_step(name):
        comms[0].b12x_ar_comm.check_health()                 # vLLM's warm-up runs real steps
        seen.append((name, _regimes(comms)))

    Worker.DURING = warm_up_step
    worker = Worker()
    worker.compile_or_warm_up_model()
    assert seen == [("compile_or_warm_up_model", ["startup"] * 4)]
    assert _regimes(comms) == ["startup"] * 4 and adapter_module.serving_armed()
    comms[0].b12x_ar_comm.check_health()                     # the engine's first real step
    assert _regimes(comms) == ["serving"] * 4                 # every session of the process
    report = comms[3].sircl.report()
    assert report["wait"] == "serving:20s" and report["session_stats"]["wait_regime"] == "serving"
    assert "wait=serving:20s" in receipt.line(report)

    def during(name):
        comms[1].b12x_ar_comm.check_health()                 # an asynchronous step output completes meanwhile
        seen.append((name, _regimes(comms)))

    Worker.DURING = during
    for name in ("sleep", "wake_up", "reload_weights", "update_weights", "profile",
                 "determine_available_memory"):
        seen.clear()
        getattr(worker, name)(*(({},) if name == "update_weights" else ()))
        assert seen == [(name, ["startup"] * 4)]
        assert _regimes(comms) == ["startup"] * 4             # until the next completed step
        comms[2].b12x_ar_comm.check_health()
        assert _regimes(comms) == ["serving"] * 4
    Worker.DURING = None
    worker.execute_model(None)                               # steps are not wrapped
    assert _regimes(comms) == ["serving"] * 4 and not adapter_module.startup_held()


def test_an_unpinned_worker_warns_and_sessions_keep_the_startup_regime(stub, caplog):
    plugin.register()
    with caplog.at_level(logging.WARNING, logger="sircl.vllm.communicator"):
        comms = _communicators("tp:0", [0, 1, 2, 3])
    assert "worker_regimes" not in shims.installed()
    assert any("worker_regimes refuses to load" in record.getMessage() for record in caplog.records)
    comms[0].b12x_ar_comm.check_health()
    assert _regimes(comms) == ["startup"] * 4                 # never armed without the shim
    assert adapter_module.enter_serving_all("a test") == ["tp:0"] * 4
    assert adapter_module.enter_serving_all("a test") == []   # already serving: nothing changes
    with adapter_module.startup_all("a test"):
        comms[0].b12x_ar_comm.check_health()
        assert _regimes(comms) == ["startup"] * 4
    comms[0].b12x_ar_comm.check_health()
    assert _regimes(comms) == ["startup"] * 4


def test_a_group_with_the_tensor_parallel_ranks_shares_its_session_and_carries_torch_reductions(stub):
    """GLM-5.3 at TP8 with SIRCL_NCCL=never: the EP group (same eight ranks) all-reduces weight amax with max."""
    import torch.distributed as dist

    stub.environ.update({"SIRCL_NCCL": "never", "SIRCL_GROUPS": "tp"})
    plugin.register()
    tp = _communicators("tp:0", list(range(8)))
    ep = _communicators("ep:0", list(range(8)))
    assert all(comm.sircl.shared_from == "tp:0" and comm.sircl.session is tp[rank].sircl.session
               for rank, comm in enumerate(ep))
    assert ep[0].sircl.report()["session"] == "shared:tp:0"
    amax = [torch.tensor([float(rank), 7.0 - rank, -1.0]) for rank in range(8)]

    def reduce(rank):
        value = amax[rank].clone()
        assert dist.all_reduce(value, op=dist.ReduceOp.MAX, group=ep[rank].device_group) is None
        low = amax[rank].clone()
        work = dist.all_reduce(low, op=dist.ReduceOp.MIN, group=ep[rank].device_group, async_op=True)
        assert work.wait()
        total = amax[rank].clone()
        dist.all_reduce(total, group=tp[rank].device_group)
        shared = torch.full((3,), float(rank))
        dist.broadcast(shared, src=5, group=ep[rank].device_group)
        return value, low, total, shared

    for value, low, total, shared in run_ranks(8, reduce):
        assert value.tolist() == [7.0, 7.0, -1.0] and low.tolist() == [0.0, 0.0, -1.0]
        assert total.tolist() == [28.0, 28.0, -8.0] and shared.tolist() == [5.0, 5.0, 5.0]
    rows = {(row["collective"], row["method"]) for row in ep[0].sircl.report()["decisions"]}
    assert {("torch.all_reduce", "max"), ("torch.all_reduce", "min"), ("torch.broadcast", "bytes")} <= rows
    assert not [event for event in _events() if event[0].startswith("stock")]

    def unsupported(rank):
        with pytest.raises(guard.NcclAcrossRelayError) as raised:
            dist.all_reduce(amax[rank].clone(), op=dist.ReduceOp.PRODUCT, group=ep[rank].device_group)
        return str(raised.value)

    message = run_ranks(8, unsupported)[0]
    assert "group ep:0" in message and "SIRCL_NCCL=auto" in message and "NCCL_SKIP_TREE_CONNECT=1" in message
    assert "SIRCL also carries the torch.distributed calls all_reduce (sum, max, min), broadcast" in message


def test_a_group_with_other_ranks_has_no_session_and_its_refusal_names_the_remedies(stub):
    import torch.distributed as dist

    # Without point-to-point channels for PP groups (SIRCL_P2P_GROUPS): the PP group's collectives are refused.
    stub.environ.update({"SIRCL_NCCL": "never", "SIRCL_GROUPS": "tp", "SIRCL_P2P_GROUPS": "tp"})
    plugin.register()
    _communicators("tp:0", list(range(8)))
    pp = _communicators("pp:0", [0, 4], world=8)          # not the tensor-parallel ranks: no shared session
    assert all(comm.sircl.session is None and comm.sircl.shared_from is None for comm in pp)

    def refused(rank):
        with pytest.raises(guard.NcclAcrossRelayError) as raised:
            dist.all_reduce(torch.ones(2), group=pp[rank].device_group)
        return str(raised.value)

    message = run_ranks(2, refused)[0]
    assert "give the group a SIRCL session" in message and "shares that group's session" in message


def _pin_dcp(monkeypatch, root: Path) -> None:
    files = pins.SLOT_FILES + pins.DCP_FILES + pins.DCP_TRANSPORT_FILES
    monkeypatch.setattr(pins, "SUPPORTED", (pins.VllmBuild("stub", "0", "test stand-in", pins.hashes(root, files)),))


def _dcp_group(comm, rank_in_group):
    """The DCP group coordinator as vLLM's MLA DCP manager reads it."""
    return types.SimpleNamespace(device_communicator=comm, world_size=4, rank_in_group=rank_in_group,
                                 device_group=comm.device_group,
                                 all_gather=lambda tensor, dim: comm.all_gather(tensor, dim))


def test_tp8_with_both_dcp4_groups_carries_every_dcp_collective_on_sircl(stub, monkeypatch):
    """GLM-5.3 at TP8 with DCP4 and SIRCL_NCCL=never: sessions on DCP groups 0-3 and 4-7 at once."""
    _pin_dcp(monkeypatch, stub.root)
    stub.environ.update({"SIRCL_NCCL": "never", "SIRCL_GROUPS": "tp,dcp"})
    plugin.register()
    import vllm.distributed.parallel_state as parallel_state

    parallel_state.TP_RANKS = list(range(8))
    tp = _communicators("tp:0", list(range(8)))
    low = _communicators("dcp:0", [0, 1, 2, 3], world=8)
    high = _communicators("dcp:0", [4, 5, 6, 7], world=8)
    assert {shim for shim in ("dcp_all_to_all", "dcp_b12x_transport")} <= set(shims.installed())
    assert all(comm.sircl.session is not None and comm.sircl.shared_from is None for comm in low + high)
    assert low[0].sircl.record["fabric"] == high[0].sircl.record["fabric"] == "cycle:0-1-2-3-4-5-6-7"
    assert all(comm.sircl.session.prepared_links == [True] for comm in low + high)   # link kernels before capture
    assert low[3].sircl.placement.positions == (0, 1, 2, 3) and high[0].sircl.placement.positions == (4, 5, 6, 7)
    assert all(comm.sircl.session is not tp[0].sircl.session for comm in low + high)
    from vllm.v1.attention.ops import dcp

    # 30 values per head and two packed LSE slots: 64-byte rows, so the exchange is whole packs per rank.
    outputs = [torch.randn(5, 8, 30).to(torch.bfloat16) for _ in range(8)]
    kv = [torch.randn(3, 64).to(torch.bfloat16) for _ in range(8)]

    def body(rank):
        comm = (low if rank < 4 else high)[rank % 4]
        manager = dcp.MLADCPManager(_dcp_group(comm, rank % 4), use_b12x=True)
        assert manager.b12x_transport is None                # the in-machine transport is never chosen
        combined = manager.combine(outputs[rank], torch.zeros(5, 8))
        query = manager.query_gather(torch.full((2, 2, 4), float(rank)))
        manager.init_kv_gather(None, 0)
        gathered = torch.empty(12, 64, dtype=torch.bfloat16)
        assert manager.kv_gather(gathered, kv[rank]) is None
        return combined, query, gathered

    results = run_ranks(8, body)
    for rank, (combined, query, gathered) in enumerate(results):
        members = range(0, 4) if rank < 4 else range(4, 8)
        local = rank % 4
        expected = sum(outputs[m][:, local * 2:(local + 1) * 2].float() for m in members).to(torch.bfloat16)
        assert torch.equal(combined, expected)
        assert query[0, :, 0].tolist() == [float(m) for m in members for _ in range(2)]
        assert torch.equal(gathered, torch.cat([kv[m] for m in members]))
    rows = {(row["collective"], row["method"]) for row in low[1].sircl.report()["decisions"]}
    assert ("torch.all_gather_into_tensor", "bytes") in rows and ("all_to_all", "scatter") in rows
    assert not [event for event in _events() if event[0].startswith("stock")]
    # A group SIRCL does not own keeps vLLM's transport choice.
    plain = types.SimpleNamespace(device_communicator=types.SimpleNamespace(sircl=None), world_size=4)
    assert dcp.MLADCPManager(plain, use_b12x=True).b12x_transport is not None


@pytest.mark.parametrize("setting,text", [
    ({"VLLM_USE_DIRECT_DCP_KV_GATHER": "1"}, "VLLM_USE_DIRECT_DCP_KV_GATHER=1 maps peer GPU memory"),
    ({"VLLM_USE_DIRECT_DCP_A2A": "true"}, "VLLM_USE_DIRECT_DCP_A2A=true maps peer GPU memory"),
    ({"VLLM_MLA_PREFILL_DCP_OVERLAP": "1"}, "gathers the prefill context on PyNccl directly"),
])
def test_dcp_settings_that_bypass_sircl_are_refused_at_setup(stub, monkeypatch, setting, text):
    _pin_dcp(monkeypatch, stub.root)
    stub.environ.update({"SIRCL_NCCL": "never", "SIRCL_GROUPS": "tp,dcp", **setting})
    plugin.register()
    import vllm.distributed.parallel_state as parallel_state

    parallel_state.TP_RANKS = list(range(8))
    with pytest.raises(SirclSetupError, match=text):
        _communicators("dcp:0", [0, 1, 2, 3], world=8)


# -- fused all-reduce + residual add + RMSNorm -------------------------------------------------


def _pin_norm(monkeypatch, root: Path) -> None:
    build = pins.VllmBuild("stub", "0", "test stand-in", pins.hashes(root, pins.NORM_FILES))
    monkeypatch.setattr(pins, "SUPPORTED", (build,))


class _StandInKernel:
    """``FusedAddRmsNorm``'s interface on an emulated session: one session all-reduce, then the stand-in norm."""

    def __init__(self, session, hidden, max_rows):
        self.session = session
        self.hidden = hidden
        self.max_rows = max_rows

    def supports_fused_add_rms_norm(self):
        return not self.session.poisoned

    def allreduce_add_rms_norm(self, hidden_states, residual, weight, epsilon):
        from vllm.ir.ops.layernorm import add_rms_norm

        rows = hidden_states.numel() // self.hidden
        if (hidden_states.dtype != torch.bfloat16 or hidden_states.shape[-1] != self.hidden
                or not 1 <= rows <= self.max_rows):
            return None
        reduced = self.session.all_reduce(hidden_states.contiguous())
        normed, z = add_rms_norm(reduced, residual, weight, epsilon)
        residual.copy_(z)
        return normed, residual

    def stats(self):
        return {"hidden": self.hidden, "max_rows": self.max_rows, "prepared": ["oneshot", "twoshot"]}


def _stand_in_kernels(monkeypatch, *, hidden, max_rows):
    from sparkring_sircl.vllm import norm_fusion

    monkeypatch.setattr(norm_fusion, "bind_kernel",
                        lambda session, size: _StandInKernel(session, size, max_rows))
    monkeypatch.setattr(norm_fusion, "model_hidden_size", lambda: hidden)


def test_fused_norm_runs_the_post_all_reduce_norm_as_one_collective_with_the_unfused_bits(stub, monkeypatch):
    """GLM-5.3 at TP8 with SIRCL_FUSED_NORM=1: decode-sized norm sites fuse, others keep vLLM's helper."""
    from sparkring_sircl.vllm import norm_fusion, receipt

    hidden = 64
    _pin_norm(monkeypatch, stub.root)
    _stand_in_kernels(monkeypatch, hidden=hidden, max_rows=4)
    stub.environ.update({"SIRCL_NCCL": "never", "SIRCL_FUSED_NORM": "1"})
    from vllm.models.common.ops import fused_allreduce_rms_norm as helper
    from vllm.models.deepseek_v32.nvidia import model    # binds the helper before the shim exists

    plugin.register()
    comms = _communicators("tp:0", list(range(8)))
    assert shims.installed()["fused_allreduce_rms_norm"] == "stub"
    original = helper.fused_allreduce_rms_norm._sircl_original
    assert model.fused_allreduce_rms_norm is helper.fused_allreduce_rms_norm is not original
    from vllm.models.deepseek_v32.nvidia import mtp       # imported after the shim: binds the wrapper

    assert mtp.fused_allreduce_rms_norm is helper.fused_allreduce_rms_norm
    assert all(isinstance(comm.sircl_fused_add_rms_norm, norm_fusion.SirclFusedNorm) for comm in comms)
    assert comms[0].sircl.record["fused_norm"] == "on"
    assert comms[0].sircl.record["fused_norm_detail"]["provider"] == "vllm_c"
    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.layernorm import RMSNorm

    class Shifted(RMSNorm):
        def forward_native(self, x, residual=None):
            return super().forward_native(x, residual)

    norm, shifted = RMSNorm(hidden, eps=1e-5), Shifted(hidden, eps=1e-5)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(hidden).to(torch.bfloat16))
        shifted.weight.copy_(norm.weight)
    partials = {rows: [torch.randn(rows, hidden).to(torch.bfloat16) for _ in range(8)] for rows in (3, 6)}
    residuals = {rows: torch.randn(rows, hidden).to(torch.bfloat16) for rows in (3, 6)}

    def body(rank):
        parallel_state.set_tp_group(_TpGroup(comms[rank]))
        out = {}
        for rows in (3, 6):                           # 3 rows fuse; 6 exceed the kernel's row limit
            residual = residuals[rows].clone()
            out[("fused", rows)] = model.layer_boundary(partials[rows][rank].clone(), residual, norm)
            residual = residuals[rows].clone()
            out[("unfused", rows)] = original(partials[rows][rank].clone(), residual, norm)
        residual = residuals[3].clone()
        out[("subclass", 3)] = model.layer_boundary(partials[3][rank].clone(), residual, shifted)
        return out

    results = run_ranks(8, body)
    from vllm.ir.ops.layernorm import add_rms_norm

    for path, rows in [("fused", 3), ("unfused", 3), ("fused", 6), ("unfused", 6), ("subclass", 3)]:
        expected = add_rms_norm(reference_sum(partials[rows]), residuals[rows], norm.weight.data, 1e-5)
        for out in results:
            assert torch.equal(out[(path, rows)][0], expected[0])     # normed rows
            assert torch.equal(out[(path, rows)][1], expected[1])     # new residual
    rows = {(row["collective"], row["method"]): row["calls"] for row in comms[5].sircl.counters.snapshot()}
    assert rows[("all_reduce", "fused_rms_norm")] == 1     # the 3-row site; the others ran the plain path
    assert rows[("all_reduce", "direct")] == 4
    assert not [event for event in _events() if event[0].startswith("stock")]
    assert "fused_norm=on" in receipt.line(comms[5].sircl.report(stats=False))


@pytest.mark.parametrize("case,text", [
    ("native", "with the native provider here"),
    ("batch-invariant", "batch-invariant"),
    ("unpinned", "fused_allreduce_rms_norm refuses to load"),
    ("hub", "no SIRCL session of its own"),
    ("kernel", "SIRCL_FUSED_NORM=1 cannot run"),
])
def test_fused_norm_is_refused_on_every_rank_where_it_cannot_match_vllms_own_path(stub, monkeypatch, case,
                                                                                  text):
    from sparkring_sircl.vllm import norm_fusion

    if case != "unpinned":
        _pin_norm(monkeypatch, stub.root)
    if case != "kernel":
        _stand_in_kernels(monkeypatch, hidden=64, max_rows=4)
    else:                                # the real kernels cannot bind an emulated session
        monkeypatch.setattr(norm_fusion, "model_hidden_size", lambda: 256)
    stub.environ.update({"SIRCL_FUSED_NORM": "1"})
    if case == "hub":                    # NCCL may run on the full ring; the TP group gets no session
        stub.environ.update({"SIRCL_GROUPS": "dcp", "NCCL_ALGO": "Ring", "NCCL_SKIP_TREE_CONNECT": "1"})
    else:
        stub.environ["SIRCL_NCCL"] = "never"
    plugin.register()
    if case == "native":
        import vllm.ir.ops.layernorm as ir_layernorm

        monkeypatch.setattr(ir_layernorm, "PROVIDER", "native")
    if case == "batch-invariant":
        import vllm.envs

        monkeypatch.setattr(vllm.envs, "VLLM_BATCH_INVARIANT", True)
    with pytest.raises(SirclSetupError, match=text) as caught:
        _communicators("tp:0", list(range(8)))
    assert "unset SIRCL_FUSED_NORM" in str(caught.value) and "rank 7" in str(caught.value)


def test_fused_norm_is_off_by_default_and_its_setting_is_strict(stub):
    from sparkring_sircl.vllm import receipt, settings

    stub.environ["SIRCL_NCCL"] = "never"
    plugin.register()
    comms = _communicators("tp:0", [0, 1, 2, 3])
    assert all(comm.sircl_fused_add_rms_norm is None for comm in comms)
    assert "fused_allreduce_rms_norm" not in shims.installed()
    assert comms[0].sircl.record["fused_norm"] == "off"
    assert "fused_norm=off" in receipt.line(comms[0].sircl.record)
    assert settings.fused_norm({}) is False and settings.fused_norm({"SIRCL_FUSED_NORM": "0"}) is False
    assert settings.fused_norm({"SIRCL_FUSED_NORM": " 1 "}) is True
    with pytest.raises(SettingError, match="SIRCL_FUSED_NORM must be 0 or 1"):
        settings.fused_norm({"SIRCL_FUSED_NORM": "on"})


# -- Qwen3.8 hyper-connection prefill row ownership --------------------------------------------------


def _pin_qwen_hc(monkeypatch, root: Path) -> None:
    build = pins.VllmBuild("stub", "0", "test stand-in", pins.hashes(root, pins.QWEN_HC_FILES))
    monkeypatch.setattr(pins, "SUPPORTED", (build,))


def _qwen_prefill(comms, partials, rows):
    """Every rank runs one eager prefill with row ownership; returns (owner, outputs, PyNccl slot after)."""
    from vllm.distributed import parallel_state
    from vllm.models.qwen4_exp.nvidia import model

    def body(rank):
        parallel_state.set_tp_group(_TpGroup(comms[rank]))
        owner, outputs = model.prefill(rows, [layer[rank] for layer in partials])
        return owner, outputs, comms[rank].pynccl_comm

    return run_ranks(len(comms), body)


@pytest.mark.parametrize("scatter", [False, True], ids=["composed", "session-scatter"])
def test_qwen_hc_prefill_rows_on_a_path_reduce_scatter_and_gather_through_sircl(stub, monkeypatch, scatter):
    from sparkring_sircl.vllm import mhc

    module = emulation.session_module(f"sircl_emulated_qwen_hc_{scatter}", scatter_available=scatter)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    stub.environ.update({"SIRCL_SESSION_MODULE": module.__name__, "VLLM_QWEN3_8_HC_PREFILL_MODE": "shard"})
    _pin_qwen_hc(monkeypatch, stub.root)
    plugin.register()
    comms = _communicators("tp:0", [0, 1, 2, 3])
    assert shims.installed()["qwen_hc_prefill_shard"] == "stub"
    assert all(comm.sircl.limits.scatter_dtypes == (("bfloat16",) if scatter else ()) for comm in comms)
    assert comms[0].sircl.record["mhc"] == "sircl"
    partials = [[torch.randn(16, 64).to(torch.bfloat16) for _ in range(4)] for _ in range(3)]
    results = _qwen_prefill(comms, partials, 16)
    for rank, (owner, outputs, slot) in enumerate(results):
        assert isinstance(owner.group, mhc.SirclPrefillComm) and (owner.reductions, owner.gathers) == (3, 3)
        assert slot is comms[rank].pynccl_comm and slot.disabled          # restored after create
        for layer, output in enumerate(outputs):
            assert torch.equal(output, reference_sum(partials[layer]))   # exact rows, gathered back
    assert sorted(event[2] for event in _events() if event[0] == "pynccl") == ["disabled"] * 4
    assert not [event for event in _events() if event[0].startswith("stock")]
    rows = {(row["collective"], row["method"]): row["calls"] for row in comms[0].sircl.counters.snapshot()}
    assert rows[("reduce_scatter", "scatter" if scatter else "allreduce_slice")] == 3
    assert sum(calls for (collective, _), calls in rows.items() if collective == "all_gather") == 3


def test_vllms_own_qwen_hc_path_on_a_path_group_fails_on_every_rank_without_reaching_nccl(stub, monkeypatch):
    stub.environ["VLLM_QWEN3_8_HC_PREFILL_MODE"] = "shard"
    _pin_qwen_hc(monkeypatch, stub.root)
    plugin.register()
    comms = _communicators("tp:0", [0, 1, 2, 3])
    from vllm.distributed import parallel_state
    from vllm.models.qwen4_exp.nvidia import hc_prefill

    original = hc_prefill.create._sircl_original

    def body(rank):
        parallel_state.set_tp_group(_TpGroup(comms[rank]))
        try:
            original(None, 16)
        except RuntimeError as exc:
            return str(exc)
        return "no error"

    assert all("requires the TP NCCL communicator" in text for text in run_ranks(4, body))
    assert not [event for event in _events() if event[0].startswith("stock")]


@pytest.mark.parametrize("environment, match", [
    ({}, "hyper-connection prefill row ownership of tp:0.*refuses to load.*"
         "--env VLLM_QWEN3_8_HC_PREFILL_MODE=off"),
    ({"SIRCL_GROUPS": "dcp"}, "VLLM_QWEN3_8_HC_PREFILL_MODE=shard calls PyNccl.*no SIRCL session to carry them"),
], ids=["unpinned", "no-session"])
def test_qwen_hc_prefill_rows_are_refused_where_sircl_cannot_carry_them(stub, monkeypatch, environment, match):
    if environment:
        _pin_qwen_hc(monkeypatch, stub.root)
    stub.environ.update({"VLLM_QWEN3_8_HC_PREFILL_MODE": "shard", **environment})
    plugin.register()
    with pytest.raises(SirclSetupError, match=match):
        _communicators("tp:0", [0, 1, 2, 3])


def test_a_pair_keeps_vllms_own_qwen_hc_path_and_the_mode_is_read_as_vllm_reads_it(stub, monkeypatch):
    from sparkring_sircl.vllm import qwen_hc

    stub.environ["VLLM_QWEN3_8_HC_PREFILL_MODE"] = "shard"
    _pin_qwen_hc(monkeypatch, stub.root)
    plugin.register()
    comms = _communicators("tp:0", [0, 1])                  # a pair: NCCL may run, PyNccl is built
    assert comms[0].pynccl_comm.available and "qwen_hc_prefill_shard" not in shims.installed()
    assert comms[0].sircl.record["mhc"] == "pynccl"
    assert qwen_hc.requested({"VLLM_QWEN3_8_HC_PREFILL_MODE": "shard"}) and not qwen_hc.requested({})
    # vLLM reads the variable with os.getenv and no normalization; only "shard" gives rows to their owners.
    assert not qwen_hc.requested({"VLLM_QWEN3_8_HC_PREFILL_MODE": "control"})
    assert not qwen_hc.requested({"VLLM_QWEN3_8_HC_PREFILL_MODE": "Shard"})


# -- session schedules and the link collectives ------------------------------------------------------


@pytest.mark.parametrize("environment, links, text", [
    ({}, True, "large:auto,gather:auto,scatter:pieces"),
    ({"SIRCL_LARGE_SCHEDULE": "ring", "SIRCL_GATHER_SCHEDULE": "ring", "SIRCL_SCATTER_SCHEDULE": "ring"}, True,
     "large:ring,gather:ring,scatter:ring"),
    ({"SIRCL_LARGE_SCHEDULE": "pieces", "SIRCL_GATHER_SCHEDULE": "pieces"}, False,
     "large:pieces,gather:pieces,scatter:pieces"),
], ids=["defaults", "ring", "pieces"])
def test_every_prepare_compiles_the_link_collectives_when_a_schedule_can_use_them(stub, monkeypatch, environment,
                                                                                links, text):
    """Chain and ring kernels compile in prepare, before any capture, never while peers wait."""
    from sparkring_sircl.vllm import receipt, sessionapi

    module = emulation.session_module("sircl_emulated_links", scatter_available=True)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    stub.environ.update({"SIRCL_SESSION_MODULE": module.__name__, "VLLM_GLM53_MHC_PREFILL_SHARD": "1",
                         **environment})
    _pin_mhc(monkeypatch, stub.root)
    plugin.register()
    comms = _communicators("tp:0", [0, 1, 2, 3])
    for comm in comms:
        # the slot's dtypes, the extra dtypes of a group NCCL may not run, and the BF16 reduce-scatter
        assert comm.sircl.session.prepared_links == [links] * 3
    assert comms[0].sircl.record["schedules"] == text
    assert f"schedules={text}" in receipt.line(comms[0].sircl.record)
    assert comms[0].sircl.report()["session_stats"]["large_schedule"] == text.split(",")[0].split(":")[1]

    class Older:
        large_schedule = "ring"

        def prepare(self, dtypes, *, padded_gather=False, algorithms=None, scatter=False):
            return None

    assert sessionapi.link_keywords(Older()) == {}
    assert sessionapi.schedule_text(Older()) is None


@pytest.mark.parametrize("environment, links, chain_min, ring_min", [
    ({"SIRCL_LINK_SLOT_BYTES": "1048576"},
     "slot:1048576,chunk:524288,gather:524288,scatter:524288,reduce:524288",
     "reduce:8388608,gather:8388608,scatter:4194304", "reduce:4194304,gather:8388608,scatter:4194304"),
    # A gather piece of 1 MiB grows the slot to 1 MiB without SIRCL_LINK_SLOT_BYTES; the others keep 512 KiB.
    ({"SIRCL_GATHER_LINK_CHUNK_BYTES": "1048576", "SIRCL_SCATTER_LINK_CHUNK_BYTES": "262144",
      "SIRCL_RING_MIN_BYTES": "0", "SIRCL_CHAIN_MIN_BYTES": "2097152"},
     "slot:1048576,chunk:524288,gather:1048576,scatter:262144,reduce:524288",
     "reduce:2097152,gather:2097152,scatter:2097152", "reduce:0,gather:0,scatter:0"),
], ids=["link-slot", "pieces-and-minimums"])
def test_the_receipts_state_the_sessions_link_pieces_and_ring_minimum(stub, monkeypatch, environment, links,
                                                                      chain_min, ring_min):
    """--link-slot, the per-collective --*-link-chunk options, --chain-min and --ring-min set the session's
    variables; the receipt shows the slot, the link piece and each collective's piece, and each collective's
    chain and ring minimums."""
    from sparkring_sircl.vllm import receipt, sessionapi

    module = emulation.session_module("sircl_emulated_link_sizes")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    stub.environ.update({"SIRCL_SESSION_MODULE": module.__name__, **environment})
    plugin.register()
    comms = _communicators("tp:0", [0, 1, 2, 3])
    record = comms[0].sircl.record
    assert (record["links"], record["chain_min"], record["ring_min"]) == (links, chain_min, ring_min)
    fields = receipt.line(record).split()
    position = fields.index(f"ring_min={ring_min}")
    assert fields[position - 1] == f"chain_min={chain_min}" and fields[position + 1] == f"links={links}"
    assert fields[position - 2].startswith("schedules=")
    stats = comms[0].sircl.report()["session_stats"]
    assert stats["link_chunks"]["gather"] == int(links.split("gather:")[1].split(",")[0])
    assert {key: str(value) for key, value in stats["ring_mins"].items()} == dict(
        item.split(":") for item in ring_min.split(","))
    assert stats["chain_min_bytes"] == (2097152 if "SIRCL_CHAIN_MIN_BYTES" in environment else None)

    class Older:
        link_slot_bytes = 1 << 20
        link_chunk_bytes = 1 << 19

    assert sessionapi.link_text(Older()) == "slot:1048576,chunk:524288"     # no per-collective pieces
    del Older.link_chunk_bytes
    assert sessionapi.link_text(Older()) is None and sessionapi.ring_min(Older()) is None
    Older.ring_min_bytes, Older.chain_min_bytes = 2 << 20, 2 << 20      # one size for all three collectives
    assert (sessionapi.ring_min(Older()), sessionapi.chain_min(Older())) == (2 << 20, 2 << 20)


@pytest.mark.parametrize("environment, limit", [({}, 131072), ({"SIRCL_ONESHOT_MAX_BYTES": "65536"}, 65536)],
                         ids=["default", "oneshot-max"])
def test_the_receipts_state_the_sessions_oneshot_limit(stub, monkeypatch, environment, limit):
    """--oneshot-max sets SIRCL_ONESHOT_MAX_BYTES; the receipt shows the limit the session read."""
    from sparkring_sircl.vllm import receipt, sessionapi

    module = emulation.session_module("sircl_emulated_oneshot_limit")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    stub.environ.update({"SIRCL_SESSION_MODULE": module.__name__, **environment})
    plugin.register()
    comms = _communicators("tp:0", [0, 1, 2, 3])
    assert comms[0].sircl.record["oneshot_max"] == limit
    fields = receipt.line(comms[0].sircl.record).split()
    position = fields.index(f"oneshot_max={limit}")
    assert fields[position - 1].startswith("dispatch=") and fields[position + 1].startswith("gather=")
    assert comms[0].sircl.report()["session_stats"]["oneshot_max_bytes"] == limit

    class Older:
        oneshot_max_bytes = True

    assert sessionapi.oneshot_limit(Older()) is None and sessionapi.oneshot_limit(object()) is None


# -- vLLM micro-batching ----------------------------------------------------------------------------


def _vllm_config(monkeypatch, **parallel):
    """Make the communicator read a vLLM config with these parallel settings (the module _communicators uses)."""
    import importlib

    communicator = importlib.import_module("sparkring_sircl.vllm.communicator")   # purged between tests
    config = types.SimpleNamespace(parallel_config=types.SimpleNamespace(**parallel))
    monkeypatch.setattr(communicator, "_current_vllm_config", lambda: config)


@pytest.mark.parametrize("parallel, ranks, refused", [
    ({"enable_dbo": True, "ubatch_size": 0}, [0, 1, 2, 3], True),     # a path: NCCL may not run
    ({"enable_dbo": False, "ubatch_size": 2}, [0, 1, 2, 3], True),
    ({"enable_dbo": True, "ubatch_size": 0}, [0, 1], True),           # a pair: NCCL may run, SIRCL still serves
    ({"enable_dbo": False, "ubatch_size": 1}, [0, 1, 2, 3], False),
], ids=["dbo-path", "ubatches-path", "dbo-pair", "off"])
def test_micro_batching_is_refused_on_every_group_with_a_session(stub, monkeypatch, parallel, ranks, refused):
    _vllm_config(monkeypatch, **parallel)
    plugin.register()
    if refused:
        with pytest.raises(SirclSetupError, match=r"SIRCL cannot set up tp:0: vLLM's micro-batching "
                                                  r"\(--enable-dbo, or --ubatch-size above 1\)"):
            _communicators("tp:0", ranks)
    else:
        assert all(comm.sircl.session is not None for comm in _communicators("tp:0", ranks))


def test_micro_batching_is_left_alone_on_a_group_without_a_session(stub, monkeypatch):
    _vllm_config(monkeypatch, enable_dbo=True, ubatch_size=0)
    stub.environ["SIRCL_GROUPS"] = "dcp"                    # a pair's TP group: NCCL runs it, no session
    plugin.register()
    comms = _communicators("tp:0", [0, 1])
    assert all(comm.sircl.session is None for comm in comms)



def test_the_receipts_state_where_vllm_and_b12x_were_imported_from(stub, monkeypatch):
    """A source overlay serves when the receipts name its directories (receipt fields vllm and b12x)."""
    from sparkring_sircl.vllm import receipt

    plugin.register()
    comms = _communicators("tp:0", [0, 1, 2, 3])
    record = comms[0].sircl.record
    assert record["vllm"] == os.path.dirname(sys.modules["vllm"].__file__)
    line = receipt.line(record)
    assert f" vllm={record['vllm']} state=ready" in line and line.endswith("state=ready")
    monkeypatch.delitem(sys.modules, "b12x", raising=False)
    assert comms[0].sircl.report()["b12x"] is None
    monkeypatch.setitem(sys.modules, "b12x", types.SimpleNamespace(__file__="/opt/sparkring-overlay/b12x/__init__.py"))
    assert comms[0].sircl.report()["b12x"] == "/opt/sparkring-overlay/b12x"
    assert receipt.package_dir("sircl_no_such_package") is None



@pytest.mark.parametrize("environment, blocks", [({}, 32), ({"SIRCL_LARGE_BLOCKS": "4"}, 4)],
                         ids=["default", "large-blocks"])
def test_the_receipts_session_statistics_state_the_large_grid_cap(stub, monkeypatch, environment, blocks):
    """--large-blocks sets SIRCL_LARGE_BLOCKS; the receipt's session statistics show the cap the session read,
    which check and bundle-check report."""
    module = emulation.session_module("sircl_emulated_large_blocks")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    stub.environ.update({"SIRCL_SESSION_MODULE": module.__name__, **environment})
    plugin.register()
    comms = _communicators("tp:0", [0, 1, 2, 3])
    assert comms[0].sircl.report()["session_stats"]["large_blocks"] == blocks



@pytest.mark.parametrize("positions", [list(range(8)), [0, 1]], ids=["tp8-ring", "tp2-pair"])
def test_with_sircl_nccl_never_every_group_receipt_passes_the_nccl_free_check(stub, positions):
    """TP8 on the ring of eight and TP2 on a pair, SIRCL_NCCL=never: the tensor-parallel and expert-parallel
    groups' receipts show nccl=none, pynccl=skipped and no NCCL row after the calls they carry, which is what
    check --require-no-nccl and bundle-check --require-no-nccl require."""
    import torch.distributed as dist

    from sparkring_sircl.vllm.serve import checks

    world = len(positions)
    stub.environ.update({"SIRCL_NCCL": "never", "SIRCL_GROUPS": "tp",
                         "SIRCL_RANK_POSITIONS": ",".join(str(p) for p in positions)})
    plugin.register()
    tp = _communicators("tp:0", list(range(world)))
    ep = _communicators("ep:0", list(range(world)))

    def step(rank):
        reduced = tp[rank].all_reduce(torch.full((4, 64), float(rank)))
        amax = torch.tensor([float(rank)])
        dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=ep[rank].device_group)
        return reduced, amax

    for reduced, amax in run_ranks(world, step):
        assert reduced[0, 0].item() == sum(range(world)) and amax.item() == world - 1
    receipts = {rank: [tp[rank].sircl.report(), ep[rank].sircl.report()] for rank in range(world)}
    lines, problems = checks.nccl_free_findings(receipts, world)
    assert problems == [] and lines == ["NCCL-free receipts: every rank's groups (ep:0, tp:0) show nccl=none, "
                                        "pynccl=skipped and no NCCL decision row"]
    assert not [event for event in _events() if event[0].startswith("stock")]


def test_dcp_sessions_are_built_without_the_tensor_parallel_sessions_schedule_and_link_settings(stub, monkeypatch):
    """GLM-5.3 at TP8 with DCP4: SIRCL_LARGE_SCHEDULE=ring and larger link sizes reach the TP session only."""
    from sparkring_sircl.vllm.settings import TP_SESSION_VARIABLES

    _pin_dcp(monkeypatch, stub.root)
    tp_only = {"SIRCL_LARGE_SCHEDULE": "ring", "SIRCL_GATHER_SCHEDULE": "chain", "SIRCL_LINK_SLOTS": "12",
               "SIRCL_LINK_SLOT_BYTES": "2097152", "SIRCL_REDUCE_LINK_CHUNK_BYTES": "2097152",
               "SIRCL_CHAIN_MIN_BYTES": "0"}
    assert set(tp_only) <= set(TP_SESSION_VARIABLES)
    stub.environ.update({"SIRCL_NCCL": "never", "SIRCL_GROUPS": "tp,dcp", **tp_only})
    plugin.register()
    import vllm.distributed.parallel_state as parallel_state

    parallel_state.TP_RANKS = list(range(8))
    tp = _communicators("tp:0", list(range(8)))
    low = _communicators("dcp:0", [0, 1, 2, 3], world=8)
    high = _communicators("dcp:0", [4, 5, 6, 7], world=8)
    session = tp[0].sircl.session
    assert (session.large_schedule, session.gather_schedule, session.link_slot_bytes) == ("ring", "chain", 2097152)
    assert session.link_chunk_for("reduce") == 2097152 and session.chain_min_for("reduce") == 0
    for comm in low + high:
        dcp = comm.sircl.session
        assert dcp is not session
        assert (dcp.large_schedule, dcp.gather_schedule, dcp.scatter_schedule) == ("auto", "auto", "pieces")
        assert (dcp.link_slot_bytes, dcp.link_chunk_for("reduce"), dcp.chain_min_for("reduce")) == (
            512 << 10, 512 << 10, 8 << 20)


@pytest.mark.parametrize("mode", ["never", "auto", "topology"])
def test_a_tuning_tables_nccl_marks_route_no_call_in_any_nccl_mode(stub, monkeypatch, mode):
    """TP2 on a cabled pair whose session's table marks NCCL faster at every size it decides. The receipt names
    the table, the mode the adapter resolved (topology is auto) and the NCCL rule. Every eager all-reduce within
    the dispatch ceiling stays on SIRCL under never, auto and topology, whatever the table measured; above the
    ceiling only the opt-in's rules (SIRCL_LARGE_ALLREDUCE=auto) send an eager all-reduce to NCCL."""
    from sparkring_sircl.vllm import settings
    from sparkring_sircl.vllm.planner import NCCL, SIRCL, TensorMeta, plan_all_reduce

    table = "b" * 16
    session_class = stub.sessions.AllReduce
    stats = session_class.stats
    monkeypatch.setattr(session_class, "stats", lambda self: {**stats(self), "tuning": {"table": table}})
    monkeypatch.setattr(session_class, "tuned_choice",
                        lambda self, collective, nbytes, mode=None: object() if nbytes >= 16 else None, raising=False)
    monkeypatch.setattr(session_class, "tuned_backend", lambda self, collective, nbytes, mode=None: "nccl",
                        raising=False)
    stub.environ["SIRCL_NCCL"] = mode
    plugin.register()
    comms = _communicators("tp:0", [0, 1])                  # a pair: NCCL may run under the opt-in
    adapter = comms[0].sircl
    resolved = "never" if mode == "never" else "auto"
    record = adapter.report()
    assert record["tuning"] == table and "tuning=" + table in receipt_line(adapter.record)
    assert (record["nccl_mode"], record["nccl_rule"]) == (resolved, settings.NCCL_RULE)
    assert record["nccl_rule"] == "NCCL: opt-in only (auto); tables choose among SIRCL options"
    assert record["nccl"] == ("none" if mode == "never" else "all")
    assert not hasattr(adapter.policy, "tuned") and adapter.config.nccl_mode == resolved
    for elements in (8, 4096, 32768):                       # 16 B, 8 KiB and 64 KiB, the dispatch ceiling
        meta = TensorMeta((elements,), "bfloat16", 2)
        assert plan_all_reduce(meta, adapter.limits, adapter.policy, capturing=False).backend == SIRCL, elements
    above = plan_all_reduce(TensorMeta((65536,), "bfloat16", 2), adapter.limits, adapter.policy, capturing=False)
    assert above.backend == (SIRCL if mode == "never" else NCCL)
    inputs = [torch.full((4096,), float(rank + 1)).to(torch.bfloat16) for rank in range(2)]
    for result in run_ranks(2, lambda rank: comms[rank].all_reduce(inputs[rank])):
        assert torch.equal(result, reference_sum(inputs))
    rows = {(row["collective"], row["backend"]) for row in adapter.report()["decisions"]}
    assert ("all_reduce", "sircl") in rows and not [row for row in rows if row[1] == "nccl"]
    assert not [event for event in _events() if event[0].startswith("stock")]


def test_the_adapter_keeps_nccl_off_unless_told_otherwise():
    from sparkring_sircl.vllm import settings

    assert settings.nccl_mode({}) == "never" and settings.NCCL_MODES == ("never", "auto")
    assert settings.nccl_mode({"SIRCL_NCCL": "auto"}) == "auto"
    assert settings.nccl_mode({"SIRCL_NCCL": "topology"}) == "auto"          # another name for auto
    assert settings.nccl_mode({"SIRCL_NCCL": " Topology "}) == "auto"
    with pytest.raises(settings.SettingError, match="SIRCL_NCCL must be one of never, auto, topology"):
        settings.nccl_mode({"SIRCL_NCCL": "sometimes"})
    assert settings.NCCL_RULE == "NCCL: opt-in only (auto); tables choose among SIRCL options"
    environ = {"SIRCL_FABRIC": "ring:8", "SIRCL_NCCL": "topology"}
    assert adapter_module.AdapterConfig.from_env(2, environ).nccl_mode == "auto"
    direct = adapter_module.AdapterConfig(adapter_module.Layout.ring(8), (0, 1), ("tp",), "topology", "auto",
                                          None, None)
    assert direct.nccl_mode == "auto"


# The tensor-parallel slot (vllm/tp_slot.py) on emulated ranks: its three size limits, the first-call log of each
# collective and the session's output tensors returned as they are.

def _slots(world: int = 2, *, single_node: bool = False) -> list[SirclRingAllReduce]:
    groups = emulated_groups(list(range(world)))
    routes = [{peer: ("rocep1s0f0",) for peer in range(world) if peer != rank} for rank in range(world)]
    return run_ranks(world, lambda rank: SirclRingAllReduce(groups[rank], object(), torch.device("cpu"),
                                                             peer_routes=routes[rank], single_node=single_node))


def test_the_slot_reports_its_sessions_limits_and_zeros_while_disabled(stub):
    import dataclasses

    from sparkring_sircl.vllm.tp_slot import NO_SLOT_LIMITS, SlotLimits

    slots = _slots()
    # The stub's settings: capacity 1 MiB, dispatch limit 64 KiB, all-gather shards up to 48 KiB.
    expected = SlotLimits(dispatch_bytes=64 << 10, capacity_bytes=1 << 20, gather_bytes=48 << 10)
    for slot in slots:
        assert not slot.disabled and slot.size_limits == expected
        assert slot.size_limits.dispatch_bytes == slot.runtime.dispatch_limit_bytes
        assert slot.size_limits.capacity_bytes == slot.runtime.max_size
    # Read-only: the record is frozen and the slot has no setter for it.
    with pytest.raises(dataclasses.FrozenInstanceError):
        slots[0].size_limits.gather_bytes = 0
    with pytest.raises(AttributeError):
        slots[0].size_limits = NO_SLOT_LIMITS
    # The limits are fixed once the session exists; the environment is read at construction.
    stub.environ["SIRCL_ALLGATHER_MAX_BYTES"] = str(16 << 10)
    stub.environ["SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES"] = str(16 << 10)
    assert all(slot.size_limits == expected for slot in slots)
    run_ranks(2, lambda rank: slots[rank].close())
    assert NO_SLOT_LIMITS == SlotLimits(0, 0, 0)
    assert all(slot.disabled and slot.size_limits == NO_SLOT_LIMITS for slot in slots)
    for slot in _slots(single_node=True):
        assert slot.disabled and slot.runtime is None and slot.size_limits == NO_SLOT_LIMITS


def test_each_collectives_first_call_on_a_slot_logs_one_line(stub):
    slots = _slots()
    records: list[logging.LogRecord] = []

    class Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    log = logging.getLogger("sircl.vllm.tp_slot")
    keep, level = Keep(), log.level
    log.addHandler(keep)
    log.setLevel(logging.DEBUG)
    try:
        reduced = [torch.full((4, 512), float(rank + 1), dtype=torch.bfloat16) for rank in range(2)]   # 4096 B
        shards = [torch.full((2, 256), float(rank + 1), dtype=torch.bfloat16) for rank in range(2)]     # 1024 B
        for _ in range(3):
            run_ranks(2, lambda rank: slots[rank].custom_all_reduce(reduced[rank]))
            run_ranks(2, lambda rank: slots[rank].all_gather(shards[rank], 0))
        # An all-reduce the slot does not route (float32) reaches no collective and logs nothing.
        assert slots[0].custom_all_reduce(torch.ones(4, 512)) is None
    finally:
        log.removeHandler(keep)
        log.setLevel(level)
    lines = sorted((record.levelno, record.getMessage()) for record in records
                   if "first call" in record.getMessage())
    assert lines == sorted([
        (logging.INFO, "SIRCL ring all-reduce, first call on this slot: 4096 bytes of bfloat16."),
        (logging.DEBUG, "SIRCL ring all-reduce, first call on this slot: 4096 bytes of bfloat16."),
        (logging.INFO, "SIRCL ring all-gather, first call on this slot: 1024 bytes of bfloat16."),
        (logging.DEBUG, "SIRCL ring all-gather, first call on this slot: 1024 bytes of bfloat16."),
    ])


def test_the_slots_collectives_return_the_sessions_output_tensors(stub):
    slots = _slots()
    returned: dict[str, list[torch.Tensor]] = {"all_reduce": [], "all_gather": []}
    for slot in slots:
        for name, outputs in returned.items():
            def recording(inp, _collective=getattr(slot.runtime, name), _outputs=outputs, **options):
                out = _collective(inp, **options)
                _outputs.append(out)
                return out

            setattr(slot.runtime, name, recording)
    inputs = [torch.full((4, 512), float(rank + 1), dtype=torch.bfloat16) for rank in range(2)]
    reduced = run_ranks(2, lambda rank: slots[rank].custom_all_reduce(inputs[rank]))
    gathered = run_ranks(2, lambda rank: slots[rank].all_gather(inputs[rank], 0))
    for out in reduced:
        assert any(out is session_out for session_out in returned["all_reduce"])
        assert torch.equal(out, reference_sum(inputs))
    for out in gathered:
        assert any(out is session_out for session_out in returned["all_gather"])
        assert torch.equal(out, torch.cat(inputs, dim=0))
    assert len(returned["all_reduce"]) == len(returned["all_gather"]) == 2


def test_the_route_map_variable_keeps_its_name():
    from sparkring_sircl.vllm import tp_slot

    # One module constant names the route-map variable.
    assert [name for name, value in vars(tp_slot).items() if value == "SIRCL_PEER_ROUTES"] == ["ENV_PEER_ROUTES"]


def receipt_line(record):
    from sparkring_sircl.vllm import receipt

    return receipt.line(record)
