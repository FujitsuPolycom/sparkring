"""The SIRCL-EP all-to-all manager: exchange paths, buffers and numerical parity, offline.

vLLM's ``all2all`` module, its ``CudaCommunicator`` and ``libsircl`` are stand-ins
here: ``AgRsAll2AllManager`` records the size exchanges the pipeline would make,
and ``FakeLib`` models ``sirclEpDispatch``/``sirclEpCombine`` from their
published contract (``spark_transport/libsircl/kernels/sircl_p2p.cu``,
``namespace sircl_ep``):

- every rank's slice of the dispatch output holds, per peer, the rows that peer
  routed to this rank in that peer's token order, then a padded tail the kernel
  masks with a non-local expert ID and a zero weight;
- a receiver fails the call when a peer routes more rows than ``max_tokens``;
- combine returns to each peer the expert output of the rows that peer sent, and
  sums for every own token the one row each peer returns for it, indexed by the
  token's slot in the rows this rank sent that peer.

The model reads and writes the buffers through the pointers the manager passes,
as the kernel does, so a wrong pointer, row offset or stride shows up as a wrong
result rather than as a passing test. A layer runs on every rank before any
combine, as the pipeline runs it, and the model records every
``(rank, num_tokens, max_tokens)`` triple so a test can prove that the ranks agreed
on one exchange stride.
"""
import ctypes
import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SOURCE = Path(__file__).with_name("sircl_ep_all2all.py")
NAME = "sircl_ep_all2all_under_test"

ELEMENT_SIZE = {
    torch.float32: 4,
    torch.int32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
}


def _scalar(value):
    """The integer a ctypes call argument carries."""
    return getattr(value, "value", value)


def _view(pointer, dtype, shape):
    """The tensor a device pointer addresses, for the CPU memory these tests use."""
    count = 1
    for size in shape:
        count *= size
    nbytes = count * ELEMENT_SIZE[dtype]
    if nbytes == 0:
        return torch.empty(shape, dtype=dtype)
    buffer = (ctypes.c_char * nbytes).from_address(pointer)
    return torch.frombuffer(buffer, dtype=torch.uint8).view(dtype).view(shape)


# -- the stand-in library: the kernel's contract, one process -------------------------------


class Cluster:
    """Every rank's routing inputs, its dispatch slot map and the calls the library saw."""

    def __init__(self, world=4, num_experts=32, top_k=4, hidden_dim=8, dtype=torch.bfloat16):
        self.world = world
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.inputs = {}
        self.expert_out = {}
        self.send_slots = {}
        self.counts_by_rank = {}
        self.dispatches = []
        self.combines = []
        self.workspaces = []
        self.error = b""
        self.fail_next = 0

    def local_experts(self):
        return self.num_experts // self.world

    def mask_for(self, rank):
        """The padded tail's expert ID: a local expert of the next rank, so it is never local."""
        return ((rank + 1) % self.world) * self.local_experts()

    def routes_to(self, rank, expert):
        local = self.local_experts()
        return local * rank <= expert < local * (rank + 1)

    def strides(self):
        """The ``(rank, num_tokens, max_tokens)`` of every dispatch, in call order."""
        return list(self.dispatches)

    def max_tokens_by_rank(self):
        return {rank: max_tokens for rank, _, max_tokens in self.dispatches}


class FakeLib:
    """``libsircl`` as ctypes presents it: functions carrying ``argtypes`` and ``restype``."""

    class _Fn:
        def __init__(self, fn):
            self._fn = fn
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self._fn(*args)

    def __init__(self, cluster, rank):
        self.cluster = cluster
        self.rank = rank
        self.sirclEpWorkspaceBytes = FakeLib._Fn(_workspace_bytes)
        self.sirclEpDispatch = FakeLib._Fn(self._dispatch)
        self.sirclEpCombine = FakeLib._Fn(self._combine)
        self.ncclGetLastError = FakeLib._Fn(lambda comm: cluster.error)

    def _dispatch(self, hidden_in, topk_ids_in, topk_weights_in, hidden_out, topk_ids_out, topk_weights_out,
                  recv_counts_out, workspace, workspace_bytes, num_tokens, max_tokens, hidden_bytes, top_k,
                  num_experts, comm, stream):
        cluster = self.cluster
        rank, world = self.rank, cluster.world
        num_tokens, max_tokens = _scalar(num_tokens), _scalar(max_tokens)
        hidden_bytes, top_k, num_experts = _scalar(hidden_bytes), _scalar(top_k), _scalar(num_experts)
        cluster.dispatches.append((rank, num_tokens, max_tokens))
        cluster.workspaces.append((rank, "dispatch", workspace.value, _scalar(workspace_bytes)))
        if cluster.fail_next:
            refused, cluster.fail_next = cluster.fail_next, 0
            return refused
        if num_tokens > max_tokens or num_experts % world or hidden_bytes % 16:
            cluster.error = b"sirclEpDispatch: invalid capacity"
            return 1
        if (num_tokens == 0) != (not hidden_in.value):
            raise AssertionError("dispatch must pass a null input pointer exactly when it has no tokens")

        hidden_dim = hidden_bytes // ELEMENT_SIZE[cluster.dtype]
        rows = world * max_tokens
        h_out = _view(hidden_out.value, cluster.dtype, (rows, hidden_dim))
        ids_out = _view(topk_ids_out.value, torch.int32, (rows, top_k))
        wts_out = _view(topk_weights_out.value, torch.float32, (rows, top_k))

        # The rows this rank receives: each peer's tokens routed to this rank, in the peer's order.
        counts = []
        for peer in range(world):
            hidden_p, ids_p, weights_p = cluster.inputs[peer]
            selected = [
                token
                for token in range(ids_p.shape[0])
                if any(cluster.routes_to(rank, int(expert)) for expert in ids_p[token])
            ]
            if len(selected) > max_tokens:
                cluster.error = (
                    b"sirclEpDispatch: a peer routed more rows than max_tokens: received "
                    + str(len(selected)).encode()
                    + b" rows, the receive expected "
                    + str(max_tokens).encode()
                )
                return 1
            counts.append(len(selected))
            base, keep = peer * max_tokens, len(selected)
            if keep:
                h_out[base:base + keep].copy_(hidden_p[selected])
                ids_out[base:base + keep].copy_(ids_p[selected])
                wts_out[base:base + keep].copy_(weights_p[selected])
            h_out[base + keep:base + max_tokens].zero_()
            ids_out[base + keep:base + max_tokens].fill_(cluster.mask_for(rank))
            wts_out[base + keep:base + max_tokens].zero_()

        # The slot of every own token in the rows this rank sends to each peer.
        _, ids_me, _ = cluster.inputs[rank]
        if num_tokens != ids_me.shape[0]:
            raise AssertionError("dispatch must describe exactly the tokens of its own rank")
        slots = [[-1] * num_tokens for _ in range(world)]
        for peer in range(world):
            slot = 0
            for token in range(num_tokens):
                if any(cluster.routes_to(peer, int(expert)) for expert in ids_me[token]):
                    slots[peer][token] = slot
                    slot += 1

        cluster.send_slots[rank] = slots
        cluster.counts_by_rank[rank] = counts
        if recv_counts_out.value:
            _view(recv_counts_out.value, torch.int32, (world,)).copy_(torch.tensor(counts, dtype=torch.int32))
        return 0

    def _combine(self, expert_out_in, hidden_out, workspace, workspace_bytes, num_tokens, max_tokens, hidden_dim,
                 datatype, comm, stream):
        cluster = self.cluster
        rank, world = self.rank, cluster.world
        num_tokens, max_tokens = _scalar(num_tokens), _scalar(max_tokens)
        hidden_dim = _scalar(hidden_dim)
        cluster.combines.append((rank, num_tokens, max_tokens))
        cluster.workspaces.append((rank, "combine", workspace.value, _scalar(workspace_bytes)))
        if (num_tokens == 0) != (not hidden_out.value):
            raise AssertionError("combine must pass a null output pointer exactly when it has no tokens")
        own = cluster.expert_out[rank]
        if expert_out_in.value != own.data_ptr():
            raise AssertionError("combine must read the expert output of its own slice")
        if num_tokens == 0:
            return 0
        slots = cluster.send_slots[rank]
        out = _view(hidden_out.value, cluster.dtype, (num_tokens, hidden_dim))
        acc = out.float()
        acc.zero_()
        for peer in range(world):
            # The rows this rank sent peer now sit in the peer's slice, in this rank's block of it.
            sent = cluster.expert_out[peer][rank * max_tokens:]
            for token in range(num_tokens):
                slot = slots[peer][token]
                if slot >= 0:
                    acc[token] += sent[slot].float()
        out.copy_(acc.to(cluster.dtype))
        return 0


def _workspace_bytes(world, max_tokens, hidden_bytes):
    """``sirclEpWorkspaceBytes``: the layouts of ``sccl_engine_ep_workspace_bytes``, 64-byte aligned."""

    def align(value):
        return (value + 63) & ~63

    indices = align(world * max_tokens * 4)
    return 64 + 2 * indices + align(world * max_tokens * hidden_bytes)


# -- stand-ins for vLLM ---------------------------------------------------------------------


class AgRsAll2AllManager:
    """vLLM's manager: the size exchange and the fallbacks the SIRCL-EP manager defers to."""

    def __init__(self, cpu_group, tcp_store_group=None):
        self.cpu_group = cpu_group
        self.tcp_store_group = tcp_store_group
        self.size_requests = []
        self.dispatch_calls = 0
        self.combine_calls = 0

    def _get_comm_group(self, is_sequence_parallel=False):
        return self.cpu_group

    def _get_sizes(self, num_tokens, group):
        self.size_requests.append(num_tokens)
        return group.sizes

    def dispatch(self, hidden_states, topk_weights, topk_ids, is_sequence_parallel=False, extra_tensors=None):
        self.dispatch_calls += 1
        return ("agrs-dispatch", int(hidden_states.shape[0]))

    def combine(self, hidden_states, is_sequence_parallel=False):
        self.combine_calls += 1
        return ("agrs-combine", int(hidden_states.shape[0]))


class MoEPrepareAndFinalizeNaiveDPEPModular:
    @staticmethod
    def prepare(*args, **kwargs):
        return ("prepared", args, kwargs)


class CudaCommunicator:
    def __init__(self, cpu_group, all2all_manager, use_all2all=True, *args, **kwargs):
        self.cpu_group = cpu_group
        self.all2all_manager = all2all_manager
        self.use_all2all = use_all2all


def _module(name, **attributes):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    for key, value in attributes.items():
        setattr(module, key, value)
    parent, _, leaf = name.rpartition(".")
    if parent:
        setattr(sys.modules[parent], leaf, module)
    return module


@pytest.fixture
def ep(monkeypatch):
    """The module under test, loaded against stand-ins for vLLM's all2all interface."""
    _module("vllm")
    _module("vllm.logger", init_logger=lambda name: logging.getLogger(name))
    _module("vllm.distributed")
    _module("vllm.distributed.device_communicators")
    _module("vllm.distributed.device_communicators.all2all", AgRsAll2AllManager=AgRsAll2AllManager)
    _module("vllm.distributed.device_communicators.cuda_communicator", CudaCommunicator=CudaCommunicator)
    _module("vllm.model_executor")
    _module("vllm.model_executor.layers")
    _module("vllm.model_executor.layers.fused_moe")
    _module("vllm.model_executor.layers.fused_moe.prepare_finalize")
    _module(
        "vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep",
        MoEPrepareAndFinalizeNaiveDPEPModular=MoEPrepareAndFinalizeNaiveDPEPModular,
    )

    spec = importlib.util.spec_from_file_location(NAME, SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = module
    spec.loader.exec_module(module)
    for registered in list(module._LIVE_MANAGERS):
        module._LIVE_MANAGERS.discard(registered)
    yield module
    sys.modules.pop(NAME, None)


def _manager_class(module):
    """The manager with the device and metadata seams of these CPU tests overridden."""

    class TestManager(module.SirclEpAll2AllManager):
        capturing = False
        shared_tokens = None

        def _capturing(self):
            return self.capturing

        def _shared_tokens(self):
            return self.shared_tokens

        def _on_device(self, tensor):
            return True

        def _stream_ptr(self, device):
            return ctypes.c_void_p(0)

    return TestManager


class Node:
    """One rank: its group, its ``libsircl`` stand-in and its manager.

    ``dp_bound`` models vLLM's ``ForwardContext.dp_metadata``: ``"auto"`` reports the
    true maximum token count of the group, an integer reports a stale or partial
    value, and ``None`` reports that no metadata is available.
    """

    def __init__(self, module, cluster, rank, *, static_cap=96, sizes=None, expert_map=None, device_comm=True,
                 dp_bound="auto"):
        self.module = module
        self.cluster = cluster
        self.rank = rank
        self.expert_map = expert_map
        self.dp_bound = dp_bound
        self.group = types.SimpleNamespace(
            world_size=cluster.world,
            rank_in_group=rank,
            sizes=list(sizes) if sizes is not None else [0] * cluster.world,
            device_communicator=None,
        )
        self.lib = FakeLib(cluster, rank)
        self.device_comm = None
        if device_comm:
            self.device_comm = types.SimpleNamespace(
                pynccl_comm=types.SimpleNamespace(
                    nccl=types.SimpleNamespace(lib=self.lib),
                    comm=0x1000 + rank,
                    world_size=cluster.world,
                    rank=rank,
                    disabled=False,
                )
            )
        self.manager = _manager_class(module)(self.group, None, device_comm=self.device_comm)
        self.manager._static_max_tokens = static_cap if static_cap else 0

    @property
    def capturing(self):
        """Whether the manager believes it is inside a CUDA graph capture."""
        return self.manager.capturing

    @capturing.setter
    def capturing(self, value):
        self.manager.capturing = value

    def dispatch(self, hidden=None, weights=None, ids=None, **kwargs):
        if self.dp_bound == "auto":
            counts = [tensor.shape[0] for tensor, _, _ in self.cluster.inputs.values()]
            self.manager.shared_tokens = max(counts, default=0) or None
        else:
            self.manager.shared_tokens = self.dp_bound
        if hidden is None:
            hidden, ids, weights = self.cluster.inputs[self.rank]
        self.module._ACTIVE_META.current = (self.cluster.num_experts, self.expert_map)
        try:
            return self.manager.dispatch(hidden, weights, ids, **kwargs)
        finally:
            self.module._ACTIVE_META.current = None

    def combine(self, hidden_states, **kwargs):
        return self.manager.combine(hidden_states, **kwargs)


def _expert_map(cluster, rank):
    """The rank's slice of a linear expert map: local IDs on this rank, ``-1`` elsewhere."""
    local = cluster.local_experts()
    mapping = torch.full((cluster.num_experts,), -1, dtype=torch.int32)
    mapping[rank * local:(rank + 1) * local] = torch.arange(local, dtype=torch.int32)
    return mapping


def _inputs(cluster, counts, seed, target_ranks=None):
    """Per-rank routing inputs with these token counts; ``target_ranks`` restricts expert IDs."""
    generator = torch.Generator().manual_seed(seed)
    if target_ranks is None:
        low, high = 0, cluster.num_experts
    else:
        low = min(target_ranks) * cluster.local_experts()
        high = (max(target_ranks) + 1) * cluster.local_experts()
    for rank, count in enumerate(counts):
        hidden = torch.randn((count, cluster.hidden_dim), generator=generator).to(cluster.dtype).contiguous()
        ids = torch.randint(low, high, (count, cluster.top_k), generator=generator, dtype=torch.int32).contiguous()
        weights = torch.rand((count, cluster.top_k), generator=generator, dtype=torch.float32).contiguous()
        cluster.inputs[rank] = (hidden, ids, weights)


def _expert(h_out, ids_out, wts_out):
    """A row-independent expert: a row's output depends only on that row's content."""
    return (h_out.float() * (1.0 + wts_out.sum(dim=1, keepdim=True))).to(h_out.dtype).contiguous()


def _expected(hidden, ids, weights, cluster):
    """Combine's result from which peers each token reaches: one contribution per peer."""
    value = (hidden.float() * (1.0 + weights.sum(dim=1, keepdim=True))).to(cluster.dtype).float()
    out = torch.zeros((hidden.shape[0], hidden.shape[1]), dtype=torch.float32)
    for token in range(hidden.shape[0]):
        peers = {int(expert) // cluster.local_experts() for expert in ids[token]}
        for _ in peers:
            out[token] += value[token]
    return out.to(cluster.dtype)


def _nodes(ep, cluster, **kwargs):
    """One manager per rank, all with the same configuration, as the group is configured."""
    return {
        rank: Node(ep, cluster, rank, expert_map=_expert_map(cluster, rank), **kwargs)
        for rank in range(cluster.world)
    }


def _layer(nodes):
    """One MoE layer on every rank: dispatch, the row-local expert, then combine."""
    dispatched = {rank: node.dispatch() for rank, node in nodes.items()}
    cluster = nodes[0].cluster
    cluster.expert_out = {
        rank: _expert(h_out, ids_out, wts_out) for rank, (h_out, wts_out, ids_out) in dispatched.items()
    }
    combined = {rank: node.combine(cluster.expert_out[rank]) for rank, node in nodes.items()}
    return dispatched, combined


# -- the exchange itself --------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_the_static_path_matches_the_analytic_exchange(ep, dtype):
    cluster = Cluster(dtype=dtype)
    _inputs(cluster, [5, 5, 5, 5], seed=7)
    nodes = _nodes(ep, cluster, static_cap=96)
    dispatched, combined = _layer(nodes)
    for rank in range(cluster.world):
        hidden, ids, weights = cluster.inputs[rank]
        assert torch.equal(combined[rank], _expected(hidden, ids, weights, cluster))
        # The exchange runs at the run's constant capacity, so every slice carries a padded tail.
        assert dispatched[rank][0].shape == (cluster.world * 96, cluster.hidden_dim)
        assert nodes[rank].manager.stats()["static_dispatches"] == 1
        assert nodes[rank].manager.stats()["size_exchanges"] == 0
        assert nodes[rank].manager.size_requests == []


def test_unequal_token_counts_share_one_exchange_stride(ep):
    """DP ranks holding 6, 5, 12 and 0 tokens must still agree on the stride."""
    cluster = Cluster()
    counts = [6, 5, 12, 0]
    _inputs(cluster, counts, seed=9)
    nodes = _nodes(ep, cluster, static_cap=96)
    _, combined = _layer(nodes)
    strides = cluster.max_tokens_by_rank()
    assert set(strides.values()) == {96}, strides
    for rank in range(cluster.world):
        hidden, ids, weights = cluster.inputs[rank]
        assert torch.equal(combined[rank], _expected(hidden, ids, weights, cluster))
        assert nodes[rank].manager.stats()["static_dispatches"] == 1
        assert cluster.dispatches[rank][1] == counts[rank]


def test_without_dp_metadata_every_rank_takes_the_dynamic_path(ep):
    cluster = Cluster()
    counts = [6, 5, 12, 0]
    _inputs(cluster, counts, seed=13)
    nodes = _nodes(ep, cluster, static_cap=96, dp_bound=None, sizes=counts)
    _, combined = _layer(nodes)
    assert set(cluster.max_tokens_by_rank().values()) == {12}
    for rank in range(cluster.world):
        hidden, ids, weights = cluster.inputs[rank]
        assert torch.equal(combined[rank], _expected(hidden, ids, weights, cluster))
        assert nodes[rank].manager.stats()["dynamic_dispatches"] == 1
        assert nodes[rank].manager.stats()["size_exchanges"] == 1


def test_a_bound_above_the_cap_is_dynamic_on_every_rank(ep):
    cluster = Cluster()
    counts = [6, 5, 12, 0]
    _inputs(cluster, counts, seed=17)
    nodes = _nodes(ep, cluster, static_cap=8, sizes=counts)
    _, combined = _layer(nodes)
    assert set(cluster.max_tokens_by_rank().values()) == {12}
    for rank in range(cluster.world):
        hidden, ids, weights = cluster.inputs[rank]
        assert torch.equal(combined[rank], _expected(hidden, ids, weights, cluster))
        assert nodes[rank].manager.stats()["static_dispatches"] == 0


def test_the_static_path_agrees_with_the_dynamic_path_when_counts_are_equal(ep):
    cluster = Cluster()
    _inputs(cluster, [6, 6, 6, 6], seed=19)
    results = {}
    for static_cap in (0, 96):
        cluster.dispatches.clear()
        cluster.send_slots.clear()
        nodes = _nodes(ep, cluster, static_cap=static_cap, sizes=[6, 6, 6, 6])
        _, combined = _layer(nodes)
        results[static_cap] = combined
        for rank, node in nodes.items():
            if static_cap:
                assert node.manager.stats()["static_dispatches"] == 1
            else:
                assert node.manager.stats()["size_exchanges"] == 1
    for rank in range(cluster.world):
        assert torch.equal(results[0][rank], results[96][rank]), f"rank {rank} disagrees between paths"


def test_ranks_that_receive_nothing_still_combine_their_own_tokens(ep):
    cluster = Cluster()
    _inputs(cluster, [6, 6, 6, 6], seed=23, target_ranks=(0, 1))
    nodes = _nodes(ep, cluster, static_cap=96)
    _, combined = _layer(nodes)
    for rank in range(cluster.world):
        hidden, ids, weights = cluster.inputs[rank]
        assert torch.equal(combined[rank], _expected(hidden, ids, weights, cluster))
        # Every token targets ranks 0 and 1, so ranks 2 and 3 receive no rows at all.
        for idle in (2, 3):
            assert cluster.counts_by_rank[idle] == [0] * cluster.world


def test_a_masked_tail_never_contributes_to_combine(ep):
    cluster = Cluster()
    _inputs(cluster, [3, 3, 3, 3], seed=29)
    nodes = _nodes(ep, cluster, static_cap=96)
    dispatched, _ = _layer(nodes)
    rank = 1
    h_out, wts_out, ids_out = dispatched[rank]
    max_tokens = h_out.shape[0] // cluster.world
    local = cluster.local_experts()
    for peer in range(cluster.world):
        base = peer * max_tokens
        real = cluster.counts_by_rank[rank][peer]
        assert real < max_tokens, "this test needs a padded tail on every slice"
        tail_ids = ids_out[base + real:base + max_tokens]
        assert torch.equal(tail_ids, torch.full_like(tail_ids, cluster.mask_for(rank)))
        assert not wts_out[base + real:base + max_tokens].any()
        # The mask names a local expert of the next rank, never one of this rank's own.
        assert not ((tail_ids >= rank * local) & (tail_ids < (rank + 1) * local)).any()


def test_the_counts_buffer_reports_what_each_peer_routed(ep):
    cluster = Cluster()
    _inputs(cluster, [6, 6, 6, 6], seed=31)
    nodes = _nodes(ep, cluster, static_cap=96)
    rank = 2
    nodes[rank].dispatch()
    arena = nodes[rank].manager._active_call.arena
    assert torch.equal(arena.counts, torch.tensor(cluster.counts_by_rank[rank], dtype=torch.int32))


def test_an_idle_step_dispatch_and_combine_are_no_ops(ep):
    cluster = Cluster()
    _inputs(cluster, [0, 0, 0, 0], seed=37)
    nodes = _nodes(ep, cluster, static_cap=96, sizes=[0, 0, 0, 0])
    dispatched, combined = _layer(nodes)
    for rank in range(cluster.world):
        assert dispatched[rank][0].shape == (cluster.world, cluster.hidden_dim)
        assert combined[rank].shape == (0, cluster.hidden_dim)
        assert nodes[rank].manager.stats()["combines"] == 1


# -- buffers, capture and synchronization ---------------------------------------------------


def test_steady_state_dispatch_and_combine_allocate_nothing(ep, monkeypatch):
    cluster = Cluster()
    _inputs(cluster, [4, 4, 4, 4], seed=41)
    nodes = _nodes(ep, cluster, static_cap=96)
    first_dispatched, first_combined = _layer(nodes)

    allocations = []
    real_empty, real_zeros = torch.empty, torch.zeros

    def counting_empty(*args, **kwargs):
        allocations.append(("empty", args))
        return real_empty(*args, **kwargs)

    def counting_zeros(*args, **kwargs):
        allocations.append(("zeros", args))
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", counting_empty)
    monkeypatch.setattr(torch, "zeros", counting_zeros)
    second_dispatched, second_combined = _layer(nodes)
    assert allocations == []
    for rank in range(cluster.world):
        assert second_dispatched[rank][0].data_ptr() == first_dispatched[rank][0].data_ptr()
        assert second_combined[rank].data_ptr() == first_combined[rank].data_ptr()


def test_combine_uses_the_workspace_its_dispatch_was_given(ep):
    """The workspace is the kernel's cross-call state; combine must name the buffer dispatch used."""
    cluster = Cluster()
    _inputs(cluster, [4, 4, 4, 4], seed=43)
    nodes = _nodes(ep, cluster, static_cap=96)
    _layer(nodes)
    node = nodes[0]
    node.dispatch()
    dispatch_workspace = node.manager._workspace
    # A growth elsewhere in the pipeline would leave a different current workspace; the combined
    # call must still describe the one its own dispatch wrote.
    node.manager._workspace = torch.empty_like(dispatch_workspace)
    cluster.workspaces.clear()
    node.combine(cluster.expert_out[0])
    assert [pointer for _, kind, pointer, _ in cluster.workspaces if kind == "combine"] == [
        dispatch_workspace.data_ptr()
    ]


def test_the_static_path_reads_nothing_back_from_the_device(ep, monkeypatch):
    cluster = Cluster()
    _inputs(cluster, [5, 5, 5, 5], seed=43)
    nodes = _nodes(ep, cluster, static_cap=96)

    def refuse(*args, **kwargs):
        raise AssertionError("the static path must not read a result back to the host")

    for name in ("item", "tolist", "cpu", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, refuse)
    monkeypatch.setattr(torch.cuda, "synchronize", refuse)
    _layer(nodes)
    for node in nodes.values():
        assert node.manager.stats()["static_dispatches"] == 1
        assert node.manager.size_requests == []


def test_tokens_across_dp_reads_the_synchronized_metadata(ep, monkeypatch):
    def with_metadata(metadata):
        monkeypatch.setitem(
            sys.modules,
            "vllm.forward_context",
            types.SimpleNamespace(get_forward_context=lambda: types.SimpleNamespace(dp_metadata=metadata)),
        )

    with_metadata(
        types.SimpleNamespace(
            max_tokens_across_dp_cpu=torch.tensor([12], dtype=torch.int32),
            num_tokens_across_dp_cpu=torch.tensor([6, 0, 12, 5], dtype=torch.int32),
        )
    )
    assert ep._tokens_across_dp() == 12

    # The CED hook rewrites the per-rank counts without the maximum: the larger one bounds the step.
    with_metadata(
        types.SimpleNamespace(
            max_tokens_across_dp_cpu=torch.tensor([64], dtype=torch.int32),
            num_tokens_across_dp_cpu=torch.tensor([6, 0, 12, 5], dtype=torch.int32),
        )
    )
    assert ep._tokens_across_dp() == 64

    # Only the per-rank counts: their maximum is the bound.
    with_metadata(types.SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([6, 0, 12, 5], dtype=torch.int32)))
    assert ep._tokens_across_dp() == 12

    # A scalar rather than a vector, and plain integers, are both accepted.
    with_metadata(types.SimpleNamespace(max_tokens_across_dp_cpu=9))
    assert ep._tokens_across_dp() == 9

    # No metadata, no bound.
    with_metadata(None)
    assert ep._tokens_across_dp() is None

    # A device-resident count would synchronize the device: no bound rather than a read.
    with_metadata(
        types.SimpleNamespace(
            max_tokens_across_dp_cpu=torch.tensor([12], dtype=torch.int32, device="meta"),
            num_tokens_across_dp_cpu=None,
        )
    )
    assert ep._tokens_across_dp() is None

    # vLLM's forward context is not importable at all.
    monkeypatch.setitem(sys.modules, "vllm.forward_context", None)
    assert ep._tokens_across_dp() is None


def test_the_dynamic_path_makes_one_size_exchange_per_layer(ep):
    cluster = Cluster()
    _inputs(cluster, [3, 3, 3, 3], seed=47)
    nodes = _nodes(ep, cluster, static_cap=0, sizes=[3, 3, 3, 3])
    for _ in range(3):
        _layer(nodes)
    for node in nodes.values():
        assert node.manager.size_requests == [3, 3, 3]
        assert node.manager.stats()["size_exchanges"] == 3
        assert node.manager.stats()["static_dispatches"] == 0


def test_growth_retires_buffers_instead_of_freeing_them(ep):
    cluster = Cluster()
    _inputs(cluster, [4, 4, 4, 4], seed=53)
    nodes = _nodes(ep, cluster, static_cap=96, sizes=[256, 256, 256, 256])
    dispatched, _ = _layer(nodes)
    captured = dispatched[1][0].untyped_storage().data_ptr()
    assert nodes[1].manager.stats()["arena_allocations"] == 1  # the first call: workspace and arena

    for node in nodes.values():
        node.manager._static_max_tokens = 0  # a prefill-sized step takes the dynamic path
    for rank in range(cluster.world):
        cluster.inputs[rank] = (
            torch.zeros((256, cluster.hidden_dim), dtype=cluster.dtype),
            torch.zeros((256, cluster.top_k), dtype=torch.int32),
            torch.zeros((256, cluster.top_k), dtype=torch.float32),
        )
    prefill, _ = _layer(nodes)
    assert prefill[1][0].shape == (cluster.world * 256, cluster.hidden_dim)
    retired = [tensor.untyped_storage().data_ptr() for tensor in nodes[1].manager._retired]
    assert captured in retired, "the buffer a graph may point at must stay alive"
    assert nodes[1].manager.stats()["arena_allocations"] == 2


def test_capture_uses_reserved_buffers_and_refuses_to_allocate(ep, monkeypatch):
    cluster = Cluster()
    _inputs(cluster, [4, 4, 4, 4], seed=59)
    nodes = _nodes(ep, cluster, static_cap=96)
    _layer(nodes)  # warm, as vLLM's per-bucket warmup does before capture

    def refuse(*args, **kwargs):
        raise AssertionError("nothing may allocate inside a CUDA graph capture")

    for node in nodes.values():
        node.capturing = True
    monkeypatch.setattr(torch, "empty", refuse)
    _layer(nodes)
    for node in nodes.values():
        assert node.manager.stats()["static_dispatches"] == 2


def test_capture_ignores_the_dp_bound_and_uses_the_constant(ep):
    """A capture runs the static path with the run's capacity even with no metadata at all."""
    cluster = Cluster()
    _inputs(cluster, [6, 6, 6, 6], seed=61)
    nodes = _nodes(ep, cluster, static_cap=96)
    _layer(nodes)  # warm: static at the constant capacity, so the arena covers it
    for node in nodes.values():
        node.dp_bound = None  # no metadata at capture time
        node.capturing = True
    cluster.dispatches.clear()
    _layer(nodes)
    assert set(cluster.max_tokens_by_rank().values()) == {96}
    for rank, node in nodes.items():
        assert node.manager.stats()["static_dispatches"] == 2
        assert node.manager.size_requests == []


def test_a_capture_size_above_the_cap_keeps_its_own_capacity(ep, caplog):
    cluster = Cluster()
    _inputs(cluster, [12, 12, 12, 12], seed=67)
    nodes = _nodes(ep, cluster, static_cap=8, sizes=[12, 12, 12, 12])
    _layer(nodes)  # the bound is above the cap, so this warm step is dynamic and sizes the arena
    for node in nodes.values():
        node.capturing = True
    cluster.dispatches.clear()
    with caplog.at_level(logging.WARNING):
        _layer(nodes)
    assert set(cluster.max_tokens_by_rank().values()) == {12}
    assert "capture size 12 exceeds" in caplog.text


def test_capture_without_warm_buffers_falls_back_to_the_pipeline_default(ep):
    cluster = Cluster()
    _inputs(cluster, [4, 4, 4, 4], seed=71)
    nodes = _nodes(ep, cluster, static_cap=96)
    nodes[0].capturing = True
    assert nodes[0].dispatch()[0] == "agrs-dispatch"
    assert nodes[0].manager.stats()["agrs_fallbacks"] == 1


# -- the stats summary ----------------------------------------------------------------------


def test_the_stats_summary_is_logged_on_first_static_interval_and_exit(ep, monkeypatch, caplog):
    monkeypatch.setattr(ep, "STATS_INTERVAL", 2)
    cluster = Cluster()
    _inputs(cluster, [4, 4, 4, 4], seed=73)
    node = Node(ep, cluster, 0, static_cap=96, expert_map=_expert_map(cluster, 0))
    with caplog.at_level(logging.INFO):
        _layer({0: node, **{rank: Node(ep, cluster, rank, static_cap=96,
                                       expert_map=_expert_map(cluster, rank))
                            for rank in range(1, cluster.world)}})
        node.dispatch()  # second dispatch of this manager reaches the interval
        ep._log_at_exit()
    lines = [record.getMessage() for record in caplog.records if ep.STATS_TAG in record.getMessage()]
    assert any("reason=first-static" in line for line in lines)
    assert any("reason=interval" in line for line in lines)
    assert any("reason=atexit" in line for line in lines)
    summary = next(line for line in lines if "reason=atexit" in line)
    assert "static_cap=96" in summary
    assert "static_dispatches=2" in summary
    assert "size_exchanges=0" in summary
    assert "dispatches=2" in summary


# -- eligibility and failure -----------------------------------------------------------------


@pytest.mark.parametrize(
    "bound, cap, expected",
    [(0, 96, 0), (1, 96, 96), (12, 96, 96), (96, 96, 96), (97, 96, 0), (12, 8, 0), (5, 0, 0)],
)
def test_the_capacity_is_the_run_constant_or_zero(bound, cap, expected, ep):
    assert ep._shared_capacity(bound, cap) == expected


def test_a_step_above_the_static_cap_takes_the_dynamic_path(ep):
    cluster = Cluster()
    _inputs(cluster, [0, 0, 0, 0], seed=79)
    cluster.inputs[0] = (
        torch.zeros((97, cluster.hidden_dim), dtype=cluster.dtype),
        torch.zeros((97, cluster.top_k), dtype=torch.int32),
        torch.zeros((97, cluster.top_k), dtype=torch.float32),
    )
    node = Node(ep, cluster, 0, static_cap=96, sizes=[97, 0, 0, 0], expert_map=_expert_map(cluster, 0))
    node.dispatch()
    assert node.manager.stats()["dynamic_dispatches"] == 1
    assert node.manager.size_requests == [97]


def test_a_peer_routing_more_rows_than_the_capacity_fails_loudly(ep):
    """Metadata that understates the group's bound is the residual risk, and it fails loudly."""
    cluster = Cluster()
    _inputs(cluster, [2, 2, 2, 2], seed=83)
    node = Node(ep, cluster, 0, static_cap=96, expert_map=_expert_map(cluster, 0), dp_bound=4)
    node.dispatch()  # capacity 96 covers this step
    cluster.inputs[1] = (
        torch.zeros((100, cluster.hidden_dim), dtype=cluster.dtype),
        torch.zeros((100, cluster.top_k), dtype=torch.int32),  # every token routes to rank 0
        torch.full((100, cluster.top_k), 0.25, dtype=torch.float32),
    )
    with pytest.raises(RuntimeError, match="sirclEpDispatch failed with rc=1"):
        node.dispatch()


def test_the_first_failure_disables_the_exchange_and_falls_back(ep):
    cluster = Cluster()
    _inputs(cluster, [4, 4, 4, 4], seed=89)
    node = Node(ep, cluster, 1, static_cap=96, expert_map=_expert_map(cluster, 1))
    cluster.fail_next = 1
    assert node.dispatch()[0] == "agrs-dispatch"
    assert "sirclEpDispatch returned rc=1" in node.manager._disabled_reason
    assert node.dispatch()[0] == "agrs-dispatch"
    assert node.manager.stats()["agrs_fallbacks"] == 2
    assert node.manager.stats()["dispatches"] == 0


def test_combine_without_a_dispatch_defers_to_the_pipeline_default(ep):
    cluster = Cluster()
    _inputs(cluster, [4, 4, 4, 4], seed=97)
    node = Node(ep, cluster, 0, static_cap=96, expert_map=_expert_map(cluster, 0))
    assert node.combine(torch.zeros((4, cluster.hidden_dim), dtype=cluster.dtype))[0] == "agrs-combine"


@pytest.mark.parametrize("reason", ["extra_tensors", "non_linear_map", "two_ranks", "no_device_comm_group"])
def test_unsupported_calls_take_the_pipeline_default(ep, reason):
    world = 2 if reason == "two_ranks" else 4
    cluster = Cluster(world=world)
    _inputs(cluster, [4] * world, seed=101)
    mapping = _expert_map(cluster, 0)
    node = Node(
        ep,
        cluster,
        0,
        static_cap=96,
        sizes=[4] * world,
        expert_map=torch.zeros_like(mapping) if reason == "non_linear_map" else mapping,
        device_comm=reason != "no_device_comm_group",
    )
    extra = [torch.zeros(1)] if reason == "extra_tensors" else None
    assert node.dispatch(extra_tensors=extra)[0] == "agrs-dispatch"
    assert node.manager.stats()["agrs_fallbacks"] == 1


# -- installation ---------------------------------------------------------------------------


def test_install_replaces_the_all2all_manager_on_the_cuda_communicator(ep, monkeypatch):
    monkeypatch.setenv(ep.ENV_STATIC_A2A, "1")
    monkeypatch.setenv(ep.ENV_STATIC_MAX_TOKENS, "64")
    assert ep.install_sircl_ep_all2all() is True
    cpu_group = types.SimpleNamespace(world_size=4, rank_in_group=0)
    communicator = CudaCommunicator.__new__(CudaCommunicator)
    CudaCommunicator.__init__(communicator, cpu_group, AgRsAll2AllManager(cpu_group))
    assert isinstance(communicator.all2all_manager, ep.SirclEpAll2AllManager)
    assert communicator.all2all_manager._static_max_tokens == 64


def test_install_leaves_a_different_all2all_manager_in_place(ep):
    assert ep.install_sircl_ep_all2all() is True
    cpu_group = types.SimpleNamespace(world_size=4, rank_in_group=0)
    installed = object()
    communicator = CudaCommunicator.__new__(CudaCommunicator)
    CudaCommunicator.__init__(communicator, cpu_group, installed)
    assert communicator.all2all_manager is installed
