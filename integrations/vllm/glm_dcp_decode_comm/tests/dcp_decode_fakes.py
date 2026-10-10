"""Host-only stand-ins for the CPU tests of ``glm_dcp_decode_comm.runtime``.

- :class:`FakeCuda` replaces torch's CUDA stream surface (current stream,
  stream creation, the stream context, capture state, ``record_stream``) with
  recording objects for one test;
- :class:`World` models one DCP group for one step: every rank's query inputs
  and attention outputs, and what rank ``rank`` receives from its peers in
  each exchange (through :mod:`glm_dcp_decode_comm.reference`);
- the layer, manager, communicator, adapter, collectives and session classes
  carry the module and class names of vLLM's B12X implementation and of the
  pinned SIRCL build, so the runtime's identity checks see what serving shows them.
"""

from __future__ import annotations

import contextlib
import functools
from types import SimpleNamespace

import torch

from glm_dcp_decode_comm import layout as L
from glm_dcp_decode_comm import reference


class FakeStream:
    def __init__(self, name: str, log: list) -> None:
        self.name = name
        self._log = log

    def wait_stream(self, other: "FakeStream") -> None:
        self._log.append(("wait", self.name, other.name))

    def __repr__(self) -> str:
        return f"<stream {self.name}>"


class FakeCuda:
    """The CUDA stream surface the runtime uses, recorded in ``log``."""

    def __init__(self, monkeypatch) -> None:
        self.log: list = []
        self.main = FakeStream("main", self.log)
        self.current = self.main
        self.capturing = False
        self.created: list[FakeStream] = []
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: self.current)
        monkeypatch.setattr(torch.cuda, "Stream", self._new_stream)
        monkeypatch.setattr(torch.cuda, "stream", self._stream)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: self.capturing)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "device", lambda device=None: contextlib.nullcontext())
        monkeypatch.setattr(torch.Tensor, "record_stream",
                            lambda tensor, stream: self.log.append(("record", stream.name)), raising=False)

    def _new_stream(self, device=None, priority=0):
        stream = FakeStream(f"side{len(self.created)}(priority {priority})", self.log)
        self.created.append(stream)
        return stream

    @contextlib.contextmanager
    def _stream(self, stream):
        saved, self.current = self.current, stream
        self.log.append(("enter", stream.name))
        try:
            yield
        finally:
            self.current = saved
            self.log.append(("leave", stream.name))


class _Session:
    def __init__(self, world: "World") -> None:
        self.world_model = world
        self.world_size = world.world
        self.rank = world.rank
        self.max_gather_bytes = 2 << 20
        self.scatter_available = True
        self.gathers: list[dict] = []

    def all_gather(self, inp, *, dim=-1, out=None, stream=None):
        gathered = self.world_model.gather_last(inp)
        self.gathers.append({"shape": tuple(inp.shape), "dim": dim, "out": out is not None})
        out.copy_(gathered)
        return out


class _Collectives:
    def __init__(self, session, gather_chunk: int, scatter_op: int) -> None:
        self._runtime = session
        self.gather_chunk = gather_chunk
        self.gather_max = 2 << 20
        self.scatter_op_bytes = scatter_op

    def _scatter_op_limit(self) -> int:
        return self.scatter_op_bytes


class _Communicator:
    def __init__(self, adapter, world: "World") -> None:
        self.sircl = adapter
        self.device = torch.device("cpu")
        self.world_model = world
        self.calls: list[str] = []

    def all_gather(self, inp, dim=-1):
        self.calls.append("all_gather")
        return self.world_model.gather_last(inp.reshape(inp.shape[0], -1)).view(
            *inp.shape[:-1], inp.shape[-1] * self.world_model.world)


class RoceOneshotAllReduce(_Session):
    __module__ = "sparkring_sircl.oneshot.runtime"


class SirclDcpCollectives(_Collectives):
    __module__ = "sparkring_sircl.vllm.dcp_collectives"


class SirclCudaCommunicator(_Communicator):
    __module__ = "sparkring_sircl.vllm.communicator"


Session, Collectives, Communicator = RoceOneshotAllReduce, SirclDcpCollectives, SirclCudaCommunicator


class B12xMLASparseImpl:
    """The B12X sparse MLA implementation's surface (the class name is what the runtime checks)."""

    lse_base_on_e = True

    def __init__(self, world: int, rank: int, heads: int, max_tokens: int) -> None:
        self.dcp_world_size = world
        self.dcp_rank = rank
        self._physical_selection_provider = None
        self.topk_indices_buffer = torch.zeros((max_tokens, 2048), dtype=torch.int32)
        self._input_num_heads = world * heads
        self._kernel_num_heads = (world * heads + 7) // 8 * 8      # whole groups of eight, as B12X runs them
        self.q_buffer = torch.zeros((max_tokens, self._kernel_num_heads, L.HEAD_DIM), dtype=torch.bfloat16)
        self.borrowed: list[int] = []

    def _borrow_workspaces(self, *, input_num_heads=None, include_ckv=False):
        heads = self._input_num_heads if input_num_heads is None else input_num_heads
        self.borrowed.append(heads)
        return [self.q_buffer[:, :heads] if heads <= self.q_buffer.shape[1] else
                torch.zeros((self.q_buffer.shape[0], heads, L.HEAD_DIM), dtype=torch.bfloat16),
                torch.zeros(16, dtype=torch.uint8)]


class Indexer:
    def __init__(self, cuda: FakeCuda) -> None:
        self.cuda = cuda
        self.weight = torch.linspace(-1, 1, 6144 * 8).view(6144, 8)
        self.calls: list[str] = []

    def wk_weights_proj(self, hidden_states):
        self.calls.append(self.cuda.current.name)
        return (hidden_states.float() @ self.weight,)


class World:
    """One DCP group for one step, seen from rank ``rank``; peers' tensors are generated, never run."""

    def __init__(self, *, world: int = 4, rank: int = 1, heads: int = 8, rows: int = 3, seed: int = 7) -> None:
        self.world, self.rank, self.heads, self.rows = world, rank, heads, rows
        g = torch.Generator().manual_seed(seed)
        self.cache = torch.randn((64, L.ROPE_DIM), generator=g)
        self.positions = torch.randint(0, 64, (rows,), generator=g, dtype=torch.int64)
        q = torch.randn((world, rows, heads, 512 + 64 + 64), generator=g).to(torch.bfloat16)
        # ql_nope as the image makes it: bmm(...).transpose(0, 1), so heads are its outer stride.
        self.ql_nope = [torch.randn((heads, rows, L.QL_NOPE_DIM), generator=g).to(torch.bfloat16).transpose(0, 1)
                        for _ in range(world)]
        self.q_pe = [q[s][:, :, 512:576] for s in range(world)]  # a view with the image's strides
        self.queries = [reference.image_query(self.positions, self.q_pe[s], self.cache, self.ql_nope[s], rows)
                        for s in range(world)]
        self.out = [torch.randn((rows, world * heads, L.V_DIM), generator=g).to(torch.bfloat16) for _ in range(world)]
        self.lse = [torch.randn((rows, world * heads), generator=g) for _ in range(world)]
        for lse in self.lse:
            lse[0, 0] = float("-inf")      # an empty local shard
        self.lse[0][1, 1] = float("nan")   # cleaned to -inf by both combines
        self.lse[2][1, 2] = float("inf")

    def gather_last(self, inp: torch.Tensor) -> torch.Tensor:
        """``[rows, X]`` of this rank -> ``[rows, world * X]``: the peers' rows are their queries."""
        rows = inp.shape[0]
        parts = [inp.reshape(rows, -1) if s == self.rank else self.queries[s][:rows].reshape(rows, -1)
                 for s in range(self.world)]
        return torch.cat(parts, dim=-1)

    def image_gather(self, query: torch.Tensor) -> torch.Tensor:
        """The image's ``query_gather``: concatenation along the head dimension."""
        rows = query.shape[0]
        return torch.cat([query if s == self.rank else self.queries[s][:rows] for s in range(self.world)], dim=1)

    def image_combine(self, attn_out, lse) -> torch.Tensor:
        records = torch.stack([reference.image_pack(attn_out if s == self.rank else self.out[s],
                                                    lse if s == self.rank else self.lse[s], self.world)[self.rank]
                               for s in range(self.world)])
        return reference.image_combine(records, True)

    def packed_exchange(self, attn_out, lse, recv) -> None:
        sends = [reference.stage_send(attn_out if s == self.rank else self.out[s],
                                      lse if s == self.rank else self.lse[s], self.world, s)
                 for s in range(self.world)]
        recv.copy_(reference.exchange(sends, self.rank))


def sircl_shim(world: World, calls: list):
    """SIRCL's ``dcp_all_to_all`` replacement as the manager's partial sees it (marked ``_sircl_original``)."""

    def dcp_a2a_lse_reduce(cp_attn_out, cp_attn_lse, cp_group, ctx=None, return_lse=False, is_lse_base_on_e=True,
                           seq_lens=None, query_start_loc=None):
        calls.append("image_combine")
        return world.image_combine(cp_attn_out, cp_attn_lse)

    dcp_a2a_lse_reduce._sircl_original = object()
    return dcp_a2a_lse_reduce


class Layer:
    """An attention layer's attributes (a plain object: the selection provider keeps a weak reference to it)."""

    def __init__(self, **attributes) -> None:
        self.__dict__.update(attributes)


def make_layer(cuda: FakeCuda, world: World, *, indexer: bool = True, gather_chunk: int = 1 << 20,
               scatter_op: int = 1 << 20, name: str = "model.layers.0.self_attn.attn"):
    """One attention layer of rank ``world.rank`` with the pinned SIRCL build's DCP communicator objects."""
    session = Session(world)
    collectives = Collectives(session, gather_chunk, scatter_op)
    adapter = SimpleNamespace(dcp=collectives, session=session, shared_from=None)
    communicator = Communicator(adapter, world)
    group = SimpleNamespace(device_communicator=communicator, rank_in_group=world.rank, world_size=world.world)
    calls: list[str] = []

    def query_gather(query):
        calls.append("image_query_gather")
        return world.image_gather(query)

    manager = SimpleNamespace(group=group, use_a2a=True, padded_num_heads=None, b12x_transport=None,
                              query_gather=query_gather)
    manager.combine = functools.partial(sircl_shim(world, calls), cp_group=group, is_lse_base_on_e=True)
    impl = B12xMLASparseImpl(world.world, world.rank, world.heads, max_tokens=16)
    layer = Layer(
        impl=impl, dcp_manager=manager, num_heads=world.heads, kv_lora_rank=512, qk_rope_head_dim=64,
        W_UK_T=torch.zeros(1), rotary_emb=SimpleNamespace(cos_sin_cache=world.cache), layer_name=name,
        indexer=Indexer(cuda) if indexer else None, skip_topk=False, use_pcp=False, _fp8_query=False,
        image_calls=calls, session=session, communicator=communicator)
    return layer


def metadata(rows: int, *, prefills: int = 0):
    return SimpleNamespace(num_actual_tokens=rows, num_prefills=prefills, num_decode_tokens=rows - prefills,
                           req_id_per_token=torch.zeros(rows, dtype=torch.int32), block_table=torch.zeros(1),
                           cp_kv_cache_interleave_size=1, block_size=64)


def kernels_stub():
    """``kernels`` with the references in place of the Triton kernels."""

    def rope_cat(positions, q_pe, cos_sin_cache, ql_nope, q_cat, rows):
        q_cat[:rows] = reference.image_query(positions, q_pe, cos_sin_cache, ql_nope, rows)

    def wire_combine(recv, out, lse, world, rank, heads, is_base_e):
        return reference.wire_combine(recv, out, lse, world, rank, is_base_e)

    return SimpleNamespace(rope_cat=rope_cat, wire_combine=wire_combine)
