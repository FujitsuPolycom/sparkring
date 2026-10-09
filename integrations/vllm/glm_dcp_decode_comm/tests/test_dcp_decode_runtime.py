"""The worker-side helpers on host tensors: every flag combination, the query, gather, stream, selection and
combine paths, and the refusal of a DCP session the plugin was not built against.

Each test drives the helpers that ``__init__.py`` places into the attention module in the order of the edited
``forward`` and ``_sparse_indexer_and_attn`` statements, for one rank of a modeled DCP group
(:class:`dcp_decode_fakes.World`); the Triton kernels are the torch references and the packed exchange is modeled with
:func:`reference.stage_send`, so a result equals the image's bit for bit exactly when the plumbing is right.
"""

from __future__ import annotations

import functools
import itertools
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import glm_dcp_decode_comm as pkg
from glm_dcp_decode_comm import layout as L
from glm_dcp_decode_comm import reference, runtime

from dcp_decode_fakes import FakeCuda, World, kernels_stub, make_layer, metadata

FLAG_KEYS = tuple(pkg.FLAGS)


@pytest.fixture
def env(monkeypatch):
    cuda = FakeCuda(monkeypatch)
    monkeypatch.setattr(runtime, "_STREAMS", {})
    monkeypatch.setattr(runtime, "COUNTS", {})
    monkeypatch.setattr(runtime, "_AUDITS", {})
    monkeypatch.setattr(runtime, "_ONCE", set())
    monkeypatch.setattr(runtime.SelectionProvider, "_cache", {})
    monkeypatch.setattr(runtime, "_kernels", kernels_stub)
    monkeypatch.setattr(runtime, "SETTINGS", runtime.Settings())
    state = SimpleNamespace(cuda=cuda, metadata=None, packed=[], filters=[])
    monkeypatch.setattr(runtime, "_metadata", lambda layer: (state.metadata, None, torch.zeros((8, 64, 1)), None))

    def fits(dcp, rows, heads):
        return (L.heads_supported(heads)
                and dcp._runtime.world_size * L.wire_chunk_bytes(rows, heads) <= dcp.scatter_op_bytes)

    def packed(rt, attn_out, lse, recv, rows, heads):
        state.packed.append(cuda.current.name)
        rt.world_model.packed_exchange(attn_out, lse, recv)
        return True

    monkeypatch.setattr(runtime, "scatter_pack_fits", fits)
    monkeypatch.setattr(runtime, "packed_all_to_all", packed)

    def selection_filter(impl, metadata_, kv_cache, num_tokens):
        state.filters.append(num_tokens)
        return torch.arange(num_tokens * 4).view(num_tokens, 4), torch.full((num_tokens,), 4)

    monkeypatch.setattr(runtime.SelectionProvider, "_filter", staticmethod(selection_filter))
    monkeypatch.setattr(pkg, "_RUNTIME", None)
    monkeypatch.setattr(pkg, "_SETTINGS", None)
    monkeypatch.setattr(pkg, "_ROOTS", {})

    def configure(**flags):
        values = {key: False for key in FLAG_KEYS}
        values.update(flags)
        values.setdefault("comm_priority", -1)
        values.setdefault("audit", False)
        monkeypatch.setattr(pkg, "_SETTINGS", values)
        monkeypatch.setattr(pkg, "_RUNTIME", None)
        pkg._rt()

    state.configure = configure
    return state


def image_fused_q(calls):
    def fused_q(positions, q_pe, cos_sin_cache, index_q, index_cache, ql_nope, q_scale, index_weights, softmax_scale,
                head_scale, has_indexer=True, index_rope_interleave=False, quantize_mqa=True):
        calls.append("image_fused_q")
        rows = q_pe.shape[0]
        return "index_q_fp8", "index_weights", reference.image_query(positions, q_pe, cos_sin_cache, ql_nope,
                                                                     rows)[:, :, L.QL_NOPE_DIM:].contiguous()

    return fused_q


def run_forward(layer, world: World, fused_calls: list, *, attn_out=None, lse=None, ql_nope=None):
    """The edited statements of ``forward`` and ``_sparse_indexer_and_attn`` for one decode call."""
    rank, rows = world.rank, world.rows
    hidden = torch.linspace(-1, 1, rows * 6144).view(rows, 6144)
    pkg.wk_fork(layer, hidden)
    if layer.indexer is not None and not layer.skip_topk:
        pkg.wk_join(layer, hidden)
    ql_nope = world.ql_nope[rank] if ql_nope is None else ql_nope
    q_pe = world.q_pe[rank]
    pkg.query(layer, world.positions, q_pe, ql_nope)
    _, _, mqa_q = pkg.fused_q(layer, image_fused_q(fused_calls))(
        world.positions, q_pe, world.cache, None, None, ql_nope, None, None, 0.0, 0.0)
    mqa_q_arg = (ql_nope[:rows], mqa_q[:rows])
    if isinstance(mqa_q_arg, tuple) and not pkg.has_query(layer, mqa_q_arg):
        mqa_q_arg = torch.cat(mqa_q_arg, dim=-1)
    gathered = pkg.gather(layer, mqa_q_arg)
    attn_out = world.out[rank] if attn_out is None else attn_out
    lse = world.lse[rank] if lse is None else lse
    combined = pkg.combine(layer)(attn_out, lse, seq_lens=None, query_start_loc=None)
    return pkg.finish(layer, "o_proj"), gathered, combined


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    view = {2: torch.int16, 4: torch.int32}[a.element_size()]
    return a.shape == b.shape and bool(torch.equal(a.contiguous().view(view), b.contiguous().view(view)))


COMBINATIONS = list(itertools.product((False, True), repeat=len(FLAG_KEYS)))


@pytest.mark.parametrize("indexer", (True, False), ids=("indexer", "no-indexer"))
@pytest.mark.parametrize("bits", COMBINATIONS, ids=lambda bits: "".join("1" if b else "0" for b in bits))
def test_every_flag_combination_matches_the_image(env, bits, indexer):
    # The five flags in every combination, on a DSA indexer layer and on one without it: the gathered query and
    # the combined output equal the image's bit for bit, each item replaces exactly the image calls it names, and
    # every side-stream fork is joined into the main stream by the end of forward.
    flags = dict(zip(FLAG_KEYS, bits))
    env.configure(**flags)
    world = World()
    layer = make_layer(env.cuda, world, indexer=indexer)
    env.metadata = metadata(world.rows)
    fused_calls: list[str] = []
    result, gathered, combined = run_forward(layer, world, fused_calls)

    assert result == "o_proj"
    assert same(gathered, world.image_gather(world.queries[world.rank]))
    assert same(combined, world.image_combine(world.out[world.rank], world.lse[world.rank]))
    query_path = flags["query_pack"] or flags["overlap"]
    assert ("image_query_gather" in layer.image_calls) == (not query_path)
    assert ("image_combine" in layer.image_calls) == (not flags["a2a_fused"])
    assert ("image_fused_q" in fused_calls) == (not query_path or indexer)
    assert len(env.packed) == int(flags["a2a_fused"])
    if indexer:
        # wk_fork is the first edited statement, so the wk stream (priority 0) is the first one created.
        expected = env.cuda.created[0].name if flags["wk_overlap"] else "main"
        assert layer.indexer.calls == [expected]
        assert not flags["wk_overlap"] or expected.endswith("(priority 0)")
    else:
        assert layer.indexer is None
    if query_path:
        early = runtime.COUNTS.get("gather_early", 0)
        late = runtime.COUNTS.get("gather_late", 0)
        assert (early, late) == ((1, 0) if flags["overlap"] else (0, 1))
        on_workspace = gathered.untyped_storage().data_ptr() == layer.impl.q_buffer.untyped_storage().data_ptr()
        assert on_workspace == (not (flags["overlap"] and indexer))
        assert layer.session.gathers == [{"shape": (world.rows, world.heads * L.HEAD_DIM), "dim": -1, "out": True}]
    state = layer._glm_dcp_decode_comm_state if any(bits) else None
    if state is not None:
        assert state.stash is None and state.wk is None
    assert env.cuda.current is env.cuda.main
    log = env.cuda.log
    for stream in env.cuda.created:
        if any(entry[:2] == ("enter", stream.name) for entry in log):
            last_leave = max(i for i, entry in enumerate(log) if entry == ("leave", stream.name))
            assert ("wait", "main", stream.name) in log[last_leave:], (stream.name, log)
    if flags["a2a_fused"]:
        assert env.packed == ([s.name for s in env.cuda.created if "priority -1" in s.name][:1]
                              if flags["overlap"] else ["main"])


def test_flags_off_make_the_image_calls_in_order(env):
    world = World()
    layer = make_layer(env.cuda, world)
    env.metadata = metadata(world.rows)
    fused_calls: list[str] = []
    _, gathered, combined = run_forward(layer, world, fused_calls)
    assert fused_calls == ["image_fused_q"]
    assert layer.image_calls == ["image_query_gather", "image_combine"]
    assert layer.indexer.calls == ["main"]
    assert env.cuda.created == [] and env.cuda.log == []
    assert "_glm_dcp_decode_comm_state" not in vars(layer)


# -- query and gather -----------------------------------------------------------------------------------------


def test_query_is_built_from_the_image_statements_and_claimed_by_its_attention_call(env):
    env.configure(query_pack=True)
    world = World(rows=5)
    layer = make_layer(env.cuda, world, indexer=False)
    env.metadata = metadata(world.rows)
    rank = world.rank
    pkg.query(layer, world.positions, world.q_pe[rank], world.ql_nope[rank])
    stash = layer._glm_dcp_decode_comm_state.stash
    assert tuple(stash.q_cat.shape) == (5, world.heads, L.HEAD_DIM) and stash.q_cat.dtype == torch.bfloat16
    assert same(stash.q_cat, world.queries[rank])
    assert stash.source == world.ql_nope[rank].data_ptr() and stash.workspace and not stash.issued
    mqa = (world.ql_nope[rank][:5], stash.q_cat[:, :, L.QL_NOPE_DIM:])
    assert pkg.has_query(layer, mqa)
    gathered = pkg.gather(layer, mqa)
    assert tuple(gathered.shape) == (5, world.world * world.heads, L.HEAD_DIM)


def test_a_stale_query_is_dropped_and_the_call_runs_the_image_path(env):
    env.configure(query_pack=True, overlap=True)
    world = World()
    layer = make_layer(env.cuda, world, indexer=False)
    env.metadata = metadata(world.rows)
    rank = world.rank
    pkg.query(layer, world.positions, world.q_pe[rank], world.ql_nope[rank])
    other = world.ql_nope[rank].clone()
    mqa = (other[:world.rows], world.q_pe[rank][:world.rows])
    assert not pkg.has_query(layer, mqa)
    assert runtime.COUNTS["query_stale"] == 1
    assert layer._glm_dcp_decode_comm_state.stash is None
    comm = env.cuda.created[0].name
    assert ("wait", "main", comm) in env.cuda.log          # the early gather was joined when dropped


def test_an_unconsumed_query_is_dropped_and_joined_by_finish(env):
    env.configure(overlap=True)
    world = World()
    layer = make_layer(env.cuda, world, indexer=True)
    env.metadata = metadata(world.rows)
    rank = world.rank
    pkg.query(layer, world.positions, world.q_pe[rank], world.ql_nope[rank])
    assert pkg.finish(layer, 7) == 7
    assert runtime.COUNTS["query_unconsumed"] == 1
    assert layer._glm_dcp_decode_comm_state.stash is None
    assert env.cuda.log[-1] == ("wait", "main", env.cuda.created[0].name)


def test_a_shard_above_one_op_goes_through_the_communicators_gather(env):
    env.configure(query_pack=True)
    world = World(rows=4)
    layer = make_layer(env.cuda, world, gather_chunk=4 * world.heads * L.HEAD_DIM * 2 - 16)
    env.metadata = metadata(world.rows)
    _, gathered, _ = run_forward(layer, world, [])
    assert layer.session.gathers == [] and layer.communicator.calls == ["all_gather"]
    assert runtime.COUNTS["gather_split_ops"] == 1
    assert same(gathered, world.image_gather(world.queries[world.rank]))


@pytest.mark.parametrize("nbytes,row,fits", [(4096, 1160, False), (4096, 1024, True), (1 << 20, 9216, True),
                                              ((1 << 20) + 16, 9216, False), (48, 16, True), (40, 40, False)])
def test_one_gather_op_is_decided_by_size_and_rows(nbytes, row, fits):
    session = SimpleNamespace(max_gather_bytes=2 << 20)
    dcp = SimpleNamespace(_runtime=session, gather_chunk=1 << 20)
    assert runtime.gather_fits_one_op(dcp, nbytes, row) is fits


def test_padded_kernel_heads_give_the_gathered_query_a_buffer_of_its_own(env):
    # Six heads on four ranks are 24 gathered heads, which B12X runs as 24: the workspace aliases. Five heads on
    # four ranks are 20, which B12X runs as 24: forward_mqa's query view is not contiguous, so the gather may not
    # write the workspace (the attention copies the query in, as the image does).
    for heads, aliased in ((6, True), (5, False)):
        env.configure(query_pack=True)
        world = World(heads=heads)
        layer = make_layer(env.cuda, world, indexer=False)
        env.metadata = metadata(world.rows)
        _, gathered, _ = run_forward(layer, world, [])
        on_workspace = gathered.untyped_storage().data_ptr() == layer.impl.q_buffer.untyped_storage().data_ptr()
        assert on_workspace is aliased
        assert layer.impl.borrowed == ([24] if aliased else [])
        assert same(gathered, world.image_gather(world.queries[world.rank]))


def test_the_workspace_view_is_used_only_when_its_shape_is_the_gathered_query(env):
    env.configure(query_pack=True)
    world = World()
    layer = make_layer(env.cuda, world, indexer=False)
    layer.impl.q_buffer = torch.zeros((16, world.world * world.heads + 1, L.HEAD_DIM), dtype=torch.bfloat16)
    layer.impl._borrow_workspaces = lambda **kwargs: [layer.impl.q_buffer, torch.zeros(16)]
    env.metadata = metadata(world.rows)
    _, gathered, _ = run_forward(layer, world, [])
    assert gathered.untyped_storage().data_ptr() != layer.impl.q_buffer.untyped_storage().data_ptr()
    assert same(gathered, world.image_gather(world.queries[world.rank]))


# -- side streams -------------------------------------------------------------------------------------------


def test_the_communicator_wrapper_moves_only_dcp_session_collectives(env, monkeypatch):
    env.configure(overlap=True)
    seen = []

    class Communicator:
        __module__ = "sparkring_sircl.vllm.communicator"

        def __init__(self, adapter):
            self.sircl = adapter
            self.device = torch.device("cpu")

    for name in runtime._WRAPPED_METHODS:
        setattr(Communicator, name, lambda self, *args, _name=name, **kwargs: seen.append(
            (_name, env.cuda.current.name)) or torch.zeros(1))
    assert runtime.install_comm_stream(Communicator) is True
    assert runtime.install_comm_stream(Communicator) is False
    dcp_session = object()
    dcp_comm = Communicator(SimpleNamespace(dcp=SimpleNamespace(_runtime=dcp_session), session=dcp_session,
                                            shared_from=None))
    tp_comm = Communicator(SimpleNamespace(dcp=None, session=object(), shared_from=None))
    plain = Communicator(None)
    dcp_comm.all_gather(torch.zeros(2), -1)
    tp_comm.all_reduce(torch.zeros(2))
    plain.reduce_scatter(torch.zeros(2))
    comm = env.cuda.created[0].name
    assert seen == [("all_gather", comm), ("all_reduce", "main"), ("reduce_scatter", "main")]
    entered = env.cuda.log.index(("enter", comm))
    assert env.cuda.log[entered - 1] == ("wait", comm, "main")          # the fork waits for main
    assert env.cuda.log[entered + 1:entered + 3] == [("leave", comm), ("wait", "main", comm)]  # and is joined

    # A group that shares a DCP group's session runs on the communication stream too.
    owner = SimpleNamespace(dcp=SimpleNamespace(_runtime=dcp_session), session=dcp_session)
    adapter_module = ModuleType("sparkring_sircl.vllm.adapter")
    adapter_module.live_adapters = lambda: [owner]
    package, vllm = ModuleType("sparkring_sircl"), ModuleType("sparkring_sircl.vllm")
    package.vllm, vllm.adapter = vllm, adapter_module
    monkeypatch.setitem(sys.modules, "sparkring_sircl", package)
    monkeypatch.setitem(sys.modules, "sparkring_sircl.vllm", vllm)
    monkeypatch.setitem(sys.modules, "sparkring_sircl.vllm.adapter", adapter_module)
    sharer = Communicator(SimpleNamespace(dcp=None, session=dcp_session, shared_from="dcp:0"))
    sharer.broadcast(torch.zeros(2), 0)
    assert seen[-1] == ("broadcast", comm)

    env.configure(overlap=False)
    dcp_comm.all_to_all_single(torch.zeros(2), torch.zeros(2))
    assert seen[-1] == ("all_to_all_single", "main")


def test_the_wrapper_refuses_a_communicator_without_a_collective():
    class Communicator:
        pass

    with pytest.raises(pkg.PatchRefused, match="all_reduce not found"):
        runtime.install_comm_stream(Communicator)


def test_the_communication_stream_takes_the_configured_priority(env):
    env.configure(overlap=True, comm_priority=-3)
    stream = runtime.side_stream("comm", torch.device("cuda", 0))
    assert stream.name.endswith("(priority -3)") and runtime.side_stream("comm", torch.device("cuda", 0)) is stream
    env.cuda.capturing = True
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        runtime.side_stream("wk", torch.device("cuda", 0))


# -- selection reuse ----------------------------------------------------------------------------------------


def test_the_selection_is_computed_on_indexer_layers_and_reused_by_the_layers_after_them(env):
    env.configure(selection_reuse=True)
    world = World()
    first = make_layer(env.cuda, world, indexer=True, name="layers.0")
    second = make_layer(env.cuda, world, indexer=False, name="layers.1")
    second.impl.topk_indices_buffer = first.impl.topk_indices_buffer
    env.metadata = metadata(world.rows)
    for layer in (first, second):
        pkg.query(layer, world.positions, world.q_pe[world.rank], world.ql_nope[world.rank])
        assert isinstance(layer.impl._physical_selection_provider, runtime.SelectionProvider)
    ask = dict(num_tokens=world.rows, num_prefills=0, num_decode_tokens=world.rows)
    computed = first.impl._physical_selection_provider.get_b12x_physical_selection(**ask)
    reused = second.impl._physical_selection_provider.get_b12x_physical_selection(**ask)
    assert reused is computed and env.filters == [world.rows]
    # An MTP draft step reuses the top-k on the layer that computed it: it converts again.
    first.skip_topk = True
    first.impl._physical_selection_provider.get_b12x_physical_selection(**ask)
    assert env.filters == [world.rows, world.rows]
    # Prefill and mixed batches go to the image.
    assert second.impl._physical_selection_provider.get_b12x_physical_selection(
        num_tokens=world.rows, num_prefills=1, num_decode_tokens=world.rows - 1) is None
    assert runtime.COUNTS == {"selection_computed": 2, "selection_reused": 1}


def test_a_provider_the_layer_already_has_is_kept(env):
    env.configure(selection_reuse=True)
    world = World()
    layer = make_layer(env.cuda, world)
    taken = object()
    layer.impl._physical_selection_provider = taken
    env.metadata = metadata(world.rows)
    pkg.query(layer, world.positions, world.q_pe[world.rank], world.ql_nope[world.rank])
    assert layer.impl._physical_selection_provider is taken


# -- the fused combine --------------------------------------------------------------------------------------


def test_the_fused_combine_repairs_unaligned_inputs_on_its_own_rank(env):
    env.configure(a2a_fused=True)
    world = World()
    layer = make_layer(env.cuda, world)
    env.metadata = metadata(world.rows)
    rank = world.rank
    out = world.out[rank]
    shifted = torch.empty(out.numel() + 1, dtype=out.dtype)[1:].view(out.shape)
    shifted.copy_(out)
    assert shifted.data_ptr() % 16
    _, _, combined = run_forward(layer, world, [], attn_out=shifted)
    assert env.packed == ["main"]
    assert same(combined, world.image_combine(out, world.lse[rank]))


@pytest.mark.parametrize("case", ("not-the-shim", "other-group", "seq-lens", "too-large", "heads", "a2a-backend-off"))
def test_the_fused_combine_declines_to_the_image_combine(env, case):
    env.configure(a2a_fused=True)
    world = World(heads=6 if case == "heads" else 8)    # six heads per rank: not a power of two
    layer = make_layer(env.cuda, world, scatter_op=(1 << 20) if case != "too-large" else 1024)
    env.metadata = metadata(world.rows)
    rank = world.rank
    kwargs = {"seq_lens": None, "query_start_loc": None}
    if case == "not-the-shim":
        image = layer.dcp_manager.combine
        layer.dcp_manager.combine = functools.partial(lambda *a, **k: image.func(*a, **k), **image.keywords)
    elif case == "other-group":
        layer.dcp_manager.combine = functools.partial(layer.dcp_manager.combine.func, cp_group=object(),
                                                      is_lse_base_on_e=True)
    elif case == "seq-lens":
        kwargs["seq_lens"] = torch.zeros(1)
    elif case == "a2a-backend-off":
        layer.dcp_manager.use_a2a = False
    out, lse = world.out[rank], world.lse[rank]
    combined = runtime.combine_fn(layer)(out, lse, **kwargs)
    assert env.packed == []
    assert runtime.COUNTS["combine_image"] == 1
    assert same(combined, world.image_combine(out, lse))


def test_the_fused_combine_takes_the_lse_base_from_the_images_combine(env, monkeypatch):
    env.configure(a2a_fused=True)
    world = World()
    layer = make_layer(env.cuda, world)
    layer.dcp_manager.combine = functools.partial(layer.dcp_manager.combine.func, cp_group=layer.dcp_manager.group,
                                                  is_lse_base_on_e=False)
    env.metadata = metadata(world.rows)
    bases = []
    stub = kernels_stub()
    monkeypatch.setattr(runtime, "_kernels", lambda: SimpleNamespace(
        rope_cat=stub.rope_cat, wire_combine=lambda *args: bases.append(args[-1]) or stub.wire_combine(*args)))
    runtime.combine_fn(layer)(world.out[world.rank], world.lse[world.rank], seq_lens=None, query_start_loc=None)
    assert bases == [False]


# -- audit --------------------------------------------------------------------------------------------------


def test_audit_mode_compares_every_exact_item_with_the_image(env):
    env.configure(**{key: True for key in FLAG_KEYS}, audit=True)
    runtime._AUDITS[0] = runtime._Audit(torch.device("cpu"))
    world = World()
    layer = make_layer(env.cuda, world)
    env.metadata = metadata(world.rows)
    fused_calls: list[str] = []
    run_forward(layer, world, fused_calls)
    assert fused_calls == ["image_fused_q"]            # audit keeps the image's fused_q as the reference
    summary = runtime._audit_summary(runtime._AUDITS[0].exact, runtime._AUDITS[0].slots)
    assert set(summary) == {"query", "gathered_query", "wk", "combine"}
    assert all(item["calls"] == 1 and item["compared_words"] > 0 and item["differing_words"] == 0
               for item in summary.values()), summary
    assert "image_query_gather" in layer.image_calls and "image_combine" in layer.image_calls


# -- configurations and sessions ----------------------------------------------------------------------------


@pytest.mark.parametrize("case,reason", [
    ("no-dcp", "no decode context parallelism"), ("pcp", "prefill context parallelism"),
    ("impl", "is not B12X sparse MLA"), ("fp8", "FP8 query path"), ("transport", "B12X PCIe DCP transport"),
    ("padded", "padded query heads"), ("widths", "latent or RoPE width")])
def test_configurations_the_items_do_not_cover_keep_the_image_path(env, case, reason):
    env.configure(query_pack=True, a2a_fused=True)
    world = World()
    layer = make_layer(env.cuda, world)
    layer.dcp_manager.group.device_communicator = SimpleNamespace()   # not SIRCL's: declining must come first
    if case == "no-dcp":
        layer.impl.dcp_world_size = 1
    elif case == "pcp":
        layer.use_pcp = True
    elif case == "impl":
        layer.impl = SimpleNamespace(dcp_world_size=4)
    elif case == "fp8":
        layer._fp8_query = True
    elif case == "transport":
        layer.dcp_manager.b12x_transport = object()
    elif case == "padded":
        layer.dcp_manager.padded_num_heads = 16
    else:
        layer.kv_lora_rank = 256
    env.metadata = metadata(world.rows)
    fused_calls: list[str] = []
    run_forward(layer, world, fused_calls)
    state = layer._glm_dcp_decode_comm_state
    assert not state.ok and reason in state.reason
    assert layer.image_calls == ["image_query_gather", "image_combine"] and fused_calls == ["image_fused_q"]


def test_prefill_and_mixed_batches_keep_the_image_path(env):
    env.configure(query_pack=True, a2a_fused=True)
    world = World()
    layer = make_layer(env.cuda, world)
    env.metadata = metadata(world.rows, prefills=1)
    fused_calls: list[str] = []
    run_forward(layer, world, fused_calls)
    assert layer.image_calls == ["image_query_gather", "image_combine"] and fused_calls == ["image_fused_q"]


@pytest.mark.parametrize("case,message", [
    ("not-sircl", "not SIRCL's sparkring_sircl.vllm.communicator.SirclCudaCommunicator"),
    ("no-session", "holds no DCP session"),
    ("collectives", "its DCP collectives are"),
    ("session", "its session is"),
    ("no-scatter", "no scatter collectives"),
    ("version", "sparkring_sircl 0.2.0 is loaded"),
    ("file", "not from the verified file"),
])
def test_a_dcp_session_it_was_not_built_against_refuses(env, monkeypatch, tmp_path, case, message):
    env.configure(a2a_fused=True)
    world = World()
    layer = make_layer(env.cuda, world)
    adapter = layer.communicator.sircl
    if case == "not-sircl":
        layer.dcp_manager.group.device_communicator = SimpleNamespace(sircl=adapter)
    elif case == "no-session":
        adapter.dcp = None
    elif case == "collectives":
        adapter.dcp = SimpleNamespace(_runtime=adapter.session)
    elif case == "session":
        other = SimpleNamespace()
        adapter.session, adapter.dcp._runtime = other, other
    elif case == "no-scatter":
        adapter.session.scatter_available = False
    elif case == "version":
        monkeypatch.setattr(runtime, "SETTINGS", runtime.Settings(a2a_fused=True, sircl_version=pkg.SIRCL_VERSION))
        monkeypatch.setitem(sys.modules, "sparkring_sircl", SimpleNamespace(__version__="0.2.0"))
    else:
        monkeypatch.setattr(runtime, "SETTINGS", runtime.Settings(a2a_fused=True, sircl_root=str(tmp_path)))
        elsewhere = SimpleNamespace(__file__=str(tmp_path / "copy" / "communicator.py"))
        monkeypatch.setitem(sys.modules, "sparkring_sircl.vllm.communicator", elsewhere)
    env.metadata = metadata(world.rows)
    with pytest.raises(pkg.PatchRefused, match=message):
        runtime.combine_fn(layer)
    with pytest.raises(pkg.PatchRefused):                 # every later call refuses too
        runtime.combine_fn(layer)


def test_the_verified_session_files_pass_the_load_check(env, monkeypatch, tmp_path):
    for relative in ("vllm/communicator.py", "vllm/dcp_collectives.py", "oneshot/runtime.py"):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text("")
    monkeypatch.setattr(runtime, "SETTINGS", runtime.Settings(a2a_fused=True, sircl_root=str(tmp_path),
                                                              sircl_version=pkg.SIRCL_VERSION))
    for module, relative in (("sparkring_sircl.vllm.communicator", "vllm/communicator.py"),
                             ("sparkring_sircl.vllm.dcp_collectives", "vllm/dcp_collectives.py"),
                             ("sparkring_sircl.oneshot.runtime", "oneshot/runtime.py")):
        monkeypatch.setitem(sys.modules, module, SimpleNamespace(__file__=str(tmp_path / relative)))
    monkeypatch.setitem(sys.modules, "sparkring_sircl", SimpleNamespace(__version__=pkg.SIRCL_VERSION))
    world = World()
    layer = make_layer(env.cuda, world)
    runtime._session_check(layer.communicator)
