"""Column gathers staged on the session's links (``executor.ColumnGather``, ``SIRCL_COLUMN_GATHER``).

An all-gather along a dimension with more than one row in front of it, which the session would carry on its
ring or chain as a dimension-0 gather of the same shard, runs as that gather into a staging buffer plus one
local copy. The tests check the decision from rank-invariant facts, the layout arithmetic of the copy for
every row width and dtype, bit-identity with the tiled gather on emulated ranks (odd shapes, non-contiguous
and unaligned inputs), the staging buffer kept across calls, the switch, the adapter's receipt and the
serve bundle's option. :class:`LinkSession` models a real session's link decisions on CPU tensors.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import pytest
torch = pytest.importorskip("torch")

from sparkring_sircl import env as env_mod  # noqa: E402
from sparkring_sircl.vllm import adapter as adapter_module  # noqa: E402
from sparkring_sircl.vllm import emulation, executor, guard, planner  # noqa: E402
from sparkring_sircl.vllm.adapter import AdapterConfig, GroupAdapter, GroupPlacement  # noqa: E402
from sparkring_sircl.vllm.emulation import EmulatedFabric, EmulatedRingSession, EmulationError, run_ranks  # noqa: E402
from sparkring_sircl.vllm.fabric import Layout  # noqa: E402
from sparkring_sircl.vllm.planner import Plan, Policy, SessionLimits, TensorMeta  # noqa: E402

WORLD = 4
POLICY = Policy(None, "never", "sircl")
LIMITS = SessionLimits(world=WORLD, capacity=1 << 20, dispatch=64 << 10, gather=48 << 10,
                       reduce_dtypes=("bfloat16",), gather_piece=32 << 10)


class LinkGathers:
    """``all_gather_large`` with a real session's link decisions, on an emulated session.

    A gather runs as one ``link`` op (``ring`` or ``chain``) when every rank's shard is one piece of the
    output (one row in front of ``dim``), a positive multiple of 16 bytes, with an output of at least
    ``link_min`` bytes, from and into 16-byte-aligned memory; otherwise as tiles. ``gather_uses_ring`` and
    ``gather_uses_chain`` answer from the shape alone, as the session's do. Every op is one exchange whose
    signature names its kind, so ranks that disagree fail with ``RankDivergence``; ``ops`` records
    ``(kind, dim, shape)`` per call.
    """

    gather_piece_bytes = 32 << 10
    link: str | None = "ring"
    link_min = 0

    def _link_shape(self, inp, dim: int) -> bool:
        if inp.dim() == 0:
            return False
        dim %= inp.dim()
        shard = inp.numel() * inp.element_size()
        output = shard * self.world_size
        return (math.prod(inp.shape[:dim]) == 1 and shard > 0 and shard % 16 == 0 and output < 1 << 31
                and output >= self.link_min)

    def gather_uses_ring(self, inp, dim: int = -1, *, mode=None) -> bool:
        return self.link == "ring" and self._link_shape(inp, dim)

    def gather_uses_chain(self, inp, dim: int = -1, *, mode=None) -> bool:
        return self.link == "chain" and self._link_shape(inp, dim)

    def all_gather_large(self, inp, *, dim: int = -1, out=None, stream=None):
        dim %= inp.dim()
        shape = list(inp.shape)
        shape[dim] *= self.world_size
        if out is not None and (list(out.shape) != shape or out.dtype != inp.dtype or not out.is_contiguous()):
            raise EmulationError("out must be a contiguous tensor of the gathered shape")
        aligned = inp.data_ptr() % 16 == 0 and (out is None or out.data_ptr() % 16 == 0)
        kind = self.link if self.link and aligned and self._link_shape(inp, dim) else "tiles"
        self.__dict__.setdefault("ops", []).append((kind, dim, tuple(inp.shape)))
        shards = self._launch(f"all_gather_large:{kind}", "bytes", inp.numel() * inp.element_size(),
                              inp.detach().contiguous().clone())
        result = torch.cat(shards, dim=dim)
        if out is None:
            return result
        out.copy_(result)
        return out


class LinkSession(LinkGathers, EmulatedRingSession):
    def __init__(self, fabric, rank, *, link: str | None = "ring", link_min: int = 0) -> None:
        super().__init__(fabric, rank, max_size=1 << 20, dispatch_limit_bytes=64 << 10, max_gather_bytes=48 << 10)
        self.link = link
        self.link_min = link_min
        self.ops: list[tuple[str, int, tuple[int, ...]]] = []


class TilesOnly(EmulatedRingSession):
    """A session with ``all_gather_large`` and no link reports."""

    gather_piece_bytes = 32 << 10

    def all_gather_large(self, inp, *, dim=-1, out=None, stream=None):
        raise AssertionError("not called")


def _sessions(link="ring", link_min=0, world=WORLD):
    fabric = EmulatedFabric(world)
    return [LinkSession(fabric, rank, link=link, link_min=link_min) for rank in range(world)]


def _inputs(shape, dtype, seed, world=WORLD):
    generator = torch.Generator().manual_seed(seed)
    if dtype == torch.bool:
        return [torch.randint(0, 2, shape, generator=generator).bool() for _ in range(world)]
    if dtype.is_complex:
        return [torch.randn(shape, generator=generator, dtype=dtype) for _ in range(world)]
    if dtype.is_floating_point:
        return [torch.randn(shape, generator=generator).to(dtype) for _ in range(world)]
    return [torch.randint(-100, 100, shape, generator=generator).to(dtype) for _ in range(world)]


def _bits(tensor):
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64, 16: torch.int64}[tensor.element_size()]
    return tensor.contiguous().view(view)


def _same(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _plan(x, dim):
    return planner.plan_all_gather(TensorMeta.of(x), dim, LIMITS, POLICY, capturing=False)


# -- the decision ---------------------------------------------------------------------------------------------


def test_the_route_follows_the_plan_the_rows_in_front_and_the_sessions_link_report():
    ring, chain, none = _sessions("ring")[0], _sessions("chain")[0], _sessions(None)[0]
    shard = torch.zeros(512, 72, dtype=torch.bfloat16)              # 73,728 bytes: above one op
    large = _plan(shard, -1)
    assert large.method == "large"
    assert executor.column_gather_route(ring, large, shard, -1) == "ring"
    assert executor.column_gather_route(chain, large, shard, 1) == "chain"
    assert executor.column_gather_route(none, large, shard, -1) is None
    # Dimension 0, or one row in front of the dimension: the shard is one piece of the output already.
    assert executor.column_gather_route(ring, _plan(shard, 0), shard, 0) is None
    one_row = torch.zeros(1, 512, 72, dtype=torch.bfloat16)
    assert executor.column_gather_route(ring, _plan(one_row, 1), one_row, 1) is None
    assert executor.column_gather_route(ring, _plan(one_row, 2), one_row, 2) == "ring"
    # One session op, or a plan that does not reach all_gather_large.
    small = torch.zeros(8, 72, dtype=torch.bfloat16)
    assert _plan(small, -1).method == "direct"
    assert executor.column_gather_route(ring, _plan(small, -1), small, -1) is None
    for method in ("rows", "tiles", "bytes_large"):
        assert executor.column_gather_route(ring, Plan("all_gather", "sircl", method, 3, 1), shard, -1) == "ring"
    # A shard that is not a whole number of 16-byte packs, an output below the session's minimum.
    odd = torch.zeros(999, 37, dtype=torch.bfloat16)                # 73,926 bytes
    assert executor.column_gather_route(ring, _plan(odd, -1), odd, -1) is None
    assert executor.column_gather_route(_sessions("ring", link_min=WORLD * 73728 + 16)[0], large, shard, -1) is None
    assert executor.column_gather_route(_sessions("ring", link_min=WORLD * 73728)[0], large, shard, -1) == "ring"
    # Sessions that do not offer the operation or do not report their links.
    fabric = EmulatedFabric(WORLD)
    assert executor.column_gather_route(TilesOnly(fabric, 0), large, shard, -1) is None
    assert executor.column_gather_route(EmulatedRingSession(fabric, 1), large, shard, -1) is None


def test_the_route_reads_no_pointer_and_no_layout():
    ring = _sessions("ring")[0]
    contiguous = torch.zeros(512, 72, dtype=torch.bfloat16)
    transposed = torch.zeros(72, 512, dtype=torch.bfloat16).t()
    offset = torch.zeros(512 * 72 + 8, dtype=torch.bfloat16)[1:1 + 512 * 72].view(512, 72)
    assert not transposed.is_contiguous() and offset.data_ptr() % 16
    assert {executor.column_gather_route(ring, _plan(contiguous, -1), x, -1)
            for x in (contiguous, transposed, offset)} == {"ring"}


# -- the layout arithmetic of the copy -----------------------------------------------------------------------


@pytest.mark.parametrize("shape,dim,dtype", [
    ((6, 5, 7), 1, torch.bfloat16),        # rows of 70 bytes: 2-byte view
    ((6, 5, 7), 2, torch.bfloat16),        # rows of 14 bytes
    ((3, 8), 1, torch.int8),               # rows of 8 bytes: 8-byte view
    ((5, 3), 1, torch.uint8),              # rows of 3 bytes: 1-byte view
    ((4, 3), -1, torch.float32),           # rows of 12 bytes: 4-byte view
    ((2, 3, 4), 1, torch.float64),
    ((3, 2), 1, torch.complex64),
    ((9, 3), -1, torch.bool),
    ((4, 6), 1, torch.float16),
    ((7, 2, 3), 2, torch.int64),
    ((2, 3, 5, 4), 2, torch.bfloat16),     # a middle dimension of four
    ((8192, 41), -1, torch.bfloat16),      # many rows of 82 bytes
])
@pytest.mark.parametrize("world", [2, 3, 8])
def test_interleaved_rows_are_the_concatenation(shape, dim, dtype, world):
    shards = _inputs(shape, dtype, seed=11 * world, world=world)
    staging = torch.cat([shard.contiguous().reshape(-1).view(torch.uint8) for shard in shards])
    out_shape = list(shape)
    out_shape[dim % len(shape)] *= world
    out = torch.empty(out_shape, dtype=dtype)
    outer = math.prod(shape[:dim % len(shape)])
    nbytes = shards[0].numel() * shards[0].element_size()
    executor.interleave_rows(staging, out.reshape(-1).view(torch.uint8), world, outer, nbytes // outer)
    assert _same(out, torch.cat(shards, dim=dim))


# -- staged against tiled, on emulated ranks ----------------------------------------------------------------


@pytest.mark.parametrize("shape,dtype,dim,layout,link,route", [
    ((512, 72), torch.bfloat16, -1, "contiguous", "ring", "ring"),
    ((512, 72), torch.bfloat16, -1, "contiguous", "chain", "chain"),
    ((64, 8, 40), torch.float32, 1, "contiguous", "ring", "ring"),       # a middle dimension
    ((2, 300, 72), torch.bfloat16, 2, "contiguous", "chain", "chain"),   # three dimensions
    ((600, 77), torch.bfloat16, -1, "contiguous", "ring", "ring"),       # rows of 154 bytes
    ((8000, 13), torch.int8, -1, "contiguous", "ring", "ring"),          # rows of 13 bytes
    ((8000, 13), torch.bool, -1, "contiguous", "ring", "ring"),          # a byte-view dtype
    ((1024, 10), torch.complex64, -1, "contiguous", "chain", "chain"),
    ((512, 72), torch.bfloat16, -1, "transposed", "ring", "ring"),       # non-contiguous input
    ((512, 72), torch.bfloat16, -1, "offset", "ring", "ring"),           # 2 bytes past a 16-byte boundary
    ((999, 37), torch.bfloat16, -1, "contiguous", "ring", None),         # shard not whole packs: tiles
    ((512, 72), torch.bfloat16, -1, "contiguous", None, None),           # no links: tiles
])
def test_staged_column_gathers_equal_the_tiled_gather_bit_for_bit(shape, dtype, dim, layout, link, route):
    sessions = _sessions(link)
    inputs = _inputs(shape, dtype, seed=31)
    staged = [executor.ColumnGather(True) for _ in range(WORLD)]
    tiled = [executor.ColumnGather(False) for _ in range(WORLD)]

    def placed(tensor):
        if layout == "transposed":
            return tensor.transpose(-1, -2).contiguous().transpose(-1, -2)
        if layout == "offset":
            base = torch.empty(tensor.numel() + 8, dtype=tensor.dtype)
            view = base[1:1 + tensor.numel()].view(tensor.shape)
            view.copy_(tensor)
            return view
        return tensor

    def body(rank):
        x = placed(inputs[rank])
        plan = _plan(x, dim)
        assert plan.method.endswith("large")
        new = executor.all_gather(sessions[rank], plan, x, dim, WORLD, column_gather=staged[rank])
        old = executor.all_gather(sessions[rank], plan, x, dim, WORLD, column_gather=tiled[rank])
        plain = executor.all_gather(sessions[rank], plan, x, dim, WORLD)
        return new, old, plain

    results = run_ranks(WORLD, body)
    expected = torch.cat(inputs, dim=dim)
    for rank, (new, old, plain) in enumerate(results):
        assert _same(new, old) and _same(new, plain) and _same(new, expected)
        assert staged[rank].last_route == route and tiled[rank].last_route is None
        assert staged[rank].calls == ({route: 1} if route else {})
        kinds = [op[0] for op in sessions[rank].ops]
        if route:
            # The staged call is one dimension-0 link op on the shard's bytes; the others gather along dim.
            assert sessions[rank].ops[0] == (route, 0, (inputs[rank].numel() * inputs[rank].element_size(),))
            assert kinds == [route, "tiles", "tiles"]
            assert staged[rank].staging_bytes == WORLD * inputs[rank].numel() * inputs[rank].element_size()
        else:
            assert kinds == ["tiles"] * 3 and staged[rank].staging_bytes == 0


def test_ranks_with_different_switches_diverge_instead_of_mixing_paths():
    sessions = _sessions("ring")
    inputs = _inputs((512, 72), torch.bfloat16, seed=37)

    def body(rank):
        gather = executor.ColumnGather(rank != 1)
        return executor.all_gather(sessions[rank], _plan(inputs[rank], -1), inputs[rank], -1, WORLD,
                                   column_gather=gather)

    with pytest.raises(emulation.RankDivergence):
        run_ranks(WORLD, body)


# -- the staging buffer -------------------------------------------------------------------------------------


def test_the_staging_buffer_grows_to_the_largest_call_and_is_reused():
    gather = executor.ColumnGather(True)
    assert gather.staging_bytes == 0
    first = gather.staging(4096, "cpu")
    pointer = first.data_ptr()
    assert first.numel() == 4096 and first.is_contiguous() and pointer % 16 == 0
    assert gather.staging(1024, "cpu").data_ptr() == pointer and gather.staging_bytes == 4096
    grown = gather.staging(8192, "cpu")
    assert grown.numel() == 8192 and gather.staging_bytes == 8192
    assert gather.staging(4096, "cpu").data_ptr() == grown.data_ptr()

    sessions = _sessions("ring")
    shapes = [(512, 72), (384, 72), (1024, 72), (512, 72)]
    steps = [_inputs(shape, torch.bfloat16, seed=41 + index) for index, shape in enumerate(shapes)]
    kept = [executor.ColumnGather(True) for _ in range(WORLD)]

    def body(rank):
        found = []
        for step in steps:
            x = step[rank]
            out = executor.all_gather(sessions[rank], _plan(x, -1), x, -1, WORLD, column_gather=kept[rank])
            found.append((out, kept[rank].staging_bytes, kept[rank]._buffer.data_ptr()))
        return found

    for entries in run_ranks(WORLD, body):
        assert [entry[1] for entry in entries] == [WORLD * 73728, WORLD * 73728, WORLD * 147456, WORLD * 147456]
        assert entries[2][2] == entries[3][2]
        for index, entry in enumerate(entries):
            assert _same(entry[0], torch.cat(steps[index], dim=-1))
    assert kept[0].calls == {"ring": 4}
    assert kept[0].snapshot() == {"enabled": True, "calls": {"ring": 4}, "staging_bytes": WORLD * 147456}


# -- the switch ---------------------------------------------------------------------------------------------


def test_the_switch_is_documented_and_parsed_strictly():
    documented = {variable.name: variable for variable in env_mod.VARIABLES}
    assert documented["SIRCL_COLUMN_GATHER"].default == "1"
    assert executor.ColumnGather.from_environment({}).enabled
    assert executor.ColumnGather.from_environment({"SIRCL_COLUMN_GATHER": "1"}).describe() == "on"
    assert executor.ColumnGather.from_environment({"SIRCL_COLUMN_GATHER": " 0 "}).describe() == "off"
    with pytest.raises(ValueError, match="SIRCL_COLUMN_GATHER must be 0 or 1"):
        executor.ColumnGather.from_environment({"SIRCL_COLUMN_GATHER": "on"})


# -- the adapter and its receipt -----------------------------------------------------------------------------


@pytest.fixture
def link_sessions(monkeypatch):
    module = emulation.session_module("sircl_emulated_column_gather")
    module.AllReduce = type("AllReduce", (LinkGathers, module.AllReduce), {"link": "chain"})
    monkeypatch.setitem(sys.modules, "sircl_emulated_column_gather", module)
    monkeypatch.setenv("SIRCL_SESSION_MODULE", "sircl_emulated_column_gather")
    monkeypatch.setenv("SIRCL_ALLREDUCE_CAPACITY_BYTES", str(1 << 20))
    monkeypatch.setenv("SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES", str(64 << 10))
    monkeypatch.setenv("SIRCL_ALLGATHER_MAX_BYTES", str(48 << 10))
    for name in ("SIRCL_PEER_ROUTES", "NCCL_ALGO", "NCCL_SKIP_TREE_CONNECT", "SIRCL_TOPOLOGY", "SIRCL_COLUMN_GATHER"):
        monkeypatch.delenv(name, raising=False)
    yield module
    for live in adapter_module.live_adapters():
        live.close()
    guard.reset()


def _tp4(receipt_dir, environ):
    config = AdapterConfig(Layout.parse("ring:8"), tuple(range(4)), ("tp", "dcp"), "topology", "auto", None,
                           str(receipt_dir))
    groups = emulation.emulated_groups([0, 1, 2, 3])

    def body(rank):
        placement = GroupPlacement.of("tp:0", [0, 1, 2, 3], rank, config, environ=environ)
        return GroupAdapter(placement, config=config, cpu_group=groups[rank], device_group=object(),
                            device=torch.device("cpu"), nccl=None, single_node=False, environ=environ,
                            capturing=lambda: False)

    return run_ranks(4, body)


@pytest.mark.parametrize("switch,route", [(None, "chain"), ("0", None)])
def test_the_adapter_stages_column_gathers_and_reports_them(link_sessions, tmp_path, switch, route):
    environ = {} if switch is None else {"SIRCL_COLUMN_GATHER": switch}
    adapters = _tp4(tmp_path, environ)
    shards = _inputs((512, 72), torch.bfloat16, seed=43)
    rows = _inputs((512, 72), torch.bfloat16, seed=47)
    results = run_ranks(4, lambda rank: (adapters[rank].all_gather(shards[rank], -1),
                                         adapters[rank].all_gather(rows[rank], 0)))
    for gathered, stacked in results:
        assert _same(gathered, torch.cat(shards, dim=-1)) and _same(stacked, torch.cat(rows, dim=0))
    first = adapters[0]
    kinds = [op[0] for op in first.session.ops]
    assert kinds == ([route, "chain"] if route else ["tiles", "chain"])
    assert first.column_gather.calls == ({route: 1} if route else {})
    first.write_receipt()
    record = json.loads((tmp_path / "rank0-tp-0.json").read_text())
    assert record["column_gather"] == ("on" if route else "off")
    assert record["column_gather_detail"] == {"enabled": route is not None, "calls": {route: 1} if route else {},
                                              "staging_bytes": 4 * 73728 if route else 0}
    assert f"column_gather={'on' if route else 'off'}" in adapter_module.receipt.line(record)


# -- the serve bundle's option -------------------------------------------------------------------------------


def test_the_bundle_option_sets_the_switch_only_when_given():
    pytest.importorskip("yaml")
    from sparkring_sircl.vllm.serve import bundle, staging
    from sparkring_sircl.vllm.serve.sitefile import ServeSite

    site = {"schema": "sircl-ring-site/v1", "image": "sha256:aba309e4610c", "lan_interface": "enP7s7",
            "control_port": 29650, "remote_dir": "/tmp/sircl-ring", "docker": "sudo -n docker",
            "ring": [{"name": f"spark{i}", "ssh": f"user@192.0.2.{20 + i}", "lan_address": f"192.0.2.{20 + i}"}
                     for i in range(8)]}

    def environment(**options):
        plan = bundle.build_bundle(ServeSite.from_json(site), bundle.BundleOptions(positions=tuple(range(8)), **options),
                                   staged_digest=staging.staged_tree().digest, library=staging.library_name())
        return [rank.environment for rank in plan.ranks]

    assert all("SIRCL_COLUMN_GATHER" not in env for env in environment())
    assert {env["SIRCL_COLUMN_GATHER"] for env in environment(column_gather=False)} == {"0"}
    assert {env["SIRCL_COLUMN_GATHER"] for env in environment(column_gather=True)} == {"1"}
    parser = argparse.ArgumentParser()
    bundle.add_arguments(parser.add_subparsers(dest="command"))
    base = ["bundle", "--site", "site.json", "--positions", "0-7"]
    assert parser.parse_args(base).column_gather is None
    assert parser.parse_args(base + ["--column-gather", "off"]).column_gather == "off"
