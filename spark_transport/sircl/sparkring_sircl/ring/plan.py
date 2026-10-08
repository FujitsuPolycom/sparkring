"""Configurations, route maps and the launch plan of the ring harness (offline).

A configuration is one or more groups that run at the same time; every Spark
belongs to at most one group. Built-in configurations on a ring of ``N``:

- ``pairs``: adjacent pairs (0-1, 2-3, ...), each owning the cable between its
  two Sparks;
- ``path4``: Sparks 0-3, a path whose end ranks reach each other through two
  relays;
- ``two-tp4``: Sparks 0-3 and 4-7 at the same time (``N >= 8``);
- ``ring``: every Spark of the ring;
- ``path4-large``: Sparks 0-3, large all-reduces only (1, 8, 32 and 64 MiB),
  each compared with the targets of a 64 MiB BF16 all-reduce on a path of four
  (8 ms, stretch 5 ms) and with NCCL's bus bandwidth on a four-Spark cycle
  (10-13 GB/s), and reduce-scatters of ``[rows, 4096]`` BF16 at 8, 32 and
  64 MiB, each checked bit for bit against the host reference and the same
  rows of the session's all-reduce in ops of at most the capacity (the
  rank-ordered float32 sum rounded once; a chain op rounds once per hop, so
  the chain schedule's rows can differ in the last place and are only
  counted); a reduce-scatter carries half the bytes of a two-shot all-reduce over the
  path's middle cable, so its 64 MiB targets are half the all-reduce's (4 ms,
  stretch 2.5 ms); all-gathers of ``[rows, 4096]`` BF16 shards along
  dimension 0 (the mHC gather) at 4 and 16 MiB per rank, as tiles of the
  session's all-gather and as chain all-gathers (each rank's shard travels
  toward both ends of the path over the chain links, forwarded by the
  progress threads) with pieces of 256 and 512 KiB; the 16 MiB rows are
  marked against 3 ms (stretch 2.5 ms; the path bound is 0.75 of the 64 MiB
  output per cable direction, about 2.1 ms). The all-reduces run both as two-shot pieces and as chain
  ops (``all_reduce_large`` with the chain schedule: neighbors only, each
  cable direction carrying the message once) with chunks of 256 KiB, 512 KiB
  and 1 MiB; the 64 MiB chain rows are marked against 4 ms (stretch 3.3 ms);
- ``two-tp4-large``: the ``path4-large`` cases on Sparks 0-3 and 4-7 at the
  same time;
- ``path4-crossover``: Sparks 0-3, all-reduces of 256 KiB to 4 MiB, all-gathers
  of the same output sizes (``[rows, 4096]`` BF16 shards along dimension 0)
  and reduce-scatters of the same input sizes, each with the ring, the chain
  and the pieces, tiles or scatter ops, the ring rows at every size (the
  session's ring minimum does not apply to a ring row): the data for the
  default chain and ring minimums (``SIRCL_CHAIN_MIN_BYTES``,
  ``SIRCL_RING_MIN_BYTES``);
- ``ring-large`` (or ``ring8-large`` on a ring of eight): the whole ring, large
  all-reduces only (8, 32, 64 and 96 MiB; 96 MiB is an 8,192-token prefill
  chunk at hidden size 6,144), as two-shot pieces (lanes through up to three
  relays) and as chain ops over the cycle used as a chain (neighbors only, the
  closing cable unused), with the chain chunk sweep;
- ``dcp4``: the decode context parallel (DCP) exchanges of GLM-5.3 at tensor
  parallel size 8 with DCP groups of four, on a ring of exactly eight: groups
  0-3 and 4-7 (paths whose ends reach each other through the two middle
  Sparks) run at the same time, and every rank also holds a tensor-parallel
  session over the whole ring (the ``world_session`` option). Per row count,
  at decode (1 to 128 rows, eager and in CUDA graph replay) and for a prefill
  chunk of 8,192 rows (eager): the all-to-all of vLLM's ``a2a`` combine
  (``[4, rows, 8, 514]`` BF16: 8 heads per rank after the combine, each with
  512 latent output values and the FP32 log-sum-exp in two BF16 slots), the
  query all-gather (``[rows, 8, 576]`` BF16 along the heads), the all-gather of
  the indexer's top-k candidates (``[rows, 2048, 2]`` float32, 16 KiB per row),
  and at decode the tensor-parallel all-reduce of ``[rows, 6144]`` BF16 on the
  world session. Every output is checked bit for bit against a host reference;
- ``ring-swing`` (or ``ring8-swing`` on a ring of eight; a ring whose size is a
  power of two): the whole ring, BF16 all-reduces of 1 MiB, 1 MiB + 16 bytes,
  1.5 MiB and 2 MiB with the Swing schedule (``2 log2(N)`` phases, each toward
  one peer at ring distance 1 or ``|rho(k)|``; on the ring of eight 1/2, 1/4,
  1/8, 1/8, 1/4 and 1/2 of the message per phase at distances 1, 1, 3, 3, 1
  and 1) beside the session's default all-reduce of the same sizes (two-shot),
  eager and in CUDA graph replay. Swing outputs are checked bit for bit against
  the Swing reference (partial sums rounded to BF16 after every step, identical
  on every rank), the others against the rank-ordered sum;
- ``ring-latency`` (or ``ring8-latency`` on a ring of eight): the whole ring,
  small all-reduces for the one-shot to two-shot crossover
  (``SIRCL_ONESHOT_MAX_BYTES``) and the progress thread's posting order. Per
  message size (4 KiB to 64 KiB, and 96 and 128 KiB, the decode all-reduce
  sizes up to the session's default crossover), the one-shot and the two-shot
  all-reduce with each posting order (``rank`` and ``ring-farthest`` unless
  ``post_orders`` says otherwise), the orders of one size and algorithm run
  back to back, eager and in CUDA graph replay; every output is checked bit for
  bit against the rank-ordered sum. Comparing orders needs a session that
  switches its posting order at run time (``set_post_order``); without it only
  the first order runs. The summary prints the crossover per order and mode and,
  beside every row, :mod:`sparkring_sircl.latency_model` for the row's posting
  order;
- ``path4-latency``: the ``ring-latency`` cases on Sparks 0-3, a path whose end
  ranks reach each other through two relays, with the posting orders ``rank``
  and ``farthest`` (``ring-farthest`` folds the group size, which does not
  follow a path). A session without ``set_post_order`` builds ``farthest`` as
  every rank's explicit peer list (``posting.resolve`` on the plan's layout).

A group is either the whole ring or consecutive Sparks along it (a path that
owns only the cables between its members). With the ``world_session`` option
every rank also holds a session over the whole ring, as the tensor-parallel
group of a deployment whose DCP groups are the configuration's groups; global
rank ``i`` then sits on Spark ``i``. Route maps follow
:mod:`sparkring_sircl.routes`; the plan lists every rank's map, its relays,
the relay load factor of each group, the forward windows of its relayed lanes
and the sizes that would exceed the relay queue rule without them.

Options set the CPU placement of every rank (``performance``: the launching
thread on performance cores and the progress thread on a performance core of
its own; ``none``: no pinning), session variables of every rank
(``session_env``: documented ``SIRCL_*`` names of :mod:`sparkring_sircl.env`
without a harness option of their own, for example the chain geometry) and,
with ``large``, the large-message cases:
two-shot all-reduces up to the capacity, ``all_reduce_large`` above it, and
``all_gather_large`` of flat and row-shaped shards.
"""

from __future__ import annotations

import dataclasses
import hashlib
import shlex
from collections.abc import Sequence
from pathlib import Path

from .. import env as env_mod
from . import nccl as nccl_mod
from .. import routes as routes_mod
from .site import DEFAULT_DOCKER, Site

ALLREDUCE_SIZES = tuple(sorted(
    {16 << shift for shift in range(14)}
    | {8192 - 16, 8192 + 16, 28672 - 16, 28672, 28672 + 16, 32768 - 16, 32768 + 16, 65536 - 16,
       65536 + 16, 131072 - 16}
))
ALLGATHER_SIZES = tuple(sorted({16 << shift for shift in range(14)} | {28672, 155648 - 16, 155648}))
# (rows, columns) of BF16 shards gathered along the last dimension; the odd ones take the padded path.
ALLGATHER_SHAPES = ((64, 1216), (5, 1000), (4, 9), (3, 7), (1, 77824))
# Large-message cases (``--large``): all-reduce message sizes and all-gather shard sizes.
LARGE_ALLREDUCE_SIZES = (256 << 10, 1 << 20, 2 << 20, 8 << 20, 32 << 20, 64 << 20, 96 << 20)
LARGE_ALLGATHER_SIZES = (1 << 20, 4 << 20, 16 << 20)
LARGE_CAPACITY = 2 << 20
# The path4-large configuration: message sizes, and (bytes, target ms, stretch ms).
PATH4_LARGE_SIZES = (1 << 20, 8 << 20, 32 << 20, 64 << 20)
PATH4_LARGE_TARGETS = ((64 << 20, 8.0, 5.0),)
# Chain all-reduces in the path4-large configurations: chunk sizes swept, and (bytes, target ms, stretch ms).
CHAIN_CHUNK_SWEEP = (256 << 10, 512 << 10, 1 << 20)
PATH4_CHAIN_TARGETS = ((64 << 20, 4.0, 3.3),)
# The sessions' chain slot (SIRCL_CHAIN_SLOT_BYTES, left at its default by the harness): the largest
# chain chunk a session accepts.
CHAIN_SLOT_BYTES = 1 << 20
LARGE_SCHEDULES = ("auto", "pieces", "chain", "ring")
# Chain all-gathers in the path4-large configurations: shard bytes per rank of [rows, 4096] BF16 along
# dimension 0, link piece sizes swept, and (shard bytes, target ms, stretch ms).
PATH4_CHAIN_GATHER_SIZES = (4 << 20, 16 << 20)
# Message bytes of the path4-crossover sweep (all-reduce messages, all-gather outputs, reduce-scatter
# inputs) around the ring minimum.
CROSSOVER_SIZES = (256 << 10, 512 << 10, 1 << 20, 2 << 20, 4 << 20)
LINK_CHUNK_SWEEP = (256 << 10, 512 << 10)
PATH4_GATHER_TARGETS = ((16 << 20, 3.0, 2.5),)
# Ring rows of the path4-large configurations (the ring closed through the relays; bounds 4.19 ms and
# 2.1 ms) and of ring-large on a ring of eight (bound 7.34 ms for 96 MiB): (collective, bytes, target ms,
# stretch ms); a reduce-scatter's bytes are its input, an all-gather's its shard.
PATH4_RING_TARGETS = (("all_reduce", 64 << 20, 4.5, 4.3), ("reduce_scatter", 64 << 20, 2.3, 2.2),
                      ("all_gather", 16 << 20, 2.3, 2.2))
RING8_RING_TARGETS = (("all_reduce", 96 << 20, 8.0, 7.6),)
# The sessions' link slot (SIRCL_LINK_SLOT_BYTES unless --session-env sets it): the largest link piece.
LINK_SLOT_BYTES = 512 << 10
GATHER_ROW_BYTES = 4096 * 2
# The ring-large configuration: whole-ring all-reduce sizes (96 MiB: 8,192 tokens at hidden size 6,144).
RING_LARGE_SIZES = (8 << 20, 32 << 20, 64 << 20, 96 << 20)
# The tune command's collectives and sizes: per-rank bytes from 4 KiB to 128 MiB, every power of two, or
# every fourth with --quick.
TUNE_COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")
TUNE_SIZES = tuple(4096 << shift for shift in range(16))
TUNE_QUICK_SIZES = tuple(4096 << shift for shift in range(0, 16, 2))
# Grid caps --large-blocks may name without SIRCL_LARGE_BLOCKS: the session's counters cover grids up to the
# larger of SIRCL_LARGE_BLOCKS and 32 blocks.
LARGE_BLOCKS_SWEEP_LIMIT = 32
# Largest link piece or chain chunk the tune command sweeps; the worker sizes the link slots to hold it.
TUNE_MAX_PIECE = 2 << 20
# The tune command prunes no candidate below this size: there large-message families still amortize their
# fixed costs, and every candidate is cheap to measure.
TUNE_PRUNE_FROM = 4 << 20
# Session variables a run cannot set through --session-env: the harness derives them per rank, or sets
# them from options of their own.
HARNESS_SESSION_VARIABLES = {
    "SIRCL_PEER_ROUTES": "the plan's route maps", "SIRCL_LAYOUT": "the plan's group layouts",
    "SIRCL_DEVICES": "the plan's route maps", "SIRCL_LARGE_PIECE_BYTES": "--large-piece",
    "SIRCL_FORWARD_WINDOW_BYTES": "--forward-window", "SIRCL_STARTUP_WAIT_S": "--startup-wait",
    "SIRCL_SERVING_WAIT_S": "--serving-wait", "SIRCL_SPIN_LIMIT": "--spin-limit",
    "SIRCL_LARGE_SCHEDULE": "--large-schedules", "SIRCL_CHAIN_CHUNK_BYTES": "--chain-chunks",
    "SIRCL_TUNING_TABLE": "--tuning-table",
    "SIRCL_CALL_PROFILE": "--eager-profile",
}
# Eager calls a session's call profile keeps under --eager-profile (more than any case's timed calls).
EAGER_PROFILE_CALLS = 8192
EAGER_PATHS = ("session", "adapter")
# Reduce-scatters of [rows, 4096] BF16 in the path4-large configuration, and their (bytes, target ms, stretch ms).
PATH4_LARGE_REDUCE_SCATTER_SIZES = (8 << 20, 32 << 20, 64 << 20)
PATH4_LARGE_REDUCE_SCATTER_TARGETS = ((64 << 20, 4.0, 2.5),)
REDUCE_SCATTER_ROW_BYTES = 4096 * 2
# The dcp4 configuration: GLM-5.3's DCP exchanges at tensor-parallel size 8 (64 query heads, 8 per rank)
# with DCP groups of four, in vLLM's a2a combine mode.
DCP_HEADS = 8                      # query heads per rank after the combine
DCP_A2A_VALUES = 512 + 2           # per head: 512 latent output values, the FP32 log-sum-exp in two BF16 slots
DCP_QUERY_VALUES = 512 + 64        # per head of the query gather: latent and RoPE values (BF16)
DCP_INDEXER_ROW_BYTES = 2048 * 2 * 4   # per row of the indexer gather: 2048 (score, position) float32 pairs
DCP_DECODE_ROWS = (1, 2, 4, 8, 16, 32, 64, 128)
DCP_PREFILL_ROWS = (8192,)
TP_HIDDEN = 6144                   # hidden width of the tensor-parallel all-reduce
# The ring-swing configuration: Swing all-reduce message sizes (BF16 bytes), within the 2 MiB capacity.
RING_SWING_SIZES = (1 << 20, (1 << 20) + 16, 3 << 19, 2 << 20)
# The ring-latency configuration: all-reduce sizes (BF16 bytes; 12 KiB per decode row at hidden size 6,144)
# and the posting orders compared.
RING_LATENCY_SIZES = (4 << 10, 8 << 10, 12 << 10, 16 << 10, 24 << 10, 28 << 10, 32 << 10, 36 << 10, 48 << 10,
                      64 << 10, 96 << 10, 128 << 10)
RING_LATENCY_POST_ORDERS = ("rank", "ring-farthest")
PATH4_LATENCY_POST_ORDERS = ("rank", "farthest")
POST_ORDERS = ("rank", "ring-farthest", "farthest")
NCCL_CYCLE_BUSBW_GBPS = (10.0, 13.0)
HAIRPIN_QUEUE_BYTES = routes_mod.DEFAULT_HAIRPIN_QUEUE
RELAY_QUEUE_SHARE = routes_mod.RELAY_QUEUE_SHARE
CPU_POLICIES = ("performance", "none")
CONFIGURATIONS = ("pairs", "path4", "two-tp4", "ring", "path4-large", "two-tp4-large", "ring-large", "dcp4",
                  "ring-swing", "ring-latency", "path4-latency", "path4-crossover")
CONTAINER_PREFIX = "sircl-ring"
PACKAGE = Path(__file__).resolve().parents[1]


class PlanError(ValueError):
    """A configuration cannot run on the described ring."""


def builtin_groups(name: str, ring_size: int) -> tuple[tuple[int, ...], ...]:
    if name == "pairs":
        return tuple((i, i + 1) for i in range(0, ring_size - 1, 2))
    if name in ("path4", "path4-large", "path4-latency", "path4-crossover"):
        if ring_size < 5:
            raise PlanError("path4 needs a ring of at least five Sparks (four consecutive, not the whole ring)")
        return ((0, 1, 2, 3),)
    if name in ("two-tp4", "two-tp4-large"):
        if ring_size < 8:
            raise PlanError("two-tp4 needs a ring of at least eight Sparks")
        return ((0, 1, 2, 3), (4, 5, 6, 7))
    if name == "dcp4":
        if ring_size != 8:
            raise PlanError("dcp4 needs a ring of exactly eight Sparks: its tensor-parallel session spans the ring")
        return ((0, 1, 2, 3), (4, 5, 6, 7))
    if name in ("ring-latency", f"ring{ring_size}-latency"):
        return (tuple(range(ring_size)),)
    if name in ("ring-swing", f"ring{ring_size}-swing"):
        if ring_size < 2 or ring_size & (ring_size - 1):
            raise PlanError("ring-swing needs a ring whose size is a power of two: Swing pairs ranks by powers of two")
        return (tuple(range(ring_size)),)
    if name in ("ring", "ring-large") or name in (f"ring{ring_size}", f"ring{ring_size}-large"):
        return (tuple(range(ring_size)),)
    raise PlanError(f"unknown configuration {name!r}; built-in ones are {', '.join(CONFIGURATIONS)}")


def parse_groups(text: str) -> tuple[tuple[int, ...], ...]:
    """``0-3;4-7`` or ``0,1;2,3``: groups separated by ``;``, members by ``,`` or ``a-b`` ranges."""
    groups = []
    for part in text.split(";"):
        members: list[int] = []
        for item in part.split(","):
            item = item.strip()
            if not item:
                continue
            first, _, last = item.partition("-")
            try:
                members.extend(range(int(first), int(last or first) + 1))
            except ValueError:
                raise PlanError(f"group {part!r} is not a list of Spark positions") from None
        if members:
            groups.append(tuple(members))
    if not groups:
        raise PlanError("no groups given")
    return tuple(groups)


def group_layout(ring_size: int, members: Sequence[int]) -> routes_mod.Layout:
    """The group's fabric: the whole ring, or the cables between consecutive members."""
    members = tuple(members)
    if len(members) < 2 or len(set(members)) != len(members):
        raise PlanError(f"group {members} needs at least two distinct Sparks")
    if any(not 0 <= member < ring_size for member in members):
        raise PlanError(f"group {members} names a Spark outside the ring of {ring_size}")
    if len(members) == ring_size:
        if sorted(members) != list(range(ring_size)):
            raise PlanError("a whole-ring group lists every Spark once")
        return routes_mod.Layout(routes_mod.Fabric.ring(ring_size), members)
    for a, b in zip(members, members[1:]):
        if b != (a + 1) % ring_size:
            raise PlanError(f"group {members} is neither the whole ring nor consecutive Sparks along it")
    fabric = routes_mod.Fabric(tuple(routes_mod.Cable(a, 0, b, 1) for a, b in zip(members, members[1:])))
    return routes_mod.Layout(fabric, members)


def ring_text(layout: routes_mod.Layout, maps) -> str:
    """The ring that closes a group's chain: closed by cables, or through relays with one ring lane
    per relay hairpin queue (the sessions' check, ``routes.ring_window``), or why it cannot run."""
    order = routes_mod.chain_order(layout, maps)
    if order is None:
        return "no ring: the ranks do not form a chain of cable neighbors"
    window, problems = routes_mod.ring_window(layout, maps, order)
    if problems:
        return "no ring: " + "; ".join(problems)
    queues = routes_mod.ring_queues(layout, maps, order)
    if not queues:
        return f"ring over {list(order)}, every edge a cable"
    relays = sorted({key[0] for key in queues})
    return (f"ring over {list(order)}: rank {order[-1]} reaches rank {order[0]} through the relays of Sparks {relays}, one ring "
            f"lane per relay hairpin queue ({len(queues)} queues), a window of {window} bytes per lane")


def layout_text(layout: routes_mod.Layout) -> str:
    cables = ",".join(cable.text for cable in layout.fabric.cables)
    return f"cables={cables};positions={','.join(str(p) for p in layout.positions)}"


@dataclasses.dataclass(frozen=True)
class GroupPlan:
    index: int
    positions: tuple[int, ...]
    global_ranks: tuple[int, ...]
    layout: str
    lanes: int
    max_relays: int
    relay_lanes: int
    relay_load: float
    route_texts: tuple[str, ...]
    lanes_detail: tuple[dict, ...]
    warnings: tuple[str, ...]
    ring: str = ""          # the ring that closes the group's chain: its relayed lanes, or why it cannot run
    tuning_table: str = ""  # hash of the tuning table whose key matches the group (--tuning-table), else empty
    # (name, value) of that table's settings (tuning.SETTINGS), which its sessions take where --session-env
    # leaves them unset
    tuning_settings: tuple[tuple[str, int], ...] = ()


@dataclasses.dataclass(frozen=True)
class RankPlan:
    global_rank: int
    group: int
    group_rank: int
    position: int
    host: str
    ssh: str
    lan_address: str
    container: str
    peer_routes: str
    docker: str = DEFAULT_DOCKER        # the Spark's Docker command (site file ``docker``)
    world_peer_routes: str = ""         # route map of the world session (``world_session``), else empty


@dataclasses.dataclass(frozen=True)
class Options:
    correctness_iterations: int = 3
    eager_iterations: int = 300
    graph_iterations: int = 1000
    warmup_iterations: int = 20
    spin_limit: int = 5_000_000
    lane_check_ms: int = 2000
    worker_timeout_s: int = 1500
    seed: int = 20261006
    transport_only: bool = False
    allreduce_sizes: tuple[int, ...] = ALLREDUCE_SIZES
    allgather_sizes: tuple[int, ...] = ALLGATHER_SIZES
    allgather_shapes: tuple[tuple[int, int], ...] = ALLGATHER_SHAPES
    cpu_policy: str = "performance"
    post_barrier_warmup: int = 10
    large: bool = False
    large_capacity: int = LARGE_CAPACITY
    large_allreduce_sizes: tuple[int, ...] = LARGE_ALLREDUCE_SIZES
    large_allgather_sizes: tuple[int, ...] = LARGE_ALLGATHER_SIZES
    large_iterations: int = 20
    large_only: bool = False            # skip the small cases (the path4-large configuration)
    large_piece_bytes: int | None = None   # SIRCL_LARGE_PIECE_BYTES; None: the session default
    targets: tuple[tuple[int, float, float], ...] = ()   # (bytes, target ms, stretch ms) of all-reduces
    large_reduce_scatter_sizes: tuple[int, ...] = ()   # reduce-scatters of [rows, 4096] BF16 (bytes per rank)
    reduce_scatter_targets: tuple[tuple[int, float, float], ...] = ()   # (bytes, target ms, stretch ms)
    large_schedules: tuple[str, ...] = ("auto",)   # SIRCL_LARGE_SCHEDULE of each large all-reduce case
    chain_chunks: tuple[int, ...] = ()            # chain chunk sizes swept; empty: the session default
    chain_targets: tuple[tuple[int, float, float], ...] = ()   # (bytes, target ms, stretch ms) of chain ops
    chain_gather_sizes: tuple[int, ...] = ()    # all-gathers of [rows, 4096] BF16 along dim 0 (bytes per rank)
    gather_schedules: tuple[str, ...] = ("auto",)   # SIRCL_GATHER_SCHEDULE of each of those cases
    link_chunks: tuple[int, ...] = ()           # link piece sizes swept; empty: the session default
    gather_link_chunks: tuple[int, ...] = ()    # all-gather link pieces swept; empty: link_chunks
    scatter_link_chunks: tuple[int, ...] = ()   # reduce-scatter link pieces swept; empty: link_chunks
    reduce_link_chunks: tuple[int, ...] = ()    # ring all-reduce link pieces swept; empty: link_chunks
    gather_targets: tuple[tuple[int, float, float], ...] = ()   # (bytes per rank, target ms, stretch ms)
    scatter_schedules: tuple[str, ...] = ("auto",)   # SIRCL_SCATTER_SCHEDULE of each reduce-scatter case
    ring_targets: tuple[tuple[str, int, float, float], ...] = ()   # (collective, bytes, target, stretch ms)
    host_send_gbps: float = 24.0                # a Spark's NIC, host memory to network (the summary's bounds)
    host_recv_gbps: float = 26.8                # a Spark's NIC, network to host memory (the summary's bounds)
    cable_gbps: float = 24.0                    # one cable direction over both lanes (the summary's bounds)
    startup_wait_s: float = 300.0       # SIRCL_STARTUP_WAIT_S: setup, compilation and warm-up
    serving_wait_s: float = 20.0        # SIRCL_SERVING_WAIT_S: the timed cases
    forward_window: int | None = None   # SIRCL_FORWARD_WINDOW_BYTES of every session; None: the default
    dcp_decode_rows: tuple[int, ...] | None = None    # DCP exchanges at decode row counts (eager and graph)
    dcp_prefill_rows: tuple[int, ...] | None = None   # DCP exchanges at prefill chunk row counts (eager)
    dcp_heads: int = DCP_HEADS               # query heads per rank after the a2a combine
    dcp_gathers: bool = False                # with the DCP query and indexer all-gathers
    world_session: bool = False              # every rank also holds a session over the whole ring
    tp_hidden: int = TP_HIDDEN               # row width of the world session's all-reduce at decode
    swing_sizes: tuple[int, ...] = ()        # Swing all-reduce message sizes (BF16 bytes), beside the default
    latency_sizes: tuple[int, ...] = ()      # one-shot and two-shot all-reduce sizes (BF16 bytes), per posting order
    post_orders: tuple[str, ...] = ()        # posting orders of the latency cases; the first builds the session
    latency_relay_us: float = 0.75           # latency model: added one-way latency per NIC relay (us)
    latency_post_us: float = 0.30            # latency model: posting time per lane where not measured (us)
    latency_write_us: float = 0.0            # latency model: one-way latency of a direct write (us; 0: excluded)
    session_env: tuple[tuple[str, str], ...] = ()   # (name, value) session variables set on every rank
    large_blocks: tuple[int, ...] = ()       # grid caps of two-shot and large-message launches swept per case
    baseline: str = ""                       # "nccl": NCCL's all-reduce and all-gather beside every case
    nccl_library: str = ""                   # the NCCL library of the baseline; empty: nccl.DEFAULT_LIBRARY
    nccl_env: tuple[tuple[str, str], ...] = ()   # (name, value) overrides of the baseline's NCCL environment
    tune: bool = False                       # the tune command: every candidate per collective, size and mode
    tune_collectives: tuple[str, ...] = TUNE_COLLECTIVES
    tune_sizes: tuple[int, ...] = TUNE_SIZES     # per-rank bytes (all-reduce message, all-gather shard, ...)
    tune_modes: tuple[str, ...] = ("graph", "eager")
    tune_grids: tuple[int, ...] = (4, 8, 16, 32)    # launch grid caps of the one-shot, two-shot, tiles, scatter ops
    tune_pieces: tuple[int, ...] = (262144, 524288, 1048576)   # chain chunks and link pieces
    tune_staggers: tuple[int, ...] = (0, 1)  # ring staggers of link 2 (reduce-scatter) and link 3 (all-gather)
    tune_large_from: int = 262144            # smallest size of the pieces, chain and ring candidates
    tune_prune: float = 1.5                  # slower than the fastest by this factor at two sizes in a row: dropped
    tune_prune_from: int = TUNE_PRUNE_FROM   # smallest size at which a candidate can be dropped
    tuning_tables: tuple[str, ...] = ()      # local paths of the tuning tables the sessions choose from
    eager_profile: bool = False              # eager rows record the session's call stage times (SIRCL_CALL_PROFILE)
    eager_path: str = "session"              # adapter: small and large collectives through the vLLM adapter's planner

    def __post_init__(self) -> None:
        if self.cpu_policy not in CPU_POLICIES:
            raise PlanError(f"CPU policy must be one of {', '.join(CPU_POLICIES)}, got {self.cpu_policy!r}")
        if self.large_capacity < 4096 or self.large_capacity % 4096:
            raise PlanError("the large-message capacity must be a positive multiple of 4096 bytes")
        sizes = (*self.large_allreduce_sizes, *self.large_allgather_sizes)
        if any(size <= 0 or size % 16 for size in sizes):
            raise PlanError("large-message sizes are positive multiples of 16 bytes")
        if any(size <= 0 or size % REDUCE_SCATTER_ROW_BYTES for size in self.large_reduce_scatter_sizes):
            raise PlanError("reduce-scatter sizes are positive multiples of one [1, 4096] BF16 row (8192 bytes)")
        if self.forward_window is not None and self.forward_window < 0:
            raise PlanError("the forward window is a byte count; 0 turns forward windows off")
        if self.large_piece_bytes is not None and (self.large_piece_bytes <= 0 or self.large_piece_bytes % 16):
            raise PlanError("the large-message piece is a positive multiple of 16 bytes")
        if not 0 < self.serving_wait_s <= self.startup_wait_s <= 4294:
            raise PlanError("wait limits must satisfy 0 < serving <= startup <= 4294 seconds")
        if not self.large_schedules or any(name not in LARGE_SCHEDULES for name in self.large_schedules):
            raise PlanError(f"large-message schedules are drawn from {', '.join(LARGE_SCHEDULES)}")
        known = {variable.name for variable in env_mod.VARIABLES}
        for name, value in self.session_env:
            if name in HARNESS_SESSION_VARIABLES:
                raise PlanError(f"{name} is set by {HARNESS_SESSION_VARIABLES[name]}, not --session-env")
            if name not in known or not name.startswith("SIRCL_"):
                raise PlanError(f"{name} is not a documented session variable (python -m sparkring_sircl.env)")
            if not value or not all(ch.isalnum() or ch in "._,:/=+-" for ch in value):
                raise PlanError(f"the value of {name} is empty or holds characters other than letters, digits "
                                f"and ._,:/=+-")
        if not self.gather_schedules or any(name not in LARGE_SCHEDULES for name in self.gather_schedules):
            raise PlanError(f"all-gather schedules are drawn from {', '.join(LARGE_SCHEDULES)}")
        if not self.scatter_schedules or any(name not in LARGE_SCHEDULES for name in self.scatter_schedules):
            raise PlanError(f"reduce-scatter schedules are drawn from {', '.join(LARGE_SCHEDULES)}")
        if self.host_send_gbps <= 0 or self.host_recv_gbps <= 0 or self.cable_gbps <= 0:
            raise PlanError("the host interface's send and receive rates and the cable rate are positive GB/s")
        if any(size <= 0 or size % GATHER_ROW_BYTES for size in self.chain_gather_sizes):
            raise PlanError("chain all-gather sizes are positive multiples of one [1, 4096] BF16 row (8192 bytes)")
        link_text = dict(self.session_env).get("SIRCL_LINK_SLOT_BYTES", str(LINK_SLOT_BYTES))
        if not link_text.isdigit():
            raise PlanError(f"SIRCL_LINK_SLOT_BYTES {link_text!r} is not a byte count")
        pieces = (*self.link_chunks, *self.gather_link_chunks, *self.scatter_link_chunks, *self.reduce_link_chunks)
        if any(chunk <= 0 or chunk % 16 or chunk > int(link_text) for chunk in pieces):
            raise PlanError(f"link pieces are positive multiples of 16 bytes up to the link slot of {link_text} "
                            f"bytes (--session-env SIRCL_LINK_SLOT_BYTES=<bytes> for larger pieces)")
        slot_text = dict(self.session_env).get("SIRCL_CHAIN_SLOT_BYTES", str(CHAIN_SLOT_BYTES))
        if not slot_text.isdigit():
            raise PlanError(f"SIRCL_CHAIN_SLOT_BYTES {slot_text!r} is not a byte count")
        slot = int(slot_text)
        if any(chunk <= 0 or chunk % 16 or chunk > slot for chunk in self.chain_chunks):
            raise PlanError(f"chain chunks are positive multiples of 16 bytes up to the chain slot of "
                            f"{slot} bytes")
        if any(rows <= 0 for rows in (*(self.dcp_decode_rows or ()), *(self.dcp_prefill_rows or ())))                 or self.dcp_heads <= 0:
            raise PlanError("DCP row counts and heads are positive")
        if any(size <= 0 or size % 16 for size in self.swing_sizes):
            raise PlanError("Swing all-reduce sizes are positive multiples of 16 bytes")
        if any(size <= 0 or size % 16 for size in self.latency_sizes):
            raise PlanError("latency all-reduce sizes are positive multiples of 16 bytes")
        if any(order not in POST_ORDERS for order in self.post_orders) or len(set(self.post_orders)) != len(
                self.post_orders):
            raise PlanError(f"posting orders are distinct names from {', '.join(POST_ORDERS)}")
        if min(self.latency_relay_us, self.latency_post_us, self.latency_write_us) < 0:
            raise PlanError("latency model parameters are non-negative microseconds")
        if self.tp_hidden <= 0 or self.tp_hidden % 8:
            raise PlanError("the tensor-parallel row width is a positive multiple of 8 BF16 values")
        if self.tune:
            from .. import tuning as tuning_mod

            if not self.tune_collectives or any(c not in tuning_mod.COLLECTIVES for c in self.tune_collectives):
                raise PlanError(f"tune collectives are drawn from {', '.join(tuning_mod.COLLECTIVES)}")
            if not self.tune_modes or any(mode not in tuning_mod.MODES for mode in self.tune_modes):
                raise PlanError(f"tune modes are drawn from {', '.join(tuning_mod.MODES)}")
            if (not self.tune_sizes or list(self.tune_sizes) != sorted(set(self.tune_sizes))
                    or any(size <= 0 or size % 16 for size in self.tune_sizes)):
                raise PlanError("tune sizes are increasing positive multiples of 16 bytes")
            if any(g < 1 or g & (g - 1) or g > LARGE_BLOCKS_SWEEP_LIMIT for g in self.tune_grids):
                raise PlanError(f"tune grids are powers of two up to {LARGE_BLOCKS_SWEEP_LIMIT}")
            if any(p < 4096 or p % 4096 or p > TUNE_MAX_PIECE for p in self.tune_pieces):
                raise PlanError(f"tune pieces are multiples of 4096 bytes up to {TUNE_MAX_PIECE}")
            if any(not 0 <= d <= 4 for d in self.tune_staggers) or self.tune_prune <= 1.0 or self.tune_large_from < 16:
                raise PlanError("tune staggers are 0 to 4, the pruning factor is above 1 and the large-message "
                                "candidates start at a positive size")
        if self.eager_path not in EAGER_PATHS:
            raise PlanError(f"the eager path is one of {', '.join(EAGER_PATHS)}, got {self.eager_path!r}")
        if self.tune_prune_from < 0:
            raise PlanError("the tune pruning starts at a non-negative size")
        if self.tuning_tables and self.tune:
            raise PlanError("a tune run measures its candidates without a tuning table; --tuning-table applies "
                            "to run")
        if any(not str(path).strip() for path in self.tuning_tables):
            raise PlanError("a tuning table is named by its path")
        if self.baseline not in ("", "nccl"):
            raise PlanError(f"the baseline is nccl or none, got {self.baseline!r}")
        if self.nccl_library and not self.nccl_library.startswith("/"):
            raise PlanError("the NCCL library is an absolute path inside the image")
        for name, value in self.nccl_env:
            try:
                nccl_mod.check_override(name, value)
            except nccl_mod.NcclError as error:
                raise PlanError(str(error)) from None
        if self.nccl_env and self.baseline != "nccl":
            raise PlanError("--nccl-env applies to the NCCL baseline (--baseline nccl)")
        grid_text = dict(self.session_env).get("SIRCL_LARGE_BLOCKS", str(LARGE_BLOCKS_SWEEP_LIMIT))
        grid_limit = max(LARGE_BLOCKS_SWEEP_LIMIT, int(grid_text) if grid_text.isdigit() else 0)
        if any(cap < 1 or cap & (cap - 1) or cap > grid_limit for cap in self.large_blocks) \
                or len(set(self.large_blocks)) != len(self.large_blocks):
            raise PlanError(f"grid caps are distinct powers of two up to {grid_limit} (larger caps need "
                            f"--session-env SIRCL_LARGE_BLOCKS=<cap>)")


# The schedule and piece options of the link cases, and the cases each applies to.
CASE_OPTIONS = ("--gather-schedules", "--gather-link-chunks", "--scatter-schedules", "--scatter-link-chunks",
                "--reduce-link-chunks", "--chain-chunks")


def unused_case_options(options: Options, given: Sequence[str]) -> list[str]:
    """Why each of the command-line options ``given`` (from :data:`CASE_OPTIONS`) applies to no case of a
    configuration with ``options``."""
    links = ("chain", "ring")
    gathers = bool(options.large and options.chain_gather_sizes)
    scatters = bool(options.large and options.large_reduce_scatter_sizes)
    reduces = bool(options.large and options.large_allreduce_sizes)
    rules = {
        "--gather-schedules": (gathers, "the link all-gather cases (--chain-gather-sizes)"),
        "--gather-link-chunks": (gathers and any(s in links for s in options.gather_schedules),
                                 "the chain and ring all-gather cases (--chain-gather-sizes with --gather-schedules "
                                 "chain or ring)"),
        "--scatter-schedules": (scatters, "the reduce-scatter cases (--reduce-scatter-sizes)"),
        "--scatter-link-chunks": (scatters and any(s in links for s in options.scatter_schedules),
                                  "the chain and ring reduce-scatter cases (--reduce-scatter-sizes with "
                                  "--scatter-schedules chain or ring)"),
        "--reduce-link-chunks": (reduces and "ring" in options.large_schedules,
                                 "the ring all-reduce cases (--large-sizes with --large-schedules ring)"),
        "--chain-chunks": (reduces and "chain" in options.large_schedules,
                           "the chain all-reduce cases (--large-sizes with --large-schedules chain)"),
    }
    return [f"{name} applies to no case: it sweeps {rules[name][1]}" for name in given if not rules[name][0]]


def configuration_options(name: str, options: Options) -> Options:
    """The options of one configuration: the ``-large`` configurations run their large cases only, ``dcp4``
    its DCP exchanges with the world session."""
    if options.tune:
        if options.transport_only:
            raise PlanError("tune measures CUDA kernels; the transport-only mode does not apply")
        return dataclasses.replace(options, large=True, large_only=True, large_allreduce_sizes=(),
                                   large_allgather_sizes=(), dcp_decode_rows=(), dcp_prefill_rows=(),
                                   latency_sizes=(), swing_sizes=())
    if name == "dcp4":
        if options.transport_only:
            raise PlanError("dcp4 checks CUDA kernels; the transport-only mode does not apply")
        return dataclasses.replace(options, large=True, large_only=True, large_allreduce_sizes=(),
                                   large_allgather_sizes=(),
                                   dcp_decode_rows=(DCP_DECODE_ROWS if options.dcp_decode_rows is None
                                                    else options.dcp_decode_rows),
                                   dcp_prefill_rows=(DCP_PREFILL_ROWS if options.dcp_prefill_rows is None
                                                     else options.dcp_prefill_rows),
                                   dcp_gathers=True, world_session=True)
    if name == "path4-latency" or name == "ring-latency" or (name.startswith("ring") and name.endswith("-latency")):
        if options.transport_only:
            raise PlanError(f"{name} checks CUDA kernels; the transport-only mode does not apply")
        if "SIRCL_POST_ORDER" in dict(options.session_env):
            raise PlanError(f"{name} sets the posting orders itself (--post-orders), not --session-env")
        orders = PATH4_LATENCY_POST_ORDERS if name == "path4-latency" else RING_LATENCY_POST_ORDERS
        return dataclasses.replace(options, large=True, large_only=True, large_allreduce_sizes=(),
                                   large_allgather_sizes=(),
                                   latency_sizes=options.latency_sizes or RING_LATENCY_SIZES,
                                   post_orders=options.post_orders or orders)
    if name == "ring-swing" or (name.startswith("ring") and name.endswith("-swing")):
        if options.transport_only:
            raise PlanError("ring-swing checks CUDA kernels; the transport-only mode does not apply")
        return dataclasses.replace(options, large=True, large_only=True, large_allreduce_sizes=(),
                                   large_allgather_sizes=(), swing_sizes=options.swing_sizes or RING_SWING_SIZES)
    if name == "ring-large" or (name.startswith("ring") and name.endswith("-large")):
        return dataclasses.replace(options, large=True, large_only=True, large_allreduce_sizes=RING_LARGE_SIZES,
                                   large_allgather_sizes=(), large_schedules=("pieces", "chain", "ring"),
                                   chain_chunks=options.chain_chunks or CHAIN_CHUNK_SWEEP,
                                   link_chunks=options.link_chunks or LINK_CHUNK_SWEEP,
                                   ring_targets=RING8_RING_TARGETS)
    if name == "path4-crossover":
        # Ring against chain against pieces, tiles or scatter ops around the ring minimum.
        return dataclasses.replace(options, large=True, large_only=True,
                                   large_allreduce_sizes=CROSSOVER_SIZES, large_allgather_sizes=(),
                                   large_schedules=("pieces", "chain", "ring"),
                                   chain_gather_sizes=tuple(size // 4 for size in CROSSOVER_SIZES),
                                   gather_schedules=("pieces", "chain", "ring"),
                                   large_reduce_scatter_sizes=CROSSOVER_SIZES,
                                   scatter_schedules=("pieces", "chain", "ring"))
    if name not in ("path4-large", "two-tp4-large"):
        return options
    return dataclasses.replace(options, large=True, large_only=True,
                               large_allreduce_sizes=PATH4_LARGE_SIZES, large_allgather_sizes=(),
                               targets=PATH4_LARGE_TARGETS,
                               large_reduce_scatter_sizes=PATH4_LARGE_REDUCE_SCATTER_SIZES,
                               reduce_scatter_targets=PATH4_LARGE_REDUCE_SCATTER_TARGETS,
                               large_schedules=("pieces", "chain", "ring"),
                               chain_chunks=options.chain_chunks or CHAIN_CHUNK_SWEEP,
                               chain_targets=PATH4_CHAIN_TARGETS,
                               chain_gather_sizes=PATH4_CHAIN_GATHER_SIZES,
                               gather_schedules=("pieces", "chain", "ring"),
                               link_chunks=options.link_chunks or LINK_CHUNK_SWEEP,
                               gather_targets=PATH4_GATHER_TARGETS,
                               scatter_schedules=("pieces", "chain", "ring"),
                               ring_targets=PATH4_RING_TARGETS)


@dataclasses.dataclass(frozen=True)
class ConfigurationPlan:
    name: str
    run_id: str
    source_digest: str
    image: str
    lan_interface: str
    leader_address: str
    control_port: int
    remote_dir: str
    gid_index: int | None
    groups: tuple[GroupPlan, ...]
    ranks: tuple[RankPlan, ...]
    options: Options
    world_layout: str = ""   # fabric of the world session (``world_session``), else empty
    tuning_tables: tuple[tuple[str, str], ...] = ()   # (hash, local path) of every named tuning table
    world_tuning_table: str = ""   # hash of the table that matches the world session, else empty

    @property
    def world(self) -> int:
        return len(self.ranks)

    @property
    def run_dir(self) -> str:
        return f"{self.remote_dir}/runs/{self.run_id}/{self.name}"

    @property
    def capacity(self) -> int:
        largest = max(self.options.allreduce_sizes)
        return max(largest, self.options.large_capacity) if self.options.large else largest

    @property
    def gather_capacity(self) -> int:
        shapes = [rows * cols * 2 for rows, cols in self.options.allgather_shapes]
        largest = max([*self.options.allgather_sizes, *shapes])
        return max(largest, self.options.large_capacity) if self.options.large else largest

    def to_json(self) -> dict:
        return {
            "schema": "sircl-ring-plan/v1",
            "configuration": self.name,
            "run_id": self.run_id,
            "source_digest": self.source_digest,
            "image": self.image,
            "lan_interface": self.lan_interface,
            "rendezvous": f"tcp://{self.leader_address}:{self.control_port}",
            "world": self.world,
            "gid_index": self.gid_index,
            "capacity": self.capacity,
            "gather_capacity": self.gather_capacity,
            "options": dataclasses.asdict(self.options),
            "groups": [dataclasses.asdict(group) for group in self.groups],
            "ranks": [dataclasses.asdict(rank) for rank in self.ranks],
            "world_layout": self.world_layout,
            "tuning_tables": [list(entry) for entry in self.tuning_tables],
            "world_tuning_table": self.world_tuning_table,
        }


def source_digest() -> str:
    """Digest of the package sources the harness ships to the Sparks (16 hex digits)."""
    digest = hashlib.sha256()
    for name, path in staged_files():
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def staged_files() -> list[tuple[str, Path]]:
    """``(name in the staged tree, source)`` of every file the harness ships to the Sparks:
    :func:`package_files` under ``sparkring_sircl/``, and SparkRing's RoCE GID resolver at the top of
    the tree, where :mod:`sparkring_sircl.roce_gid` imports it."""
    from .. import roce_gid

    resolver = roce_gid.source_file()
    files = [(path.relative_to(PACKAGE.parent).as_posix(), path) for path in package_files()]
    return files + [(resolver.name, resolver)]


# Modules of the vLLM adapter subpackage that ``run --eager-path adapter`` calls on the Sparks (the planner
# and executor and what they import); none of them imports vLLM.
ADAPTER_PATH_FILES = frozenset({"__init__.py", "fabric.py", "planner.py", "executor.py"})


def package_files() -> list[Path]:
    """Files of ``sparkring_sircl`` the harness needs (of the vLLM adapter subpackage, only
    :data:`ADAPTER_PATH_FILES`)."""
    files = []
    for path in sorted(PACKAGE.rglob("*")):
        relative = path.relative_to(PACKAGE).parts
        if not path.is_file() or "__pycache__" in relative:
            continue
        if relative[0] == "vllm" and (len(relative) != 2 or relative[1] not in ADAPTER_PATH_FILES):
            continue
        if path.suffix in (".py", ".c", ".h", ".json"):
            files.append(path)
    return files


def build_plan(site: Site, name: str, run_id: str, *, groups: Sequence[Sequence[int]] | None = None,
               options: Options = Options(), digest: str | None = None) -> ConfigurationPlan:
    """The launch plan of one configuration (``groups`` overrides the built-in ones)."""
    chosen = tuple(tuple(g) for g in groups) if groups is not None else builtin_groups(name, site.size)
    used = [p for group in chosen for p in group]
    if len(used) != len(set(used)):
        raise PlanError(f"configuration {name}: a Spark belongs to two groups")
    layouts = [group_layout(site.size, members) for members in chosen]
    derived = [routes_mod.derive_routes(layout, 2) for layout in layouts]
    problems = routes_mod.isolation_problems(derived)
    if problems:
        raise PlanError(f"configuration {name}: " + "; ".join(problems))
    group_plans, rank_plans = [], []
    global_rank = 0
    safe_run = shlex.quote(run_id) == run_id and run_id.replace("-", "").isalnum()
    if not safe_run:
        raise PlanError("run id must be letters, digits and dashes")
    for index, (members, layout, routes) in enumerate(zip(chosen, layouts, derived)):
        if options.baseline == "nccl" and nccl_mod.baseline_kind(layout.fabric.kind, layout.world) is None:
            raise PlanError(f"configuration {name}: the NCCL baseline runs on pairs and whole cycles; group "
                            f"{members} is a {layout.fabric.kind} of {layout.world} Sparks")
        relays = routes.max_relays()
        if relays > routes_mod.DEFAULT_MAX_RELAYS:
            raise PlanError(f"group {members} needs {relays} relays; the qualified limit is "
                            f"{routes_mod.DEFAULT_MAX_RELAYS}")
        busiest, load = routes_mod.relay_load(routes)
        warnings = []
        limit = RELAY_QUEUE_SHARE * HAIRPIN_QUEUE_BYTES
        window = routes_mod.DEFAULT_FORWARD_WINDOW if options.forward_window is None else options.forward_window
        maps = [routes.route_map(rank) for rank in range(layout.world)]
        windows = [routes_mod.forward_windows(layout, maps, rank, max_window=window) for rank in range(layout.world)]
        paced = sorted({w for table in windows for row in table for w in row if w})
        for label, sizes in (("all-reduce", options.allreduce_sizes), ("all-gather", options.allgather_sizes)):
            over = [size for size in sizes if load * size > limit]
            if over:
                pacing = (f"; forward windows of {', '.join(str(w) for w in paced)} bytes pace the relayed lanes"
                          if paced else "; forward windows are off")
                warnings.append(f"{label} sizes from {min(over)} bytes put {load:g} x size above "
                                f"{RELAY_QUEUE_SHARE:.0%} of a {HAIRPIN_QUEUE_BYTES >> 10} KiB relay queue{pacing}")
        details = []
        for rank in range(layout.world):
            for peer, lanes in sorted(routes.ranks[rank].items()):
                for lane in lanes:
                    details.append({"rank": rank, "peer": peer, **lane.to_json(),
                                    "forward_window": windows[rank][peer][lane.lane]})
        ranks_of_group = tuple(range(global_rank, global_rank + layout.world))
        group_plans.append(GroupPlan(index, tuple(members), ranks_of_group, layout_text(layout), 2, relays,
                                     busiest, load, tuple(routes.route_text(r) for r in range(layout.world)),
                                     tuple(details), tuple(warnings), ring_text(layout, maps)))
        for group_rank, position in enumerate(members):
            host = site.host(position)
            rank_plans.append(RankPlan(
                global_rank, index, group_rank, position, host.name, host.ssh, host.lan_address,
                f"{CONTAINER_PREFIX}-{run_id}-{name}-r{global_rank}", routes.route_text(group_rank),
                host.docker,
            ))
            global_rank += 1
    world_layout = ""
    world_facts = None
    if options.world_session:
        # The world session is the tensor-parallel group of a deployment whose DCP groups are the
        # configuration's groups: every Spark of the ring, world rank i on Spark i.
        if [rank.position for rank in rank_plans] != list(range(site.size)):
            raise PlanError(f"configuration {name}: the world session needs every Spark of the ring, global rank "
                            "i on Spark i")
        world = group_layout(site.size, tuple(range(site.size)))
        world_routes = routes_mod.derive_routes(world, 2)
        if world_routes.max_relays() > routes_mod.DEFAULT_MAX_RELAYS:
            raise PlanError(f"configuration {name}: the world session needs {world_routes.max_relays()} relays; "
                            f"the qualified limit is {routes_mod.DEFAULT_MAX_RELAYS}")
        world_layout = layout_text(world)
        world_facts = (world, world_routes.max_relays())
        rank_plans = [dataclasses.replace(rank, world_peer_routes=world_routes.route_text(rank.position))
                      for rank in rank_plans]
    tables: tuple[tuple[str, str], ...] = ()
    world_table = ""
    if options.tuning_tables:
        group_plans, tables, world_table = assign_tuning_tables(options.tuning_tables, group_plans, layouts,
                                                                world_facts)
        from .. import tuning as tuning_mod

        given = dict(options.session_env)
        for group in group_plans:
            conflicts = tuning_mod.settings_conflicts(dict(group.tuning_settings), given)
            if conflicts:
                raise PlanError(f"configuration {name}: group {group.index}'s sessions take tuning table "
                                f"{group.tuning_table}, whose choices need more than --session-env "
                                f"{', '.join(conflicts)}; drop those settings, or name a table tuned under them")
    leader = rank_plans[0].lan_address
    built = ConfigurationPlan(name, run_id, digest or source_digest(), site.image, site.lan_interface,
                              leader, site.control_port,
                              site.remote_dir, site.gid_index, tuple(group_plans), tuple(rank_plans), options,
                              world_layout, tables, world_table)
    if options.latency_sizes and max(options.latency_sizes) > built.capacity:
        raise PlanError(f"configuration {name}: all-reduces of up to {max(options.latency_sizes)} bytes exceed the "
                        f"capacity of {built.capacity} bytes")
    if options.swing_sizes:
        if max(options.swing_sizes) > built.capacity:
            raise PlanError(f"configuration {name}: Swing messages of up to {max(options.swing_sizes)} bytes exceed "
                            f"the capacity of {built.capacity} bytes")
        if any(len(group.positions) & (len(group.positions) - 1) for group in group_plans):
            raise PlanError(f"configuration {name}: Swing needs groups whose size is a power of two")
    return built


def assign_tuning_tables(paths: Sequence[str], groups: Sequence[GroupPlan], layouts: Sequence[routes_mod.Layout],
                         world: tuple[routes_mod.Layout, int] | None = None
                         ) -> tuple[list[GroupPlan], tuple[tuple[str, str], ...], str]:
    """Each group's tuning table among ``paths``: the one whose key matches the group's shape, size, lanes,
    relays and this package's build, as its sessions choose at setup (``tuning.select_table``); the
    (hash, path) of every distinct table; and the world session's table (``world``: its layout and most
    relays). Two different tables that match one group are refused."""
    from .. import tuning as tuning_mod

    tables: dict[str, tuning_mod.Table] = {}
    for path in paths:
        try:
            table = tuning_mod.Table.load(path)
        except tuning_mod.TuningError as error:
            raise PlanError(str(error)) from None
        tables.setdefault(table.hash, table)

    def matching(label: str, own: dict) -> str:
        found = [digest for digest, table in tables.items() if not table.mismatches(own)]
        if len(found) > 1:
            raise PlanError(f"{label}: several different tuning tables match it: "
                            + ", ".join(tables[digest].source for digest in found))
        return found[0] if found else ""

    assigned = []
    for group, layout in zip(groups, layouts):
        own = tuning_mod.facts(layout.identity(), layout.world, group.lanes, group.max_relays)
        digest = matching(f"group {group.index}", own)
        settings = tuple(sorted(tables[digest].settings.items())) if digest else ()
        assigned.append(dataclasses.replace(group, tuning_table=digest, tuning_settings=settings))
    world_table = ""
    if world is not None:
        layout, relays = world
        world_table = matching("the world session", tuning_mod.facts(layout.identity(), layout.world, 2, relays))
    return assigned, tuple((digest, table.source) for digest, table in tables.items()), world_table


def tune_families_text(plan: ConfigurationPlan) -> list[str]:
    """The candidate families the tune command measures per group and collective, chain and ring as the
    routes allow them (``routes.chain_order``, ``routes.ring_window``), and a note per family a group
    cannot run."""
    options = plan.options
    lines = []
    nccl = ", NCCL" if options.baseline == "nccl" else ""
    for group in plan.groups:
        layout = routes_mod.Layout.parse(group.layout)
        maps = [routes_mod.derive_routes(layout, group.lanes).route_map(rank) for rank in range(layout.world)]
        order = routes_mod.chain_order(layout, maps)
        problems = routes_mod.ring_window(layout, maps, order)[1] if order is not None else ["no chain"]
        links = (", chain" if order is not None else "") + (", ring" if order is not None and not problems else "")
        lines.append(f"  tune families, group {group.index}: all_reduce: one-shot, two-shot, pieces{links}{nccl}; "
                     f"all_gather: tiles{links}{nccl}; reduce_scatter: scatter ops{links}; all_to_all: scatter ops")
        if order is None:
            lines.append(f"    note: group {group.index} is not a chain of cable neighbors: no chain or ring candidates")
        elif problems:
            lines.append(f"    note: group {group.index} cannot run the ring ({'; '.join(problems)}): no ring candidates")
    return lines


def render_text(plan: ConfigurationPlan) -> str:
    lines = [
        f"configuration {plan.name} (run {plan.run_id}): {plan.world} ranks in {len(plan.groups)} group(s)",
        f"  image {plan.image}; GPU 0; host networking; control exchange tcp://{plan.leader_address}:"
        f"{plan.control_port} over {plan.lan_interface}",
        f"  remote run directory {plan.run_dir}; package sources {plan.remote_dir}/src/{plan.source_digest}",
        f"  capacity {plan.capacity} bytes (all-reduce), {plan.gather_capacity} bytes per shard (all-gather)",
    ]
    for group in plan.groups:
        lines.append(f"  group {group.index}: Sparks {list(group.positions)} (global ranks "
                     f"{list(group.global_ranks)}), {group.lanes} lanes per peer, up to {group.max_relays} "
                     f"relay(s), busiest relay queue {group.relay_lanes} lanes (load factor {group.relay_load:g})")
        lines.append(f"    fabric {group.layout}")
        if plan.tuning_tables:
            settings = ", ".join(f"{setting}={value}" for setting, value in group.tuning_settings)
            lines.append((f"    tuning table {group.tuning_table}"
                          + (f"; its sessions take {settings} where --session-env leaves them unset"
                             if settings else "")) if group.tuning_table
                         else "    no tuning table matches this group: the rules choose")
        for rank, text in enumerate(group.route_texts):
            lines.append(f"    rank {rank}: SIRCL_PEER_ROUTES={text}")
        for detail in group.lanes_detail:
            if detail["relays"]:
                window = detail.get("forward_window", 0)
                pacing = f", forward window {window} bytes" if window else ", no forward window"
                lines.append(f"    lane: rank {detail['rank']} -> rank {detail['peer']} lane {detail['lane']} "
                             f"{detail['local']} -> {detail['remote']} via Sparks {detail['relays']}{pacing}")
        for warning in group.warnings:
            lines.append(f"    note: {warning}")
        if group.ring:
            lines.append(f"    {group.ring}")
    if plan.world_layout:
        lines.append(f"  world session (tensor parallel over every rank, beside the group sessions): fabric "
                     f"{plan.world_layout}")
        for rank in plan.ranks:
            lines.append(f"    rank {rank.global_rank}: SIRCL_PEER_ROUTES={rank.world_peer_routes}")
    for rank in plan.ranks:
        lines.append(f"  rank {rank.global_rank}: {rank.host} ({rank.ssh}, LAN {rank.lan_address}) group "
                     f"{rank.group} rank {rank.group_rank}, container {rank.container}, "
                     f"docker command \"{rank.docker}\"")
    options = plan.options
    lines.append(f"  all-reduce BF16 sizes {list(options.allreduce_sizes)}")
    lines.append(f"  all-gather BF16 shard sizes {list(options.allgather_sizes)} and last-dimension shapes "
                 f"{[list(shape) for shape in options.allgather_shapes]}")
    if options.transport_only:
        lines.append(f"  transport only: CPU-staged one-shot ops over every size above, {options.correctness_iterations} "
                     f"checked and {options.eager_iterations} timed per size; no CUDA kernel runs")
    else:
        lines.append(f"  per size: {options.correctness_iterations} checked calls, {options.eager_iterations} timed "
                     f"eager calls, {options.graph_iterations} timed graph replays; spin limit {options.spin_limit}")
    lines.append(f"  CPU placement {options.cpu_policy}; {options.post_barrier_warmup} untimed calls after "
                 "every barrier; forward windows "
                 + ("at the session default" if options.forward_window is None
                    else "off" if options.forward_window == 0 else f"up to {options.forward_window} bytes"))
    lines.append(f"  flag-wait limits: {options.startup_wait_s:g} s during setup and warm-up, "
                 f"{options.serving_wait_s:g} s for the timed cases; large-message pieces "
                 + (f"{options.large_piece_bytes} bytes" if options.large_piece_bytes else "at the session default"))
    for size, target, stretch in options.targets:
        lines.append(f"  target: {size} bytes all-reduced in at most {target:g} ms (stretch {stretch:g} ms); "
                     f"NCCL on a four-Spark cycle: {NCCL_CYCLE_BUSBW_GBPS[0]:g}-{NCCL_CYCLE_BUSBW_GBPS[1]:g} GB/s "
                     "bus bandwidth")
    if plan.tuning_tables:
        lines.append("  tuning tables, written beside plan.json on every Spark (SIRCL_TUNING_TABLE): "
                     + ", ".join(f"{digest} ({source})" for digest, source in plan.tuning_tables)
                     + "; where a table decides, its choice replaces the rules; rows that force a schedule, "
                       "piece, stagger or grid cap run with the table suspended")
        if plan.world_layout:
            lines.append("  world session: " + (f"tuning table {plan.world_tuning_table}" if plan.world_tuning_table
                                                else "no tuning table matches: the rules choose"))
    if options.eager_profile or options.eager_path != "session":
        lines.append(f"  eager path: {options.eager_path}"
                     + ("; eager rows of all_reduce, all_gather, all_reduce_large and all_gather_large record the "
                        f"sessions' call stage times (SIRCL_CALL_PROFILE={EAGER_PROFILE_CALLS})"
                        if options.eager_profile else ""))
    if options.session_env:
        lines.append("  session variables on every rank: " + " ".join(f"{name}={value}"
                                                                     for name, value in options.session_env))
    if options.tune:
        lines.append(f"  tune: {', '.join(options.tune_collectives)} at {len(options.tune_sizes)} sizes of "
                     f"{options.tune_sizes[0]} to {options.tune_sizes[-1]} bytes per rank, modes "
                     f"{', '.join(options.tune_modes)}; candidates: one-shot and two-shot (grids "
                     f"{list(options.tune_grids)}) within the capacity, and from {options.tune_large_from} bytes "
                     f"pieces, tiles and scatter ops (the same grids), chain (pieces {list(options.tune_pieces)}) "
                     f"and ring (those pieces, staggers {list(options.tune_staggers)} on the partials and on the "
                     f"forwarded pieces), Swing where the session "
                     f"offers it{', NCCL' if options.baseline == 'nccl' else ''}; a candidate "
                     f"{options.tune_prune:g} times the fastest at two sizes in a row from {options.tune_prune_from} bytes "
                     f"on, its ratio not falling, is dropped; "
                     f"{options.correctness_iterations} checked, {options.eager_iterations} eager, "
                     f"{options.graph_iterations} graph and {options.large_iterations} large-message calls timed")
    if options.tune:
        lines.extend(tune_families_text(plan))
    if options.baseline == "nccl":
        effective = nccl_mod.environment(plan.lan_interface, plan.gid_index, options.nccl_library, options.nccl_env)
        lines.append("  NCCL baseline: NCCL's all-reduce and all-gather (dimension 0) of every case's size and mode; "
                     "NCCL's environment on every rank (the transport is read from its log, NCCL_DEBUG=INFO "
                     "INIT,NET, after a warm-up all-reduce; sockets on a Spark with RoCE stop the run):")
        lines.append("    " + " ".join(f"{name}={value}" for name, value in effective.items()))
    if options.large_blocks:
        lines.append(f"  grid caps of two-shot and large-message launches {list(options.large_blocks)}: every "
                     "large, DCP, Swing and two-shot latency case once per cap (the small cases and one-shot rows "
                     "once)")
    if options.large and options.large_schedules != ("auto",):
        chunks = list(options.chain_chunks) if options.chain_chunks else "the session default"
        lines.append(f"  large all-reduce schedules {list(options.large_schedules)}; chain chunks {chunks}")
    for size, target, stretch in options.chain_targets:
        lines.append(f"  chain target: {size} bytes all-reduced in at most {target:g} ms (stretch {stretch:g} ms)")
    for name, size, target, stretch in options.ring_targets:
        lines.append(f"  ring target: {name} of {size} bytes in at most {target:g} ms (stretch {stretch:g} ms)")
    for kind, label in (("gather", "all-gather"), ("scatter", "reduce-scatter"), ("reduce", "ring all-reduce")):
        own = getattr(options, f"{kind}_link_chunks")
        if own:
            lines.append(f"  {label} link pieces {list(own)}")
    if options.chain_gather_sizes:
        chunks = list(options.link_chunks) if options.link_chunks else "the session default"
        lines.append(f"  all-gather of [rows, 4096] BF16 along dimension 0: {list(options.chain_gather_sizes)} bytes "
                     f"per rank, schedules {list(options.gather_schedules)}, link pieces {chunks}")
    for size, target, stretch in options.gather_targets:
        lines.append(f"  all-gather target: {size} bytes per rank gathered in at most {target:g} ms (stretch "
                     f"{stretch:g} ms)")
    if options.large_only:
        lines.append("  large cases only: the small all-reduce and all-gather cases are skipped")
    if options.swing_sizes:
        lines.append(f"  Swing all-reduce of BF16 messages of {list(options.swing_sizes)} bytes beside the session's "
                     "default all-reduce of the same sizes, eager and graph; Swing outputs checked bit for bit "
                     "against the Swing reference (partial sums rounded after every step)")
    if options.latency_sizes:
        lines.append(f"  one-shot and two-shot all-reduce of BF16 messages of {list(options.latency_sizes)} bytes with "
                     f"the posting orders {list(options.post_orders)} back to back, eager and graph; the summary gives "
                     f"the crossover per order and the latency model (relay {options.latency_relay_us:g} us, posting "
                     f"{options.latency_post_us:g} us per lane where the session does not measure it, direct write "
                     f"{options.latency_write_us:g} us)")
    decode_rows, prefill_rows = options.dcp_decode_rows or (), options.dcp_prefill_rows or ()
    if decode_rows or prefill_rows:
        heads = options.dcp_heads
        lines.append(f"  DCP exchanges at decode rows {list(decode_rows)} (eager and graph) and prefill rows "
                     f"{list(prefill_rows)} (eager): all-to-all of [W, rows, {heads}, {DCP_A2A_VALUES}] BF16"
                     + (f", query all-gather of [rows, {heads}, {DCP_QUERY_VALUES}] BF16 and indexer all-gather of "
                        f"{DCP_INDEXER_ROW_BYTES} bytes per row" if options.dcp_gathers else "")
                     + (f", and at decode the world session's all-reduce of [rows, {options.tp_hidden}] BF16"
                        if options.world_session else "")
                     + "; every output checked bit for bit against a host reference")
    lines.append(f"  bounds: a Spark's NIC host interface sends {options.host_send_gbps:g} GB/s and receives "
                 f"{options.host_recv_gbps:g} GB/s, a cable direction carries {options.cable_gbps:g} GB/s "
                 "(sparkring_sircl.bounds)")
    if options.large_reduce_scatter_sizes and options.scatter_schedules != ("auto",):
        lines.append(f"  reduce-scatter schedules {list(options.scatter_schedules)}")
    if options.large_reduce_scatter_sizes:
        lines.append(f"  reduce-scatter of [rows, 4096] BF16: {list(options.large_reduce_scatter_sizes)} bytes per "
                     "rank; scatter ops in ops of the large-message piece, checked against the host reference and "
                     "the same rows of the session's all-reduce in ops of at most the capacity; chain "
                     "reduce-scatters checked against the chain reference, the elements that differ from the "
                     "rank-ordered sum counted")
    if options.large and (options.large_allreduce_sizes or options.large_allgather_sizes):
        if options.transport_only:
            sizes = [size for size in options.large_allreduce_sizes if size <= plan.capacity]
            lines.append(f"  large: CPU-staged one-shot ops of {sizes} bytes (capacity {plan.capacity})")
        else:
            lines.append(f"  large: all-reduce {list(options.large_allreduce_sizes)} bytes (two-shot up to the "
                         f"capacity {plan.capacity}, all_reduce_large above), all_gather_large shards "
                         f"{list(options.large_allgather_sizes)} bytes flat and as rows of 4096 bytes; "
                         f"{options.large_iterations} timed calls each")
    return "\n".join(lines)
