"""SIRCL collectives of one decode context parallel (DCP) group.

It runs the collectives of GLM-5.3's decode context parallelism on a SIRCL
ring session. SIRCL's communicator passes the group's fabric positions, route
map and relay-safe op size derived from the layout, and the setup votes run
through :mod:`.groupops`, so emulated ranks can drive it in CPU tests. Without
those arguments it applies the eight-Spark ring rules below.

``SirclDcpCollectives`` owns a second ``sparkring_sircl.oneshot.AllReduce`` runtime
(its own pinned region, queue pairs, flags and proxy thread) over the ranks of
one DCP group and runs the three collectives the group needs:

- all-gather along any dimension (query, LSE and indexer candidates): a gather
  along dimension ``k`` of a contiguous tensor is the last-dimension gather of
  its ``[prod(shape[:k]), prod(shape[k:])]`` view, which sircl's kernel
  writes in the concatenated layout directly; shards above the sub-gather
  limit are gathered in row ranges, each a complete one-shot op;
- reduce-scatter along dimension 0 (the head-major LSE combine): one scatter
  op for messages within the scatter op limit, else a loop over head
  sub-ranges with the runtime's strided chunks;
- all-to-all of a ``[world, rows, ...]`` buffer (the packed ``a2a`` combine):
  one scatter op, or a loop over row ranges with strided chunks.

Routes. Ring rank ``r`` is ring position ``r``, so a DCP group of consecutive
ranks occupies consecutive ring positions and its group rank ``j`` sits at
position ``positions[j]``. Each peer is reached through the local RDMA
functions that the ring's peer-to-device rule assigns by clockwise ring
distance (``route_functions``): distance 1-3 through port 0
(primary and secondary), 5-7 through port 1, 4 with one stripe each way. The
hardware relays of the ring carry the non-neighbour paths, so a DCP 4 group
needs no IP route between its ends. Peers are posted farthest first by
physical distance.

Relay load. A relayed frame crosses each intermediate Spark through one
hairpin queue of 512 KiB per egress function and direction
(measured on the eight-Spark ring: every relayed sircl
frame of one function and direction takes queue 0 of one hairpin instance,
and three eight-rank collectives started losing packets when that queue held
75-110 % of 512 KiB). ``relay_queue_bytes`` computes the busiest queue's bytes
for one op in which every rank sends the same number of bytes to every peer
(the one-shot all-gather and the scatter op share this pattern): three times
the per-peer bytes on the full ring, once on a DCP 4 path, nothing for a
neighbour pair. The default sub-gather and scatter op sizes keep that load at
75 % of the queue on the ring (128 KiB per peer, the measured drop-free sizes
of the all-gather and of 1 MiB scatter messages) and 50 % on a DCP 4 path
(256 KiB per peer, the measured drop-free DCP 4 scatter of 1 MiB messages).

Size thresholds. Above ``GLM_DCP_RDMA_MAX_BYTES_{GATHER,REDUCE_SCATTER,
ALL_TO_ALL}`` an eager collective is better served by NCCL's pipelined ring
(measured on the eight-Spark ring: at DCP 8 the sircl indexer gather of a
512 KiB shard, three sub-gathers, took 355 us against 321 us on PyNccl; at
1 MiB and 2 MiB reduce-scatter messages sircl took 88 and 179 us against
211 and 244 us, so the lines cross near 3 MiB). The defaults are those
measured or interpolated crossovers; 0 means no threshold. Whether a message
above a threshold leaves sircl is decided by the communicator
(``communicator.py``), which knows whether the group has a PyNccl
communicator and whether a CUDA graph is being captured: captured (decode)
collectives always stay on sircl, so the thresholds only move eager
(prefill and large eager decode) calls.

Every setting is voted over the group's CPU process group before the runtime is
built, and the runtime's own setup exchange refuses differing protocol
configurations, so a group either serves every rank from sircl or fails to
start on every rank. Dispatch decisions depend only on dtype, shape,
contiguity and the voted limits, never on pointer values, so all ranks of a
group route a collective the same way.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch

from . import groupops, sessionapi
from .settings import TP_SESSION_VARIABLES

logger = logging.getLogger("sircl.vllm.dcp_collectives")

REQUIRED_SIRCL_API_VERSION = 1
PACK_BYTES = 16
RING = 8
PREPARED_DTYPES = (torch.bfloat16,)
GATHER_DTYPES_DECLINED = (torch.bool,)

ENV_CAPACITY = "GLM_DCP_RDMA_CAPACITY_BYTES"
ENV_GATHER_MAX = "GLM_DCP_RDMA_GATHER_MAX_BYTES"
ENV_GATHER_CHUNK = "GLM_DCP_RDMA_GATHER_CHUNK_BYTES"
ENV_SCATTER_OP = "GLM_DCP_RDMA_SCATTER_OP_BYTES"
ENV_MAX_GATHER = "GLM_DCP_RDMA_MAX_BYTES_GATHER"
ENV_MAX_REDUCE_SCATTER = "GLM_DCP_RDMA_MAX_BYTES_REDUCE_SCATTER"
ENV_MAX_ALL_TO_ALL = "GLM_DCP_RDMA_MAX_BYTES_ALL_TO_ALL"
ENV_PROXY_CPU = "GLM_DCP_RDMA_PROXY_CPU"
ENV_SPIN_LIMIT = "GLM_DCP_RDMA_SPIN_LIMIT"
DEFAULT_CAPACITY_BYTES = 2 * 1024 * 1024
DEFAULT_GATHER_MAX_BYTES = 2 * 1024 * 1024

# One hairpin queue of a ring relay: queue 0 of the egress function's hairpin
# instance, at the device's maximum size.
HAIRPIN_QUEUE_BYTES = 512 * 1024
# Queue fill above which the settings are logged as outside the measured
# drop-free range (packet loss measured on the ring began at 75 %).
RELAY_FILL_WARN = 0.75

# Per-peer bytes of one sircl op (a sub-gather's shard, or a scatter op's
# chunk) that keep the busiest hairpin queue within the measured drop-free
# range: 128 KiB on the full ring (three relayed flows per queue: 75 % of the
# queue; eight-rank all-gathers stayed drop-free to 155,648-byte shards and
# 1 MiB scatter messages ran without RDMA errors, 2 MiB messages with 3,441
# and 2,478 of them, at DCP 8 on the eight-Spark ring), 256 KiB on a DCP 4 path
# (one relayed flow per queue: 50 %; 1 MiB DCP 4 scatter messages ran clean).
RELAYED_PER_PEER_BYTES = {4: 262144, 8: 131072}

# Eager-call thresholds above which the communicator prefers PyNccl's ring
# where the group has one (bytes of this rank's shard for gathers, of the
# whole message for the scatters; 0: none). Measured and interpolated on the
# eight-Spark ring (graph p50, slowest rank):
# DCP 8 gathers: sircl 177.5 us at 256 KiB against NCCL 280.3, 355.4 at
#   512 KiB against 320.8; the lines cross at about 448 KiB.
# DCP 8 reduce-scatter: 88.3 us at 1 MiB against 210.7, 178.9 at 2 MiB against
#   243.8; sircl adds 88 us per MiB (1 MiB ops) and NCCL 33 us per MiB, so
#   they cross near 3 MiB; the all-to-all message of the same rows is 0.4 %
#   larger and shares the threshold.
# DCP 2 (a neighbour pair, no relay): sircl stays below NCCL at every
#   measured size (512 KiB: gathers 45 against 100 us, reduce-scatter 28
#   against 99); NCCL's two-rank ring reaches about 168 Gb/s against about
#   120-150 Gb/s for sircl's serialized stage, wire and copy phases, so
#   the crossover sits above the 2 MiB sub-op size for gathers (modelled
#   near 3 MiB) and near 8-12 MiB for the scatters.
# DCP 4 has no PyNccl communicator (communicator.py), so its thresholds are
#   unused unless GLM_DCP_RDMA_PYNCCL=keep is forced; none by default.
THRESHOLD_DEFAULTS = {
    2: {"gather": 2 * 1024 * 1024, "reduce_scatter": 8 * 1024 * 1024, "all_to_all": 8 * 1024 * 1024},
    4: {"gather": 0, "reduce_scatter": 0, "all_to_all": 0},
    8: {"gather": 458752, "reduce_scatter": 3 * 1024 * 1024, "all_to_all": 3 * 1024 * 1024},
}
THRESHOLD_ENV = {"gather": ENV_MAX_GATHER, "reduce_scatter": ENV_MAX_REDUCE_SCATTER, "all_to_all": ENV_MAX_ALL_TO_ALL}

# Ring fabric functions: key -> (RDMA device, port). Port 0 of ring position r
# is cabled to port 1 of position r + 1.
FUNCTIONS = {
    "P0": ("rocep1s0f0", 0),
    "S0": ("roceP2p1s0f0", 0),
    "P1": ("rocep1s0f1", 1),
    "S1": ("roceP2p1s0f1", 1),
}


def route_functions(src: int, dst: int) -> tuple[str, str]:
    """Local function keys, in stripe order, that ring position ``src`` uses to reach ``dst``.

    The ring's peer-to-device rule: clockwise distance 1-3
    leaves through port 0, 5-7 through port 1, and the opposite Spark (4) uses
    one stripe each way, the end at positions 0-3 listing port 0 primary and
    port 1 secondary and the other end the mirror.
    """
    d = (dst - src) % RING
    if d == 0:
        raise ValueError("a rank has no route to itself")
    if d <= 3:
        return ("P0", "S0")
    if d >= 5:
        return ("P1", "S1")
    return ("P0", "S1") if src % RING < 4 else ("P1", "S0")


def ring_distance(a: int, b: int) -> int:
    """Number of cables between ring positions ``a`` and ``b`` on the shortest path."""
    d = (b - a) % RING
    return min(d, RING - d)


def group_positions(global_ranks: Sequence[int]) -> tuple[int, ...]:
    """Ring positions of a DCP group from its global ranks (ring rank r is ring position r)."""
    positions = tuple(int(r) % RING for r in global_ranks)
    if len(set(positions)) != len(positions) or not 2 <= len(positions) <= RING:
        raise ValueError(f"a DCP group must occupy 2-8 distinct ring positions, got {list(global_ranks)}")
    return positions


def peer_routes(rank: int, positions: Sequence[int]) -> dict[int, tuple[str, ...]]:
    """``{group peer rank: (local RDMA devices in stripe order)}`` for group rank ``rank``."""
    return {
        peer: tuple(FUNCTIONS[f][0] for f in route_functions(positions[rank], positions[peer]))
        for peer in range(len(positions)) if peer != rank
    }


def posting_order(rank: int, positions: Sequence[int]) -> tuple[int, ...]:
    """Peers of group rank ``rank`` by decreasing physical distance, clockwise before counter-clockwise.

    The proxy's ``ring-farthest`` order folds the *group* size, which matches
    the physical ring only for the full ring (DCP 8); an explicit list keeps
    the longest relay paths first for the paths of DCP 4 as well.
    """
    src = positions[rank]

    def key(peer: int) -> tuple[int, int]:
        dst = positions[peer]
        clockwise = (dst - src) % RING
        return (-ring_distance(src, dst), 0 if clockwise <= RING // 2 else 1)

    return tuple(sorted((p for p in range(len(positions)) if p != rank), key=key))


def stripe_relays(src: int, dst: int, function_key: str) -> tuple[int, ...]:
    """Ring positions a stripe from ``src`` to ``dst`` on local function ``function_key`` is relayed through.

    Port 0 leaves clockwise, port 1 counter-clockwise; the relays are the
    positions strictly between the ends along that direction.
    """
    port = FUNCTIONS[function_key][1]
    clockwise = (dst - src) % RING
    hops = clockwise if port == 0 else RING - clockwise
    step = 1 if port == 0 else -1
    return tuple((src + step * k) % RING for k in range(1, hops))


def relay_queue_loads(positions: Sequence[int], per_peer_bytes: int) -> dict[tuple[int, int, str], float]:
    """Bytes through each relay hairpin queue for one op in which every rank sends ``per_peer_bytes`` to every peer.

    Keys are ``(relay position, egress port, function class)``: the hairpin
    instance of one egress function and direction.
    The two stripes of a peer carry half the bytes each on their own function.
    """
    loads: dict[tuple[int, int, str], float] = {}
    for src in positions:
        for dst in positions:
            if src == dst:
                continue
            for key in route_functions(src, dst):
                port = FUNCTIONS[key][1]
                for relay in stripe_relays(src, dst, key):
                    loads[(relay, port, key[0])] = loads.get((relay, port, key[0]), 0.0) + per_peer_bytes / 2
    return loads


def relay_queue_bytes(positions: Sequence[int], per_peer_bytes: int) -> int:
    """The busiest relay hairpin queue's bytes for one op of ``per_peer_bytes`` to every peer (0 without relays)."""
    loads = relay_queue_loads(positions, per_peer_bytes)
    return int(math.ceil(max(loads.values()))) if loads else 0


def relay_fill(positions: Sequence[int], per_peer_bytes: int) -> float:
    """``relay_queue_bytes`` as a fraction of one hairpin queue."""
    return relay_queue_bytes(positions, per_peer_bytes) / HAIRPIN_QUEUE_BYTES


def env_int(name: str, default: int) -> int:
    """A non-negative integer from the environment, else ``default``."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {raw}")
    return value


def gather_chunk_default(world_size: int, capacity: int, per_peer_bytes: int | None = None) -> int:
    """Sub-gather shard limit when ``GLM_DCP_RDMA_GATHER_CHUNK_BYTES`` is 0.

    ``per_peer_bytes`` is the relay-safe size SIRCL's communicator derives from
    the layout; without it the eight-Spark ring values apply.
    """
    if per_peer_bytes is not None:
        return min(capacity, per_peer_bytes)
    return min(capacity, RELAYED_PER_PEER_BYTES.get(world_size, capacity))


def scatter_op_default(world_size: int, capacity: int, per_peer_bytes: int | None = None) -> int:
    """Scatter op message limit when ``GLM_DCP_RDMA_SCATTER_OP_BYTES`` is 0: ``world_size`` relay-safe chunks."""
    per_peer = RELAYED_PER_PEER_BYTES.get(world_size) if per_peer_bytes is None else per_peer_bytes
    return capacity if per_peer is None else min(capacity, per_peer * world_size)


def threshold_defaults(world_size: int) -> dict[str, int]:
    """The eager-call thresholds (bytes; 0 none) per collective kind for a group of ``world_size`` ranks."""
    return dict(THRESHOLD_DEFAULTS.get(world_size, {"gather": 0, "reduce_scatter": 0, "all_to_all": 0}))


def within_threshold(nbytes: int, limit: int) -> bool:
    """True when a message of ``nbytes`` is at or below ``limit`` (0: always)."""
    return limit == 0 or nbytes <= limit


def local_settings(world_size: int, per_peer_bytes: int | None = None) -> tuple[str | None, dict | None]:
    """This rank's reason for not taking part (or None) and its voted settings."""
    try:
        capacity = env_int(ENV_CAPACITY, DEFAULT_CAPACITY_BYTES)
        gather_max = env_int(ENV_GATHER_MAX, DEFAULT_GATHER_MAX_BYTES)
        gather_chunk = env_int(ENV_GATHER_CHUNK, 0)
        scatter_op = env_int(ENV_SCATTER_OP, 0)
        thresholds = {kind: env_int(name, -1) for kind, name in THRESHOLD_ENV.items()}
    except ValueError as exc:
        return str(exc), None
    if capacity < PACK_BYTES or capacity % PACK_BYTES:
        return f"{ENV_CAPACITY}={capacity} must be a positive multiple of {PACK_BYTES}", None
    if gather_max < PACK_BYTES or gather_max % PACK_BYTES:
        return f"{ENV_GATHER_MAX}={gather_max} must be a positive multiple of {PACK_BYTES}", None
    if gather_chunk % PACK_BYTES or gather_chunk > gather_max:
        return f"{ENV_GATHER_CHUNK}={gather_chunk} must be a multiple of {PACK_BYTES} up to {ENV_GATHER_MAX}", None
    if not gather_chunk:
        gather_chunk = gather_chunk_default(world_size, gather_max, per_peer_bytes)
    if scatter_op % (world_size * PACK_BYTES) or scatter_op > capacity:
        return (f"{ENV_SCATTER_OP}={scatter_op} must be a multiple of {world_size * PACK_BYTES} "
                f"({world_size} chunks of whole packs) up to {ENV_CAPACITY}"), None
    if not scatter_op:
        scatter_op = scatter_op_default(world_size, capacity, per_peer_bytes)
    defaults = threshold_defaults(world_size)
    for kind, value in thresholds.items():
        if value < 0:
            thresholds[kind] = defaults[kind]
        elif value % PACK_BYTES:
            return f"{THRESHOLD_ENV[kind]}={value} must be a multiple of {PACK_BYTES} (0: no threshold)", None
    topology = os.environ.get("SIRCL_TOPOLOGY") or "direct"
    if topology != "direct":
        return f"SIRCL_TOPOLOGY={topology}; ring sessions have the single topology direct", None
    try:
        oneshot = sessionapi.load()
    except Exception as exc:  # noqa: BLE001 - missing package or a broken native build
        return f"the ring session package is not importable: {type(exc).__name__}: {exc}", None
    api = getattr(oneshot, "API_VERSION", None)
    if api != REQUIRED_SIRCL_API_VERSION:
        return f"ring session API version {api}, plugin needs {REQUIRED_SIRCL_API_VERSION}", None
    if not hasattr(oneshot.AllReduce, "reduce_scatter") or not hasattr(oneshot.AllReduce, "all_to_all"):
        return "the ring session has no scatter collectives", None
    if not oneshot.is_supported():
        return "needs an integrated GPU with an active RDMA device", None
    return None, {
        "capacity": capacity,
        "gather_max": gather_max,
        "gather_chunk": gather_chunk,
        "scatter_op": scatter_op,
        "max_gather": thresholds["gather"],
        "max_reduce_scatter": thresholds["reduce_scatter"],
        "max_all_to_all": thresholds["all_to_all"],
        "spin_limit": os.environ.get(ENV_SPIN_LIMIT, "") or os.environ.get("SIRCL_SPIN_LIMIT", ""),
        "proxy_cpu": os.environ.get(ENV_PROXY_CPU, ""),
        "api_version": REQUIRED_SIRCL_API_VERSION,
    }


def vote(group, local, *, compare: bool) -> str | None:
    """Gather every rank's ``local`` over ``group``; the text of any failure or disagreement.

    ``local`` is ``(reason, settings)`` with ``compare`` (settings must then
    match rank 0's, rank-local keys excepted) or an error text (None for
    success) without it.
    """
    votes = groupops.all_gather_object(group, local)
    reasons = [v[0] if compare else v for v in votes]
    failures = [f"rank {i}: {reason}" for i, reason in enumerate(reasons) if reason]
    if failures:
        return "; ".join(failures)
    if compare:
        reference = {k: v for k, v in votes[0][1].items() if k != "proxy_cpu"}
        differing = [f"rank {i}: {v[1]}" for i, v in enumerate(votes)
                     if {k: x for k, x in v[1].items() if k != "proxy_cpu"} != reference]
        if differing:
            return f"settings differ from rank 0's {reference}: " + "; ".join(differing)
    return None


@contextmanager
def environment(overrides: dict[str, str | None]) -> Iterator[None]:
    """Set (or remove, for None) environment variables for the duration of the block."""
    saved = {name: os.environ.get(name) for name in overrides}
    try:
        for name, value in overrides.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class Declined(Exception):
    """A collective that sircl does not take; the message is the logged reason."""


class SirclDcpCollectives:
    """One DCP group's sircl runtime and its all-gather, reduce-scatter and all-to-all."""

    def __init__(self, *, cpu_group, device: torch.device, global_ranks: Sequence[int],
                 positions: Sequence[int] | None = None,
                 routes: dict[int, tuple[str, ...]] | None = None,
                 per_peer_bytes: int | None = None, layout: str | None = None) -> None:
        self.group = cpu_group
        self.device = device
        self.rank = groupops.rank(cpu_group)
        self.world_size = groupops.size(cpu_group)
        self.positions = (group_positions(global_ranks) if positions is None
                          else tuple(int(p) for p in positions))
        if len(self.positions) != self.world_size:
            raise ValueError("global ranks must match the process group")
        self.routes = peer_routes(self.rank, self.positions) if routes is None else dict(routes)
        self.order = posting_order(self.rank, self.positions)
        self._runtime = None
        self._announced: set[str] = set()
        reason, settings = local_settings(self.world_size, per_peer_bytes)
        verdict = vote(cpu_group, (reason, settings), compare=True)
        if verdict is not None:
            raise RuntimeError(f"sircl DCP collectives unavailable: {verdict}")
        assert settings is not None
        self.capacity = int(settings["capacity"])
        self.gather_max = int(settings["gather_max"])
        self.gather_chunk = int(settings["gather_chunk"])
        self.scatter_op_bytes = int(settings["scatter_op"])
        self.thresholds = {
            "all-gather": int(settings["max_gather"]),
            "reduce-scatter": int(settings["max_reduce_scatter"]),
            "all-to-all": int(settings["max_all_to_all"]),
        }
        # The eight-Spark relay-load model applies only to its own ring rules.
        ring_rules = positions is None and routes is None
        self.relay_fill = {
            "gather": relay_fill(self.positions, self.gather_chunk) if ring_rules else 0.0,
            "scatter": (relay_fill(self.positions, self.scatter_op_bytes // self.world_size)
                        if ring_rules else 0.0),
        }
        if self.rank == 0:
            for name, fill in self.relay_fill.items():
                if fill > RELAY_FILL_WARN:
                    logger.warning(
                        "SIRCL DCP collectives: the %s op size puts %.0f%% of a %d-byte relay hairpin queue in flight "
                        "per op, above the %.0f%% at which eight-rank collectives started dropping packets "
                        "on the eight-Spark ring; watch rdma_errors and rx_out_of_buffer",
                        name, fill * 100, HAIRPIN_QUEUE_BYTES, RELAY_FILL_WARN * 100)

        oneshot = sessionapi.load()

        # The runtime reads its posting order, proxy placement and dispatch
        # limits from SIRCL_* at construction, which the tensor-parallel
        # runtime of the same process set for the whole ring; the DCP runtime
        # needs its own values for the duration of the constructor. The
        # tensor-parallel session's schedule and link settings are removed
        # (TP_SESSION_VARIABLES): this session keeps their defaults.
        overrides: dict[str, str | None] = {
            **dict.fromkeys(TP_SESSION_VARIABLES),
            "SIRCL_POST_ORDER": ",".join(str(p) for p in self.order),
            "SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES": str(self.capacity),
            "SIRCL_SWING_ABOVE_BYTES": "0",
            "SIRCL_PROGRESS_CPU": settings["proxy_cpu"] or None,
        }
        if settings["spin_limit"]:
            overrides["SIRCL_SPIN_LIMIT"] = settings["spin_limit"]
        try:
            with environment(overrides):
                runtime = oneshot.AllReduce(
                    exchange_group=cpu_group, device=device, max_size=self.capacity,
                    max_gather_bytes=self.gather_max, peer_routes=self.routes,
                    algorithm="oneshot", **({} if layout is None else {"layout": layout}),
                )
        except Exception as exc:  # noqa: BLE001 - sircl coordinated the ranks
            raise RuntimeError(f"sircl DCP setup failed on group rank {self.rank}") from exc
        error = None
        try:
            if not runtime.scatter_available:
                raise RuntimeError("the runtime's scatter collectives are unavailable")
            runtime.prepare(PREPARED_DTYPES, padded_gather=True, scatter=True, **sessionapi.link_keywords(runtime))
        except Exception as exc:  # noqa: BLE001 - reported through the vote below
            error = f"{type(exc).__name__}: {exc}"
        verdict = vote(cpu_group, error, compare=False)
        if verdict is not None:
            runtime.close()
            raise RuntimeError(f"sircl DCP prepare failed: {verdict}")
        self._runtime = runtime

    # -- diagnostics ---------------------------------------------------------------

    def describe(self) -> str:
        rt = self._runtime
        limits = ", ".join(
            f"{kind} above {limit} bytes" if limit else f"{kind} never"
            for kind, limit in self.thresholds.items())
        return (
            f"{self.world_size}-rank DCP group at ring positions {list(self.positions)}: hcas="
            f"{','.join(rt.hca_names)} stripes={rt.lane_count} posting order {list(self.order)}, "
            f"gather shard <= {self.gather_max} bytes in sub-gathers of <= {self.gather_chunk}, "
            f"reduce-scatter and all-to-all messages in scatter ops of <= {self.scatter_op_bytes} bytes "
            f"(capacity {self.capacity}; relay hairpin queue fill per op: gather {self.relay_fill['gather']:.0%}, "
            f"scatter {self.relay_fill['scatter']:.0%}), eager calls leave sircl for NCCL where it exists: "
            f"{limits}, spin limit {rt.spin_limit}"
        )

    def _announce(self, kind: str, detail: str) -> None:
        if kind in self._announced:
            return
        self._announced.add(kind)
        (logger.info if self.rank == 0 else logger.debug)(
            "sircl DCP %s is live: first routed call is %s.", kind, detail)

    def check_health(self) -> None:
        if self._runtime is not None:
            self._runtime.check_health()

    @contextmanager
    def capture(self, stream: torch.cuda.Stream | None = None):
        if self._runtime is None:
            yield
            return
        with self._runtime.capture(stream=stream):
            yield

    def close(self) -> None:
        runtime, self._runtime = self._runtime, None
        if runtime is not None:
            runtime.close()

    def stats(self) -> dict:
        return {} if self._runtime is None else self._runtime.stats()

    # -- size policy -----------------------------------------------------------------

    def threshold(self, kind: str) -> int:
        """The eager-call threshold (bytes; 0 none) of ``kind``: all-gather, reduce-scatter or all-to-all."""
        return self.thresholds[kind]

    def prefers_sircl(self, kind: str, nbytes: int) -> bool:
        """True when ``nbytes`` of ``kind`` is within the group's sircl threshold (the communicator adds the capture and fallback conditions)."""
        return within_threshold(nbytes, self.thresholds[kind])

    @staticmethod
    def gather_bytes(inp: torch.Tensor) -> int:
        """This rank's shard bytes: the quantity the gather threshold is compared with."""
        return inp.numel() * inp.element_size()

    @staticmethod
    def message_bytes(inp: torch.Tensor) -> int:
        """The whole message bytes of a reduce-scatter input or an all-to-all send buffer."""
        return inp.numel() * inp.element_size()

    # -- eligibility -----------------------------------------------------------------

    def _check_tensor(self, inp: torch.Tensor, kind: str) -> None:
        if self._runtime is None:
            raise RuntimeError("the sircl DCP runtime is closed")
        if inp.device != self.device:
            raise Declined(f"{kind}: tensor on {inp.device}, runtime on {self.device}")
        if not inp.is_contiguous():
            raise Declined(f"{kind}: non-contiguous input")
        if inp.dim() == 0 or inp.numel() == 0:
            raise Declined(f"{kind}: empty or scalar input")
        if inp.is_complex() or inp.is_sparse or inp.dtype in GATHER_DTYPES_DECLINED:
            raise Declined(f"{kind}: dtype {inp.dtype}")

    # -- all-gather --------------------------------------------------------------------

    @staticmethod
    def _normalize_dim(inp: torch.Tensor, dim: int) -> int:
        if dim < 0:
            dim += inp.dim()
        if not 0 <= dim < inp.dim():
            raise Declined(f"all-gather: dim {dim} out of range for {tuple(inp.shape)}")
        return dim

    def all_gather(self, inp: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Concatenate every rank's ``inp`` along ``dim``; raises ``Declined`` when sircl does not take it."""
        self._check_tensor(inp, "all-gather")
        dim = self._normalize_dim(inp, dim)
        rt = self._runtime
        W = self.world_size
        shape = list(inp.shape)
        out_shape = list(shape)
        out_shape[dim] *= W
        outer = math.prod(shape[:dim])
        inner = math.prod(shape[dim:])
        elem = inp.element_size()
        nbytes = inp.numel() * elem
        row_bytes = inner * elem
        chunk = min(self.gather_chunk, rt.max_gather_bytes)
        out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
        if outer == 1:
            # Concatenation along the first non-trivial dimension: shard s is
            # the s-th block of the flat output.
            flat_in = inp.reshape(-1)
            flat_out = out.view(W, -1)
            if nbytes <= chunk:
                rt.all_gather(flat_in, dim=0, out=flat_out.view(-1))
            else:
                self._gather_flat_in_pieces(flat_in, flat_out, chunk)
        else:
            if row_bytes % PACK_BYTES or nbytes % PACK_BYTES:
                # The runtime's padded path handles rows that are not whole
                # packs with its own staging copy; it needs the whole shard.
                if nbytes > rt.max_gather_bytes:
                    raise Declined(f"all-gather: {nbytes}-byte shard with {row_bytes}-byte rows above {rt.max_gather_bytes}")
                rt.all_gather(inp.reshape(outer, inner), dim=-1, out=out.view(outer, W * inner))
            else:
                view = inp.reshape(outer, inner)
                out_rows = out.view(outer, W * inner)
                rows_per_op = max(1, min(outer, chunk // row_bytes))
                for r0 in range(0, outer, rows_per_op):
                    r1 = min(outer, r0 + rows_per_op)
                    rt.all_gather(view[r0:r1], dim=-1, out=out_rows[r0:r1])
        self._announce("all-gather", f"{tuple(inp.shape)} {str(inp.dtype).replace('torch.', '')} along dim {dim}")
        return out

    def _gather_flat_in_pieces(self, flat_in: torch.Tensor, flat_out: torch.Tensor, chunk: int) -> None:
        """Gather a flat shard above the sub-gather limit in element ranges through a scratch tile."""
        rt = self._runtime
        W = self.world_size
        elem = flat_in.element_size()
        n = flat_in.numel()
        per_op = max(PACK_BYTES // elem, (chunk // elem) // (PACK_BYTES // elem) * (PACK_BYTES // elem))
        scratch = torch.empty(W * per_op, dtype=flat_in.dtype, device=flat_in.device)
        for e0 in range(0, n, per_op):
            e1 = min(n, e0 + per_op)
            count = e1 - e0
            if (count * elem) % PACK_BYTES == 0:
                tile = scratch[: W * count]
                rt.all_gather(flat_in[e0:e1], dim=0, out=tile)
                flat_out[:, e0:e1].copy_(tile.view(W, count))
            else:
                # the padded path returns a new tensor laid out [W, count]
                flat_out[:, e0:e1].copy_(rt.all_gather(flat_in[e0:e1], dim=0).view(W, count))

    # -- scatter chunking ------------------------------------------------------------

    def _scatter_op_limit(self) -> int:
        """Bytes of the largest scatter op: the relay-safe setting, never above the runtime's capacity."""
        return min(self.scatter_op_bytes, self._runtime.max_size)

    def _scatter_pieces(self, rows: int, row_bytes: int) -> Iterator[tuple[int, int]]:
        """``(byte offset within a peer's block, chunk bytes)`` of the ops that scatter ``rows`` rows per peer.

        Each op's message is ``world_size`` chunks and must fit the scatter op
        limit, so a chunk holds as many whole rows as fit, or, when one row
        does not fit, a 16-byte-aligned byte range of one row.
        """
        per_op = (self._scatter_op_limit() // self.world_size) // PACK_BYTES * PACK_BYTES
        if per_op < PACK_BYTES:
            raise Declined(f"scatter: op limit {self._scatter_op_limit()} below one pack per rank")
        if row_bytes <= per_op:
            rows_per_op = per_op // row_bytes
            for r0 in range(0, rows, rows_per_op):
                count = min(rows_per_op, rows - r0)
                yield r0 * row_bytes, count * row_bytes
        else:
            for r in range(rows):
                for c0 in range(0, row_bytes, per_op):
                    yield r * row_bytes + c0, min(per_op, row_bytes - c0)

    # -- reduce-scatter ------------------------------------------------------------

    def reduce_scatter(self, inp: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Sum ``inp`` across the group and keep this rank's ``world_size``-th along ``dim``."""
        self._check_tensor(inp, "reduce-scatter")
        if inp.dtype not in PREPARED_DTYPES:
            raise Declined(f"reduce-scatter: dtype {inp.dtype} not prepared")
        dim = self._normalize_dim(inp, dim)
        rt = self._runtime
        W = self.world_size
        if dim != 0:
            # The scatter dimension first, with one copy (vLLM's own reduce_scatter
            # does the same for its NCCL path).
            inp = inp.movedim(dim, 0).contiguous()
        if inp.shape[0] % W:
            raise Declined(f"reduce-scatter: dim-0 extent {inp.shape[0]} not divisible by {W}")
        heads = inp.shape[0] // W
        row_bytes = (inp.numel() // inp.shape[0]) * inp.element_size()
        if row_bytes % PACK_BYTES or inp.data_ptr() % PACK_BYTES:
            raise Declined(f"reduce-scatter: {row_bytes}-byte rows or an unaligned pointer")
        out_shape = list(inp.shape)
        out_shape[0] = heads
        out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
        nbytes = inp.numel() * inp.element_size()
        if nbytes <= self._scatter_op_limit():
            rt.reduce_scatter(inp, out=out)
        else:
            # Every op scatters one piece of each peer's block of ``heads`` rows:
            # whole rows while a row fits, else byte ranges of one row.
            elem = inp.element_size()
            flat_in = inp.view(-1)
            flat_out = out.view(-1)
            for offset, chunk in self._scatter_pieces(heads, row_bytes):
                rt.reduce_scatter(
                    flat_in[offset // elem:], out=flat_out[offset // elem:(offset + chunk) // elem],
                    chunk_bytes=chunk, src_stride_bytes=heads * row_bytes,
                )
        result = out if dim == 0 else out.movedim(0, dim).contiguous()
        self._announce("reduce-scatter", f"{tuple(inp.shape)} {str(inp.dtype).replace('torch.', '')} along dim {dim}")
        return result

    # -- all-to-all ----------------------------------------------------------------

    def all_to_all_rows(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        """Exchange ``send[p]`` to rank ``p`` into ``recv[s]`` from rank ``s`` for ``[world, rows, ...]`` buffers."""
        self._check_tensor(send, "all-to-all")
        rt = self._runtime
        W = self.world_size
        if send.dim() < 2 or send.shape[0] != W or recv.shape != send.shape or recv.dtype != send.dtype:
            raise Declined(f"all-to-all: send {tuple(send.shape)} and recv {tuple(recv.shape)} must be [world, rows, ...]")
        if not recv.is_contiguous() or send.data_ptr() % PACK_BYTES or recv.data_ptr() % PACK_BYTES:
            raise Declined("all-to-all: non-contiguous receive buffer or unaligned pointers")
        rows = send.shape[1]
        row_bytes = (send.numel() // (W * rows)) * send.element_size()
        if row_bytes % PACK_BYTES:
            raise Declined(f"all-to-all: {row_bytes}-byte rows")
        nbytes = send.numel() * send.element_size()
        if nbytes <= self._scatter_op_limit():
            rt.all_to_all(send, recv)
        else:
            elem = send.element_size()
            flat_send, flat_recv = send.view(-1), recv.view(-1)
            for offset, chunk in self._scatter_pieces(rows, row_bytes):
                rt.all_to_all(
                    flat_send[offset // elem:], flat_recv[offset // elem:],
                    chunk_bytes=chunk, src_stride_bytes=rows * row_bytes,
                    dst_stride_bytes=rows * row_bytes,
                )
        self._announce("all-to-all", f"{tuple(send.shape)} {str(send.dtype).replace('torch.', '')}")

    def all_to_all_single(self, output: torch.Tensor, input_: torch.Tensor) -> None:
        """``dist.all_to_all_single`` with equal splits over flat buffers."""
        self._check_tensor(input_, "all-to-all")
        rt = self._runtime
        W = self.world_size
        nbytes = input_.numel() * input_.element_size()
        if output.numel() != input_.numel() or output.dtype != input_.dtype or not output.is_contiguous():
            raise Declined("all-to-all: output must match the input")
        if nbytes % (W * PACK_BYTES) or input_.data_ptr() % PACK_BYTES or output.data_ptr() % PACK_BYTES:
            raise Declined(f"all-to-all: {nbytes} bytes are not {W} chunks of whole packs, or unaligned pointers")
        if nbytes > self._scatter_op_limit():
            # rows of one 16-byte pack each: the row-range loop of all_to_all_rows
            per_pack = PACK_BYTES // input_.element_size()
            self.all_to_all_rows(input_.view(W, -1, per_pack), output.view(W, -1, per_pack))
            return
        rt.all_to_all(input_, output)
        self._announce("all-to-all", f"{tuple(input_.shape)} {str(input_.dtype).replace('torch.', '')}")


def suppress_declined(fn, *args, **kwargs):
    """Run ``fn``; return ``(result, None)`` or ``(None, reason)`` when it declined."""
    try:
        return fn(*args, **kwargs), None
    except Declined as exc:
        return None, str(exc)


__all__ = [
    "Declined",
    "FUNCTIONS",
    "HAIRPIN_QUEUE_BYTES",
    "RELAYED_PER_PEER_BYTES",
    "RELAY_FILL_WARN",
    "RING",
    "SirclDcpCollectives",
    "THRESHOLD_DEFAULTS",
    "THRESHOLD_ENV",
    "environment",
    "gather_chunk_default",
    "group_positions",
    "local_settings",
    "peer_routes",
    "posting_order",
    "relay_fill",
    "relay_queue_bytes",
    "relay_queue_loads",
    "ring_distance",
    "route_functions",
    "scatter_op_default",
    "stripe_relays",
    "suppress_declined",
    "threshold_defaults",
    "vote",
    "within_threshold",
]
