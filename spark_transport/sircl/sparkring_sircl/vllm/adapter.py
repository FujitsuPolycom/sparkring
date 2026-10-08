"""SIRCL for one vLLM process group: placement, NCCL policy, session and dispatch.

:class:`GroupPlacement` is computed before vLLM builds a group's device
communicator (it decides whether PyNccl may be built); :class:`GroupAdapter`
is built right after, with the group's CPU process group, and then serves
every collective of the group. Neither imports vLLM: the vLLM-facing
communicator (:mod:`.communicator`) passes the stock NCCL paths in as
:class:`NcclPaths`, and the CPU tests pass emulated groups and recorders.

Sessions. A group of a kind named in ``SIRCL_GROUPS`` (default ``tp,dcp``)
gets a ring session:

- the tensor-parallel group through vLLM's RoCE slot class
  (:class:`.tp_slot.SirclRingAllReduce`): the instance vLLM already built for
  the slot, or one built here with the route map the layout derives
  (:meth:`.fabric.GroupTopology.route_map`); a route map given in
  ``SIRCL_PEER_ROUTES`` must equal it;
- a decode-context-parallel group through :class:`.dcp_collectives.SirclDcpCollectives`,
  routed over the tensor-parallel group's fabric (a subgroup routes over its
  parent's cables).

A group NCCL may not run on (:class:`.fabric.NcclPolicy`) gets the extra
all-reduce dtypes float16 and float32 prepared, because no NCCL path exists
for them. A tensor-parallel or decode-context-parallel group NCCL may not run
on must have a session, or setup fails. Another group on such a placement
whose global ranks equal, in order, those of a group with a session (the
expert-parallel group a mixture-of-experts model gets without expert
parallelism has the tensor-parallel group's ranks) shares that session: both
groups' collectives are issued by each rank's one thread in program order, so
every rank sees the same sequence on the shared session. The decision is
voted. Any other group there is built without PyNccl and without a session;
any collective on it is refused when issued.

Point-to-point channels. A pipeline-parallel group, and a tensor-parallel or
decode-context-parallel group with its own session, gets point-to-point
channels when ``SIRCL_P2P_GROUPS`` names its kind (:mod:`.p2p`; the module
``SIRCL_P2P_MODULE`` names provides them). Setup is collective and voted; a PP
group whose channels cannot be built fails setup on every rank, a session
group keeps its session and refuses send and receive (the receipt says why).
A group with the same global ranks in the same order shares a group's
channels (voted). Send, receive and batched send/receive of the device
communicator, and the ``torch.distributed`` point-to-point calls and (on a
group without a collective session) broadcasts that reach the tripwire, run on
the channels wherever the group has a channel to the peer
(:func:`.planner.plan_point_to_point`).

Direct ``torch.distributed`` calls. A group with a session (own or shared)
registers :meth:`GroupAdapter.carry_torch` with the guard's tripwire, which
then runs refused ``torch.distributed`` calls through this adapter's plans:
``all_reduce`` with sum (the session's rank-ordered sum), max or min (every
rank's values, then the element-wise extremum: exact), ``broadcast``,
``all_gather`` and ``all_gather_into_tensor`` (bytes in rank order),
``reduce_scatter_tensor`` with sum, and ``all_to_all_single`` with equal
splits. vLLM issues such calls outside the device communicator, for example
the weight ``amax`` reductions of online quantization at startup and the
decode-context-parallel KV gathers of chunked prefill. Any other call stays
refused.

Fail-stop. Session setup is collective and voted; a failure raises on every
rank. After setup, a collective the plan refuses raises
:class:`SirclDispatchError` on every rank alike (plans are rank-invariant),
and session failures propagate. Nothing falls back to NCCL on a group whose
cabling forbids it.

Capture and health fan-out. vLLM enters only the tensor-parallel, pipeline
and data-parallel groups' capture contexts and checks only the
tensor-parallel communicator's ``b12x_ar_comm.check_health`` after a step.
:func:`capture_all` and :func:`check_all_health` cover every live session of
the process, and the communicator chains them into that slot.

Flag-wait regimes. A session waits for a late peer at most the limit of its
regime: ``startup`` (minutes: compilation, warm-up, graph capture) or
``serving`` (seconds: a longer lag means a failed rank). Sessions start in the
startup regime. The ``worker_regimes`` shim (:mod:`.shims`) runs the worker's
methods that can make one rank late (warm-up and graph capture, memory
profiling, sleep and wake-up, weight reloads, profiler start and stop) inside
:func:`startup_all`, and arms the serving regime when the warm-up method
returns. vLLM's warm-up issues real steps through ``execute_model``, so
post-step checks also run during warm-up; they keep the startup regime while
any startup block is in progress. The first post-step check after warm-up
outside such a block (the engine's first real step) puts every session of the
process in the serving regime. Without the shim the serving regime is never
armed and sessions keep the startup limit.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import threading
import time
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import torch

from . import executor, fabric, guard, groupops, planner, receipt, sessionapi, settings, tp4
from . import p2p as p2p_mod
from .fabric import FabricError, GroupTopology, Layout, NcclPolicy
from .planner import NCCL, REFUSE, SIRCL, P2PLimits, Plan, Policy, SessionLimits, TensorMeta

logger = logging.getLogger("sircl.vllm.adapter")

SESSION_KINDS = ("tp", "dcp")
WAIT_REGIMES = ("startup", "serving")
SUBGROUP_KINDS = ("dcp",)
EXTRA_REDUCE_DTYPES = (torch.float16, torch.float32)


class SirclDispatchError(RuntimeError):
    """A collective that neither SIRCL nor (for this group's cabling) NCCL may carry."""


class SirclSetupError(RuntimeError):
    """A group's SIRCL setup is impossible with this configuration."""


# -- configuration ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AdapterConfig:
    layout: Layout
    positions: tuple[int, ...]       # fabric position of every global rank of the instance
    groups: tuple[str, ...]
    nccl_mode: str
    large: str
    relay_override: int | None
    receipt_dir: str | None
    p2p_groups: tuple[str, ...] = settings.DEFAULT_P2P_GROUPS
    p2p_module: str = settings.DEFAULT_P2P_MODULE

    @classmethod
    def from_env(cls, world: int, environ=None) -> "AdapterConfig":
        layout = Layout.parse(settings.fabric_text(environ))
        positions = settings.rank_positions(world, environ)
        outside = [p for p in positions if p >= layout.size]
        if outside:
            raise settings.SettingError(
                f"SIRCL_RANK_POSITIONS {list(positions)} name positions {outside} outside the "
                f"{layout.describe()} layout")
        return cls(layout, positions, settings.groups(environ), settings.nccl_mode(environ),
                   settings.large_allreduce(environ), settings.relay_per_peer_bytes(environ),
                   settings.receipt_dir(environ), settings.p2p_groups(environ), settings.p2p_module(environ))

    def cabled(self, global_a: int, global_b: int) -> bool:
        a, b = self.positions[global_a], self.positions[global_b]
        return any({a, b} == {cable.a, cable.b} for cable in self.layout.cables)


@dataclasses.dataclass(frozen=True)
class GroupPlacement:
    """Where a group sits and what NCCL may do there; computed before vLLM builds its communicator."""

    name: str
    kind: str
    global_ranks: tuple[int, ...]
    rank: int
    positions: tuple[int, ...]
    topology: GroupTopology | None
    policy: NcclPolicy
    reason: str
    session: bool                       # this group gets a SIRCL session
    raw_policy: NcclPolicy = NcclPolicy.ALL   # what the cabling alone allows (before SIRCL_NCCL and NCCL settings)

    @property
    def world(self) -> int:
        return len(self.global_ranks)

    @property
    def suppress_pynccl(self) -> bool:
        """PyNccl's constructor all-reduces over the whole group; only allowed where NCCL may."""
        return self.world > 1 and not self.policy.allows("all_reduce")

    @classmethod
    def of(cls, unique_name: str, global_ranks: Sequence[int], rank: int, config: AdapterConfig,
           *, parent_ranks: Sequence[int] | None = None, environ=None) -> "GroupPlacement":
        kind = unique_name.split(":")[0]
        ranks = tuple(int(r) for r in global_ranks)
        if max(ranks, default=0) >= len(config.positions):
            raise SirclSetupError(f"group {unique_name} has global ranks {list(ranks)} beyond the "
                                  f"{len(config.positions)} positions SIRCL knows")
        positions = tuple(config.positions[r] for r in ranks)
        session = kind in config.groups and kind in SESSION_KINDS and len(ranks) > 1
        topology = None
        if session:
            parent = None
            if kind in SUBGROUP_KINDS:
                if parent_ranks is None:
                    raise SirclSetupError(f"subgroup {unique_name} needs its parent group's ranks")
                parent = tuple(config.positions[r] for r in parent_ranks)
            topology = fabric.describe_group(config.layout, positions, parent=parent)
            raw_policy, raw_reason = topology.nccl_policy, topology.nccl_reason
        else:
            raw_policy, raw_reason = fabric.nccl_policy_of(config.layout, positions)
        policy, reason = guard.effective_policy(raw_policy, raw_reason, nccl_mode=config.nccl_mode,
                                                environ=environ)
        return cls(unique_name, kind, ranks, int(rank), positions, topology, policy, reason, session,
                   raw_policy)

    def guard_entry(self, config: AdapterConfig, *, carried: bool | None = None) -> guard.GuardedGroup:
        carried = self.session if carried is None else carried
        remedy = guard.remedies(self.raw_policy, self.policy, carried=carried) if self.world > 1 else ""
        return guard.GuardedGroup(self.name, self.global_ranks, self.policy, self.reason,
                                  cabled=config.cabled, remedy=remedy)


def resolver_for(config: AdapterConfig, environ=None) -> Callable[[Sequence[int]], guard.GuardedGroup | None]:
    """Classify an NCCL process group SIRCL did not build (the default group, sibling groups)."""

    def resolve(ranks: Sequence[int]) -> guard.GuardedGroup | None:
        try:
            positions = [config.positions[r] for r in ranks]
            raw, why = fabric.nccl_policy_of(config.layout, positions)
        except (IndexError, FabricError):
            return guard.GuardedGroup(f"ranks {list(ranks)}", tuple(ranks), NcclPolicy.NONE,
                                      "the group's ranks are not on the SIRCL layout")
        policy, reason = guard.effective_policy(raw, why, nccl_mode=config.nccl_mode, environ=environ)
        return guard.GuardedGroup(f"process group of ranks {list(ranks)}", tuple(ranks), policy,
                                  reason, cabled=config.cabled,
                                  remedy=guard.remedies(raw, policy, carried=False))

    return resolve


# -- live sessions of this process ---------------------------------------------------------

_LIVE: "weakref.WeakSet[GroupAdapter]" = weakref.WeakSet()
_LIVE_LOCK = threading.Lock()
# Regime changes are serialized: vLLM may complete an asynchronous step's
# output (and run the post-step check) on another thread while a startup
# block begins.
_REGIME_LOCK = threading.RLock()
_STARTUP_HOLDS = 0                      # startup_all blocks in progress
_SERVING_ARMED = False                  # the worker's warm-up has returned (startup_all then_serve)


def live_adapters() -> list["GroupAdapter"]:
    with _LIVE_LOCK:
        return sorted(_LIVE, key=lambda adapter: adapter.placement.name)


def _enter_regime(adapters: Sequence["GroupAdapter"], regime: str, reason: str) -> list[str]:
    changed = [adapter.placement.name for adapter in adapters if adapter.enter_regime(regime)]
    if changed:
        limits = sorted({text.partition(":")[2] for adapter in adapters
                         if (text := adapter._wait_text()) and text.startswith(regime)})
        logger.info("SIRCL %s regime on %s (flag waits up to %s): %s", regime, ", ".join(changed),
                    ", ".join(limits) or "the session's limit", reason)
    return changed


def check_all_health() -> None:
    """Fail-stop check of every SIRCL session in this process, run by vLLM's worker after each step.

    Once the worker's warm-up has returned (the serving regime is armed) and
    no :func:`startup_all` block is in progress, a completed step means the
    engine is serving: every session enters the serving regime first (a few
    attribute reads when it already is).
    """
    adapters = live_adapters()
    with _REGIME_LOCK:
        if _SERVING_ARMED and not _STARTUP_HOLDS:
            _enter_regime(adapters, "serving", "a step completed after warm-up")
    for adapter in adapters:
        adapter.check_health()


def enter_serving_all(reason: str = "requested") -> list[str]:
    """Put every live session of this process in the serving regime; returns the groups that changed."""
    with _REGIME_LOCK:
        return _enter_regime(live_adapters(), "serving", reason)


def enter_startup_all(reason: str = "requested") -> list[str]:
    """Put every live session in the startup regime until a later post-step check outside a startup block."""
    with _REGIME_LOCK:
        return _enter_regime(live_adapters(), "startup", reason)


@contextlib.contextmanager
def startup_all(reason: str = "start-up work", *, then_serve: bool = False) -> Iterator[None]:
    """The startup regime on every live session for the block and until the next completed step.

    Wraps work after which one rank may reach its next collective much later
    than its peers. Post-step checks that run during the block leave the
    regime alone; once the serving regime is armed, the first one after the
    last block ends selects it. ``then_serve`` (the worker's warm-up) arms the
    serving regime when the block ends without an exception.
    """
    global _STARTUP_HOLDS, _SERVING_ARMED
    with _REGIME_LOCK:
        _STARTUP_HOLDS += 1
        _enter_regime(live_adapters(), "startup", reason)
    completed = False
    try:
        yield
        completed = True
    finally:
        with _REGIME_LOCK:
            _STARTUP_HOLDS -= 1
            if completed and then_serve:
                _SERVING_ARMED = True


def startup_held() -> bool:
    """True while a :func:`startup_all` block is in progress in this process."""
    return _STARTUP_HOLDS > 0


def serving_armed() -> bool:
    """True once the worker's warm-up has returned: post-step checks then select the serving regime."""
    return _SERVING_ARMED


def reset_regimes_for_tests() -> None:
    global _STARTUP_HOLDS, _SERVING_ARMED
    with _REGIME_LOCK:
        _STARTUP_HOLDS = 0
        _SERVING_ARMED = False


@contextlib.contextmanager
def capture_all(stream: Any = None) -> Iterator[None]:
    """Enter the CUDA graph capture context of every live SIRCL session."""
    with contextlib.ExitStack() as stack:
        for adapter in live_adapters():
            context = adapter.session_capture(stream)
            if context is not None:
                stack.enter_context(context)
        yield


# -- the adapter ---------------------------------------------------------------------------


class NcclPaths:
    """The stock vLLM implementations SIRCL may hand a collective to (set by the communicator)."""

    def __init__(self, **methods: Callable[..., Any]) -> None:
        self._methods = methods

    def __getattr__(self, name: str) -> Callable[..., Any]:
        try:
            return self._methods[name]
        except KeyError:
            raise AttributeError(name) from None


def _capturing_now() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


class GroupAdapter:
    """One rank's SIRCL view of one vLLM group."""

    def __init__(
        self,
        placement: GroupPlacement,
        *,
        config: AdapterConfig,
        cpu_group: Any,
        device_group: Any,
        device: Any,
        slot: Any = None,
        nccl: NcclPaths | None = None,
        communicator: Any = None,
        communicator_class: type | None = None,
        capturing: Callable[[], bool] = _capturing_now,
        single_node: bool | None = None,
        environ=None,
        shape: p2p_mod.Shape | None = None,
        parent_ranks: Sequence[int] | None = None,
    ) -> None:
        self.placement = placement
        self.config = config
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.device = device
        self.nccl = nccl
        self.communicator = communicator
        self.capturing = capturing
        self._environ = environ
        self.counters = receipt.Counters()
        # Column gathers staged on the session's links (executor.ColumnGather; SIRCL_COLUMN_GATHER).
        self.column_gather = executor.ColumnGather.from_environment(environ)
        self.fused_norm: Any = None             # bound fused all-reduce + RMSNorm (norm_fusion)
        self.slot = None
        self.dcp = None
        self.session = None
        self.limits: SessionLimits | None = None
        self.shared_from: str | None = None     # the group whose session this group shares
        self.prepared_extra: tuple[str, ...] = ()
        self.per_peer: tuple[int | None, str] = (None, "no relayed lane")
        self._tp4 = False
        self._closed = False
        self._receipt_dirty = False
        self._receipt_written = 0.0
        self._receipt_version = -1      # counters version of the last write
        self._stats_written = 0.0
        self._session_stats: dict[str, Any] | None = None
        self._receipt_failed = False
        self.p2p: Any = None                    # this group's point-to-point channels (own or shared)
        self.p2p_plan: p2p_mod.GroupChannels | None = None
        self.p2p_shared_from: str | None = None
        self.p2p_reason = ""                    # why the group has no channels
        self._shape = shape
        self._parent_ranks = None if parent_ranks is None else tuple(int(r) for r in parent_ranks)
        guard.register(device_group, placement.guard_entry(config))
        nccl_above: tuple[tuple[str, int], ...] = ()
        try:
            if placement.session:
                if placement.topology is not None:
                    self.per_peer = placement.topology.per_peer_op_bytes(config.relay_override)
                if placement.kind == "tp":
                    self._build_tp(slot, single_node, environ)
                elif placement.kind == "dcp":
                    nccl_above = self._build_dcp()
            if self.session is None and placement.world > 1 and placement.policy is NcclPolicy.NONE:
                self._share_session()
            if (placement.kind in SESSION_KINDS and placement.world > 1
                    and placement.policy is NcclPolicy.NONE and self.session is None):
                raise SirclSetupError(
                    f"group {placement.name} (global ranks {list(placement.global_ranks)}) may not "
                    f"use NCCL ({placement.reason}) and has no SIRCL session; add "
                    f"{placement.kind} to SIRCL_GROUPS or change the placement")
            if placement.kind == "tp" and communicator_class is not None:
                self._tp4 = tp4.unavailable_reason(placement.topology, kind="tp",
                                                   communicator_class=communicator_class) is None
            if placement.world > 1:
                self._build_p2p()
                if self.p2p is None:
                    self._share_p2p()
        except BaseException:
            guard.unregister(device_group)
            self.close()
            raise
        ranks = placement.global_ranks
        self.policy = Policy(placement.topology, config.nccl_mode, config.large, nccl_above,
                             placement.policy, placement.reason,
                             lambda a, b: config.cabled(ranks[a], ranks[b]),
                             tuned=(sessionapi.tuned_backend(self.session)
                                    if self.session is not None and placement.policy is not NcclPolicy.NONE
                                    else None))
        if self.session is not None:
            guard.register(device_group, placement.guard_entry(config, carried=True))
            guard.register_carrier(device_group, self.carry_torch)
        if self.p2p is not None:
            guard.register_p2p_carrier(device_group, placement.global_ranks, self.carry_p2p)
        with _LIVE_LOCK:
            _LIVE.add(self)
        self.record = self._record()
        logger.info("%s", receipt.line(self.record))
        self.write_receipt()

    # -- sessions ----------------------------------------------------------------------

    def _routes(self, environ) -> dict[int, tuple[str, ...]]:
        """This rank's route map from the layout; a given ``SIRCL_PEER_ROUTES`` must equal it.

        The check is voted, so a map that is wrong on one rank fails setup on
        every rank instead of leaving the others waiting in the session's setup
        exchange.
        """
        import os

        topology = self.placement.topology
        assert topology is not None
        derived = topology.route_map(self.placement.rank)
        env = os.environ if environ is None else environ
        given = env.get("SIRCL_PEER_ROUTES", "").strip() if self.placement.kind == "tp" else ""
        error = None
        if given:
            try:
                routes = fabric.parse_routes(given, world=self.placement.world, rank=self.placement.rank)
                fabric.check_routes(topology, self.placement.rank, routes)
            except FabricError as exc:
                error = str(exc)
        verdict = groupops.vote(self.cpu_group, error, compare=False)
        if verdict is not None:
            raise SirclSetupError(f"route maps of {self.placement.name} do not match SIRCL_FABRIC "
                                  f"{self.config.layout.describe()}: {verdict}")
        return derived

    def _build_tp(self, slot: Any, single_node: bool | None, environ) -> None:
        from .tp_slot import SirclRingAllReduce

        routes = self._routes(environ)
        if slot is not None and not isinstance(slot, SirclRingAllReduce) and not getattr(slot, "disabled", True):
            raise SirclSetupError(
                f"group {self.placement.name} already has the "
                f"{getattr(slot, 'backend_name', type(slot).__name__)} transport in vLLM's RoCE slot; "
                "SIRCL and another RDMA transport cannot serve one group (select one per run)")
        if not isinstance(slot, SirclRingAllReduce):
            slot = SirclRingAllReduce(self.cpu_group, self.device_group, self.device,
                                      global_ranks=self.placement.global_ranks, peer_routes=routes,
                                      layout=self.placement.topology.session_layout(),
                                      single_node=single_node)
        self.slot = slot
        if slot.disabled or slot.runtime is None:
            return
        self.session = slot.runtime
        reduce = ["bfloat16"]
        if self.placement.policy is NcclPolicy.NONE:
            error = None
            try:
                self.session.prepare(EXTRA_REDUCE_DTYPES, padded_gather=slot.all_gather_max_bytes > 0,
                                     **sessionapi.link_keywords(self.session))
            except Exception as exc:  # noqa: BLE001 - voted below
                error = f"{type(exc).__name__}: {exc}"
            verdict = groupops.vote(self.cpu_group, error, compare=False)
            if verdict is not None:
                raise SirclSetupError(f"SIRCL could not prepare float16 and float32 all-reduce for "
                                      f"{self.placement.name}: {verdict}")
            reduce += ["float16", "float32"]
            self.prepared_extra = ("float16", "float32")
        scatter = self._prepare_tp_scatter(environ)
        self.limits = SessionLimits.of(self.session, reduce_dtypes=reduce, scatter_dtypes=scatter,
                                       per_peer_op_bytes=self.per_peer[0],
                                       gather=slot.all_gather_max_bytes)

    def _prepare_tp_scatter(self, environ) -> tuple[str, ...]:
        """BF16 session reduce-scatter for prefill row ownership, when the session has it.

        GLM-5.3-Flash's mHC rows (``VLLM_GLM53_MHC_PREFILL_SHARD``) and Qwen3.8's
        hyper-connection rows (``VLLM_QWEN3_8_HC_PREFILL_MODE=shard``)
        reduce-scatter every sublayer output. Without the session's
        reduce-scatter the planner composes it as an all-reduce and this rank's
        rows (same bits). Prepared only on a group NCCL may not run, with one of
        the two settings and ``scatter_available`` on every rank (voted, like
        every preparation).
        """
        from . import mhc, qwen_hc

        if self.placement.policy is not NcclPolicy.NONE or not (mhc.requested(environ)
                                                                or qwen_hc.requested(environ)):
            return ()
        available = bool(getattr(self.session, "scatter_available", False))
        verdict = groupops.vote(self.cpu_group, (None, available), compare=True)
        if verdict is not None:
            raise SirclSetupError(f"ranks of {self.placement.name} disagree on the session reduce-scatter: "
                                  f"{verdict}")
        if not available:
            return ()
        import torch

        error = None
        try:
            self.session.prepare((torch.bfloat16,), scatter=True, **sessionapi.link_keywords(self.session))
        except Exception as exc:  # noqa: BLE001 - voted below
            error = f"{type(exc).__name__}: {exc}"
        verdict = groupops.vote(self.cpu_group, error, compare=False)
        if verdict is not None:
            raise SirclSetupError(f"SIRCL could not prepare the BF16 reduce-scatter of {self.placement.name}: "
                                  f"{verdict}")
        return ("bfloat16",)

    def _share_session(self) -> None:
        """Share the session of this rank's group with the same global ranks in the same order (voted)."""
        placement = self.placement
        owners = [adapter for adapter in live_adapters()
                  if adapter.session is not None and adapter.shared_from is None and not adapter._closed
                  and adapter.placement.global_ranks == placement.global_ranks
                  and adapter.placement.rank == placement.rank]
        owners.sort(key=lambda adapter: adapter.placement.kind != "tp")
        owner = owners[0] if owners else None
        verdict = groupops.vote(self.cpu_group, (None, owner.placement.name if owner else None), compare=True)
        if verdict is not None:
            raise SirclSetupError(f"ranks of {placement.name} disagree on the group whose SIRCL session it "
                                  f"shares: {verdict}")
        if owner is None:
            return
        self.session = owner.session
        self.limits = owner.limits
        self.column_gather = owner.column_gather
        self.per_peer = owner.per_peer
        self.prepared_extra = owner.prepared_extra
        self.shared_from = owner.placement.name
        logger.info("SIRCL group %s shares the session of %s (global ranks %s)", placement.name,
                    owner.placement.name, list(placement.global_ranks))

    def _build_dcp(self) -> tuple[tuple[str, int], ...]:
        from .dcp_collectives import SirclDcpCollectives

        routes = self._routes(None)
        dcp = SirclDcpCollectives(cpu_group=self.cpu_group, device=self.device,
                                  global_ranks=self.placement.global_ranks,
                                  positions=self.placement.positions, routes=routes,
                                  per_peer_bytes=self.per_peer[0],
                                  layout=self.placement.topology.session_layout())
        self.dcp = dcp
        self.session = dcp._runtime
        self.limits = SessionLimits.of(self.session, reduce_dtypes=("bfloat16",),
                                       scatter_dtypes=("bfloat16",), all_to_all=True,
                                       per_peer_op_bytes=dcp.gather_chunk,
                                       gather=min(dcp.gather_max, self.session.max_gather_bytes),
                                       scatter_op=dcp.scatter_op_bytes)
        names = {"all-gather": "all_gather", "reduce-scatter": "reduce_scatter",
                 "all-to-all": "all_to_all"}
        return tuple((names[kind], int(limit)) for kind, limit in dcp.thresholds.items())

    # -- point-to-point channels -----------------------------------------------------------

    def _p2p_failed(self, required: bool, why: str) -> None:
        if required:
            raise SirclSetupError(f"SIRCL point-to-point channels of {self.placement.name} (global ranks "
                                  f"{list(self.placement.global_ranks)}): {why}")
        self.p2p_reason = f"its point-to-point channels could not be built: {why}"
        logger.warning("SIRCL group %s has no point-to-point channels: %s", self.placement.name, why)

    def _build_p2p(self) -> None:
        """Point-to-point channels of a PP group, or of a session group with its own session (voted)."""
        placement = self.placement
        kind = placement.kind
        if kind not in p2p_mod.P2P_KINDS:
            self.p2p_reason = f"{kind} groups get no point-to-point channels of their own"
            return
        if kind not in self.config.p2p_groups:
            self.p2p_reason = f"SIRCL_P2P_GROUPS does not name {kind}"
            return
        if kind in SESSION_KINDS and (self.session is None or self.shared_from is not None):
            self.p2p_reason = "the group has no SIRCL session of its own"
            return
        # A PP group NCCL may not run cannot pass its stages' tensors without channels.
        required = kind == "pp" and placement.policy is NcclPolicy.NONE
        error = None
        module = None
        plan = None
        try:
            plan = p2p_mod.plan(self.config.layout, self.config.positions, self.config.groups,
                                self.config.p2p_groups, self._shape, kind=kind, global_ranks=placement.global_ranks,
                                parent_ranks=self._parent_ranks, environ=self._environ)
            if plan is None:
                raise SirclSetupError("the instance plan has no channels for this group")
            if required and plan.unavailable:
                # Every pair of a PP group carries tensors (adjacent stages) or the sampled tokens (the last
                # stage), and NCCL may not carry any of them here.
                raise SirclSetupError("pairs without a channel: " + "; ".join(
                    f"{a}->{b}: {why}" for (a, b), why in sorted(plan.unavailable.items())))
            module = sessionapi.load_p2p_module(self.config.p2p_module)
        except Exception as exc:  # noqa: BLE001 - voted below
            error = f"{type(exc).__name__}: {exc}"
        verdict = groupops.vote(self.cpu_group, error, compare=False)
        if verdict is not None:
            self._p2p_failed(required, verdict)
            return
        try:
            channels = module.PointToPoint(
                exchange_group=self.cpu_group, device=self.device, peer_routes=plan.route_map(placement.rank),
                layout=plan.layout_text, windows=[[list(lanes) for lanes in row] for row in plan.windows],
                unavailable=dict(plan.unavailable))
        except Exception as exc:  # noqa: BLE001 - setup is collective: every rank raises alike
            self._p2p_failed(required, f"{type(exc).__name__}: {exc}")
            return
        error = None
        try:
            channels.prepare()
        except Exception as exc:  # noqa: BLE001 - voted below
            error = f"{type(exc).__name__}: {exc}"
        verdict = groupops.vote(self.cpu_group, error, compare=False)
        if verdict is not None:
            with contextlib.suppress(Exception):
                channels.close()
            self._p2p_failed(required, f"compiling the point-to-point kernels failed: {verdict}")
            return
        self.p2p = channels
        self.p2p_plan = plan
        self.p2p_reason = ""

    def _share_p2p(self) -> None:
        """Share the channels of this rank's group with the same global ranks in the same order (voted)."""
        placement = self.placement
        owners = [adapter for adapter in live_adapters()
                  if adapter.p2p is not None and adapter.p2p_shared_from is None and not adapter._closed
                  and adapter.placement.global_ranks == placement.global_ranks
                  and adapter.placement.rank == placement.rank]
        owners.sort(key=lambda adapter: (adapter.placement.kind != "tp", adapter.placement.name))
        owner = owners[0] if owners else None
        verdict = groupops.vote(self.cpu_group, (None, owner.placement.name if owner else None), compare=True)
        if verdict is not None:
            raise SirclSetupError(f"ranks of {placement.name} disagree on the group whose point-to-point channels it "
                                  f"shares: {verdict}")
        if owner is None:
            return
        self.p2p = owner.p2p
        self.p2p_plan = owner.p2p_plan
        self.p2p_shared_from = owner.placement.name
        self.p2p_reason = ""

    def p2p_limits(self) -> P2PLimits:
        """This rank's channels for the planner: peers with a channel, relayed peers, and why others have none."""
        rank = self.placement.rank
        if self.p2p is None or self._closed:
            return P2PLimits(frozenset(), reason=self.p2p_reason or "the group has no point-to-point channels")
        world = self.placement.world
        peers = frozenset(peer for peer in range(world) if self.p2p.has_channel(peer))
        problems = tuple((peer, str(self.p2p.channel_problem(peer))) for peer in range(world)
                         if peer != rank and peer not in peers)
        relayed = self.p2p_plan.relayed_peers(rank) if self.p2p_plan is not None else frozenset()
        return P2PLimits(peers, frozenset(relayed) & peers, problems)

    def _plan_p2p(self, collective: str, peer: int, nbytes: int = 0) -> Plan:
        return planner.plan_point_to_point(collective, self.policy, self.placement.rank, int(peer),
                                           p2p=self.p2p_limits(), capturing=self._now_capturing(), nbytes=nbytes)

    def _settle_p2p(self, plan: Plan, pair: tuple[int, int], key: tuple[str, str, str] | None = None) -> Plan:
        self._count(key or plan.key(), plan.reason)
        if plan.backend == REFUSE:
            raise SirclDispatchError(f"SIRCL refuses {plan.collective} on {self.placement.name} (rank "
                                     f"{self.placement.rank}): {plan.reason}")
        if plan.backend == NCCL:
            self.placement.guard_entry(self.config).check(plan.collective, pair=pair)
        return plan

    def _pair(self, peer: int) -> tuple[int, int]:
        ranks = self.placement.global_ranks
        return ranks[self.placement.rank], ranks[peer]

    def send(self, tensor: torch.Tensor, peer: int, fallback: Callable[[], Any]) -> None:
        """The device communicator's send to group rank ``peer``: the channel, NCCL (``fallback``) or a refusal."""
        plan = self._settle_p2p(self._plan_p2p("send", peer, tensor.numel() * tensor.element_size()),
                                self._pair(peer))
        if plan.backend == SIRCL:
            self.p2p.send(tensor, peer)
            return None
        return fallback()

    def recv(self, tensor: torch.Tensor, peer: int, fallback: Callable[[], torch.Tensor]) -> torch.Tensor:
        """The device communicator's receive from group rank ``peer`` into ``tensor``."""
        plan = self._settle_p2p(self._plan_p2p("recv", peer, tensor.numel() * tensor.element_size()),
                                self._pair(peer))
        if plan.backend == SIRCL:
            return self.p2p.recv(tensor, peer)
        return fallback()

    def batch_isend_irecv(self, ops: Sequence[tuple[str, torch.Tensor, int]], fallback: Callable[[], Any]) -> Any:
        """``(kind, tensor, group peer)`` ops of one batch: all on the channels, or all on NCCL."""
        plans = [self._plan_p2p(kind, peer, tensor.numel() * tensor.element_size()) for kind, tensor, peer in ops]
        backends = {plan.backend for plan in plans}
        if REFUSE in backends or len(backends) > 1:
            refused = next((plan for plan in plans if plan.backend == REFUSE), None)
            why = refused.reason if refused is not None else (
                "the batch mixes pairs SIRCL's channels carry with pairs only NCCL carries")
            self._count(("batch_isend_irecv", "refuse", "refuse"), why)
            raise SirclDispatchError(f"SIRCL refuses batch_isend_irecv on {self.placement.name} (rank "
                                     f"{self.placement.rank}): {why}")
        if backends == {NCCL}:
            for plan, (_, _, peer) in zip(plans, ops):
                self._settle_p2p(plan, self._pair(peer))
            return fallback()
        self._count(("batch_isend_irecv", "sircl", "batch"), f"{len(ops)} ops on the group's channels")
        for plan in plans:
            self._count(plan.key(), plan.reason)
        for work in self.p2p.batch_isend_irecv(list(ops)):
            work.wait()
        return None

    # -- torch.distributed point-to-point calls the tripwire hands over ---------------------

    def _group_peer(self, arguments: Mapping[str, Any], local: str, global_: str) -> int | None:
        peer = arguments.get(local)
        if peer is not None:
            return int(peer)
        value = arguments.get(global_)
        if value is None or int(value) not in self.placement.global_ranks:
            return None
        return self.placement.global_ranks.index(int(value))

    def carry_p2p(self, operation: str, arguments: Mapping[str, Any]) -> Any:
        """Run a ``torch.distributed`` point-to-point call (or a broadcast, on a group without a collective
        session) on the channels; ``guard.NOT_CARRIED`` when the channels do not cover it."""
        if self.p2p is None or self._closed:
            return guard.NOT_CARRIED
        if operation in ("isend", "irecv", "send", "recv"):
            tensor = arguments.get("tensor")
            sending = operation in ("isend", "send")
            peer = (self._group_peer(arguments, "group_dst", "dst") if sending
                    else self._group_peer(arguments, "group_src", "src"))
            if not isinstance(tensor, torch.Tensor) or peer is None or (arguments.get("tag") or 0) != 0:
                return guard.NOT_CARRIED
            plan = self._plan_p2p("send" if sending else "recv", peer, tensor.numel() * tensor.element_size())
            if plan.backend == NCCL:
                return guard.NOT_CARRIED
            self._settle_p2p(plan, self._pair(peer), (f"torch.{operation}", plan.backend, plan.method))
            work = self.p2p.isend(tensor, peer) if sending else self.p2p.irecv(tensor, peer)
            if operation in ("isend", "irecv"):
                return work
            work.wait()
            return None if sending else self.placement.global_ranks[peer]
        if operation == "batch_isend_irecv":
            ops = []
            for op in arguments.get("p2p_op_list") or ():
                name = getattr(getattr(op, "op", None), "__name__", "")
                tensor = getattr(op, "tensor", None)
                peer = getattr(op, "group_peer", None)
                if peer is None and getattr(op, "peer", None) in self.placement.global_ranks:
                    peer = self.placement.global_ranks.index(op.peer)
                if name not in ("isend", "irecv") or not isinstance(tensor, torch.Tensor) or peer is None:
                    return guard.NOT_CARRIED
                ops.append(("send" if name == "isend" else "recv", tensor, int(peer)))
            plans = [self._plan_p2p(kind, peer, t.numel() * t.element_size()) for kind, t, peer in ops]
            if not ops or any(plan.backend != SIRCL for plan in plans):
                return guard.NOT_CARRIED
            self._count(("torch.batch_isend_irecv", "sircl", "batch"), f"{len(ops)} ops on the group's channels")
            return self.p2p.batch_isend_irecv(ops)
        if operation == "broadcast" and self.session is None:
            tensor = arguments.get("tensor")
            source = self._group_peer(arguments, "group_src", "src")
            if not isinstance(tensor, torch.Tensor) or source is None:
                return guard.NOT_CARRIED
            rank = self.placement.rank
            peers = [peer for peer in range(self.placement.world) if peer != rank] if rank == source else [source]
            nbytes = tensor.numel() * tensor.element_size()
            plans = [self._plan_p2p("send" if rank == source else "recv", peer, nbytes) for peer in peers]
            if any(plan.backend != SIRCL for plan in plans):
                return guard.NOT_CARRIED
            self._count(("torch.broadcast", "sircl", "p2p"), "a broadcast as sends from the source on the group's "
                                                             "point-to-point channels")
            if rank == source:
                works = [self.p2p.isend(tensor, peer) for peer in peers]
            else:
                works = [self.p2p.irecv(tensor, source)]
            if arguments.get("async_op"):
                return _WorkGroup(works)
            for work in works:
                work.wait()
            return None
        return guard.NOT_CARRIED

    # -- capture and health --------------------------------------------------------------

    # The class-level methods are called on purpose: SIRCL's communicator replaces
    # the slot instance's capture and check_health with the process-wide
    # versions below, which call back into these.

    def session_capture(self, stream: Any = None):
        if self.slot is not None and not self.slot.disabled:
            return type(self.slot).capture(self.slot, stream=stream)
        if self.dcp is not None:
            return type(self.dcp).capture(self.dcp, stream=stream)
        return None

    def enter_regime(self, regime: str) -> bool:
        """Select the session's flag-wait regime (``startup`` or ``serving``); True when it changed.

        A session without regimes is left alone. The receipt is rewritten at
        the next health check.
        """
        if regime not in WAIT_REGIMES:
            raise ValueError(f"wait regime must be one of {WAIT_REGIMES}, got {regime!r}")
        session = self.session
        if (session is None and self.p2p is None) or self._closed:
            return False
        changed = False
        for target in (session, self.p2p if self.p2p_shared_from is None else None):
            if target is None:
                continue
            method = getattr(target, "enter_serving" if regime == "serving" else "enter_startup", None)
            if not callable(method) or getattr(target, "wait_regime", None) == regime:
                continue
            method()
            changed = True
        if changed:
            self._receipt_dirty = True
        return changed

    def check_health(self) -> None:
        if self.slot is not None:
            type(self.slot).check_health(self.slot)
        if self.dcp is not None:
            type(self.dcp).check_health(self.dcp)
        if self.p2p is not None and self.p2p_shared_from is None:
            self.p2p.check_health()
        self._refresh_receipt()

    def _refresh_receipt(self) -> None:
        """Rewrite the receipt from the worker's post-step health check, outside any forward pass.

        With fresh session statistics (``stats()`` reads a device counter)
        after a step that added a decision row or a regime change, and at least
        every ``REFRESH_SECONDS``; with the current counts alone (statistics of
        the last refresh) at most every ``COUNT_REFRESH_SECONDS`` while counts
        change. A failed write is logged once and never stops serving.
        """
        if not self.config.receipt_dir or self._closed:
            return
        now = time.monotonic()
        stats_due = self._receipt_dirty or now - self._stats_written >= receipt.REFRESH_SECONDS
        counts_due = (self.counters.version != self._receipt_version
                      and now - self._receipt_written >= receipt.COUNT_REFRESH_SECONDS)
        if not stats_due and not counts_due:
            return
        try:
            self.write_receipt(stats=stats_due)
        except OSError as exc:
            if not self._receipt_failed:
                self._receipt_failed = True
                logger.warning("SIRCL receipt for %s could not be written: %s", self.placement.name, exc)

    # -- dispatch ------------------------------------------------------------------------

    def _now_capturing(self) -> bool:
        return bool(self.capturing())

    def _count(self, key: tuple[str, str, str], reason: str) -> None:
        if self.counters.count(key, reason):
            self._receipt_dirty = True

    def _settle(self, plan: Plan) -> Plan:
        self._count(plan.key(), plan.reason)
        if plan.backend == REFUSE:
            raise SirclDispatchError(
                f"SIRCL refuses {plan.collective} on {self.placement.name} (rank "
                f"{self.placement.rank}): {plan.reason}")
        if plan.backend == NCCL:
            self.placement.guard_entry(self.config).check(plan.collective)
            if self.nccl is None:
                raise SirclDispatchError(f"{plan.collective} on {self.placement.name} planned for "
                                         "NCCL, but no NCCL path is attached")
        return plan

    def all_reduce(self, inp: torch.Tensor, *, in_place: bool = False) -> torch.Tensor:
        capturing = self._now_capturing()
        if self._tp4 and tp4.admits(self.communicator, inp, capturing=capturing):
            self._count(("all_reduce", "tp4", "four-rank"), "the four-rank sessions admit it")
            return self.nccl.all_reduce(inp)
        plan = self._settle(planner.plan_all_reduce(TensorMeta.of(inp), self.limits, self.policy,
                                                    capturing=capturing))
        if plan.backend == NCCL:
            return self.nccl.all_reduce_in_place(inp) if in_place else self.nccl.all_reduce(inp)
        return executor.all_reduce(self.session, plan, inp, self.limits, self.policy,
                                   capturing=capturing, out=inp if in_place else None)

    def all_gather(self, inp: torch.Tensor, dim: int = -1) -> torch.Tensor:
        capturing = self._now_capturing()
        plan = self._settle(planner.plan_all_gather(TensorMeta.of(inp), dim, self.limits,
                                                    self.policy, capturing=capturing))
        if plan.backend == NCCL:
            return self.nccl.all_gather(inp, dim)
        return executor.all_gather(self.session, plan, inp, dim, self.placement.world,
                                   column_gather=self.column_gather)

    def all_gatherv(self, inp: torch.Tensor | list[torch.Tensor], dim: int = 0,
                    sizes: list[int] | None = None):
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        if isinstance(inp, (list, tuple)):
            capturing = self._now_capturing()
            plans = [planner.plan_all_gatherv(TensorMeta.of(t), sizes, self.limits, self.policy,
                                              capturing=capturing) for t in inp]
            if any(plan.backend == NCCL for plan in plans) and all(p.backend != REFUSE for p in plans):
                for plan in plans:
                    self._count(plan.key(), plan.reason)
                self.placement.guard_entry(self.config).check("all_gather")
                return self.nccl.all_gatherv(inp, dim, sizes)
            return [self._all_gatherv_one(t, sizes) for t in inp]
        return self._all_gatherv_one(inp, sizes)

    def _all_gatherv_one(self, inp: torch.Tensor, sizes: list[int] | None):
        capturing = self._now_capturing()
        plan = self._settle(planner.plan_all_gatherv(TensorMeta.of(inp), sizes, self.limits,
                                                     self.policy, capturing=capturing))
        if plan.backend == NCCL:
            return self.nccl.all_gatherv(inp, 0, sizes)
        return executor.all_gatherv(self.session, plan, inp, sizes, self.placement.rank,
                                    self.placement.world)

    def reduce_scatter(self, inp: torch.Tensor, dim: int = -1) -> torch.Tensor:
        capturing = self._now_capturing()
        plan = self._settle(planner.plan_reduce_scatter(TensorMeta.of(inp), dim, self.limits,
                                                        self.policy, capturing=capturing))
        if plan.backend == NCCL:
            return self.nccl.reduce_scatter(inp, dim)
        return executor.reduce_scatter(self.session, plan, inp, dim, self.placement.rank,
                                       self.limits, self.policy, capturing=capturing)

    def reduce_scatterv(self, inp: torch.Tensor, dim: int = -1, sizes: list[int] | None = None):
        capturing = self._now_capturing()
        plan = self._settle(planner.plan_reduce_scatterv(TensorMeta.of(inp), sizes, self.limits,
                                                         self.policy, capturing=capturing))
        if plan.backend == NCCL:
            return self.nccl.reduce_scatterv(inp, dim, sizes)
        return executor.reduce_scatterv(self.session, plan, inp, dim, sizes, self.placement.rank,
                                        self.limits, self.policy, capturing=capturing)

    def gather(self, inp: torch.Tensor, dst: int = 0, dim: int = -1):
        capturing = self._now_capturing()
        plan = self._settle(planner.plan_bytes_gather("gather", TensorMeta.of(inp), self.limits,
                                                      self.policy, capturing=capturing,
                                                      nccl_operation="gather"))
        if plan.backend == NCCL:
            return self.nccl.gather(inp, dst, dim)
        return executor.gather(self.session, plan, inp, dst, dim, self.placement.rank,
                               self.limits, self.policy, capturing=capturing)

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        capturing = self._now_capturing()
        plan = self._settle(planner.plan_bytes_gather("broadcast", TensorMeta.of(tensor), self.limits,
                                                      self.policy, capturing=capturing,
                                                      nccl_operation="broadcast"))
        if plan.backend == NCCL:
            return self.nccl.broadcast(tensor, src)
        return executor.broadcast(self.session, plan, tensor, src, self.limits, self.policy,
                                  capturing=capturing)

    def all_to_all_single(self, output: torch.Tensor, input_: torch.Tensor) -> torch.Tensor:
        capturing = self._now_capturing()
        plan = self._settle(planner.plan_all_to_all(TensorMeta.of(input_), self.limits, self.policy,
                                                    capturing=capturing))
        if plan.backend == NCCL:
            return self.nccl.all_to_all_single(output, input_)
        return executor.all_to_all_single(self.session, plan, output, input_, self.placement.rank,
                                          self.limits, self.policy, capturing=capturing)

    # -- torch.distributed calls the tripwire hands over ----------------------------------

    def carry_torch(self, operation: str, arguments: Mapping[str, Any]) -> Any:
        """Run a refused ``torch.distributed`` call on the session, or return ``guard.NOT_CARRIED``.

        ``all_reduce`` with sum, max or min and ``broadcast`` of a tensor are
        carried; ``async_op`` calls get a completed work object, because the
        session's ops are already ordered on the caller's stream.
        """
        if self.session is None or self._closed:
            return guard.NOT_CARRIED
        if operation in _TENSOR_PAIRS:
            if not self._carry_pair(operation, arguments):
                return guard.NOT_CARRIED
            return _CompletedWork() if arguments.get("async_op") else None
        tensor = arguments.get("tensor")
        if not isinstance(tensor, torch.Tensor):
            return guard.NOT_CARRIED
        if operation == "all_gather":
            outputs = arguments.get("tensor_list")
            world = self.placement.world
            if (not isinstance(outputs, (list, tuple)) or len(outputs) != world
                    or any(not isinstance(t, torch.Tensor) or t.numel() != tensor.numel() for t in outputs)):
                return guard.NOT_CARRIED
            self._count((f"torch.{operation}", "sircl", "bytes"), "torch.distributed call carried on the session")
            rows = self._gather_flat(tensor)
            for source, target in enumerate(outputs):
                target.copy_(rows[source].view_as(target))
            return _CompletedWork() if arguments.get("async_op") else None
        if operation == "all_reduce":
            name = _reduce_op_name(arguments.get("op"))
            if name is None:
                return guard.NOT_CARRIED
            self._count((f"torch.{operation}", "sircl", name), "torch.distributed call carried on the session")
            self._carry_all_reduce(tensor, name)
        elif operation == "broadcast":
            source = arguments.get("group_src")
            if source is None:
                global_source = arguments.get("src")
                if global_source is None or int(global_source) not in self.placement.global_ranks:
                    return guard.NOT_CARRIED
                source = self.placement.global_ranks.index(int(global_source))
            self._count((f"torch.{operation}", "sircl", "bytes"), "torch.distributed call carried on the session")
            result = self.broadcast(tensor, int(source))
            if result is not tensor:
                tensor.copy_(result)
        else:
            return guard.NOT_CARRIED
        return _CompletedWork() if arguments.get("async_op") else None

    def _gather_flat(self, tensor: torch.Tensor) -> torch.Tensor:
        """Every rank's ``tensor`` as rows of a ``[world, numel]`` tensor, in rank order."""
        flat = tensor.reshape(-1).contiguous()
        return self.all_gather(flat, 0).view(self.placement.world, -1)

    def _carry_pair(self, operation: str, arguments: Mapping[str, Any]) -> bool:
        """The output-and-input calls: True when carried, False to leave the refusal standing."""
        out_name, in_name = _TENSOR_PAIRS[operation]
        output, source = arguments.get(out_name), arguments.get(in_name)
        if not isinstance(output, torch.Tensor) or not isinstance(source, torch.Tensor):
            return False
        world = self.placement.world
        if not output.is_contiguous() or output.dtype != source.dtype:
            return False
        if operation == "all_gather_into_tensor":
            if output.numel() != world * source.numel():
                return False
            self._count((f"torch.{operation}", "sircl", "bytes"), "torch.distributed call carried on the session")
            if source.numel():
                output.copy_(self._gather_flat(source).view_as(output))
            return True
        if operation == "reduce_scatter_tensor":
            if _reduce_op_name(arguments.get("op")) != "sum" or source.numel() != world * output.numel():
                return False
            self._count((f"torch.{operation}", "sircl", "sum"), "torch.distributed call carried on the session")
            if output.numel():
                chunk = self.reduce_scatter(source.contiguous().view(world, -1), 0)
                output.copy_(chunk.reshape(output.shape))
            return True
        # all_to_all_single with equal splits
        splits = [arguments.get("output_split_sizes"), arguments.get("input_split_sizes")]
        if any(sizes is not None and len(set(int(size) for size in sizes)) > 1 for sizes in splits):
            return False
        if not source.is_contiguous() or output.numel() != source.numel() or source.numel() % world:
            return False
        self._count((f"torch.{operation}", "sircl", "equal"), "torch.distributed call carried on the session")
        if source.numel():
            self.all_to_all_single(output.view(-1), source.view(-1))
        return True

    def _carry_all_reduce(self, tensor: torch.Tensor, name: str) -> None:
        if tensor.numel() == 0:
            return
        if name == "sum":
            tensor.copy_(self.all_reduce(tensor.contiguous()).view_as(tensor))
            return
        rows = self._gather_flat(tensor)
        reduced = rows.amax(dim=0) if name == "max" else rows.amin(dim=0)
        tensor.copy_(reduced.view_as(tensor))

    # -- reporting and lifecycle -----------------------------------------------------------

    def _record(self) -> dict[str, Any]:
        placement = self.placement
        topology = placement.topology
        session = self.session
        limits = self.limits
        return {
            "group": placement.name,
            "global_rank": placement.global_ranks[placement.rank],
            "rank": placement.rank,
            "world": placement.world,
            "global_ranks": list(placement.global_ranks),
            "layout": self.config.layout.describe(),
            "fabric": topology.fabric.describe() if topology is not None else "-",
            "positions": list(placement.positions),
            "nccl": placement.policy.value,
            "nccl_reason": placement.reason,
            "pynccl": "skipped" if placement.suppress_pynccl else "built",
            "session": ("none" if session is None else
                        f"shared:{self.shared_from}" if self.shared_from else "ring"),
            "session_shared_with": self.shared_from,
            "lanes": getattr(session, "lane_count", None),
            "hcas": list(getattr(session, "hca_names", ()) or ()),
            "capacity": limits.capacity if limits else None,
            "dispatch": limits.dispatch if limits else None,
            "oneshot_max": sessionapi.oneshot_limit(session) if session is not None else None,
            "gather": limits.gather if limits else None,
            "op_per_peer": self.per_peer[0],
            "op_per_peer_basis": self.per_peer[1],
            "large_piece": limits.large_piece if limits else None,
            "gather_piece": limits.gather_piece if limits else None,
            "mhc": self._mhc_mode(),
            "schedules": sessionapi.schedule_text(session) if session is not None else None,
            "chain_min": sessionapi.chain_min(session) if session is not None else None,
            "ring_min": sessionapi.ring_min(session) if session is not None else None,
            "links": sessionapi.link_text(session) if session is not None else None,
            "tuning": sessionapi.tuning_table(session) if session is not None else None,
            "fused_norm": (None if placement.kind != "tp" else
                           "on" if self.fused_norm is not None else "off"),
            "fused_norm_detail": self.fused_norm.describe() if self.fused_norm is not None else None,
            "column_gather": self.column_gather.describe() if session is not None else None,
            "p2p": self._p2p_text(),
            "p2p_detail": self._p2p_detail(),
            "wait": self._wait_text(),
            "vllm": receipt.package_dir("vllm"),
            "b12x": receipt.package_dir("b12x"),
            "reduce_dtypes": list(limits.reduce_dtypes) if limits else [],
            "relays": topology.max_relays() if topology is not None else None,
            "relay_factor": topology.relay_factor() if topology is not None else None,
            "four_rank_sessions": self._tp4,
            "state": "ready",
        }

    def attach_fused_norm(self, fused: Any) -> None:
        """Record bound fused all-reduce + RMSNorm kernels (:mod:`.norm_fusion`) and rewrite the receipt."""
        self.fused_norm = fused
        self.record = self._record()
        logger.info("%s", receipt.line(self.record))
        self.write_receipt()

    def _mhc_mode(self) -> str | None:
        """Prefill row ownership on this group: carried by SIRCL, by PyNccl, or off.

        The receipt field ``mhc`` covers GLM-5.3-Flash's mHC rows and Qwen3.8's
        hyper-connection rows (``VLLM_QWEN3_8_HC_PREFILL_MODE=shard``).
        """
        from . import mhc, qwen_hc

        if self.placement.kind != "tp":
            return None
        if not (mhc.requested(self._environ) or qwen_hc.requested(self._environ)):
            return "off"
        return "sircl" if self.placement.policy is NcclPolicy.NONE else "pynccl"

    def _p2p_text(self) -> str:
        """``peers=<n>,slots=<K>x<S>,windows=...``, ``shared:<group>``, or ``none`` (the detail says why)."""
        if self.p2p is None:
            return "none"
        if self.p2p_shared_from is not None:
            return f"shared:{self.p2p_shared_from}"
        stats = self.p2p.stats() if callable(getattr(self.p2p, "stats", None)) else {}
        text = f"peers={len(stats.get('channels', ()))}"
        if stats.get("slots"):
            text += f",slots={stats['slots']}x{stats.get('slot_bytes')}"
        if self.p2p_plan is not None:
            text += "," + self.p2p_plan.window_text()
        return text

    def _p2p_detail(self) -> dict[str, Any]:
        if self.p2p is None:
            return {"channels": None, "reason": self.p2p_reason}
        limits = self.p2p_limits()
        return {"channels": sorted(limits.peers), "relayed": sorted(limits.relayed),
                "unavailable": {str(peer): why for peer, why in limits.problems},
                "shared_from": self.p2p_shared_from,
                "plan_basis": self.p2p_plan.basis if self.p2p_plan is not None else None}

    def _wait_text(self) -> str | None:
        """The session's flag-wait regime and limit, ``startup:600s``, when the session has regimes."""
        regime = getattr(self.session, "wait_regime", None)
        limit = getattr(self.session, "wait_limit_s", None)
        if not isinstance(regime, str) or not isinstance(limit, (int, float)):
            return None
        return f"{regime}:{limit:g}s"

    def report(self, *, stats: bool = True) -> dict[str, Any]:
        """The receipt record; ``stats=False`` reuses the session statistics of the last report."""
        record = dict(self.record)
        record["wait"] = self._wait_text()
        record["vllm"] = receipt.package_dir("vllm")      # b12x loads with the model, after group setup
        record["b12x"] = receipt.package_dir("b12x")
        record["decisions"] = self.counters.snapshot()
        record["column_gather_detail"] = self.column_gather.snapshot() if self.session is not None else None
        if stats and self.session is not None and not self._closed:
            try:
                self._session_stats = self.session.stats()
            except Exception as exc:  # noqa: BLE001 - diagnostics only
                self._session_stats = {"error": f"{type(exc).__name__}: {exc}"}
        if self._session_stats is not None:
            record["session_stats"] = self._session_stats
        record["state"] = "closed" if self._closed else (
            "poisoned" if getattr(self.session, "poisoned", False) else "ready")
        return record

    def write_receipt(self, *, stats: bool = True) -> None:
        if self.config.receipt_dir:
            now = time.monotonic()
            self._receipt_dirty = False
            self._receipt_written = now
            if stats:
                self._stats_written = now
            self._receipt_version = self.counters.version
            receipt.write(self.report(stats=stats), self.config.receipt_dir)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self.write_receipt()
        with _LIVE_LOCK:
            _LIVE.discard(self)
        guard.unregister(self.device_group)
        self.fused_norm = None
        dcp, self.dcp = self.dcp, None
        if dcp is not None:
            with contextlib.suppress(Exception):
                dcp.close()
        channels, self.p2p = self.p2p, None
        if channels is not None and self.p2p_shared_from is None:
            with contextlib.suppress(Exception):
                channels.close()
        self.session = None             # a shared session stays open: its owner closes it


# torch.distributed calls of an output and an input tensor: (output argument, input argument).
_TENSOR_PAIRS = {"all_gather_into_tensor": ("output_tensor", "input_tensor"),
                 "reduce_scatter_tensor": ("output", "input"),
                 "all_to_all_single": ("output", "input")}


class _WorkGroup:
    """One work object for several transfers (a broadcast as sends): ``wait`` orders the caller's stream after
    all of them."""

    def __init__(self, works: Sequence[Any]) -> None:
        self._works = list(works)

    def wait(self, timeout: Any = None) -> bool:
        for work in self._works:
            work.wait()
        return True

    def is_completed(self) -> bool:
        return all(work.is_completed() for work in self._works)

    def is_success(self) -> bool:
        return all(work.is_success() for work in self._works)

    def exception(self) -> None:
        return None


class _CompletedWork:
    """The work object of a carried ``async_op`` call: already ordered on the caller's stream."""

    def wait(self, timeout: Any = None) -> bool:
        return True

    def is_completed(self) -> bool:
        return True

    def is_success(self) -> bool:
        return True

    def exception(self) -> None:
        return None


def _reduce_op_name(op: Any) -> str | None:
    """``sum``, ``max`` or ``min`` for a ``torch.distributed.ReduceOp``; None for any other op."""
    import torch.distributed as dist

    for name, member in (("sum", dist.ReduceOp.SUM), ("max", dist.ReduceOp.MAX), ("min", dist.ReduceOp.MIN)):
        try:
            if op == member:
                return name
        except Exception:  # noqa: BLE001 - an op type that does not compare
            return None
    return None
