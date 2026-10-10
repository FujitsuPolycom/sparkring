"""SIRCL ring-session collectives behind vLLM's RoCE all-reduce interface.

``SirclRingAllReduce`` provides what the RoCE path of the vLLM fork in image
``aba309e4610c`` (vllm 0.1.dev21553+gab86b7073) calls on
``CudaCommunicator.b12x_ar_comm``:

- ``__init__(group, device_group, device, *, global_ranks=None)``, called by
  ``CudaCommunicator.__init__``; ``group`` is the tensor-parallel group's CPU
  (gloo) group, over which the capability vote and the session's setup
  exchange run;
- ``disabled`` and ``backend_name`` (``_log_all_reduce_backend_selection``);
- ``should_custom_ar`` / ``custom_all_reduce`` (``CudaCommunicator.all_reduce``);
- ``should_all_gather`` / ``all_gather`` (``CudaCommunicator.all_gather``);
- ``capture(stream=...)`` (``GroupCoordinator.graph_capture``);
- ``check_health`` (the worker's post-step check, after every step's output
  reaches the host);
- ``close`` (``CudaCommunicator.destroy``);
- ``supports_fused_add_rms_norm`` (False, as for the RoCE path it replaces).

Dispatch, decided only from dtype, shape, contiguity and size so that every
rank routes the same collective the same way:

- all-reduce: contiguous BF16 CUDA tensors whose size is a multiple of 16
  bytes and at most ``SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES``; the registered
  capacity is ``SIRCL_ALLREDUCE_CAPACITY_BYTES``. Both default to 131072.
  Every other all-reduce goes to vLLM's next backend, unless SIRCL's
  communicator (:mod:`.communicator`) owns the group and carries it itself.
- all-gather: shards the session supports (contiguous, concatenated along
  dim 0 or the last dim) of at most ``SIRCL_ALLGATHER_MAX_BYTES`` (default
  131072); 0 leaves every all-gather to the next backend.
- Only BF16 launchers are compiled, in ``prepare`` at construction, before any
  CUDA graph capture; other dtypes are declined here, so no kernel compiles
  inside a serving step or a capture.

Failures stop every rank instead of falling back: a rank without a route map,
without an importable session package, without an integrated GPU and RDMA
device, or with different limits makes construction raise on all ranks (vote
over the CPU group); so does a failed ``prepare`` on any rank (second vote). A
single-node group is left to vLLM's other backends (``disabled``), as the
adapter this class replaces does.

Route map: ``SIRCL_PEER_ROUTES`` (``peer=device[/device],...``: every other
rank once, one or two distinct devices for each, the same lane count for every
peer), or the same map passed as ``peer_routes`` by SIRCL's communicator,
which derives it from the physical fabric and also passes the session's layout
(``layout``) so the session checks every lane against it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from . import fabric, groupops, sessionapi

logger = logging.getLogger("sircl.vllm.tp_slot")

REQUIRED_SIRCL_API_VERSION = 1
PACK_BYTES = 16
DEFAULT_LIMIT_BYTES = 131072
PREPARED_DTYPES = (torch.bfloat16,)
MAX_LANES = 2

ENV_PEER_ROUTES = "SIRCL_PEER_ROUTES"
ENV_CAPACITY = "SIRCL_ALLREDUCE_CAPACITY_BYTES"
ENV_DISPATCH = "SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES"
ENV_GATHER = "SIRCL_ALLGATHER_MAX_BYTES"
ENV_TOPOLOGY = "SIRCL_TOPOLOGY"
ENV_SPIN_LIMIT = "SIRCL_SPIN_LIMIT"
# All-gather tiers, read by the session itself; voted here so that ranks with
# different values fail together with a readable message.
ENV_GATHER_ALGORITHM = "SIRCL_ALLGATHER_ALGORITHM"
ENV_GATHER_ONESHOT = "SIRCL_ALLGATHER_ONESHOT_MAX_BYTES"
ENV_GATHER_SWING = "SIRCL_ALLGATHER_SWING_MAX_BYTES"
GATHER_ALGORITHMS = ("auto", "oneshot", "swing")


def attach_sircl_logging() -> None:
    """Send the ``sircl`` loggers' records through vLLM's handlers.

    vLLM configures handlers on its ``vllm`` logger only, so INFO records of
    other packages would otherwise be dropped.
    """
    source = logging.getLogger("vllm")
    target = logging.getLogger("sircl")
    if target.handlers or not source.handlers:
        return
    for handler in source.handlers:
        target.addHandler(handler)
    target.setLevel(logging.INFO)
    target.propagate = False


def env_bytes(name: str, default: int) -> int:
    """A non-negative byte count from the environment (plain integer)."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {raw}")
    return value


def gather_tiers() -> tuple[str, int | None, int | None]:
    """The all-gather tier settings (algorithm, one-shot limit, Swing limit); None keeps the session's default."""
    algorithm = os.environ.get(ENV_GATHER_ALGORITHM, "").strip() or "auto"
    if algorithm not in GATHER_ALGORITHMS:
        raise ValueError(f"{ENV_GATHER_ALGORITHM}={algorithm} must be one of {', '.join(GATHER_ALGORITHMS)}")
    oneshot_max = env_bytes(ENV_GATHER_ONESHOT, -1)
    swing_max = env_bytes(ENV_GATHER_SWING, -1)
    if swing_max > 0 and swing_max % PACK_BYTES:
        raise ValueError(f"{ENV_GATHER_SWING}={swing_max} must be a multiple of {PACK_BYTES}")
    return algorithm, None if oneshot_max < 0 else oneshot_max, None if swing_max < 0 else swing_max


@dataclass(frozen=True)
class SlotLimits:
    """The slot's size limits in bytes, read once from its session when the session exists, for SIRCL's own
    adapter (vLLM does not read them).

    ``dispatch_bytes`` is the largest all-reduce message the slot hands to the session (the session's dispatch
    limit), ``capacity_bytes`` the session's registered buffer capacity and ``gather_bytes`` the all-gather shard
    limit the settings vote agreed (0 routes no all-gather). A slot without a session reports
    :data:`NO_SLOT_LIMITS`, every limit zero.
    """

    dispatch_bytes: int = 0
    capacity_bytes: int = 0
    gather_bytes: int = 0

    @classmethod
    def of_session(cls, session, gather: int) -> "SlotLimits":
        return cls(dispatch_bytes=int(session.dispatch_limit_bytes), capacity_bytes=int(session.max_size),
                   gather_bytes=int(gather))


NO_SLOT_LIMITS = SlotLimits()


def _element_type(tensor: torch.Tensor) -> str:
    """A tensor's element type as the logs name it (``bfloat16``)."""
    return str(tensor.dtype).rpartition(".")[2]


def describe_gather_tiers(runtime) -> str:
    """How the runtime splits all-gathers by shard size (sessions without tiers: one-shot)."""
    swing = int(getattr(runtime, "gather_swing_max_bytes", 0) or 0)
    algorithm = getattr(runtime, "gather_algorithm", "oneshot")
    oneshot_max = int(getattr(runtime, "gather_oneshot_max_bytes", 0) or 0)
    if not swing or algorithm == "oneshot" or (algorithm == "auto" and swing <= oneshot_max):
        return "one-shot"
    if algorithm == "swing":
        return f"Swing up to {swing} bytes"
    return f"one-shot up to {oneshot_max} bytes, Swing up to {swing}"


def peer_routes(world_size: int, rank: int) -> dict[int, tuple[str, ...]]:
    """This rank's ``SIRCL_PEER_ROUTES``, checked against the group.

    The map must name every other rank of the group exactly once, with one or
    two distinct local RDMA devices per peer and the same width for every peer.
    Checking here lets the vote report a bad map on
    every rank instead of one rank failing before the session's setup exchange
    and leaving the others waiting.
    """
    raw = os.environ.get(ENV_PEER_ROUTES, "")
    if not raw:
        raise ValueError(f"{ENV_PEER_ROUTES} is not set")
    try:
        return fabric.parse_routes(raw, world=world_size, rank=rank, name=ENV_PEER_ROUTES)
    except fabric.FabricError as exc:
        raise ValueError(str(exc)) from None


def require_peer_routes() -> None:
    """Refuse to start without a well-formed route map (the slot plugin's ``register`` check).

    Without a map a ring session has no devices toward its peers. Only the
    syntax is checked here; the rank and world size are known
    when the communicator constructs the slot.
    """
    raw = os.environ.get(ENV_PEER_ROUTES, "")
    if not raw:
        raise RuntimeError(f"SIRCL slot: {ENV_PEER_ROUTES} is not set; launch every rank with its "
                           "route map (peer=device[/device],...)")
    entries = [entry for entry in raw.split(",") if entry.strip()]
    if not entries or any("=" not in entry for entry in entries):
        raise RuntimeError(f"SIRCL slot: {ENV_PEER_ROUTES}={raw!r} is not peer=device[/device],...")


class SirclRingAllReduce:
    """vLLM RoCE all-reduce slot implemented with the ring session ``AllReduce``."""

    backend_name = "SIRCL"

    def __init__(
        self,
        group,
        device_group,
        device: torch.device,
        *,
        global_ranks: Sequence[int] | None = None,
        peer_routes: Mapping[int, Sequence[str]] | None = None,
        layout: str | None = None,
        single_node: bool | None = None,
    ) -> None:
        self.disabled = True
        self.group = group
        self.device_group = device_group
        self.device = device
        self.rank = groupops.rank(group)
        self.world_size = groupops.size(group)
        self.global_ranks = tuple(
            int(rank) for rank in (global_ranks if global_ranks is not None else range(self.world_size))
        )
        self._runtime = None
        self._limits = NO_SLOT_LIMITS
        self._first_calls_logged: set[str] = set()
        self._routes = None if peer_routes is None else {
            int(peer): tuple(devices) for peer, devices in peer_routes.items()}
        if len(self.global_ranks) != self.world_size:
            raise ValueError("global ranks must match the process group")
        if device_group is None:
            raise RuntimeError("the SIRCL ring all-reduce needs a CUDA process group")

        if single_node is None:
            from vllm.distributed.parallel_state import in_the_same_node_as

            single_node = all(in_the_same_node_as(group, source_rank=0))
        if single_node:
            logger.info("Single-node tensor-parallel group: SIRCL ring session not used.")
            return

        reason, settings = self._local_settings()
        verdict = self._vote((reason, settings), compare=True)
        if verdict is not None:
            raise RuntimeError(f"SIRCL ring all-reduce unavailable: {verdict}")
        capacity, _dispatch, gather, topology, spin_limit, _tiers = settings

        oneshot = sessionapi.load()

        try:
            # Setup exchange over the CPU (gloo) group: a torch NCCL group
            # would add a communicator that vLLM otherwise never creates. The
            # session refuses differing protocol configurations on every rank
            # inside this call.
            if self._routes is None:
                runtime = oneshot.AllReduce.from_exchange_group(
                    exchange_group=group, device=device, max_size=capacity, max_gather_bytes=gather,
                )
            else:
                extra = {} if layout is None else {"layout": layout}
                runtime = oneshot.AllReduce(
                    exchange_group=group, device=device, max_size=capacity,
                    max_gather_bytes=gather, peer_routes=self._routes, **extra,
                )
        except Exception as exc:  # noqa: BLE001 - the session coordinated the ranks
            raise RuntimeError(f"SIRCL ring session setup failed on rank {self.rank}") from exc
        error = None
        try:
            runtime.prepare(PREPARED_DTYPES, padded_gather=gather > 0, **sessionapi.link_keywords(runtime))
        except Exception as exc:  # noqa: BLE001 - reported through the vote below
            error = f"{type(exc).__name__}: {exc}"
        verdict = self._vote(error, compare=False)
        if verdict is not None:
            runtime.close()
            raise RuntimeError(f"SIRCL ring session prepare failed: {verdict}")
        self._runtime = runtime
        self._limits = SlotLimits.of_session(runtime, gather)
        self.disabled = False
        if self.rank == 0:
            logger.info(
                "Using SIRCL ring collectives on the %d-rank tensor-parallel group: "
                "topology=%s, hcas=%s, stripes=%d, BF16 all-reduce <= %d bytes "
                "(capacity %d), all-gather shard <= %d bytes (%s), spin limit %s.",
                self.world_size, topology, ",".join(runtime.hca_names), runtime.lane_count,
                runtime.dispatch_limit_bytes, runtime.max_size, gather, describe_gather_tiers(runtime),
                spin_limit,
            )

    def _local_settings(self):
        """This rank's reason for not taking part (or None) and its settings."""
        try:
            if self._routes is None:
                peer_routes(self.world_size, self.rank)
            else:
                fabric.parse_routes(fabric.format_routes(self._routes), world=self.world_size,
                                    rank=self.rank, name="peer_routes")
            capacity = env_bytes(ENV_CAPACITY, DEFAULT_LIMIT_BYTES)
            dispatch = env_bytes(ENV_DISPATCH, capacity)
            gather = env_bytes(ENV_GATHER, DEFAULT_LIMIT_BYTES)
            tiers = gather_tiers()
        except (ValueError, fabric.FabricError) as exc:
            return str(exc), None
        if capacity < PACK_BYTES or capacity % PACK_BYTES:
            return f"{ENV_CAPACITY}={capacity} must be a positive multiple of {PACK_BYTES}", None
        if not PACK_BYTES <= dispatch <= capacity or dispatch % PACK_BYTES:
            return (f"{ENV_DISPATCH}={dispatch} must be a multiple of {PACK_BYTES} "
                    f"within the capacity {capacity}"), None
        if gather % PACK_BYTES:
            return f"{ENV_GATHER}={gather} must be a multiple of {PACK_BYTES}", None
        topology = os.environ.get(ENV_TOPOLOGY) or "direct"
        if topology != "direct":
            return f"{ENV_TOPOLOGY}={topology}; ring sessions have the single topology direct", None
        spin_limit = os.environ.get(ENV_SPIN_LIMIT, "")
        try:
            oneshot = sessionapi.load()
        except Exception as exc:  # noqa: BLE001 - missing package or a broken native build
            return f"the ring session package is not importable: {type(exc).__name__}: {exc}", None
        api = getattr(oneshot, "API_VERSION", None)
        if api != REQUIRED_SIRCL_API_VERSION:
            return f"ring session API version {api}, slot needs {REQUIRED_SIRCL_API_VERSION}", None
        if not oneshot.is_supported(self.device):
            return "needs an integrated GPU with an active RDMA device", None
        if tiers != ("auto", None, None) and not hasattr(oneshot.AllReduce, "select_gather_algorithm"):
            return (f"{ENV_GATHER_ALGORITHM}, {ENV_GATHER_ONESHOT} and {ENV_GATHER_SWING} need a "
                    "session with the Swing all-gather"), None
        return None, (capacity, dispatch, gather, topology, spin_limit, tiers)

    def _vote(self, local, *, compare: bool) -> str | None:
        """Gather every rank's ``local`` over the CPU group; the text of any failure.

        ``local`` is ``(reason, settings)`` with ``compare`` (settings must then
        match rank 0's) or an error text (None for success) without it.
        """
        return groupops.vote(self.group, local, compare=compare)

    # -- limits, for SIRCL's adapter -------------------------------------------------

    @property
    def size_limits(self) -> SlotLimits:
        """The slot's size limits (:class:`SlotLimits`), fixed once the session exists; every limit is 0 while
        the slot is disabled."""
        return NO_SLOT_LIMITS if self.disabled else self._limits

    @property
    def runtime(self):
        """The ring session, or None while disabled (SIRCL's communicator drives it directly)."""
        return self._runtime

    # -- interface used by vLLM ----------------------------------------------------

    def check_health(self) -> None:
        if not self.disabled and self._runtime is not None:
            self._runtime.check_health()

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        return (not self.disabled and inp.dtype in PREPARED_DTYPES
                and self._runtime.should_allreduce(inp))

    def _run_collective(self, name: str, collective: Callable[..., torch.Tensor], inp: torch.Tensor,
                        **options) -> torch.Tensor:
        """``collective(inp, **options)``, one of the session's collectives, whose output tensor is returned as
        it is. The slot's first call of each collective ``name`` logs one line with the message size in bytes
        and the element type (rank 0 at INFO, the other ranks at DEBUG); later calls log nothing."""
        if name not in self._first_calls_logged:
            self._first_calls_logged.add(name)
            logger.log(logging.INFO if self.rank == 0 else logging.DEBUG,
                       "SIRCL ring %s, first call on this slot: %d bytes of %s.",
                       name, int(inp.nbytes), _element_type(inp))
        return collective(inp, **options)

    def custom_all_reduce(self, inp: torch.Tensor) -> torch.Tensor | None:
        if not self.should_custom_ar(inp):
            return None
        return self._run_collective("all-reduce", self._runtime.all_reduce, inp)

    def should_all_gather(self, inp: torch.Tensor, dim: int) -> bool:
        return (not self.disabled and self._limits.gather_bytes > 0
                and self._runtime.should_all_gather(inp, dim))

    def all_gather(self, inp: torch.Tensor, dim: int) -> torch.Tensor:
        return self._run_collective("all-gather", self._runtime.all_gather, inp, dim=dim)

    def supports_fused_add_rms_norm(self) -> bool:
        return False

    @contextmanager
    def capture(self, stream: torch.cuda.Stream | None = None):
        if self.disabled or self._runtime is None:
            yield
            return
        with self._runtime.capture(stream=stream):
            yield

    def close(self) -> None:
        runtime, self._runtime = self._runtime, None
        self.disabled = True
        if runtime is not None:
            runtime.close()


__all__ = [
    "NO_SLOT_LIMITS",
    "SirclRingAllReduce",
    "SlotLimits",
    "attach_sircl_logging",
    "describe_gather_tiers",
    "env_bytes",
    "gather_tiers",
    "peer_routes",
    "require_peer_routes",
]
