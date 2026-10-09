"""CPU reference of the ring-session surface, with emulated ranks in one process.

:class:`EmulatedRingSession` implements the session methods the adapter uses
(:class:`.sessionapi.RingSession`) with the session's size, dispatch, capture
and fail-stop contracts on CPU tensors. The
ranks of one group are threads that share an :class:`EmulatedFabric`; every
collective is one rendezvous of all ranks. It exists for two purposes:

- the vLLM adapter's CPU tests drive the adapter's dispatch, chunking,
  composition, capture and fail-stop logic through it;
- it states, as executable code, the arithmetic the adapter relies on from a
  real session: all-reduce sums the ranks' contributions in rank order
  ``0 .. W-1`` in float32 and rounds once to the input dtype, so every rank
  gets identical bits; gathers and all-to-all copy bytes unchanged; a
  reduce-scatter chunk equals the same chunk of the all-reduce.

It does not model the wire, relays, timing or GPU memory ordering. Ranks that
issue different collectives at the same step fail with :class:`RankDivergence`
instead of hanging, which is how the tests detect rank-variant dispatch.
"""

from __future__ import annotations

import contextlib
import dataclasses
import threading
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import torch

PACK = 16
REDUCE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
DEFAULT_TIMEOUT = 20.0
# Flag-wait limits of the two regimes, the core's defaults. The emulation
# records the regime; its waits use the fabric's timeout.
STARTUP_WAIT_S = 600.0
SERVING_WAIT_S = 20.0
# The session package's chain and ring minimums of each collective when SIRCL_CHAIN_MIN_BYTES and
# SIRCL_RING_MIN_BYTES are unset (DEFAULT_CHAIN_MINS and DEFAULT_RING_MINS of oneshot/runtime.py).
DEFAULT_CHAIN_MINS = {"reduce": 8 << 20, "gather": 8 << 20, "scatter": 4 << 20}
DEFAULT_RING_MINS = {"reduce": 4 << 20, "gather": 8 << 20, "scatter": 4 << 20}


class EmulationError(RuntimeError):
    """A contract of the ring session was violated."""


class RankDivergence(EmulationError):
    """Ranks of one group issued different collectives at the same step."""


class SessionPoisoned(EmulationError):
    """A wait timed out or the session was closed; every later call fails."""


class CaptureCompileError(EmulationError):
    """A launcher that prepare() did not compile was needed inside a capture."""


@dataclasses.dataclass
class OpRecord:
    rank: int
    op: str
    nbytes: int
    capturing: bool


class EmulatedFabric:
    """The rendezvous all ranks of one emulated group share."""

    def __init__(self, world: int, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        if world < 2:
            raise ValueError("an emulated group needs at least two ranks")
        self.world = world
        self.timeout = timeout
        self._barrier = threading.Barrier(world)
        self._slots: list[Any] = [None] * world
        self._lock = threading.Lock()
        self.records: list[OpRecord] = []
        self.failed_ranks: set[int] = set()

    def exchange(self, rank: int, signature: tuple, payload: Any) -> list[Any]:
        """Deposit ``payload``; return every rank's payload once all ranks deposited theirs."""
        if rank in self.failed_ranks:
            # A failed rank never publishes; its peers' waits time out.
            raise SessionPoisoned(f"rank {rank} has failed and publishes nothing")
        self._slots[rank] = (signature, payload)
        try:
            self._barrier.wait(self.timeout)
            entries = list(self._slots)
            self._barrier.wait(self.timeout)
        except threading.BrokenBarrierError:
            raise SessionPoisoned(
                f"rank {rank} timed out waiting for its peers at {signature[0]}") from None
        signatures = [entry[0] for entry in entries]
        if any(sig != signatures[0] for sig in signatures):
            raise RankDivergence("ranks issued different collectives: " + "; ".join(
                f"rank {r}: {sig}" for r, sig in enumerate(signatures)))
        return [entry[1] for entry in entries]

    def fail(self, rank: int) -> None:
        """Make ``rank`` stop publishing; the others time out and poison (fault injection)."""
        self.failed_ranks.add(rank)
        self._barrier.abort()

    def count(self, op: str | None = None) -> int:
        with self._lock:
            return sum(1 for record in self.records if op is None or record.op == op)

    def record(self, record: OpRecord) -> None:
        with self._lock:
            self.records.append(record)


def _minimums(name: str, defaults: dict[str, int]) -> dict[str, int]:
    """Per-collective minimums as a session reads them: ``defaults``, or the variable's one size for all."""
    import os

    raw = os.environ.get(name, "").strip()
    return {collective: int(raw) for collective in defaults} if raw else dict(defaults)


def _bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


class EmulatedRingSession:
    """One rank's emulated ring session."""

    def __init__(
        self,
        fabric: EmulatedFabric,
        rank: int,
        *,
        max_size: int = 2 * 1024 * 1024,
        dispatch_limit_bytes: int | None = None,
        max_gather_bytes: int = 2 * 1024 * 1024,
        lane_count: int = 2,
        hca_names: Sequence[str] = ("rocep1s0f0", "roceP2p1s0f0"),
        scatter_available: bool = True,
        device: torch.device | str = "cpu",
        current_stream: Callable[[], object] = lambda: None,
        capturing: Callable[[], bool] | None = None,
        startup_wait_s: float = STARTUP_WAIT_S,
        serving_wait_s: float = SERVING_WAIT_S,
    ) -> None:
        dispatch = max_size if dispatch_limit_bytes is None else dispatch_limit_bytes
        for name, value in (("max_size", max_size), ("dispatch_limit_bytes", dispatch),
                            ("max_gather_bytes", max_gather_bytes)):
            if value < 0 or value % PACK:
                raise EmulationError(f"{name} must be a non-negative multiple of 16")
        if not PACK <= dispatch <= max_size:
            raise EmulationError("the dispatch ceiling must lie in [16, capacity]")
        self.fabric = fabric
        self.rank = rank
        self.world_size = fabric.world
        self.max_size = max_size
        self.dispatch_limit_bytes = dispatch
        self.max_gather_bytes = max_gather_bytes
        self.lane_count = lane_count
        self.hca_names = tuple(hca_names)
        self.scatter_available = scatter_available
        self.device = torch.device(device)
        self._current_stream = current_stream
        self._capturing_override = capturing
        self._capturing = False
        self._capture_stream: object | None = None
        self._prepared: set[torch.dtype] = set()
        self._prepared_gather = False
        self._prepared_scatter = False
        self._closed = False
        self._failure: str | None = None
        self._step = 0
        self.startup_wait_s = float(startup_wait_s)
        self.serving_wait_s = float(serving_wait_s)
        self.wait_regime = "startup"
        # The schedules a real session reads from the environment (recorded; every op runs as one exchange).
        import os

        self.large_schedule = os.environ.get("SIRCL_LARGE_SCHEDULE", "auto")
        self.gather_schedule = os.environ.get("SIRCL_GATHER_SCHEDULE", "auto")
        self.scatter_schedule = os.environ.get("SIRCL_SCATTER_SCHEDULE", "pieces")
        # The one-shot limit of the auto all-reduce, the chain and ring minimums of each collective (one
        # size for all three when the variable is set), and the link collectives' slot and pieces with the
        # session's slot rule, read the same way (recorded only).
        self.oneshot_max_bytes = int(os.environ.get("SIRCL_ONESHOT_MAX_BYTES", 131072))
        self.large_blocks = int(os.environ.get("SIRCL_LARGE_BLOCKS", 32))
        self._chain_mins = _minimums("SIRCL_CHAIN_MIN_BYTES", DEFAULT_CHAIN_MINS)
        self._ring_mins = _minimums("SIRCL_RING_MIN_BYTES", DEFAULT_RING_MINS)
        configured = {name: int(os.environ.get(name, 0) or 0)
                      for name in ("SIRCL_LINK_CHUNK_BYTES", "SIRCL_GATHER_LINK_CHUNK_BYTES",
                                   "SIRCL_SCATTER_LINK_CHUNK_BYTES", "SIRCL_REDUCE_LINK_CHUNK_BYTES")}
        wanted = -(-max(configured.values()) // 4096) * 4096
        self.link_slot_bytes = int(os.environ.get("SIRCL_LINK_SLOT_BYTES",
                                                  wanted if 512 << 10 < wanted <= 1 << 20 else 512 << 10))
        self.link_chunk_bytes = configured["SIRCL_LINK_CHUNK_BYTES"] or min(512 << 10, self.link_slot_bytes)
        self._link_chunks = {collective: configured[f"SIRCL_{collective.upper()}_LINK_CHUNK_BYTES"]
                             for collective in ("gather", "scatter", "reduce")
                             if configured[f"SIRCL_{collective.upper()}_LINK_CHUNK_BYTES"]}
        self.prepared_links: list[bool] = []

    # -- state ------------------------------------------------------------------

    @property
    def poisoned(self) -> bool:
        return self._failure is not None

    def _require_open(self) -> None:
        if self._closed:
            raise SessionPoisoned("the session is closed")
        if self._failure is not None:
            raise SessionPoisoned(self._failure)

    def check_health(self) -> None:
        self._require_open()

    # -- flag-wait regimes (recorded, not timed) ---------------------------------------

    @property
    def wait_limit_s(self) -> float:
        return self.startup_wait_s if self.wait_regime == "startup" else self.serving_wait_s

    def enter_startup(self) -> None:
        self.wait_regime = "startup"

    def enter_serving(self) -> None:
        self.wait_regime = "serving"

    @contextlib.contextmanager
    def startup(self) -> Iterator["EmulatedRingSession"]:
        previous = self.wait_regime
        self.wait_regime = "startup"
        try:
            yield self
        finally:
            self.wait_regime = previous

    def capturing(self) -> bool:
        if self._capturing_override is not None:
            return bool(self._capturing_override())
        return self._capturing

    @contextlib.contextmanager
    def capture(self, stream: object | None = None, *, channel_id: object = None) -> Iterator["EmulatedRingSession"]:
        self._capturing = True
        self._capture_stream = stream if stream is not None else self._current_stream()
        try:
            yield self
        finally:
            self._capturing = False
            self._capture_stream = None

    def prepare(self, dtypes: Sequence[torch.dtype] = (torch.bfloat16,), *, padded_gather: bool = False,
                algorithms: Sequence[str] | None = None, scatter: bool = False, links: bool = False) -> None:
        if self.capturing():
            raise CaptureCompileError("prepare() is refused inside a CUDA graph capture")
        if scatter and not self.scatter_available:
            raise EmulationError("scatter collectives are unavailable on this session")
        self.prepared_links.append(bool(links))
        self._prepared.update(dtypes)
        self._prepared_gather = True
        self._prepared_scatter = self._prepared_scatter or scatter

    # -- eligibility --------------------------------------------------------------

    def _plain(self, inp: torch.Tensor) -> bool:
        return inp.device == self.device and inp.is_contiguous() and inp.dim() > 0 and not inp.is_sparse

    def _reduce_eligible(self, inp: torch.Tensor, limit: int) -> bool:
        nbytes = _bytes(inp)
        return (self._plain(inp) and inp.dtype in REDUCE_DTYPES and 0 < nbytes <= limit
                and nbytes % PACK == 0)

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        self._require_open()
        return self._reduce_eligible(inp, self.dispatch_limit_bytes)

    def should_all_gather(self, inp: torch.Tensor, dim: int = -1) -> bool:
        self._require_open()
        if not self._plain(inp) or inp.is_complex() or inp.dtype == torch.bool:
            return False
        if not -inp.dim() <= dim < inp.dim():
            return False
        dim = dim % inp.dim()
        return dim in (0, inp.dim() - 1) and 0 < _bytes(inp) <= self.max_gather_bytes

    def _scatter_geometry(self, inp: torch.Tensor, chunk_bytes: int | None,
                          stride_bytes: int | None) -> tuple[int, int] | None:
        nbytes = _bytes(inp)
        world = self.world_size
        if chunk_bytes is None:
            if nbytes == 0 or nbytes % (world * PACK):
                return None
            chunk = nbytes // world
            stride = chunk
        else:
            chunk = int(chunk_bytes)
            stride = chunk if stride_bytes is None else int(stride_bytes)
            if chunk <= 0 or chunk % PACK or stride % PACK or stride < chunk:
                return None
            if (world - 1) * stride + chunk > nbytes:
                return None
        if world * chunk > self.max_size:
            return None
        return chunk, stride

    def should_reduce_scatter(self, inp: torch.Tensor, *, chunk_bytes: int | None = None,
                              src_stride_bytes: int | None = None) -> bool:
        self._require_open()
        return (self.scatter_available and self._plain(inp) and inp.dtype in REDUCE_DTYPES
                and self._scatter_geometry(inp, chunk_bytes, src_stride_bytes) is not None)

    def should_all_to_all(self, inp: torch.Tensor, *, chunk_bytes: int | None = None,
                          src_stride_bytes: int | None = None) -> bool:
        self._require_open()
        return (self.scatter_available and self._plain(inp) and not inp.is_complex()
                and self._scatter_geometry(inp, chunk_bytes, src_stride_bytes) is not None)

    # -- collectives ----------------------------------------------------------------

    def _launch(self, op: str, key: object, nbytes: int, payload: Any) -> list[Any]:
        self._require_open()
        capturing = self.capturing()
        if capturing:
            stream = self._current_stream()
            if self._capture_stream is not None and stream is not self._capture_stream:
                raise EmulationError("collectives of one capture must use one stream")
            if op == "all_reduce" and key not in self._prepared:
                raise CaptureCompileError(f"all-reduce launcher for {key} was not prepared")
            if op == "all_gather" and not self._prepared_gather:
                raise CaptureCompileError("all-gather launcher was not prepared")
            if op in ("reduce_scatter", "all_to_all") and not self._prepared_scatter:
                raise CaptureCompileError(f"{op} launcher was not prepared")
        self.fabric.record(OpRecord(self.rank, op, nbytes, capturing))
        self._step += 1
        try:
            return self.fabric.exchange(self.rank, (op, nbytes, str(key), self._step), payload)
        except SessionPoisoned as exc:
            self._failure = (f"SIRCL collective on group rank {self.rank} timed out; the session "
                             f"is poisoned and rank data is untrustworthy ({exc})")
            raise SessionPoisoned(self._failure) from None

    def all_reduce(self, inp: torch.Tensor, *, out: torch.Tensor | None = None,
                   stream: object = None, **ignored: Any) -> torch.Tensor:
        self._require_open()
        if not self._reduce_eligible(inp, self.max_size):
            raise EmulationError("input is not eligible for the ring all-reduce")
        if out is not None and (out.shape != inp.shape or out.dtype != inp.dtype
                                or not out.is_contiguous()):
            raise EmulationError("out must be a contiguous tensor like the input")
        contributions = self._launch("all_reduce", inp.dtype, _bytes(inp), inp.detach().clone())
        total = torch.zeros(inp.shape, dtype=torch.float32)
        for tensor in contributions:          # rank order 0 .. W-1, one rounding
            total += tensor.to(torch.float32)
        result = total.to(inp.dtype)
        if out is None:
            return result
        out.copy_(result)
        return out

    def all_gather(self, inp: torch.Tensor, *, dim: int = -1, out: torch.Tensor | None = None,
                   stream: object = None) -> torch.Tensor:
        if not self.should_all_gather(inp, dim):
            raise EmulationError("input is not eligible for the ring all-gather")
        shards = self._launch("all_gather", "bytes", _bytes(inp), inp.detach().clone())
        result = torch.cat(shards, dim=dim % inp.dim())
        if out is None:
            return result
        if out.shape != result.shape or out.dtype != result.dtype or not out.is_contiguous():
            raise EmulationError("out must be a contiguous tensor of the gathered shape")
        out.copy_(result)
        return out

    @staticmethod
    def _chunk(flat: torch.Tensor, index: int, chunk: int, stride: int) -> torch.Tensor:
        return flat[index * stride: index * stride + chunk]

    def reduce_scatter(self, inp: torch.Tensor, *, out: torch.Tensor | None = None,
                       stream: object = None, chunk_bytes: int | None = None,
                       src_stride_bytes: int | None = None) -> torch.Tensor:
        if not self.should_reduce_scatter(inp, chunk_bytes=chunk_bytes,
                                          src_stride_bytes=src_stride_bytes):
            raise EmulationError("input is not eligible for the ring reduce-scatter")
        geometry = self._scatter_geometry(inp, chunk_bytes, src_stride_bytes)
        assert geometry is not None
        chunk, stride = geometry
        flat = inp.detach().reshape(-1).view(torch.uint8)
        payload = [self._chunk(flat, j, chunk, stride).clone() for j in range(self.world_size)]
        sent = self._launch("reduce_scatter", inp.dtype, self.world_size * chunk, payload)
        total = None
        for source in sent:                    # rank order, float32, one rounding
            part = source[self.rank].view(inp.dtype).to(torch.float32)
            total = part if total is None else total + part
        result = total.to(inp.dtype)
        if out is None:
            return result
        if _bytes(out) != chunk or out.dtype != inp.dtype or not out.is_contiguous():
            raise EmulationError("out must hold one chunk of the input's dtype")
        out.view(-1).copy_(result)
        return out

    def all_to_all(self, inp: torch.Tensor, out: torch.Tensor, *, stream: object = None,
                   chunk_bytes: int | None = None, src_stride_bytes: int | None = None,
                   dst_stride_bytes: int | None = None) -> torch.Tensor:
        if not self.should_all_to_all(inp, chunk_bytes=chunk_bytes, src_stride_bytes=src_stride_bytes):
            raise EmulationError("input is not eligible for the ring all-to-all")
        geometry = self._scatter_geometry(inp, chunk_bytes, src_stride_bytes)
        assert geometry is not None
        chunk, stride = geometry
        dst_stride = chunk if dst_stride_bytes is None else int(dst_stride_bytes)
        if dst_stride % PACK or dst_stride < chunk or not out.is_contiguous():
            raise EmulationError("invalid all-to-all destination stride")
        if (self.world_size - 1) * dst_stride + chunk > _bytes(out):
            raise EmulationError("the all-to-all output is too small for its chunks")
        flat = inp.detach().reshape(-1).view(torch.uint8)
        payload = [self._chunk(flat, j, chunk, stride).clone() for j in range(self.world_size)]
        sent = self._launch("all_to_all", "bytes", self.world_size * chunk, payload)
        target = out.view(-1).view(torch.uint8)
        for source, chunks in enumerate(sent):
            target[source * dst_stride: source * dst_stride + chunk].copy_(chunks[self.rank])
        return out

    # -- lifecycle -------------------------------------------------------------------

    def link_chunk_for(self, collective: str) -> int:
        """The link piece of ``collective`` (``gather``, ``scatter`` or ``reduce``), as a session reports it."""
        return self._link_chunks.get(collective, self.link_chunk_bytes)

    def chain_min_for(self, collective: str) -> int:
        """The chain minimum of ``collective`` (``reduce``, ``gather`` or ``scatter``), as a session reports it."""
        return self._chain_mins[collective]

    def ring_min_for(self, collective: str) -> int:
        """The ring minimum of ``collective``, as a session reports it."""
        return self._ring_mins[collective]

    @property
    def chain_min_bytes(self) -> int | None:
        values = set(self._chain_mins.values())
        return values.pop() if len(values) == 1 else None

    @property
    def ring_min_bytes(self) -> int | None:
        values = set(self._ring_mins.values())
        return values.pop() if len(values) == 1 else None

    def stats(self) -> dict[str, Any]:
        return {
            "rank": self.rank, "world_size": self.world_size, "max_size": self.max_size,
            "dispatch_limit_bytes": self.dispatch_limit_bytes,
            "max_gather_bytes": self.max_gather_bytes, "lane_count": self.lane_count,
            "hca_names": list(self.hca_names),
            "prepared": sorted(_dtype_name(dtype) for dtype in self._prepared),
            "ops": self.fabric.count(), "poisoned": self.poisoned,
            "wait_regime": self.wait_regime, "wait_limit_s": self.wait_limit_s,
            "startup_wait_s": self.startup_wait_s, "serving_wait_s": self.serving_wait_s,
            "large_schedule": self.large_schedule, "gather_schedule": self.gather_schedule,
            "scatter_schedule": self.scatter_schedule, "oneshot_max_bytes": self.oneshot_max_bytes,
            "large_blocks": self.large_blocks,
            "chain_min_bytes": self.chain_min_bytes, "ring_min_bytes": self.ring_min_bytes,
            "chain_mins": dict(self._chain_mins), "ring_mins": dict(self._ring_mins),
            "link_slot_bytes": self.link_slot_bytes,
            "link_chunks": {collective: self.link_chunk_for(collective) for collective in ("gather", "scatter",
                                                                                          "reduce")},
            "link_chunk_bytes": self.link_chunk_bytes,
        }

    def close(self, *, abort: bool = False) -> None:
        self._closed = True


def run_ranks(world: int, body: Callable[[int], Any], *, timeout: float = 60.0) -> list[Any]:
    """Run ``body(rank)`` for every rank in its own thread; re-raise the first failure."""
    results: list[Any] = [None] * world
    errors: list[BaseException | None] = [None] * world

    def target(rank: int) -> None:
        try:
            results[rank] = body(rank)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors[rank] = exc

    threads = [threading.Thread(target=target, args=(rank,), daemon=True) for rank in range(world)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError("an emulated rank did not finish")
    for error in errors:
        if error is not None:
            raise error
    return results


def reference_sum(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Fixed rank-order float32 sum rounded once to the inputs' dtype."""
    total = torch.zeros(tensors[0].shape, dtype=torch.float32)
    for tensor in tensors:
        total += tensor.to(torch.float32)
    return total.to(tensors[0].dtype)


class EmulatedGroup:
    """One rank's view of an emulated CPU process group (see :mod:`.groupops`).

    All ranks of one group share ``objects`` (the CPU object exchange) and
    ``data`` (the fabric the group's ring session uses).
    """

    def __init__(self, objects: EmulatedFabric, data: EmulatedFabric, rank: int,
                 global_ranks: Sequence[int]) -> None:
        self.objects = objects
        self.data = data
        self.rank = rank
        self.global_ranks = tuple(global_ranks)
        self._steps = 0

    def sircl_rank(self) -> int:
        return self.rank

    def sircl_size(self) -> int:
        return self.objects.world

    def sircl_ranks(self) -> list[int]:
        return list(self.global_ranks)

    def sircl_all_gather_object(self, value: Any) -> list[Any]:
        self._steps += 1
        return self.objects.exchange(self.rank, ("all_gather_object", self._steps), value)


def emulated_groups(global_ranks: Sequence[int], *, timeout: float = DEFAULT_TIMEOUT) -> list[EmulatedGroup]:
    """One :class:`EmulatedGroup` per rank of a group of ``global_ranks``."""
    world = len(global_ranks)
    objects = EmulatedFabric(world, timeout=timeout)
    data = EmulatedFabric(world, timeout=timeout)
    return [EmulatedGroup(objects, data, rank, global_ranks) for rank in range(world)]


def session_module(name: str = "sircl_emulated", **defaults: Any):
    """A session package (``API_VERSION``, ``is_supported``, ``AllReduce``) of emulated sessions.

    ``AllReduce(exchange_group=EmulatedGroup, ...)`` reads, like a real session,
    ``SIRCL_PEER_ROUTES`` when no ``peer_routes`` are given (and refuses to
    start without a map) and ``SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES``
    for the dispatch ceiling. Every constructed session is kept in the module's
    ``created`` list.
    """
    import os
    import types

    from . import fabric as fabric_module

    module = types.ModuleType(name)
    module.API_VERSION = 1
    module.SUPPORTED_DTYPES = REDUCE_DTYPES
    module.created = []
    module.is_supported = lambda device=None: True

    class AllReduce(EmulatedRingSession):
        def __init__(self, *, exchange_group: EmulatedGroup, device: Any = "cpu",
                     max_size: int = 2 * 1024 * 1024, max_gather_bytes: int = 2 * 1024 * 1024,
                     peer_routes: Any = None, algorithm: str | None = None, **ignored: Any) -> None:
            options = dict(defaults)
            raw_dispatch = os.environ.get("SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES", "").strip()
            dispatch = int(raw_dispatch) if raw_dispatch else max_size
            dispatch = min(dispatch, max_size)
            world = exchange_group.sircl_size()
            rank = exchange_group.sircl_rank()
            if peer_routes is None:
                text = os.environ.get("SIRCL_PEER_ROUTES", "")
                if not text:
                    raise EmulationError("a ring session needs a route map")
                peer_routes = fabric_module.parse_routes(text, world=world, rank=rank)
            self.peer_routes = {int(peer): tuple(devices) for peer, devices in dict(peer_routes).items()}
            self.algorithm = algorithm or "auto"
            self.exchange_group = exchange_group
            lanes = {len(devices) for devices in self.peer_routes.values()}
            hcas = tuple(dict.fromkeys(d for devices in self.peer_routes.values() for d in devices))
            super().__init__(exchange_group.data, rank, max_size=max_size,
                             dispatch_limit_bytes=dispatch, max_gather_bytes=max_gather_bytes,
                             lane_count=lanes.pop() if len(lanes) == 1 else 0,
                             hca_names=hcas, device=device, **options)
            # Setup is collective: every rank must construct before any uses it.
            exchange_group.sircl_all_gather_object(("setup", self.max_size, self.max_gather_bytes,
                                                    self.dispatch_limit_bytes))
            module.created.append(self)

        @classmethod
        def from_exchange_group(cls, *, exchange_group: EmulatedGroup, device: Any,
                                max_size: int = 2 * 1024 * 1024, eager_buffer_bytes: int | None = None,
                                max_gather_bytes: int = 2 * 1024 * 1024, **ignored: Any) -> "AllReduce":
            return cls(exchange_group=exchange_group, device=device,
                       max_size=max(max_size, eager_buffer_bytes or 0),
                       max_gather_bytes=max_gather_bytes)

    module.AllReduce = AllReduce
    return module
