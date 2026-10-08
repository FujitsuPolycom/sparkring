"""SIRCL ring session: collectives of one group over RDMA on DGX Spark.

A session is one rank's membership in one group's collective instance (2 to
16 ranks; Spark fabrics hold 2 to 8): a pinned host arena that the GB10 GPU
addresses at its host pointer and every opened RDMA device registers with
plain ``ibv_reg_mr`` (GB10 offers no GPU-memory registration), a device
command ring in that arena, one reliable-connected queue pair per (device,
peer) that carries a lane, and a native progress thread that posts the RDMA
writes (``_roce_proxy.c``). Every collective is one kernel launch that stages
its input, rings the progress thread, waits for every peer's lane flags and
reduces or gathers in place.

Construction is collective over ``exchange_group`` (a CPU process group,
gloo in vLLM): each rank checks its route map, opens its devices, builds its
setup record and connection record, and the ranks exchange both with one
all-gather. Setup fails on every rank, with one message naming each failing
rank and reason, when any rank failed locally, when a shared setting differs
from rank 0's, when lane counts differ, or when lanes do not pair. Then every
rank connects its lanes, the ranks exchange verdicts, every rank proves every
lane with one small write, and the progress threads start.

Contract:

- one collective in flight per session, in launch order across streams (an
  event orders streams outside a capture; one stream per capture inside);
- all-reduce of float16, bfloat16 or float32, contiguous, sizes a multiple of
  16 bytes up to the capacity ``max_size`` (the dispatch predicate accepts up
  to ``dispatch_limit_bytes``); sums in float32 in rank order 0..W-1 with one
  rounding, so every rank stores identical bits, with the one-shot and the
  two-shot algorithm alike;
- all-gather of any plain dtype along dimension 0 or the last dimension, shards
  up to ``max_gather_bytes``; unaligned shapes take a padded path;
- ``all_reduce_large`` and ``all_gather_large``: messages of any size. On a
  group whose ranks form a chain of cable neighbors, an all-reduce from
  ``chain_min_for("reduce")`` on is one chain op (``_chain_cute.py``: half the message
  reduces along the chain each way and the results travel back, pipelined in
  chunks between neighbors only); its bits are identical on every rank and
  can differ in the last place from the one-shot result, because each hop
  rounds. Other all-reduces run as ops of at most ``large_piece_bytes``, and
  all-gathers as ops of at most ``gather_piece_bytes``
  (:mod:`sparkring_sircl.pieces`), with the one-shot bits; capturable after
  ``prepare``;
- forward windows: with a layout, every lane whose path crosses relays posts
  its stripes in chunks and keeps at most its window of bytes unacknowledged,
  so the relays' hairpin queues stay within ``RELAY_QUEUE_SHARE`` of their
  size at every message size;
- wait limits: a kernel waits for a peer's flag at most the session's wait
  limit, in seconds of the GPU's clock. A session starts in the startup
  regime (``startup_wait_s``, default 600 s: peers may lag while they compile
  kernels, warm up or capture graphs); :meth:`enter_serving` switches to the
  serving regime (``serving_wait_s``, default 20 s), :meth:`enter_startup`
  and the :meth:`startup` context switch back. The limit lives in the command
  ring, so it applies to eager calls and graph replays alike from the next
  launch on. ``prepare`` measures how many flag polls the GPU makes per
  second (``stats()["poll_rate_per_s"]``);
- fail-stop: a flag wait that exceeds the wait limit poisons the session;
  :meth:`check_health` raises from then on, and later launches do nothing. A
  collective is never retried on another backend.
- posting order: in every network phase the progress thread posts the
  lanes peer after peer in the rank's posting order (``post_order`` or
  ``SIRCL_POST_ORDER``: ``rank``, ``ring-farthest``, ``farthest`` or an
  explicit peer list; :mod:`sparkring_sircl.posting`), resolved once the
  route map and the layout are known and handed to the native layer as an
  explicit peer list. Unset, it is ``farthest`` (most relays first) for a
  session with a layout and ``rank`` without one. The order decides when
  each write starts, not what it carries, so ranks need not agree on it;
- one-shot limit: the auto all-reduce runs one-shot up to
  ``oneshot_max_bytes``. ``SIRCL_ONESHOT_MAX_BYTES`` sets it; unset, a session
  with a layout and the two-shot all-reduce takes the latency model's limit
  for its layout, lane count and posting order
  (:func:`sparkring_sircl.latency_model.oneshot_limit`, at most 131,072
  bytes and the capacity: 28,672 on the ring of eight farthest first, 73,728
  on a path of four), and a session without a layout 131,072. Every rank
  derives it from agreed inputs, and the setup agreement compares it
  (``stats()['oneshot_max_source']`` names where it came from).

This build carries the one-shot and two-shot all-reduce, the all-gather and
the scatter collectives (reduce-scatter and all-to-all; host side in
``_scatter_ops``): messages of any size in ops of at most
``large_piece_bytes``, the reduce-scatter with the one-shot bits.
``scatter_available`` needs multi-phase posting, and ``prepare(scatter=True)``
raises without it. The Swing all-reduce needs a kernel this build does not
have: ``_available`` reports Swing False, and asking for it explicitly is a
setup error. Phase tracing (``SIRCL_TRACE``) and the direct posting mode
(``SIRCL_POST_MODE=direct``) are unsupported.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .. import callprofile as callprofile_mod
from .. import latency_model
from .. import pieces
from .. import posting as posting_mod
from .. import protocol as proto
from .. import scatter_plan as _scatter_plan
from ..agreement import agreement_failures
from .. import roce_gid
from .. import routes as routes_mod
from .. import tuning as tuning_mod
from . import _allgather_cute, _chain_cute, _links_cute, _oneshot_cute, _scatter_ops, _timed_wait, _twoshot_cute
from ._proxy import ABI_VERSION, Layout, Proxy
from ._proxy import chain_layout as _proxy_chain_layout
from ._proxy import link_layout as _proxy_link_layout

logger = logging.getLogger("sircl")

API_VERSION = 1
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
SUPPORTED_WORLD_SIZES = tuple(range(2, 17))
DEFAULT_MAX_SIZE = 2 * 1024 * 1024
DEFAULT_MAX_GATHER_BYTES = 16 * 1024 * 1024
DEFAULT_THREADS = 512
DEFAULT_BLOCKS = 8
DEFAULT_GID_INDEX = 3
DEFAULT_SPIN_LIMIT = 20_000_000
DEFAULT_ONESHOT_MAX_BYTES = 131072
DEFAULT_LANE_CHECK_MS = 2000
DEFAULT_LARGE_BLOCKS = 32
# SIRCL_FLAG_POLLERS: "one-block" (block 0 polls the peers' flags in host memory and hands arrival to the
# other blocks in device memory) or "every-block" (every block polls them).
FLAG_POLLERS = ("one-block", "every-block")
DEFAULT_FLAG_POLLERS = "one-block"
# Op size of all_reduce_large when the capacity is smaller; the arena's slots hold it.
# On a path of four, 4 MiB two-shot ops reduced 64 MiB in 8.1 ms against 9.1 ms
# for 2 MiB and 10.8 ms for 1 MiB.
DEFAULT_LARGE_PIECE_BYTES = 4 << 20
LARGE_SCHEDULES = ("auto", "chain", "ring", "pieces")
DEFAULT_CHAIN_SLOTS = 4
DEFAULT_CHAIN_SLOT_BYTES = 1 << 20
DEFAULT_CHAIN_CHUNK_BYTES = 512 << 10
DEFAULT_CHAIN_BLOCKS = 4
DEFAULT_CHAIN_UNROLL = 4
MAX_EVENT_TRACE = 1 << 24
CLOCK_PROBE_ROUNDS = 16
# The collectives whose chain and ring schedules start at a size of their own, and what the size counts:
# the all-reduce's message, the all-gather's output, the reduce-scatter's input.
MIN_COLLECTIVES = ("reduce", "gather", "scatter")
# Smallest collective auto runs as a chain op, and a ring schedule as a ring op: the sizes from which the
# chain and the ring beat two-shot pieces, tiles and scatter ops on Sparks 0-3 (ring harness
# configurations path4-crossover and path4-large). SIRCL_CHAIN_MIN_BYTES and
# SIRCL_RING_MIN_BYTES set one size for all three.
DEFAULT_CHAIN_MINS = {"reduce": 8 << 20, "gather": 8 << 20, "scatter": 4 << 20}
DEFAULT_RING_MINS = {"reduce": 4 << 20, "gather": 8 << 20, "scatter": 4 << 20}
# Chain links (the chain all-gather of all_gather_large and the chain reduce-scatter): schedules and
# ring geometry.
GATHER_SCHEDULES = ("auto", "chain", "ring", "pieces")
SCATTER_SCHEDULES = ("auto", "chain", "ring", "pieces")
# The fewest link slots of a session without SIRCL_LINK_SLOTS or a tuning table's; a session of W ranks
# takes protocol.default_link_slots(W) (2 W, at least this).
DEFAULT_LINK_SLOTS = 8
DEFAULT_LINK_SLOT_BYTES = 512 << 10
DEFAULT_LINK_CHUNK_BYTES = 512 << 10
# The link collectives with a piece of their own, and the variables that set them (unset: the session's
# link piece, SIRCL_LINK_CHUNK_BYTES).
LINK_COLLECTIVES = {"gather": "SIRCL_GATHER_LINK_CHUNK_BYTES", "scatter": "SIRCL_SCATTER_LINK_CHUNK_BYTES",
                    "reduce": "SIRCL_REDUCE_LINK_CHUNK_BYTES"}
# Without SIRCL_LINK_SLOT_BYTES the link slot grows to the largest configured piece up to this size (link
# area: 4 links x slots x slot bytes, twice).
MAX_AUTO_LINK_SLOT_BYTES = 1 << 20
# The ring reduce-scatter's stagger without SIRCL_RING_STAGGER, and the ring all-gather's without
# SIRCL_RING_GATHER_STAGGER: 1 when the link slots hold it.
DEFAULT_RING_STAGGER = 1
DEFAULT_RING_GATHER_STAGGER = 1
DEFAULT_LINK_BLOCKS = 4
DEFAULT_LINK_UNROLL = 4
DEFAULT_STARTUP_WAIT_S = 600.0
DEFAULT_SERVING_WAIT_S = 20.0
# The command ring holds the wait limit as 32-bit microseconds.
MAX_WAIT_S = 0xFFFFFFFF / 1e6
WAIT_REGIMES = ("startup", "serving")
POLL_RATE_PROBE_POLLS = 200_000
DEFAULT_FORWARD_WINDOW_BYTES = routes_mod.DEFAULT_FORWARD_WINDOW
DEFAULT_FORWARD_CHUNK_BYTES = routes_mod.DEFAULT_FORWARD_CHUNK
DEFAULT_HAIRPIN_QUEUE_BYTES = routes_mod.DEFAULT_HAIRPIN_QUEUE
# Chunks a forward window may hold (the native layer's limit: its credit minus four).
MAX_FORWARD_CHUNKS = 60
ALGORITHMS = proto.ALGORITHMS
ALGORITHM_CHOICES = proto.ALGORITHM_CHOICES
LARGE_ALGORITHMS = proto.LARGE_ALGORITHMS
SCATTER_MODES = ("reduce", "copy")
MAX_LANES = proto.MAX_LANES
MAX_DEVICES = proto.MAX_DEVICES
PACK_BYTES = proto.PACK_BYTES
# Kernels this build carries; the others make their algorithm unavailable.
_BUILT_ALGORITHMS = frozenset({"oneshot", "twoshot", "scatter"})
_DTYPE_NAMES = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}
# A context that does nothing, reused by every call that needs no device, stream or tuning context.
_NO_CONTEXT = contextlib.nullcontext()


def _whole_call(op: str):
    """Profile a large-message method from entry to exit (``SIRCL_CALL_PROFILE``; captures excluded)."""
    def wrap(method):
        @functools.wraps(method)
        def run(self, inp, *args, **kwargs):
            profile = self._profile
            if profile is None or torch.cuda.is_current_stream_capturing():
                return method(self, inp, *args, **kwargs)
            call = profile.begin(op)
            result = method(self, inp, *args, **kwargs)
            profile.end(call, "large", inp.numel() * inp.element_size())
            return result
        return run
    return wrap
_ENV_LOCK = threading.Lock()


# -- environment ------------------------------------------------------------------


def _env_text(*names: str, default: str) -> str:
    """The first non-empty value among the environment variables ``names``, else ``default``."""
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and raw.strip():
            return raw.strip()
    return default


def _env_int(*names: str, default: int) -> int:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and raw.strip():
            try:
                return int(raw.strip(), 0)
            except ValueError:
                raise ValueError(f"{name}={raw} is not an integer") from None
    return default


def _env_minimums(name: str, defaults: Mapping[str, int]) -> dict[str, int]:
    """Per-collective minimums: ``defaults``, or the value of ``name`` for every collective."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return dict(defaults)
    value = _env_int(name, default=0)
    if value < 0:
        raise ValueError(f"{name}={value} must not be negative")
    return {collective: value for collective in defaults}


def _explicit_gid_index() -> int | None:
    for name in ("SIRCL_GID_INDEX", "NCCL_IB_GID_INDEX"):
        raw = os.environ.get(name)
        if raw is not None and raw.strip():
            value = _env_int(name, default=DEFAULT_GID_INDEX)
            if not 0 <= value <= 255:
                raise ValueError(f"{name}={raw} is outside 0-255")
            return value
    return None


def default_gid_index() -> int:
    """``SIRCL_GID_INDEX``, else ``NCCL_IB_GID_INDEX``, else 3."""
    explicit = _explicit_gid_index()
    return DEFAULT_GID_INDEX if explicit is None else explicit


def _device_list(text: str) -> tuple[str, ...]:
    """Device names of ``SIRCL_DEVICES`` or ``NCCL_IB_HCA`` (``^``, ``=`` and ``:port`` stripped)."""
    items = []
    for item in text.split(","):
        item = item.strip().lstrip("=^")
        if item:
            items.append(item.split(":")[0])
    return tuple(items)


def _active_devices(gid_index: int | None = None, root: Path = Path("/sys/class/infiniband")) -> tuple[str, ...]:
    found = []
    for device in sorted(root.glob("*")):
        try:
            if "ACTIVE" not in (device / "ports" / "1" / "state").read_text():
                continue
            if gid_index is not None:
                gid = (device / "ports" / "1" / "gids" / str(gid_index)).read_text()
                if gid.strip().replace(":", "").strip("0") == "":
                    continue
        except OSError:
            continue
        found.append(device.name)
    return tuple(found)


def discover_hcas(gid_index: Optional[int] = None) -> tuple[str, ...]:
    """Devices to open when no route map names them.

    ``SIRCL_DEVICES``, else NCCL's ``NCCL_IB_HCA``, else every active device
    with a populated GID at ``gid_index`` (default :func:`default_gid_index`);
    at most four.
    """
    explicit = _device_list(_env_text("SIRCL_DEVICES", "NCCL_IB_HCA", default=""))
    if explicit:
        return explicit[:MAX_DEVICES]
    index = default_gid_index() if gid_index is None else int(gid_index)
    return _active_devices(index)[:MAX_DEVICES]


def is_supported(device: torch.device | int | str | None = None) -> bool:
    """True on an integrated GPU (pinned host memory read in place) with an active RDMA device."""
    if not torch.cuda.is_available():
        return False
    index = torch.cuda.current_device() if device is None else torch.device(device).index
    props = torch.cuda.get_device_properties(index if index is not None else 0)
    if not getattr(props, "is_integrated", False):
        return False
    return len(discover_hcas()) > 0


def _normalize_device(device: torch.device | int | str) -> torch.device:
    if isinstance(device, int):
        device = torch.device("cuda", device)
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("a SIRCL session runs on a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _capture_id(stream: torch.cuda.Stream) -> int:
    """CUDA's id of the capture ``stream`` is in, 0 outside a capture."""
    from cuda.bindings import runtime as cudart

    info = cudart.cudaStreamGetCaptureInfo(stream.cuda_stream)
    if info[0] != cudart.cudaError_t.cudaSuccess:
        return 0
    if info[1] != cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
        return 0
    return int(info[2])


def _device_pointer(host_ptr: int) -> int:
    from cuda.bindings import runtime as cudart

    err, ptr = cudart.cudaHostGetDevicePointer(host_ptr, 0)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaHostGetDevicePointer failed: {err}")
    return int(ptr)


def _grid_blocks(size_packs: int, threads: int, max_blocks: int,
                 packs_per_thread: int = proto.DEFAULT_PACKS_PER_THREAD) -> int:
    """Power-of-two grid of about ``packs_per_thread`` packs per thread, capped at ``max_blocks``."""
    return proto.grid_blocks(size_packs, threads, max_blocks, packs_per_thread)


def _exchange(local: object, group: ProcessGroup) -> list[object]:
    gathered: list[object] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, local, group=group)
    return gathered


@contextmanager
def _environment(**overrides: str | None):
    """Set environment variables the native layer reads at creation and start."""
    with _ENV_LOCK:
        saved = {name: os.environ.get(name) for name in overrides}
        try:
            for name, value in overrides.items():
                if value is not None:
                    os.environ[name] = value
            yield
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


class RoceOneshotAllReduce:
    """A SIRCL ring session; see the module docstring."""

    algorithm_choices = ALGORITHM_CHOICES

    def __init__(
        self,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_size: int = DEFAULT_MAX_SIZE,
        max_gather_bytes: int = DEFAULT_MAX_GATHER_BYTES,
        hca_names: Optional[Sequence[str]] = None,
        peer_routes: Optional[Mapping[int, Sequence[str]]] = None,
        gid_index: Optional[int] = None,
        threads: Optional[int] = None,
        blocks: Optional[int] = None,
        algorithm: Optional[str] = None,
        post_order: Optional[str] = None,
        dispatch_limit_bytes: Optional[int] = None,
        progress_cpu: Optional[str] = None,
        spin_limit: Optional[int] = None,
        layout: routes_mod.Layout | str | None = None,
        lane_check_ms: int = DEFAULT_LANE_CHECK_MS,
        startup_wait_s: Optional[float] = None,
        serving_wait_s: Optional[float] = None,
    ) -> None:
        self.device = _normalize_device(device)
        self.rank = dist.get_rank(group=exchange_group)
        self.world_size = dist.get_world_size(group=exchange_group)
        self._group = exchange_group
        self._closed = False
        self._lock = threading.Lock()
        self._proxy: Optional[Proxy] = None
        self._launchers: dict[tuple, Callable[..., None]] = {}
        self._gather_launcher: Optional[Callable[..., None]] = None
        self._gather_buffers: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._align_buffers: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._stream_event = torch.cuda.Event()
        self._last_stream: Optional[torch.cuda.Stream] = None
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._capture_id = 0
        self.topology = "direct"
        self._layout_identity_object: routes_mod.Layout | None = None
        self._lane_check: dict[str, Any] = {"result": "not run"}
        self._forward_table: list[list[int]] = []
        self.relay_safe_bytes: Optional[int] = None
        self.wait_regime = "startup"
        self.poll_rate_per_s: Optional[float] = None
        self.chain_order: Optional[tuple[int, ...]] = None
        self.chain_index: Optional[int] = None
        self.chain_available = False
        self._chain_offset = 0
        self.link_available = False
        self.ring_available = False
        self.ring_window_bytes = 0
        self.ring_problem: Optional[str] = None
        self._link_offset = 0
        self.startup_wait_s = DEFAULT_STARTUP_WAIT_S
        self.serving_wait_s = DEFAULT_SERVING_WAIT_S
        error: Optional[str] = None
        try:
            self._configure(max_size, max_gather_bytes, hca_names, peer_routes, gid_index, threads,
                            blocks, algorithm, post_order, dispatch_limit_bytes, progress_cpu,
                            spin_limit, layout, startup_wait_s, serving_wait_s)
        except Exception as exc:  # noqa: BLE001 - reported to every rank below
            error = f"{type(exc).__name__}: {exc}"
        blob = b""
        if error is None:
            try:
                self._allocate()
                with _environment(SIRCL_POST_ORDER=self._post_order_text):
                    self._proxy = Proxy(
                        world_size=self.world_size, rank=self.rank, hca_names=self.hca_names,
                        peer_lane_devices=self._lane_devices, lane_count=self.lane_count,
                        gid_indices=self.gid_indices, region_ptr=self._region.data_ptr(),
                        region_bytes=self._region_bytes, slot_bytes=self._slot_bytes,
                    )
                blob = self._proxy.local_blob()
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
        statuses = _exchange((error, blob, self._setup_record() if error is None else {}), exchange_group)
        failures = agreement_failures(statuses, self._layout_identity_object)
        if failures:
            self.close()
            raise RuntimeError("SIRCL session setup failed: " + "; ".join(failures))
        self._verdict("queue-pair connection", lambda: self._proxy.connect([s[1] for s in statuses]))
        route_maps = [{int(peer): tuple(devices) for peer, devices in status[2]["route_map"].items()}
                      for status in statuses]

        def check_and_start() -> None:
            self._plan_relays(route_maps)
            self._plan_chain(route_maps)
            started = time.perf_counter()
            self._proxy.lane_check(lane_check_ms)
            self._lane_check = {"result": "passed", "lanes": self.lane_count * (self.world_size - 1),
                                "seconds": round(time.perf_counter() - started, 4)}
            if any(any(row) for row in self._forward_table):
                self._proxy.set_forward(self._forward_table, self.forward_chunk_bytes)
            if self.event_trace:
                self._proxy.set_trace(self.event_trace)
            with _environment(SIRCL_PROGRESS_CPU=self._progress_cpu):
                self._proxy.start()

        self._verdict("lane check", check_and_start)
        if self.rank == 0:
            logger.info(
                "SIRCL session ready: %d ranks, %d lane(s) per peer, devices %s, capacity %d, "
                "dispatch limit %d, gather capacity %d, traffic class %d, windowed lanes %d, "
                "large pieces %d, gather pieces %d",
                self.world_size, self.lane_count, ",".join(self.hca_names), self.max_size,
                self.dispatch_limit_bytes, self.max_gather_bytes, self._proxy.traffic_class,
                sum(1 for row in self._forward_table for window in row if window),
                self.large_piece_bytes, self.gather_piece_bytes,
            )

    # -- construction ---------------------------------------------------------------

    def _configure(self, max_size, max_gather_bytes, hca_names, peer_routes, gid_index, threads,
                   blocks, algorithm, post_order, dispatch_limit_bytes, progress_cpu, spin_limit,
                   layout, startup_wait_s=None, serving_wait_s=None) -> None:
        if self.world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(f"SIRCL sessions have 2 to 16 ranks, got {self.world_size}")
        topology = _env_text("SIRCL_TOPOLOGY", default="direct")
        if topology != "direct":
            raise ValueError(f"SIRCL_TOPOLOGY={topology} is unsupported: ring sessions have one "
                             "transport topology, direct (relays are configured on the fabric)")
        if int(_env_int("SIRCL_TRACE", default=0)) > 0:
            raise ValueError("SIRCL_TRACE (phase tracing) is unsupported by this SIRCL build")
        self.event_trace = _env_int("SIRCL_EVENT_TRACE", default=0)
        if not 0 <= self.event_trace <= MAX_EVENT_TRACE:
            raise ValueError(f"SIRCL_EVENT_TRACE must be 0 to {MAX_EVENT_TRACE} records")
        self.max_size = int(max_size)
        if self.max_size < PACK_BYTES or self.max_size % PACK_BYTES:
            raise ValueError(f"max_size {self.max_size} must be a positive multiple of 16 bytes")
        limit = dispatch_limit_bytes
        if limit is None:
            limit = _env_int("SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES", default=self.max_size)
        self.dispatch_limit_bytes = int(limit)
        if not PACK_BYTES <= self.dispatch_limit_bytes <= self.max_size or self.dispatch_limit_bytes % PACK_BYTES:
            raise ValueError(f"SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES {self.dispatch_limit_bytes} must be a "
                             f"multiple of 16 between 16 and the capacity {self.max_size}")
        self.max_gather_bytes = int(max_gather_bytes)
        if self.max_gather_bytes < 0 or self.max_gather_bytes % PACK_BYTES:
            raise ValueError(f"max_gather_bytes {self.max_gather_bytes} must be a multiple of 16 bytes")
        threads = _env_int("SIRCL_THREADS", default=DEFAULT_THREADS) if threads is None else int(threads)
        blocks = _env_int("SIRCL_BLOCKS", default=DEFAULT_BLOCKS) if blocks is None else int(blocks)
        if threads % 32 or not 32 <= threads <= 1024:
            raise ValueError("SIRCL_THREADS must be a multiple of 32 between 32 and 1024")
        if blocks < 1 or blocks & (blocks - 1):
            raise ValueError("SIRCL_BLOCKS must be a positive power of two")
        packs_per_thread = _env_int("SIRCL_PACKS_PER_THREAD", default=proto.DEFAULT_PACKS_PER_THREAD)
        if not 1 <= packs_per_thread <= 64:
            raise ValueError("SIRCL_PACKS_PER_THREAD must be between 1 and 64")
        large_blocks = _env_int("SIRCL_LARGE_BLOCKS", default=DEFAULT_LARGE_BLOCKS)
        if large_blocks < 1 or large_blocks & (large_blocks - 1) or large_blocks > 1024:
            raise ValueError("SIRCL_LARGE_BLOCKS must be a power of two between 1 and 1024")
        self._threads, self._blocks, self._packs_per_thread = threads, blocks, packs_per_thread
        self._large_blocks = large_blocks
        # Who polls the peers' flags in host memory in the one-shot, two-shot and all-gather kernels: block 0
        # alone, handing arrival to the other blocks in device memory, or every block of the grid.
        pollers = os.environ.get("SIRCL_FLAG_POLLERS", "").strip() or DEFAULT_FLAG_POLLERS
        if pollers not in FLAG_POLLERS:
            raise ValueError(f"SIRCL_FLAG_POLLERS={pollers!r}: one of {', '.join(FLAG_POLLERS)}")
        self.flag_pollers = pollers
        self._one_block_polls = pollers == "one-block"
        # Arrival counters exist for every grid class up to the largest cap set_large_blocks accepts.
        self._counter_layout = proto.CounterLayout(max(blocks, large_blocks, DEFAULT_LARGE_BLOCKS))
        self._counter_classes = self._counter_layout.classes
        self.spin_limit = int(_env_int("SIRCL_SPIN_LIMIT", default=DEFAULT_SPIN_LIMIT)
                              if spin_limit is None else spin_limit)
        if not 1 <= self.spin_limit < 1 << 32:
            raise ValueError("SIRCL_SPIN_LIMIT must be a positive 32-bit poll count")
        for name, given, default in (("SIRCL_STARTUP_WAIT_S", startup_wait_s, DEFAULT_STARTUP_WAIT_S),
                                     ("SIRCL_SERVING_WAIT_S", serving_wait_s, DEFAULT_SERVING_WAIT_S)):
            if given is None:
                raw = os.environ.get(name, "").strip()
                try:
                    given = float(raw) if raw else default
                except ValueError:
                    raise ValueError(f"{name}={raw} is not a number of seconds") from None
            if not 1e-6 <= float(given) <= MAX_WAIT_S:
                raise ValueError(f"{name} {given} must be between 1e-6 and {MAX_WAIT_S:.0f} seconds")
            if name == "SIRCL_STARTUP_WAIT_S":
                self.startup_wait_s = float(given)
            else:
                self.serving_wait_s = float(given)
        # Empty: the layout's default, chosen once the layout is known.
        self._post_order_name = (post_order if post_order is not None
                                 else os.environ.get("SIRCL_POST_ORDER", "")).strip()
        if self._post_order_name not in ("", "farthest"):
            proto.post_order(self.rank, self.world_size, self._post_order_name)
        self._progress_cpu = progress_cpu if progress_cpu is not None else os.environ.get("SIRCL_PROGRESS_CPU")
        post_mode = _env_text("SIRCL_POST_MODE", default="verbs")
        if post_mode != "verbs":
            raise ValueError(f"SIRCL_POST_MODE={post_mode} is unsupported by this SIRCL build; use verbs")
        self.post_mode = post_mode

        # Route map, layout and devices.
        if isinstance(layout, str):
            layout = routes_mod.Layout.parse(layout)
        if layout is None and os.environ.get("SIRCL_LAYOUT", "").strip():
            layout = routes_mod.Layout.parse(os.environ["SIRCL_LAYOUT"])
        self._layout_identity_object = layout
        if peer_routes is None:
            text = os.environ.get("SIRCL_PEER_ROUTES", "")
            if not text.strip():
                raise ValueError("a session on a switchless fabric needs a route map: set "
                                 "SIRCL_PEER_ROUTES or pass peer_routes")
            peer_routes = routes_mod.parse_peer_routes(text)
        routes = {int(peer): tuple(devices) for peer, devices in peer_routes.items()}
        max_relays = _env_int("SIRCL_MAX_RELAYS", default=routes_mod.DEFAULT_MAX_RELAYS)
        active = _active_devices()
        self.lane_count = routes_mod.validate_route_map(
            self.rank, self.world_size, routes, layout=layout, max_relays=max_relays,
            available_devices=active or None,
        )
        named = list(dict.fromkeys(d for _, devices in sorted(routes.items()) for d in devices))
        if hca_names is None:
            listed = _device_list(os.environ.get("SIRCL_DEVICES", ""))
            hca_names = listed or tuple(named)
        self.hca_names = tuple(hca_names)
        missing = [device for device in named if device not in self.hca_names]
        if missing or not 1 <= len(self.hca_names) <= MAX_DEVICES:
            raise ValueError(f"opened devices {self.hca_names} must be 1 to {MAX_DEVICES} devices "
                             f"including every device of the route map (missing {missing})")
        self.peer_routes = tuple(routes.get(peer, ()) for peer in range(self.world_size))
        if not self._post_order_name:
            # Farthest first where the layout gives every lane's relays, else rank order.
            self._post_order_name = "farthest" if layout is not None else "rank"
        # Resolved here, where the layout and the route map are known; the native layer receives
        # the order as an explicit peer list.
        self.post_order_peers = posting_mod.resolve(self._post_order_name, self.rank, self.world_size,
                                                    layout, routes)
        self._post_order_text = ",".join(str(peer) for peer in self.post_order_peers)
        # Measured choices (SIRCL_TUNING_TABLE: one or more table paths, comma-separated). The session takes the
        # table whose key matches its own facts (a process holds sessions of several group shapes); its hash
        # joins the setup agreement, so every rank decides from the same table. No match: the rules choose.
        self._tuning: Optional[tuning_mod.Table] = None
        self._tuning_unmatched: dict[str, list[str]] = {}
        self._tuning_counts: dict[str, int] = {}
        self._tuning_unusable: dict[str, int] = {}
        self._tuning_lock = threading.RLock()
        self._tuning_suspended = 0           # untuned() depth: the table decides nothing while above 0
        # Host timestamps of eager calls (SIRCL_CALL_PROFILE): None when off; _profiled is the call whose
        # launch is being made.
        keep = _env_int("SIRCL_CALL_PROFILE", default=0)
        self._profile: Optional[callprofile_mod.CallProfile] = (
            callprofile_mod.CallProfile(keep, path=_env_text("SIRCL_CALL_PROFILE_FILE", default=""),
                                        rank=self.rank, gpu=_env_int("SIRCL_CALL_PROFILE_GPU", default=0) == 1)
            if keep > 0 else None)
        self._profiled: Optional[callprofile_mod.Call] = None
        table_paths = tuning_mod.table_paths(os.environ.get("SIRCL_TUNING_TABLE", ""))
        if table_paths:
            if layout is None:
                raise ValueError("SIRCL_TUNING_TABLE needs the session's layout (pass layout= or set SIRCL_LAYOUT)")
            self._tuning, self._tuning_unmatched = tuning_mod.select_table(table_paths, self.tuning_facts())
        # The settings the table's choices ran under and need (tuning.SETTINGS): each one the environment
        # leaves unset takes the table's value, so the choices run as measured. They size the arena below
        # and join the setup agreement like any other setting.
        self._table_settings: dict[str, int] = dict(self._tuning.settings) if self._tuning is not None else {}
        self._lane_devices = tuple(tuple(self.hca_names.index(d) for d in self.peer_routes[peer])
                                   for peer in range(self.world_size))
        explicit = gid_index if gid_index is not None else _explicit_gid_index()
        self.gid_index = DEFAULT_GID_INDEX if explicit is None else int(explicit)
        if explicit is not None:
            self.gid_indices = tuple(int(explicit) for _ in self.hca_names)
        else:
            self.gid_indices = tuple(roce_gid.resolve_device_gid_index(name) for name in self.hca_names)

        # The large-message op size sizes the slots too, so all_reduce_large runs
        # ops of at least DEFAULT_LARGE_PIECE_BYTES whatever the capacity.
        capacity_piece = self.max_size // PACK_BYTES * PACK_BYTES
        piece = _env_int("SIRCL_LARGE_PIECE_BYTES", default=self._table_settings.get(
            "SIRCL_LARGE_PIECE_BYTES", max(DEFAULT_LARGE_PIECE_BYTES, capacity_piece)))
        if piece < PACK_BYTES or piece % PACK_BYTES or piece > proto.OP_BYTES_MASK:
            raise ValueError(f"SIRCL_LARGE_PIECE_BYTES {piece} must be a positive multiple of 16 below 2^30")
        self._configured_large_piece = piece
        self.large_piece_bytes = piece

        # Algorithms: one-shot always; two-shot and Swing need multi-phase posting.
        slot_bytes = proto.slot_bytes_for(max(self.max_size, piece), self.max_gather_bytes)
        self.multi_phase = proto.multi_phase_available(self.lane_count, slot_bytes)
        power_of_two = self.world_size & (self.world_size - 1) == 0
        self._available = {
            "oneshot": True,
            "twoshot": self.multi_phase and "twoshot" in _BUILT_ALGORITHMS,
            "swing": self.multi_phase and power_of_two and "swing" in _BUILT_ALGORITHMS,
        }
        self.scatter_available = self.multi_phase and "scatter" in _BUILT_ALGORITHMS
        self.algorithm = algorithm or _env_text("SIRCL_ALLREDUCE_ALGORITHM", default="auto")
        if self.algorithm not in ALGORITHM_CHOICES:
            raise ValueError(f"SIRCL_ALLREDUCE_ALGORITHM {self.algorithm!r} is not one of {ALGORITHM_CHOICES}")
        self.large_algorithm = _env_text("SIRCL_LARGE_ALGORITHM", default="twoshot")
        if self.large_algorithm not in LARGE_ALGORITHMS:
            raise ValueError(f"SIRCL_LARGE_ALGORITHM must be one of {LARGE_ALGORITHMS}")
        explicit_algorithms = {self.algorithm} - {"auto", "oneshot"}
        if os.environ.get("SIRCL_LARGE_ALGORITHM"):
            explicit_algorithms.add(self.large_algorithm)
        for name in sorted(explicit_algorithms):
            if not self._available[name]:
                raise ValueError(f"the {name} all-reduce is unavailable: this build has no {name} kernel")
        self.oneshot_max_bytes, self.oneshot_max_source = self._oneshot_limit(layout)
        self.swing_above_bytes = _env_int("SIRCL_SWING_ABOVE_BYTES", default=0)
        if self.oneshot_max_bytes < 0 or self.swing_above_bytes < 0:
            raise ValueError("SIRCL_ONESHOT_MAX_BYTES and SIRCL_SWING_ABOVE_BYTES must not be negative")
        if self.swing_above_bytes and not self._available["swing"]:
            raise ValueError("SIRCL_SWING_ABOVE_BYTES needs the Swing all-reduce, which is unavailable")
        self._slot_bytes = slot_bytes

        # Large messages and forward windows of relayed lanes.
        self.forward_window_bytes = _env_int("SIRCL_FORWARD_WINDOW_BYTES", default=DEFAULT_FORWARD_WINDOW_BYTES)
        self.forward_chunk_bytes = _env_int("SIRCL_FORWARD_CHUNK_BYTES", default=DEFAULT_FORWARD_CHUNK_BYTES)
        self.hairpin_queue_bytes = _env_int("SIRCL_HAIRPIN_QUEUE_BYTES", default=DEFAULT_HAIRPIN_QUEUE_BYTES)
        if self.forward_window_bytes < 0:
            raise ValueError("SIRCL_FORWARD_WINDOW_BYTES must not be negative (0 turns forward windows off)")
        if self.forward_window_bytes:
            chunk = self.forward_chunk_bytes
            if chunk < PACK_BYTES or chunk % PACK_BYTES or chunk > self.forward_window_bytes:
                raise ValueError(f"SIRCL_FORWARD_CHUNK_BYTES {chunk} must be a positive multiple of 16 "
                                 f"within the forward window {self.forward_window_bytes}")
            if self.forward_window_bytes // chunk > MAX_FORWARD_CHUNKS:
                raise ValueError(f"a forward window holds at most {MAX_FORWARD_CHUNKS} chunks: "
                                 f"SIRCL_FORWARD_WINDOW_BYTES {self.forward_window_bytes} with chunks of {chunk}")
        if self.hairpin_queue_bytes < 4096:
            raise ValueError("SIRCL_HAIRPIN_QUEUE_BYTES must be at least 4096")
        self.gather_piece_bytes = self.max_gather_bytes // PACK_BYTES * PACK_BYTES

        # Chain schedule of all_reduce_large: settings, and whether the arena gets a chain area.
        self.large_schedule = _env_text("SIRCL_LARGE_SCHEDULE", default="auto")
        if self.large_schedule not in LARGE_SCHEDULES:
            raise ValueError(f"SIRCL_LARGE_SCHEDULE {self.large_schedule!r} is not one of {LARGE_SCHEDULES}")
        self.chain_slots = _env_int("SIRCL_CHAIN_SLOTS", default=DEFAULT_CHAIN_SLOTS)
        self.chain_slot_bytes = _env_int("SIRCL_CHAIN_SLOT_BYTES", default=self._table_settings.get(
            "SIRCL_CHAIN_SLOT_BYTES", DEFAULT_CHAIN_SLOT_BYTES))
        self.chain_blocks = _env_int("SIRCL_CHAIN_BLOCKS", default=DEFAULT_CHAIN_BLOCKS)
        self.chain_unroll = _env_int("SIRCL_CHAIN_UNROLL", default=DEFAULT_CHAIN_UNROLL)
        self._chain_mins = _env_minimums("SIRCL_CHAIN_MIN_BYTES", DEFAULT_CHAIN_MINS)
        self._ring_mins = _env_minimums("SIRCL_RING_MIN_BYTES", DEFAULT_RING_MINS)
        chunk = _env_int("SIRCL_CHAIN_CHUNK_BYTES", default=min(DEFAULT_CHAIN_CHUNK_BYTES, self.chain_slot_bytes))
        proto.ChainLayout(self.lane_count, self.chain_slots, self.chain_slot_bytes)   # validates the geometry
        if self.chain_blocks < 1 or self.chain_blocks > 64:
            raise ValueError("SIRCL_CHAIN_BLOCKS must be 1 to 64")
        if not 1 <= self.chain_unroll <= 8:
            raise ValueError("SIRCL_CHAIN_UNROLL must be 1 to 8")
        self._check_chain_chunk(chunk)
        self.chain_chunk_bytes = chunk
        fabric_full = layout is not None and sorted(layout.positions) == list(layout.fabric.positions)
        # The chain kernel runs its send-slot wait in a warp apart from the lane waits.
        chain_threads = self._threads >= 64
        self._chain_region = self.large_schedule != "pieces" and fabric_full and chain_threads
        if self.large_schedule == "chain" and not fabric_full:
            raise ValueError("SIRCL_LARGE_SCHEDULE=chain needs a layout whose every Spark hosts a rank of the "
                             "session (pass layout=)")
        if self.large_schedule == "chain" and not chain_threads:
            raise ValueError(f"SIRCL_LARGE_SCHEDULE=chain needs at least 64 threads per block (SIRCL_THREADS), "
                             f"got {self._threads}")

        # Chain links of all_gather_large: settings, and whether the arena gets a link area.
        self.gather_schedule = _env_text("SIRCL_GATHER_SCHEDULE", default="auto")
        if self.gather_schedule not in GATHER_SCHEDULES:
            raise ValueError(f"SIRCL_GATHER_SCHEDULE {self.gather_schedule!r} is not one of {GATHER_SCHEDULES}")
        self.link_slots = _env_int("SIRCL_LINK_SLOTS", default=self._table_settings.get(
            "SIRCL_LINK_SLOTS", proto.default_link_slots(self.world_size)))
        configured = {name: _env_int(name, default=0)
                      for name in ("SIRCL_LINK_CHUNK_BYTES", *LINK_COLLECTIVES.values())}
        # The slot holds the largest configured piece: rounded up to 4096 bytes, at least the default and,
        # unless SIRCL_LINK_SLOT_BYTES sets it, at most MAX_AUTO_LINK_SLOT_BYTES; and the tuning table's slot,
        # which holds its chosen pieces.
        wanted = -(-max(configured.values()) // 4096) * 4096
        auto_slot = wanted if DEFAULT_LINK_SLOT_BYTES < wanted <= MAX_AUTO_LINK_SLOT_BYTES else DEFAULT_LINK_SLOT_BYTES
        auto_slot = max(auto_slot, self._table_settings.get("SIRCL_LINK_SLOT_BYTES", 0))
        self.link_slot_bytes = _env_int("SIRCL_LINK_SLOT_BYTES", default=auto_slot)
        self.link_blocks = _env_int("SIRCL_LINK_BLOCKS", default=DEFAULT_LINK_BLOCKS)
        self.link_unroll = _env_int("SIRCL_LINK_UNROLL", default=DEFAULT_LINK_UNROLL)
        link_chunk = configured["SIRCL_LINK_CHUNK_BYTES"] or min(DEFAULT_LINK_CHUNK_BYTES, self.link_slot_bytes)
        proto.LinkLayout(self.lane_count, self.link_slots, self.link_slot_bytes)   # validates the geometry
        if not 1 <= self.link_blocks <= 64 or not 1 <= self.link_unroll <= 8:
            raise ValueError("SIRCL_LINK_BLOCKS must be 1 to 64 and SIRCL_LINK_UNROLL 1 to 8")
        self._check_link_chunk(link_chunk, "SIRCL_LINK_CHUNK_BYTES")
        self.link_chunk_bytes = link_chunk
        self.ring_stagger = self._stagger_setting("SIRCL_RING_STAGGER", DEFAULT_RING_STAGGER)
        self.ring_gather_stagger = self._stagger_setting("SIRCL_RING_GATHER_STAGGER", DEFAULT_RING_GATHER_STAGGER)
        self._link_chunks: dict[str, int] = {}
        for collective, name in LINK_COLLECTIVES.items():
            if configured[name]:
                self._check_link_chunk(configured[name], name)
                self._link_chunks[collective] = configured[name]
        self.scatter_schedule = _env_text("SIRCL_SCATTER_SCHEDULE", default="pieces")
        if self.scatter_schedule not in SCATTER_SCHEDULES:
            raise ValueError(f"SIRCL_SCATTER_SCHEDULE {self.scatter_schedule!r} is not one of {SCATTER_SCHEDULES}")
        gather_links = self.gather_schedule != "pieces" and self.max_gather_bytes > 0
        scatter_links = self.scatter_schedule != "pieces"
        reduce_links = self.large_schedule == "ring"
        # A tuning table that chooses a link schedule (a ring schedule, a chain all-gather or reduce-scatter)
        # needs the link area whatever the configured schedules.
        table_links = self._tuning is not None and any(
            choice.schedule == "ring" or (choice.schedule == "chain" and collective != "all_reduce")
            for collective, _, choice in self._tuning.decided())
        self._link_region = fabric_full and chain_threads and (gather_links or scatter_links or reduce_links
                                                               or table_links)
        for name, schedule in (("SIRCL_LARGE_SCHEDULE", self.large_schedule),
                               ("SIRCL_GATHER_SCHEDULE", self.gather_schedule),
                               ("SIRCL_SCATTER_SCHEDULE", self.scatter_schedule)):
            if schedule == "ring" and not self._link_region:
                raise ValueError(f"{name}=ring needs a layout whose every Spark hosts a rank of the session "
                                 "(pass layout=) and at least 64 threads per block")
        if self.gather_schedule == "chain" and not (self._link_region and gather_links):
            raise ValueError("SIRCL_GATHER_SCHEDULE=chain needs a layout whose every Spark hosts a rank of the "
                             "session (pass layout=), at least 64 threads per block and an enabled all-gather")
        if self.scatter_schedule == "chain" and not self._link_region:
            raise ValueError("SIRCL_SCATTER_SCHEDULE=chain needs a layout whose every Spark hosts a rank of the "
                             "session (pass layout=) and at least 64 threads per block")

    def _check_chain_chunk(self, chunk: int) -> None:
        if chunk < PACK_BYTES or chunk % PACK_BYTES or chunk > self.chain_slot_bytes:
            raise ValueError(f"chain chunk of {chunk} bytes must be a multiple of 16 up to the chain slot of "
                             f"{self.chain_slot_bytes} bytes")

    def _check_link_chunk(self, chunk: int, name: str = "link piece") -> None:
        if chunk < PACK_BYTES or chunk % PACK_BYTES or chunk > self.link_slot_bytes:
            raise ValueError(f"{name} of {chunk} bytes must be a multiple of 16 up to the link slot of "
                             f"{self.link_slot_bytes} bytes (SIRCL_LINK_SLOT_BYTES; without it the slot holds "
                             f"configured pieces up to {MAX_AUTO_LINK_SLOT_BYTES} bytes)")

    def _stagger_setting(self, name: str, default: int) -> int:
        """A ring stagger variable: ``auto`` gives ``default`` when the link slots hold it, else 0."""
        text = _env_text(name, default="auto")
        if text == "auto":
            return default if self.link_slots >= proto.ring_stagger_slots(self.world_size, default) else 0
        try:
            value = int(text)
        except ValueError:
            raise ValueError(f"{name}={text} is not auto or a number of rounds") from None
        self._check_ring_stagger(value)
        return value

    def _check_ring_stagger(self, stagger: int) -> None:
        needed = proto.ring_stagger_slots(self.world_size, stagger)
        if not 0 <= stagger <= proto.MAX_RING_STAGGER or self.link_slots < needed:
            raise ValueError(f"ring stagger {stagger} must be 0 to {proto.MAX_RING_STAGGER} rounds and needs "
                             f"{needed} link slots (SIRCL_LINK_SLOTS), the session has {self.link_slots}")

    def set_ring_stagger(self, stagger: int) -> None:
        """Stagger of later ring reduce-scatters and all-reduces (rounds a relay leaves after the partial
        it extends arrived); every rank must set the same value before the same op."""
        self._check_ring_stagger(int(stagger))
        self.ring_stagger = int(stagger)

    def set_ring_gather_stagger(self, stagger: int) -> None:
        """Stagger of later ring all-gathers and of the all-gather part of ring all-reduces (rounds a
        forward of a finished piece leaves after the piece arrived); every rank must set the same value
        before the same op."""
        self._check_ring_stagger(int(stagger))
        self.ring_gather_stagger = int(stagger)

    def chain_min_for(self, collective: str) -> int:
        """Smallest ``collective`` that ``auto`` runs as a chain op: ``reduce`` (all-reduce message bytes),
        ``gather`` (all-gather output bytes) or ``scatter`` (reduce-scatter input bytes)."""
        return self._chain_mins[self._min_collective(collective)]

    def ring_min_for(self, collective: str) -> int:
        """Smallest ``collective`` a ``ring`` schedule runs as a ring op (sizes as in
        :meth:`chain_min_for`); below it the schedule runs as ``auto`` does."""
        return self._ring_mins[self._min_collective(collective)]

    def set_chain_min_bytes(self, nbytes: Optional[int], collective: Optional[str] = None) -> None:
        """Chain minimum of later calls, of every collective (``collective`` None) or of one; None
        restores the default. Every rank must set the same value before the same op."""
        self._set_minimum(self._chain_mins, DEFAULT_CHAIN_MINS, nbytes, collective)

    def set_ring_min_bytes(self, nbytes: Optional[int], collective: Optional[str] = None) -> None:
        """Ring minimum of later calls, of every collective (``collective`` None) or of one; None
        restores the default. Every rank must set the same value before the same op."""
        self._set_minimum(self._ring_mins, DEFAULT_RING_MINS, nbytes, collective)

    @property
    def chain_min_bytes(self) -> Optional[int]:
        """The chain minimum all three collectives share, or None while they differ (the defaults).
        Assigning a size sets it for all three; assigning None restores the defaults."""
        values = set(self._chain_mins.values())
        return values.pop() if len(values) == 1 else None

    @chain_min_bytes.setter
    def chain_min_bytes(self, nbytes: Optional[int]) -> None:
        self.set_chain_min_bytes(nbytes)

    @property
    def ring_min_bytes(self) -> Optional[int]:
        """The ring minimum all three collectives share, or None while they differ (the defaults).
        Assigning a size sets it for all three; assigning None restores the defaults."""
        values = set(self._ring_mins.values())
        return values.pop() if len(values) == 1 else None

    @ring_min_bytes.setter
    def ring_min_bytes(self, nbytes: Optional[int]) -> None:
        self.set_ring_min_bytes(nbytes)

    @staticmethod
    def _min_collective(collective: str) -> str:
        if collective not in MIN_COLLECTIVES:
            raise ValueError(f"collectives with minimums are {', '.join(MIN_COLLECTIVES)}, not {collective!r}")
        return collective

    def _set_minimum(self, minimums: dict[str, int], defaults: Mapping[str, int], nbytes: Optional[int],
                     collective: Optional[str]) -> None:
        if nbytes is not None and int(nbytes) < 0:
            raise ValueError(f"a minimum of {nbytes} bytes must not be negative")
        for name in MIN_COLLECTIVES if collective is None else (self._min_collective(collective),):
            minimums[name] = defaults[name] if nbytes is None else int(nbytes)

    def link_chunk_for(self, collective: str) -> int:
        """The link piece of ``collective``: ``gather`` (chain and ring all-gathers), ``scatter`` (chain and
        ring reduce-scatters) or ``reduce`` (ring all-reduces); without a piece of its own, the session's
        link piece (``link_chunk_bytes``)."""
        if collective not in LINK_COLLECTIVES:
            raise ValueError(f"link collectives are {', '.join(LINK_COLLECTIVES)}, not {collective!r}")
        return self._link_chunks.get(collective, self.link_chunk_bytes)

    def set_link_chunk_bytes(self, chunk: int, collective: Optional[str] = None) -> None:
        """Piece size of later link ops: the session's link piece (``collective`` None), which every
        collective without a piece of its own uses, or the piece of ``collective`` (``gather``,
        ``scatter``, ``reduce``; 0 returns it to the session's piece). Every rank must set the same value
        before the same op."""
        if collective is None:
            self._check_link_chunk(int(chunk))
            self.link_chunk_bytes = int(chunk)
            return
        if collective not in LINK_COLLECTIVES:
            raise ValueError(f"link collectives are {', '.join(LINK_COLLECTIVES)}, not {collective!r}")
        if int(chunk) == 0:
            self._link_chunks.pop(collective, None)
            return
        self._check_link_chunk(int(chunk), LINK_COLLECTIVES[collective])
        self._link_chunks[collective] = int(chunk)

    def set_chain_chunk_bytes(self, chunk: int) -> None:
        """Chunk size of later chain ops; every rank must set the same value before the same op."""
        self._check_chain_chunk(int(chunk))
        self.chain_chunk_bytes = int(chunk)

    # -- measured choices ----------------------------------------------------------------------

    @staticmethod
    def _mode() -> str:
        """``graph`` inside a CUDA graph capture, else ``eager``: every rank captures the same ops."""
        return "graph" if torch.cuda.is_current_stream_capturing() else "eager"

    def tuning_facts(self) -> dict[str, Any]:
        """The key fields a tuning table must carry for this session (``tuning.facts``): its group shape,
        size, lanes and relays and this build's hashes and version. Needs the session's layout."""
        layout = self._layout_identity_object
        if layout is None:
            raise ValueError("tuning facts need the session's layout")
        relays = routes_mod.derive_routes(layout, self.lane_count).max_relays()
        return tuning_mod.facts(layout.identity(), self.world_size, self.lane_count, relays)

    def tuned_choice(self, collective: str, nbytes: int, mode: Optional[str] = None) -> Optional[tuning_mod.Choice]:
        """The tuning table's SIRCL choice for ``collective`` (``all_reduce``, ``all_gather``,
        ``reduce_scatter``, ``all_to_all``) of ``nbytes`` per rank in ``mode`` (the current one when None),
        or None without a table or below its smallest measured size."""
        if self._tuning is None:
            return None
        return self._tuning.decide(collective, int(nbytes), mode or self._mode())

    def tuned_backend(self, collective: str, nbytes: int, mode: Optional[str] = None) -> str:
        """``nccl`` where the tuning table measured NCCL faster than every SIRCL candidate, else ``sircl``
        (also without a table). Whether NCCL may run on the group is the caller's policy."""
        if self._tuning is None:
            return "sircl"
        return self._tuning.backend(collective, int(nbytes), mode or self._mode())

    def _apply_choice(self, collective: str, choice: tuning_mod.Choice, nbytes: int) -> bool:
        """Set the session's settings to ``choice`` for one op of ``nbytes``; False, changing nothing, when
        this session cannot run it (a schedule it lacks, a piece above its slots, a stagger its slots cannot
        hold, a grid above its counters, an all-reduce algorithm for a message above the capacity, which
        those algorithms run in one op). Every rank has the same settings, so every rank decides alike."""
        kind = {"all_reduce": "reduce", "all_gather": "gather", "reduce_scatter": "scatter"}.get(collective)
        if choice.grid is not None and choice.grid > self._counter_layout.blocks:
            return False
        if collective == "all_reduce" and choice.algorithm is not None and nbytes > self.max_size:
            return False
        if choice.schedule is not None:
            if kind is None:
                return False
            if choice.schedule == "ring" and not self.ring_available:
                return False
            if choice.schedule == "chain" and not (self.chain_available if kind == "reduce" else self.link_available):
                return False
            if choice.piece is not None and choice.schedule in ("ring", "chain"):
                limit = self.chain_slot_bytes if (choice.schedule == "chain" and kind == "reduce") else self.link_slot_bytes
                if choice.piece > limit:
                    return False
            if choice.stagger is not None and (choice.schedule != "ring" or kind == "gather" or
                                               proto.ring_stagger_slots(self.world_size, choice.stagger)
                                               > self.link_slots):
                return False
            if choice.gather_stagger is not None and (choice.schedule != "ring" or kind == "scatter" or
                                                      proto.ring_stagger_slots(self.world_size, choice.gather_stagger)
                                                      > self.link_slots):
                return False
        if choice.algorithm is not None and not self._available.get(choice.algorithm):
            return False
        if choice.grid is not None:
            self._large_blocks = choice.grid
            self._blocks = choice.grid
        if choice.schedule is not None:
            if collective == "all_reduce":
                self.large_schedule = choice.schedule
            elif collective == "all_gather":
                self.gather_schedule = choice.schedule
            else:
                self.scatter_schedule = choice.schedule
            if choice.schedule in ("ring", "chain"):
                # The table chose this schedule at this size: the minimums do not apply to the op.
                self._ring_mins[kind] = 0
                self._chain_mins[kind] = 0
            if choice.piece is not None:
                if choice.schedule == "chain" and kind == "reduce":
                    self.chain_chunk_bytes = choice.piece
                elif choice.schedule in ("ring", "chain"):
                    self._link_chunks[kind] = choice.piece
            if choice.stagger is not None:
                self.ring_stagger = choice.stagger
            if choice.gather_stagger is not None:
                self.ring_gather_stagger = choice.gather_stagger
        return True

    @contextlib.contextmanager
    def _tuned_op(self, collective: str, nbytes: int, *, mode: Optional[str] = None, count: bool = True):
        """One op of ``collective`` of ``nbytes`` with the tuning table's choice for ``mode`` (the current
        one when None) applied, the session's settings restored after it; counts the op under its choice
        (or as unusable when the choice cannot run here) unless ``count`` is False. Without a table, inside
        :meth:`untuned` or without a decision, nothing changes. Nested uses apply the same choice again."""
        if self._tuning is None or self._tuning_suspended:
            yield None
            return
        mode = mode or self._mode()
        choice = self._tuning.decide(collective, int(nbytes), mode)
        if choice is None:
            yield None
            return
        with self._tuning_lock:
            saved = (self._large_blocks, self._blocks, self.large_schedule, self.gather_schedule,
                     self.scatter_schedule, dict(self._ring_mins), dict(self._chain_mins), dict(self._link_chunks),
                     self.chain_chunk_bytes, self.ring_stagger, self.ring_gather_stagger)
            applied = self._apply_choice(collective, choice, int(nbytes))
            if count:
                label = f"{collective}/{mode}/{choice.label()}"
                counts = self._tuning_counts if applied else self._tuning_unusable
                counts[label] = counts.get(label, 0) + 1
            try:
                yield choice if applied else None
            finally:
                (self._large_blocks, self._blocks, self.large_schedule, self.gather_schedule, self.scatter_schedule,
                 ring_mins, chain_mins, link_chunks, self.chain_chunk_bytes, self.ring_stagger,
                 self.ring_gather_stagger) = saved
                self._ring_mins.clear()
                self._ring_mins.update(ring_mins)
                self._chain_mins.clear()
                self._chain_mins.update(chain_mins)
                self._link_chunks.clear()
                self._link_chunks.update(link_chunks)

    @contextlib.contextmanager
    def untuned(self):
        """Ops inside run under the session's own settings and rules, not the tuning table: for a caller
        that fixes a schedule, piece, stagger or grid of its own (the ring harness's forced variants).
        Nests. Every rank must enter it around the same ops, as for any setting."""
        with self._tuning_lock:
            self._tuning_suspended += 1
        try:
            yield
        finally:
            with self._tuning_lock:
                self._tuning_suspended -= 1

    def _tuning_stats(self) -> Optional[dict[str, Any]]:
        """The table this session decides from (None without one), its decisions so far by
        ``collective/mode/choice``, the choices it could not run, the table's settings with the session's own
        value of each (``settings``: name -> {"table", "session"}; they differ where the environment set
        another value), and the named tables that did not match."""
        if self._tuning is None and not self._tuning_unmatched:
            return None
        info: dict[str, Any] = {"table": None, "unmatched": dict(self._tuning_unmatched)}
        if self._tuning is not None:
            own = {"SIRCL_LINK_SLOTS": self.link_slots, "SIRCL_LINK_SLOT_BYTES": self.link_slot_bytes,
                   "SIRCL_CHAIN_SLOT_BYTES": self.chain_slot_bytes,
                   "SIRCL_LARGE_PIECE_BYTES": self._configured_large_piece}
            info.update(table=self._tuning.hash, path=self._tuning.source, key=dict(self._tuning.key),
                        decisions=dict(sorted(self._tuning_counts.items())),
                        unusable=dict(sorted(self._tuning_unusable.items())),
                        settings={name: {"table": value, "session": own[name]}
                                  for name, value in sorted(self._table_settings.items())})
        return info

    def launch_grid(self, kind: str, packs: int) -> int:
        """Grid of one launch of ``kind`` moving ``packs`` 16-byte packs: the smallest power of two of at
        least one block per ``SIRCL_THREADS * SIRCL_PACKS_PER_THREAD`` packs, capped by ``SIRCL_BLOCKS`` for
        ``oneshot`` and ``gather`` (the plain all-gather) and by the large-message cap for ``twoshot`` and
        ``gather-tiles``. The one place a launch grid is chosen; it depends on the kind and size only."""
        if kind in ("oneshot", "gather"):
            cap = self._blocks
        elif kind in ("twoshot", "gather-tiles"):
            cap = self._large_blocks
        else:
            raise ValueError(f"no launch grid for {kind!r}")
        return _grid_blocks(int(packs), self._threads, cap, self._packs_per_thread)

    @property
    def blocks(self) -> int:
        """Largest grid of one-shot and plain all-gather launches (``SIRCL_BLOCKS``)."""
        return self._blocks

    def set_blocks(self, blocks: int) -> None:
        """Largest grid of later one-shot and plain all-gather launches: a power of two up to the counter
        layout's grid. Rank-local like ``set_large_blocks``."""
        value = int(blocks)
        if value < 1 or value & (value - 1) or value > self._counter_layout.blocks:
            raise ValueError(f"grids are powers of two up to {self._counter_layout.blocks}, got {blocks}")
        self._blocks = value

    @property
    def large_blocks(self) -> int:
        """Largest grid of two-shot and large-message launches (``SIRCL_LARGE_BLOCKS``)."""
        return self._large_blocks

    def set_large_blocks(self, blocks: int) -> None:
        """Largest grid of later two-shot and large-message launches: a power of two up to the counter
        layout's grid (the larger of ``SIRCL_BLOCKS``, ``SIRCL_LARGE_BLOCKS`` and 32). The grid changes only
        how a launch splits its work, not what ranks exchange, so ranks may differ; a CUDA graph keeps the
        grid it was captured with."""
        value = int(blocks)
        if value < 1 or value & (value - 1) or value > self._counter_layout.blocks:
            raise ValueError(f"large-message grids are powers of two up to {self._counter_layout.blocks}, "
                             f"got {blocks}")
        self._large_blocks = value

    def _allocate(self) -> None:
        self._layout = Layout(self.world_size, self._slot_bytes)
        expected = proto.ArenaLayout(self.world_size, self._slot_bytes).as_tuple()
        reported = (self._layout.recv_off, self._layout.flag_off, self._layout.send_off,
                    self._layout.ctrl_off, self._layout.total_bytes, self._layout.flag_stride,
                    self._layout.slots)
        if reported != expected:
            raise RuntimeError(f"native arena layout {reported} differs from the protocol's {expected}")
        self._region_bytes = self._layout.total_bytes
        if self._chain_region:
            self._chain_layout = proto.ChainLayout(self.lane_count, self.chain_slots, self.chain_slot_bytes)
            native = _proxy_chain_layout(self.lane_count, self.chain_slots, self.chain_slot_bytes)
            if native != self._chain_layout.as_tuple():
                raise RuntimeError(f"native chain layout {native} differs from the protocol's "
                                   f"{self._chain_layout.as_tuple()}")
            self._chain_offset = proto.chain_offset(self._layout.total_bytes)
            self._region_bytes = self._chain_offset + self._chain_layout.total_bytes
        if self._link_region:
            self._link_layout = proto.LinkLayout(self.lane_count, self.link_slots, self.link_slot_bytes)
            native = _proxy_link_layout(self.lane_count, self.link_slots, self.link_slot_bytes)
            if native != self._link_layout.as_tuple():
                raise RuntimeError(f"native link layout {native} differs from the protocol's "
                                   f"{self._link_layout.as_tuple()}")
            self._link_offset = proto.chain_offset(self._region_bytes)
            self._region_bytes = self._link_offset + self._link_layout.total_bytes
        with torch.cuda.device(self.device):
            # Zeroed: flags and command words start at 0 and the first sequence is 1.
            self._region = torch.zeros(self._region_bytes, dtype=torch.uint8, pin_memory=True)
            self._counters = torch.zeros(self._counter_layout.words, dtype=torch.int32, device=self.device)
            # Arrival words of the one-block flag polling: namespace 0, namespace 1.
            self._arrival = torch.zeros(2, dtype=torch.int32, device=self.device)
            self._chain_counters = torch.zeros(_chain_cute.COUNTER_WORDS, dtype=torch.int32, device=self.device)
            self._link_counters = torch.zeros(_links_cute.COUNTER_WORDS, dtype=torch.int32, device=self.device)
            # Arrival counters of the chain reduce-scatter's pieces, then one decision word per block.
            self._piece_counters = torch.zeros(
                _links_cute.PIECE_COUNTERS + _links_cute.DECISION_WORDS if self._link_region else 1,
                dtype=torch.int32, device=self.device)
        host = self._region.data_ptr()
        if _device_pointer(host) != host:
            raise RuntimeError("SIRCL needs pinned host memory that the GPU addresses at its host "
                               "pointer (an integrated GPU with unified addressing)")
        self._recv_base = host + self._layout.recv_off
        self._flag_base = host + self._layout.flag_off
        self._send_base = host + self._layout.send_off
        self._ctrl_base = host + self._layout.ctrl_off
        self._ctrl_words = self._region[self._layout.ctrl_off:self._layout.ctrl_off + proto.FLAG_STRIDE].view(torch.int32)
        self._ctrl_np = self._ctrl_words.numpy()
        self._set_regime("startup")
        self._epoch_address = self._counters.data_ptr()
        self._poison_address = self._epoch_address + 4 * self._counter_layout.poison_word
        self._arrival_address = self._arrival.data_ptr()

    def _setup_record(self) -> dict[str, Any]:
        layout = self._layout_identity_object
        return {
            "api_version": API_VERSION,
            "proxy_abi": ABI_VERSION,
            "world_size": self.world_size,
            "lane_count": self.lane_count,
            "slot_bytes": self._slot_bytes,
            "slots": self._layout.slots,
            "flag_stride": self._layout.flag_stride,
            "max_size": self.max_size,
            "dispatch_limit_bytes": self.dispatch_limit_bytes,
            "max_gather_bytes": self.max_gather_bytes,
            "spin_limit": self.spin_limit,
            "startup_wait_s": self.startup_wait_s,
            "serving_wait_s": self.serving_wait_s,
            "threads": self._threads,
            "blocks": self._blocks,
            "packs_per_thread": self._packs_per_thread,
            "flag_pollers": self.flag_pollers,
            "algorithm": self.algorithm,
            "large_algorithm": self.large_algorithm,
            "oneshot_max_bytes": self.oneshot_max_bytes,
            "swing_above_bytes": self.swing_above_bytes,
            "multi_phase": self.multi_phase,
            "scatter": self.scatter_available,
            "available": dict(self._available),
            "tuning_table": self._tuning.hash if self._tuning is not None else None,
            "large_blocks": self._large_blocks,
            "large_piece_bytes": self._configured_large_piece,
            "forward_window_bytes": self.forward_window_bytes,
            "forward_chunk_bytes": self.forward_chunk_bytes,
            "hairpin_queue_bytes": self.hairpin_queue_bytes,
            "large_schedule": self.large_schedule,
            "chain_region": self._chain_region,
            "chain_slots": self.chain_slots,
            "chain_slot_bytes": self.chain_slot_bytes,
            "chain_chunk_bytes": self.chain_chunk_bytes,
            "chain_blocks": self.chain_blocks,
            "chain_unroll": self.chain_unroll,
            "chain_mins": dict(self._chain_mins),
            "ring_mins": dict(self._ring_mins),
            "gather_schedule": self.gather_schedule,
            "scatter_schedule": self.scatter_schedule,
            "link_region": self._link_region,
            "link_slots": self.link_slots,
            "link_slot_bytes": self.link_slot_bytes,
            "link_chunk_bytes": self.link_chunk_bytes,
            "link_chunks": {collective: self.link_chunk_for(collective) for collective in LINK_COLLECTIVES},
            "ring_stagger": self.ring_stagger,
            "ring_gather_stagger": self.ring_gather_stagger,
            "link_blocks": self.link_blocks,
            "link_unroll": self.link_unroll,
            "post_mode": self.post_mode,
            "traffic_class": self._proxy.traffic_class if self._proxy is not None else None,
            "layout": layout.identity() if layout is not None else None,
            "devices": list(self.hca_names),
            "gid_indices": list(self.gid_indices),
            "lane_counts": [len(devices) for devices in self.peer_routes],
            "route_map": {str(peer): list(devices) for peer, devices in enumerate(self.peer_routes) if devices},
        }

    def _plan_relays(self, route_maps: Sequence[Mapping[int, Sequence[str]]]) -> None:
        """Forward windows of this rank's lanes and the piece sizes, from every rank's agreed route map.

        Every input is agreed, so every rank derives the same piece sizes.
        Without a layout the hops of a lane are unknown: no windows, and the
        piece sizes stay at the capacities.
        """
        layout = self._layout_identity_object
        self._forward_table = [[0] * self.lane_count for _ in range(self.world_size)]
        self.relay_safe_bytes = None
        if layout is None:
            return
        self._forward_table = routes_mod.forward_windows(
            layout, route_maps, self.rank, max_window=self.forward_window_bytes,
            chunk=self.forward_chunk_bytes or DEFAULT_FORWARD_CHUNK_BYTES,
            queue_bytes=self.hairpin_queue_bytes)
        if self.forward_window_bytes:
            return
        queues = routes_mod.relay_queues(layout, route_maps)
        busiest = max((len(members) for members in queues.values()), default=0)
        safe = pieces.relay_safe_bytes(busiest, self.lane_count, self.hairpin_queue_bytes,
                                       routes_mod.RELAY_QUEUE_SHARE)
        self.relay_safe_bytes = safe
        if safe is not None:
            self.gather_piece_bytes = max(PACK_BYTES, min(self.gather_piece_bytes, safe))
            per_peer = self.world_size if self._available["twoshot"] else 1
            self.large_piece_bytes = max(PACK_BYTES, min(self.large_piece_bytes, safe * per_peer))

    def _plan_chain(self, route_maps: Sequence[Mapping[int, Sequence[str]]]) -> None:
        """The chain order of the session's ranks and this rank's neighbors, from the agreed
        layout and route maps; configures the native chain schedule (before it starts)."""
        self.chain_order = None
        self.chain_index = None
        self.chain_available = False
        self.link_available = False
        if not (self._chain_region or self._link_region):
            return
        order = routes_mod.chain_order(self._layout_identity_object, route_maps)
        if order is None:
            for name, schedule in (("SIRCL_LARGE_SCHEDULE", self.large_schedule),
                                   ("SIRCL_GATHER_SCHEDULE", self.gather_schedule)):
                if schedule == "chain":
                    raise ValueError(f"{name}=chain: the ranks do not form a chain of cable neighbors joined by "
                                     "direct lanes")
            return
        index = order.index(self.rank)
        prev_rank = order[index - 1] if index > 0 else -1
        next_rank = order[index + 1] if index < len(order) - 1 else -1
        self.chain_order = tuple(order)
        self.chain_index = index
        self._chain_prev, self._chain_next = prev_rank, next_rank
        if self._chain_region:
            self._proxy.set_chain(prev_rank, next_rank, self.chain_slots, self.chain_slot_bytes, self._chain_offset)
            self.chain_available = True
        if self._link_region:
            ring_prev, ring_next, window = self._plan_ring(order, index, route_maps)
            self._proxy.set_links(prev_rank, next_rank, index, self.link_slots, self.link_slot_bytes,
                                  self._link_offset, ring_prev=ring_prev, ring_next=ring_next, ring_window=window)
            self.link_available = True
            self.ring_available = ring_next >= 0
            self.ring_window_bytes = window

    def _plan_ring(self, order: Sequence[int], index: int,
                   route_maps: Sequence[Mapping[int, Sequence[str]]]) -> tuple[int, int, int]:
        """``(ring prev, ring next, window)`` of this rank on the ring that closes the chain ``order``,
        or ``(-1, -1, 0)`` when the ring cannot run (``ring_problem`` says why).

        The last rank reaches the first over the closing cable of a cycle or, on a path, through
        relays. During a ring op the ring's lanes are the only relayed traffic, and relays forward
        each function's lanes through that function's own hairpin queue, so the ring runs when no
        relay queue carries two ring lanes (``routes.ring_window``); each relayed ring lane then keeps
        at most 75 % of a queue unacknowledged. Every rank decides the same: the decision follows the
        agreed layout and route maps. A ring schedule named at construction makes a ring that cannot
        run a setup error.
        """
        world = len(order)
        window, problems = routes_mod.ring_window(self._layout_identity_object, route_maps, order,
                                                  chunk=proto.LINK_WINDOW_CHUNK,
                                                  queue_bytes=self.hairpin_queue_bytes)
        if problems:
            self.ring_problem = "; ".join(problems)
            for name, schedule in (("SIRCL_LARGE_SCHEDULE", self.large_schedule),
                                   ("SIRCL_GATHER_SCHEDULE", self.gather_schedule),
                                   ("SIRCL_SCATTER_SCHEDULE", self.scatter_schedule)):
                if schedule == "ring":
                    raise ValueError(f"{name}=ring: the ring cannot run: {self.ring_problem}")
            return -1, -1, 0
        ring_next = order[(index + 1) % world]
        relayed = any(rank == self.rank for members in
                      routes_mod.ring_queues(self._layout_identity_object, route_maps, order).values()
                      for rank, _, _ in members)
        return order[(index - 1) % world], ring_next, window if relayed else 0

    def _verdict(self, what: str, action: Callable[[], None]) -> None:
        error = None
        if self._proxy is None:
            error = "no native context"
        else:
            try:
                action()
            except Exception as exc:  # noqa: BLE001 - reported to every rank
                error = str(exc)
        verdicts = _exchange(error, self._group)
        failures = [f"rank {index}: {verdict}" for index, verdict in enumerate(verdicts) if verdict is not None]
        if failures:
            self.close()
            raise RuntimeError(f"SIRCL session {what} failed: " + "; ".join(failures))

    @classmethod
    def from_exchange_group(cls, *, exchange_group: ProcessGroup, device, max_size: int = DEFAULT_MAX_SIZE,
                            eager_buffer_bytes: Optional[int] = None,
                            max_gather_bytes: int = DEFAULT_MAX_GATHER_BYTES, **_ignored: Any):
        capacity = max(int(max_size), int(eager_buffer_bytes or 0))
        return cls(exchange_group=exchange_group, device=device, max_size=capacity,
                   max_gather_bytes=max_gather_bytes)

    @classmethod
    def from_process_group(cls, *, process_group: ProcessGroup, device, max_size: int = DEFAULT_MAX_SIZE,
                           max_input_bytes: Optional[int] = None, **_ignored: Any):
        capacity = max(int(max_size), int(max_input_bytes or 0))
        return cls(exchange_group=process_group, device=device, max_size=capacity)

    # -- wait limits --------------------------------------------------------------------

    def _set_regime(self, regime: str) -> None:
        if regime not in WAIT_REGIMES:
            raise ValueError(f"wait regime must be one of {WAIT_REGIMES}, got {regime!r}")
        seconds = self.startup_wait_s if regime == "startup" else self.serving_wait_s
        micros = max(1, min(0xFFFFFFFF, int(round(seconds * 1e6))))
        # A plain host store: the next launch's kernels read it, eager or replayed.
        self._ctrl_np[proto.Ctrl.WAIT_LIMIT_US] = micros - (1 << 32) if micros >= 1 << 31 else micros
        self.wait_regime = regime

    @property
    def wait_limit_s(self) -> float:
        """The flag-wait limit of launches from now on, in seconds."""
        return self.startup_wait_s if self.wait_regime == "startup" else self.serving_wait_s

    def enter_startup(self) -> None:
        """Wait up to ``startup_wait_s`` for peers: compilation, warm-up and graph capture."""
        self._set_regime("startup")

    def enter_serving(self) -> None:
        """Wait up to ``serving_wait_s`` for peers: steady serving, where a longer lag means a failure."""
        self._set_regime("serving")

    @contextmanager
    def startup(self):
        """The startup regime for the duration of the block, then the previous one."""
        previous = self.wait_regime
        self._set_regime("startup")
        try:
            yield self
        finally:
            self._set_regime(previous)

    def _measure_poll_rate(self) -> None:
        """Time a fixed number of flag polls on this GPU (an own-rank flag line nobody writes)."""
        probe = _timed_wait.get_probe(self.device.index)
        line = proto.flag_index(0, self.rank, 0, 0, self.world_size, self.lane_count)
        result = torch.zeros(1, dtype=torch.int64, device=self.device)
        probe(self._flag_base + line * proto.FLAG_STRIDE, POLL_RATE_PROBE_POLLS, result.data_ptr())
        elapsed_ns = int(result.item())
        self.poll_rate_per_s = POLL_RATE_PROBE_POLLS / (elapsed_ns * 1e-9) if elapsed_ns > 0 else None

    # -- policy -----------------------------------------------------------------------

    @property
    def supports_all_peer_auxiliary(self) -> bool:
        return False

    def _require_open(self) -> None:
        if self._closed or self._proxy is None:
            raise RuntimeError("SIRCL session is closed")

    def _allreduce_eligible(self, inp: torch.Tensor, limit: int) -> bool:
        if inp.dtype not in SUPPORTED_DTYPES or not inp.is_cuda:
            return False
        if inp.device != self.device or not inp.is_contiguous():
            return False
        nbytes = inp.numel() * inp.element_size()
        return 0 < nbytes <= limit and nbytes % PACK_BYTES == 0

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        """Dispatch predicate: eligible all-reduce up to the dispatch limit (rank-invariant)."""
        self._require_open()
        return self._allreduce_eligible(inp, self.dispatch_limit_bytes)

    def _oneshot_limit(self, layout: routes_mod.Layout | None) -> tuple[int, str]:
        """The one-shot limit and where it comes from: ``SIRCL_ONESHOT_MAX_BYTES`` when set; with a
        layout and the two-shot all-reduce, the latency model's limit for the layout, the lane count and
        the posting order (a rank's own peer list counts as ``farthest``, so every rank derives the same
        value); else ``DEFAULT_ONESHOT_MAX_BYTES``."""
        if os.environ.get("SIRCL_ONESHOT_MAX_BYTES", "").strip():
            return (_env_int("SIRCL_ONESHOT_MAX_BYTES", default=DEFAULT_ONESHOT_MAX_BYTES),
                    "SIRCL_ONESHOT_MAX_BYTES")
        if layout is None or not self._available["twoshot"]:
            return DEFAULT_ONESHOT_MAX_BYTES, "default"
        order = self._post_order_name if self._post_order_name in posting_mod.POST_ORDERS else "farthest"
        limit = latency_model.oneshot_limit(layout, self.lane_count, order,
                                            cap=min(self.max_size, DEFAULT_ONESHOT_MAX_BYTES))
        return limit, f"latency model ({order})"

    def select_algorithm(self, nbytes: int, *, mode: Optional[str] = None) -> str:
        """The all-reduce algorithm of a message of ``nbytes``: the tuning table's choice for ``mode``
        (the current one when None; a CUDA graph capture is ``graph``) when it names an available algorithm
        and :meth:`untuned` is not in effect, else the rules."""
        if self._tuning is not None and not self._tuning_suspended:
            choice = self._tuning.decide("all_reduce", int(nbytes), mode or self._mode())
            if choice is not None and choice.algorithm is not None and self._available.get(choice.algorithm):
                return choice.algorithm
        return proto.select_algorithm(
            int(nbytes), algorithm=self.algorithm, large_algorithm=self.large_algorithm,
            oneshot_max_bytes=self.oneshot_max_bytes, swing_above_bytes=self.swing_above_bytes,
            available=self._available,
        )

    def prepare_channels(self, channel_ids: Sequence[str]) -> None:
        return None

    def for_stream(self, stream: object = None, *, channel_id: Optional[str] = None):
        return self

    # -- compilation --------------------------------------------------------------------

    def prepare(self, dtypes: Sequence[torch.dtype] = (torch.bfloat16,), *, padded_gather: bool = False,
                algorithms: Optional[Sequence[str]] = None, scatter: bool = False, links: bool = False) -> None:
        """Compile every launcher this session can need and allocate scratch, outside any capture.

        ``links`` also compiles the link collectives the configured schedules do not select (the chain
        all-gather and, with ``scatter``, the chain reduce-scatter), for schedules switched at run time.

        The two-shot launcher is compiled for every dtype whenever the two-shot
        all-reduce is available, because ``all_reduce_large`` uses it at any
        message size.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SIRCL prepare() is refused inside a CUDA graph capture")
        unknown = sorted(set(algorithms or ()) - set(ALGORITHMS))
        if unknown:
            raise ValueError(f"unknown all-reduce algorithms {unknown}")
        if scatter and not (self.scatter_available or self.link_available):
            raise RuntimeError("the scatter collectives are unavailable in this session: they need multi-phase "
                               "posting (at most two lanes per peer, slots below 2^30 bytes) or the chain links")
        boundaries = {self.max_size}
        for threshold in (self.oneshot_max_bytes, self.swing_above_bytes):
            if 0 < threshold < self.max_size:
                boundaries.add(threshold + PACK_BYTES)
        # The one-shot launcher serves every small message; the two-shot one, when
        # available, every piece of all_reduce_large above the one-shot limit.
        wanted = {"oneshot"} | set(algorithms or ()) | {self.select_algorithm(size) for size in boundaries}
        if self._tuning is not None:
            # Every algorithm and schedule the tuning table can choose is ready before any capture.
            chosen = self._tuning.chosen()
            wanted |= {choice.algorithm for choice in chosen
                       if choice.algorithm is not None and self._available.get(choice.algorithm)}
            links = links or any(choice.schedule in ("chain", "ring") for choice in chosen)
        if self._available["twoshot"]:
            wanted.add("twoshot")
        for name in sorted(wanted):
            if not self._available[name]:
                raise RuntimeError(f"the {name} all-reduce is unavailable in this session")
        with torch.cuda.device(self.device):
            for dtype in dtypes:
                if dtype not in SUPPORTED_DTYPES:
                    raise ValueError(f"unsupported all-reduce dtype {dtype}")
                for name in sorted(wanted):
                    self._reduce_launcher(name, dtype, capturing=False)
            if self.max_gather_bytes > 0:
                self._all_gather_launcher(capturing=False)
                self._tiled_gather_launcher(capturing=False)
            if self.chain_available:
                for dtype in dtypes:
                    self._chain_launcher(dtype, capturing=False)
            if self.link_available and self.max_gather_bytes > 0 and (links or self.gather_schedule != "pieces"):
                self._gather_chain_launcher(capturing=False)
            if scatter and self.link_available and (links or self.scatter_schedule != "pieces"):
                for dtype in dtypes:
                    self._scatter_chain_launcher(dtype, capturing=False)
            schedules = (self.large_schedule, self.gather_schedule, self.scatter_schedule)
            if self.ring_available and (links or "ring" in schedules):
                for dtype in dtypes:
                    self._ring_launcher("reduce", dtype, capturing=False)
                    if scatter:
                        self._ring_launcher("scatter", dtype, capturing=False)
                if self.max_gather_bytes > 0:
                    self._ring_launcher("gather", None, capturing=False)
            if scatter and self.scatter_available:
                _scatter_ops.prepare(self, dtypes)
            if self.poll_rate_per_s is None:
                # A diagnostic: a failed probe leaves the rate unknown and changes nothing else.
                try:
                    self._measure_poll_rate()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("SIRCL poll-rate probe failed on rank %d: %s", self.rank, exc)
            if self.event_trace:
                # event_trace_records maps kernel times with the clock probe: compiled here, not
                # between collectives.
                _timed_wait.get_clock_probe(self.device.index)
            self._aligned_scratch(0, self._region[:PACK_BYTES])
            self._kernel_trace_address()
            if padded_gather and self.max_gather_bytes > 0:
                self._gather_scratch(PACK_BYTES)

    def _oneshot_launcher(self, dtype: torch.dtype, capturing: bool) -> Callable[..., None]:
        key = ("oneshot", dtype)
        launcher = self._launchers.get(key)
        if launcher is None:
            if capturing:
                raise RuntimeError(f"SIRCL one-shot all-reduce for {dtype} was not prepared before CUDA "
                                   "graph capture; call prepare()")
            if dtype not in SUPPORTED_DTYPES:
                raise ValueError(f"unsupported all-reduce dtype {dtype}")
            launcher = _oneshot_cute.get_launcher(
                _DTYPE_NAMES[dtype], self.world_size, self.rank, self._threads, self._layout.slots,
                self._layout.flag_stride, self.lane_count, self.device.index, False,
                one_block_polls=self._one_block_polls,
            )
            self._launchers[key] = launcher
        return launcher

    def _twoshot_launcher(self, dtype: torch.dtype, capturing: bool) -> Callable[..., None]:
        key = ("twoshot", dtype)
        launcher = self._launchers.get(key)
        if launcher is None:
            if capturing:
                raise RuntimeError(f"SIRCL two-shot all-reduce for {dtype} was not prepared before CUDA "
                                   "graph capture; call prepare()")
            if dtype not in SUPPORTED_DTYPES:
                raise ValueError(f"unsupported all-reduce dtype {dtype}")
            launcher = _twoshot_cute.get_launcher(
                _DTYPE_NAMES[dtype], self.world_size, self.rank, self._threads, self._layout.slots,
                self._layout.flag_stride, self.lane_count, self.device.index, False,
                one_block_polls=self._one_block_polls,
            )
            self._launchers[key] = launcher
        return launcher

    def _chain_launcher(self, dtype: torch.dtype, capturing: bool) -> Callable[..., None]:
        key = ("chain", dtype)
        launcher = self._launchers.get(key)
        if launcher is None:
            if capturing:
                raise RuntimeError(f"SIRCL chain all-reduce for {dtype} was not prepared before CUDA graph "
                                   "capture; call prepare()")
            if dtype not in SUPPORTED_DTYPES:
                raise ValueError(f"unsupported all-reduce dtype {dtype}")
            launcher = _chain_cute.get_launcher(
                _DTYPE_NAMES[dtype], self.world_size, self.chain_index, self._chain_prev, self._chain_next,
                self.rank, self._threads, self.lane_count, self.chain_slots, self.chain_slot_bytes,
                self.chain_blocks, self.device.index, self.chain_unroll, self.event_trace,
            )
            self._launchers[key] = launcher
        return launcher

    def _gather_chain_launcher(self, capturing: bool) -> Callable[..., None]:
        key = ("link-gather",)
        launcher = self._launchers.get(key)
        if launcher is None:
            if capturing:
                raise RuntimeError("SIRCL chain all-gather was not prepared before CUDA graph capture; call prepare()")
            launcher = _links_cute.get_gather_launcher(
                self.world_size, self.chain_index, self._chain_prev, self._chain_next, self.rank, self.chain_order,
                self._threads, self.lane_count, self.link_slots, self.link_slot_bytes, self.link_blocks,
                self.link_unroll, self.device.index,
            )
            self._launchers[key] = launcher
        return launcher

    def _scatter_chain_launcher(self, dtype: torch.dtype, capturing: bool) -> Callable[..., None]:
        key = ("link-scatter", dtype)
        launcher = self._launchers.get(key)
        if launcher is None:
            if capturing:
                raise RuntimeError(f"SIRCL chain reduce-scatter for {dtype} was not prepared before CUDA graph "
                                   "capture; call prepare(..., scatter=True)")
            if dtype not in SUPPORTED_DTYPES:
                raise ValueError(f"unsupported reduce-scatter dtype {dtype}")
            launcher = _links_cute.get_scatter_launcher(
                _DTYPE_NAMES[dtype], self.world_size, self.chain_index, self._chain_prev, self._chain_next,
                self.rank, self.chain_order, self._threads, self.lane_count, self.link_slots, self.link_slot_bytes,
                self.link_blocks, self.link_unroll, self.device.index,
            )
            self._launchers[key] = launcher
        return launcher

    def _ring_launcher(self, mode: str, dtype: Optional[torch.dtype], capturing: bool) -> Callable[..., None]:
        name = "bytes" if dtype is None else _DTYPE_NAMES[dtype]
        key = ("link-ring", mode, name)
        launcher = self._launchers.get(key)
        if launcher is None:
            if capturing:
                raise RuntimeError(f"SIRCL ring {mode} for {name} was not prepared before CUDA graph capture; "
                                   "call prepare(..., links=True)")
            launcher = _links_cute.get_ring_launcher(
                mode, name, self.world_size, self.chain_index, self.rank, self.chain_order, self._threads,
                self.lane_count, self.link_slots, self.link_slot_bytes, self.link_blocks, self.link_unroll,
                self.device.index, self.event_trace,
            )
            self._launchers[key] = launcher
        return launcher

    def _launch_ring(self, launcher: Callable[..., None], inp: torch.Tensor, out: torch.Tensor, chunk_bytes: int,
                     stride_bytes: int, capturing: bool, piece_bytes: Optional[int] = None) -> None:
        """One ring op: ``inp`` and ``out`` 16-byte aligned, chunks of ``chunk_bytes`` every ``stride_bytes``
        (the all-gather: the shard's bytes, twice), in pieces of ``piece_bytes`` (default the session's
        link piece)."""
        piece = self.link_chunk_bytes if piece_bytes is None else int(piece_bytes)
        self._order_stream(capturing)
        launcher(inp.data_ptr(), out.data_ptr(), chunk_bytes // PACK_BYTES, stride_bytes // PACK_BYTES,
                 piece // PACK_BYTES, self._region.data_ptr() + self._link_offset,
                 self._link_counters.data_ptr(), self._ctrl_base, self._poison_address, self.spin_limit,
                 self._kernel_trace_address(), self.ring_stagger, self.ring_gather_stagger)
        self._mark_stream(capturing)

    def _scatter_geometry(self, inp: torch.Tensor, chunk_bytes: Optional[int], src_stride_bytes: Optional[int]):
        """The scatter geometry of a link reduce-scatter of ``inp``, or None (dtype, device, contiguity, size)."""
        if not self.link_available or inp.dtype not in SUPPORTED_DTYPES:
            return None
        if not inp.is_cuda or inp.device != self.device or inp.dim() == 0 or not inp.is_contiguous():
            return None
        geometry = _scatter_plan.scatter_geometry(inp.numel() * inp.element_size(), self.world_size, None,
                                                  chunk_bytes, src_stride_bytes)
        if geometry is None:
            return None
        if (geometry.src_stride_bytes * self.world_size >= 1 << 35
                or geometry.chunk_bytes * self.world_size >= 1 << 31):
            return None
        return geometry

    def scatter_uses_ring(self, inp: torch.Tensor, *, chunk_bytes: Optional[int] = None,
                          src_stride_bytes: Optional[int] = None, mode: Optional[str] = None) -> bool:
        """Whether ``reduce_scatter`` runs as one ring reduce-scatter (schedule ``ring`` with a ring, a link
        reduce-scatter's geometry whose ``W`` chunks hold at least ``ring_min_for("scatter")``; the same on
        every rank for the same arguments), with the tuning table's choice for ``mode`` applied as the op
        applies it."""
        with self._tuned_op("reduce_scatter", inp.numel() * inp.element_size(), mode=mode, count=False):
            return self._scatter_uses_ring(inp, chunk_bytes, src_stride_bytes)

    def _scatter_uses_ring(self, inp: torch.Tensor, chunk_bytes: Optional[int],
                           src_stride_bytes: Optional[int]) -> bool:
        if self.scatter_schedule != "ring" or not self.ring_available:
            return False
        geometry = self._scatter_geometry(inp, chunk_bytes, src_stride_bytes)
        return (geometry is not None
                and self._schedule("ring", geometry.chunk_bytes * self.world_size, "scatter") == "ring")

    def _reduce_scatter_ring(self, inp: torch.Tensor, out: Optional[torch.Tensor], stream: object,
                             chunk_bytes: Optional[int], src_stride_bytes: Optional[int]) -> torch.Tensor:
        """One ring reduce-scatter (``scatter_uses_ring`` holds for the arguments)."""
        with self._lock:
            self.check_health()
            geometry = self._scatter_geometry(inp, chunk_bytes, src_stride_bytes)
            chunk = geometry.chunk_bytes
            if inp.data_ptr() % PACK_BYTES:
                raise ValueError("the SIRCL reduce-scatter needs a 16-byte aligned input pointer")
            out = self._scatter_output(inp, out, chunk, chunk_bytes)
            context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
            with torch.cuda.device(self.device), context:
                capturing = torch.cuda.is_current_stream_capturing()
                launcher = self._ring_launcher("scatter", inp.dtype, capturing)
                self._launch_ring(launcher, inp, out, chunk, geometry.src_stride_bytes, capturing,
                                  self.link_chunk_for("scatter"))
            if not capturing:
                self.check_health()
            return out

    def _scatter_output(self, inp: torch.Tensor, out: Optional[torch.Tensor], chunk: int,
                        chunk_bytes: Optional[int]) -> torch.Tensor:
        """``out`` checked, or allocated: one chunk, shaped as rows when the chunks are whole rows."""
        if out is None:
            if chunk_bytes is None and inp.shape[0] % self.world_size == 0:
                return torch.empty((inp.shape[0] // self.world_size, *inp.shape[1:]), dtype=inp.dtype,
                                   device=inp.device)
            return torch.empty(chunk // inp.element_size(), dtype=inp.dtype, device=inp.device)
        if (out.dtype != inp.dtype or out.device != inp.device or not out.is_contiguous()
                or out.numel() * out.element_size() != chunk or out.data_ptr() % PACK_BYTES):
            raise ValueError("out must be a 16-byte aligned contiguous tensor of one chunk in the input's dtype")
        return out

    def scatter_uses_chain(self, inp: torch.Tensor, *, chunk_bytes: Optional[int] = None,
                           src_stride_bytes: Optional[int] = None, mode: Optional[str] = None) -> bool:
        """Whether ``reduce_scatter(inp, chunk_bytes=, src_stride_bytes=)`` runs as one chain reduce-scatter
        (the same on every rank for the same arguments): the chain links exist, the schedule allows it, the
        dtype is float16, bfloat16 or float32, the input is a contiguous tensor on the session's device in a
        scatter geometry (``scatter_plan.scatter_geometry``), and its chunk holds at most
        ``PIECE_COUNTERS`` link pieces; with the tuning table's choice for ``mode`` applied as the op
        applies it."""
        with self._tuned_op("reduce_scatter", inp.numel() * inp.element_size(), mode=mode, count=False):
            return self._scatter_uses_chain(inp, chunk_bytes, src_stride_bytes)

    def _scatter_uses_chain(self, inp: torch.Tensor, chunk_bytes: Optional[int],
                            src_stride_bytes: Optional[int]) -> bool:
        if not self.link_available or inp.dtype not in SUPPORTED_DTYPES:
            return False
        if not inp.is_cuda or inp.device != self.device or inp.dim() == 0 or not inp.is_contiguous():
            return False
        geometry = _scatter_plan.scatter_geometry(inp.numel() * inp.element_size(), self.world_size, None,
                                                  chunk_bytes, src_stride_bytes)
        if geometry is None:
            return False
        schedule = self._schedule(self.scatter_schedule, geometry.chunk_bytes * self.world_size, "scatter")
        if schedule not in ("auto", "chain"):
            return False
        if -(-geometry.chunk_bytes // self.link_chunk_for("scatter")) > _links_cute.PIECE_COUNTERS:
            return False
        if (geometry.src_stride_bytes * self.world_size >= 1 << 35
                or geometry.chunk_bytes * self.world_size >= 1 << 31):
            return False
        if schedule == "auto" and geometry.chunk_bytes * self.world_size < self.chain_min_for("scatter"):
            return False
        return True

    def _reduce_scatter_chain(self, inp: torch.Tensor, out: Optional[torch.Tensor], stream: object,
                              chunk_bytes: Optional[int], src_stride_bytes: Optional[int]) -> torch.Tensor:
        """One chain reduce-scatter (``scatter_uses_chain`` holds for the arguments)."""
        with self._lock:
            self.check_health()
            geometry = _scatter_plan.scatter_geometry(inp.numel() * inp.element_size(), self.world_size, None,
                                                      chunk_bytes, src_stride_bytes)
            chunk = geometry.chunk_bytes
            if inp.data_ptr() % PACK_BYTES:
                raise ValueError("the SIRCL reduce-scatter needs a 16-byte aligned input pointer")
            if out is None:
                if chunk_bytes is None and inp.shape[0] % self.world_size == 0:
                    out = torch.empty((inp.shape[0] // self.world_size, *inp.shape[1:]), dtype=inp.dtype,
                                      device=inp.device)
                else:
                    out = torch.empty(chunk // inp.element_size(), dtype=inp.dtype, device=inp.device)
            if (out.dtype != inp.dtype or out.device != inp.device or not out.is_contiguous()
                    or out.numel() * out.element_size() != chunk or out.data_ptr() % PACK_BYTES):
                raise ValueError("out must be a 16-byte aligned contiguous tensor of one chunk in the input's dtype")
            context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
            with torch.cuda.device(self.device), context:
                capturing = torch.cuda.is_current_stream_capturing()
                launcher = self._scatter_chain_launcher(inp.dtype, capturing)
                # The partial from the higher chain indices of a middle rank's own chunk waits here
                # until the other partial arrives.
                scratch = torch.empty(chunk, dtype=torch.uint8, device=self.device)
                self._order_stream(capturing)
                launcher(inp.data_ptr(), out.data_ptr(), scratch.data_ptr(), chunk // PACK_BYTES,
                         geometry.src_stride_bytes // PACK_BYTES, self.link_chunk_for("scatter") // PACK_BYTES,
                         self._region.data_ptr() + self._link_offset, self._link_counters.data_ptr(),
                         self._piece_counters.data_ptr(), self._ctrl_base, self._poison_address, self.spin_limit)
                self._mark_stream(capturing)
            if not capturing:
                self.check_health()
            return out

    def gather_uses_chain(self, inp: torch.Tensor, dim: int = -1, *, mode: Optional[str] = None) -> bool:
        """Whether ``all_gather_large(inp, dim=dim)`` runs as one chain all-gather (the same on every rank
        for the same shape): the chain links exist, the schedule allows it, every rank's shard lands in one
        piece of the output (``dim`` is the first dimension of size above 1), and the shard is a multiple of
        16 bytes on a 16-byte-aligned tensor; with the tuning table's choice for ``mode`` applied as the op
        applies it."""
        with self._tuned_op("all_gather", inp.numel() * inp.element_size(), mode=mode, count=False):
            return self._gather_uses_chain(inp, dim)

    def _gather_uses_chain(self, inp: torch.Tensor, dim: int) -> bool:
        if not self.link_available or not self._gather_shape(inp, dim):
            return False
        shard = inp.numel() * inp.element_size()
        schedule = self._schedule(self.gather_schedule, shard * self.world_size, "gather")
        if schedule not in ("auto", "chain"):
            return False
        return schedule == "chain" or shard * self.world_size >= self.chain_min_for("gather")

    def gather_uses_ring(self, inp: torch.Tensor, dim: int = -1, *, mode: Optional[str] = None) -> bool:
        """Whether ``all_gather_large(inp, dim=dim)`` runs as one ring all-gather: schedule ``ring`` with a
        ring, an output of at least ``ring_min_for("gather")`` and the shard of a chain all-gather
        (:meth:`gather_uses_chain`); with the tuning table's choice for ``mode`` applied as the op applies
        it."""
        with self._tuned_op("all_gather", inp.numel() * inp.element_size(), mode=mode, count=False):
            return self._gather_uses_ring(inp, dim)

    def _gather_uses_ring(self, inp: torch.Tensor, dim: int) -> bool:
        if not self._gather_shape(inp, dim):
            return False
        output = inp.numel() * inp.element_size() * self.world_size
        return self._schedule(self.gather_schedule, output, "gather") == "ring"

    def _gather_shape(self, inp: torch.Tensor, dim: int) -> bool:
        """Every rank's shard lands in one piece of the output, a multiple of 16 bytes."""
        if self.max_gather_bytes <= 0 or inp.dim() == 0:
            return False
        dim = dim % inp.dim()
        shard = inp.numel() * inp.element_size()
        outer = 1
        for extent in inp.shape[:dim]:
            outer *= int(extent)
        return not (outer != 1 or shard == 0 or shard % PACK_BYTES or shard * self.world_size >= 1 << 31)

    def _launch_gather_chain(self, src: torch.Tensor, out: torch.Tensor, capturing: bool) -> None:
        """One chain all-gather: ``src`` contiguous and 16-byte aligned, ``out`` every rank's shard in rank
        order."""
        launcher = self._gather_chain_launcher(capturing)
        shard_packs = src.numel() * src.element_size() // PACK_BYTES
        self._order_stream(capturing)
        launcher(src.data_ptr(), out.data_ptr(), shard_packs, self.link_chunk_for("gather") // PACK_BYTES,
                 self._region.data_ptr() + self._link_offset, self._link_counters.data_ptr(), self._ctrl_base,
                 self._poison_address, self.spin_limit)
        self._mark_stream(capturing)

    def large_reduce_plan(self, nbytes: int, *, aligned: bool = True,
                          mode: Optional[str] = None) -> tuple[pieces.ReducePiece, ...]:
        """The ops ``all_reduce_large`` runs for a message of ``nbytes`` (16-byte-aligned tensors
        unless ``aligned`` is False) in ``mode`` (the current one when None), the tuning table's choice
        applied as the op applies it; the same on every rank."""
        with self._tuned_op("all_reduce", int(nbytes), mode=mode, count=False):
            return self._large_reduce_plan(int(nbytes), aligned)

    def _large_reduce_plan(self, nbytes: int, aligned: bool) -> tuple[pieces.ReducePiece, ...]:
        chain_from = ring_from = None
        schedule = self._schedule(self.large_schedule, int(nbytes), "reduce")
        if aligned and schedule == "ring":
            ring_from = PACK_BYTES * self.world_size
        elif self.chain_available and aligned and schedule != "pieces":
            chain_from = PACK_BYTES if schedule == "chain" else max(PACK_BYTES, self.chain_min_for("reduce"))
        return pieces.reduce_plan(int(nbytes), self.large_piece_bytes, chain_from, ring_from=ring_from,
                                  ring_world=self.world_size)

    def _schedule(self, schedule: str, nbytes: int, collective: str) -> str:
        """A schedule as it runs for ``collective`` of ``nbytes`` (the all-reduce's message, the
        all-gather's output, the reduce-scatter's input): ``ring`` runs as ``auto`` without a ring and below
        ``ring_min_for(collective)``. The same on every rank for the same sizes."""
        if schedule == "ring" and (not self.ring_available or nbytes < self.ring_min_for(collective)):
            return "auto"
        return schedule

    def _launch_chain(self, launcher: Callable[..., None], inp: torch.Tensor, out: torch.Tensor,
                      capturing: bool) -> None:
        """One chain op: ``inp`` and ``out`` contiguous, 16-byte aligned, a multiple of 16 bytes."""
        packs = inp.numel() * inp.element_size() // PACK_BYTES
        a_packs, b_packs = proto.chain_halves(packs)
        self._order_stream(capturing)
        launcher(inp.data_ptr(), out.data_ptr(), a_packs, b_packs, self.chain_chunk_bytes // PACK_BYTES,
                 self._region.data_ptr() + self._chain_offset, self._chain_counters.data_ptr(), self._ctrl_base,
                 self._poison_address, self.spin_limit, self._kernel_trace_address())
        self._mark_stream(capturing)

    def _reduce_launcher(self, algorithm: str, dtype: torch.dtype, capturing: bool) -> Callable[..., None]:
        if algorithm == "oneshot":
            return self._oneshot_launcher(dtype, capturing)
        if algorithm == "twoshot":
            return self._twoshot_launcher(dtype, capturing)
        raise RuntimeError(f"the {algorithm} all-reduce is unavailable: this build has no {algorithm} kernel")

    def _tiled_gather_launcher(self, capturing: bool) -> Callable[..., None]:
        launcher = self._launchers.get(("gather-tiles",))
        if launcher is None:
            if capturing:
                raise RuntimeError("SIRCL all-gather was not prepared before CUDA graph capture; call prepare()")
            launcher = _allgather_cute.get_tiled_launcher(
                self.world_size, self.rank, self._threads, self._layout.slots, self._layout.flag_stride,
                self.lane_count, self.device.index, one_block_polls=self._one_block_polls,
            )
            self._launchers[("gather-tiles",)] = launcher
        return launcher

    def _all_gather_launcher(self, capturing: bool) -> Callable[..., None]:
        if self._gather_launcher is None:
            if capturing:
                raise RuntimeError("SIRCL all-gather was not prepared before CUDA graph capture; call prepare()")
            self._gather_launcher = _allgather_cute.get_launcher(
                self.world_size, self.rank, self._threads, self._layout.slots, self._layout.flag_stride,
                self.lane_count, self.device.index, one_block_polls=self._one_block_polls,
            )
        return self._gather_launcher

    # -- counters and streams -------------------------------------------------------------

    def _counter_addresses(self, blocks: int) -> tuple[int, int]:
        stage = self._epoch_address + 4 * self._counter_layout.stage_word(int(blocks))
        tail = self._epoch_address + 4 * self._counter_layout.tail_word(int(blocks))
        return stage, tail

    def _phase_counters(self, blocks: int) -> tuple[int, int]:
        first = self._epoch_address + 4 * self._counter_layout.phase_word(int(blocks), 1)
        return first, 4 * self._counter_classes

    def _order_stream(self, capturing: bool) -> None:
        """Order this launch after the previous one. Outside a capture, a launch on another stream than the
        previous launch's first waits for an event recorded on that stream (everything queued there so
        far), and a launch on the same stream needs nothing; inside a capture, every launch of one capture
        must use one stream."""
        current = torch.cuda.current_stream(self.device)
        if capturing:
            capture_id = _capture_id(current)
            if capture_id != self._capture_id:
                self._capture_id = capture_id
                self._capture_stream = current
            elif current != self._capture_stream:
                raise RuntimeError("SIRCL collectives of one CUDA graph capture must use one stream")
            return
        last = self._last_stream
        if last is not None and current != last:
            self._stream_event.record(last)
            current.wait_event(self._stream_event)
        self._last_stream = current

    def _mark_stream(self, capturing: bool) -> None:
        """After a launch: nothing to record (:meth:`_order_stream` orders the next launch)."""
        return None

    @contextmanager
    def capture(self, stream: object = None, *, channel_id: Optional[str] = None):
        """CUDA graph capture context: drops the cross-stream event, resets capture tracking on exit."""
        self._last_stream = None
        self._capture_stream = None
        self._capture_id = 0
        try:
            yield self
        finally:
            self._capture_stream = None
            self._capture_id = 0
            self._last_stream = None

    # -- scratch ------------------------------------------------------------------------------

    def _aligned_scratch(self, which: int, like: torch.Tensor) -> torch.Tensor:
        if self._align_buffers is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("SIRCL alignment scratch must be allocated before CUDA graph capture; "
                                   "call prepare()")
            # Large-message ops may exceed the capacity; the scratch holds either.
            size = max(self.max_size, self.large_piece_bytes)
            self._align_buffers = (torch.empty(size, dtype=torch.uint8, device=self.device),
                                   torch.empty(size, dtype=torch.uint8, device=self.device))
        nbytes = like.numel() * like.element_size()
        return self._align_buffers[which][:nbytes].view(like.dtype).view(like.shape)

    def _gather_scratch(self, padded: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._gather_buffers is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("SIRCL all-gather scratch must be allocated before CUDA graph capture; "
                                   "call prepare(padded_gather=True)")
            capacity = (self.max_gather_bytes + PACK_BYTES - 1) // PACK_BYTES * PACK_BYTES
            self._gather_buffers = (
                torch.empty(capacity, dtype=torch.uint8, device=self.device),
                torch.empty(self.world_size * capacity, dtype=torch.uint8, device=self.device),
            )
        staged, gathered = self._gather_buffers
        return staged[:padded], gathered[: self.world_size * padded]

    # -- all-reduce ----------------------------------------------------------------------------

    def all_reduce(self, inp: torch.Tensor, *, out: Optional[torch.Tensor] = None, stream: object = None,
                   channel_id: Optional[str] = None, peer_input_ptrs: Optional[Sequence[int]] = None,
                   algorithm: Optional[str] = None) -> torch.Tensor:
        """Sum ``inp`` over the session's ranks into ``out`` (allocated like ``inp`` when omitted).

        Accepts eligible messages up to the capacity ``max_size``; every rank
        must pass the same ``algorithm``. A poisoned session raises.
        """
        del channel_id, peer_input_ptrs
        with self._lock:
            call = self._profile.begin("all_reduce") if self._profile is not None else None
            self.check_health()
            if not self._allreduce_eligible(inp, self.max_size):
                raise ValueError("input is not eligible for the SIRCL all-reduce")
            self._check_reduce_out(inp, out)
            if call is not None:
                call.mark("checked")
            nbytes = inp.numel() * inp.element_size()
            context = torch.cuda.stream(stream) if stream is not None else _NO_CONTEXT
            device = _NO_CONTEXT if torch.cuda.current_device() == self.device.index else torch.cuda.device(self.device)
            with device, context:
                capturing = torch.cuda.is_current_stream_capturing()
                tuned = (_NO_CONTEXT if algorithm is not None or self._tuning is None
                         else self._tuned_op("all_reduce", nbytes))
                with tuned:
                    chosen = algorithm or self.select_algorithm(nbytes)
                    if chosen not in ALGORITHMS or not self._available[chosen]:
                        raise ValueError(f"SIRCL all-reduce algorithm {chosen!r} is unavailable here")
                    launcher = self._reduce_launcher(chosen, inp.dtype, capturing)
                    if out is None:
                        out = torch.empty_like(inp)
                    if call is not None:
                        call.mark("dispatched")
                        self._profiled = None if capturing else call
                    try:
                        self._launch_reduce(chosen, launcher, inp, out, capturing)
                    finally:
                        self._profiled = None
                    if call is not None:
                        call.mark("launched")
            if call is not None and not capturing:
                self._profile.end(call, chosen, nbytes)
            return out

    @staticmethod
    def _check_reduce_out(inp: torch.Tensor, out: Optional[torch.Tensor]) -> None:
        if out is not None and (out.shape != inp.shape or out.dtype != inp.dtype
                                or out.device != inp.device or not out.is_contiguous()):
            raise ValueError("out must be a contiguous tensor on the input's device matching the input")

    def _launch_reduce(self, algorithm: str, launcher: Callable[..., None], inp: torch.Tensor,
                       out: torch.Tensor, capturing: bool) -> None:
        """One all-reduce op: ``inp`` and ``out`` contiguous, a multiple of 16 bytes, within one slot."""
        nbytes = inp.numel() * inp.element_size()
        packs = nbytes // PACK_BYTES
        src = inp
        if inp.data_ptr() % PACK_BYTES:
            src = self._aligned_scratch(0, inp)
            src.copy_(inp)
        dst = out if out.data_ptr() % PACK_BYTES == 0 else self._aligned_scratch(1, out)
        self._order_stream(capturing)
        self._launch_begin()
        if algorithm == "twoshot":
            grid = self.launch_grid("twoshot", packs)
            stage, tail = self._counter_addresses(grid)
            phase, _ = self._phase_counters(grid)
            launcher(src.data_ptr(), dst.data_ptr(), packs, nbytes, self._recv_base, self._flag_base,
                     self._send_base, self._ctrl_base, self._slot_bytes, self._epoch_address, stage, phase,
                     tail, self._poison_address, self.spin_limit, grid, 0, arrival_address=self._arrival_address)
        else:
            grid = self.launch_grid("oneshot", packs)
            stage, tail = self._counter_addresses(grid)
            launcher(src.data_ptr(), dst.data_ptr(), packs, nbytes, self._recv_base, self._flag_base,
                     self._send_base, self._ctrl_base, self._slot_bytes, self._epoch_address, stage, tail,
                     self._poison_address, self.spin_limit, grid, 0, 1 if capturing else 0,
                     arrival_address=self._arrival_address)
        self._launch_end()
        if dst is not out:
            out.copy_(dst)
        self._mark_stream(capturing)

    @_whole_call("all_reduce_large")
    def all_reduce_large(self, inp: torch.Tensor, *, out: Optional[torch.Tensor] = None,
                         stream: object = None) -> torch.Tensor:
        """Sum a message of any size over the session's ranks.

        :meth:`large_reduce_plan` names the ops. On a chain of cable neighbors
        (``chain_available``) a message from ``chain_min_for("reduce")`` on
        (``SIRCL_LARGE_SCHEDULE``: ``auto``; ``chain`` always, ``pieces`` never)
        is one chain op: every rank stores identical bits, half A summed in
        chain order and half B in reverse chain order with one rounding per hop
        (they can differ in the last place from the one-shot result). Otherwise
        ops of at most ``large_piece_bytes`` (default the larger of 4 MiB and the
        capacity), each the algorithm ``select_algorithm`` names for its size,
        with the one-shot bits. A tail of fewer than 16 bytes travels
        zero-padded as a one-shot op. Capturable on one stream after
        ``prepare``.
        """
        with self._lock:
            self.check_health()
            if (inp.dtype not in SUPPORTED_DTYPES or not inp.is_cuda or inp.device != self.device
                    or inp.is_sparse):
                raise ValueError("input is not eligible for the SIRCL large all-reduce: a dense float16, "
                                 "bfloat16 or float32 tensor on the session's device")
            if out is not None and (out.shape != inp.shape or out.dtype != inp.dtype
                                    or out.device != inp.device or not out.is_contiguous()):
                raise ValueError("out must be a contiguous tensor on the input's device matching the input")
            context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
            with torch.cuda.device(self.device), context, self._tuned_op("all_reduce", inp.numel() * inp.element_size()):
                capturing = torch.cuda.is_current_stream_capturing()
                src = inp if inp.is_contiguous() else inp.contiguous()
                if out is None:
                    out = torch.empty_like(src, memory_format=torch.contiguous_format)
                item = src.element_size()
                aligned = src.data_ptr() % PACK_BYTES == 0 and out.data_ptr() % PACK_BYTES == 0
                plan = self.large_reduce_plan(src.numel() * item, aligned=aligned)
                launchers = {}
                for piece in plan:
                    if piece.chain:
                        launchers["chain"] = self._chain_launcher(src.dtype, capturing)
                        continue
                    if piece.ring:
                        launchers["ring"] = self._ring_launcher("reduce", src.dtype, capturing)
                        continue
                    size = PACK_BYTES if piece.padded else piece.nbytes
                    name = self.select_algorithm(size)
                    if name not in launchers:
                        launchers[name] = self._reduce_launcher(name, src.dtype, capturing)
                flat_in, flat_out = src.reshape(-1), out.view(-1)
                for piece in plan:
                    first, count = piece.offset // item, piece.nbytes // item
                    if piece.chain:
                        self._launch_chain(launchers["chain"], flat_in[first:first + count],
                                           flat_out[first:first + count], capturing)
                        continue
                    if piece.ring:
                        chunk = piece.nbytes // self.world_size
                        self._launch_ring(launchers["ring"], flat_in[first:first + count],
                                          flat_out[first:first + count], chunk, chunk, capturing,
                                          self.link_chunk_for("reduce"))
                        continue
                    name = self.select_algorithm(PACK_BYTES if piece.padded else piece.nbytes)
                    if piece.padded:
                        staged = self._aligned_scratch(0, self._region[:PACK_BYTES]).view(src.dtype)
                        result = self._aligned_scratch(1, self._region[:PACK_BYTES]).view(src.dtype)
                        staged.zero_()
                        staged[:count].copy_(flat_in[first:first + count])
                        self._launch_reduce(name, launchers[name], staged, result, capturing)
                        flat_out[first:first + count].copy_(result[:count])
                    else:
                        self._launch_reduce(name, launchers[name], flat_in[first:first + count],
                                            flat_out[first:first + count], capturing)
            if not capturing:
                self.check_health()
            return out

    # -- all-gather ----------------------------------------------------------------------------

    def should_all_gather(self, inp: torch.Tensor, dim: int = -1) -> bool:
        """Eligible: contiguous plain CUDA tensor gathered along dimension 0 or the last one, up to the capacity."""
        self._require_open()
        if self.max_gather_bytes <= 0 or not inp.is_cuda or inp.dim() == 0:
            return False
        if inp.device != self.device or not inp.is_contiguous():
            return False
        if inp.is_complex() or inp.is_sparse or inp.dtype == torch.bool:
            return False
        dim = dim + inp.dim() if dim < 0 else dim
        if dim not in (0, inp.dim() - 1):
            return False
        nbytes = inp.numel() * inp.element_size()
        return 0 < nbytes <= self.max_gather_bytes

    def all_gather(self, inp: torch.Tensor, *, dim: int = -1, out: Optional[torch.Tensor] = None,
                   stream: object = None) -> torch.Tensor:
        """Concatenate every rank's ``inp`` along ``dim`` in rank order."""
        with self._lock:
            call = self._profile.begin("all_gather") if self._profile is not None else None
            self.check_health()
            if not self.should_all_gather(inp, dim):
                raise ValueError("input is not eligible for the SIRCL all-gather")
            dim = dim + inp.dim() if dim < 0 else dim
            shape = list(inp.shape)
            shape[dim] *= self.world_size
            if out is not None and (list(out.shape) != shape or out.dtype != inp.dtype
                                    or out.device != inp.device or not out.is_contiguous()):
                raise ValueError("out must be a contiguous tensor of the gathered shape")
            nbytes = inp.numel() * inp.element_size()
            if call is not None:
                call.mark("checked")
            context = torch.cuda.stream(stream) if stream is not None else _NO_CONTEXT
            device = _NO_CONTEXT if torch.cuda.current_device() == self.device.index else torch.cuda.device(self.device)
            tuned = _NO_CONTEXT if self._tuning is None else self._tuned_op("all_gather", nbytes)
            with device, context, tuned:
                capturing = torch.cuda.is_current_stream_capturing()
                if capturing:
                    call = None
                launcher = self._all_gather_launcher(capturing)
                row_bytes = nbytes if dim == 0 else inp.shape[-1] * inp.element_size()
                direct = (nbytes % PACK_BYTES == 0 and row_bytes % PACK_BYTES == 0
                          and inp.data_ptr() % PACK_BYTES == 0
                          and (out is None or out.data_ptr() % PACK_BYTES == 0))
                if direct:
                    if out is None:
                        out = torch.empty(shape, dtype=inp.dtype, device=inp.device)
                    if call is not None:
                        call.mark("dispatched")
                        self._profiled = call
                    self._order_stream(capturing)
                    self._launch_gather(launcher, inp.data_ptr(), out.data_ptr(), nbytes, row_bytes // PACK_BYTES)
                    self._mark_stream(capturing)
                    if call is not None:
                        call.mark("launched")
                        self._profile.end(call, "direct", nbytes)
                    return out
                padded = (nbytes + PACK_BYTES - 1) // PACK_BYTES * PACK_BYTES
                staged, gathered = self._gather_scratch(padded)
                staged[:nbytes].copy_(inp.reshape(-1).view(torch.uint8))
                if call is not None:
                    call.mark("dispatched")
                    self._profiled = call
                self._order_stream(capturing)
                self._launch_gather(launcher, staged.data_ptr(), gathered.data_ptr(), padded, padded // PACK_BYTES)
                self._mark_stream(capturing)
                if call is not None:
                    call.mark("launched")
                stacked = (gathered.view(self.world_size, padded)[:, :nbytes].reshape(-1)
                           .view(inp.dtype).reshape(self.world_size, *inp.shape))
                result = stacked.movedim(0, dim).reshape(shape)
                if out is None:
                    out = result.clone(memory_format=torch.contiguous_format)
                else:
                    out.copy_(result)
                if call is not None:
                    self._profile.end(call, "padded", nbytes)
                return out

    def _launch_begin(self) -> None:
        """Mark the start of the profiled call's kernel launch (and record its device event)."""
        call = self._profiled
        if call is not None:
            if self._profile.gpu:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                call.events = (start, end)
            call.mark("launch_begin")

    def _launch_end(self) -> None:
        """Mark the end of the profiled call's kernel launch; later launches belong to no call."""
        call = self._profiled
        if call is not None:
            call.mark("launch_end")
            if call.events is not None:
                call.events[1].record()
            self._profiled = None

    def call_profile(self, reset: bool = False) -> Optional[dict[str, Any]]:
        """The summary of the profiled eager calls (``SIRCL_CALL_PROFILE``; :mod:`sparkring_sircl.callprofile`),
        None when profiling is off; ``reset`` drops the calls kept. Synchronizes the device when the
        profile holds device events."""
        if self._profile is None:
            return None
        if self._profile.gpu:
            torch.cuda.synchronize(self.device)
        summary = self._profile.summary(lambda start, end: start.elapsed_time(end))
        if reset:
            self._profile.reset()
        return summary

    def _launch_gather(self, launcher, input_address: int, output_address: int, nbytes: int,
                       row_packs: int) -> None:
        grid = self.launch_grid("gather", nbytes // PACK_BYTES)
        stage, tail = self._counter_addresses(grid)
        self._launch_begin()
        launcher(input_address, output_address, nbytes // PACK_BYTES, nbytes, row_packs, self._recv_base,
                 self._flag_base, self._send_base, self._ctrl_base, self._slot_bytes, self._epoch_address,
                 stage, tail, self._poison_address, self.spin_limit, grid, arrival_address=self._arrival_address)
        self._launch_end()

    @_whole_call("all_gather_large")
    def all_gather_large(self, inp: torch.Tensor, *, dim: int = -1, out: Optional[torch.Tensor] = None,
                         stream: object = None) -> torch.Tensor:
        """Concatenate every rank's ``inp`` along any dimension ``dim``, at any size.

        The shard is viewed as rows of bytes (:func:`sparkring_sircl.pieces.gather_view`)
        and moved in ops of at most ``gather_piece_bytes``
        (:func:`sparkring_sircl.pieces.gather_plan`). Rows that are 16-byte
        multiples on 16-byte-aligned tensors go straight from the input to the
        output (tiled kernel); other shapes go through the padded gather
        scratch. Bytes are copied unchanged. Capturable on one stream after
        ``prepare`` (with ``padded_gather=True`` for the padded shapes).
        """
        with self._lock:
            self.check_health()
            if self.max_gather_bytes <= 0:
                raise RuntimeError("the SIRCL all-gather is disabled in this session (max_gather_bytes=0)")
            if not inp.is_cuda or inp.device != self.device or inp.dim() == 0 or inp.is_sparse:
                raise ValueError("input is not eligible for the SIRCL large all-gather: a dense tensor with "
                                 "at least one dimension on the session's device")
            dim = dim % inp.dim()
            shape = list(inp.shape)
            shape[dim] *= self.world_size
            if out is not None and (list(out.shape) != shape or out.dtype != inp.dtype
                                    or out.device != inp.device or not out.is_contiguous()):
                raise ValueError("out must be a contiguous tensor of the gathered shape")
            context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
            with torch.cuda.device(self.device), context, self._tuned_op("all_gather", inp.numel() * inp.element_size()):
                capturing = torch.cuda.is_current_stream_capturing()
                src = inp if inp.is_contiguous() else inp.contiguous()
                if out is None:
                    out = torch.empty(shape, dtype=inp.dtype, device=inp.device)
                outer, inner = pieces.gather_view(tuple(src.shape), dim, src.element_size())
                if outer * inner == 0:
                    return out
                links_aligned = src.data_ptr() % PACK_BYTES == 0 and out.data_ptr() % PACK_BYTES == 0
                if self.gather_uses_ring(src, dim) and links_aligned:
                    shard = src.numel() * src.element_size()
                    self._launch_ring(self._ring_launcher("gather", None, capturing), src, out, shard, shard,
                                      capturing, self.link_chunk_for("gather"))
                    if not capturing:
                        self.check_health()
                    return out
                if self.gather_uses_chain(src, dim) and links_aligned:
                    self._launch_gather_chain(src, out, capturing)
                    if not capturing:
                        self.check_health()
                    return out
                world = self.world_size
                src_bytes = src.reshape(-1).view(torch.uint8)
                out_bytes = out.view(-1).view(torch.uint8)
                aligned = (inner % PACK_BYTES == 0 and src.data_ptr() % PACK_BYTES == 0
                           and out.data_ptr() % PACK_BYTES == 0)
                plan = pieces.gather_plan(outer, inner, self.gather_piece_bytes)
                if aligned:
                    launcher = self._tiled_gather_launcher(capturing)
                    inner_packs = inner // PACK_BYTES
                    for tile in plan:
                        packs = tile.nbytes // PACK_BYTES
                        grid = self.launch_grid("gather-tiles", packs)
                        stage, tail = self._counter_addresses(grid)
                        self._order_stream(capturing)
                        launcher(src.data_ptr() + tile.source_offset(inner),
                                 out.data_ptr() + tile.output_offset(inner, world), packs, tile.nbytes,
                                 tile.cols // PACK_BYTES, inner_packs, world * inner_packs, inner_packs,
                                 self._recv_base, self._flag_base, self._send_base, self._ctrl_base,
                                 self._slot_bytes, self._epoch_address, stage, tail, self._poison_address,
                                 self.spin_limit, grid, arrival_address=self._arrival_address)
                        self._mark_stream(capturing)
                else:
                    launcher = self._all_gather_launcher(capturing)
                    out_rows = out_bytes.view(outer, world, inner)
                    for tile in plan:
                        start = tile.source_offset(inner)
                        size = pieces.padded(tile.nbytes)
                        staged, gathered = self._gather_scratch(size)
                        staged[:tile.nbytes].copy_(src_bytes[start:start + tile.nbytes])
                        self._order_stream(capturing)
                        grid = self.launch_grid("gather", size // PACK_BYTES)
                        stage, tail = self._counter_addresses(grid)
                        launcher(staged.data_ptr(), gathered.data_ptr(), size // PACK_BYTES, size,
                                 size // PACK_BYTES, self._recv_base, self._flag_base, self._send_base,
                                 self._ctrl_base, self._slot_bytes, self._epoch_address, stage, tail,
                                 self._poison_address, self.spin_limit, grid,
                                 arrival_address=self._arrival_address)
                        self._mark_stream(capturing)
                        parts = gathered.view(world, size)[:, :tile.nbytes]
                        if tile.cols == inner:
                            rows = parts.reshape(world, tile.rows, inner).transpose(0, 1)
                            out_rows[tile.row:tile.row + tile.rows].copy_(rows)
                        else:
                            out_rows[tile.row, :, tile.col:tile.col + tile.cols].copy_(parts)
            if not capturing:
                self.check_health()
            return out

    # -- scatter collectives: reduce-scatter and all-to-all (``_scatter_ops``) ------------------

    def should_reduce_scatter(self, inp: torch.Tensor, *, chunk_bytes: Optional[int] = None,
                              src_stride_bytes: Optional[int] = None) -> bool:
        """Eligible (rank-invariant): float16, bfloat16 or float32 in a scatter geometry, any size."""
        if (self.scatter_uses_ring(inp, chunk_bytes=chunk_bytes, src_stride_bytes=src_stride_bytes)
                or self.scatter_uses_chain(inp, chunk_bytes=chunk_bytes, src_stride_bytes=src_stride_bytes)):
            return True
        return _scatter_ops.should_reduce_scatter(self, inp, chunk_bytes=chunk_bytes,
                                                  src_stride_bytes=src_stride_bytes)

    def should_all_to_all(self, inp: torch.Tensor, *, chunk_bytes: Optional[int] = None,
                          src_stride_bytes: Optional[int] = None) -> bool:
        """Eligible (rank-invariant): any plain dtype in a scatter geometry, any size."""
        return _scatter_ops.should_all_to_all(self, inp, chunk_bytes=chunk_bytes, src_stride_bytes=src_stride_bytes)

    def reduce_scatter(self, inp: torch.Tensor, *, out: Optional[torch.Tensor] = None, stream: object = None,
                       chunk_bytes: Optional[int] = None,
                       src_stride_bytes: Optional[int] = None) -> torch.Tensor:
        """Chunk ``rank`` of the sum into ``out``. On a chain (``scatter_uses_chain``): one chain
        reduce-scatter, each owner's rows ``round((L + x) + R)`` (``references.chain_reduce_scatter``).
        Otherwise the rank-ordered float32 sum, rounded once, in ops of at most ``large_piece_bytes``
        (the one-shot and two-shot bits at every size). With a tuning table, its choice for the input's
        bytes and the current mode."""
        with self._tuned_op("reduce_scatter", inp.numel() * inp.element_size()):
            if self.scatter_uses_ring(inp, chunk_bytes=chunk_bytes, src_stride_bytes=src_stride_bytes):
                return self._reduce_scatter_ring(inp, out, stream, chunk_bytes, src_stride_bytes)
            if self.scatter_uses_chain(inp, chunk_bytes=chunk_bytes, src_stride_bytes=src_stride_bytes):
                return self._reduce_scatter_chain(inp, out, stream, chunk_bytes, src_stride_bytes)
            return _scatter_ops.reduce_scatter(self, inp, out=out, stream=stream, chunk_bytes=chunk_bytes,
                                               src_stride_bytes=src_stride_bytes)

    def all_to_all(self, inp: torch.Tensor, out: torch.Tensor, *, stream: object = None,
                   chunk_bytes: Optional[int] = None, src_stride_bytes: Optional[int] = None,
                   dst_stride_bytes: Optional[int] = None) -> torch.Tensor:
        """Chunk ``p`` of ``inp`` to rank ``p``; rank ``s``'s chunk at ``s * dst_stride_bytes`` of ``out``
        (with a tuning table, in its launch grid for the input's bytes and the current mode)."""
        with self._tuned_op("all_to_all", inp.numel() * inp.element_size()):
            return _scatter_ops.all_to_all(self, inp, out, stream=stream, chunk_bytes=chunk_bytes,
                                           src_stride_bytes=src_stride_bytes, dst_stride_bytes=dst_stride_bytes)

    # -- health and diagnostics -------------------------------------------------------------------

    def check_health(self) -> None:
        """Raise when the progress thread failed or a flag wait timed out (two host reads)."""
        if self._proxy is not None and self._proxy.failed():
            raise RuntimeError(f"SIRCL progress thread failed on rank {self.rank}: {self._proxy.error()}")
        failed_seq = int(self._ctrl_np[proto.Ctrl.ERROR_SEQ]) & 0xFFFFFFFF
        if failed_seq:
            peer = int(self._ctrl_np[proto.Ctrl.MISSING_PEER])
            lane = int(self._ctrl_np[proto.Ctrl.MISSING_LANE])
            kind = int(self._ctrl_np[proto.Ctrl.ERROR_KIND])
            if kind == proto.ErrorKind.CHAIN_CHUNK:
                what = f"chain chunk {failed_seq - 1} from rank {peer} lane {lane}"
            elif kind == proto.ErrorKind.CHAIN_SLOT:
                what = f"its progress thread to free the chain send slot of chunk {failed_seq - 1}"
            else:
                what = f"rank {peer} lane {lane} at sequence {failed_seq}"
            raise RuntimeError(
                f"SIRCL collective on rank {self.rank} timed out waiting for {what} (wait limit "
                f"{self.wait_limit_s:g} s, {self.wait_regime} regime); the session is poisoned (later launches "
                "do nothing) and rank data is untrustworthy"
            )
        if self._closed:
            raise RuntimeError("SIRCL session is closed")

    @property
    def poisoned(self) -> bool:
        proxy_failed = self._proxy is not None and self._proxy.failed()
        return proxy_failed or int(self._ctrl_np[proto.Ctrl.ERROR_SEQ]) != 0

    def stats(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "world_size": self.world_size,
            "rank": self.rank,
            "devices": list(self.hca_names),
            "gid_indices": list(self.gid_indices),
            "peer_routes": routes_mod.format_peer_routes(
                {peer: devices for peer, devices in enumerate(self.peer_routes) if devices}),
            "lane_count": self.lane_count,
            "max_size": self.max_size,
            "dispatch_limit_bytes": self.dispatch_limit_bytes,
            "max_gather_bytes": self.max_gather_bytes,
            "slot_bytes": self._slot_bytes,
            "epoch": int(self._counters[0].item()),
            "error_seq": int(self._ctrl_np[proto.Ctrl.ERROR_SEQ]) & 0xFFFFFFFF,
            "error_peer": int(self._ctrl_np[proto.Ctrl.MISSING_PEER]),
            "error_lane": int(self._ctrl_np[proto.Ctrl.MISSING_LANE]),
            "ctrl_seq": int(self._ctrl_np[proto.Ctrl.DOORBELL]) & 0xFFFFFFFF,
            "spin_limit": self.spin_limit,
            "spin_limit_s": (round(self.spin_limit / self.poll_rate_per_s, 6)
                             if self.poll_rate_per_s else None),
            "wait_regime": self.wait_regime,
            "wait_limit_s": self.wait_limit_s,
            "startup_wait_s": self.startup_wait_s,
            "serving_wait_s": self.serving_wait_s,
            "poll_rate_per_s": round(self.poll_rate_per_s) if self.poll_rate_per_s else None,
            "threads": self._threads,
            "blocks": self._blocks,
            "packs_per_thread": self._packs_per_thread,
            "flag_pollers": self.flag_pollers,
            "algorithm": self.algorithm,
            "large_algorithm": self.large_algorithm,
            "oneshot_max_bytes": self.oneshot_max_bytes,
            "oneshot_max_source": self.oneshot_max_source,
            "swing_above_bytes": self.swing_above_bytes,
            "multi_phase": self.multi_phase,
            "scatter": self.scatter_available,
            "algorithms_available": [name for name in ALGORITHMS if self._available[name]],
            "layout": (self._layout_identity_object.identity()
                       if self._layout_identity_object is not None else None),
            "lane_check": dict(self._lane_check),
            "post_order": self._post_order_name,
            "post_order_peers": list(self.post_order_peers),
            "large_blocks": self._large_blocks,
            "large_piece_bytes": self.large_piece_bytes,
            "gather_piece_bytes": self.gather_piece_bytes,
            "relay_safe_bytes": self.relay_safe_bytes,
            "scatter_op_bytes": _scatter_ops.op_bytes(self) if self.scatter_available else None,
            "operations": {
                "all_reduce": [name for name in ALGORITHMS if self._available[name]],
                "all_gather": self.max_gather_bytes > 0,
                "reduce_scatter": self.scatter_available,
                "all_to_all": self.scatter_available,
            },
            "forward_chunk_bytes": self.forward_chunk_bytes,
            "forward_windows": {str(peer): list(row) for peer, row in enumerate(self._forward_table)
                                if any(row)},
            "large_schedule": self.large_schedule,
            "chain_available": self.chain_available,
            "chain_order": list(self.chain_order) if self.chain_order is not None else None,
            "chain_slots": self.chain_slots,
            "chain_slot_bytes": self.chain_slot_bytes,
            "chain_chunk_bytes": self.chain_chunk_bytes,
            "chain_blocks": self.chain_blocks,
            "chain_unroll": self.chain_unroll,
            "chain_min_bytes": self.chain_min_bytes,
            "ring_min_bytes": self.ring_min_bytes,
            "chain_mins": dict(self._chain_mins),
            "ring_mins": dict(self._ring_mins),
            "gather_schedule": self.gather_schedule,
            "scatter_schedule": self.scatter_schedule,
            "event_trace": self.event_trace,
            "link_available": self.link_available,
            "ring_available": self.ring_available,
            "ring_window_bytes": self.ring_window_bytes,
            "ring_problem": self.ring_problem,
            "link_slots": self.link_slots,
            "link_slot_bytes": self.link_slot_bytes,
            "link_chunk_bytes": self.link_chunk_bytes,
            "link_chunks": {collective: self.link_chunk_for(collective) for collective in LINK_COLLECTIVES},
            "ring_stagger": self.ring_stagger,
            "ring_gather_stagger": self.ring_gather_stagger,
            "link_blocks": self.link_blocks,
            "link_unroll": self.link_unroll,
            "tuning": self._tuning_stats(),
        }
        if self._proxy is not None and not self._closed:
            info.update(self._proxy.stats())
        return info

    # -- event trace ----------------------------------------------------------------------

    def _kernel_trace_address(self) -> int:
        """The chain kernel's trace buffer (allocated on first use), or 0 without a trace."""
        if not self.event_trace:
            return 0
        buffer = getattr(self, "_kernel_trace", None)
        if buffer is None:
            words = (_chain_cute.TRACE_HEADER_BYTES + self.event_trace * _chain_cute.TRACE_RECORD_BYTES) // 8
            buffer = torch.zeros(words, dtype=torch.int64, device=self.device)
            self._kernel_trace = buffer
        return buffer.data_ptr()

    def _gpu_clock_offset(self, rounds: int = CLOCK_PROBE_ROUNDS) -> tuple[int, int]:
        """``(offset, error)`` in nanoseconds with CLOCK_REALTIME = ``%globaltimer`` + offset,
        within error: the GPU stamps its timer when it sees a word the host writes, and the round
        with the shortest host round trip bounds the offset."""
        probe = _timed_wait.get_clock_probe(self.device.index)
        buffer = torch.zeros(rounds + 1, dtype=torch.int64, pin_memory=True)
        words = buffer.numpy()
        flag = words[:1].view("uint32")
        with torch.cuda.device(self.device):
            torch.cuda.current_stream().synchronize()
            probe(buffer.data_ptr(), buffer.data_ptr() + 8, rounds, 1_000_000)
            best = None
            for r in range(rounds):
                sent = time.clock_gettime_ns(time.CLOCK_REALTIME)
                flag[0] = r + 1
                while int(words[1 + r]) == 0:
                    if time.clock_gettime_ns(time.CLOCK_REALTIME) - sent > 2_000_000_000:
                        raise RuntimeError("SIRCL clock probe: the GPU did not answer within 2 s")
                seen = time.clock_gettime_ns(time.CLOCK_REALTIME)
                stamp = int(words[1 + r])
                if best is None or seen - sent < best[1]:
                    best = ((sent + seen) // 2 - stamp, seen - sent)
            torch.cuda.current_stream().synchronize()
        return best[0], (best[1] + 1) // 2

    def event_trace_records(self, reset: bool = True) -> dict[str, Any]:
        """The event trace since the last call (``SIRCL_EVENT_TRACE`` records; empty without one):
        native records of the chain streams and links, and the chain kernel's records mapped to
        CLOCK_REALTIME (``offset_ns`` within ``offset_error_ns``). Call it when the session's
        stream is idle; ``reset`` restarts the kernel's buffer. Records are
        ``(ns, source, event, stream, value)`` with ``source`` ``native`` or ``kernel`` and
        ``event`` a ``protocol.TraceEvent`` name, sorted by time."""
        self._require_open()
        result: dict[str, Any] = {"enabled": bool(self.event_trace), "clock": "CLOCK_REALTIME", "records": [],
                                  "lost": {"native": 0, "kernel": 0}}
        if not self.event_trace:
            return result
        native, lost = self._proxy.take_trace(self.event_trace)
        records = [(ns, "native", proto.TraceEvent(event).name, stream, value) for ns, event, stream, value in native]
        result["lost"]["native"] = lost
        buffer = getattr(self, "_kernel_trace", None)
        if buffer is not None:
            with torch.cuda.device(self.device):
                torch.cuda.current_stream().synchronize()
                words = buffer.cpu()
            claimed = int(words[0].item()) & 0xFFFFFFFF
            count = min(claimed, self.event_trace)
            offset, error = self._gpu_clock_offset()
            result.update(offset_ns=offset, offset_error_ns=error)
            body = words[_chain_cute.TRACE_HEADER_BYTES // 8:].view(torch.int32).reshape(-1, 4)[:count].tolist()
            for lo, hi, value, packed in body:
                stamp = ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)
                event, stream = packed & 0xFFFF, (packed >> 16) & 0xFFFF
                records.append((stamp + offset, "kernel", proto.TraceEvent(event).name, stream,
                                value & 0xFFFFFFFF))
            result["lost"]["kernel"] = claimed - count
            if reset:
                buffer.zero_()
        records.sort(key=lambda record: record[0])
        result["records"] = records
        return result

    def oneshot_trace(self, reset: bool = True) -> dict[str, Any]:
        return {"oneshot_calls": 0, "enabled": False}

    def twoshot_trace(self, reset: bool = True) -> dict[str, Any]:
        return {"calls": 0, "enabled": False}

    def close(self) -> None:
        """Synchronize the device, stop the progress thread and release the RDMA resources (idempotent)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            with contextlib.suppress(Exception):
                torch.cuda.synchronize(self.device)
            if self._profile is not None and self._profile.path:
                with contextlib.suppress(Exception):
                    self._profile.dump(lambda start, end: start.elapsed_time(end))
            if self._proxy is not None:
                self._proxy.close()
                self._proxy = None

    def __del__(self) -> None:  # pragma: no cover - defensive teardown
        # No module-global lookups: at interpreter exit the module's globals may already be None.
        try:
            self.close()
        except Exception:  # noqa: BLE001 - teardown of a collected session
            pass


AllReduce = RoceOneshotAllReduce

__all__ = [
    "ALGORITHMS", "ALGORITHM_CHOICES", "API_VERSION", "AllReduce", "DEFAULT_LARGE_BLOCKS",
    "DEFAULT_CHAIN_CHUNK_BYTES", "DEFAULT_CHAIN_MINS", "DEFAULT_RING_MINS", "DEFAULT_LARGE_PIECE_BYTES",
    "DEFAULT_MAX_GATHER_BYTES", "DEFAULT_SERVING_WAIT_S", "LARGE_SCHEDULES",
    "DEFAULT_STARTUP_WAIT_S", "WAIT_REGIMES",
    "DEFAULT_MAX_SIZE", "MAX_DEVICES", "MAX_LANES", "RoceOneshotAllReduce", "SCATTER_MODES",
    "SUPPORTED_DTYPES", "SUPPORTED_WORLD_SIZES", "agreement_failures", "default_gid_index",
    "discover_hcas", "is_supported",
]
