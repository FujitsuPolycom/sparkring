"""Point-to-point channels of one group: send, receive and batched send/receive over RDMA on DGX Spark.

A :class:`PointToPoint` is one rank's point-to-point context of one group (2
to 16 ranks): a pinned host arena that the GB10 GPU addresses at its host
pointer and every opened RDMA device registers with plain ``ibv_reg_mr``, one
reliable-connected queue pair per (peer, lane) of every channel, a native
progress thread (``p2p/_p2p_proxy.c``) and the send and receive kernels
(``p2p/_kernels.py``). It is independent of the group's collective session.

Construction is collective over ``exchange_group`` (a CPU process group, gloo
in vLLM): each rank checks its route map, opens its devices, allocates the
arena and publishes its connection record and setup record in one
all-gather. Setup fails on every rank, with one message naming each failing
rank and reason, when any rank failed locally, when a shared setting differs
from rank 0's (slots, slot bytes, chunk, kernel geometry, wait limits, the
layout, the channel table, the window table), when lane counts differ or
when lanes do not pair. Then every rank connects its lanes, the ranks exchange
verdicts, every rank proves every lane with one small write and the progress
threads start.

Contract:

- channels: every ordered pair of ``channels`` (default every pair) is a
  first-in first-out channel; a receive takes the oldest message of its
  channel not yet received, and must name that message's byte count (a
  different count fails the group's channels on both ranks);
- any tensor of any dtype on the context's device; contiguous, 16-byte
  aligned tensors of whole 16-byte packs go straight between the tensor and
  the arena, others through an aligned scratch tensor;
- :meth:`isend` and :meth:`irecv` run on the channel's own CUDA stream after
  the caller's current stream reached the call and return a :class:`P2PWork`
  whose ``wait()`` makes the caller's current stream wait (the host does not
  block) and whose ``is_completed()`` queries the transfer's event. Every
  direction of every pair owns one non-blocking stream that no other code
  receives, created on first use: a receive kernel holds its stream until the
  peer's item arrives. The GPU runs streams on ``CUDA_DEVICE_MAX_CONNECTIONS``
  hardware queues (default 8); with more streams in use in the process than
  queues, a waiting receive also holds back what follows it on its queue;
  :meth:`send` and :meth:`recv` wait at once; :meth:`batch_isend_irecv` issues
  a list of ops, each on its channel's stream, its sends before its receives;
- issue order: a rank's GPU commands share the GPU's hardware queues, so a
  rank issues the sends a peer waits for before receives that wait on that
  peer (vLLM's pipeline order does; a batch does it itself). A send of more
  than two rounds of slots (``2 * slots * slot_bytes``) waits for the
  receiver while it runs, and holds back what follows it on its queue;
- outside CUDA graph capture only (a captured call raises);
- wait limits and fail-stop as SIRCL's collective sessions: every kernel wait
  is limited by the context's wait limit (``startup`` regime, 600 s;
  :meth:`enter_serving`, 20 s); a timeout, a size mismatch or a failed RDMA
  write poisons the context on every rank of the group and
  :meth:`check_health` raises from then on;
- close: as a collective session's (:mod:`sparkring_sircl.teardown`):
  collective over the exchange group unless ``abort`` is set, it waits for
  this rank's transfers, votes every rank's health in round 1, stops the
  progress thread, waits for every rank's stop in round 2 (only after round 1
  completed) and destroys the native context; its result is kept as
  ``close_result``. A stream synchronization or destruction that fails while
  the channels wait for this rank's transfers is this rank's failure, voted in
  round 1; the arena stays allocated when a verbs object cannot be released or
  a stream's work was not shown complete.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Optional

import torch
import torch.distributed as dist

from .. import roce_gid
from .. import routes as routes_mod
from .. import teardown as teardown_mod
from ..agreement import agreement_failures
from . import _kernels, _native, budget
from .protocol import API_VERSION, CONTROL_BYTES, Control, ErrorKind, P2PLayout, describe_header, items, padded
from .settings import DEFAULT_LANE_CHECK_MS, P2PSettings, SettingError, explicit_gid_index

logger = logging.getLogger("sircl.p2p")

WAIT_REGIMES = ("startup", "serving")
PACK = 16


def _exchange(local: object, group: Any) -> list[object]:
    gathered: list[object] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, local, group=group)
    return gathered


def _device_pointer(host_ptr: int) -> int:
    from cuda.bindings import runtime as cudart

    err, ptr = cudart.cudaHostGetDevicePointer(host_ptr, 0)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaHostGetDevicePointer failed: {err}")
    return int(ptr)


def _normalize_device(device: torch.device | int | str) -> torch.device:
    if isinstance(device, int):
        device = torch.device("cuda", device)
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("SIRCL point-to-point channels run on a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _pairs(world: int, channels: Optional[Iterable[tuple[int, int]]]) -> list[list[bool]]:
    """Symmetric channel table: every pair, or the pairs given (in either order)."""
    table = [[False] * world for _ in range(world)]
    if channels is None:
        for a in range(world):
            for b in range(world):
                table[a][b] = a != b
        return table
    for a, b in channels:
        a, b = int(a), int(b)
        if a == b or not (0 <= a < world and 0 <= b < world):
            raise ValueError(f"channel ({a}, {b}) does not join two ranks of a group of {world}")
        table[a][b] = table[b][a] = True
    return table


class P2PWork:
    """The work object of one transfer: ``wait()`` orders the caller's current stream after it."""

    def __init__(self, owner: "PointToPoint", event: torch.cuda.Event, keep: tuple[Any, ...], *, kind: str,
                 peer: int, nbytes: int) -> None:
        self._owner = owner
        self._event = event
        self._keep = keep
        self.kind = kind
        self.peer = peer
        self.nbytes = nbytes

    def wait(self, timeout: Any = None) -> bool:
        torch.cuda.current_stream(self._owner.device).wait_event(self._event)
        return True

    def is_completed(self) -> bool:
        done = bool(self._event.query())
        if done:
            self._keep = ()
        return done

    def is_success(self) -> bool:
        return self.is_completed() and not self._owner.poisoned

    def exception(self) -> Optional[BaseException]:
        return None

    def synchronize(self) -> None:
        self._event.synchronize()

    def source_rank(self) -> int:
        return self.peer

    def result(self) -> list:
        return []


def _check(result: tuple, what: str) -> tuple:
    from cuda.bindings import driver as cu

    if result[0] != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what} failed: {result[0]}")
    return result[1:]


def _dedicated_stream(device: torch.device) -> tuple[torch.cuda.ExternalStream, int]:
    """A non-blocking CUDA stream of the device's primary context that no other code receives.

    Streams from ``torch.cuda.Stream()`` come from a pool of 32 per priority that
    torch hands out in turn, so a process that asks for more shares them. A
    receive kernel waits on its stream until its peer sends; on a shared
    stream it would also hold back whatever else was queued there, including
    the send it waits for. Every channel direction therefore owns its stream.
    """
    from cuda.bindings import driver as cu

    _check(cu.cuInit(0), "cuInit")
    (handle,) = _check(cu.cuDeviceGet(device.index), "cuDeviceGet")
    (context,) = _check(cu.cuDevicePrimaryCtxRetain(handle), "cuDevicePrimaryCtxRetain")
    try:
        _check(cu.cuCtxPushCurrent(context), "cuCtxPushCurrent")
        try:
            (stream,) = _check(cu.cuStreamCreate(cu.CUstream_flags.CU_STREAM_NON_BLOCKING), "cuStreamCreate")
        finally:
            _check(cu.cuCtxPopCurrent(), "cuCtxPopCurrent")
    finally:
        cu.cuDevicePrimaryCtxRelease(handle)
    raw = int(stream)
    return torch.cuda.ExternalStream(raw, device=device), raw


def _destroy_stream(raw: int) -> None:
    from cuda.bindings import driver as cu

    _check(cu.cuStreamDestroy(cu.CUstream(raw)), "cuStreamDestroy")


class _Channel:
    """One direction of one pair on this rank: its own stream and the next item of the channel."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.stream: Optional[torch.cuda.Stream] = None
        self.raw_stream = 0
        self.next_item = 0
        self.messages = 0
        self.bytes = 0

    def get_stream(self) -> torch.cuda.Stream:
        if self.stream is None:
            self.stream, self.raw_stream = _dedicated_stream(self.device)
        return self.stream

    def release(self) -> tuple[Optional[str], bool]:
        """Wait for the stream's work and destroy the stream: ``(what failed or None, whether the stream's work
        is known complete)``. The work is unknown after a failed synchronization; a failed destroy only leaks
        the stream."""
        if self.stream is None:
            return None, True
        stream, raw, self.stream, self.raw_stream = self.stream, self.raw_stream, None, 0
        problems, settled = [], True
        try:
            stream.synchronize()
        except Exception as exc:  # noqa: BLE001 - voted in round 1
            problems.append(f"stream synchronization failed: {type(exc).__name__}: {exc}")
            settled = False
        try:
            _destroy_stream(raw)
        except Exception as exc:  # noqa: BLE001 - voted in round 1
            problems.append(f"stream destruction failed: {type(exc).__name__}: {exc}")
        return "; ".join(problems) or None, settled


class PointToPoint:
    """One rank's point-to-point channels of one group; see the module docstring."""

    def __init__(
        self,
        *,
        exchange_group: Any,
        device: torch.device | int | str,
        peer_routes: Optional[Mapping[int, Sequence[str]]] = None,
        layout: routes_mod.Layout | str | None = None,
        channels: Optional[Iterable[tuple[int, int]]] = None,
        windows: Optional[Sequence[Sequence[Sequence[int]]]] = None,
        unavailable: Optional[Mapping[tuple[int, int], str]] = None,
        settings: Optional[P2PSettings] = None,
        hca_names: Optional[Sequence[str]] = None,
        gid_index: Optional[int] = None,
        lane_check_ms: int = DEFAULT_LANE_CHECK_MS,
        library: Optional[str] = None,
        start_item: int = 0,
    ) -> None:
        self.device = _normalize_device(device)
        self.rank = dist.get_rank(group=exchange_group)
        self.world_size = dist.get_world_size(group=exchange_group)
        self._group = exchange_group
        self._closed = False
        self.close_result: Optional[str] = None
        # The teardown rounds' channel (teardown.tensor_exchange); the GPU emulation binds its thread group's.
        self._teardown_exchange = teardown_mod.tensor_exchange(exchange_group)
        # Its ordinal among this rank's objects of the group: the teardown notes name the close by it.
        self._teardown_ordinal = teardown_mod.register(exchange_group, self.rank)
        self._lock = threading.Lock()
        self._native: Optional[_native.Native] = None
        self._library_path = library
        self.wait_regime = "startup"
        self._lane_check: dict[str, Any] = {"result": "not run"}
        self.channel_problems: dict[int, str] = {}
        error: Optional[str] = None
        blob = b""
        record: dict[str, Any] = {}
        try:
            self._configure(peer_routes, layout, channels, windows, unavailable, settings, hca_names, gid_index,
                            start_item)
            self._allocate()
            self._native = _native.Native(
                world_size=self.world_size, rank=self.rank, hca_names=self.hca_names,
                peer_lane_devices=self._lane_devices, lane_count=self.lane_count, gid_indices=self.gid_indices,
                channels=self._channel_table[self.rank], region_ptr=self._base, region_bytes=self.layout.total_bytes,
                slots=self.settings.slots, slot_bytes=self.settings.slot_bytes,
                library=_native.load(library) if library else None,
            )
            blob = self._native.local_blob()
            record = self._setup_record()
        except Exception as exc:  # noqa: BLE001 - reported to every rank below
            error = f"{type(exc).__name__}: {exc}"
        statuses = _exchange((error, blob, record), exchange_group)
        failures = agreement_failures(statuses, self._layout_object if error is None else None)
        if failures:
            # Every rank fails here alike and nothing was posted: no rounds.
            self.close(abort=True)
            raise RuntimeError("SIRCL point-to-point setup failed: " + "; ".join(failures))
        self._verdict("queue-pair connection", lambda: self._native.connect([s[1] for s in statuses]))

        def check_and_start() -> None:
            started = time.perf_counter()
            self._native.lane_check(lane_check_ms)
            self._lane_check = {"result": "passed",
                                "lanes": self.lane_count * sum(self._channel_table[self.rank]),
                                "seconds": round(time.perf_counter() - started, 4)}
            if any(window for row in self.windows for window in row):
                self._native.set_windows(self.windows, self.settings.chunk_bytes)
            if self._start_item:
                self._native.set_base(self._start_item)
            self._native.start()

        self._verdict("lane check", check_and_start)
        if self.rank == 0:
            logger.info("SIRCL point-to-point channels ready: %d ranks, %d channel pairs, %d lane(s), %d slots of %d "
                        "bytes, windowed lanes %d", self.world_size,
                        sum(sum(row) for row in self._channel_table) // 2, self.lane_count, self.settings.slots,
                        self.settings.slot_bytes, sum(1 for row in self.windows for window in row if window))

    # -- construction -------------------------------------------------------------------------

    def _configure(self, peer_routes, layout, channels, windows, unavailable, settings, hca_names, gid_index,
                   start_item) -> None:
        if not 2 <= self.world_size <= 16:
            raise ValueError(f"SIRCL point-to-point groups have 2 to 16 ranks, got {self.world_size}")
        self.settings = settings if settings is not None else P2PSettings.from_env()
        if isinstance(layout, str):
            layout = routes_mod.Layout.parse(layout)
        self._layout_object = layout
        self._channel_table = _pairs(self.world_size, channels)
        for (a, b), reason in (unavailable or {}).items():
            self._drop_pair(int(a), int(b), reason)
        if peer_routes is None:
            if layout is None:
                raise ValueError("point-to-point channels need a route map (peer_routes) or a layout to derive it")
            derived = routes_mod.derive_routes(layout, 2)
            peer_routes = derived.route_map(self.rank)
        routes = {int(peer): tuple(devices) for peer, devices in peer_routes.items()}
        self.lane_count = routes_mod.validate_route_map(self.rank, self.world_size, routes, layout=layout,
                                                        max_relays=self.settings.max_relays)
        self.peer_routes = tuple(routes.get(peer, ()) for peer in range(self.world_size))
        named = list(dict.fromkeys(d for _, devices in sorted(routes.items()) for d in devices))
        self.hca_names = tuple(hca_names) if hca_names is not None else tuple(named)
        missing = [device for device in named if device not in self.hca_names]
        if missing or not 1 <= len(self.hca_names) <= 4:
            raise ValueError(f"opened devices {self.hca_names} must be 1 to 4 devices including every device of the "
                             f"route map (missing {missing})")
        explicit = gid_index if gid_index is not None else explicit_gid_index()
        if explicit is not None:
            self.gid_indices = tuple(int(explicit) for _ in self.hca_names)
        else:
            self.gid_indices = tuple(roce_gid.resolve_device_gid_index(name) for name in self.hca_names)
        self.layout = P2PLayout(self.world_size, self.lane_count, self.settings.slots, self.settings.slot_bytes)
        if windows is None:
            windows = self._default_windows(layout, routes)
        table = [[[int(w) for w in lanes] for lanes in row] for row in windows]
        if len(table) != self.world_size or any(len(row) != self.world_size for row in table) or any(
                len(lanes) != self.lane_count for row in table for lanes in row):
            raise ValueError(f"the window table needs [rank][peer][lane] entries for {self.world_size} ranks and "
                             f"{self.lane_count} lanes")
        # A pair has a channel only when every lane of both directions that crosses relays has a window; the
        # table is the same on every rank, so every rank drops the same pairs.
        if layout is not None:
            derived = routes_mod.derive_routes(layout, self.lane_count)
            for a in range(self.world_size):
                for b in range(self.world_size):
                    if a == b or not self._channel_table[a][b]:
                        continue
                    for lane in derived.lanes_to(a, b):
                        if lane.relays and table[a][b][lane.lane] == 0:
                            self._drop_pair(a, b, self.channel_problems.get(b if a == self.rank else a, (
                                f"lane {lane.lane} from rank {a} to rank {b} crosses relays {list(lane.relays)} "
                                "and has no forward window")))
                            break
        self._window_table = [[[w if self._channel_table[a][b] else 0 for w in table[a][b]]
                               for b in range(self.world_size)] for a in range(self.world_size)]
        self.windows = self._window_table[self.rank]
        self._lane_devices = tuple(
            tuple(self.hca_names.index(d) for d in self.peer_routes[peer])
            if peer != self.rank and self._channel_table[self.rank][peer] else ()
            for peer in range(self.world_size))
        if not any(self._channel_table[self.rank]):
            raise ValueError(f"rank {self.rank} has no point-to-point channel left: "
                             + "; ".join(f"rank {peer}: {reason}" for peer, reason in sorted(self.channel_problems.items())))
        self._start_item = int(start_item) & 0xFFFFFFFF

    def _drop_pair(self, a: int, b: int, reason: str) -> None:
        """Remove the channels between ``a`` and ``b`` (both directions), keeping the reason for this rank."""
        if not (0 <= a < self.world_size and 0 <= b < self.world_size) or a == b:
            raise ValueError(f"pair ({a}, {b}) does not join two ranks of a group of {self.world_size}")
        self._channel_table[a][b] = self._channel_table[b][a] = False
        if self.rank in (a, b):
            self.channel_problems.setdefault(b if a == self.rank else a, reason)

    def _default_windows(self, layout, routes) -> list[list[list[int]]]:
        """Windows of this group's lanes when they are the only relayed traffic on their queues."""
        zero = [[[0] * self.lane_count for _ in range(self.world_size)] for _ in range(self.world_size)]
        if layout is None:
            return zero
        lane_set = budget.LaneSet.of("group", layout, self.lane_count,
                                     [(a, b) for a in range(self.world_size) for b in range(self.world_size)
                                      if self._channel_table[a][b]])
        table, unavailable = budget.group_windows(lane_set, self.lane_count, max_window=self.settings.window_bytes,
                                                  chunk=self.settings.chunk_bytes,
                                                  queue_bytes=self.settings.hairpin_queue_bytes)
        for (rank, peer), reason in unavailable.items():
            if self.rank in (rank, peer):
                self.channel_problems.setdefault(peer if rank == self.rank else rank, reason)
        return table

    def _allocate(self) -> None:
        native_layout = _native.layout(self.world_size, self.lane_count, self.settings.slots, self.settings.slot_bytes,
                                       library=_native.load(self._library_path) if self._library_path else None)
        if native_layout != self.layout.as_tuple():
            raise RuntimeError(f"native point-to-point layout {native_layout} differs from the protocol's "
                               f"{self.layout.as_tuple()}")
        with torch.cuda.device(self.device):
            # Zeroed: every tag word starts at 0 and the first item's tag is 1.
            self._region = torch.zeros(self.layout.total_bytes + 4096, dtype=torch.uint8, pin_memory=True)
        host = self._region.data_ptr()
        self._base = host + (-host) % 4096
        if _device_pointer(self._base) != self._base:
            raise RuntimeError("SIRCL point-to-point channels need pinned host memory that the GPU addresses at its "
                               "host pointer (an integrated GPU with unified addressing)")
        offset = self._base - host
        self._control = self._region[offset:offset + CONTROL_BYTES].view(torch.int32).numpy()
        if self._start_item:
            self._set_start_words(offset)
        self._set_regime("startup")
        self._out = [_Channel(self.device) for _ in range(self.world_size)]
        self._in = [_Channel(self.device) for _ in range(self.world_size)]
        for channel in (*self._out, *self._in):
            channel.next_item = self._start_item

    def _set_start_words(self, offset: int) -> None:
        """Arena words for channels that start at item ``start_item`` (tests of the 32-bit wrap)."""
        base = self._start_item
        signed = base - (1 << 32) if base >= 1 << 31 else base
        region = self._region[offset:offset + self.layout.total_bytes]
        for peer in range(self.world_size):
            if peer == self.rank:
                continue
            block = self.layout.block(peer)
            for area in (self.layout.ready_off, self.layout.consumed_off):
                region[block + area:block + area + 4 * self.settings.slots].view(torch.int32).fill_(signed)
            for slot in range(self.settings.slots):
                for lane in range(self.lane_count):
                    at = self.layout.flag(peer, slot, lane)
                    region[at:at + 4].view(torch.int32).fill_(signed)
            for area in (self.layout.sent_off, self.layout.credit_off):
                region[block + area:block + area + 4].view(torch.int32).fill_(signed)

    def _setup_record(self) -> dict[str, Any]:
        layout = self._layout_object
        return {
            "api_version": API_VERSION,
            "p2p_abi": _native.ABI_VERSION,
            "world_size": self.world_size,
            "lane_count": self.lane_count,
            "settings": self.settings.record(),
            "layout": layout.identity() if layout is not None else None,
            "channels": [list(row) for row in self._channel_table],
            "windows": self._window_table,
            "start_item": self._start_item,
            "traffic_class": self._native.traffic_class if self._native is not None else None,
            "devices": list(self.hca_names),
            "gid_indices": list(self.gid_indices),
            "lane_counts": [len(devices) for devices in self.peer_routes],
            "route_map": {str(peer): list(devices) for peer, devices in enumerate(self.peer_routes) if devices},
        }

    def _verdict(self, what: str, action) -> None:
        error = None
        if self._native is None:
            error = "no native context"
        else:
            try:
                action()
            except Exception as exc:  # noqa: BLE001 - reported to every rank
                error = str(exc)
        verdicts = _exchange(error, self._group)
        failures = [f"rank {index}: {verdict}" for index, verdict in enumerate(verdicts) if verdict is not None]
        if failures:
            # Every rank fails here alike, after its lane checks ended and before any transfer: no rounds.
            self.close(abort=True)
            raise RuntimeError(f"SIRCL point-to-point {what} failed: " + "; ".join(failures))

    # -- wait limits -------------------------------------------------------------------------

    def _set_regime(self, regime: str) -> None:
        if regime not in WAIT_REGIMES:
            raise ValueError(f"wait regime must be one of {WAIT_REGIMES}, got {regime!r}")
        seconds = self.settings.startup_wait_s if regime == "startup" else self.settings.serving_wait_s
        micros = max(1, min(0xFFFFFFFF, int(round(seconds * 1e6))))
        # A plain host store: kernels read it when a launch starts.
        self._control[Control.WAIT_LIMIT_US] = micros - (1 << 32) if micros >= 1 << 31 else micros
        self.wait_regime = regime

    @property
    def wait_limit_s(self) -> float:
        return self.settings.startup_wait_s if self.wait_regime == "startup" else self.settings.serving_wait_s

    @property
    def startup_wait_s(self) -> float:
        return self.settings.startup_wait_s

    @property
    def serving_wait_s(self) -> float:
        return self.settings.serving_wait_s

    def enter_startup(self) -> None:
        self._set_regime("startup")

    def enter_serving(self) -> None:
        self._set_regime("serving")

    @contextmanager
    def startup(self):
        previous = self.wait_regime
        self._set_regime("startup")
        try:
            yield self
        finally:
            self._set_regime(previous)

    # -- kernels ------------------------------------------------------------------------------

    def _launcher(self, kind: str):
        s = self.settings
        return _kernels.get_launcher(kind, s.threads, self.lane_count, s.slots, s.slot_bytes, s.blocks, s.unroll,
                                     self.device.index)

    def prepare(self) -> None:
        """Compile the send and receive kernels and load their modules (an empty launch of each)."""
        self._require_open()
        with torch.cuda.device(self.device):
            for kind in ("send", "recv"):
                launcher = self._launcher(kind)
                probe = torch.empty(PACK, dtype=torch.uint8, device=self.device)
                launcher(probe.data_ptr(), 0, 0, 0, 0, self._base + self.layout.block(0), self._base, 0)
            torch.cuda.current_stream(self.device).synchronize()

    # -- transfers ----------------------------------------------------------------------------

    def has_channel(self, peer: int) -> bool:
        return 0 <= peer < self.world_size and peer != self.rank and bool(self._channel_table[self.rank][peer])

    def channel_problem(self, peer: int) -> Optional[str]:
        """Why there is no channel toward ``peer`` (None when there is one)."""
        if not 0 <= peer < self.world_size or peer == self.rank:
            return f"rank {peer} is not another rank of this group of {self.world_size}"
        if not self._channel_table[self.rank][peer]:
            return self.channel_problems.get(peer, f"ranks {self.rank} and {peer} have no point-to-point channel")
        return None

    def _require_open(self) -> None:
        if self._closed or self._native is None:
            raise RuntimeError("SIRCL point-to-point channels are closed")

    def _check_call(self, tensor: torch.Tensor, peer: int) -> None:
        self._require_open()
        problem = self.channel_problem(int(peer))
        if problem is not None:
            raise ValueError(f"SIRCL point-to-point toward rank {peer}: {problem}")
        if not isinstance(tensor, torch.Tensor) or tensor.device != self.device:
            raise ValueError(f"SIRCL point-to-point tensors live on {self.device}")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SIRCL point-to-point transfers run outside CUDA graph capture")
        self.check_health()

    @staticmethod
    def _direct(tensor: torch.Tensor) -> bool:
        nbytes = tensor.numel() * tensor.element_size()
        return tensor.is_contiguous() and tensor.data_ptr() % PACK == 0 and nbytes % PACK == 0

    def _launch(self, kind: str, data_address: int, nbytes: int, first: int, peer: int) -> None:
        count = items(nbytes, self.settings.slot_bytes)
        self._launcher(kind)(data_address, padded(nbytes) // PACK, nbytes % PACK, first, count,
                             self._base + self.layout.block(peer), self._base, peer)

    def isend(self, tensor: torch.Tensor, peer: int) -> P2PWork:
        """Send ``tensor`` to group rank ``peer``; returns the work object (see the module docstring)."""
        self._check_call(tensor, peer)
        peer = int(peer)
        nbytes = tensor.numel() * tensor.element_size()
        caller = torch.cuda.current_stream(self.device)
        ready = torch.cuda.Event()
        ready.record(caller)
        with self._lock:
            channel = self._out[peer]
            stream = channel.get_stream()
            first = channel.next_item
            channel.next_item += items(nbytes, self.settings.slot_bytes)
            channel.messages += 1
            channel.bytes += nbytes
            done = torch.cuda.Event()
            keep: tuple[Any, ...] = (tensor,)
            with torch.cuda.device(self.device), torch.cuda.stream(stream):
                stream.wait_event(ready)
                if self._direct(tensor):
                    source = tensor
                else:
                    source = torch.zeros(padded(nbytes), dtype=torch.uint8, device=self.device)
                    if nbytes:
                        source[:nbytes].copy_(tensor.contiguous().view(-1).view(torch.uint8))
                    keep = (tensor, source)
                self._launch("send", source.data_ptr(), nbytes, first, peer)
                tensor.record_stream(stream)
                done.record(stream)
        return P2PWork(self, done, keep, kind="send", peer=peer, nbytes=nbytes)

    def irecv(self, tensor: torch.Tensor, peer: int) -> P2PWork:
        """Receive the next message of group rank ``peer`` into ``tensor`` (its byte count must match)."""
        self._check_call(tensor, peer)
        peer = int(peer)
        nbytes = tensor.numel() * tensor.element_size()
        caller = torch.cuda.current_stream(self.device)
        ready = torch.cuda.Event()
        ready.record(caller)
        with self._lock:
            channel = self._in[peer]
            stream = channel.get_stream()
            first = channel.next_item
            channel.next_item += items(nbytes, self.settings.slot_bytes)
            channel.messages += 1
            channel.bytes += nbytes
            done = torch.cuda.Event()
            keep: tuple[Any, ...] = (tensor,)
            with torch.cuda.device(self.device), torch.cuda.stream(stream):
                stream.wait_event(ready)
                if self._direct(tensor):
                    self._launch("recv", tensor.data_ptr(), nbytes, first, peer)
                else:
                    landing = torch.empty(max(PACK, padded(nbytes)), dtype=torch.uint8, device=self.device)
                    self._launch("recv", landing.data_ptr(), nbytes, first, peer)
                    if nbytes:
                        if tensor.is_contiguous():
                            tensor.view(-1).view(torch.uint8).copy_(landing[:nbytes])
                        else:
                            tensor.copy_(landing[:nbytes].view(tensor.dtype).view(tensor.shape))
                    keep = (tensor, landing)
                tensor.record_stream(stream)
                done.record(stream)
        return P2PWork(self, done, keep, kind="recv", peer=peer, nbytes=nbytes)

    def send(self, tensor: torch.Tensor, peer: int) -> None:
        """Send ``tensor`` to ``peer``, ordered on the caller's current stream (the host does not block)."""
        self.isend(tensor, peer).wait()

    def recv(self, tensor: torch.Tensor, peer: int) -> torch.Tensor:
        """Receive into ``tensor`` from ``peer``, ordered on the caller's current stream; returns ``tensor``."""
        self.irecv(tensor, peer).wait()
        return tensor

    def batch_isend_irecv(self, ops: Sequence[tuple[str, torch.Tensor, int]]) -> list[P2PWork]:
        """Issue ``(kind, tensor, peer)`` ops (kind ``send`` or ``recv``), each on its channel's stream, and return
        their work objects in list order.

        The sends are issued before the receives. Sends and receives are on different channels, and every
        channel keeps the list's order, so this matches messages exactly as the list does; it keeps every send
        of the batch ahead of the batch's receives on the GPU's hardware queues, where a waiting receive would
        hold back the commands queued behind it.
        """
        for kind, tensor, peer in ops:
            if kind not in ("send", "recv"):
                raise ValueError(f"a batched op is send or recv, got {kind!r}")
            self._check_call(tensor, peer)
        works: list[Optional[P2PWork]] = [None] * len(ops)
        for wanted in ("send", "recv"):
            for index, (kind, tensor, peer) in enumerate(ops):
                if kind == wanted:
                    works[index] = self.isend(tensor, peer) if kind == "send" else self.irecv(tensor, peer)
        return works  # type: ignore[return-value]

    # -- health, statistics, teardown -----------------------------------------------------------

    def _error_text(self) -> Optional[str]:
        control = getattr(self, "_control", None)
        if control is None:
            return None
        tag = int(control[Control.ERROR_TAG]) & 0xFFFFFFFF
        poison = int(control[Control.POISON])
        if not tag and not poison:
            return None
        kind = int(control[Control.ERROR_KIND])
        peer = int(control[Control.ERROR_PEER])
        lane = int(control[Control.ERROR_LANE])
        item = (tag - 1) & 0xFFFFFFFF
        if kind == ErrorKind.FLAG:
            what = (f"timed out waiting for item {item} from rank {peer} lane {lane} (wait limit {self.wait_limit_s:g} "
                    f"s, {self.wait_regime} regime)")
        elif kind == ErrorKind.SLOT:
            what = (f"timed out waiting for a free send slot for item {item} toward rank {peer}: rank {peer} did not "
                    f"receive the earlier items (wait limit {self.wait_limit_s:g} s, {self.wait_regime} regime)")
        elif kind == ErrorKind.SIZE:
            expected = int(control[Control.ERROR_EXPECTED]) & 0xFFFFFFFF
            got = int(control[Control.ERROR_GOT]) & 0xFFFFFFFF
            what = (f"received item {item} from rank {peer} as {describe_header(got)}, but the receive expected "
                    f"{describe_header(expected)}: the two ranks issued messages of different sizes on this channel")
        else:
            what = "stopped"
        return what

    def _failure_text(self) -> Optional[str]:
        """This rank's failure (a failed progress thread or a kernel's error), or None."""
        if self._native is not None and self._native.failed():
            local = self._error_text()
            detail = f"; this rank's kernel {local}" if local else ""
            return f"SIRCL point-to-point channels failed on rank {self.rank}: {self._native.error()}{detail}"
        local = self._error_text()
        if local:
            return (f"SIRCL point-to-point channels on rank {self.rank}: a kernel {local}; the channels are "
                    "poisoned and received data is untrustworthy")
        return None

    def check_health(self) -> None:
        """Raise when the channels failed on this or another rank of the group (host reads only)."""
        failure = self._failure_text()
        if failure is not None:
            raise RuntimeError(failure)
        if self._closed:
            raise RuntimeError("SIRCL point-to-point channels are closed")

    @property
    def poisoned(self) -> bool:
        failed = self._native is not None and self._native.failed()
        return failed or bool(int(self._control[Control.POISON])) or bool(int(self._control[Control.ERROR_TAG]))

    def stats(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "world_size": self.world_size,
            "rank": self.rank,
            "devices": list(self.hca_names),
            "gid_indices": list(self.gid_indices),
            "lane_count": self.lane_count,
            "channels": [peer for peer in range(self.world_size) if self.has_channel(peer)],
            "channel_problems": {str(peer): reason for peer, reason in self.channel_problems.items()},
            "slots": self.settings.slots,
            "slot_bytes": self.settings.slot_bytes,
            "chunk_bytes": self.settings.chunk_bytes,
            "blocks": self.settings.blocks,
            "windows": {str(peer): list(row) for peer, row in enumerate(self.windows) if any(row)},
            "arena_bytes": self.layout.total_bytes,
            "wait_regime": self.wait_regime,
            "wait_limit_s": self.wait_limit_s,
            "lane_check": dict(self._lane_check),
            "layout": self._layout_object.identity() if self._layout_object is not None else None,
            "sent": {str(peer): {"messages": c.messages, "bytes": c.bytes} for peer, c in enumerate(self._out)
                     if c.messages},
            "received": {str(peer): {"messages": c.messages, "bytes": c.bytes} for peer, c in enumerate(self._in)
                         if c.messages},
        }
        if self._native is not None:
            info.update(self._native.stats())
        return info

    def close(self, *, abort: bool = False) -> Optional[str]:
        """Refuse further work, wait for this rank's transfers, hold the teardown rounds over the exchange group
        (unless ``abort``) and release the RDMA resources; the close result, None after a healthy close (see the
        module docstring). Idempotent: a later call returns the first one's result."""
        with self._lock:
            if self._closed:
                return self.close_result
            self._closed = True
            released = [channel.release() for channel in (*getattr(self, "_out", ()), *getattr(self, "_in", ()))]
            problems = [problem for problem, _ in released if problem]
            settled = all(done for _, done in released)
            parts = [self._failure_text()]
            if problems:
                more = f" (and {len(problems) - 1} more channel(s))" if len(problems) > 1 else ""
                parts.append(f"SIRCL point-to-point channels on rank {self.rank}: {problems[0]}{more}")
            own = "; ".join(part for part in parts if part) or None
            # A setup that failed before the settings were read has no native context and holds no round.
            limit = (self.wait_limit_s if hasattr(self, "settings") else 0.0) + teardown_mod.SLACK_S
            what = f"SIRCL point-to-point teardown on rank {self.rank}"
            try:
                result = teardown_mod.close_native(
                    self._native, group=self._group, exchange=self._teardown_exchange,
                    own_failure=own, limit_s=limit, abort=abort, arena=getattr(self, "_region", None),
                    what=what, unsettled=not settled, kind="channels",
                    ordinal=getattr(self, "_teardown_ordinal", 0))
            except BaseException as exc:
                # close_native keeps every Exception; this is an interrupt mid-teardown. The outcome is unknown:
                # the native context stays referenced and the arena allocated, and the close has failed.
                if getattr(self, "_region", None) is not None:
                    teardown_mod.retain(self._region)
                interrupted = f"{what}: interrupted: {type(exc).__name__}: {exc}"
                self.close_result = f"{own}; {interrupted}" if own else interrupted
                raise
            self._native = None
            self.close_result = result
        if result is not None:
            logger.warning("SIRCL point-to-point close: %s", result)
        return result

    def __del__(self) -> None:  # pragma: no cover - defensive teardown
        # No rounds: garbage collection runs in no agreed order across ranks.
        try:
            self.close(abort=True)
        except Exception:  # noqa: BLE001 - teardown of a collected context
            pass


__all__ = ["P2PWork", "PointToPoint", "SettingError"]
