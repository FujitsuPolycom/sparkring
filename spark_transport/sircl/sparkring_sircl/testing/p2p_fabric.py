"""In-process point-to-point contexts on the verbs stand-in, with the kernels' part played on the host.

:class:`LocalChannels` creates one native point-to-point context per rank of a
group through the production binding (:class:`sparkring_sircl.p2p._native.Native`)
on the test library (:func:`.p2p_build.build_shared_library`), with route maps
from :mod:`sparkring_sircl.routes`, connects them over an ideal in-memory fabric
(delivery by GID) and starts their progress threads. :meth:`LocalChannels.send`
and :meth:`LocalChannels.recv` play the send and receive kernels
(``p2p/_kernels.py``) on the host with the same slots, headers, tags and words,
one item after another. Torch-free: payloads are bytes.
"""

from __future__ import annotations

import ctypes
import time
from collections.abc import Iterable, Sequence

from .. import routes as routes_mod
from ..p2p import _native, budget
from ..p2p import protocol as proto
from .fabric import FakeFabric

WAIT_S = 20.0


class ChannelFailure(RuntimeError):
    """A host-played kernel found a failure (the control line's record, as a kernel writes it)."""


class LocalChannels:
    """Point-to-point contexts of every rank of one group in this process."""

    def __init__(self, library_path: str, layout: routes_mod.Layout, *, lanes: int = 2, slots: int = 4,
                 slot_bytes: int = 8192, channels: Iterable[tuple[int, int]] | None = None,
                 windows: Sequence[Sequence[Sequence[int]]] | None = None, chunk_bytes: int = 4096,
                 max_window: int = 16384, node_base: int = 0, reset: bool = True) -> None:
        self.library = _native.load(library_path)
        self.fabric = FakeFabric(self.library)
        if reset:
            self.fabric.reset()
            self.library.fv_set_ideal(1)
        self.layout = layout
        self.world = layout.world
        self.lanes = lanes
        self.slots = slots
        self.slot_bytes = slot_bytes
        self.arena = proto.P2PLayout(self.world, lanes, slots, slot_bytes)
        self.routes = routes_mod.derive_routes(layout, lanes)
        every = [(a, b) for a in range(self.world) for b in range(self.world) if a < b]
        chosen = sorted({tuple(sorted(pair)) for pair in (every if channels is None else channels)})
        self.table = [[False] * self.world for _ in range(self.world)]
        for a, b in chosen:
            self.table[a][b] = self.table[b][a] = True
        if windows is None:
            lane_set = budget.LaneSet.of("group", layout, lanes,
                                         [(a, b) for a in range(self.world) for b in range(self.world)
                                          if self.table[a][b]])
            windows, _ = budget.group_windows(lane_set, lanes, max_window=max_window, chunk=chunk_bytes)
        self.windows = windows
        self.contexts: list[_native.Native] = []
        self._buffers = []
        self.addresses: list[int] = []
        self.sent_items = [[0] * self.world for _ in range(self.world)]
        self.received_items = [[0] * self.world for _ in range(self.world)]
        for rank in range(self.world):
            node = node_base + rank
            for index, role in enumerate(routes_mod.ROLES):
                self.fabric.add_device(f"n{node}.{role.device}", node, role.port, int(role.secondary),
                                       FakeFabric.gid(node, index))
        self.fabric.start()
        for rank in range(self.world):
            node = node_base + rank
            route_map = self.routes.route_map(rank)
            named = list(dict.fromkeys(d for _, devices in sorted(route_map.items()) for d in devices))
            names = [f"n{node}.{device}" for device in named]
            lane_devices = [() if peer == rank or not self.table[rank][peer] else
                            tuple(named.index(d) for d in route_map[peer]) for peer in range(self.world)]
            buffer = ctypes.create_string_buffer(self.arena.total_bytes + 4096)
            address = ctypes.addressof(buffer) + (-ctypes.addressof(buffer)) % 4096
            self._buffers.append(buffer)
            self.addresses.append(address)
            self.contexts.append(_native.Native(
                world_size=self.world, rank=rank, hca_names=names, peer_lane_devices=lane_devices, lane_count=lanes,
                gid_indices=[3] * len(names), channels=self.table[rank], region_ptr=address,
                region_bytes=self.arena.total_bytes, slots=slots, slot_bytes=slot_bytes, library=self.library))
        blobs = [context.local_blob() for context in self.contexts]
        for context in self.contexts:
            context.connect(blobs)
        for context in self.contexts:
            context.lane_check(10000)
        for rank, context in enumerate(self.contexts):
            if any(window for row in windows[rank] for window in row):
                context.set_windows(windows[rank], chunk_bytes)
            context.start()

    # -- words --------------------------------------------------------------------------------

    def _word(self, rank: int, offset: int) -> int:
        return _native.load_acquire_u32(self.addresses[rank] + offset, library=self.library)

    def _store(self, rank: int, offset: int, value: int) -> None:
        _native.store_release_u32(self.addresses[rank] + offset, value, library=self.library)

    def _wait(self, rank: int, test, what: str, peer: int, lane: int, kind: int, tag: int) -> None:
        deadline = time.monotonic() + WAIT_S
        while not test():
            if self._word(rank, 4 * proto.Control.POISON):
                raise ChannelFailure(f"rank {rank} is poisoned while waiting for {what}")
            if time.monotonic() > deadline:
                self.fail(rank, peer, lane, kind, tag, 0, 0)
                raise ChannelFailure(f"rank {rank} waited more than {WAIT_S} s for {what}")
            time.sleep(0.00005)

    def fail(self, rank: int, peer: int, lane: int, kind: int, tag: int, expected: int, got: int) -> None:
        """Record a failure in ``rank``'s control line as a kernel does: the error words, the tag, the poison."""
        for word, value in ((proto.Control.ERROR_PEER, peer), (proto.Control.ERROR_LANE, lane),
                            (proto.Control.ERROR_KIND, kind), (proto.Control.ERROR_EXPECTED, expected),
                            (proto.Control.ERROR_GOT, got)):
            self._store(rank, 4 * word, value)
        self._store(rank, 4 * proto.Control.ERROR_TAG, tag)
        self._store(rank, 4 * proto.Control.POISON, 1)

    # -- the kernels' part ----------------------------------------------------------------------

    def send(self, rank: int, peer: int, payload: bytes) -> None:
        """Stage every item of ``payload`` toward ``peer`` as the send kernel does."""
        arena, slots = self.arena, self.slots
        count = proto.items(len(payload), self.slot_bytes)
        padded = payload + bytes(proto.padded(len(payload)) - len(payload))
        for index in range(count):
            g = self.sent_items[rank][peer]
            m, tag = g % slots, proto.tag(g)
            sent = arena.word(peer, arena.sent_off)
            target = (tag - slots) & 0xFFFFFFFF
            self._wait(rank, lambda: ((self._word(rank, sent) - target) & 0xFFFFFFFF) < (1 << 31),
                       f"send slot {m} toward rank {peer}", peer, 255, proto.ErrorKind.SLOT, tag)
            first = index * self.slot_bytes
            item = padded[first:first + proto.item_bytes(len(payload), self.slot_bytes, index)]
            ctypes.memmove(self.addresses[rank] + arena.send_slot(peer, m), item, len(item))
            self._store(rank, arena.word(peer, arena.desc_off, m), proto.header(len(payload), self.slot_bytes, index))
            self._store(rank, arena.word(peer, arena.ready_off, m), tag)
            self.sent_items[rank][peer] = (g + 1) & 0xFFFFFFFF

    def recv(self, rank: int, peer: int, nbytes: int) -> bytes:
        """Receive the next message of ``nbytes`` from ``peer`` as the receive kernel does."""
        arena, slots = self.arena, self.slots
        out = bytearray()
        for index in range(proto.items(nbytes, self.slot_bytes)):
            g = self.received_items[rank][peer]
            m, tag = g % slots, proto.tag(g)
            for lane in range(self.lanes):
                flag = arena.flag(peer, m, lane)
                self._wait(rank, lambda: self._word(rank, flag) == tag, f"item {g} from rank {peer} lane {lane}",
                           peer, lane, proto.ErrorKind.FLAG, tag)
            expected = proto.header(nbytes, self.slot_bytes, index)
            got = self._word(rank, arena.header(peer, m))
            if got != expected:
                self.fail(rank, peer, 255, proto.ErrorKind.SIZE, tag, expected, got)
                raise ChannelFailure(f"rank {rank}: item {g} from rank {peer} is {proto.describe_header(got)}, "
                                     f"the receive expected {proto.describe_header(expected)}")
            size = expected & proto.BYTES_MASK
            out += ctypes.string_at(self.addresses[rank] + arena.recv_slot(peer, m), size)
            self._store(rank, arena.word(peer, arena.consumed_off, m), tag)
            self.received_items[rank][peer] = (g + 1) & 0xFFFFFFFF
        return bytes(out[:nbytes])

    def close(self) -> None:
        for context in self.contexts:
            context.close()
        self.contexts = []
        self.fabric.stop()


__all__ = ["ChannelFailure", "LocalChannels"]
