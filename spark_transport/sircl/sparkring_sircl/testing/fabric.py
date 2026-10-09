"""ctypes control of the in-memory verbs stand-in, and in-process sessions on it.

:class:`FakeFabric` binds the stand-in's control interface (``fake_verbs.h``)
from the simulator library (:func:`.native_build.build_shared_library`) and
runs a scheduler thread that executes posted writes. :class:`LocalSession`
creates one native context per rank of a session through the production
binding (:class:`sparkring_sircl.oneshot._proxy.Proxy`), with route maps from
:mod:`sparkring_sircl.routes` and, optionally, the forward windows that
:func:`sparkring_sircl.routes.forward_windows` derives, and plays the kernels'
part of the command ring for one-shot and two-shot ops: the two-shot
emulation (:func:`run_twoshot_ops`) stages, reduces and gathers with the same
chunk split, offsets and flag lines as the two-shot kernel
(``oneshot/_twoshot_cute.py``), summing 32-bit words modulo 2^32 in rank order.
Torch-free: payloads are ctypes buffers.
"""

from __future__ import annotations

import ctypes
import threading
import time
from collections.abc import Sequence

from .. import protocol as proto
from .. import routes as routes_mod
from ..oneshot import _proxy


class FakeFabric:
    """The stand-in's control interface on a loaded simulator library."""

    def __init__(self, library: ctypes.CDLL) -> None:
        self.lib = library
        u8p, i32, u32, u64 = ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_uint32, ctypes.c_uint64
        signatures = {
            "fv_reset": (None, []),
            "fv_add_device": (i32, [ctypes.c_char_p, i32, i32, i32, u8p]),
            "fv_add_cable": (i32, [i32, i32, i32, i32, u32]),
            "fv_set_relay": (None, [i32, i32]),
            "fv_set_dest_tag": (i32, [i32, u8p, u32]),
            "fv_set_ideal": (None, [i32]),
            "fv_set_latency": (None, [u64, u64]),
            "fv_set_rate": (None, [u64]),
            "fv_set_ack_delay": (None, [u64]),
            "fv_set_recording": (None, [i32]),
            "fv_inject_failure": (None, [u32, u32]),
            "fv_fail_teardown": (None, [i32]),
            "fv_progress": (u64, [u64, u64]),
            "fv_pending": (u64, []),
        }
        for name, (restype, argtypes) in signatures.items():
            function = getattr(library, name)
            function.restype = restype
            function.argtypes = argtypes
        self._running = False
        self._thread: threading.Thread | None = None

    def reset(self) -> None:
        self.stop()
        self.lib.fv_reset()

    @staticmethod
    def gid(node: int, role: int) -> bytes:
        return bytes(10) + b"\xff\xff" + bytes((10, node, role, 1))

    def add_device(self, name: str, node: int, port: int, function: int, gid: bytes) -> int:
        buffer = (ctypes.c_uint8 * 16)(*gid)
        index = self.lib.fv_add_device(name.encode(), node, port, function, buffer)
        if index < 0:
            raise RuntimeError(f"the stand-in refused device {name}")
        return index

    def start(self, seed: int = 1, *, spin: bool = False) -> None:
        """Run the scheduler thread; ``spin`` keeps it polling (a timed fabric) instead of napping."""
        if self._running:
            return
        self._running = True
        self.lib.fv_progress(0, seed)
        nap = 0.0 if spin else 0.0001

        def pump() -> None:
            while self._running:
                if self.lib.fv_progress(32, 0) == 0:
                    time.sleep(nap)

        self._thread = threading.Thread(target=pump, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None


class LocalSession:
    """Every rank of one session in this process, over an ideal fabric (delivery by GID)."""

    def __init__(self, library_path: str, layout: routes_mod.Layout, *, lanes: int = 2,
                 slot_bytes: int = 16384, node_base: int = 0, reset: bool = True,
                 forward_window: int = 0, forward_chunk: int = routes_mod.DEFAULT_FORWARD_CHUNK,
                 region_bytes: int | None = None) -> None:
        """``region_bytes``: each rank's registered region (default the arena; larger for a link area
        after it)."""
        self.library = _proxy.load(library_path)
        self.fabric = FakeFabric(self.library)
        if reset:
            self.fabric.reset()
            self.library.fv_set_ideal(1)
        self.layout = layout
        self.routes = routes_mod.derive_routes(layout, lanes)
        self.world = layout.world
        self.lanes = lanes
        self.slot_bytes = slot_bytes
        self.arena = proto.ArenaLayout(self.world, slot_bytes)
        maps = [self.routes.route_map(rank) for rank in range(self.world)]
        self.forward_chunk = forward_chunk
        self.windows = [routes_mod.forward_windows(layout, maps, rank, max_window=forward_window,
                                                   chunk=forward_chunk) for rank in range(self.world)]
        self._buffers = []
        self.addresses = []
        self.proxies: list[_proxy.Proxy] = []
        for rank in range(self.world):
            node = node_base + rank
            for index, role in enumerate(routes_mod.ROLES):
                self.fabric.add_device(f"n{node}.{role.device}", node, role.port, int(role.secondary),
                                       FakeFabric.gid(node, index))
        for rank in range(self.world):
            node = node_base + rank
            route_map = self.routes.route_map(rank)
            named = list(dict.fromkeys(d for _, devices in sorted(route_map.items()) for d in devices))
            names = [f"n{node}.{device}" for device in named]
            lane_devices = [() if peer == rank else tuple(named.index(d) for d in route_map[peer])
                            for peer in range(self.world)]
            region = self.arena.total_bytes if region_bytes is None else int(region_bytes)
            buffer = ctypes.create_string_buffer(region + 4096)
            address = ctypes.addressof(buffer) + (-ctypes.addressof(buffer)) % 4096
            self._buffers.append(buffer)
            self.addresses.append(address)
            self.proxies.append(_proxy.Proxy(
                world_size=self.world, rank=rank, hca_names=names, peer_lane_devices=lane_devices,
                lane_count=lanes, gid_indices=[3] * len(names), region_ptr=address,
                region_bytes=region, slot_bytes=slot_bytes, library=self.library,
            ))

    def connect(self, lane_check_ms: int = 2000) -> None:
        self.fabric.start()
        blobs = [proxy.local_blob() for proxy in self.proxies]
        for proxy in self.proxies:
            proxy.connect(blobs)
        for proxy in self.proxies:
            proxy.lane_check(lane_check_ms)
        for proxy, table in zip(self.proxies, self.windows):
            if any(any(row) for row in table):
                proxy.set_forward(table, self.forward_chunk)
        for proxy in self.proxies:
            proxy.start()

    # -- the kernels' part of the command ring ---------------------------------------

    def _word(self, rank: int, offset: int) -> ctypes.c_uint32:
        return ctypes.c_uint32.from_address(self.addresses[rank] + offset)

    def ring_oneshot(self, rank: int, seq: int, payload: bytes) -> None:
        slot = proto.slot_of(seq)
        ctypes.memmove(self.addresses[rank] + self.arena.send_off + slot * self.slot_bytes, payload, len(payload))
        ctrl = self.arena.ctrl_off
        self._word(rank, ctrl + 4 * (proto.Ctrl.OP_WORD + slot)).value = proto.op_word(proto.Op.ONESHOT, len(payload))
        self._word(rank, ctrl + 4 * proto.Ctrl.NBYTES).value = len(payload)
        self.library.roce_store_release_u32(self.addresses[rank] + ctrl + 4 * proto.Ctrl.DOORBELL,
                                            seq & 0xFFFFFFFF)

    def ring_twoshot(self, rank: int, seq: int, payload: bytes) -> None:
        """Stage every chunk but the own one, then ring the doorbell with op code 1."""
        slot = proto.slot_of(seq)
        send = self.addresses[rank] + self.arena.send_off + slot * self.slot_bytes
        packs = len(payload) // proto.PACK_BYTES
        for chunk, (first, count) in enumerate(proto.chunks(packs, self.world)):
            if chunk != rank and count:
                lo, hi = first * proto.PACK_BYTES, (first + count) * proto.PACK_BYTES
                ctypes.memmove(send + lo, payload[lo:hi], hi - lo)
        ctrl = self.arena.ctrl_off
        self._word(rank, ctrl + 4 * (proto.Ctrl.OP_WORD + slot)).value = proto.op_word(proto.Op.TWOSHOT, len(payload))
        self._word(rank, ctrl + 4 * proto.Ctrl.NBYTES).value = len(payload)
        self.library.roce_store_release_u32(self.addresses[rank] + ctrl + 4 * proto.Ctrl.DOORBELL,
                                            seq & 0xFFFFFFFF)

    def release_phase(self, rank: int, seq: int, phase: int, reduced: bytes, offset: int) -> None:
        """Write ``reduced`` into ``send[slot]`` at ``offset``, then ring the phase doorbell."""
        send = self.addresses[rank] + self.arena.send_off + proto.slot_of(seq) * self.slot_bytes
        if reduced:
            ctypes.memmove(send + offset, reduced, len(reduced))
        word = self.arena.ctrl_off + 4 * proto.phase_doorbell_word(phase)
        self.library.roce_store_release_u32(self.addresses[rank] + word, seq & 0xFFFFFFFF)

    def wait_flags(self, rank: int, seq: int, timeout: float = 10.0, namespace: int = 0) -> None:
        deadline = time.monotonic() + timeout
        for source in range(self.world):
            if source == rank:
                continue
            for lane in range(self.lanes):
                line = proto.flag_index(namespace, source, proto.slot_of(seq), lane, self.world, self.lanes)
                address = self.addresses[rank] + self.arena.flag_off + line * proto.FLAG_STRIDE
                while self.library.roce_load_acquire_u32(address) != seq & 0xFFFFFFFF:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"rank {rank} waited for rank {source} lane {lane} at sequence {seq}")
                    time.sleep(0.0001)

    def received(self, rank: int, source: int, seq: int, nbytes: int) -> bytes:
        slot = proto.slot_of(seq)
        start = self.addresses[rank] + self.arena.recv_off + (source * proto.SLOTS + slot) * self.slot_bytes
        return ctypes.string_at(start, nbytes)

    def close(self) -> None:
        for proxy in self.proxies:
            proxy.close()
        self.proxies = []
        self.fabric.stop()


def payload(rank: int, seq: int, nbytes: int) -> bytes:
    """A deterministic payload of ``nbytes`` for (rank, sequence)."""
    seed = (rank * 1_000_003 + seq * 7919) & 0xFFFFFFFF
    out = bytearray(nbytes)
    for index in range(0, nbytes, 4):
        seed = (seed * 1103515245 + 12345) & 0xFFFFFFFF
        out[index:index + 4] = seed.to_bytes(4, "little")
    return bytes(out)


def run_oneshot_ops(session: LocalSession, sizes: Sequence[int], first_seq: int = 1) -> None:
    """Every rank stages one payload per op; every rank checks every peer's bytes."""
    for offset, nbytes in enumerate(sizes):
        seq = first_seq + offset
        for rank in range(session.world):
            session.ring_oneshot(rank, seq, payload(rank, seq, nbytes))
        for rank in range(session.world):
            session.wait_flags(rank, seq)
            for source in range(session.world):
                if source != rank and session.received(rank, source, seq, nbytes) != payload(source, seq, nbytes):
                    raise AssertionError(f"rank {rank} received wrong bytes from rank {source} at sequence {seq}")


def _add_words(total: bytearray, data: bytes) -> None:
    """``total += data`` as little-endian 32-bit words modulo 2^32."""
    for index in range(0, len(data), 4):
        value = int.from_bytes(total[index:index + 4], "little") + int.from_bytes(data[index:index + 4], "little")
        total[index:index + 4] = (value & 0xFFFFFFFF).to_bytes(4, "little")


def run_twoshot_ops(session: LocalSession, sizes: Sequence[int], first_seq: int = 1) -> None:
    """Two-shot ops: every rank reduces its chunk in rank order, gathers the others' and checks the sum."""
    world = session.world
    for offset, nbytes in enumerate(sizes):
        seq = first_seq + offset
        packs = nbytes // proto.PACK_BYTES
        inputs = [payload(rank, seq, nbytes) for rank in range(world)]
        expected = bytearray(nbytes)
        for data in inputs:
            _add_words(expected, data)
        for rank in range(world):
            session.ring_twoshot(rank, seq, inputs[rank])
        reduced = []
        for rank in range(world):
            session.wait_flags(rank, seq, namespace=0)
            first, count = proto.chunk(packs, world, rank)
            lo, hi = first * proto.PACK_BYTES, (first + count) * proto.PACK_BYTES
            total = bytearray(hi - lo)
            for source in range(world):
                part = inputs[rank][lo:hi] if source == rank else session.received(rank, source, seq, hi)[lo:hi]
                _add_words(total, part)
            reduced.append(bytes(total))
            session.release_phase(rank, seq, 1, bytes(total), lo)
        for rank in range(world):
            session.wait_flags(rank, seq, namespace=1)
            output = bytearray(nbytes)
            for source in range(world):
                first, count = proto.chunk(packs, world, source)
                lo, hi = first * proto.PACK_BYTES, (first + count) * proto.PACK_BYTES
                output[lo:hi] = reduced[rank] if source == rank else session.received(rank, source, seq, hi)[lo:hi]
            if output != expected:
                raise AssertionError(f"rank {rank} gathered a wrong sum at sequence {seq} ({nbytes} bytes)")
