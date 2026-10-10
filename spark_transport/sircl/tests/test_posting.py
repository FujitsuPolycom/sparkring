"""Posting orders (``sparkring_sircl.posting``) and, on the simulator build, the native posting order."""

from __future__ import annotations

import ctypes

import pytest

from sparkring_sircl import posting
from sparkring_sircl import protocol as proto
from sparkring_sircl import routes
from sparkring_sircl.testing import fabric


@pytest.mark.parametrize("size", [2, 3, 4, 6, 8])
def test_farthest_first_equals_ring_farthest_on_a_ring_in_ring_order(size):
    layout = routes.Layout.parse(f"ring:{size}")
    group = routes.derive_routes(layout, 2)
    for rank in range(size):
        expected = proto.ring_farthest(rank, size)
        assert posting.farthest_first(layout, rank, group.route_map(rank)) == expected
        assert posting.resolve("farthest", rank, size, layout, group.route_map(rank)) == expected


def test_farthest_first_on_paths_and_subgroups():
    path = routes.Layout.parse("path:0-3")
    group = routes.derive_routes(path, 2)
    assert posting.lane_relays(path, 0, group.route_map(0)) == {1: (0, 0), 2: (1, 1), 3: (2, 2)}
    assert [posting.farthest_first(path, rank, group.route_map(rank)) for rank in range(4)] == [
        (3, 2, 1), (3, 2, 0), (0, 3, 1), (0, 1, 2)]
    # ring-farthest folds the group size: on a path it posts the far end last.
    assert proto.ring_farthest(0, 4) == (2, 1, 3)
    alternate = routes.Layout.parse("ring:8:0,2,4,6")
    group = routes.derive_routes(alternate, 2)
    assert posting.lane_relays(alternate, 0, group.route_map(0)) == {1: (1, 1), 2: (3, 3), 3: (1, 1)}
    assert posting.farthest_first(alternate, 0, group.route_map(0)) == (2, 1, 3)
    ring = routes.Layout.parse("ring:8")
    single = routes.derive_routes(ring, 1)
    assert posting.farthest_first(ring, 0, single.route_map(0)) == (4, 3, 5, 2, 6, 1, 7)


def test_resolve_names_lists_and_refusals():
    layout = routes.Layout.parse("ring:4")
    group = routes.derive_routes(layout, 2)
    assert posting.resolve(None, 1, 4) == (0, 2, 3) == posting.resolve(" rank ", 1, 4)
    assert posting.resolve("ring-farthest", 1, 4) == proto.ring_farthest(1, 4)
    assert posting.resolve("3,0,2", 1, 4) == (3, 0, 2)
    with pytest.raises(ValueError, match="layout and route map"):
        posting.resolve("farthest", 1, 4)
    for bad in ("0,2", "nearest", "0,2,2"):
        with pytest.raises(ValueError):
            posting.resolve(bad, 1, 4)
    partial = {peer: devices for peer, devices in group.route_map(1).items() if peer != 3}
    with pytest.raises(ValueError, match="every peer"):
        posting.resolve("farthest", 1, 4, layout, partial)
    with pytest.raises(routes.RouteError, match="no known role"):
        posting.lane_relays(layout, 0, {1: ("mlx5_9",)})


class _Event(ctypes.Structure):
    _fields_ = [("sequence", ctypes.c_uint64), ("phase", ctypes.c_uint32), ("status", ctypes.c_uint32),
                ("src_device", ctypes.c_int32), ("dst_device", ctypes.c_int32), ("qp_num", ctypes.c_uint32),
                ("flags", ctypes.c_uint32), ("wr_id", ctypes.c_uint64), ("local_addr", ctypes.c_uint64),
                ("remote_addr", ctypes.c_uint64), ("length", ctypes.c_uint32), ("inline_word", ctypes.c_uint32),
                ("relays", ctypes.c_uint32), ("reason", ctypes.c_uint32), ("relay_nodes", ctypes.c_uint64)]


def _posted_peers(library, devices: set[int], seq: int) -> tuple[int, ...]:
    """Peers of the flag writes ``devices`` posted for ``seq``, in posting order (one entry per peer)."""
    library.fv_event_count.restype = ctypes.c_uint64
    library.fv_event.argtypes = [ctypes.c_uint64, ctypes.POINTER(_Event)]
    peers: list[int] = []
    event = _Event()
    for index in range(library.fv_event_count()):
        library.fv_event(index, ctypes.byref(event))
        if event.phase != 0 or not event.flags & 2 or event.inline_word != seq or event.src_device not in devices:
            continue
        peer = event.wr_id & 0xFF
        if not peers or peers[-1] != peer:
            peers.append(peer)
    return tuple(peers)


def test_native_posting_order_switches_between_ops_and_counts_posting(simulator_library, monkeypatch):
    from sparkring_sircl.oneshot import _proxy

    if not hasattr(_proxy.Proxy, "set_post_order"):
        pytest.skip("the native layer has no run-time posting order (Proxy.set_post_order)")
    monkeypatch.delenv("SIRCL_POST_ORDER", raising=False)
    layout = routes.Layout.parse("ring:8")
    session = fabric.LocalSession(str(simulator_library), layout, lanes=2)
    rank0 = {0, 1, 2, 3}     # the stand-in numbers devices in the order LocalSession adds them, rank 0 first
    try:
        session.connect()
        proxy = session.proxies[0]
        fabric.run_oneshot_ops(session, [256, 4096], first_seq=1)
        assert _posted_peers(session.library, rank0, 1) == (1, 2, 3, 4, 5, 6, 7)
        assert proxy.post_order() == (1, 2, 3, 4, 5, 6, 7)
        farthest = posting.farthest_first(layout, 0, session.routes.route_map(0))
        proxy.set_post_order(farthest)
        fabric.run_oneshot_ops(session, [256], first_seq=3)
        assert _posted_peers(session.library, rank0, 3) == farthest == (4, 3, 5, 2, 6, 1, 7)
        assert proxy.post_order() == farthest
        # Every phase posts each of its 14 lanes at once (no forward windows here).
        stats = proxy.stats()
        assert stats["post_phases_total"] == 3 and stats["post_lanes_total"] == 3 * 14
        assert 0 < stats["post_ns_max"] <= stats["post_ns_total"]
        with pytest.raises(ValueError, match="not a peer of rank 0 named once"):
            proxy.set_post_order((1, 1, 2, 3, 4, 5, 6))
        with pytest.raises(ValueError, match="names all 7 peers"):
            proxy.set_post_order((1, 2))
        assert proxy.post_order() == farthest
        # A two-shot op posts two phases, each over every peer.
        fabric.run_twoshot_ops(session, [4096], first_seq=4)
        stats = proxy.stats()
        assert stats["post_phases_total"] == 5 and stats["post_lanes_total"] == 5 * 14
        assert not any(item.failed() for item in session.proxies)
    finally:
        session.close()
