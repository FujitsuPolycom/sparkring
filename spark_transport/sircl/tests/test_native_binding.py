"""The Python binding of the native layer, on the simulator build of the library."""

from __future__ import annotations

import pytest

from sparkring_sircl import protocol as proto
from sparkring_sircl import routes
from sparkring_sircl.oneshot import _proxy
from sparkring_sircl.testing import fabric


def test_abi_and_layout_agree_with_the_protocol(simulator_library):
    lib = _proxy.load(simulator_library)
    assert lib.roce_abi_version() == _proxy.ABI_VERSION
    for world, slot in ((2, 4096), (3, 16384), (8, 155648), (16, 1 << 20)):
        native = _proxy.Layout(world, slot, library=lib)
        reported = (native.recv_off, native.flag_off, native.send_off, native.ctrl_off,
                    native.total_bytes, native.flag_stride, native.slots)
        assert reported == proto.ArenaLayout(world, slot).as_tuple()
    for world, slot in ((1, 4096), (17, 4096), (4, 1000), (4, 0)):
        with pytest.raises(ValueError, match=f"world size {world}"):
            _proxy.Layout(world, slot, library=lib)


def test_a_mismatched_library_is_refused(simulator_library, monkeypatch):
    monkeypatch.setattr(_proxy, "ABI_VERSION", _proxy.ABI_VERSION + 1)
    monkeypatch.setattr(_proxy, "_LIBRARIES", {})
    with pytest.raises(RuntimeError, match="unexpected native ABI version"):
        _proxy.load(simulator_library)


@pytest.mark.parametrize("layout_text, lanes", [
    ("ring:2", 2), ("ring:3", 2), ("ring:4", 1), ("path:0-3", 2), ("ring:8", 2),
])
def test_sessions_through_the_binding(simulator_library, layout_text, lanes):
    session = fabric.LocalSession(str(simulator_library), routes.Layout.parse(layout_text), lanes=lanes)
    try:
        session.connect()
        fabric.run_oneshot_ops(session, [16, 4096, 16000, 16384, 48], first_seq=1)
        stats = session.proxies[0].stats()
        assert stats["ops_posted"] == 5 and stats["lane_count"] == lanes and stats["last_seq"] == 5
        assert stats["post_mode"] == "verbs" and stats["multi_phase"]
        assert sum(stats["bytes_posted_per_hca"]) == (session.world - 1) * (16 + 4096 + 16000 + 16384 + 48)
        assert not any(proxy.failed() for proxy in session.proxies)
    finally:
        session.close()


def test_a_failed_write_fails_the_progress_thread(simulator_library):
    session = fabric.LocalSession(str(simulator_library), routes.Layout.parse("ring:3"), lanes=2)
    try:
        session.connect()
        # The first flag write of sequence 2 on any queue pair fails.
        session.library.fv_inject_failure(0, 2)
        fabric.run_oneshot_ops(session, [64], first_seq=1)
        for rank in range(3):
            session.ring_oneshot(rank, 2, fabric.payload(rank, 2, 64))
        import time

        deadline = time.monotonic() + 5
        while not any(proxy.failed() for proxy in session.proxies) and time.monotonic() < deadline:
            time.sleep(0.01)
        failed = [proxy for proxy in session.proxies if proxy.failed()]
        assert failed and "RDMA write of sequence 2" in failed[0].error()
        assert "remote access error" in failed[0].error()
    finally:
        session.close()


def test_native_refusals_name_the_problem(simulator_library):
    lib = _proxy.load(simulator_library)
    import ctypes

    buffer = ctypes.create_string_buffer(proto.ArenaLayout(2, 4096).total_bytes + 4096)
    address = ctypes.addressof(buffer) + (-ctypes.addressof(buffer)) % 4096
    common = dict(world_size=2, rank=0, gid_indices=[3], region_ptr=address,
                  region_bytes=proto.ArenaLayout(2, 4096).total_bytes, slot_bytes=4096, library=lib)
    with pytest.raises(RuntimeError, match="not found"):
        _proxy.Proxy(hca_names=["missing-device"], peer_lane_devices=[(), (0,)], lane_count=1, **common)
    with pytest.raises(ValueError, match="lane devices"):
        _proxy.Proxy(hca_names=["x"], peer_lane_devices=[(), (0, 0)], lane_count=1, **common)
    with pytest.raises(RuntimeError, match="two lanes toward rank 1"):
        _proxy.Proxy(hca_names=["x"], peer_lane_devices=[(), (0, 0)], lane_count=2, **common)


@pytest.mark.parametrize("layout_text, lanes", [("ring:2", 2), ("ring:3", 1), ("path:0-3", 2), ("ring:8", 2)])
def test_twoshot_ops_through_the_binding(simulator_library, layout_text, lanes):
    session = fabric.LocalSession(str(simulator_library), routes.Layout.parse(layout_text), lanes=lanes)
    try:
        session.connect()
        # 48 bytes over 4 or 8 ranks leaves empty chunks: flag-only lanes.
        fabric.run_twoshot_ops(session, [16, 48, 4096, 16384, 16000], first_seq=1)
        stats = session.proxies[0].stats()
        assert stats["ops_posted"] == 5 and stats["later_phases_posted"] == 5
        assert not any(proxy.failed() for proxy in session.proxies)
    finally:
        session.close()


def test_forward_windows_pace_relayed_lanes_through_the_binding(simulator_library):
    session = fabric.LocalSession(str(simulator_library), routes.Layout.parse("path:0-3"), lanes=2,
                                  slot_bytes=65536, forward_window=16384, forward_chunk=4096)
    try:
        assert session.windows[0] == [[0, 0], [0, 0], [16384, 16384], [16384, 16384]]
        session.connect()
        fabric.run_oneshot_ops(session, [16, 4096, 65536, 32768], first_seq=1)
        fabric.run_twoshot_ops(session, [65536, 48, 16, 65520], first_seq=5)
        stats = session.proxies[0].stats()
        assert stats["forward_chunks_posted"] > 0
        assert 0 < stats["forward_max_unacked_bytes"] <= 16384 + 4
        assert stats["ops_posted"] == 8 and stats["later_phases_posted"] == 4
        # Rank 1 reaches only rank 3 through a relay.
        assert session.windows[1] == [[0, 0], [0, 0], [0, 0], [16384, 16384]]
        assert not any(proxy.failed() for proxy in session.proxies)
    finally:
        session.close()


@pytest.mark.parametrize("setting, proof", [("0", False), ("", True)])
def test_forward_proof_setting_and_window_counters_through_the_binding(simulator_library, monkeypatch, setting,
                                                                        proof):
    monkeypatch.setenv("SIRCL_FORWARD_PROOF", setting)
    session = fabric.LocalSession(str(simulator_library), routes.Layout.parse("path:0-3"), lanes=2,
                                  slot_bytes=65536, forward_window=16384, forward_chunk=4096)
    try:
        session.connect()
        fabric.run_twoshot_ops(session, [65536, 65520, 32768, 65536], first_seq=1)
        stats = session.proxies[0].stats()
        assert stats["forward_proof"] is proof
        assert 0 <= stats["forward_wait_max_ns"] <= stats["forward_wait_ns"]
        assert (stats["forward_waits"] == 0) == (stats["forward_wait_ns"] == 0)
        assert stats["forward_proven_bytes"] >= 0 and (proof or stats["forward_proven_bytes"] == 0)
        assert 0 < stats["forward_max_unacked_bytes"] <= 16384 + 4
        assert not any(p.failed() for p in session.proxies)
    finally:
        session.close()


def test_forward_window_refusals_name_the_lane(simulator_library):
    session = fabric.LocalSession(str(simulator_library), routes.Layout.parse("ring:3"), lanes=1)
    try:
        proxy = session.proxies[0]
        with pytest.raises(RuntimeError, match="own lane"):
            proxy.set_forward([[4096], [0], [0]], 4096)
        with pytest.raises(RuntimeError, match="must hold 1 to"):
            proxy.set_forward([[0], [1024], [0]], 4096)
        with pytest.raises(RuntimeError, match="multiple of 16"):
            proxy.set_forward([[0], [8192], [0]], 1000)
        with pytest.raises(ValueError, match="one entry per rank"):
            proxy.set_forward([[0]], 4096)
        proxy.set_forward([[0], [8192], [0]], 4096)
        proxy.set_forward(None)
    finally:
        session.close()


def test_chain_layout_and_refusals_through_the_binding(simulator_library):
    lib = _proxy.load(simulator_library)
    for lanes, slots, slot in ((1, 2, 4096), (2, 4, 1 << 20), (2, 32, 8192)):
        assert _proxy.chain_layout(lanes, slots, slot, library=lib) == proto.ChainLayout(lanes, slots, slot).as_tuple()
    with pytest.raises(ValueError, match="unsupported chain geometry"):
        _proxy.chain_layout(2, 1, 4096, library=lib)
    session = fabric.LocalSession(str(simulator_library), routes.Layout.parse("path:0-2"), lanes=2)
    try:
        proxy = session.proxies[1]
        with pytest.raises(RuntimeError, match="does not fit"):
            proxy.set_chain(0, 2, 2, 4096, proto.chain_offset(session.arena.total_bytes))
        with pytest.raises(RuntimeError, match="chain neighbors"):
            proxy.set_chain(1, 2, 2, 4096, 0)
        proxy.set_chain(0, 2, 0, 0, 0)
        stats = proxy.stats()
        assert stats["chain_ops"] == 0 and stats["chain_chunks_posted"] == 0
    finally:
        session.close()


def test_link_layout_and_refusals_through_the_binding(simulator_library):
    lib = _proxy.load(simulator_library)
    for lanes, slots, slot in ((1, 2, 4096), (2, 8, 512 << 10), (2, 32, 8192)):
        assert _proxy.link_layout(lanes, slots, slot, library=lib) == proto.LinkLayout(lanes, slots, slot).as_tuple()
    with pytest.raises(ValueError, match="unsupported link geometry"):
        _proxy.link_layout(2, 1, 4096, library=lib)
    session = fabric.LocalSession(str(simulator_library), routes.Layout.parse("path:0-2"), lanes=2)
    try:
        proxy = session.proxies[1]
        with pytest.raises(RuntimeError, match="does not fit"):
            proxy.set_links(0, 2, 1, 2, 4096, proto.chain_offset(session.arena.total_bytes))
        with pytest.raises(RuntimeError, match="link neighbors"):
            proxy.set_links(1, 2, 1, 2, 4096, 0)
        with pytest.raises(RuntimeError, match="link neighbors"):
            proxy.set_links(0, 2, 0, 2, 4096, 0)
        offset = proto.chain_offset(session.arena.total_bytes)
        # The neighbor checks come before the geometry's, so an area that does not fit still names them.
        with pytest.raises(RuntimeError, match="ring neighbors"):
            proxy.set_links(0, 2, 1, 2, 4096, offset, ring_prev=2, ring_next=0)
        with pytest.raises(RuntimeError, match="ring window"):
            proxy.set_links(0, 2, 1, 2, 4096, offset, ring_prev=0, ring_next=2, ring_window=1000)
        proxy.set_links(0, 2, 1, 0, 0, 0, ring_prev=0, ring_next=2)
        stats = proxy.stats()
        assert stats["link_ops"] == 0 and stats["link_items_posted"] == 0 and stats["link_credits_sent"] == 0
        assert offset > 0
    finally:
        session.close()
