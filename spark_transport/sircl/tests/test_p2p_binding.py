"""The point-to-point binding (``p2p/_native.py``) on the test build of the native layer.

The contexts run their real progress threads over the in-memory verbs stand-in
(:class:`sparkring_sircl.testing.p2p_fabric.LocalChannels`); the host plays the
kernels' part with the same slots, words and headers.
"""

from __future__ import annotations

import ctypes
import time

import pytest

from sparkring_sircl import routes
from sparkring_sircl.p2p import _native
from sparkring_sircl.p2p import protocol as proto


@pytest.fixture(scope="session")
def p2p_library():
    from conftest import WORK, _require_compiler

    _require_compiler()
    from sparkring_sircl.testing import p2p_build

    return p2p_build.build_shared_library(WORK / ".build" / "sim")


def _payload(source: int, peer: int, index: int, nbytes: int) -> bytes:
    seed = (source * 131 + peer * 17 + index * 7) & 0xFF
    return bytes((seed + i * 13) & 0xFF for i in range(nbytes))


def _until(test, seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while not test():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


def test_the_native_layout_is_the_protocol_layout(p2p_library):
    lib = _native.load(p2p_library)
    assert lib.p2p_abi_version() == proto.ABI_VERSION
    for world, lanes, slots, slot_bytes in ((2, 1, 2, 4096), (4, 2, 8, 524288), (8, 2, 32, 1 << 20),
                                            (3, 2, 4, 8192)):
        assert _native.layout(world, lanes, slots, slot_bytes, library=lib) == \
            proto.P2PLayout(world, lanes, slots, slot_bytes).as_tuple()
    for world, lanes, slots, slot_bytes in ((1, 1, 8, 4096), (4, 0, 8, 4096), (4, 2, 3, 4096), (4, 2, 64, 4096),
                                            (4, 2, 8, 4000)):
        with pytest.raises(ValueError, match="unsupported point-to-point geometry"):
            _native.layout(world, lanes, slots, slot_bytes, library=lib)


def test_a_legacy_point_to_point_override_is_refused(tmp_path, monkeypatch):
    from native_stub import NativeStub

    monkeypatch.setattr(_native, "_LIBRARIES", {}, raising=False)
    legacy = NativeStub("p2p", _native.ABI_VERSION, None)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path, use_errno=True: legacy)
    with pytest.raises(RuntimeError, match="no local feature identity"):
        _native.load(tmp_path / "legacy.so")
    with pytest.raises(RuntimeError, match="local features 0x0; this binding requires 0x1"):
        _native._declare(NativeStub("p2p", _native.ABI_VERSION, 0))
    complete = NativeStub("p2p", _native.ABI_VERSION, _native.FEATURE_DESTROY_COUNT)
    assert _native._declare(complete) is complete


def test_the_point_to_point_source_build_reports_its_local_features(p2p_library):
    assert _native.local_features(_native.load(p2p_library)) == _native.FEATURE_DESTROY_COUNT


def test_a_library_of_another_abi_is_refused(p2p_library, monkeypatch):
    monkeypatch.setattr(_native, "ABI_VERSION", proto.ABI_VERSION + 1)
    monkeypatch.setattr(_native, "_LIBRARIES", {}, raising=False)
    with pytest.raises(RuntimeError, match="unexpected point-to-point native ABI version"):
        _native.load(p2p_library)


@pytest.mark.parametrize("layout_text, lanes", [("ring:2", 1), ("ring:4", 2), ("path:0-3", 2), ("ring:8", 2),
                                                ("path:0-2", 1)])
def test_every_pair_carries_exact_messages_in_order(p2p_library, layout_text, lanes):
    group = routes.Layout.parse(layout_text)
    channels = __import__("sparkring_sircl.testing.p2p_fabric", fromlist=["LocalChannels"]).LocalChannels(
        str(p2p_library), group, lanes=lanes, slots=4, slot_bytes=8192)
    sizes = [16, 4096, 4112, 8192, 8208, 3 * 8192 + 48, 40000] if group.world <= 4 else [16, 8208, 20000]
    try:
        world = group.world
        for source in range(world):
            for peer in range(world):
                if peer == source:
                    continue
                for index, nbytes in enumerate(sizes):
                    payload = _payload(source, peer, index, nbytes)
                    channels.send(source, peer, payload)
                    assert channels.recv(peer, source, nbytes) == payload
        items = sum(proto.items(n, 8192) for n in sizes)
        for rank, context in enumerate(channels.contexts):
            stats = context.stats()
            assert not context.failed(), context.error()
            assert stats["items_posted"] == (world - 1) * items
            assert stats["bytes_posted"] == (world - 1) * sum(proto.padded(n) for n in sizes)
            assert _until(lambda c=context: c.stats()["items_released"] == (world - 1) * items)
            for peer in range(world):
                if peer != rank:
                    assert stats["per_peer"][str(peer)]["items_posted"] == items
    finally:
        channels.close()


def test_a_sender_stages_two_rounds_of_slots_ahead_and_then_waits_for_credit(p2p_library):
    """A send slot is free once its item's write completed; the receiver's credit then holds the posting of the
    next round, so an idle receiver lets a sender stage ``2 * slots`` items and holds the next one."""
    import threading

    from sparkring_sircl.testing.p2p_fabric import LocalChannels

    channels = LocalChannels(str(p2p_library), routes.Layout.parse("path:0-3"), lanes=2, slots=2, slot_bytes=4096)
    try:
        messages = [_payload(0, 3, i, 4096) for i in range(6)]
        for message in messages[:4]:
            channels.send(0, 3, message)
        assert _until(lambda: channels.contexts[0].stats()["items_posted"] == 2)
        blocked = threading.Thread(target=channels.send, args=(0, 3, messages[4]))
        blocked.start()
        blocked.join(0.3)
        assert blocked.is_alive()                         # slot 0 still holds item 2, which waits for credit
        assert channels.contexts[0].stats()["items_posted"] == 2
        assert channels.recv(3, 0, 4096) == messages[0]
        blocked.join(10)
        assert not blocked.is_alive()
        assert channels.recv(3, 0, 4096) == messages[1]   # credit for item 3, whose write frees send slot 1
        channels.send(0, 3, messages[5])
        assert [channels.recv(3, 0, 4096) for _ in range(4)] == messages[2:]
        assert not any(context.failed() for context in channels.contexts)
    finally:
        channels.close()


def test_relayed_lanes_stay_within_their_windows(p2p_library):
    from sparkring_sircl.testing.p2p_fabric import LocalChannels

    channels = LocalChannels(str(p2p_library), routes.Layout.parse("path:0-3"), lanes=2, slots=4, slot_bytes=65536,
                             chunk_bytes=4096, max_window=16384)
    try:
        assert any(w for row in channels.windows for lanes in row for w in lanes)
        for index in range(6):
            payload = _payload(0, 3, index, 65536 * 2 + 32)
            channels.send(0, 3, payload)
            assert channels.recv(3, 0, len(payload)) == payload
        stats = channels.contexts[0].stats()
        largest = max(w for lanes in channels.windows[0] for w in lanes)
        assert 0 < stats["window_max_unacked_bytes"] <= largest
        assert stats["proven_bytes"] >= 0 and not channels.contexts[0].failed()
    finally:
        channels.close()


def test_channels_exist_only_between_the_pairs_named(p2p_library):
    from sparkring_sircl.testing.p2p_fabric import LocalChannels

    channels = LocalChannels(str(p2p_library), routes.Layout.parse("ring:4"), lanes=2, channels=[(0, 1), (2, 3)])
    try:
        channels.send(2, 3, b"x" * 64)
        assert channels.recv(3, 2, 64) == b"x" * 64
        assert set(channels.contexts[0].stats()["per_peer"]) <= {"1"}
        assert channels.contexts[2].stats()["per_peer"]["3"]["items_posted"] == 1
    finally:
        channels.close()


def test_a_receive_of_another_size_stops_every_rank_and_names_the_rank_that_found_it(p2p_library):
    from sparkring_sircl.testing.p2p_fabric import ChannelFailure, LocalChannels

    channels = LocalChannels(str(p2p_library), routes.Layout.parse("ring:4"), lanes=2)
    try:
        channels.send(1, 2, b"y" * 4096)
        with pytest.raises(ChannelFailure, match="the receive expected 4112 bytes"):
            channels.recv(2, 1, 4112)
        assert _until(lambda: all(context.failed() for context in channels.contexts))
        assert "recorded a failure" in channels.contexts[2].error()
        assert "kind 3" in channels.contexts[2].error()
        for rank in (0, 1, 3):
            assert "stopped after a failure on rank 2" in channels.contexts[rank].error()
            assert channels.contexts[rank].stats()["abort_from_rank"] == 2
    finally:
        channels.close()


def test_native_refusals_name_the_problem(p2p_library):
    from sparkring_sircl.testing.p2p_fabric import LocalChannels

    lib = _native.load(p2p_library)
    arena = proto.P2PLayout(2, 1, 2, 4096)
    buffer = ctypes.create_string_buffer(arena.total_bytes + 8192)
    address = ctypes.addressof(buffer) + (-ctypes.addressof(buffer)) % 4096
    common = dict(world_size=2, rank=0, gid_indices=[3], region_ptr=address, slots=2, slot_bytes=4096,
                  lane_count=1, library=lib)
    with pytest.raises(RuntimeError, match="not found"):
        _native.Native(hca_names=["missing-device"], peer_lane_devices=[(), (0,)], channels=[False, True],
                       region_bytes=arena.total_bytes, **common)
    with pytest.raises(ValueError, match="has no channel with rank 0 and names lane devices"):
        _native.Native(hca_names=["missing-device"], peer_lane_devices=[(), (0,)], channels=[False, False],
                       region_bytes=arena.total_bytes, **common)
    channels = LocalChannels(str(p2p_library), routes.Layout.parse("path:0-2"), lanes=1)
    try:
        context = channels.contexts[0]
        with pytest.raises(RuntimeError, match="before the progress thread starts"):
            context.set_base(5)
        with pytest.raises(ValueError, match="one entry per rank"):
            context.set_windows([[4096]], 4096)
    finally:
        channels.close()
