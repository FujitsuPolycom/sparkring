"""Point-to-point protocol arithmetic, settings and forward-window budgets (torch-free, any host)."""

from __future__ import annotations

import pytest

from sparkring_sircl import routes
from sparkring_sircl.p2p import budget, protocol
from sparkring_sircl.p2p.settings import P2PSettings, SettingError
from sparkring_sircl.protocol import ProtocolError

S = 8192


def test_the_arena_layout_keeps_every_area_in_its_block():
    layout = protocol.P2PLayout(8, 2, 8, 512 << 10)
    assert layout.recv_off == 0 and layout.send_off == 8 * (512 << 10) and layout.flag_off == 2 * 8 * (512 << 10)
    assert layout.desc_off == layout.flag_off + 8 * 2 * 128
    assert [layout.ready_off, layout.consumed_off, layout.sent_off, layout.credit_off] == [
        layout.desc_off + 128 * k for k in range(1, 5)]
    assert layout.block_bytes % 4096 == 0 and layout.block_bytes >= layout.credit_off + 128
    assert layout.total_bytes == protocol.CONTROL_BYTES + 8 * layout.block_bytes
    assert layout.block(3) == protocol.CONTROL_BYTES + 3 * layout.block_bytes
    assert layout.header(2, 5) == layout.flag(2, 5, 0) + 4
    assert layout.flag(2, 5, 1) == layout.flag(2, 5, 0) + 128
    assert len(layout.as_tuple()) == protocol.LAYOUT_WORDS
    # The defaults: about 8 MiB per peer.
    assert protocol.P2PLayout(2, 2, protocol.DEFAULT_SLOTS, protocol.DEFAULT_SLOT_BYTES).block_bytes == (8 << 20) + 4096


@pytest.mark.parametrize("args", [(1, 2, 8, S), (17, 2, 8, S), (4, 3, 8, S), (4, 2, 6, S), (4, 2, 64, S),
                                  (4, 2, 1, S), (4, 2, 8, S + 16), (4, 2, 8, 0), (4, 2, 8, 1 << 30)])
def test_the_layout_refuses_geometry_the_wire_cannot_carry(args):
    with pytest.raises(ProtocolError):
        protocol.P2PLayout(*args)


@pytest.mark.parametrize("nbytes, count", [(0, 1), (1, 1), (15, 1), (16, 1), (S, 1), (S + 1, 2), (3 * S - 5, 3),
                                           (3 * S, 3)])
def test_items_cover_the_padded_message_in_slots(nbytes, count):
    assert protocol.items(nbytes, S) == count
    sizes = [protocol.item_bytes(nbytes, S, index) for index in range(count)]
    assert sum(sizes) == protocol.padded(nbytes)
    assert all(size % 16 == 0 and size <= S for size in sizes)
    headers = [protocol.header(nbytes, S, index) for index in range(count)]
    assert [bool(h & protocol.LAST) for h in headers] == [False] * (count - 1) + [True]
    assert headers[-1] & protocol.TAIL_MASK == nbytes % 16
    assert [h & protocol.BYTES_MASK for h in headers] == sizes


def test_headers_tell_messages_of_different_sizes_apart():
    # The first differing item tells a receive of another size, or another message boundary, apart.
    assert protocol.header(17, S, 0) != protocol.header(32, S, 0)
    assert protocol.header(S, S, 0) != protocol.header(2 * S, S, 0)          # last bit
    assert protocol.header(0, S, 0) == protocol.LAST
    assert "last item of a message of 16k+3 bytes" in protocol.describe_header(protocol.header(35, S, 0))
    assert protocol.tag(0xFFFFFFFF) == 0 and protocol.tag(4) == 5


def test_item_stripes_split_packs_over_the_lanes():
    assert protocol.item_stripes(48, 2) == ((0, 2), (2, 1))
    assert protocol.item_stripes(0, 2) == ((0, 0), (0, 0))
    with pytest.raises(ProtocolError):
        protocol.item_stripes(20, 2)


def test_settings_from_the_environment_and_their_refusals():
    settings = P2PSettings.from_env({"SIRCL_P2P_SLOTS": "4", "SIRCL_P2P_SLOT_BYTES": "65536",
                                     "SIRCL_SERVING_WAIT_S": "2.5"})
    assert (settings.slots, settings.slot_bytes, settings.serving_wait_s) == (4, 65536, 2.5)
    assert P2PSettings.from_env({}) == P2PSettings()
    for environ in ({"SIRCL_P2P_SLOTS": "6"}, {"SIRCL_P2P_SLOT_BYTES": "1000"}, {"SIRCL_P2P_BLOCKS": "0"},
                    {"SIRCL_P2P_THREADS": "48"}, {"SIRCL_P2P_WINDOW_BYTES": "1000"},
                    {"SIRCL_P2P_CHUNK_BYTES": "20"}, {"SIRCL_STARTUP_WAIT_S": "x"}):
        with pytest.raises(SettingError):
            P2PSettings.from_env(environ)


def test_pipeline_groups_follow_vllms_rank_layout():
    assert budget.pp_groups(8, tp=4, pp=2) == ((0, 4), (1, 5), (2, 6), (3, 7))
    assert budget.pp_groups(4, tp=2, pp=2) == ((0, 2), (1, 3))
    assert budget.pp_groups(8, tp=2, pp=4) == ((0, 2, 4, 6), (1, 3, 5, 7))
    assert budget.pp_groups(8, tp=1, pp=8) == (tuple(range(8)),)
    assert budget.pp_groups(8, tp=2, pp=2, dp=2) == ((0, 2), (1, 3), (4, 6), (5, 7))
    with pytest.raises(ValueError):
        budget.pp_groups(6, tp=4, pp=2)


def _ring8(positions):
    return routes.Layout(routes.Fabric.ring(8), tuple(positions))


def _totals(allocation, lane_sets, reserved):
    used = dict(reserved)
    for lanes in lane_sets:
        for key, members in lanes.queues().items():
            used[key] = used.get(key, 0) + sum(allocation.windows.get((lanes.name, r, p, lane), 0) for r, p, lane in members)
    return used


def test_pipeline_pairs_between_two_tp4_groups_share_the_queues_the_sessions_leave():
    queue = routes.DEFAULT_HAIRPIN_QUEUE
    share = budget.queue_share(queue)
    sessions = [budget.LaneSet.of(f"tp:{i}", routes.Layout(routes.Fabric.path(members), tuple(members)))
                for i, members in enumerate(((0, 1, 2, 3), (4, 5, 6, 7)))]
    reserved: dict = {}
    for lanes in sessions:
        for key, value in budget.session_reservation(lanes, budget.SessionSettings()).items():
            reserved[key] = reserved.get(key, 0) + value
    # A path of four: two lanes of 128 KiB through each middle relay queue in each direction and function.
    assert reserved[(1, False, 0)] == 2 * 131072
    pp = [budget.LaneSet.of(f"pp:{i}", _ring8(pair)) for i, pair in enumerate(((0, 4), (1, 5), (2, 6), (3, 7)))]
    allocation = budget.allocate([pp], reserved=reserved, max_window=131072, chunk=32768, queue_bytes=queue)
    assert not allocation.unavailable
    # Spark 5's primary queue toward Spark 6 holds 4->6 and 4->7 of tp:1 (256 KiB) and the PP lanes 2->6 and
    # 3->7: 128 KiB left for two lanes. Every PP lane crosses such a queue in one direction or the other.
    assert set(allocation.windows.values()) == {65536}
    assert len(allocation.windows) == 4 * 2 * 2
    assert all(total <= share for total in _totals(allocation, pp, reserved).values())


def test_a_queue_the_sessions_fill_leaves_its_pipeline_lanes_without_a_window():
    full = budget.SessionSettings(forward_window=0)
    tp = budget.LaneSet.of("tp:0", routes.Layout(routes.Fabric.path((0, 1, 2, 3)), (0, 1, 2, 3)))
    reserved = budget.session_reservation(tp, full)
    assert set(reserved.values()) == {budget.queue_share(routes.DEFAULT_HAIRPIN_QUEUE)}
    pp = budget.LaneSet.of("pp:0", _ring8((0, 4)))
    allocation = budget.allocate([[pp]], reserved=reserved, max_window=131072, chunk=32768)
    pairs = allocation.unavailable_pairs("pp:0")
    assert set(pairs) == {(0, 1), (1, 0)}
    assert pairs[(0, 1)].startswith("relay 1 primary function toward port 0 has 0 of its 393216 bytes left for 1 "
                                    "point-to-point lanes")


def test_pipeline_lanes_take_their_windows_before_the_session_groups_channels():
    tp = budget.LaneSet.of("tp:0", routes.Layout(routes.Fabric.path((0, 1, 2, 3)), (0, 1, 2, 3)))
    reserved = budget.session_reservation(tp, budget.SessionSettings())
    pp = budget.LaneSet.of("pp:0", _ring8((0, 4)))
    tp_p2p = dataclass_rename(tp, "tp:0/p2p")
    allocation = budget.allocate([[pp], [tp_p2p]], reserved=reserved, max_window=131072, chunk=32768)
    # The relay queue of Spark 1 toward Spark 2 (primary function): 256 KiB reserved, the PP lane gets the rest.
    assert allocation.windows[("pp:0", 0, 1, 0)] == 131072
    # The TP group's own relayed channels find no room left on their primary-function lanes (a pair needs a
    # window on every relayed lane); their secondary-function lanes share the other queues.
    assert set(allocation.unavailable_pairs("tp:0/p2p")) == {(0, 2), (0, 3), (1, 3), (2, 0), (3, 0), (3, 1)}
    assert {w for (name, _, _, lane), w in allocation.windows.items() if name == "tp:0/p2p"} == {65536}


def dataclass_rename(lanes: budget.LaneSet, name: str) -> budget.LaneSet:
    return budget.LaneSet(name, lanes.layout, lanes.route_maps, lanes.pairs)


def test_one_group_alone_gets_the_routes_rule_windows():
    lanes = budget.LaneSet.of("pp:0", _ring8((0, 2, 4, 6)), pairs=[(a, b) for a in range(4) for b in range(4)
                                                                   if a != b and abs(a - b) in (1, 3)])
    table, unavailable = budget.group_windows(lanes, 2, max_window=131072, chunk=32768)
    assert not unavailable
    # Adjacent stages two Sparks apart: one relay; every relayed lane has a window, direct lanes none.
    assert table[0][1] == [131072, 131072] and table[0][3] == [131072, 131072]
    assert table[0][2] == [0, 0]


def test_windows_off_leave_relayed_channels_unavailable():
    lanes = budget.LaneSet.of("pp:0", _ring8((0, 4)))
    table, unavailable = budget.group_windows(lanes, 2, max_window=0, chunk=32768)
    assert set(unavailable) == {(0, 1), (1, 0)} and "SIRCL_P2P_WINDOW_BYTES=0" in unavailable[(0, 1)]
