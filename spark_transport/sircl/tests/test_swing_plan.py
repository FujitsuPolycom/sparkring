"""Swing all-reduce plan and relay bound (``sparkring_sircl.swing_plan``)."""

from __future__ import annotations

import pytest

from sparkring_sircl import protocol as proto
from sparkring_sircl import routes
from sparkring_sircl import swing_plan as swing
from sparkring_sircl.pieces import padded
from vectors import load


def test_steps_equal_the_vectors():
    for key, entry in load("numeric.json")["swing"].items():
        world = int(key.split("=")[1])
        assert list(proto.swing_chunk_owners(world)) == entry["chunk_owners"]
        for rank_text, phases in entry["phases"].items():
            rank = int(rank_text)
            steps = swing.swing_steps(world, rank)
            count = len(steps)
            assert 2 * count == len(phases) == swing.phase_count(world)
            for step in steps:
                sent = phases[step.step]
                kept = phases[2 * count - 1 - step.step]
                assert [step.peer, *step.send, 0] == sent
                assert [step.peer, *step.keep, 1] == kept
            for phase, expected in enumerate(phases):
                assert list(swing.phase_range(world, rank, phase)) == expected[:3]


@pytest.mark.parametrize("world", [2, 4, 8, 16])
def test_steps_partition_and_pair(world):
    owners = proto.swing_chunk_owners(world)
    for rank in range(world):
        steps = swing.swing_steps(world, rank)
        held = (0, world)
        for step in steps:
            # The sent and kept ranges are the two equal halves of the range held before the step.
            lower, upper = sorted([step.send, step.keep])
            assert lower[0] == held[0] and lower[1] == upper[0] and upper[1] == held[1]
            assert step.send[1] - step.send[0] == step.keep[1] - step.keep[0]
            # The peer keeps what this rank sends, and sends back what this rank keeps.
            mirror = swing.swing_steps(world, step.peer)[step.step]
            assert mirror.peer == rank and mirror.keep == step.send and mirror.send == step.keep
            held = step.keep
        assert held == (owners.index(rank), owners.index(rank) + 1)


@pytest.mark.parametrize("world", [2, 4, 8])
def test_descriptor_words_decode_to_the_phases(world):
    for rank in range(world):
        words = swing.descriptor_words(world, rank)
        assert len(words) == swing.phase_count(world)
        for phase, word in enumerate(words):
            decoded = proto.decode_descriptor(word)
            peer, first, end = swing.phase_range(world, rank, phase)
            assert (decoded.peer, decoded.first, decoded.end) == (peer, first, end)
            assert decoded.namespace == (0 if phase < len(words) // 2 else 1)


def test_availability_needs_a_power_of_two_and_multi_phase_ops():
    assert [w for w in range(1, 18) if swing.available(w)] == [2, 4, 8, 16]
    assert not swing.available(8, multi_phase=False)
    for world in (3, 5, 6, 7):
        with pytest.raises(proto.ProtocolError):
            swing.swing_steps(world, 0)


def _exact_queue_load(layout, lanes, message):
    """Bytes through every relay queue in one Swing op of ``message`` bytes (exact stripes)."""
    world = layout.world
    derived = routes.derive_routes(layout, lanes)
    maps = [derived.route_map(rank) for rank in range(world)]
    packs = message // 16
    load = {}
    for key, members in routes.relay_queues(layout, maps).items():
        total = 0
        for rank, peer, lane in members:
            for step in swing.swing_steps(world, rank):
                if step.peer != peer:
                    continue
                for first, end in (step.send, step.keep):
                    lo, hi = swing.position_packs(packs, world, first, end)
                    total += 16 * proto.stripe(hi - lo, lanes, lane)[1]
        load[key] = total
    return maps, load


def test_relay_bound_on_the_ring_of_eight():
    layout = routes.Layout.parse("ring:8")
    maps, _ = _exact_queue_load(layout, 2, 16)
    # Only the distance-3 exchange of the third step crosses relays: one lane per
    # queue, one eighth of the message per lane over the two phases.
    bound = swing.relay_safe_message_bytes(layout, maps, 2)
    assert bound == (393216 - 64) * 8
    _, load = _exact_queue_load(layout, 2, bound)
    assert max(load.values()) <= 0.75 * 524288


@pytest.mark.parametrize("text, lanes", [("path:0-3", 2), ("ring:8:0,2,4,6", 2), ("ring:8:0,1,2,3", 2),
                                         ("ring:8:0,2,4,6", 1), ("ring:8:0,4", 2), ("ring:8", 1)])
def test_relay_bound_keeps_every_queue_within_its_share(text, lanes):
    layout = routes.Layout.parse(text)
    derived = routes.derive_routes(layout, lanes)
    maps = [derived.route_map(rank) for rank in range(layout.world)]
    bound = swing.relay_safe_message_bytes(layout, maps, lanes)
    assert bound is not None and bound % 16 == 0
    for message in (bound, padded(bound // 3), 16):
        _, load = _exact_queue_load(layout, lanes, message)
        assert max(load.values()) <= 0.75 * 524288


@pytest.mark.parametrize("text", ["ring:2", "ring:4", "ring:8:0,1"])
def test_no_relay_bound_without_relayed_swing_lanes(text):
    layout = routes.Layout.parse(text)
    derived = routes.derive_routes(layout, 2)
    maps = [derived.route_map(rank) for rank in range(layout.world)]
    assert swing.relay_safe_message_bytes(layout, maps, 2) is None


def test_relay_bound_refuses_groups_without_swing():
    layout = routes.Layout.parse("ring:6")
    derived = routes.derive_routes(layout, 2)
    with pytest.raises(proto.ProtocolError):
        swing.relay_safe_message_bytes(layout, [derived.route_map(r) for r in range(6)], 2)


@pytest.mark.parametrize("world", [2, 4, 8, 16])
def test_phase_send_volume_halves_then_doubles(world):
    nbytes = float(1 << 21)
    sends = swing.phase_send_bytes(world, nbytes)
    steps = swing.phase_count(world) // 2
    forward = tuple(nbytes / 2 ** (step + 1) for step in range(steps))
    assert sends == forward + forward[::-1]
    # Every rank sends 2 (W - 1) / W of the message, the volume of the two-shot all-reduce.
    assert sum(sends) == pytest.approx(2 * (world - 1) / world * nbytes)
    assert swing.phase_distances(8) == (1, 1, 3, 3, 1, 1)
    assert swing.phase_distances(16) == (1, 1, 3, 5, 5, 3, 1, 1)


def test_the_bounds_model_carries_the_swing_schedule():
    from sparkring_sircl import bounds

    nbytes = 2 << 20
    transfers = bounds.flows("all_reduce", "swing", 8, nbytes)
    egress = [0.0] * 8
    for source, _, size in transfers:
        egress[source] += size
    assert egress == [1.75 * nbytes] * 8
    layout = routes.Layout.parse("ring:8")
    loads = bounds.loads(routes.derive_routes(layout, 2), transfers)
    # The host interface binds Swing on the ring of eight (1.75 of the message each way); no cable direction
    # carries more than 1.5 of it. The two-shot all-reduce of the same message is cable-bound at 2.
    assert max(loads.ingress) == 1.75 * nbytes and max(loads.cables.values()) == 1.5 * nbytes
    assert loads.limit() == "host"
    swing_us = bounds.bound_seconds(layout, 2, "all_reduce", "swing", nbytes) * 1e6
    twoshot_us = bounds.bound_seconds(layout, 2, "all_reduce", "twoshot", nbytes) * 1e6
    assert swing_us == pytest.approx(1.75 * nbytes / 24e3) and twoshot_us == pytest.approx(2 * nbytes / 24e3)
    with pytest.raises(bounds.BoundError, match="all-reduce schedule"):
        bounds.flows("reduce_scatter", "swing", 8, nbytes)
    with pytest.raises(bounds.BoundError, match="power-of-two"):
        bounds.flows("all_reduce", "swing", 6, nbytes)


def test_phase_serial_bound_on_the_ring_of_eight():
    from sparkring_sircl import bounds

    layout = routes.Layout.parse("ring:8")
    nbytes = 2 << 20
    unit = nbytes / 24e9
    phases = swing.phase_seconds(layout, 2, nbytes)
    # The distance-3 phases put two transfers of 1/8 of the message on every other cable direction: a quarter
    # of the message at the cable rate, twice their host-interface time.
    assert phases == pytest.approx((unit / 2, unit / 4, unit / 4, unit / 4, unit / 4, unit / 2))
    assert swing.phase_send_bytes(8, nbytes)[2] / 24e9 == pytest.approx(unit / 8)
    # Run phase by phase, Swing's least time equals the two-shot bound; the whole-op bound is the host's.
    assert sum(phases) == pytest.approx(bounds.bound_seconds(layout, 2, "all_reduce", "twoshot", nbytes))
    assert sum(phases) > bounds.bound_seconds(layout, 2, "all_reduce", "swing", nbytes)
    # A sum of per-phase maxima is never below the maximum of the summed loads; groups without Swing raise.
    path = routes.Layout.parse("path:0-3")
    assert sum(swing.phase_seconds(path, 2, nbytes)) >= bounds.bound_seconds(path, 2, "all_reduce", "swing",
                                                                             nbytes)
    with pytest.raises(proto.ProtocolError):
        swing.phase_seconds(routes.Layout.parse("ring:6"), 2, nbytes)
