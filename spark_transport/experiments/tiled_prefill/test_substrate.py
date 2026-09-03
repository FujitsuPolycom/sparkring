from __future__ import annotations

from dataclasses import replace

import pytest

from spark_transport.experiments.tiled_prefill.substrate import (
    CapacityClass,
    MAX_TILE_GENERATION,
    TileCreditUnavailable,
    TileTicket,
    TiledCapacityPlanner,
    UnexpectedTileTicket,
    bidirectional_ring_allreduce_schedule,
    counter_rotating_allreduce_schedule,
    sequential_allgather_schedule,
    sequential_allreduce_schedule,
    split_relay_allgather_schedule,
    validate_tile_acquisition,
    verify_allgather_schedule,
    verify_allreduce_schedule,
    verify_bidirectional_ring_allreduce_schedule,
)


def test_arbitrary_query_widths_share_bounded_capacity_sessions() -> None:
    planner = TiledCapacityPlanner.tp4_bf16_allreduce()

    plans = {
        query_rows: planner.plan(query_rows)
        for query_rows in (
            1, 40, 41, 511, 512, 513, 4095, 4096, 4097, 8192
        )
    }

    assert plans[1].capacity_class == "latency_q40"
    assert plans[40].capacity_class == "latency_q40"
    assert plans[41].capacity_class == "medium_q512"
    assert plans[512].capacity_class == "medium_q512"
    assert plans[513].capacity_class == "streaming_q4096"
    assert plans[4096].capacity_class == "streaming_q4096"
    assert plans[4097].capacity_class == "extended_q8192"
    assert plans[8192].capacity_class == "extended_q8192"
    assert plans[1].session_key == plans[40].session_key
    assert plans[41].session_key == plans[512].session_key
    assert plans[513].session_key == plans[4096].session_key
    assert len({plan.session_key for plan in plans.values()}) == 4
    assert plans[4095].active_bytes == 4095 * 6144 * 2


def test_glm53_width4096_q2048_maps_to_exact_sixteen_mibibytes() -> None:
    planner = TiledCapacityPlanner.tp4_bf16_allreduce(
        elements_per_query_row=4096
    )

    plan = planner.plan(2048)

    assert plan.family == "tp4_allreduce_bf16_4096"
    assert plan.active_bytes == 16 * 1024 * 1024
    assert plan.tile_count == 32
    assert plan.capacity_class == "streaming_q4096"
    assert plan.session_key.bytes_per_query_row == 4096 * 2
    assert sum(tile.active_bytes for tile in plan.tiles) == plan.active_bytes

    maximum = planner.plan(8192)
    assert maximum.active_bytes == 64 * 1024 * 1024
    assert maximum.tile_count == 128


@pytest.mark.parametrize("elements", [0, -1, True])
def test_tp4_planner_rejects_invalid_row_geometry(elements: object) -> None:
    with pytest.raises(ValueError, match="elements_per_query_row"):
        TiledCapacityPlanner.tp4_bf16_allreduce(
            elements_per_query_row=elements  # type: ignore[arg-type]
        )


def test_operation_is_split_into_exact_generation_tagged_tile_descriptors() -> None:
    planner = TiledCapacityPlanner.tp4_bf16_allreduce()

    plan = planner.plan(513, first_tile_ordinal=6)

    assert plan.collective == "all_reduce"
    assert plan.dtype == "bf16"
    assert plan.reduction_algebra == "sum"
    assert plan.tile_bytes == 512 * 1024
    assert plan.slots_per_edge == 8
    assert len(plan.tiles) == 13
    assert plan.tiles[0].ticket == TileTicket(generation=1, slot=6)
    assert plan.tiles[1].ticket == TileTicket(generation=1, slot=7)
    assert plan.tiles[2].ticket == TileTicket(generation=2, slot=0)
    assert plan.tiles[-1].ticket == TileTicket(generation=3, slot=2)
    assert [tile.offset_bytes for tile in plan.tiles] == [
        index * plan.tile_bytes for index in range(13)
    ]
    assert all(
        tile.active_bytes == plan.tile_bytes for tile in plan.tiles[:-1]
    )
    assert plan.tiles[-1].active_bytes == 6144 * 2
    assert sum(tile.active_bytes for tile in plan.tiles) == plan.active_bytes


def test_cumulative_credit_guards_reuse_and_generation_mismatch() -> None:
    slots_per_edge = 8

    validate_tile_acquisition(
        TileTicket(generation=1, slot=7),
        expected_ordinal=7,
        consumed_through=-1,
        slots_per_edge=slots_per_edge,
    )
    with pytest.raises(TileCreditUnavailable, match="ordinal 0"):
        validate_tile_acquisition(
            TileTicket(generation=2, slot=0),
            expected_ordinal=8,
            consumed_through=-1,
            slots_per_edge=slots_per_edge,
        )
    validate_tile_acquisition(
        TileTicket(generation=2, slot=0),
        expected_ordinal=8,
        consumed_through=0,
        slots_per_edge=slots_per_edge,
    )
    with pytest.raises(UnexpectedTileTicket, match="expected generation=2 slot=0"):
        validate_tile_acquisition(
            TileTicket(generation=3, slot=0),
            expected_ordinal=8,
            consumed_through=0,
            slots_per_edge=slots_per_edge,
        )


def test_ticket_generation_is_bounded_and_zero_generation_is_unrepresentable() -> None:
    slots_per_edge = 8
    maximum_ordinal = MAX_TILE_GENERATION * slots_per_edge - 1

    assert TileTicket.from_ordinal(
        maximum_ordinal, slots_per_edge
    ) == TileTicket(generation=MAX_TILE_GENERATION, slot=7)
    with pytest.raises(OverflowError, match="generation exhausted"):
        TileTicket.from_ordinal(maximum_ordinal + 1, slots_per_edge)
    with pytest.raises(ValueError, match="nonnegative"):
        TileTicket.from_ordinal(-1, slots_per_edge)
    with pytest.raises(ValueError, match="positive"):
        TileTicket.from_ordinal(0, 0)


def test_counter_rotating_tree_halves_the_verified_bandwidth_critical_path() -> None:
    payload_bytes = 512 * 6144 * 2

    sequential = verify_allreduce_schedule(
        sequential_allreduce_schedule(payload_bytes)
    )
    counter_rotating = verify_allreduce_schedule(
        counter_rotating_allreduce_schedule(payload_bytes)
    )

    assert sequential.complete
    assert counter_rotating.complete
    assert sequential.contributor_masks == {"whole": (15, 15, 15, 15)}
    assert counter_rotating.contributor_masks == {
        "lower": (15, 15, 15, 15),
        "upper": (15, 15, 15, 15),
    }
    assert sequential.edge_bytes_per_rank == (payload_bytes, payload_bytes)
    assert counter_rotating.edge_bytes_per_rank == (
        payload_bytes,
        payload_bytes,
    )
    assert sequential.stage_edge_bytes_per_rank == (
        (payload_bytes, 0),
        (0, payload_bytes),
    )
    assert counter_rotating.stage_edge_bytes_per_rank == (
        (payload_bytes // 2, payload_bytes // 2),
        (payload_bytes // 2, payload_bytes // 2),
    )
    assert sequential.ideal_critical_path_bytes_per_rank == 2 * payload_bytes
    assert counter_rotating.ideal_critical_path_bytes_per_rank == payload_bytes


def test_bidirectional_ring_allreduce_reduces_q2048_wire_bytes() -> None:
    payload_bytes = 2048 * 4096 * 2

    verification = verify_bidirectional_ring_allreduce_schedule(
        bidirectional_ring_allreduce_schedule(payload_bytes)
    )

    assert verification.complete
    assert verification.stage_count == 6
    assert verification.edge_bytes_by_rank == (
        (12 * 1024 * 1024, 12 * 1024 * 1024),
    ) * 4
    assert verification.total_bytes_per_rank == 24 * 1024 * 1024
    assert verification.ideal_critical_path_bytes_per_rank == (
        12 * 1024 * 1024
    )
    assert all(
        masks == (15,) * 8
        for masks in verification.contributor_masks_by_rank
    )


def test_symbolic_expressions_expose_changed_bf16_association() -> None:
    payload_bytes = 40 * 6144 * 2

    sequential = verify_allreduce_schedule(
        sequential_allreduce_schedule(payload_bytes)
    )
    counter_rotating = verify_allreduce_schedule(
        counter_rotating_allreduce_schedule(payload_bytes)
    )

    sequential_fingerprint = sequential.association_fingerprints["whole"]
    assert len(set(sequential_fingerprint)) == 1
    assert (
        counter_rotating.association_fingerprints["lower"]
        == sequential_fingerprint
    )
    assert (
        counter_rotating.association_fingerprints["upper"]
        != sequential_fingerprint
    )


def test_split_relay_allgather_proves_completeness_at_half_the_critical_bytes() -> None:
    shard_bytes = 512 * 38720 * 2

    sequential = verify_allgather_schedule(
        sequential_allgather_schedule(shard_bytes)
    )
    split_relay = verify_allgather_schedule(
        split_relay_allgather_schedule(shard_bytes)
    )

    assert sequential.complete
    assert split_relay.complete
    assert all(len(segments) == 8 for segments in split_relay.segments_by_rank)
    assert sequential.edge_bytes_by_rank == (
        (shard_bytes, 2 * shard_bytes),
    ) * 4
    assert split_relay.edge_bytes_by_rank == (
        (3 * shard_bytes // 2, 3 * shard_bytes // 2),
    ) * 4
    assert sequential.ideal_critical_path_bytes_per_rank == 3 * shard_bytes
    assert (
        split_relay.ideal_critical_path_bytes_per_rank
        == 3 * shard_bytes // 2
    )


def test_every_q1_q4096_has_exact_tiles_and_reversible_tickets() -> None:
    planner = TiledCapacityPlanner.tp4_bf16_allreduce()

    for query_rows in range(1, 4097):
        first_ordinal = query_rows % 17
        plan = planner.plan(
            query_rows,
            first_tile_ordinal=first_ordinal,
        )
        assert sum(tile.active_bytes for tile in plan.tiles) == plan.active_bytes
        assert all(0 < tile.active_bytes <= plan.tile_bytes for tile in plan.tiles)
        assert tuple(
            tile.ticket.ordinal(plan.slots_per_edge) for tile in plan.tiles
        ) == tuple(
            range(first_ordinal, first_ordinal + len(plan.tiles))
        )
        assert plan.tiles[-1].offset_bytes + plan.tiles[-1].active_bytes == (
            plan.active_bytes
        )


def test_q4096_streams_through_a_fixed_four_mibibyte_pool_per_edge() -> None:
    plan = TiledCapacityPlanner.tp4_bf16_allreduce().plan(4096)

    assert plan.active_bytes == 48 * 1024 * 1024
    assert plan.tile_count == 96
    assert plan.arena_bytes_per_edge == 4 * 1024 * 1024


def test_session_key_includes_wire_geometry_but_excludes_exact_query_width() -> None:
    standard = TiledCapacityPlanner.tp4_bf16_allreduce()
    alternate_geometry = TiledCapacityPlanner(
        family="tp4_allreduce_bf16_6144",
        collective="all_reduce",
        dtype="bf16",
        reduction_algebra="sum",
        bytes_per_query_row=6144 * 2,
        capacity_classes=(CapacityClass("latency_q40", 40),),
        tile_bytes=256 * 1024,
        slots_per_edge=8,
    )

    assert standard.plan(1).session_key == standard.plan(40).session_key
    assert standard.plan(1).session_key != alternate_geometry.plan(1).session_key
    assert standard.plan(1).session_key.capacity_query_rows == 40
    assert standard.plan(1).session_key.tile_bytes == 512 * 1024


def test_planner_rejects_ambiguous_or_unbounded_session_geometry() -> None:
    common = {
        "family": "test_family",
        "collective": "all_reduce",
        "dtype": "bf16",
        "reduction_algebra": "sum",
        "bytes_per_query_row": 128,
        "tile_bytes": 1024,
        "slots_per_edge": 8,
    }

    with pytest.raises(ValueError, match="at least one capacity"):
        TiledCapacityPlanner(capacity_classes=(), **common)
    with pytest.raises(ValueError, match="strictly increasing"):
        TiledCapacityPlanner(
            capacity_classes=(
                CapacityClass("large", 512),
                CapacityClass("small", 40),
            ),
            **common,
        )
    with pytest.raises(ValueError, match="tile_bytes"):
        TiledCapacityPlanner(
            capacity_classes=(CapacityClass("small", 40),),
            **{**common, "tile_bytes": 0},
        )


def test_schedule_verifiers_reject_incomplete_or_acausal_plans() -> None:
    allreduce = sequential_allreduce_schedule(40 * 6144 * 2)
    incomplete = replace(allreduce, exchanges=allreduce.exchanges[:1])
    assert not verify_allreduce_schedule(incomplete).complete

    allgather = split_relay_allgather_schedule(512 * 38720 * 2)
    acausal = replace(
        allgather,
        transfers=tuple(
            replace(transfer, stage=1 - transfer.stage)
            for transfer in allgather.transfers
        ),
    )
    with pytest.raises(ValueError, match="unavailable segments"):
        verify_allgather_schedule(acausal)
