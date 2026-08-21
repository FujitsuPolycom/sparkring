from __future__ import annotations

from dataclasses import replace

import pytest

from . import decode_payload_contract as decode_contract
from . import dynamic_payload_planner as prototype


def test_registry_extends_the_decode_formula_contract_without_drift() -> None:
    assert set(prototype.FAMILY_REGISTRY) == set(
        decode_contract.PAYLOAD_FAMILY_BY_KEY
    )
    for tag, family in prototype.FAMILY_REGISTRY.items():
        assert family.bytes_per_q.coefficient == (
            decode_contract.PAYLOAD_FAMILY_BY_KEY[tag].bytes_per_query_row
        )
        assert family.arena.registered_horizon_q == 4096
        assert family.arena.capacity_bytes > (
            family.byte_geometry(4096)[2]
        )


@pytest.mark.parametrize("query_rows", range(1, 41))
@pytest.mark.parametrize("family_tag", prototype.FAMILY_REGISTRY)
def test_every_registered_decode_family_q1_q40_is_latency_admitted(
    family_tag: str,
    query_rows: int,
) -> None:
    decision = prototype.admit_payload(
        prototype.descriptor_for(family_tag, query_rows)
    )
    assert decision.code is prototype.DecisionCode.ADMIT_LATENCY
    assert decision.tier is prototype.TransportTier.LATENCY
    assert decision.chunk_count == 1


@pytest.mark.parametrize("query_rows", [48, 512, 4096])
@pytest.mark.parametrize("family_tag", prototype.FAMILY_REGISTRY)
def test_larger_and_prefill_like_widths_admit_without_enumeration(
    family_tag: str,
    query_rows: int,
) -> None:
    decision = prototype.admit_payload(
        prototype.descriptor_for(family_tag, query_rows)
    )
    assert decision.admitted
    assert decision.code in {
        prototype.DecisionCode.ADMIT_LATENCY,
        prototype.DecisionCode.ADMIT_CHUNKED,
    }


def test_large_vocabulary_width_uses_chunked_tier() -> None:
    decision = prototype.admit_payload(
        prototype.descriptor_for("vocabulary", 4096)
    )
    assert decision.code is prototype.DecisionCode.ADMIT_CHUNKED
    assert decision.chunk_count is not None
    assert decision.chunk_count > 1


def test_unknown_family_does_not_admit_on_same_byte_collision() -> None:
    indexer_q3 = prototype.descriptor_for("indexer", 3)
    collision = replace(indexer_q3, family_tag="unregistered_49152_family")
    assert collision.payload_bytes == 49_152
    decision = prototype.admit_payload(collision)
    assert decision.code is prototype.DecisionCode.REJECT_UNKNOWN_FAMILY
    assert not decision.admitted


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("collective", "all_reduce"),
        ("dtype", "bf16"),
        ("layout", "contiguous-row-major:[Q,8192]"),
        ("payload_bytes", 49_151),
    ],
)
def test_known_family_contract_mismatch_fails_closed(
    field: str,
    value: object,
) -> None:
    descriptor = replace(
        prototype.descriptor_for("indexer", 3),
        **{field: value},
    )
    decision = prototype.admit_payload(descriptor)
    assert decision.code is prototype.DecisionCode.REJECT_CONTRACT_MISMATCH
    assert not decision.admitted


def test_noncontiguous_payload_fails_closed() -> None:
    decision = prototype.admit_payload(
        prototype.descriptor_for("indexer", 3, contiguous=False)
    )
    assert decision.code is prototype.DecisionCode.REJECT_NONCONTIGUOUS


def test_arena_boundary_chunks_then_fails_closed_on_overflow() -> None:
    family = prototype.FAMILY_REGISTRY["vocabulary"]
    bytes_per_q = family.byte_geometry(1)[2]
    maximum_fitting_q = family.arena.capacity_bytes // bytes_per_q

    fitting = prototype.admit_payload(
        prototype.descriptor_for("vocabulary", maximum_fitting_q)
    )
    overflowing = prototype.admit_payload(
        prototype.descriptor_for("vocabulary", maximum_fitting_q + 1)
    )

    assert fitting.code is prototype.DecisionCode.ADMIT_CHUNKED
    assert overflowing.code is prototype.DecisionCode.REJECT_OVERFLOW
    assert overflowing.arena_bytes is not None
    assert overflowing.arena_capacity_bytes is not None
    assert overflowing.arena_bytes > overflowing.arena_capacity_bytes


def test_padding_metrics_are_byte_weighted_and_exact() -> None:
    metrics = prototype.padding_metrics(
        (24,),
        (
            prototype.CensusSample(
                "indexer",
                23,
                10,
                prototype.CensusRoute.PADDED,
            ),
        ),
    )
    assert len(metrics) == 1
    metric = metrics[0]
    # Indexer all-gather has a 4x output footprint: 65,536 bytes per Q.
    assert metric.target_bucket == 24
    assert metric.padding_rows == 1
    assert metric.logical_bytes == 23 * 65_536 * 10
    assert metric.padding_waste_bytes == 65_536 * 10
    assert metric.padding_waste_ppm == 43_479


def test_census_proposal_is_restart_only_and_deterministic() -> None:
    census = (
        prototype.CensusSample(
            "indexer",
            23,
            4_112,
            prototype.CensusRoute.PADDED,
        ),
        prototype.CensusSample(
            "vocabulary",
            69,
            800,
            prototype.CensusRoute.PADDED,
        ),
        prototype.CensusSample(
            "dcp_query",
            143,
            600,
            prototype.CensusRoute.EAGER,
        ),
        # Duplicate entries exercise deterministic aggregation.
        prototype.CensusSample(
            "vocabulary",
            512,
            100,
            prototype.CensusRoute.EAGER,
        ),
        prototype.CensusSample(
            "vocabulary",
            512,
            200,
            prototype.CensusRoute.EAGER,
        ),
    )
    buckets = (1, 2, 3, 4, 5, 8, 16, 24, 40, 72, 144)

    forward = prototype.propose_next_restart_plan(
        buckets,
        census,
        maximum_new_buckets=3,
    )
    reverse = prototype.propose_next_restart_plan(
        buckets,
        tuple(reversed(census)),
        maximum_new_buckets=3,
    )

    assert forward == reverse
    assert prototype.canonical_plan_json(forward) == (
        prototype.canonical_plan_json(reverse)
    )
    assert forward["activation"] == "next_restart_only"
    assert forward["live_capture"] is False
    assert len(forward["additions"]) == 3
    assert forward["before"]["padding_waste_bytes"] > (
        forward["after"]["padding_waste_bytes"]
    )
    assert forward["before"]["eager_observations"] > (
        forward["after"]["eager_observations"]
    )


def test_census_unknown_family_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown family"):
        prototype.propose_next_restart_plan(
            (1, 2, 4),
            (
                prototype.CensusSample(
                    "same_bytes_but_unknown",
                    3,
                    1_000,
                    prototype.CensusRoute.EAGER,
                ),
            ),
        )
