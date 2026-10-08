"""Setup agreement: one verdict on every rank from all ranks' setup records."""

from __future__ import annotations

import pytest

from sparkring_sircl import routes
from sparkring_sircl.agreement import agreement_failures


def _records(world: int, layout_text: str = "ring"):
    layout = routes.Layout.parse(f"ring:{world}") if layout_text == "ring" else routes.Layout.parse(layout_text)
    derived = routes.derive_routes(layout, 2)
    records = []
    for rank in range(world):
        route_map = derived.route_map(rank)
        records.append((None, b"record", {
            "api_version": 1, "proxy_abi": 1, "world_size": world, "lane_count": 2, "slot_bytes": 131072,
            "max_size": 131072, "dispatch_limit_bytes": 131072, "spin_limit": 20_000_000, "threads": 512,
            "algorithm": "auto", "layout": layout.identity(),
            "devices": sorted({d for devices in route_map.values() for d in devices}),
            "gid_indices": [3, 3, 3, 3],
            "lane_counts": [0 if peer == rank else 2 for peer in range(world)],
            "route_map": {str(peer): list(devices) for peer, devices in route_map.items()},
        }))
    return layout, records


@pytest.mark.parametrize("world", [2, 3, 4, 8])
def test_equal_configurations_pass(world):
    layout, records = _records(world)
    assert agreement_failures(records, layout) == []
    assert agreement_failures(records, None) == []


@pytest.mark.parametrize("field, value", [("spin_limit", 1), ("threads", 256), ("algorithm", "oneshot"),
                                          ("slot_bytes", 4096), ("api_version", 2)])
def test_one_differing_field_fails_every_rank_with_both_values(field, value):
    layout, records = _records(4)
    error, blob, record = records[2]
    records[2] = (error, blob, {**record, field: value})
    failures = agreement_failures(records, layout)
    assert len(failures) == 1
    assert f"rank 2 configuration differs from rank 0: {{'{field}'" in failures[0]
    assert repr(value) in failures[0]


def test_rank_local_fields_may_differ():
    layout, records = _records(3)
    error, blob, record = records[1]
    records[1] = (error, blob, {**record, "gid_indices": [5, 5, 5, 5], "devices": ["a", "b"]})
    assert agreement_failures(records, layout) == []


def test_a_native_failure_on_one_rank_is_reported_by_every_rank():
    layout, records = _records(3)
    records[1] = ("RDMA device rocep1s0f0 port 1 is not active", b"", {})
    assert agreement_failures(records, layout) == ["rank 1: RDMA device rocep1s0f0 port 1 is not active"]


def test_lane_counts_must_be_the_agreed_count():
    layout, records = _records(3)
    error, blob, record = records[0]
    records[0] = (error, blob, {**record, "lane_counts": [0, 2, 1]})
    assert any("rank 0 has lane counts [0, 2, 1]" in f for f in agreement_failures(records, layout))


def test_unpaired_lanes_fail_the_agreement():
    layout, records = _records(8)
    error, blob, record = records[4]
    route_map = dict(record["route_map"])
    route_map["0"] = ["rocep1s0f0", "roceP2p1s0f1"]           # the opposite rank's devices, not ours
    records[4] = (error, blob, {**record, "route_map": route_map})
    failures = agreement_failures(records, layout)
    assert failures and all("arrives on rank" in failure for failure in failures)
