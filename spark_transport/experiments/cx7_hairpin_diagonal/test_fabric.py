"""Behavior tests for the four-rank hardware-diagonal operator rig."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spark_transport.experiments.cx7_hairpin_diagonal import fabric


def _port(rank: int, direction: str, function: int) -> dict[str, object]:
    port = 0 if direction == "clockwise" else 1
    peer_rank = (rank + (1 if direction == "clockwise" else -1)) % 4
    peer_direction = (
        "counter_clockwise" if direction == "clockwise" else "clockwise"
    )
    domain = "" if function == 0 else "P2"
    return {
        "function": function,
        "netdev": f"en{domain}p1s0f{port}np{port}",
        "rdma_device": f"roce{domain}p1s0f{port}",
        "ipv4_cidr": f"198.18.{rank}.{1 + port * 2 + function}/32",
        "mac": f"02:{rank:02x}:{port:02x}:{function:02x}:00:01",
        "peer_rank": peer_rank,
        "peer_direction": peer_direction,
        "peer_function": function,
    }


def _document(*, functions: int = 2) -> dict[str, object]:
    ranks = []
    for rank in range(4):
        ranks.append(
            {
                "rank": rank,
                "ssh_alias": f"spark-r{rank}",
                "management_netdev": "enP7s7",
                "ports": {
                    direction: [
                        _port(rank, direction, function)
                        for function in range(functions)
                    ]
                    for direction in ("clockwise", "counter_clockwise")
                },
            }
        )
    return {
        "schema": "sparkring-cx7-hardware-diagonal-fabric/v1",
        "status": "research-only",
        "group_id": 77,
        "socket_direct_functions": functions,
        "flow_label_base": 1,
        "shared_diagonal_flow_label": False,
        "standard_ether_type": 0x0800,
        "marked_ether_type": 0x88B5,
        "standard_udp_destination_port": 4791,
        "roce_gid_index": 3,
        "bounded_runtime_seconds": 2,
        "expected_ethernet_mtu": 9000,
        "expected_roce_mtu": 4096,
        "orchestration": {
            "host_helper_path": "/opt/sparkring/bin/sparkring-cx7-fabric-helper",
            "remote_state_root": "/run/sparkring/cx7-fabric",
        },
        "ranks": ranks,
    }


def _write_topology(tmp_path: Path, document: object | None = None) -> Path:
    path = tmp_path / "fabric.json"
    path.write_text(json.dumps(document or _document()), encoding="utf-8")
    return path


def _qualified_gate(plan: fabric.FabricPlan) -> dict[str, object]:
    return {
        "schema": fabric.HARDWARE_GATE_SCHEMA,
        "status": "qualified",
        "topology_sha256": plan.topology_sha256,
        "plan_sha256": plan.sha256,
        "source_ethertype_rewrite_in_hw": True,
        "intermediate_ethertype_restore_in_hw": True,
        "remote_payload_byte_match": True,
        "rx_icrc_encapsulated_delta": 0,
        "cleanup_verified": True,
    }


def test_topology_runtime_limit_matches_the_native_marker_helper(
    tmp_path: Path,
) -> None:
    document = _document()
    document["bounded_runtime_seconds"] = 7200

    topology = fabric.load_topology(_write_topology(tmp_path, document))

    assert topology.bounded_runtime_seconds == 7200
    document["bounded_runtime_seconds"] = 7201
    with pytest.raises(fabric.FabricError, match="bounded_runtime_seconds"):
        fabric.load_topology(_write_topology(tmp_path, document))


def test_plan_generates_reciprocal_tp4_paths_and_reserved_flows(
    tmp_path: Path,
) -> None:
    topology = fabric.load_topology(_write_topology(tmp_path))
    plan = fabric.build_plan(topology)

    assert len(plan.paths) == 32
    assert len([path for path in plan.paths if path.kind == "direct"]) == 16
    diagonals = [path for path in plan.paths if path.kind == "diagonal"]
    assert len(diagonals) == 16
    assert len(plan.routes) == 32
    assert len(plan.markers) == 16
    assert len(plan.tc_rules) == 16
    assert sorted(marker.flow_label for marker in plan.markers) == [
        value for value in range(1, 9) for _ in range(2)
    ]
    assert sorted(marker.udp_source_port for marker in plan.markers) == [
        value for value in range(49153, 49161) for _ in range(2)
    ]
    assert plan.hardware_gate_status == "unproven"
    assert plan.apply_permitted is False
    assert {path.destination_rank for path in diagonals if path.source_rank == 0} == {
        2
    }
    assert {path.intermediate_rank for path in diagonals if path.source_rank == 0} == {
        1,
        3,
    }
    for marker in plan.markers:
        assert marker.match_ether_type == 0x0800
        assert marker.marked_ether_type == 0x88B5
        assert marker.match_udp_destination_port == 4791
    for rule in plan.tc_rules:
        assert rule.match_ether_type == 0x88B5
        assert rule.restore_ether_type == 0x0800
        source = next(path for path in diagonals if path.name == rule.path_name)
        assert rule.match_ethernet_source == next(
            rank for rank in plan.ranks if rank.rank == source.source_rank
        ).port(source.direction, source.function).mac
        assert rule.ingress_netdev != rule.egress_netdev
        reverse = next(
            item
            for item in plan.tc_rules
            if item.source_rank == rule.destination_rank
            and item.destination_rank == rule.source_rank
            and item.intermediate_rank == rule.intermediate_rank
            and item.function == rule.function
        )
        assert reverse.ingress_netdev == rule.egress_netdev
        assert reverse.egress_netdev == rule.ingress_netdev
        assert reverse.match_ethernet_source == rule.rewrite_ethernet_destination
        assert reverse.rewrite_ethernet_destination == rule.match_ethernet_source
        marker = next(item for item in plan.markers if item.path_name == rule.path_name)
        reverse_marker = next(
            item for item in plan.markers if item.path_name == reverse.path_name
        )
        assert reverse_marker.flow_label == marker.flow_label
        assert reverse_marker.udp_source_port == marker.udp_source_port


def test_udp_destination_marker_is_a_non_executable_negative_control(
    tmp_path: Path,
) -> None:
    plan = fabric.build_plan(fabric.load_topology(_write_topology(tmp_path)))
    control = plan.to_dict()["negative_controls"]["udp_port_mark_and_restore"]

    assert control["status"] == "unsupported"
    assert control["executable"] is False
    assert control["reason"] == "post-ICRC inner-header modification"
    assert not any(
        hasattr(marker, "marked_udp_destination_port") for marker in plan.markers
    )


def test_plan_exposes_exact_route_marker_and_restore_commands(tmp_path: Path) -> None:
    plan = fabric.build_plan(fabric.load_topology(_write_topology(tmp_path)))
    diagonal = next(path for path in plan.paths if path.kind == "diagonal")
    route = next(item for item in plan.routes if item.path_name == diagonal.name)
    marker = next(item for item in plan.markers if item.path_name == diagonal.name)
    rule = next(item for item in plan.tc_rules if item.path_name == diagonal.name)

    route_command = fabric.route_command(route, add=True)
    neighbor_command = fabric.neighbor_command(route, add=True)
    marker_command = fabric.marker_command(plan, marker, apply=True)
    rule_command = fabric.tc_rule_command(rule, add=True)

    assert route_command == [
        "ip",
        "route",
        "add",
        f"{route.destination_ipv4}/32",
        "dev",
        route.source_netdev,
        "src",
        route.source_ipv4,
        "scope",
        "link",
    ]
    assert neighbor_command == [
        "ip",
        "neigh",
        "add",
        route.destination_ipv4,
        "lladdr",
        route.next_hop_mac,
        "nud",
        "permanent",
        "dev",
        route.source_netdev,
    ]
    assert "--flow-label" in marker_command
    assert marker_command[marker_command.index("--flow-label") + 1] == str(
        marker.flow_label
    )
    assert marker_command[marker_command.index("--udp-source-port") + 1] == str(
        marker.udp_source_port
    )
    assert marker_command[marker_command.index("--match-ether-type") + 1] == "0x0800"
    assert marker_command[marker_command.index("--marked-ether-type") + 1] == "0x88b5"
    assert "--marked-udp-destination-port" not in marker_command
    assert "skip_sw" in rule_command
    assert rule_command[rule_command.index("protocol") + 1] == "0x88b5"
    assert rule_command[rule_command.index("src_mac") + 1] == (
        rule.match_ethernet_source
    )
    assert rule_command[rule_command.index("dst_mac") + 1] == (
        rule.match_ethernet_destination
    )
    assert ["munge", "eth", "type", "set", "0x0800"] == rule_command[
        rule_command.index("munge") : rule_command.index("munge") + 5
    ]
    assert rule.rewrite_ethernet_destination in rule_command
    assert rule.egress_netdev == rule_command[-1]
    assert not any("docker" in value.lower() or "model" in value.lower() for value in (
        route_command + neighbor_command + marker_command + rule_command
    ))


def test_cleanup_commands_are_exact_and_idempotent_by_contract(tmp_path: Path) -> None:
    plan = fabric.build_plan(fabric.load_topology(_write_topology(tmp_path)))
    diagonal = next(path for path in plan.paths if path.kind == "diagonal")
    route = next(item for item in plan.routes if item.path_name == diagonal.name)
    marker = next(item for item in plan.markers if item.path_name == diagonal.name)
    rule = next(item for item in plan.tc_rules if item.path_name == diagonal.name)

    assert fabric.route_command(route, add=False)[:4] == [
        "ip",
        "route",
        "del",
        f"{route.destination_ipv4}/32",
    ]
    assert fabric.neighbor_command(route, add=False) == [
        "ip",
        "neigh",
        "del",
        route.destination_ipv4,
        "dev",
        route.source_netdev,
    ]
    marker_cleanup = fabric.marker_command(plan, marker, apply=False)
    assert marker_cleanup[-1] == "--if-present"
    assert marker.path_name in marker_cleanup
    assert fabric.tc_rule_command(rule, add=False) == [
        "tc",
        "filter",
        "delete",
        "dev",
        rule.ingress_netdev,
        "ingress",
        "protocol",
        "0x88b5",
        "pref",
        str(rule.preference),
        "handle",
        str(rule.handle),
        "flower",
    ]


def test_apply_gate_requires_remote_bytes_zero_icrc_and_cleanup(
    tmp_path: Path,
) -> None:
    plan = fabric.build_plan(fabric.load_topology(_write_topology(tmp_path)))
    receipt = _qualified_gate(plan)

    fabric.require_hardware_gate(plan, receipt)
    for field in (
        "remote_payload_byte_match",
        "source_ethertype_rewrite_in_hw",
        "intermediate_ethertype_restore_in_hw",
        "cleanup_verified",
    ):
        failed = dict(receipt)
        failed[field] = False
        with pytest.raises(fabric.FabricError, match="hardware gate"):
            fabric.require_hardware_gate(plan, failed)
    failed = dict(receipt)
    failed["rx_icrc_encapsulated_delta"] = 1
    with pytest.raises(fabric.FabricError, match="hardware gate"):
        fabric.require_hardware_gate(plan, failed)


def test_apply_manifest_requires_all_four_token_and_hardware_gate(
    tmp_path: Path,
) -> None:
    plan = fabric.build_plan(fabric.load_topology(_write_topology(tmp_path)))
    gate = _qualified_gate(plan)

    with pytest.raises(fabric.FabricError, match="all-four authorization token"):
        fabric.build_apply_manifest(plan, gate, authorization_token="wrong")
    failed_gate = dict(gate, remote_payload_byte_match=False)
    with pytest.raises(fabric.FabricError, match="hardware gate"):
        fabric.build_apply_manifest(
            plan,
            failed_gate,
            authorization_token=fabric.AUTHORIZATION_TOKEN,
        )

    manifest = fabric.build_apply_manifest(
        plan,
        gate,
        authorization_token=fabric.AUTHORIZATION_TOKEN,
    )

    assert manifest["schema"] == fabric.APPLY_MANIFEST_SCHEMA
    assert manifest["status"] == "research-only"
    assert manifest["authorized_ranks"] == [0, 1, 2, 3]
    assert manifest["bounded_runtime_seconds"] == 2
    assert manifest["receipt_schema"] == fabric.RECEIPT_SCHEMA
    assert [phase["name"] for phase in manifest["apply_phases"]] == [
        "mtu_snapshot",
        "endpoint_routes",
        "intermediate_qdiscs",
        "intermediate_rules",
        "source_markers",
        "mtu_verify",
    ]
    assert manifest["cleanup"]["automatic_on_exit"] is True
    assert manifest["cleanup"]["idempotent"] is True
    assert [phase["name"] for phase in manifest["cleanup"]["phases"]] == [
        "source_markers",
        "intermediate_rules",
        "intermediate_qdiscs",
        "endpoint_neighbors",
        "endpoint_routes",
        "mtu_restore",
        "mtu_verify",
    ]
    commands = [
        command
        for phase in manifest["apply_phases"] + manifest["cleanup"]["phases"]
        for command in phase["commands"]
    ]
    assert {command["rank"] for command in commands} == {0, 1, 2, 3}
    assert all(isinstance(command["argv"], list) for command in commands)
    assert not any(
        "docker" in argument.lower() or "model" in argument.lower()
        for command in commands
        for argument in command["argv"]
    )


def test_rocenante_selection_uses_24_origin_qps_and_balances_each_rank_function(
    tmp_path: Path,
) -> None:
    complete = fabric.build_plan(fabric.load_topology(_write_topology(tmp_path)))
    selected = fabric.build_rocenante_plan(complete)
    inventory = fabric.rocenante_inventory(selected)

    direct = [path for path in selected.paths if path.kind == "direct"]
    diagonal = [path for path in selected.paths if path.kind == "diagonal"]
    assert len(direct) == 16
    assert len(diagonal) == 8
    assert len(selected.paths) == 24
    assert len(selected.routes) == 8
    assert len(selected.markers) == 8
    assert len(selected.tc_rules) == 8
    assert inventory["origin_qps_per_rank"] == {str(rank): 6 for rank in range(4)}
    assert inventory["direct_origin_qps"] == 16
    assert inventory["forwarded_origin_qps"] == 8
    assert inventory["total_origin_qps"] == 24
    assert all(
        {
            path.function
            for path in diagonal
            if path.source_rank == rank
        }
        == {0, 1}
        for rank in range(4)
    )
    assert {
        (path.intermediate_rank, path.function)
        for path in diagonal
        if {path.source_rank, path.destination_rank} == {0, 2}
    } == {(1, 0), (3, 1)}
    assert {
        (path.intermediate_rank, path.function)
        for path in diagonal
        if {path.source_rank, path.destination_rank} == {1, 3}
    } == {(2, 0), (0, 1)}
    assert inventory["diagonal_edge_function_traversals"] == {
        "0-1/function0": 2,
        "0-1/function1": 2,
        "0-3/function0": 0,
        "0-3/function1": 4,
        "1-2/function0": 4,
        "1-2/function1": 0,
        "2-3/function0": 2,
        "2-3/function1": 2,
    }
    assert inventory["origin_qps_per_rank_and_function"] == {
        f"rank{rank}/function{function}": 3
        for rank in range(4)
        for function in (0, 1)
    }
    assert len(inventory["plan_sha256"]) == 64


def test_rocenante_cleanup_is_scoped_to_eight_selected_diagonals(
    tmp_path: Path,
) -> None:
    complete = fabric.build_plan(fabric.load_topology(_write_topology(tmp_path)))
    selected = fabric.build_rocenante_plan(complete)
    cleanup = fabric.build_cleanup_manifest(selected)

    assert cleanup["schema"] == fabric.CLEANUP_MANIFEST_SCHEMA
    assert cleanup["automatic_on_exit"] is True
    assert cleanup["idempotent"] is True
    phases = {phase["name"]: phase["commands"] for phase in cleanup["phases"]}
    assert len(phases["source_markers"]) == 8
    assert len(phases["intermediate_rules"]) == 8
    assert len(phases["intermediate_qdiscs"]) == 8
    assert len(phases["endpoint_neighbors"]) == 8
    assert len(phases["endpoint_routes"]) == 8
    assert len(phases["mtu_restore"]) == 4
    assert len(phases["mtu_verify"]) == 4
    for command in phases["intermediate_rules"]:
        argv = command["argv"]
        assert argv[2] == "delete"
        assert argv[argv.index("protocol") + 1] == "0x88b5"
        assert "pref" in argv and "handle" in argv
    for command in phases["intermediate_qdiscs"]:
        argv = command["argv"]
        assert argv[:3] == [
            "/opt/sparkring/bin/sparkring-cx7-fabric-helper",
            "qdisc",
            "cleanup",
        ]
        assert argv[-1] == "--if-owned-and-empty"


def test_rocenante_native_path_arguments_pair_reciprocal_routes(
    tmp_path: Path,
) -> None:
    complete = fabric.build_plan(fabric.load_topology(_write_topology(tmp_path)))
    selected = fabric.build_rocenante_plan(complete)
    arguments = fabric.rocenante_native_path_arguments(selected)

    assert set(arguments) == {"0", "1", "2", "3"}
    assert all(len(values) == 6 for values in arguments.values())
    assert set(arguments["0"]) == {
        "1,0,rocep1s0f0,3,1",
        "1,1,roceP2p1s0f0,3,1",
        "2,0,rocep1s0f0,3,2",
        "2,1,roceP2p1s0f1,3,2",
        "3,0,rocep1s0f1,3,1",
        "3,1,roceP2p1s0f1,3,1",
    }
    assert set(arguments["2"]) == {
        "0,0,rocep1s0f1,3,2",
        "0,1,roceP2p1s0f0,3,2",
        "1,0,rocep1s0f1,3,1",
        "1,1,roceP2p1s0f1,3,1",
        "3,0,rocep1s0f0,3,1",
        "3,1,roceP2p1s0f0,3,1",
    }


def test_shared_diagonal_flow_label_matches_six_qp_runtime(tmp_path: Path) -> None:
    document = _document()
    document["flow_label_base"] = 16383
    document["shared_diagonal_flow_label"] = True
    complete = fabric.build_plan(fabric.load_topology(_write_topology(tmp_path, document)))
    selected = fabric.build_rocenante_plan(complete)

    assert {marker.flow_label for marker in selected.markers} == {16383}
    assert {marker.udp_source_port for marker in selected.markers} == {65535}
    assert len(
        {
            (marker.source_rank, marker.rdma_device, marker.udp_source_port)
            for marker in selected.markers
        }
    ) == 8
    assert len(selected.tc_rules) == 8


def test_adjacent_gateway_routes_and_tc_identity_overrides_are_exact(
    tmp_path: Path,
) -> None:
    document = _document()
    initial = fabric.build_rocenante_plan(
        fabric.build_plan(fabric.load_topology(_write_topology(tmp_path, document)))
    )
    document["endpoint_route_strategy"] = "adjacent_gateway"
    document["tc_rule_identity_overrides"] = {
        rule.path_name: {"preference": 50000 + index, "handle": 1000 + index}
        for index, rule in enumerate(initial.tc_rules)
    }
    selected = fabric.build_rocenante_plan(
        fabric.build_plan(fabric.load_topology(_write_topology(tmp_path, document)))
    )

    for index, rule in enumerate(selected.tc_rules):
        assert rule.preference == 50000 + index
        assert rule.handle == 1000 + index
    route = selected.routes[0]
    assert route.gateway_ipv4 is not None
    assert route.permanent_final_neighbor is False
    command = fabric.route_command(route, add=True)
    assert command[:5] == [
        "ip",
        "route",
        "add",
        f"{route.destination_ipv4}/32",
        "via",
    ]
    assert "scope" not in command
    with pytest.raises(fabric.FabricError, match="has no permanent"):
        fabric.neighbor_command(route, add=True)

    cleanup = fabric.build_cleanup_manifest(selected)
    phases = {phase["name"]: phase["commands"] for phase in cleanup["phases"]}
    assert phases["endpoint_neighbors"] == []
    assert len(phases["endpoint_routes"]) == 8


def test_tc_identity_override_rejects_unknown_path(tmp_path: Path) -> None:
    document = _document()
    document["tc_rule_identity_overrides"] = {
        "not-a-generated-diagonal": {"preference": 50000, "handle": 1000}
    }

    with pytest.raises(fabric.FabricError, match="unknown diagonal paths"):
        fabric.build_plan(fabric.load_topology(_write_topology(tmp_path, document)))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["ranks"][0]["ports"]["clockwise"][0].update(
                mac="REPLACE_SOURCE_MAC"
            ),
            "unresolved",
        ),
        (
            lambda value: value["ranks"][0]["ports"]["clockwise"][0].update(
                peer_rank=0
            ),
            "reciprocal cycle endpoint",
        ),
        (
            lambda value: value.update(flow_label_base=16380),
            "alias a RoCE UDP source port",
        ),
        (
            lambda value: value["ranks"][0]["ports"]["clockwise"][0].update(
                ipv4_cidr="0.0.0.0/32"
            ),
            "unicast endpoint",
        ),
    ],
)
def test_topology_rejects_unresolved_nonreciprocal_colliding_or_broad_paths(
    tmp_path: Path, mutate, message: str
) -> None:
    document = _document()
    mutate(document)

    with pytest.raises(fabric.FabricError, match=message):
        fabric.load_topology(_write_topology(tmp_path, document))
