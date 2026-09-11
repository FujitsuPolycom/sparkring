"""CPU-only coverage of data-network plans and retained connection identities."""

from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path

import pytest

from scripts.deploy_network import NetworkPlanError, plan_network, verify_network


ROOT = Path(__file__).resolve().parents[1]


def network_fixture():
    fabric = json.loads(
        (ROOT / "runtime/glm53-spark-mtp3-mesh/fabric.example.json").read_text()
    )
    spec = {"owner": "test-mesh", "network": {"backend": "NetworkManager"}, "hosts": []}
    inventory = {}
    for host in fabric["ranks"]:
        rank = host["rank"]
        desired = {
            "host": host["ssh_alias"],
            "rank": rank,
            "management_netdev": host["management_netdev"],
            "backup_dir": f"/var/tmp/sparkring-network-backup-r{rank}",
            "data_interfaces": [],
        }
        observed = {
            "schema": "sparkring-deploy-host-inventory/v1",
            "rank": rank,
            "ssh_target": host["ssh_alias"],
            "management": {
                "address": f"192.0.2.{10 + rank}",
                "interface": host["management_netdev"],
                "route_to_controller": {
                    "dst": "192.0.2.100",
                    "dev": host["management_netdev"],
                },
            },
            "routes": [
                {
                    "dst": "default",
                    "dev": host["management_netdev"],
                    "gateway": "192.0.2.1",
                },
                {"dst": "192.0.2.0/24", "dev": host["management_netdev"]},
            ],
            "interfaces": [],
            "rdma": [],
            "docker": {"containers": []},
            "network": {
                "backend": "NetworkManager",
                "network_manager_active": True,
                "networkd_active": False,
                "rdma_resources": [],
                "connections": [],
            },
            "paths": {
                "/etc/NetworkManager/system-connections": {
                    "exists": True,
                    "type": "directory",
                },
                "/etc/netplan": {"exists": True, "type": "directory"},
            },
        }
        for index, (direction, functions) in enumerate(host["ports"].items()):
            for port in functions:
                role = ("cw" if direction == "clockwise" else "ccw") + (
                    "_primary" if port["function"] == 0 else "_secondary"
                )
                address = port["ipv4_cidr"].replace("/32", "/24")
                desired["data_interfaces"].append(
                    {
                        "role": role,
                        "netdev": port["netdev"],
                        "rdma_device": port["rdma_device"],
                        "address": address,
                    }
                )
                identifier = str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"fixture/{rank}/{role}")
                )
                nm = {
                    "available": True,
                    "managed": True,
                    "connection_uuid": identifier,
                    "connection_name": f"existing-{role}",
                    "owner": "system",
                    "autoconnect": True,
                    "ipv4_method": "manual",
                    "ipv4_addresses": [address],
                    "ethernet_mtu": 9000,
                    "ipv4_never_default": True,
                    "ipv4_ignore_auto_dns": True,
                    "ipv6_method": "link-local",
                    "ipv6_never_default": True,
                }
                observed["interfaces"].append(
                    {
                        "name": port["netdev"],
                        "mac": port["mac"],
                        "mtu": 9000,
                        "operstate": "UP",
                        "ipv4": [address],
                        "master": None,
                        "network_manager": nm,
                        "hw_tc_offload": True,
                        "hw_tc_offload_fixed": False,
                    }
                )
                observed["network"]["connections"].append(
                    {
                        "uuid": identifier,
                        "name": nm["connection_name"],
                        "interface": port["netdev"],
                        "owner": "system",
                    }
                )
                pci = f"000{port['function']}:01:00.{index}"
                observed["rdma"].append(
                    {
                        "device": port["rdma_device"],
                        "netdev": port["netdev"],
                        "pci_address": pci,
                        "driver": "mlx5_core",
                        "port": 1,
                        "gid_index": 3,
                        "gid": "::ffff:" + port["ipv4_cidr"].split("/")[0],
                        "gid_type": "RoCE v2",
                        "gid_netdev": port["netdev"],
                        "active_mtu": 4096,
                        "state": "ACTIVE",
                        "link_layer": "Ethernet",
                        "devlink": {
                            "available": True,
                            "device": "pci/" + pci,
                            "eswitch_mode": "legacy",
                            "eswitch_inline_mode": "none",
                            "eswitch_encap_mode": "basic",
                            "parameters": {
                                "hairpin_num_queues": {
                                    "value": 4,
                                    "cmode": "driverinit",
                                    "allowed_values": [],
                                },
                                "hairpin_queue_size": {
                                    "value": 1024,
                                    "cmode": "driverinit",
                                    "allowed_values": [],
                                },
                                "flow_steering_mode": {
                                    "value": "hmfs",
                                    "cmode": "runtime",
                                    "allowed_values": [],
                                },
                            },
                        },
                    }
                )
        spec["hosts"].append(desired)
        inventory[host["ssh_alias"]] = observed
    return spec, inventory


def unconfigured(spec, inventory, rank=0, port_index=0):
    observed = inventory[f"spark-r{rank}"]
    interface = observed["interfaces"][port_index]
    previous = interface["network_manager"]["connection_uuid"]
    interface["ipv4"] = []
    interface["mtu"] = 1500
    interface["network_manager"].update(connection_uuid=None, connection_name=None)
    observed["network"]["connections"] = [
        c for c in observed["network"]["connections"] if c["uuid"] != previous
    ]
    return interface


def test_configured_hosts_are_noop_without_adopting_foreign_connections():
    spec, inventory = network_fixture()
    before = copy.deepcopy((spec, inventory))
    plan = plan_network(spec, inventory)
    assert all(host["action"] == "none" for host in plan["hosts"])
    assert all(
        not host["apply"] and not host["driver_steps"] and not host["rollback"]
        for host in plan["hosts"]
    )
    assert all(
        port["created_connection_uuid"] is None
        for host in plan["hosts"]
        for port in host["interfaces"]
    )
    assert verify_network(spec, inventory)["data_functions"] == 16
    assert (spec, inventory) == before


def test_empty_interface_gets_persistent_named_profile_and_scoped_rollback():
    spec, inventory = network_fixture()
    unconfigured(spec, inventory)
    host = plan_network(spec, inventory)["hosts"][0]
    assert host["action"] == "configure"
    assert host["interfaces"][0]["endpoint_locator"] == "198.18.1.1/32"
    create, activate = host["apply"]
    assert create["argv"][:6] == ["sudo", "-n", "nmcli", "connection", "add", "type"]
    assert create["argv"][create["argv"].index("ipv4.addresses") + 1] == "198.18.1.1/24"
    assert "ipv4.never-default" in create["argv"] and "link-local" in create["argv"]
    assert create["requires_stopped_models"] and create["requires_no_rdma_users"]
    assert activate["argv"][-1] == host["interfaces"][0]["created_connection_uuid"]
    assert host["rollback"][0]["only_if_owned_uuid"] == activate["argv"][-1]
    assert host["backup"][0]["argv"] == [
        "test",
        "!",
        "-e",
        spec["hosts"][0]["backup_dir"],
    ]
    assert all("enP7s7" not in c["argv"] for c in host["apply"])


def test_owned_profile_is_noop_after_activation():
    spec, inventory = network_fixture()
    interface = unconfigured(spec, inventory)
    prepared = plan_network(spec, inventory)["hosts"][0]["interfaces"][0]
    identifier = prepared["created_connection_uuid"]
    spec["owned_connection_uuids"] = [identifier]
    interface["ipv4"] = [prepared["address"]]
    interface["mtu"] = 9000
    interface["network_manager"].update(
        connection_uuid=identifier, connection_name=prepared["created_connection_name"]
    )
    inventory["spark-r0"]["network"]["connections"].append(
        {"uuid": identifier, "name": prepared["created_connection_name"]}
    )
    assert plan_network(spec, inventory)["hosts"][0]["action"] == "none"


def test_known_replacement_retains_and_reactivates_exact_previous_uuid():
    spec, inventory = network_fixture()
    interface = inventory["spark-r0"]["interfaces"][0]
    previous = interface["network_manager"]["connection_uuid"]
    interface["mtu"] = 1500
    with pytest.raises(NetworkPlanError, match="unowned connection"):
        plan_network(spec, inventory)
    spec["hosts"][0]["data_interfaces"][0]["replace_connection_uuid"] = previous
    host = plan_network(spec, inventory)["hosts"][0]
    assert host["interfaces"][0]["previous_connection_uuid"] == previous
    assert host["rollback"][-1]["argv"] == [
        "sudo",
        "-n",
        "nmcli",
        "connection",
        "up",
        "uuid",
        previous,
    ]
    assert all("modify" not in c["argv"] for c in host["apply"])


def test_wrong_replacement_uuid_cannot_displace_connection():
    spec, inventory = network_fixture()
    inventory["spark-r0"]["interfaces"][0]["mtu"] = 1500
    spec["hosts"][0]["data_interfaces"][0]["replace_connection_uuid"] = str(
        uuid.uuid4()
    )
    with pytest.raises(NetworkPlanError, match="replacement UUID"):
        plan_network(spec, inventory)


def test_profile_name_collision_refuses_even_when_uuid_differs():
    spec, inventory = network_fixture()
    unconfigured(spec, inventory)
    inventory["spark-r0"]["network"]["connections"].append(
        {"name": "sparkring-test-mesh-r0-cw-primary", "uuid": str(uuid.uuid4())}
    )
    with pytest.raises(NetworkPlanError, match="already belongs"):
        plan_network(spec, inventory)


def test_driver_reload_is_separate_and_requires_rediscovery():
    spec, inventory = network_fixture()
    params = inventory["spark-r0"]["rdma"][0]["devlink"]["parameters"]
    params["hairpin_num_queues"]["value"] = 0
    params["hairpin_queue_size"]["value"] = 0
    host = plan_network(spec, inventory)["hosts"][0]
    assert not host["apply"]
    assert len(host["driver_steps"]) == 3
    reload = host["driver_steps"][-1]
    assert reload["argv"] == [
        "sudo",
        "-n",
        "devlink",
        "dev",
        "reload",
        "pci/0000:01:00.0",
        "action",
        "driver_reinit",
    ]
    assert reload["risk"] == "driver-reload" and reload["stop_after"]
    assert (
        reload["requires_independent_management"] and reload["requires_no_rdma_users"]
    )
    assert [cmd["argv"][-3] for cmd in host["rollback"][:2]] == ["0", "0"]
    assert all("switchdev" not in c["argv"] for c in host["driver_steps"])


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("flow_steering_mode", "dmfs", "hmfs"),
        ("flow_steering_mode", None, "hmfs"),
        ("hairpin_num_queues", None, "driverinit"),
        ("hairpin_queue_size", "unknown", "driverinit"),
    ],
)
def test_unproven_driver_capability_refuses(field, value, match):
    spec, inventory = network_fixture()
    inventory["spark-r0"]["rdma"][0]["devlink"]["parameters"][field]["value"] = value
    with pytest.raises(NetworkPlanError, match=match):
        plan_network(spec, inventory)


def test_incompatible_driver_modes_are_not_automatically_replaced():
    spec, inventory = network_fixture()
    inventory["spark-r0"]["rdma"][0]["devlink"]["eswitch_mode"] = "switchdev"
    with pytest.raises(NetworkPlanError, match="legacy/none/basic"):
        plan_network(spec, inventory)


def test_disabled_offload_requires_proven_toggle_support():
    spec, inventory = network_fixture()
    interface = inventory["spark-r0"]["interfaces"][0]
    interface["hw_tc_offload"] = False
    interface["hw_tc_offload_fixed"] = True
    with pytest.raises(NetworkPlanError, match="cannot be enabled"):
        plan_network(spec, inventory)
    interface["hw_tc_offload_fixed"] = False
    host = plan_network(spec, inventory)["hosts"][0]
    assert host["driver_steps"][0]["argv"][-2:] == ["hw-tc-offload", "on"]
    assert host["rollback"][0]["argv"][-2:] == ["hw-tc-offload", "off"]


@pytest.mark.parametrize(
    "route",
    [
        {"dst": "198.18.0.0/15", "dev": "tun0"},
        {"dst": "198.18.1.100/32", "dev": "enP7s7"},
        {"dst": "default", "dev": "enp1s0f0np0"},
    ],
)
def test_management_and_vpn_route_overlap_refuses(route):
    spec, inventory = network_fixture()
    inventory["spark-r0"]["routes"].append(route)
    with pytest.raises(NetworkPlanError, match="overlap|default route"):
        plan_network(spec, inventory)


def test_wrong_controller_return_interface_refuses():
    spec, inventory = network_fixture()
    inventory["spark-r0"]["management"]["route_to_controller"]["dev"] = "enp1s0f0np0"
    with pytest.raises(NetworkPlanError, match="independent management"):
        plan_network(spec, inventory)


def test_unknown_network_backend_refuses():
    spec, inventory = network_fixture()
    inventory["spark-r2"]["network"]["backend"] = "systemd-networkd"
    with pytest.raises(NetworkPlanError, match="NetworkManager"):
        plan_network(spec, inventory)


@pytest.mark.parametrize(
    "value", ["198.18.1.1/32", "198.18.1.0/24", "198.18.1.255/24", "::1/128"]
)
def test_address_geometry_is_explicit(value):
    spec, inventory = network_fixture()
    spec["hosts"][0]["data_interfaces"][0]["address"] = value
    with pytest.raises(NetworkPlanError, match="/24|unicast"):
        plan_network(spec, inventory)


def test_cable_subnet_mismatch_refuses():
    spec, inventory = network_fixture()
    spec["hosts"][0]["data_interfaces"][0]["address"] = "198.18.9.1/24"
    with pytest.raises(NetworkPlanError, match="endpoints must share"):
        plan_network(spec, inventory)


def test_network_changes_wait_for_running_models_and_rdma_users():
    spec, inventory = network_fixture()
    unconfigured(spec, inventory)
    inventory["spark-r0"]["docker"]["containers"] = [
        {"name": "model", "state": "running"}
    ]
    inventory["spark-r0"]["network"]["rdma_resources"] = [{"type": "qp", "id": 1}]
    host = plan_network(spec, inventory)["hosts"][0]
    assert not host["apply_permitted"]
    assert len(host["blocked_by"]) == 2


def test_backup_archive_includes_only_proven_existing_sources():
    spec, inventory = network_fixture()
    unconfigured(spec, inventory)
    inventory["spark-r0"]["paths"]["/etc/netplan"] = {"exists": False, "type": None}
    commands = plan_network(spec, inventory)["hosts"][0]["backup"]
    archive = next(c["argv"] for c in commands if c["id"] == "archive-network-config")
    assert archive[-1] == "etc/NetworkManager/system-connections"
    assert "etc/netplan" not in archive
    del inventory["spark-r0"]["paths"]["/etc/netplan"]
    with pytest.raises(NetworkPlanError, match="backup source"):
        plan_network(spec, inventory)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("gid", "::ffff:198.18.1.99", "GID index 3"),
        ("gid_type", "IB/RoCE v1", "RoCE v2"),
        ("gid_index", 1, "GID index 3"),
        ("active_mtu", 2048, "MTU"),
        ("state", "DOWN", "not active"),
    ],
)
def test_fresh_verification_checks_rdma_state_not_only_addresses(field, value, match):
    spec, inventory = network_fixture()
    inventory["spark-r0"]["rdma"][0][field] = value
    with pytest.raises(NetworkPlanError, match=match):
        verify_network(spec, inventory)


def test_verification_does_not_call_an_unapplied_plan_ready():
    spec, inventory = network_fixture()
    unconfigured(spec, inventory)
    with pytest.raises(NetworkPlanError, match="do not match"):
        verify_network(spec, inventory)


def test_missing_interface_driver_identity_refuses():
    spec, inventory = network_fixture()
    inventory["spark-r0"]["rdma"][0]["driver"] = None
    with pytest.raises(NetworkPlanError, match="verified mlx5"):
        plan_network(spec, inventory)


@pytest.mark.parametrize('field,value', [('ipv4_addresses', ['198.18.99.1/24']),
                                        ('ethernet_mtu', 1500), ('ipv4_addresses', None),
                                        ('ethernet_mtu', None)])
def test_live_settings_do_not_prove_saved_connection_settings(field, value):
    spec, inventory = network_fixture()
    inventory['spark-r0']['interfaces'][0]['network_manager'][field] = value
    with pytest.raises(NetworkPlanError):
        verify_network(spec, inventory)


@pytest.mark.parametrize('binding', ['different-interface', None])
def test_network_readiness_requires_gid_interface_binding(binding):
    spec, inventory = network_fixture()
    inventory['spark-r0']['rdma'][0]['gid_netdev'] = binding
    with pytest.raises(NetworkPlanError, match='GID.*interface'):
        verify_network(spec, inventory)


def test_synthetic_command_text_cannot_enter_interface_identity():
    spec, inventory = network_fixture()
    spec["hosts"][0]["data_interfaces"][0]["netdev"] = "eth0; reboot"
    with pytest.raises(NetworkPlanError, match="invalid"):
        plan_network(spec, inventory)
