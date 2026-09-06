"""Host-discovery tests use recorded Linux output; they never contact a host."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import subprocess

import pytest

from scripts.deploy_inventory import (
    MANAGED_PATHS,
    SCHEMA,
    _collect_local,
    _request,
    collect_command,
    summarise_inventory,
    validate_inventory,
)


def host_inventory():
    """One ARM64 host with independent management and four Socket Direct functions."""
    interfaces = [
        {
            "name": "eno1",
            "mac": "02:00:00:00:10:01",
            "mtu": 1500,
            "operstate": "UP",
            "master": None,
            "ipv4": ["192.0.2.10/24"],
            "network_manager": {
                "available": True,
                "managed": True,
                "connection_uuid": "40ae1d90-b77a-4cba-b04a-1ce84d4f64a0",
                "connection_name": "Management",
                "owner": "system",
            },
            "hw_tc_offload": False,
            "hw_tc_offload_fixed": False,
        }
    ]
    rdma = []
    for index, netdev in enumerate(
        ("enp1s0f0np0", "enP2p1s0f0np0", "enp1s0f1np1", "enP2p1s0f1np1")
    ):
        address = f"198.18.{index + 1}.1"
        interfaces.append(
            {
                "name": netdev,
                "mac": f"02:00:00:00:20:{index:02x}",
                "mtu": 9000,
                "operstate": "UP",
                "master": None,
                "ipv4": [address + "/32"],
                "hw_tc_offload": True,
                "hw_tc_offload_fixed": False,
                "network_manager": {
                    "available": True,
                    "managed": True,
                    "connection_uuid": None,
                    "connection_name": None,
                    "owner": None,
                },
            }
        )
        rdma.append(
            {
                "device": f"mlx5_{index}",
                "port": 1,
                "netdev": netdev,
                "pci_address": f"000{index // 2}:01:00.{index % 2}",
                "driver": "mlx5_core",
                "gid_index": 3,
                "gid": "::ffff:" + address,
                "gid_netdev": netdev,
                "gid_type": "RoCE v2",
                "active_mtu": 4096,
                "state": "PORT_ACTIVE",
                "link_state": "Up",
                "link_layer": "Ethernet",
                "devlink": {
                    "available": True,
                    "parameters": {
                        "hairpin_num_queues": {
                            "value": 4,
                            "cmode": "driverinit",
                            "allowed_values": None,
                        },
                        "hairpin_queue_size": {
                            "value": 1024,
                            "cmode": "driverinit",
                            "allowed_values": None,
                        },
                        "flow_steering_mode": {
                            "value": "hmfs",
                            "cmode": "runtime",
                            "allowed_values": None,
                        },
                    },
                    "eswitch_mode": "legacy",
                    "eswitch_inline_mode": "none",
                    "eswitch_encap_mode": "basic",
                    "error": None,
                },
                "error": None,
            }
        )
    return {
        "schema": SCHEMA,
        "rank": 0,
        "ssh_target": "operator@node-a",
        "collected_at": "2026-09-06T00:00:00+00:00",
        "platform": {"system": "Linux", "architecture": "aarch64"},
        "gpu": {
            "available": True,
            "devices": [
                {
                    "name": "NVIDIA GB10",
                    "driver_version": "580.00",
                    "uuid": "GPU-example",
                }
            ],
            "error": None,
        },
        "docker": {
            "available": True,
            "server_version": "28.0.0",
            "runtimes": ["nvidia", "runc"],
            "containers": [],
            "error": None,
        },
        "toolkit": {
            "available": True,
            "version": "NVIDIA Container Toolkit CLI version 1.17.8",
            "error": None,
        },
        "privilege": {"root": False, "sudo_noninteractive": True, "error": None},
        "tools": {
            "python3": "/usr/bin/python3",
            "nmcli": "/usr/bin/nmcli",
            "devlink": "/usr/sbin/devlink",
        },
        "management": {
            "address": "192.0.2.10",
            "interface": "eno1",
            "controller_address": "192.0.2.99",
            "route_to_controller": {
                "dst": "192.0.2.99",
                "dev": "eno1",
                "prefsrc": "192.0.2.10",
            },
            "error": None,
        },
        "interfaces": interfaces,
        "routes": [{"dst": "default", "gateway": "192.0.2.1", "dev": "eno1"}],
        "rdma": rdma,
        "network": {
            "backend": "NetworkManager",
            "network_manager_active": True,
            "networkd_active": False,
            "connections": [],
            "rdma_resources": [],
            "errors": [],
        },
        "paths": {
            "/": {
                "exists": True,
                "type": "directory",
                "nonempty": True,
                "mode": "0755",
                "owner_uid": 0,
                "free_bytes": 500_000_000_000,
                "error": None,
            }
        },
        "managed": {"installation_present": False, "units": {}},
    }


def test_prepared_inventory_is_copied_and_summarised():
    original = host_inventory()
    observed = validate_inventory(original, require_ready=True)
    observed["rdma"].clear()
    assert len(original["rdma"]) == 4
    text = summarise_inventory(original)
    assert "NVIDIA GB10" in text
    assert "RDMA: 4 observed" in text
    assert "Running containers: 0" in text
    assert "not tested" in text
    assert "blocker" not in text


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda doc: doc["platform"].update(architecture="x86_64"), "Linux ARM64"),
        (
            lambda doc: doc["privilege"].update(root=False, sudo_noninteractive=None),
            "noninteractive sudo",
        ),
        (lambda doc: doc["rdma"].pop(), "four distinct RDMA"),
        (lambda doc: doc["rdma"][0].update(netdev="eno1"), "management traffic"),
        (
            lambda doc: doc["management"].update(route_to_controller=None),
            "return route",
        ),
        (
            lambda doc: doc["interfaces"][1]["ipv4"].append("192.0.2.10/24"),
            "more than one",
        ),
    ],
)
def test_unsafe_host_observation_cannot_authorize_preparation(change, message):
    document = host_inventory()
    change(document)
    validate_inventory(document)
    with pytest.raises(ValueError, match=message):
        validate_inventory(document, require_ready=True)


def test_unavailable_optional_facts_are_not_reported_as_absent_or_healthy():
    document = host_inventory()
    document["gpu"].update(
        available=None, devices=None, error="nvidia-smi query failed"
    )
    document["tools"]["devlink"] = None
    document["docker"].update(
        available=None, containers=None, error="permission denied"
    )
    document["managed"]["installation_present"] = None
    text = summarise_inventory(document)
    assert "GPU: unavailable" in text
    assert "Running containers: unavailable" in text
    assert "Managed mesh files: unavailable" in text
    assert "Tools absent from PATH: devlink" in text


def test_running_work_and_nonempty_directories_remain_visible():
    document = host_inventory()
    document["docker"]["containers"] = [{"name": "user-model", "state": "running"}]
    document["paths"]["/srv/models"] = {
        "exists": True,
        "type": "directory",
        "nonempty": True,
    }
    document["managed"]["installation_present"] = True
    text = summarise_inventory(document)
    assert "Running containers: 1" in text
    assert "Nonempty paths: /srv/models" in text
    assert "Managed mesh files: present" in text


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("rank", True, "rank"),
        ("ssh_target", "-oProxyCommand=bad", "ssh_target"),
        ("management_address", "127.0.0.1", "network host"),
        ("paths", ["/srv/../etc"], "parent traversal"),
        ("paths", ["/srv/models\n"], "absolute"),
    ],
)
def test_probe_arguments_are_literal_and_validated(key, value, message):
    args = {"rank": 0, "ssh_target": "node-a", "management_address": "192.0.2.10"}
    args[key] = value
    with pytest.raises(ValueError, match=message):
        collect_command(**args)


def test_probe_is_a_complete_self_contained_python_command(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("command generation executed a process"),
    )
    argv = collect_command(
        rank=2,
        ssh_target="operator@node-c",
        management_address="192.0.2.12",
        paths=("/srv/models",),
    )
    assert argv[:2] == ["python3", "-c"]
    ast.parse(argv[2])
    assert "operator@node-c" in argv[2]
    assert "/etc/NetworkManager/system-connections" in argv[2]
    assert "docker run" not in argv[2]
    assert "systemctl start" not in argv[2]
    assert "health.key" not in argv[2]


class LinuxFixture:
    """Recorded command shapes and small sysfs files for an ARM64 host."""

    def __init__(self, root: Path):
        self.root = root
        self.calls = []
        self.inventory = host_inventory()
        for item in self.inventory["rdma"]:
            device = root / "sys/class/infiniband" / item["device"]
            (device / "ports/1/gids").mkdir(parents=True)
            (device / "ports/1/gid_attrs/ndevs").mkdir(parents=True)
            (device / "ports/1/gid_attrs/types").mkdir(parents=True)
            (device / "ports/1/gids/3").write_text(item["gid"])
            (device / "ports/1/gid_attrs/ndevs/3").write_text(item["netdev"])
            (device / "ports/1/gid_attrs/types/3").write_text("RoCE v2")
            (device / "device").mkdir()
            (device / "device/uevent").write_text(
                "DRIVER=mlx5_core\nPCI_SLOT_NAME=" + item["pci_address"]
            )
        (root / "etc/NetworkManager/system-connections").mkdir(parents=True)
        (root / "etc/netplan").mkdir(parents=True)
        (root / "srv/models").mkdir(parents=True)
        (root / "srv/models/retained-model-marker").write_bytes(b"retained")

    def run(self, argv, **kwargs):
        self.calls.append(argv)
        command = tuple(argv)
        text, code = "", 0
        if command == ("sudo", "-n", "id", "-u"):
            text = "0"
        elif command[0] == "nvidia-smi":
            text = "NVIDIA GB10, 580.00, GPU-example"
        elif command[0] == "nvidia-ctk":
            text = "NVIDIA Container Toolkit CLI version 1.17.8"
        elif command[:2] == ("docker", "info"):
            text = json.dumps(
                {"ServerVersion": "28.0.0", "Runtimes": {"nvidia": {}, "runc": {}}}
            )
        elif command[:2] == ("docker", "ps"):
            text = json.dumps(
                {
                    "ID": "a" * 64,
                    "Names": "existing-model",
                    "Image": "sha256:" + "b" * 64,
                    "State": "running",
                    "Status": "Up 1 hour",
                }
            )
        elif command[:4] == ("ip", "-j", "-4", "address"):
            rows = [
                {
                    "ifname": item["name"],
                    "address": item["mac"],
                    "mtu": item["mtu"],
                    "operstate": item["operstate"],
                    "addr_info": [
                        {
                            "family": "inet",
                            "local": value.split("/")[0],
                            "prefixlen": int(value.split("/")[1]),
                        }
                        for value in item["ipv4"]
                    ],
                }
                for item in self.inventory["interfaces"]
            ]
            text = json.dumps(rows)
        elif command[:5] == ("ip", "-j", "-4", "route", "show"):
            text = json.dumps(self.inventory["routes"])
        elif command[:5] == ("ip", "-j", "-4", "route", "get"):
            text = json.dumps([self.inventory["management"]["route_to_controller"]])
        elif command[:2] == ("systemctl", "show"):
            if command[2] == "NetworkManager.service":
                text = "LoadState=loaded\nActiveState=active\nSubState=running"
            else:
                text = "LoadState=not-found\nActiveState=inactive\nSubState=dead"
        elif command == ("nmcli", "-g", "UUID", "connection", "show"):
            text = "40ae1d90-b77a-4cba-b04a-1ce84d4f64a0"
        elif command[:3] == ("nmcli", "--escape", "no"):
            field = command[4]
            if command[5] == "device":
                text = "yes\n" + (
                    "40ae1d90-b77a-4cba-b04a-1ce84d4f64a0"
                    if command[-1] == "eno1"
                    else "--"
                )
            else:
                text = {
                    "connection.id": "Management",
                    "connection.interface-name": "eno1",
                    "connection.permissions": "",
                    "ipv4.method": "auto",
                    "ipv6.method": "auto",
                    "connection.autoconnect": "yes",
                }.get(field, "no")
        elif command[:2] == ("ethtool", "-k"):
            text = "Features:\nhw-tc-offload: " + (
                "off" if command[-1] == "eno1" else "on"
            )
        elif command == ("ibdev2netdev",):
            text = "\n".join(
                f"{item['device']} port 1 ==> {item['netdev']} (Up)"
                for item in self.inventory["rdma"]
            )
        elif command[0] == "ibv_devinfo":
            text = "\tactive_mtu: 4096 (5)\n\tlink_layer: Ethernet\n\tstate: PORT_ACTIVE (4)"
        elif command[:5] == ("devlink", "-j", "dev", "param", "show"):
            name = command[-1]
            value = {
                "hairpin_num_queues": 4,
                "hairpin_queue_size": 1024,
                "flow_steering_mode": "hmfs",
            }[name]
            cmode = "runtime" if name == "flow_steering_mode" else "driverinit"
            text = json.dumps(
                {
                    "param": {
                        command[5]: [
                            {
                                "name": name,
                                "type": "string" if isinstance(value, str) else "u32",
                                "values": [{"cmode": cmode, "value": value}],
                            }
                        ]
                    }
                }
            )
        elif command[:5] == ("devlink", "-j", "dev", "eswitch", "show"):
            text = json.dumps(
                {
                    "dev": {
                        command[5]: {
                            "mode": "legacy",
                            "inline-mode": "none",
                            "encap-mode": "basic",
                        }
                    }
                }
            )
        elif command == ("rdma", "-j", "resource", "show", "qp"):
            text = '[{"ifname":"mlx5_0","lqpn":21,"type":"RC","state":"RTS","pid":123}]'
        else:
            raise AssertionError(f"Unexpected probe command: {command}")
        return subprocess.CompletedProcess(argv, code, text, "")

    def collect(self):
        options = _request(
            0, "operator@node-a", "192.0.2.10", ("/srv/models",), "192.0.2.99"
        )
        return _collect_local(
            options,
            run=self.run,
            root=self.root,
            tool_path=lambda command: "/usr/bin/" + command,
            platform_info=("Linux", "aarch64"),
            euid=1000,
        )


def test_collector_parses_linux_fixtures_without_remote_access(tmp_path):
    fixture = LinuxFixture(tmp_path)
    result = fixture.collect()
    validate_inventory(result, require_ready=True)
    assert result["gpu"]["devices"][0]["name"] == "NVIDIA GB10"
    assert result["docker"]["containers"][0]["name"] == "existing-model"
    assert result["network"]["backend"] == "NetworkManager"
    assert result["network"]["connections"][0]["owner"] == "system"
    assert result["interfaces"][0]["network_manager"]["ipv4_method"] == "auto"
    assert result["interfaces"][1]["hw_tc_offload"] is True
    assert result["rdma"][0]["active_mtu"] == 4096
    assert result["rdma"][0]["gid"] == "::ffff:198.18.1.1"
    assert result["rdma"][0]["driver"] == "mlx5_core"
    assert result["rdma"][0]["state"] == "PORT_ACTIVE"
    assert result["rdma"][0]["devlink"]["parameters"]["hairpin_num_queues"] == {
        "value": 4,
        "cmode": "driverinit",
        "allowed_values": None,
    }
    assert result["rdma"][0]["devlink"]["eswitch_encap_mode"] == "basic"
    assert result["network"]["rdma_resources"][0]["state"] == "RTS"
    assert result["paths"]["/srv/models"]["nonempty"] is True
    assert result["paths"]["/"]["free_bytes"] > 0
    assert result["managed"]["installation_present"] is False
    assert set(MANAGED_PATHS) <= set(result["paths"])
    assert all(argv[0] not in ("ssh", "scp") for argv in fixture.calls)


def test_collector_handles_missing_tools_and_permission_failures(tmp_path):
    options = _request(0, "node-a", "192.0.2.10", (), "192.0.2.99")
    document = _collect_local(
        options,
        root=tmp_path,
        tool_path=lambda command: None,
        run=lambda *args, **kwargs: pytest.fail("missing tool was executed"),
        platform_info=("Linux", "aarch64"),
        euid=1000,
    )
    validate_inventory(document)
    assert document["gpu"]["available"] is None
    assert document["privilege"]["sudo_noninteractive"] is None
    assert document["network"]["rdma_resources"] is None
    assert document["network"]["backend"] == "unknown"
    with pytest.raises(ValueError, match="sudo"):
        validate_inventory(document, require_ready=True)


def test_validation_rejects_forged_boolean_and_unobserved_mapping():
    document = host_inventory()
    document["privilege"]["root"] = "false"
    with pytest.raises(ValueError, match="true, false, or null"):
        validate_inventory(document)
    document = host_inventory()
    document["rdma"][0]["netdev"] = "unobserved"
    with pytest.raises(ValueError, match="unobserved"):
        validate_inventory(document)


def test_validation_rejects_duplicate_functions_and_bad_gid():
    document = host_inventory()
    document["rdma"][1] = copy.deepcopy(document["rdma"][0])
    with pytest.raises(ValueError, match="distinct"):
        validate_inventory(document)
    document = host_inventory()
    document["rdma"][0]["gid"] = "not-an-ipv6-address"
    with pytest.raises(ValueError, match="GID"):
        validate_inventory(document)
