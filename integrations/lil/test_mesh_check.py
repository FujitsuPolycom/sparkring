"""Mesh checks compare the proposed trial to installed state without real SSH."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from spark_transport.fabric.cx7_hairpin_diagonal.fabric import Port, Rank

spec = importlib.util.spec_from_file_location(
    "lil_mesh_check", Path(__file__).with_name("mesh_check.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(monkeypatch, rank=0):
    nodes = []
    for i in range(4):
        ports = {}
        for direction, other in (
            ("clockwise", (i + 1) % 4),
            ("counter_clockwise", (i - 1) % 4),
        ):
            for function in (0, 1):
                ports[direction, function] = Port(
                    direction=direction,
                    function=function,
                    netdev=f"eth{function}",
                    mac=f"02:00:00:00:0{i}:0{function}",
                    peer_rank=other,
                    peer_direction="counter_clockwise"
                    if direction == "clockwise"
                    else "clockwise",
                    peer_function=function,
                    rdma_device=f"dev-{direction}-{function}",
                    ipv4=f"198.18.{i}.{1 + function * 2 + (direction == 'clockwise')}",
                )
        nodes.append(
            Rank(
                rank=i,
                ssh_alias=f"spark{i}",
                management_netdev="mgmt0",
                ports=tuple(ports.values()),
            )
        )
    topology = SimpleNamespace(rank=lambda i: nodes[i])
    config = {
        "rank": rank,
        "container_image": "sha256:" + "a" * 64,
        "container_id": "b" * 64,
    }
    site = {"management_addresses": [f"192.0.2.{i + 1}" for i in range(4)]}
    fake = SimpleNamespace(
        load_config=lambda path: (config, site, topology, None, "identity")
    )
    monkeypatch.setitem(sys.modules, "managed_service", fake)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *a, **k: json.dumps(
            [{"Id": config["container_id"], "State": {"Running": False}}]
        ),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stdout=json.dumps({"ready": True, "identity": "identity"})
        ),
    )
    network = {
        "HOST_IP": site["management_addresses"][rank],
        "MASTER_ADDR": site["management_addresses"][0],
        "SOCKET_IFNAME": "mgmt0",
        "NCCL_IB_HCA": "dev-clockwise-0,dev-counter_clockwise-0",
        "NCCL_IB_GID_INDEX": "3",
    }
    directions = (
        ("clockwise", "counter_clockwise")
        if rank % 2 == 0
        else ("counter_clockwise", "clockwise")
    )
    for slot, d in enumerate(directions):
        for f in (0, 1):
            port = nodes[rank].port(d, f)
            peer = nodes[port.peer_rank].port(port.peer_direction, f)
            prefix = (
                "SPARK_TP4_" if f == 0 else "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_"
            )
            network.update(
                {
                    prefix + f"DEVICE{slot}": port.rdma_device,
                    prefix + f"PEER{slot}": peer.ipv4,
                    prefix + f"GID{slot}": "3",
                }
            )
    return {
        "rank": rank,
        "host": f"spark{rank}",
        "image": config["container_image"],
        "network": network,
    }


@pytest.mark.parametrize("rank", range(4))
def test_exact_installed_fabric_accepts_supervised_trial(monkeypatch, rank):
    expected = fixture(monkeypatch, rank)
    assert "supervised trial only" in module.verify_installed_mesh(expected)


@pytest.mark.parametrize(
    "field", ["SPARK_TP4_PEER0", "SPARK_TP4_DEVICE1", "NCCL_IB_GID_INDEX"]
)
def test_wrong_proposed_fabric_rejected(monkeypatch, field):
    expected = fixture(monkeypatch)
    expected["network"][field] = "wrong"
    with pytest.raises(ValueError, match="differ"):
        module.verify_installed_mesh(expected)


def test_running_managed_model_rejected(monkeypatch):
    expected = fixture(monkeypatch)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *a, **k: json.dumps([{"Id": "b" * 64, "State": {"Running": True}}]),
    )
    with pytest.raises(ValueError, match="Stop the managed model"):
        module.verify_installed_mesh(expected)


def test_mesh_check_is_bound_in_export():
    text = Path(__file__).with_name("export.py").read_text()
    assert "mesh_check.command(" in text
