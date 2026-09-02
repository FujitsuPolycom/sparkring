"""Offline tests for blank-Spark cluster bootstrap."""

from __future__ import annotations

import ipaddress
import json

import pytest

from scripts.sparkring_bootstrap import (
    BootstrapError,
    DEFAULT_INTERFACES,
    DEFAULT_RDMA_DEVICES,
    NodeFacts,
    _network_apply_script,
    _parse_probe,
    authorize_local_key,
    build_cluster_document,
    parse_target,
    render_rank_netplan,
)
from scripts.sparkring_cluster import validate_cluster


def facts(size: int) -> list[NodeFacts]:
    return [
        NodeFacts(
            rank=rank,
            target=f"user{rank}@192.0.2.{rank + 10}",
            hostname=f"spark-{rank}",
            management_interface="enP7s7",
            management_address=ipaddress.ip_address(f"192.0.2.{rank + 10}"),
            fabric_interfaces=DEFAULT_INTERFACES,
            rdma_devices=DEFAULT_RDMA_DEVICES,
        )
        for rank in range(size)
    ]


@pytest.mark.parametrize("size", [4, 6])
def test_generated_inventory_is_a_complete_cycle(size):
    document = build_cluster_document(
        facts(size), name=f"ring-{size}", fabric_supernet="198.18.0.0/21"
    )

    cluster = validate_cluster(document)

    assert len(cluster.ranks) == size
    assert len(cluster.topology.edges) == size
    for rank in cluster.ranks:
        assert len(rank.ring_ports) == 2
        assert len(rank.neighbour_ranks) == 2
        assert rank.management.interface not in {
            port.interface for port in rank.ring_ports
        }


def test_standard_port_orientation_connects_f0_to_next_ranks_f1():
    cluster = validate_cluster(
        build_cluster_document(
            facts(4), name="orientation", fabric_supernet="198.18.0.0/21"
        )
    )

    edge = cluster.topology.by_id("r0-r1")
    rank0 = next(port for port in cluster.rank(0).ring_ports if port.edge == edge.id)
    rank1 = next(port for port in cluster.rank(1).ring_ports if port.edge == edge.id)

    assert rank0.interface == "enp1s0f0np0"
    assert rank0.rdma_device == "rocep1s0f0"
    assert rank1.interface == "enp1s0f1np1"
    assert rank1.rdma_device == "rocep1s0f1"
    assert rank0.address == ipaddress.ip_address("198.18.0.10")
    assert rank1.address == ipaddress.ip_address("198.18.0.11")


def test_management_overlap_is_rejected():
    with pytest.raises(BootstrapError, match="overlaps a management address"):
        build_cluster_document(
            facts(4), name="overlap", fabric_supernet="192.0.0.0/21"
        )


def test_target_requires_username_and_literal_ipv4():
    assert parse_target("operator@192.0.2.10") == (
        "operator",
        ipaddress.ip_address("192.0.2.10"),
    )
    with pytest.raises(BootstrapError, match="username@IPv4"):
        parse_target("spark-r1")


def test_probe_identifies_management_and_cx7_mapping():
    rows = [
        {
            "ifname": "enP7s7",
            "addr_info": [{"family": "inet", "local": "192.0.2.10"}],
        },
        {"ifname": "enp1s0f0np0", "addr_info": []},
        {"ifname": "enp1s0f1np1", "addr_info": []},
    ]
    output = (
        "spark-zero\n__SPARKRING_ADDR__\n"
        + json.dumps(rows)
        + "\n__SPARKRING_RDMA__\n"
        + "rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)\n"
        + "rocep1s0f1 port 1 ==> enp1s0f1np1 (Down)\n"
    )

    observed = _parse_probe(
        0, "operator@192.0.2.10", ipaddress.ip_address("192.0.2.10"), output
    )

    assert observed.hostname == "spark-zero"
    assert observed.management_interface == "enP7s7"
    assert observed.fabric_interfaces == DEFAULT_INTERFACES


def test_generated_netplan_contains_only_fabric_interfaces():
    cluster = validate_cluster(
        build_cluster_document(
            facts(4), name="netplan", fabric_supernet="198.18.0.0/21"
        )
    )

    text = render_rank_netplan(cluster, 0)

    assert "enp1s0f0np0:" in text
    assert "enp1s0f1np1:" in text
    assert "enP7s7" not in text
    assert "198.18.0.10/24" in text
    assert "198.18.3.11/24" in text
    assert "mtu: 9000" in text

    apply_script = _network_apply_script(cluster, 0)
    assert "rollback()" in apply_script
    assert "if ! netplan apply" in apply_script
    assert "ip -4 -o addr show dev enP7s7" in apply_script


def test_network_apply_waits_for_the_recorded_management_address():
    cluster = validate_cluster(
        build_cluster_document(
            facts(4), name="management-wait", fabric_supernet="198.18.0.0/21"
        )
    )

    apply_script = _network_apply_script(cluster, 0)

    assert (
        "# Netplan may reload the management connection. Accept only the recorded "
        "IPv4 address and fail after five seconds.\n"
        "management_ready=0\n"
        "management_attempt=0\n"
        'while test "$management_attempt" -lt 50; do\n'
        "  if ip -4 -o addr show dev enP7s7 | "
        "grep -F -q -- ' 192.0.2.10/'; then\n"
        "    management_ready=1\n"
        "    break\n"
        "  fi\n"
        '  management_attempt=$((management_attempt + 1))\n'
        "  sleep 0.1\n"
        "done"
    ) in apply_script


def test_network_apply_rolls_back_when_management_wait_expires():
    cluster = validate_cluster(
        build_cluster_document(
            facts(4), name="management-timeout", fabric_supernet="198.18.0.0/21"
        )
    )

    apply_script = _network_apply_script(cluster, 0)

    assert (
        'if test "$management_ready" -ne 1; then\n'
        "  rollback\n"
        "  netplan generate\n"
        "  netplan apply\n"
        "  echo 'management invariant failed; fabric netplan rolled back' >&2\n"
        "  exit 91\n"
        "fi"
    ) in apply_script


def test_head_self_authorization_is_idempotent_and_preserves_existing_keys(tmp_path):
    public = tmp_path / "bootstrap.pub"
    key = "ssh-ed25519 QkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkI= bootstrap@spark"
    public.write_text(key + "\n", encoding="utf-8")
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    authorized = ssh_dir / "authorized_keys"
    authorized.write_text(
        "ssh-ed25519 Q0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0M= existing@spark\n",
        encoding="utf-8",
    )

    authorize_local_key(public, ssh_dir=ssh_dir)
    authorize_local_key(public, ssh_dir=ssh_dir)

    lines = authorized.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines.count(key) == 1
    assert any(ssh_dir.glob("authorized_keys.before-sparkring-*"))
