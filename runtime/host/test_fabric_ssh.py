"""Fabric routing follows enrolled topology and rejects wrong-node paths."""
import copy
import ipaddress
from types import SimpleNamespace

import pytest

from runtime.host import fabric_ssh, topology
from runtime.host.test_appliance import nodes


def cluster(size):
    found = nodes(size)
    return {"name": "test", "plan": topology.build_spec(found, found[0]["node_id"])}


@pytest.mark.parametrize("size", [2, 4])
def test_bulk_paths_are_derived_from_actual_fabric_addresses(size):
    value = cluster(size)
    original = fabric_ssh.routes(value)
    changed = copy.deepcopy(value)
    for host in changed["plan"]["spec"]["hosts"]:
        for port in host["data_interfaces"]:
            address = ipaddress.IPv4Interface(port["address"])
            shifted = ipaddress.IPv4Address(int(address.ip) + 256 * 24)
            port["address"] = str(shifted) + "/24"
    result = fabric_ssh.routes(changed)
    assert result[0]["address"] is None
    for before, after in zip(original[1:], result[1:], strict=True):
        assert int(ipaddress.ip_address(after["address"])) - int(ipaddress.ip_address(before["address"])) == 256 * 24
    if size == 4:
        assert [row["parent"] for row in result] == [None, 0, 1, 0]


def test_no_management_address_is_used_as_a_bulk_destination(tmp_path):
    value = cluster(4)
    def run(argv, **kwargs):
        assert argv[:2] == ["ssh", "-G"]
        return SimpleNamespace(stdout="hostname 192.0.2.8\nuser root\nport 22\nidentityfile /root/key\nuserknownhostsfile /root/known\n")
    transport = fabric_ssh.Transport(value, tmp_path / "ssh", root=tmp_path, run=run)
    text = transport.config.read_text()
    assert 'HostName "192.0.2.8"' not in text
    assert 'HostKeyAlias "192.0.2.8"' in text
    assert 'ProxyJump "sparkring-r1"' in text
    assert "ForwardAgent" not in text
    assert transport.command(0, ["docker", "save", "fixture"]) == ["docker", "save", "fixture"]


def test_fabric_peer_must_match_the_enrolled_node_id(tmp_path):
    value = cluster(2)
    def run(argv, **kwargs):
        if argv[:2] == ["ssh", "-G"]:
            return SimpleNamespace(stdout="hostname 192.0.2.8\nuser root\nport 22\n")
        if argv[:4] == ["ip", "-j", "route", "get"]:
            return SimpleNamespace(stdout='[{"dev":"enp1s0f0np0"}]')
        return SimpleNamespace(stdout="wrong-node\n")
    transport = fabric_ssh.Transport(value, tmp_path / "ssh", root=tmp_path, run=run)
    with pytest.raises(ValueError, match="node identity"):
        transport.verify()
