"""Management routing and optional settings, without host access."""
import base64
import copy
import json
import shlex

import pytest

from runtime.host import bootstrap, control, settings


def fixture(size=4):
    nodes = [{"id": str(i), "hostname": f"spark{i}", "public_key": base64.b64encode(bytes([i + 1]) * 32).decode(),
              "routes": [], "uplink": "eth0"} for i in range(size)]
    edges = []
    for rank in range(1 if size == 2 else size):
        peer = (rank + 1) % size
        edges.append([{"id": str(rank), "netdev": "port0", "address": f"fe80::{rank + 1}:1", "mac": f"02:00:00:00:{rank:02x}:01"},
                      {"id": str(peer), "netdev": "port0" if size == 2 else "port1", "address": f"fe80::{peer + 1}:2", "mac": f"02:00:00:00:{peer:02x}:02"}])
    return nodes, edges


@pytest.mark.parametrize("size", [2, 4])
def test_tree_routing_has_one_authenticated_path_to_head_and_every_peer(size):
    nodes, edges = fixture(size)
    plans = control.plan(nodes, edges, "0")
    by_id = {p["id"]: p for p in plans}
    for source in plans:
        for destination in plans:
            if source == destination:
                continue
            current, seen = source, set()
            while current != destination:
                assert current["id"] not in seen
                seen.add(current["id"])
                exact = [p for p in current["peers"] if destination["address"] + "/32" in p["allowed_ips"]]
                hop = exact[0] if exact else next(p for p in current["peers"] if "0.0.0.0/0" in p["allowed_ips"])
                current = by_id[hop["id"]]
            assert len(seen) <= 3
        rendered = control.render(source, base64.b64encode(b"a" * 32).decode())
        assert "MTU = 1420" in rendered
        assert "PostUp" not in rendered  # no arbitrary shell instructions
    assert sum(len(p["peers"]) for p in plans) == 2 * (size - 1)


def test_management_prefix_conflict_and_disconnected_graph_are_rejected():
    nodes, edges = fixture()
    nodes[0]["routes"] = [{"dst": "10.0.0.0/8", "dev": "existing"}]
    with pytest.raises(ValueError, match="overlaps"):
        control.plan(nodes, edges, "0")
    nodes[0]["routes"] = []
    with pytest.raises(ValueError, match="pair or"):
        control.plan(nodes, edges[:-1], "0")


def test_no_uplink_sharing_never_adds_default_route_or_nat():
    nodes, edges = fixture()
    for plan in control.plan(nodes, edges, "0", share_uplink=False):
        assert all("0.0.0.0/0" not in p["allowed_ips"] for p in plan["peers"])
        assert not any("MASQUERADE" in rule for _, rule in control.firewall(plan))


def test_optional_env_is_literal_and_has_defaults(tmp_path):
    assert settings.load()["SPARKRING_SSH_USER"] == "root"
    path = tmp_path / ".env"
    path.write_text("SPARKRING_NAME=home\nSPARKRING_SHARE_INTERNET=no\n")
    assert settings.load(path)["SPARKRING_NAME"] == "home"
    for value in ("SPARKRING_NAME=$(id)", "PASSWORD=secret", "SPARKRING_SSH_PORT=9999", "SPARKRING_NAME=x\nSPARKRING_NAME=y"):
        path.write_text(value)
        with pytest.raises(ValueError):
            settings.load(path)


def test_ssh_hops_resolve_link_scope_on_the_jump_host_and_keep_key_local(tmp_path):
    route = [{"user": "root", "address": "fe80::1", "interface": "port0", "port": 22},
             {"user": "cody", "address": "fe80::2", "interface": "port1", "port": 22}]
    command = bootstrap.ssh_argv(route, tmp_path, identity=tmp_path / "private")
    proxy = next(a.removeprefix("ProxyCommand=") for a in command if a.startswith("ProxyCommand="))
    nested = shlex.split(proxy)
    assert nested[nested.index("-W") + 1] == "[fe80::2%%port1]:22"
    assert command[-1] == "cody@fe80::2%port1"
    assert "ForwardAgent=yes" not in json.dumps(command)
    assert "StrictHostKeyChecking=yes" in command
    assert command[command.index("-i") + 1] == str(tmp_path / "private")
    assert nested[nested.index("-i") + 1] == str(tmp_path / "private")


def test_bootstrap_does_not_invent_an_identity_file_when_using_existing_ssh_auth(tmp_path):
    route = [{"user": "root", "address": "fe80::1", "interface": "port0", "port": 22}]
    assert "-i" not in bootstrap.ssh_argv(route, tmp_path)


def test_bootstrap_discovery_uses_authenticated_identity_and_rejects_wrong_neighbor_mac():
    a = {"id": "a", "hostname": "a", "architecture": "aarch64", "routes": [], "os": {}, "functions": [
        {"netdev": "port0", "mac": "02:00:00:00:00:01", "addresses": ["fe80::1"]}],
        "neighbors": [{"dev": "port0", "dst": "fe80::2", "lladdr": "02:00:00:00:00:02"}]}
    b = copy.deepcopy(a)
    b.update(id="b", hostname="b", functions=[{"netdev": "port0", "mac": "02:00:00:00:00:02", "addresses": ["fe80::2"]}], neighbors=[])

    class FakeSSH:
        def login(self, route):
            pass

        def inventory(self, route):
            return b if route else a

    result = bootstrap.discover(FakeSSH())
    assert result["head"] == "a" and len(result["edges"]) == 1
    b["functions"][0]["mac"] = "02:00:00:00:00:ff"
    with pytest.raises(ValueError, match="matched"):
        bootstrap.discover(FakeSSH())
