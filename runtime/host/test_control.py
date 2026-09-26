"""Management routing, administration link checks and optional settings, without host access."""
import base64
import copy
import json
import os
import shlex
import subprocess

import pytest

from runtime.host import bootstrap, control, control_node, node, settings


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


CONTROL_PRIVATE = base64.b64encode(b"k" * 32).decode()
CONTROL_PUBLIC = base64.b64encode(b"p" * 32).decode()


class ControlHost:
    """One Spark's administration links as fake sysfs files and command answers."""

    def __init__(self, root, config, *, interface=True):
        self.root, self.interface, self.calls = root, interface, []
        self.link_local = {link["netdev"]: [link["address"]] for link in config["links"]}
        node.save(root, "/etc/sparkring/control.json", config, mode=0o600)
        (root / "etc/sparkring/control.key").write_text(CONTROL_PRIVATE + "\n")
        (root / "etc/ssh").mkdir(parents=True)
        (root / "etc/ssh/ssh_host_ed25519_key.pub").write_text("ssh-ed25519 " + CONTROL_PUBLIC + " host\n")
        for link in config["links"]:
            self.set_mac(link["netdev"], link["mac"])

    def set_mac(self, netdev, mac):
        directory = self.root / "sys/class/net" / netdev
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "address").write_text(mac + "\n")

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        stdout, code = "", 0
        if argv[:5] == ["ip", "-j", "-6", "address", "show"]:
            stdout = json.dumps([{"addr_info": [{"local": a, "scope": "link"} for a in self.link_local[argv[-1]]]}])
        elif argv[:4] == ["ip", "-j", "link", "show"]:
            code = 0 if self.interface else 1
        elif argv[:2] in (["wg", "pubkey"], ["wg", "show"]):
            stdout = CONTROL_PUBLIC + "\n"
        return subprocess.CompletedProcess(argv, code, stdout, "")

    def refreshed(self):
        return [call[4] for call in self.calls if call[:3] == ["wg", "set", control.INTERFACE]]


def head_config():
    config = control.plan(*fixture(), "0")[0]
    assert [p["netdev"] for p in config["peers"]] == ["port0", "port1"]
    return config


def break_link(host, config, netdev, kind):
    link = next(link for link in config["links"] if link["netdev"] == netdev)
    if kind == "mac":
        host.set_mac(netdev, "02:00:00:00:00:ff")
    elif kind == "link-local":
        host.link_local[netdev] = [link["address"], "fe80::99"]
    else:
        (host.root / "sys/class/net" / netdev / "address").unlink()


@pytest.mark.parametrize("kind", ["mac", "link-local", "absent"])
def test_up_refreshes_healthy_peers_and_names_the_failed_link(tmp_path, kind):
    config = head_config()
    host = ControlHost(tmp_path, config)
    break_link(host, config, "port1", kind)
    with pytest.raises(ValueError, match="port1") as raised:
        control_node.up(root=tmp_path, run=host)
    assert "port0" not in str(raised.value)
    healthy = next(p for p in config["peers"] if p["netdev"] == "port0")
    assert host.refreshed() == [healthy["key"]]
    assert not any(call[0] == "wg-quick" for call in host.calls)
    # Forwarding and firewall rules still apply, so the healthy links carry traffic.
    assert ["sysctl", "-w", f"net.ipv4.conf.{control.INTERFACE}.forwarding=1"] in host.calls
    assert any(call[:2] == ["iptables", "-w"] for call in host.calls)


def test_a_changed_mac_is_never_refreshed_or_queried(tmp_path):
    config = head_config()
    host = ControlHost(tmp_path, config)
    for netdev in ("port0", "port1"):
        break_link(host, config, netdev, "mac")
    with pytest.raises(ValueError, match="port0: MAC 02:00:00:00:00:ff differs"):
        control_node.up(root=tmp_path, run=host)
    assert host.refreshed() == []
    assert not any(call[:2] == ["ip", "-j"] and "-6" in call for call in host.calls)


@pytest.mark.skipif(os.name != "posix", reason="file modes need POSIX")
@pytest.mark.parametrize("kind", ["mac", "link-local", "absent"])
def test_up_creates_the_interface_without_the_endpoints_of_failed_links(tmp_path, kind):
    config = head_config()
    host = ControlHost(tmp_path, config, interface=False)
    break_link(host, config, "port0", kind)
    with pytest.raises(ValueError, match="port0") as raised:
        control_node.up(root=tmp_path, run=host)
    assert "port1" not in str(raised.value)
    [up] = [call for call in host.calls if call[0] == "wg-quick"]
    path = tmp_path / "run/sparkring-control" / (control.INTERFACE + ".conf")
    assert up == ["wg-quick", "up", str(path)]
    rendered = path.read_text()
    failed, healthy = (next(p for p in config["peers"] if p["netdev"] == name) for name in ("port0", "port1"))
    assert "PublicKey = " + failed["key"] in rendered and failed["endpoint"] not in rendered
    assert "Endpoint = " + healthy["endpoint"] in rendered
    assert path.stat().st_mode & 0o777 == 0o600 and path.parent.stat().st_mode & 0o777 == 0o700
    assert host.refreshed() == [healthy["key"]]
    assert ["sysctl", "-w", f"net.ipv4.conf.{control.INTERFACE}.forwarding=1"] in host.calls
    assert any(call[:2] == ["iptables", "-w"] for call in host.calls)


def test_up_creates_the_interface_from_its_configuration_when_every_link_passes(tmp_path):
    config = head_config()
    host = ControlHost(tmp_path, config, interface=False)
    assert control_node.up(root=tmp_path, run=host) == {"control_up": True}
    assert ["wg-quick", "up", control.INTERFACE] in host.calls
    assert host.refreshed() == [p["key"] for p in config["peers"]]
    assert not (tmp_path / "run/sparkring-control").exists()


def test_render_leaves_out_the_endpoints_of_named_links():
    config = head_config()
    private = base64.b64encode(b"a" * 32).decode()
    full, partial = control.render(config, private), control.render(config, private, without_endpoint={"port1"})
    assert full.count("Endpoint = ") == 2 and partial.count("Endpoint = ") == 1
    assert partial == full.replace("Endpoint = " + config["peers"][1]["endpoint"] + "\n", "")


def test_refresh_endpoint_checks_and_sets_one_peer_only(tmp_path):
    config = head_config()
    host = ControlHost(tmp_path, config)
    peer = control_node.refresh_endpoint("port1", root=tmp_path, run=host)
    assert peer == next(p for p in config["peers"] if p["netdev"] == "port1")
    assert host.calls == [["ip", "-j", "-6", "address", "show", "dev", "port1"],
                          ["wg", "set", control.INTERFACE, "peer", peer["key"], "endpoint", peer["endpoint"]]]

    host.calls.clear()
    break_link(host, config, "port1", "mac")
    with pytest.raises(ValueError, match="port1: MAC"):
        control_node.refresh_endpoint("port1", root=tmp_path, run=host)
    with pytest.raises(ValueError, match="port2: no administration network link"):
        control_node.refresh_endpoint("port2", root=tmp_path, run=host)
    assert host.refreshed() == []


def test_underlay_lists_every_failing_link_and_require_raises(tmp_path):
    assert control_node.underlay(root=tmp_path, run=pytest.fail) == []
    control_node.require_underlay(root=tmp_path, run=pytest.fail)
    config = head_config()
    host = ControlHost(tmp_path, config)
    assert control_node.underlay(root=tmp_path, run=host) == []
    break_link(host, config, "port0", "absent")
    break_link(host, config, "port1", "link-local")
    problems = control_node.underlay(root=tmp_path, run=host)
    assert [p["netdev"] for p in problems] == ["port0", "port1"]
    assert problems[0]["error"] == "interface is not present"
    assert control_node.underlay("port1", root=tmp_path, run=host) == problems[1:]
    with pytest.raises(ValueError, match="port0: interface is not present; port1: link-local"):
        control_node.require_underlay(root=tmp_path, run=host)
