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


@pytest.mark.parametrize(("text", "expected"), [
    ("none", None), ("None", None), ("850Mbit", 106_250_000), ("850mbit", 106_250_000), ("2Gbit", 250_000_000),
    ("1.5Gbit", 187_500_000), ("1Mbit", 125_000)])
def test_download_limit_is_a_rate_in_bits_per_second(text, expected):
    assert settings.download_limit(text) == expected


@pytest.mark.parametrize("text", ["850", "850M", "100MB/s", "850 Mbit", "0.5Mbit", "-1Gbit", "fast", ""])
def test_download_limit_refuses_other_forms_with_an_example(text):
    with pytest.raises(ValueError, match="850Mbit or 2Gbit"):
        settings.download_limit(text)


def test_download_limit_is_a_settings_file_preference(tmp_path):
    assert settings.load()["SPARKRING_DOWNLOAD_LIMIT"] == "none"
    path = tmp_path / ".env"
    path.write_text("SPARKRING_DOWNLOAD_LIMIT=850Mbit\n")
    assert settings.load(path)["SPARKRING_DOWNLOAD_LIMIT"] == "850Mbit"
    path.write_text("SPARKRING_DOWNLOAD_LIMIT=100MB\n")
    with pytest.raises(ValueError, match="850Mbit"):
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
    with pytest.raises(ValueError, match="none of its fabric functions has that address"):
        bootstrap.discover(FakeSSH())


def pair(**head):
    """Inventories of a Spark pair joined by one cable, with head fields overridden."""
    a = {"id": "hw-a", "machine_id": "m-a", "hostname": "a", "architecture": "aarch64", "functions": [
        {"netdev": "port0", "mac": "02:00:00:00:00:01", "addresses": ["fe80::1"]}],
        "neighbors": [{"dev": "port0", "dst": "fe80::2", "lladdr": "02:00:00:00:00:02", "answered": True}]}
    b = {"id": "hw-b", "machine_id": "m-b", "hostname": "b", "architecture": "aarch64", "functions": [
        {"netdev": "port0", "mac": "02:00:00:00:00:02", "addresses": ["fe80::2"]}], "neighbors": []}
    a.update(head)
    return a, b


class PairSSH:
    """Signs in to every route and records each route's last address."""

    def __init__(self, a, b):
        self.a, self.b, self.logins = a, b, []

    def login(self, route):
        self.logins.append(route[-1]["address"])

    def inventory(self, route):
        if route and route[-1]["address"] != "fe80::2":
            raise AssertionError("signed in to " + route[-1]["address"])
        return self.b if route else self.a


def test_fabric_identity_hashes_node_guids_and_keeps_machine_id_without_rdma():
    one = bootstrap.fabric_identity(["4cbb:4703:00e8:aa43\n", "4cbb:4703:00e8:aa47\n"], "m")
    assert one == bootstrap.fabric_identity(["4CBB:4703:00E8:AA47", "4cbb:4703:00e8:aa43"], "other")
    assert one != bootstrap.fabric_identity(["4cbb:4703:002c:931f"], "m") and len(one) == 32
    assert bootstrap.fabric_identity([], "m") == "m"


def test_discovery_tells_apart_sparks_that_share_a_machine_id():
    a, b = pair(machine_id="same")
    b["machine_id"] = "same"
    result = bootstrap.discover(PairSSH(a, b))
    assert {n["id"] for n in result["nodes"]} == {"hw-a", "hw-b"}
    assert len(result["warnings"]) == 1 and result["warnings"][0].startswith("a and b share /etc/machine-id")
    b["machine_id"] = "m-b"
    assert bootstrap.discover(PairSSH(a, b))["warnings"] == []


def test_discovery_refuses_a_neighbor_that_is_the_same_spark():
    a, b = pair()
    b["id"] = "hw-a"
    with pytest.raises(ValueError, match="reached itself"):
        bootstrap.discover(PairSSH(a, b))


def test_discovery_skips_cached_addresses_that_did_not_answer():
    phantom = {"dev": "port0", "dst": "fe80::6531:4cc1:4038:6c3d", "lladdr": "02:00:00:00:00:77", "answered": False}
    a, b = pair()
    a["neighbors"].insert(0, phantom)
    transport = PairSSH(a, b)
    assert len(bootstrap.discover(transport)["nodes"]) == 2 and transport.logins == ["fe80::2"]
    # An address that the authenticated Spark no longer holds is skipped even
    # when its entry carries that Spark's MAC.
    a["neighbors"].insert(0, dict(phantom, lladdr="02:00:00:00:00:02", answered=None))
    transport = PairSSH(a, b)
    assert len(bootstrap.discover(transport)["nodes"]) == 2 and transport.logins == ["fe80::2"]


def test_discovery_tries_neighbors_that_ignore_echo_when_none_answered():
    a, b = pair()
    a["neighbors"][0]["answered"] = False
    transport = PairSSH(a, b)
    assert len(bootstrap.discover(transport)["nodes"]) == 2 and transport.logins == ["fe80::2"]


def test_discovery_tries_the_next_address_after_one_gives_no_ssh_answer():
    a, b = pair()
    a["neighbors"].insert(0, {"dev": "port1", "dst": "fe80::2", "lladdr": "02:00:00:00:00:02", "answered": True})
    a["functions"].append({"netdev": "port1", "mac": "02:00:00:00:00:03", "addresses": ["fe80::3"]})

    class SharedCable(PairSSH):
        def login(self, route):
            super().login(route)
            if route[-1]["interface"] == "port1":
                raise bootstrap.Unanswered("fe80::2 on port1 did not answer SSH")

    transport = SharedCable(a, b)
    result = bootstrap.discover(transport)
    assert len(result["nodes"]) == 2 and transport.logins == ["fe80::2", "fe80::2"]
    assert result["routes"]["hw-b"][-1]["interface"] == "port0"


def test_discovery_names_skipped_addresses_when_no_pair_is_found():
    a, b = pair()
    a["neighbors"] = [{"dev": "port0", "dst": "fe80::1:2", "lladdr": "02:00:00:00:00:05", "answered": True},
                      {"dev": "port0", "dst": "fe80::6531:4cc1:4038:6c3d", "lladdr": "02:00:00:00:00:77", "answered": False}]

    class Refused(PairSSH):
        def login(self, route):
            raise ValueError(bootstrap.login_failure(route[-1], "ssh: connect to host port 22: Connection refused\n"))

    with pytest.raises(ValueError, match="refused SSH on port 22"):
        bootstrap.discover(Refused(a, b))
    with pytest.raises(ValueError, match=r"Found 1 Spark \(a\); setup needs two or four\. Skipped neighbor addresses "
                                         r"that did not answer: fe80::6531:4cc1:4038:6c3d on a's port0 \(no echo reply\)\."):
        bootstrap.discover(PairSSH(a, b), select=lambda peer: False)


@pytest.mark.parametrize("errors, cause", [
    ("code@fe80::2%port0: Permission denied (publickey,password).\n", "did not accept the password or account code"),
    ("ssh: connect to host fe80::2%port0 port 22: Connection refused\n", "--worker-bundle"),
    ("ssh: connect to host fe80::2%port0 port 22: Connection timed out\n", "did not answer SSH"),
    ("Connection closed by fe80::2%port0 port 22\n", "about 2 minutes"),
    ("Host key verification failed.\n", "has no recorded SSH host key yet"),
    ("kex_exchange_identification: read: Connection reset by peer\n", "closed the SSH connection"),
    ("something unexpected\n", "SSH sign-in to code@fe80::2 on port0 failed"),
])
def test_login_failure_names_the_cause_and_keeps_the_ssh_message(tmp_path, errors, cause):
    def run(argv, **kwargs):
        kwargs["stderr"].write("Warning: Permanently added 'fe80::2%port0' (ED25519) to the list of known hosts.\n" + errors)
        return subprocess.CompletedProcess(argv, 255)

    route = [{"user": "code", "address": "fe80::2", "interface": "port0", "port": 22}]
    with pytest.raises(ValueError) as failure:
        bootstrap.SSH(tmp_path, run=run).login(route)
    message = str(failure.value)
    assert cause in message and message.endswith("(ssh: " + errors.strip().removeprefix("ssh: ") + ")")
    assert len(message.splitlines()) == 1
    assert isinstance(failure.value, bootstrap.Unanswered) == ("timed out" in errors)
    assert "Permanently added" not in message and "worker-bundle" not in message.replace(cause, "")
    bootstrap.SSH(tmp_path, run=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0)).login(route)


def test_a_changed_host_key_is_told_apart_from_an_unknown_one(tmp_path):
    changed = ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
               "@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @\n"
               "Offending ED25519 key in /var/lib/sparkring/controller/ssh/known_hosts:2\n"
               "Host key verification failed.\n")
    unknown = ("No ED25519 host key is known for fe80::2%port0 and you have requested strict checking.\n"
               "Host key verification failed.\n")
    messages = []
    for errors in (changed, unknown):
        def run(argv, errors=errors, **kwargs):
            kwargs["stderr"].write(errors)
            return subprocess.CompletedProcess(argv, 255)

        route = [{"user": "code", "address": "fe80::2", "interface": "port0", "port": 22}]
        with pytest.raises(ValueError) as failure:
            bootstrap.SSH(tmp_path, run=run).login(route)
        messages.append(str(failure.value))
    assert "differs from the one recorded" in messages[0] and "known_hosts:2" in messages[0]
    assert "has no recorded SSH host key yet" in messages[1]


def test_control_configuration_names_this_spark_by_hardware_or_machine_id(tmp_path):
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc/machine-id").write_text("factory\n")
    for device, guid in (("rocep1s0f0", "4cbb:4703:00e8:aa43"), ("rocep1s0f1", "4cbb:4703:00e8:aa44")):
        (tmp_path / "sys/class/infiniband" / device).mkdir(parents=True)
        (tmp_path / "sys/class/infiniband" / device / "node_guid").write_text(guid + "\n")
    hardware = bootstrap.fabric_identity(["4cbb:4703:00e8:aa43", "4cbb:4703:00e8:aa44"], "factory")
    assert control_node.identities(tmp_path) == {"factory", hardware}
    with pytest.raises(ValueError, match="belongs to another machine"):
        control_node.configure({"control": {"schema": "sparkring-control/v1", "id": "another"}}, root=tmp_path)


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


# Fallback paths of the administration tunnel. A pair's Sparks have four
# ConnectX functions on two ports, both ports cabled; the primary path is the
# port 1 cable, as on a pair whose discovery found that cable first.
FUNCTIONS = (("enp1s0f0np0", "rocep1s0f0"), ("enP2p1s0f0np0", "roceP2p1s0f0"),
             ("enp1s0f1np1", "rocep1s0f1"), ("enP2p1s0f1np1", "roceP2p1s0f1"))
LAN_NETDEV = "enP7s7"


def mac(letter, index):
    return f"02:00:00:00:0{'ab'.index(letter) + 1}:0{index}"


def spark(letter, lan_address):
    """The inventory of Spark ``hw-LETTER`` (bootstrap.probe); its functions' link-local addresses are fe80::LETTERn."""
    return {"id": "hw-" + letter, "machine_id": "m-" + letter, "hostname": "spark-" + letter, "architecture": "aarch64",
            "public_key": base64.b64encode(letter.encode() * 32).decode(), "routes": [],
            "uplink": LAN_NETDEV, "api_address": lan_address,
            "functions": [{"device": device, "netdev": netdev, "mac": mac(letter, index), "carrier": True,
                           "addresses": [f"fe80::{letter}{index}"]} for index, (netdev, device) in enumerate(FUNCTIONS)],
            "neighbors": []}


def neighbor(netdev, letter, index, answered=True):
    return {"dev": netdev, "dst": f"fe80::{letter}{index}", "lladdr": mac(letter, index), "answered": answered}


def pair_inventories():
    a, b = spark("a", "198.51.100.200"), spark("b", "198.51.100.137")
    # Node A sees each worker function on its own cable; Socket Direct also
    # shows it a crossed function, and one cache entry did not answer.
    a["neighbors"] = [neighbor(netdev, "b", index) for index, (netdev, _) in enumerate(FUNCTIONS)]
    a["neighbors"] += [neighbor("enp1s0f0np0", "b", 1), neighbor("enP2p1s0f1np1", "b", 0, answered=False)]
    b["neighbors"] = [neighbor("enp1s0f1np1", "a", 2)]
    return a, b


def pair_plan(**changes):
    a, b = pair_inventories()
    a.update(changes)
    edge = [{"id": "hw-a", "netdev": "enp1s0f1np1", "address": "fe80::a2", "mac": mac("a", 2)},
            {"id": "hw-b", "netdev": "enp1s0f1np1", "address": "fe80::b2", "mac": mac("b", 2)}]
    return control.plan([a, b], [edge], "hw-a"), {"hw-a": a, "hw-b": b}


def cable(local, peer, netdev, index):
    return {"via": "cable", "netdev": netdev, "mac": mac(local, index), "address": f"fe80::{local}{index}",
            "peer": f"fe80::{peer}{index}"}


def test_plan_lists_the_other_cables_then_the_lan_as_fallbacks_in_the_same_order_on_both_ends():
    (head, worker), inventories = pair_plan()
    [to_worker], [to_head] = head["peers"], worker["peers"]
    assert to_worker["endpoint"] == "[fe80::b2%enp1s0f1np1]:51871"
    # The port 0 cable first, parallel functions before crossed ones; then the
    # primary cable's other function; then the LAN.
    assert to_worker["alternates"] == [cable("a", "b", "enP2p1s0f0np0", 1), cable("a", "b", "enp1s0f0np0", 0),
                                       cable("a", "b", "enP2p1s0f1np1", 3),
                                       {"via": "lan", "netdev": LAN_NETDEV, "peer": "198.51.100.137"}]
    assert to_head["alternates"] == [cable("b", "a", "enP2p1s0f0np0", 1), cable("b", "a", "enp1s0f0np0", 0),
                                     cable("b", "a", "enP2p1s0f1np1", 3),
                                     {"via": "lan", "netdev": LAN_NETDEV, "peer": "198.51.100.200"}]
    assert (to_worker["address"], to_head["address"]) == ("10.253.255.2", "10.253.255.1")
    for config in (head, worker):
        control.validate(config)
    # An installed configuration without fallbacks gains exactly these.
    assert control.extend([control.base(c) for c in (head, worker)], inventories) == [head, worker]
    assert control.render(head, CONTROL_PRIVATE) == control.render(control.base(head), CONTROL_PRIVATE)


def test_plan_without_a_lan_address_or_second_cable_lists_what_exists():
    (head, worker), _ = pair_plan(api_address=None)
    assert [p["via"] for p in head["peers"][0]["alternates"]] == ["cable"] * 3
    nodes, edges = fixture(2)
    assert all(p["alternates"] == [] for config in control.plan(nodes, edges, "0") for p in config["peers"])


def test_ring_tree_links_get_lan_fallbacks_between_their_two_sparks():
    nodes, edges = fixture(4)
    for rank, n in enumerate(nodes):
        n.update(api_address=f"198.51.100.{130 + rank}")
    plans = {p["id"]: p for p in control.plan(nodes, edges, "0")}
    for config in plans.values():
        for peer in config["peers"]:
            assert peer["alternates"] == [{"via": "lan", "netdev": "eth0", "peer": f"198.51.100.{130 + int(peer['id'])}"}]
    head_rules = control.firewall(plans["0"])
    for peer in plans["0"]["peers"]:
        assert ("iptables", ["INPUT", "-i", "eth0", "-s", f"198.51.100.{130 + int(peer['id'])}/32", "-p", "udp",
                             "--dport", "51871", "-j", "ACCEPT"]) in head_rules


def test_firewall_opens_the_tunnel_port_on_fallbacks_only_to_that_peer():
    (head, _), _ = pair_plan()
    tunnel = [rule for rule in control.firewall(head) if "51871" in rule[1]]
    assert tunnel == [
        ("ip6tables", ["INPUT", "-i", "enp1s0f1np1", "-s", "fe80::/10", "-p", "udp", "--dport", "51871", "-j", "ACCEPT"]),
        ("ip6tables", ["INPUT", "-i", "enP2p1s0f0np0", "-s", "fe80::b1/128", "-p", "udp", "--dport", "51871", "-j", "ACCEPT"]),
        ("ip6tables", ["INPUT", "-i", "enp1s0f0np0", "-s", "fe80::b0/128", "-p", "udp", "--dport", "51871", "-j", "ACCEPT"]),
        ("ip6tables", ["INPUT", "-i", "enP2p1s0f1np1", "-s", "fe80::b3/128", "-p", "udp", "--dport", "51871", "-j", "ACCEPT"]),
        ("iptables", ["INPUT", "-i", LAN_NETDEV, "-s", "198.51.100.137/32", "-p", "udp", "--dport", "51871", "-j", "ACCEPT"])]


@pytest.mark.parametrize("change, message", [
    (lambda path: path.update(via="wifi"), "cable or the LAN"),
    (lambda path: path.update(address="2001:db8::1"), "link-local"),
    (lambda path: path.update(extra=1), "unexpected fields"),
    (lambda path: path.update(netdev="bad name"), "Invalid fabric interface"),
])
def test_fallback_paths_are_validated(change, message):
    (head, _), _ = pair_plan()
    change(head["peers"][0]["alternates"][0])
    with pytest.raises(ValueError, match=message):
        control.validate(head)


def test_a_fallback_lan_address_inside_the_administration_subnet_is_refused():
    (head, _), _ = pair_plan()
    head["peers"][0]["alternates"][-1]["peer"] = "10.253.255.3"
    with pytest.raises(ValueError, match="outside the administration subnet"):
        control.validate(head)


class TunnelHost(ControlHost):
    """One Spark of the pair: carrier files, fallback functions, and a WireGuard peer that ``wg set`` moves."""

    def __init__(self, root, config, *, now):
        super().__init__(root, config)
        self.peer = config["peers"][0]
        for path in self.peer["alternates"]:
            if path["via"] == "cable":
                self.set_mac(path["netdev"], path["mac"])
                self.link_local[path["netdev"]] = [path["address"]]
        for name, _ in FUNCTIONS:
            self.carrier(name, True)
        self.carrier(LAN_NETDEV, True)
        self.endpoint, self.handshake, self.received = self.peer["endpoint"], now - 30, 1000
        self.contacted, self.said = [], []

    def carrier(self, netdev, up):
        directory = self.root / "sys/class/net" / netdev
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "carrier").write_text("1\n" if up else "0\n")
        (directory / "operstate").write_text("up\n" if up else "down\n")

    def __call__(self, argv, **kwargs):
        if argv == ["wg", "show", control.INTERFACE, "dump"]:
            self.calls.append(list(argv))
            dump = (f"PRIVATE\tPUBLIC\t51871\toff\n{self.peer['key']}\t(none)\t{self.endpoint}\t10.253.255.0/29\t"
                    f"{self.handshake}\t{self.received}\t2000\t15\n")
            return subprocess.CompletedProcess(argv, 0, dump, "")
        if argv[:3] == ["wg", "set", control.INTERFACE]:
            self.endpoint = argv[-1]
        return super().__call__(argv, **kwargs)

    def up(self, now):
        self.calls.clear()
        return control_node.up(root=self.root, run=self, now=lambda: now, contact=self.contacted.append,
                               say=self.said.append)


def unplug(host, cable_port, up=False):
    """Set the carrier of both functions of one ConnectX port, as a cable change does."""
    for name, device in FUNCTIONS:
        if device.endswith(f"f{cable_port}"):
            host.carrier(name, up)


T = 1_000_000


def test_an_unplugged_primary_cable_moves_the_peer_to_the_other_cable_and_back(tmp_path):
    (head, _), _ = pair_plan()
    host = TunnelHost(tmp_path, head, now=T)
    assert host.up(T) == {"control_up": True}
    assert host.refreshed() == [] and host.contacted == []
    # The primary cable loses carrier on both of its functions.
    unplug(host, 1)
    assert host.up(T + 20) == {"control_up": True}
    assert host.endpoint == "[fe80::b1%enP2p1s0f0np0]:51871" and host.contacted == ["10.253.255.2"]
    assert host.said == ["sr-control: peer 10.253.255.2 now uses cable enP2p1s0f0np0 (cable enp1s0f1np1: link down)"]
    # The worker answered over the port 0 cable; the peer stays there.
    host.received += 60
    host.up(T + 40)
    assert host.refreshed() == [] and host.contacted == ["10.253.255.2"]
    report = node.control_report(root=tmp_path, run=host, now=lambda: T + 40)
    assert report["peers"][0]["path"] == {"via": "cable", "netdev": "enP2p1s0f0np0", "address": "fe80::b1",
                                          "primary": False}
    assert node.tunnel_problems(report) == ["admin tunnel to 10.253.255.2 runs over cable enP2p1s0f0np0 "
                                            "(primary cable enp1s0f1np1: no link)"]
    # The cable returns: once its link has passed on two refreshes, the peer returns to it.
    unplug(host, 1, up=True)
    host.up(T + 60)
    assert host.refreshed() == []
    host.up(T + 80)
    assert host.endpoint == "[fe80::b2%enp1s0f1np1]:51871" and host.contacted == ["10.253.255.2"] * 2
    assert host.said[-1] == ("sr-control: peer 10.253.255.2 now uses cable enp1s0f1np1 "
                             "(cable enp1s0f1np1 passes its link check again)")
    host.handshake = T + 90
    host.up(T + 100)
    assert host.refreshed() == []
    assert node.tunnel_problems(node.control_report(root=tmp_path, run=host, now=lambda: T + 100)) == []


def test_a_missing_primary_function_is_carried_by_a_fallback_without_failing_the_refresh(tmp_path):
    (head, _), _ = pair_plan()
    host = TunnelHost(tmp_path, head, now=T)
    break_link(host, head, "enp1s0f1np1", "absent")
    assert host.up(T) == {"control_up": True}
    assert host.endpoint == "[fe80::b1%enP2p1s0f0np0]:51871"
    assert host.said[-1] == "sr-control: enp1s0f1np1: interface is not present; a fallback path carries its peer"
    # Without any usable path, the failed link is named as before.
    for name, _ in FUNCTIONS:
        host.carrier(name, False)
    host.carrier(LAN_NETDEV, False)
    with pytest.raises(ValueError, match="enp1s0f1np1: interface is not present"):
        host.up(T + 20)


def test_a_fallback_that_does_not_answer_is_skipped_for_the_next(tmp_path):
    (head, _), _ = pair_plan()
    host = TunnelHost(tmp_path, head, now=T)
    unplug(host, 1)
    host.up(T)
    assert host.endpoint == "[fe80::b1%enP2p1s0f0np0]:51871"
    # No handshake and no byte arrived over that cable by the next refresh.
    host.up(T + 20)
    assert host.endpoint == "[fe80::b0%enp1s0f0np0]:51871"
    assert host.said[-1] == ("sr-control: peer 10.253.255.2 now uses cable enp1s0f0np0 "
                             "(no answer over cable enP2p1s0f0np0)")
    host.up(T + 40)
    # The primary cable's other function has no carrier either; the LAN is next.
    assert host.endpoint == "198.51.100.137:51871" and host.contacted == ["10.253.255.2"] * 3
    host.handshake = T + 50
    host.up(T + 60)
    assert host.refreshed() == []


def test_a_peer_moved_by_the_other_sparks_packets_stays_where_they_arrive(tmp_path):
    """The worker's own links pass; Node A moved the tunnel to the port 0 cable, and WireGuard followed."""
    (_, worker), _ = pair_plan()
    host = TunnelHost(tmp_path, worker, now=T)
    host.up(T)
    host.endpoint = "[fe80::a1%enP2p1s0f0np0]:51871"
    for step in range(1, 6):
        host.handshake = T + 20 * step - 5
        host.up(T + 20 * step)
        assert host.refreshed() == [] and host.contacted == []
    # Node A returns to the primary cable; WireGuard follows again.
    host.endpoint = "[fe80::a2%enp1s0f1np1]:51871"
    host.up(T + 120)
    assert host.refreshed() == []


def test_a_failed_endpoint_change_is_logged_and_chosen_again_at_the_next_refresh(tmp_path):
    (head, _), _ = pair_plan()
    host = TunnelHost(tmp_path, head, now=T)
    unplug(host, 1)
    real = host.__call__

    def vanished(argv, **kwargs):
        if argv[:3] == ["wg", "set", control.INTERFACE]:
            host.calls.append(list(argv))
            return subprocess.CompletedProcess(argv, 1, "", "Name or service not known")
        return real(argv, **kwargs)

    assert control_node.up(root=tmp_path, run=vanished, now=lambda: T, contact=host.contacted.append,
                           say=host.said.append) == {"control_up": True}
    assert host.endpoint == head["peers"][0]["endpoint"] and host.contacted == []
    assert host.said == ["sr-control: peer 10.253.255.2: wg set sr-control peer: Name or service not known"]
    host.up(T + 20)
    assert host.endpoint == "[fe80::b1%enP2p1s0f0np0]:51871" and host.contacted == ["10.253.255.2"]


def test_a_stale_interface_index_is_set_again_without_moving_the_peer(tmp_path):
    (head, _), _ = pair_plan()
    host = TunnelHost(tmp_path, head, now=T)
    host.endpoint = "[fe80::b2%17]:51871"
    host.up(T)
    assert host.endpoint == "[fe80::b2%enp1s0f1np1]:51871" and host.contacted == [] and host.said == []


def test_a_failed_return_to_the_primary_holds_it_back_with_a_doubling_delay():
    handshake, state, current, outcomes = T - 10, None, 0, []
    # (time, links usable, bytes received): the primary is down, comes back at
    # T + 40, and gives no answer after the return at T + 60.
    for at, links, received in [(T, [False, True], 0), (T + 20, [False, True], 5), (T + 40, [True, True], 5),
                                (T + 60, [True, True], 5), (T + 80, [True, True], 5), (T + 100, [True, True], 6)]:
        target, check, reason, state = control.choose(state, links, current, handshake, received, at, has_endpoint=True)
        outcomes.append((target, check, reason))
        current = target if target is not None else current
    assert outcomes == [(1, True, "link"),
                        (1, False, None),       # bytes arrived over the fallback: confirmed
                        (1, False, None),       # the primary passes, not yet for PRIMARY_SETTLE
                        (0, True, "primary"),
                        (1, True, "answer"),    # no answer on the primary: back to the fallback
                        (1, False, None)]
    assert state["held"] == T + 80 + control.PRIMARY_RETRY and state["returns"] == 1
    # The fallback keeps its handshakes; the primary is tried again when the hold ends.
    later = T + 80 + control.PRIMARY_RETRY
    target, _, reason, state = control.choose(state, [True, True], 1, later - 10, 6, later, has_endpoint=True)
    assert (target, reason) == (0, "primary")
    target, _, _, state = control.choose(state, [True, True], 0, later - 10, 6, later + 20, has_endpoint=True)
    assert target == 1 and state["returns"] == 2 and state["held"] == later + 20 + 2 * control.PRIMARY_RETRY
    # A primary link that goes down and up again is tried at once.
    _, _, _, state = control.choose(state, [False, True], 1, later + 30, 7, later + 40, has_endpoint=True)
    assert state["held"] is None and state["returns"] == 0


def test_when_every_path_fails_the_peer_settles_on_the_most_preferred_usable_one():
    handshake, state, current, seen = T - 1000, None, 0, []
    for at in range(T, T + 200, 20):
        target, _, _, state = control.choose(state, [False, True, True], current, handshake, 0, at, has_endpoint=True)
        seen.append(target)
        current = target if target is not None else current
    # Each fallback gets one refresh to answer; then the first usable one stays.
    assert seen[:3] == [1, 2, 1] and set(seen[3:]) == {1}
    assert state["verify"] is None and set(state["failed"]) == {"1", "2"}
    # A handshake after the failures clears them.
    _, _, _, state = control.choose(state, [False, True, True], 1, T + 300, 0, T + 310, has_endpoint=True)
    assert state["failed"] == {}


def test_a_path_without_a_handshake_for_the_stale_limit_is_left():
    target, _, _, state = control.choose(None, [True, True], 0, T, 0, T, has_endpoint=True)
    assert target == 0
    target, _, _, state = control.choose(state, [True, True], 0, T, 0, T + control.HANDSHAKE_STALE, has_endpoint=True)
    assert target == 0
    target, check, reason, state = control.choose(state, [True, True], 0, T, 0, T + control.HANDSHAKE_STALE + 1,
                                                  has_endpoint=True)
    assert (target, check, reason) == (1, True, "handshake") and state["held"] is not None


def test_configure_adds_fallbacks_to_an_installed_configuration_and_refuses_other_changes(tmp_path):
    (head, _), _ = pair_plan()
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc/machine-id").write_text("hw-a\n")
    node.save(tmp_path, "/etc/sparkring/control.json", control.base(head), mode=0o600)
    calls = []

    def run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    document = {"control": head, "ssh_key": "ssh-ed25519 " + CONTROL_PUBLIC + " controller"}
    assert control_node.configure(document, root=tmp_path, run=run) == {
        "configured": True, "address": "10.253.255.1", "fallback_paths": 4}
    assert node.read(tmp_path, "/etc/sparkring/control.json") == head
    assert calls == [["systemctl", "start", "--no-block", "sparkring-control-refresh.service"]]
    # Replacing the fallbacks removes the firewall rules of paths it does not list.
    replaced = copy.deepcopy(head)
    replaced["peers"][0]["alternates"][-1]["peer"] = "198.51.100.140"
    calls.clear()
    control_node.configure({**document, "control": replaced}, root=tmp_path, run=run)
    assert [call for call in calls if "-D" in call] == [
        ["iptables", "-w", "-D", "INPUT", "-i", LAN_NETDEV, "-s", "198.51.100.137/32", "-p", "udp", "--dport", "51871",
         "-m", "comment", "--comment", "sparkring-control", "-j", "ACCEPT"]]
    for change in (lambda c: c.update(address="10.253.255.3"), lambda c: c["peers"][0].update(key=CONTROL_PUBLIC),
                   lambda c: c["peers"][0].update(endpoint="[fe80::b9%enp1s0f1np1]:51871")):
        other = copy.deepcopy(replaced)
        change(other)
        with pytest.raises(ValueError, match="different control network"):
            control_node.configure({**document, "control": other}, root=tmp_path, run=run)
    assert node.read(tmp_path, "/etc/sparkring/control.json") == replaced


def test_setup_admin_fallback_extends_every_installed_spark(tmp_path, monkeypatch, capsys):
    from runtime.common import installer
    from runtime.host import single_uplink
    (head, worker), inventories = pair_plan()
    targets = ["root@" + config["address"] for config in (head, worker)]
    installer.write(tmp_path / "enrolled.json", {"targets": targets})
    installed = {targets[0]: control.base(head), targets[1]: control.base(worker)}
    probes = {targets[0]: inventories["hw-a"], targets[1]: inventories["hw-b"]}
    configured = {}

    def invoke(target, argv, data=None):
        if argv[:3] == ["sudo", "-n", "cat"]:
            return json.dumps(installed[target])
        if argv[:4] == ["sudo", "-n", "python3", "-I"]:
            return json.dumps(probes[target])
        assert argv == ["sudo", "-n", "/usr/bin/sparkring", "node", "control-configure"]
        configured[target] = json.loads(data)
        return "{}"

    updated = []
    monkeypatch.setattr(single_uplink, "match_revisions", lambda t, nodes, directory: updated.append(t) or nodes)

    def run(*flags):
        args = single_uplink._arguments(["--admin-fallback", *flags])[0]
        return single_uplink.admin_fallback(args, tmp_path, "ssh-ed25519 KEY", tmp_path / "setup", invoke=invoke,
                                            collect=lambda t: [], root=tmp_path)

    with pytest.raises(ValueError, match="no SparkRing administration network"):
        run("--plan")
    node.save(tmp_path, "/etc/sparkring/control.json", installed[targets[0]], mode=0o600)
    assert run("--plan") == 0
    out = capsys.readouterr().out
    assert ("  spark-a to 10.253.255.2: cable enP2p1s0f0np0, cable enp1s0f0np0, cable enP2p1s0f1np1, "
            "LAN 198.51.100.137") in out
    assert configured == {} and updated == []
    assert run("--yes") == 0
    assert configured == {targets[0]: {"control": head, "ssh_key": "ssh-ed25519 KEY"},
                          targets[1]: {"control": worker, "ssh_key": "ssh-ed25519 KEY"}}
    assert updated == [targets]
    installed.update({target: document["control"] for target, document in configured.items()})
    configured.clear()
    assert run("--yes") == 0
    assert configured == {} and "Every Spark already has these fallback paths." in capsys.readouterr().out
