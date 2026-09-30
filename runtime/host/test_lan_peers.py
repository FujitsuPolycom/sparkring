"""Preparing cabled Sparks through the LAN, with fake inventories, commands and SSH."""
import json
import subprocess

import pytest

from runtime.host import lan_peers

HEAD = {"hostname": "spark-3286", "uplink": "enP7s7",
        "functions": [{"netdev": "p0", "mac": "4C:BB:47:E9:32:87", "carrier": True},
                      {"netdev": "p1", "mac": "4c:bb:47:e9:32:88", "carrier": False}],
        "neighbors": [{"dev": "p0", "dst": "fe80::4ebb:47ff:fee9:a10", "lladdr": "4c:bb:47:e9:0a:10"},
                      {"dev": "p0", "dst": "fe80::4ebb:47ff:fee9:328b", "lladdr": "4c:bb:47:e9:32:87"},
                      {"dev": "enP7s7", "dst": "fe80::9", "lladdr": "4c:bb:47:e6:ed:fd"}]}


def test_peer_macs_are_other_hosts_on_fabric_links_only():
    assert lan_peers.peer_macs(HEAD) == ["4c:bb:47:e9:0a:10"]


def test_match_pairs_a_fabric_mac_with_the_lan_mac_just_below_it():
    hosts = {"4c:bb:47:e9:0a:0f": "192.168.0.232", "4c:bb:47:e6:ed:fd": "192.168.0.193", "f0:68:e3:b3:13:ce": "192.168.0.239",
             "4c:bb:47:e9:0a:05": "192.168.0.9"}
    assert lan_peers.match(["4c:bb:47:e9:0a:10", "4c:bb:47:e9:0a:14"], hosts) == {
        "4c:bb:47:e9:0a:0f": ("192.168.0.232", "4c:bb:47:e9:0a:10")}
    assert lan_peers.match(["4c:bb:47:e9:0a:10"], {"4c:bb:47:e9:0a:10": "192.168.0.5"}) == {}


def test_lan_hosts_sweeps_the_subnet_only_when_no_cabled_spark_is_in_the_arp_table():
    table = {"rows": [{"dst": "192.168.0.254", "lladdr": "00:11:22:33:44:55", "state": ["REACHABLE"]}]}
    pinged = []

    def run(argv, **kwargs):
        if argv[:4] == ["ip", "-j", "-4", "neigh"]:
            return subprocess.CompletedProcess(argv, 0, json.dumps(table["rows"]), "")
        rows = [{"addr_info": [{"family": "inet", "local": "192.168.0.242", "prefixlen": 29}]}]
        return subprocess.CompletedProcess(argv, 0, json.dumps(rows), "")

    class Ping:
        def __init__(self, argv, **kwargs):
            pinged.append(argv[-1])
            if argv[-1] == "192.168.0.244":
                table["rows"].append({"dst": "192.168.0.244", "lladdr": "4C:BB:47:E9:0A:0F", "state": ["STALE"]})

        def wait(self):
            return 0

    hosts = lan_peers.lan_hosts("enP7s7", ["4c:bb:47:e9:0a:10"], run=run, popen=Ping)
    assert hosts["4c:bb:47:e9:0a:0f"] == "192.168.0.244"
    assert pinged == ["192.168.0.241", "192.168.0.243", "192.168.0.244", "192.168.0.245", "192.168.0.246"]
    pinged.clear()
    assert lan_peers.lan_hosts("enP7s7", ["4c:bb:47:e9:0a:10"], run=run, popen=Ping) == hosts and pinged == []


def test_waiting_polls_until_the_cabled_spark_answers():
    alone = dict(HEAD, neighbors=[])
    inventories = [alone, alone, HEAD]
    clock, said = [0.0], []

    def sleep(seconds):
        clock[0] += seconds

    head, macs = lan_peers.wait_for_peers(lambda: inventories.pop(0), clock=lambda: clock[0], sleep=sleep, say=said.append)
    assert macs == ["4c:bb:47:e9:0a:10"] and len(said) == 1 and clock[0] == 2 * lan_peers.POLL_SECONDS


def test_waiting_returns_at_once_without_a_cable_and_stops_at_the_deadline():
    uncabled = dict(HEAD, neighbors=[], functions=[dict(f, carrier=False) for f in HEAD["functions"]])
    assert lan_peers.wait_for_peers(lambda: uncabled, sleep=pytest.fail) == (uncabled, [])
    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds

    head, macs = lan_peers.wait_for_peers(lambda: dict(HEAD, neighbors=[]), clock=lambda: clock[0], sleep=sleep, say=lambda _: None)
    assert macs == [] and clock[0] >= lan_peers.WAIT_SECONDS


def test_prepare_signs_in_over_the_lan_and_runs_worker_preparation(monkeypatch):
    peer = {"hostname": "spark-0a0f", "functions": [{"netdev": "p0", "mac": "4c:bb:47:e9:0a:10"}]}
    calls = []

    class Transport:
        def login(self, route):
            calls.append(("login", route[0]["address"], route[0]["interface"], route[0]["user"]))

        def inventory(self, route):
            return peer

    monkeypatch.setattr(lan_peers, "lan_hosts", lambda interface, macs, run: {"4c:bb:47:e9:0a:0f": "192.168.0.232"})
    monkeypatch.setattr(lan_peers.packages, "transfer", lambda transport, route, archive, target: calls.append(("transfer", archive)))

    def root_command(transport, route, argv):
        calls.append(("root", route[0]["address"], argv[-3:]))

    prepared = lan_peers.prepare(Transport(), root_command, head=HEAD, user="code", archive=lambda: "bundle.tar", say=lambda _: None)
    assert prepared == [peer]
    assert calls == [("login", "192.168.0.232", None, "code"), ("transfer", "bundle.tar"),
                     ("root", "192.168.0.232", ["--apply", "--prepare", "--yes"])]
    peer["functions"][0]["mac"] = "4c:bb:47:e9:77:77"
    with pytest.raises(ValueError, match="not the Spark on the cable"):
        lan_peers.prepare(Transport(), root_command, head=HEAD, user="code", archive=lambda: "bundle.tar", say=lambda _: None)
    assert lan_peers.prepare(Transport(), root_command, head=dict(HEAD, neighbors=[]), user="code", archive=pytest.fail) == []


def test_a_lan_hop_is_a_private_ipv4_address_without_a_zone(tmp_path):
    from runtime.host import bootstrap
    hop = {"user": "code", "address": "192.168.0.232", "interface": None, "port": 22}
    assert bootstrap.ssh_argv([hop], tmp_path)[-1] == "code@192.168.0.232"
    for bad in (dict(hop, address="8.8.8.8"), dict(hop, port=2222)):
        with pytest.raises(ValueError, match="private IPv4"):
            bootstrap.validate_hop(bad)
    assert "192.168.0.232 on the LAN did not accept" in bootstrap.login_failure(hop, "Permission denied (password).")
