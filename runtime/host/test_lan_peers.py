"""Preparing cabled Sparks through the LAN, with fake inventories, commands and SSH."""
import json
import subprocess

import pytest

from runtime.host import lan_peers

HEAD = {"hostname": "spark-b", "uplink": "enP7s7",
        "functions": [{"netdev": "p0", "mac": "02:00:00:FA:43:98", "carrier": True},
                      {"netdev": "p1", "mac": "02:00:00:fa:43:99", "carrier": False}],
        "neighbors": [{"dev": "p0", "dst": "fe80::ff:fefa:1b21", "lladdr": "02:00:00:fa:1b:21"},
                      {"dev": "p0", "dst": "fe80::ff:fefa:439c", "lladdr": "02:00:00:fa:43:98"},
                      {"dev": "enP7s7", "dst": "fe80::9", "lladdr": "02:00:00:f7:ff:0e"}]}


def test_peer_macs_are_other_hosts_on_fabric_links_only():
    assert lan_peers.peer_macs(HEAD) == ["02:00:00:fa:1b:21"]


def test_match_pairs_a_fabric_mac_with_the_lan_mac_just_below_it():
    hosts = {"02:00:00:fa:1b:20": "192.0.2.232", "02:00:00:f7:ff:0e": "192.0.2.193", "02:00:01:00:00:17": "192.0.2.239",
             "02:00:00:fa:1b:16": "192.0.2.9"}
    assert lan_peers.match(["02:00:00:fa:1b:21", "02:00:00:fa:1b:25"], hosts) == {
        "02:00:00:fa:1b:20": ("192.0.2.232", "02:00:00:fa:1b:21")}
    assert lan_peers.match(["02:00:00:fa:1b:21"], {"02:00:00:fa:1b:21": "192.0.2.5"}) == {}


def test_lan_hosts_sweeps_the_subnet_only_when_no_cabled_spark_is_in_the_arp_table():
    table = {"rows": [{"dst": "192.0.2.254", "lladdr": "00:11:22:33:44:55", "state": ["REACHABLE"]}]}
    pinged = []

    def run(argv, **kwargs):
        if argv[:4] == ["ip", "-j", "-4", "neigh"]:
            return subprocess.CompletedProcess(argv, 0, json.dumps(table["rows"]), "")
        rows = [{"addr_info": [{"family": "inet", "local": "192.0.2.242", "prefixlen": 29}]}]
        return subprocess.CompletedProcess(argv, 0, json.dumps(rows), "")

    class Ping:
        def __init__(self, argv, **kwargs):
            pinged.append(argv[-1])
            if argv[-1] == "192.0.2.244":
                table["rows"].append({"dst": "192.0.2.244", "lladdr": "02:00:00:FA:1B:20", "state": ["STALE"]})

        def wait(self):
            return 0

    hosts = lan_peers.lan_hosts("enP7s7", ["02:00:00:fa:1b:21"], run=run, popen=Ping)
    assert hosts["02:00:00:fa:1b:20"] == "192.0.2.244"
    assert pinged == ["192.0.2.241", "192.0.2.243", "192.0.2.244", "192.0.2.245", "192.0.2.246"]
    pinged.clear()
    assert lan_peers.lan_hosts("enP7s7", ["02:00:00:fa:1b:21"], run=run, popen=Ping) == hosts and pinged == []


def test_waiting_polls_until_the_cabled_spark_answers():
    alone = dict(HEAD, neighbors=[])
    inventories = [alone, alone, HEAD]
    clock, said = [0.0], []

    def sleep(seconds):
        clock[0] += seconds

    head, macs = lan_peers.wait_for_peers(lambda: inventories.pop(0), clock=lambda: clock[0], sleep=sleep, say=said.append)
    assert macs == ["02:00:00:fa:1b:21"] and len(said) == 1 and clock[0] == 2 * lan_peers.POLL_SECONDS


def test_waiting_returns_at_once_without_a_cable_and_stops_at_the_deadline():
    uncabled = dict(HEAD, neighbors=[], functions=[dict(f, carrier=False) for f in HEAD["functions"]])
    assert lan_peers.wait_for_peers(lambda: uncabled, sleep=pytest.fail) == (uncabled, [])
    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds

    head, macs = lan_peers.wait_for_peers(lambda: dict(HEAD, neighbors=[]), clock=lambda: clock[0], sleep=sleep, say=lambda _: None)
    assert macs == [] and clock[0] >= lan_peers.WAIT_SECONDS


def test_prepare_signs_in_over_the_lan_and_runs_worker_preparation(monkeypatch):
    peer = {"hostname": "spark-a", "functions": [{"netdev": "p0", "mac": "02:00:00:fa:1b:21"}]}
    calls = []

    class Transport:
        def login(self, route):
            calls.append(("login", route[0]["address"], route[0]["interface"], route[0]["user"]))

        def inventory(self, route):
            return peer

    monkeypatch.setattr(lan_peers, "lan_hosts", lambda interface, macs, run: {"02:00:00:fa:1b:20": "192.0.2.232"})
    monkeypatch.setattr(lan_peers.packages, "transfer", lambda transport, route, archive, target: calls.append(("transfer", archive)))

    def root_command(transport, route, argv):
        calls.append(("root", route[0]["address"], argv[-3:]))

    prepared = lan_peers.prepare(Transport(), root_command, head=HEAD, user="operator", archive=lambda: "bundle.tar", say=lambda _: None)
    assert prepared == [peer]
    assert calls == [("login", "192.0.2.232", None, "operator"), ("transfer", "bundle.tar"),
                     ("root", "192.0.2.232", ["--apply", "--prepare", "--yes"])]
    peer["functions"][0]["mac"] = "02:00:00:fa:88:88"
    with pytest.raises(ValueError, match="not the Spark on the cable"):
        lan_peers.prepare(Transport(), root_command, head=HEAD, user="operator", archive=lambda: "bundle.tar", say=lambda _: None)
    assert lan_peers.prepare(Transport(), root_command, head=dict(HEAD, neighbors=[]), user="operator", archive=pytest.fail) == []


def test_a_lan_hop_is_a_private_ipv4_address_without_a_zone(tmp_path):
    from runtime.host import bootstrap
    hop = {"user": "operator", "address": "192.0.2.232", "interface": None, "port": 22}
    assert bootstrap.ssh_argv([hop], tmp_path)[-1] == "operator@192.0.2.232"
    for bad in (dict(hop, address="8.8.8.8"), dict(hop, port=2222)):
        with pytest.raises(ValueError, match="private IPv4"):
            bootstrap.validate_hop(bad)
    assert "192.0.2.232 on the LAN did not accept" in bootstrap.login_failure(hop, "Permission denied (password).")
