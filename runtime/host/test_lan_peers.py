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
    hosts = {"4c:bb:47:e9:0a:0f": "fe80::6c9c:52a:557b:fbbd", "4c:bb:47:e6:ed:fd": "fe80::2", "f0:68:e3:b3:13:ce": "fe80::3",
             "4c:bb:47:e9:0a:05": "fe80::4"}
    assert lan_peers.match(["4c:bb:47:e9:0a:10", "4c:bb:47:e9:0a:14"], hosts) == {
        "4c:bb:47:e9:0a:0f": ("fe80::6c9c:52a:557b:fbbd", "4c:bb:47:e9:0a:10")}
    assert lan_peers.match(["4c:bb:47:e9:0a:10"], {"4c:bb:47:e9:0a:10": "fe80::5"}) == {}


def test_lan_hosts_keeps_only_addresses_that_answered():
    def run(argv, **kwargs):
        if argv[0] == "ping":
            return subprocess.CompletedProcess(argv, 0, "64 bytes from fe80::1%enP7s7: icmp_seq=1\n", "")
        rows = [{"dst": "fe80::1", "lladdr": "4C:BB:47:E9:0A:0F"}, {"dst": "fe80::7", "lladdr": "00:11:22:33:44:55"}]
        return subprocess.CompletedProcess(argv, 0, json.dumps(rows), "")

    assert lan_peers.lan_hosts("enP7s7", run=run) == {"4c:bb:47:e9:0a:0f": "fe80::1"}


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

    monkeypatch.setattr(lan_peers, "lan_hosts", lambda interface, run: {"4c:bb:47:e9:0a:0f": "fe80::6c9c:52a:557b:fbbd"})
    monkeypatch.setattr(lan_peers.packages, "transfer", lambda transport, route, archive, target: calls.append(("transfer", archive)))

    def root_command(transport, route, argv):
        calls.append(("root", route[0]["address"], argv[-3:]))

    prepared = lan_peers.prepare(Transport(), root_command, head=HEAD, user="code", archive=lambda: "bundle.tar", say=lambda _: None)
    assert prepared == [peer]
    assert calls == [("login", "fe80::6c9c:52a:557b:fbbd", "enP7s7", "code"), ("transfer", "bundle.tar"),
                     ("root", "fe80::6c9c:52a:557b:fbbd", ["--apply", "--prepare", "--yes"])]
    peer["functions"][0]["mac"] = "4c:bb:47:e9:77:77"
    with pytest.raises(ValueError, match="not the Spark on the cable"):
        lan_peers.prepare(Transport(), root_command, head=HEAD, user="code", archive=lambda: "bundle.tar", say=lambda _: None)
    assert lan_peers.prepare(Transport(), root_command, head=dict(HEAD, neighbors=[]), user="code", archive=pytest.fail) == []
