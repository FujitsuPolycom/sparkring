"""Fabric bandwidth check against simulated Sparks; no test starts ib_write_bw or contacts a host.

The ``Sparks`` simulation answers the RoCE GID reads, the test server
program and the test client that ``fabric_bandwidth`` sends, records every
call in order and fails a test that starts a second test while one runs.
The client outputs are lines that ``ib_write_bw`` printed on a four-Spark ring.
"""
import ast
import ipaddress
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time

import pytest

from runtime.common import installer
from runtime.host import controller, fabric_bandwidth as bandwidth, placement
from runtime.host.test_fabric_ssh import cluster

HEADER = (" #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]\n")
RULE = "---------------------------------------------------------------------------------------\n"
HEALTHY_LINE = " 1048576    38096            0.00               213.05 \t\t   0.025397\n"
DEGRADED_LINE = " 1048576    39035            0.00               118.66 \t\t   0.014145\n"


def output(line):
    """An ib_write_bw client's standard output around one result line."""
    return (RULE + "                    RDMA_Write Bidirectional BW Test\n Link type       : Ethernet\n"
            " GID index       : 3\n" + RULE + HEADER + line + RULE)


HEALTHY = (0, output(HEALTHY_LINE), "")
DEGRADED = (0, output(DEGRADED_LINE), "")
REFUSED = (1, "", "Couldn't connect to 198.18.1.1:18620\n"
                  "Unable to open file descriptor for socket connection Unable to init the socket connection\n")


def arguments(source, name):
    """The arguments of the ``name(*(...))`` call that ends a program's source."""
    line = source.rstrip().splitlines()[-1]
    assert line.startswith(name + "(*") and line.endswith(")")
    return ast.literal_eval(line[len(name) + 2:-1])


def mapped(address):
    return ipaddress.IPv6Address("::ffff:" + str(address)).exploded


class Server:
    def __init__(self, sparks, port):
        self.sparks, self.port = sparks, port

    def announcement(self, timeout):
        return self.sparks.announce or {"port": self.port}

    def stop(self):
        if self.sparks.running is not None:
            self.sparks.events.append(("stop",) + self.sparks.running[:2])
            self.sparks.running = None
        return self.sparks.server_text


class Sparks:
    """Simulated Sparks of a setup plan: RoCE GID entries, test servers and test clients.

    ``clients`` maps (server rank, server RDMA device) to the client's
    ``(returncode, stdout, stderr)``; HEALTHY otherwise. ``gids`` replaces
    ``(rank, device)`` entries; ``unread`` maps ranks to the stderr of a GID
    read that fails.
    """

    def __init__(self, plan, *, clients=None, gids=None, unread=None, announce=None, server_text=""):
        self.hosts = plan["spec"]["hosts"]
        self.clients = clients or {}
        self.unread = unread or {}
        self.announce, self.server_text = announce, server_text
        self.events, self.running = [], None
        self.gids = {(host["rank"], port["rdma_device"]): [mapped(ipaddress.IPv4Interface(port["address"]).ip), "RoCE v2"]
                     for host in self.hosts for port in host["data_interfaces"]}
        self.gids.update(gids or {})

    def address(self, rank, device):
        port = next(p for p in self.hosts[rank]["data_interfaces"] if p["rdma_device"] == device)
        return str(ipaddress.IPv4Interface(port["address"]).ip)

    def run(self, rank, argv, *, timeout):
        if argv[:3] == ["python3", "-I", "-c"]:
            devices, index = arguments(argv[3], "read_gids")
            assert index == 3 and self.running is None
            self.events.append(("gids", rank))
            if rank in self.unread:
                return 255, "", self.unread[rank]
            return 0, json.dumps({device: self.gids.get((rank, device), [None, None]) for device in devices}) + "\n", ""
        assert self.running is not None, "a test client ran without a test server"
        server_rank, server_device, port = self.running
        device = argv[argv.index("-d") + 1]
        assert argv[:4] == ["timeout", "-k", "5", str(bandwidth.CLIENT_SECONDS)]
        assert argv[4:] == [*bandwidth.ib_write_bw_argv(device), "-p", str(port), self.address(server_rank, server_device)]
        assert timeout >= bandwidth.CLIENT_SECONDS
        self.events.append(("client", rank, device))
        return self.clients.get((server_rank, server_device), HEALTHY)

    def start(self, rank, argv):
        assert self.running is None, "two tests ran at once"
        assert argv[:3] == ["python3", "-I", "-c"]
        test, ports, ready, limit = arguments(argv[3], "serve")
        device = test[test.index("-d") + 1]
        assert test == bandwidth.ib_write_bw_argv(device) and ports == list(bandwidth.PORTS)
        assert (ready, limit) == (bandwidth.READY_SECONDS, bandwidth.SERVER_SECONDS)
        self.running = (rank, device, ports[0])
        self.events.append(("server", rank, device))
        return Server(self, ports[0])


def checked(size, **options):
    value = cluster(size)
    sparks = Sparks(value["plan"], **{k: v for k, v in options.items() if k != "busy"})
    return value, sparks, bandwidth.check(value, access=sparks, busy=options.get("busy"), now=lambda: 1000.0)


# Parsing and the threshold.

def test_parses_the_bw_average_column_of_healthy_and_degraded_runs():
    assert bandwidth.parse_bandwidth(output(HEALTHY_LINE)) == 213.05
    assert bandwidth.parse_bandwidth(output(DEGRADED_LINE)) == 118.66
    assert bandwidth.verdict(213.05) == "healthy" and bandwidth.verdict(118.66) == "degraded"


def test_output_without_a_result_line_or_in_other_units_has_no_bandwidth():
    assert bandwidth.parse_bandwidth(REFUSED[1] + REFUSED[2]) is None
    assert bandwidth.parse_bandwidth("") is None and bandwidth.parse_bandwidth(None) is None
    # Without --report_gbits the columns are MiB/s; such a line is not read as Gb/s.
    assert bandwidth.parse_bandwidth(output(HEALTHY_LINE).replace("Gb/sec", "MiB/sec")) is None
    # Another message size is another test.
    assert bandwidth.parse_bandwidth(output(HEALTHY_LINE.replace("1048576", "65536"))) is None


@pytest.mark.parametrize("gbps, verdict", [(190.0, "healthy"), (190.01, "healthy"), (189.99, "degraded"),
                                           (0.0, "degraded"), (213.05, "healthy")])
def test_threshold_edges(gbps, verdict):
    assert bandwidth.HEALTHY_GBPS == 190.0
    assert bandwidth.verdict(gbps) == verdict
    line = f" 1048576    38096            0.00               {gbps:.2f} \t\t   0.025397\n"
    assert bandwidth.verdict(bandwidth.parse_bandwidth(output(line))) == verdict


# Cables and sequencing.

def test_a_ring_is_measured_cable_by_cable_and_function_by_function():
    _, sparks, document = checked(4)
    reads = [event for event in sparks.events if event[0] == "gids"]
    # Every RoCE GID entry is read once per Spark before any test starts.
    assert sparks.events[:4] == reads == [("gids", rank) for rank in range(4)]
    tests = sparks.events[4:]
    expected = []
    for rank in range(4):
        for device, peer in (("rocep1s0f0", "rocep1s0f1"), ("roceP2p1s0f0", "roceP2p1s0f1")):
            expected += [("server", rank, device), ("client", (rank + 1) % 4, peer), ("stop", rank, device)]
    assert tests == expected
    assert [bandwidth.cable_text(cable) for cable in document["cables"]] == [
        "spark0 port 0 ↔ spark1 port 1", "spark1 port 0 ↔ spark2 port 1",
        "spark2 port 0 ↔ spark3 port 1", "spark3 port 0 ↔ spark0 port 1"]
    assert document["verdict"] == "healthy" and document["layout"] == "ring"
    assert all(function["gbps"] == 213.05 for cable in document["cables"] for function in cable["functions"])


def test_a_pair_measures_both_functions_of_the_cable_between_its_ports_0():
    _, sparks, document = checked(2)
    assert [event for event in sparks.events if event[0] != "gids"] == [
        ("server", 0, "rocep1s0f0"), ("client", 1, "rocep1s0f0"), ("stop", 0, "rocep1s0f0"),
        ("server", 0, "roceP2p1s0f0"), ("client", 1, "roceP2p1s0f0"), ("stop", 0, "roceP2p1s0f0")]
    [cable] = document["cables"]
    assert cable["ends"] == [{"rank": 0, "hostname": "spark0", "port": 0}, {"rank": 1, "hostname": "spark1", "port": 0}]
    assert document["layout"] == "pair" and document["notes"] == []


def test_a_pair_notes_the_cable_between_its_ports_1_that_it_cannot_measure():
    from runtime.host import topology
    value = cluster(2)
    nodes = value["plan"]["nodes"]
    # Add LLDP of a second cable between the two ports 1, seen from both ends.
    for rank, current in enumerate(nodes):
        peer = nodes[1 - rank]
        for role in ("ccw_primary", "ccw_secondary"):
            port, other = topology.endpoints(current)[role], topology.endpoints(peer)[role]
            current["lldp"]["lldp"]["interface"].append({port["netdev"]: {
                "chassis": {peer["hostname"]: {"id": {"type": "mac", "value": other["mac"]}}},
                "port": {"id": {"type": "mac", "value": other["mac"]}}}})
    document = bandwidth.check(value, access=Sparks(value["plan"]))
    assert len(document["cables"]) == 1
    assert document["notes"] == ["The cable between the ports 1 is not measured: pair models do not use it, and "
                                 "SparkRing gives it no fabric addresses."]
    assert bandwidth.lines(document)[-1] == "Note: " + document["notes"][0]


def test_a_degraded_function_marks_its_cable_with_the_repair_steps():
    _, sparks, document = checked(4, clients={(1, "rocep1s0f0"): DEGRADED, (1, "roceP2p1s0f0"): DEGRADED})
    cable = document["cables"][1]
    assert [function["verdict"] for function in cable["functions"]] == ["degraded", "degraded"]
    assert cable["verdict"] == "degraded" and document["verdict"] == "degraded"
    assert cable["repair"] == [
        "To repair: reboot both spark1 and spark2, then run sudo sparkring cabling --bandwidth again.",
        "Restarting the link or the network driver does not clear this.",
        "If the cable is still degraded after the reboot, reseat it at both ends."]
    text = bandwidth.lines(document)
    at = text.index("spark1 port 0 ↔ spark2 port 1: degraded")
    assert text[at + 1:at + 6] == ["  enp1s0f0np0 ↔ enp1s0f1np1      118.66 Gb/s  degraded",
                                   "  enP2p1s0f0np0 ↔ enP2p1s0f1np1  118.66 Gb/s  degraded",
                                   *("  " + line for line in cable["repair"])]
    assert text[0] == "Fabric bandwidth, both directions at once (190 Gb/s or more per link is healthy):"
    assert text[1:4] == ["spark0 port 0 ↔ spark1 port 1: healthy",
                         "  enp1s0f0np0 ↔ enp1s0f1np1      213.05 Gb/s  healthy",
                         "  enP2p1s0f0np0 ↔ enP2p1s0f1np1  213.05 Gb/s  healthy"]


def test_a_test_that_cannot_run_fails_with_its_reason_and_the_next_function_still_runs():
    missing = (127, "", "timeout: failed to run command 'ib_write_bw': No such file or directory\n")
    timed_out = (124, "", "")
    _, sparks, document = checked(4, clients={(0, "rocep1s0f0"): REFUSED, (0, "roceP2p1s0f0"): missing,
                                              (2, "rocep1s0f0"): timed_out, (3, "rocep1s0f0"): (0, "garbage", "")})
    first = document["cables"][0]["functions"]
    assert first[0] == {**first[0], "gbps": None, "verdict": "failed",
                        "reason": "spark1: Unable to open file descriptor for socket connection Unable to init the "
                                  "socket connection"}
    assert first[1]["reason"] == "ib_write_bw is not installed on spark1 (Debian package perftest)"
    assert document["cables"][2]["functions"][0]["reason"] == "the test on spark3 did not finish within 30 seconds"
    assert document["cables"][3]["functions"][0]["reason"] == "the test on spark0 printed no bandwidth result"
    assert [cable["verdict"] for cable in document["cables"]] == ["failed", "healthy", "failed", "failed"]
    assert document["verdict"] == "failed"
    # Every server was stopped, also after a failed client.
    assert sum(event[0] == "server" for event in sparks.events) == sum(event[0] == "stop" for event in sparks.events) == 8


def test_a_server_that_does_not_start_fails_the_function_without_a_client():
    value = cluster(2)
    for announce, text, reason in [
            ({"error": "the test server stopped before it listened", "returncode": 127,
              "output": ["timeout: failed to run command 'ib_write_bw': No such file or directory"]}, "",
             "ib_write_bw is not installed on spark0 (Debian package perftest)"),
            ({"error": "TCP ports 18620-18639 are all in use", "returncode": None, "output": []}, "",
             "spark0: TCP ports 18620-18639 are all in use")]:
        sparks = Sparks(value["plan"], announce=announce, server_text=text)
        document = bandwidth.check(value, access=sparks)
        assert {function["reason"] for function in document["cables"][0]["functions"]} == {reason}
        assert not any(event[0] == "client" for event in sparks.events)

    class Silent(Sparks):
        def start(self, rank, argv):
            server = super().start(rank, argv)
            server.announcement = lambda timeout: None
            return server
    sparks = Silent(value["plan"], server_text="sudo: a password is required")
    document = bandwidth.check(value, access=sparks)
    assert document["cables"][0]["functions"][0]["reason"] == "spark0: sudo: a password is required"
    assert sparks.running is None


# RoCE GID entries.

def test_a_function_whose_gid_entry_lacks_its_address_is_reported_instead_of_tested():
    _, sparks, document = checked(4, gids={(2, "rocep1s0f1"): [mapped("198.18.9.9"), "RoCE v2"],
                                           (3, "roceP2p1s0f1"): ["0000:0000:0000:0000:0000:0000:0000:0000", "RoCE v2"],
                                           (0, "roceP2p1s0f1"): [mapped("198.18.104.2"), "RoCE v1"],
                                           (3, "rocep1s0f0"): [None, None]})
    first, second = document["cables"][1]["functions"]
    assert first["verdict"] == "failed" and first["reason"] == (
        "RoCE GID 3 of enp1s0f1np1 on spark2 holds 198.18.9.9; the test needs its fabric address 198.18.2.2 there")
    assert second["verdict"] == "healthy"
    assert document["cables"][2]["functions"][1]["reason"] == (
        "RoCE GID 3 of enP2p1s0f1np1 on spark3 holds no IPv4 address; the test needs its fabric address "
        "198.18.103.2 there")
    assert document["cables"][3]["functions"][1]["reason"] == (
        "RoCE GID 3 of enP2p1s0f1np1 on spark0 is a RoCE v1 entry; the test needs RoCE v2")
    assert document["cables"][3]["functions"][0]["reason"] == "RoCE GID 3 of enp1s0f0np0 on spark3 could not be read"
    # No server starts for a function whose entries are wrong.
    started = [event[1:] for event in sparks.events if event[0] == "server"]
    assert (1, "rocep1s0f0") not in started and (2, "roceP2p1s0f0") not in started
    assert (3, "rocep1s0f0") not in started and (3, "roceP2p1s0f0") not in started
    assert len(started) == 4


def test_a_spark_whose_gid_entries_cannot_be_read_fails_its_functions():
    _, sparks, document = checked(2, unread={1: "ssh: connect to host 192.0.2.11 port 22: Connection timed out"})
    reasons = {function["reason"] for function in document["cables"][0]["functions"]}
    assert reasons == {"spark1 could not be read: ssh: connect to host 192.0.2.11 port 22: Connection timed out"}
    assert not any(event[0] == "server" for event in sparks.events)


# Serving models.

def deployment(root, slot, profile="qwen38-flash-next-tp2", *, stopped=False):
    """A recorded active deployment on ``slot`` whose last operation is an up, or a completed down."""
    directory = root / "deployments" / (profile + ("" if slot is None else f"-on-{slot[0]}-{slot[1]}"))
    installer.write(directory / "deployment.lock.json", {"selection": {"profile": profile},
                                                         "site": {"placement": list(slot) if slot else None}})
    installer.write(directory / "state.json", {"operation": "down" if stopped else "up", "complete": True})
    placement.record(root, slot, directory)
    return directory


def test_serving_names_the_sparks_of_each_running_model(tmp_path):
    assert bandwidth.serving(tmp_path, 4) == {}
    deployment(tmp_path, (0, 1))
    deployment(tmp_path, (2, 3), "glm53-flash-nvfp4-spark-tp2", stopped=True)
    assert bandwidth.serving(tmp_path, 4) == {rank: {"profile": "qwen38-flash-next-tp2", "placement": (0, 1)}
                                              for rank in (0, 1)}
    whole = tmp_path / "whole"
    deployment(whole, None, "glm53-flash-nvfp4-spark-tp2")
    assert set(bandwidth.serving(whole, 2)) == {0, 1}


def test_cables_of_a_serving_half_are_skipped_and_the_others_measured(tmp_path):
    deployment(tmp_path, (0, 1))
    busy = bandwidth.serving(tmp_path, 4)
    _, sparks, document = checked(4, busy=busy)
    assert [cable["verdict"] for cable in document["cables"]] == ["skipped", "skipped", "healthy", "skipped"]
    assert document["cables"][0]["reason"] == ("qwen38-flash-next-tp2 serves on spark0 and spark1, and the test "
                                               "would slow it")
    assert {event[1] for event in sparks.events if event[0] in ("server", "client")} == {2, 3}
    assert document["verdict"] == "healthy"
    assert ("spark1 port 0 ↔ spark2 port 1: not measured; qwen38-flash-next-tp2 serves on spark1, and the test "
            "would slow it (--while-serving measures it anyway)") in bandwidth.lines(document)


def test_while_serving_measures_every_cable_and_notes_the_model(tmp_path):
    deployment(tmp_path, (0, 1))
    value = cluster(4)
    sparks = Sparks(value["plan"])
    document = bandwidth.check(value, access=sparks, busy=bandwidth.serving(tmp_path, 4), allow_serving=True)
    assert [cable["verdict"] for cable in document["cables"]] == ["healthy"] * 4
    assert document["cables"][0]["notes"] == ["Measured while qwen38-flash-next-tp2 serves on spark0 and spark1; "
                                              "model traffic can lower the result."]
    assert document["cables"][2]["notes"] == []


def recorded(tmp_path, size):
    value = cluster(size)
    installer.write(tmp_path / "cluster.json", value)
    return value


def test_command_refuses_when_a_model_serves_on_every_cable_and_saves_nothing(tmp_path):
    value = recorded(tmp_path, 2)
    deployment(tmp_path, None)
    sparks = Sparks(value["plan"])
    with pytest.raises(ValueError) as refused:
        bandwidth.command(state_root=tmp_path, access=sparks)
    assert str(refused.value) == (
        "Every fabric cable is in use: qwen38-flash-next-tp2 serves on every Spark. The bandwidth test fills each "
        "cable for several seconds per link and would slow the model. Stop it first (sudo sparkring down "
        "--execute), or add --while-serving to measure anyway.")
    assert sparks.events == [] and not (tmp_path / bandwidth.RECORD).exists()
    assert bandwidth.command(state_root=tmp_path, access=sparks, allow_serving=True) == 0
    assert (tmp_path / bandwidth.RECORD).exists()


def test_refusal_names_each_half_and_its_stop_command(tmp_path):
    deployment(tmp_path, (0, 1))
    deployment(tmp_path, (2, 3), "glm53-flash-nvfp4-spark-tp2")
    assert bandwidth.refusal(bandwidth.serving(tmp_path, 4)) == (
        "Every fabric cable is in use: qwen38-flash-next-tp2 serves on Sparks 0 and 1; glm53-flash-nvfp4-spark-tp2 "
        "serves on Sparks 2 and 3. The bandwidth test fills each cable for several seconds per link and would slow "
        "the models. Stop them first (sudo sparkring down --on 0,1 --execute and sudo sparkring down --on 2,3 "
        "--execute), or add --while-serving to measure anyway.")


def test_command_takes_the_installation_lock(tmp_path):
    from runtime.common import process_lock
    value = recorded(tmp_path, 2)
    with process_lock.hold(tmp_path / "install.lock"):
        with pytest.raises(ValueError, match="Another operation is active"):
            bandwidth.command(state_root=tmp_path, access=Sparks(value["plan"]))


def test_command_without_a_recorded_cluster_explains_where_it_runs(tmp_path):
    with pytest.raises(ValueError, match="No pair or ring is recorded on this Spark"):
        bandwidth.command(state_root=tmp_path, access=None)


# The JSON document and the saved result.

def test_json_output_is_one_document_that_is_also_saved(tmp_path, capsys):
    value = recorded(tmp_path, 4)
    sparks = Sparks(value["plan"], clients={(3, "rocep1s0f0"): DEGRADED})
    assert bandwidth.command(json_output=True, state_root=tmp_path, access=sparks) == 1
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert captured.err.splitlines()[0] == "Measuring spark0 port 0 ↔ spark1 port 1"
    assert document == json.loads((tmp_path / bandwidth.RECORD).read_text())
    assert set(document) == {"schema", "measured_at", "cluster_id", "layout", "threshold_gbps", "test", "verdict",
                             "notes", "cables"}
    assert document["schema"] == "sparkring-fabric-bandwidth/v1" and document["cluster_id"] == value["plan"]["id"]
    assert document["test"] == {"program": "ib_write_bw", "bidirectional": True, "message_bytes": 1048576,
                                "seconds": 5, "gid_index": 3}
    assert document["threshold_gbps"] == 190.0 and document["verdict"] == "degraded"
    cable = document["cables"][3]
    assert set(cable) == {"ends", "functions", "verdict", "reason", "notes", "repair"}
    assert set(cable["functions"][0]) == {"function", "server", "client", "gbps", "verdict", "reason"}
    assert cable["functions"][0]["server"] == {"rank": 3, "hostname": "spark3", "netdev": "enp1s0f0np0",
                                               "rdma_device": "rocep1s0f0", "address": "198.18.4.1"}
    assert cable["functions"][0]["client"]["rank"] == 0
    assert [function["gbps"] for function in cable["functions"]] == [118.66, 213.05]
    assert cable["verdict"] == "degraded" and cable["repair"][0].startswith("To repair: reboot both spark3 and spark0")


def test_terminal_output_ends_with_the_table_and_exits_0_when_every_cable_is_healthy(tmp_path, capsys):
    value = recorded(tmp_path, 2)
    assert bandwidth.command(state_root=tmp_path, access=Sparks(value["plan"])) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Measuring spark0 port 0 ↔ spark1 port 0",
        "Fabric bandwidth, both directions at once (190 Gb/s or more per link is healthy):",
        "spark0 port 0 ↔ spark1 port 0: healthy",
        "  enp1s0f0np0 ↔ enp1s0f0np0      213.05 Gb/s  healthy",
        "  enP2p1s0f0np0 ↔ enP2p1s0f0np0  213.05 Gb/s  healthy"]


def test_cabling_bandwidth_flag_runs_the_check(monkeypatch, capsys):
    import os
    from runtime.host import cabling
    calls = []
    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(bandwidth, "command", lambda **options: calls.append(options) or 1)
    assert cabling.main(["--bandwidth", "--json", "--while-serving"]) == 1
    assert calls == [{"json_output": True, "allow_serving": True}]
    with pytest.raises(SystemExit):
        cabling.main(["--while-serving"])
    assert "--while-serving applies to --bandwidth" in capsys.readouterr().err


# Status.

def saved(tmp_path, document):
    bandwidth.save(tmp_path, document)
    return bandwidth.summary(tmp_path, now=lambda: document["measured_at"] + 3 * 3600)


def test_status_says_never_measured_without_a_saved_result(tmp_path):
    value = bandwidth.summary(tmp_path)
    assert value == {"state": "never-measured"}
    assert bandwidth.status_lines(value) == ["Fabric bandwidth: never measured; sudo sparkring cabling --bandwidth "
                                             "measures it"]


def test_status_shows_a_healthy_result_with_its_age(tmp_path):
    value_cluster, _, document = checked(4)
    value = saved(tmp_path, document)
    assert value["state"] == "measured" and value["age_seconds"] == 3 * 3600
    assert bandwidth.status_lines(value, value_cluster["plan"]["id"]) == [
        "Fabric bandwidth: healthy on all 4 cables, measured 3 h ago"]
    _, _, pair = checked(2)
    assert bandwidth.status_lines(saved(tmp_path, pair)) == ["Fabric bandwidth: healthy, measured 3 h ago"]


def test_status_highlights_a_degraded_cable_with_its_repair_lines(tmp_path):
    _, _, document = checked(4, clients={(1, "rocep1s0f0"): DEGRADED, (1, "roceP2p1s0f0"): DEGRADED})
    assert bandwidth.status_lines(saved(tmp_path, document)) == [
        "Fabric bandwidth: 1 of 4 cables degraded, measured 3 h ago",
        "  spark1 port 0 ↔ spark2 port 1: degraded, 118.66 and 118.66 Gb/s (healthy is 190 or more)",
        "    To repair: reboot both spark1 and spark2, then run sudo sparkring cabling --bandwidth again.",
        "    Restarting the link or the network driver does not clear this.",
        "    If the cable is still degraded after the reboot, reseat it at both ends."]


def test_status_names_cables_that_failed_or_were_skipped(tmp_path):
    _, _, document = checked(4, clients={(0, "rocep1s0f0"): REFUSED})
    assert bandwidth.status_lines(saved(tmp_path, document)) == [
        "Fabric bandwidth: 1 of 4 cables not measured, measured 3 h ago",
        "  spark0 port 0 ↔ spark1 port 1: not measured: spark1: Unable to open file descriptor for socket connection "
        "Unable to init the socket connection"]
    root = tmp_path / "serving"
    deployment(root, (2, 3))
    _, _, document = checked(4, busy=bandwidth.serving(root, 4))
    assert bandwidth.status_lines(saved(tmp_path, document)) == [
        "Fabric bandwidth: healthy on 1 of 4 cables, measured 3 h ago",
        "  spark1 port 0 ↔ spark2 port 1: not measured; qwen38-flash-next-tp2 serves on spark2, and the test would "
        "slow it",
        "  spark2 port 0 ↔ spark3 port 1: not measured; qwen38-flash-next-tp2 serves on spark2 and spark3, and the "
        "test would slow it",
        "  spark3 port 0 ↔ spark0 port 1: not measured; qwen38-flash-next-tp2 serves on spark3, and the test would "
        "slow it"]


def test_status_treats_a_result_of_another_setup_or_an_unreadable_one_as_unmeasured(tmp_path):
    _, _, document = checked(2)
    value = saved(tmp_path, document)
    assert bandwidth.status_lines(value, "f" * 64) == [
        "Fabric bandwidth: never measured on this setup (the saved result is from another setup of the Sparks); "
        "sudo sparkring cabling --bandwidth measures it"]
    (tmp_path / bandwidth.RECORD).write_text("{not json")
    value = bandwidth.summary(tmp_path)
    assert value["state"] == "unreadable"
    assert bandwidth.status_lines(value)[0].startswith("Fabric bandwidth: the saved result cannot be read (")


def test_sparkring_status_prints_the_saved_result_and_never_measures(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(controller, "STATE", tmp_path)
    value = recorded(tmp_path, 4)
    monkeypatch.setattr(controller.node, "status", lambda: {"state": "network-configured", "next_action": "sparkring models"})
    monkeypatch.setattr(controller.discovery, "ssh", lambda host, argv: json.dumps(
        {"state": "network-configured", "hostname": "spark"}))
    measure = bandwidth.check
    monkeypatch.setattr(bandwidth, "check", lambda *args, **kwargs: pytest.fail("status measured the fabric"))
    monkeypatch.setattr(bandwidth, "Access", lambda *args, **kwargs: pytest.fail("status contacted the fabric"))
    assert controller.lifecycle(["status"]) == 0
    assert ("Fabric bandwidth: never measured; sudo sparkring cabling --bandwidth measures it"
            in capsys.readouterr().out.splitlines())
    assert controller.lifecycle(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["fabric_bandwidth"] == {"state": "never-measured"}
    sparks = Sparks(value["plan"], clients={(2, "rocep1s0f0"): DEGRADED})
    bandwidth.save(tmp_path, measure(value, access=sparks, now=lambda: time.time() - 120))
    assert controller.lifecycle(["status"]) == 0
    lines = capsys.readouterr().out.splitlines()
    at = lines.index("Fabric bandwidth: 1 of 4 cables degraded, measured 2 min ago")
    assert lines[at + 1] == "  spark2 port 0 ↔ spark3 port 1: degraded, 118.66 and 213.05 Gb/s (healthy is 190 or more)"
    assert lines[at + 2].startswith("    To repair: reboot both spark2 and spark3")
    assert controller.lifecycle(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["fabric_bandwidth"]["verdict"] == "degraded"


# Setup.

def test_setup_warns_about_a_degraded_cable_and_saves_the_result(tmp_path):
    value = cluster(4)
    sparks = Sparks(value["plan"], clients={(1, "rocep1s0f0"): DEGRADED, (1, "roceP2p1s0f0"): DEGRADED})
    said = []
    document = bandwidth.after_setup(tmp_path, value, access=sparks, say=said.append)
    assert document["verdict"] == "degraded"
    assert json.loads((tmp_path / bandwidth.RECORD).read_text()) == document
    assert said[0] == "Measure the bandwidth of each fabric cable (about 30 seconds per cable)"
    # The table leaves the repair steps to the warning that follows it.
    at = said.index("WARNING: the fabric cable spark1 port 0 ↔ spark2 port 1 is degraded (118.66 and 118.66 Gb/s; "
                    "healthy is 190 or more).")
    assert said[at:] == ["WARNING: the fabric cable spark1 port 0 ↔ spark2 port 1 is degraded (118.66 and 118.66 Gb/s; "
                         "healthy is 190 or more).",
                         "  Models run, but prompt processing over this cable is slower.",
                         "  To repair: reboot both spark1 and spark2, then run sudo sparkring cabling --bandwidth again.",
                         "  Restarting the link or the network driver does not clear this.",
                         "  If the cable is still degraded after the reboot, reseat it at both ends."]
    assert sum(line.startswith("  To repair:") for line in said) == 1


def test_setup_check_that_cannot_run_is_a_warning(tmp_path, monkeypatch):
    value = cluster(2)
    said = []
    monkeypatch.setattr(bandwidth, "save", lambda root, document: (_ for _ in ()).throw(OSError("No space left")))
    assert bandwidth.after_setup(tmp_path, value, access=Sparks(value["plan"]), say=said.append) is None
    assert said[-1] == ("Warning: the fabric bandwidth check could not run (No space left); sudo sparkring cabling "
                        "--bandwidth runs it again.")


def test_setup_measures_nothing_while_a_model_serves(tmp_path):
    value = cluster(4)
    deployment(tmp_path, (2, 3))
    sparks = Sparks(value["plan"])
    said = []
    assert bandwidth.after_setup(tmp_path, value, access=sparks, say=said.append) is None
    assert sparks.events == [] and not (tmp_path / bandwidth.RECORD).exists()
    assert said == ["Fabric bandwidth: not measured, because a model is serving; sudo sparkring cabling --bandwidth "
                    "measures it after the model stops."]


# The programs that run on the Sparks and the processes on Node A.

def test_access_runs_node_a_commands_locally_and_worker_commands_over_ssh_as_root(monkeypatch):
    access = bandwidth.Access(cluster(2)["plan"])
    monkeypatch.setattr(bandwidth.os, "geteuid", lambda: 0, raising=False)
    assert access.argv(0, ["cat", "x"]) == ["cat", "x"]
    monkeypatch.setattr(bandwidth.os, "geteuid", lambda: 1000, raising=False)
    assert access.argv(0, ["cat", "x"]) == ["sudo", "-n", "cat", "x"]
    assert access.argv(1, ["cat", "a b"]) == ["ssh", *bandwidth.SSH_OPTIONS, "root@192.0.2.11", "sudo -n cat 'a b'"]


def test_server_reads_the_announcement_and_stops_the_program_by_closing_its_input():
    program = ("import json, sys\nprint('Warning: some note', flush=True)\nprint(json.dumps({'port': 18620}), flush=True)\n"
               "sys.stdin.read()\nprint(json.dumps({'returncode': -15, 'output': []}), flush=True)\n")
    # Access.start opens the program's pipes the same way; it would add sudo on Node A.
    server = bandwidth.Server(subprocess.Popen([sys.executable, "-c", program], stdin=subprocess.PIPE,
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    assert server.announcement(10) == {"port": 18620}
    assert server.stop() == "Warning: some note"
    assert server.process.returncode == 0
    assert server.stop() == "Warning: some note"


def test_programs_sent_to_the_sparks_are_self_contained():
    source = bandwidth.server_program("rocep1s0f0")[3]
    compile(source, "serve", "exec")
    assert arguments(source, "serve") == (bandwidth.ib_write_bw_argv("rocep1s0f0"), list(bandwidth.PORTS), 20, 60)
    assert bandwidth.ib_write_bw_argv("rocep1s0f0") == ["ib_write_bw", "-b", "-d", "rocep1s0f0", "-x", "3", "-s", "1048576",
                                                 "-D", "5", "-F", "--report_gbits"]
    assert bandwidth.client_argv("rocep1s0f1", "198.18.1.1", 18621)[-3:] == ["-p", "18621", "198.18.1.1"]
    source = bandwidth.gid_program(["rocep1s0f0"])[3]
    compile(source, "gids", "exec")
    assert arguments(source, "read_gids") == (["rocep1s0f0"], 3)


LINUX = Path("/proc/net/tcp").exists() and shutil.which("timeout") is not None
# A stand-in for ib_write_bw: listens on the port after -p, prints a result line and exits.
FAKE_SERVER = """
import socket, sys, time
port = int(sys.argv[sys.argv.index("-p") + 1])
listener = socket.create_server(("127.0.0.1", port))
time.sleep(float(sys.argv[1]))
print(" 1048576    38096            0.00               213.05 \\t\\t   0.025397")
"""


def free_ports(count):
    sockets = [socket.create_server(("127.0.0.1", 0)) for _ in range(count)]
    ports = [s.getsockname()[1] for s in sockets]
    for s in sockets:
        s.close()
    return ports


@pytest.mark.skipif(not LINUX, reason="serve reads /proc/net/tcp and runs timeout(1)")
def test_serve_announces_the_port_once_the_server_listens_and_skips_ports_in_use(capsys):
    taken, free = free_ports(2)
    with socket.create_server(("127.0.0.1", taken)):
        bandwidth.serve([sys.executable, "-c", FAKE_SERVER, "0.5"], [taken, free], 10, 30, stop=threading.Event().wait)
    first, last = map(json.loads, capsys.readouterr().out.splitlines())
    assert first == {"port": free}
    assert last["returncode"] == 0 and last["output"][-1].split()[3] == "213.05"


@pytest.mark.skipif(not LINUX, reason="serve reads /proc/net/tcp and runs timeout(1)")
def test_serve_reports_a_server_that_stops_before_it_listens_and_one_it_was_told_to_stop(capsys):
    [port] = free_ports(1)
    bandwidth.serve(["false"], [port], 10, 30, stop=threading.Event().wait)
    [line] = map(json.loads, capsys.readouterr().out.splitlines())
    assert line == {"error": "the test server stopped before it listened", "returncode": 1, "output": []}
    started = time.monotonic()
    bandwidth.serve([sys.executable, "-c", FAKE_SERVER, "60"], [port], 10, 30, stop=lambda: time.sleep(1))
    first, last = map(json.loads, capsys.readouterr().out.splitlines())
    assert first == {"port": port} and last["returncode"] != 0
    assert time.monotonic() - started < 15
