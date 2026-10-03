"""The API Spark's address and port report, the endpoint check and the terminal question."""
import json
import pathlib
import subprocess
from types import SimpleNamespace

import pytest

from runtime.host import api_endpoint
from runtime.host.install_errors import NeedsInput
from runtime.host.test_fabric_ssh import cluster

# `ip -j -4 address show` and `ss -H -ltnp` of a Node A that serves a model on port 8015, as on DGX4-2,
# with documentation addresses in place of its LAN addresses.
ADDRESSES = [
    {"ifname": "lo", "operstate": "UNKNOWN", "addr_info": [{"family": "inet", "local": "127.0.0.1", "prefixlen": 8}]},
    {"ifname": "enP7s7", "operstate": "UP", "addr_info": [{"family": "inet", "local": "198.51.100.10", "prefixlen": 24}]},
    {"ifname": "wlP9s9", "operstate": "UP", "addr_info": [{"family": "inet", "local": "198.51.100.11", "prefixlen": 24}]},
    {"ifname": "enp1s0f0np0", "operstate": "UP", "addr_info": [{"family": "inet", "local": "198.18.0.1", "prefixlen": 24}]},
    {"ifname": "sr-control", "operstate": "UNKNOWN",
     "addr_info": [{"family": "inet", "local": "10.253.255.1", "prefixlen": 32}]},
    {"ifname": "docker0", "operstate": "DOWN", "addr_info": [{"family": "inet", "local": "198.51.100.254", "prefixlen": 16}]},
]
LISTENERS = """\
LISTEN 0      4096    10.253.255.1:47835 0.0.0.0:* users:(("VLLM::Worker_TP",pid=3259984,fd=169))
LISTEN 0      4096       127.0.0.1:11000 0.0.0.0:* users:(("dashboard-servi",pid=1531,fd=6))
LISTEN 0      4096   127.0.0.53%lo:53    0.0.0.0:* users:(("systemd-resolve",pid=1691,fd=15))
LISTEN 0      4096         0.0.0.0:22    0.0.0.0:* users:(("sshd",pid=2558,fd=3),("systemd",pid=1,fd=328))
LISTEN 0      128     10.253.255.1:2222  0.0.0.0:* users:(("sshd",pid=4842,fd=3))
LISTEN 0      2048         0.0.0.0:8015  0.0.0.0:* users:(("python3",pid=3255358,fd=41))
LISTEN 0      4096           [::1]:631      [::]:* users:(("cupsd",pid=513916,fd=6))
LISTEN 0      4096               *:29778       *:* users:(("VLLM::Worker_TP",pid=3259984,fd=50))
"""
CONTAINER = "4d" * 32


@pytest.fixture
def probed(monkeypatch):
    """``_probe()`` on that Node A, whose vLLM processes run in the model container of deployment ``f...f``."""
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["ip", "-j"]:
            return SimpleNamespace(stdout=json.dumps(ADDRESSES), returncode=0)
        if argv[0] == "ss":
            return SimpleNamespace(stdout=LISTENERS, returncode=0)
        if "inspect" in argv:
            assert argv[-1:] == [CONTAINER]
            return SimpleNamespace(stdout=json.dumps([{"Id": CONTAINER, "Name": "/sr-ring-r0",
                                                       "Config": {"Labels": {"io.sparkring.deployment": "f" * 64}}}]),
                                   returncode=0)
        raise AssertionError(argv)
    real = pathlib.Path.read_text

    def read_text(path, *args, **kwargs):
        text = path.as_posix()
        if text.startswith("/proc/"):
            pid = text.split("/")[2]
            if pid in ("3259984", "3255358"):
                return f"0::/system.slice/docker-{CONTAINER}.scope\n"
            return "0::/system.slice/ssh.service\n"
        if text == "/etc/sparkring/control.json":
            return json.dumps({"schema": "sparkring-control/v1", "subnet": "10.253.255.0/29"})
        return real(path, *args, **kwargs)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(pathlib.Path, "read_text", read_text)
    return api_endpoint._probe(), calls


def test_the_report_names_each_listener_and_the_container_that_holds_it(probed):
    document, calls = probed
    assert calls[:2] == [["ip", "-j", "-4", "address", "show"], ["ss", "-H", "-ltnp"]]
    assert document["control_subnet"] == "10.253.255.0/29"
    assert {(row["interface"], row["address"], row["state"]) for row in document["addresses"]} >= {
        ("enP7s7", "198.51.100.10", "UP"), ("docker0", "198.51.100.254", "DOWN")}
    by_port = {row["port"]: row for row in document["listeners"]}
    assert by_port[8015]["address"] == "0.0.0.0" and by_port[8015]["processes"] == [
        {"name": "python3", "pid": 3255358,
         "container": {"id": CONTAINER, "name": "sr-ring-r0", "deployment": "f" * 64}}]
    assert by_port[53]["address"] == "127.0.0.53" and by_port[631]["address"] == "::1"
    assert by_port[29778]["address"] == "*" and by_port[22]["processes"][1] == {"name": "systemd", "pid": 1,
                                                                                 "container": None}


def test_ports_held_on_every_address_or_on_the_listen_address_block_the_api(probed):
    document, _ = probed
    assert [row["port"] for row in api_endpoint.holders(document, "0.0.0.0", 11000)] == [11000]
    assert api_endpoint.holders(document, "198.51.100.10", 11000) == []
    assert [row["address"] for row in api_endpoint.holders(document, "198.51.100.10", 8015)] == ["0.0.0.0"]
    assert [row["address"] for row in api_endpoint.holders(document, "198.51.100.11", 29778)] == ["*"]
    # An IPv6 loopback listener does not hold an IPv4 port.
    assert api_endpoint.holders(document, "0.0.0.0", 631) == []


def test_the_question_offers_the_spark_s_own_networks_only(probed):
    document, _ = probed
    value = cluster(2)
    assert api_endpoint.offered(value, None, document) == [{"address": "198.51.100.10", "interface": "enP7s7"},
                                                           {"address": "198.51.100.11", "interface": "wlP9s9"}]
    lines, answers = [], ["", "8016"]
    chosen = api_endpoint.ask(value, None, document, 8015, read=lambda prompt: answers.pop(0), write=lines.append)
    assert chosen == {"api_port": 8016}
    assert lines == ["Model API (Enter keeps the first choice):",
                     "  1. Every address of Node A, shown as http://192.0.2.10:8015/v1",
                     "  2. Only 198.51.100.10 (enP7s7)", "  3. Only 198.51.100.11 (wlP9s9)"]
    answers = ["2", "8015"]
    assert api_endpoint.ask(value, None, document, 8015, read=lambda prompt: answers.pop(0),
                            write=lines.append) == {"api_bind": "198.51.100.10"}
    for given, message in ((["4"], "Choose one of the listed API addresses"),
                           (["", "80"], "--api-port takes a whole number from 1024 to 65535")):
        with pytest.raises(NeedsInput, match=message):
            api_endpoint.ask(value, None, document, 8015, read=lambda prompt: given.pop(0), write=lines.append)


def test_the_model_s_running_deployment_may_hold_its_port(probed, tmp_path):
    document, _ = probed
    value = cluster(2)
    other = tmp_path / "other"
    other.mkdir()
    (other / "deployment.lock.json").write_text(json.dumps({"id": "e" * 64, "backend": "compose", "site": {
        "name": "x", "ranks": [{"rank": 0, "host": "a"}]}}))
    with pytest.raises(NeedsInput, match=r"TCP port 8015 on Node A is in use by python3 \(pid 3255358, container "
                                         r"sr-ring-r0\) at 0.0.0.0:8015"):
        api_endpoint.check(value, None, {"api_port": 8015}, document, port=8015, allowed=[other])
    (other / "deployment.lock.json").write_text(json.dumps({"id": "f" * 64, "backend": "compose", "site": {
        "name": "x", "ranks": [{"rank": 0, "host": "a"}]}}))
    api_endpoint.check(value, None, {"api_port": 8015}, document, port=8015, allowed=[other, tmp_path / "missing"])


def test_a_half_on_sparks_2_and_3_is_checked_on_spark_2(probed):
    document, _ = probed
    value = cluster(4)
    asked = []

    def invoke(host, argv, timeout):
        asked.append((host, argv[:3]))
        return json.dumps(document)
    assert api_endpoint.inspect(value, (2, 3), invoke=invoke) == document
    assert asked == [(value["plan"]["spec"]["hosts"][2]["host"], ["sudo", "-n", "python3"])]
    document = {**document, "addresses": [*document["addresses"], {"interface": "lo", "address": "127.0.0.2"}]}
    with pytest.raises(NeedsInput) as caught:
        api_endpoint.check(value, (2, 3), {"api_bind": "127.0.0.2"}, document, port=8000, created=True,
                           allow_loopback=True)
    assert str(caught.value) == ("--api-bind 127.0.0.2 is a loopback address of Spark 2 (spark2); Node A, which checks "
                                 "the model through its API, could not reach it. Choose another address. Nothing has "
                                 "been changed.")
