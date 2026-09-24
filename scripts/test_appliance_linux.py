"""Opt-in network-namespace rehearsal; no physical interface or DGX is touched.

SPARKRING_LINUX_LAB=1 WG=/path/to/wg sudo -E python -m pytest this_file -q
"""
import ipaddress
import os
import subprocess
import sys

import pytest

from runtime.host import control
from runtime.host.test_control import fixture

pytestmark = pytest.mark.skipif(sys.platform != "linux" or os.environ.get("SPARKRING_LINUX_LAB") != "1",
                                reason="opt-in isolated Linux network namespace rehearsal")


@pytest.mark.parametrize("size", [2, 4])
def test_real_wireguard_tree_and_worker_default_route(tmp_path, size):
    wg = os.environ.get("WG", "wg")
    prefix = f"srtest{os.getpid()}-{size}"
    namespaces = [f"{prefix}-{i}" for i in range(size)]
    created = []

    def run(argv, *, namespace=None, data=None):
        if namespace:
            argv = ["ip", "netns", "exec", namespace, *argv]
        return subprocess.run(argv, input=data, capture_output=True, text=True, check=True).stdout

    nodes, edges = fixture(size)
    try:
        for ns in namespaces:
            run(["ip", "netns", "add", ns])
            created.append(ns)
            run(["ip", "link", "set", "lo", "up"], namespace=ns)
        for index, (left, right) in enumerate(edges):
            a, b = f"sra{os.getpid()}{index}", f"srb{os.getpid()}{index}"
            run(["ip", "link", "add", a, "type", "veth", "peer", "name", b])
            for end, temporary in ((left, a), (right, b)):
                ns = namespaces[int(end["id"])]
                run(["ip", "link", "set", temporary, "netns", ns])
                run(["ip", "link", "set", temporary, "name", end["netdev"]], namespace=ns)
                run(["ip", "link", "set", end["netdev"], "address", end["mac"]], namespace=ns)
                run(["ip", "link", "set", end["netdev"], "up"], namespace=ns)
                run(["ip", "-6", "addr", "add", end["address"] + "/64", "dev", end["netdev"], "nodad"], namespace=ns)
        private = {}
        for row in nodes:
            value = run([wg, "genkey"]).strip()
            private[row["id"]] = value
            row["public_key"] = run([wg, "pubkey"], data=value + "\n").strip()
        plans = control.plan(nodes, edges, "0", share_uplink=True)
        for plan in plans:
            ns = namespaces[int(plan["id"])]
            config = tmp_path / (plan["id"] + ".conf")
            rows = control.render(plan, private[plan["id"]]).splitlines()
            config.write_text("\n".join(r for r in rows if not r.startswith(("Address =", "MTU ="))) + "\n")
            config.chmod(0o600)
            run(["ip", "link", "add", control.INTERFACE, "type", "wireguard"], namespace=ns)
            run([wg, "setconf", control.INTERFACE, str(config)], namespace=ns)
            run(["ip", "addr", "add", plan["address"] + "/32", "dev", control.INTERFACE], namespace=ns)
            run(["ip", "link", "set", control.INTERFACE, "mtu", "1420", "up"], namespace=ns)
            run(["sysctl", "-w", "net.ipv4.conf.sr-control.forwarding=1"], namespace=ns)
            for peer in plan["peers"]:
                for address in peer["allowed_ips"]:
                    run(["ip", "route", "add", address, "dev", control.INTERFACE], namespace=ns)
        # A head-only address represents a resource outside the workers' fabric.
        run(["ip", "addr", "add", "192.0.2.254/32", "dev", "lo"], namespace=namespaces[0])
        for plan in plans:
            ns = namespaces[int(plan["id"])]
            for peer in plans:
                run(["ping", "-n", "-c", "1", "-W", "4", peer["address"]], namespace=ns)
            if not plan["head"]:
                run(["ping", "-n", "-c", "1", "-W", "4", "192.0.2.254"], namespace=ns)
            assert ipaddress.ip_address(plan["address"]) in ipaddress.ip_network(plan["subnet"])
    finally:
        for ns in reversed(created):
            run(["ip", "netns", "delete", ns])
