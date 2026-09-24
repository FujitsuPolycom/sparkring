"""Discover candidates without trusting advertisements; authenticate through SSH."""
import ipaddress
import json
import re
import shlex
import subprocess


def avahi_candidates(text):
    found = {}
    for line in text.splitlines():
        parts = line.split(";")
        if len(parts) < 9 or parts[0] != "=" or parts[2] != "IPv4" or parts[4] != "_sparkring._tcp":
            continue
        address = ipaddress.IPv4Address(parts[7])
        if address.is_loopback or address.is_multicast or address.is_unspecified or parts[8] != "22":
            continue
        found[str(address)] = {"address": str(address), "hostname": parts[6], "authenticated": False}
    return sorted(found.values(), key=lambda item: ipaddress.IPv4Address(item["address"]))


def discover(*, run=subprocess.run):
    result = run(["avahi-browse", "--resolve", "--terminate", "--parsable", "_sparkring._tcp"],
                 capture_output=True, text=True, timeout=30, check=True)
    return avahi_candidates(result.stdout)


def target(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9._-]+@(?:[0-9]{1,3}\.){3}[0-9]{1,3}", value):
        raise ValueError("Node target must be username@management-IPv4")
    address = ipaddress.IPv4Address(value.split("@", 1)[1])
    if address.is_loopback or address.is_unspecified or address.is_multicast:
        raise ValueError("Use a node's management address")
    return value


def ssh(node, args, *, data=None, run=subprocess.run, timeout=180):
    result = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target(node), shlex.join(args)],
                 input=data, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{node}: {result.stderr.strip()}")
    return result.stdout


def inspect_node(node, rank, witness, *, invoke=ssh):
    address = target(node).split("@", 1)[1]
    result = json.loads(invoke(node, ["sudo", "-n", "/usr/bin/sparkring", "node", "inspect", "--rank", str(rank),
                                     "--target", node, "--management", address, "--witness", witness]))
    if result["facts"]["management"]["address"] != address or result["facts"]["ssh_target"] != node:
        raise ValueError("Authenticated host inventory differs from its SSH target")
    return result
