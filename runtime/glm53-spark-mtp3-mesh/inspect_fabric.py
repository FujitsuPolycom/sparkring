#!/usr/bin/env python3
"""Read-only local validation of configured mesh routes, hardware rules and marker leases."""
from __future__ import annotations

import argparse
import importlib.util
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mtp3_mesh_profile", HERE / "profile.py")
profile = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = profile
spec.loader.exec_module(profile)


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError(f"Read-only command failed: {argv!r}: {result.stderr.strip()}")
    return result.stdout


def hex_value(value):
    return value if type(value) is int else int(str(value), 16)


def verify_rule(rule, entries):
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise ValueError("Hardware filter inventory must be a list of objects")
    matches = [entry for entry in entries if entry.get("pref") == rule.preference
               and isinstance(entry.get("options"), dict)
               and entry["options"].get("handle") == rule.handle]
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous hardware rule {rule.preference}/{rule.handle}")
    options = matches[0]["options"]
    keys = options.get("keys", {})
    if not isinstance(keys, dict):
        raise ValueError("Hardware rule keys must be an object")
    if any(not isinstance(keys.get(name), str) for name in ("src_mac", "dst_mac")):
        raise ValueError("Hardware rule MAC values must be strings")
    if (keys.get("src_mac", "").lower() != rule.match_ethernet_source.lower()
            or keys.get("dst_mac", "").lower() != rule.match_ethernet_destination.lower()
            or hex_value(keys.get("eth_type", "0")) != 0x88B5
            or options.get("skip_sw") is not True or options.get("in_hw") is not True):
        raise ValueError("Rule match or hardware-only placement differs from the plan")
    actions = options.get("actions", [])
    if not isinstance(actions, list) or any(not isinstance(action, dict) for action in actions):
        raise ValueError("Hardware rule actions must be a list of objects")
    if [action.get("kind") for action in actions] != ["pedit", "pedit", "mirred"]:
        raise ValueError("Expected EtherType restore, destination-MAC rewrite and redirect")
    first = actions[0].get("keys", [])
    second = actions[1].get("keys", [])
    if (not isinstance(first, list) or not isinstance(second, list)
            or any(not isinstance(item, dict) for item in [*first, *second])):
        raise ValueError("Hardware rewrite keys must be lists of objects")
    destination = bytes.fromhex(rule.rewrite_ethernet_destination.replace(":", ""))
    expected = [(12, 0x08000000, 0xFFFF), (0, int.from_bytes(destination[:4], "big"), 0),
                (4, int.from_bytes(destination[4:] + b"\0\0", "big"), 0xFFFF)]
    actual = [(item.get("offset"), hex_value(item.get("val", "0")), hex_value(item.get("mask", "0")))
              for item in [*first, *second]]
    if actual != expected or any(item.get("htype") != "eth" or item.get("cmd") != "set" for item in [*first, *second]):
        raise ValueError("Hardware rewrite bytes differ from the plan")
    if actions[2].get("to_dev") != rule.egress_netdev or actions[2].get("mirred_action") != "redirect":
        raise ValueError("Hardware redirect uses a different egress interface")
    for action in actions:
        stats = action.get("stats", {})
        if not isinstance(stats, dict):
            raise ValueError("Hardware action counters must be an object")
        if stats.get("sw_packets", 0) or stats.get("drops", 0):
            raise ValueError("The hardware rule reports software packets or drops")
    return {"preference": rule.preference, "handle": rule.handle, "in_hw": True,
            "software_packet_counter_available": all("sw_packets" in a.get("stats", {}) for a in actions)}


def remaining_seconds(argv, stat_text, uptime, ticks):
    if not isinstance(argv, list) or argv.count("--run-seconds") != 1:
        raise ValueError("Marker process has no bounded lifetime")
    position = argv.index("--run-seconds") + 1
    if position >= len(argv):
        raise ValueError("Marker lifetime value is missing")
    lifetime = int(argv[position])
    if not 1 <= lifetime <= 7200:
        raise ValueError("Marker lifetime is outside the supported bound")
    if (type(uptime) not in (int, float) or not math.isfinite(uptime) or uptime < 0
            or type(ticks) not in (int, float) or not math.isfinite(ticks) or ticks <= 0):
        raise ValueError("Process clock values must be finite with positive tick frequency")
    if not isinstance(stat_text, str) or ")" not in stat_text:
        raise ValueError("Malformed process stat record")
    fields = stat_text.rsplit(")", 1)[1].split()
    if len(fields) < 20:
        raise ValueError("Process stat record has no start time")
    start_ticks = int(fields[19])
    if start_ticks < 0 or start_ticks / ticks > uptime:
        raise ValueError("Process start time is outside the current boot")
    return lifetime - (uptime - start_ticks / ticks)


def inspect(site_path, rank, minimum_remaining):
    site, topology, plan = profile.load_site(site_path)
    local = topology.rank(rank)
    management = json.loads(command(["ip", "-j", "-4", "addr", "show", "dev", local.management_netdev]))
    addresses = {item["local"] for link in management for item in link.get("addr_info", [])}
    if site["management_addresses"][rank] not in addresses:
        raise ValueError("Management address does not identify this rank")
    for port in local.ports:
        links = json.loads(command(["ip", "-j", "addr", "show", "dev", port.netdev]))
        if len(links) != 1 or links[0]["mtu"] != 9000 or links[0]["address"].lower() != port.mac.lower():
            raise ValueError(f"Link identity or Ethernet MTU differs: {port.netdev}")
        if port.ipv4 not in {x.get("local") for x in links[0].get("addr_info", [])}:
            raise ValueError(f"Configured address is absent: {port.netdev}")
        gid = Path("/sys/class/infiniband") / port.rdma_device / "ports/1/gids/3"
        if ipaddress.IPv6Address(gid.read_text().strip()).ipv4_mapped != ipaddress.IPv4Address(port.ipv4):
            raise ValueError(f"RoCE GID 3 differs from the configured source: {port.rdma_device}")
    for route in plan.routes:
        if route.source_rank != rank:
            continue
        found = json.loads(command(["ip", "-j", "-4", "route", "show", "exact", route.destination_ipv4 + "/32"]))
        if len(found) != 1 or found[0].get("dev") != route.source_netdev or found[0].get("gateway") != route.gateway_ipv4:
            raise ValueError(f"Opposite-peer route differs: {route.destination_ipv4}")
    rules = [verify_rule(rule, json.loads(command(["tc", "-j", "-s", "filter", "show", "dev", rule.ingress_netdev, "ingress"])))
             for rule in plan.tc_rules if rule.intermediate_rank == rank]
    binary = Path(site["marker_binary"])
    if profile.sha(binary) != site["marker_binary_sha256"]:
        raise ValueError("Native marker binary digest differs")
    found = subprocess.run(["pgrep", "-f", "^" + re.escape(str(binary)) + " "], capture_output=True, text=True, timeout=10)
    if found.returncode not in (0, 1):
        raise RuntimeError("Cannot enumerate the configured marker processes")
    expected_devices = {marker.rdma_device for marker in plan.markers if marker.source_rank == rank}
    observed = []
    uptime = float(Path("/proc/uptime").read_text().split()[0])
    for value in found.stdout.split():
        pid = int(value)
        root = Path("/proc") / str(pid)
        argv = root.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")
        if argv[0] != str(binary) or "--device" not in argv:
            raise ValueError("Marker process changed during inspection")
        def argument(name):
            if argv.count(name) != 1 or argv.index(name) + 1 >= len(argv):
                raise ValueError(f"Missing or duplicated marker argument: {name}")
            return argv[argv.index(name) + 1]
        device = argument("--device")
        if device not in expected_devices:
            continue
        if (argv.count("--attach") != 1 or argument("--source-port") != "65535"
                or argument("--replacement-ethertype").lower() != "0x88b5"):
            raise ValueError("Marker process arguments differ from the profile")
        remaining = remaining_seconds(argv, root.joinpath("stat").read_text(), uptime, os.sysconf("SC_CLK_TCK"))
        if remaining < minimum_remaining:
            raise ValueError(f"Marker lease has insufficient time: {remaining:.1f} seconds")
        observed.append({"pid": pid, "device": device, "remaining_seconds": round(remaining, 1)})
    if len(observed) != len(expected_devices) or {x["device"] for x in observed} != expected_devices:
        raise ValueError("Expected exactly one marker process on each selected function")
    return {"schema": "sparkring-mtp3-mesh-readiness/v1", "status": "research-only", "rank": rank,
            "ready": True, "rules": rules, "markers": observed,
            "limitation": "Read-only identity and lease snapshot, not an end-to-end RC traffic or model correctness test."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=range(4), required=True)
    parser.add_argument("--minimum-remaining", type=int, default=900)
    args = parser.parse_args()
    if not 0 <= args.minimum_remaining <= 7200:
        parser.error("minimum remaining time must be in [0,7200]")
    try:
        print(json.dumps(inspect(args.site, args.rank, args.minimum_remaining), indent=2))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(json.dumps({"ready": False, "error": str(error)}))
        raise SystemExit(1)
