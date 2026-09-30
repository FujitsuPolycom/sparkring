"""Prepare cabled Sparks through the LAN before fabric discovery.

A factory Spark's fabric connections ask for DHCP, which a direct cable never
answers, so its fabric address exists only while a DHCP attempt lasts (see
seed.dhcp_without_lease). When a cabled Spark is also on Node A's LAN, setup
signs in to it there once, where the link is stable, installs SparkRing from
Node A's worker bundle and runs the worker preparation: it turns DHCP off on
the Spark's fabric connections and opens the preparation SSH service (port
2222) to Node A's key. Fabric discovery then signs in there with that key.
"""
import json
import re
import subprocess
import time

from runtime.host import packages

# A factory fabric port retries DHCP for about 3 minutes, then pauses for 5.
WAIT_SECONDS = 480
POLL_SECONDS = 10


def mac_number(mac):
    return int(mac.replace(":", "").lower(), 16)


def cabled(inventory):
    """Whether a fabric function of the Spark inventory describes has a link."""
    return any(f.get("carrier") for f in inventory["functions"])


def peer_macs(inventory):
    """MACs of other hosts' fabric functions in the neighbor cache of the Spark inventory describes."""
    own = {str(f["mac"]).lower() for f in inventory["functions"]}
    fabric = {f["netdev"] for f in inventory["functions"]}
    return sorted({n["lladdr"].lower() for n in inventory["neighbors"]
                   if n.get("dev") in fabric and n.get("lladdr") and n["lladdr"].lower() not in own})


def lan_hosts(interface, *, run=subprocess.run):
    """{MAC: link-local address} of the hosts that answer the all-nodes echo on the LAN interface."""
    output = run(["ping", "-6", "-n", "-w", "2", "-I", interface, "ff02::1"],
                 capture_output=True, text=True, timeout=10).stdout
    replied = set(re.findall(r"from (fe80:[0-9a-f:]+)", output))
    rows = json.loads(run(["ip", "-j", "-6", "neigh", "show", "dev", interface],
                          capture_output=True, text=True, check=True, timeout=10).stdout)
    return {row["lladdr"].lower(): row["dst"].split("%")[0] for row in rows
            if row.get("lladdr") and row.get("dst", "").split("%")[0] in replied}


def match(fabric_macs, hosts):
    """{LAN MAC: (link-local address, fabric MAC)} for LAN hosts that are cabled Sparks.

    A Spark's LAN port and ConnectX functions take consecutive MAC addresses,
    the LAN port first, so a cabled Spark's fabric MAC exceeds its LAN MAC by
    one to eight within the same vendor prefix.
    """
    found = {}
    for mac, address in sorted(hosts.items()):
        for fabric in fabric_macs:
            if fabric[:8] == mac[:8] and 0 < mac_number(fabric) - mac_number(mac) <= 8:
                found.setdefault(mac, (address, fabric))
    return found


def wait_for_peers(inventory_of_head, *, clock=time.monotonic, sleep=time.sleep, say=print):
    """Node A's inventory and the fabric MACs of the Sparks cabled to it.

    Polls while Node A has a cabled fabric function but sees no other host on
    it, for at most WAIT_SECONDS.
    """
    deadline = clock() + WAIT_SECONDS
    announced = False
    while True:
        head = inventory_of_head()
        macs = peer_macs(head)
        if macs or not cabled(head) or clock() >= deadline:
            return head, macs
        if not announced:
            say("Waiting for the Spark on the fabric cable to answer; a factory Spark retries its fabric "
                f"connection every few minutes (at most {WAIT_SECONDS // 60} minutes)")
            announced = True
        sleep(POLL_SECONDS)


def prepare(transport, root_command, *, head, user, archive, run=subprocess.run, say=print):
    """Install SparkRing and run worker preparation on every cabled Spark found on Node A's LAN.

    ``head`` is Node A's inventory; ``archive`` returns the worker bundle.
    Returns the inventories of the prepared Sparks, empty when none of the
    cabled Sparks is on the LAN.
    """
    interface = head.get("uplink")
    macs = peer_macs(head)
    if not interface or not macs:
        return []
    found = match(macs, lan_hosts(interface, run=run))
    prepared = []
    for mac, (address, fabric) in found.items():
        route = [{"user": user, "address": address, "interface": interface, "port": 22}]
        say(f"Cabled Spark on the LAN at {address} on {interface} (fabric MAC {fabric}); signing in there as {user}")
        transport.login(route)
        peer = transport.inventory(route)
        if fabric not in {str(f["mac"]).lower() for f in peer["functions"]}:
            raise ValueError(f"{peer['hostname']} at {address} on {interface} has no fabric function with MAC {fabric}; "
                             "it is not the Spark on the cable")
        destination = "/var/tmp/sparkring-enroll-" + str(time.time_ns())
        packages.transfer(transport, route, archive(), destination)
        say(f"Install SparkRing on {peer['hostname']} and turn off DHCP on its fabric connections")
        root_command(transport, route, ["python3", "-I", destination + "/install.py", "--apply", "--prepare", "--yes"])
        prepared.append(peer)
    return prepared
