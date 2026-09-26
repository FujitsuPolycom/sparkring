"""Keep each fabric port's IPv4 address in the RoCE GID index that SparkRing pins.

The kernel lists one RoCE v2 GID for each IPv4 address of a port's netdev. When
the link goes down while a model or mesh holds that GID entry, as on the
Sparks cabled to one that restarts, the address returns in another GID index
and the pinned index stays empty until nothing holds the old entry. Once the
holders have stopped, deleting and adding the address again puts it back.
"""
import ipaddress
import json
from pathlib import Path
import time

from runtime.host import node

SETTLE_SECONDS = 10


def readd_address(netdev, ipv4, call):
    """Delete and add one IPv4 address with its prefix and flags, re-registering its RoCE GIDs."""
    links = json.loads(call(["ip", "-j", "-4", "addr", "show", "dev", netdev]).stdout)
    entries = [entry for link in links for entry in link.get("addr_info", []) if entry.get("local") == ipv4]
    if len(entries) != 1:
        raise ValueError(f"{netdev} does not hold its fabric address {ipv4}")
    address = f"{ipv4}/{entries[0]['prefixlen']}"
    extra = ((["broadcast", entries[0]["broadcast"]] if entries[0].get("broadcast") else [])
             + (["noprefixroute"] if entries[0].get("noprefixroute") else []))
    call(["ip", "addr", "del", address, "dev", netdev])
    call(["ip", "addr", "add", address, *extra, "dev", netdev])


def stale_ports(hcas, index, *, call=node.call, root=Path("/")):
    """(netdev, IPv4 address) of each RDMA device in ``hcas`` whose GID ``index`` lacks its RoCE v2 address.

    Each device's netdev comes from its PCI function and must hold exactly one
    IPv4 address. This reads sysfs and ``ip`` only; an empty index is stale.
    """
    stale = []
    for device in hcas:
        base = root / "sys/class/infiniband" / device
        netdevs = [entry.name for entry in (base / "device/net").iterdir()]
        if len(netdevs) != 1:
            raise ValueError(f"{device} has no single network interface")
        links = json.loads(call(["ip", "-j", "-4", "addr", "show", "dev", netdevs[0]]).stdout)
        addresses = [entry["local"] for link in links for entry in link.get("addr_info", [])]
        if len(addresses) != 1:
            raise ValueError(f"{netdevs[0]} must hold exactly one IPv4 address")
        port = base / "ports/1"
        try:
            gid = ipaddress.IPv6Address((port / f"gids/{index}").read_text().strip())
            owner = (port / f"gid_attrs/ndevs/{index}").read_text().strip()
            kind = (port / f"gid_attrs/types/{index}").read_text().strip()
        except (OSError, ValueError):
            gid = owner = kind = None
        if gid is None or owner != netdevs[0] or kind != "RoCE v2" or str(gid.ipv4_mapped) != addresses[0]:
            stale.append((netdevs[0], addresses[0]))
    return stale


def check(hcas, index, **options):
    """Raise ValueError naming each port whose GID ``index`` lacks its address."""
    stale = stale_ports(hcas, index, **options)
    if stale:
        raise ValueError(f"RoCE GID index {index} lacks the address of "
                         + ", ".join(f"{netdev} ({ipv4})" for netdev, ipv4 in stale))
    return {"ok": True}


def serve(hcas, index, *, call=node.call, root=Path("/"), sleep=time.sleep, clock=time.monotonic):
    """Re-add each address missing from GID ``index`` and wait until every port holds its address.

    Nothing may hold the stale entries: the model on this Spark has stopped.
    The kernel registers the GIDs of an added address asynchronously, so the
    check is repeated for up to ``SETTLE_SECONDS``.
    """
    repaired = stale_ports(hcas, index, call=call, root=root)
    for netdev, ipv4 in repaired:
        readd_address(netdev, ipv4, call)
    deadline = clock() + SETTLE_SECONDS
    while True:
        try:
            check(hcas, index, call=call, root=root)
            break
        except ValueError:
            if clock() >= deadline:
                raise
            sleep(0.5)
    return {"ok": True, "repaired": [netdev for netdev, _ in repaired]}
