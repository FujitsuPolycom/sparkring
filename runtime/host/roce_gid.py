"""Keep each fabric port's IPv4 address in the RoCE GID index that SparkRing pins.

The kernel lists one RoCE v2 GID for each IPv4 address of a port's netdev. When
the link goes down while a model or mesh holds that GID entry, as on the
Sparks cabled to one that restarts, the address returns in another GID index
and the pinned index stays empty until nothing holds the old entry. Once the
holders have stopped, deleting and adding the address again puts it back.

The installer's serving containers pin one index for every HCA of a rank: the
image's prepared B12X RoCE transport takes one index for all of its HCAs
(``B12X_ROCE_GID_INDEX``, else ``NCCL_IB_GID_INDEX``), and the profiles set
``NCCL_IB_GID_INDEX`` to the pinned index. Restoring that index before a model
starts is therefore what keeps those settings valid. The shared resolver in
``integrations/vllm/spark_roce_gid.py`` locates each address's RoCE v2 GID; a
failed check names the index it found or the resolver's error.
"""
import json
from pathlib import Path
import time

from integrations.vllm import spark_roce_gid
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


def locate(hcas, *, call=node.call, root=Path("/")):
    """(netdev, IPv4 address, resolved index or resolution error) of each RDMA device in ``hcas``.

    Each device's netdev comes from its PCI function and must hold exactly one
    IPv4 address. The index is that of the address's RoCE v2 GID owned by the
    netdev. This reads sysfs and ``ip`` only.
    """
    sysfs = root / "sys/class/infiniband"
    ports = []
    for device in hcas:
        netdevs = [entry.name for entry in (sysfs / device / "device/net").iterdir()]
        if len(netdevs) != 1:
            raise ValueError(f"{device} has no single network interface")
        links = json.loads(call(["ip", "-j", "-4", "addr", "show", "dev", netdevs[0]]).stdout)
        addresses = [entry["local"] for link in links for entry in link.get("addr_info", [])]
        if len(addresses) != 1:
            raise ValueError(f"{netdevs[0]} must hold exactly one IPv4 address")
        try:
            where = spark_roce_gid.resolve_gid_index(device, addresses[0], netdev=netdevs[0], root=sysfs)
        except ValueError as error:
            where = error
        ports.append((netdevs[0], addresses[0], where))
    return ports


def stale_ports(hcas, index, *, call=node.call, root=Path("/")):
    """(netdev, IPv4 address) of each RDMA device in ``hcas`` whose RoCE v2 GID is not at ``index``."""
    return [(netdev, ipv4) for netdev, ipv4, where in locate(hcas, call=call, root=root) if where != index]


def check(hcas, index, **options):
    """Raise ValueError naming each port whose RoCE v2 GID is not at ``index``, and where it is."""
    stale = [(netdev, ipv4, where) for netdev, ipv4, where in locate(hcas, **options) if where != index]
    if stale:
        raise ValueError(f"RoCE GID index {index} lacks the address of " + ", ".join(
            f"{netdev} ({ipv4}): " + (f"its RoCE v2 GID is at index {where}" if isinstance(where, int)
                                      else str(where))
            for netdev, ipv4, where in stale))
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
