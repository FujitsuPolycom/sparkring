"""Find the RoCE v2 GID table index that carries an IPv4 fabric address.

The kernel lists every GID of an RDMA port under
``/sys/class/infiniband/<device>/ports/<port>/gids/<index>``, the GID type in
``gid_attrs/types/<index>`` and the network interface that owns the entry in
``gid_attrs/ndevs/<index>``. Each IPv4 address of an interface appears as an
IPv4-mapped GID (``::ffff:a.b.c.d``) once per RoCE version. The index of the
RoCE v2 entry is not a property of the address: it depends on the other
addresses the interface holds (IPv6 link-local entries usually occupy the
first two indices), and an address registered again while its previous entry
is still referenced, as on a Spark whose cabled neighbor restarted under a
running model, returns at another index.

Resolution reads the table and selects the single RoCE v2 entry whose GID is
the IPv4-mapped form of the address. Entries of other types, IPv6 GIDs and
empty (all-zero) entries are ignored. No match and several matches are errors
that list the RoCE v2 IPv4 entries present.

The module uses only the standard library so the SIRCL transport can import it
next to its adapter inside a serving container. It also runs as a read-only
host command that prints the resolved index::

    python3 integrations/vllm/spark_roce_gid.py rocep1s0f0 198.18.0.1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import ipaddress
import os
from pathlib import Path
import sys
from typing import Iterable, Sequence

SYSFS_ROOT = Path("/sys/class/infiniband")
ROCE_V2 = "RoCE v2"


class GidResolutionError(ValueError):
    """No single RoCE v2 GID carries the requested IPv4 address."""


@dataclass(frozen=True)
class GidEntry:
    """One populated GID table entry of an RDMA port.

    ``type`` and ``netdev`` are None when the kernel did not report them.
    """

    index: int
    gid: str
    type: str | None = None
    netdev: str | None = None

    @property
    def ipv4(self) -> str | None:
        """The IPv4 address of an IPv4-mapped GID, else None."""
        try:
            mapped = ipaddress.IPv6Address(self.gid).ipv4_mapped
        except ValueError:
            return None
        if mapped is None or int(mapped) == 0:
            return None
        return str(mapped)

    @property
    def roce_v2(self) -> bool:
        return self.type is not None and " ".join(self.type.split()).lower() == ROCE_V2.lower()

    def describe(self) -> str:
        value = self.ipv4 or self.gid
        return (f"index {self.index} ({value}, {self.type or 'type unknown'}, "
                f"{self.netdev or 'interface unknown'})")


def ipv4_mapped_gid(address: str | ipaddress.IPv4Address) -> str:
    """The sysfs text of the IPv4-mapped GID of ``address``.

    Built from the address bytes: ``IPv6Address.exploded`` writes an
    IPv4-mapped address with a dotted-quad tail on newer Python releases, while
    the kernel always writes eight groups of four hexadecimal digits.
    """
    packed = bytes(10) + bytes((0xFF, 0xFF)) + ipaddress.IPv4Address(address).packed
    return ":".join(packed[i:i + 2].hex() for i in range(0, 16, 2))


def _root(root: Path | str | None) -> Path:
    # Resolved at call time so tests can point SYSFS_ROOT at a fake tree.
    return Path(SYSFS_ROOT if root is None else root)


def _attribute(port_dir: Path, kind: str, index: str) -> str | None:
    try:
        value = (port_dir / "gid_attrs" / kind / index).read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return value or None


def read_gid_table(device: str, port: int = 1, *, root: Path | str | None = None) -> tuple[GidEntry, ...]:
    """The populated GID entries of ``device`` port ``port``, in index order.

    Raises GidResolutionError when the port has no GID table under ``root``.
    """
    port_dir = _root(root) / device / "ports" / str(port)
    try:
        names = os.listdir(port_dir / "gids")
    except OSError as error:
        raise GidResolutionError(
            f"{device} port {port} has no readable GID table under {port_dir}: {error.strerror or error}"
        ) from None
    entries = []
    for name in sorted((item for item in names if item.isdigit()), key=int):
        try:
            text = (port_dir / "gids" / name).read_text(encoding="ascii").strip().lower()
            value = ipaddress.IPv6Address(text)
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if int(value) == 0:
            continue
        entries.append(GidEntry(int(name), text, _attribute(port_dir, "types", name),
                                _attribute(port_dir, "ndevs", name)))
    return tuple(entries)


def select_gid_index(entries: Iterable[GidEntry], *, ipv4: str | None = None, netdev: str | None = None,
                     where: str = "the GID table") -> int:
    """The index of the single RoCE v2 IPv4-mapped entry that matches.

    ``ipv4`` restricts the match to that address and ``netdev`` to entries the
    named interface owns. Without ``ipv4``, the table (after the ``netdev``
    restriction) must hold exactly one RoCE v2 IPv4 entry. Raises
    GidResolutionError naming the entries present when none or several match.
    """
    entries = tuple(entries)
    address = None if ipv4 is None else str(ipaddress.IPv4Address(ipv4))
    usable = [entry for entry in entries if entry.roce_v2 and entry.ipv4 is not None]
    matches = [entry for entry in usable
               if (address is None or entry.ipv4 == address) and (netdev is None or entry.netdev == netdev)]
    if len(matches) == 1:
        return matches[0].index
    target = f"IPv4 address {address}" if address is not None else "IPv4 address"
    scope = f" on {netdev}" if netdev is not None else ""
    if not matches:
        present = "; ".join(entry.describe() for entry in usable) or "none"
        raise GidResolutionError(f"{where} has no RoCE v2 GID for {target}{scope} "
                                 f"(RoCE v2 IPv4 GIDs present: {present})")
    raise GidResolutionError(f"{where} has several RoCE v2 GIDs for {target}{scope}: "
                             + "; ".join(entry.describe() for entry in matches))


def resolve_gid_index(device: str, ipv4: str | None = None, *, port: int = 1, netdev: str | None = None,
                      root: Path | str | None = None) -> int:
    """The GID index of ``device`` port ``port`` whose RoCE v2 GID carries ``ipv4``.

    See select_gid_index for ``ipv4`` and ``netdev``.
    """
    return select_gid_index(read_gid_table(device, port, root=root), ipv4=ipv4, netdev=netdev,
                            where=f"{device} port {port}")


def device_netdev(device: str, *, root: Path | str | None = None) -> str | None:
    """The network interface of ``device``'s PCI function, or None when not exactly one is visible."""
    try:
        names = os.listdir(_root(root) / device / "device" / "net")
    except OSError:
        return None
    return names[0] if len(names) == 1 else None


def resolve_device_gid_index(device: str, *, port: int = 1, root: Path | str | None = None) -> int:
    """The index of the single RoCE v2 IPv4 GID owned by ``device``'s own interface.

    For callers that know the RDMA device but not its fabric address. When the
    interface is not visible, every RoCE v2 IPv4 entry of the port is
    considered. An interface holding several IPv4 addresses has several such
    entries and is an error.
    """
    return resolve_gid_index(device, port=port, netdev=device_netdev(device, root=root), root=root)


def main(argv: Sequence[str] | None = None, *, root: Path | str | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print the RoCE v2 GID index that carries an IPv4 address. Reads sysfs only.")
    parser.add_argument("device", help="RDMA device, for example rocep1s0f0")
    parser.add_argument("ipv4", nargs="?", help="fabric IPv4 address; default: the single IPv4 "
                        "address of the device's own interface")
    parser.add_argument("--port", type=int, default=1, help="RDMA port number (default 1)")
    parser.add_argument("--netdev", help="only consider entries owned by this interface")
    args = parser.parse_args(argv)
    try:
        if args.ipv4 is None and args.netdev is None:
            index = resolve_device_gid_index(args.device, port=args.port, root=root)
        else:
            index = resolve_gid_index(args.device, args.ipv4, port=args.port, netdev=args.netdev, root=root)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    print(index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
