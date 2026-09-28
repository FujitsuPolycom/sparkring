"""RoCE v2 GID resolution against fake sysfs GID tables."""

from __future__ import annotations

from pathlib import Path

import pytest

import spark_roce_gid as resolver

DEVICE, NETDEV, ADDRESS = "rocep1s0f0", "enp1s0f0np0", "198.18.0.1"
LINK_LOCAL = "fe80:0000:0000:0000:5e25:73ff:fe01:0203"


def write_port(root: Path, entries: dict[int, tuple[str, str | None, str | None]], *,
               device: str = DEVICE, netdevs: tuple[str, ...] = (NETDEV,), size: int = 16) -> Path:
    """A sysfs port with ``size`` GID slots; ``entries`` maps index to (gid, type, netdev).

    Slots absent from ``entries`` hold the all-zero GID and no attributes, as
    the kernel reports unused or invalid entries.
    """
    base = root / device
    for name in netdevs:
        (base / "device/net" / name).mkdir(parents=True)
    port = base / "ports/1"
    for kind in ("gids", "gid_attrs/types", "gid_attrs/ndevs"):
        (port / kind).mkdir(parents=True)
    for index in range(size):
        gid, kind, owner = entries.get(index, ("0000:0000:0000:0000:0000:0000:0000:0000", None, None))
        (port / "gids" / str(index)).write_text(gid + "\n")
        if kind is not None:
            (port / "gid_attrs/types" / str(index)).write_text(kind + "\n")
        if owner is not None:
            (port / "gid_attrs/ndevs" / str(index)).write_text(owner + "\n")
    return port


def standard_table(address: str = ADDRESS, v2_index: int = 3, netdev: str = NETDEV) -> dict:
    """IPv6 link-local entries at 0 and 1, the IPv4 RoCE v1 entry at 2 and RoCE v2 at ``v2_index``."""
    mapped = resolver.ipv4_mapped_gid(address)
    return {0: (LINK_LOCAL, "IB/RoCE v1", netdev), 1: (LINK_LOCAL, "RoCE v2", netdev),
            2: (mapped, "IB/RoCE v1", netdev), v2_index: (mapped, "RoCE v2", netdev)}


def test_ipv4_mapped_gid_uses_the_sysfs_text_form():
    assert resolver.ipv4_mapped_gid("198.18.0.1") == "0000:0000:0000:0000:0000:ffff:c612:0001"


def test_the_single_roce_v2_entry_of_the_address_is_selected(tmp_path):
    write_port(tmp_path, standard_table())
    assert resolver.resolve_gid_index(DEVICE, ADDRESS, root=tmp_path) == 3
    assert resolver.resolve_gid_index(DEVICE, ADDRESS, netdev=NETDEV, root=tmp_path) == 3
    assert resolver.resolve_device_gid_index(DEVICE, root=tmp_path) == 3


def test_an_address_that_moved_to_another_index_is_found_there(tmp_path):
    # The cabled neighbor restarted while the old entry was held: index 3 is
    # empty and the address's RoCE v2 GID returned at index 5.
    write_port(tmp_path, standard_table(v2_index=5))
    assert resolver.resolve_gid_index(DEVICE, ADDRESS, root=tmp_path) == 5
    assert resolver.resolve_device_gid_index(DEVICE, root=tmp_path) == 5


def test_an_ipv6_disabled_interface_resolves_its_lower_index(tmp_path):
    mapped = resolver.ipv4_mapped_gid(ADDRESS)
    write_port(tmp_path, {0: (mapped, "IB/RoCE v1", NETDEV), 1: (mapped, "RoCE v2", NETDEV)})
    assert resolver.resolve_gid_index(DEVICE, ADDRESS, root=tmp_path) == 1


def test_ipv6_and_roce_v1_entries_are_never_selected(tmp_path):
    table = standard_table()
    del table[3]
    table[4] = ("2001:0db8:0000:0000:0000:0000:0000:0001", "RoCE v2", NETDEV)
    write_port(tmp_path, table)
    with pytest.raises(resolver.GidResolutionError,
                       match=r"rocep1s0f0 port 1 has no RoCE v2 GID for IPv4 address 198\.18\.0\.1 "
                             r"\(RoCE v2 IPv4 GIDs present: none\)"):
        resolver.resolve_gid_index(DEVICE, ADDRESS, root=tmp_path)
    with pytest.raises(resolver.GidResolutionError, match="no RoCE v2 GID"):
        resolver.resolve_device_gid_index(DEVICE, root=tmp_path)


def test_a_missing_address_names_the_roce_v2_entries_present(tmp_path):
    write_port(tmp_path, standard_table(address="198.18.0.9", v2_index=5))
    with pytest.raises(resolver.GidResolutionError,
                       match=r"no RoCE v2 GID for IPv4 address 198\.18\.0\.1 .*"
                             r"index 5 \(198\.18\.0\.9, RoCE v2, enp1s0f0np0\)"):
        resolver.resolve_gid_index(DEVICE, ADDRESS, root=tmp_path)


def test_an_entry_owned_by_another_interface_does_not_match_the_netdev(tmp_path):
    write_port(tmp_path, standard_table(netdev="enp1s0f0np0.100"))
    assert resolver.resolve_gid_index(DEVICE, ADDRESS, root=tmp_path) == 3
    with pytest.raises(resolver.GidResolutionError, match=r"on enp1s0f0np0 \("):
        resolver.resolve_gid_index(DEVICE, ADDRESS, netdev=NETDEV, root=tmp_path)


def test_several_matching_entries_are_ambiguous(tmp_path):
    mapped = resolver.ipv4_mapped_gid(ADDRESS)
    table = standard_table()
    table[7] = (mapped, "RoCE v2", "br-fabric")
    write_port(tmp_path, table)
    with pytest.raises(resolver.GidResolutionError,
                       match=r"several RoCE v2 GIDs for IPv4 address 198\.18\.0\.1: index 3 .*; index 7 "):
        resolver.resolve_gid_index(DEVICE, ADDRESS, root=tmp_path)
    assert resolver.resolve_gid_index(DEVICE, ADDRESS, netdev=NETDEV, root=tmp_path) == 3


def test_an_interface_with_two_addresses_needs_the_address(tmp_path):
    table = standard_table()
    table.update({4: (resolver.ipv4_mapped_gid("198.18.0.9"), "IB/RoCE v1", NETDEV),
                  5: (resolver.ipv4_mapped_gid("198.18.0.9"), "RoCE v2", NETDEV)})
    write_port(tmp_path, table)
    with pytest.raises(resolver.GidResolutionError, match="several RoCE v2 GIDs for IPv4 address on enp1s0f0np0"):
        resolver.resolve_device_gid_index(DEVICE, root=tmp_path)
    assert resolver.resolve_gid_index(DEVICE, "198.18.0.9", root=tmp_path) == 5


def test_a_device_without_one_visible_interface_considers_the_whole_port(tmp_path):
    write_port(tmp_path, standard_table(v2_index=6), netdevs=())
    assert resolver.device_netdev(DEVICE, root=tmp_path) is None
    assert resolver.resolve_device_gid_index(DEVICE, root=tmp_path) == 6


def test_a_missing_device_or_port_is_reported(tmp_path):
    with pytest.raises(resolver.GidResolutionError, match="rocep1s0f1 port 1 has no readable GID table"):
        resolver.resolve_gid_index("rocep1s0f1", ADDRESS, root=tmp_path)


def test_selection_accepts_parsed_entries_and_normalizes_type_spacing():
    entries = [resolver.GidEntry(3, resolver.ipv4_mapped_gid(ADDRESS), "RoCE  v2", NETDEV),
               resolver.GidEntry(2, resolver.ipv4_mapped_gid(ADDRESS), "IB/RoCE v1", NETDEV)]
    assert resolver.select_gid_index(entries, ipv4=ADDRESS, netdev=NETDEV) == 3
    with pytest.raises(ValueError):
        resolver.select_gid_index(entries, ipv4="not-an-address")


def test_the_command_prints_the_index_or_the_error(tmp_path, capsys):
    write_port(tmp_path, standard_table(v2_index=5))
    assert resolver.main([DEVICE, ADDRESS], root=tmp_path) == 0
    assert capsys.readouterr().out == "5\n"
    assert resolver.main([DEVICE], root=tmp_path) == 0
    assert capsys.readouterr().out == "5\n"
    assert resolver.main([DEVICE, "--netdev", NETDEV], root=tmp_path) == 0
    assert capsys.readouterr().out == "5\n"
    assert resolver.main([DEVICE, "198.18.0.2"], root=tmp_path) == 1
    assert "no RoCE v2 GID for IPv4 address 198.18.0.2" in capsys.readouterr().err
