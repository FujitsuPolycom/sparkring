"""A fabric address that left its pinned RoCE GID index is found and re-added."""
import json
from types import SimpleNamespace

import pytest

from runtime.host import roce_gid

DEVICE, NETDEV, ADDRESS = "rocep1s0f0", "enp1s0f0np0", "198.18.0.1"
MAPPED = "0000:0000:0000:0000:0000:ffff:c612:0001"
ZERO = "0000:0000:0000:0000:0000:0000:0000:0000"


class Port:
    """sysfs and ``ip`` of one RDMA function whose GID ``index`` holds ``slot``.

    Deleting the address empties every entry that carries it; adding it
    registers its RoCE v2 GID at index 3, the lowest free index, unless
    ``refill`` is false because another holder still references the old entry.
    """

    def __init__(self, root, slot=ADDRESS, owner=NETDEV, kind="RoCE v2", refill=True, index=3):
        self.root, self.refill, self.commands = root, refill, []
        base = root / "sys/class/infiniband" / DEVICE
        (base / "device/net" / NETDEV).mkdir(parents=True)
        self.port = base / "ports/1"
        (self.port / "gid_attrs/ndevs").mkdir(parents=True)
        (self.port / "gid_attrs/types").mkdir(parents=True)
        (self.port / "gids").mkdir()
        for number in range(8):
            self.write(number, None, None, None)
        self.write(index, slot, owner, kind)

    def write(self, index, slot, owner, kind):
        (self.port / f"gids/{index}").write_text((MAPPED if slot else ZERO) + "\n")
        for name, value in (("ndevs", owner), ("types", kind)):
            path = self.port / "gid_attrs" / name / str(index)
            if slot and value:
                path.write_text(value + "\n")
            elif path.exists():
                path.unlink()

    def call(self, argv, **kwargs):
        self.commands.append(argv)
        if argv[:4] == ["ip", "-j", "-4", "addr"]:
            info = [{"local": ADDRESS, "prefixlen": 24, "broadcast": "198.18.0.255", "noprefixroute": True}]
            return SimpleNamespace(stdout=json.dumps([{"addr_info": info}]))
        if argv[:3] == ["ip", "addr", "del"]:
            for path in (self.port / "gids").iterdir():
                if path.read_text().strip() == MAPPED:
                    self.write(int(path.name), None, None, None)
        if argv[:3] == ["ip", "addr", "add"] and self.refill:
            self.write(3, ADDRESS, NETDEV, "RoCE v2")
        return SimpleNamespace(stdout="")

    def changes(self):
        return [argv for argv in self.commands if "show" not in argv]


READD = [["ip", "addr", "del", ADDRESS + "/24", "dev", NETDEV],
         ["ip", "addr", "add", ADDRESS + "/24", "broadcast", "198.18.0.255", "noprefixroute", "dev", NETDEV]]


def test_a_port_that_holds_its_address_is_left_alone(tmp_path):
    port = Port(tmp_path)
    assert roce_gid.locate([DEVICE], call=port.call, root=tmp_path) == [(NETDEV, ADDRESS, 3)]
    assert roce_gid.stale_ports([DEVICE], 3, call=port.call, root=tmp_path) == []
    assert roce_gid.serve([DEVICE], 3, call=port.call, root=tmp_path) == {"ok": True, "repaired": []}
    assert port.changes() == []


@pytest.mark.parametrize("slot,owner,kind", [(None, None, None), (ADDRESS, "enp1s0f1np1", "RoCE v2"),
                                             (ADDRESS, NETDEV, "IB/RoCE v1")])
def test_an_empty_or_foreign_index_is_re_added_with_its_prefix_and_flags(tmp_path, slot, owner, kind):
    port = Port(tmp_path, slot, owner, kind)
    assert roce_gid.stale_ports([DEVICE], 3, call=port.call, root=tmp_path) == [(NETDEV, ADDRESS)]
    assert roce_gid.serve([DEVICE], 3, call=port.call, root=tmp_path) == {"ok": True, "repaired": [NETDEV]}
    assert port.changes() == READD


def test_an_address_that_moved_is_reported_at_its_index_and_restored(tmp_path):
    # The cabled neighbor restarted under a running model: the address's RoCE
    # v2 GID returned at index 5 and index 3 is empty.
    port = Port(tmp_path, index=5)
    assert roce_gid.locate([DEVICE], call=port.call, root=tmp_path) == [(NETDEV, ADDRESS, 5)]
    with pytest.raises(ValueError, match=r"RoCE GID index 3 lacks the address of enp1s0f0np0 \(198\.18\.0\.1\): "
                                         r"its RoCE v2 GID is at index 5$"):
        roce_gid.check([DEVICE], 3, call=port.call, root=tmp_path)
    assert roce_gid.check([DEVICE], 5, call=port.call, root=tmp_path) == {"ok": True}
    assert roce_gid.serve([DEVICE], 3, call=port.call, root=tmp_path) == {"ok": True, "repaired": [NETDEV]}
    assert port.changes() == READD
    assert roce_gid.locate([DEVICE], call=port.call, root=tmp_path) == [(NETDEV, ADDRESS, 3)]


def test_an_entry_still_held_elsewhere_fails_after_the_settle_time(tmp_path):
    port = Port(tmp_path, None, None, None, refill=False)
    times = iter([0.0, 1.0, 11.0])
    with pytest.raises(ValueError, match=f"RoCE GID index 3 lacks the address of {NETDEV} \\({ADDRESS}\\): "
                                         "rocep1s0f0 port 1 has no RoCE v2 GID for IPv4 address 198.18.0.1"):
        roce_gid.serve([DEVICE], 3, call=port.call, root=tmp_path, sleep=lambda seconds: None,
                       clock=lambda: next(times))
