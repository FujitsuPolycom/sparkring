"""Layouts of the relay plan installer: which Sparks of a ring form which groups.

A layout is a set of disjoint groups on one physical ring of DGX Sparks: ring
position ``i``'s port 0 is cabled to position ``i + 1``'s port 1, wrapping
around, in the order of the site file. A group is either the whole ring or
consecutive Sparks along it, including runs that wrap from the last position
to the first (``7,0,1,2``). A group owns the cables between its members and no
other cable. Its fabric is built by the ring harness's
:func:`sparkring_sircl.ring.plan.group_layout`, so this installer, the ring
harness and SIRCL sessions derive lanes from the same fabric.

Built-in layouts of an eight-Spark ring:

- ``ring8``: one group of all eight Sparks, the universal relay table;
- ``2xTP4``: two groups, Sparks 0-3 and Sparks 4-7;
- ``4xTP2``: four adjacent pairs, 0-1, 2-3, 4-5 and 6-7 (no relays).

Custom layouts name their groups: ``0-3;4-7``, ``7,0,1,2``, or ``7-2`` (a range
whose end is smaller than its start wraps around the ring).

Relay egress. A relay forwards a frame out of its other port, either through
the function of the lane's own class on that port (``same``) or through the
sibling Socket Direct function of the other class on that port (``sibling``).
Either way the relay writes the next Spark's MAC of the lane's class into the
frame, so the frame arrives on the same function. Measured on the ring:
relaying out of the sibling function lowered burst loss to 0.06 % at 2 MiB.
A layout whose only group is the whole ring uses ``same``: that is the
universal relay table ``ring8`` names, whose objects deployments built on the
table rely on. Every other layout uses ``sibling``. ``--relay-egress``
overrides either default.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from .. import routes
from ..ring import plan as ring_plan

RELAY_EGRESS = ("same", "sibling")
BUILTIN_LAYOUTS = ("ring8", "2xTP4", "4xTP2")


class FabricError(ValueError):
    """A layout, the facts of its Sparks or its relay plan breaks a rule of the installer."""


@dataclasses.dataclass(frozen=True)
class Group:
    """One group: its members (ring positions, in rank order) on a ring of ``ring_size`` Sparks."""

    members: tuple[int, ...]
    ring_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", tuple(int(member) for member in self.members))
        self.layout()

    def layout(self) -> routes.Layout:
        """The group's fabric and rank positions, exactly as the ring harness builds them."""
        try:
            return ring_plan.group_layout(self.ring_size, self.members)
        except (ring_plan.PlanError, routes.RouteError) as error:
            raise FabricError(str(error)) from None

    @property
    def whole_ring(self) -> bool:
        return len(self.members) == self.ring_size

    @property
    def order(self) -> tuple[int, ...]:
        """Members in fabric order: the whole ring from position 0, a path from its first member."""
        return tuple(range(self.ring_size)) if self.whole_ring else self.members

    @property
    def kind(self) -> str:
        if self.whole_ring:
            return "cycle"
        return "pair" if len(self.members) == 2 else "path"

    @property
    def label(self) -> str:
        """The group's identity, for example ``cycle:0-1-2-3-4-5-6-7`` or ``path:7-0-1-2``."""
        return f"{self.kind}:{'-'.join(str(position) for position in self.order)}"

    @property
    def cables(self) -> tuple[str, ...]:
        return tuple(cable.text for cable in self.layout().fabric.cables)


@dataclasses.dataclass(frozen=True)
class FabricLayout:
    """Disjoint groups on one ring and the relay egress of their relays."""

    name: str
    ring_size: int
    groups: tuple[Group, ...]
    relay_egress: str

    def __post_init__(self) -> None:
        if self.relay_egress not in RELAY_EGRESS:
            raise FabricError(f"relay egress must be one of {', '.join(RELAY_EGRESS)}, got {self.relay_egress!r}")
        if not self.groups:
            raise FabricError(f"layout {self.name} has no groups")
        seen: dict[int, str] = {}
        cables: dict[str, str] = {}
        for group in self.groups:
            if group.ring_size != self.ring_size:
                raise FabricError(f"group {group.label} is on a ring of {group.ring_size}, the layout on "
                                  f"{self.ring_size}")
            for member in group.members:
                if member in seen:
                    raise FabricError(f"layout {self.name}: Spark {member} belongs to groups {seen[member]} and "
                                      f"{group.label}; a Spark belongs to at most one group")
                seen[member] = group.label
            for cable in group.cables:
                if cable in cables:
                    raise FabricError(f"layout {self.name}: cable {cable} is owned by {cables[cable]} and {group.label}")
                cables[cable] = group.label

    @property
    def positions(self) -> tuple[int, ...]:
        return tuple(sorted(member for group in self.groups for member in group.members))

    def select(self, selector: str | None) -> tuple[Group, ...]:
        """The groups an operation acts on: all of them, or the one ``selector`` names.

        ``selector`` is a group index (``0``, ``1``, ...), a label
        (``path:4-5-6-7``) or the group's members in the custom syntax (``4-7``).
        """
        if selector is None:
            return self.groups
        text = selector.strip()
        if text.isdigit() and int(text) < len(self.groups):
            return (self.groups[int(text)],)
        for group in self.groups:
            if text == group.label:
                return (group,)
        try:
            wanted = parse_groups(text, self.ring_size)
        except FabricError:
            wanted = ()
        for group in self.groups:
            if len(wanted) == 1 and set(wanted[0]) == set(group.members):
                return (group,)
        raise FabricError(f"layout {self.name} has no group {selector!r}; its groups are "
                          + ", ".join(f"{index} ({group.label})" for index, group in enumerate(self.groups)))


def parse_groups(text: str, ring_size: int) -> tuple[tuple[int, ...], ...]:
    """``0-3;4-7``, ``7,0,1,2`` or ``7-2``: groups separated by ``;``, members by ``,``.

    A range ``a-b`` lists ``a, a+1, ..., b`` along the ring; when ``b < a`` it
    wraps from the last position to position 0.
    """
    groups = []
    for part in text.split(";"):
        members: list[int] = []
        for item in part.split(","):
            item = item.strip()
            if not item:
                continue
            first_text, _, last_text = item.partition("-")
            try:
                first = int(first_text)
                last = int(last_text) if last_text else first
            except ValueError:
                raise FabricError(f"group {part.strip()!r} is not a list of Spark positions") from None
            if not (0 <= first < ring_size and 0 <= last < ring_size):
                raise FabricError(f"group {part.strip()!r} names a position outside the ring of {ring_size}")
            span = (last - first) % ring_size
            members.extend((first + step) % ring_size for step in range(span + 1))
        if members:
            groups.append(tuple(members))
    if not groups:
        raise FabricError("no groups given")
    return tuple(groups)


def default_relay_egress(groups: Sequence[Group]) -> str:
    """``same`` for a layout whose only group is the whole ring, ``sibling`` otherwise."""
    return "same" if len(groups) == 1 and groups[0].whole_ring else "sibling"


def builtin(name: str, ring_size: int) -> tuple[str, tuple[tuple[int, ...], ...]]:
    """Canonical name and member lists of a built-in layout."""
    key = name.strip().lower()
    if key in ("ring", f"ring{ring_size}"):
        return f"ring{ring_size}", (tuple(range(ring_size)),)
    if key in ("2xtp4", "4xtp2"):
        if ring_size != 8:
            raise FabricError(f"layout {name} is defined on a ring of 8 Sparks; the site lists {ring_size}")
        if key == "2xtp4":
            return "2xTP4", ((0, 1, 2, 3), (4, 5, 6, 7))
        return "4xTP2", ((0, 1), (2, 3), (4, 5), (6, 7))
    raise FabricError(f"unknown layout {name!r}; built-in layouts are {', '.join(BUILTIN_LAYOUTS)} "
                      "(ring<N> for the whole ring of another size), or give --groups")


def resolve(*, ring_size: int, layout: str | None = None, groups: str | None = None, name: str | None = None,
            relay_egress: str | None = None) -> FabricLayout:
    """The layout of an operation: a built-in name, or custom groups (``groups``) with an optional name."""
    if (layout is None) == (groups is None):
        raise FabricError("give exactly one of --layout and --groups")
    if layout is not None:
        canonical, member_lists = builtin(layout, ring_size)
    else:
        member_lists = parse_groups(groups, ring_size)
        canonical = name or "custom"
    built = tuple(Group(members, ring_size) for members in member_lists)
    return FabricLayout(canonical, ring_size, built, relay_egress or default_relay_egress(built))
