"""The operator's description of the Sparks the harness may use.

A site file (JSON, schema ``sircl-ring-site/v1``) names the serving image, the
wired-LAN interface that carries the control exchange, a TCP port for it, the
remote working directory, and the Sparks: every Spark in cabling order
(``ring``), each with its SSH target and its wired-LAN address. Addresses and
SSH users are site data and stay out of version control;
``site.example.json`` shows the form with placeholders.

The optional ``cabling`` field says how the listed Sparks are cabled; entry
``i`` of ``ring`` is fabric position ``i``:

- ``"ring"`` (the default when absent): entry ``i``'s port 0 is cabled to
  entry ``i + 1``'s port 1, wrapping around, so every Spark has both ports
  cabled;
- ``"path"``: the same cables without the one that closes the ring, so the
  first entry's port 1 and the last entry's port 0 are free (a line of
  Sparks; two entries are a pair cabled port 0 to port 1);
- a list of cables written ``"<position>.port<p>-<position>.port<q>"``, for
  example ``["0.port0-1.port0"]`` for two Sparks cabled port 0 to port 0. The
  cables must join every listed Spark into one path or cycle, each port
  holding at most one cable.

The ring harness plans on every cabling (``Site.load(path,
cablings=CABLINGS)``). Other readers of the site file model ring cabling only
and take the default ``cablings=("ring",)``, which refuses a site cabled any
other way instead of planning lanes over cables that do not exist.

The optional ``docker`` field is the command that runs Docker on the Sparks,
``docker`` when absent. A ring entry's own ``docker`` field overrides it for
that Spark, for example ``"sudo -n docker"`` where the SSH user is not in the
docker group and passwordless sudo is allowed. Every Docker invocation of the
harness on a Spark starts with that Spark's command. It is 1 to 8 words of
letters, digits and ``_.@:/+=-`` (nothing a shell interprets), and its last
word is the Docker executable: ``docker`` or a path ending in ``/docker``.

The control exchange (setup records, verdicts, barriers) runs over the wired
LAN, never over fabric addresses: NIC relays forward only tagged RDMA traffic,
and a fabric address reaches only cabled neighbours.
"""

from __future__ import annotations

import dataclasses
import ipaddress
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from .. import routes as routes_mod

SCHEMA = "sircl-ring-site/v1"
DEFAULT_DOCKER = "docker"
# The cablings a site file can state: ``ring`` and ``path`` by name, ``cables`` as an explicit cable list.
CABLINGS = ("ring", "path", "cables")
_PLACEHOLDER = re.compile(r"REPLACE|<|>")
_SAFE = re.compile(r"^[A-Za-z0-9_.@:/+-]+$")
_DOCKER_WORD = re.compile(r"^[A-Za-z0-9_.@:/+=-]+$")
_DOCKER_MAX_WORDS = 8


class SiteError(ValueError):
    """The site description is incomplete or inconsistent."""


def docker_problem(value: object) -> str | None:
    """Why ``value`` cannot be a Docker command prefix, or None when it can."""
    if not isinstance(value, str):
        return 'must be a string such as "docker" or "sudo -n docker"'
    words = value.split()
    if not 1 <= len(words) <= _DOCKER_MAX_WORDS:
        return f"must hold 1 to {_DOCKER_MAX_WORDS} words"
    for word in words:
        if _PLACEHOLDER.search(word) or not _DOCKER_WORD.match(word):
            return f"has the word {word!r}, a placeholder or characters a shell would read"
    if words[-1] != "docker" and not words[-1].endswith("/docker"):
        return "must end with the Docker executable (docker, or a path ending in /docker)"
    return None


def _docker_text(value: object) -> object:
    """``value`` with its words separated by single spaces (other types pass through to validation)."""
    return " ".join(value.split()) if isinstance(value, str) else value


def _cabling_of(value: object) -> tuple[str, tuple[str, ...]]:
    """(cabling, cable texts) of a site file's ``cabling`` field."""
    if value is None or value == "ring":
        return "ring", ()
    if value == "path":
        return "path", ()
    if isinstance(value, Sequence) and not isinstance(value, str) and value \
            and all(isinstance(item, str) for item in value):
        return "cables", tuple(" ".join(item.split()) for item in value)
    raise SiteError('cabling must be "ring", "path" or a list of cables such as ["0.port0-1.port0"]')


@dataclasses.dataclass(frozen=True)
class Host:
    name: str
    ssh: str
    lan_address: str
    docker: str = DEFAULT_DOCKER


@dataclasses.dataclass(frozen=True)
class Site:
    image: str
    lan_interface: str
    control_port: int
    remote_dir: str
    ring: tuple[Host, ...]
    gid_index: int | None = None
    docker: str = DEFAULT_DOCKER
    cabling: str = "ring"            # one of CABLINGS
    cables: tuple[str, ...] = ()     # the cable texts of cabling "cables", else empty

    @property
    def size(self) -> int:
        return len(self.ring)

    def host(self, position: int) -> Host:
        return self.ring[position]

    def fabric(self) -> routes_mod.Fabric:
        """Every cable of the site between the listed Sparks (positions are entry indices)."""
        if self.cabling == "ring":
            return routes_mod.Fabric.ring(self.size)
        if self.cabling == "path":
            return routes_mod.Fabric.path(range(self.size))
        return routes_mod.Fabric.parse(self.cables)

    def cabled_ports(self, position: int) -> tuple[int, ...]:
        """The ports of the Spark at ``position`` that hold a cable of the site."""
        fabric = self.fabric()
        return tuple(port for port in (0, 1) if fabric.cable_at(position, port) is not None)

    def cabling_text(self) -> str:
        """The cabling in words, for plans and errors."""
        if self.cabling == "ring":
            return f"a ring of {self.size} (entry i port 0 to entry i + 1 port 1, wrapping around)"
        if self.cabling == "path":
            return f"a path of {self.size} (entry i port 0 to entry i + 1 port 1, the ends' outer ports free)"
        return f"the cables {', '.join(self.cables)}"

    @classmethod
    def from_json(cls, document: Mapping[str, object], *, cablings: Sequence[str] = ("ring",)) -> "Site":
        """The site of ``document``; a cabling outside ``cablings`` is refused (the caller plans those only)."""
        if document.get("schema") != SCHEMA:
            raise SiteError(f"site schema must be {SCHEMA}")
        try:
            image = str(document["image"])
            interface = str(document["lan_interface"])
            port = int(document["control_port"])
            remote_dir = str(document.get("remote_dir", "/tmp/sircl-ring"))
            entries = list(document["ring"])
        except (KeyError, TypeError, ValueError) as error:
            raise SiteError(f"site file is missing or malforms a field: {error}") from None
        gid = document.get("gid_index")
        docker = _docker_text(document.get("docker", DEFAULT_DOCKER))
        cabling, cables = _cabling_of(document.get("cabling"))
        hosts = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise SiteError(f"ring entry {index} is not an object")
            hosts.append(Host(str(entry.get("name", f"spark{index}")), str(entry.get("ssh", "")),
                              str(entry.get("lan_address", "")), _docker_text(entry.get("docker", docker))))
        site = cls(image, interface, port, remote_dir, tuple(hosts), None if gid is None else int(gid), docker,
                   cabling, cables)
        site.validate()
        if site.cabling not in cablings:
            raise SiteError(f"the site's Sparks are cabled as {site.cabling_text()} (cabling {site.cabling!r}); "
                            f"this command plans cabling {' or '.join(repr(name) for name in cablings)} only")
        return site

    @classmethod
    def load(cls, path: str | Path, *, cablings: Sequence[str] = ("ring",)) -> "Site":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")), cablings=cablings)

    def validate(self) -> None:
        problems = []
        for field, value in (("image", self.image), ("lan_interface", self.lan_interface),
                             ("remote_dir", self.remote_dir)):
            if not value or _PLACEHOLDER.search(value) or not _SAFE.match(value):
                problems.append(f"{field} {value!r} is a placeholder or holds characters a shell would read")
        problem = docker_problem(self.docker)
        if problem:
            problems.append(f"docker {self.docker!r} {problem}")
        if not 1024 <= self.control_port <= 65000:
            problems.append("control_port must be between 1024 and 65000")
        if not self.remote_dir.startswith("/"):
            problems.append("remote_dir must be an absolute path on the Sparks")
        if not 2 <= self.size <= 16:
            problems.append(f"the ring lists {self.size} Sparks; 2 to 16 are supported")
        elif self.cabling == "cables":
            problems.extend(self._cable_problems())
        seen_names, seen_addresses = set(), set()
        for index, host in enumerate(self.ring):
            if not host.ssh or _PLACEHOLDER.search(host.ssh) or not _SAFE.match(host.ssh):
                problems.append(f"ring entry {index} ({host.name}) needs an SSH target")
            problem = docker_problem(host.docker)
            if problem:
                problems.append(f"ring entry {index} ({host.name}) docker {host.docker!r} {problem}")
            try:
                address = ipaddress.IPv4Address(host.lan_address)
            except ValueError:
                problems.append(f"ring entry {index} ({host.name}) needs a wired-LAN IPv4 address")
                continue
            if address in seen_addresses:
                problems.append(f"wired-LAN address {address} appears twice")
            if host.name in seen_names:
                problems.append(f"Spark name {host.name} appears twice")
            seen_addresses.add(address)
            seen_names.add(host.name)
        if self.gid_index is not None and not 0 <= self.gid_index <= 255:
            problems.append("gid_index must be in 0-255")
        if problems:
            raise SiteError("; ".join(problems))

    def _cable_problems(self) -> list[str]:
        """Why an explicit cable list does not join the listed Sparks into one path or cycle."""
        try:
            fabric = self.fabric()
        except routes_mod.RouteError as error:
            return [f"cabling: {error}"]
        outside = sorted({p for p in fabric.positions if not 0 <= p < self.size})
        if outside:
            return [f"cabling names positions {outside}; the ring lists positions 0 to {self.size - 1}"]
        missing = sorted(set(range(self.size)) - set(fabric.positions))
        if missing:
            return [f"cabling leaves positions {missing} without a cable"]
        reached, frontier = {0}, [0]
        while frontier:
            here = frontier.pop()
            for port in (0, 1):
                step = fabric.cross(here, port)
                if step is not None and step.dst not in reached:
                    reached.add(step.dst)
                    frontier.append(step.dst)
        if len(reached) != self.size:
            return [f"cabling splits the Sparks: positions {sorted(set(range(self.size)) - reached)} are not "
                    "reached from position 0"]
        return []
