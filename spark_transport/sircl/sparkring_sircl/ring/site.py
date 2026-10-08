"""The operator's description of the Sparks the harness may use.

A site file (JSON, schema ``sircl-ring-site/v1``) names the serving image, the
wired-LAN interface that carries the control exchange, a TCP port for it, the
remote working directory, and the ring: every Spark in cabling order (entry
``i``'s port 0 is cabled to entry ``i + 1``'s port 1, wrapping around), each
with its SSH target and its wired-LAN address. Addresses and SSH users are
site data and stay out of version control; ``site.example.json`` shows the
form with placeholders.

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
from collections.abc import Mapping
from pathlib import Path

SCHEMA = "sircl-ring-site/v1"
DEFAULT_DOCKER = "docker"
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

    @property
    def size(self) -> int:
        return len(self.ring)

    def host(self, position: int) -> Host:
        return self.ring[position]

    @classmethod
    def from_json(cls, document: Mapping[str, object]) -> "Site":
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
        hosts = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise SiteError(f"ring entry {index} is not an object")
            hosts.append(Host(str(entry.get("name", f"spark{index}")), str(entry.get("ssh", "")),
                              str(entry.get("lan_address", "")), _docker_text(entry.get("docker", docker))))
        site = cls(image, interface, port, remote_dir, tuple(hosts), None if gid is None else int(gid), docker)
        site.validate()
        return site

    @classmethod
    def load(cls, path: str | Path) -> "Site":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

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
