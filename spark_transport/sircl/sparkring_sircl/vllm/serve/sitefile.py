"""The site file as the serve launcher reads it: the ring harness's site plus two per-Spark keys.

The launcher reads the ring harness's site file (schema
``sircl-ring-site/v1``) through :class:`sparkring_sircl.ring.site.Site`, and
two keys of its own:

- ``model_path`` (ring entry): the absolute checkpoint directory on that
  Spark, for Sparks that hold the same checkpoint at different paths. It
  overrides ``--model-path PATH`` for that Spark; ``--model-path N=PATH``
  overrides it in turn (:func:`sparkring_sircl.vllm.serve.plan.resolve_model_paths`).
- ``sudo`` (top level, or a ring entry for that Spark): the command prefix of
  the launcher's host-side file operations on the Spark, outside containers:
  creating the cache and run directories, and the preflight checks of the
  model and cache directories (existence, file sizes, digests, free space).
  ``"sudo -n"`` runs them as root, for directories such as
  ``/srv/sparkring/<cluster>`` that only root may read; ``""`` runs them as
  the SSH user. Without the key, a Spark whose Docker command starts with
  ``sudo`` (for example ``"sudo -n docker"``) uses that command's words before
  the Docker executable (``"sudo -n"``); any other Spark runs them as the SSH
  user.

The ring harness ignores both keys. A ``model_path`` is 1 to 4,096 characters
of letters, digits and ``_.@:/+-``, starts with ``/`` and has no ``..``
component, because it becomes a ``docker run --mount`` source, whose syntax
reserves ``,`` and ``=``. A ``sudo`` value is empty or 1 to 8 words of
letters, digits and ``_.@:/+=-`` whose first word is ``sudo`` or a path ending
in ``/sudo``.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...ring.site import Site, SiteError

MODEL_PATH = "model_path"
SUDO = "sudo"
_PATH = re.compile(r"^/[A-Za-z0-9_.@:/+-]{0,4095}$")
_WORD = re.compile(r"^[A-Za-z0-9_.@:/+=-]+$")
_PLACEHOLDER = re.compile(r"REPLACE|<|>")
_MAX_WORDS = 8


def path_problem(value: object) -> str | None:
    """Why ``value`` cannot be a model directory, or None when it can."""
    if not isinstance(value, str) or _PLACEHOLDER.search(value) or not _PATH.match(value):
        return "must be an absolute path of letters, digits and _.@:/+-"
    if ".." in value.split("/"):
        return "must not contain a '..' component"
    return None


def sudo_problem(value: object) -> str | None:
    """Why ``value`` cannot prefix host-side file operations, or None when it can ("" is none)."""
    if not isinstance(value, str):
        return 'must be a string such as "sudo -n", or "" for none'
    words = value.split()
    if not words:
        return None
    if len(words) > _MAX_WORDS:
        return f"must hold at most {_MAX_WORDS} words"
    for word in words:
        if _PLACEHOLDER.search(word) or not _WORD.match(word):
            return f"has the word {word!r}, a placeholder or characters a shell would read"
    if words[0] != "sudo" and not words[0].endswith("/sudo"):
        return "must start with sudo (or a path ending in /sudo)"
    return None


def derived_sudo(docker: str) -> str:
    """The words of a Docker command before the Docker executable, when the command starts with sudo."""
    words = docker.split()
    if len(words) > 1 and (words[0] == "sudo" or words[0].endswith("/sudo")):
        return " ".join(words[:-1])
    return ""


@dataclasses.dataclass(frozen=True)
class ServeSite:
    site: Site
    model_paths: Mapping[int, str]       # ring position -> model directory from the site file
    sudo: Mapping[int, str]              # ring position -> prefix of host-side file operations ("" none)
    sudo_sources: Mapping[int, str]      # ring position -> where that prefix comes from

    @classmethod
    def of(cls, site: Site) -> "ServeSite":
        """A site without the launcher's keys: prefixes derived from each Spark's Docker command."""
        return cls.from_json({}, site=site)

    @classmethod
    def from_json(cls, document: Mapping[str, Any], *, site: Site | None = None) -> "ServeSite":
        site = site if site is not None else Site.from_json(document)
        entries = list(document.get("ring", []))
        problems = []
        top = document.get(SUDO)
        if top is not None:
            problem = sudo_problem(top)
            if problem:
                problems.append(f"{SUDO} {top!r} {problem}")
        paths: dict[int, str] = {}
        prefixes: dict[int, str] = {}
        sources: dict[int, str] = {}
        for index, host in enumerate(site.ring):
            entry = entries[index] if index < len(entries) and isinstance(entries[index], Mapping) else {}
            if MODEL_PATH in entry:
                value = entry[MODEL_PATH]
                problem = path_problem(value)
                if problem:
                    problems.append(f"ring entry {index} ({host.name}) {MODEL_PATH} {value!r} {problem}")
                else:
                    paths[index] = value.rstrip("/") or "/"
            if SUDO in entry:
                value = entry[SUDO]
                problem = sudo_problem(value)
                if problem:
                    problems.append(f"ring entry {index} ({host.name}) {SUDO} {value!r} {problem}")
                    continue
                prefixes[index], sources[index] = " ".join(value.split()), "site file, ring entry"
            elif top is not None and isinstance(top, str):
                prefixes[index], sources[index] = " ".join(top.split()), "site file"
            elif derived_sudo(host.docker):
                prefixes[index], sources[index] = derived_sudo(host.docker), "the Spark's Docker command"
            else:
                prefixes[index], sources[index] = "", "default"
        if problems:
            raise SiteError("; ".join(problems))
        return cls(site, paths, prefixes, sources)

    @classmethod
    def load(cls, path: str | Path) -> "ServeSite":
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise SiteError(f"site file {path} is not valid JSON: {error}") from None
        return cls.from_json(document)
