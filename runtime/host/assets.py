"""This Spark's checkpoint survey for one profile, printed by ``sparkring node assets``.

The survey (``runtime.host.checkpoint_search``) only reads: it looks for copies
of the profile's pinned checkpoint on this Spark and classifies them by content
evidence. ``sparkring install`` runs the same survey on every Spark before it
plans the checkpoint; this module runs it locally, for diagnostics, and returns
a summary without per-file entries.
"""
from pathlib import Path
import re

from runtime.common import installer, setup
from runtime.host import checkpoint_search

CONTROLLER_CLUSTER = "/var/lib/sparkring/controller/cluster.json"
WORKSPACES = "/srv/sparkring"
# The cluster name sparkring setup uses when none is given.
DEFAULT_CLUSTER = "sparkring"
CANDIDATE_KEYS = ("path", "layout", "commit", "branches", "home", "sparkring", "found_by", "counts")
SEARCH_KEYS = ("complete", "stopped", "passes", "seconds", "entries", "unvisited", "skipped_mounts",
               "large_directories", "unreadable")


def cluster_name(root="/"):
    """The name of the cluster this Spark belongs to, which names its checkpoint directories.

    Node A records it in the controller's cluster record. Every Spark of a
    cluster holds that cluster's workspace ``/srv/sparkring/<name>``; when the
    record is absent and exactly one such workspace exists, its name is used.
    Otherwise the name is setup's default, ``sparkring``.
    """
    base = Path(root)
    try:
        name = installer.read(base / CONTROLLER_CLUSTER.lstrip("/"))["name"]
        if isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9-]{0,39}", name):
            return name
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        names = [entry.name for entry in (base / WORKSPACES.lstrip("/")).iterdir()
                 if entry.is_dir() and not entry.is_symlink() and not entry.name.endswith("-managed")
                 and re.fullmatch(r"[a-z][a-z0-9-]{0,39}", entry.name)]
    except OSError:
        names = []
    return names[0] if len(names) == 1 else DEFAULT_CLUSTER


def summary(profile, cluster, survey):
    """The survey without per-file entries: where copies are, their layout and how many files match."""
    owned = survey.get("owned") or {}
    return {"schema": "sparkring-node-assets/v1", "profile": profile, "cluster": cluster, "host": survey.get("host"),
            "repository": survey.get("repository"), "revision": survey.get("revision"),
            "owned": {"path": owned.get("path"), "state": owned.get("state"), "fstype": owned.get("fstype"),
                      "free_bytes": owned.get("free_bytes"), "files": len(owned.get("files") or {})},
            "docker": survey.get("docker"),
            "candidates": [{key: candidate.get(key) for key in CANDIDATE_KEYS} for candidate in survey.get("candidates") or []],
            "not_used": survey.get("not_used") or [], "named": survey.get("named") or [],
            "search": {key: (survey.get("search") or {}).get(key) for key in SEARCH_KEYS},
            "errors": len((survey.get("search") or {}).get("errors") or [])}


def discover(profile, *, cluster=None, root="/"):
    """Survey this Spark for the profile's pinned checkpoint and return the summary; never writes.

    ``cluster`` names the cluster whose checkpoint directory is reported under
    ``owned`` (default: ``cluster_name``). ``root`` prefixes every host path,
    for tests against a fixture tree.
    """
    card = setup.selection(profile)
    pins = installer.checkpoint_pins(card)
    name = cluster or cluster_name(root)
    owned = installer.checkpoint_directory(name, card)
    result = checkpoint_search.survey(pins, checkpoint_search.options(owned=owned, root=root))
    return summary(profile, name, result)
