"""The ``sparkring`` group: what an ordinary user may read, and what stays root's.

SparkRing's records are root's: the controller state under
``/var/lib/sparkring``, the administration network's configuration
(``/etc/sparkring/control.json``, mode 0600) and the deployment records. The
read-only commands do not need any of those records directly. Members of the
``sparkring`` group read the artifacts the package and the root commands
publish for them:

- the host agent's observation, ``/run/sparkring/status.json`` (mode 0640);
- the installation logs under ``/var/log/sparkring`` (mode 0640);
- the operator view of the controller's status,
  ``/var/lib/sparkring/operator-status.json`` (mode 0640), whose site
  addresses are redacted (``privilege.redact``).

SSH keys, tokens, the administration network's configuration and the
controller's own records stay root-only. Membership is granted with
``sudo usermod -aG sparkring USER``; the Debian package creates the group.
Every function here degrades gracefully on a host without the group (a
developer machine), leaving today's root-only behavior unchanged.
"""
from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Any

try:
    import grp
except ImportError:  # A host without POSIX groups (Windows CPU checks) has no sparkring group.
    grp = None

GROUP = "sparkring"

# An IPv4 site address: a fabric address (10.253.x.x), a management address or
# an API host inside a status document. Ports, ranks and hostnames stay.
_ADDRESS = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
REDACTED = "<site-address>"


def group_gid(name: str = GROUP):
    """The group's numeric id, or None on a host without the group."""
    if grp is None:
        return None
    try:
        return grp.getgrnam(name).gr_gid
    except (KeyError, OSError):
        return None


def running_as_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def member(gids=None) -> bool:
    """Whether this process may read the group's artifacts: root, or a member of GROUP."""
    if running_as_root():
        return True
    gid = group_gid()
    if gid is None:
        return False
    return gid in (os.getgroups() if gids is None else gids)


def reader_request() -> str:
    """How an operator gains read access, for error messages."""
    if group_gid() is None:
        return "run the command with sudo"
    return (f"join the {GROUP} group (sudo usermod -aG {GROUP} USER, then sign in again) "
            "or run the command with sudo")


def share(path: Path, mode: int = 0o640) -> bool:
    """Make a root-written file readable by GROUP: its group becomes GROUP and its mode ``mode``.

    False, and no change, when this process is not root or the host has no
    GROUP: a developer's tree and a non-root run keep the previous modes.
    """
    if not running_as_root():
        return False
    gid = group_gid()
    if gid is None:
        return False
    os.chown(path, -1, gid)
    path.chmod(mode)
    return True


def share_directory(path: Path, mode: int = 0o750) -> bool:
    """Give a root-written directory GROUP's group and the setgid bit, so files created in it
    inherit the group; ``mode`` carries the group-read and -enter bits. Same limits as ``share``."""
    if not running_as_root():
        return False
    gid = group_gid()
    if gid is None:
        return False
    os.chown(path, -1, gid)
    path.chmod(mode | stat.S_ISGID)
    return True


def redact(value) -> Any:
    """Replace every IPv4 site address in strings, lists and mappings with REDACTED.

    The operator view is written from records that hold management, fabric and
    API addresses; redaction keeps those root-only while the ranks, states,
    model names and ports the status shows stay. Non-string leaves pass through.
    """
    if isinstance(value, str):
        return _ADDRESS.sub(REDACTED, value)
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    return value
