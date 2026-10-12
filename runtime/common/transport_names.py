"""The names people read for SparkRing's collective transports.

A transport has an identifier that commands, deployment locks, image locks and
status JSON keep unchanged (``--transport prepared``, ``transports: [...]``,
``tp_collective_transport: rocenante``), and a display name that installer
messages, ``sparkring status`` and the dashboard print. This module is the one
place that maps the first to the second. It imports only the standard library,
so the dashboard service can use it without loading the installer.

The ``prepared`` transport is RoCEnante: the RoCE all-reduce of Local Inference
Lab's b12x (``b12x.comm.roce``) with SparkRing's adaptive peer-path integration,
driven through b12x's prepared-execution API, which gives the identifier its
name. Workers that run it report ``tp_collective_transport`` as ``rocenante``.
"""
from __future__ import annotations

PREPARED_NAME = "RoCEnante (b12x)"

# Installer transport identifiers (--transport values and lock `backend`/`transports` entries) whose display
# name differs from the identifier. Installer messages print sircl, nccl and libsircl as they are typed.
DISPLAY_NAMES = {"prepared": PREPARED_NAME}

# Values of runtime-status's per-worker `tp_collective_transport`, as the dashboard shows them.
COLLECTIVE_NAMES = {"sircl": "SIRCL", "rocenante": PREPARED_NAME, "nccl": "NCCL"}


def display_name(identifier) -> str:
    """The name people read for an installer transport identifier; an unknown identifier reads as itself."""
    return DISPLAY_NAMES.get(str(identifier), str(identifier))


def collective_name(value) -> str:
    """The name people read for a runtime-status ``tp_collective_transport`` value; an unknown value reads as itself."""
    return COLLECTIVE_NAMES.get(str(value), str(value))
