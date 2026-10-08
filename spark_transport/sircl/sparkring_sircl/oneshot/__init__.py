"""SIRCL ring sessions: the session class, its kernels and the native binding.

Exports (resolved on first use, so the torch-free native binding
``sparkring_sircl.oneshot._proxy`` imports without torch or CUTLASS):

- ``AllReduce`` (also ``RoceOneshotAllReduce``): the session class, with
  ``should_allreduce``, ``select_algorithm``, ``prepare``, ``all_reduce``,
  ``should_all_gather``, ``all_gather``, ``should_reduce_scatter``,
  ``reduce_scatter``, ``should_all_to_all``, ``all_to_all``, ``capture``,
  ``check_health``, ``stats`` and ``close``;
- ``API_VERSION``, ``SUPPORTED_DTYPES``, ``SUPPORTED_WORLD_SIZES``,
  ``DEFAULT_MAX_SIZE``, ``DEFAULT_MAX_GATHER_BYTES``, ``ALGORITHMS``,
  ``ALGORITHM_CHOICES``, ``SCATTER_MODES``, ``MAX_LANES``, ``MAX_DEVICES``;
- ``is_supported``, ``discover_hcas``, ``default_gid_index``;
- the modules ``runtime``, ``_compile`` and ``_cute_intrinsics``.
"""

from __future__ import annotations

import importlib
from typing import Any

_FROM_RUNTIME = (
    "ALGORITHMS", "ALGORITHM_CHOICES", "API_VERSION", "AllReduce", "DEFAULT_MAX_GATHER_BYTES",
    "DEFAULT_MAX_SIZE", "MAX_DEVICES", "MAX_LANES", "RoceOneshotAllReduce", "SCATTER_MODES",
    "SUPPORTED_DTYPES", "SUPPORTED_WORLD_SIZES", "default_gid_index", "discover_hcas", "is_supported",
)
_MODULES = ("runtime", "_compile", "_cute_intrinsics", "_oneshot_cute", "_allgather_cute", "_proxy")

__all__ = list(_FROM_RUNTIME)


def __getattr__(name: str) -> Any:
    if name in _FROM_RUNTIME:
        return getattr(importlib.import_module(".runtime", __name__), name)
    if name in _MODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
