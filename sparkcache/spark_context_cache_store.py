"""Fail-closed loader for the persistent context-cache storage engine.

The engine is single-sourced at
``sparkcache/persistent_context_cache/cache_manifest.py``.
This shim resolves it relative to this file so the same import works from
the repository tree and from a staged bundle. Import failure is a hard
error: a build that stages the connector without its storage engine must
not come up.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_CANDIDATES = (
    # Staged bundle layout (canonical): engine beneath the sparkcache package.
    _HERE.parent / "persistent_context_cache" / "cache_manifest.py",
    # Repository layout.
    _HERE.parents[2] / "experiments" / "persistent_context_cache" / "cache_manifest.py",
)
_ENGINE = next((path for path in _CANDIDATES if path.is_file()), None)

if _ENGINE is None:
    raise ImportError(
        "spark-context-cache storage engine absent; tried: "
        + ", ".join(str(path) for path in _CANDIDATES)
    )

_spec = importlib.util.spec_from_file_location("spark_context_cache_engine", _ENGINE)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("spark_context_cache_engine", _module)
_spec.loader.exec_module(_module)

CacheIdentity = _module.CacheIdentity
CommitConflict = _module.CommitConflict
ContextChunk = _module.ContextChunk
IncompleteEntry = _module.IncompleteEntry
LookupResult = _module.LookupResult
ManifestStore = _module.ManifestStore
StateRecord = _module.StateRecord

__all__ = [
    "CacheIdentity",
    "CommitConflict",
    "ContextChunk",
    "IncompleteEntry",
    "LookupResult",
    "ManifestStore",
    "StateRecord",
]
