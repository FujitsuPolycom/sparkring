"""One status line and one JSON receipt per rank and vLLM group.

Every SIRCL communicator logs a receipt line when its group is set up and can
write the same record as JSON (``SIRCL_RECEIPT_DIR/rank<global>-<group>.json``).
The record states what this rank will do with each collective of the group:

- where the group sits (layout, fabric, positions) and what NCCL may do there,
  including whether PyNccl was built;
- the ring session's identity and agreed limits (lanes, devices, capacity,
  dispatch ceiling, all-gather capacity, relay-safe op size and its basis);
- the group's point-to-point channels (``p2p``: the peers this rank has a
  channel with, the slots and the forward windows, ``shared:<group>``, or
  ``none`` with the reason in ``p2p_detail``);
- on a group with a session, ``column_gather`` (``on`` or ``off``) and
  ``column_gather_detail``: the column gathers staged on the session's links
  per route and the staging bytes kept (``executor.ColumnGather``);
- the plan counters (:meth:`Counters.snapshot`): calls per collective,
  backend and method, refreshed whenever the receipt is written again
  (point-to-point rows ``send/sircl/direct``, ``recv/sircl/relayed``,
  ``torch.isend/sircl/...`` and ``torch.broadcast/sircl/p2p`` show what
  carried each transfer);
- the directories the process imported ``vllm`` and ``b12x`` from
  (:func:`package_dir`), which show whether a source overlay serves.

A group rewrites its receipt from the worker's post-step check: with fresh
session statistics after a step in which a new (collective, backend, method)
row first appeared or the flag-wait regime changed, and at least every
:data:`REFRESH_SECONDS`; with the current counts alone (statistics of the last
refresh) at most every :data:`COUNT_REFRESH_SECONDS` while counts change; and
when it closes. The file on disk therefore lists every row that has occurred,
with counts at most :data:`COUNT_REFRESH_SECONDS` older than the last step
that ran a post-step check.

A run's evidence therefore shows, per rank, that no collective of a relayed
group reached NCCL (no ``nccl`` backend row for that group).
"""

from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "sircl-vllm-receipt/v1"
REFRESH_SECONDS = 60.0
COUNT_REFRESH_SECONDS = 1.0


class Counters:
    """Calls per (collective, backend, method), thread-safe."""

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str, str], int] = {}
        self._reasons: dict[tuple[str, str, str], str] = {}
        self._lock = threading.Lock()
        self.version = 0                 # calls counted so far

    def count(self, key: tuple[str, str, str], reason: str = "") -> bool:
        """Count one call; True when ``key`` is counted for the first time."""
        with self._lock:
            new = key not in self._counts
            self._counts[key] = self._counts.get(key, 0) + 1
            self._reasons.setdefault(key, reason)
            self.version += 1
            return new

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"collective": key[0], "backend": key[1], "method": key[2], "calls": count,
                 "reason": self._reasons.get(key, "")}
                for key, count in sorted(self._counts.items())
            ]

    def total(self, *, backend: str | None = None, collective: str | None = None) -> int:
        with self._lock:
            return sum(count for key, count in self._counts.items()
                       if (backend is None or key[1] == backend)
                       and (collective is None or key[0] == collective))


def _value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value) or "-"
    if value is None:
        return "-"
    return str(value)


LINE_FIELDS = (
    "group", "global_rank", "rank", "world", "layout", "fabric", "positions", "nccl",
    "pynccl", "session", "lanes", "hcas", "capacity", "dispatch", "oneshot_max", "gather", "op_per_peer",
    "large_piece", "gather_piece", "schedules", "chain_min", "ring_min", "links", "tuning", "mhc", "fused_norm",
    "column_gather", "p2p", "wait", "vllm",
    "state",
)


def package_dir(name: str) -> str | None:
    """The directory of the top-level package ``name`` this process imported, or None when it has not imported
    it; reads ``sys.modules`` and imports nothing."""
    module = sys.modules.get(name)
    origin = getattr(module, "__file__", None)
    if origin:
        return os.path.dirname(origin)
    locations = list(getattr(module, "__path__", None) or ())
    return str(locations[0]) if locations else None


def line(record: Mapping[str, Any]) -> str:
    """``SIRCL receipt key=value ...`` with the fields of :data:`LINE_FIELDS`."""
    return "SIRCL receipt " + " ".join(f"{name}={_value(record.get(name))}" for name in LINE_FIELDS)


def write(record: Mapping[str, Any], directory: str | os.PathLike[str]) -> Path:
    """Write ``record`` as ``rank<global>-<group>.json`` atomically; returns the path."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    group = str(record.get("group", "group")).replace(":", "-")
    path = folder / f"rank{record.get('global_rank', 'x')}-{group}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"schema": SCHEMA, **record}, indent=2, sort_keys=True,
                                    default=str), encoding="utf-8")
    os.replace(temporary, path)
    return path
