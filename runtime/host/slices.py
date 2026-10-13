"""On-demand admission through the legacy ring gate and per-position file locks.

There is no queue worker or polling. File descriptors, not PID records, own
locks. A failed admission closes every descriptor it acquired.
"""
import contextlib
import json
import os
import time
import uuid
from pathlib import Path

from runtime.common import process_lock
from runtime.host import placement


def label(positions):
    return "the whole ring" if positions is None else "Sparks " + ",".join(map(str, positions))


def _owner(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"pid": value}
    except (OSError, ValueError):
        return {}


def _held(path, *, shared=False):
    if not path.exists():
        return False
    try:
        with process_lock.hold(path, shared=shared):
            return False
    except ValueError:
        return True


def active(root):
    """Read live owners once. Stale metadata counts for nothing."""
    root = Path(root)
    gate = root / "install.lock"
    if _held(gate, shared=True):
        return [{"operation": "ring-wide operation", **_owner(gate), "positions": None}]
    found = {}
    for path in sorted((root / "slice-locks").glob("spark-*.lock")):
        owner = _owner(path)
        if _held(path):
            key = owner.get("id", path.name)
            found[key] = {"positions": [int(path.stem.split("-")[1])],
                          "operation": "operation", **owner}
    return list(found.values())


def conflict(root, positions):
    return [row for row in active(root) if placement.overlap(positions, row.get("positions"))]


def conflict_error(root, positions):
    holders = conflict(root, positions)
    detail = "; ".join(f"{row['operation']} (pid {row.get('pid', 'unknown')}) holds {label(row['positions'])}"
                       for row in holders)
    error = ValueError("Another operation is active. Blocked: " + (detail or "another operation holds the ring gate or these Sparks")
                       + "; retry after it finishes")
    error.activity = {"schema": "sparkring-activity/v1", "grade": "blocked", "positions": positions,
                      "reasons": holders, "message": str(error)}
    return error


def affected(root, positions):
    """Include whole deployments whose stop or rollback a replacement can cause."""
    if positions is None:
        return None
    result = set(positions)
    for slot in placement.slots(root):
        if not placement.overlap(slot, positions):
            continue
        journal = placement.journal(root, slot)
        if placement.exists(placement.recorded(root, slot)) or (journal and not journal.get("complete")):
            if slot is None:
                return None
            result.update(slot)
    return tuple(sorted(result))


@contextlib.contextmanager
def hold(root, positions, operation, *, expand=False):
    root = Path(root)
    with contextlib.ExitStack() as stack:
        # The admission lock prevents two partial acquisitions and protects
        # the snapshot used to expand a replacement's affected positions.
        with process_lock.hold(root / "admission.lock", wait=True):
            selected = affected(root, positions) if expand else positions
            owner = {"id": uuid.uuid4().hex, "pid": os.getpid(), "operation": operation,
                     "positions": list(selected) if selected is not None else None, "since": time.time()}
            try:
                stack.enter_context(process_lock.hold(root / "install.lock", shared=selected is not None, owner=owner))
                for position in sorted(set(selected or ())):
                    stack.enter_context(process_lock.hold(root / "slice-locks" / f"spark-{position}.lock", owner=owner))
            except ValueError:
                stack.close()
                raise conflict_error(root, selected) from None
        yield owner


def busy(root, positions):
    return bool(conflict(root, positions))
