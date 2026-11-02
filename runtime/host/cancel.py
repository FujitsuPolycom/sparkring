"""Cancelling an installation: one request record on Node A and the steps an installation honours it at.

``sudo sparkring install --cancel`` names the installations running at that
moment in ``install-cancel.json``, one entry per operation with the time and
the operator. A running installation reads its own entry before each step of
its model switch (``runtime.host.rollout``): it stops there, stops the
candidate deployment through its own adapter, brings back the model it
replaced and records the switch as ``cancelled``. With no installation
running, the same command settles the model switch an earlier installation
left unfinished, under the admission of that switch's Sparks.

The request stands outside admission: it takes no installation lock and needs
no hold override, because the operation it ends already holds them. It names
operations by the identifier their lock records, so a request that outlives
its installation cancels nothing later.
"""
from __future__ import annotations

import time
from pathlib import Path

from runtime.common import installer, process_lock
from runtime.host import hold, node

SCHEMA = "sparkring-install-cancel/v1"
RESULT_SCHEMA = "sparkring-install-cancel-result/v1"
NAME = "install-cancel.json"
# Serializes a request's read-modify-write with another request or clear; it is
# never held while an installation runs.
LOCK = "install-cancel.lock"
WAIT_SECONDS = 120
POLL_SECONDS = 2
# The operation string an admission records for an installation is ``install [PROFILE]``.
INSTALL = "install"

# What the boundary names to an operator: the switch step the installation never started.
BOUNDARIES = {"preparing": "preparing the deployment",
              "stopping-previous": "stopping the model it replaces",
              "starting": "starting the selected model",
              "verifying": "checking the selected model",
              "committing": "committing the switch"}


class Cancelled(RuntimeError):
    """An installation stopped at a step boundary because ``install --cancel`` asked it to."""

    def __init__(self, record, boundary):
        self.record, self.boundary = record, boundary
        super().__init__("Installation cancelled at your request while "
                         + BOUNDARIES.get(boundary, boundary) + ".")


def path(state_root):
    return Path(state_root) / NAME


def read(state_root):
    """The standing request, or None: no record, or one this package does not read.

    A damaged request cancels nothing: a record that cannot be read must not
    stop an installation, because the installation is what reports its own
    outcome.
    """
    try:
        value = installer.read(path(state_root))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) and value.get("schema") == SCHEMA else None


def entries(state_root):
    """``{operation id: {"at", "by"}}`` of the standing request."""
    record = read(state_root)
    operations = (record or {}).get("operations")
    return dict(operations) if isinstance(operations, dict) else {}


def request(state_root, operations, *, now=None, who=None):
    """Record that the installations ``operations`` must stop at their next step boundary."""
    ids = sorted({str(item) for item in operations if item})
    if not ids:
        raise ValueError("No installation is running to cancel")
    moment = time.time() if now is None else now
    operator = who or hold.operator()
    with process_lock.hold(Path(state_root) / LOCK, wait=True):
        record = read(state_root) or {"schema": SCHEMA, "operations": {}}
        record["operations"] = {**(record.get("operations") or {}),
                                **{item: {"at": moment, "by": operator} for item in ids}}
        node.save(Path(state_root), NAME, record, mode=0o600)
    return record


def pending(state_root, operation):
    """The entry naming ``operation``, or None."""
    return entries(state_root).get(operation)


def check(state_root, operation, boundary):
    """Raise ``Cancelled`` when ``install --cancel`` named this installation."""
    entry = pending(state_root, operation)
    if entry is not None:
        raise Cancelled(entry, boundary)


def clear(state_root, operation=None):
    """Forget ``operation``'s entry, and the record when no other entry stands."""
    with process_lock.hold(Path(state_root) / LOCK, wait=True):
        record = read(state_root)
        if record is None:
            return None
        operations = dict(record.get("operations") or {})
        if operation is None:
            operations.clear()
        else:
            operations.pop(operation, None)
        if operations:
            node.save(Path(state_root), NAME, {**record, "operations": operations}, mode=0o600)
            return {**record, "operations": operations}
        path(state_root).unlink(missing_ok=True)
        return {**record, "operations": operations}


def installations(rows):
    """The rows of ``slices.active`` that are installations this command can stop."""
    return [row for row in rows if str(row.get("operation") or "").split(" ")[0] == INSTALL]


def describe(rows):
    """``install PROFILE on Sparks 0-1``: what the request names, as the operator sees it."""
    return " and ".join(
        f"{row.get('operation') or INSTALL} on "
        + ("every Spark" if row.get("positions") is None else "Sparks " + ",".join(map(str, row["positions"])))
        for row in rows)


def cancelled_since(state_root, moment):
    """``[(slot, record)]`` of switches recorded cancelled at or after ``moment``."""
    from runtime.host import placement as placements
    found = []
    for slot in placements.slots(state_root):
        try:
            record = placements.journal(state_root, slot)
        except (OSError, ValueError):
            continue
        if not record or record.get("state") != placements.CANCELLED:
            continue
        if ((record.get("cancelled") or {}).get("at") or 0) >= moment:
            found.append((slot, record))
    return found


def run(state_root, *, now=None, who=None, say=print, sleep=time.sleep, active=None, wait=WAIT_SECONDS,
        clock=time.monotonic):
    """``sparkring install --cancel``: ask the running installation to stop, or settle an unfinished switch.

    Returns the result document. ``operations`` names what the request held,
    ``switches`` the model switches this command settled, ``busy`` the slots
    another operation kept, and ``message`` the one line an operator reads.
    ``state`` is ``requested`` while a running installation finishes a step it
    cannot abandon, ``failed`` when a switch this command settled still needs
    attention, and otherwise ``cancelled`` or ``idle``.
    """
    from runtime.host import placement as placements, slices
    state_root = Path(state_root)
    active = slices.active if active is None else active
    document = {"schema": RESULT_SCHEMA, "operations": [], "switches": [], "busy": []}
    running = installations(active(state_root))
    if running:
        record = request(state_root, [row.get("id") for row in running], now=now, who=who)
        document["operations"] = sorted(record["operations"])
        moment = min(entry["at"] for entry in record["operations"].values())
        say(f"Asked {describe(running)} to stop at its next step boundary.")
        deadline = clock() + wait
        while clock() < deadline:
            left = [row for row in installations(active(state_root)) if row.get("id") in document["operations"]]
            if not left:
                break
            sleep(POLL_SECONDS)
        left = [row for row in installations(active(state_root)) if row.get("id") in document["operations"]]
        if left:
            # The installation is inside a step it cannot abandon, such as a
            # checkpoint download; the request outlives this command and the
            # installation honours it at the next boundary.
            document["state"] = "requested"
            document["message"] = (f"The request stands; {describe(left)} stops at its next step boundary. "
                                   "Its result names the switch it leaves.")
            say(document["message"])
            return document
        # The installation clears its own entry as it stops; one that died
        # first leaves it here, and it names an operation that no longer runs.
        for operation in document["operations"]:
            clear(state_root, operation)
        stopped = cancelled_since(state_root, moment)
        document["state"] = "cancelled"
        document["message"] = ("The installation stopped at the request and recorded the switch cancelled."
                               if stopped else
                               "The installation finished before it reached the request; its own result describes "
                               "what it did.")
        say(document["message"])
    else:
        # up, down and restart take the same locks as an installation and are
        # not one: a request cannot stop them, and settling a switch under a
        # running one would fight it.
        others = [row for row in active(state_root)
                  if str(row.get("operation") or "") in ("up", "down", "restart")]
        if others:
            document["state"] = "busy"
            document["message"] = (f"{describe(others)} is running; install --cancel stops installations. Repeat it "
                                   "when that operation finishes.")
            say(document["message"])
            return document
        say("No installation is running.")
    settled, busy = settle(state_root, now=now, who=who, say=say)
    document["switches"], document["busy"] = settled, busy
    attention = [item for item in settled if item["state"] != placements.CANCELLED]
    if attention:
        # A switch that stopped with a model it could not bring back is not
        # settled: its slot stays reserved and status keeps reporting it until
        # an operator or a later installation acts on it.
        first = next((item for item in attention if item.get("error")), attention[0])
        document["state"] = "failed"
        document["message"] = (f"The model switch to {Path(first.get('candidate') or '').name} still needs attention"
                               + (f": {first['error']}" if first.get("error") else "")
                               + ". sparkring status names it.")
    elif busy:
        document["state"] = "busy"
        document["message"] = ("Another operation holds "
                               + ", ".join(item["placement"] or "the whole ring" for item in busy)
                               + "; its model switch is left for it. Repeat the command when it finishes.")
    elif settled and document.get("state") != "cancelled":
        document["state"] = "cancelled"
        document["message"] = ("Cancelled the unfinished model switch to "
                               + ", ".join(Path(item["candidate"] or "").name for item in settled) + ".")
    elif not settled and document.get("state") != "cancelled":
        document["state"] = "idle"
        document["message"] = "No installation is running and no model switch is unfinished."
    say(document["message"])
    return document


def settle(state_root, *, apply=None, verify=None, now=None, who=None, say=print):
    """Settle every unfinished model switch under its own admission; returns ``(settled, busy)``.

    A switch's Sparks may belong to another operation that started after the
    request, so each one is settled while holding that slot's admission. A slot
    another operation keeps is reported and left alone, and a switch another
    cancel settled between the scan and the admission is skipped.

    Each settled entry names its placement, candidate, state and ``recovered``,
    and ``error`` the restore failure that left the switch needing attention
    when its state is not ``placements.CANCELLED``.
    """
    from runtime.host import placement as placements, retained_source, rollout, slices
    state_root = Path(state_root)
    cache = state_root / "retained-sources"

    def adapter(path, action):
        return retained_source.apply(Path(path), action, cache=cache)

    apply = adapter if apply is None else apply
    verify = (lambda path: apply(path, "verify")) if verify is None else verify
    settled, busy = [], []
    for slot, record in rollout.unfinished(state_root):
        try:
            guard = slices.hold(state_root, slot, "install --cancel")
            guard.__enter__()
        except ValueError as error:
            say(f"{placements.text(slot)}: {error}")
            busy.append({"placement": list(slot) if slot else None, "reason": str(error)})
            continue
        try:
            current = placements.journal(state_root, slot)
            if not current or current.get("state") not in placements.PENDING:
                continue
            candidate = Path(current.get("candidate") or "")
            say(f"Stopping the unfinished model switch to {candidate.name} on {placements.text(slot)}.")
            result = rollout.settle(state_root, slot, apply=apply, verify=verify, now=now, who=who, say=say)
            settled.append({"placement": list(slot) if slot else None, "candidate": result.get("candidate"),
                            "state": result.get("state"), "recovered": bool(result.get("recovered")),
                            "error": (result.get("cancelled") or {}).get("error")})
        finally:
            guard.__exit__(None, None, None)
    return settled, busy
