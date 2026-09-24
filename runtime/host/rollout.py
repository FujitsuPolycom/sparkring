"""Prepare a replacement before downtime, then recover the retained deployment on failure."""
from pathlib import Path

from runtime.common import installer
from runtime.host import node


def execute(directory, previous, *, state_root, prepare, apply, verify):
    """All mutations pass through deployment adapters; the active pointer commits last."""
    directory = Path(directory).resolve()
    previous = Path(previous).resolve() if previous else None
    state_root = Path(state_root)
    record = {"schema": "sparkring-install-transaction/v1", "candidate": str(directory),
              "previous": str(previous) if previous else None, "state": "preparing", "complete": False}
    resume = False
    journal = state_root / "transaction.json"
    if journal.exists():
        retained = installer.read(journal)
        pending = retained["state"] in ("stopping-previous", "starting", "verifying", "recovering-previous", "needs-attention")
        if pending:
            if retained["candidate"] != str(directory) or retained["state"] in ("recovering-previous", "needs-attention"):
                raise ValueError("A previous model switch needs attention; inspect " + str(journal))
            record = retained
            previous = Path(record["previous"]) if record["previous"] else None
            resume = True
    def save(state):
        record["state"] = state
        node.save(state_root, "transaction.json", record, mode=0o600)
    # No prior deployment is stopped if prerequisites, transfer, space or asset
    # verification fail. The existing active pointer remains authoritative.
    if not resume:
        save("preparing")
        try:
            prepare(directory)
        except Exception as error:
            record["error"] = str(error)
            save("preparation-failed")
            raise
    switched = False
    try:
        if previous is not None and previous != directory:
            save("stopping-previous")
            switched = True
            apply(previous, "down")
        save("starting")
        apply(directory, "up")
        save("verifying")
        observed = verify(directory)
        record["verification"] = observed
        node.save(state_root, "active.json", {"path": str(directory)}, mode=0o600)
        record["complete"] = True
        save("complete")
        return record
    except Exception as error:
        record["error"] = str(error)
        if switched:
            save("recovering-previous")
            try:
                apply(directory, "down")
                # A failure while stopping the previous cluster may have left
                # only some ranks stopped. Reconcile that retained operation
                # before asking its own source version to bring it back.
                apply(previous, "down")
                apply(previous, "up")
                record["recovery_verification"] = verify(previous)
                node.save(state_root, "active.json", {"path": str(previous)}, mode=0o600)
                record["recovered"] = True
                save("failed-recovered")
            except Exception as recovery:
                record["recovered"] = False
                record["recovery_error"] = str(recovery)
                save("needs-attention")
        else:
            save("failed")
        raise RuntimeError("Installation did not complete; " + record["state"] + ": " + str(error)) from error


def active(state_root):
    path = Path(state_root) / "active.json"
    return Path(installer.read(path)["path"]) if path.exists() else None
