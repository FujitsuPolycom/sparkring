"""Prepare a replacement before downtime, then recover the retained deployment on failure."""
from pathlib import Path

from runtime.common import installer
from runtime.host import node, placement as placements

# What the candidate's stop before preparation tells the operator, by the unfinished operation it clears.
UNFINISHED = {"up": "This model's last start did not complete; it stops on every Spark first.",
              "down": "This model's last stop did not complete; it finishes stopping on every Spark first."}


def execute(directory, previous, *, state_root, prepare, apply, verify, supersede=False, serving=None,
            unfinished=None, placement=None, displaced=()):
    """All mutations pass through deployment adapters; the active pointer commits last.

    ``placement`` selects the slot (``runtime.host.placement``) whose switch
    record and active pointer this switch uses: the whole cluster's
    ``transaction.json`` and ``active.json`` for None, a ring half's under
    ``slots/`` otherwise. ``displaced`` lists running deployments of the
    conflicting slots. They stop after the previous deployment and before the
    candidate starts; once the candidate is verified the conflicting slots
    record no deployment, and when the switch fails the displaced deployments
    start again with the previous deployment.

    When the candidate is the active deployment, ``serving(directory)`` tells
    whether it serves on every rank; one that does not stops before it starts
    again.

    ``unfinished(directory)`` names the candidate's own last operation when it
    did not complete (``installer.unfinished``). A deployment whose start or
    stop did not complete accepts no preparation until it stops, so such a
    candidate stops through its own deployment before preparation: the stop
    checks ownership labels, stops only that deployment's containers and
    refuses to repeat a stop action whose outcome is uncertain. The
    transaction records the operation as ``unfinished``.
    """
    directory = Path(directory).resolve()
    previous = Path(previous).resolve() if previous else None
    state_root = Path(state_root)
    displaced = [Path(path).resolve() for path in displaced]
    record = {"schema": "sparkring-install-transaction/v1", "candidate": str(directory),
              "previous": str(previous) if previous else None, "state": "preparing", "complete": False}
    if displaced:
        record["displaced"] = [str(path) for path in displaced]
    resume = False
    slot = placements.slot_directory(state_root, placement)
    journal = slot / "transaction.json"
    if journal.exists():
        retained = installer.read(journal)
        pending = retained["state"] in ("stopping-previous", "starting", "verifying", "recovering-previous", "needs-attention")
        # The caller holds install.lock, so a journal left mid-switch belongs
        # to an install that stopped, for example because a Spark restarted.
        # Installing the same candidate resumes it; another candidate may
        # replace it, like a switch whose recovery failed.
        abandoned = retained["state"] in ("recovering-previous", "needs-attention") or retained["candidate"] != str(directory)
        if pending and abandoned and supersede:
            # The caller has confirmed no unexpected GPU workload is running.
            # Stop the abandoned candidate through its own deployment, then
            # replace it; the active pointer still names the last good model.
            print("Replacing an unfinished model switch: " + retained["candidate"])
            try:
                apply(Path(retained["candidate"]), "down")
            except Exception as error:  # noqa: BLE001 - recorded; GPUs were confirmed idle
                retained["supersede_stop_error"] = str(error)
                print("Its own stop step was refused; no unexpected GPU workload is running, continuing.")
            record["superseded"] = retained
            pending = False
        if pending:
            if abandoned:
                raise ValueError(f"An unfinished switch to {Path(retained['candidate']).name} stopped while "
                                 f"{retained['state']}; install that model again to resume it, or confirm that no "
                                 f"other GPU workload is running to replace it ({journal})")
            record = retained
            previous = Path(record["previous"]) if record["previous"] else None
            displaced = [Path(path) for path in record.get("displaced") or []]
            resume = True
    def save(state):
        record["state"] = state
        node.save(slot, "transaction.json", record, mode=0o600)
    # No other deployment is stopped if prerequisites, transfer, space or asset
    # verification fail. The existing active pointer remains authoritative.
    # Only a candidate whose own start or stop did not complete stops before
    # preparation, also when it is the active deployment.
    stopped = False
    if not resume:
        save("preparing")
        try:
            left = unfinished(directory) if unfinished is not None else None
            if left in UNFINISHED:
                print(UNFINISHED[left])
                record["unfinished"] = left
                apply(directory, "down")
                stopped = True
            prepare(directory)
        except Exception as error:
            record["error"] = str(error)
            save("preparation-failed")
            raise
    switched = False
    restore = ([previous] if previous is not None and previous != directory else []) + displaced
    try:
        if previous is not None and previous != directory:
            save("stopping-previous")
            switched = True
            apply(previous, "down")
        elif previous == directory and not stopped and serving is not None and not serving(directory):
            print("The installed model does not serve on every Spark; it stops on every Spark and starts again.")
            save("stopping-previous")
            apply(directory, "down")
        for other in displaced:
            save("stopping-previous")
            switched = True
            apply(other, "down")
        save("starting")
        apply(directory, "up")
        save("verifying")
        observed = verify(directory)
        record["verification"] = observed
        node.save(slot, "active.json", {"path": str(directory)}, mode=0o600)
        placements.clear_conflicting(state_root, placement)
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
                # only some ranks stopped. Reconcile each retained operation
                # before asking its own source version to bring it back.
                for other in restore:
                    apply(other, "down")
                for other in restore:
                    apply(other, "up")
                record["recovery_verification"] = [verify(other) for other in restore]
                if len(restore) == 1:
                    record["recovery_verification"] = record["recovery_verification"][0]
                if previous is not None and previous != directory:
                    node.save(slot, "active.json", {"path": str(previous)}, mode=0o600)
                record["recovered"] = True
                save("failed-recovered")
            except Exception as recovery:
                record["recovered"] = False
                record["recovery_error"] = str(recovery)
                save("needs-attention")
        else:
            save("failed")
        raise RuntimeError("Installation did not complete; " + record["state"] + ": " + str(error)) from error


def active(state_root, placement=None):
    """The deployment the slot of ``placement`` records active; the whole cluster's for None."""
    return placements.recorded(state_root, placement)
