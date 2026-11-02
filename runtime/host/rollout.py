"""Prepare a replacement before downtime, then recover the retained deployment on failure."""
import time
from pathlib import Path

from runtime.common import installer
from runtime.host import cancel, hold, node
from runtime.host import placement as placements

# What the candidate's stop before preparation tells the operator, by the unfinished operation it clears.
UNFINISHED = {"up": "This model's last start did not complete; it stops on every Spark first.",
              "down": "This model's last stop did not complete; it finishes stopping on every Spark first."}


def execute(directory, previous, *, state_root, prepare, apply, verify, supersede=False, serving=None,
            unfinished=None, placement=None, displaced=(), cancelled=None):
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

    ``cancelled(boundary)`` is asked before each step the switch records, and
    raises ``cancel.Cancelled`` to stop it there. A switch that has not yet
    stopped anything records ``cancelled``; one that has brings back every
    deployment it stopped first, and one whose restore fails records
    ``needs-attention`` with the restore's error.
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
        pending = retained["state"] in placements.PENDING
        # The caller holds every affected Spark's lock, so a journal left mid-switch belongs
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
    def save(state, *, honour=True):
        if honour and cancelled is not None:
            cancelled(state)
        record["state"] = state
        node.save(slot, "transaction.json", record, mode=0o600)

    def note_cancel(stop):
        record["cancelled"] = {"at": time.time(), "by": (stop.record or {}).get("by"),
                               "boundary": stop.boundary}

    def bring_back(paths):
        """Stop the candidate and bring back every deployment this switch stopped."""
        targets = []
        for path in paths:
            resolved = Path(path).resolve()
            if resolved not in targets:
                targets.append(resolved)
        apply(directory, "down")
        # A stop of the previous cluster may have left only some ranks stopped.
        # Reconcile each retained operation before asking its own source
        # version to bring it back. The candidate is already stopped above.
        for other in targets:
            if other != directory:
                apply(other, "down")
        for other in targets:
            apply(other, "up")
        observed = [verify(other) for other in targets]
        return observed[0] if len(observed) == 1 else observed

    # No other deployment is stopped if prerequisites, transfer, space or asset
    # verification fail. The existing active pointer remains authoritative.
    # Only a candidate whose own start or stop did not complete stops before
    # preparation, also when it is the active deployment.
    stopped = attempted = stopped_active = False
    if not resume:
        try:
            save("preparing")
        except cancel.Cancelled as stop:
            # Nothing has changed yet: the previous model serves and the
            # candidate is not prepared, so the record is the whole outcome.
            note_cancel(stop)
            save("cancelled", honour=False)
            raise
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
            save("preparation-failed", honour=False)
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
            stopped_active = True
        for other in displaced:
            save("stopping-previous")
            switched = True
            apply(other, "down")
        save("starting")
        attempted = True
        apply(directory, "up")
        save("verifying")
        observed = verify(directory)
        record["verification"] = observed
        # The commit itself cannot be cancelled, so the request is read once
        # more while the switch is still undoable.
        if cancelled is not None:
            cancelled("committing")
        node.save(slot, "active.json", {"path": str(directory)}, mode=0o600)
        placements.clear_conflicting(state_root, placement)
        record["complete"] = True
        # The commit is not a step a cancel can undo: the active pointer and
        # the conflicting slots' records changed together.
        save("complete", honour=False)
        return record
    except cancel.Cancelled as stop:
        # The operator asked the installation to stop: record where, and leave
        # the slot as it was, with the model this switch replaced serving.
        # What to bring back belongs to the journal, not to this call: a
        # resumed switch may already have stopped it, and a restart of the
        # active deployment starts it again under ``attempted``.
        note_cancel(stop)
        back = restore or ([previous] if previous is not None
                           and (stopped_active or attempted or resume) else [])
        try:
            if resume or switched or stopped_active or attempted:
                record["recovery_verification"] = bring_back(back)
                if previous is not None and previous != directory:
                    node.save(slot, "active.json", {"path": str(previous)}, mode=0o600)
                if back:
                    record["recovered"] = True
            save("cancelled", honour=False)
        except Exception as recovery:
            record["recovered"] = False
            record["cancellation_error"] = str(recovery)
            save("needs-attention", honour=False)
            raise RuntimeError("Installation cancelled at your request, but restoring "
                               + (back[0].name if back else "the previous model") + " failed: "
                               + str(recovery)) from recovery
        raise
    except Exception as error:
        record["error"] = str(error)
        if switched:
            save("recovering-previous", honour=False)
            try:
                record["recovery_verification"] = bring_back(restore)
                if previous is not None and previous != directory:
                    node.save(slot, "active.json", {"path": str(previous)}, mode=0o600)
                record["recovered"] = True
                save("failed-recovered", honour=False)
            except Exception as recovery:
                record["recovered"] = False
                record["recovery_error"] = str(recovery)
                save("needs-attention", honour=False)
        else:
            save("failed", honour=False)
        raise RuntimeError("Installation did not complete; " + record["state"] + ": " + str(error)) from error


def active(state_root, placement=None):
    """The deployment the slot of ``placement`` records active; the whole cluster's for None."""
    return placements.recorded(state_root, placement)


def unfinished(state_root):
    """``[(slot, record)]`` of every slot whose model switch stopped before it finished.

    A record that cannot be read is left out: ``sparkring status`` reports what
    it can read rather than failing, and admission reads the record itself.
    """
    found = []
    for slot in placements.slots(state_root):
        try:
            record = placements.journal(state_root, slot)
        except (OSError, ValueError):
            continue
        if record and record.get("state") in placements.PENDING:
            found.append((slot, record))
    return found


def switch_line(slot, record, size=None):
    """The first half of one ``sparkring status`` line for a model switch in progress or stopped."""
    candidate = Path(record.get("candidate") or "").name
    previous = Path(record["previous"]).name if record.get("previous") else None
    switch = f"{previous} to {candidate}" if previous else candidate
    return (f"Model switch: {switch} on {placements.text(slot, size)}, "
            f"recorded {record.get('state')}")


# What settles a switch that stopped before it finished, appended to switch_line's line.
SWITCH_NEXT = "sudo sparkring install --cancel stops it and records it cancelled, or install that model again to resume it"


def settle(state_root, placement, *, apply, verify=None, now=None, who=None, say=print):
    """Stop an unfinished switch's candidate, bring back what it stopped, and record it cancelled.

    The installation that started the switch is gone: another run replaced it,
    a Spark restarted, or the operator stopped it. Its own record is then the
    only description of what it stopped, so the restore follows ``previous``
    and ``displaced`` and asks each deployment's own adapter. A switch that
    committed before it stopped recorded the candidate as active; the restore
    writes the deployment that serves again in its place.
    """
    slot = placements.slot_directory(state_root, placement)
    journal = slot / "transaction.json"
    if not journal.exists():
        raise ValueError(f"No model switch is recorded under {slot}")
    record = installer.read(journal)
    if record.get("state") not in placements.PENDING:
        raise ValueError(f"The model switch under {slot} is {record.get('state')}, not unfinished")
    directory = Path(record["candidate"])
    previous = Path(record["previous"]) if record.get("previous") else None
    # The switch stops the previous deployment and the displaced ones only
    # after it records the step that does so; a switch that stopped while
    # preparing stopped nothing, and a stopped model needs no restart.
    stopped = record["state"] != "preparing"
    back = ([previous] if previous is not None and previous != directory else []) if stopped else []
    back = back + ([Path(path) for path in record.get("displaced") or []] if stopped else [])
    if not back and stopped and previous is not None and previous == directory:
        back = [previous]
    verify = (lambda path: apply(path, "verify")) if verify is None else verify
    errors = []

    def attempt(action, path):
        try:
            apply(path, action)
        except Exception as error:  # noqa: BLE001 - recorded and reported, one step must not hide the rest
            errors.append(f"{path.name} {action}: {error}")

    attempt("down", directory)
    for path in back:
        attempt("down", path)
    for path in back:
        attempt("up", path)
    verification = []
    for path in back:
        try:
            verification.append(verify(path))
        except Exception as error:  # noqa: BLE001 - recorded as this switch's verification
            errors.append(f"{path.name} verify: {error}")
    record["cancelled"] = {"at": time.time() if now is None else now, "by": who or hold.operator(),
                           "boundary": record["state"]}
    if not errors:
        try:
            committed = placements.recorded(state_root, placement) == directory
            if committed:
                # The switch committed and stopped before it recorded completion:
                # the slot names the candidate, which is now stopped.
                if previous is not None:
                    placements.record(state_root, placement, previous)
                else:
                    placements.forget(state_root, placement, directory)
        except (OSError, ValueError):
            # An unreadable active record is reported by status; it does not
            # change what this switch did to the models.
            pass
    if back:
        record["recovered"] = not errors
        record["recovery_verification"] = verification[0] if len(verification) == 1 else verification
    if errors:
        record["cancelled"]["error"] = "; ".join(errors)
        record["state"] = "needs-attention"
    else:
        record["state"] = "cancelled"
    node.save(slot, "transaction.json", record, mode=0o600)
    say("Its candidate stopped; " + ("the model it replaced serves again." if record.get("recovered")
                                     else "no earlier model was stopped." if not back
                                     else "the model it replaced was not confirmed serving: "
                                          + record["cancelled"]["error"]))
    return record
