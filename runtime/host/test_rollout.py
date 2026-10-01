"""A stranger must not supply stop/start/rollback commands around installation."""
import pytest

from runtime.common import installer
from runtime.host import node, rollout


class Cluster:
    def __init__(self, previous, candidate, fail=None):
        self.previous, self.candidate, self.fail = previous, candidate, fail
        self.running = previous
        self.events = []

    def prepare(self, directory):
        assert self.running == self.previous
        self.events.append("prepare")
        if self.fail == "prepare":
            raise RuntimeError("Node 2 has insufficient storage")

    def apply(self, directory, action):
        self.events.append((directory.name, action))
        if action == "down":
            if self.running == directory:
                self.running = None
            if directory == self.previous and self.fail == "stop-once":
                self.fail = None
                raise RuntimeError("Previous stop returned a partial failure")
        else:
            assert self.running is None
            self.running = directory

    def verify(self, directory):
        self.events.append((directory.name, "verify"))
        if directory == self.candidate and self.fail == "verify":
            raise RuntimeError("Candidate API failed its response check")
        assert self.running == directory
        return {"api_healthy": True}


def test_one_workflow_prepares_switches_and_commits_the_active_pointer(tmp_path):
    before, after = tmp_path / "old", tmp_path / "candidate"
    node.save(tmp_path, "active.json", {"path": str(before)})
    cluster = Cluster(before, after)
    result = rollout.execute(after, before, state_root=tmp_path, prepare=cluster.prepare, apply=cluster.apply, verify=cluster.verify)
    assert cluster.events == ["prepare", ("old", "down"), ("candidate", "up"), ("candidate", "verify")]
    assert result["complete"] and rollout.active(tmp_path) == after


def test_prepare_failure_leaves_the_previous_model_running(tmp_path):
    before, after = tmp_path / "old", tmp_path / "candidate"
    node.save(tmp_path, "active.json", {"path": str(before)})
    cluster = Cluster(before, after, fail="prepare")
    with pytest.raises(RuntimeError, match="insufficient storage"):
        rollout.execute(after, before, state_root=tmp_path, prepare=cluster.prepare, apply=cluster.apply, verify=cluster.verify)
    assert cluster.running == before and cluster.events == ["prepare"]
    assert rollout.active(tmp_path) == before
    assert installer.read(tmp_path / "transaction.json")["state"] == "preparation-failed"


def test_failed_candidate_is_stopped_and_previous_model_recovered_automatically(tmp_path):
    before, after = tmp_path / "old", tmp_path / "candidate"
    node.save(tmp_path, "active.json", {"path": str(before)})
    cluster = Cluster(before, after, fail="verify")
    with pytest.raises(RuntimeError, match="failed-recovered"):
        rollout.execute(after, before, state_root=tmp_path, prepare=cluster.prepare, apply=cluster.apply, verify=cluster.verify)
    assert cluster.running == before
    assert cluster.events[-4:] == [("candidate", "down"), ("old", "down"), ("old", "up"), ("old", "verify")]
    assert rollout.active(tmp_path) == before


def test_first_install_has_no_previous_model_to_stop(tmp_path):
    after = tmp_path / "candidate"
    cluster = Cluster(None, after)
    result = rollout.execute(after, None, state_root=tmp_path, prepare=cluster.prepare, apply=cluster.apply, verify=cluster.verify)
    assert result["complete"]
    assert cluster.events == ["prepare", ("candidate", "up"), ("candidate", "verify")]


def test_partial_previous_stop_is_reconciled_before_recovery(tmp_path):
    before, after = tmp_path / "old", tmp_path / "candidate"
    node.save(tmp_path, "active.json", {"path": str(before)})
    cluster = Cluster(before, after, fail="stop-once")
    with pytest.raises(RuntimeError, match="failed-recovered"):
        rollout.execute(after, before, state_root=tmp_path, prepare=cluster.prepare, apply=cluster.apply, verify=cluster.verify)
    assert cluster.running == before
    assert ("candidate", "up") not in cluster.events


@pytest.mark.parametrize("stop_refused", [False, True])
def test_unfinished_switch_is_replaced_by_an_approved_install(tmp_path, stop_refused):
    before, stuck, after = tmp_path / "old", tmp_path / "stuck", tmp_path / "candidate"
    node.save(tmp_path, "active.json", {"path": str(before)})
    node.save(tmp_path, "transaction.json", {"schema": "sparkring-install-transaction/v1", "candidate": str(stuck),
                                             "previous": str(before), "state": "needs-attention", "complete": False})
    cluster = Cluster(before, after)
    original = cluster.apply

    def apply(directory, action):
        if directory == stuck and stop_refused:
            raise ValueError("Previous operation is incomplete or uncertain")
        return original(directory, action)
    result = rollout.execute(after, before, state_root=tmp_path, prepare=cluster.prepare, apply=apply,
                             verify=cluster.verify, supersede=True)
    assert result["complete"] and rollout.active(tmp_path) == after
    assert result["superseded"]["candidate"] == str(stuck)
    assert ("supersede_stop_error" in result["superseded"]) == stop_refused


def test_unfinished_switch_still_blocks_without_approval(tmp_path):
    before, stuck, after = tmp_path / "old", tmp_path / "stuck", tmp_path / "candidate"
    node.save(tmp_path, "transaction.json", {"schema": "sparkring-install-transaction/v1", "candidate": str(stuck),
                                             "previous": str(before), "state": "needs-attention", "complete": False})
    cluster = Cluster(before, after)
    with pytest.raises(ValueError, match="unfinished switch to stuck stopped while needs-attention"):
        rollout.execute(after, before, state_root=tmp_path, prepare=cluster.prepare, apply=cluster.apply, verify=cluster.verify)


def interrupted(tmp_path, state):
    """A switch from ``old`` to ``stuck`` whose install stopped in ``state``, as after a Spark restart."""
    before, stuck = tmp_path / "old", tmp_path / "stuck"
    node.save(tmp_path, "active.json", {"path": str(before)})
    node.save(tmp_path, "transaction.json", {"schema": "sparkring-install-transaction/v1", "candidate": str(stuck),
                                             "previous": str(before), "state": state, "complete": False})
    return before, stuck


@pytest.mark.parametrize("state", ["stopping-previous", "starting", "verifying"])
def test_a_switch_interrupted_mid_step_is_replaced_by_another_approved_install(tmp_path, state):
    before, stuck = interrupted(tmp_path, state)
    after, events = tmp_path / "candidate", []
    result = rollout.execute(after, before, state_root=tmp_path, prepare=lambda directory: events.append("prepare"),
                             apply=lambda directory, action: events.append((directory.name, action)),
                             verify=lambda directory: events.append((directory.name, "verify")) or {}, supersede=True)
    # The abandoned candidate stops through its own deployment before
    # anything else changes; then the switch runs from the last good model.
    assert events == [("stuck", "down"), "prepare", ("old", "down"), ("candidate", "up"), ("candidate", "verify")]
    assert result["complete"] and rollout.active(tmp_path) == after
    assert result["superseded"]["state"] == state


def test_installing_the_interrupted_candidate_again_resumes_it(tmp_path):
    before, stuck = interrupted(tmp_path, "starting")
    events = []
    result = rollout.execute(stuck, before, state_root=tmp_path, prepare=lambda directory: events.append("prepare"),
                             apply=lambda directory, action: events.append((directory.name, action)),
                             verify=lambda directory: events.append((directory.name, "verify")) or {}, supersede=True)
    assert "superseded" not in result and "prepare" not in events
    assert result["complete"] and rollout.active(tmp_path) == stuck


def test_an_interrupted_switch_blocks_another_install_without_approval(tmp_path):
    before, stuck = interrupted(tmp_path, "starting")
    with pytest.raises(ValueError, match="install that model again to resume it"):
        rollout.execute(tmp_path / "candidate", before, state_root=tmp_path, prepare=lambda directory: None,
                        apply=lambda directory, action: None, verify=lambda directory: {})
    assert installer.read(tmp_path / "transaction.json")["candidate"] == str(stuck)


@pytest.mark.parametrize("serves", [True, False])
def test_the_active_model_restarts_only_when_it_does_not_serve(tmp_path, capsys, serves):
    current = tmp_path / "current"
    node.save(tmp_path, "active.json", {"path": str(current)})
    events = []
    result = rollout.execute(current, rollout.active(tmp_path), state_root=tmp_path,
                             prepare=lambda directory: events.append("prepare"),
                             apply=lambda directory, action: events.append(action),
                             verify=lambda directory: events.append("verify") or {},
                             serving=lambda directory: events.append("serving") or serves)
    assert events == ["prepare", "serving", *([] if serves else ["down"]), "up", "verify"]
    assert result["complete"] and rollout.active(tmp_path) == current
    assert ("does not serve on every Spark" in capsys.readouterr().out) is not serves


def recorded(events):
    """Adapters that record ``(directory name, action)`` for apply and verify and ``prepare`` for prepare."""
    return {"prepare": lambda directory: events.append("prepare"),
            "apply": lambda directory, action: events.append((directory.name, action)),
            "verify": lambda directory: events.append((directory.name, "verify")) or {}}


@pytest.mark.parametrize("active", [None, "old", "candidate"])
def test_a_candidate_whose_start_did_not_complete_stops_before_it_is_prepared_again(tmp_path, capsys, active):
    candidate = tmp_path / "candidate"
    previous = tmp_path / active if active else None
    if previous:
        node.save(tmp_path, "active.json", {"path": str(previous)})
    events = []
    result = rollout.execute(candidate, previous, state_root=tmp_path, **recorded(events),
                             serving=lambda directory: events.append("serving") or False,
                             unfinished=lambda directory: "up" if directory.name == "candidate" else None)
    # Another active model keeps serving until the switch; the candidate
    # itself, when active, just stopped and needs no serving check.
    switch = [("old", "down")] if active == "old" else []
    assert events == [("candidate", "down"), "prepare", *switch, ("candidate", "up"), ("candidate", "verify")]
    assert result["complete"] and result["unfinished"] == "up" and rollout.active(tmp_path) == candidate
    assert "This model's last start did not complete; it stops on every Spark first." in capsys.readouterr().out


def test_an_unfinished_stop_finishes_before_preparation(tmp_path, capsys):
    events = []
    result = rollout.execute(tmp_path / "candidate", None, state_root=tmp_path, **recorded(events),
                             unfinished=lambda directory: "down")
    assert events == [("candidate", "down"), "prepare", ("candidate", "up"), ("candidate", "verify")]
    assert result["complete"] and result["unfinished"] == "down"
    assert "This model's last stop did not complete; it finishes stopping on every Spark first." in capsys.readouterr().out


def test_an_unfinished_preparation_repeats_without_a_stop(tmp_path, capsys):
    events = []
    result = rollout.execute(tmp_path / "candidate", None, state_root=tmp_path, **recorded(events),
                             unfinished=lambda directory: "prepare")
    assert events == ["prepare", ("candidate", "up"), ("candidate", "verify")]
    assert result["complete"] and "unfinished" not in result
    assert "did not complete" not in capsys.readouterr().out


def test_a_refused_stop_of_the_unfinished_candidate_prepares_nothing_and_keeps_the_active_model(tmp_path):
    before, candidate = tmp_path / "old", tmp_path / "candidate"
    node.save(tmp_path, "active.json", {"path": str(before)})
    events = []

    def apply(directory, action):
        events.append((directory.name, action))
        raise ValueError("stop:spark1: prior outcome is uncertain; inspect host state before creating a recovery plan")
    with pytest.raises(ValueError, match="prior outcome is uncertain"):
        rollout.execute(candidate, before, state_root=tmp_path, prepare=lambda directory: events.append("prepare"),
                        apply=apply, verify=lambda directory: {}, unfinished=lambda directory: "down")
    assert events == [("candidate", "down")]
    journal = installer.read(tmp_path / "transaction.json")
    assert journal["state"] == "preparation-failed" and journal["unfinished"] == "down"
    assert rollout.active(tmp_path) == before


def test_a_restart_that_fails_leaves_no_other_model_to_recover(tmp_path):
    current = tmp_path / "current"
    node.save(tmp_path, "active.json", {"path": str(current)})
    events = []
    def apply(directory, action):
        events.append(action)
        if action == "up":
            raise RuntimeError("Ring check still fails")
    with pytest.raises(RuntimeError, match="failed: Ring check still fails"):
        rollout.execute(current, current, state_root=tmp_path, prepare=lambda directory: None, apply=apply,
                        verify=lambda directory: {}, serving=lambda directory: False)
    assert events == ["down", "up"]
    assert installer.read(tmp_path / "transaction.json")["state"] == "failed"
