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
    with pytest.raises(ValueError, match="needs attention"):
        rollout.execute(after, before, state_root=tmp_path, prepare=cluster.prepare, apply=cluster.apply, verify=cluster.verify)


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
