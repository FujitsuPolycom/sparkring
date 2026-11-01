"""Cancelling an installation: the request record, the switch it stops, and the switch it settles."""
import json
import os

import pytest

from runtime.common import installer
from runtime.host import cancel, node, rollout

NOW = 1_800_000_000.0


class Cluster:
    """Adapters over two deployments, as ``test_rollout`` uses them."""

    def __init__(self, previous, candidate):
        self.previous, self.candidate = previous, candidate
        self.running = previous
        self.events = []

    def prepare(self, directory):
        self.events.append("prepare")

    def apply(self, directory, action):
        self.events.append((directory.name, action))
        if action == "down":
            if self.running == directory:
                self.running = None
        else:
            self.running = directory

    def verify(self, directory):
        self.events.append((directory.name, "verify"))
        assert self.running == directory
        return {"api_healthy": True}


def raise_at(boundary, who="alice"):
    """A ``cancelled`` hook that asks to stop at one boundary of the switch."""
    def cancelled(state):
        if state == boundary:
            raise cancel.Cancelled({"by": who, "at": NOW}, state)
    return cancelled


def interrupted(tmp_path, state):
    """A switch from ``old`` to ``stuck`` whose installation stopped in ``state``, as after a SIGTERM."""
    before, stuck = tmp_path / "old", tmp_path / "stuck"
    node.save(tmp_path, "active.json", {"path": str(before)})
    node.save(tmp_path, "transaction.json", {"schema": "sparkring-install-transaction/v1", "candidate": str(stuck),
                                             "previous": str(before), "state": state, "complete": False})
    return before, stuck


def test_a_request_keeps_the_operator_and_time_of_each_ask(tmp_path):
    first = cancel.request(tmp_path, ["b"], now=NOW, who="alice")
    assert first["operations"] == {"b": {"at": NOW, "by": "alice"}}
    both = cancel.request(tmp_path, ["a"], now=NOW + 1, who="bob")
    assert both["operations"] == {"a": {"at": NOW + 1, "by": "bob"}, "b": {"at": NOW, "by": "alice"}}
    assert cancel.pending(tmp_path, "a") == {"at": NOW + 1, "by": "bob"}
    assert cancel.pending(tmp_path, "c") is None
    assert cancel.clear(tmp_path, "a")["operations"] == {"b": {"at": NOW, "by": "alice"}}
    assert cancel.clear(tmp_path, "b")["operations"] == {}
    assert cancel.read(tmp_path) is None
    with pytest.raises(ValueError, match="No installation is running"):
        cancel.request(tmp_path, [], now=NOW, who="alice")


def test_a_damaged_request_cancels_nothing(tmp_path):
    (tmp_path / cancel.NAME).write_text("{not json", encoding="utf-8")
    assert cancel.read(tmp_path) is None
    # A record that cannot be read must not stop an installation: the
    # installation is what reports the outcome.
    cancel.check(tmp_path, "a", "starting")
    (tmp_path / cancel.NAME).write_text(json.dumps({"schema": "other/v1", "operations": {"a": {}}}),
                                        encoding="utf-8")
    cancel.check(tmp_path, "a", "starting")


def test_a_cancel_before_any_stop_leaves_the_serving_model_alone(tmp_path):
    before, after = tmp_path / "old", tmp_path / "candidate"
    node.save(tmp_path, "active.json", {"path": str(before)})
    cluster = Cluster(before, after)
    with pytest.raises(cancel.Cancelled, match="preparing the deployment"):
        rollout.execute(after, before, state_root=tmp_path, prepare=cluster.prepare, apply=cluster.apply,
                        verify=cluster.verify, cancelled=raise_at("preparing"))
    assert cluster.events == [] and cluster.running == before
    record = installer.read(tmp_path / "transaction.json")
    assert record["state"] == "cancelled" and record["complete"] is False
    assert record["cancelled"]["by"] == "alice" and record["cancelled"]["boundary"] == "preparing"
    assert "recovered" not in record and record["cancelled"].get("at")
    assert rollout.active(tmp_path) == before


@pytest.mark.parametrize("boundary", ["starting", "verifying", "committing"])
def test_a_cancel_mid_switch_brings_back_the_model_it_replaced(tmp_path, boundary):
    before, after = tmp_path / "old", tmp_path / "candidate"
    node.save(tmp_path, "active.json", {"path": str(before)})
    cluster = Cluster(before, after)
    with pytest.raises(cancel.Cancelled, match="Installation cancelled at your request"):
        rollout.execute(after, before, state_root=tmp_path, prepare=cluster.prepare, apply=cluster.apply,
                        verify=cluster.verify, cancelled=raise_at(boundary))
    # The candidate never serves: whatever started stops, then the previous
    # model is reconciled and started again.
    assert cluster.events[-4:] == [("candidate", "down"), ("old", "down"), ("old", "up"), ("old", "verify")]
    assert cluster.running == before
    record = installer.read(tmp_path / "transaction.json")
    assert record["state"] == "cancelled" and record["recovered"] is True
    assert record["cancelled"]["boundary"] == boundary
    assert rollout.active(tmp_path) == before


def test_a_cancel_as_the_switch_would_stop_the_running_model_stops_nothing(tmp_path):
    before, after = tmp_path / "old", tmp_path / "candidate"
    node.save(tmp_path, "active.json", {"path": str(before)})
    cluster = Cluster(before, after)
    # The step is recorded before the stop it describes, so a cancel there
    # finds both models as the switch began.
    with pytest.raises(cancel.Cancelled):
        rollout.execute(after, before, state_root=tmp_path, prepare=cluster.prepare, apply=cluster.apply,
                        verify=cluster.verify, cancelled=raise_at("stopping-previous"))
    assert cluster.events == ["prepare"] and cluster.running == before
    record = installer.read(tmp_path / "transaction.json")
    assert record["state"] == "cancelled" and "recovered" not in record


def test_a_cancelled_restart_of_the_active_model_starts_it_again(tmp_path):
    current = tmp_path / "current"
    node.save(tmp_path, "active.json", {"path": str(current)})
    events = []

    def apply(directory, action):
        events.append((directory.name, action))

    with pytest.raises(cancel.Cancelled):
        rollout.execute(current, rollout.active(tmp_path), state_root=tmp_path,
                        prepare=lambda directory: events.append("prepare"), apply=apply,
                        verify=lambda directory: events.append((directory.name, "verify")) or {},
                        serving=lambda directory: True, cancelled=raise_at("verifying"))
    # The switch restarted the only model; a cancel stops it and starts it again.
    assert events == ["prepare", ("current", "up"), ("current", "down"), ("current", "up"), ("current", "verify")]
    record = installer.read(tmp_path / "transaction.json")
    assert record["state"] == "cancelled" and record["recovered"] is True


def test_a_cancelled_resumed_switch_brings_back_the_model_the_earlier_run_stopped(tmp_path):
    before, stuck = interrupted(tmp_path, "starting")
    events = []
    with pytest.raises(cancel.Cancelled):
        rollout.execute(stuck, before, state_root=tmp_path, prepare=lambda directory: events.append("prepare"),
                        apply=lambda directory, action: events.append((directory.name, action)),
                        verify=lambda directory: events.append((directory.name, "verify")) or {},
                        cancelled=raise_at("stopping-previous"))
    # A resumed switch skips preparation, and its first recorded step is the
    # stop the earlier run already performed.
    assert events == [("stuck", "down"), ("old", "down"), ("old", "up"), ("old", "verify")]
    record = installer.read(tmp_path / "transaction.json")
    assert record["state"] == "cancelled" and record["recovered"] is True


def test_a_cancelled_first_install_stops_its_candidate_and_records_it(tmp_path):
    after = tmp_path / "candidate"
    events = []
    with pytest.raises(cancel.Cancelled):
        rollout.execute(after, None, state_root=tmp_path, prepare=lambda directory: events.append("prepare"),
                        apply=lambda directory, action: events.append((directory.name, action)),
                        verify=lambda directory: events.append((directory.name, "verify")) or {},
                        cancelled=raise_at("verifying"))
    assert events == ["prepare", ("candidate", "up"), ("candidate", "down")]
    record = installer.read(tmp_path / "transaction.json")
    assert record["state"] == "cancelled" and "recovered" not in record


def test_the_commit_is_not_a_boundary_a_cancel_stops(tmp_path):
    before, after = tmp_path / "old", tmp_path / "candidate"
    node.save(tmp_path, "active.json", {"path": str(before)})
    cluster = Cluster(before, after)
    result = rollout.execute(after, before, state_root=tmp_path, prepare=cluster.prepare, apply=cluster.apply,
                             verify=cluster.verify, cancelled=raise_at("complete"))
    assert result["complete"] and rollout.active(tmp_path) == after
    assert cluster.running == after and "cancelled" not in result


def test_a_cancelled_switch_does_not_block_the_next_install(tmp_path):
    before, after = tmp_path / "old", tmp_path / "candidate"
    node.save(tmp_path, "active.json", {"path": str(before)})
    cluster = Cluster(before, after)
    with pytest.raises(cancel.Cancelled):
        rollout.execute(after, before, state_root=tmp_path, prepare=cluster.prepare, apply=cluster.apply,
                        verify=cluster.verify, cancelled=raise_at("starting"))
    later = tmp_path / "later"
    result = rollout.execute(later, before, state_root=tmp_path, prepare=cluster.prepare, apply=cluster.apply,
                             verify=cluster.verify)
    assert result["complete"] and rollout.active(tmp_path) == later


def test_a_failed_restore_while_cancelling_keeps_the_slot_blocked(tmp_path):
    before, after = tmp_path / "old", tmp_path / "candidate"
    node.save(tmp_path, "active.json", {"path": str(before)})
    events = []

    def apply(directory, action):
        events.append((directory.name, action))
        if action == "up" and directory == before:
            raise RuntimeError("Node 1 did not answer")

    with pytest.raises(RuntimeError, match="cancelled at your request, but restoring old failed"):
        rollout.execute(after, before, state_root=tmp_path, prepare=lambda directory: None, apply=apply,
                        verify=lambda directory: {}, cancelled=raise_at("starting"))
    record = installer.read(tmp_path / "transaction.json")
    assert record["state"] == "needs-attention" and record["recovered"] is False
    assert "Node 1 did not answer" in record["cancellation_error"]


def test_settling_an_unfinished_switch_stops_its_candidate_and_restarts_the_previous(tmp_path):
    before, stuck = interrupted(tmp_path, "starting")
    assert str(before).endswith("old")
    events = []
    settled, busy = cancel.settle(tmp_path, apply=lambda directory, action: events.append((directory.name, action)),
                                  verify=lambda directory: events.append((directory.name, "verify")) or {"ok": True})
    assert events == [("stuck", "down"), ("old", "down"), ("old", "up"), ("old", "verify")]
    assert busy == []
    record = installer.read(tmp_path / "transaction.json")
    assert record["state"] == "cancelled" and record["recovered"] is True
    assert record["cancelled"]["boundary"] == "starting"
    assert settled == [{"placement": None, "candidate": str(stuck), "state": "cancelled", "recovered": True}]
    # Nothing is left for a second run to settle.
    assert cancel.settle(tmp_path, apply=lambda directory, action: None) == ([], [])


def test_settling_a_committed_switch_rewrites_the_active_record(tmp_path):
    before, stuck = tmp_path / "old", tmp_path / "stuck"
    # The switch wrote the candidate active and stopped before it recorded completion.
    node.save(tmp_path, "active.json", {"path": str(stuck)})
    node.save(tmp_path, "transaction.json", {"schema": "sparkring-install-transaction/v1", "candidate": str(stuck),
                                             "previous": str(before), "state": "verifying", "complete": False})
    cancel.settle(tmp_path, apply=lambda directory, action: None, verify=lambda directory: {},
                 say=lambda line: None)
    assert rollout.active(tmp_path) == before


def test_settling_reports_a_restore_it_could_not_complete(tmp_path):
    interrupted(tmp_path, "verifying")

    def apply(directory, action):
        if action == "up":
            raise RuntimeError("Node 1 did not answer")

    settled, busy = cancel.settle(tmp_path, apply=apply, verify=lambda directory: {}, say=lambda line: None)
    record = installer.read(tmp_path / "transaction.json")
    assert record["state"] == "needs-attention"
    assert settled[0]["state"] == "needs-attention" and settled[0]["recovered"] is False
    assert busy == [] and len(rollout.unfinished(tmp_path)) == 1


def test_settling_leaves_a_switch_another_operation_holds(tmp_path):
    from runtime.host import slices
    interrupted(tmp_path, "starting")
    with slices.hold(tmp_path, None, "install other"):
        settled, busy = cancel.settle(tmp_path, apply=lambda directory, action: None,
                                      verify=lambda directory: {}, say=lambda line: None)
    assert settled == [] and busy and "Another operation is active" in busy[0]["reason"]
    # The switch still belongs to the operation that holds it.
    assert installer.read(tmp_path / "transaction.json")["state"] == "starting"


def test_settling_skips_a_switch_that_became_settled_first(tmp_path, monkeypatch):
    interrupted(tmp_path, "cancelled")
    stale = (None, {"candidate": str(tmp_path / "stuck"), "previous": str(tmp_path / "old"), "state": "starting"})
    monkeypatch.setattr(rollout, "unfinished", lambda root: [stale])
    applied = []
    settled, busy = cancel.settle(tmp_path, apply=lambda directory, action: applied.append(action),
                                  verify=lambda directory: {}, say=lambda line: None)
    assert (settled, busy, applied) == ([], [], [])


def test_status_names_the_switch_and_the_command_that_settles_it(tmp_path):
    _, stuck = interrupted(tmp_path, "starting")
    found = rollout.unfinished(tmp_path)
    assert [(slot, record["candidate"]) for slot, record in found] == [(None, str(stuck))]
    assert rollout.switch_line(found[0][0], found[0][1], 2) == (
        "Model switch: old to stuck on both Sparks, recorded starting")
    assert "install --cancel" in rollout.SWITCH_NEXT and "install that model again" in rollout.SWITCH_NEXT
    # A settled switch is not reported again.
    cancel.settle(tmp_path, apply=lambda directory, action: None, verify=lambda directory: {},
                  say=lambda line: None)
    assert rollout.unfinished(tmp_path) == []


def test_the_operator_view_names_an_unfinished_switch_too(tmp_path):
    from runtime.host import operator_view
    interrupted(tmp_path, "starting")
    document = operator_view.view(tmp_path)
    assert document["switches"] == [{"placement": None, "candidate": "stuck", "previous": "old",
                                    "state": "starting"}]
    text = "\n".join(operator_view.lines(document))
    assert "Model switch: old to stuck on every Spark, recorded starting" in text
    assert "sudo sparkring install --cancel" in text


def test_a_record_that_cannot_be_read_is_not_reported(tmp_path):
    from runtime.host import operator_view
    interrupted(tmp_path, "starting")
    (tmp_path / "transaction.json").write_text("{not json", encoding="utf-8")
    assert rollout.unfinished(tmp_path) == []
    assert operator_view.view(tmp_path)["switches"] == []


def test_a_cancelled_switch_reserves_no_spark_and_no_rollback_target(tmp_path):
    from runtime.host import checkpoints, slices
    node.save(tmp_path, "transaction.json", {"schema": "sparkring-install-transaction/v1",
                                             "candidate": str(tmp_path / "stuck"),
                                             "previous": str(tmp_path / "old"), "state": "cancelled",
                                             "complete": False})
    # The candidate is stopped and the replaced model serves, so a model on
    # part of the ring needs no admission over the rest of it.
    assert slices.affected(tmp_path, (0, 1)) == (0, 1)
    assert os.path.realpath(tmp_path / "stuck") not in checkpoints.roles(tmp_path)
    # A switch that stopped mid-step still reserves the ring and keeps its candidate.
    node.save(tmp_path, "transaction.json", {"schema": "sparkring-install-transaction/v1",
                                             "candidate": str(tmp_path / "stuck"),
                                             "previous": str(tmp_path / "old"), "state": "starting",
                                             "complete": False})
    assert slices.affected(tmp_path, (0, 1)) is None
    assert checkpoints.roles(tmp_path)[os.path.realpath(tmp_path / "stuck")] == "switching"


def test_cancel_with_nothing_running_reports_nothing_to_do(tmp_path):
    lines = []
    document = cancel.run(tmp_path, say=lines.append, sleep=lambda seconds: None, active=lambda root: [])
    assert document["state"] == "idle" and document["switches"] == [] and document["busy"] == []
    assert cancel.read(tmp_path) is None
    assert "No installation is running" in lines[0]


def test_cancel_asks_the_running_installation_to_stop(tmp_path):
    rows = [{"id": "op-1", "operation": "install glm53", "positions": None}]
    lines = []

    def active(root):
        return list(rows)

    def sleep(seconds):
        # The installation reads the request at its next step boundary and leaves.
        rows.clear()

    document = cancel.run(tmp_path, say=lines.append, sleep=sleep, active=active, now=NOW, who="alice",
                          clock=lambda: 0.0)
    assert document["state"] == "cancelled" and document["operations"] == ["op-1"]
    assert "install glm53 on every Spark" in lines[0]
    assert cancel.read(tmp_path) is None


def test_cancel_reports_an_installation_that_finished_before_reaching_the_request(tmp_path):
    rows = [{"id": "op-1", "operation": "install glm53", "positions": None}]
    lines = []

    def sleep(seconds):
        rows.clear()

    document = cancel.run(tmp_path, say=lines.append, sleep=sleep, active=lambda root: list(rows),
                          now=NOW, who="alice", clock=lambda: 0.0)
    assert document["state"] == "cancelled"
    assert "finished before it reached the request" in document["message"]


def test_cancel_that_the_installation_cannot_reach_yet_leaves_its_request_standing(tmp_path):
    rows = [{"id": "op-1", "operation": "install glm53", "positions": [0, 1]}]
    lines = []
    document = cancel.run(tmp_path, say=lines.append, sleep=lambda seconds: None, active=lambda root: list(rows),
                          now=NOW, who="alice", clock=lambda: 0.0, wait=0)
    assert document["state"] == "requested"
    assert "install glm53 on Sparks 0,1" in document["message"]
    assert cancel.pending(tmp_path, "op-1") is not None


def test_cancel_does_not_settle_a_switch_under_a_running_lifecycle_operation(tmp_path):
    interrupted(tmp_path, "starting")
    lines = []
    rows = [{"id": "op-2", "operation": "down", "positions": [0, 1]}]
    document = cancel.run(tmp_path, say=lines.append, sleep=lambda seconds: None, active=lambda root: list(rows))
    assert document["state"] == "busy" and "down on Sparks 0,1 is running" in document["message"]
    # The switch is left for the operation that owns the locks.
    assert installer.read(tmp_path / "transaction.json")["state"] == "starting"
    assert len(rollout.unfinished(tmp_path)) == 1


def test_the_command_needs_no_cluster_and_refuses_other_options(tmp_path, monkeypatch):
    from runtime.host import install_workflow
    monkeypatch.setattr(install_workflow.controller, "STATE", tmp_path)
    monkeypatch.setattr(install_workflow.cancel, "run",
                        lambda root, **kwargs: {"schema": cancel.RESULT_SCHEMA, "state": "idle",
                                                "message": "No installation is running and no model switch is "
                                                           "unfinished.", "operations": [], "switches": [],
                                                "busy": []})
    assert install_workflow.main(["--cancel"]) == 0
    for refused in (["--profile", "glm53-flash"], ["--yes"], ["--events", str(tmp_path / "events.jsonl")]):
        with pytest.raises(SystemExit):
            install_workflow.main(["--cancel", *refused])
