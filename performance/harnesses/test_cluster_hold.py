"""cluster_hold: take a hold only when none stands, release only the hold this run took."""
import json
import subprocess

import pytest

from performance.harnesses import cluster_hold

NODE_A = ["ssh", "-o", "BatchMode=yes", "code@node-a"]


class Node:
    """Stands in for ``sparkring hold`` on Node A, reached through ``runner``."""

    def __init__(self, standing=None, *, unsupported=False, refuse_take=None):
        self.standing, self.unsupported, self.refuse_take = standing, unsupported, refuse_take
        self.calls = []

    def __call__(self, argv, **kwargs):
        script = argv[-1]
        self.calls.append(script)
        assert "sudo -n /usr/bin/sparkring hold" in script and script.rstrip().endswith("--json 2>&1")
        if self.unsupported:
            out = "usage: sparkring ... invalid choice: 'hold'"
            return subprocess.CompletedProcess(argv, 2, out, "")
        if "--release" in script:
            self.standing = None
            return subprocess.CompletedProcess(argv, 0, json.dumps({"state": "released"}), "")
        if "--reason" in script:
            if self.refuse_take:
                return subprocess.CompletedProcess(argv, 1, self.refuse_take, "")
            self.standing = {"holder": "code", "reason": "acceptance run of qwen", "since": 1000.0, "until": 44200.0}
            return subprocess.CompletedProcess(argv, 0, json.dumps({"state": "held", "hold": self.standing}), "")
        return subprocess.CompletedProcess(argv, 0, json.dumps({"state": "held" if self.standing else "none",
                                                                 "hold": self.standing}), "")


@pytest.fixture(autouse=True)
def holding(monkeypatch):
    monkeypatch.delenv(cluster_hold.SKIP, raising=False)


def test_takes_a_hold_for_the_run_and_releases_it():
    node = Node()
    with cluster_hold.held("acceptance run of qwen", ssh=NODE_A, where="Node A", runner=node) as field:
        assert field["state"] == "held" and field["holder"] == "code" and node.standing is not None
        assert "--until 12h" in node.calls[1]
    assert node.standing is None and field["released"] is True
    assert sum("--release" in call for call in node.calls) == 1


def test_a_standing_hold_is_left_alone():
    standing = {"holder": "alice", "reason": "checkpoint experiment", "since": 5.0, "until": None}
    node = Node(dict(standing))
    with cluster_hold.held("acceptance run of qwen", ssh=NODE_A, runner=node) as field:
        assert field["state"] == "already held" and field["holder"] == "alice"
    assert node.standing == standing and not any("--reason" in c or "--release" in c for c in node.calls)


def test_a_replaced_hold_is_not_released():
    node = Node()
    with cluster_hold.held("acceptance run of qwen", ssh=NODE_A, runner=node) as field:
        node.standing = {"holder": "bob", "reason": "override", "since": 2000.0, "until": None}
    assert field["released"] is False and "replaced" in field["release_detail"]
    assert node.standing["holder"] == "bob"


def test_failures_are_recorded_and_the_run_goes_ahead(monkeypatch):
    node = Node(refuse_take="sudo: a password is required for code@node-a")
    with cluster_hold.held("acceptance run of qwen", ssh=NODE_A, runner=node) as field:
        pass
    assert field["state"] == "not held" and "NODE_A" in field["detail"] and "node-a" not in field["detail"]
    assert cluster_hold.take("x", ssh=NODE_A, runner=Node(unsupported=True))["state"] == "unsupported"

    def broken(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 60)
    assert cluster_hold.take("x", ssh=NODE_A, runner=broken)["state"] == "not held"
    monkeypatch.setenv(cluster_hold.SKIP, "skip")
    calls = Node()
    assert cluster_hold.take("x", runner=calls)["detail"] == f"{cluster_hold.SKIP}=skip" and calls.calls == []


def test_sigterm_during_the_run_still_releases():
    import os
    import signal
    node = Node()
    with pytest.raises(SystemExit):
        with cluster_hold.held("acceptance run of qwen", ssh=NODE_A, runner=node):
            os.kill(os.getpid(), signal.SIGTERM)
    assert node.standing is None
