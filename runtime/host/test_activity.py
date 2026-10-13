"""Activity uses recorded state and at most one metrics sample per neighbour."""
import json
import os

import pytest

from runtime.host import activity, dashboard, node, placement, slices
from runtime.host.install_errors import NeedsInput


@pytest.mark.skipif(os.name != "posix", reason="Linux shared flock")
def test_grading_and_yes_never_bypasses_blocked(tmp_path):
    quiet = tmp_path / "quiet"
    assert activity.assess(tmp_path, (4, 5), quiet_path=quiet)["grade"] == "clear"
    with slices.hold(tmp_path, (0, 1), "install fixture"):
        warning = activity.assess(tmp_path, (4, 5), quiet_path=quiet)
        assert warning["grade"] == "may-be-affected"
        assert "Node A's download disk" in warning["message"] and "Sparks 0,1" in warning["message"]
        with pytest.raises(NeedsInput):
            activity.confirm(warning, interactive=False)
        activity.confirm(warning, yes=True)
        blocked = activity.assess(tmp_path, (0, 1), quiet_path=quiet)
        assert blocked["grade"] == "blocked"
        with pytest.raises(NeedsInput) as caught:
            activity.confirm(blocked, yes=True)
        assert caught.value.document()["details"]["activity"]["grade"] == "blocked"


def test_one_metrics_read_per_neighbour_and_quiet_suppresses_all(tmp_path, monkeypatch):
    node.save(tmp_path, "cluster.json", {})
    monkeypatch.setattr(placement, "layout_of", lambda cluster: {"shape": "cycle", "size": 8})
    monkeypatch.setattr(activity.installer, "connection", lambda lock: {"api_url": lock["api"]})
    for slot in ((0, 1), (2, 3), (6, 7)):
        directory = tmp_path / ("deployment-" + str(slot[0]))
        node.save(directory, "deployment.lock.json", {"api": f"http://192.0.2.{slot[0] + 10}:8000/v1"})
        node.save(directory, "state.json", {"operation": "up", "complete": True})
        placement.record(tmp_path, slot, directory)
    seen = []
    def metrics(url):
        seen.append(url)
        return {"running": 4, "waiting": 0}
    quiet = tmp_path / "quiet"
    value = activity.assess(tmp_path, (4, 5), metrics=metrics, quiet_path=quiet)
    assert len(seen) == len(set(seen)) == 2
    assert all(url.endswith("/metrics") for url in seen)
    assert value["grade"] == "may-be-affected"
    assert len(value["reasons"]) == 2
    dashboard.quiet_hold(quiet, "decode-benchmark", ttl=60)
    seen.clear()
    value = activity.assess(tmp_path, (4, 5), metrics=metrics, quiet_path=quiet)
    assert seen == []
    assert "decode-benchmark" in value["message"]
    assert all(row["traffic"] == "unknown" for row in value["observations"])


def test_review_changes_only_for_overlapping_deployments(tmp_path):
    first = activity.fingerprint(tmp_path, (4, 5))
    other = tmp_path / "other"
    node.save(other, "deployment.lock.json", {})
    placement.record(tmp_path, (0, 1), other)
    assert activity.fingerprint(tmp_path, (4, 5)) == first
    placement.record(tmp_path, (4, 5), other)
    with pytest.raises(NeedsInput, match="changed since preview"):
        activity.require_review(tmp_path, (4, 5), first)


def test_json_warning_is_serializable(tmp_path):
    value = activity.assess(tmp_path, None, quiet_path=tmp_path / "quiet")
    assert json.loads(json.dumps(value))["schema"] == "sparkring-activity/v1"
