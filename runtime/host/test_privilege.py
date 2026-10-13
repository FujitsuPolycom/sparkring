"""The sparkring group: what it may read, what stays root's, and the redaction of the operator view.

Offline: no host is contacted, no group is created and, unless a test fakes
root, the running user's own permissions decide the outcomes (a non-root run
exercises the no-op paths; the root paths are faked).
"""
import json
import os
import stat
import time
from pathlib import Path

from runtime.host import controller, node, operator_view, privilege, progress, recovery


# The real publication; the autouse fixture no_operator_view (conftest.py) replaces it.
REAL_PUBLISH = operator_view.publish


def test_group_membership_follows_the_process_groups(monkeypatch):
    monkeypatch.setattr(privilege, "group_gid", lambda name=privilege.GROUP: 4242)
    monkeypatch.setattr(os, "geteuid", lambda: 1000, raising=False)
    assert privilege.member([1000, 4242]) is True
    assert privilege.member([1000]) is False
    # Root reads regardless of group membership.
    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
    assert privilege.member([]) is True


def test_without_the_group_nobody_but_root_reads(monkeypatch):
    monkeypatch.setattr(privilege, "group_gid", lambda name=privilege.GROUP: None)
    monkeypatch.setattr(os, "geteuid", lambda: 1000, raising=False)
    assert privilege.member([4242]) is False
    assert privilege.reader_request() == "run the command with sudo"


def test_redact_removes_site_addresses_and_keeps_ports_ranks_and_names():
    document = {
        "api_url": "http://192.0.2.50:8000/v1", "rank": 3, "model": "glm53",
        "hosts": ["root@192.0.2.10", "sparkops@198.51.100.1"],
        "note": "route to 198.51.100.17 missing", "port": 8000,
        "nested": {"address": "203.0.113.9", "state": "serving"},
    }
    redacted = privilege.redact(document)
    text = json.dumps(redacted)
    assert "192.0.2." not in text and "198.51.100." not in text and "203.0.113." not in text
    assert redacted["api_url"] == f"http://{privilege.REDACTED}:8000/v1"
    assert redacted["rank"] == 3 and redacted["port"] == 8000 and redacted["model"] == "glm53"
    assert redacted["hosts"] == [f"root@{privilege.REDACTED}", f"sparkops@{privilege.REDACTED}"]
    assert redacted["nested"]["address"] == privilege.REDACTED
    # The input is untouched.
    assert document["api_url"] == "http://192.0.2.50:8000/v1"


def test_share_is_a_no_off_change_without_root_or_the_group(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    target.write_text("{}\n", encoding="utf-8")
    before = stat.S_IMODE(target.stat().st_mode)
    monkeypatch.setattr(privilege, "group_gid", lambda name=privilege.GROUP: None)
    assert privilege.share(target, 0o640) is False
    assert privilege.share_directory(tmp_path, 0o750) is False
    assert stat.S_IMODE(target.stat().st_mode) == before


def test_share_sets_the_group_and_the_group_readable_mode(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    target.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(privilege, "running_as_root", lambda: True)
    monkeypatch.setattr(privilege, "group_gid", lambda name=privilege.GROUP: 4242)
    chowned = []
    monkeypatch.setattr(os, "chown", lambda path, uid, gid: chowned.append((Path(path), gid)))
    assert privilege.share(target, 0o640) is True
    assert chowned == [(target, 4242)]
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_share_directory_sets_the_setgid_bit(tmp_path, monkeypatch):
    monkeypatch.setattr(privilege, "running_as_root", lambda: True)
    monkeypatch.setattr(privilege, "group_gid", lambda name=privilege.GROUP: 4242)
    monkeypatch.setattr(os, "chown", lambda path, uid, gid: None)
    assert privilege.share_directory(tmp_path, 0o750) is True
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o2750


def test_installation_logs_are_group_readable_as_root_and_root_only_elsewhere(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(privilege, "running_as_root", lambda: True)
    monkeypatch.setattr(privilege, "group_gid", lambda name=privilege.GROUP: 4242)
    monkeypatch.setattr(os, "chown", lambda path, uid, gid: None)
    with progress.run("setup"):
        pass
    for name in ("install.log", "install-details.log"):
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == 0o640
    assert stat.S_IMODE(tmp_path.stat().st_mode) & stat.S_ISGID


def test_installation_logs_stay_root_only_without_the_group(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(privilege, "group_gid", lambda name=privilege.GROUP: None)
    with progress.run("setup"):
        pass
    for name in ("install.log", "install-details.log"):
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == 0o600


def test_unreadable_names_a_directory_this_user_may_not_enter(tmp_path):
    closed = tmp_path / "closed"
    closed.mkdir()
    closed.chmod(0o000)
    try:
        assert progress.unreadable(closed) is True
    finally:
        closed.chmod(0o750)
    assert progress.unreadable(tmp_path) is False
    assert progress.unreadable(tmp_path / "absent") is False


def test_logs_names_the_group_when_the_log_dir_is_closed_to_this_user(monkeypatch, capsys):
    monkeypatch.setattr(progress, "unreadable", lambda path: True)
    monkeypatch.setattr(privilege, "group_gid", lambda name=privilege.GROUP: 4242)
    assert progress.main([]) == 2
    assert "sparkring group" in capsys.readouterr().err


def test_logs_without_the_group_names_sudo(monkeypatch, capsys):
    monkeypatch.setattr(progress, "unreadable", lambda path: True)
    monkeypatch.setattr(privilege, "group_gid", lambda name=privilege.GROUP: None)
    assert progress.main([]) == 2
    assert "sudo" in capsys.readouterr().err


DOCUMENT = {
    "schema": operator_view.SCHEMA, "written_at": time.time() - 12,
    "node": {"state": "network-configured", "next_action": "sparkring models", "rank": 0},
    "cluster": {"plan_id": "plan-1", "size": 4},
    "deployments": [{"id": "glm53", "profile": "glm53", "model": "GLM-5.3", "positions": [0, 1, 2, 3],
                     "api_port": 8000, "backend": "compose", "stopped": False,
                     "recovery": {"enabled": True, "failures": 1, "stopped": False}}],
    "recovery_timer": True,
}


def test_the_operator_view_lines_describe_the_recorded_state():
    lines = operator_view.lines(DOCUMENT, now=lambda: DOCUMENT["written_at"] + 12)
    assert "recorded 12s ago" in lines[0]
    assert any("network-configured" in line for line in lines)
    assert any("glm53" in line and "serving" in line and "8000" in line for line in lines)
    assert any("automatic recovery: on" in line and "1 failed attempt" in line for line in lines)


def test_publish_writes_the_redacted_view_group_readable(tmp_path, monkeypatch):
    state = tmp_path / "controller"
    state.mkdir()
    (state / "cluster.json").write_text(json.dumps({"plan": {"id": "plan-1"}}), encoding="utf-8")
    observation = {"schema": "sparkring-node-status/v1", "state": "network-configured",
                   "hostname": "spark-a", "rank": 0, "warnings": ["peer 192.0.2.11 did not answer"]}
    monkeypatch.setattr(privilege, "running_as_root", lambda: True)
    monkeypatch.setattr(privilege, "group_gid", lambda name=privilege.GROUP: 4242)
    monkeypatch.setattr(os, "chown", lambda path, uid, gid: None)
    monkeypatch.setattr("runtime.host.node.status", lambda **kwargs: observation)
    target = tmp_path / "operator-status.json"
    assert REAL_PUBLISH(state, target=target) is True
    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["schema"] == operator_view.SCHEMA
    assert written["node"]["state"] == "network-configured"
    assert "192.0.2." not in target.read_text(encoding="utf-8")
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_publish_skips_a_non_root_caller_a_groupless_host_and_a_node_without_a_controller(
        tmp_path, monkeypatch):
    state = tmp_path / "controller"
    state.mkdir()
    (state / "cluster.json").write_text(json.dumps({"plan": {}}), encoding="utf-8")
    monkeypatch.setattr(privilege, "running_as_root", lambda: False)
    monkeypatch.setattr(privilege, "group_gid", lambda name=privilege.GROUP: 4242)
    assert REAL_PUBLISH(state, target=tmp_path / "view.json") is False
    monkeypatch.setattr(privilege, "running_as_root", lambda: True)
    monkeypatch.setattr(privilege, "group_gid", lambda name=privilege.GROUP: None)
    assert REAL_PUBLISH(state, target=tmp_path / "view.json") is False
    monkeypatch.setattr(privilege, "group_gid", lambda name=privilege.GROUP: 4242)
    assert REAL_PUBLISH(tmp_path / "empty", target=tmp_path / "view.json") is False
    assert not (tmp_path / "view.json").exists()


def test_load_rejects_a_document_that_is_not_an_operator_view(tmp_path):
    target = tmp_path / "operator-status.json"
    target.write_text(json.dumps({"schema": "other/v9"}), encoding="utf-8")
    assert operator_view.load(target) is None
    target.write_text("not json", encoding="utf-8")
    assert operator_view.load(target) is None
    assert operator_view.load(tmp_path / "absent.json") is None


def _refused(*args, **kwargs):
    raise PermissionError(13, "Permission denied", "/var/lib/sparkring/controller")


def test_status_as_a_group_member_prints_the_recorded_view_and_changes_nothing(monkeypatch, capsys):
    monkeypatch.setattr(privilege, "running_as_root", lambda: False)
    monkeypatch.setattr(privilege, "member", lambda gids=None: True)
    monkeypatch.setattr(controller, "active_deployments", _refused)
    monkeypatch.setattr(operator_view, "load", lambda target=operator_view.PATH: dict(DOCUMENT))
    published = []
    monkeypatch.setattr(operator_view, "publish", lambda *args, **kwargs: published.append(args) or False)
    result = controller.main(["status"])
    assert result == 0
    out = capsys.readouterr().out
    assert "Operator view" in out and "glm53" in out
    # A read by a group member records nothing and refreshes nothing.
    assert published == []


def test_status_refuses_a_user_outside_the_group(monkeypatch, capsys):
    monkeypatch.setattr(privilege, "running_as_root", lambda: False)
    monkeypatch.setattr(privilege, "member", lambda gids=None: False)
    monkeypatch.setattr(controller, "active_deployments", _refused)
    result = controller.main(["status"])
    assert result == 2
    assert "sparkring group" in capsys.readouterr().err


def test_status_as_a_group_member_refuses_the_live_refresh(monkeypatch, capsys):
    monkeypatch.setattr(privilege, "running_as_root", lambda: False)
    monkeypatch.setattr(privilege, "member", lambda gids=None: True)
    monkeypatch.setattr(controller, "active_deployments", _refused)
    monkeypatch.setattr(operator_view, "load", lambda target=operator_view.PATH: dict(DOCUMENT))
    assert controller.main(["status", "--refresh"]) == 2
    assert "needs root" in capsys.readouterr().err


def test_status_without_a_recorded_view_says_what_writes_it(monkeypatch, capsys):
    monkeypatch.setattr(privilege, "running_as_root", lambda: False)
    monkeypatch.setattr(privilege, "member", lambda gids=None: True)
    monkeypatch.setattr(controller, "active_deployments", _refused)
    monkeypatch.setattr(operator_view, "load", lambda target=operator_view.PATH: None)
    assert controller.main(["status"]) == 2
    assert "sudo sparkring status" in capsys.readouterr().err


def test_status_as_root_on_a_readable_controller_keeps_the_live_path(monkeypatch, capsys):
    monkeypatch.setattr(privilege, "running_as_root", lambda: True)
    monkeypatch.setattr(controller, "active_deployments", lambda report=False: [])
    monkeypatch.setattr(node, "status", lambda **kwargs: {"state": "not-configured", "next_action": "sparkring setup"})
    monkeypatch.setattr(operator_view, "publish", lambda *args, **kwargs: False)
    assert controller.main(["status"]) == 0
    assert "Operator view" not in capsys.readouterr().out


def test_recover_status_as_a_group_member_reads_the_recorded_view(monkeypatch, capsys):
    monkeypatch.setattr(privilege, "running_as_root", lambda: False)
    monkeypatch.setattr(privilege, "member", lambda gids=None: True)
    monkeypatch.setattr(recovery, "describe", _refused)
    monkeypatch.setattr(operator_view, "load", lambda target=operator_view.PATH: dict(DOCUMENT))
    result = recovery.main(["status"])
    assert result == 0
    out = capsys.readouterr().out
    assert "glm53" in out and "automatic recovery: on" in out


def test_recover_status_refuses_a_user_outside_the_group(monkeypatch, capsys):
    monkeypatch.setattr(privilege, "running_as_root", lambda: False)
    monkeypatch.setattr(privilege, "member", lambda gids=None: False)
    monkeypatch.setattr(recovery, "describe", _refused)
    assert recovery.main(["status"]) == 2
    assert "sparkring group" in capsys.readouterr().err
