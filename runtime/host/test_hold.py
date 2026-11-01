"""Operator holds: the record, the refusals other operators get, overrides and where a hold is shown."""
import json
import os
import time
from pathlib import Path

import pytest

from runtime.host import activity, checkpoints, controller, dashboard_admin, hold, operator_view, privilege, storage
from runtime.host.install_errors import NeedsInput

NOW = 1_800_000_000.0


@pytest.fixture
def as_operator(monkeypatch):
    """Run as the operator ``name``, as sudo records it."""
    def switch(name):
        monkeypatch.setenv("SUDO_USER", name)
    return switch


def _overrides(root):
    path = Path(root) / hold.OVERRIDES
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def test_until_takes_a_duration_or_a_future_time():
    assert hold.parse_until("90m", NOW) == NOW + 5400
    assert hold.parse_until("4h", NOW) == NOW + 4 * 3600
    assert hold.parse_until("2d", NOW) == NOW + 2 * 86400
    assert hold.parse_until("2027-03-15T12:00+00:00", NOW) == 1_805_112_000.0
    with pytest.raises(ValueError, match="not in the future"):
        hold.parse_until("2020-01-01T00:00", NOW)
    for text in ("soon", "0h", "4 h", "-1d", ""):
        with pytest.raises(ValueError, match="--until takes"):
            hold.parse_until(text, NOW)


def test_the_operator_is_the_sudo_login(monkeypatch):
    monkeypatch.setenv("SUDO_USER", "alice")
    assert hold.operator() == "alice"
    monkeypatch.setenv("SUDO_USER", "root")
    monkeypatch.setattr(hold.getpass, "getuser", lambda: "root")
    assert hold.operator() == "root"


def test_take_show_replace_and_expire(tmp_path):
    record = hold.take(tmp_path, "  checkpoint\nexperiment ", NOW + 3600, now=NOW, who="alice")
    assert record["reason"] == "checkpoint experiment" and record["holder"] == "alice"
    assert hold.current(tmp_path, NOW + 10) == record
    assert hold.foreign(tmp_path, NOW + 10, who="alice") is None
    assert hold.foreign(tmp_path, NOW + 10, who="bob") == record
    # The holder replaces their own hold without an override.
    replaced = hold.take(tmp_path, "acceptance run", None, now=NOW + 20, who="alice")
    assert hold.current(tmp_path, NOW + 10 ** 6) == replaced
    assert _overrides(tmp_path) == []
    hold.take(tmp_path, "short", NOW + 60, now=NOW + 30, who="alice")
    assert hold.current(tmp_path, NOW + 59) is not None and hold.current(tmp_path, NOW + 90) is None
    assert hold.foreign(tmp_path, NOW + 90, who="bob") is None
    assert "since" in hold.text(replaced) and hold.text(replaced).startswith("Held by alice since ")
    assert hold.text(replaced).endswith(": acceptance run")


def test_reason_is_required_and_bounded(tmp_path):
    for reason in ("", "   ", "x" * (hold.REASON_LIMIT + 1)):
        with pytest.raises(ValueError, match="--reason is 1 to"):
            hold.take(tmp_path, reason, now=NOW, who="alice")


def test_an_unreadable_or_foreign_schema_record_is_no_hold(tmp_path):
    (tmp_path / hold.NAME).write_text("{not json", encoding="utf-8")
    assert hold.current(tmp_path, NOW) is None
    (tmp_path / hold.NAME).write_text(json.dumps({"schema": "other/v1", "holder": "alice"}), encoding="utf-8")
    assert hold.current(tmp_path, NOW) is None


def test_another_operators_hold_needs_an_override_to_replace_or_release(tmp_path):
    hold.take(tmp_path, "checkpoint experiment", now=NOW, who="alice")
    with pytest.raises(NeedsInput, match="Held by alice.*--override-hold") as caught:
        hold.take(tmp_path, "mine now", now=NOW, who="bob")
    assert caught.value.field == "hold" and caught.value.details["hold"]["holder"] == "alice"
    with pytest.raises(NeedsInput, match="ask alice"):
        hold.release(tmp_path, now=NOW, who="bob")
    assert hold.current(tmp_path, NOW)["holder"] == "alice"
    cleared = hold.release(tmp_path, override=True, now=NOW + 5, who="bob")
    assert cleared["holder"] == "alice" and hold.current(tmp_path, NOW + 5) is None
    [entry] = _overrides(tmp_path)
    assert entry["schema"] == hold.OVERRIDE_SCHEMA and entry["operator"] == "bob"
    assert entry["operation"] == "hold --release" and entry["hold"]["reason"] == "checkpoint experiment"
    assert hold.release(tmp_path, now=NOW, who="bob") is None


def test_activity_refuses_another_operators_hold_until_overridden(tmp_path, as_operator):
    quiet = tmp_path / "quiet"
    as_operator("alice")
    hold.take(tmp_path, "acceptance run", now=time.time())
    own = activity.assess(tmp_path, (0, 1), quiet_path=quiet, operation="up")
    assert own["grade"] == "clear"
    as_operator("bob")
    held = activity.assess(tmp_path, (0, 1), quiet_path=quiet, operation="up")
    assert held["grade"] == "held" and held["reasons"][0]["operation"] == "up"
    assert "Held by alice" in held["message"] and "sparkring up" in held["message"]
    assert json.loads(json.dumps(held))["reasons"][0]["hold"]["holder"] == "alice"
    # --yes approves advisory warnings only; it never overrides a hold.
    for yes in (False, True):
        with pytest.raises(NeedsInput, match="--override-hold") as caught:
            activity.confirm(held, yes=yes, interactive=False)
        assert caught.value.field == "hold"
    assert _overrides(tmp_path) == []
    # A plan shows the hold and changes nothing.
    activity.confirm(held, plan=True)
    assert _overrides(tmp_path) == []
    activity.confirm(held, interactive=False, override_hold=True)
    [entry] = _overrides(tmp_path)
    assert entry["operator"] == "bob" and entry["operation"] == "up" and entry["positions"] == [0, 1]
    assert entry["hold"]["holder"] == "alice"


def test_a_hold_applies_to_down_and_names_the_command(tmp_path, as_operator):
    as_operator("alice")
    hold.take(tmp_path, "measuring", now=time.time())
    as_operator("bob")
    value = activity.assess(tmp_path, None, quiet_path=tmp_path / "quiet", operation="down")
    assert value["grade"] == "held"
    value = activity.assess(tmp_path, None, quiet_path=tmp_path / "quiet", command="recover off")
    assert "sparkring recover off" in value["message"] and value["reasons"][0]["operation"] == "recover off"


def test_an_override_still_needs_yes_for_advisory_warnings(tmp_path, as_operator):
    quiet = tmp_path / "quiet"
    as_operator("alice")
    hold.take(tmp_path, "measuring", now=time.time())
    as_operator("bob")
    from runtime.host import dashboard
    dashboard.quiet_hold(quiet, "decode-benchmark", ttl=60)
    value = activity.assess(tmp_path, (4, 5), quiet_path=quiet)
    assert value["grade"] == "held"
    assert {row["grade"] for row in value["reasons"]} == {"held", "may-be-affected"}
    with pytest.raises(NeedsInput, match="--yes"):
        activity.confirm(value, interactive=False, override_hold=True)
    # The refused run changed nothing, so it records no override.
    assert _overrides(tmp_path) == []
    activity.confirm(value, yes=True, interactive=False, override_hold=True)
    assert len(_overrides(tmp_path)) == 1


@pytest.mark.skipif(os.name != "posix", reason="Linux shared flock")
def test_a_lock_blocks_even_with_an_override(tmp_path, as_operator):
    from runtime.host import slices
    as_operator("alice")
    hold.take(tmp_path, "measuring", now=time.time())
    as_operator("bob")
    with slices.hold(tmp_path, (0, 1), "install fixture"):
        value = activity.assess(tmp_path, (0, 1), quiet_path=tmp_path / "quiet")
        assert value["grade"] == "blocked"
        with pytest.raises(NeedsInput) as caught:
            activity.confirm(value, yes=True, override_hold=True)
        assert caught.value.field == "activity"
    assert _overrides(tmp_path) == []


def test_require_for_commands_outside_the_assessment(tmp_path, as_operator, capsys):
    as_operator("alice")
    assert hold.require(tmp_path, "sparkring storage --release") is None
    hold.take(tmp_path, "measuring", now=time.time())
    assert hold.require(tmp_path, "sparkring storage --release") is None
    as_operator("bob")
    with pytest.raises(NeedsInput, match="sparkring storage --release would change"):
        hold.require(tmp_path, "sparkring storage --release")
    assert hold.require(tmp_path, "sparkring storage --release", plan=True)["holder"] == "alice"
    assert _overrides(tmp_path) == []
    hold.require(tmp_path, "sparkring storage --release", override=True)
    assert [entry["operation"] for entry in _overrides(tmp_path)] == ["sparkring storage --release"]
    assert "the override is recorded" in capsys.readouterr().err


@pytest.mark.parametrize("module", [storage, checkpoints])
def test_releases_refuse_during_another_operators_hold(module, tmp_path, as_operator, monkeypatch):
    monkeypatch.setattr(module, "_administrator", lambda: True)
    released = []
    monkeypatch.setattr(module, "release_cluster", lambda *args, **kwargs: released.append(args) or {"state": "released"})
    as_operator("alice")
    hold.take(tmp_path, "measuring", now=time.time())
    as_operator("bob")
    path = "/srv/sparkring/tp4/cache/old"
    assert module.main(["--release", path, "--yes"], state_root=tmp_path, invoke=lambda *a, **k: None) == 3
    assert released == []
    assert module.main(["--release", path, "--yes", "--override-hold"], state_root=tmp_path,
                       invoke=lambda *a, **k: None) == 0
    assert len(released) == 1 and len(_overrides(tmp_path)) == 1


def test_automatic_recovery_waits_during_any_hold(tmp_path, monkeypatch, as_operator):
    from runtime.host import placement, recovery, slices
    import contextlib
    monkeypatch.setattr(controller, "STATE", tmp_path)
    monkeypatch.setattr(placement, "from_lock", lambda lock: None)
    monkeypatch.setattr(slices, "hold", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(recovery, "load", lambda **kwargs: pytest.fail("a held cluster is not acted on"))
    as_operator("alice")
    hold.take(tmp_path, "acceptance run", now=time.time())
    outcome = recovery.attempt(tmp_path / "deployment", {}, {}, {}, apply=lambda operation: pytest.fail("acted"))
    assert outcome["state"] == "held" and "Held by alice" in outcome["summary"]
    assert "acts after the hold ends" in outcome["summary"]
    assert recovery.PRECEDENCE.index("held") < recovery.PRECEDENCE.index("busy")


def test_the_operator_view_shows_a_standing_hold_first(tmp_path, as_operator):
    as_operator("alice")
    record = hold.take(tmp_path, "checkpoint experiment", now=NOW, who="alice")
    document = {"schema": operator_view.SCHEMA, "written_at": NOW, "deployments": [],
                "hold": {key: record[key] for key in ("holder", "reason", "since", "until")}}
    lines = operator_view.lines(document, now=lambda: NOW + 5)
    assert lines[1].startswith("Held by alice since ") and lines[1].endswith(": checkpoint experiment")
    expired = {**document, "hold": {**document["hold"], "until": NOW + 1}}
    assert not any(line.startswith("Held by") for line in operator_view.lines(expired, now=lambda: NOW + 5))


def test_the_command_takes_shows_and_releases(tmp_path, monkeypatch, capsys, as_operator):
    (tmp_path / "cluster.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(controller, "STATE", tmp_path)
    monkeypatch.setattr(privilege, "running_as_root", lambda: True)
    monkeypatch.setattr(dashboard_admin, "refresh_site", lambda **kwargs: True)
    as_operator("alice")
    assert hold.main(["--reason", "acceptance run", "--until", "4h"]) == 0
    assert capsys.readouterr().out.startswith("Held by alice since ")
    assert hold.main(["--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["state"] == "held" and shown["hold"]["reason"] == "acceptance run"
    assert shown["hold"]["until"] - shown["hold"]["since"] == pytest.approx(4 * 3600)
    as_operator("bob")
    assert hold.main(["--release"]) == 3
    assert "ask alice" in capsys.readouterr().err
    assert hold.main(["--release", "--override-hold"]) == 0
    assert capsys.readouterr().out.strip() == "Hold released."
    assert hold.main([]) == 0 and capsys.readouterr().out.strip() == "No hold stands."
    for argv in (["--until", "4h"], ["--release", "--reason", "x"], ["--reason", "x", "--until", "yesterday"]):
        assert hold.main(argv) == 2
    monkeypatch.setattr(privilege, "running_as_root", lambda: False)
    assert hold.main(["--reason", "x"]) == 2
    assert "Run sudo sparkring hold" in capsys.readouterr().err
    # A member who cannot read hold.json reads the operator view, which may carry an expired hold.
    def unreadable(root, now=None):
        raise PermissionError(root)
    monkeypatch.setattr(hold, "current", unreadable)
    monkeypatch.setattr(privilege, "member", lambda: True)
    published = {"holder": "alice", "reason": "acceptance run", "since": time.time() - 7200}
    for until, state in ((time.time() + 3600, "held"), (time.time() - 60, "none")):
        monkeypatch.setattr(operator_view, "load", lambda: {"hold": {**published, "until": until}})
        assert hold.main(["--json"]) == 0
        assert json.loads(capsys.readouterr().out)["state"] == state


def test_the_top_level_command_dispatches_hold():
    source = (Path(__file__).resolve().parents[2] / "scripts" / "sparkring.py").read_text(encoding="utf-8")
    assert 'raw[0] == "hold"' in source and "from runtime.host.hold import main" in source
