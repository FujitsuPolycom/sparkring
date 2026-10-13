"""The dashboard's root actions and the commands that call them: the site file written from the
deployment records, its refresh after install, up, down and automatic recovery (a failed refresh
never fails them) and the reload that makes the running service re-read it, the service's start at
the end of setup, its persistent off switch and setup's plan line. No host is contacted and no
systemctl runs."""
import json
from pathlib import Path
import subprocess

import pytest

from runtime.common import installer
from runtime.host import controller, dashboard, dashboard_admin, fabric_bandwidth, install_workflow
from runtime.host import single_uplink
from runtime.host.test_fabric_ssh import cluster
from runtime.host.test_recovery import DEAD, Clock, pair, run  # noqa: F401 (fixture)
from runtime.host.test_setup_prompts import four_sparks  # noqa: F401 (fixture)

# The real functions; the autouse fixture no_dashboard_service (conftest.py) replaces them.
REAL_REFRESH_SITE = dashboard_admin.refresh_site
REAL_AFTER_SETUP = dashboard_admin.after_setup
REAL_RELOAD_SERVICE = dashboard_admin.reload_service
REAL_PAGE_URL = dashboard_admin.page_url
RELOAD = ["--no-block", "try-reload-or-restart", dashboard_admin.UNIT]
NODE_A_LAN = "192.0.2.50"
ROOT = Path(__file__).resolve().parents[2]


class Systemctl:
    def __init__(self, fail=()):
        self.calls, self.fail = [], set(fail)

    def __call__(self, argv, **kwargs):
        self.calls.append(argv[1:])
        if argv[1] in self.fail:
            raise subprocess.CalledProcessError(1, argv, stderr="Unit failed")
        return subprocess.CompletedProcess(argv, 0, "", "")


@pytest.fixture
def records(tmp_path, monkeypatch):
    """A controller with the whole ring's deployment (stopped) and one on Sparks 2 and 3."""
    state = tmp_path / "controller"
    for name, placement in (("whole", None), ("profile-on-2-3", [2, 3])):
        directory = state / "deployments" / name
        directory.mkdir(parents=True)
        site = {"name": "test-p-" + name[:4] + "-abc123", "ranks": [{"rank": 0, "host": "root@192.0.2.10"},
                                                                    {"rank": 1, "host": "root@192.0.2.11"}],
                **({"placement": placement} if placement else {})}
        (directory / "deployment.lock.json").write_text(json.dumps({
            "id": "f" * 64, "backend": "compose", "selection": {"profile": "p-" + name}, "site": site}))
        slot = state if placement is None else state / "slots" / "2-3"
        slot.mkdir(parents=True, exist_ok=True)
        (slot / "active.json").write_text(json.dumps({"path": str(directory)}))
    (state / "deployments" / "whole" / "state.json").write_text(json.dumps({"operation": "down", "complete": True}))
    monkeypatch.setattr(controller, "STATE", state)
    monkeypatch.setattr(controller, "recorded_layout", lambda: {"size": 4})
    monkeypatch.setattr(installer, "connection", lambda lock: {
        "api_url": "http://192.0.2.12:8015/v1" if lock["site"].get("placement") else f"http://{NODE_A_LAN}:8000/v1",
        "model": "M", "port": 8015 if lock["site"].get("placement") else 8000})
    monkeypatch.setattr(dashboard, "SITE_PATH", tmp_path / "dashboard" / "site.json")
    monkeypatch.setattr(dashboard_admin, "SITE_PATH", tmp_path / "dashboard" / "site.json")
    return state


def test_the_site_file_lists_every_slots_active_deployment(records, tmp_path):
    site = dashboard_admin.write_site()
    whole, arc = site["deployments"]
    assert (whole["positions"], whole["api_host"], whole["api_port"], whole["stopped"]) == (
        [0, 1, 2, 3], NODE_A_LAN, 8000, True)
    assert (arc["id"], arc["positions"], arc["api_host"], arc["stopped"]) == ("profile-on-2-3", [2, 3], "192.0.2.12",
                                                                             True)
    # Container names come from installer.containers, the names the installer gives each rank.
    assert arc["containers"] == [{"position": 2, "name": "sr-test-p-prof-abc123-r0"},
                                 {"position": 3, "name": "sr-test-p-prof-abc123-r1"}]
    written = json.loads((tmp_path / "dashboard" / "site.json").read_text(encoding="utf-8"))
    assert dashboard.deployments_from_site(written, 4)[1]["api_port"] == 8015


def test_site_preserves_thinking_selection_and_only_public_serving_defaults(records):
    path = records / "deployments" / "whole" / "deployment.lock.json"
    lock = json.loads(path.read_text())
    lock["selection"]["target_variant"] = "nvfp4"
    lock["serving"] = {"reasoning_effort": "high", "unrelated": "private"}
    path.write_text(json.dumps(lock))
    site = dashboard_admin.site_from_records(records)
    row = dashboard.deployments_from_site(site, 4)[0]
    assert row["checkpoint"] == "nvfp4"
    assert row["serving"] == {"reasoning_effort": "high"}
    assert "private" not in json.dumps(site)


def test_a_failed_refresh_logs_one_line_and_never_raises(records, monkeypatch):
    def broken(output=None):
        raise PermissionError(13, "Permission denied", "/var/lib/sparkring/dashboard/site.json")
    monkeypatch.setattr(dashboard_admin, "write_site", broken)
    lines = []
    assert REAL_REFRESH_SITE(log=lines.append) is False
    assert len(lines) == 1 and lines[0].startswith("Dashboard: its site file was not refreshed (PermissionError")
    assert lines[0].endswith("sudo sparkring dashboard export writes it")
    monkeypatch.setattr(dashboard_admin, "write_site", lambda output=None: {"deployments": []})
    assert REAL_REFRESH_SITE(log=lines.append) is True and len(lines) == 1


@pytest.fixture
def reloads(tmp_path, monkeypatch):
    """The real refresh and reload request, with the site write stubbed and systemctl recorded."""
    if controller.STATE == Path("/var/lib/sparkring/controller"):
        monkeypatch.setattr(controller, "STATE", tmp_path / "state")
    monkeypatch.setattr(dashboard_admin, "DISABLED_PATH", tmp_path / "dashboard.disabled")
    monkeypatch.setattr(dashboard_admin, "refresh_site", REAL_REFRESH_SITE)
    monkeypatch.setattr(dashboard_admin, "reload_service", REAL_RELOAD_SERVICE)
    monkeypatch.setattr(dashboard_admin, "write_site", lambda output=None: {"deployments": []})
    systemctl = Systemctl()
    monkeypatch.setattr(dashboard_admin, "_systemctl", lambda *args, run=None: systemctl([None, *args]))
    return systemctl


def test_a_refresh_asks_the_running_service_to_reread_the_site_file(reloads, tmp_path):
    lines = []
    assert dashboard_admin.refresh_site(log=lines.append) is True
    assert reloads.calls == [RELOAD] and lines == []
    (tmp_path / "dashboard.disabled").write_text("off\n")
    assert dashboard_admin.refresh_site(log=lines.append) is True and reloads.calls == [RELOAD]


def test_a_refused_reload_logs_one_line_and_keeps_the_refresh(records, tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_admin, "DISABLED_PATH", tmp_path / "dashboard.disabled")
    monkeypatch.setattr(dashboard_admin, "reload_service", REAL_RELOAD_SERVICE)
    lines = []
    assert REAL_REFRESH_SITE(log=lines.append, run=Systemctl(fail={"--no-block"})) is True
    assert len(lines) == 1 and "was not asked to re-read its site file" in lines[0] and "Unit failed" in lines[0]
    assert (tmp_path / "dashboard" / "site.json").exists()
    missing = REAL_RELOAD_SERVICE(log=lines.append, run=lambda argv, **kwargs: (_ for _ in ()).throw(
        FileNotFoundError(2, "No such file or directory", "systemctl")))
    assert missing is False and len(lines) == 2 and "systemctl" in lines[1]


def test_export_reloads_the_service_only_for_its_own_site_file(records, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dashboard_admin.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(dashboard_admin, "DISABLED_PATH", tmp_path / "dashboard.disabled")
    monkeypatch.setattr(dashboard_admin, "reload_service", REAL_RELOAD_SERVICE)
    systemctl = Systemctl()
    assert dashboard_admin.export(str(dashboard_admin.SITE_PATH), run=systemctl) == 0
    assert systemctl.calls == [RELOAD] and "Wrote" in capsys.readouterr().out
    assert dashboard_admin.export(str(tmp_path / "copy-for-another-host.json"), run=systemctl) == 0
    assert systemctl.calls == [RELOAD]


@pytest.mark.parametrize("command", ["up", "down"])
def test_up_and_down_reload_the_service(command, reloads, monkeypatch):
    monkeypatch.setattr(controller, "lifecycle", lambda argv: 0)
    assert controller.main([command, "profile"]) == 0 and reloads.calls == [RELOAD]


def test_install_reloads_the_service(reloads, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(install_workflow, "execute", lambda args: {"state": "installed"})
    install_workflow.main(["--profile", "qwen38-flash-next-tp2", "--yes", "--json"])
    assert reloads.calls == [RELOAD]


def test_automatic_recovery_reloads_the_service_after_its_restart(pair, reloads):  # noqa: F811
    clock = Clock()
    run(clock, api=DEAD)
    clock.advance(60)
    assert run(clock, api=DEAD)["state"] == "recovered" and reloads.calls == [RELOAD]


@pytest.mark.parametrize("command", ["up", "down"])
def test_up_and_down_refresh_the_site_file_whether_they_finish_or_fail(command, monkeypatch, capsys):
    refreshed = []
    monkeypatch.setattr(dashboard_admin, "refresh_site", lambda **kwargs: refreshed.append(command))
    monkeypatch.setattr(controller, "lifecycle", lambda argv: 0)
    assert controller.main([command, "profile"]) == 0 and refreshed == [command]

    def fails(argv):
        raise ValueError("the deployment did not start")
    monkeypatch.setattr(controller, "lifecycle", fails)
    assert controller.main([command, "profile"]) == 2 and refreshed == [command, command]
    monkeypatch.setattr(controller, "lifecycle", lambda argv: 0)
    assert controller.main(["status"]) == 0 and len(refreshed) == 2


def test_install_refreshes_the_site_file_and_a_failed_refresh_keeps_its_result(records, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path / "logs"))
    calls = []
    monkeypatch.setattr(install_workflow, "execute", lambda args: calls.append("execute") or (_ for _ in ()).throw(
        ValueError("no checkpoint")))

    def broken(output=None):
        calls.append("refresh")
        raise OSError("disk full")
    monkeypatch.setattr(dashboard_admin, "write_site", broken)
    monkeypatch.setattr(dashboard_admin, "refresh_site", REAL_REFRESH_SITE)
    assert install_workflow.main(["--profile", "qwen38-flash-next-tp2", "--yes", "--json"]) == 2
    assert calls == ["execute", "refresh"]
    out, err = capsys.readouterr()
    assert json.loads(out)["state"] == "failed" and json.loads(out)["message"] == "no checkpoint"
    assert sum(line.startswith("Dashboard: its site file was not refreshed") for line in err.splitlines()) == 1
    # A plan changes no record and writes no site file.
    calls.clear()
    monkeypatch.setattr(install_workflow, "execute", lambda args: calls.append("execute") or {"state": "planned"})
    install_workflow.main(["--profile", "qwen38-flash-next-tp2", "--plan", "--json"])
    assert calls == ["execute"]


def test_automatic_recovery_refreshes_the_site_file_after_its_restart(pair, monkeypatch):  # noqa: F811
    refreshed = []
    monkeypatch.setattr(dashboard_admin, "refresh_site", lambda **kwargs: refreshed.append("refresh"))
    clock = Clock()
    run(clock, api=DEAD)
    assert refreshed == []                       # a check that only observes changes no record
    clock.advance(60)
    assert run(clock, api=DEAD)["state"] == "recovered"
    assert refreshed == ["refresh"]


def test_setup_starts_the_dashboard_after_recording_the_fabric(four_sparks, monkeypatch, tmp_path):  # noqa: F811
    _, plan, _, targets = four_sparks
    order = []
    monkeypatch.setattr(controller, "apply", lambda value, directory, **options: value)
    monkeypatch.setattr(fabric_bandwidth, "after_setup", lambda root, value: None)
    monkeypatch.setattr(controller.fabric, "finish_setup", lambda *args, **kwargs: order.append("fabric"))
    monkeypatch.setattr(dashboard_admin, "after_setup", lambda **kwargs: order.append("dashboard"))
    assert controller.setup([*targets, "--apply", "--yes", "--skip-enroll", "--output", str(tmp_path / "setup")]) == 0
    assert order == ["fabric", "dashboard"]


def test_automated_setup_starts_the_dashboard_after_recording_the_fabric(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "STATE", tmp_path / "state")
    value = cluster(2)
    installer.write(controller.STATE / "cluster.json", value)
    monkeypatch.setattr(single_uplink.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(single_uplink.distribution, "installed", lambda root: True)
    monkeypatch.setattr(single_uplink, "identity_key", lambda directory: (tmp_path / "key", "controller public key"))
    monkeypatch.setattr(single_uplink, "record_reason", lambda base: None)
    monkeypatch.setattr(single_uplink.controller, "collect", lambda targets: value["plan"]["nodes"])
    monkeypatch.setattr(single_uplink, "match_revisions", lambda targets, nodes, directory: nodes)
    monkeypatch.setattr(single_uplink.node, "read", lambda root, name: {"node_id": value["plan"]["nodes"][0]["node_id"]})
    monkeypatch.setattr(single_uplink.topology, "build_spec", lambda *args, **kwargs: value["plan"])
    monkeypatch.setattr(single_uplink.controller, "apply", lambda plan, directory, **options: plan)
    monkeypatch.setattr(fabric_bandwidth, "after_setup", lambda root, record: None)
    order = []
    monkeypatch.setattr(single_uplink.fabric, "finish_setup", lambda *args, **kwargs: order.append("fabric"))
    monkeypatch.setattr(dashboard_admin, "after_setup", lambda **kwargs: order.append("dashboard"))
    assert single_uplink.main(["--yes"]) == 0
    assert order == ["fabric", "dashboard"]


def test_after_setup_enables_and_restarts_the_service_unless_it_was_turned_off(records, tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_admin, "DISABLED_PATH", tmp_path / "dashboard.disabled")
    refreshed = []
    monkeypatch.setattr(dashboard_admin, "refresh_site", lambda **kwargs: refreshed.append(True))
    systemctl = Systemctl()
    assert REAL_AFTER_SETUP(run=systemctl) is True
    assert systemctl.calls == [["enable", dashboard_admin.UNIT], ["restart", dashboard_admin.UNIT]] and refreshed
    (tmp_path / "dashboard.disabled").write_text("off\n")
    systemctl.calls.clear()
    assert REAL_AFTER_SETUP(run=systemctl) is False and systemctl.calls == [] and len(refreshed) == 2


def test_a_service_that_does_not_start_never_fails_setup(records, tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_admin, "DISABLED_PATH", tmp_path / "dashboard.disabled")
    monkeypatch.setattr(dashboard_admin, "refresh_site", lambda **kwargs: True)
    lines = []
    assert REAL_AFTER_SETUP(run=Systemctl(fail={"restart"}), log=lines.append) is False
    assert len(lines) == 1 and "did not start" in lines[0] and "Unit failed" in lines[0]


def test_disable_keeps_the_dashboard_off_until_enable(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dashboard_admin, "DISABLED_PATH", tmp_path / "etc" / "dashboard.disabled")
    monkeypatch.setattr(dashboard_admin.os, "geteuid", lambda: 0, raising=False)
    systemctl = Systemctl()
    assert dashboard_admin.switch("disable", run=systemctl) == 0
    assert (tmp_path / "etc" / "dashboard.disabled").exists()
    assert systemctl.calls == [["disable", "--now", dashboard_admin.UNIT]]
    assert dashboard_admin.plan_line(NODE_A_LAN, "192.0.2.10") == (
        "SparkRing dashboard: off (sudo sparkring dashboard enable turns it on)")
    assert dashboard_admin.switch("enable", run=systemctl) == 0
    assert not (tmp_path / "etc" / "dashboard.disabled").exists()
    assert systemctl.calls[-1] == ["enable", "--now", dashboard_admin.UNIT]
    monkeypatch.setattr(dashboard_admin.os, "geteuid", lambda: 1000, raising=False)
    with pytest.raises(ValueError, match="sudo"):
        dashboard_admin.switch("disable", run=systemctl)


def test_the_plan_names_the_listen_address_and_port(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_admin, "DISABLED_PATH", tmp_path / "dashboard.disabled")
    assert dashboard_admin.plan_line(NODE_A_LAN, "192.0.2.10") == (
        f"SparkRing dashboard: http://{NODE_A_LAN}:8090/ on Node A and http://127.0.0.1:8090/ on Node A itself, read-only, "
        "LAN and loopback only (sparkring-dashboard.service; sudo sparkring dashboard disable turns it off)")
    assert dashboard_admin.plan_line(None, "192.0.2.10").startswith("SparkRing dashboard: http://192.0.2.10:8090/")
    summarize = (ROOT / "runtime" / "host" / "controller.py").read_text(encoding="utf-8")
    assert 'print(dashboard_admin.plan_line(api_address, plan["spec"]["hosts"][0].get("management_address")))' in summarize


class UnitState:
    """systemctl is-enabled / is-active for the dashboard unit: exit 0 for the states in `on`."""

    def __init__(self, *on, missing=False):
        self.on, self.missing, self.calls = set(on), missing, []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv[1:])
        if self.missing:
            raise FileNotFoundError("systemctl")
        return subprocess.CompletedProcess(argv, 0 if argv[1] in self.on else 1, "", "")


def test_the_install_card_names_the_page_only_while_its_service_is_on(tmp_path, monkeypatch):
    from runtime.host.test_dashboard import NODE_A_LAN as FABRIC_NODE_A_LAN, fabric_file
    import socket
    monkeypatch.setattr(dashboard_admin, "DISABLED_PATH", tmp_path / "dashboard.disabled")
    monkeypatch.setattr(socket, "gethostname", lambda: "spark0")
    path = fabric_file(tmp_path)
    page = f"http://{FABRIC_NODE_A_LAN}:8090/"
    assert REAL_PAGE_URL(run=UnitState("is-enabled"), fabric_path=path) == page
    assert REAL_PAGE_URL(run=UnitState("is-active"), fabric_path=path) == page     # running, though not enabled
    off = UnitState()
    assert REAL_PAGE_URL(run=off, fabric_path=path) is None
    assert off.calls == [["is-enabled", "--quiet", dashboard_admin.UNIT], ["is-active", "--quiet", dashboard_admin.UNIT]]
    assert REAL_PAGE_URL(run=UnitState(missing=True), fabric_path=path) is None    # no systemctl on this host
    (tmp_path / "dashboard.disabled").write_text("")
    assert REAL_PAGE_URL(run=UnitState("is-enabled", "is-active"), fabric_path=path) is None


def test_the_package_turns_the_dashboard_on_unless_it_was_turned_off():
    packaging = ROOT / "packaging" / "debian"
    postinst = (packaging / "postinst").read_text(encoding="utf-8")
    assert "if [ ! -e /etc/sparkring/dashboard.disabled ]; then" in postinst
    assert "systemctl enable sparkring-dashboard.service" in postinst
    assert "systemctl restart sparkring-dashboard.service ||" in postinst      # a failed start never fails the package
    unit = (packaging / "sparkring-dashboard.service").read_text(encoding="utf-8")
    assert "ConditionPathExists=!/etc/sparkring/dashboard.disabled" in unit
    assert "ConditionPathExists=/var/lib/sparkring/controller/fabric.json" in unit
    assert str(dashboard_admin.DISABLED_PATH).replace("\\", "/").endswith("/etc/sparkring/dashboard.disabled")


def test_the_dashboard_command_routes_its_root_actions(monkeypatch):
    seen = []
    monkeypatch.setattr(dashboard_admin, "switch", lambda command: seen.append(command) or 0)
    monkeypatch.setattr(dashboard_admin, "export", lambda output: seen.append(("export", output)) or 0)
    assert dashboard.main(["disable"]) == 0 and dashboard.main(["enable"]) == 0
    assert dashboard.main(["export", "--output", "site.json"]) == 0
    assert seen == ["disable", "enable", ("export", "site.json")]
