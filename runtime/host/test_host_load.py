"""Host-load settings: placement drop-ins from SIRCL's core classifier, the host agent's interval while a
model serves (and the staleness its readers apply), and the recovery timer's SSH through master
connections. Fake systemctl, ssh and sysfs; no host is contacted."""
import json
from pathlib import Path
import subprocess

import pytest

from runtime.host import discovery, host_load, node, recovery
from scripts import sparkring_node

ROOT = Path(__file__).resolve().parents[2]
PACKAGING = ROOT / "packaging" / "debian"
EFFICIENCY = {*range(0, 5), *range(10, 15)}
MODEL = {"name": "sr-ring8-glm53-nvfp4-tp8-0a1b2c-r3", "state": "running"}


class Run:
    def __init__(self):
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")


def settings_file(root, **values):
    path = root / "etc" / "sparkring" / "host-load.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": host_load.SCHEMA, **values}), encoding="utf-8")


# -- settings ---------------------------------------------------------------------------------------
def test_without_a_settings_file_every_job_keeps_its_present_behaviour(tmp_path):
    assert host_load.load(root=tmp_path) == {"placement": "default", "agent_serving_interval": 30,
                                             "recovery_ssh_reuse": False, "control_refresh_interval": 20}
    assert host_load.agent_interval({"containers": [MODEL]}, root=tmp_path) == node.AGENT_INTERVAL == 30
    assert host_load.recovery_invoke(root=tmp_path, ssh=discovery.ssh) is discovery.ssh


def test_unusable_values_fall_back_to_their_defaults(tmp_path):
    settings_file(tmp_path, placement="fastest", agent_serving_interval=5, recovery_ssh_reuse="yes")
    assert host_load.load(root=tmp_path) == host_load.DEFAULTS
    (tmp_path / "etc" / "sparkring" / "host-load.json").write_text("not json")
    assert host_load.load(root=tmp_path) == host_load.DEFAULTS


@pytest.mark.parametrize("item", ["placement=fast", "agent_serving_interval=10", "agent_serving_interval=601",
                                  "recovery_ssh_reuse=maybe", "nice=19", "placement"])
def test_assignments_are_refused_before_anything_changes(item):
    with pytest.raises(ValueError):
        host_load.parse_assignments([item])


def test_assignments():
    assert host_load.parse_assignments(["placement=efficiency", "agent_serving_interval=120",
                                        "recovery_ssh_reuse=on"]) == {
        "placement": "efficiency", "agent_serving_interval": 120, "recovery_ssh_reuse": True}


# -- placement --------------------------------------------------------------------------------------
def test_efficiency_placement_writes_the_dashboards_priority_into_each_units_drop_in(tmp_path):
    run = Run()
    result = host_load.apply_placement({**host_load.DEFAULTS, "placement": "efficiency"}, root=tmp_path, run=run,
                                       cpus=EFFICIENCY)
    assert result == {"placement": "efficiency", "cpus": sorted(EFFICIENCY), "restarted": list(host_load.RESTARTED)}
    for unit in host_load.UNITS:
        text = host_load.drop_in_path(unit, root=tmp_path).read_text(encoding="utf-8")
        assert "[Service]\nCPUAffinity=0-4,10-14\nNice=19\nCPUSchedulingPolicy=idle\nIOSchedulingClass=idle\n" in text
    assert run.calls == [["systemctl", "daemon-reload"],
                         ["systemctl", "try-restart", "sparkring-agent.service", "sparkring-ssh-mux.service"]]
    host_load.apply_placement(host_load.DEFAULTS, root=tmp_path, run=run)
    assert not any(host_load.drop_in_path(unit, root=tmp_path).exists() for unit in host_load.UNITS)


def test_set_applies_a_placement_change_only(tmp_path, monkeypatch):
    monkeypatch.setattr(host_load, "efficiency_cpus", lambda: EFFICIENCY)
    run = Run()
    result = host_load.set_local(["placement=efficiency", "agent_serving_interval=120"], root=tmp_path, run=run)
    assert result["settings"] == {"placement": "efficiency", "agent_serving_interval": 120, "recovery_ssh_reuse": False,
                                  "control_refresh_interval": 20}
    assert result["placement"]["cpus"] == sorted(EFFICIENCY) and len(run.calls) == 2
    assert json.loads((tmp_path / "etc/sparkring/host-load.json").read_text())["schema"] == host_load.SCHEMA
    run.calls.clear()
    assert "placement" not in host_load.set_local(["agent_serving_interval=60"], root=tmp_path, run=run)
    assert run.calls == []


def test_the_core_set_comes_from_sircls_classifier(tmp_path):
    """The GB10 layout of SIRCL's CPU tests: one X925 core reports a higher capacity than its siblings."""
    root = tmp_path / "cpu"
    root.mkdir()
    (root / "online").write_text("0-19\n")
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("")
    for cpu in range(20):
        (root / f"cpu{cpu}").mkdir()
        capacity = 580 if cpu in EFFICIENCY else (1024 if cpu == 19 else 1017)
        (root / f"cpu{cpu}" / "cpu_capacity").write_text(f"{capacity}\n")
    assert host_load.efficiency_cpus(root, cpuinfo) == EFFICIENCY
    from runtime.host import dashboard
    assert dashboard.efficiency_cpus(root, cpuinfo) == EFFICIENCY


# -- the host agent's pace ------------------------------------------------------------------------------
def test_a_running_model_container_paces_the_agent_by_the_setting(tmp_path):
    settings_file(tmp_path, agent_serving_interval=120)
    assert host_load.serving({"containers": [MODEL]})
    assert not host_load.serving({"containers": [{**MODEL, "state": "exited"}]})
    assert not host_load.serving({"containers": [{"name": "netadm", "state": "running"}]})
    assert host_load.agent_interval({"containers": [MODEL]}, root=tmp_path) == 120
    assert host_load.agent_interval({"containers": []}, root=tmp_path) == 30


def test_the_agent_records_its_interval_and_a_lowered_setting_ends_the_wait(tmp_path, monkeypatch):
    settings_file(tmp_path, agent_serving_interval=120)
    monkeypatch.setattr(node, "snapshot", lambda **kwargs: {"observed_at": 1, "containers": [MODEL]})
    slept, recorded = [], []

    class Stop(Exception):
        pass

    def sleep(seconds):
        slept.append(seconds)
        recorded.append(node.read(tmp_path, "/run/sparkring/status.json").get("interval_seconds"))
        if len(slept) == 2:
            settings_file(tmp_path, agent_serving_interval=30)
        if len(slept) == 3:
            raise Stop
    with pytest.raises(Stop):
        node.agent(root=tmp_path, sleep=sleep)
    # The first document announces 120 s and the agent waits in 30 s steps; lowering the setting
    # ends the wait at 60 s, and the next document, at the default pace, carries no interval.
    assert slept == [30, 30, 30] and recorded == [120, 120, None]


def test_the_agent_document_is_unchanged_at_the_default_pace(tmp_path, monkeypatch):
    monkeypatch.setattr(node, "snapshot", lambda **kwargs: {"observed_at": 1, "containers": [MODEL]})
    node.agent(root=tmp_path, once=True)
    assert node.read(tmp_path, "/run/sparkring/status.json") == {"observed_at": 1, "containers": [MODEL]}


@pytest.mark.parametrize("interval, age, stale", [(None, 89, False), (None, 91, True), (120, 260, False),
                                                  (120, 280, True), (30, 91, True)])
def test_readers_judge_a_status_document_by_its_interval(tmp_path, interval, age, stale):
    document = {"observed_at": 1000.0, "state": "network-configured",
                **({"interval_seconds": interval} if interval else {})}
    node.save(tmp_path, "/run/sparkring/status.json", document)
    result = node.status(root=tmp_path, now=lambda: 1000.0 + age)
    assert (result["state"] == "stale") is stale


# -- the recovery timer's SSH ---------------------------------------------------------------------------
def test_recovery_ssh_goes_through_the_master_connections_when_the_setting_is_on(tmp_path):
    settings_file(tmp_path, recovery_ssh_reuse=True)
    seen = []
    invoke = host_load.recovery_invoke(root=tmp_path, ssh=lambda node, args, **kw: seen.append((node, args, kw)) or "{}")
    invoke("root@192.0.2.11", ["/usr/bin/sparkring", "node", "status"])
    assert seen == [("root@192.0.2.11", ["/usr/bin/sparkring", "node", "status"],
                     {"options": ["-o", "ControlMaster=no", "-o", "ControlPath=/run/sparkring/ssh-mux/%C"]})]


def test_ssh_places_extra_options_before_the_target():
    run = []

    def fake(argv, **kwargs):
        run.append(argv)
        return subprocess.CompletedProcess(argv, 0, "ok", "")
    assert discovery.ssh("root@192.0.2.11", ["true"], run=fake, options=host_load.mux_options()) == "ok"
    assert run[0] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ControlMaster=no",
                      "-o", "ControlPath=/run/sparkring/ssh-mux/%C", "root@192.0.2.11", "true"]


def test_the_recovery_check_takes_its_ssh_from_the_setting(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(host_load, "recovery_invoke", lambda: sentinel)
    monkeypatch.setattr(recovery.controller, "active_deployments", lambda: [(None, Path("deployment"))])
    seen = []
    monkeypatch.setattr(recovery, "check_one", lambda directory, **kwargs: seen.append(kwargs["invoke"]) or
                        {"state": "serving", "summary": "", "deployment": str(directory)})
    recovery.check()
    assert seen == [sentinel]


def test_masters_keep_one_connection_per_spark_and_back_off_after_an_exit():
    started = []

    class Process:
        def __init__(self, argv, **kwargs):
            self.argv, self.code = argv, None
            started.append(self)

        def poll(self):
            return self.code

        def terminate(self):
            self.code = -15

        def wait(self, timeout=None):
            return self.code
    now = [100.0]
    masters = host_load.Masters(popen=Process, clock=lambda: now[0], directory="/run/sparkring/ssh-mux")
    masters.keep(["root@192.0.2.11", "root@192.0.2.12"])
    assert [p.argv[-1] for p in started] == ["root@192.0.2.11", "root@192.0.2.12"]
    assert started[0].argv[:3] == ["ssh", "-M", "-N"] and "ServerAliveInterval=10" in started[0].argv
    started[0].code = 255                      # the Spark went away
    masters.keep(["root@192.0.2.11", "root@192.0.2.12"])
    assert len(started) == 2                   # backoff: not yet started again
    now[0] += host_load.MUX_BACKOFF_SECONDS[0]
    masters.keep(["root@192.0.2.11", "root@192.0.2.12"])
    assert len(started) == 3 and started[2].argv[-1] == "root@192.0.2.11"
    masters.keep([])                           # the setting was turned off
    assert all(p.code is not None for p in started)


def test_the_master_service_follows_the_setting(tmp_path, monkeypatch):
    kept = []

    class Masters:
        def keep(self, targets):
            kept.append(list(targets))

        def stop(self, target=None):
            kept.append("stopped")
    monkeypatch.setattr(host_load, "recorded_targets", lambda: ["root@192.0.2.11"])
    host_load.mux_main(root=tmp_path, sleep=lambda s: None, masters=Masters(), rounds=1)
    settings_file(tmp_path, recovery_ssh_reuse=True)
    host_load.mux_main(root=tmp_path, sleep=lambda s: None, masters=Masters(), rounds=1)
    assert kept == [[], "stopped", ["root@192.0.2.11"], "stopped"]


def test_turning_reuse_on_and_off_starts_and_stops_the_master_service_on_node_a(tmp_path):
    run = Run()
    host_load.set_local(["recovery_ssh_reuse=on"], root=tmp_path, run=run)
    assert run.calls == []                     # a worker: no controller record, no master service
    (tmp_path / "var/lib/sparkring/controller").mkdir(parents=True)
    (tmp_path / "var/lib/sparkring/controller/cluster.json").write_text("{}")
    host_load.set_local(["recovery_ssh_reuse=off"], root=tmp_path, run=run)
    host_load.set_local(["recovery_ssh_reuse=on"], root=tmp_path, run=run)
    assert run.calls == [["systemctl", "disable", "--now", host_load.MUX_UNIT],
                         ["systemctl", "enable", "--now", host_load.MUX_UNIT]]


def test_the_detection_bound_of_a_shared_connection():
    """A command on a master whose Spark went away waits at most the keepalive (10 s × 2) beyond a
    fresh login's 10 s connect timeout; the recovery timer's 60 s period is unchanged."""
    assert host_load.MUX_ALIVE_SECONDS * host_load.MUX_ALIVE_COUNT == 20
    unit = (PACKAGING / "sparkring-recover.timer").read_text(encoding="utf-8")
    assert "OnUnitInactiveSec=60" in unit


# -- commands and packaging ------------------------------------------------------------------------------
def test_node_commands_and_the_fabric_wide_command_are_registered(monkeypatch, capsys, tmp_path):
    source = (ROOT / "scripts" / "sparkring.py").read_text(encoding="utf-8")
    assert 'raw[0] == "host-load"' in source and "from runtime.host.host_load import main" in source
    monkeypatch.setattr(host_load, "show", lambda **kwargs: {"settings": host_load.DEFAULTS})
    assert sparkring_node.main(["host-load", "show"]) == 0
    assert json.loads(capsys.readouterr().out)["settings"] == host_load.DEFAULTS


def test_set_with_all_changes_every_recorded_spark_over_ssh(monkeypatch, capsys):
    monkeypatch.setattr(host_load, "fabric_targets", lambda: ["root@192.0.2.10", "operator@192.0.2.11"])
    calls = []
    monkeypatch.setattr(discovery, "ssh", lambda target, argv, **kw: calls.append((target, argv)) or '{"ok": true}')
    assert host_load.main(["set", "placement=efficiency", "--all"]) == 0
    assert calls == [("root@192.0.2.10", ["/usr/bin/sparkring", "node", "host-load", "set", "placement=efficiency"]),
                     ("operator@192.0.2.11", ["sudo", "-n", "/usr/bin/sparkring", "node", "host-load", "set",
                                              "placement=efficiency"])]
    assert host_load.main(["set", "placement=fast", "--all"]) == 2 and len(calls) == 2


def test_the_master_service_unit_and_package_records():
    unit = (PACKAGING / "sparkring-ssh-mux.service").read_text(encoding="utf-8")
    assert "ExecStart=/usr/bin/sparkring node ssh-mux" in unit and "Environment=HOME=/root" in unit
    assert "ConditionPathExists=/var/lib/sparkring/controller/cluster.json" in unit
    units = next(line for line in (PACKAGING / "prerm").read_text().splitlines() if line.startswith("UNITS="))
    assert "sparkring-ssh-mux.service" in units.split("=", 1)[1].strip('"').split()
    postinst = (PACKAGING / "postinst").read_text(encoding="utf-8")
    assert "sparkring-ssh-mux" not in postinst            # off until recovery_ssh_reuse=on starts it
    assert "/usr/bin/sparkring node host-load apply" in postinst


# -- Node A's own rank, without SSH -----------------------------------------------------------------
REAL_LOCAL_STATUS = recovery.local_status
REAL_LOCAL_COMMAND = recovery.local_command


@pytest.fixture
def ring_of_eight(tmp_path, monkeypatch):
    """A recorded cluster of eight Sparks (rank 0 is Node A) and the lock of a TP8 deployment on it."""
    hosts = [f"root@192.0.2.{10 + rank}" for rank in range(8)]
    state = tmp_path / "controller"
    state.mkdir()
    (state / "cluster.json").write_text(json.dumps({"plan": {"spec": {"hosts": [{"host": h} for h in hosts]}}}))
    monkeypatch.setattr(recovery.controller, "STATE", state)
    monkeypatch.setattr(recovery, "local_status", REAL_LOCAL_STATUS)
    monkeypatch.setattr(recovery, "local_command", REAL_LOCAL_COMMAND)
    lock = {"id": "f" * 64, "backend": "compose",
            "site": {"name": "ring8-glm53-0a1b2c", "ranks": [{"rank": r, "host": h} for r, h in enumerate(hosts)]}}
    return hosts, lock


def test_the_recovery_timer_reads_node_a_without_signing_in_to_it(ring_of_eight, monkeypatch):
    hosts, lock = ring_of_eight
    assert recovery.local_target() == hosts[0]
    monkeypatch.setattr(node, "status", lambda: {"state": "network-configured", "boot_id": "local"})
    local_runs = []
    monkeypatch.setattr(recovery.subprocess, "run", lambda argv, **kw: local_runs.append(argv) or
                        subprocess.CompletedProcess(argv, 0, json.dumps({"present": True, "running": True}), ""))
    logins = []

    def ssh(host, args, **kwargs):
        logins.append(host)
        if args[-1] == "status":
            return json.dumps({"state": "network-configured", "boot_id": host})
        return json.dumps({"present": True, "running": True})
    documents = recovery.node_documents(hosts, invoke=ssh)
    assert documents[hosts[0]] == {"state": "network-configured", "boot_id": "local"}
    rows = recovery.probe_containers(lock, invoke=ssh)
    assert all(row["running"] for row in rows)
    # One run of a TP8 check: 7 status reads and 7 container probes over SSH, none to Node A.
    assert hosts[0] not in logins and len(logins) == 14 and sorted(set(logins)) == sorted(hosts[1:])
    [argv] = local_runs
    assert argv[:4] == ["python3", "-I", "-B", "-c"] and argv[-3] == "sr-ring8-glm53-0a1b2c-r0"


def test_a_failing_local_probe_is_reported_like_an_unreachable_rank(ring_of_eight, monkeypatch):
    hosts, lock = ring_of_eight
    monkeypatch.setattr(recovery.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(
        argv, 1, "", "Cannot connect to the Docker daemon"))
    rows = recovery.probe_containers(lock, invoke=lambda host, args, **kw: json.dumps({"running": True}))
    assert rows[0]["running"] is None and "Docker daemon" in rows[0]["error"]
    assert all(row["running"] for row in rows[1:])


def test_without_a_cluster_record_every_rank_is_read_over_ssh(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery.controller, "STATE", tmp_path / "none")
    monkeypatch.setattr(recovery, "local_status", REAL_LOCAL_STATUS)
    assert recovery.local_target() is None
    seen = []
    recovery.node_documents(["root@192.0.2.10"], invoke=lambda host, args, **kw: seen.append(host) or "{}")
    assert seen == ["root@192.0.2.10"]


# -- the administration tunnel's checks ------------------------------------------------------------
def test_a_slower_check_interval_is_a_timer_drop_in_and_twenty_seconds_removes_it(tmp_path):
    run = Run()
    result = host_load.set_local(["control_refresh_interval=300"], root=tmp_path, run=run)
    assert result["control"] == {"control_refresh_interval": 300, "drop_in": True}
    text = host_load.control_drop_in_path(root=tmp_path).read_text(encoding="utf-8")
    assert "[Timer]\nOnUnitInactiveSec=\nOnUnitInactiveSec=300\n" in text      # the reset line drops the unit's 20 s
    assert run.calls == [["systemctl", "daemon-reload"], ["systemctl", "try-restart", host_load.CONTROL_TIMER]]
    host_load.set_local(["control_refresh_interval=20"], root=tmp_path, run=run)
    assert not host_load.control_drop_in_path(root=tmp_path).exists()
    with pytest.raises(ValueError):
        host_load.parse_assignments(["control_refresh_interval=10"])


def follow_ups(host):
    return [call for call in host.calls if call[:1] == ["systemd-run"]]


def test_a_slow_check_follows_up_in_twenty_seconds_while_a_peer_moves(tmp_path):
    from runtime.host.test_control import T, TunnelHost, pair_plan, unplug
    (head, _), _ = pair_plan()
    host = TunnelHost(tmp_path, head, now=T)
    assert host.up(T) == {"control_up": True} and follow_ups(host) == []       # 20 s period: no follow-ups
    settings_file(tmp_path, control_refresh_interval=300)
    host.up(T + 20)
    assert follow_ups(host) == []                                              # steady on the primary path
    unplug(host, 1)
    host.up(T + 40)                                                            # the cable is pulled: a move
    [call] = follow_ups(host)
    assert call == ["systemd-run", "--quiet", "--collect", "--unit=sparkring-control-follow-up", "--on-active=20",
                    "--timer-property=AccuracySec=1s", "/usr/bin/systemctl", "start", "--no-block",
                    "sparkring-control-refresh.service"]
    host.handshake = T + 50                                                    # the fallback answered
    host.up(T + 60)
    assert follow_ups(host)                                                    # on a fallback: keep checking
    unplug(host, 1, up=True)
    host.up(T + 80)
    host.handshake = T + 85
    host.up(T + 100)                                                           # back on the primary path
    host.handshake = T + 115
    host.up(T + 120)
    assert follow_ups(host) == []


def test_link_changes_and_driver_restarts_start_a_check():
    dispatcher = (PACKAGING / "sparkring-control-link").read_text(encoding="utf-8")
    assert "up|down|pre-down) ;;" in dispatcher and "exec systemctl start --no-block sparkring-control-refresh.service" in dispatcher
    rules = (PACKAGING / "90-sparkring-control.rules").read_text(encoding="utf-8")
    assert ('ACTION=="add", SUBSYSTEM=="net", DRIVERS=="mlx5_core", TAG+="systemd", '
            'ENV{SYSTEMD_WANTS}+="sparkring-control-refresh.service"') in rules
    service = (PACKAGING / "sparkring-control-refresh.service").read_text(encoding="utf-8")
    assert "after interface reload" not in service and "link changes and ConnectX driver restarts" in service
    build = (ROOT / "scripts" / "build_deb.py").read_text(encoding="utf-8")
    assert 'dispatchers / "90-sparkring-control-link"' in build and 'udev_rules / "90-sparkring-control.rules"' in build


@pytest.mark.skipif(not __import__("shutil").which("sh"), reason="needs a POSIX shell")
def test_the_dispatcher_starts_a_check_for_recorded_administration_links_only(tmp_path):
    script = (PACKAGING / "sparkring-control-link").read_text(encoding="utf-8")
    config = tmp_path / "control.json"
    config.write_text(json.dumps({"peers": [{"netdev": "enp1s0f1np1"}]}, indent=2))
    started = tmp_path / "started"
    script = script.replace("/etc/sparkring/control.json", config.as_posix()).replace(
        "exec systemctl start --no-block sparkring-control-refresh.service", f"echo started >> {started.as_posix()}")
    path = tmp_path / "dispatcher"
    path.write_text(script, encoding="utf-8", newline="\n")
    for interface, event in (("enp1s0f1np1", "up"), ("enp1s0f1np1", "dhcp4-change"), ("docker0", "up"),
                             ("enp1s0f1np1", "down")):
        subprocess.run(["sh", path.as_posix(), interface, event], check=True, timeout=10)
    assert started.read_text().splitlines() == ["started", "started"]


def test_the_check_runs_at_the_dashboards_priority_with_efficiency_placement():
    assert "sparkring-control-refresh.service" in host_load.UNITS
    assert "sparkring-control-refresh.service" not in host_load.RESTARTED       # a oneshot: its next run uses it
