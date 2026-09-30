"""Automatic model recovery on a simulated pair and ring: guards, actions, backoff and status; no host access."""
import json
import urllib.error

import pytest

from runtime.common import installer, process_lock
from runtime.host import controller, node, recovery
from runtime.host.test_fabric_ssh import cluster
from scripts import installer_runner, sparkring

PROFILE = "qwen38-flash-next-tp2"
BOOT = "11111111-1111-4111-8111-111111111111"
REBOOTED = "22222222-2222-4222-8222-222222222222"
SERVING = {"ok": True, "status": 200, "error": None}
DEAD = {"ok": False, "status": 503, "error": "HTTP 503 from /health: the model engine has stopped"}


class Sparks:
    """Rank operations, Docker state and SSH of simulated Sparks.

    ``running`` holds each rank's container state; the rank operations of
    the plain installer runner change it as the real ones would. SSH answers
    ``sparkring node status`` and the recovery container probe; any other
    command fails the test.
    """

    def __init__(self, hosts):
        self.hosts = hosts
        self.running = {rank: False for rank in range(len(hosts))}
        self.exit_codes = {rank: 0 for rank in range(len(hosts))}
        self.boots = {host: BOOT for host in hosts}
        self.unreachable = set()
        self.fail = set()
        self.operations = []
        self.mesh = {host: {"active": ["sparkring-test-mesh.service"], "failed": [], "markers": []} for host in hosts}

    def rank_operation(self, runner, target, argv, timeout):
        operation, rank = argv[1], int(argv[2])
        self.operations.append((operation, rank))
        if operation in self.fail:
            return {"returncode": 1, "stdout": "", "stderr": f"{operation} failed on rank {rank}", "uncertain": False}
        if operation == "status":
            if target in self.unreachable:
                return {"returncode": 1, "stdout": "", "uncertain": False,
                        "stderr": f"ssh: connect to host {target.split('@')[1]} port 2222: Connection timed out"}
            return {"returncode": 0, "stderr": "", "uncertain": False, "stdout": json.dumps(self.observation(rank))}
        if operation == "start":
            self.running[rank] = True
        elif operation == "stop":
            self.running[rank] = False
        elif operation == "running" and not self.running[rank]:
            return {"returncode": 1, "stdout": "", "stderr": "Rank is not running", "uncertain": False}
        elif operation == "stopped" and self.running[rank]:
            return {"returncode": 1, "stdout": "", "stderr": "Container is still running", "uncertain": False}
        return {"returncode": 0, "stdout": "ok", "stderr": "", "uncertain": False}

    def observation(self, rank):
        """The ``installer status`` observation of a rank's container."""
        running = self.running[rank]
        value = {"schema": "sparkring-model-observation/v1", "rank": rank, "present": True, "running": running,
                 "health": "healthy" if running and rank == 0 else None, "container_name": f"sr-test-r{rank}"}
        if not running:
            value.update(exit_code=self.exit_codes[rank], finished_at="2026-09-30T17:02:11.5Z")
        return value

    def ssh(self, host, argv, **kwargs):
        if host in self.unreachable:
            raise RuntimeError(f"ssh: connect to host {host.split('@')[1]} port 2222: Connection timed out")
        rank = self.hosts.index(host)
        if argv in (["/usr/bin/sparkring", "node", "status"],
                    ["sudo", "-n", "/usr/bin/sparkring", "node", "status", "--refresh"]):
            document = {"state": "network-configured", "hostname": f"spark{rank}", "boot_id": self.boots[host]}
            if len(self.hosts) == 4:
                document["mesh"] = self.mesh[host]
            return json.dumps(document)
        if recovery.CONTAINER in argv:
            return json.dumps({**self.observation(rank), "name": argv[-3], "owned": True})
        pytest.fail(f"unexpected SSH command on {host}: {argv}")


@pytest.fixture
def pair(tmp_path, monkeypatch):
    """A running Qwen TP2 deployment on two simulated Sparks, started by sparkring up."""
    from runtime.common import distribution
    from runtime.host import discovery
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(controller, "STATE", tmp_path / "state")
    value = cluster(2)
    node.save(controller.STATE, "cluster.json", value)
    monkeypatch.setattr(distribution, "identity", lambda _: "a" * 40)
    monkeypatch.setattr(distribution, "bundle", lambda root, dest: dest.write_bytes(b"retained source"))
    sparks = Sparks([host["host"] for host in value["plan"]["spec"]["hosts"]])
    monkeypatch.setattr(discovery, "ssh", sparks.ssh)
    monkeypatch.setattr(installer_runner.Runner, "_call", lambda runner, target, argv, timeout:
                        sparks.rank_operation(runner, target, argv, timeout))
    assert controller.lifecycle(["up", PROFILE, "--execute"]) == 0
    sparks.directory = (controller.STATE / "deployments" / PROFILE).resolve()
    assert all(sparks.running.values())
    recovery.started(sparks.directory)
    sparks.operations.clear()
    return sparks


class Clock:
    def __init__(self, value=1_000_000.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def run(clock, api=SERVING, **options):
    return recovery.check(now=clock, api=lambda url: api, tunnel=options.pop("tunnel", {}), **options)


def record(sparks):
    return recovery.record_of(recovery.load(), sparks.directory)


def state(sparks):
    return installer.read(sparks.directory / "state.json")


def test_up_records_the_generation_and_every_boot(pair):
    value = record(pair)
    assert value["enabled"] is True and value["failures"] == 0
    assert value["generation"] == state(pair)["generation"] == 1
    assert value["boots"] == {host: BOOT for host in pair.hosts}
    assert recovery.path() == controller.STATE.parent / "recovery.json"


def test_serving_model_is_left_alone(pair):
    clock = Clock()
    assert run(clock)["state"] == "serving"
    assert pair.operations == [] and state(pair)["generation"] == 1


def test_model_stopped_everywhere_starts_through_up_with_the_pairs_gid_repair(pair):
    """Failure 2: every container stopped; one confirmed finding runs the up code path."""
    clock = Clock()
    pair.running.update({0: False, 1: False})
    pair.exit_codes.update({0: 0, 1: 255})
    first = run(clock)
    assert first["state"] == "confirming" and first["finding"]["state"] == "stopped"
    assert first["finding"]["next_action"] == recovery.UP and pair.operations == []
    clock.advance(60)
    assert run(clock)["state"] == "recovered"
    operations = [operation for operation, _ in pair.operations]
    # up on a completed up probes each rank, finds none running and runs a new generation.
    assert "stop" not in operations and operations.count("running") >= 2
    assert sorted(rank for operation, rank in pair.operations if operation == "gid-serve") == [0, 1]
    assert all(pair.running.values())
    value = record(pair)
    assert (value["generation"], value["failures"], value["last"]["result"]) == (state(pair)["generation"], 0, "succeeded")
    assert state(pair) == {**state(pair), "operation": "up", "complete": True, "generation": 2}
    log = (controller.STATE.parent / "logs" / "install.log").read_text()
    assert "Automatic recovery of qwen38-flash-next-tp2: The model is not running on any Spark." in log


def test_partial_model_stops_everywhere_then_starts(pair):
    """Failure 3: the worker's container died while rank 0's still runs."""
    clock = Clock()
    pair.running[1] = False
    pair.exit_codes[1] = 255
    finding = run(clock)["finding"]
    assert finding["state"] == "partial" and finding["next_action"] == recovery.DOWN_UP
    assert "rank 1 (spark1): stopped, exit code 255 at 2026-09-30 17:02:11 UTC" in finding["details"]
    clock.advance(60)
    assert run(clock)["state"] == "recovered"
    operations = [operation for operation, _ in pair.operations]
    assert operations.index("stop") < operations.index("gid-serve")
    assert sorted(rank for operation, rank in pair.operations if operation == "gid-serve") == [0, 1]
    assert state(pair) == {**state(pair), "operation": "up", "complete": True, "generation": 3}


def test_running_containers_whose_api_fails_are_restarted(pair):
    clock = Clock()
    assert run(clock, api=DEAD)["finding"]["details"] == ["API: " + DEAD["error"]]
    clock.advance(60)
    assert run(clock, api=DEAD)["state"] == "recovered"
    assert ("stop", 0) in pair.operations and ("gid-serve", 1) in pair.operations


def test_a_spark_restarted_since_the_generation_started_is_a_restart(pair):
    clock = Clock()
    pair.boots[pair.hosts[1]] = REBOOTED
    finding = run(clock)["finding"]
    assert finding["state"] == "rebooted" and finding["details"] == ["rank 1 (spark1) restarted since the model started"]
    clock.advance(60)
    assert run(clock)["state"] == "recovered"
    # The new generation's boots are the baseline for the next check.
    assert record(pair)["boots"][pair.hosts[1]] == REBOOTED
    clock.advance(60)
    assert run(clock)["state"] == "serving"


def test_an_unreachable_spark_is_only_waited_for(pair):
    clock = Clock()
    pair.running[1] = False
    pair.unreachable.add(pair.hosts[1])
    result = run(clock)
    assert result["state"] == "waiting" and pair.operations == []
    assert record(pair)["waiting"]["hosts"] == ["rank 1 (root@192.0.2.11)"]
    clock.advance(60)
    assert run(clock)["state"] == "waiting" and pair.operations == [] and record(pair)["pending"] is None
    pair.unreachable.clear()
    clock.advance(60)
    assert run(clock)["state"] == "confirming" and record(pair)["waiting"] is None


def test_an_unreachable_worker_behind_a_down_admin_link_names_the_link(pair):
    """Failure 4: the admin tunnel's link lost carrier while the model still serves."""
    clock = Clock()
    pair.unreachable.add(pair.hosts[1])
    tunnel = {"interface_up": True, "peers": [{
        "id": "b", "address": "192.0.2.11", "allowed_ips": ["192.0.2.11/32"], "upstream": False,
        "netdev": "enp1s0f1np1", "carrier": False, "operstate": "down", "endpoint": "[fe80::2%enp1s0f1np1]:51871",
        "handshake_age_s": 2760}]}
    finding = run(clock, tunnel=tunnel)["finding"]
    assert finding["summary"] == "SparkRing cannot reach rank 1 (root@192.0.2.11); the model's API still answers"
    assert finding["details"] == ["rank 1 (root@192.0.2.11) does not answer: the admin tunnel has no recent handshake "
                                  "(last 46 min ago); Node A's enp1s0f1np1: no link"]
    assert finding["next_action"] == "reconnect the cable of Node A's enp1s0f1np1; it carries the admin tunnel"
    assert pair.operations == []


def test_a_completed_down_is_never_undone(pair):
    clock = Clock()
    assert controller.lifecycle(["down", "--execute"]) == 0
    pair.operations.clear()
    for _ in range(3):
        assert run(clock)["state"] == "inactive"
        clock.advance(60)
    assert pair.operations == []


def test_recovery_skips_while_another_operation_holds_the_install_lock(pair):
    clock = Clock()
    pair.running.update({0: False, 1: False})
    with process_lock.hold(controller.STATE / "install.lock"):
        assert run(clock)["state"] == "busy"
        clock.advance(60)
        assert run(clock)["state"] == "busy"
    assert pair.operations == [] and record(pair).get("pending") is None


def test_off_does_nothing_and_on_resets_failures(pair, capsys):
    clock = Clock()
    pair.running.update({0: False, 1: False})
    assert sparkring.main(["recover", "off"]) == 0
    assert "Automatic recovery: off" in capsys.readouterr().out
    for _ in range(3):
        assert run(clock)["state"] == "off"
        clock.advance(60)
    assert pair.operations == []
    recovery.update(pair.directory, failures=3, stopped=True)
    assert sparkring.main(["recover", "on"]) == 0
    assert (record(pair)["enabled"], record(pair)["failures"], record(pair)["stopped"]) == (True, 0, False)


def test_failed_attempts_back_off_and_stop_after_the_limit(pair):
    clock = Clock()
    pair.running.update({0: False, 1: False})
    pair.fail.add("start")
    run(clock)
    clock.advance(60)
    assert run(clock)["state"] == "failed"
    value = record(pair)
    assert value["failures"] == 1 and value["next_attempt_at"] == clock.value + recovery.BACKOFF[0]
    assert value["last"]["result"] == "failed" and "start-workers" in value["last"]["error"]
    # The state the failed attempt left lets the retry act; it is always down, then up.
    assert value["left"] == {**value["left"], "operation": "up", "complete": False}
    pair.operations.clear()
    clock.advance(60)
    assert run(clock)["state"] == "backoff" and pair.operations == []
    clock.advance(recovery.BACKOFF[0])
    assert run(clock)["state"] == "failed" and ("stop", 1) in pair.operations
    assert record(pair)["next_attempt_at"] == clock.value + recovery.BACKOFF[1]
    clock.advance(60)
    assert run(clock)["state"] == "backoff"
    clock.advance(recovery.BACKOFF[1])
    assert run(clock)["state"] == "failed"
    assert record(pair)["stopped"] is True and record(pair)["failures"] == recovery.FAILURE_LIMIT
    pair.operations.clear()
    clock.advance(3600)
    assert run(clock)["state"] == "stopped" and pair.operations == []
    # A manual up resets it.
    pair.fail.clear()
    assert controller.lifecycle(["down", "--execute"]) == 0
    assert controller.lifecycle(["up", "--execute"]) == 0
    assert (record(pair)["failures"], record(pair)["stopped"]) == (0, False)


def test_an_interrupted_attempt_counts_as_failed_and_its_state_is_retried(pair):
    clock = Clock()
    pair.running.update({0: False, 1: False})

    def interrupted(operation):
        raise KeyboardInterrupt

    run(clock)
    clock.advance(60)
    with pytest.raises(KeyboardInterrupt):
        run(clock, apply=interrupted)
    assert record(pair)["attempt"]["action"] == "start"
    clock.advance(60)
    # The next run records the interruption as a failed attempt and backs off.
    assert run(clock)["state"] == "backoff"
    value = record(pair)
    assert (value["attempt"], value["failures"], value["last"]["result"]) == (None, 1, "interrupted")
    clock.advance(recovery.BACKOFF[0])
    assert run(clock)["state"] == "recovered"
    assert (record(pair)["failures"], record(pair)["last"]["result"]) == (0, "succeeded")


def four_spark_ring(tmp_path, monkeypatch):
    from runtime.host import discovery
    monkeypatch.setattr(controller, "STATE", tmp_path / "state")
    value = cluster(4)
    node.save(controller.STATE, "cluster.json", value)
    directory = tmp_path / "state" / "deployments" / "ring"
    lock = {"id": "d" * 64, "backend": "compose", "selection": {"profile": "qwen38-flash-next-qad-tp4"},
            "site": {"name": "ring", "ranks": [{"rank": rank, "host": host["host"]}
                                               for rank, host in enumerate(value["plan"]["spec"]["hosts"])]}}
    lock["selection"].update(target_variant=None, model_repository="local-inference-lab/Qwen3.8-Flash-Next-NVFP4",
                             model_revision="0" * 40, release="dev-image")
    installer.write(directory / "deployment.lock.json", lock)
    installer.write(directory / "state.json", {"generation": 1, "operation": "up", "complete": True})
    node.save(controller.STATE, "active.json", {"path": str(directory)})
    sparks = Sparks([row["host"] for row in lock["site"]["ranks"]])
    sparks.directory = directory.resolve()
    monkeypatch.setattr(discovery, "ssh", sparks.ssh)
    recovery.update(directory, api_url="http://192.0.2.10:8000/v1")
    return sparks


def test_ring_mesh_processes_left_without_a_supervisor_are_left_to_the_operator(tmp_path, monkeypatch):
    sparks = four_spark_ring(tmp_path, monkeypatch)
    sparks.running.update({2: False})
    sparks.mesh[sparks.hosts[2]] = {"active": [], "markers": [4242], "failed": [
        {"unit": "sparkring-test-mesh.service", "result": "watchdog", "error": None}]}
    clock = Clock()
    applied = []
    for _ in range(3):
        result = run(clock, apply=applied.append)
        clock.advance(60)
    assert result["state"] == "report" and applied == []
    assert result["finding"]["state"] == "mesh-cleanup"
    assert "rank 2 (spark2): sparkring-test-mesh.service failed: systemd result watchdog" in result["finding"]["details"]


def test_ring_with_a_failed_mesh_unit_starts_through_up(tmp_path, monkeypatch):
    """Failure 1: every Spark restarted and the mesh unit failed at boot; no model runs."""
    sparks = four_spark_ring(tmp_path, monkeypatch)
    for host in sparks.hosts:
        sparks.mesh[host] = {"active": [], "markers": [], "failed": [
            {"unit": "sparkring-test-mesh.service", "result": "exit-code",
             "error": "ValueError: Management address does not identify this rank"}]}
    monkeypatch.setattr(controller, "_hairpin_problem", lambda: None)
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path / "logs"))
    clock = Clock()
    applied = []
    finding = run(clock, apply=applied.append)["finding"]
    assert finding["state"] == "stopped" and finding["next_action"] == recovery.UP
    assert ("rank 0 (spark0): sparkring-test-mesh.service failed: ValueError: Management address does not identify "
            "this rank") in finding["details"]
    clock.advance(60)
    run(clock, apply=applied.append)
    assert applied == ["up"]


def test_ring_waits_for_the_hairpin_setting(tmp_path, monkeypatch):
    sparks = four_spark_ring(tmp_path, monkeypatch)
    monkeypatch.setattr(controller, "_hairpin_problem", lambda: "rank 2 (spark2): the ConnectX hairpin setting is not in effect")
    clock = Clock()
    applied = []
    run(clock, apply=applied.append)
    clock.advance(60)
    result = run(clock, apply=applied.append)
    assert applied == [] and result["finding"]["next_action"] == "on Node A: sudo sparkring hairpin"
    assert sparks.operations == []


def test_ring_member_without_a_mesh_report_is_unknown(tmp_path, monkeypatch):
    sparks = four_spark_ring(tmp_path, monkeypatch)
    sparks.running.update({1: False})
    sparks.mesh[sparks.hosts[3]] = None
    clock = Clock()
    applied = []
    for _ in range(2):
        result = run(clock, apply=applied.append)
        clock.advance(60)
    assert result["finding"]["state"] == "unknown" and applied == []


def test_api_health_reads_vllm_health_without_a_proxy():
    class Answer:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    urls = []

    def opener(url, timeout):
        urls.append((url, timeout))
        return Answer()
    assert recovery.api_health("http://192.0.2.10:8000/v1", opener=opener) == SERVING
    assert urls == [("http://192.0.2.10:8000/health", recovery.API_TIMEOUT)]

    def dead(url, timeout):
        raise urllib.error.HTTPError(url, 503, "Service Unavailable", {}, None)
    assert recovery.api_health("http://192.0.2.10:8000/v1", opener=dead) == DEAD

    def refused(url, timeout):
        raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))
    answer = recovery.api_health("http://192.0.2.10:8000/v1", opener=refused)
    assert answer["ok"] is False and "Connection refused" in answer["error"]


def test_status_lines_name_the_choice_the_wait_and_the_attempts():
    base = {"enabled": True, "failures": 0, "stopped": False}
    assert recovery.status_lines({**base, "enabled": False}) == [
        "Automatic recovery: off | turn on: sudo sparkring recover on"]
    lines = recovery.status_lines({**base, "failures": 1, "waiting": {"hosts": ["rank 1 (spark1)"], "since": 999_999_000},
                                   "last": {"action": "restart", "started_at": 999_999_000, "result": "failed",
                                            "error": "ready: phase failed"},
                                   "next_attempt_at": 2_000_000_000}, now=lambda: 1_000_000_000)
    assert lines[0] == "Automatic recovery: on"
    assert lines[1].startswith("  waiting for rank 1 (spark1) since ")
    assert lines[2].startswith("  last attempt: restart at ") and lines[2].endswith(", failed: ready: phase failed")
    assert lines[3].startswith("  next attempt: not before ")
    stopped = recovery.status_lines({**base, "failures": 3, "stopped": True})
    assert stopped == ["Automatic recovery: stopped after 3 failed attempts | sudo sparkring up --execute turns it "
                       "back on"]


def status(monkeypatch, capsys, *, snapshot=None, api=SERVING):
    """``sparkring status --refresh`` text lines; Node A's own document is ``snapshot``."""
    monkeypatch.setattr(recovery, "api_health", lambda url: api)
    monkeypatch.setattr(recovery, "timer_enabled", lambda: True)
    monkeypatch.setattr(controller.node, "snapshot",
                        lambda: snapshot or {"state": "network-configured", "next_action": "sparkring models"})
    capsys.readouterr()
    assert controller.lifecycle(["status", "--refresh"]) == 0
    return capsys.readouterr().out.splitlines()


def test_status_says_the_model_stopped_everywhere_and_how_to_start_it(pair, monkeypatch, capsys):
    """Failure 2: status said "up complete" and printed raw JSON observations."""
    pair.running.update({0: False, 1: False})
    pair.exit_codes.update({0: 0, 1: 255})
    lines = status(monkeypatch, capsys)
    assert lines[:3] == ["The model is not running on any Spark | next: sudo sparkring up --execute",
                         "  rank 0 (spark0): stopped, exit code 0 at 2026-09-30 17:02:11 UTC",
                         "  rank 1 (spark1): stopped, exit code 255 at 2026-09-30 17:02:11 UTC"]
    at = lines.index("Model containers:")
    assert lines[at + 1:at + 3] == ["  rank 0 root@192.0.2.10: stopped, exit code 0 at 2026-09-30 17:02:11 UTC",
                                    "  rank 1 root@192.0.2.11: stopped, exit code 255 at 2026-09-30 17:02:11 UTC"]
    assert not any(line.lstrip().startswith(("{", "[")) for line in lines)
    assert "Automatic recovery: on" in lines


def test_status_says_the_model_runs_on_one_spark_only(pair, monkeypatch, capsys):
    """Failure 3: rank 0's container stayed up and reported healthy while every request failed."""
    pair.running[1] = False
    pair.exit_codes[1] = 255
    lines = status(monkeypatch, capsys, api=DEAD)
    assert lines[0] == ("The model runs on rank 0 (spark0) but stopped on rank 1 (spark1) | next: "
                        "sudo sparkring down --execute, then sudo sparkring up --execute")
    assert "  API: HTTP 503 from /health: the model engine has stopped" in lines


def test_status_of_a_serving_model_keeps_node_a_first(pair, monkeypatch, capsys):
    lines = status(monkeypatch, capsys)
    assert lines[0] == "network-configured | next: sparkring models"
    assert "Model: The model runs on every Spark and its API answers" in lines
    assert controller.lifecycle(["status", "--refresh", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["model"]["state"] == "serving"


def test_status_names_the_down_admin_link_of_an_unreachable_worker(pair, monkeypatch, capsys):
    """Failure 4: the admin tunnel lost its link while the model kept serving over the other port."""
    pair.unreachable.add(pair.hosts[1])
    snapshot = {"state": "network-configured", "next_action": "sparkring models", "control": {
        "interface_up": True, "peers": [{"id": "b", "address": "192.0.2.11", "allowed_ips": ["192.0.2.11/32"],
                                         "upstream": False, "netdev": "enp1s0f1np1", "carrier": False,
                                         "operstate": "down", "endpoint": "[fe80::2%enp1s0f1np1]:51871",
                                         "handshake_age_s": 2760}]}}
    lines = status(monkeypatch, capsys, snapshot=snapshot)
    assert lines[0] == ("SparkRing cannot reach rank 1 (root@192.0.2.11); the model's API still answers | next: "
                        "reconnect the cable of Node A's enp1s0f1np1; it carries the admin tunnel")
    row = next(index for index, line in enumerate(lines) if line.startswith("  rank 1 (root@192.0.2.11): unreachable"))
    assert lines[row + 1] == "    admin tunnel: no recent handshake (last 46 min ago); Node A's enp1s0f1np1: no link"


def test_status_of_a_ring_whose_mesh_failed_at_boot(tmp_path, monkeypatch, capsys):
    """Failure 1: every rank said "Missing mesh network objects" and next "sparkring status --refresh"."""
    from runtime.host import retained_source
    sparks = four_spark_ring(tmp_path, monkeypatch)
    for host in sparks.hosts:
        sparks.mesh[host] = {"active": [], "markers": [], "failed": [
            {"unit": "sparkring-test-mesh.service", "result": "exit-code",
             "error": "ValueError: Management address does not identify this rank"}]}
    observations = [{"host": host, "result": sparks.observation(rank)} for rank, host in enumerate(sparks.hosts)]
    saved = {"profile": "qwen38-flash-next-qad-tp4", "state": {"generation": 1, "operation": "up", "complete": True},
             "api_url": "http://192.0.2.10:8000/v1", "observations": observations}
    monkeypatch.setattr(retained_source, "apply", lambda *a, **k: dict(saved))
    lines = status(monkeypatch, capsys)
    assert lines[0] == "The model is not running on any Spark | next: sudo sparkring up --execute"
    assert ("  rank 2 (spark2): sparkring-test-mesh.service failed: ValueError: Management address does not "
            "identify this rank") in lines
    assert ("    mesh: sparkring-test-mesh.service failed: ValueError: Management address does not identify this "
            "rank") in lines


def test_package_ships_the_timer_that_runs_one_check_after_another():
    from pathlib import Path
    from runtime.host.test_persistence import PACKAGING, unit_settings
    timer = unit_settings((PACKAGING / recovery.TIMER).read_text())
    assert timer[("Timer", "Unit")] == ["sparkring-recover.service"]
    # Measured from the end of the previous run, so a long restart is never checked again while it runs.
    assert timer[("Timer", "OnUnitInactiveSec")] == ["60"]
    assert timer[("Install", "WantedBy")] == ["timers.target"]
    service = unit_settings((PACKAGING / "sparkring-recover.service").read_text())
    assert service[("Service", "Type")] == ["oneshot"]
    assert service[("Service", "ExecStart")] == ["/usr/bin/sparkring", "recover", "--auto"]
    assert service[("Service", "TimeoutStartSec")] == ["infinity"]
    assert ("Service", "Restart") not in service and ("Install", "WantedBy") not in service
    # Removal records an enabled timer and reinstalling restores it, like the other enabled units.
    units = next(line for line in (PACKAGING / "prerm").read_text().splitlines() if line.startswith("UNITS="))
    assert recovery.TIMER in units.split("=", 1)[1].strip('"').split()
    assert "sparkring-recover" not in (PACKAGING / "postinst").read_text()
    assert recovery.TIMER_FILE == Path("/usr/lib/systemd/system/sparkring-recover.timer")
