"""Automatic model recovery on a simulated pair and ring: guards, actions, backoff and status; no host access."""
import contextlib
import json
import subprocess
import urllib.error

import pytest

from runtime.common import installer, process_lock
from runtime.host import controller, node, recovery
from runtime.host.test_fabric_ssh import cluster
from scripts import installer_runner, sparkring

PROFILE = "qwen38-flash-next-tp2"
BOOT = "11111111-1111-4111-8111-111111111111"
REBOOTED = "22222222-2222-4222-8222-222222222222"
SERVING = {"ok": True, "status": 200, "dead": False, "error": None}
DEAD = {"ok": False, "status": 503, "dead": True, "error": "HTTP 503 from /health: the model engine has stopped"}
SILENT = {"ok": False, "status": None, "dead": False,
          "error": "no answer from http://192.0.2.10:8000/health: timed out; /v1/models: no answer: timed out"}


class Sparks:
    """Rank operations, Docker state and SSH of simulated Sparks.

    ``running`` holds each rank's container state; the rank operations of
    the plain installer runner change it as the real ones would. SSH answers
    ``sparkring node status`` (``documents`` replaces a Spark's document) and
    the recovery container probe (it fails on the ranks in ``probe_fails``);
    any other command fails the test.
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
        self.documents = {}
        self.probe_fails = set()

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
            return json.dumps(self.documents.get(host, document))
        if recovery.CONTAINER in argv:
            if rank in self.probe_fails:
                raise RuntimeError(f"{host}: Cannot connect to the Docker daemon at unix:///var/run/docker.sock")
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
    """Every container stopped: the second check that finds it runs the up code path."""
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
    """The worker's container stopped while rank 0's still runs: down, then up."""
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
    """The admin tunnel's only link lost carrier while the model serves over the other port."""
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


def test_failed_attempts_back_off_and_stop_after_the_limit(pair, monkeypatch, capsys):
    assert (recovery.BACKOFF, recovery.FAILURE_LIMIT) == ((120, 300, 900), 4)
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
    for wait in recovery.BACKOFF[1:]:
        assert record(pair)["next_attempt_at"] == clock.value + wait
        clock.advance(60)
        assert run(clock)["state"] == "backoff"
        clock.advance(wait)
        assert run(clock)["state"] == "failed"
    value = record(pair)
    assert value["stopped"] is True and value["failures"] == recovery.FAILURE_LIMIT
    assert value["stop_reason"] == "4 failed attempts in a row"
    lines = status(monkeypatch, capsys)
    assert ("Automatic recovery: stopped: 4 failed attempts in a row | sudo sparkring up --execute turns it back on"
            in lines)
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
    """Every Spark restarted and each mesh unit failed at boot, so no model runs: up starts it."""
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
    assert (answer["ok"], answer["dead"]) == (False, False) and "Connection refused" in answer["error"]
    # /health without an answer asks /v1/models; an answer there counts as serving.
    asked = []

    def slow_health(url, timeout):
        asked.append(url)
        if url.endswith("/health"):
            raise TimeoutError("timed out")
        return Answer()
    answer = recovery.api_health("http://192.0.2.10:8000/v1", opener=slow_health)
    assert answer["ok"] is True and asked == ["http://192.0.2.10:8000/health", "http://192.0.2.10:8000/v1/models"]


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
    stopped = recovery.status_lines({**base, "failures": 4, "stopped": True})
    assert stopped == ["Automatic recovery: stopped: 4 failed attempts in a row | sudo sparkring up --execute turns "
                       "it back on"]
    assert recovery.status_lines(base, backend="glm-managed") == [
        "Automatic recovery: not available for this deployment (glm-managed); restart it by hand"]
    assert recovery.status_lines(base, state={"generation": 2, "operation": "down", "complete": True}) == [
        "Automatic recovery: on; idle until the next sudo sparkring up --execute or sudo sparkring install"]


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
    """Every container stopped: the first line says so and names up; containers are text lines, not JSON."""
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
    """Rank 0's container runs and its engine stopped while the worker's container stopped."""
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
    """A worker unreachable because the admin tunnel's only link lost carrier: the line names that link."""
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


def test_status_names_the_fallback_path_of_a_worker_reached_without_its_primary_cable(pair, monkeypatch, capsys):
    """The primary cable lost carrier; Node A reaches the worker over its LAN address, and status says so."""
    snapshot = {"state": "network-configured", "next_action": "sparkring models", "control": {
        "interface_up": True, "peers": [{"id": "b", "address": "192.0.2.11", "upstream": False,
                                         "netdev": "enp1s0f1np1", "carrier": False, "operstate": "down",
                                         "endpoint": "198.51.100.137:51871", "handshake_age_s": 12, "fallbacks": 2,
                                         "path": {"via": "lan", "netdev": "enP7s7", "address": "198.51.100.137",
                                                  "primary": False}}]}}
    lines = status(monkeypatch, capsys, snapshot=snapshot)
    assert "Model: The model runs on every Spark and its API answers" in lines
    row = next(index for index, line in enumerate(lines) if line.startswith("  rank 1 ("))
    assert lines[row + 1] == "    admin tunnel: over LAN 198.51.100.137 (primary cable enp1s0f1np1: no link)"
    assert recovery.tunnel_reason(snapshot["control"], pair.hosts[1]) is None


def test_status_of_a_ring_whose_mesh_failed_at_boot(tmp_path, monkeypatch, capsys):
    """Mesh units that failed at boot are named with their log line; the next step is up, not status."""
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


def test_a_manual_down_after_an_interrupted_attempt_is_not_taken_for_recoverys_own(pair):
    clock = Clock()
    recovery.update(pair.directory, attempt={"started_at": clock.value, "action": "restart", "reason": "test",
                                             "generation": state(pair)["generation"]})
    assert controller.lifecycle(["down", "--execute"]) == 0
    assert record(pair)["attempt"] is None
    pair.operations.clear()
    clock.advance(600)
    assert run(clock)["state"] == "inactive" and pair.operations == []


def test_status_of_another_deployment_shows_no_recovery_state(pair, monkeypatch, capsys):
    lines = status(monkeypatch, capsys)
    assert "Automatic recovery: on" in lines
    node.save(controller.STATE, "active.json", {"path": str(controller.STATE / "deployments" / "other")})
    capsys.readouterr()
    assert controller.lifecycle(["status", PROFILE]) == 0
    assert not any(line.startswith("Automatic recovery") for line in capsys.readouterr().out.splitlines())


def test_api_without_any_answer_must_stay_silent_before_a_restart(pair):
    """Timeouts and refused connections, unlike vLLM's 503, act only after API_SILENCE seconds."""
    clock = Clock()
    for _ in range(recovery.API_SILENCE // 60):
        result = run(clock, api=SILENT)
        assert result["state"] == "confirming" and result["finding"]["state"] == "api-failing"
        clock.advance(60)
    assert pair.operations == []
    # An answer in between starts the window again.
    assert run(clock)["state"] == "serving"
    clock.advance(60)
    assert run(clock, api=SILENT)["state"] == "confirming"
    for _ in range(recovery.API_SILENCE // 60):
        clock.advance(60)
        result = run(clock, api=SILENT)
    assert result["state"] == "recovered" and ("stop", 0) in pair.operations


def test_restarts_that_keep_failing_stop_after_the_restart_limit(pair, monkeypatch, capsys):
    """A model that stops again after every successful restart is restarted RESTART_LIMIT times, then left."""
    clock = Clock()
    for _ in range(recovery.RESTART_LIMIT):
        pair.running[1] = False
        run(clock)
        clock.advance(60)
        assert run(clock)["state"] == "recovered"
        clock.advance(1800)
    pair.running[1] = False
    pair.operations.clear()
    run(clock)
    clock.advance(60)
    result = run(clock)
    assert result["state"] == "stopped" and pair.operations == []
    reason = record(pair)["stop_reason"]
    assert reason.startswith("the model stopped again after 3 restarts within 6 hours")
    assert any(line.startswith("Automatic recovery: stopped: " + reason) for line in status(monkeypatch, capsys))
    # Restarts older than the window do not count.
    assert recovery.recent_restarts({"restarts": [0, clock.value - 60]}, clock.value) == [clock.value - 60]


@pytest.mark.parametrize("change", ["off", "other-deployment", "lock", "recovered"])
def test_an_attempt_rechecks_its_record_deployment_lock_and_model_before_acting(pair, monkeypatch, change):
    clock = Clock()
    pair.running.update({0: False, 1: False})
    run(clock)
    clock.advance(60)
    observed, held = recovery.observe, contextlib.ExitStack()

    def observe(*args, **kwargs):
        finding = observed(*args, **kwargs)
        if change == "off":
            recovery.update(pair.directory, enabled=False)
        elif change == "other-deployment":
            node.save(controller.STATE, "active.json", {"path": str(controller.STATE / "deployments" / "other")})
            (controller.STATE / "deployments" / "other").mkdir()
            installer.write(controller.STATE / "deployments" / "other" / "deployment.lock.json", {})
        elif change == "lock":
            held.enter_context(process_lock.hold(controller.STATE / "install.lock"))
        else:
            pair.running.update({0: True, 1: True})
        monkeypatch.setattr(recovery, "observe", observed)
        return finding

    monkeypatch.setattr(recovery, "observe", observe)
    with held:
        result = run(clock)
    expected = {"off": "off", "other-deployment": "changed", "lock": "busy", "recovered": "changed"}[change]
    assert result["state"] == expected and pair.operations == []
    assert record(pair).get("attempt") is None


def test_ring_member_with_a_stale_or_missing_status_report_is_unknown(tmp_path, monkeypatch):
    sparks = four_spark_ring(tmp_path, monkeypatch)
    sparks.running.update({1: False})
    # A stale document keeps the mesh block it had when its agent last wrote it.
    sparks.documents[sparks.hosts[2]] = {"state": "stale", "age_seconds": 412.0, "hostname": "spark2",
                                         "mesh": {"active": ["sparkring-test-mesh.service"], "failed": [], "markers": []}}
    sparks.documents[sparks.hosts[3]] = {"state": "agent-unavailable"}
    clock = Clock()
    applied = []
    for _ in range(3):
        result = run(clock, apply=applied.append)
        clock.advance(60)
    assert result["finding"]["state"] == "unknown" and applied == []
    assert "rank 2 (spark2): its status report is stale (412 s old)" in result["finding"]["details"]
    assert "rank 3 (root@192.0.2.13): its status service has not reported yet" in result["finding"]["details"]


def test_ring_guard_is_checked_again_under_the_lock(tmp_path, monkeypatch):
    """Marker processes left without a mesh unit between the check and the attempt stop the attempt."""
    sparks = four_spark_ring(tmp_path, monkeypatch)
    sparks.running.update({0: False, 1: False, 2: False, 3: False})
    monkeypatch.setattr(controller, "_hairpin_problem", lambda: None)
    clock = Clock()
    applied = []
    run(clock, apply=applied.append)
    clock.advance(60)
    observed = recovery.observe

    def observe(*args, **kwargs):
        finding = observed(*args, **kwargs)
        sparks.mesh[sparks.hosts[2]] = {"active": [], "markers": [4242], "failed": []}
        return finding

    monkeypatch.setattr(recovery, "observe", observe)
    result = run(clock, apply=applied.append)
    assert result["state"] == "changed" and applied == []
    assert result["finding"]["state"] == "mesh-cleanup"


def test_an_ssh_failure_while_probing_one_container_is_unknown(pair):
    clock = Clock()
    pair.running[1] = False
    pair.probe_fails.add(1)
    for _ in range(3):
        result = run(clock)
        clock.advance(60)
    assert result["state"] == "report" and result["finding"]["state"] == "unknown" and pair.operations == []
    assert result["finding"]["details"] == [
        "rank 1 (spark1): root@192.0.2.11: Cannot connect to the Docker daemon at unix:///var/run/docker.sock"]


def test_an_unreadable_recovery_file_is_moved_aside_and_recovery_continues(pair, monkeypatch, capsys):
    recovery.path().write_text("{not json")
    lines = status(monkeypatch, capsys)
    assert "Automatic recovery: state unreadable; use sudo sparkring status" in lines
    clock = Clock()
    assert run(clock)["state"] == "serving"
    assert "was unreadable; it was moved to" in capsys.readouterr().err
    aside = list(recovery.path().parent.glob("recovery.json.unreadable-*"))
    assert len(aside) == 1 and aside[0].read_text() == "{not json"
    assert recovery.load()["schema"] == recovery.SCHEMA and record(pair)["enabled"] is True


def test_a_failed_retained_source_is_named_by_its_exit_status(pair):
    clock = Clock()
    pair.running.update({0: False, 1: False})

    def retained(operation):
        raise subprocess.CalledProcessError(1, ["python3", "-c", "x = 1\n" * 4000])

    run(clock)
    clock.advance(60)
    assert run(clock, apply=retained)["state"] == "failed"
    error = record(pair)["last"]["error"]
    assert error == ("the deployment's own source exited with status 1; sudo sparkring logs --details shows its "
                     "output")
    assert len(recovery.error_text(RuntimeError("y" * 5000))) == node.ERROR_TEXT


def test_up_keeps_off_and_install_style_choice_is_explicit(pair):
    recovery.set_enabled(False, directory=pair.directory)
    assert controller.lifecycle(["up", "--execute"]) == 0
    assert record(pair)["enabled"] is False
    recovery.started(pair.directory, enabled=True)
    assert record(pair)["enabled"] is True


def test_a_failing_recovery_record_never_stops_up_or_down(pair, monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(recovery, "update", broken)
    assert controller.lifecycle(["down", "--execute"]) == 0
    assert controller.lifecycle(["up", "--execute"]) == 0
    err = capsys.readouterr().err
    assert "automatic recovery state was not updated" in err and "was not recorded" in err


def test_status_after_a_deliberate_down_says_recovery_is_idle(pair, monkeypatch, capsys):
    assert controller.lifecycle(["down", "--execute"]) == 0
    lines = status(monkeypatch, capsys)
    assert ("Automatic recovery: on; idle until the next sudo sparkring up --execute or sudo sparkring install"
            in lines)


def test_up_of_another_deployment_forgets_the_active_ones_unfinished_attempt(pair):
    assert controller.lifecycle(["down", "--execute"]) == 0
    recovery.update(pair.directory, attempt={"started_at": 1.0, "action": "restart", "reason": "test",
                                             "generation": state(pair)["generation"]})
    assert controller.lifecycle(["up", PROFILE, "--instance", "other", "--execute"]) == 0
    assert record(pair)["attempt"] is None


def test_unsupported_backends_are_recorded_and_reported_as_such(tmp_path, monkeypatch):
    sparks = four_spark_ring(tmp_path, monkeypatch)
    lock = installer.read(sparks.directory / "deployment.lock.json")
    (sparks.directory / "deployment.lock.json").unlink()
    installer.write(sparks.directory / "deployment.lock.json", {**lock, "backend": "glm-existing-mesh"})
    value = recovery.started(sparks.directory)
    assert value["supported"] is False
    assert run(Clock())["state"] == "unsupported"
    with pytest.raises(ValueError, match="does not restart this deployment"):
        recovery.set_enabled(True)


def test_prerm_stops_a_running_recovery_before_tearing_units_down():
    from runtime.host.test_persistence import PACKAGING
    text = (PACKAGING / "prerm").read_text()
    stop = text.index("systemctl stop sparkring-recover.timer sparkring-recover.service")
    assert stop < text.index("systemctl disable --now $UNITS")


def test_host_tests_never_call_systemd_for_the_timer():
    assert recovery.enable_timer() is False and recovery.timer_enabled() is None


def test_status_shows_the_recorded_address_while_checks_and_recovery_use_the_listen_address(pair, monkeypatch,
                                                                                             capsys):
    from runtime.host import api_endpoint
    api_endpoint.record(pair.directory, "llm.example.net")
    probed = []
    monkeypatch.setattr(recovery, "api_health", lambda url: probed.append(url) or SERVING)
    monkeypatch.setattr(recovery, "timer_enabled", lambda: True)
    monkeypatch.setattr(controller.node, "snapshot", lambda: {"state": "network-configured", "next_action": "x"})
    capsys.readouterr()
    assert controller.lifecycle(["status", "--refresh"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert ("http://llm.example.net:8000/v1 (SparkRing's own checks use http://192.0.2.10:8000/v1)" in lines
            and "Model: The model runs on every Spark and its API answers" in lines)
    assert controller.lifecycle(["status", "--refresh", "--json"]) == 0
    deployment = json.loads(capsys.readouterr().out)["deployment"]
    assert (deployment["api_url"], deployment["check_url"]) == ("http://llm.example.net:8000/v1",
                                                                 "http://192.0.2.10:8000/v1")
    assert recovery.check(now=lambda: 2_000_000.0, api=lambda url: probed.append(url) or SERVING,
                          tunnel={})["state"] == "serving"
    assert set(probed) == {"http://192.0.2.10:8000/v1"}


def test_up_checks_a_new_deployment_s_listen_address_on_its_api_spark(pair, monkeypatch, capsys):
    from runtime.host import api_endpoint, discovery
    reports = []
    report = {"addresses": [{"interface": "enP7s7", "address": "203.0.113.7", "prefix": 24, "state": "UP"},
                            {"interface": "lo", "address": "127.0.0.1", "prefix": 8, "state": "UNKNOWN"}],
              "listeners": [], "control_subnet": None}

    def ssh(host, argv, **kwargs):
        if argv == api_endpoint.PROBE_COMMAND:
            reports.append(host)
            return json.dumps(report)
        return pair.ssh(host, argv, **kwargs)
    monkeypatch.setattr(discovery, "ssh", ssh)
    with pytest.raises(ValueError, match="--api-bind 198.51.100.9 is not an address of Node A"):
        controller.lifecycle(["up", PROFILE, "--instance", "bound", "--api-bind", "198.51.100.9", "--plan"])
    assert not (controller.STATE / "deployments" / (PROFILE + "-bound")).exists()
    with pytest.raises(ValueError, match="Add --allow-loopback-bind to serve it that way"):
        controller.lifecycle(["up", PROFILE, "--instance", "bound", "--api-bind", "127.0.0.1", "--plan"])
    assert controller.lifecycle(["up", PROFILE, "--instance", "bound", "--api-bind", "203.0.113.7", "--api-port",
                                 "9100", "--plan"]) == 0
    lock = installer.read(controller.STATE / "deployments" / (PROFILE + "-bound") / "deployment.lock.json")
    assert lock["serving"] == {"api_bind": "203.0.113.7", "api_port": 9100}
    assert installer.connection(lock)["api_url"] == "http://203.0.113.7:9100/v1"
    assert reports == ["root@192.0.2.10"] * 3
    assert "Serving settings: --api-bind 203.0.113.7, --api-port 9100" in capsys.readouterr().out
