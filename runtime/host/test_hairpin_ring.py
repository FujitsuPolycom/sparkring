"""The ring procedure for the ConnectX hairpin setting against simulated Sparks; no SSH or systemd is used."""
import copy
import json

import pytest

from runtime.host import controller, hairpin, hairpin_ring as ring_module, node, topology
from runtime.host.install_errors import NeedsInput
from runtime.host.test_appliance import nodes
from scripts import hairpin_setting as rule

REVISION = "a" * 40
OLDER = "b" * 40
ROLE_ORDER = ("cw_primary", "cw_secondary", "ccw_primary", "ccw_secondary")
# Control addresses whose ascending order (ranks 0, 1, 3, 2) differs from rank order.
CONTROL = ["10.253.255.1", "10.253.255.2", "10.253.255.4", "10.253.255.3"]


def functions(plan, rank, states):
    """Status rows of one Spark's four functions, identified as the plan records them."""
    host = plan["spec"]["hosts"][rank]
    pci = {row["device"]: row["pci_address"] for row in plan["inventory"]["hosts"][host["host"]]["rdma"]}
    ports = sorted(host["data_interfaces"], key=lambda port: ROLE_ORDER.index(port["role"]))
    rows = []
    for port, state in zip(ports, states, strict=True):
        rows.append({"role": port["role"], "rdma_device": port["rdma_device"], "pci_address": pci[port["rdma_device"]],
                     "netdev": port["netdev"], "mac": port["mac"],
                     "values": {"hairpin_num_queues": 4, "hairpin_queue_size": 1024 if state == rule.DEFAULT else 8192},
                     "driver_reinit": None if state == rule.UNKNOWN else 0 if state in (rule.DEFAULT, rule.PENDING) else 1,
                     "reload_failed": state == rule.FAILED, "offload": "off" if state == rule.OFFLOAD_OFF else "on",
                     "state": state})
    return rows


def document(plan, rank, state=rule.IN_EFFECT, *, approved=True, armed=True, suspended=None, revision=REVISION,
             unit=None, last_run=None):
    """A sparkring-hairpin-status/v1 document; a kept Spark by default."""
    states = [state] * 4 if isinstance(state, str) else list(state)
    rows = functions(plan, rank, states)
    return {"schema": ring_module.STATUS_SCHEMA, "hostname": f"spark{rank}", "revision": revision,
            "approval": {"present": approved, "valid": approved, "error": None}, "armed": armed,
            "boot_disabled_by_kernel_command_line": False, "suspended": suspended,
            "unit": unit or {"ActiveState": "active" if armed else "inactive", "Result": "success",
                             "InvocationID": f"earlier{rank}"},
            "last_run": last_run, "function_source": "approval" if approved else "fabric", "functions": rows,
            "in_effect": all(s == rule.IN_EFFECT for s in states), "blocked_units": [], "warnings": [],
            "read_errors": [], "parameters": dict(rule.PARAMETERS)}


def kept(plan):
    """Give every node of a four-Spark plan a kept hairpin status document."""
    if len(plan["nodes"]) == 4:
        for rank, current in enumerate(plan["nodes"]):
            current["hairpin"] = document(plan, rank, revision=current.get("revision"))
    return plan


def ring_plan(*, single_uplink=False):
    found = nodes(4)
    plan = topology.build_spec(found, found[0]["node_id"])
    if single_uplink:
        for rank, host in enumerate(plan["spec"]["hosts"]):
            host["management_address"] = CONTROL[rank]
            if rank:
                host["management_netdev"] = "sr-control"
    return kept(plan)


class Spark:
    """One simulated Spark: its status document and how its next unit run ends."""

    def __init__(self, rank, status):
        self.rank = rank
        self.status = status
        self.outcome = "success"
        self.polls = 1
        self.remaining = 0
        self.runs = 0
        self.unreachable = 0
        self.busy = []
        self.approval = None
        self.revision_after_update = REVISION
        # Mesh units that the start check refused in this boot.
        self.blocked = []


class Ring:
    """Simulated Sparks behind invoke/run_local, with a fake clock and a call log."""

    def __init__(self, plan):
        self.plan = plan
        # A Spark without a status document runs a SparkRing revision that predates it.
        self.sparks = {rank: Spark(rank, copy.deepcopy(plan["nodes"][rank].get("hairpin") or {"schema": None}))
                       for rank in range(4)}
        self.calls = []
        self.events = []
        self.time = 0.0
        self.lldp_failures = 0
        self.hosts = {host["host"]: rank for rank, host in enumerate(plan["spec"]["hosts"])}

    def clock(self):
        return self.time

    def sleep(self, seconds):
        self.time += seconds

    def invoke(self, host, argv, *, timeout):
        assert argv[:2] == ["sudo", "-n"], argv
        return self.handle(self.hosts[host], list(argv[2:]))

    def run_local(self, argv, *, timeout):
        return self.handle(0, list(argv))

    def handle(self, rank, argv):
        self.calls.append((rank, tuple(argv)))
        spark = self.sparks[rank]
        if spark.unreachable:
            spark.unreachable -= 1
            raise RuntimeError(f"root@192.0.2.{10 + rank}: ssh: connect to host port 22: Connection timed out")
        if argv[:4] == [ring_module.SPARKRING, "node", "hairpin", "status"]:
            self.tick(spark)
            value = copy.deepcopy(spark.status)
            if "--busy" in argv:
                value["busy"] = spark.busy
            return json.dumps(value)
        if argv[:4] == [ring_module.SPARKRING, "node", "hairpin", "approve"]:
            rows = spark.approval or [{key: row[key] for key in ("role", "rdma_device", "pci_address", "netdev", "mac")}
                                      for row in spark.status["functions"]]
            spark.status["approval"] = {"present": True, "valid": True, "error": None}
            spark.status["function_source"] = "approval"
            return json.dumps({"schema": hairpin.APPROVAL_SCHEMA, "functions": rows})
        if argv[:4] == [ring_module.SPARKRING, "node", "hairpin", "revoke"]:
            spark.status["approval"] = {"present": False, "valid": False, "error": None}
            spark.status["armed"] = False
            return json.dumps({"revoked": True})
        if argv[:4] == [ring_module.SPARKRING, "node", "hairpin", "start"]:
            # sparkring node hairpin start --after: restart only while no newer run exists.
            after, unit = argv[argv.index("--after") + 1], spark.status["unit"]
            if unit["ActiveState"] == "activating" or (unit.get("InvocationID") or "") != after:
                return json.dumps({"started": False})
            self.start(spark, blocking=False)
            return json.dumps({"started": True})
        if argv[:4] == [ring_module.SPARKRING, "node", "hairpin", "resume"]:
            started, spark.blocked = spark.blocked, []
            self.events += [("mesh-start", spark.rank)] * len(started)
            return json.dumps({"started": started, "skipped": []})
        if argv[:2] == ["systemctl", "restart"]:
            self.start(spark, blocking="--no-block" not in argv)
            if spark.status["unit"]["ActiveState"] == "failed" and "--no-block" not in argv:
                raise RuntimeError("systemctl restart sparkring-hairpin.service: Job failed")
            return ""
        if argv == ["lldpcli", "update"]:
            return ""
        if argv[:2] == ["systemctl", "list-unit-files"]:
            return ("sparkring-mesh.service enabled enabled\n"
                    "sparkring-glm-mesh.service enabled enabled\n"
                    "sparkring-old-mesh.service disabled enabled\n")
        if argv[:3] == ["systemctl", "show", "-p"]:
            # Rank 3 runs no mesh; every rank also holds an enabled mesh unit of
            # another deployment and a disabled one, both stopped.
            # systemctl show prints the units in the order requested (sorted).
            return ("Id=sparkring-glm-mesh.service\nActiveState=inactive\nUnitFileState=enabled\n\n"
                    "Id=sparkring-mesh.service\nActiveState=" + ("inactive" if rank == 3 else "active")
                    + "\nUnitFileState=enabled\n\n"
                    "Id=sparkring-old-mesh.service\nActiveState=inactive\nUnitFileState=disabled\n")
        raise AssertionError(f"unexpected command on rank {rank}: {argv}")

    def start(self, spark, *, blocking):
        spark.runs += 1
        self.events.append(("start", spark.rank))
        spark.status["unit"] = {"ActiveState": "activating", "Result": "success",
                                "InvocationID": f"run{spark.rank}-{spark.runs}"}
        spark.remaining = spark.polls
        if blocking:
            self.finish(spark)

    def tick(self, spark):
        if spark.status["unit"]["ActiveState"] == "activating" and spark.outcome != "silent":
            spark.remaining -= 1
            if spark.remaining < 0:
                self.finish(spark)

    def finish(self, spark):
        self.events.append(("finish", spark.rank))
        status, identifier = spark.status, spark.status["unit"]["InvocationID"]
        restarted = [row["pci_address"] for row in status["functions"] if row["state"] in rule.RESTART_STATES]
        first = status["functions"][0]
        if spark.outcome == "success":
            for row in status["functions"]:
                row.update(state=rule.IN_EFFECT, values=dict(rule.PARAMETERS), offload="on", reload_failed=False,
                           driver_reinit=(row["driver_reinit"] or 0) + (row["pci_address"] in restarted))
            status.update(in_effect=True, armed=True, suspended=None)
            status["unit"].update(ActiveState="active", Result="success")
            status["last_run"] = {"invocation_id": identifier, "restarted": restarted, "error": None, "class": None,
                                  "in_effect": True}
            return
        tunnel = "the administration tunnel to 10.253.255.1"
        refused = "driver restart failed: Invalid argument"
        # As the node records them: the message, its class and, for restart
        # outcomes, the short cause of the failed function.
        errors = {"restart": (hairpin.message_m1(first, "Invalid argument"), "restart", refused),
                  "cut": (hairpin.message_m1(first, "Invalid argument", hairpin.cut_off_advice(
                      first, head=spark.rank == 0)), "restart", refused),
                  "check": (hairpin.message_m18(first, tunnel, 30), "check", hairpin.check_cause(tunnel, 30)),
                  "M3": (hairpin.message_m3(first), "M3", None),
                  "M14": (hairpin.message_m14(first, "enp9s0, MAC 02:00:00:00:00:09, rocep9s0"), "M14", None)}
        status["unit"].update(ActiveState="failed", Result="exit-code")
        if spark.outcome == "no-record":
            return
        error, kind, cause = errors[spark.outcome]
        status["last_run"] = {"invocation_id": identifier, "restarted": restarted[:1] if cause else [],
                              "error": error, "class": kind, "cause": cause,
                              "function": f"{first['netdev']} (pci/{first['pci_address']})" if cause else None,
                              "cut_off": first["netdev"] if spark.outcome == "cut" else None, "in_effect": False}

    def update(self):
        self.calls.append(("update-workers",))
        for spark in self.sparks.values():
            spark.status["revision"] = spark.revision_after_update
            if spark.status.get("schema") != ring_module.STATUS_SCHEMA:
                spark.status = document(self.plan, spark.rank, revision=spark.revision_after_update)

    def inspect(self, targets):
        self.calls.append(("inspect",))
        found = copy.deepcopy(self.plan["nodes"])
        for rank, current in enumerate(found):
            current["hairpin"] = copy.deepcopy(self.sparks[rank].status)
        return found

    def rebuild(self, found):
        self.calls.append(("rebuild",))
        if self.lldp_failures:
            self.lldp_failures -= 1
            raise ValueError("Missing reciprocal LLDP cable evidence; wait for LLDP or inspect cabling")
        plan = copy.deepcopy(self.plan)
        plan["nodes"] = found
        for current, host in zip(found, plan["network"]["hosts"], strict=True):
            # As the planner does: only a Spark whose functions are all in effect has no driver drift.
            status = current.get("hairpin") or {}
            host["driver_action"] = "none" if status.get("in_effect") else "apply"
            host["hairpin"] = copy.deepcopy(status.get("functions") or [])
        return plan

    def ensure(self, plan=None, tmp_path=None, **options):
        options.setdefault("approved", True)
        return ring_module.ensure(plan or self.plan, inspect=self.inspect, rebuild=self.rebuild,
                                  update_workers=self.update, invoke=self.invoke, run_local=self.run_local,
                                  directory=tmp_path, clock=self.clock, sleep=self.sleep, **options)

    def commands(self, *words):
        return [(rank, argv) for rank, *rest in self.calls if rest for argv in rest
                if all(word in argv for word in words)]

    def changing(self):
        """Calls that change a Spark: unit starts, approvals, revocations and resumed mesh units."""
        return [entry for entry in self.calls if len(entry) == 2 and (
            entry[1][:2] == ("systemctl", "restart") or entry[1][3:4] in (("approve",), ("revoke",), ("start",),
                                                                          ("resume",)))]

    def starts(self):
        """The ranks whose unit a call started (Node A's blocking restart, a worker's dispatch)."""
        return [rank for rank, argv in self.changing() if argv[:2] == ("systemctl", "restart") or argv[3:4] == ("start",)]


def needing(plan, ranks, state=rule.DEFAULT, **options):
    for rank in ranks:
        plan["nodes"][rank]["hairpin"] = document(plan, rank, state, approved=False, armed=False, **options)
    return plan


def test_pair_and_all_kept_ring_make_no_calls(tmp_path):
    found = nodes(2)
    pair = topology.build_spec(found, found[0]["node_id"])
    record = {}
    assert ring_module.ensure(pair, approved=False, inspect=None, rebuild=None, update_workers=None,
                              invoke=pytest.fail, run_local=pytest.fail, record=record) is pair
    assert record["state"] == "not-required" and ring_module.requirement(pair) == []
    plan = ring_plan()
    ring = Ring(plan)
    assert ring.ensure(tmp_path=tmp_path, approved=False) is plan
    assert ring.calls == [] and not (tmp_path / "hairpin.json").exists()


def test_requirement_classifies_each_spark():
    plan = ring_plan()
    plan["nodes"][1]["hairpin"] = document(plan, 1, approved=False, armed=False)
    plan["nodes"][2]["hairpin"] = document(plan, 2, [rule.DEFAULT, rule.PENDING, rule.IN_EFFECT, rule.IN_EFFECT])
    del plan["nodes"][3]["hairpin"]
    plan["nodes"][3]["revision"] = OLDER
    rows = ring_module.requirement(plan)
    assert [row["state"] for row in rows] == ["kept", "record", "restart", "update"]
    plan["nodes"][3]["hairpin"] = document(plan, 3, [rule.UNKNOWN] * 4)
    plan["nodes"][3]["revision"] = REVISION
    assert ring_module.requirement(plan)[3]["state"] == "unknown"
    # A suspended boot record keeps an in-effect Spark from being kept.
    plan["nodes"][0]["hairpin"]["suspended"] = {"state": "failed", "boot_id": "b1"}
    assert ring_module.requirement(plan)[0]["state"] == "record"


def test_consent_text_names_each_rank_and_the_update():
    plan = ring_plan(single_uplink=True)
    needing(plan, [3])
    plan["nodes"][0]["hairpin"] = document(plan, 0, approved=False, armed=False)
    for rank in (1, 2):
        plan["nodes"][rank]["revision"] = OLDER
    rows = ring_module.requirement(plan)
    text = "\n".join(ring_module.consent_lines(plan, rows))
    assert "rank 0 spark0: in effect; record it and apply it at every boot (no restart)" in text
    assert "rank 3 spark3: restart 4 functions (1024 -> 8192)" in text
    assert "ranks 1-2: update SparkRing to aaaaaaaaaaaa first" in text
    assert ring_module.NOTICE_SINGLE_UPLINK in text
    # A restart on a ring whose workers depend on the ring cables defaults to No.
    assert ring_module.consent_default(plan, rows) is False
    assert ring_module.consent_default(ring_plan(), ring_module.requirement(needing(ring_plan(), [3]))) is True
    assert ring_module.m7(rows) == (
        "This installation also applies the ConnectX hairpin setting: each function of rank 3 that needs it restarts "
        "its driver once (link down about 8 seconds, about 30 seconds per Spark); ranks 1-2 are updated to this "
        "SparkRing revision first; rank 0 only records it. SparkRing then repeats it at every boot before "
        "networking starts. Review with --plan, then repeat with --yes.")
    assert ring_module.m15_plan(rows[1], REVISION).startswith("Rank 1 (spark1) runs SparkRing bbbbbbbbbbbb")


def test_record_only_consent_says_no_restart():
    plan = ring_plan()
    for rank in range(4):
        plan["nodes"][rank]["hairpin"] = document(plan, rank, approved=False, armed=False)
    rows = ring_module.requirement(plan)
    lines = ring_module.consent_lines(plan, rows)
    assert lines[1] == "It is in effect on all 4 Sparks. SparkRing records it and applies it at every boot"
    assert "No driver restarts now." in lines[2]
    assert ring_module.m7(rows).startswith("This installation records the ConnectX hairpin setting")
    assert ring_module.consent_default(plan, rows) is True


@pytest.mark.parametrize("state,expected", [
    ("kept", "in effect; applied at every boot"),
    ("record", "in effect; record it and apply it at every boot (no restart)"),
    ("restart", "restart 4 functions after addressing (1024 -> 8192), about 8 s link loss each"),
    ("suspended", "boot restarts suspended after a failed restart; retry live"),
    ("update", "read after updating SparkRing on this Spark")])
def test_summary_lines(state, expected):
    plan = ring_plan()
    if state == "record":
        plan["nodes"][1]["hairpin"] = document(plan, 1, approved=False)
    elif state == "restart":
        needing(plan, [1])
    elif state == "suspended":
        plan["nodes"][1]["hairpin"] = document(plan, 1, suspended={"state": "failed", "boot_id": "b1"})
    elif state == "update":
        del plan["nodes"][1]["hairpin"]
    assert ring_module.summary_line(ring_module.requirement(plan)[1]) == expected


def test_unknown_statistics_refuse_before_any_change(tmp_path):
    plan = ring_plan()
    plan["nodes"][2]["hairpin"] = document(plan, 2, [rule.UNKNOWN, rule.IN_EFFECT, rule.IN_EFFECT, rule.IN_EFFECT])
    needing(plan, [1])
    ring = Ring(plan)
    with pytest.raises(NeedsInput) as caught:
        ring.ensure(tmp_path=tmp_path)
    assert caught.value.field == "driver" and "devlink reload statistics are unavailable for" in str(caught.value)
    assert ring.calls == []


def test_missing_consent_stops_before_any_change(tmp_path):
    ring = Ring(needing(ring_plan(), [2]))
    with pytest.raises(NeedsInput) as caught:
        ring.ensure(tmp_path=tmp_path, approved=False)
    assert caught.value.field == "approval" and "Review with --plan, then repeat with --yes." in str(caught.value)
    assert ring.calls == []
    assert json.loads((tmp_path / "hairpin.json").read_text())["state"] == "needs_input"


def test_older_revisions_are_updated_before_any_status_or_approval(tmp_path):
    plan = ring_plan()
    del plan["nodes"][2]["hairpin"]
    plan["nodes"][2]["revision"] = OLDER
    ring = Ring(plan)
    record = {}
    result = ring.ensure(tmp_path=tmp_path, record=record)
    assert ring.calls[0] == ("update-workers",)
    assert not ring.changing()
    # Updated Sparks carry reload statistics only in a fresh inventory, so the ring is re-inspected.
    assert ring.calls.index(("inspect",)) < ring.calls.index(("rebuild",))
    assert result["nodes"][2]["hairpin"]["revision"] == REVISION
    assert record["state"] == "complete" and record["reinspected"] is True


def test_worker_updates_authenticate_each_bulk_path_first(tmp_path, monkeypatch):
    from runtime.host import fabric_ssh, install_assets
    calls = []

    class Transport:
        def __init__(self, cluster, directory):
            calls.append("transport")

        def verify(self):
            calls.append("verify")

    class Assets:
        def __init__(self, transport, directory):
            pass

        def sync_packages(self):
            calls.append("sync")
            return {"updated": [2]}
    monkeypatch.setattr(controller, "STATE", tmp_path)
    monkeypatch.setattr(fabric_ssh, "Transport", Transport)
    monkeypatch.setattr(install_assets, "Assets", Assets)
    assert ring_module._update_workers({"plan": ring_plan()}, tmp_path / "hairpin")["updated"] == [2]
    assert calls == ["transport", "verify", "sync"]


def test_revision_still_older_after_update_gives_m15(tmp_path):
    plan = ring_plan()
    plan["nodes"][3]["revision"] = OLDER
    plan["nodes"][3]["hairpin"]["revision"] = OLDER
    ring = Ring(plan)
    ring.sparks[3].status["revision"] = OLDER
    ring.sparks[3].revision_after_update = OLDER
    with pytest.raises(NeedsInput) as caught:
        ring.ensure(tmp_path=tmp_path)
    assert "Rank 3 (spark3) still runs SparkRing bbbbbbbbbbbb after the update" in str(caught.value)
    assert not ring.changing()


def test_busy_ring_gives_m8_before_any_approval_or_restart(tmp_path):
    plan = needing(ring_plan(), [1])
    ring = Ring(plan)
    ring.sparks[3].busy = [{"kind": "unit", "unit": "sparkring-mesh.service", "active_state": "active", "processes": ["41"],
                            "detail": "sparkring-mesh.service is active"},
                           {"kind": "forwarding", "netdev": "enp1s0f0np0", "count": 2,
                            "detail": "2 forwarding rule(s) on enp1s0f0np0"}]
    ring.sparks[0].busy = [{"kind": "gpu", "pid": "900", "detail": "GPU compute process PID 900"}]
    with pytest.raises(NeedsInput) as caught:
        ring.ensure(tmp_path=tmp_path)
    error = caught.value
    assert error.field == "driver"
    assert str(error).startswith("The ConnectX hairpin setting must be applied on 1 Spark, and 2 Sparks are in use.")
    lines = error.details["lines"]
    assert "SparkRing's model: on Node A, sudo sparkring down" in lines
    assert any("stop it: ssh -t root@192.0.2.13 sudo systemctl stop sparkring-mesh.service" in line for line in lines)
    assert any("if it is already stopped, reboot that Spark" in line for line in lines)
    assert sorted(rank for rank, argv in ring.commands("--busy")) == [0, 1, 2, 3]
    assert not ring.changing()


def test_approvals_precede_every_restart(tmp_path):
    plan = needing(ring_plan(), [0, 2])
    ring = Ring(plan)
    ring.ensure(tmp_path=tmp_path)
    changes = ring.changing()
    approvals = [index for index, (rank, argv) in enumerate(changes) if "approve" in argv]
    restarts = [index for index, (rank, argv) in enumerate(changes)
                if argv[:2] == ("systemctl", "restart") or argv[3:4] == ("start",)]
    assert [changes[i][0] for i in approvals] == [0, 2]
    assert max(approvals) < min(restarts)


def test_approved_rows_that_differ_from_the_plan_stop_before_any_restart(tmp_path):
    plan = needing(ring_plan(), [0, 2])
    ring = Ring(plan)
    rows = [{key: row[key] for key in ("role", "rdma_device", "pci_address", "netdev", "mac")}
            for row in ring.sparks[2].status["functions"]]
    rows[1]["mac"] = "02:00:00:00:99:99"
    ring.sparks[2].approval = rows
    with pytest.raises(NeedsInput) as caught:
        ring.ensure(tmp_path=tmp_path)
    assert "Rank 2 (spark2): the approved ConnectX functions differ from the plan" in str(caught.value)
    assert "ssh -t root@192.0.2.12 sudo /usr/bin/sparkring node hairpin revoke" in str(caught.value)
    assert not ring.starts()


def test_node_a_runs_locally_first_and_its_failure_leaves_workers_untouched(tmp_path):
    plan = needing(ring_plan(), [0, 1, 2, 3])
    ring = Ring(plan)
    ring.sparks[0].outcome = "restart"
    with pytest.raises(NeedsInput) as caught:
        ring.ensure(tmp_path=tmp_path)
    message = str(caught.value)
    # The node's short cause, without its prefix and its own advice.
    assert message.startswith("The ConnectX hairpin setting did not complete on rank 0 (spark0): enp1s0f0np0 "
                              "(pci/0000:01:00.0): driver restart failed: Invalid argument. Later Sparks were not "
                              "changed.")
    assert "SparkRing hairpin:" not in message and "Boot runs on this Spark" not in message
    assert "power-cycle it; it starts without restarting any function and stays reachable" in message
    restarts = [(rank, argv) for rank, argv in ring.changing() if argv[:2] == ("systemctl", "restart")]
    # Node A blocks on its own run; no worker was dispatched.
    assert restarts == [(0, ("systemctl", "restart", ring_module.UNIT))]
    receipt = json.loads((tmp_path / "hairpin.json").read_text())
    assert receipt["state"] == "needs_input" and receipt["ranks"][0]["outcome"] == "failed"


@pytest.mark.parametrize("single_uplink,order", [(False, [0, 1, 2, 3]), (True, [0, 1, 3, 2])])
def test_workers_run_one_at_a_time_in_tree_or_rank_order(tmp_path, single_uplink, order):
    plan = needing(ring_plan(single_uplink=single_uplink), [0, 1, 2, 3])
    ring = Ring(plan)
    for spark in ring.sparks.values():
        spark.polls = 3
    record = {}
    result = ring.ensure(tmp_path=tmp_path, record=record)
    starts = [rank for kind, rank in ring.events if kind == "start"]
    assert starts == order
    # Each run finishes before the next one starts.
    assert ring.events == [event for rank in order for event in (("start", rank), ("finish", rank))]
    dispatched = [rank for rank, argv in ring.changing() if argv[:4] == (ring_module.SPARKRING, "node", "hairpin",
                                                                          "start")]
    assert dispatched == order[1:]
    # Each dispatch names the invocation it read, so a repeat never restarts a newer run.
    assert all(argv[4:] == ("--after", f"earlier{rank}") for rank, argv in ring.changing() if argv[3:4] == ("start",))
    assert record["state"] == "complete" and all(entry["outcome"] == "applied" for entry in record["ranks"])
    assert all(current["hairpin"]["in_effect"] for current in result["nodes"])


def test_ssh_errors_while_polling_are_tolerated(tmp_path):
    plan = needing(ring_plan(), [1])
    ring = Ring(plan)
    spark = ring.sparks[1]
    spark.polls = 2
    original = ring.start

    def start(target, *, blocking):
        original(target, blocking=blocking)
        if target.rank == 1:
            target.unreachable = 4
    ring.start = start
    ring.ensure(tmp_path=tmp_path)
    assert ring.sparks[1].status["in_effect"] and ring.time >= 4 * ring_module.POLL


def test_an_earlier_invocation_or_run_record_is_never_success(tmp_path):
    plan = needing(ring_plan(), [1])
    ring = Ring(plan)
    # The earlier run's record says in effect; the new run fails without writing one.
    ring.sparks[1].status["last_run"] = {"invocation_id": "earlier1", "error": None, "in_effect": True, "restarted": []}
    ring.sparks[1].outcome = "no-record"
    with pytest.raises(NeedsInput) as caught:
        ring.ensure(tmp_path=tmp_path)
    assert "without a run record" in str(caught.value)


def test_a_run_that_never_starts_is_silent_after_the_limit(tmp_path):
    plan = needing(ring_plan(), [2])
    ring = Ring(plan)
    ring.sparks[2].outcome = "silent"
    with pytest.raises(NeedsInput) as caught:
        ring.ensure(tmp_path=tmp_path)
    assert str(caught.value).startswith(f"Rank 2 (spark2) did not report back within {ring_module.REPORT} s")
    # A Spark armed earlier may apply the setting again at boot, so no promise of no restart is made.
    assert ("If it stays unreachable, power-cycle it; after a restart that failed or did not finish, its next boot "
            "restarts no function. Then repeat this command.") in str(caught.value)
    assert ring.time >= ring_module.REPORT


def test_an_activating_unit_is_awaited_not_restarted(tmp_path):
    plan = needing(ring_plan(), [1])
    ring = Ring(plan)
    ring.sparks[1].status["unit"] = {"ActiveState": "activating", "Result": "success", "InvocationID": "prior1"}
    ring.sparks[1].remaining = 2
    ring.ensure(tmp_path=tmp_path)
    assert 1 not in ring.starts()
    assert ring.sparks[1].status["unit"]["InvocationID"] == "prior1" and ring.sparks[1].status["in_effect"]


def test_unreachable_worker_is_never_started(tmp_path):
    plan = ring_plan()
    # Approved but not armed: the run needs no restart, so no busy check or approval reaches SSH first.
    for rank in (1, 2):
        plan["nodes"][rank]["hairpin"] = document(plan, rank, armed=False)
    ring = Ring(plan)
    ring.sparks[1].unreachable = 10**6
    with pytest.raises(NeedsInput) as caught:
        ring.ensure(tmp_path=tmp_path)
    message = str(caught.value)
    assert message.startswith("Could not start the ConnectX hairpin step on rank 1 (spark1):")
    assert "No function was restarted there; later Sparks were not changed." in message
    assert 2 not in ring.starts()


@pytest.mark.parametrize("outcome,advice", [
    ("restart", "stays reachable. Then run sudo sparkring hairpin."),
    ("check", "restarted, but the administration tunnel to 10.253.255.1 did not return within 30 s. Later Sparks "
              "were not changed. If the Spark cannot be reached, power-cycle it, then run sudo sparkring hairpin "
              "again."),
    ("M3", None)])
def test_failed_runs_give_advice_by_class(tmp_path, outcome, advice):
    plan = needing(ring_plan(), [3])
    ring = Ring(plan)
    ring.sparks[3].outcome = outcome
    with pytest.raises(NeedsInput) as caught:
        ring.ensure(tmp_path=tmp_path)
    message = str(caught.value)
    assert message.startswith("The ConnectX hairpin setting did not complete on rank 3 (spark3):")
    assert "Later Sparks were not changed." in message
    if advice:
        assert advice in message
    else:
        assert "power-cycle" not in message and "Check that port's cable and transceiver" in message
    assert caught.value.details["class"] == outcome
    assert any(line.startswith("Log: ssh -t root@192.0.2.13 sudo journalctl -u sparkring-hairpin.service")
               for line in caught.value.details["lines"])


def test_record_only_ring_gets_approvals_and_runs_without_busy_check_or_reinspection(tmp_path):
    plan = ring_plan()
    for rank in range(4):
        plan["nodes"][rank]["hairpin"] = document(plan, rank, approved=False, armed=False)
    ring = Ring(plan)
    result = ring.ensure(tmp_path=tmp_path)
    assert not ring.commands("--busy")
    assert ("inspect",) not in ring.calls and not ring.commands("lldpcli")
    assert [rank for rank, argv in ring.changing() if "approve" in argv] == [0, 1, 2, 3]
    assert [rank for kind, rank in ring.events if kind == "start"] == [0, 1, 2, 3]
    assert all(current["hairpin"]["armed"] for current in result["nodes"])


def test_a_run_that_turns_offload_on_is_followed_by_reinspection(tmp_path):
    plan = ring_plan()
    plan["nodes"][2]["hairpin"] = document(plan, 2, [rule.OFFLOAD_OFF, rule.IN_EFFECT, rule.IN_EFFECT, rule.IN_EFFECT])
    ring = Ring(plan)
    record = {}
    result = ring.ensure(tmp_path=tmp_path, record=record)
    # No restart is needed, so no busy check; the run changed offload, so the
    # plan's inventory is read again before the caller verifies the network.
    assert not ring.commands("--busy") and [rank for kind, rank in ring.events if kind == "start"] == [2]
    assert ("inspect",) in ring.calls and record["reinspected"] is True
    assert result["nodes"][2]["hairpin"]["in_effect"]


def test_reinspection_follows_lldp_update_and_retries_missing_evidence(tmp_path):
    plan = needing(ring_plan(), [1])
    ring = Ring(plan)
    ring.lldp_failures = 2
    result = ring.ensure(tmp_path=tmp_path)
    first_inspect = ring.calls.index(("inspect",))
    lldp = [index for index, entry in enumerate(ring.calls) if len(entry) == 2 and entry[1] == ("lldpcli", "update")]
    assert sorted(ring.calls[index][0] for index in lldp[:4]) == [0, 1, 2, 3] and lldp[3] < first_inspect
    assert ring.calls.count(("rebuild",)) == 3
    assert result["nodes"][1]["hairpin"]["in_effect"]


def test_driver_drift_after_reinspection_needs_input(tmp_path):
    plan = needing(ring_plan(), [1])
    ring = Ring(plan)
    rebuild = ring.rebuild

    def drifted(found):
        value = rebuild(found)
        value["network"]["hosts"][1].update(driver_action="apply", hairpin=functions(plan, 1, [rule.DEFAULT] * 4))
        return value
    ring.rebuild = drifted
    with pytest.raises(NeedsInput, match="not in effect after the hairpin step"):
        ring.ensure(tmp_path=tmp_path)


def test_adoption_records_in_effect_sparks_and_reports_m19_for_the_others(tmp_path, capsys):
    plan = ring_plan()
    plan["nodes"][1]["hairpin"] = document(plan, 1, approved=False, armed=False)
    needing(plan, [2])
    ring = Ring(plan)
    record = {}
    result = ring.ensure(tmp_path=tmp_path, restart=False, record=record)
    assert [rank for kind, rank in ring.events if kind == "start"] == [1]
    assert not ring.commands("--busy") and not any(rank == 2 for rank, _ in ring.changing())
    assert ("The ConnectX hairpin setting is not in effect on rank 2 (spark2). Adoption recorded the existing fabric "
            "without it.") in capsys.readouterr().out
    assert record["state"] == "partial" and result["nodes"][1]["hairpin"]["armed"]


def test_kernel_option_that_disables_the_unit_never_starts_a_run(tmp_path):
    plan = needing(ring_plan(), [1])
    ring = Ring(plan)
    ring.sparks[1].status["boot_disabled_by_kernel_command_line"] = True
    with pytest.raises(NeedsInput, match="sparkring.hairpin=off"):
        ring.ensure(tmp_path=tmp_path)
    assert not ring.starts()


def test_not_in_effect_names_each_spark_and_value():
    plan = ring_plan()
    statuses = [document(plan, rank) for rank in range(4)]
    assert ring_module.not_in_effect(plan, statuses) is None
    statuses[2] = document(plan, 2, [rule.DEFAULT, rule.IN_EFFECT, rule.IN_EFFECT, rule.IN_EFFECT])
    statuses[3] = "root@192.0.2.13: Connection timed out"
    netdev = statuses[2]["functions"][0]["netdev"]
    # One line per Spark, then the remedy.
    assert ring_module.not_in_effect(plan, statuses).splitlines() == [
        f"rank 2 (spark2): the ConnectX hairpin setting is not in effect on 1 of 4 functions: {netdev}: "
        "hairpin_queue_size 1024, required 8192.",
        "rank 3 (root@192.0.2.13): cannot read its ConnectX hairpin status: root@192.0.2.13: Connection timed out.",
        "On Node A, sudo sparkring hairpin applies it after asking."]


# sparkring hairpin.

@pytest.fixture
def command(tmp_path, monkeypatch):
    """sparkring hairpin on a simulated Node A with a recorded four-Spark cluster."""
    from runtime.host import install_workflow
    from scripts import sparkring
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(controller, "STATE", tmp_path / "state")
    plan = ring_plan()
    ring = Ring(plan)
    cluster = {"name": "test", "plan": plan}
    node.save(controller.STATE, "cluster.json", cluster)
    monkeypatch.setattr(install_workflow, "require_head", lambda *a, **k: plan["nodes"][0]["node_id"])
    monkeypatch.setattr(install_workflow, "check_access", lambda value: None)
    monkeypatch.setattr(install_workflow, "refresh_cluster", lambda value: {**value, "plan": ring.plan})
    monkeypatch.setattr(ring_module, "remote", ring.invoke)
    monkeypatch.setattr(ring_module, "local", ring.run_local)
    monkeypatch.setattr(ring_module.sys.stdin, "isatty", lambda: False)
    # The command uses the real clock; simulated runs finish within a few polls.
    monkeypatch.setattr(ring_module, "POLL", 0)
    monkeypatch.setattr(ring_module, "DISPATCH_RETRY", 0)

    def run(*argv):
        return sparkring.main(["hairpin", *argv])
    return ring, run


def test_hairpin_plan_prints_the_result_and_changes_nothing(command, capsys):
    ring, run = command
    needing(ring.plan, [3])
    assert run("--plan", "--json") == 0
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert result["schema"] == "sparkring-hairpin-result/v1" and result["state"] == "planned"
    assert [row["before"] for row in result["ranks"]] == ["kept", "kept", "kept", "restart"]
    assert result["ranks"][3]["action"] == "restart" and len(result["ranks"][3]["functions"]) == 4
    assert "rank 3 spark3: restart 4 functions (1024 -> 8192)" in out.err
    assert ring.calls == []


def test_hairpin_without_yes_needs_approval(command, capsys):
    ring, run = command
    needing(ring.plan, [3])
    assert run("--json") == 3
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert result["field"] == "approval" and result["message"].startswith("This command applies the ConnectX hairpin")
    # The consent block is printed once, under the message.
    assert out.err.count("Four-Spark forwarding needs the ConnectX hairpin setting") == 1
    assert out.err.index(result["message"]) < out.err.index("  Four-Spark forwarding needs")
    assert ring.calls == []


def test_hairpin_with_yes_applies_and_lists_stopped_meshes(command, capsys):
    ring, run = command
    for rank in range(4):
        ring.plan["nodes"][rank]["hairpin"] = document(ring.plan, rank, approved=False, armed=False)
    ring.sparks = {rank: Spark(rank, copy.deepcopy(ring.plan["nodes"][rank]["hairpin"])) for rank in range(4)}
    assert run("--yes", "--json") == 0
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert result["state"] == "complete" and [row["after"] for row in result["ranks"]] == ["kept"] * 4
    assert json.loads(open(result["receipt"]).read())["schema"] == ring_module.RECEIPT_SCHEMA
    assert ring_module.COMPLETE in out.err
    # Only a Spark with no running mesh is listed, with its enabled mesh units;
    # the stopped units of Sparks whose mesh runs, and disabled units, are not.
    assert "No mesh service runs on these Sparks. Start the one that should serve:" in out.err
    assert "rank 3: sparkring-mesh.service: ssh -t root@192.0.2.13 sudo systemctl start sparkring-mesh.service" in out.err
    assert result["stopped_mesh_units"] == [{"rank": 3, "unit": "sparkring-glm-mesh.service"},
                                            {"rank": 3, "unit": "sparkring-mesh.service"}]
    assert "sparkring-old-mesh.service" not in out.err and "rank 0: sparkring-glm-mesh" not in out.err


def test_hairpin_revoke_defaults_to_no(command, capsys, monkeypatch):
    ring, run = command
    monkeypatch.setattr(ring_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(controller.sys.stdin, "isatty", lambda: True)
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "")
    assert run("--revoke") == 2
    assert prompts == ["Revoke the ConnectX hairpin approval on every Spark? [y/N]: "]
    assert ring.calls == [] and "Cancelled" in capsys.readouterr().err
    assert run("--revoke", "--yes", "--json") == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "complete" and [row["after"] for row in result["ranks"]] == ["revoked"] * 4
    assert sorted(rank for rank, argv in ring.changing() if "revoke" in argv) == [0, 1, 2, 3]


def test_hairpin_on_a_pair_or_without_a_ring(command, capsys, tmp_path):
    ring, run = command
    found = nodes(2)
    node.save(controller.STATE, "cluster.json", {"name": "test", "plan": topology.build_spec(found, found[0]["node_id"])})
    assert run("--json") == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "complete" and result["message"] == ring_module.PAIR
    (controller.STATE / "cluster.json").unlink()
    assert run("--json") == 3
    assert json.loads(capsys.readouterr().out)["field"] == "setup"
    assert ring.calls == []


def test_hairpin_failure_on_a_spark_exits_3_with_details(command, capsys):
    ring, run = command
    needing(ring.plan, [2])
    ring.sparks = {rank: Spark(rank, copy.deepcopy(ring.plan["nodes"][rank]["hairpin"])) for rank in range(4)}
    ring.sparks[2].outcome = "restart"
    assert run("--yes") == 3
    err = capsys.readouterr().err
    assert "did not complete on rank 2 (spark2)" in err
    assert "  Log: ssh -t root@192.0.2.12 sudo journalctl -u sparkring-hairpin.service -n 40 --no-pager" in err


# Safety and operator texts of the ring procedure.

def test_adoption_completes_with_an_m19_spark_and_an_offload_only_spark(tmp_path, capsys):
    plan = ring_plan()
    needing(plan, [3])
    needing(plan, [2], [rule.OFFLOAD_OFF] + [rule.IN_EFFECT] * 3)
    ring = Ring(plan)
    record = {}
    result = ring.ensure(tmp_path=tmp_path, restart=False, record=record)
    # Rank 2's run turned offload on, so the ring was re-inspected; rank 3 keeps its drift.
    assert ("inspect",) in ring.calls and record["reinspected"] is True
    assert result["network"]["hosts"][3]["driver_action"] == "apply"
    assert record["state"] == "partial" and "rank 3 (spark3)" in capsys.readouterr().out
    assert [rank for kind, rank in ring.events if kind == "start"] == [2]


def test_refused_mesh_units_start_only_after_every_spark_ran(tmp_path):
    plan = needing(ring_plan(), [0, 1, 2, 3])
    ring = Ring(plan)
    ring.sparks[0].blocked = ["sparkring-mesh.service"]
    record = {}
    ring.ensure(tmp_path=tmp_path, resume=True, record=record)
    assert ring.events[-1] == ("mesh-start", 0)
    assert [event for event in ring.events if event[0] != "mesh-start"] == [
        (kind, rank) for rank in (0, 1, 2, 3) for kind in ("start", "finish")]
    assert record["ranks"][0]["resumed"] == ["sparkring-mesh.service"]
    # Without resume, as for sparkring install, no mesh unit starts.
    again = Ring(needing(ring_plan(), [1]))
    again.sparks[0].blocked = ["sparkring-mesh.service"]
    again.ensure(tmp_path=tmp_path / "install")
    assert ("mesh-start", 0) not in again.events and not again.commands("resume")


def test_a_failed_run_starts_no_refused_mesh_unit(tmp_path):
    plan = needing(ring_plan(), [0, 1])
    ring = Ring(plan)
    ring.sparks[0].blocked = ["sparkring-mesh.service"]
    ring.sparks[1].outcome = "restart"
    with pytest.raises(NeedsInput):
        ring.ensure(tmp_path=tmp_path, resume=True)
    assert not ring.commands("resume") and ("mesh-start", 0) not in ring.events


def test_a_dispatch_retry_never_restarts_a_run_in_progress(tmp_path):
    plan = needing(ring_plan(single_uplink=True), [1])
    ring = Ring(plan)
    ring.sparks[1].polls = 5
    sent, pending = [], {"dispatched": False, "status_failures": 0}
    original = ring.invoke

    def invoke(host, argv, *, timeout):
        rank = ring.hosts[host]
        if rank == 1 and "start" in argv:
            sent.append(ring.sparks[1].status["unit"]["ActiveState"])
            result = original(host, argv, timeout=timeout)
            if not pending["dispatched"]:
                # The dispatch reached the Spark, but SSH reported an error.
                pending.update(dispatched=True, status_failures=1)
                raise RuntimeError("root@192.0.2.11: Connection to 10.253.255.2 closed by remote host.")
            return result
        if rank == 1 and pending["status_failures"] and "status" in argv:
            pending["status_failures"] -= 1
            raise RuntimeError("root@192.0.2.11: ssh: connect to host port 2222: Connection timed out")
        return original(host, argv, timeout=timeout)

    ring.invoke = invoke
    ring.ensure(tmp_path=tmp_path)
    # The retry found the run in progress and left it alone.
    assert sent == ["inactive", "activating"] and ring.sparks[1].runs == 1
    assert ring.sparks[1].status["in_effect"]


@pytest.mark.parametrize("failure", ["timeout", "error"])
def test_node_a_status_that_cannot_be_read_after_its_run_gives_m9(tmp_path, failure):
    import subprocess
    plan = needing(ring_plan(), [0])
    ring = Ring(plan)
    ring.sparks[0].outcome = "restart"
    original, state = ring.run_local, {"restarted": False}

    def run_local(argv, *, timeout):
        if argv[:2] == ["systemctl", "restart"]:
            state["restarted"] = True
        elif state["restarted"] and argv[:4] == [ring_module.SPARKRING, "node", "hairpin", "status"]:
            if failure == "timeout":
                raise subprocess.TimeoutExpired(argv, timeout)
            raise RuntimeError("/usr/bin/sparkring node hairpin status: devlink: Device or resource busy")
        return original(argv, timeout=timeout)

    ring.run_local = run_local
    with pytest.raises(NeedsInput) as caught:
        ring.ensure(tmp_path=tmp_path)
    assert caught.value.field == "driver"
    if failure == "timeout":
        assert str(caught.value).startswith(f"Rank 0 (spark0) did not report back within {ring_module.REPORT} s")
    else:
        assert str(caught.value).startswith(
            "The ConnectX hairpin setting did not complete on rank 0 (spark0): systemctl restart "
            "sparkring-hairpin.service: Job failed; its hairpin status cannot be read: /usr/bin/sparkring node "
            "hairpin status: devlink: Device or resource busy.")


def test_consent_for_outdated_workers_that_show_the_setting_announces_no_restart():
    plan = ring_plan(single_uplink=True)
    plan["nodes"][0]["hairpin"] = document(plan, 0, approved=False, armed=False)
    for rank in (1, 2, 3):
        del plan["nodes"][rank]["hairpin"]
        plan["nodes"][rank]["revision"] = OLDER
        plan["network"]["hosts"][rank]["hairpin"] = functions(plan, rank, [rule.UNKNOWN] * 4)
        for row in plan["network"]["hosts"][rank]["hairpin"]:
            row["values"] = dict(rule.PARAMETERS)
    rows = ring_module.requirement(plan)
    assert [row["state"] for row in rows] == ["record", "update", "update", "update"]
    assert not ring_module.restart_expected(rows)
    lines = ring_module.consent_lines(plan, rows)
    assert lines[:3] == [
        "Four-Spark forwarding needs the ConnectX hairpin setting (hairpin_queue_size 8192):",
        "  rank 0 spark0: in effect; record it and apply it at every boot (no restart)",
        "  ranks 1-3: update SparkRing to aaaaaaaaaaaa first; they show hairpin_queue_size 8192, so SparkRing "
        "expects only to record it"]
    text = "\n".join(lines)
    assert "No driver restarts now." in text and "Each restart takes" not in text
    assert ring_module.NOTICE_SINGLE_UPLINK not in text
    assert ring_module.consent_default(plan, rows) is True
    assert ring_module.m7(rows, command=True).startswith(
        "This command applies the ConnectX hairpin setting without a driver restart: ranks 1-3 are updated to this "
        "SparkRing revision first; rank 0 only records it.")
    # Rows that do not show the setting may need a restart after the update.
    plan["network"]["hosts"][3]["hairpin"][0]["values"]["hairpin_queue_size"] = 1024
    rows = ring_module.requirement(plan)
    assert ring_module.restart_expected(rows) and ring_module.consent_default(plan, rows) is False
    assert "Each restart takes one function's link down" in "\n".join(ring_module.consent_lines(plan, rows))
    assert "  rank 3: update SparkRing to aaaaaaaaaaaa first" in ring_module.consent_lines(plan, rows)


def test_offload_only_consent_announces_no_restart():
    plan = ring_plan()
    needing(plan, [2], [rule.OFFLOAD_OFF] + [rule.IN_EFFECT] * 3)
    rows = ring_module.requirement(plan)
    lines = ring_module.consent_lines(plan, rows)
    assert lines[1] == ("  rank 2 spark2: turn on hw-tc-offload on 1 function; record it and apply it at every "
                        "boot (no restart)")
    assert lines[2:] == ring_module.NO_RESTART
    assert ring_module.m7(rows) == (
        "This installation also applies the ConnectX hairpin setting without a driver restart: rank 2 turns on "
        "hardware TC offload. SparkRing then applies it at every boot before networking starts (about 30 seconds "
        "per boot). Review with --plan, then repeat with --yes.")


def test_an_update_that_reveals_a_restart_stops_for_approval(tmp_path):
    plan = ring_plan()
    del plan["nodes"][2]["hairpin"]
    plan["nodes"][2]["revision"] = OLDER
    ring = Ring(plan)
    original = ring.update

    def update():
        original()
        ring.sparks[2].status = document(plan, 2, rule.DEFAULT, approved=False, armed=False)
    ring.update = update
    with pytest.raises(NeedsInput) as caught:
        ring.ensure(tmp_path=tmp_path, restart_approved=False)
    assert caught.value.field == "approval"
    assert str(caught.value).startswith("SparkRing was updated on rank 2. Rank 2 then needs ConnectX driver "
                                        "restarts, which the approval did not cover, so no ConnectX driver was "
                                        "restarted.")
    assert "  rank 2 spark2: restart 4 functions (1024 -> 8192)" in caught.value.details["lines"]
    assert not ring.commands("--busy") and not ring.starts() and not ring.commands("approve")


def test_m8_after_a_worker_update_says_what_changed(tmp_path):
    plan = ring_plan()
    del plan["nodes"][2]["hairpin"]
    plan["nodes"][2]["revision"] = OLDER
    needing(plan, [3])
    ring = Ring(plan)
    ring.sparks[1].busy = [{"kind": "gpu", "pid": "900", "detail": "GPU compute process PID 900"}]
    with pytest.raises(NeedsInput) as caught:
        ring.ensure(tmp_path=tmp_path)
    assert str(caught.value).startswith("SparkRing was updated on rank 2. The ConnectX hairpin setting must be "
                                        "applied on 1 Spark, and 1 Spark is in use. No ConnectX driver was "
                                        "restarted.")
    assert "Nothing has been changed" not in str(caught.value)


def test_a_suspended_spark_names_the_failed_restart_in_its_consent_line():
    plan = ring_plan()
    record = {"boot_id": "b1", "pci_address": "0000:01:00.0", "netdev": "enp1s0f0np0", "state": "failed",
              "cause": "driver restart failed: Device or resource busy", "time": 1.0}
    plan["nodes"][2]["hairpin"] = document(plan, 2, rule.DEFAULT, suspended=record)
    line = next(line for line in ring_module.consent_lines(plan, ring_module.requirement(plan)) if "rank 2" in line)
    assert line == ("  rank 2 spark2: restart 4 functions (1024 -> 8192); boot restarts suspended since enp1s0f0np0 "
                    "failed to restart at 1970-01-01 00:00:01 UTC (driver restart failed: Device or resource busy)")


@pytest.mark.parametrize("state,expected", [
    ("restart", "not in effect ({}: hairpin_queue_size 1024, required 8192); adoption does not restart it; sudo "
                "sparkring hairpin applies it afterwards"),
    ("record", "in effect; record it and apply it at every boot (no restart)")])
def test_adoption_summary_lines_announce_no_restart(state, expected):
    plan = ring_plan()
    if state == "restart":
        needing(plan, [1])
    else:
        plan["nodes"][1]["hairpin"] = document(plan, 1, approved=False)
    row = ring_module.requirement(plan)[1]
    netdevs = ", ".join(f["netdev"] for f in row["functions"])
    assert ring_module.summary_line(row, adopt=True) == expected.format(netdevs)


def test_a_failed_child_facing_restart_advises_rebooting_that_spark(tmp_path):
    plan = needing(ring_plan(single_uplink=True), [1])
    ring = Ring(plan)
    ring.sparks[1].outcome = "cut"
    with pytest.raises(NeedsInput) as caught:
        ring.ensure(tmp_path=tmp_path)
    message = str(caught.value)
    assert ("The Sparks behind enp1s0f0np0 are cut off from the administration network, and this command needs "
            "them: reboot rank 1 (sudo ssh 10.253.255.2 systemctl reboot; power-cycle it if the reboot hangs); its "
            "next boot restarts no ConnectX function. Then run sudo sparkring hairpin.") in message
    assert "it starts without restarting any function and stays reachable" not in message
    head = Ring(needing(ring_plan(), [0]))
    head.sparks[0].outcome = "cut"
    with pytest.raises(NeedsInput) as caught:
        head.ensure(tmp_path=tmp_path / "head")
    assert "The workers behind enp1s0f0np0 are cut off" in str(caught.value)
    assert "reboot Node A (sudo systemctl reboot;" in str(caught.value)


def test_messages_that_advise_revoking_give_the_command_for_node_a(tmp_path):
    plan = needing(ring_plan(single_uplink=True), [2])
    ring = Ring(plan)
    ring.sparks[2].outcome = "M14"
    with pytest.raises(NeedsInput) as caught:
        ring.ensure(tmp_path=tmp_path)
    assert "On Node A: sudo ssh 10.253.255.4 /usr/bin/sparkring node hairpin revoke" in caught.value.details["lines"]
    refused = Ring(needing(ring_plan(), [3]))
    original = refused.invoke

    def invoke(host, argv, *, timeout):
        if "approve" in argv:
            raise RuntimeError(f"{host}: SparkRing node: This Spark already approves the hairpin setting for other "
                               "ConnectX functions (cw_primary: mac x instead of y). If its card was replaced, run "
                               "sudo sparkring node hairpin revoke on it, then repeat.")
        return original(host, argv, timeout=timeout)
    refused.invoke = invoke
    with pytest.raises(NeedsInput) as caught:
        refused.ensure(tmp_path=tmp_path / "approve")
    assert caught.value.details["lines"] == [
        "On Node A: ssh -t root@192.0.2.13 sudo /usr/bin/sparkring node hairpin revoke"]
    # Other approval failures carry no revoke command; their details stay printable per key.
    unreachable = Ring(needing(ring_plan(), [3]))
    original = unreachable.invoke

    def fail(host, argv, *, timeout):
        if "approve" in argv:
            raise RuntimeError(f"{host}: ssh: connect to host port 22: Connection timed out")
        return original(host, argv, timeout=timeout)
    unreachable.invoke = fail
    with pytest.raises(NeedsInput) as caught:
        unreachable.ensure(tmp_path=tmp_path / "unreachable")
    assert "lines" not in caught.value.details
    assert "error: root@192.0.2.13: ssh: connect to host port 22: Connection timed out" in controller.detail_lines(
        caught.value.details)


def test_hairpin_refuses_unknown_statistics_before_asking(command, capsys, monkeypatch):
    ring, run = command
    ring.plan["nodes"][2]["hairpin"] = document(ring.plan, 2, [rule.UNKNOWN] + [rule.IN_EFFECT] * 3)
    monkeypatch.setattr(ring_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("asked before refusing unknown statistics"))
    assert run() == 3
    assert "devlink reload statistics are unavailable" in capsys.readouterr().err
    assert ring.calls == []


def test_hairpin_revoke_plan_changes_nothing(command, capsys):
    ring, run = command
    assert run("--revoke", "--plan") == 0
    assert "Plan only; nothing was changed. Repeat without --plan to revoke it." in capsys.readouterr().err
    assert ring.calls == []


def test_hairpin_holds_the_installation_lock(command, capsys):
    from runtime.common import process_lock
    ring, run = command
    needing(ring.plan, [3])
    with process_lock.hold(controller.STATE / "install.lock"):
        assert run("--yes") == 2
    assert "Another operation is active" in capsys.readouterr().err
    assert ring.calls == []


def test_hairpin_interrupted_leaves_an_interrupted_receipt(command, capsys):
    ring, run = command
    needing(ring.plan, [3])
    ring.sparks = {rank: Spark(rank, copy.deepcopy(ring.plan["nodes"][rank]["hairpin"])) for rank in range(4)}
    original = ring.handle

    def handle(rank, argv):
        if rank == 3 and argv[3:4] == ["start"]:
            raise KeyboardInterrupt
        return original(rank, argv)
    ring.handle = handle
    assert run("--yes", "--json") == 2
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert result["state"] == "failed" and result["message"] == "Interrupted; repeat sudo sparkring hairpin to continue."
    [receipt] = (controller.STATE / "hairpin").glob("*/hairpin.json")
    assert json.loads(receipt.read_text())["state"] == "interrupted"
    assert "A hairpin run that already started on a Spark finishes by itself" in out.err
