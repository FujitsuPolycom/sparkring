"""The ConnectX hairpin setting across a four-Spark ring, driven from Node A.

Status: implemented. No hardware evidence yet for a live run over the
administration network.

Four-Spark rings relay traffic between nonadjacent Sparks through ConnectX
hardware forwarding, which needs the hairpin setting on every fabric function
(``scripts/hairpin_setting.py`` holds the values and the in-effect rule). Each
Spark applies it with its own ``sparkring-hairpin.service``
(``runtime/host/hairpin.py``): a run of that unit restarts the functions that
need it, and the Spark enables the unit (it is then *armed*) after a run in
which every restart succeeded, so every later boot applies the setting again
before networking starts. A Spark's ``/etc/sparkring/hairpin.json`` records the
operator's approval.

This module works on the ring as a whole:

- ``requirement(plan)`` classifies every Spark of a four-Spark plan from the
  ``hairpin`` status in its inspect document and its package revision:
  ``kept`` (approved, armed, in effect, boot runs not suspended), ``record``
  (no restart needed, but an approval, arming or a suspension must be
  recorded by a unit run), ``restart`` (a function needs a driver restart),
  ``update`` (the Spark runs another SparkRing revision than Node A and is
  read again after its package is updated) and ``unknown`` (reload statistics
  unavailable; nothing is restarted).
- ``ensure(plan, ...)`` brings every Spark to ``kept``: consent, package
  updates, one idle check before any restart, approvals, then one unit run on
  Node A and then on one worker at a time, and a re-inspection of the ring.
  It stops at the first failure, so later Sparks stay untouched. Only after
  every run succeeded does it start, on request, the mesh units that the
  start check refused, so no mesh runs while a Spark restarts functions.
- ``main`` is ``sudo sparkring hairpin`` on Node A.

Worker order: on a ring whose workers SparkRing reaches only through the
administration network (``sr-control``), workers run in ascending control
address, which is the breadth-first order of the administration tree. Every
Spark between Node A and a worker has then finished its run before that worker
is dispatched, and one Spark runs at a time, so no dispatch or poll crosses a
function that another Spark is restarting. A worker's own run briefly cuts
the path to the Sparks behind its child-facing functions; they have not run
yet. Elsewhere workers run in rank order.
"""
import argparse
import concurrent.futures
import contextlib
import copy
import ipaddress
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from runtime.common import installer, process_lock
from runtime.host import control, discovery, hairpin, node, progress
from runtime.host.install_errors import NeedsInput
from scripts import hairpin_setting as rule

UNIT = "sparkring-hairpin.service"
SPARKRING = "/usr/bin/sparkring"
STATUS_SCHEMA = "sparkring-hairpin-status/v1"
RECEIPT_SCHEMA = "sparkring-hairpin-receipt/v1"
RESULT_SCHEMA = "sparkring-hairpin-result/v1"

KEPT, RECORD, RESTART, UPDATE, UNKNOWN = "kept", "record", "restart", "update", "unknown"
ACTIONS = {KEPT: "none", RECORD: "record", RESTART: "restart", UPDATE: "update", UNKNOWN: "refuse"}

# Time limits in seconds. SSH calls that dispatch or poll a worker's run use
# keep-alive options and SSH_LIMIT, so a Spark whose link is down during its
# own restart costs at most SSH_LIMIT per attempt.
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=5",
               "-o", "ServerAliveCountMax=3")
SSH_LIMIT = 20
STATUS_LIMIT = 90
DISPATCH_RETRY = 5
DISPATCH_WINDOW = 60
POLL = 5
# sparkring-hairpin.service allows 330 s to start; a run that its start timeout
# stops ends within its 60 s stop timeout, so every run reports within this.
REPORT = 400
REINSPECT = 60
REINSPECT_RETRY = 5
# Errors that a re-inspection shortly after a driver restart can show while
# LLDP and the RDMA devices come back: topology's LLDP and RDMA checks, and the
# node inventory's RDMA function count.
TRANSIENT = ("Missing reciprocal LLDP cable evidence", "missing verified RDMA function",
             "four distinct RDMA functions are required")

COMPLETE = "ConnectX hairpin setting: in effect on 4 Sparks and applied at every boot."
PAIR = "pairs do not use the ConnectX hairpin setting"
NOTICE_SINGLE_UPLINK = ("A worker reached only through the ring cables stays unreachable if one of its restarts "
                        "fails, until it is power-cycled.")
NO_RESTART = ["No driver restarts now. SparkRing records the setting and applies it at every boot",
              "before networking starts (about 30 seconds per boot)."]
RESTART_TRAILER = ["Each restart takes one function's link down for about 8 seconds: about 30 seconds",
                   "per Spark, one Spark at a time, Node A first. SparkRing checks that no model, mesh",
                   "service or RDMA program is running and restarts nothing if one is.",
                   "sparkring-hairpin.service then repeats the restarts at every boot before networking",
                   "starts, adding about 30 seconds to each boot."]
MESH_UNITS = ("sparkring-mesh.service", "sparkring-*-mesh.service")
REVOKE_ADVICE = "sparkring node hairpin revoke"


# Commands on the ring's Sparks.

def remote(host, argv, *, timeout):
    """Run argv on a worker over SSH with keep-alive options; return stdout, raise on failure."""
    result = subprocess.run(["ssh", *SSH_OPTIONS, discovery.target(host), shlex.join(argv)],
                            capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{host}: " + (_last_line(result.stderr) or f"exit status {result.returncode}"))
    return result.stdout


def local(argv, *, timeout):
    """Run argv on Node A as root; return stdout, raise on failure."""
    root = hasattr(os, "geteuid") and os.geteuid() == 0
    result = subprocess.run(list(argv) if root else ["sudo", "-n", *argv], capture_output=True, text=True,
                            encoding="utf-8", timeout=timeout)
    if result.returncode:
        raise RuntimeError(" ".join(argv[:4]) + ": " + (_last_line(result.stderr) or f"exit status {result.returncode}"))
    return result.stdout


def _last_line(text):
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


FAILURES = (RuntimeError, OSError, ValueError, subprocess.SubprocessError)


class Access:
    """Commands on the ring's Sparks: Node A locally, workers over SSH with ``sudo -n``."""

    def __init__(self, plan, *, invoke=None, run_local=None):
        self.hosts = plan["spec"]["hosts"]
        self.invoke = invoke or remote
        self.run_local = run_local or local

    def run(self, rank, argv, *, timeout):
        if rank == 0:
            return self.run_local(list(argv), timeout=timeout)
        return self.invoke(self.hosts[rank]["host"], ["sudo", "-n", *argv], timeout=timeout)

    def node(self, rank, argv, *, timeout):
        return self.run(rank, [SPARKRING, "node", "hairpin", *argv], timeout=timeout)

    def status(self, rank, *, busy=False, timeout=STATUS_LIMIT):
        document = json.loads(self.node(rank, ["status", *(["--busy"] if busy else [])], timeout=timeout))
        if not isinstance(document, dict) or document.get("schema") != STATUS_SCHEMA:
            raise ValueError(f"rank {rank}: unexpected hairpin status document")
        return document


def single_uplink(plan):
    """Whether SparkRing reaches the workers only through the administration network."""
    return any(host.get("management_netdev") == control.INTERFACE for host in plan["spec"]["hosts"][1:])


def worker_order(plan):
    """Workers in administration-tree order on single-uplink rings, otherwise in rank order."""
    hosts = plan["spec"]["hosts"]
    workers = list(range(1, len(hosts)))
    if single_uplink(plan):
        return sorted(workers, key=lambda rank: ipaddress.IPv4Address(hosts[rank]["management_address"]))
    return workers


def command_text(plan, rank, argv):
    """A command for one Spark in a form that runs on Node A."""
    words = shlex.join(argv)
    if rank == 0:
        return "sudo " + words
    host = plan["spec"]["hosts"][rank]
    if host.get("management_netdev") == control.INTERFACE:
        # Root's SSH configuration on Node A names each control address.
        return f"sudo ssh {host['management_address']} {words}"
    return f"ssh -t {host['host']} sudo {words}"


def revoke_lines(plan, rank, text):
    """The Node A form of the revoke command when a Spark's message recommends it; otherwise no line."""
    if REVOKE_ADVICE not in str(text):
        return []
    return ["On Node A: " + command_text(plan, rank, [SPARKRING, "node", "hairpin", "revoke"])]


def _short(revision):
    return revision[:12] if isinstance(revision, str) and revision else "of unknown revision"


def ranks_text(ranks):
    ranks = sorted(ranks)
    if len(ranks) == 1:
        return f"rank {ranks[0]}"
    if ranks == list(range(ranks[0], ranks[-1] + 1)):
        return f"ranks {ranks[0]}-{ranks[-1]}"
    return "ranks " + ", ".join(str(rank) for rank in ranks)


def _verb(ranks, singular, plural):
    return singular if len(ranks) == 1 else plural


def _clause(text):
    return " ".join(str(text).split()).rstrip(". ")


def _sentence(text):
    text = str(text).strip()
    return text if text.endswith((".", "!", "?")) else text + "."


# Classification.

def classify(document, revision, head):
    """One Spark's state from its hairpin status document and package revision."""
    if not isinstance(document, dict) or document.get("schema") != STATUS_SCHEMA or revision != head:
        return UPDATE
    functions = document.get("functions")
    if (not isinstance(functions, list) or len(functions) != 4
            or any(not isinstance(row, dict) or row.get("state") not in rule.STATES or row.get("state") == rule.UNKNOWN
                   for row in functions)):
        return UNKNOWN
    if any(row["state"] in rule.RESTART_STATES for row in functions):
        return RESTART
    approval = document.get("approval") or {}
    if (document.get("in_effect") is True and approval.get("valid") is True and document.get("armed") is True
            and not document.get("suspended")):
        return KEPT
    return RECORD


def _function_rows(rows):
    keys = ("role", "netdev", "pci_address", "rdma_device", "values", "driver_reinit", "reload_failed", "offload", "state")
    return [{key: row.get(key) for key in keys if key in row} for row in rows or [] if isinstance(row, dict)]


def _row(rank, host, hostname, document, revision, head, fallback=None):
    state = classify(document, revision, head)
    present = isinstance(document, dict) and document.get("schema") == STATUS_SCHEMA
    functions = document.get("functions") if present else fallback
    return {"rank": rank, "host": host["host"],
            "hostname": hostname or (document.get("hostname") if present else None) or host["host"],
            "state": state, "revision": revision,
            "approved": bool(present and (document.get("approval") or {}).get("valid")),
            "armed": bool(present and document.get("armed")),
            "suspended": document.get("suspended") if present else None,
            "in_effect": bool(present and document.get("in_effect")),
            "functions": _function_rows(functions), "status": document if present else None}


def requirement(plan):
    """Classify every Spark of a four-Spark plan; a pair gives an empty list.

    Reads the ``hairpin`` status of each inspect document in ``plan["nodes"]``
    and each Spark's package revision against Node A's. A Spark without a
    status document runs a SparkRing revision that predates the status, so it
    is ``update``. For such a Spark the function rows come from the network
    plan's ``hairpin`` rows.
    """
    nodes = plan["nodes"]
    if len(nodes) != 4:
        return []
    head = nodes[0].get("revision")
    planned = plan.get("network", {}).get("hosts") or [{}] * 4
    return [_row(rank, host, current.get("hostname"), current.get("hairpin"), current.get("revision"), head,
                 fallback=planned[rank].get("hairpin"))
            for rank, (current, host) in enumerate(zip(nodes, plan["spec"]["hosts"], strict=True))]


def required(rows):
    return any(row["state"] != KEPT for row in rows)


def expects_record(row):
    """Whether an outdated Spark's planner rows show the setting's values with offload on for all four functions.

    Such a Spark most likely needs only its setting recorded after its package
    update. The rows cannot show whether a restart applied the values, so the
    procedure reads the Spark again after the update and stops for approval
    when it then needs a restart.
    """
    functions = row["functions"]
    return (row["state"] == UPDATE and len(functions) == 4
            and all((function.get("values") or {}) == rule.PARAMETERS and function.get("offload") == rule.OFFLOAD_ON
                    for function in functions))


def restart_expected(rows):
    """Whether the step may restart a driver: a Spark needs a restart, or an outdated one may need one."""
    return any(row["state"] == RESTART or (row["state"] == UPDATE and not expects_record(row)) for row in rows)


def refuse_unknown(rows):
    """Raise M16 (field ``driver``) for Sparks whose reload statistics are unavailable; nothing is restarted."""
    unknown = [row for row in rows if row["state"] == UNKNOWN]
    if unknown:
        raise NeedsInput(" ".join(m16(row) for row in unknown), field="driver",
                         details={"ranks": [row["rank"] for row in unknown]})


def _restart_detail(functions):
    parts = []
    default = [row for row in functions if row.get("state") == rule.DEFAULT]
    sizes = sorted({(row.get("values") or {}).get("hairpin_queue_size") for row in default}
                   - {rule.HAIRPIN_QUEUE_SIZE, None})
    if sizes:
        parts.append(", ".join(str(size) for size in sizes) + f" -> {rule.HAIRPIN_QUEUE_SIZE}")
    queues = sorted({(row.get("values") or {}).get("hairpin_num_queues") for row in default}
                    - {rule.HAIRPIN_NUM_QUEUES, None})
    if queues:
        parts.append("hairpin_num_queues " + ", ".join(str(value) for value in queues)
                     + f" -> {rule.HAIRPIN_NUM_QUEUES}")
    if any(row.get("state") == rule.PENDING for row in functions):
        parts.append(f"{rule.HAIRPIN_QUEUE_SIZE} set but not yet in effect")
    if any(row.get("state") == rule.FAILED for row in functions):
        parts.append("the last driver restart failed")
    return "; ".join(parts) or f"-> {rule.HAIRPIN_QUEUE_SIZE}"


def _count(number, noun):
    return f"{number} {noun}" + ("" if number == 1 else "s")


def _suspension(row):
    """Why boot restarts are suspended on a Spark, as a clause; empty when they are not."""
    record = row.get("suspended")
    if not isinstance(record, dict):
        return ""
    return "boot restarts suspended since " + hairpin.suspension_text(record)


def _record_text(row):
    offload = [f for f in row["functions"] if f.get("state") == rule.OFFLOAD_OFF]
    if offload:
        return (f"turn on hw-tc-offload on {_count(len(offload), 'function')}; record it and apply it at every "
                "boot (no restart)")
    if row["suspended"]:
        return f"in effect; {_suspension(row)}; record the success and apply it at every boot (no restart)"
    return "in effect; record it and apply it at every boot (no restart)"


def summary_line(row, *, adopt=False):
    """The ``ConnectX hairpin:`` text of one rank in ``controller.summarize``.

    With ``adopt``, a Spark that needs a restart is shown as adoption leaves
    it: adoption restarts no function.
    """
    if row["state"] == UPDATE:
        return "read after updating SparkRing on this Spark"
    if row["state"] == UNKNOWN:
        return "devlink reload statistics unavailable; SparkRing restarts nothing"
    if adopt and row["state"] == RESTART:
        failing = [f for f in row["functions"] if f.get("state") != rule.IN_EFFECT]
        return (f"not in effect ({rule.grouped(failing)}); adoption does not restart it; sudo sparkring hairpin "
                "applies it afterwards")
    if row["suspended"] and row["state"] != KEPT:
        return "boot restarts suspended after a failed restart; retry live"
    if row["state"] == KEPT:
        return "in effect; applied at every boot"
    if row["state"] == RESTART:
        restart = [f for f in row["functions"] if f.get("state") in rule.RESTART_STATES]
        return (f"restart {_count(len(restart), 'function')} after addressing ({_restart_detail(restart)}), "
                "about 8 s link loss each")
    return _record_text(row)


def _rank_line(row):
    if row["state"] == RESTART:
        restart = [f for f in row["functions"] if f.get("state") in rule.RESTART_STATES]
        suspension = _suspension(row)
        return (f"restart {_count(len(restart), 'function')} ({_restart_detail(restart)})"
                + (f"; {suspension}" if suspension else ""))
    if row["state"] == UNKNOWN:
        return "devlink reload statistics are unavailable; SparkRing restarts nothing there"
    return _record_text(row)


def record_only(rows):
    """Whether every Spark that is not kept only needs its setting recorded, and all are in effect."""
    return all(row["state"] in (KEPT, RECORD) and row["in_effect"] for row in rows)


def consent_lines(plan, rows):
    """The installed-ring consent text; empty when every Spark is kept.

    The restart variant lists the driver restarts and appears only when a
    Spark needs one, or when an outdated Spark's rows do not show the setting
    (``restart_expected``). Otherwise the text says that no driver restarts:
    the record-only text when every Spark is in effect, and a per-rank list
    when hardware TC offload is turned on or outdated Sparks are updated first.
    """
    if not required(rows):
        return []
    header = f"Four-Spark forwarding needs the ConnectX hairpin setting (hairpin_queue_size {rule.HAIRPIN_QUEUE_SIZE})"
    if record_only(rows):
        return [header + ".",
                "It is in effect on all 4 Sparks. SparkRing records it and applies it at every boot",
                "before networking starts (about 30 seconds per boot). No driver restarts now."]
    lines = [header + ":"]
    for row in rows:
        if row["state"] not in (KEPT, UPDATE):
            lines.append(f"  rank {row['rank']} {row['hostname']}: {_rank_line(row)}")
    revision = _short(plan["nodes"][0].get("revision"))
    updates = [row for row in rows if row["state"] == UPDATE]
    expected = [row["rank"] for row in updates if expects_record(row)]
    others = [row["rank"] for row in updates if not expects_record(row)]
    if expected:
        lines.append(f"  {ranks_text(expected)}: update SparkRing to {revision} first; "
                     f"{_verb(expected, 'it shows', 'they show')} hairpin_queue_size {rule.HAIRPIN_QUEUE_SIZE}, "
                     "so SparkRing expects only to record it")
    if others:
        lines.append(f"  {ranks_text(others)}: update SparkRing to {revision} first")
    if restart_expected(rows):
        lines += RESTART_TRAILER
        if single_uplink(plan):
            lines.append(NOTICE_SINGLE_UPLINK)
    else:
        lines += NO_RESTART
        if expected:
            lines.append("If an updated Spark then needs a driver restart, SparkRing stops and asks for it first.")
    return lines


def consent_default(plan, rows):
    """Default answer: No when a restart may be needed on a single-uplink ring, Yes otherwise."""
    return not (single_uplink(plan) and restart_expected(rows))


def m7(rows, *, command=False):
    """The approval request when consent is missing; only Sparks that may restart a driver are said to."""
    subject = "This command" if command else "This installation"
    verb = "applies" if command else "also applies"
    if record_only(rows):
        return (f"{subject} records the ConnectX hairpin setting, already in effect on all 4 Sparks, and applies "
                "it at every boot (about 30 seconds per boot). No driver restarts now. Review with --plan, then "
                "repeat with --yes.")
    restart = [row["rank"] for row in rows if row["state"] == RESTART]
    possible = [row["rank"] for row in rows if row["state"] == UPDATE and not expects_record(row)]
    updated = [row["rank"] for row in rows if row["state"] == UPDATE and expects_record(row)]
    offload = [row["rank"] for row in rows if row["state"] == RECORD
               and any(f.get("state") == rule.OFFLOAD_OFF for f in row["functions"])]
    record = [row["rank"] for row in rows if row["state"] == RECORD and row["rank"] not in offload]
    parts = []
    if restart:
        parts.append(f"each function of {ranks_text(restart)} that needs it restarts its driver once (link down "
                     "about 8 seconds, about 30 seconds per Spark)")
    if possible:
        parts.append(f"{ranks_text(possible)} {_verb(possible, 'is', 'are')} updated to this SparkRing revision "
                     f"first, then {_verb(possible, 'restarts', 'restart')} the functions that need it")
    if updated:
        parts.append(f"{ranks_text(updated)} {_verb(updated, 'is', 'are')} updated to this SparkRing revision "
                     "first")
    if offload:
        parts.append(f"{ranks_text(offload)} {_verb(offload, 'turns', 'turn')} on hardware TC offload")
    if record:
        parts.append(f"{ranks_text(record)} only {_verb(record, 'records', 'record')} it")
    if restart or possible:
        return (f"{subject} {verb} the ConnectX hairpin setting: " + "; ".join(parts) + ". SparkRing then repeats "
                "it at every boot before networking starts. Review with --plan, then repeat with --yes.")
    return (f"{subject} {verb} the ConnectX hairpin setting without a driver restart: " + "; ".join(parts)
            + ". SparkRing then applies it at every boot before networking starts (about 30 seconds per boot)"
            + ("; if an updated Spark then needs a driver restart, it stops and asks for it first" if updated else "")
            + ". Review with --plan, then repeat with --yes.")


def m7_updated(rows, updated):
    """The approval request when Sparks need driver restarts that the approval did not cover, after their update."""
    restarting = [row["rank"] for row in rows if row["state"] == RESTART]
    return (f"SparkRing was updated on {ranks_text(updated)}. {ranks_text(restarting).capitalize()} then "
            f"{_verb(restarting, 'needs', 'need')} ConnectX driver restarts, which the approval did not cover, so no "
            "ConnectX driver was restarted. Each function that needs it restarts its driver once (link down about 8 "
            "seconds, about 30 seconds per Spark), and SparkRing repeats it at every boot before networking starts. "
            "Review with --plan, then repeat with --yes.")


def m15_plan(row, head):
    return (f"Rank {row['rank']} ({row['hostname']}) runs SparkRing {_short(row['revision'])}; its hairpin state is "
            f"read after the update to {_short(head)}.")


def m16(row):
    netdevs = ", ".join(f.get("netdev") or "?" for f in row["functions"] if f.get("state") in (rule.UNKNOWN, None)) \
        or "its fabric functions"
    return (f"Rank {row['rank']} ({row['hostname']}): devlink reload statistics are unavailable for {netdevs}; "
            "SparkRing cannot tell whether the hairpin setting is in effect and restarts nothing.")


def m19(row):
    return (f"The ConnectX hairpin setting is not in effect on rank {row['rank']} ({row['hostname']}). Adoption "
            "recorded the existing fabric without it. To apply it: stop the mesh service on every Spark, then run "
            "sudo sparkring hairpin.")


def mesh_hint(rows):
    """Text to append to a native-mesh refusal while Sparks lack the setting or run an older SparkRing.

    An older SparkRing reports only a running mesh, so a mesh that stopped
    when a cabled Spark restarted looks absent until ``sparkring hairpin``,
    which updates outdated Sparks, has run. Empty otherwise.
    """
    ranks = [row["rank"] for row in rows if row["state"] == RESTART]
    if ranks:
        return (f" The ConnectX hairpin setting is not in effect on {ranks_text(ranks)}, so "
                f"{_verb(ranks, 'its mesh service', 'their mesh services')} cannot start; run sudo sparkring hairpin "
                "on Node A first.")
    outdated = [row["rank"] for row in rows if row["state"] == UPDATE]
    if outdated:
        text = ranks_text(outdated)
        return (f" {text[0].upper() + text[1:]} {_verb(outdated, 'runs', 'run')} an older SparkRing, which does not "
                "report a stopped mesh service; run sudo sparkring hairpin on Node A to update "
                f"{_verb(outdated, 'it', 'them')}, then repeat the installation.")
    return ""


def rank_rows(rows, plan=None):
    """Per-rank result rows (no status documents), with the M15 plan text for outdated Sparks."""
    head = plan["nodes"][0].get("revision") if plan else None
    result = []
    for row in rows:
        entry = {"rank": row["rank"], "host": row["host"], "hostname": row["hostname"], "before": row["state"],
                 "action": ACTIONS[row["state"]], "functions": row["functions"], "after": None, "error": None}
        if row["state"] == UPDATE and plan:
            entry["message"] = m15_plan(row, head)
        if row["state"] == UNKNOWN:
            entry["message"] = m16(row)
        result.append(entry)
    return result


def not_in_effect(plan, statuses):
    """M6 for ``sparkring up``: one line per Spark that lacks the setting, then the remedy; None when all have it.

    ``statuses`` holds each Spark's status document or the error that
    prevented reading it.
    """
    lines = []
    for rank, value in enumerate(statuses):
        host = plan["spec"]["hosts"][rank]
        if not isinstance(value, dict):
            lines.append(f"rank {rank} ({host['host']}): cannot read its ConnectX hairpin status: {value}.")
            continue
        name = f"rank {rank} ({value.get('hostname') or host['host']})"
        functions = [row for row in value.get("functions") or [] if isinstance(row, dict)]
        failing = [row for row in functions if row.get("state") != rule.IN_EFFECT]
        if len(functions) != 4 or failing:
            detail = rule.grouped(failing) if failing else "its functions cannot be read"
            lines.append(f"{name}: the ConnectX hairpin setting is not in effect on {len(failing) or 4} of 4 "
                         f"functions: {detail}.")
    if not lines:
        return None
    return "\n".join(lines + ["On Node A, sudo sparkring hairpin applies it after asking."])


def read_statuses(plan, *, invoke=None):
    """Each Spark's hairpin status document, or the error that prevented reading it (for ``sparkring up``)."""
    invoke = invoke or discovery.ssh
    statuses = []
    for host in plan["spec"]["hosts"]:
        try:
            document = json.loads(invoke(host["host"], ["sudo", "-n", SPARKRING, "node", "hairpin", "status"]))
            if not isinstance(document, dict) or document.get("schema") != STATUS_SCHEMA:
                raise ValueError("unexpected hairpin status document")
            statuses.append(document)
        except FAILURES as error:
            statuses.append(str(error).splitlines()[-1] if str(error) else type(error).__name__)
    return statuses


# M8 and M9.

def busy_lines(plan, rank, hostname, findings):
    """Terminal lines for one Spark's busy findings, with stop commands that run on Node A."""
    lines = [f"rank {rank} ({hostname}):"]
    for finding in findings:
        kind, detail, unit = finding.get("kind"), finding.get("detail") or "", finding.get("unit")
        if kind == "unit" and unit and finding.get("active_state") in ("inactive", "failed"):
            lines.append(f"  - {detail}; end them: {command_text(plan, rank, ['systemctl', 'kill', unit])}")
        elif kind == "unit" and unit:
            lines.append(f"  - {detail}; stop it: {command_text(plan, rank, ['systemctl', 'stop', unit])}")
        elif kind == "forwarding" and finding.get("count"):
            lines.append(f"  - {detail}: stop its mesh service to remove them; if it is already stopped, reboot "
                         "that Spark")
        else:
            lines.append("  - " + detail)
    return lines


def m8(plan, needing, busy, *, updated=()):
    """NeedsInput for Sparks in use; busy is [(rank, hostname, findings)] with findings.

    ``updated`` names the ranks whose SparkRing package this call updated
    first; no ConnectX driver has been restarted at this point.
    """
    lines = []
    if any(finding.get("kind") == "gpu" or str(finding.get("unit") or "").endswith("-model.service")
           for _, _, findings in busy for finding in findings):
        lines.append("SparkRing's model: on Node A, sudo sparkring down")
    for rank, hostname, findings in busy:
        lines += busy_lines(plan, rank, hostname, findings)
    prefix = f"SparkRing was updated on {ranks_text(updated)}. " if updated else ""
    return NeedsInput(prefix + f"The ConnectX hairpin setting must be applied on {_count(needing, 'Spark')}, and "
                      f"{_count(len(busy), 'Spark')} {'is' if len(busy) == 1 else 'are'} in use. No ConnectX driver "
                      "was restarted. Stop what is listed, then repeat this command.", field="driver",
                      details={"sparks": [{"rank": rank, "host": plan["spec"]["hosts"][rank]["host"],
                                           "hostname": hostname, "findings": findings}
                                          for rank, hostname, findings in busy],
                               "lines": lines})


def _cut_off_advice(plan, rank, netdev):
    """M9's remedy when a failed restart cut the Sparks behind ``netdev`` off from the administration network."""
    if rank == 0:
        return (f" The workers behind {netdev} are cut off from the administration network, and this command "
                "needs them: reboot Node A (sudo systemctl reboot; power-cycle it if the reboot hangs); its next boot "
                "restarts no ConnectX function. Then run sudo sparkring hairpin.")
    return (f" The Sparks behind {netdev} are cut off from the administration network, and this command needs "
            f"them: reboot rank {rank} ({command_text(plan, rank, ['systemctl', 'reboot'])}; power-cycle it if the "
            "reboot hangs); its next boot restarts no ConnectX function. Then run sudo sparkring hairpin.")


def m9(plan, rank, hostname, *, variant, error=None, kind=None, cut_off=None):
    """NeedsInput for a unit run that never started, failed or did not report back.

    ``error`` is the reason (for a failed run, the node's short cause when it
    recorded one); ``kind`` is the node's failure class, which selects the
    advice; ``cut_off`` names the function whose failed restart cut other
    Sparks off from Node A.
    """
    lines = ["Log: " + command_text(plan, rank, ["journalctl", "-u", UNIT, "-n", "40", "--no-pager"])]
    if variant == "never-started":
        message = (f"Could not start the ConnectX hairpin step on rank {rank} ({hostname}): {_clause(error)}. No "
                   "function was restarted there; later Sparks were not changed.")
    elif variant == "silent":
        message = (f"Rank {rank} ({hostname}) did not report back within {REPORT} s after its hairpin step started. "
                   "If it stays unreachable, power-cycle it; after a restart that failed or did not finish, its next "
                   "boot restarts no function. Then repeat this command.")
    else:
        text = str(error).removeprefix(hairpin.PREFIX) if error else "its run failed"
        message = (f"The ConnectX hairpin setting did not complete on rank {rank} ({hostname}): {_sentence(text)} "
                   "Later Sparks were not changed.")
        if kind == "restart" and cut_off:
            message += _cut_off_advice(plan, rank, cut_off)
        elif kind == "restart":
            message += (" If the Spark cannot be reached, power-cycle it; it starts without restarting any function "
                        "and stays reachable. Then run sudo sparkring hairpin.")
        elif kind == "check":
            message += " If the Spark cannot be reached, power-cycle it, then run sudo sparkring hairpin again."
        lines += revoke_lines(plan, rank, text)
    return NeedsInput(message, field="driver", details={"rank": rank, "host": plan["spec"]["hosts"][rank]["host"],
                                                        "hostname": hostname, "outcome": variant, "class": kind,
                                                        "error": str(error) if error else None, "lines": lines})


# The ring procedure.

class Ring:
    """One ``ensure`` call: the plan, the Sparks' statuses and the receipt."""

    def __init__(self, plan, *, access, clock, sleep, directory, record):
        self.plan = plan
        self.access = access
        self.clock, self.sleep = clock, sleep
        self.directory = Path(directory) if directory is not None else None
        self.record = record
        self.head = plan["nodes"][0].get("revision")
        self.receipt = {"schema": RECEIPT_SCHEMA, "started_at": time.time(), "finished_at": None, "state": "running",
                        "plan_id": plan.get("id"), "ranks": []}

    def entry(self, row):
        for entry in self.receipt["ranks"]:
            if entry["rank"] == row["rank"]:
                return entry
        entry = {"rank": row["rank"], "host": row["host"], "hostname": row["hostname"], "before": row["state"],
                 "action": ACTIONS[row["state"]], "functions": row["functions"], "status_before": row["status"],
                 "invocation_before": None, "invocation_after": None, "dispatched_at": None, "finished_at": None,
                 "restarted": [], "status_after": None, "after": None, "outcome": None, "error": None}
        self.receipt["ranks"].append(entry)
        self.receipt["ranks"].sort(key=lambda value: value["rank"])
        return entry

    def save(self, state=None):
        if state:
            self.receipt["state"] = state
            self.receipt["finished_at"] = time.time()
        if self.record is not None:
            self.record.clear()
            self.record.update(copy.deepcopy(self.receipt))
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        node.save(self.directory, "hairpin.json", self.receipt, mode=0o600)
        if self.record is not None:
            self.record["path"] = str(self.directory / "hairpin.json")

    def statuses(self):
        """Read every Spark's status again and classify it against Node A's revision."""
        documents = {}
        for rank in range(4):
            try:
                documents[rank] = self.access.status(rank)
            except FAILURES as error:
                raise ValueError(f"Rank {rank} ({self.plan['spec']['hosts'][rank]['host']}): cannot read its "
                                 f"ConnectX hairpin status: {error}") from None
        head = documents[0].get("revision")
        self.head = head
        rows = []
        for rank, host in enumerate(self.plan["spec"]["hosts"]):
            document = documents[rank]
            rows.append(_row(rank, host, self.plan["nodes"][rank].get("hostname"), document,
                             document.get("revision"), head))
        return rows


def ensure(plan, *, approved, restart=True, restart_approved=True, resume=False, inspect, rebuild, update_workers,
           invoke=None, run_local=None, directory=None, record=None, clock=time.monotonic, sleep=time.sleep):
    """Bring every Spark of a four-Spark ring to ``kept``; return the resulting plan.

    The caller holds ``/var/lib/sparkring/controller/install.lock``.

    - A pair, or a ring whose Sparks are all ``kept``, returns the plan
      unchanged and runs nothing.
    - ``unknown`` Sparks stop the call with M16 before any change.
    - Without ``approved`` the call stops with M7 before any change.
    - ``update_workers()`` runs when a Spark's revision differs from Node A's;
      every Spark's status is then read again, and a revision that still
      differs stops with M15. ``restart_approved`` False means the approval
      covered no driver restart (its text was the no-restart variant); a
      Spark that needs a restart after the update then stops the call with the
      restart variant of M7 before any restart.
    - When any Spark needs a restart, every Spark's busy findings are read in
      parallel; any finding stops with M8 before any restart.
    - Approvals are recorded on every Spark that needs a run, in rank order,
      and compared with the plan before any restart (M13, M14).
    - The unit runs on Node A locally, then on one worker at a time; the
      first failure stops the call (M9) and leaves later Sparks untouched.
    - When a run changed a function (a restart, or hardware TC offload turned
      on) or a package was updated, the ring is re-inspected and
      ``rebuild(nodes)`` returns the plan, which must have no driver drift on
      the Sparks this call ran. Otherwise the plan is kept, with each Spark's
      final status in ``nodes[r]["hairpin"]``.
    - With ``resume``, every Spark then starts, in rank order, the enabled mesh
      units that its start check refused in this boot. ``sparkring hairpin``
      asks for it; ``sparkring install`` does not, because the model
      installation that follows owns the mesh.

    ``restart=False`` never dispatches a run that needs a restart: such Sparks
    get M19 (and ``unknown`` ones M16) as notices, and the call completes.
    ``invoke(host, argv, *, timeout)`` and ``run_local(argv, *, timeout)``
    return stdout and raise on failure. ``record``, a dict, receives the
    receipt, which is also written to ``directory/hairpin.json``.
    """
    if record is not None:
        record.clear()
    rows = requirement(plan)
    if not rows or not required(rows):
        if record is not None:
            record.update(state="not-required" if not rows else KEPT, ranks=[])
        return plan
    access = Access(plan, invoke=invoke, run_local=run_local)
    ring = Ring(plan, access=access, clock=clock, sleep=sleep, directory=directory, record=record)
    try:
        result = _ensure(ring, rows, approved=approved, restart=restart, restart_approved=restart_approved,
                         resume=resume, inspect=inspect, rebuild=rebuild, update_workers=update_workers)
    except NeedsInput as error:
        ring.receipt["error"] = str(error)
        ring.save("needs_input")
        raise
    except KeyboardInterrupt:
        ring.save("interrupted")
        progress.say("Interrupted. A hairpin run that already started on a Spark finishes by itself; "
                     "repeat this command to continue.")
        raise
    except BaseException as error:
        ring.receipt["error"] = str(error)
        ring.save("failed")
        raise
    return result


def _ensure(ring, rows, *, approved, restart, restart_approved, resume, inspect, rebuild, update_workers):
    plan = ring.plan
    for row in rows:
        ring.entry(row)
    if restart:
        refuse_unknown(rows)
    if not approved:
        raise NeedsInput(m7(rows), field="approval", details={"lines": consent_lines(plan, rows)})
    updated = [row["rank"] for row in rows if row["state"] == UPDATE]
    if updated:
        with progress.step("Update SparkRing on the Sparks that run another revision"):
            update_workers()
        rows = ring.statuses()
        for row in rows:
            entry = ring.entry(row)
            entry.update(before=row["state"], action=ACTIONS[row["state"]], functions=row["functions"],
                         status_before=row["status"])
        stale = [row for row in rows if row["state"] == UPDATE]
        if stale:
            raise NeedsInput(" ".join(f"Rank {row['rank']} ({row['hostname']}) still runs SparkRing "
                                      f"{_short(row['revision'])} after the update; inspect its package installation."
                                      for row in stale), field="driver",
                             details={"ranks": [row["rank"] for row in stale]})
        if restart:
            refuse_unknown(rows)
        if restart and not restart_approved and any(row["state"] == RESTART for row in rows):
            raise NeedsInput(m7_updated(rows, updated), field="approval",
                             details={"lines": consent_lines(plan, rows)})
    final = {row["rank"]: row for row in rows}
    selected = []
    for row in rows:
        entry = ring.entry(row)
        if row["state"] == KEPT:
            entry["outcome"], entry["after"] = "kept", KEPT
        elif not restart and row["state"] in (RESTART, UNKNOWN):
            text = m19(row) if row["state"] == RESTART else m16(row)
            print(text)
            entry.update(outcome="not-applied", after=row["state"], error=text)
        else:
            selected.append(row)
    ring.save()
    if restart and any(row["state"] == RESTART for row in selected):
        _idle(ring, selected, updated)
    for row in selected:
        _approve(ring, row)
    ring.save()
    restarted = False
    order = [0] + worker_order(plan)
    for rank in order:
        row = next((row for row in selected if row["rank"] == rank), None)
        if row is None:
            continue
        with progress.step(f"Node {rank}: ConnectX hairpin setting"):
            document = _run(ring, row)
        final[rank] = _row(rank, plan["spec"]["hosts"][rank], row["hostname"], document, document.get("revision"),
                           ring.head)
        entry = ring.entry(row)
        entry.update(status_after=document, after=final[rank]["state"], outcome="applied")
        restarted = restarted or bool(entry["restarted"])
        ring.save()
    # A run on a Spark that was not in effect changed a function (a restart, or
    # hardware TC offload turned on), and a package update changes what the
    # inventory reports; either makes the plan's inventory stale.
    changed = restarted or bool(updated) or any(not row["in_effect"] for row in selected)
    skipped = [entry["rank"] for entry in ring.receipt["ranks"] if entry["outcome"] == "not-applied"]
    if changed:
        plan = _reinspect(ring, inspect, rebuild, skip=skipped)
    else:
        plan = copy.deepcopy(plan)
        for rank, row in final.items():
            if row["status"] is not None:
                plan["nodes"][rank]["hairpin"] = row["status"]
    ring.receipt["reinspected"] = changed
    if resume:
        _resume(ring)
    ring.save("partial" if skipped else "complete")
    return plan


def _idle(ring, selected, updated=()):
    """Read every Spark's busy findings in parallel; any finding stops before any restart."""
    plan, access = ring.plan, ring.access

    def read(rank):
        try:
            return access.status(rank, busy=True).get("busy") or []
        except FAILURES as error:
            return [{"kind": "status", "detail": f"cannot read its hairpin status to confirm it is idle: {error}"}]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        findings = list(pool.map(read, range(4)))
    busy = [(rank, _hostname(ring, rank), found) for rank, found in enumerate(findings) if found]
    if busy:
        raise m8(plan, sum(1 for row in selected if row["state"] == RESTART), busy, updated=updated)


def _hostname(ring, rank):
    for entry in ring.receipt["ranks"]:
        if entry["rank"] == rank:
            return entry["hostname"]
    return ring.plan["spec"]["hosts"][rank]["host"]


def _expected(plan, rank):
    """The four functions of one Spark as the plan records them, by role."""
    host = plan["spec"]["hosts"][rank]
    facts = plan["inventory"]["hosts"].get(host["host"]) or {}
    pci = {row.get("device"): row.get("pci_address") for row in facts.get("rdma") or []}
    return {port["role"]: {"role": port["role"], "rdma_device": port["rdma_device"], "netdev": port["netdev"],
                           "mac": str(port["mac"]).lower(), "pci_address": pci.get(port["rdma_device"])}
            for port in host["data_interfaces"]}


def _describe(function):
    return (f"{function.get('netdev')}, MAC {function.get('mac')}, {function.get('rdma_device')} "
            f"(pci/{function.get('pci_address')})")


def _approve(ring, row):
    """Record the approval where it is missing or invalid, then compare the functions with the plan."""
    plan, access, rank = ring.plan, ring.access, row["rank"]
    document = row["status"] or {}
    if (document.get("approval") or {}).get("valid") and document.get("function_source") == "approval":
        functions = document.get("functions") or []
    else:
        try:
            approval = json.loads(access.node(rank, ["approve"], timeout=STATUS_LIMIT))
        except FAILURES as error:
            details = {"rank": rank, "host": row["host"], "error": str(error)}
            if revoke_lines(plan, rank, error):
                details["lines"] = revoke_lines(plan, rank, error)
            raise NeedsInput(f"Rank {rank} ({row['hostname']}): the hairpin approval was not recorded: {error}",
                             field="driver", details=details) from None
        functions = approval.get("functions") if isinstance(approval, dict) else None
        ring.entry(row)["approved"] = True
    expected = _expected(plan, rank)
    differences = []
    for function in functions or []:
        wanted = expected.get(function.get("role"))
        if wanted is None:
            differences.append(f"{function.get('role')}: not a fabric function of the plan")
        elif any(str(function.get(key) or "").lower() != str(wanted[key] or "").lower()
                 for key in ("rdma_device", "netdev", "mac", "pci_address")):
            differences.append(f"{function.get('role')} is {_describe(function)} but the plan records "
                               f"{_describe(wanted)}")
    if len(functions or []) != 4 or differences:
        revoke = command_text(plan, rank, [SPARKRING, "node", "hairpin", "revoke"])
        raise NeedsInput(f"Rank {rank} ({row['hostname']}): the approved ConnectX functions differ from the plan ("
                         + ("; ".join(differences) or "the approval does not list four functions")
                         + f"); nothing was restarted. If the card was replaced, run {revoke}, then sudo sparkring "
                         "hairpin on Node A.", field="driver",
                         details={"rank": rank, "host": row["host"], "differences": differences})


def _finished(document, previous):
    """The unit fields of a run that started after ``previous`` and has ended, else None."""
    unit = document.get("unit") or {}
    identifier = unit.get("InvocationID")
    if not identifier or identifier == previous:
        return None
    if unit.get("ActiveState") not in ("active", "failed", "inactive"):
        return None
    return unit


def _run(ring, row):
    """Run the unit on one Spark and wait for its outcome; return the Spark's status after the run."""
    plan, access, rank = ring.plan, ring.access, row["rank"]
    entry = ring.entry(row)
    hostname = row["hostname"]
    deadline = ring.clock() + DISPATCH_WINDOW
    error = None
    while True:
        try:
            before = access.status(rank, timeout=SSH_LIMIT if rank else STATUS_LIMIT)
            break
        except FAILURES as failure:
            error = failure
        if ring.clock() >= deadline:
            raise m9(plan, rank, hostname, variant="never-started", error=error)
        ring.sleep(DISPATCH_RETRY)
    if before.get("boot_disabled_by_kernel_command_line"):
        raise m9(plan, rank, hostname, variant="never-started",
                 error="this boot was started with sparkring.hairpin=off on the kernel command line, so "
                       f"{UNIT} does not run in it; reboot that Spark without the option, then repeat this command")
    unit = before.get("unit") or {}
    if unit.get("ActiveState") == "activating":
        # A run started earlier, for example before an interruption, is awaited, not restarted.
        previous = None
        progress.say(f"Node {rank}: waiting for the hairpin run that is already in progress")
    else:
        previous = unit.get("InvocationID")
        entry["invocation_before"] = previous
        entry["dispatched_at"] = time.time()
        ring.save()
        if rank == 0:
            error = _start_local(ring)
            if error is not None:
                try:
                    document = access.status(0)
                except FAILURES as failure:
                    # The status read after a failed or timed-out run can itself
                    # hang, for example while a hung reload holds RTNL.
                    if isinstance(error, subprocess.TimeoutExpired) or isinstance(failure, subprocess.TimeoutExpired):
                        raise m9(plan, rank, hostname, variant="silent") from None
                    raise m9(plan, rank, hostname, variant="failed",
                             error=f"{error}; its hairpin status cannot be read: {failure}") from None
                if _finished(document, previous) is None and (document.get("unit") or {}).get("ActiveState") != "activating":
                    if isinstance(error, subprocess.TimeoutExpired):
                        raise m9(plan, rank, hostname, variant="silent")
                    raise m9(plan, rank, hostname, variant="never-started", error=error)
        else:
            _dispatch(ring, rank, hostname, previous, deadline)
    document = _poll(ring, rank, hostname, previous)
    return _outcome(ring, row, document)


def _start_local(ring):
    """Restart the unit on Node A and wait for its run; return the error of systemctl, if any."""
    try:
        ring.access.run(0, ["systemctl", "restart", UNIT], timeout=REPORT)
    except subprocess.TimeoutExpired as error:
        return error
    except FAILURES as error:
        # systemctl reports a failed run as an error; the status tells which.
        return error
    return None


def _dispatch(ring, rank, hostname, previous, deadline):
    """Start the unit on a worker without waiting; retry until DISPATCH_WINDOW ends.

    ``sparkring node hairpin start --after <previous>`` starts the unit only
    while its InvocationID is still ``previous`` and no run is in progress, so
    a retry after an SSH error that hid a successful dispatch never restarts
    that run. A changed InvocationID counts as dispatched.
    """
    access, error = ring.access, None
    while True:
        try:
            access.node(rank, ["start", "--after", previous or ""], timeout=SSH_LIMIT)
            return
        except FAILURES as failure:
            error = failure
        # The request may have reached the Spark although SSH reported an error.
        with contextlib.suppress(*FAILURES):
            current = access.status(rank, timeout=SSH_LIMIT)
            unit = current.get("unit") or {}
            if unit.get("InvocationID") not in (None, "", previous) or unit.get("ActiveState") == "activating":
                return
        if ring.clock() >= deadline:
            raise m9(ring.plan, rank, hostname, variant="never-started", error=error)
        ring.sleep(DISPATCH_RETRY)


def _poll(ring, rank, hostname, previous):
    """Poll the Spark's status until the run that started after ``previous`` ends; tolerate SSH errors."""
    started = ring.clock()
    while True:
        with contextlib.suppress(*FAILURES):
            document = ring.access.status(rank, timeout=SSH_LIMIT if rank else STATUS_LIMIT)
            if _finished(document, previous) is not None:
                return document
        if ring.clock() - started >= REPORT:
            raise m9(ring.plan, rank, hostname, variant="silent")
        ring.sleep(POLL)


def _outcome(ring, row, document):
    """Check a finished run; raise M9 unless it succeeded and left the Spark in effect and armed."""
    plan, rank, hostname = ring.plan, row["rank"], row["hostname"]
    entry = ring.entry(row)
    unit = document.get("unit") or {}
    last = document.get("last_run") if isinstance(document.get("last_run"), dict) else None
    if last and last.get("invocation_id") != unit.get("InvocationID"):
        # state.json of an earlier run never describes this one.
        last = None
    entry.update(invocation_after=unit.get("InvocationID"), finished_at=time.time(), status_after=document,
                 restarted=list((last or {}).get("restarted") or []))
    if (unit.get("ActiveState") == "active" and unit.get("Result") == "success" and document.get("in_effect") is True
            and document.get("armed") is True):
        return document
    cut_off = None
    if last and last.get("error"):
        kind = last.get("class")
        if last.get("cause") and last.get("function"):
            error = f"{last['function']}: {last['cause']}"
        else:
            error = last["error"]
        cut_off = last.get("cut_off")
    elif last is None:
        error, kind = (f"its run ended ({unit.get('ActiveState')}, {unit.get('Result')}) without a run record; see "
                       + command_text(plan, rank, ["journalctl", "-u", UNIT, "-n", "40", "--no-pager"])), None
    elif document.get("in_effect") is True:
        error, kind = f"the setting is in effect but {UNIT} is not enabled for the next boot", None
    else:
        error, kind = "the setting is not in effect after its run", None
    entry.update(outcome="failed", error=error, after=classify(document, document.get("revision"), ring.head))
    raise m9(plan, rank, hostname, variant="failed", error=error, kind=kind, cut_off=cut_off)


def _reinspect(ring, inspect, rebuild, *, skip=()):
    """Re-inspect the ring after a restart or a package update and rebuild the plan from it.

    Driver drift is an error on every Spark except those in ``skip``: the
    Sparks that adoption left unchanged (M19, M16).
    """
    plan, access = ring.plan, ring.access
    targets = [host["host"] for host in plan["spec"]["hosts"]]
    identities = {current["node_id"] for current in plan["nodes"]}
    deadline = ring.clock() + REINSPECT
    with progress.step("Re-inspect the ring after the ConnectX hairpin step"):
        while True:
            for rank in range(4):
                try:
                    access.run(rank, ["lldpcli", "update"], timeout=SSH_LIMIT)
                except FAILURES as error:
                    progress.say(f"Node {rank}: lldpcli update: {error}")
            try:
                found = inspect(targets)
                if {current["node_id"] for current in found} != identities:
                    raise ValueError("Node identities changed during the ConnectX hairpin step")
                rebuilt = rebuild(found)
                break
            except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
                if not any(text in str(error) for text in TRANSIENT) or ring.clock() >= deadline:
                    raise
                ring.sleep(REINSPECT_RETRY)
    drift = [(rank, host) for rank, host in enumerate(rebuilt["network"]["hosts"])
             if host.get("driver_action") != "none" and rank not in skip]
    if drift:
        details = []
        for rank, host in drift:
            failing = [row for row in host.get("hairpin") or [] if row.get("state") != rule.IN_EFFECT]
            details.append(f"rank {rank} ({host['host']}): " + (rule.grouped(failing) if failing
                                                                 else host.get("driver_action")))
        raise NeedsInput("The ConnectX hairpin setting is not in effect after the hairpin step: " + "; ".join(details)
                         + ". On Node A, sudo sparkring hairpin retries it.", field="driver",
                         details={"lines": details})
    return rebuilt


def _resume(ring):
    """On every Spark in rank order, start the enabled mesh units that its start check refused in this boot."""
    for rank in range(4):
        entry = next((entry for entry in ring.receipt["ranks"] if entry["rank"] == rank), None)
        try:
            result = json.loads(ring.access.node(rank, ["resume"], timeout=STATUS_LIMIT))
        except FAILURES as error:
            progress.say(f"Node {rank}: could not start the mesh services that its hairpin start check refused: "
                         f"{error}")
            if entry is not None:
                entry["resume_error"] = str(error)
            continue
        started = list(result.get("started") or []) if isinstance(result, dict) else []
        for unit in started:
            progress.say(f"Node {rank}: started {unit}, which its hairpin start check had refused")
        if entry is not None:
            entry["resumed"] = started
    ring.save()


# sparkring hairpin.

def result_ranks(rows, plan, record):
    """Result rows: the requirement, with each Spark's outcome from the receipt."""
    ranks = rank_rows(rows, plan)
    for entry in (record or {}).get("ranks") or []:
        for row in ranks:
            if row["rank"] == entry["rank"]:
                row.update(after=entry.get("after"), error=entry.get("error"))
                if entry.get("outcome"):
                    row["outcome"] = entry["outcome"]
                if entry.get("resumed"):
                    row["resumed"] = entry["resumed"]
    return ranks


def stopped_meshes(plan, access):
    """(enabled mesh units of Sparks on which no mesh unit runs, as (rank, unit), problems).

    The operator starts one of them when it should serve. A Spark whose mesh
    runs is left out: its other mesh units belong to deployments that are not
    active, and a Spark runs one mesh at a time.
    """
    stopped, problems = [], []
    for rank in range(len(plan["spec"]["hosts"])):
        try:
            listing = access.run(rank, ["systemctl", "list-unit-files", "--no-legend", "--no-pager", "--type=service",
                                        *MESH_UNITS], timeout=SSH_LIMIT)
            units = sorted({line.split()[0] for line in listing.splitlines() if line.split()})
            if not units:
                continue
            shown = access.run(rank, ["systemctl", "show", "-p", "Id,ActiveState,UnitFileState", *units],
                               timeout=SSH_LIMIT)
        except FAILURES as error:
            problems.append(f"rank {rank}: cannot list its mesh services: {error}")
            continue
        found, current = [], {}
        for line in [*shown.splitlines(), ""]:
            if not line.strip():
                if current.get("Id"):
                    found.append(current)
                current = {}
                continue
            key, _, value = line.partition("=")
            current[key] = value
        if any(unit.get("ActiveState") in ("active", "activating", "reloading") for unit in found):
            continue
        stopped.extend((rank, unit["Id"]) for unit in found
                       if unit.get("ActiveState") in ("inactive", "failed")
                       and unit.get("UnitFileState") in ("enabled", "enabled-runtime"))
    return stopped, problems


def _update_workers(cluster, directory):
    from runtime.host import controller, fabric_ssh, install_assets
    transport = fabric_ssh.Transport(cluster, controller.STATE / "bulk-ssh")
    # As sparkring install does, prove that each bulk path reaches the enrolled
    # node identity before a package crosses it.
    transport.verify()
    return install_assets.Assets(transport, Path(directory) / "assets").sync_packages()


def _revoke(args, cluster, interactive, context):
    from runtime.host import controller
    plan = cluster["plan"]
    access = Access(plan)
    print("Revoke the ConnectX hairpin approval on all 4 Sparks: sparkring-hairpin.service is disabled and boot runs "
          "stop. Values in effect stay until each Spark's next boot; its mesh services then stay stopped by their "
          "start check until the setting is applied again.")
    ranks = [{"rank": rank, "host": host["host"], "hostname": node_row.get("hostname") or host["host"],
              "before": None, "action": "revoke", "functions": [], "after": None, "error": None}
             for rank, (host, node_row) in enumerate(zip(plan["spec"]["hosts"], plan["nodes"], strict=True))]
    context["ranks"] = ranks
    if args.plan:
        print("Plan only; nothing was changed. Repeat without --plan to revoke it.")
        return {"schema": RESULT_SCHEMA, "state": "planned", "ranks": ranks}
    if not args.yes:
        if not interactive:
            raise NeedsInput("Revoking the ConnectX hairpin approval needs confirmation. Review with --revoke --plan, "
                             "then repeat with --revoke --yes.", field="approval")
        controller.confirm("Revoke the ConnectX hairpin approval on every Spark?")
    failed = []
    for row in ranks:
        try:
            access.node(row["rank"], ["revoke"], timeout=STATUS_LIMIT)
            row["after"] = "revoked"
        except FAILURES as error:
            row["error"] = str(error)
            failed.append(f"rank {row['rank']}: {error}")
    if failed:
        raise RuntimeError("The ConnectX hairpin approval was not revoked everywhere: " + "; ".join(failed))
    print("ConnectX hairpin approval revoked on 4 Sparks.")
    return {"schema": RESULT_SCHEMA, "state": "complete", "ranks": ranks}


def execute(args, context):
    """``sparkring hairpin`` inside ``install.lock``; returns the result document."""
    from runtime.host import controller, install_workflow
    from scripts import deploy_network
    install_workflow.require_head(command="hairpin")
    interactive = not args.json and sys.stdin.isatty()
    with process_lock.hold(controller.STATE / "install.lock"):
        path = controller.STATE / "cluster.json"
        if not path.exists():
            raise NeedsInput("No ring is configured on this Spark. Run sudo sparkring install on Node A; on four "
                             "Sparks it applies the ConnectX hairpin setting.", field="setup")
        cluster = installer.read(path)
        install_workflow.require_head(cluster, command="hairpin")
        if len(cluster["plan"]["nodes"]) != 4:
            print("SparkRing: " + PAIR + ".")
            return {"schema": RESULT_SCHEMA, "state": "complete", "message": PAIR, "ranks": []}
        install_workflow.check_access(cluster)
        if args.revoke:
            return _revoke(args, cluster, interactive, context)
        cluster = install_workflow.refresh_cluster(cluster)
        plan = cluster["plan"]
        rows = requirement(plan)
        context["ranks"] = rank_rows(rows, plan)
        if not required(rows):
            print(COMPLETE)
            return {"schema": RESULT_SCHEMA, "state": "planned" if args.plan else "complete",
                    "ranks": context["ranks"]}
        lines = consent_lines(plan, rows)
        if args.plan:
            for line in lines:
                print(line)
            print("Plan only; nothing was changed. Repeat without --plan to apply it.")
            return {"schema": RESULT_SCHEMA, "state": "planned", "ranks": context["ranks"]}
        refuse_unknown(rows)
        if not args.yes and not interactive:
            # main prints the lines of the result once.
            raise NeedsInput(m7(rows, command=True), field="approval", details={"lines": lines})
        for line in lines:
            print(line)
        if not args.yes:
            controller.confirm("Apply the ConnectX hairpin setting?", default=consent_default(plan, rows))
        directory = controller.STATE / "hairpin" / str(time.time_ns())
        record = {}
        try:
            plan = ensure(plan, approved=True, restart_approved=restart_expected(rows), resume=True,
                          inspect=controller.collect, rebuild=lambda found: install_workflow.rebuild(cluster, found),
                          update_workers=lambda: _update_workers(cluster, directory), directory=directory,
                          record=record)
        finally:
            context["ranks"] = result_ranks(rows, plan, record)
            context["receipt"] = record.get("path")
        verified = deploy_network.verify_network(plan["spec"], plan["inventory"]["hosts"], stale_gids=True)
        print(COMPLETE)
        if verified["stale_gids"]:
            print("RoCE GID index 3 lacks the address of " + ", ".join(
                f"{entry['host']} {entry['netdev']}" for entry in verified["stale_gids"])
                + "; sudo sparkring install re-adds it before it starts the model.")
        stopped, problems = stopped_meshes(plan, Access(plan))
        for problem in problems:
            print(problem)
        if stopped:
            print("No mesh service runs on these Sparks. Start the one that should serve:")
            for rank, unit in stopped:
                print(f"  rank {rank}: {unit}: {command_text(plan, rank, ['systemctl', 'start', unit])}")
        return {"schema": RESULT_SCHEMA, "state": "complete", "ranks": context["ranks"], "receipt": record.get("path"),
                "stopped_mesh_units": [{"rank": rank, "unit": unit} for rank, unit in stopped]}


def main(argv=None):
    from runtime.host import controller
    parser = argparse.ArgumentParser(prog="sparkring hairpin",
                                     description="Apply the ConnectX hairpin setting that four-Spark forwarding needs "
                                                 "on every Spark of this ring, and at every boot. Run on Node A.")
    parser.add_argument("--plan", action="store_true", help="show what each Spark needs; change nothing")
    parser.add_argument("--yes", action="store_true", help="approve the listed driver restarts or records on an idle ring")
    parser.add_argument("--json", action="store_true", help="emit one sparkring-hairpin-result/v1 document on stdout")
    parser.add_argument("--revoke", action="store_true",
                        help="remove the approval on every Spark; boot runs stop, values in effect stay until reboot")
    args = parser.parse_args(argv)
    output = sys.stdout
    code = 0
    context = {"ranks": [], "receipt": None}
    with contextlib.redirect_stdout(sys.stderr), progress.run("hairpin"):
        try:
            result = execute(args, context)
        except NeedsInput as error:
            result, code = {"schema": RESULT_SCHEMA, **error.document()}, 3
        except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
            result, code = {"schema": RESULT_SCHEMA, "state": "failed", "message": str(error)}, 2
            progress.failure(str(error))
        except KeyboardInterrupt:
            # A unit run that already started on a Spark finishes under systemd;
            # a repeat waits for it instead of restarting it.
            result, code = {"schema": RESULT_SCHEMA, "state": "failed",
                            "message": "Interrupted; repeat sudo sparkring hairpin to continue."}, 2
        result.setdefault("ranks", context["ranks"])
        if context.get("receipt"):
            result.setdefault("receipt", context["receipt"])
        if result["state"] == "needs_input":
            print(result["message"])
            for line in controller.detail_lines(result.get("details")):
                print("  " + line)
    if args.json:
        print(json.dumps(result, indent=2), file=output)
    return code
