"""Automatic recovery of the active model deployment on Node A.

``sparkring-recover.timer`` runs ``sparkring recover --auto`` a minute after
the previous run ends. One run (``check``) observes the active deployment and,
when the model stopped serving, starts it again through the deployment's own
lifecycle code (``retained_source.apply``), as the operator's commands do:

- The model container runs on no Spark: ``up``, the code path of
  ``sudo sparkring up --execute``. A completed Compose ``up`` whose ranks all
  answer "Rank is not running" runs a new generation, which repeats every
  step: a pair's RoCE GID repair (``gid-serve``) or a ring's mesh step, and
  the container starts.
- It runs on some Sparks only, rank 0's API fails while every container
  runs, or a Spark restarted since the generation started: ``down``, then
  ``up``. A failed attempt leaves an incomplete operation, so its retry is
  ``down``, then ``up`` too.

A run acts only when every guard holds:

- recovery is on for the active deployment (``recovery.json``) and has not
  stopped after FAILURE_LIMIT consecutive failed attempts;
- the install lock (``install.lock``, ``process_lock.hold``) is free, so no
  install, setup, hairpin procedure, up or down runs;
- the deployment's last operation is a completed ``up``, or the state that
  recovery's own previous attempt left (a manual ``down`` is never undone);
- the deployment is a Compose deployment;
- every Spark answers over SSH; otherwise the run only records which Sparks
  it waits for;
- on a four-Spark ring, every Spark reports its mesh units, no mesh marker
  process runs without an active mesh unit, and the ConnectX hairpin setting
  is in effect;
- CONFIRMATIONS consecutive runs found the model not serving, and the backoff
  after a failed attempt (BACKOFF) has passed.

State is kept in ``/var/lib/sparkring/recovery.json`` (``sparkring-recovery/v1``),
one record per deployment directory. A manual ``up`` or ``install`` resets a
record's failures (``started``); ``sparkring recover on|off`` sets its choice.
"""
import argparse
import concurrent.futures
import contextlib
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

from runtime.common import compose, installer, process_lock
from runtime.host import controller, discovery, node, progress, retained_source

SCHEMA = "sparkring-recovery/v1"
TIMER = "sparkring-recover.timer"
# The timer unit the package installs; the timer is enabled only where it exists.
TIMER_FILE = Path("/usr/lib/systemd/system") / TIMER
# Seconds before the next attempt after one, two, three, and four or more
# consecutive failed attempts.
BACKOFF = (120, 300, 900, 1800)
# Consecutive failed attempts after which recovery stops until a manual up,
# install or ``sparkring recover on``.
FAILURE_LIMIT = 3
# Consecutive runs that must find the model not serving before one acts.
CONFIRMATIONS = 2
# Seconds for rank 0's /health answer.
API_TIMEOUT = 10
UP = "sudo sparkring up --execute"
DOWN_UP = "sudo sparkring down --execute, then sudo sparkring up --execute"
# States whose ``action`` recovery may perform.
ACTIONS = {"stopped": "up", "partial": "restart", "rebooted": "restart", "api-failing": "restart"}
# Node status states that carry the Spark's current boot ID.
CURRENT = ("network-configured", "existing-network-verified", "needs-attention", "not-configured")
READ_ERRORS = (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.SubprocessError)

# Run on a Spark with ``python3 -I -B -c``: one JSON document with a named
# container's state and whether it carries the deployment's label. Needs only
# Docker, so it does not depend on the Spark's SparkRing version.
CONTAINER = r'''import json,subprocess,sys
name,key,value=sys.argv[1:]
def docker(*args):
 done=subprocess.run(['docker','--context','default',*args],capture_output=True,text=True,timeout=60)
 if done.returncode: raise SystemExit(done.stderr.strip() or 'docker '+args[0]+' failed')
 return done.stdout
result={'name':name,'present':name in docker('container','ls','--all','--format','{{.Names}}').split(),'running':False}
if result['present']:
 info=json.loads(docker('inspect',name))[0]
 state=info['State']
 result.update(running=bool(state.get('Running')),health=(state.get('Health') or {}).get('Status'),
               exit_code=state.get('ExitCode'),finished_at=state.get('FinishedAt'),started_at=state.get('StartedAt'),
               owned=(info['Config'].get('Labels') or {}).get(key)==value)
print(json.dumps(result))
'''


def path():
    """``/var/lib/sparkring/recovery.json``: written by root, readable by the operator's status command."""
    return controller.STATE.parent / "recovery.json"


def load():
    """The recovery document; an empty one when none was written."""
    file = path()
    if not file.exists():
        return {"schema": SCHEMA, "deployments": {}}
    document = installer.read(file)
    if not isinstance(document, dict) or document.get("schema") != SCHEMA or not isinstance(document.get("deployments"), dict):
        raise ValueError(f"{file} is not a {SCHEMA} document")
    return document


def record_of(document, directory):
    """The record of ``directory``: recovery is on and nothing has failed for a deployment without one."""
    return {"enabled": True, "failures": 0, "stopped": False,
            **(document["deployments"].get(str(Path(directory).resolve())) or {})}


def update(directory, **changes):
    """Merge ``changes`` into the record of ``directory`` and save; return the record.

    Each writer reads the document again under ``recovery.lock``, so a check
    and a concurrent ``install`` or ``recover off`` keep each other's fields.
    Records of deployment directories that no longer exist are dropped.
    """
    file = path()
    for _ in range(50):
        try:
            with process_lock.hold(file.with_name("recovery.lock")):
                document = load()
                key = str(Path(directory).resolve())
                record = {**record_of(document, key), **changes}
                document["deployments"][key] = record
                document["deployments"] = {name: value for name, value in document["deployments"].items()
                                           if (Path(name) / "deployment.lock.json").exists()}
                node.save(file.parent, file.name, document, mode=0o644)
                return record
        except ValueError as error:
            if "Another operation is active" not in str(error):
                raise
            time.sleep(0.1)
    raise ValueError(f"{file} stayed locked by another SparkRing process")


def is_active(directory):
    """Whether ``directory`` is the deployment that ``active.json`` names."""
    file = controller.STATE / "active.json"
    return file.exists() and Path(installer.read(file)["path"]).resolve() == Path(directory).resolve()


def saved_state(directory):
    """The deployment's ``state.json`` (generation, operation, complete), or {} before its first operation."""
    file = Path(directory) / "state.json"
    return installer.read(file) if file.exists() else {}


def _state_key(state):
    return {key: state.get(key) for key in ("generation", "operation", "complete")}


def backoff(failures):
    """Seconds to wait after ``failures`` consecutive failed attempts."""
    return BACKOFF[min(max(failures, 1), len(BACKOFF)) - 1]


def enable_timer(*, run=subprocess.run):
    """Enable and start the timer on a systemd host where the package installed it; elsewhere do nothing."""
    if not (hasattr(os, "geteuid") and os.geteuid() == 0 and Path("/run/systemd/system").is_dir()
            and TIMER_FILE.exists()):
        return False
    done = run(["systemctl", "enable", "--now", TIMER], capture_output=True, text=True, timeout=60)
    if done.returncode:
        print(f"Warning: {TIMER} was not enabled: {done.stderr.strip()}", file=sys.stderr)
    return done.returncode == 0


def _parallel(function, items):
    items = list(items)
    if not items:
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(items)) as pool:
        return list(pool.map(function, items))


def node_documents(hosts, *, invoke=None):
    """Each Spark's cached ``sparkring node status`` document by SSH target.

    A Spark that cannot be read gets ``{"state": "unreachable", "error": ...}``.
    """
    invoke = invoke or discovery.ssh

    def one(host):
        try:
            return json.loads(invoke(host, ["/usr/bin/sparkring", "node", "status"]))
        except READ_ERRORS as error:
            return {"state": "unreachable", "error": (str(error).strip().splitlines() or ["no answer"])[-1]}
    return dict(zip(hosts, _parallel(one, hosts), strict=True))


def boot_ids(documents):
    """The boot ID of each Spark whose status document is current."""
    return {host: value.get("boot_id") for host, value in documents.items()
            if value.get("state") in CURRENT and value.get("boot_id")}


def probe_containers(lock, *, invoke=None):
    """Each rank's model container state, read with Docker over SSH.

    Rows carry ``rank``, ``host``, ``name``, ``present``, ``running``, and for a
    present container ``health``, ``exit_code``, ``finished_at`` and
    ``owned`` (it carries this deployment's label). A rank that cannot be
    read has ``running`` None and ``error``.
    """
    invoke = invoke or discovery.ssh

    def one(row):
        prefix = [] if row["host"].startswith("root@") else ["sudo", "-n"]
        try:
            value = json.loads(invoke(row["host"], prefix + ["python3", "-I", "-B", "-c", CONTAINER, row["name"],
                                                             compose.LABEL, lock["id"]]))
            return {"rank": row["rank"], "host": row["host"], **value}
        except READ_ERRORS as error:
            return {"rank": row["rank"], "host": row["host"], "name": row["name"], "running": None,
                    "error": (str(error).strip().splitlines() or ["no answer"])[-1]}
    return _parallel(one, installer.containers(lock))


def ranks_from_observations(observations):
    """Container rows, as ``probe_containers`` returns them, from ``retained_source`` status observations."""
    rows = []
    for rank, item in enumerate(observations or []):
        value = item.get("result") or {}
        if "error" in value or "running" not in value:
            rows.append({"rank": rank, "host": item.get("host"), "running": None,
                         "error": (str(value.get("error", "no observation")).strip().splitlines() or ["?"])[-1]})
            continue
        row = {"rank": value.get("rank", rank), "host": item.get("host"), "name": value.get("container_name") or value.get("name"),
               "present": value.get("present", True), "running": bool(value.get("running")),
               "health": value.get("health"), "exit_code": value.get("exit_code"),
               "finished_at": value.get("finished_at")}
        if "ownership_label_matches" in value:
            row["owned"] = value["ownership_label_matches"]
        rows.append(row)
    return rows


def _opener():
    # Rank 0's API is on the site's own network; a proxy from the environment does not reach it.
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open


def api_health(api_url, *, timeout=API_TIMEOUT, opener=None):
    """Rank 0's ``/health`` answer: ``{"ok", "status", "error"}``.

    vLLM answers 200 while its engine runs and 503 once the engine client has
    failed (``EngineDeadError``), the condition in which chat requests fail
    with HTTP 500 "EngineCore encountered an issue". A connection error or a
    timeout is a failed answer too.
    """
    url = api_url.removesuffix("/").removesuffix("/v1") + "/health"
    try:
        with (opener or _opener())(url, timeout=timeout) as response:
            return {"ok": response.status == 200, "status": response.status, "error": None}
    except urllib.error.HTTPError as error:
        reason = "the model engine has stopped" if error.code == 503 else error.reason
        return {"ok": False, "status": error.code, "error": f"HTTP {error.code} from /health: {reason}"}
    except (OSError, ValueError) as error:
        reason = getattr(error, "reason", None) or error
        return {"ok": False, "status": None, "error": f"no answer from {url}: {reason}"}


def _docker_time(value):
    if not value or str(value).startswith("0001-"):
        return None
    return str(value)[:19].replace("T", " ") + (" UTC" if str(value).endswith("Z") else "")


def container_text(row):
    """``running, healthy``, ``stopped, exit code 255 at 2026-09-30 17:02:11 UTC``, or why the state is unknown."""
    if row.get("running") is None:
        return "state unknown: " + str(row.get("error") or "no observation")
    if row.get("present") is False:
        return "no model container"
    if row.get("owned") is False:
        return f"the container {row.get('name')} belongs to another deployment"
    if row["running"]:
        return "running" + (f", {row['health']}" if row.get("health") else "")
    text = "stopped"
    if row.get("exit_code") is not None:
        text += f", exit code {row['exit_code']}"
    when = _docker_time(row.get("finished_at"))
    return text + (f" at {when}" if when else "")


def _serving(row):
    return bool(row.get("running")) and row.get("owned") is not False and row.get("present") is not False


def tunnel_reason(report, host):
    """Why Node A's administration tunnel does not reach ``host``, from Node A's ``control`` report; or None."""
    address = str(host).rsplit("@", 1)[-1]
    for peer in (report or {}).get("peers") or []:
        if f"{address}/32" not in (peer.get("allowed_ips") or []) and peer.get("address") != address:
            continue
        age = peer.get("handshake_age_s")
        if age is not None and age <= node.HANDSHAKE_STALE:
            return None
        return (f"the admin tunnel has no recent handshake (last {node.age_text(age)}); Node A's {peer['netdev']}: "
                + node.link_text(peer.get("carrier"), peer.get("operstate")))
    return None


def assess(ranks, *, api=None, rebooted=(), nodes=None, tunnel=None, require_mesh=False):
    """What is wrong with a deployment whose last operation is a completed up, and the next step.

    ``ranks`` are container rows (``probe_containers``); ``api`` is
    ``api_health`` or None; ``rebooted`` lists SSH targets whose boot ID
    differs from the generation's; ``nodes`` maps SSH targets to node status
    documents; ``tunnel`` is Node A's ``control`` report; ``require_mesh``
    makes a four-Spark member without a mesh report unknown.

    Returns ``state``, ``summary``, ``next_action``, ``details`` (lines) and
    ``action``: ``up``, ``restart`` (down, then up) or None when only the
    operator can act. States in order of precedence: ``unreachable``,
    ``unknown``, ``mesh-cleanup`` (marker processes without an active mesh
    unit), ``stopped``, ``partial``, ``rebooted``, ``api-failing``,
    ``mesh-failed`` and ``serving``.
    """
    nodes = nodes or {}

    def name(row):
        hostname = (nodes.get(row["host"]) or {}).get("hostname")
        return f"rank {row['rank']} ({hostname or row['host']})"

    def result(state, summary, next_action, details, affected=()):
        return {"state": state, "summary": summary, "next_action": next_action, "details": details,
                "action": ACTIONS.get(state), "ranks": [name(row) for row in affected]}

    api_line = [f"API: {api['error']}"] if api and not api.get("ok") else []
    missing = [row for row in ranks if (nodes.get(row["host"]) or {}).get("state") == "unreachable"]
    if missing:
        details, reasons = [], []
        for row in missing:
            reason = tunnel_reason(tunnel, row["host"])
            reasons.append(reason)
            details.append(f"{name(row)} does not answer: " + (reason or nodes[row["host"]].get("error") or "no answer"))
        summary = "SparkRing cannot reach " + ", ".join(name(row) for row in missing)
        if api and api.get("ok"):
            summary += "; the model's API still answers"
        links = [reason.rsplit("Node A's ", 1)[-1].split(":", 1)[0] for reason in reasons if reason]
        next_action = (f"reconnect the cable of Node A's {', '.join(links)}; it carries the admin tunnel" if links
                       else "check that " + ", ".join(name(row) for row in missing) + " is powered on and connected")
        return result("unreachable", summary, next_action, details + api_line, missing)
    unknown = [row for row in ranks if row.get("running") is None]
    mesh = {row["host"]: (nodes.get(row["host"]) or {}).get("mesh") for row in ranks}
    if require_mesh:
        unknown += [{**row, "error": "its mesh units are not reported; update SparkRing on it"}
                    for row in ranks if row not in unknown and (not isinstance(mesh[row["host"]], dict)
                                                                or "error" in mesh[row["host"]])]
    if unknown:
        return result("unknown", "The model state on " + ", ".join(name(row) for row in unknown) + " is unknown",
                      "check " + "; ".join(f"{name(row)}: {row.get('error')}" for row in unknown),
                      [f"{name(row)}: " + str(row.get("error")) for row in unknown], unknown)
    failed = [(row, unit) for row in ranks for unit in ((mesh[row["host"]] or {}).get("failed") or [])]
    mesh_lines = [f"{name(row)}: {node.mesh_failure_text(unit)}" for row, unit in failed]
    orphaned = [row for row in ranks if node.orphaned_markers(mesh[row["host"]])]
    running = [row for row in ranks if _serving(row)]
    stopped = [row for row in ranks if not _serving(row)]
    container_lines = [f"{name(row)}: {container_text(row)}" for row in stopped]
    if orphaned:
        names = ", ".join(name(row) for row in orphaned)
        return result("mesh-cleanup", f"The mesh on {names} stopped without stopping its forwarding processes",
                      f"stop them on {names} (install reference: When a model stops serving), then {DOWN_UP}",
                      [f"{name(row)}: mesh marker processes run while no mesh unit is active" for row in orphaned]
                      + mesh_lines + container_lines, orphaned)
    if not running:
        return result("stopped", "The model is not running on any Spark", UP, container_lines + mesh_lines, stopped)
    if stopped:
        return result("partial", "The model runs on " + ", ".join(name(row) for row in running) + " but stopped on "
                      + ", ".join(name(row) for row in stopped), DOWN_UP, container_lines + mesh_lines + api_line,
                      stopped)
    restarted = [row for row in ranks if row["host"] in rebooted]
    if restarted:
        names = ", ".join(name(row) for row in restarted)
        return result("rebooted", f"{names} restarted after the model started", DOWN_UP,
                      [f"{name(row)} restarted since the model started" for row in restarted] + api_line,
                      restarted)
    if api is not None and not api.get("ok"):
        return result("api-failing", "The model's containers run, but its API fails", DOWN_UP, api_line,
                      ranks[:1])
    if failed:
        return result("mesh-failed", "The mesh service failed on " + ", ".join(sorted({name(row) for row, _ in failed})),
                      UP, mesh_lines, [row for row, _ in failed])
    return result("serving", "The model runs on every Spark and its API answers", None, [])


def status_assessment(deployment, nodes, record, *, tunnel=None, api=None):
    """The ``assess`` result that ``sparkring status --refresh`` prints, or None.

    None unless the deployment's last operation is a completed up and its
    containers were observed. ``nodes`` are the status rows of every Spark;
    ``record`` is the deployment's recovery record, whose boot IDs of the
    generation give the restart signal; ``tunnel`` is Node A's ``control``
    report; ``api`` probes rank 0's API (``api_health``) while rank 0's
    container runs.
    """
    state = deployment.get("state") or {}
    if state.get("operation") != "up" or not state.get("complete") or not deployment.get("observations"):
        return None
    ranks = ranks_from_observations(deployment["observations"])
    documents = {row.get("host"): row for row in nodes or []}
    health = None
    if ranks and ranks[0].get("running") and deployment.get("api_url"):
        health = (api or api_health)(deployment["api_url"])
    record = record or {}
    boots = (record.get("boots") or {}) if record.get("generation") == state.get("generation") else {}
    current = boot_ids(documents)
    rebooted = [host for host, boot in boots.items() if boot and current.get(host) and current[host] != boot]
    return assess(ranks, api=health, rebooted=rebooted, nodes=documents, tunnel=tunnel)


def started(directory, *, enabled=None, invoke=None):
    """Record that a manual ``up`` or ``install`` started ``directory``; return the record, or None.

    Resets the failure count and backoff, forgets any unfinished attempt,
    records the generation and each Spark's boot ID for the restart signal,
    keeps the deployment's recorded choice unless ``enabled`` names one, and
    enables the timer. Never raises: the model operation itself succeeded.
    """
    try:
        directory = Path(directory).resolve()
        lock = installer.read(directory / "deployment.lock.json")
        boots = boot_ids(node_documents([row["host"] for row in lock["site"]["ranks"]], invoke=invoke))
        changes = {"failures": 0, "stopped": False, "next_attempt_at": None, "pending": None, "attempt": None,
                   "left": None, "waiting": None, "generation": saved_state(directory).get("generation"),
                   "boots": boots}
        if enabled is not None:
            changes["enabled"] = bool(enabled)
        record = update(directory, **changes)
        if record["enabled"]:
            enable_timer()
        return record
    except Exception as error:  # noqa: BLE001 - a recovery record never fails the model operation
        print("Warning: automatic recovery state was not recorded: " + str(error), file=sys.stderr)
        return None


def forget_attempt(directory):
    """Forget recovery's own unfinished attempt before a manual operation changes ``directory``.

    Afterwards only a completed ``up`` lets recovery act again, so a manual
    ``down`` is never taken for the state recovery left. Never raises.
    """
    try:
        if Path(directory, "deployment.lock.json").exists():
            update(directory, attempt=None, left=None, pending=None)
    except Exception as error:  # noqa: BLE001 - see started()
        print("Warning: automatic recovery state was not updated: " + str(error), file=sys.stderr)


def set_enabled(enabled, *, directory=None):
    """``sparkring recover on|off`` for the active deployment; ``on`` also clears failures and backoff."""
    directory = directory or controller.active_deployment()
    if directory is None:
        raise ValueError("No model deployment is active. Start one with sudo sparkring install.")
    changes = {"enabled": bool(enabled)}
    if enabled:
        changes.update(failures=0, stopped=False, next_attempt_at=None, pending=None)
    record = update(directory, **changes)
    if enabled:
        enable_timer()
    return record


def _busy():
    try:
        with process_lock.hold(controller.STATE / "install.lock"):
            return False
    except ValueError:
        return True


def _outcome(state, summary, directory=None, **fields):
    return {"state": state, "summary": summary, "deployment": str(directory) if directory else None, **fields}


def _local_tunnel():
    try:
        return node.control_report()
    except READ_ERRORS:
        return None


def check(*, now=time.time, invoke=None, api=None, apply=None, tunnel=None):
    """One run of ``sparkring recover --auto``; returns ``{"state", "summary", "deployment", ...}``.

    ``state`` is one of ``idle``, ``off``, ``stopped``, ``busy``, ``inactive``,
    ``unsupported``, ``waiting``, ``serving``, ``report`` (not serving, and
    only the operator can act), ``confirming``, ``backoff``, ``changed``,
    ``recovered`` or ``failed``. ``invoke`` replaces SSH, ``api`` the /health
    probe, ``apply`` the lifecycle call ``apply(operation)`` and ``tunnel``
    Node A's ``control`` report.
    """
    invoke = invoke or discovery.ssh
    directory = controller.active_deployment()
    if directory is None:
        return _outcome("idle", "No model deployment is active")
    directory = Path(directory).resolve()
    record = record_of(load(), directory)
    if not record["enabled"]:
        return _outcome("off", "Automatic recovery is off for this deployment", directory)
    if record["stopped"]:
        return _outcome("stopped", f"Stopped after {record['failures']} failed attempts; {UP} turns it back on", directory)
    if _busy():
        return _outcome("busy", "Another SparkRing operation is running", directory)
    state = saved_state(directory)
    if record.get("attempt"):
        record = _interrupted(directory, record, state, now())
        if record["stopped"]:
            return _outcome("stopped", f"Stopped after {record['failures']} failed attempts", directory)
    left = record.get("left")
    if not (state.get("operation") == "up" and state.get("complete")) and not (left and _state_key(state) == left):
        if record.get("pending") or record.get("waiting"):
            update(directory, pending=None, waiting=None)
        return _outcome("inactive", "The last model operation is not a completed up; recovery waits for one", directory)
    lock = installer.read(directory / "deployment.lock.json")
    if lock.get("backend") != "compose":
        return _outcome("unsupported", f"Automatic recovery covers Compose deployments; this one uses {lock.get('backend')}",
                        directory)
    finding = observe(directory, lock, record, state, invoke=invoke, api=api,
                      tunnel=tunnel if tunnel is not None else _local_tunnel())
    t = now()
    shown = {key: finding[key] for key in ("state", "summary", "next_action", "details")}
    if finding["state"] == "unreachable":
        since = (record.get("waiting") or {}).get("since") or t
        update(directory, checked_at=t, finding=shown, pending=None, waiting={"hosts": finding["ranks"], "since": since})
        return _outcome("waiting", finding["summary"], directory, finding=shown)
    changes = {"checked_at": t, "finding": shown, "waiting": None}
    if finding["state"] == "serving":
        boots = dict(record.get("boots") or {}) if record.get("generation") == state.get("generation") else {}
        # A boot ID that up could not read is recorded by the first run that sees the model serve.
        boots.update({host: boot for host, boot in finding["boots"].items() if host not in boots})
        update(directory, **changes, pending=None, failures=0, next_attempt_at=None, left=None,
               generation=state.get("generation"), boots=boots)
        return _outcome("serving", finding["summary"], directory, finding=shown)
    if finding["action"] is None:
        update(directory, **changes, pending=None)
        return _outcome("report", finding["summary"], directory, finding=shown)
    pending = record.get("pending") or {}
    count = pending.get("count", 0) + 1
    changes["pending"] = {"count": count, "since": pending.get("since") or t, "state": finding["state"]}
    if record.get("next_attempt_at") and t < record["next_attempt_at"]:
        update(directory, **changes)
        return _outcome("backoff", finding["summary"] + "; the next attempt waits for its backoff", directory,
                        finding=shown)
    if count < CONFIRMATIONS:
        update(directory, **changes)
        return _outcome("confirming", finding["summary"] + "; the next check confirms it", directory, finding=shown)
    update(directory, **changes)
    return attempt(directory, lock, state, record, finding, now=now, apply=apply, invoke=invoke)


def _interrupted(directory, record, state, t):
    """Record an attempt that stopped without recording its end, for example because Node A restarted."""
    prior = record["attempt"]
    first = prior.get("generation") or 0
    # The install lock kept other operations out while it ran, and every
    # manual operation forgets it, so a state within its generations is its own.
    ours = state.get("operation") in ("up", "down") and first <= (state.get("generation") or 0) <= first + 2
    failures = record.get("failures", 0) + 1
    return update(directory, attempt=None, failures=failures, stopped=failures >= FAILURE_LIMIT,
                  next_attempt_at=t + backoff(failures), left=_state_key(state) if ours else None,
                  last={**prior, "finished_at": None, "result": "interrupted",
                        "error": "the recovery run stopped before it finished"})


def observe(directory, lock, record, state, *, invoke, api=None, tunnel=None):
    """``assess`` of the deployment in ``directory``, with ``boots``: each Spark's current boot ID."""
    hosts = [row["host"] for row in lock["site"]["ranks"]]
    documents = node_documents(hosts, invoke=invoke)
    unreachable = any(value.get("state") == "unreachable" for value in documents.values())
    if unreachable:
        ranks = [{"rank": row["rank"], "host": row["host"], "running": None} for row in lock["site"]["ranks"]]
    else:
        ranks = probe_containers(lock, invoke=invoke)
    health = None
    if unreachable or ranks[0].get("running"):
        url = record.get("api_url")
        if not url:
            url = retained_source.apply(directory, "saved-status", cache=controller.STATE / "retained-sources")["api_url"]
            update(directory, api_url=url)
        health = (api or api_health)(url)
    boots = boot_ids(documents)
    baseline = (record.get("boots") or {}) if record.get("generation") == state.get("generation") else {}
    rebooted = [host for host in hosts if baseline.get(host) and boots.get(host) and baseline[host] != boots[host]]
    finding = assess(ranks, api=health, rebooted=rebooted, nodes=documents, tunnel=tunnel,
                     require_mesh=len(hosts) == 4)
    finding["boots"] = boots
    return finding


def attempt(directory, lock, state, record, finding, *, now=time.time, apply=None, invoke=None):
    """Start the model again under the install lock; record the result and the backoff after a failure."""
    cache = controller.STATE / "retained-sources"
    apply = apply or (lambda operation: retained_source.apply(directory, operation, cache=cache))
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(process_lock.hold(controller.STATE / "install.lock"))
        except ValueError:
            return _outcome("busy", "Another SparkRing operation started", directory)
        if _state_key(saved_state(directory)) != _state_key(state):
            update(directory, pending=None)
            return _outcome("changed", "The deployment changed during the check; the next check observes it again",
                            directory)
        if len(lock["site"]["ranks"]) == 4:
            # retained_source.apply does not check the hairpin setting; up does.
            problem = controller._hairpin_problem()
            if problem:
                shown = {"state": "hairpin", "summary": "The ConnectX hairpin setting is not in effect",
                         "next_action": "on Node A: sudo sparkring hairpin", "details": problem.splitlines()}
                update(directory, finding=shown, pending=None)
                return _outcome("report", shown["summary"], directory, finding=shown)
        start_only = finding["action"] == "up" and state.get("operation") == "up" and state.get("complete")
        operations = ["up"] if start_only else ["down", "up"]
        began = now()
        entry = {"started_at": began, "action": "start" if start_only else "restart", "reason": finding["summary"],
                 "generation": state.get("generation")}
        update(directory, attempt=entry, pending=None)
        error = None
        with progress.run("recover"):
            print(f"Automatic recovery of {lock['selection']['profile']}: {finding['summary']}.")
            print("Starting the model on every Spark." if start_only
                  else "Stopping the model on every Spark, then starting it.")
            try:
                for operation in operations:
                    apply(operation)
            except Exception as failure:  # noqa: BLE001 - every failure is recorded and backs off
                error = (str(failure).strip().splitlines() or [type(failure).__name__])[-1]
                progress.failure(traceback.format_exc())
            after = saved_state(directory)
            succeeded = error is None and after.get("operation") == "up" and after.get("complete")
            print("Automatic recovery finished: the model started." if succeeded
                  else "Automatic recovery failed: " + (error or "the model operation did not complete"))
    finished = now()
    last = {**entry, "finished_at": finished, "result": "succeeded" if succeeded else "failed", "error": error}
    if succeeded:
        documents = node_documents([row["host"] for row in lock["site"]["ranks"]], invoke=invoke)
        update(directory, attempt=None, last=last, failures=0, next_attempt_at=None, left=None,
               generation=after.get("generation"), boots=boot_ids(documents))
        return _outcome("recovered", "The model started again", directory)
    failures = record.get("failures", 0) + 1
    update(directory, attempt=None, last=last, failures=failures, stopped=failures >= FAILURE_LIMIT,
           next_attempt_at=finished + backoff(failures), left=_state_key(after))
    return _outcome("failed", "Automatic recovery failed: " + (error or "the model operation did not complete"), directory)


def clock(seconds):
    """A timestamp as local ``YYYY-MM-DD HH:MM``, or UTC where local time is unavailable."""
    moment = datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc)
    try:
        return moment.astimezone().strftime("%Y-%m-%d %H:%M")
    except (OSError, OverflowError, ValueError):
        return moment.strftime("%Y-%m-%d %H:%M UTC")


RESULTS = {"succeeded": "the model started", "failed": "failed", "interrupted": "interrupted"}


def timer_enabled(*, run=subprocess.run):
    """Whether systemd has the timer enabled; None where systemd is not running or cannot answer."""
    if not Path("/run/systemd/system").is_dir():
        return None
    try:
        answer = run(["systemctl", "is-enabled", TIMER], capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return answer == "enabled"


def status_lines(record, *, now=time.time, timer=None):
    """The lines ``sparkring status`` prints about automatic recovery of one deployment.

    ``timer`` is ``timer_enabled()``: False adds that the timer does not run.
    """
    if record is None:
        return ["Automatic recovery: state unreadable; use sudo sparkring status"]
    if not record.get("enabled", True):
        return ["Automatic recovery: off | turn on: sudo sparkring recover on"]
    if record.get("stopped"):
        lines = [f"Automatic recovery: stopped after {record.get('failures')} failed attempts | {UP} turns it back on"]
    elif timer is False:
        lines = [f"Automatic recovery: on, but {TIMER} is not enabled | sudo sparkring recover on enables it"]
    else:
        lines = ["Automatic recovery: on"]
    waiting = record.get("waiting")
    if waiting:
        lines.append(f"  waiting for {', '.join(waiting.get('hosts') or ['a Spark'])} since {clock(waiting['since'])}")
    last = record.get("last")
    if last:
        verb = "start" if last.get("action") == "start" else "restart"
        lines.append(f"  last attempt: {verb} at {clock(last['started_at'])}, {RESULTS.get(last.get('result'), last.get('result'))}"
                     + (f": {last['error']}" if last.get("error") else ""))
    upcoming = record.get("next_attempt_at")
    if not record.get("stopped"):
        if upcoming and upcoming > now():
            lines.append(f"  next attempt: not before {clock(upcoming)}")
        elif record.get("pending"):
            lines.append("  next attempt: at the next check, about a minute from the last one, if the model still "
                         "does not serve")
    return lines


def describe():
    """``sparkring recover status``: the active deployment and its record."""
    directory = controller.active_deployment()
    if directory is None:
        return {"deployment": None, "record": None, "lines": ["No model deployment is active."]}
    directory = Path(directory).resolve()
    record = record_of(load(), directory)
    lines = ["Deployment: " + str(directory), *status_lines(record)]
    finding = record.get("finding")
    if finding and record.get("checked_at"):
        lines.append(f"Last check at {clock(record['checked_at'])}: {finding['summary']}"
                     + (f" | next: {finding['next_action']}" if finding.get("next_action") else ""))
        lines += ["  " + line for line in finding.get("details") or []]
    return {"deployment": str(directory), "record": record, "lines": lines}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sparkring recover",
                                     description="Automatic restart of the active model when a Spark stops serving.")
    parser.add_argument("action", nargs="?", choices=("status", "on", "off"),
                        help="status (default), or turn automatic recovery on or off for the active deployment")
    parser.add_argument("--auto", action="store_true",
                        help="check once and restart the model if needed; sparkring-recover.timer runs this")
    parser.add_argument("--json", action="store_true", help="print one JSON document")
    args = parser.parse_args(argv)
    if args.auto and args.action:
        parser.error("--auto takes no action")
    try:
        if args.auto:
            result = check()
            print(json.dumps(result, indent=2) if args.json else f"{result['state']}: {result['summary']}")
            return 1 if result["state"] == "failed" else 0
        if args.action in ("on", "off"):
            record = set_enabled(args.action == "on")
            result = {"deployment": str(Path(controller.active_deployment()).resolve()), "record": record,
                      "lines": status_lines(record)}
        else:
            result = describe()
        print(json.dumps({key: value for key, value in result.items() if key != "lines"}, indent=2)
              if args.json else "\n".join(result["lines"]))
        return 0
    except PermissionError as error:
        print(f"SparkRing recover: {error}; use sudo", file=sys.stderr)
        return 2
    except READ_ERRORS as error:
        print("SparkRing recover: " + str(error), file=sys.stderr)
        return 2
