"""Automatic recovery of the active model deployments on Node A.

``sparkring-recover.timer`` runs ``sparkring recover --auto`` a minute after
the previous run ends. One run (``check``) observes each active deployment:
the whole cluster's, and each group's on a fabric that serves models on
part of its Sparks (``runtime.host.placement``). When a model stopped serving, the run starts it
again through the deployment's own lifecycle code (``retained_source.apply``),
as the operator's commands do:

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
  stopped (FAILURE_LIMIT consecutive failed attempts, or RESTART_LIMIT
  restarts within RESTART_WINDOW);
- the install lock (``install.lock``, ``process_lock.hold``) is free, so no
  install, setup, hairpin procedure, up or down runs. The lock covers the
  whole cluster: while one half of a ring installs, a run reports the other
  half's model as ``busy`` and checks it again at its next run;
- the deployment's last operation is a completed ``up``, or the state that
  recovery's own previous attempt left (a manual ``down`` is never undone);
- the deployment is a Compose deployment;
- every Spark answers over SSH; otherwise the run only records which Sparks
  it waits for;
- for a four-rank model, every Spark's status report is current and reports
  its mesh units, no mesh marker process runs without an active mesh unit,
  and the ConnectX hairpin setting is in effect;
- consecutive runs found the model not serving: CONFIRMATIONS runs, or for an
  API that gives no answer at all, API_SILENCE seconds; and the backoff after
  a failed attempt (BACKOFF) has passed.

Before it acts, the run takes the install lock, reads its record and the
active deployment again, and observes the deployment again; it acts on that
second observation only.

State is kept in ``/var/lib/sparkring/recovery.json`` (``sparkring-recovery/v1``),
one record per deployment directory. A manual ``up`` or ``install`` resets a
record's failures and restarts (``started``); ``sparkring recover on|off``
sets its choice.
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
# Seconds before the next attempt after one, two and three consecutive failed attempts.
BACKOFF = (120, 300, 900)
# Consecutive failed attempts after which recovery stops until a manual up,
# install or ``sparkring recover on``: with BACKOFF, about 25 minutes of trying.
FAILURE_LIMIT = 4
# Successful restarts within RESTART_WINDOW seconds after which recovery stops
# instead of restarting again. Each restart reloads the model (10 to 30
# minutes); a model that stops again this often has a cause a restart does
# not remove, such as a failing cable or Spark, and restarting would hide it.
RESTART_LIMIT = 3
RESTART_WINDOW = 6 * 3600
# Consecutive runs that must find the model not serving before one acts.
CONFIRMATIONS = 2
# Seconds rank 0's API may give no answer at all (timeouts, refused
# connections) before that alone counts as not serving. A 503 from /health,
# vLLM's answer for a stopped engine, counts after CONFIRMATIONS runs.
API_SILENCE = 300
# Seconds for each answer of rank 0's API.
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


def _empty():
    return {"schema": SCHEMA, "deployments": {}}


def load(*, repair=False):
    """The recovery document; an empty one when none was written.

    A file that is not a ``sparkring-recovery/v1`` document raises
    ValueError, or with ``repair`` is moved aside to
    ``recovery.json.unreadable-<ns>`` with a warning, so recovery starts
    from empty records rather than staying disabled.
    """
    file = path()
    if not file.exists():
        return _empty()
    try:
        document = installer.read(file)
        valid = (isinstance(document, dict) and document.get("schema") == SCHEMA
                 and isinstance(document.get("deployments"), dict))
    except ValueError:
        document, valid = None, False
    if valid:
        return document
    if not repair:
        raise ValueError(f"{file} is not a {SCHEMA} document")
    aside = file.with_name(f"{file.name}.unreadable-{time.time_ns()}")
    os.replace(file, aside)
    print(f"Warning: {file} was unreadable; it was moved to {aside} and automatic recovery starts from empty records",
          file=sys.stderr)
    return _empty()


def _write(document):
    """Replace the recovery file atomically; its content and the rename reach the disk before this returns."""
    file = path()
    if any(item.is_symlink() for item in (file, *file.parents)):
        raise ValueError("Recovery state path contains a symlink: " + str(file))
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(file.name + ".writing")
    with open(temporary, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(document, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o644)
    os.replace(temporary, file)
    try:
        descriptor = os.open(file.parent, os.O_RDONLY)
    except OSError:
        return  # A platform that cannot open a directory (Windows) cannot sync it either.
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


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
                document = load(repair=True)
                key = str(Path(directory).resolve())
                record = {**record_of(document, key), **changes}
                document["deployments"][key] = record
                document["deployments"] = {name: value for name, value in document["deployments"].items()
                                           if (Path(name) / "deployment.lock.json").exists()}
                _write(document)
                return record
        except ValueError as error:
            if "Another operation is active" not in str(error):
                raise
            time.sleep(0.1)
    raise ValueError(f"{file} stayed locked by another SparkRing process")


def is_active(directory):
    """Whether ``directory`` is the deployment its slot's ``active.json`` names."""
    from runtime.host import placement
    recorded = placement.recorded(controller.STATE, placement.of_directory(directory))
    return recorded is not None and Path(recorded).resolve() == Path(directory).resolve()


def supported(lock):
    """Whether recovery restarts the deployment of ``lock``: Compose deployments only."""
    return lock.get("backend") == "compose"


def saved_state(directory):
    """The deployment's ``state.json`` (generation, operation, complete), or {} before its first operation."""
    file = Path(directory) / "state.json"
    return installer.read(file) if file.exists() else {}


def _state_key(state):
    return {key: state.get(key) for key in ("generation", "operation", "complete")}


def may_act(state, record):
    """Whether the saved ``state`` lets recovery act: a completed up, or exactly what its own attempt left."""
    left = record.get("left")
    return bool(state.get("operation") == "up" and state.get("complete")) or bool(left and _state_key(state) == left)


def backoff(failures):
    """Seconds to wait after ``failures`` consecutive failed attempts."""
    return BACKOFF[min(max(failures, 1), len(BACKOFF)) - 1]


def recent_restarts(record, now):
    """Times of the successful restarts within RESTART_WINDOW seconds before ``now``."""
    return [moment for moment in record.get("restarts") or [] if now - moment < RESTART_WINDOW]


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


def _last_line(error):
    return (str(error).strip().splitlines() or ["no answer"])[-1][:node.ERROR_TEXT]


def node_documents(hosts, *, invoke=None):
    """Each Spark's cached ``sparkring node status`` document by SSH target.

    A Spark that cannot be read gets ``{"state": "unreachable", "error": ...}``.
    The node command marks a document older than 90 seconds ``stale`` and
    reports ``agent-unavailable`` before its agent wrote one.
    """
    invoke = invoke or discovery.ssh

    def one(host):
        try:
            return json.loads(invoke(host, ["/usr/bin/sparkring", "node", "status"]))
        except READ_ERRORS as error:
            return {"state": "unreachable", "error": _last_line(error)}
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
                    "error": _last_line(error)}
    return _parallel(one, installer.containers(lock))


def ranks_from_observations(observations):
    """Container rows, as ``probe_containers`` returns them, from ``retained_source`` status observations."""
    rows = []
    for rank, item in enumerate(observations or []):
        value = item.get("result") or {}
        if "error" in value or "running" not in value:
            rows.append({"rank": rank, "host": item.get("host"), "running": None,
                         "error": _last_line(value.get("error", "no observation"))})
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


def _ask(url, timeout, opener):
    """``(status, error)`` of one GET: the HTTP status, or None and why there was no answer."""
    try:
        with opener(url, timeout=timeout) as response:
            return response.status, None
    except urllib.error.HTTPError as error:
        return error.code, error.reason
    except (OSError, ValueError) as error:
        return None, str(getattr(error, "reason", None) or error)


def api_health(api_url, *, timeout=API_TIMEOUT, opener=None):
    """Rank 0's API answer: ``{"ok", "status", "dead", "error"}``.

    vLLM's ``/health`` answers 200 while its engine runs and 503 once the
    engine client has failed (``EngineDeadError``), the condition in which
    chat requests fail with HTTP 500 "EngineCore encountered an issue"; that
    503 sets ``dead``. Any other failure (a timeout, a refused connection,
    another status) asks ``/v1/models`` as a second probe: an answer there
    counts as serving, and without one the API is failing but not ``dead``,
    which recovery treats as a fault only once it lasts API_SILENCE seconds.
    """
    base = api_url.removesuffix("/").removesuffix("/v1")
    opener = opener or _opener()
    status, reason = _ask(base + "/health", timeout, opener)
    if status == 200:
        return {"ok": True, "status": 200, "dead": False, "error": None}
    if status == 503:
        return {"ok": False, "status": 503, "dead": True,
                "error": "HTTP 503 from /health: the model engine has stopped"}
    first = f"HTTP {status} from /health" if status else f"no answer from {base}/health: {reason}"
    second, other = _ask(base + "/v1/models", timeout, opener)
    if second == 200:
        return {"ok": True, "status": 200, "dead": False, "error": None, "note": first + "; /v1/models answered"}
    more = f"HTTP {second}" if second else f"no answer: {other}"
    return {"ok": False, "status": status, "dead": False, "error": f"{first}; /v1/models: {more}"}


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


def _tunnel_peer(report, host):
    address = str(host).rsplit("@", 1)[-1]
    for peer in (report or {}).get("peers") or []:
        if f"{address}/32" in (peer.get("allowed_ips") or []) or peer.get("address") == address:
            return peer
    return None


def tunnel_reason(report, host):
    """Why Node A's administration tunnel does not reach ``host``, from Node A's ``control`` report; or None."""
    peer = _tunnel_peer(report, host)
    if peer is None:
        return None
    age = peer.get("handshake_age_s")
    if age is not None and age <= node.HANDSHAKE_STALE:
        return None
    trying = node.fallback_path(peer)
    return (f"the admin tunnel has no recent handshake (last {node.age_text(age)})"
            + (f" over {trying}" if trying else "") + f"; Node A's {peer['netdev']}: "
            + node.link_text(peer.get("carrier"), peer.get("operstate")))


def tunnel_fallback(report, host):
    """``over LAN 192.0.2.12 (primary cable enp1s0f1np1: no link)`` when Node A reaches ``host`` over a fallback path; else None."""
    peer = _tunnel_peer(report, host)
    if peer is None:
        return None
    age = peer.get("handshake_age_s")
    return node.fallback_text(peer) if age is not None and age <= node.HANDSHAKE_STALE else None


def _mesh_problem(document):
    """Why a four-Spark member's status document cannot vouch for its mesh, or None."""
    state = (document or {}).get("state")
    if state == "stale":
        return f"its status report is stale ({int(document.get('age_seconds') or 0)} s old)"
    if state == "agent-unavailable":
        return "its status service has not reported yet"
    mesh = (document or {}).get("mesh")
    if not isinstance(mesh, dict):
        return "its mesh units are not reported; update SparkRing on it"
    if "error" in mesh:
        return "its mesh units could not be read: " + str(mesh["error"])
    return None


def assess(ranks, *, api=None, rebooted=(), nodes=None, tunnel=None, require_mesh=False):
    """What is wrong with a deployment whose last operation is a completed up, and the next step.

    ``ranks`` are container rows (``probe_containers``); ``api`` is
    ``api_health`` or None; ``rebooted`` lists SSH targets whose boot ID
    differs from the generation's; ``nodes`` maps SSH targets to node status
    documents; ``tunnel`` is Node A's ``control`` report; ``require_mesh``
    makes a four-Spark member whose status report is stale, missing or
    without a mesh report unknown.

    Returns ``state``, ``summary``, ``next_action``, ``details`` (lines),
    ``ranks`` (the ranks it names), ``api_dead`` (the API answered that its
    engine stopped) and ``action``: ``up``, ``restart`` (down, then up) or
    None when only the operator can act. States in order of precedence:
    ``unreachable``, ``unknown``, ``mesh-cleanup`` (marker processes without
    an active mesh unit), ``stopped``, ``partial``, ``rebooted``,
    ``api-failing``, ``mesh-failed`` and ``serving``.
    """
    nodes = nodes or {}

    def name(row):
        hostname = (nodes.get(row["host"]) or {}).get("hostname")
        return f"rank {row['rank']} ({hostname or row['host']})"

    def result(state, summary, next_action, details, affected=()):
        return {"state": state, "summary": summary, "next_action": next_action, "details": details,
                "action": ACTIONS.get(state), "ranks": [name(row) for row in affected],
                "api_dead": bool(api and api.get("dead"))}

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
        for row in ranks:
            problem = _mesh_problem(nodes.get(row["host"]))
            if problem and row not in unknown:
                unknown.append({**row, "error": problem})
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

    Resets the failure count, the restart history and the backoff, forgets
    any unfinished attempt, records the generation and each Spark's boot ID
    for the restart signal, keeps the deployment's recorded choice unless
    ``enabled`` names one, and enables the timer. The record of a deployment
    recovery does not restart (``supported``) carries ``supported`` false
    and enables no timer. Never raises: the model operation itself succeeded.
    """
    try:
        directory = Path(directory).resolve()
        lock = installer.read(directory / "deployment.lock.json")
        if not supported(lock):
            return update(directory, supported=False)
        boots = boot_ids(node_documents([row["host"] for row in lock["site"]["ranks"]], invoke=invoke))
        changes = {"supported": True, "failures": 0, "stopped": False, "stop_reason": None, "restarts": [],
                   "next_attempt_at": None, "pending": None, "attempt": None, "left": None, "waiting": None,
                   "generation": saved_state(directory).get("generation"), "boots": boots}
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
    """``sparkring recover on|off`` for one active deployment; ``on`` also clears failures, restarts and backoff."""
    directory = directory or controller.active_deployment()
    if directory is None:
        raise ValueError("No model deployment is active. Start one with sudo sparkring install.")
    if not supported(installer.read(Path(directory) / "deployment.lock.json")):
        raise ValueError("Automatic recovery does not restart this deployment's backend; restart it by hand")
    changes = {"enabled": bool(enabled)}
    if enabled:
        changes.update(failures=0, stopped=False, stop_reason=None, restarts=[], next_attempt_at=None, pending=None)
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


def _stopped_text(record):
    return "Stopped: " + str(record.get("stop_reason") or f"{record.get('failures')} failed attempts in a row")


def _confirmed(finding, pending, t):
    """Whether the runs recorded in ``pending`` are enough to act on ``finding``."""
    if finding["state"] == "api-failing" and not finding.get("api_dead"):
        return t - pending["since"] >= API_SILENCE
    return pending["count"] >= CONFIRMATIONS


# The outcome of a run over several deployments: the first state in this order that one of them reached.
PRECEDENCE = ("failed", "recovered", "stopped", "report", "waiting", "backoff", "confirming", "changed", "busy",
              "unsupported", "inactive", "off", "serving", "idle")


def check(*, now=time.time, invoke=None, api=None, apply=None, tunnel=None):
    """One run of ``sparkring recover --auto``; returns ``{"state", "summary", "deployment", ...}``.

    ``state`` is one of ``idle``, ``off``, ``stopped``, ``busy``, ``inactive``,
    ``unsupported``, ``waiting``, ``serving``, ``report`` (not serving, and
    only the operator can act), ``confirming``, ``backoff``, ``changed``,
    ``recovered`` or ``failed``. ``invoke`` replaces SSH, ``api`` the API
    probe, ``apply`` the lifecycle call ``apply(operation)`` and ``tunnel``
    Node A's ``control`` report.

    Each slot's active deployment is checked in turn (``check_one``). With one
    active deployment the result is its own; with several, ``deployments``
    holds each result and ``state`` the first of ``PRECEDENCE`` that one of
    them reached.
    """
    found = controller.active_deployments()
    if not found:
        return _outcome("idle", "No model deployment is active")
    results = [check_one(directory, now=now, invoke=invoke, api=api, apply=apply, tunnel=tunnel)
               for _, directory in found]
    if len(results) == 1:
        return results[0]
    state = next(state for state in PRECEDENCE if any(result["state"] == state for result in results))
    return {"state": state, "summary": "; ".join(f"{Path(result['deployment']).name}: {result['summary']}"
                                                 for result in results),
            "deployment": None, "deployments": results}


def check_one(directory, *, now=time.time, invoke=None, api=None, apply=None, tunnel=None):
    """``check`` of one active deployment."""
    invoke = invoke or discovery.ssh
    directory = Path(directory).resolve()
    record = record_of(load(repair=True), directory)
    if not record["enabled"]:
        return _outcome("off", "Automatic recovery is off for this deployment", directory)
    if record["stopped"]:
        return _outcome("stopped", f"{_stopped_text(record)}; {UP} turns it back on", directory)
    lock = installer.read(directory / "deployment.lock.json")
    if not supported(lock):
        return _outcome("unsupported", f"Automatic recovery covers Compose deployments; this one uses {lock.get('backend')}",
                        directory)
    if _busy():
        return _outcome("busy", "Another SparkRing operation is running", directory)
    state = saved_state(directory)
    if record.get("attempt"):
        record = _interrupted(directory, record, state, now())
        if record["stopped"]:
            return _outcome("stopped", _stopped_text(record), directory)
    if not may_act(state, record):
        if record.get("pending") or record.get("waiting"):
            update(directory, pending=None, waiting=None)
        return _outcome("inactive", "The last model operation is not a completed up; recovery waits for one", directory)
    tunnel = tunnel if tunnel is not None else _local_tunnel()
    finding = observe(directory, lock, record, state, invoke=invoke, api=api, tunnel=tunnel)
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
    changes["pending"] = pending = {"count": pending.get("count", 0) + 1, "since": pending.get("since") or t,
                                    "state": finding["state"]}
    if record.get("next_attempt_at") and t < record["next_attempt_at"]:
        update(directory, **changes)
        return _outcome("backoff", finding["summary"] + "; the next attempt waits for its backoff", directory,
                        finding=shown)
    if not _confirmed(finding, pending, t):
        update(directory, **changes)
        return _outcome("confirming", finding["summary"] + "; a later check confirms it", directory, finding=shown)
    restarts = recent_restarts(record, t)
    if len(restarts) >= RESTART_LIMIT:
        reason = (f"the model stopped again after {len(restarts)} restarts within {RESTART_WINDOW // 3600} hours; "
                  "find the cause before restarting it")
        update(directory, **{**changes, "pending": None}, stopped=True, stop_reason=reason)
        return _outcome("stopped", "Stopped: " + reason, directory, finding=shown)
    update(directory, **changes)
    return attempt(directory, lock, state, finding, now=now, apply=apply, invoke=invoke, api=api, tunnel=tunnel)


def _interrupted(directory, record, state, t):
    """Record an attempt that stopped without recording its end, for example because Node A restarted."""
    prior = record["attempt"]
    first = prior.get("generation") or 0
    # The install lock kept other operations out while it ran, and every
    # manual operation forgets it, so a state within its generations is its own.
    ours = state.get("operation") in ("up", "down") and first <= (state.get("generation") or 0) <= first + 2
    failures = record.get("failures", 0) + 1
    return update(directory, attempt=None, failures=failures, stopped=failures >= FAILURE_LIMIT,
                  stop_reason=f"{failures} failed attempts in a row" if failures >= FAILURE_LIMIT else None,
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


def error_text(failure):
    """One line, at most ERROR_TEXT characters, that says why a lifecycle call failed.

    The command of a retained source's subprocess holds that source's code,
    so a failed subprocess is named by its exit status; its own output is in
    the installation log.
    """
    if isinstance(failure, subprocess.CalledProcessError):
        return (f"the deployment's own source exited with status {failure.returncode}; "
                "sudo sparkring logs --details shows its output")
    return (str(failure).strip().splitlines() or [type(failure).__name__])[-1][:node.ERROR_TEXT]


def attempt(directory, lock, state, finding, *, now=time.time, apply=None, invoke=None, api=None, tunnel=None):
    """Start the model again under the install lock; record the result and the backoff after a failure.

    Under the lock it reads the record, the active deployment and the saved
    state again and observes the deployment again: a run that ``recover off``,
    another operation or a recovered model overtook does nothing.
    """
    cache = controller.STATE / "retained-sources"
    apply = apply or (lambda operation: retained_source.apply(directory, operation, cache=cache))
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(process_lock.hold(controller.STATE / "install.lock"))
        except ValueError:
            return _outcome("busy", "Another SparkRing operation started", directory)
        record = record_of(load(repair=True), directory)
        if not record["enabled"]:
            return _outcome("off", "Automatic recovery was turned off during the check", directory)
        if record["stopped"]:
            return _outcome("stopped", _stopped_text(record), directory)
        if not is_active(directory):
            update(directory, pending=None)
            return _outcome("changed", "Another deployment became active during the check", directory)
        if _state_key(saved_state(directory)) != _state_key(state):
            update(directory, pending=None)
            return _outcome("changed", "The deployment changed during the check; the next check observes it again",
                            directory)
        # Observed again under the lock: the mesh guard and the model state must still hold.
        finding = observe(directory, lock, record, state, invoke=invoke or discovery.ssh, api=api, tunnel=tunnel)
        shown = {key: finding[key] for key in ("state", "summary", "next_action", "details")}
        if finding["action"] is None:
            update(directory, finding=shown, pending=None)
            return _outcome("changed", "Observed again before acting: " + finding["summary"], directory, finding=shown)
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
                error = error_text(failure)
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
               generation=after.get("generation"), boots=boot_ids(documents),
               restarts=[*recent_restarts(record, finished), finished])
        return _outcome("recovered", "The model started again", directory)
    failures = record.get("failures", 0) + 1
    update(directory, attempt=None, last=last, failures=failures, stopped=failures >= FAILURE_LIMIT,
           stop_reason=f"{failures} failed attempts in a row" if failures >= FAILURE_LIMIT else None,
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


def status_lines(record, *, now=time.time, timer=None, state=None, backend="compose"):
    """The lines ``sparkring status`` prints about automatic recovery of one deployment.

    ``timer`` is ``timer_enabled()``: False adds that the timer does not run.
    ``state`` is the deployment's saved state: when it is not a completed up,
    recovery is idle until the next one. ``backend`` other than ``compose``
    says recovery does not restart the deployment.
    """
    if backend != "compose":
        return [f"Automatic recovery: not available for this deployment ({backend}); restart it by hand"]
    if record is None:
        return ["Automatic recovery: state unreadable; use sudo sparkring status"]
    if not record.get("enabled", True):
        return ["Automatic recovery: off | turn on: sudo sparkring recover on"]
    if record.get("stopped"):
        lines = [f"Automatic recovery: {_stopped_text(record).lower()} | {UP} turns it back on"]
    elif timer is False:
        lines = [f"Automatic recovery: on, but {TIMER} is not enabled | sudo sparkring recover on enables it"]
    elif state is not None and not may_act(state, record):
        lines = ["Automatic recovery: on; idle until the next sudo sparkring up --execute or sudo sparkring install"]
    else:
        lines = ["Automatic recovery: on"]
    waiting = record.get("waiting")
    if waiting:
        lines.append(f"  waiting for {', '.join(waiting.get('hosts') or ['a Spark'])} since {clock(waiting['since'])}")
    last = record.get("last")
    if last:
        verb = "start" if last.get("action") == "start" else "restart"
        lines.append(f"  last attempt: {verb} at {clock(last['started_at'])}, {RESULTS.get(last.get('result'), last.get('result'))}"
                     + (f": {str(last['error'])[:node.ERROR_TEXT]}" if last.get("error") else ""))
    upcoming = record.get("next_attempt_at")
    if not record.get("stopped"):
        if upcoming and upcoming > now():
            lines.append(f"  next attempt: not before {clock(upcoming)}")
        elif record.get("pending"):
            lines.append("  next attempt: at a later check, a minute or more from the last one, if the model still "
                         "does not serve")
    return lines


def describe():
    """``sparkring recover status``: each active deployment and its record.

    With one active deployment the document holds its ``deployment`` and
    ``record``; with several, ``deployments`` holds one such pair each.
    """
    found = controller.active_deployments()
    if not found:
        return {"deployment": None, "record": None, "lines": ["No model deployment is active."]}
    documents, lines = [], []
    for _, directory in found:
        directory = Path(directory).resolve()
        record = record_of(load(), directory)
        backend = installer.read(directory / "deployment.lock.json").get("backend")
        lines += ["Deployment: " + str(directory),
                  *status_lines(record, state=saved_state(directory), timer=timer_enabled(), backend=backend)]
        finding = record.get("finding")
        if finding and record.get("checked_at"):
            lines.append(f"Last check at {clock(record['checked_at'])}: {finding['summary']}"
                         + (f" | next: {finding['next_action']}" if finding.get("next_action") else ""))
            lines += ["  " + line for line in finding.get("details") or []]
        documents.append({"deployment": str(directory), "record": record})
    if len(documents) == 1:
        return {**documents[0], "lines": lines}
    return {"deployment": None, "record": None, "deployments": documents, "lines": lines}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sparkring recover",
                                     description="Automatic restart of the active models when a Spark stops serving.")
    parser.add_argument("action", nargs="?", choices=("status", "on", "off"),
                        help="status (default), or turn automatic recovery on or off for the active deployments: "
                             "the whole ring's or pair's, and each ring half's")
    parser.add_argument("--auto", action="store_true",
                        help="check once and restart the model if needed; sparkring-recover.timer runs this")
    parser.add_argument("--json", action="store_true", help="print one JSON document")
    args = parser.parse_args(argv)
    if args.auto and args.action:
        parser.error("--auto takes no action")
    try:
        if args.auto:
            result = check()
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                for item in result.get("deployments") or [result]:
                    print(f"{item['state']}: {item['summary']}")
            return 1 if result["state"] == "failed" else 0
        if args.action in ("on", "off"):
            # Applies to every slot's active deployment: the whole cluster's and each arc's.
            found = [directory for _, directory in controller.active_deployments()] or [None]
            documents, lines = [], []
            for directory in found:
                record = set_enabled(args.action == "on", directory=directory)
                directory = Path(directory or controller.active_deployment()).resolve()
                documents.append({"deployment": str(directory), "record": record})
                lines += ([f"Deployment: {directory}"] if len(found) > 1 else []) + status_lines(
                    record, state=saved_state(directory))
            result = ({**documents[0], "lines": lines} if len(documents) == 1 else
                      {"deployment": None, "record": None, "deployments": documents, "lines": lines})
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
