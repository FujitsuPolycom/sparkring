"""Automatic release of what older deployments leave on the Sparks.

Every ``sparkring install`` request that differs in package revision, image,
checkpoint or serving settings is a separate deployment
(``install_workflow.select_deployment``) with its own directory on Node A below
``/var/lib/sparkring/controller/deployments``. Once started, a deployment
leaves on every Spark its workspace (``/srv/sparkring/<cluster>/<profile>-i<identity>``:
the source checkout, the Compose and runtime-binding files and the checkpoint
receipt), its stopped model container and compile caches keyed by image and
checkpoint revision. After every successful ``sudo sparkring install`` and
``sudo sparkring up``, ``after_operation`` keeps a bounded set of deployments
and releases that data of the others.

Kept
----
A deployment is kept while any of these holds (``REASONS``):

- ``active``, ``rollback``, ``switching``: it is the active deployment
  (``active.json``), the rollback target or the candidate of an unfinished
  model switch (``transaction.json``; ``checkpoints.roles``).
- ``running``: its model container runs on a Spark.
- ``mesh``: a SparkRing mesh installed on a Spark uses its workspace
  (``storage.installed_meshes``). On a four-Spark ring the deployment that
  created the mesh holds the mesh's host marker and bundle in its workspace,
  and the mesh's model unit starts that deployment's container.
- ``unfinished``: its last operation (``state.json``) did not complete, or its
  state cannot be read; it may need recovery through its own receipts.
- ``started``: its last operation is a completed ``up``. It was started and not
  stopped since, and a stop or a repeated ``up`` reads its workspace.
- ``backend``: a managed GLM backend, whose own services own its containers.
- ``recent``: it is one of the ``retain`` most recent deployments of its
  profile that ran an operation, by the time of their last operation
  (``state.json``'s modification time). ``DEFAULT`` is 2: for each model, the
  deployment that runs or ran last and the one before it, usually the previous
  package revision or another checkpoint or serving setting of that model,
  start again without preparing a workspace, container or compile cache.

Released
--------
For every other deployment that ran an operation, on each Spark:

- its stopped model containers (``sr-<site>-r<rank>``, labelled with its lock
  ID); the ``create`` phase of its next ``up`` creates them again;
- its workspace; the ``source`` phase of its next ``up`` or installation
  creates it again from the deployment's ``source.bundle`` on Node A, and the
  ``model`` phase writes the checkpoint receipt again from the checkpoint
  directory's journal and path records, hashing only changed files;
- each compile cache that no kept deployment uses and no installer profile of
  the installed package references (class ``unreferenced`` of ``sudo sparkring
  storage``); the next start that uses it compiles and tunes again;
- each remainder of an interrupted ``sudo sparkring storage --release``.

Each Spark checks again before it removes anything
(``storage.release_batch_local``): a container must be stopped, carry the
deployment's label and not be the container that an installed mesh starts, and
each workspace and cache passes ``storage.release_local`` with the paths that
the kept deployments name there, so the installed-mesh, model-file, mount-point
and running-container checks of ``sudo sparkring storage --release`` apply.
Checkpoint directories, Docker images, the deployment directories on Node A
(lock, source bundle and receipts) and anything SparkRing's installer did not
create are never released.

A released deployment's directory on Node A records the release in
``released.json`` with the deployment's state generation (``release_record``).
While that state stands, ``sparkring down`` of the deployment reports that it
is stopped instead of verifying its containers through a workspace that is
gone, and later releases skip it. The next operation of the deployment starts
a new generation, so the record no longer applies.

Setting
-------
``retention.json`` on Node A holds the number of recent deployments kept per
profile, or ``off`` (``setting``). ``sudo sparkring storage
--retain-deployments N|off`` and the ``SPARKRING_RETAIN_DEPLOYMENTS``
preference of ``sudo sparkring install --env FILE`` save it.
"""
import concurrent.futures
import json
from pathlib import Path
import subprocess

from runtime.common import installer
from runtime.host import checkpoints, node, progress, settings, storage

DEFAULT = 2
SETTING_FILE = "retention.json"
SETTING_SCHEMA = "sparkring-retention/v1"
RELEASED_FILE = "released.json"
RELEASED_SCHEMA = "sparkring-deployment-release/v1"
RESULT_SCHEMA = "sparkring-retention-result/v1"
BATCH = ["sudo", "-n", "/usr/bin/sparkring", "node", "storage", "--release-batch"]
# Seconds that one Spark may take for its removals.
BATCH_TIMEOUT = 3600
# Why a deployment is kept, in the order the report lists them; ``recent`` names the setting's number.
REASONS = {"active": "active", "rollback": "rollback target", "switching": "unfinished model switch",
           "running": "model container running", "mesh": "holds an installed mesh's files",
           "unfinished": "last operation incomplete", "started": "started and not stopped",
           "backend": "managed GLM services", "recent": "{retain} most recent of its profile"}
# Last operations after which a deployment runs no container: the state a release record applies to.
STOPPED = ("down", "prepare")
# Refusals listed after the summary line; the report names the rest.
SHOWN_ERRORS = 5


# Setting ---------------------------------------------------------------------

def setting(state_root):
    """``(retain, text)``: recent deployments kept per profile, ``None`` when off, and the saved text.

    Without a saved setting the default applies. A setting file that cannot be
    read raises ``ValueError``, so that automatic release does not run on a
    guess.
    """
    path = Path(state_root) / SETTING_FILE
    try:
        text = installer.read(path)["retain_deployments"]
        return settings.retain_deployments(text), text
    except FileNotFoundError:
        return DEFAULT, str(DEFAULT)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"{path} cannot be read ({error}); automatic release does not run until sudo sparkring "
                         "storage --retain-deployments N|off saves a setting") from None


def save_setting(state_root, text):
    """Save the setting ``text`` (``off`` or a number); returns the parsed value."""
    retain = settings.retain_deployments(text)
    node.save(state_root, SETTING_FILE, {"schema": SETTING_SCHEMA,
                                         "retain_deployments": "off" if retain is None else str(retain)}, mode=0o600)
    return retain


def setting_text(retain):
    """The report's sentence for setting ``retain``."""
    if retain is None:
        return ("Automatic release is off; sudo sparkring storage --retain-deployments 2 turns it on.")
    kept = "only the deployments it always keeps" if retain == 0 else (
        "the most recent deployment" if retain == 1 else f"the {retain} most recent deployments")
    return (f"After each install and up, automatic release keeps {kept}" + ("" if retain == 0 else " of each profile")
            + "; sudo sparkring storage --retain-deployments N or off changes it.")


# Deployments -------------------------------------------------------------------

def _state(directory):
    """``(state, time)``: ``state.json`` of ``directory`` and its modification time.

    ``state`` is ``None`` when no operation ran and ``{"unreadable": True}``
    when the file cannot be read as a deployment state.
    """
    path = Path(directory) / "state.json"
    try:
        time = path.stat().st_mtime
    except FileNotFoundError:
        return None, None
    except OSError:
        return {"unreadable": True}, None
    try:
        state = installer.read(path)
    except (OSError, ValueError):
        return {"unreadable": True}, time
    if (not isinstance(state, dict) or type(state.get("generation")) is not int
            or not isinstance(state.get("complete"), bool)):
        return {"unreadable": True}, time
    return state, time


def release_record(directory, state=None):
    """``released.json`` of ``directory`` while it applies to the deployment's state, else ``None``.

    It applies while the state is the completed ``down`` or ``prepare`` of the
    generation it records, so the deployment runs no container and no later
    operation created its workspace again.
    """
    state = _state(directory)[0] if state is None else state
    try:
        record = installer.read(Path(directory) / RELEASED_FILE)
    except (OSError, ValueError):
        return None
    if (isinstance(record, dict) and isinstance(state, dict) and state.get("complete") is True
            and state.get("operation") in STOPPED and type(record.get("generation")) is int
            and record["generation"] == state.get("generation")):
        return record
    return None


def deployments(state_root):
    """Node A's deployments as ``storage.deployments`` reads them, with the fields the policy reads.

    Adds ``profile`` (the deployment's name when the lock names none),
    ``backend``, ``state`` and ``time`` (``_state``), ``containers`` (each
    rank's host and model container name, ``installer.container_name``) and
    ``released`` (``release_record``).
    """
    found = []
    for record in storage.deployments(state_root):
        directory = Path(record["directory"])
        try:
            lock = installer.read(directory / "deployment.lock.json")
        except (OSError, ValueError):
            lock = {}
        selection = lock.get("selection") if isinstance(lock.get("selection"), dict) else {}
        backend = lock.get("backend", "compose")
        try:
            names = [{"host": row["host"], "rank": row["rank"],
                      "name": installer.container_name({**lock, "backend": backend}, row["rank"])}
                     for row in lock["site"]["ranks"]]
        except (KeyError, TypeError):
            names = []
        state, time = _state(directory)
        found.append({**record, "profile": selection.get("profile") or record["name"], "backend": backend,
                      "state": state, "time": time, "containers": names, "released": release_record(directory, state)})
    return found


# Policy ------------------------------------------------------------------------------

def _reasons(records, role, nodes, retain):
    """``{deployment name: [reason, ...]}`` in ``REASONS`` order; an empty list releases it."""
    found = {record["name"]: set() for record in records}
    by_id = {record["id"]: record for record in records}
    for record in records:
        if record["directory"] in role:
            found[record["name"]].add(role[record["directory"]])
        state = record["state"]
        if state is not None:
            if state.get("unreadable") or not state.get("complete"):
                found[record["name"]].add("unfinished")
            elif state.get("operation") == "up":
                found[record["name"]].add("started")
        if record["backend"] != "compose":
            found[record["name"]].add("backend")
    for node_entry in nodes:
        if "error" in node_entry:
            continue
        for entry in node_entry.get("model_containers") or ():
            if entry.get("running") and entry.get("deployment") in by_id:
                found[by_id[entry["deployment"]]["name"]].add("running")
        for item in node_entry.get("items") or ():
            if item["kind"] == "workspace" and item.get("meshes"):
                for record in records:
                    if record["workspace"] == item["path"] or record["id"] == item.get("deployment"):
                        found[record["name"]].add("mesh")
    groups = {}
    for record in records:
        if record["state"] is not None:
            groups.setdefault(record["profile"], []).append(record)
    for group in groups.values():
        for record in sorted(group, key=lambda value: (-(value["time"] or 0), value["name"]))[:retain]:
            found[record["name"]].add("recent")
    return {name: [code for code in REASONS if code in codes] for name, codes in found.items()}


def _releasable(item, host, kept, released):
    """Whether automatic release removes listed ``item`` from ``host``.

    ``kept`` are the deployments the policy keeps and ``released`` the lock
    IDs of those it releases. ``item`` must be one that ``sudo sparkring
    storage`` proposes for release (``storage.classify``'s ``release``), which
    excludes items of class ``installed`` or ``profile``, items with model
    files and items that running containers use. A workspace must belong to a
    released deployment, and no kept deployment may use the item.
    """
    if item["kind"] not in storage.RELEASED or not item.get("release"):
        return False
    if item["kind"] == "workspace" and item.get("deployment") not in released:
        return False
    return item["kind"] == "releasing" or not any(storage._uses(record, host, item["path"]) for record in kept)


def plan(records, role, nodes, retain):
    """What the policy keeps, and what it releases on each Spark.

    ``records`` are ``deployments``, ``role`` is ``checkpoints.roles`` and
    ``nodes`` are the Sparks' listings after ``storage.classify`` (none plans
    from Node A's records alone, without the ``running`` and ``mesh``
    reasons). Returns ``{"deployments": [{"name", "profile", "kept",
    "reasons", "started"}], "released": [name, ...], "nodes": [{"rank",
    "host", "hostname", "containers", "paths", "frees_bytes", "complete"} or
    {"rank", "host", "error"}]}``; ``containers`` are ``{"deployment",
    "name", "bytes"}`` and ``paths`` ``{"path", "kind", "deployment",
    "frees_bytes"}``, workspaces first.
    """
    reasons = _reasons(records, role, nodes, retain)
    kept = [record for record in records if reasons[record["name"]]]
    released = {record["id"] for record in records if not reasons[record["name"]] and record["state"] is not None}
    by_id = {record["id"]: record for record in records}
    actions = []
    for node_entry in nodes:
        where = {"rank": node_entry.get("rank"), "host": node_entry.get("host")}
        if "error" in node_entry:
            actions.append({**where, "error": node_entry["error"]})
            continue
        host = node_entry.get("host")
        containers = []
        for entry in node_entry.get("model_containers") or ():
            record = by_id.get(entry.get("deployment"))
            if record is None or record["id"] not in released or entry.get("running"):
                continue
            if entry["name"] in {value["name"] for value in record["containers"] if value["host"] == host}:
                containers.append({"deployment": record["id"], "name": entry["name"], "bytes": entry.get("bytes")})
        order = {"workspace": 0, "cache": 1, "releasing": 2}
        paths = [{"path": item["path"], "kind": item["kind"], "deployment": item.get("deployment"),
                  "frees_bytes": item.get("frees_bytes"), "complete": item.get("complete")}
                 for item in sorted(node_entry.get("items") or (), key=lambda value: (order.get(value["kind"], 3),
                                                                                     value["path"]))
                 if _releasable(item, host, kept, released)]
        sizes = [entry["bytes"] for entry in containers] + [entry["frees_bytes"] for entry in paths]
        actions.append({**where, "hostname": node_entry.get("hostname"), "containers": containers,
                        "paths": [{key: value for key, value in entry.items() if key != "complete"} for entry in paths],
                        "frees_bytes": sum(size for size in sizes if size), "complete":
                        all(size is not None for size in sizes) and all(entry["complete"] is not False for entry in paths)})
    listed = [{"name": record["name"], "profile": record["profile"], "kept": bool(reasons[record["name"]]),
               "reasons": reasons[record["name"]], "started": record["state"] is not None}
              for record in sorted(records, key=lambda value: (value["profile"], -(value["time"] or 0), value["name"]))]
    return {"deployments": listed, "released": sorted(record["name"] for record in records if record["id"] in released),
            "nodes": actions}


def reason_text(code, retain):
    return REASONS[code].format(retain=retain)


def view(state_root, records, role, nodes):
    """The policy as ``sudo sparkring storage`` reports it, with what the next release would free.

    ``nodes`` are the classified listings of that report. Returns ``{"setting",
    "retain", "error"}`` and, unless automatic release is off or its setting
    cannot be read, ``plan``'s fields with ``frees_bytes`` and ``complete``
    over every Spark.
    """
    try:
        retain, text = setting(state_root)
    except ValueError as error:
        return {"setting": None, "retain": None, "error": str(error)}
    result = {"setting": text, "retain": retain, "error": None}
    if retain is None:
        return result
    decided = plan(records, role, nodes, retain)
    pending = [entry for entry in decided["nodes"] if "error" not in entry]
    return {**result, **decided, "frees_bytes": sum(entry["frees_bytes"] for entry in pending),
            "complete": len(pending) == len(decided["nodes"]) and all(entry["complete"] for entry in pending)}


def describe(retention):
    """The report's lines for ``view``'s result."""
    if retention.get("error"):
        return ["Automatic release: " + retention["error"]]
    retain = retention["retain"]
    if retain is None:
        return [setting_text(None)]
    listed = retention["deployments"]
    kept = [entry for entry in listed if entry["kept"]]
    released = [entry for entry in listed if entry["started"] and not entry["kept"]]
    idle = len(listed) - len(kept) - len(released)
    lines = [f"Deployments on Node A: {len(listed)}" + (f", {idle} never started" if idle else "")]
    width = max([len(entry["name"]) for entry in kept] + [0])
    for entry in kept:
        lines.append(f"    kept  {entry['name']:<{width}}  "
                     + ", ".join(reason_text(code, retain) for code in entry["reasons"]))
    holding = sum(1 for entry in retention["nodes"] for _ in entry.get("containers") or ()) + sum(
        1 for entry in retention["nodes"] for _ in entry.get("paths") or ())
    if released and holding:
        noun = "deployment" if len(released) == 1 else "deployments"
        lines.append(f"    The next install or up releases the containers, workspaces and caches of {len(released)} "
                     f"older {noun}: {storage._measured(retention['frees_bytes'], retention['complete'])}")
    elif holding:
        lines.append("    The next install or up releases unused compile caches: "
                     + storage._measured(retention["frees_bytes"], retention["complete"]))
    elif released:
        noun = "deployment holds" if len(released) == 1 else "deployments hold"
        lines.append(f"    The {len(released)} older {noun} nothing on the Sparks.")
    lines.append(setting_text(retain))
    return lines


# Release ---------------------------------------------------------------------------

def _pending(records, role, retain):
    """The deployments that Node A's records alone would release and that hold data a release did not remove."""
    decided = plan(records, role, [], retain)
    names = set(decided["released"])
    return [record for record in records if record["name"] in names
            and not (record["released"] and record["released"].get("complete"))]


def release(state_root, invoke, retain, *, write=print):
    """Release on every Spark what the policy does not keep; the caller holds ``install.lock``.

    Nothing is listed or removed when Node A's records alone release no
    deployment that still holds data. A Spark that cannot be listed, or Sparks
    running different package revisions, stop the release before anything is
    removed. Each Spark then removes its share (``storage.release_batch_local``);
    refusals are reported per entry. Writes ``released.json`` for every
    released deployment and prints ``summary`` lines. Returns the result
    document (``RESULT_SCHEMA``).
    """
    state_root = Path(state_root)
    result = {"schema": RESULT_SCHEMA, "state": "nothing", "retain": retain, "released": [], "freed_bytes": 0,
              "nodes": [], "errors": []}
    cluster = checkpoints._cluster(state_root)
    if cluster is None:
        return result
    records, role = deployments(state_root), checkpoints.roles(state_root)
    if not _pending(records, role, retain):
        return result
    nodes = storage._survey(checkpoints._hosts(cluster), cluster["name"], records, invoke, measure=[])
    failed = [entry for entry in nodes if "error" in entry]
    if failed:
        raise ValueError("; ".join(f"Node {entry['rank']} {entry['host']} could not be listed: {entry['error']}"
                                   for entry in failed) + "; nothing was released")
    revisions = {entry.get("package_revision") for entry in nodes}
    if len(revisions) > 1:
        raise ValueError("the Sparks run different SparkRing package revisions; nothing was released")
    storage.classify(nodes, records, role, storage.profile_references())
    decided = plan(records, role, nodes, retain)
    kept = [record for record in records if any(entry["kept"] and entry["name"] == record["name"]
                                                for entry in decided["deployments"])]

    def one(action):
        host = action["host"]
        in_use = sorted({value for record in kept for value in storage._on(record, host)[0]})
        opaque = sorted({value for record in kept for value in storage._on(record, host)[1]})
        request = storage._request(host, cluster["name"], records, in_use=in_use, opaque=opaque,
                                   containers=[{"deployment": entry["deployment"], "name": entry["name"]}
                                               for entry in action["containers"]],
                                   paths=[entry["path"] for entry in action["paths"]])
        try:
            answer = json.loads(invoke(host, list(BATCH), data=json.dumps(request), timeout=BATCH_TIMEOUT))
            if not isinstance(answer, dict):
                raise ValueError("unexpected answer from sparkring node storage --release-batch")
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
            text = str(error).strip()
            return {**action, "error": text.splitlines()[-1][:300] if text else type(error).__name__}
        return {**action, "results": answer}

    work = [action for action in decided["nodes"] if action["containers"] or action["paths"]]
    answers = []
    if work:
        with progress.step("Release what older deployments hold on the Sparks", phase="release-older"), \
                concurrent.futures.ThreadPoolExecutor(max_workers=len(work)) as pool:
            answers = list(pool.map(one, work))
    by_id = {record["id"]: record for record in records}
    removed, incomplete, caches, remainders = set(), set(), set(), set()
    for action in answers:
        name = checkpoints._node_name(action)
        owners = {entry["path"]: entry for entry in action["paths"]}
        if "error" in action:
            result["errors"].append(f"{name}: {action['error']}")
            incomplete.update(entry["deployment"] for entry in action["containers"])
            incomplete.update(entry["deployment"] for entry in action["paths"] if entry["kind"] == "workspace")
            result["nodes"].append({key: action[key] for key in ("rank", "host", "hostname", "error")})
            continue
        answer = action["results"]
        for entry in answer.get("containers") or ():
            if "error" in entry:
                result["errors"].append(f"{name}: {entry['error']}")
                incomplete.add(entry.get("deployment"))
            elif entry.get("state") == "removed":
                removed.add(entry["deployment"])
                result["freed_bytes"] += entry.get("freed_bytes") or 0
        for entry in answer.get("paths") or ():
            planned = owners.get(entry.get("path"), {})
            if "error" in entry:
                result["errors"].append(f"{name}: {entry['error']}")
                if planned.get("kind") == "workspace":
                    incomplete.add(planned.get("deployment"))
                continue
            if entry.get("state") != "released":
                continue
            result["freed_bytes"] += entry.get("freed_bytes") or 0
            if planned.get("kind") == "workspace":
                removed.add(planned.get("deployment"))
            elif planned.get("kind") == "cache":
                caches.add(entry["path"])
            elif planned.get("kind") == "releasing":
                remainders.add(entry["path"])
        result["nodes"].append({"rank": action["rank"], "host": action["host"], "hostname": action["hostname"],
                                **answer})
    for record in records:
        if record["name"] not in decided["released"]:
            continue
        node.save(record["directory"], RELEASED_FILE,
                  {"schema": RELEASED_SCHEMA, "generation": record["state"].get("generation"),
                   "complete": record["id"] not in incomplete,
                   "removed": record["id"] in removed or bool(record["released"] and record["released"].get("removed"))},
                  mode=0o600)
    result["released"] = sorted(by_id[value]["name"] for value in removed if value in by_id)
    result.update(caches=len(caches), remainders=len(remainders),
                  state="released" if removed or caches or remainders else "nothing")
    for line in summary(result):
        write(line)
    return result


def summary(result):
    """The lines that report a release: one summary line, then up to ``SHOWN_ERRORS`` refusals."""
    count, size = len(result["released"]), storage._size(result["freed_bytes"])
    lines = []
    if count:
        noun = "deployment's" if count == 1 else "deployments'"
        lines.append(f"Released {count} older {noun} containers, workspaces and caches: {size}")
    elif result.get("caches"):
        lines.append(f"Released {result['caches']} unused compile cache{'' if result['caches'] == 1 else 's'}: {size}")
    elif result.get("remainders"):
        lines.append(f"Released the remainders of interrupted storage releases: {size}")
    errors = result["errors"]
    if errors:
        lines.append(("Some data stays on the Sparks" if lines else "Older deployments were not released")
                     + "; sudo sparkring storage lists it:")
        lines.extend("  " + error for error in errors[:SHOWN_ERRORS])
        if len(errors) > SHOWN_ERRORS:
            lines.append(f"  and {len(errors) - SHOWN_ERRORS} more")
    return lines


def after_operation(state_root, invoke, *, preference=None, write=print):
    """Automatic release after a successful ``install`` or ``up``; the caller holds ``install.lock``.

    ``preference`` is the ``SPARKRING_RETAIN_DEPLOYMENTS`` value of the
    installation's preferences file, saved before it applies. Never raises:
    the model operation before it completed, so a failure is printed and
    returned with state ``failed``.
    """
    try:
        if preference:
            save_setting(state_root, preference)
        retain, _ = setting(state_root)
        if retain is None:
            return {"schema": RESULT_SCHEMA, "state": "off"}
        return release(state_root, invoke, retain, write=write)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        write(f"Older deployments were not released: {error}. sudo sparkring storage lists what they hold.")
        return {"schema": RESULT_SCHEMA, "state": "failed", "message": str(error)}
