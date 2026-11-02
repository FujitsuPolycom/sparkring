"""Root actions around ``sparkring dashboard`` on Node A: its site file, its service and setup's plan line.

The serving collector (``runtime.host.dashboard``) starts no process and needs no privilege.
The actions here run as root, in the operator's commands:

- ``write_site`` turns the controller's deployment records (root only) into the site file
  (``sparkring-dashboard-site/v1``, 0644) that the collector reads.
- ``refresh_site`` is called by ``sparkring install``, ``up``, ``down``, setup and automatic
  recovery when they finish, so the page follows every change of deployment records. It writes
  the site file, then asks the running service to re-read it (``reload_service``). It never
  raises: a failure costs one line on stderr and the command carries on.
- ``after_setup`` enables and restarts ``sparkring-dashboard.service`` at the end of setup.
- ``switch`` is ``sudo sparkring dashboard enable|disable``. ``disable`` records DISABLED_PATH,
  which the unit's condition, the package's postinst and ``after_setup`` respect, so the
  dashboard stays off across reboots, upgrades and setups.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

from runtime.host.dashboard import (
    CONTAINER_RE,
    DEFAULT_PORT,
    FABRIC_PATH,
    HISTORY_VALUES,
    SITE_PATH,
    SITE_SCHEMA,
    NODE_REPORTS_PATH,
    QUIET_PATH,
    quiet_state,
    service_url,
)

DISABLED_PATH = Path("/etc/sparkring/dashboard.disabled")
UNIT = "sparkring-dashboard.service"


def refresh_nodes(*, state_root=None, output=None, invoke=None, quiet_path=None, now=time.time):
    """Cache every recorded Spark's public agent facts for the unprivileged dashboard.

    Called by the existing recovery timer, including when no deployment is
    active. Reads cached node status through recovery's existing SSH client,
    never refreshes an agent or runs Docker. Quiet mode and dashboard disable
    suppress these additional reads. Returns full documents only in memory so
    recovery can reuse its node-status reads during the same timer run.
    """
    from runtime.host import controller, host_load, node, recovery
    state_root = Path(state_root or controller.STATE)
    if DISABLED_PATH.exists() or quiet_state(Path(quiet_path or QUIET_PATH), now())["quiet"]:
        return {}
    try:
        hosts = host_load.recorded_targets(state_root)[:256]
        if not hosts:
            return {}
        output = Path(output or NODE_REPORTS_PATH)
        # The service reads this public directory; process_lock's default is private.
        output.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        from runtime.common import process_lock
        # Serialize collection and replacement: a slower timer must not overwrite
        # a newer cache, or discard retained observations after a transient failure.
        with process_lock.hold(output.with_suffix(".lock")):
            documents = recovery.node_documents(hosts, invoke=invoke)
            public = cached_node_rows(state_root, output, documents, now())
            node.save(output.parent, output.name, {"schema": "sparkring-dashboard-nodes/v1", "written_at": now(), "nodes": public})
        return documents
    except (OSError, ValueError, KeyError, TypeError) as error:
        _stderr(f"Dashboard: agent facts could not be cached ({type(error).__name__})")
        return {}


def cached_node_rows(state_root, output, documents, checked_at):
    """Project public facts by setup identity, retaining failed reads at their original age."""
    from runtime.host.dashboard import read_json
    from runtime.host.node_containers import MAX_CONTAINERS, thinking_kwargs

    def read(path):
        try:
            return read_json(path)
        except (OSError, ValueError):
            return {}

    positions = read(state_root / "fabric.json").get("positions", [])
    identities = {p["node_id"]: p for p in positions if p.get("node_id")}
    previous = {d["node_id"]: d for d in read(output).get("nodes", []) if d.get("node_id")}
    targets = {}
    for host in documents:
        address = host.rsplit("@", 1)[-1].strip("[]")
        matches = [p["node_id"] for p in identities.values()
                   if address in ((p.get("management") or {}).get("address"), p.get("hostname"), p.get("lan_address"))]
        if len(matches) == 1:
            targets[host] = matches[0]
    fields = ("name", "image", "image_id", "managed", "engine", "served_model", "tp", "pp", "dp", "dcp",
              "nnodes", "node_rank", "api_port", "api_bind", "master_addr", "master_port", "headless", "dp_addr", "dp_port", "dp_rank")
    public = {}

    def project(document):
        from runtime.host.dashboard import spark_cached_details
        row = {k: document[k] for k in ("node_id", "hostname", "observed_at", "interval_seconds", "gpu_compute")
               if k in document and isinstance(document[k], (str, int, float, bool))}
        row.update(spark_cached_details(document))
        # Publish only bounded counters; agent documents can contain private fields.
        from runtime.host.dashboard_rates import number, plugin_fresh
        telemetry = document.get("telemetry")
        row["telemetry"] = {}
        for deployment, snapshot in list(telemetry.items())[:128] if isinstance(telemetry, dict) else []:
            if not isinstance(deployment, str) or not 0 < len(deployment) <= 256:
                continue
            if plugin_fresh(snapshot, None, checked_at):
                row["telemetry"][deployment] = {
                    k: snapshot[k] for k in (*HISTORY_VALUES, "at", "monotonic") if number(snapshot.get(k))
                } | {"schema": "sparkring-stats/v1", "run_id": snapshot["run_id"],
                     "step_timing": bool(snapshot.get("step_timing"))}
        containers = document.get("model_containers")
        row["model_containers"] = []
        for source in containers[:MAX_CONTAINERS] if isinstance(containers, list) else []:
            if not isinstance(source, dict):
                continue
            target = {k: (source[k][:512] if isinstance(source[k], str) else source[k]) for k in fields
                      if k in source and isinstance(source[k], (str, int, float, bool))}
            kwargs = thinking_kwargs(source.get("default_chat_template_kwargs"))
            if kwargs is not None:
                target["default_chat_template_kwargs"] = kwargs
            row["model_containers"].append(target)
        return row

    # Setup positions remain visible even if a response has no identity, or the
    # cluster's host list temporarily lacks a target. No list index is an identity.
    for node_id, position in identities.items():
        public[node_id] = {**project(previous.get(node_id, {})), "node_id": node_id,
                           "hostname": position.get("hostname"), "position": position["position"],
                           "status_read": "unavailable", "checked_at": checked_at}
    for host, document in documents.items():
        if not isinstance(document, dict):
            continue
        node_id = document.get("node_id")
        expected = targets.get(host)
        if expected and node_id != expected:
            public[expected]["status_read"] = "identity-mismatch" if node_id else document.get("state", "unavailable")
            continue
        if not node_id or (identities and node_id not in identities):
            continue
        # A stale agent response must not replace a newer observation already saved.
        prior = previous.get(node_id, {})
        observed = document.get("observed_at")
        prior_at = prior.get("observed_at")
        selected = prior if isinstance(prior_at, (int, float)) and (
            not isinstance(observed, (int, float)) or prior_at > observed) else document
        public[node_id] = {**project(selected), "status_read": document.get("state", "ok"), "checked_at": checked_at}
        if node_id in identities:
            public[node_id]["position"] = identities[node_id]["position"]
    return list(public.values())


def operation_times(directory: Path) -> dict:
    """Project completed lifecycle events, never record modification/check times."""
    from runtime.common import installer
    from runtime.host.dashboard_rates import number
    try:
        state = installer.read(directory / "state.json")
        operation = state.get("operation")
        if not state.get("complete") or operation not in ("up", "down"):
            return {}
        completed = state.get("completed_at")
        if completed is None and isinstance(state.get("receipt"), str):
            receipt_path = (directory / state["receipt"]).resolve()
            if not receipt_path.is_relative_to(directory.resolve() / "operations"):
                return {}
            receipt = installer.read(receipt_path)
            completed = receipt.get("completed_at") if receipt.get("complete") else None
        if number(completed) and completed > 0:
            return {"stopped_at" if operation == "down" else "serving_at": completed}
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return {}


def deployment_row(directory, slot=None, size=None):
    """Public deployment fields shared by site export and active installation progress.

    A record the installed package cannot render again (its checkpoint or
    profile is no longer listed) takes its API and model from its own
    rank-0 container record (``installer.recorded_connection``).
    """
    from runtime.common import installer
    from runtime.host import api_endpoint
    from runtime.host import placement as placements
    lock = installer.read(Path(directory) / "deployment.lock.json")
    try:
        connection = installer.connection(lock)
    except (ValueError, KeyError):
        connection = installer.recorded_connection(directory, lock)
    host = urllib.parse.urlsplit(connection["api_url"]).hostname
    positions = list(slot) if slot else list(range(size or len(lock["site"]["ranks"])))
    shown = api_endpoint.recorded(directory)
    containers = []
    try:
        for row in installer.containers(lock):
            if CONTAINER_RE.fullmatch(row["name"]):
                containers.append({"position": positions[row["rank"]], "name": row["name"]})
    except (ValueError, KeyError, TypeError, IndexError):
        containers = []
    return {"id": Path(directory).name, "deployment_id": lock["id"], "profile": placements.profile_of(directory),
                 "checkpoint": lock["selection"].get("target_variant"),
                 "serving": {k: v for k, v in (lock.get("serving") or {}).items()
                             if k in ("reasoning_effort", "thinking")},
                 "model": connection["model"], "positions": positions, "api_host": host,
                 "api_port": connection["port"],
                 "display_url": api_endpoint.url(connection["api_url"], shown) if shown else None,
                 "stopped": placements.stopped(directory), "containers": containers,
                 **operation_times(Path(directory))}


def site_from_records(state_root: Path) -> dict:
    """The sparkring-dashboard-site/v1 document of every slot's active deployment on this controller.

    Each row: the deployment's directory name as `id`, its profile, served model name, positions
    (the whole fabric or the slot's arc), the API address and port SparkRing's own checks use,
    the display URL when one is recorded, whether its last operation stopped it, and each rank's
    model container name (``installer.containers``). Checkpoint selection and the public
    thinking settings let the reader resolve defaults from the shipped thinking records."""
    from runtime.host import controller
    from runtime.host import placement as placements
    rows = []
    size = None
    try:
        size = controller.recorded_layout()["size"]
    except (ValueError, OSError, KeyError):
        pass
    for slot in placements.slots(state_root):
        directory = placements.recorded(state_root, slot)
        if not placements.exists(directory):
            continue
        try:
            rows.append(deployment_row(directory, slot, size))
        except (ValueError, KeyError, OSError, TypeError, IndexError) as error:
            # One record the controller cannot describe must not keep every other deployment off the page.
            _stderr(f"Dashboard: deployment {Path(directory).name} is not shown ({type(error).__name__}: {error})")
    from runtime.host import ring_name
    return {"schema": SITE_SCHEMA, "written_at": time.time(), "deployments": rows,
            "display_name": ring_name.read(state_root)}


def external_rows(path) -> list:
    """The endpoints the external list names (dashboard.read_external); [] without the file."""
    from runtime.host import dashboard
    return dashboard.read_external(path)


def write_external(path, rows) -> None:
    """Write the external list file with ``rows``, world-readable, replacing it atomically."""
    from runtime.host import dashboard, node
    path = Path(path)
    document = {"schema": dashboard.EXTERNAL_SCHEMA, "endpoints": rows}
    node.save(path.parent, path.name, document, mode=0o644)


def external(args) -> int:
    """``sparkring dashboard external list|add|remove``: the endpoints shown as read-only cards.

    ``add`` validates the entry as the service does and replaces an entry of the same id; ``remove``
    drops one. The service re-reads the file within FILES_SECONDS; no restart. Prints the list."""
    from runtime.host import dashboard
    path = Path(args.file)
    rows = external_rows(path)
    if args.action == "add":
        entry = {"id": args.id, "label": args.label, "kind": args.kind, "host": args.host, "port": args.port}
        if getattr(args, "position", None) is not None:
            entry["position"] = args.position
        rows = dashboard.external_rows({"schema": dashboard.EXTERNAL_SCHEMA,
                                        "endpoints": [r for r in rows if r["id"] != args.id] + [entry]}, str(path))
        write_external(path, rows)
    elif args.action == "remove":
        if not any(r["id"] == args.id for r in rows):
            raise ValueError(f"no endpoint {args.id} in {path}")
        write_external(path, [r for r in rows if r["id"] != args.id])
        rows = dashboard.read_external(path)
    if not rows:
        print(f"No endpoints listed in {path}.")
    for row in rows:
        where = f" on position {row['position']}" if row.get("position") is not None else ""
        print(f"{row['id']}: {row['label']} ({row['kind']}) at http://{row['host']}:{row['port']}{where}")
    return 0


def write_site(output: Path | None = None) -> dict:
    """Write the site file from this controller's records; returns the document."""
    from runtime.host import controller, node
    site = site_from_records(controller.STATE)
    output = Path(output or SITE_PATH)
    node.save(output.parent, output.name, site, mode=0o644)
    return site


def _stderr(line):
    print(line, file=sys.stderr, flush=True)


def refresh_site(*, log=None, run=None, reload=True) -> bool:
    """Write the site file again after a command changed deployment records and ask the service to
    re-read it; never raises. True when the file was written.

    The dashboard shows the previous records until the next refresh or
    `sudo sparkring dashboard export` when the write fails. With ``reload=False``,
    the collector picks up the file at its existing cadence without a service call."""
    try:
        from runtime.common import process_lock
        from runtime.host import controller, operator_view
        # The group-readable operator view follows the same changes, so a
        # member of the sparkring group reads an up-to-date status.
        with process_lock.hold(controller.STATE / "publication.lock", wait=True):
            write_site()
            operator_view.publish()
    except Exception as error:  # noqa: BLE001 - the dashboard must never fail an installation
        (log or _stderr)(f"Dashboard: its site file was not refreshed ({type(error).__name__}: {error}); "
                         "sudo sparkring dashboard export writes it")
        return False
    if reload:
        reload_service(log=log, run=run)
    return True


def reload_service(*, log=None, run=None) -> bool:
    """Ask a running dashboard to re-read its host files at once (systemctl reload sends SIGHUP); a
    stopped service is left stopped. Never raises; True when systemctl accepted the request.

    The service also compares the site file's inode and time every second, so a failed request
    delays the page by at most that second."""
    if DISABLED_PATH.exists():
        return False
    try:
        _systemctl("--no-block", "try-reload-or-restart", UNIT, run=run)
        return True
    except Exception as error:  # noqa: BLE001 - the dashboard must never fail the command that wrote the file
        (log or _stderr)(f"Dashboard: {UNIT} was not asked to re-read its site file ({type(error).__name__}: "
                         f"{error}); it notices the new file within a second")
        return False


def _systemctl(*args, run=None):
    try:
        return (run or subprocess.run)(["systemctl", *args], check=True, capture_output=True, text=True, timeout=60)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"systemctl {' '.join(args)}: {(error.stderr or '').strip() or error}") from error
    except OSError as error:                 # no systemctl on this host
        raise RuntimeError(f"systemctl {' '.join(args)}: {error}") from error


def listen_address(api_address, management_address) -> str:
    """Where the dashboard on Node A listens: the LAN address setup records for Node A (where the
    model API of a model on Node A serves), else its management address. The collector's
    default_bind reads the same values from the fabric document."""
    return api_address or management_address or "127.0.0.1"


def plan_line(api_address, management_address) -> str:
    """The line setup's plan prints about the dashboard."""
    if DISABLED_PATH.exists():
        return "SparkRing dashboard: off (sudo sparkring dashboard enable turns it on)"
    address = listen_address(api_address, management_address)
    loopback = "" if address == "127.0.0.1" else f" and http://127.0.0.1:{DEFAULT_PORT}/ on Node A itself"
    return (f"SparkRing dashboard: http://{address}:{DEFAULT_PORT}/ on Node A{loopback}, read-only, LAN and loopback only "
            f"({UNIT}; sudo sparkring dashboard disable turns it off)")


def service_on(*, run=None) -> bool:
    """Whether the dashboard service on this host is enabled or running, and not turned off with
    `sparkring dashboard disable`. False where systemctl is absent or cannot answer."""
    if DISABLED_PATH.exists():
        return False
    for query in ("is-enabled", "is-active"):
        try:
            if (run or subprocess.run)(["systemctl", query, "--quiet", UNIT], check=False, capture_output=True,
                                       text=True, timeout=10).returncode == 0:
                return True
        except (OSError, subprocess.SubprocessError):
            return False
    return False


def page_url(*, run=None, fabric_path=None) -> str | None:
    """The address of the SparkRing dashboard page when its service is on (``service_on``), else None.

    The address follows the service's default bind rule (``dashboard.service_url``): Node A's LAN
    address from the fabric document, else loopback."""
    if not service_on(run=run):
        return None
    try:
        return service_url(Path(fabric_path or FABRIC_PATH)) + "/"
    except (OSError, ValueError, KeyError, TypeError):
        return None


def after_setup(*, log=None, run=None) -> bool:
    """Enable and restart the dashboard on Node A at the end of setup, then write its site file;
    never raises. A dashboard turned off with `sparkring dashboard disable` stays off."""
    say = log or _stderr
    started = False
    if not DISABLED_PATH.exists():
        try:
            _systemctl("enable", UNIT, run=run)
            _systemctl("restart", UNIT, run=run)
            started = True
        except Exception as error:  # noqa: BLE001 - setup has finished; the dashboard is optional
            say(f"Dashboard: {UNIT} did not start ({type(error).__name__}: {error}); "
                f"systemctl status {UNIT} shows why")
    refresh_site(log=say)
    return started


def export(output, *, run=None) -> int:
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise ValueError("Run sudo sparkring dashboard export on Node A: the deployment records are root's")
    site = write_site(Path(output))
    print(f"Wrote {output}: {len(site['deployments'])} deployment(s).")
    if Path(output) == Path(SITE_PATH):     # the service's own file, not a copy for another host
        reload_service(run=run)
    return 0


def switch(command, *, run=None) -> int:
    """`sudo sparkring dashboard enable|disable`: turn the service on or off for good."""
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise ValueError(f"Run sudo sparkring dashboard {command}")
    if command == "disable":
        DISABLED_PATH.parent.mkdir(parents=True, exist_ok=True)
        DISABLED_PATH.write_text("Written by sparkring dashboard disable: the dashboard stays off. "
                                 "sudo sparkring dashboard enable turns it on.\n", encoding="utf-8")
        _systemctl("disable", "--now", UNIT, run=run)
        print(f"Dashboard off: {UNIT} is stopped and disabled; setup and package upgrades leave it off.")
        return 0
    DISABLED_PATH.unlink(missing_ok=True)
    _systemctl("enable", "--now", UNIT, run=run)
    print(f"Dashboard on: {UNIT} serves http://NODE_A_LAN_ADDRESS:{DEFAULT_PORT}/ on Node A.")
    return 0
