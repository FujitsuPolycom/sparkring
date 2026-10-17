"""The operator view: the group-readable status document the read-only commands show.

``sparkring status`` and every command that refreshes the dashboard's site
file (``dashboard_admin.refresh_site``) also write
``/var/lib/sparkring/operator-status.json`` (``sparkring-operator-status/v1``,
mode 0640, group ``sparkring``, site addresses redacted by
``runtime.host.privilege.redact``). A member of the ``sparkring`` group reads
that document with ``sparkring status`` and ``sparkring recover status``
without sudo and cannot change anything: writing it is root's alone, and the
controller's own records, the administration network's configuration and the
deployment records are never made group-readable.

The document is a snapshot: its ``written_at`` says when a root command last
recorded it. Live observation (``sparkring status --refresh``, which signs in
to every Spark) stays root's.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from runtime.common import sparkcache
from runtime.host import privilege

PATH = Path("/var/lib/sparkring/operator-status.json")
SCHEMA = "sparkring-operator-status/v1"
# Fields of the host agent's observation the operator view keeps; the rest
# (paths, tunnel detail) stays root's.
NODE_FIELDS = ("state", "next_action", "rank", "size", "hostname", "error", "warnings",
               "age_seconds", "containers", "model_ready")


def view(state_root: Path) -> dict:
    """The operator view of the controller state at ``state_root``; every address is redacted.

    Reads the records root can read: the cluster record for the plan and the
    fabric's size, each slot's active deployment lock for the served model and
    its placement, the recovery record and the host agent's observation. A
    record that cannot be read is left out; the command that publishes the
    view must never fail because one file is odd.
    """
    from runtime.common import installer
    from runtime.host import node, placement as placements, recovery
    from runtime.host import controller

    state_root = Path(state_root or controller.STATE)
    document: dict = {"schema": SCHEMA, "written_at": time.time()}
    try:
        observation = node.status()
        document["node"] = {key: observation[key] for key in NODE_FIELDS if key in observation}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        cluster = installer.read(state_root / "cluster.json")
        from runtime.host import ring_name
        document["cluster"] = {"name": ring_name.effective(state_root, cluster), "plan_id": cluster["plan"].get("id"),
                               "size": placements.layout_of(cluster)["size"]}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        recovery_document = recovery.load()
    except (OSError, ValueError):
        recovery_document = {"deployments": {}}
    deployments = []
    try:
        slots = placements.slots(state_root)
    except (OSError, ValueError):
        slots = []
    for slot in slots:
        try:
            recorded = placements.recorded(state_root, slot)
            if recorded is None:
                continue
            directory = Path(recorded)
            if not placements.exists(directory):
                continue
            lock = installer.read(directory / "deployment.lock.json")
            try:
                connection = installer.connection(lock)
            except (ValueError, KeyError):
                # The installed package no longer lists the record's checkpoint or profile.
                connection = installer.recorded_connection(directory, lock)
            record = recovery.record_of(recovery_document, directory)
            deployments.append({
                "id": directory.name,
                "profile": placements.profile_of(directory),
                "model": connection["model"],
                "positions": list(slot) if slot else list(range(len(lock["site"]["ranks"]))),
                "api_port": connection["port"],
                "backend": lock.get("backend"),
                "sparkcache_size": sparkcache.deployment_budget(lock),
                **sparkcache.deployment_options(lock),
                "stopped": placements.stopped(directory),
                "recovery": {"enabled": bool(record.get("enabled")),
                             "failures": record.get("failures", 0),
                             "stopped": bool(record.get("stopped"))},
            })
        except (OSError, ValueError, KeyError, TypeError):
            continue
    document["deployments"] = deployments
    try:
        document["recovery_timer"] = recovery.timer_enabled()
    except (OSError, ValueError, RuntimeError, TypeError):
        document["recovery_timer"] = None
    public = privilege.redact(document)
    if document.get("cluster", {}).get("name"):
        # This intentionally public label must print verbatim, even if it resembles an address.
        public["cluster"]["name"] = document["cluster"]["name"]
    return public


def publish(state_root: Path | None = None, *, target: Path = PATH) -> bool:
    """Write the operator view; root only, never raises; False when skipped.

    Skipped when this process is not root, the host lacks the ``sparkring``
    group, or ``state_root`` holds no controller record (a Spark that is not
    the controller publishes nothing).
    """
    from runtime.host import controller

    if not privilege.running_as_root() or privilege.group_gid() is None:
        return False
    root = Path(state_root) if state_root is not None else controller.STATE
    state_root = root
    if not (state_root / "cluster.json").exists():
        return False
    try:
        document = view(state_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        import uuid
        temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".writing")
        temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)
        privilege.share(target, 0o640)
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def load(target: Path = PATH):
    """The last published operator view, or None when none was published or it is unreadable."""
    try:
        document = json.loads(Path(target).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) and document.get("schema") == SCHEMA else None


def lines(document: dict, *, now=time.time) -> list[str]:
    """What ``sparkring status`` and ``sparkring recover status`` print from a published view.

    Each line says the state as it was recorded; the header names the record's
    age so a reader never mistakes the snapshot for a live observation.
    """
    age = max(0, int(now() - document.get("written_at", now())))
    node = document.get("node") or {}
    result = [f"Operator view, recorded {age}s ago (an operator's sudo sparkring status refreshes it):"]
    result.append((node.get("state") or "unknown") + " | next: " + (node.get("next_action") or "sudo sparkring status"))
    for warning in node.get("warnings") or []:
        result.append("  warning: " + warning)
    if node.get("error"):
        result.append("  error: " + node["error"])
    cluster = document.get("cluster") or {}
    if cluster.get("name"):
        result.append("Ring: " + cluster["name"])
    if cluster.get("size"):
        result.append(f"Fabric: {cluster['size']} Sparks" + (f", plan {cluster['plan_id']}" if cluster.get("plan_id") else ""))
    for deployment in document.get("deployments") or []:
        placement = (", ".join(map(str, deployment["positions"]))
                     if deployment.get("positions") else "whole fabric")
        state = "stopped" if deployment.get("stopped") else "serving"
        result.append(f"{deployment['profile']} on {placement}: {state}"
                      f" | model {deployment['model']} | API port {deployment['api_port']}"
                      f" | transport {deployment.get('backend') or 'prepared'}")
        recovery_record = deployment.get("recovery") or {}
        if deployment.get("sparkcache_size") is not None:
            result.append(sparkcache.status_line(deployment["sparkcache_size"]))
        result.extend(sparkcache.option_lines(deployment))
        if recovery_record:
            timer = document.get("recovery_timer")
            result.append("  automatic recovery: "
                          + ("on" if recovery_record.get("enabled") else "off")
                          + ("" if timer is None else f", timer {'enabled' if timer else 'not enabled'}")
                          + (f", {recovery_record['failures']} failed attempt(s)" if recovery_record.get("failures") else ""))
    if not document.get("deployments"):
        result.append("No model deployment is recorded.")
    return result
