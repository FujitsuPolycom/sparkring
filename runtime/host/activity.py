"""One-shot activity assessment; no collector, scheduled task or inference request."""
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

from runtime.common import installer
from runtime.host import dashboard, placement, slices
from runtime.host.install_errors import NeedsInput

SCHEMA = "sparkring-activity/v1"
METRICS_TIMEOUT = 0.2
METRICS_LIMIT = 64 * 1024


def transfer_resource(root, owner_positions, requested):
    """Name forwarding Sparks from the recorded bulk-copy tree, without probing them."""
    resource = "Node A's download disk and network, including checkpoint downloads and image-copy routes"
    try:
        from runtime.host import fabric_ssh
        cluster = installer.read(Path(root) / "cluster.json")
        routes = {row["rank"]: row for row in fabric_ssh.routes(cluster)}
        forwarding = set()
        for position in owner_positions:
            parent = routes[position]["parent"]
            while parent is not None:
                forwarding.add(parent)
                parent = routes[parent]["parent"]
        shared = forwarding & set(requested or ())
        if shared:
            resource += " through " + slices.label(sorted(shared))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return resource


def fingerprint(root, positions):
    """Bind a review to the fabric and overlapping deployment generations."""
    root = Path(root)
    values = {}
    def read(path):
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
    for name in ("cluster.json", "fabric.json"):
        values[name] = read(root / name)
    # An installed package can change profile defaults between preview and
    # confirmation even when the active deployment has not changed.
    values["controller_package"] = read(installer.ROOT / "distribution.json")
    for slot in placement.slots(root):
        if not placement.overlap(slot, positions):
            continue
        home = placement.slot_directory(root, slot)
        for name in ("active.json", "transaction.json"):
            values[str(home / name)] = read(home / name)
        directory = placement.recorded(root, slot)
        if directory:
            for name in ("deployment.lock.json", "state.json"):
                values[str(directory / name)] = read(directory / name)
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def require_review(root, positions, expected):
    if expected and fingerprint(root, positions) != expected:
        raise NeedsInput("The deployment, fabric or package changed since preview; review a fresh plan.", field="review")


def read_metrics(url):
    """One bounded GET with no proxy, redirect, retry or inference."""
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(url, timeout=METRICS_TIMEOUT) as response:
            return dashboard.parse_metrics(response.read(METRICS_LIMIT).decode("utf-8", errors="replace"))
    except (OSError, ValueError, urllib.error.URLError):
        return None


def assess(root, positions, *, metrics=None, quiet_path=None, probe=True):
    """Inspect locks, benchmark holders and neighbours once at an operator's request."""
    root = Path(root)
    reasons = []
    def reason(grade, where, activity, resource):
        message = (f"Another operation is active: {activity} on {slices.label(where)} holds {resource} needed by this operation."
                   if grade == "blocked" else
                   f"{activity} on {slices.label(where)} shares {resource}; this operation may be affected, probably not.")
        reasons.append({"grade": grade, "positions": where, "activity": activity,
                        "resource": resource, "message": message})
    for owner in slices.active(root):
        where = owner.get("positions")
        if placement.overlap(positions, where):
            reason("blocked", where, owner["operation"], slices.label(where))
        elif owner["operation"].startswith("install"):
            reason("may-be-affected", where, owner["operation"],
                   transfer_resource(root, where, positions))
    quiet = dashboard.quiet_state(Path(quiet_path or dashboard.QUIET_PATH), time.time())
    for holder in quiet["holders"]:
        reason("may-be-affected", None, "benchmark " + str(holder.get("label") or "measurement"), "the fabric network")
    if quiet["operator"]:
        reason("may-be-affected", None, "dashboard quiet mode (traffic is not sampled)", "the fabric network")
    observations = []
    if not any(row["grade"] == "blocked" for row in reasons):
        try:
            cluster = installer.read(root / "cluster.json")
            layout = placement.layout_of(cluster)
        except (OSError, ValueError, KeyError, TypeError):
            layout = None
        if layout:
            selected = set(placement.positions(positions, layout["size"]))
            neighbours = {p for p in range(layout["size"]) if any(
                abs(p - mine) == 1 or (layout["shape"] == "cycle" and abs(p - mine) == layout["size"] - 1)
                for mine in selected)} - selected
            seen = set()
            for slot, directory in placement.running(root).items():
                other = set(placement.positions(slot, layout["size"]))
                if selected & other or not neighbours & other:
                    continue
                lock = installer.read(directory / "deployment.lock.json")
                url = installer.connection(lock)["api_url"].split("/v1", 1)[0].rstrip("/") + "/metrics"
                if url in seen:
                    continue
                seen.add(url)
                value = (metrics or read_metrics)(url) if probe and not quiet["quiet"] else None
                observations.append({"positions": sorted(other), "traffic": "unknown" if value is None else "sampled"})
                if value and (value.get("running", 0) >= 4 or value.get("waiting", 0) > 0):
                    reason("may-be-affected", sorted(other), "heavy serving traffic", "neighbouring fabric links")
    grade = next((name for name in ("blocked", "may-be-affected")
                  if any(row["grade"] == name for row in reasons)), "clear")
    return {"schema": SCHEMA, "grade": grade, "positions": list(positions) if positions is not None else None,
            "reasons": reasons, "observations": observations,
            "message": " ".join(row["message"] for row in reasons) or
                       f"No conflicting activity is recorded for {slices.label(positions)}."}


def confirm(document, *, yes=False, plan=False, interactive=None):
    if document["grade"] == "clear":
        return
    print(document["message"], file=sys.stderr)
    if plan:
        return
    if document["grade"] == "blocked":
        raise NeedsInput(document["message"], field="activity", details={"activity": document})
    if yes:
        return
    interactive = sys.stdin.isatty() if interactive is None else interactive
    if not interactive:
        raise NeedsInput("Review the activity warning and repeat with --yes to proceed.", field="activity",
                         details={"activity": document})
    if input("Continue despite this activity? [y/N]: ").strip().lower() not in ("y", "yes"):
        raise NeedsInput("Operation cancelled.", field="activity", details={"activity": document})
