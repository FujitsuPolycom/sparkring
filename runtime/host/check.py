"""``sudo sparkring check``: functional checks and the transport check of the running models, and the tester report.

For each slot's active deployment (``--on`` selects the model on one arc of
the fabric, ``runtime.host.placement``) the command:

1. sends the acceptance harness's functional checks to the model's API
   (``performance/harnesses/acceptance/checks.py``): counting, arithmetic and
   code, and a tool call, an image and thinking where the profile serves them,
   each greedy with the profile's request settings;
2. for a deployment on SIRCL ring sessions, reads every rank's receipts and
   container log through the deployment's own source and judges them against
   its transport section (``runtime/host/transport_receipts.py``), recording
   the verdict that ``sparkring status`` prints.

The result is ``sparkring-check/v1`` (``--json``). The exit status is 0 when
every check passed and every SIRCL verdict is ``as-expected``, 1 otherwise.
It changes no Spark: requests go to the model and SSH commands only read.

``--report DIR`` also writes ``DIR/sparkring-report-<UTC time>/``
(``sparkring-test-report/v1``), the bundle a tester attaches to a Test report
issue: the last installation's result, ``sparkring fabric show`` and the last
``sparkring fabric verify`` report, the status, this check, the receipts
with the tuning table in effect, the last 200 lines of the installation's
details log and of each rank's container log, and Node A's environment
(package, image, DGX OS, driver, Docker, container toolkit, ConnectX
firmware). Management and LAN addresses, host names, MAC addresses and
account names are replaced by placeholders; fabric addresses (198.18.0.0/15)
stay. A file that still names a private item after replacement is left out
and listed in ``report.json``. Review the bundle before attaching it.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from runtime.common import distribution, installer, transport
from runtime.host import api_endpoint, controller, fabric, node, progress, retained_source, transport_receipts

SCHEMA = "sparkring-check/v1"
REPORT_SCHEMA = "sparkring-test-report/v1"
TAIL = 200
MAC = re.compile(r"(?i)(?<![0-9a-f:])(?:[0-9a-f]{2}:){5}[0-9a-f]{2}(?![0-9a-f:])")


def deployments(on=None):
    """``[(placement, directory)]`` of the deployments to check: every slot's active one, or the ``--on`` arc's."""
    from runtime.host import placement as placements
    if on:
        slot = placements.parse(on, controller.recorded_layout())
        path = controller.active_deployment(placement=slot)
        return [(slot, path)] if path is not None else []
    return controller.active_deployments(report=True)


def functional(lock, *, client=None, log=lambda line: None):
    """The acceptance harness's functional checks against the deployment's API."""
    sys.path.insert(0, str(installer.ROOT))
    from performance.harnesses.acceptance import checks, profile_info, runners
    card = lock["selection"]
    default = installer.setup.selection(card["profile"])["target_variant"]
    info = profile_info.load(card["profile"], checkpoint=card["target_variant"]
                             if card["target_variant"] != default else None)
    connection = installer.connection(lock)
    chat = checks.Chat(client or runners.HttpClient(), connection["api_url"], connection["model"],
                       thinking_off=info.thinking_off, thinking_on=info.thinking_on)
    return checks.run_functional(chat, features=info.features, log=log)


def transport_check(directory, lock, *, cache):
    """The transport verdict of a deployment: ``{"backend": "prepared"}`` or ``{"backend": "nccl"}`` without
    receipts, else the SIRCL receipt verdict."""
    section = lock.get("transport")
    if not section:
        return {"backend": "prepared"}
    if section.get("backend") == "nccl":
        return {"backend": "nccl"}
    try:
        return retained_source.apply(directory, "transport", cache=cache)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        return {"backend": "sircl", "verdict": "unknown", "problems": [str(error).strip().splitlines()[-1][:300]]}


def run(on=None, *, client=None, say=print):
    """``sparkring-check/v1`` of every deployment checked."""
    cache = controller.STATE / "retained-sources"
    rows = []
    for slot, directory in deployments(on):
        lock = installer.read(Path(directory) / "deployment.lock.json")
        shown = api_endpoint.present(installer.connection(lock), api_endpoint.recorded(directory))
        say(f"{lock['selection']['profile']} at {shown['api_url']}" + (f" (positions {','.join(map(str, slot))})"
                                                                         if slot else ""))
        try:
            checked = functional(lock, client=client, log=lambda line: say("  " + line))
        except (OSError, ValueError, KeyError) as error:
            checked = {"ok": False, "error": str(error), "checks": [], "passed": 0, "failed": 0, "skipped": 0}
            say(f"  functional checks could not run: {error}")
        verdict = transport_check(directory, lock, cache=cache)
        if verdict.get("backend") == "prepared":
            say("  Transport: prepared")
        else:
            say("  " + (transport_receipts.text({"problems": ["no detail"], **verdict}) or "Transport: sircl"))
        ok = checked.get("ok", False) and verdict.get("verdict", "as-expected") == "as-expected"
        rows.append({"deployment": str(directory), "profile": lock["selection"]["profile"],
                     "placement": list(slot) if slot else None, "api_url": shown["api_url"],
                     "functional": checked, "transport": verdict, "ok": ok})
    return {"schema": SCHEMA, "checked_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "deployments": rows, "ok": bool(rows) and all(row["ok"] for row in rows)}


# The tester report.

def _tail(path, lines=TAIL):
    try:
        return "\n".join(Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]) + "\n"
    except OSError:
        return None


def _output(argv):
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def environment(lock=None):
    """Node A's software and firmware facts that a test report needs."""
    record = {}
    try:
        record = json.loads((installer.ROOT / "distribution.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    firmware = _output(["ethtool", "-i", "enp1s0f0np0"]) or ""
    return {"package_version": record.get("version"), "source_revision": distribution.identity(installer.ROOT),
            "image_id": (lock or {}).get("selection", {}).get("image_id"),
            "image_release": (lock or {}).get("selection", {}).get("release"),
            "dgx_os": (_tail("/etc/dgx-release", 20) or "").strip() or None,
            "driver": _output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
            "docker": _output(["docker", "version", "--format", "{{.Server.Version}}"]),
            "container_toolkit": next(iter((_output(["nvidia-ctk", "--version"]) or "").splitlines()), None),
            "connectx_firmware": next((line.split(":", 1)[1].strip() for line in firmware.splitlines()
                                       if line.startswith("firmware-version:")), None)}


def private_items(state=None):
    """``(replacements, names, users)`` of the cluster: addresses, SSH targets and host names to hide."""
    state = Path(state or controller.STATE)
    replacements, names, users = {}, set(), set()
    try:
        cluster = installer.read(state / "cluster.json")
    except (OSError, ValueError):
        cluster = {}
    hosts = ((cluster.get("plan") or {}).get("spec") or {}).get("hosts") or []
    for rank, host in enumerate(hosts):
        target = host.get("host") or ""
        user, separator, address = target.rpartition("@")
        if separator:
            users.add(user)
            replacements[target] = f"USER@SPARK_{rank}"
        replacements[address or target] = f"SPARK_{rank}"
        if host.get("management_address"):
            replacements[host["management_address"]] = f"MANAGEMENT_{rank}"
    for rank, row in enumerate((cluster.get("plan") or {}).get("nodes") or []):
        if isinstance(row, dict) and row.get("hostname"):
            names.add(row["hostname"])
            replacements[row["hostname"]] = f"spark-{rank}"
    if cluster.get("api_address"):
        replacements[cluster["api_address"]] = "LAN_ADDRESS"
    try:
        document = fabric.read_document(state)
        for row in document["positions"]:
            names.add(row["hostname"])
            replacements[row["hostname"]] = f"spark-{row['position']}"
            if row.get("lan_address"):
                replacements[row["lan_address"]] = f"LAN_ADDRESS_{row['position']}"
            if (row.get("management") or {}).get("address"):
                replacements[row["management"]["address"]] = f"MANAGEMENT_{row['position']}"
    except (OSError, ValueError, KeyError):
        pass
    users.discard("root")
    return replacements, sorted(names), sorted(users)


def sanitize(text, replacements, names, users):
    """``text`` with the cluster's private items and every MAC address replaced; PrivateDataError on leftovers."""
    sys.path.insert(0, str(installer.ROOT))
    from performance.harnesses.acceptance import record
    numbered = {}
    text = MAC.sub(lambda match: numbered.setdefault(match.group(0).lower(), f"MAC_{len(numbered) + 1}"), text)
    return record.sanitize_text(text, replace=replacements, names=names, users=users)


def report(output, result, *, state=None, now=None):
    """Write the tester bundle under ``output``; returns its directory and the files it holds or left out."""
    sys.path.insert(0, str(installer.ROOT))
    from performance.harnesses.acceptance import throughput
    state = Path(state or controller.STATE)
    moment = now or datetime.datetime.now(datetime.timezone.utc)
    target = Path(output) / f"sparkring-report-{moment.strftime('%Y%m%dT%H%M%SZ')}"
    target.mkdir(parents=True, exist_ok=False, mode=0o700)
    replacements, names, users = private_items(state)
    files, left_out = {}, {}
    first = result["deployments"][0] if result["deployments"] else None
    lock = installer.read(Path(first["deployment"]) / "deployment.lock.json") if first else None
    documents = {"check.json": result, "environment.json": environment(lock)}
    try:
        documents["fabric-show.json"] = {"document": fabric.read_document(state),
                                         "latest_verification": fabric.latest_report(state)}
        documents["fabric-verify.json"] = fabric.latest_report(state)
    except (OSError, ValueError):
        pass
    status = {"node_a": node.status(), "deployments": []}
    for row in result["deployments"]:
        directory = Path(row["deployment"])
        try:
            status["deployments"].append(installer.status(directory))
        except (OSError, ValueError, KeyError) as error:
            status["deployments"].append({"deployment": str(directory), "error": str(error)})
        saved = directory / "install-result.json"
        if saved.exists():
            documents[f"install-result-{directory.name}.json"] = installer.read(saved)
        verdict = row.get("transport") or {}
        receipts = Path(verdict["receipts"]) if verdict.get("receipts") else None
        if receipts is not None and receipts.is_dir():
            for path in sorted(receipts.glob("*.json")):
                value = json.loads(path.read_text(encoding="utf-8"))
                if path.name.startswith("log-rank"):
                    files[f"logs/{directory.name}/{path.stem.removeprefix('log-')}.log"] = (
                        "\n".join((value.get("log") or {}).get("tail") or []) + "\n")
                    value = {key: item for key, item in value.items() if key != "log"} | {
                        "nccl_lines": {key: (value.get("log") or {}).get(key) for key in ("init", "library")}}
                documents[f"receipts/{directory.name}/{path.name}"] = value
        if lock and row["deployment"] == first["deployment"] and (lock.get("transport") or {}).get("backend") == \
                "sircl":
            try:
                documents["tuning-table.json"] = transport.tuning_in_effect(
                    state, fabric.read_document(state), {"image_id": lock["selection"]["image_id"]})[0]
            except (OSError, ValueError):
                pass
    documents["status.json"] = status
    details = _tail(progress.directory() / "install-details.log")
    if details is not None:
        files["logs/install-details.log"] = details
    for name, value in documents.items():
        files[name] = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    written = []
    for name, text in sorted(files.items()):
        try:
            clean = sanitize(text, replacements, names, users)
        except throughput.PrivateDataError as error:
            left_out[name] = str(error)
            continue
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_text(clean, encoding="utf-8", newline="\n")
        written.append(name)
    manifest = {"schema": REPORT_SCHEMA, "created_at": moment.strftime("%Y-%m-%dT%H:%M:%SZ"), "files": written,
                "left_out": left_out,
                "sanitized": "management and LAN addresses, host names, SSH accounts and MAC addresses replaced; "
                             "fabric addresses kept; review before attaching"}
    (target / "report.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"directory": str(target), "files": written, "left_out": left_out}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sparkring check", description=(
        "Check the running models: functional requests to each API and, on SIRCL ring sessions, which transport "
        "carried each collective. Sends requests; changes no Spark. Run on Node A with sudo."))
    parser.add_argument("--on", metavar="ARC", help="only the model on these Sparks, such as 0,1, 0-3 or 6-1")
    parser.add_argument("--json", action="store_true", help="print the sparkring-check/v1 result")
    parser.add_argument("--report", type=Path, metavar="DIR",
                        help="also write the sanitized tester bundle (sparkring-test-report/v1) under DIR")
    args = parser.parse_args(argv)
    say = (lambda line: print(line, file=sys.stderr)) if args.json else print
    try:
        if not hasattr(os, "geteuid") or os.geteuid() != 0:
            raise ValueError("Run sudo sparkring check on Node A: it reads the controller's deployment records")
        result = run(args.on, say=say)
        if not result["deployments"]:
            raise ValueError("No model deployment is active; sudo sparkring install starts one")
        if args.report is not None:
            result["report"] = report(args.report, result)
            say(f"Report: {result['report']['directory']}" + (
                f" ({len(result['report']['left_out'])} files left out; see report.json)"
                if result["report"]["left_out"] else "") + ". Review it, then attach it to a Test report issue.")
    except (ValueError, OSError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print("SparkRing: " + str(error), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        say("All checks passed." if result["ok"] else "Some checks failed; see above.")
    return 0 if result["ok"] else 1
