"""Execute reviewed network changes, with backups and a stop after driver reload."""

from __future__ import annotations

import copy
import inspect
import json

from .deploy_engine import plan_digest, seal_plan
from .deploy_inventory import _collect_local, _request
from .deploy_network import plan_network


def _guard(facts, host):
    """Compare configuration, not counters, timestamps, or transient GPU usage."""
    names = {p["netdev"] for p in host["data_interfaces"]}
    devices = {p["rdma_device"] for p in host["data_interfaces"]}
    return {
        "management": facts["management"],
        "routes": facts["routes"],
        "interfaces": [i for i in facts["interfaces"] if i["name"] in names],
        "rdma": [i for i in facts["rdma"] if i["device"] in devices],
        "connections": facts["network"]["connections"],
    }


def _network_local(payload, operation, *, collect=None, run=None):
    """Remote helper. Its source and the inventory probe travel in the reviewed plan."""
    import os
    from pathlib import Path
    import subprocess

    probe = collect or _collect_local
    invoke = run or subprocess.run
    host = payload["host"]
    root = Path(host["backup_dir"])
    journal = root / "execution.json"

    def idle(facts):
        containers = facts.get("docker", {}).get("containers")
        resources = facts.get("network", {}).get("rdma_resources")
        if containers is None or any(c.get("state") == "running" for c in containers):
            raise ValueError("Stop containers before changing data networking")
        if resources is None or resources:
            raise ValueError("RDMA users remain, or their state is unavailable")

    def call(record):
        result = invoke(record["argv"], capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(f"{record['id']}: {result.stderr.strip()}")
        if record.get("stdout_path"):
            path = Path(record["stdout_path"])
            if path.parent != root or path.is_symlink():
                raise ValueError(
                    "Backup capture must stay inside its dedicated directory"
                )
            with path.open("x") as stream:
                stream.write(result.stdout)
            path.chmod(0o600)

    facts = probe(payload["request"])
    if operation == "check":
        if _guard(facts, host) != payload["guard"]:
            raise ValueError(
                "Host networking changed since discovery; discover and plan again"
            )
        if payload["commands"]:
            idle(facts)
        return {"checked": True}
    if operation == "verify":
        if not journal.is_file() or journal.is_symlink():
            raise ValueError("Network execution journal is missing")
        record = json.loads(journal.read_text())
        if record.get("plan_sha256") != payload["identity"] or not record.get(
            "complete"
        ):
            raise ValueError(
                "Network execution journal is incomplete or belongs to another plan"
            )
        # A configuration fingerprint detects drift when resuming an executed plan.
        if _guard(facts, host) != record["after"]:
            raise ValueError(
                "Networking changed after execution; discover and plan again"
            )
        return {"complete": True, "rediscover": payload["rediscover"]}
    if operation != "apply":
        raise ValueError("Unknown network operation")
    if os.geteuid() != 0:
        raise ValueError("Network apply requires sudo")
    _network_local(payload, "check", collect=probe, run=invoke)
    # Never traverse a symlink or reuse another operation's backup directory.
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Backup path contains a symlink")
    if root.exists():
        raise ValueError(
            "Backup directory already exists; inspect its journal before recovery"
        )
    root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.mkdir(mode=0o700)
    record = {"plan_sha256": payload["identity"], "commands": [], "complete": False}

    def save():
        temporary = root / "execution.writing"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.chmod(0o600)
        temporary.replace(journal)

    save()
    for command in payload["commands"]:
        if command["risk"] != "read-only":
            idle(probe(payload["request"]))
        entry = {"id": command["id"], "argv": command["argv"], "state": "running"}
        record["commands"].append(entry)
        save()
        call(command)
        entry["state"] = "succeeded"
        save()
        if command.get("stop_after"):
            break
    after = probe(payload["request"])
    route = after["management"].get("route_to_controller") or {}
    if route.get("dev") != host["management_netdev"] or after["management"].get(
        "error"
    ):
        raise ValueError("Independent management path no longer verifies")
    record.update(complete=True, after=_guard(after, host))
    save()
    return {"complete": True, "rediscover": payload["rediscover"]}


def remote_command(payload, operation):
    source = "\n\n".join(
        inspect.getsource(f) for f in (_guard, _collect_local, _network_local)
    )
    source += (
        "\nimport json\nprint(json.dumps(_network_local("
        + repr(payload)
        + ", "
        + repr(operation)
        + ")))\n"
    )
    return ["sudo", "-n", "python3", "-c", source]


def build_network_plan(preparation, inventory):
    """Build an executable plan. Driver reload changes one function, then stops."""
    spec = copy.deepcopy(preparation["spec"])
    identity = plan_digest({"spec": spec, "inventory": inventory})
    for host in spec["hosts"]:
        host["backup_dir"] += "/" + identity[:16]
    network = plan_network(spec, inventory["hosts"])
    for host in network["hosts"]:
        if not host["apply_permitted"]:
            raise ValueError(host["host"] + ": " + " ".join(host["blocked_by"]))
    reload_host = next(
        (
            h
            for h in network["hosts"]
            if any(c.get("stop_after") for c in h["driver_steps"])
        ),
        None,
    )
    payloads = []
    for host, proposed in zip(spec["hosts"], network["hosts"], strict=True):
        commands = []
        if proposed["action"] != "none" and (
            reload_host is None or proposed is reload_host
        ):
            # Directory ownership is established atomically by the remote helper.
            commands = [
                c
                for c in proposed["backup"]
                if c["id"]
                not in (
                    "unused-backup-path",
                    "backup-not-symlink",
                    "create-backup-directory",
                )
            ]
            if reload_host is None:
                commands += proposed["apply"] + proposed["driver_steps"]
            else:
                for c in proposed["driver_steps"]:
                    commands.append(c)
                    if c.get("stop_after"):
                        break
        request = _request(
            host["rank"],
            host["host"],
            host["management_address"],
            (),
            spec["controller_address"],
        )
        payloads.append(
            {
                "host": host,
                "request": request,
                "guard": _guard(inventory["hosts"][host["host"]], host),
                "identity": identity,
                "commands": commands,
                "rediscover": reload_host is not None,
            }
        )
    phases = [
        {
            "id": "check-network-inventory",
            "actions": [
                {
                    "host": p["host"]["host"],
                    "argv": remote_command(p, "check"),
                    "risk": "read-only",
                }
                for p in payloads
            ],
        }
    ]
    actions = []
    for p in payloads:
        if p["commands"]:
            actions.append(
                {
                    "host": p["host"]["host"],
                    "argv": remote_command(p, "apply"),
                    "risk": "driver-reload" if p["rediscover"] else "mutates-host",
                    "verify": {
                        "argv": remote_command(p, "verify"),
                        "json": {"complete": True},
                    },
                    "timeout": 1800,
                }
            )
    if actions:
        phases.append({"id": "configure-data-network", "actions": actions})
    return seal_plan(
        {
            "schema": "sparkring-deploy-plan/v1",
            "operation": "network",
            "rediscover_required": True,
            "driver_reload": reload_host is not None,
            "network": network,
            "phases": phases,
        }
    )
