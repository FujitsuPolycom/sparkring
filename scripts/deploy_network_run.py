"""Execute reviewed network changes, with backups and a stop after driver reload."""

from __future__ import annotations

import copy
import inspect
import json

from runtime.host.control import INTERFACE as ADMINISTRATION_NETDEV

from .deploy_engine import plan_digest, seal_plan
from .deploy_inventory import _collect_local, _request
from .deploy_network import plan_network


def _guard(facts, host):
    """Compare configuration, not counters, timestamps, or transient GPU usage."""
    names = {p["netdev"] for p in host["data_interfaces"]}
    devices = {p["rdma_device"] for p in host["data_interfaces"]}
    rdma = []
    for row in facts["rdma"]:
        if row["device"] not in devices:
            continue
        # devlink.reload counts driver restarts, which the ConnectX hairpin
        # step performs between discovery and this comparison.
        if isinstance(row.get("devlink"), dict) and "reload" in row["devlink"]:
            row = {**row, "devlink": {k: v for k, v in row["devlink"].items() if k != "reload"}}
        rdma.append(row)
    return {
        "management": facts["management"],
        "routes": facts["routes"],
        "interfaces": [i for i in facts["interfaces"] if i["name"] in names],
        "rdma": rdma,
        # NetworkManager lists connections in activation order; compare them as a set.
        "connections": sorted(facts["network"]["connections"] or [],
                              key=lambda c: (str(c.get("uuid")), str(c.get("interface")))),
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
        # Links may change under GPU-less containers (network helpers,
        # registries) but not under GPU work or a process holding RDMA queues.
        containers = facts.get("docker", {}).get("containers")
        resources = facts.get("network", {}).get("rdma_resources")
        compute = facts.get("gpu", {}).get("compute_processes")
        if compute:
            raise ValueError("Stop GPU work before changing data networking (PIDs "
                             + ", ".join(map(str, compute)) + ")")
        if compute is None and (containers is None or any(c.get("state") == "running" for c in containers)):
            raise ValueError("Stop containers before changing data networking")
        # Kernel management queue pairs (GSI/SMI) have no pid and always exist.
        if resources is None or any(r.get("pid") is not None for r in resources):
            raise ValueError("RDMA users remain, or their state is unavailable")
        if payload.get("require_idle_gpu"):
            result = invoke(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=30)
            if result.returncode or result.stdout.strip():
                raise ValueError("GPU compute users remain, or their state is unavailable")

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
    # Activation applies addresses asynchronously. Record the post-change
    # fingerprint only once every planned fabric address is live, so later
    # verification compares settled configuration rather than a transition.
    import time
    planned = {p["netdev"]: p["address"] for p in host["data_interfaces"] if p.get("address")}
    for _ in range(30):
        after = probe(payload["request"])
        live = {i["name"]: i.get("ipv4") or [] for i in after["interfaces"]}
        if all(address in live.get(name, []) for name, address in planned.items()):
            break
        time.sleep(1)
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


def build_network_plan(preparation, inventory, *, defer_driver=False):
    """Build an executable plan for the data network.

    ``defer_driver=True`` serves SparkRing setup, where sparkring-hairpin.service
    applies the ConnectX hairpin setting. The plan executes NetworkManager
    changes only, lists the hosts whose ``driver_action`` is not ``none`` in
    ``driver_pending``, and lets GPU or RDMA users block only hosts with
    NetworkManager changes.

    ``defer_driver=False`` serves ``deploy_suite.py apply-plan`` on hosts without
    the SparkRing package. It also executes driver steps: a driver reload
    changes one function, then the plan stops for rediscovery. It refuses hosts
    whose hairpin state is unknown, and driver reloads on hosts whose only
    management path is the administration network, which a reload interrupts.
    """
    spec = copy.deepcopy(preparation["spec"])
    identity = plan_digest({"spec": spec, "inventory": inventory})
    for host in spec["hosts"]:
        host["backup_dir"] += "/" + identity[:16]
    network = plan_network(spec, inventory["hosts"])
    pending = [h["host"] for h in network["hosts"] if h["driver_action"] != "none"]
    for host in network["hosts"]:
        if defer_driver and host["action"] == "none":
            continue
        if not host["apply_permitted"]:
            raise ValueError(host["host"] + ": " + " ".join(host["blocked_by"]))
    if not defer_driver:
        for host in network["hosts"]:
            if host["driver_action"] == "unknown":
                raise ValueError(
                    host["host"] + ": devlink reload statistics are unavailable; SparkRing cannot "
                    "tell whether the hairpin setting is in effect and restarts nothing. "
                    "Rediscover the host, then plan again."
                )
            if host["management_netdev"] == ADMINISTRATION_NETDEV and any(
                c["risk"] == "driver-reload" for c in host["driver_steps"]
            ):
                raise ValueError(
                    host["host"] + f": a driver reload would interrupt its only management path "
                    f"({ADMINISTRATION_NETDEV}). On Node A, sudo sparkring hairpin applies the "
                    "ConnectX hairpin setting."
                )
    reload_host = None if defer_driver else next(
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
        driver = [] if defer_driver else proposed["driver_steps"]
        if (
            proposed["action"] != "none"
            or (not defer_driver and proposed["driver_action"] != "none")
        ) and (reload_host is None or proposed is reload_host):
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
                commands += proposed["apply"] + driver
            else:
                for c in driver:
                    commands.append(c)
                    if c.get("stop_after"):
                        break
        request = _request(
            host["rank"],
            host["host"],
            host["management_address"],
            (),
            host.get("controller_probe_address", spec["controller_address"]),
        )
        payloads.append(
            {
                "host": host,
                "request": request,
                "guard": _guard(inventory["hosts"][host["host"]], host),
                "identity": identity,
                "commands": commands,
                "rediscover": reload_host is not None,
                "require_idle_gpu": spec.get("require_idle_gpu", False),
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
            "driver_pending": pending,
            "network": network,
            "phases": phases,
        }
    )
