"""Lease-scoped serving trials using shared container rendering and saved rollback.

This module creates only run-owned container names. Production containers are
stopped/started by their captured immutable IDs, never deleted or recreated.
"""

from __future__ import annotations

from dataclasses import replace
import json
import getpass
import socket
import re
import shlex
import time
import urllib.request

from runtime.common.container_spec import Bind, ContainerSpec, docker_create
from .contracts import Refused, require, check_policy
from .execution import lease_valid
from .io import checked


def snapshot_spec(snapshot):
    """Preserve the captured envelope; reject settings the renderer cannot retain."""
    config, host = snapshot["Config"], snapshot["HostConfig"]
    require(
        not host.get("Privileged"),
        "Privileged serving snapshots require a reviewed adapter",
    )
    require(not host.get("VolumesFrom"), "Inherited container volumes are not admitted")
    for key in (
        "CapDrop",
        "CpuShares",
        "CpuPeriod",
        "CpuQuota",
        "CpusetMems",
        "NanoCpus",
        "MemoryReservation",
        "PidsLimit",
        "OomKillDisable",
        "ReadonlyRootfs",
        "DeviceCgroupRules",
        "ExtraHosts",
        "Dns",
        "PidMode",
        "UTSMode",
        "PortBindings",
    ):
        require(
            not host.get(key),
            "Snapshot setting needs an explicit renderer mapping: " + key,
        )
    requests = host.get("DeviceRequests", [])
    require(
        len(requests) == 1
        and requests[0].get("Count") == -1
        and not requests[0].get("DeviceIDs")
        and requests[0].get("Capabilities") == [["gpu"]],
        "Qualification expects the captured single-node all-GPU request",
    )
    require(
        host.get("NetworkMode") == "host" and host.get("IpcMode") == "host",
        "Qualification expects the recorded host-network/IPC deployment",
    )
    require(
        all(m["Type"] == "bind" for m in snapshot["Mounts"]),
        "Named volumes require a reviewed cache-isolation adapter",
    )
    environment = {}
    for item in config.get("Env", []):
        key, separator, value = item.partition("=")
        require(
            separator and key not in environment,
            "Ambiguous environment in serving snapshot",
        )
        environment[key] = value
    devices = []
    for device in host.get("Devices", []):
        require(
            device["PathOnHost"] == device["PathInContainer"]
            and device.get("CgroupPermissions") == "rwm",
            "Device mapping cannot be represented faithfully",
        )
        devices.append(device["PathOnHost"])
    limits = {item["Name"]: item for item in host.get("Ulimits", []) or []}
    require(
        set(limits) <= {"memlock"},
        "Additional resource limits require a reviewed adapter",
    )
    limit = limits.get("memlock", dict(Soft=-1, Hard=-1))
    require(
        limit["Soft"] == limit["Hard"],
        "Asymmetric memlock limits require a reviewed adapter",
    )
    return ContainerSpec(
        name=snapshot["Name"].lstrip("/"),
        image_id=snapshot["Image"],
        entrypoint=tuple(config.get("Entrypoint") or ()),
        command=tuple(config.get("Cmd") or ()),
        environment=environment,
        mounts=tuple(
            Bind(m["Source"], m["Destination"], not m["RW"]) for m in snapshot["Mounts"]
        ),
        init=bool(host.get("Init")),
        restart_policy="no",
        memory=host.get("Memory") or None,
        memory_swap=host.get("MemorySwap") or None,
        shm_size=host.get("ShmSize") or None,
        devices=tuple(devices),
        memlock=limit["Soft"],
        cap_add=tuple(host.get("CapAdd") or ()),
        security_opt=tuple(host.get("SecurityOpt") or ()),
        user=config.get("User") or None,
        working_dir=config.get("WorkingDir") or None,
        cpuset_cpus=host.get("CpusetCpus") or None,
        health_mode="disabled",
    )


def isolated_spec(
    snapshot,
    *,
    run_id,
    role,
    rank,
    cache_mappings,
    image_id=None,
    native=False,
    environment=None,
    command=None,
):
    require(
        re.fullmatch(r"[a-z0-9][a-z0-9-]{1,47}", run_id), "Invalid qualification run ID"
    )
    require(
        role in ("control", "candidate") and rank in (0, 1),
        "Unknown qualification role/rank",
    )
    spec = snapshot_spec(snapshot)
    require(cache_mappings, "Explicit writable mount isolation is required")
    mounts = []
    for mount in spec.mounts:
        if not mount.read_only:
            require(
                mount.target in cache_mappings,
                "Writable mount is not isolated: " + mount.target,
            )
            source = cache_mappings[mount.target]
            require(
                source != mount.source
                and source.startswith("/")
                and ".." not in source.split("/")
                and "," not in source,
                "Qualification cache cannot reuse the serving mount",
            )
            mount = replace(mount, source=source)
        mounts.append(mount)
    args = tuple(command) if command is not None else spec.command
    entrypoint = spec.entrypoint
    if native:
        require(image_id is not None, "Native qualification requires an exact image ID")
        index = next(
            (i for i, item in enumerate(args) if item in ("serve", "--serve")), None
        )
        require(index is not None, "Snapshot has no recognizable serving interface")
        args = ("serve", *args[index + 1 :])
        entrypoint = ("/opt/venv/bin/python", "/opt/sparkring/bin/native-image.py")
    return replace(
        spec,
        name="sr-upgrade-" + run_id + "-" + role + "-r" + str(rank),
        image_id=image_id or spec.image_id,
        entrypoint=entrypoint,
        command=args,
        environment={**spec.environment, **(environment or {})},
        mounts=tuple(mounts),
        labels={
            "sparkring.upgrade.run": run_id,
            "sparkring.upgrade.role": role,
            "sparkring.upgrade.rank": str(rank),
        },
    )


def metrics(url):
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=5) as response:
        text = response.read(4 * 1024**2).decode()
    result = {"running": 0.0, "waiting": 0.0, "steps": 0.0}
    names = {
        "num_requests_running": "running",
        "num_requests_waiting": "waiting",
        "iteration_tokens_total_count": "steps",
    }
    found = set()
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        match = re.match(r"vllm:([a-z_]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if match and match[1] in names:
            key = names[match[1]]
            result[key] += float(match[2])
            found.add(key)
    require(
        found == set(result), "Server does not provide the idle-observation counters"
    )
    return result


class Pair:
    def __init__(self, policy, lease, hosts, *, run_id, gate_id, hostnames=None):
        require(
            len(hosts) == 2 and len(set(hosts)) == 2,
            "A serving trial requires two distinct leased SSH hosts",
        )
        require(
            all(
                re.fullmatch(r"[a-z_][a-z0-9_-]*@[A-Za-z0-9.-]+", host)
                for host in hosts
            ),
            "Invalid SSH host identity",
        )
        self.policy, self.lease, self.hosts, self.run_id = (
            policy,
            lease,
            tuple(hosts),
            run_id,
        )
        self.gate = {"id": gate_id, "resources": list(hosts)}
        self.hostnames = tuple(hostnames) if hostnames is not None else ()
        require(
            not self.hostnames or len(self.hostnames) == 2,
            "Expected two hostname identities",
        )

    def call(self, rank, argv, *, seconds=120, input_bytes=None):
        if self.policy.get("_path"):
            check_policy(self.policy)
        deadline = lease_valid(
            self.lease, self.policy, self.gate, time.monotonic() + seconds
        )
        require(rank in (0, 1), "Unknown host rank")
        local = (
            self.hostnames
            and socket.gethostname() == self.hostnames[rank]
            and getpass.getuser() == self.hosts[rank].split("@")[0]
        )
        if local:
            return checked(
                list(argv),
                seconds=max(0.1, deadline - time.monotonic()),
                limit=16 * 1024**2,
                input_bytes=input_bytes,
            )
        remote = list(argv)
        if self.hostnames:
            wrapper = "import socket,subprocess,sys; assert socket.gethostname()==sys.argv[1], 'SSH hostname identity differs'; raise SystemExit(subprocess.call(sys.argv[2:]))"
            remote = ["python3", "-c", wrapper, self.hostnames[rank], *remote]
        return checked(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                self.hosts[rank],
                shlex.join(remote),
            ],
            seconds=max(0.1, deadline - time.monotonic()),
            limit=16 * 1024**2,
            input_bytes=input_bytes,
        )

    def inspect(self, rank, identifier):
        require(
            re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", identifier),
            "Invalid container identifier",
        )
        return json.loads(self.call(rank, ["docker", "inspect", identifier]))[0]

    def assert_idle(self, url, *, observation_seconds=10):
        before = metrics(url)
        require(
            before["running"] == before["waiting"] == 0,
            "Serving requests are active; trial deferred",
        )
        time.sleep(observation_seconds)
        after = metrics(url)
        require(
            after["running"] == after["waiting"] == 0
            and after["steps"] == before["steps"],
            "Serving activity changed; trial deferred",
        )
        return after

    def stop_saved(self, snapshots):
        for rank, snapshot in enumerate(snapshots):
            actual = self.inspect(rank, snapshot["Id"])
            require(
                actual["Image"] == snapshot["Image"]
                and actual["Name"] == snapshot["Name"],
                "Rollback container identity differs",
            )
            self.call(
                rank, ["docker", "stop", "--time", "90", snapshot["Id"]], seconds=120
            )

    def start_saved(self, snapshots):
        for rank in (1, 0):
            snapshot = snapshots[rank]
            actual = self.inspect(rank, snapshot["Id"])
            require(
                actual["Image"] == snapshot["Image"]
                and actual["Name"] == snapshot["Name"],
                "Saved rollback identity differs",
            )
            self.call(rank, ["docker", "start", snapshot["Id"]])

    def create(self, rank, spec):
        require(
            spec.labels.get("sparkring.upgrade.run") == self.run_id,
            "Serving spec is not owned by this trial",
        )
        require(
            spec.name.startswith("sr-upgrade-" + self.run_id + "-"),
            "Container name escapes the trial",
        )
        result = self.call(rank, docker_create(spec))
        actual = self.inspect(rank, spec.name)
        require(
            actual["Image"] == spec.image_id
            and actual["Config"]["Labels"].get("sparkring.upgrade.run") == self.run_id,
            "Created container differs from its trial identity",
        )
        return result.decode().strip()

    def start(self, rank, name):
        self.owned(rank, name)
        self.call(rank, ["docker", "start", name])
        return (
            "ssh -t "
            + self.hosts[rank]
            + " "
            + shlex.quote("docker logs -f --tail 80 " + name)
        )

    def owned(self, rank, name):
        actual = self.inspect(rank, name)
        require(
            actual["Config"].get("Labels", {}).get("sparkring.upgrade.run")
            == self.run_id,
            "Container is not owned by this run",
        )
        return actual

    def stop(self, rank, name):
        self.owned(rank, name)
        self.call(rank, ["docker", "stop", "--time", "90", name], seconds=120)


def wait_ready(url, model, *, seconds=900):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                url.rstrip("/") + "/v1/models", timeout=5
            ) as response:
                data = json.loads(response.read(1024**2))
            if any(row.get("id") == model for row in data.get("data", [])):
                return data
        except (OSError, ValueError):
            pass
        time.sleep(5)
    raise Refused("Model did not become ready inside its startup budget")
