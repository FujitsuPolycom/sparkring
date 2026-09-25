#!/usr/bin/env python3
"""Qwen serving variants derived from the installer's Qwen deployment; run as root on Node A.

Each rank's installer Compose file is the base. A variant can mount another
checkpoint (every cache path then names that checkpoint's revision, so compiled
kernels and prepared weights are never shared between checkpoints), change the
MTP speculative configuration, append vLLM arguments and set environment
variables. `up` stops the installer's Qwen containers, starts the variant on
every rank (workers first) and waits for rank 0 to report healthy. `restore`
removes variant containers and restarts the installer containers.

Usage: qwab.py up VARIANT.json | down NAME | restore
/var/tmp/qwab/hosts.json lists the worker SSH commands, e.g. [["ssh", "10.253.255.2"]].
"""
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

BASE = Path("/var/tmp/qwab")
REVISION = "60215d26cf5e"
HOSTS = [None, *json.loads((BASE / "hosts.json").read_text())]


def sh(argv, *, data=None, check=True):
    result = subprocess.run(argv, input=data, capture_output=True, text=True)
    if check and result.returncode:
        raise SystemExit(f"{' '.join(argv[:4])} failed: {result.stderr.strip()[-2000:]}")
    return result.stdout


def on(host, command, *, data=None, check=True):
    argv = ["bash", "-c", command] if HOSTS[host] is None else [*HOSTS[host], command]
    return sh(argv, data=data, check=check)


def installer_container(host):
    """The newest installer Qwen container on a host: (name, rank, compose path)."""
    rows = on(host, "docker ps -a --filter label=io.sparkring.rank --format '{{.CreatedAt}}\t{{.Names}}' | grep qwen | sort -r").splitlines()
    if not rows:
        raise SystemExit(f"host {host}: no installer Qwen container")
    name = rows[0].split("\t")[1]
    labels = json.loads(on(host, f"docker inspect {name} --format '{{{{json .Config.Labels}}}}'"))
    return name, int(labels["io.sparkring.rank"]), labels["com.docker.compose.project.config_files"]


def derive(text, variant, rank):
    document = yaml.safe_load(text)
    name = f"qwab-{variant['name']}-r{rank}"
    document["name"] = name
    (service,) = document["services"].values()
    service["container_name"] = name
    service["labels"] = {"io.sparkring.experiment": variant["name"]}
    environment = service["environment"]
    if variant.get("checkpoint"):
        for volume in service["volumes"]:
            if volume.get("target") == "/models/target":
                volume["source"] = variant["checkpoint"]
        revision = variant["revision"][:12]
        for key, value in list(environment.items()):
            if isinstance(value, str) and REVISION in value:
                environment[key] = value.replace(REVISION, revision)
    command = [str(part) for part in service["command"]]
    if variant.get("spec"):
        index = command.index("--speculative-config") + 1
        spec = json.loads(command[index])
        spec.update(variant["spec"])
        command[index] = json.dumps(spec, separators=(",", ":"))
    for flag, value in variant.get("set_args", {}).items():
        command[command.index(flag) + 1] = value
    for flag in variant.get("remove_args", []):
        index = command.index(flag)
        del command[index:index + 2]
    command += variant.get("add_args", [])
    service["command"] = command
    environment.update(variant.get("env", {}))
    service["volumes"] += [{"type": "bind", "read_only": True, **volume} for volume in variant.get("volumes", [])]
    return yaml.safe_dump(document, sort_keys=False)


def up(path):
    variant = json.loads(Path(path).read_text())
    found = [installer_container(host) for host in range(len(HOSTS))]
    (BASE / "installer.json").write_text(json.dumps(found))
    for host, (_, rank, compose) in enumerate(found):
        text = derive(on(host, f"cat {compose}"), variant, rank)
        on(host, f"mkdir -p {BASE}/run/{variant['name']} && cat > {BASE}/run/{variant['name']}/compose.yaml", data=text)
    for host in range(len(HOSTS)):
        on(host, "for c in $(docker ps --filter label=io.sparkring.rank --format '{{.Names}}'; "
                 "docker ps --filter label=io.sparkring.experiment --format '{{.Names}}'); do docker stop -t 15 $c >/dev/null; done")
    for host in [*range(1, len(HOSTS)), 0]:
        on(host, f"docker compose -f {BASE}/run/{variant['name']}/compose.yaml up -d")
    name = f"qwab-{variant['name']}-r0"
    start = time.time()
    while time.time() - start < 3000:
        state = json.loads(sh(["docker", "inspect", name]))[0]["State"]
        if not state.get("Running"):
            raise SystemExit(f"{name} exited: {state.get('ExitCode')}")
        if state.get("Health", {}).get("Status") == "healthy":
            print(f"{variant['name']}: healthy after {time.time() - start:.0f}s", flush=True)
            return
        time.sleep(5)
    raise SystemExit(f"{variant['name']}: not healthy after 50 minutes")


def down(name):
    for host in range(len(HOSTS)):
        on(host, f"test -f {BASE}/run/{name}/compose.yaml && docker compose -f {BASE}/run/{name}/compose.yaml down --timeout 15", check=False)
    print(f"{name}: removed", flush=True)


def restore():
    for host in range(len(HOSTS)):
        on(host, f"for f in {BASE}/run/*/compose.yaml; do test -f $f && docker compose -f $f down --timeout 15; done", check=False)
    found = json.loads((BASE / "installer.json").read_text())
    for host in [*range(1, len(HOSTS)), 0]:
        on(host, f"docker start {found[host][0]}")
    print("installer containers restarted", flush=True)


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "up":
        up(sys.argv[2])
    elif action == "down":
        down(sys.argv[2])
    elif action == "restore":
        restore()
    else:
        raise SystemExit(__doc__)
