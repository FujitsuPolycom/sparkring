#!/usr/bin/env python3
"""Environment-only variants of an installed four-Spark deployment.

Run as root on Node A. For each Spark, the newest installer container whose name
contains MATCH is the base: its Compose file is copied unchanged except for the
container name, an experiment label, the removed runtime-binding mount and the
variant's environment overrides. Usage:
  envab.py up MATCH NAME KEY=VALUE... | down NAME | restore MATCH
/var/tmp/dsab/hosts.json lists the worker SSH commands.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

BASE = Path("/var/tmp/dsab")
HOSTS = [None, *json.loads((BASE / "hosts.json").read_text())]


def sh(argv, *, data=None, check=True):
    result = subprocess.run(argv, input=data, capture_output=True, text=True)
    if check and result.returncode:
        raise SystemExit(f"{' '.join(argv[:4])} failed: {result.stderr.strip()[-2000:]}")
    return result.stdout


def on(host, command, *, data=None, check=True):
    argv = ["bash", "-c", command] if HOSTS[host] is None else [*HOSTS[host], command]
    return sh(argv, data=data, check=check)


def base_container(host, match):
    rows = on(host, "docker ps -a --filter label=io.sparkring.rank --format '{{.CreatedAt}}\t{{.Names}}' "
                    f"| grep {match} | sort -r").splitlines()
    if not rows:
        raise SystemExit(f"host {host}: no installer container matching {match}")
    name = rows[0].split("\t")[1]
    labels = json.loads(on(host, f"docker inspect {name} --format '{{{{json .Config.Labels}}}}'"))
    return name, int(labels["io.sparkring.rank"]), labels["com.docker.compose.project.config_files"]


def derive(text, name, rank, env):
    document = yaml.safe_load(text)
    document["name"] = f"envab-{name}-r{rank}"
    (service,) = document["services"].values()
    service["container_name"] = f"envab-{name}-r{rank}"
    service["labels"] = {"io.sparkring.experiment": name}
    service["environment"] = {**service["environment"], **env}
    service["volumes"] = [v for v in service["volumes"] if v.get("target") != "/run/sparkring/runtime-binding.json"]
    return yaml.safe_dump(document, sort_keys=False)


def up(match, name, pairs):
    env = dict(pair.split("=", 1) for pair in pairs)
    found = [base_container(host, match) for host in range(len(HOSTS))]
    (BASE / f"envab-{match}.json").write_text(json.dumps(found))
    for host, (_, rank, compose) in enumerate(found):
        text = derive(on(host, f"cat {compose}"), name, rank, env)
        on(host, f"mkdir -p {BASE}/envab/{name} && cat > {BASE}/envab/{name}/compose.yaml", data=text)
    for host in range(len(HOSTS)):
        on(host, "for c in $(docker ps --filter label=io.sparkring.rank --format '{{.Names}}'; "
                 "docker ps --filter label=io.sparkring.experiment --format '{{.Names}}'); do docker stop -t 15 $c >/dev/null; done")
    for host in [*range(1, len(HOSTS)), 0]:
        on(host, f"docker compose -f {BASE}/envab/{name}/compose.yaml up -d")
    container, start = f"envab-{name}-r0", time.time()
    while time.time() - start < 3600:
        state = json.loads(sh(["docker", "inspect", container]))[0]["State"]
        if not state.get("Running"):
            raise SystemExit(f"{container} exited: {state.get('ExitCode')}")
        if state.get("Health", {}).get("Status") == "healthy":
            print(f"{name}: healthy after {time.time() - start:.0f}s", flush=True)
            return
        time.sleep(10)
    raise SystemExit(f"{name}: not healthy after 60 minutes")


def down(name):
    for host in range(len(HOSTS)):
        on(host, f"test -f {BASE}/envab/{name}/compose.yaml && docker compose -f {BASE}/envab/{name}/compose.yaml down --timeout 15", check=False)


def restore(match):
    found = json.loads((BASE / f"envab-{match}.json").read_text())
    for host in [*range(1, len(HOSTS)), 0]:
        on(host, f"docker start {found[host][0]}")


if __name__ == "__main__":
    if sys.argv[1] == "up":
        up(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif sys.argv[1] == "down":
        down(sys.argv[2])
    elif sys.argv[1] == "restore":
        restore(sys.argv[2])
