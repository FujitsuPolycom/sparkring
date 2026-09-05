#!/usr/bin/env python3
"""Read-only four-rank container and HTTP readiness check for a rendered launch."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("mesh_readiness_profile", HERE / "profile.py")
profile = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = profile
SPEC.loader.exec_module(profile)
ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
POLL_SECONDS = 2.0


def load_launch(launch):
    site, topology, _ = profile.load_site(Path(launch) / "site.json")
    aliases = [topology.rank(rank).ssh_alias for rank in range(4)]
    if any(not isinstance(alias, str) or not ALIAS.fullmatch(alias) for alias in aliases):
        raise ValueError("SSH aliases must contain only letters, digits, dots, underscores and hyphens")
    if len(set(aliases)) != 4:
        raise ValueError("Expected four distinct SSH aliases")
    environment = profile.defaults(Path(launch) / "rank0.env")
    ports = []
    for key in ("PORT", "SPARKRING_LIVENESS_PORT"):
        value = environment.get(key, "")
        if not re.fullmatch(r"[0-9]+", value) or not 1 <= int(value) <= 65535:
            raise ValueError(f"{key} must be an explicit literal port in rank0.env")
        ports.append(int(value))
    if ports[0] == ports[1] or environment.get("SPARKRING_LIVENESS_ENABLED") != "1":
        raise ValueError("Readiness requires a distinct enabled liveness port")
    address = site["management_addresses"][0]
    return {
        "containers": [{"rank": rank, "ssh_alias": aliases[rank],
                        "name": site["container_prefix"] + f"-r{rank}"} for rank in range(4)],
        "urls": [f"http://{address}:{ports[0]}/health", f"http://{address}:{ports[1]}/liveness"],
    }


def inspect_command(alias, name):
    if not isinstance(alias, str) or not ALIAS.fullmatch(alias):
        raise ValueError("Invalid SSH alias")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name):
        raise ValueError("Invalid container name")
    remote = shlex.join(["docker", "inspect", "--format", "{{json .State}}", name])
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", alias, remote]


def remaining(deadline, clock=time.monotonic):
    budget = deadline - clock()
    if budget <= 0:
        raise TimeoutError("Readiness deadline exceeded")
    return min(5.0, budget)


def inspect_container(target, timeout):
    result = subprocess.run(inspect_command(target["ssh_alias"], target["name"]),
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Container inspection failed for rank {target['rank']}: {result.stderr.strip()}")
    state = json.loads(result.stdout)
    if not isinstance(state, dict) or type(state.get("Running")) is not bool:
        raise ValueError("Docker returned an invalid container state")
    health = state.get("Health", {})
    if not isinstance(health, dict):
        raise ValueError("Docker returned an invalid container health state")
    return {**target, "running": state["Running"], "health": health.get("Status"),
            "status": state.get("Status"),
            "ready": state["Running"] is True and health.get("Status") == "healthy"}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def http_ready(url, timeout):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(url, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"Readiness endpoint returned HTTP {response.status}: {url}")
    return {"url": url, "status": 200}


def sample(plan, deadline, *, inspect=inspect_container, request=http_ready, clock=time.monotonic):
    result = {"ready": False, "containers": [], "http": []}
    try:
        for target in plan["containers"]:
            result["containers"].append(inspect(target, remaining(deadline, clock)))
        if len(result["containers"]) != 4 or not all(row["ready"] for row in result["containers"]):
            return result
        for url in plan["urls"]:
            result["http"].append(request(url, remaining(deadline, clock)))
        remaining(deadline, clock)
        result["ready"] = len(result["http"]) == 2
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
    return result


def wait(plan, timeout, *, probe=sample, clock=time.monotonic, sleep=time.sleep):
    if not math.isfinite(timeout) or not 0 < timeout <= 900:
        raise ValueError("Timeout must be finite and greater than zero through 900 seconds")
    started = clock()
    deadline = started + timeout
    receipt = {"schema": "sparkring-managed-model-readiness/v1", "ready": False, "samples": []}
    while clock() < deadline:
        observation = probe(plan, deadline)
        receipt["samples"].append(observation)
        if observation["ready"]:
            receipt["ready"] = True
            break
        sleep(max(0.0, min(POLL_SECONDS, deadline - clock())))
    receipt["elapsed_seconds"] = clock() - started
    if not receipt["ready"]:
        receipt["error"] = "Readiness deadline exceeded; inspect per-rank container and HTTP results"
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True, help="Rendered directory containing site.json and rank0.env")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output", type=Path, help="Optional absent receipt path; never overwrite an existing file")
    args = parser.parse_args()
    if args.output is not None and (args.output.exists() or args.output.is_symlink()):
        parser.error("Output receipt must not already exist")
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 900:
        parser.error("Timeout must be finite and greater than zero through 900 seconds")
    receipt = wait(load_launch(args.launch), args.timeout)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
    print(json.dumps(receipt, indent=2))
    print("READY" if receipt["ready"] else "NOT READY")
    return 0 if receipt["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
