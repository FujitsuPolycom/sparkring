#!/usr/bin/env python3
"""Read-only four-rank container and HTTP readiness check for a rendered launch."""
from __future__ import annotations

import argparse
import hashlib
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


def registered_targets(launch, site, topology, targets):
    """Accept workers without healthchecks only when their TP4 image matches its receipt."""
    manifest = Path(launch) / "fabric-plan.json"
    if not manifest.is_file():
        return False
    rendered = json.loads(manifest.read_text())
    image = rendered.get("image", {})
    schema = image.get("schema") if isinstance(image, dict) else None
    adapters = {profile.r35.SCHEMA: (profile.r35, "r35", "/opt/sparkring/bin/sparkring"),
                profile.candidate.SCHEMA: (profile.candidate, "candidate", profile.candidate.ENTRYPOINT)}
    if schema not in adapters:
        return False
    adapter, release, wrapper = adapters[schema]
    checked = adapter.validate_receipt(image)
    selected = adapter.profile_contract(checked["installed"])["profiles"].get(site.get("runtime_profile"), {})
    if (selected.get("tensor_parallel_size") != 4 or selected.get("node_count") != 4
            or rendered.get("schema") != "sparkring-mtp3-mesh-render/v1"
            or rendered.get("topology_sha256") != topology.sha256):
        raise ValueError("Registered readiness requires the matching rendered TP4 profile")
    required = {"site.json", "fabric.json", "launch-rank.sh", *(f"rank{rank}.env" for rank in range(4))}
    if set(rendered.get("files", {})) != required:
        raise ValueError("Registered readiness requires the complete rendered launch inventory")
    for name in required:
        path = Path(launch) / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != rendered["files"][name]:
            raise ValueError("Rendered readiness input changed: " + name)
    for target in targets:
        environment = profile.defaults(Path(launch) / f"rank{target['rank']}.env")
        expected = {"IMAGE_ID": checked["image_id"], "IMAGE_REF": checked["image_reference"],
                    "SPARKRING_RUNTIME_RELEASE": release, "SOURCE_IMAGE_PROFILE": site["runtime_profile"],
                    "NODE_RANK": str(target["rank"]), "SPARKRING_NODE_RANK": str(target["rank"])}
        if any(environment.get(key) != value for key, value in expected.items()):
            raise ValueError("Rank readiness identity differs from the verified image/profile")
        target.update(image_id=checked["image_id"], runtime_profile=site["runtime_profile"],
                      runtime_release=release, wrapper=wrapper,
                      decode_context_parallel_size=selected["decode_context_parallel_size"],
                      headless_worker=target["rank"] != 0)
    return True


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
    targets = [{"rank": rank, "ssh_alias": aliases[rank],
                "name": site["container_prefix"] + f"-r{rank}"} for rank in range(4)]
    registered = registered_targets(launch, site, topology, targets)
    return {
        "containers": targets,
        "urls": [f"http://{address}:{ports[0]}/health", f"http://{address}:{ports[1]}/liveness"],
        "stable_samples_required": 2 if registered else 1,
    }


def inspect_command(alias, name):
    if not isinstance(alias, str) or not ALIAS.fullmatch(alias):
        raise ValueError("Invalid SSH alias")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name):
        raise ValueError("Invalid container name")
    remote = shlex.join(["docker", "inspect", "--format", "{{json .}}", name])
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
    container = json.loads(result.stdout)
    if (not isinstance(container, dict) or container.get("Name") != "/" + target["name"]
            or not isinstance(container.get("Id"), str) or not re.fullmatch("[0-9a-f]{64}", container["Id"])
            or not isinstance(container.get("Image"), str) or not re.fullmatch("sha256:[0-9a-f]{64}", container["Image"])):
        raise ValueError("Docker returned a different or malformed container identity")
    state = container.get("State")
    if (not isinstance(state, dict)
            or any(type(state.get(key)) is not bool for key in ("Running", "Paused", "Restarting", "Dead", "OOMKilled"))
            or not isinstance(state.get("Status"), str)
            or not isinstance(state.get("StartedAt"), str) or not state["StartedAt"]
            or type(state.get("Pid")) is not int or state["Pid"] < 0
            or type(container.get("RestartCount")) is not int or container["RestartCount"] < 0):
        raise ValueError("Docker returned an invalid container state")
    health = state.get("Health", {})
    if not isinstance(health, dict):
        raise ValueError("Docker returned an invalid container health state")
    stable_running = (state["Running"] and state["Status"] == "running" and state["Pid"] > 0
                      and not any(state[key] for key in ("Paused", "Restarting", "Dead", "OOMKilled")))
    health_required = True
    configured_health = True
    if "image_id" in target:
        config = container.get("Config")
        if container.get("Image") != target["image_id"] or not isinstance(config, dict):
            raise ValueError("Container image differs from the verified readiness image")
        command = config.get("Cmd")
        environment = config.get("Env")
        if (config.get("Entrypoint") != ["/opt/venv/bin/python"]
                or not isinstance(command, list) or any(not isinstance(value, str) for value in command)
                or command[:2] != [target["wrapper"], "serve"]
                or not isinstance(environment, list)):
            raise ValueError("Registered container is not using its verified serving wrapper")
        env = {}
        for value in environment:
            if not isinstance(value, str) or "=" not in value:
                raise ValueError("Registered container environment is malformed")
            key, content = value.split("=", 1)
            if key in env:
                raise ValueError("Registered container environment has duplicate names")
            env[key] = content
        expected = {"NODE_RANK": str(target["rank"]), "SPARKRING_NODE_RANK": str(target["rank"]),
                    "SOURCE_IMAGE_PROFILE": target["runtime_profile"], "SPARKRING_PROFILE_MODE": "custom"}
        if any(env.get(key) != value for key, value in expected.items()):
            raise ValueError("Registered container rank or profile differs from the launch")
        for flag, value in (("--node-rank", target["rank"]), ("--tensor-parallel-size", 4),
                            ("--nnodes", 4), ("--decode-context-parallel-size", target["decode_context_parallel_size"])):
            if command.count(flag) != 1 or command.index(flag) + 1 >= len(command) or command[command.index(flag) + 1] != str(value):
                raise ValueError("Registered container rank or topology arguments differ")
        if command.count("--headless") != int(target["headless_worker"]):
            raise ValueError("Registered container headless role differs from its rank")
        healthcheck = config.get("Healthcheck")
        if healthcheck is not None and not isinstance(healthcheck, dict):
            raise ValueError("Registered container health configuration is malformed")
        test = (healthcheck or {}).get("Test")
        if test is not None and (not isinstance(test, list) or any(not isinstance(value, str) for value in test)):
            raise ValueError("Registered container health test is malformed")
        if healthcheck is not None and (not test or (test != ["NONE"] and (test[0] not in ("CMD", "CMD-SHELL") or len(test) < 2))):
            raise ValueError("Registered container health test is malformed")
        absent_health = healthcheck is None or test == ["NONE"]
        configured_health = target["headless_worker"] or not absent_health
        health_required = not (target["headless_worker"] and absent_health and not health)
    passed = stable_running and configured_health and (health.get("Status") == "healthy" if health_required else True)
    return {**target, "id": container["Id"], "image": container.get("Image"),
            "running": state["Running"], "health": health.get("Status"), "health_required": health_required,
            "status": state["Status"], "pid": state["Pid"], "started_at": state["StartedAt"],
            "restart_count": container["RestartCount"], "ready": passed}


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
    previous = None
    stable_samples = 0
    while clock() < deadline:
        observation = probe(plan, deadline)
        receipt["samples"].append(observation)
        if observation["ready"]:
            required = plan.get("stable_samples_required", 1)
            identities = tuple((row["id"], row["pid"], row["started_at"], row["restart_count"])
                               for row in observation.get("containers", [])) if required > 1 else ()
            stable_samples = stable_samples + 1 if identities == previous else 1
            previous = identities
            if required == 1 or (len(identities) == 4 and stable_samples >= required):
                receipt["ready"] = True
                break
        else:
            previous, stable_samples = None, 0
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
