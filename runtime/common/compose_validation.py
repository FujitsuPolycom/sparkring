"""Validate standalone SparkRing Compose recipes without a Docker daemon or GPUs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from runtime.common import compose, profiles, standalone_compose

VARIABLES = ("SPARKRING_MODEL_DIR", "SPARKRING_CACHE_DIR", "SPARKRING_MASTER_ADDR",
             "SPARKRING_HOST_IP", "SPARKRING_INTERFACE")
SCOPE = "Offline configuration and simulated rank handshake only. No images pulled, containers started, hosts contacted or inference performed. GPU, RDMA, image contents and actual model files remain untested."


class UniqueLoader(yaml.SafeLoader):
    """Allow generated YAML merges, but reject duplicate explicit mapping keys."""


def mapping(loader, node):
    keys = set()
    for key_node, _ in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in keys:
            raise ValueError("Duplicate or non-string YAML mapping key: " + str(key))
        keys.add(key)
    loader.flatten_mapping(node)
    return yaml.SafeLoader.construct_mapping(loader, node)


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)


def parse(text):
    value = yaml.load(text, Loader=UniqueLoader)
    if not isinstance(value, dict):
        raise ValueError("Expected a SparkRing Compose mapping")
    return value


def differences(actual, expected, path="$", limit=12):
    result = []
    if type(actual) is not type(expected):
        return [path]
    if isinstance(expected, dict):
        for key in sorted(set(actual) | set(expected)):
            if key not in actual or key not in expected:
                result.append(path + "." + str(key))
            else:
                result.extend(differences(actual[key], expected[key], path + "." + str(key)))
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            result.append(path + ".length")
        for index, (left, right) in enumerate(zip(actual, expected)):
            result.extend(differences(left, right, f"{path}[{index}]"))
    elif actual != expected:
        result.append(path)
    return result[:limit]


def inspect_recipe(text):
    document = parse(text)
    meta = document.get("x-sparkring", {})
    profile = meta.get("profile") if isinstance(meta, dict) else None
    if profile not in standalone_compose.SUPPORTED:
        raise ValueError("This validator expects a generated SparkRing TP2 recipe with x-sparkring metadata")
    variants = ("nvfp4-spark", "nvfp4-qad") if profile.startswith("glm53-") else (None,)
    for variant in variants:
        expected = parse(standalone_compose.render(profile, variant))
        if meta == expected["x-sparkring"]:
            break
    else:
        raise ValueError("Profile release or checkpoint identity differs from the registered recipe")
    # Reject external inputs before invoking Compose. Config can otherwise read
    # includes, env files and other local resources even without a daemon.
    if set(document) - set(expected):
        raise ValueError("Unsupported top-level fields: " + ", ".join(sorted(set(document) - set(expected))))
    for key in document:
        changed = differences(document[key], expected[key], "$." + key)
        if changed:
            raise ValueError("Recipe differs from the profile at " + ", ".join(changed))
    if not {"name", "x-sparkring", "services"} <= document.keys():
        raise ValueError("Recipe requires name, x-sparkring and services")
    return profile, variant, document


def environment(rank):
    # Preserve executable/OS configuration, not credentials or ambient Compose
    # selectors. --env-file also prevents an unrelated working-directory .env.
    allowed = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP",
               "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "LANG", "LC_ALL",
               "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432",
               "SYSTEMDRIVE", "ALLUSERSPROFILE", "HOMEDRIVE", "HOMEPATH"}
    values = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    values.update(SPARKRING_MODEL_DIR="/srv/mock/models", SPARKRING_CACHE_DIR="/srv/mock/cache",
                  SPARKRING_MASTER_ADDR="198.18.250.1", SPARKRING_HOST_IP=f"198.18.250.{rank+1}",
                  SPARKRING_INTERFACE="mockfabric0")
    return values


def resolve(text, project, rank, env, *, run=subprocess.run):
    argv = compose.compose_command(project)
    if rank is not None:
        argv += ["--profile", f"rank{rank}"]
    return run([*argv, "config", "--format", "json", "--no-path-resolution"],
               input=text, encoding="utf-8", text=True, capture_output=True, timeout=30, env=env)


def expected_service(profile, variant, rank, env):
    card, specs = standalone_compose.specifications(profile, variant)
    value = compose.service(specs[rank], card["image_reference"])
    value.update(profiles=[f"rank{rank}"], pull_policy="missing")
    value["environment"]["VLLM_HOST_IP"] = env["SPARKRING_HOST_IP"]
    for key in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"):
        value["environment"][key] = env["SPARKRING_INTERFACE"]
    if "MASTER_ADDR" in value["environment"]:
        value["environment"]["MASTER_ADDR"] = env["SPARKRING_MASTER_ADDR"]
    command = value["command"]
    command[command.index("--master-addr") + 1] = env["SPARKRING_MASTER_ADDR"]
    for mount in value["volumes"]:
        mount["source"] = env["SPARKRING_MODEL_DIR" if mount["read_only"] else "SPARKRING_CACHE_DIR"]
    return compose.escape(compose.normalize_service(value))


def option(command, flag):
    if command.count(flag) != 1 or command.index(flag) + 1 == len(command):
        raise ValueError("Mock rank needs exactly one " + flag)
    return command[command.index(flag) + 1]


def mock_pair(services):
    """Simulate rank registration from resolved argv, without sockets or processes.

    This validates rank relationships, not vLLM execution or NCCL behavior.
    The worker registers first and waits until its matching API rank arrives.
    """
    registrations = []
    events = []
    for rank in (1, 0):
        service = services[rank]
        command = service["command"]
        if option(command, "--node-rank") != str(rank):
            raise ValueError("Mock rank identity differs from its selected service")
        if option(command, "--nnodes") != "2" or option(command, "--tensor-parallel-size") != "2":
            raise ValueError("Mock pair requires two nodes and TP2")
        if ("--headless" in command) != (rank == 1):
            raise ValueError("Only the worker may be headless")
        registrations.append({"rank": rank, "master": option(command, "--master-addr"),
                              "port": option(command, "--master-port"), "model": option(command, "--served-model-name"),
                              "image": service["image"], "host_ip": service["environment"]["VLLM_HOST_IP"]})
        events.append("worker registered; waiting for head" if rank == 1 else "head registered")
    worker, head = registrations
    if any(worker[key] != head[key] for key in ("master", "port", "model", "image")):
        raise ValueError("Mock ranks disagree on rendezvous, model or image")
    if head["host_ip"] != head["master"] or worker["host_ip"] == head["host_ip"]:
        raise ValueError("Mock rank addresses are not a distinct worker and rank-0 master")
    events.append("two compatible ranks registered; simulated pair ready")
    return events


def validate(path, *, run=subprocess.run):
    path = Path(path)
    raw = path.read_bytes()
    report = {"schema": "sparkring-compose-validation/v1", "file": str(path),
              "sha256": hashlib.sha256(raw).hexdigest(), "passed": False,
              "checks": [], "scope": SCOPE}
    name = "canonical recipe and immutable image/checkpoint"
    try:
        # subprocess text mode translates LF on Windows. Normalize file CRLF
        # first, or piping a Windows-saved recipe produces CRCRLF and can turn
        # folded interpolation strings into invalid multiline expressions.
        text = raw.decode("utf-8-sig").replace("\r\n", "\n")
        profile, variant, document = inspect_recipe(text)
        report.update(profile=profile, variant=variant)
        report["checks"].append({"name": name, "passed": True})
        name = "Docker Compose CLI"
        version = run(["docker", "--context", "default", "compose", "version", "--short"],
                      encoding="utf-8", text=True, capture_output=True, timeout=30, env=environment(0))
        if version.returncode:
            raise ValueError(version.stderr.strip() or "Compose CLI unavailable")
        report["compose_version"] = version.stdout.strip()
        report["checks"].append({"name": name, "passed": True})
        resolved = {}
        for rank in (0, 1):
            name = f"rank{rank}: Compose resolution, rank isolation and profile settings"
            env = environment(rank)
            response = resolve(text, document["name"], rank, env, run=run)
            if response.returncode:
                raise ValueError(response.stderr.strip())
            services = json.loads(response.stdout).get("services", {})
            if set(services) != {f"rank{rank}"}:
                raise ValueError("Compose must select only the requested rank")
            resolved[rank] = services[f"rank{rank}"]
            changed = differences(compose.normalize_service(resolved[rank]), expected_service(profile, variant, rank, env))
            if changed:
                raise ValueError("Resolved settings differ at " + ", ".join(changed))
            report["checks"].append({"name": name, "passed": True})
        name = "no rank selected by default"
        response = resolve(text, document["name"], None, environment(0), run=run)
        if response.returncode or json.loads(response.stdout).get("services"):
            raise ValueError("Default Compose configuration must select no services")
        report["checks"].append({"name": name, "passed": True})
        for variable in VARIABLES:
            name = "missing input rejected: " + variable
            env = environment(0)
            del env[variable]
            response = resolve(text, document["name"], 0, env, run=run)
            if response.returncode == 0 or variable not in response.stderr:
                raise ValueError("Compose did not reject the missing required variable")
            report["checks"].append({"name": name, "passed": True})
        name = "mock two-rank registration"
        report["mock_events"] = mock_pair(resolved)
        report["checks"].append({"name": name, "passed": True})
        report["passed"] = True
    except (ValueError, KeyError, TypeError, OSError, yaml.YAMLError, subprocess.SubprocessError) as error:
        report["checks"].append({"name": name, "passed": False, "error": str(error)[:2000]})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sparkring validate-compose", description=__doc__)
    parser.add_argument("file", nargs="?", type=Path)
    parser.add_argument("--all", action="store_true", help="validate the three maintained standalone TP2 recipes")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path, help="save a new JSON validation report")
    args = parser.parse_args(argv)
    if bool(args.file) == args.all:
        parser.error("Select a file or --all")
    paths = ([profiles.ROOT / "profiles" / profile / "compose/standalone.yaml" for profile in standalone_compose.SUPPORTED]
             if args.all else [args.file])
    try:
        if args.output and args.output.exists():
            raise ValueError("Report already exists; choose a new output path")
        reports = [validate(path) for path in paths]
        summary = {"passed": all(report["passed"] for report in reports), "reports": reports, "scope": SCOPE}
        if args.output:
            from runtime.common.installer import write
            write(args.output, summary)
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            for report in reports:
                print(f"{'PASS' if report['passed'] else 'FAIL'} {report.get('profile', report['file'])}")
                for check in report["checks"]:
                    print(f"  {'PASS' if check['passed'] else 'FAIL'} {check['name']}" + (": " + check["error"] if "error" in check else ""))
            print(SCOPE)
        return 0 if summary["passed"] else 1
    except (OSError, ValueError) as error:
        print("Compose validation: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
