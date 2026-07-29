#!/usr/bin/env python3
"""Fail-closed four-rank launcher for the public-functional SparkRing lane.

Dry-run planning is the default. Remote mutation requires both a mutating
subcommand and ``--execute``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sparkring_site import SiteConfig, SiteConfigError, load_site

SCHEMA = "sparkring-public-launch/v1"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
_ABS = re.compile(r"^/[A-Za-z0-9._/+@:-]*[A-Za-z0-9._+@:-]$")
_ENV = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_PLACEHOLDER = re.compile(r"{([a-z0-9_]+)}")
_ALLOWED_PLACEHOLDERS = {
    "api_port",
    "master_addr",
    "master_port",
    "model_path",
    "peer0_addr",
    "peer0_device",
    "peer0_gid",
    "peer0_rank",
    "peer1_addr",
    "peer1_device",
    "peer1_gid",
    "peer1_rank",
    "rank",
    "world_size",
}


class LaunchConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LaunchConfig:
    engine: str
    container_name: str
    model_host_path: str
    shm_size: str
    startup_timeout_seconds: int
    environment: dict[str, str]
    extra_vllm_args: tuple[str, ...]


@dataclass(frozen=True)
class RemoteAction:
    rank: int
    ssh_target: str
    argv: tuple[str, ...]

    @property
    def shell_command(self) -> str:
        return shlex.join(self.argv)


def _exact_keys(value: dict, expected: set[str], where: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise LaunchConfigError(f"{where}: unknown key {unknown[0]!r}")
    if missing:
        raise LaunchConfigError(f"{where}: missing key {missing[0]!r}")


def load_launch(path: Path) -> LaunchConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchConfigError(f"{path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise LaunchConfigError(f"{path}: root must be an object")
    _exact_keys(
        raw,
        {
            "schema",
            "engine",
            "container_name",
            "model_host_path",
            "shm_size",
            "startup_timeout_seconds",
            "environment",
            "extra_vllm_args",
        },
        str(path),
    )
    if raw["schema"] != SCHEMA:
        raise LaunchConfigError(f"{path}: unsupported schema {raw['schema']!r}")
    engine = raw["engine"]
    if engine not in ("docker", "podman"):
        raise LaunchConfigError(f"{path}: engine must be docker or podman")
    name = raw["container_name"]
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise LaunchConfigError(f"{path}: invalid container_name")
    model_host_path = raw["model_host_path"]
    if not isinstance(model_host_path, str) or not _ABS.fullmatch(model_host_path):
        raise LaunchConfigError(f"{path}: model_host_path must be shell-safe absolute")
    shm_size = raw["shm_size"]
    if not isinstance(shm_size, str) or not re.fullmatch(r"[1-9][0-9]*[gGmM]", shm_size):
        raise LaunchConfigError(f"{path}: shm_size must look like 16g")
    timeout = raw["startup_timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 30 <= timeout <= 7200:
        raise LaunchConfigError(
            f"{path}: startup_timeout_seconds must be an integer in [30,7200]"
        )
    environment = raw["environment"]
    if not isinstance(environment, dict):
        raise LaunchConfigError(f"{path}: environment must be an object")
    checked_env: dict[str, str] = {}
    for key, value in environment.items():
        if not isinstance(key, str) or not _ENV.fullmatch(key):
            raise LaunchConfigError(f"{path}: invalid environment key {key!r}")
        if not isinstance(value, str) or "\x00" in value or "\n" in value:
            raise LaunchConfigError(f"{path}: environment {key} must be one line")
        _validate_placeholders(value, f"environment.{key}")
        checked_env[key] = value
    extra = raw["extra_vllm_args"]
    if not isinstance(extra, list) or not all(
        isinstance(value, str) and value and "\x00" not in value and "\n" not in value
        for value in extra
    ):
        raise LaunchConfigError(f"{path}: extra_vllm_args must be non-empty strings")
    for index, value in enumerate(extra):
        _validate_placeholders(value, f"extra_vllm_args[{index}]")
    return LaunchConfig(
        engine=engine,
        container_name=name,
        model_host_path=model_host_path,
        shm_size=shm_size,
        startup_timeout_seconds=timeout,
        environment=checked_env,
        extra_vllm_args=tuple(extra),
    )


def _validate_placeholders(value: str, where: str) -> None:
    observed = set(_PLACEHOLDER.findall(value))
    unknown = sorted(observed - _ALLOWED_PLACEHOLDERS)
    if unknown:
        raise LaunchConfigError(f"{where}: unknown placeholder {{{unknown[0]}}}")


def _context(site: SiteConfig, rank_id: int) -> dict[str, str]:
    rank = site.rank(rank_id)
    peers = sorted(rank.transport_peers, key=lambda peer: peer.rank)
    ports = {port.peer_rank: port for port in rank.ring_ports}
    master = site.rank(site.serving.master_rank)
    return {
        "api_port": str(site.serving.api_port),
        "master_addr": str(master.management.address),
        "master_port": str(site.serving.master_port),
        "model_path": site.runtime.model_path,
        "peer0_addr": str(peers[0].address),
        "peer0_device": ports[peers[0].rank].rdma_device,
        "peer0_gid": str(ports[peers[0].rank].roce_gid_index),
        "peer0_rank": str(peers[0].rank),
        "peer1_addr": str(peers[1].address),
        "peer1_device": ports[peers[1].rank].rdma_device,
        "peer1_gid": str(ports[peers[1].rank].roce_gid_index),
        "peer1_rank": str(peers[1].rank),
        "rank": str(rank_id),
        "world_size": str(len(site.ranks)),
    }


def _expand(value: str, context: dict[str, str]) -> str:
    return _PLACEHOLDER.sub(lambda match: context[match.group(1)], value)


def container_name(config: LaunchConfig, rank: int) -> str:
    return f"{config.container_name}-r{rank}"


def _base_environment(site: SiteConfig, rank_id: int) -> dict[str, str]:
    context = _context(site, rank_id)
    rank = site.rank(rank_id)
    return {
        "GLOO_SOCKET_IFNAME": rank.management.interface,
        "NCCL_IB_HCA": f"{context['peer0_device']},{context['peer1_device']}",
        "NCCL_SOCKET_IFNAME": rank.management.interface,
        "RANK": context["rank"],
        "WORLD_SIZE": context["world_size"],
        "MASTER_ADDR": context["master_addr"],
        "MASTER_PORT": context["master_port"],
        "SPARKRING_IMAGE_DIGEST": site.runtime.container_image_digest,
        "SPARKRING_MODEL_CONFIG_SHA256": _model_config_sha(site),
        "SPARKRING_MODEL_PATH": site.runtime.model_path,
        "SPARKRING_MODEL_REPOSITORY": site.runtime.model_repo,
        "SPARKRING_MODEL_REVISION": site.runtime.model_revision,
        "SPARKRING_RUNTIME_MANIFEST": "/opt/sparkring/runtime-manifest.json",
        "SPARK_TP4_DEVICE0": context["peer0_device"],
        "SPARK_TP4_DEVICE1": context["peer1_device"],
        "SPARK_TP4_GID0": context["peer0_gid"],
        "SPARK_TP4_GID1": context["peer1_gid"],
        "SPARK_TP4_PEER0": context["peer0_addr"],
        "SPARK_TP4_PEER1": context["peer1_addr"],
    }


def _model_config_sha(site: SiteConfig) -> str:
    # The immutable runtime lock owns config.json identity. checkpoint_sha256
    # may name a safetensors index and is deliberately not substituted here.
    lock = json.loads(
        (Path(__file__).resolve().parents[1] / "runtime/runtime-lock.json").read_text(
            encoding="utf-8"
        )
    )
    model = lock["model"]
    if (
        model["repository"] != site.runtime.model_repo
        or model["revision"] != site.runtime.model_revision
    ):
        raise LaunchConfigError(
            "site model identity differs from runtime/runtime-lock.json"
        )
    return str(model["config_sha256"])


def start_actions(site: SiteConfig, config: LaunchConfig) -> list[RemoteAction]:
    actions: list[RemoteAction] = []
    for rank in site.ranks:
        context = _context(site, rank.id)
        environment = _base_environment(site, rank.id)
        environment.update(
            {key: _expand(value, context) for key, value in config.environment.items()}
        )
        argv = [
            config.engine,
            "run",
            "--detach",
            "--name",
            container_name(config, rank.id),
            "--label",
            "org.sparkring.managed=true",
            "--label",
            f"org.sparkring.site={site.name}",
            "--network",
            "host",
            "--ipc",
            "host",
            "--gpus",
            "all",
            "--shm-size",
            config.shm_size,
            "--ulimit",
            "memlock=-1:-1",
            "--cap-add",
            "IPC_LOCK",
            "--device",
            "/dev/infiniband:/dev/infiniband",
            "--volume",
            f"{config.model_host_path}:{site.runtime.model_path}:ro",
            "--volume",
            f"{site.paths.jit_cache_dir}:{site.paths.jit_cache_dir}",
            "--volume",
            f"{site.paths.context_cache_dir}:{site.paths.context_cache_dir}",
        ]
        for key, value in sorted(environment.items()):
            argv.extend(("--env", f"{key}={value}"))
        argv.extend(
            (
                site.runtime.container_image,
                "serve",
                site.runtime.model_path,
                "--tensor-parallel-size",
                str(site.serving.tensor_parallel_size),
                "--decode-context-parallel-size",
                str(site.serving.decode_context_parallel_size),
                "--max-model-len",
                str(site.serving.max_model_len),
                "--kv-cache-memory-bytes",
                str(site.serving.kv_cache_bytes_per_rank),
                "--max-num-seqs",
                str(site.serving.max_num_seqs),
                "--port",
                str(site.serving.api_port),
                "--distributed-executor-backend",
                "mp",
                "--nnodes",
                str(len(site.ranks)),
                "--node-rank",
                str(rank.id),
                "--master-addr",
                context["master_addr"],
                "--master-port",
                context["master_port"],
            )
        )
        argv.extend(_expand(value, context) for value in config.extra_vllm_args)
        if rank.id != site.serving.master_rank:
            argv.append("--headless")
        actions.append(RemoteAction(rank.id, rank.ssh_target, tuple(argv)))
    return actions


def simple_actions(
    site: SiteConfig, config: LaunchConfig, operation: str
) -> list[RemoteAction]:
    actions = []
    for rank in site.ranks:
        name = container_name(config, rank.id)
        if operation == "stop":
            inspect = shlex.join(
                (
                    config.engine,
                    "container",
                    "inspect",
                    "--format",
                    '{{index .Config.Labels "org.sparkring.managed"}}',
                    name,
                )
            )
            remove = shlex.join((config.engine, "rm", "--force", name))
            script = (
                f"managed=$({inspect} 2>/dev/null) || exit 0; "
                '[ "$managed" = true ] || exit 73; '
                f"exec {remove}"
            )
            argv = ("sh", "-c", script)
        elif operation == "status":
            argv = (
                config.engine,
                "inspect",
                "--format",
                "{{.State.Status}}",
                name,
            )
        elif operation == "verify-rollback":
            argv = (
                "sh",
                "-c",
                f"! {shlex.join((config.engine, 'container', 'inspect', name))} "
                ">/dev/null 2>&1",
            )
        else:
            raise ValueError(operation)
        actions.append(RemoteAction(rank.id, rank.ssh_target, tuple(argv)))
    return actions


def run_remote(action: RemoteAction, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            action.ssh_target,
            "--",
            "sh",
            "-lc",
            action.shell_command,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def execute(actions: list[RemoteAction], timeout: int) -> dict[int, dict]:
    results: dict[int, dict] = {}
    if not actions:
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(actions)) as pool:
        pending = {
            action.rank: pool.submit(run_remote, action, timeout) for action in actions
        }
        for rank, future in pending.items():
            try:
                result = future.result()
                results[rank] = {
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            except subprocess.TimeoutExpired as exc:
                results[rank] = {
                    "exit_code": 124,
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "remote command timed out",
                }
    return results


def plan_document(command: str, actions: list[RemoteAction]) -> dict:
    return {
        "schema": "sparkring-public-launch-plan/v1",
        "command": command,
        "mutates_remote": command in ("start", "stop"),
        "actions": [
            {
                "rank": action.rank,
                "ssh_target": action.ssh_target,
                "remote_command": action.shell_command,
            }
            for action in actions
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", required=True)
    parser.add_argument("--launch-config", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("command", choices=("plan", "start", "stop", "status", "verify-rollback"))
    args = parser.parse_args(argv)
    try:
        site = load_site(args.site)
        config = load_launch(Path(args.launch_config))
        actions = (
            start_actions(site, config)
            if args.command in ("plan", "start")
            else simple_actions(site, config, args.command)
        )
    except (OSError, KeyError, json.JSONDecodeError, SiteConfigError, LaunchConfigError) as exc:
        parser.error(str(exc))

    if not args.execute:
        print(json.dumps(plan_document(args.command, actions), indent=2))
        if args.command != "plan":
            print(
                f"DRY RUN: {args.command} made no remote connection; add --execute",
                file=sys.stderr,
            )
        return 0
    if args.command == "plan":
        parser.error("plan never executes; omit --execute")

    results = execute(actions, config.startup_timeout_seconds)
    failed = [rank for rank, result in results.items() if result["exit_code"] != 0]
    if args.command == "start" and failed:
        started = {
            rank for rank, result in results.items() if result["exit_code"] == 0
        }
        rollback = [
            action
            for action in simple_actions(site, config, "stop")
            if action.rank in started
        ]
        rollback_results = execute(rollback, config.startup_timeout_seconds)
    else:
        rollback_results = None
    print(
        json.dumps(
            {
                "schema": "sparkring-public-launch-result/v1",
                "command": args.command,
                "passed": not failed,
                "results": results,
                "rollback_results": rollback_results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
