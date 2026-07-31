#!/usr/bin/env python3
"""Dry-run-first four-rank launcher for the public EXL3 profile."""

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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sparkring_site import SiteConfig, SiteConfigError, load_site  # noqa: E402


SCHEMA = "sparkring-public-exl3-launch/v1"
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*")
STANDARD_PROFILE_ID = "glm52-exl3-tr3-3.25bpw"
HUGE_CONTEXT_PROFILE_ID = "glm52-exl3-tr3-3.25bpw-native-1m"
PROFILE_IDS = frozenset((STANDARD_PROFILE_ID, HUGE_CONTEXT_PROFILE_ID))
RECIPE_PATH = ROOT / "recipes/glm52-exl3-tr3-3.25bpw.json"
DEFAULT_MAX_NUM_BATCHED_TOKENS = 4096
EXPERIMENT_MAX_NUM_BATCHED_TOKENS = frozenset((2048, 3072, 4096))


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class Profile:
    document: dict

    def __getattr__(self, name: str):
        try:
            return self.document[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True)
class RemoteAction:
    rank: int
    ssh_target: str
    argv: tuple[str, ...]

    @property
    def shell_command(self) -> str:
        return shlex.join(self.argv)


def load_profile(path: Path) -> Profile:
    document = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "profile_id",
        "engine",
        "container_name",
        "image",
        "image_id",
        "model_host_path",
        "model_container_path",
        "jit_cache_host_path",
        "model_repository",
        "model_revision",
        "model_config_sha256",
        "model_index_sha256",
        "model_tier_bitmap_sha256",
        "model_manifest_sha256",
        "model_shard_count",
        "model_weight_bytes",
        "shm_size",
        "startup_timeout_seconds",
        "environment",
        "extra_vllm_args",
    }
    if set(document) != required:
        raise ProfileError(
            f"profile keys differ: missing={sorted(required - set(document))}, "
            f"unknown={sorted(set(document) - required)}"
        )
    if document["schema"] != SCHEMA:
        raise ProfileError("wrong profile schema")
    if document["profile_id"] not in PROFILE_IDS:
        raise ProfileError("wrong public profile id")
    if document["engine"] not in ("docker", "podman"):
        raise ProfileError("engine must be docker or podman")
    if not NAME.fullmatch(document["container_name"]):
        raise ProfileError("invalid container name")
    if not IMAGE_ID.fullmatch(document["image_id"]):
        raise ProfileError(
            "image_id must be the exact derived sha256 image ID; "
            "copy launch.template.json to an ignored local file after building"
        )
    if not HEX40.fullmatch(document["model_revision"]):
        raise ProfileError("model revision must be 40 lowercase hex")
    for name in (
        "model_config_sha256",
        "model_index_sha256",
        "model_tier_bitmap_sha256",
        "model_manifest_sha256",
    ):
        if not HEX64.fullmatch(document[name]):
            raise ProfileError(f"{name} must be 64 lowercase hex")
    if document["model_shard_count"] != 81:
        raise ProfileError("public EXL3 contract requires 81 model shards")
    if document["model_weight_bytes"] != 339069245936:
        raise ProfileError(
            "public EXL3 contract requires exact pinned model weight bytes"
        )
    if not all(
        isinstance(path_value, str) and path_value.startswith("/")
        for path_value in (
            document["model_host_path"],
            document["model_container_path"],
            document["jit_cache_host_path"],
        )
    ):
        raise ProfileError("model paths must be absolute")
    if not isinstance(document["environment"], dict):
        raise ProfileError("environment must be an object")
    if not isinstance(document["extra_vllm_args"], list):
        raise ProfileError("extra_vllm_args must be a list")
    _validate_contract(document)
    return Profile(document)


def _option_values(arguments: list[str], option: str) -> list[str]:
    return [
        arguments[index + 1]
        for index, value in enumerate(arguments[:-1])
        if value == option
    ]


def _replace_option_value(
    arguments: list[str],
    option: str,
    value: str,
) -> list[str]:
    indices = [
        index
        for index, argument in enumerate(arguments[:-1])
        if argument == option
    ]
    if len(indices) != 1:
        raise ProfileError(f"public EXL3 contract requires one {option}")
    updated = list(arguments)
    updated[indices[0] + 1] = value
    return updated


def _effective_max_num_batched_tokens(requested: int | None) -> int:
    effective = (
        DEFAULT_MAX_NUM_BATCHED_TOKENS if requested is None else requested
    )
    if effective not in EXPERIMENT_MAX_NUM_BATCHED_TOKENS:
        allowed = ", ".join(
            str(value) for value in sorted(EXPERIMENT_MAX_NUM_BATCHED_TOKENS)
        )
        raise ProfileError(
            "public EXL3 max batched tokens experiment must be one of "
            f"{allowed}"
        )
    return effective


def _validate_contract(document: dict) -> None:
    if document["model_repository"] != "willfalco/GLM-5.2-EXL3-TR3-3.25bpw":
        raise ProfileError("wrong EXL3 model repository")
    environment = document["environment"]
    if document["profile_id"] == HUGE_CONTEXT_PROFILE_ID:
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        expected_environment = recipe["serving"]["environment"]
        if environment != expected_environment:
            missing = sorted(set(expected_environment) - set(environment))
            extra = sorted(set(environment) - set(expected_environment))
            changed = sorted(
                name
                for name in set(environment) & set(expected_environment)
                if environment[name] != expected_environment[name]
            )
            raise ProfileError(
                "native-1m EXL3 environment differs from the published recipe: "
                f"missing={missing}, extra={extra}, changed={changed}"
            )
        if document["extra_vllm_args"] != recipe["serving"]["vllm_args"]:
            raise ProfileError(
                "native-1m EXL3 vLLM arguments differ from the published recipe"
            )
    context_profile = environment.get("SPARKRING_EXL3_CONTEXT_PROFILE")
    if document["profile_id"] == STANDARD_PROFILE_ID:
        if context_profile is not None:
            raise ProfileError(
                "standard public EXL3 profile must not select a context profile"
            )
        context_expected = {
            "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS": "262144",
            "VLLM_SPARK_KV_CACHE_MEMORY_BYTES": "7000000000",
            "VLLM_SPARK_MAX_MODEL_LEN": "262144",
        }
    elif context_profile == "native-1m-kv9gb":
        context_expected = {
            "SPARKRING_EXL3_CONTEXT_PROFILE": "native-1m-kv9gb",
            "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS": "1048576",
            "VLLM_SPARK_KV_CACHE_MEMORY_BYTES": "9000000000",
            "VLLM_SPARK_MAX_MODEL_LEN": "1048576",
        }
    else:
        raise ProfileError(
            "native-1m public EXL3 profile requires "
            "SPARKRING_EXL3_CONTEXT_PROFILE=native-1m-kv9gb"
        )
    common_expected = {
        "SPARK_CONTEXT_CACHE_ENABLE": "0",
        "B12X_NSA_CONTIGUOUS_PREFILL_BLOCK_K": "auto",
        "VLLM_B12X_MLA_CKV_GATHER": "1",
        "VLLM_EXL3_TRELLIS_MIN_M": "1",
        "VLLM_EXL3_TRELLIS_MAX_M": "32",
        "VLLM_SPARK_KV_CACHE_DTYPE": "nvfp4_ds_mla",
        "VLLM_SPARK_MAX_NUM_BATCHED_TOKENS": "4096",
        "VLLM_SPARK_MAX_NUM_SEQS": "8",
        "VLLM_SPARK_MAX_QUERY_ROWS": "32",
        "VLLM_SPARK_MTP_TOKENS": "3",
        "VLLM_SPARK_NCCL_TRANSPORT_MODE": "switchless_ib",
        **context_expected,
    }
    for name, value in common_expected.items():
        if environment.get(name) != value:
            raise ProfileError(f"public EXL3 contract requires {name}={value}")

    mode = environment.get("VLLM_SPARK_MTP_MODE_ID")
    if mode == "fixed-mtp3":
        mode_expected = {
            "SPARK_ADAPTIVE_MTP_CONTROL": "0",
            "SPARK_GLM52_MTP_INDEX_REUSE": "0",
            "VLLM_ADAPTIVE_SPEC_DEPTHS": None,
            "VLLM_SPARK_MTP_ADAPTIVE_WINDOW": "0",
            "VLLM_SPARK_TRUE_ADAPTIVE_DRAFT": "0",
        }
        expected_spec = {
            "method": "mtp",
            "num_speculative_tokens": 3,
            "moe_backend": "triton",
            "draft_sample_method": "greedy",
        }
    elif mode == "adaptive-mtp2-3-window32":
        mode_expected = {
            "CUDA_DEVICE_MAX_CONNECTIONS": "32",
            "SPARK_ADAPTIVE_MTP_CONTROL": "1",
            "SPARK_GLM52_MTP_INDEX_REUSE": "1",
            "VLLM_ADAPTIVE_SPEC_DEPTHS": "2,3",
            "VLLM_SPARK_COMPILATION_PROFILE": (
                "public-exl3-adaptive-mtp2-3-q32"
            ),
            "VLLM_SPARK_MTP_ADAPTIVE_WINDOW": "32",
            "VLLM_SPARK_TRUE_ADAPTIVE_DRAFT": "1",
        }
        expected_spec = {
            "method": "mtp",
            "num_speculative_tokens": 3,
            "moe_backend": "triton",
            "draft_sample_method": "greedy",
            "adaptive_speculative_tokens_window": 32,
        }
    else:
        raise ProfileError(
            "public EXL3 contract requires fixed-mtp3 or "
            "adaptive-mtp2-3-window32"
        )
    for name, value in mode_expected.items():
        if name not in environment or environment[name] != value:
            raise ProfileError(f"public EXL3 contract requires {name}={value}")

    for name in (
        "HYBRID_NF3",
        "VLLM_PREFIX_CACHE_RETENTION_INTERVAL",
        "VLLM_SPARK_NF3_PROFILE",
        "VLLM_SPARK_NF3_SINGLE_COMPILE_RANGE",
        "VLLM_SPARK_NF3_STARTUP_PROFILE_MAX_TOKENS",
        "VLLM_SPARK_NF3_WORKSPACE_RESERVE_BYTES",
    ):
        if name not in environment or environment[name] is not None:
            raise ProfileError(f"public EXL3 contract must explicitly unset {name}")
    args = document["extra_vllm_args"]
    options = {
        "--quantization": "exl3",
        "--moe-backend": "b12x",
        "--dcp-comm-backend": "ag_rs",
        "--attention-backend": "B12X_MLA_SPARSE",
        "--kv-cache-dtype": "nvfp4_ds_mla",
        "--max-cudagraph-capture-size": "32",
        "--max-num-batched-tokens": "4096",
    }
    for option, value in options.items():
        if _option_values(args, option) != [value]:
            raise ProfileError(f"public EXL3 contract requires {option} {value}")
    speculative = _option_values(args, "--speculative-config")
    if len(speculative) != 1:
        raise ProfileError("public EXL3 contract requires one speculative config")
    spec = json.loads(speculative[0])
    if spec != expected_spec:
        raise ProfileError(
            f"public EXL3 contract requires exact {mode} speculative config"
        )
    if "--enforce-eager" in args:
        raise ProfileError("public EXL3 rank-sliced profile requires CUDA graphs")


def _context(site: SiteConfig, rank_id: int) -> dict[str, str]:
    rank = site.rank(rank_id)
    peers = {peer.rank: peer for peer in rank.transport_peers}
    ports = {port.peer_rank: port for port in rank.ring_ports}
    ordered = [rank_id ^ 1, rank_id ^ 3]
    master = site.rank(site.serving.master_rank)
    return {
        "rank": str(rank_id),
        "world_size": str(len(site.ranks)),
        "master_addr": str(master.management.address),
        "master_port": str(site.serving.master_port),
        "peer0_addr": str(peers[ordered[0]].address),
        "peer0_device": ports[ordered[0]].rdma_device,
        "peer0_gid": str(ports[ordered[0]].roce_gid_index),
        "peer1_addr": str(peers[ordered[1]].address),
        "peer1_device": ports[ordered[1]].rdma_device,
        "peer1_gid": str(ports[ordered[1]].roce_gid_index),
    }


def _validate_site(site: SiteConfig) -> None:
    if len(site.ranks) != 4:
        raise ProfileError("public EXL3 profile requires exactly four ranks")
    expected = {
        "tensor_parallel_size": 4,
        "decode_context_parallel_size": 4,
    }
    for name, value in expected.items():
        if getattr(site.serving, name) != value:
            raise ProfileError(f"site serving.{name} must equal {value}")


def _base_environment(site: SiteConfig, profile: Profile, rank_id: int) -> dict:
    context = _context(site, rank_id)
    rank = site.rank(rank_id)
    if context["peer0_gid"] != context["peer1_gid"]:
        raise ProfileError(f"rank {rank_id} has mismatched RoCE GID indices")
    return {
        "GLOO_SOCKET_IFNAME": rank.management.interface,
        "MASTER_ADDR": context["master_addr"],
        "MASTER_PORT": context["master_port"],
        "NCCL_IB_GID_INDEX": context["peer0_gid"],
        "NCCL_IB_HCA": f"{context['peer0_device']},{context['peer1_device']}",
        "NCCL_IB_SUBNET_PREFIX_LEN": "24",
        "NCCL_SOCKET_IFNAME": rank.management.interface,
        "RANK": context["rank"],
        "SPARKRING_IMAGE_DIGEST": profile.image_id,
        "SPARKRING_MODEL_CONFIG_SHA256": profile.model_config_sha256,
        "SPARKRING_MODEL_INDEX_SHA256": profile.model_index_sha256,
        "SPARKRING_MODEL_PATH": profile.model_container_path,
        "SPARKRING_MODEL_REPOSITORY": profile.model_repository,
        "SPARKRING_MODEL_REVISION": profile.model_revision,
        "SPARKRING_MODEL_TIER_BITMAP_SHA256": profile.model_tier_bitmap_sha256,
        "SPARK_TP4_DEVICE0": context["peer0_device"],
        "SPARK_TP4_DEVICE1": context["peer1_device"],
        "SPARK_TP4_GID0": context["peer0_gid"],
        "SPARK_TP4_GID1": context["peer1_gid"],
        "SPARK_TP4_PEER0": context["peer0_addr"],
        "SPARK_TP4_PEER1": context["peer1_addr"],
        "WORLD_SIZE": context["world_size"],
    }


def container_name(profile: Profile, rank: int) -> str:
    return f"{profile.container_name}-r{rank}"


def model_verification_script(profile: Profile) -> str:
    checks = [
        ("config.json", profile.model_config_sha256),
        ("model.safetensors.index.json", profile.model_index_sha256),
        ("tier_bitmap.json", profile.model_tier_bitmap_sha256),
        ("MANIFEST.sha256", profile.model_manifest_sha256),
    ]
    metadata_checks = " && ".join(
        f"test \"$(sha256sum -- "
        f"{shlex.quote(profile.model_host_path + '/' + name)} "
        f"| awk '{{print $1}}')\" = {expected}"
        for name, expected in checks
    )
    shard_check = (
        "import json,os,sys;"
        "p=sys.argv[1];"
        "i=json.load(open(os.path.join(p,'model.safetensors.index.json')));"
        "s=set(i['weight_map'].values());"
        f"assert len(s)=={profile.model_shard_count};"
        "assert all(os.path.isfile(os.path.join(p,f)) for f in s);"
        f"assert sum(os.path.getsize(os.path.join(p,f)) for f in s)"
        f"=={profile.model_weight_bytes}"
    )
    return (
        f"{metadata_checks} && python3 -c {shlex.quote(shard_check)} "
        f"{shlex.quote(profile.model_host_path)}"
    )


def start_actions(
    site: SiteConfig,
    profile: Profile,
    *,
    max_num_batched_tokens: int | None = None,
) -> list[RemoteAction]:
    _validate_site(site)
    effective_batch_tokens = _effective_max_num_batched_tokens(
        max_num_batched_tokens
    )
    effective_extra_vllm_args = _replace_option_value(
        profile.extra_vllm_args,
        "--max-num-batched-tokens",
        str(effective_batch_tokens),
    )
    actions = []
    for rank in site.ranks:
        context = _context(site, rank.id)
        environment = _base_environment(site, profile, rank.id)
        environment.update(profile.environment)
        environment["VLLM_SPARK_MAX_NUM_BATCHED_TOKENS"] = str(
            effective_batch_tokens
        )
        command = [
            profile.engine,
            "run",
            "--detach",
            "--name",
            container_name(profile, rank.id),
            "--label",
            "org.sparkring.managed=true",
            "--label",
            f"org.sparkring.exl3-profile={profile.profile_id}",
            "--network",
            "host",
            "--ipc",
            "host",
            "--gpus",
            "all",
            "--shm-size",
            profile.shm_size,
            "--ulimit",
            "memlock=-1:-1",
            "--cap-add",
            "IPC_LOCK",
            "--device",
            "/dev/infiniband:/dev/infiniband",
            "--volume",
            f"{profile.model_host_path}:{profile.model_container_path}:ro",
            "--volume",
            f"{profile.jit_cache_host_path}:/cache/jit",
            profile.image,
            "serve",
            profile.model_container_path,
            "--tensor-parallel-size",
            "4",
            "--decode-context-parallel-size",
            "4",
            "--max-model-len",
            profile.environment["VLLM_SPARK_MAX_MODEL_LEN"],
            "--kv-cache-memory-bytes",
            profile.environment["VLLM_SPARK_KV_CACHE_MEMORY_BYTES"],
            "--max-num-seqs",
            profile.environment["VLLM_SPARK_MAX_NUM_SEQS"],
            "--port",
            str(site.serving.api_port),
            "--distributed-executor-backend",
            "mp",
            "--nnodes",
            "4",
            "--node-rank",
            str(rank.id),
            "--master-addr",
            context["master_addr"],
            "--master-port",
            context["master_port"],
        ]
        env_args: list[str] = []
        for name, value in sorted(environment.items()):
            env_args.extend(("--env", name if value is None else f"{name}={value}"))
        image_index = command.index(profile.image)
        command[image_index:image_index] = env_args
        command.extend(effective_extra_vllm_args)
        if rank.id != site.serving.master_rank:
            command.append("--headless")
        model_checks = (
            f"test \"$({profile.engine} image inspect --format '{{{{.Id}}}}' "
            f"{shlex.quote(profile.image)})\" = {shlex.quote(profile.image_id)}"
            f" && {model_verification_script(profile)}"
            f" && exec {shlex.join(command)}"
        )
        actions.append(
            RemoteAction(rank.id, rank.ssh_target, ("sh", "-lc", model_checks))
        )
    return actions


def simple_actions(
    site: SiteConfig, profile: Profile, operation: str
) -> list[RemoteAction]:
    actions = []
    for rank in site.ranks:
        name = container_name(profile, rank.id)
        if operation == "status":
            argv = (profile.engine, "inspect", "--format", "{{.State.Status}}", name)
        elif operation == "stop":
            script = (
                f"managed=$({profile.engine} inspect --format "
                f"'{{{{index .Config.Labels \"org.sparkring.exl3-profile\"}}}}' "
                f"{shlex.quote(name)} 2>/dev/null) || exit 0; "
                f"test \"$managed\" = {shlex.quote(profile.profile_id)} || exit 73; "
                f"exec {profile.engine} rm --force {shlex.quote(name)}"
            )
            argv = ("sh", "-lc", script)
        elif operation == "verify-image":
            script = (
                f"test \"$({profile.engine} image inspect --format "
                f"'{{{{.Id}}}}' {shlex.quote(profile.image)})\" = "
                f"{shlex.quote(profile.image_id)}"
            )
            argv = ("sh", "-lc", script)
        elif operation == "verify-model":
            script = model_verification_script(profile)
            argv = ("sh", "-lc", script)
        else:
            raise ValueError(operation)
        actions.append(RemoteAction(rank.id, rank.ssh_target, argv))
    return actions


def execute(actions: list[RemoteAction], timeout: int) -> dict[int, dict]:
    def one(action: RemoteAction):
        remote = shlex.join(("sh", "-lc", action.shell_command))
        return subprocess.run(
            ["ssh", "-o", "BatchMode=yes", action.ssh_target, remote],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(actions)) as pool:
        pending = {action.rank: pool.submit(one, action) for action in actions}
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        choices=tuple(sorted(EXPERIMENT_MAX_NUM_BATCHED_TOKENS)),
        help=(
            "bounded public A/B override; profile default remains 4096 and "
            "only 2048, 3072, or 4096 are accepted"
        ),
    )
    parser.add_argument(
        "command",
        choices=("plan", "start", "stop", "status", "verify-image", "verify-model"),
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "plan" and args.execute:
            raise ProfileError("plan is always offline; remove --execute")
        if (
            args.max_num_batched_tokens is not None
            and args.command not in ("plan", "start")
        ):
            raise ProfileError(
                "--max-num-batched-tokens applies only to plan or start"
            )
        site = load_site(args.site)
        profile = load_profile(Path(args.profile))
        effective_batch_tokens = _effective_max_num_batched_tokens(
            args.max_num_batched_tokens
        )
        actions = (
            start_actions(
                site,
                profile,
                max_num_batched_tokens=args.max_num_batched_tokens,
            )
            if args.command in ("plan", "start")
            else simple_actions(site, profile, args.command)
        )
    except (OSError, KeyError, json.JSONDecodeError, SiteConfigError, ProfileError) as exc:
        parser.error(str(exc))
    document = {
        "schema": "sparkring-public-exl3-plan/v1",
        "command": args.command,
        "mutates_remote": args.command in ("start", "stop"),
        "profile_attestation": {
            "profile_id": profile.profile_id,
            "image_id": profile.image_id,
            "model_revision": profile.model_revision,
        },
        "effective_settings": {
            "default_max_num_batched_tokens": (
                DEFAULT_MAX_NUM_BATCHED_TOKENS
            ),
            "max_num_batched_tokens": effective_batch_tokens,
        },
        "experiment_overrides": (
            {
                "max_num_batched_tokens": effective_batch_tokens,
            }
            if args.max_num_batched_tokens is not None
            and effective_batch_tokens != DEFAULT_MAX_NUM_BATCHED_TOKENS
            else {}
        ),
        "actions": [
            {
                "rank": action.rank,
                "ssh_target": action.ssh_target,
                "remote_command": action.shell_command,
            }
            for action in actions
        ],
    }
    if not args.execute:
        print(json.dumps(document, indent=2))
        return 0
    results = execute(actions, profile.startup_timeout_seconds)
    print(json.dumps({"plan": document, "results": results}, indent=2))
    return 0 if all(item["exit_code"] == 0 for item in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
