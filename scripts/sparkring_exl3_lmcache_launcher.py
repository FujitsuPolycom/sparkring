#!/usr/bin/env python3
"""Dry-run-first launcher for the EXL3 CS512 four-server LMCache profile."""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sparkring_exl3_launcher as exl3  # noqa: E402
from sparkring_site import SiteConfigError, load_site  # noqa: E402


PROFILE_ID = exl3.LMCACHE_CS512_PROFILE_ID
SERVER_PREFIX = "glm52-sparkring-lmcache-cs512-server"
CONFIRMATION = "START-EXL3-LMCACHE-CS512-ALL-FOUR"
EXPERIMENTAL_MEMORY_PROFILES = {
    "kv4gb-480k-l1-0.5gb": {
        "kv_cache_memory_bytes": "4000000000",
        "max_model_len": "491520",
        "lmcache_l1_size_gb": 0.5,
    }
}


def recipe_lmcache(experiment: dict | None = None) -> dict:
    recipe = json.loads(exl3.RECIPE_PATH.read_text(encoding="utf-8"))
    config = copy.deepcopy(recipe["serving"].get("lmcache"))
    if not isinstance(config, dict):
        raise exl3.ProfileError("published EXL3 recipe has no LMCache contract")
    if experiment is not None:
        config["l1_size_gb"] = experiment["lmcache_l1_size_gb"]
    return config


def experiment_label(experiment_id: str | None) -> str:
    return PROFILE_ID if experiment_id is None else f"{PROFILE_ID}-exp-{experiment_id}"


def selected_experiment(experiment_id: str | None) -> dict | None:
    if experiment_id is None:
        return None
    try:
        return EXPERIMENTAL_MEMORY_PROFILES[experiment_id]
    except KeyError as exc:
        raise exl3.ProfileError(
            f"unknown experimental memory profile {experiment_id!r}"
        ) from exc


def server_name(rank: int) -> str:
    return f"{SERVER_PREFIX}-r{rank}"


def shell_action(rank, script: str) -> exl3.RemoteAction:
    return exl3.RemoteAction(rank.id, rank.ssh_target, ("sh", "-lc", script))


def server_start_actions(
    site,
    profile: exl3.Profile,
    *,
    experiment_id: str | None = None,
) -> list[exl3.RemoteAction]:
    experiment = selected_experiment(experiment_id)
    config = recipe_lmcache(experiment)
    effective_label = experiment_label(experiment_id)
    actions = []
    for rank in site.ranks:
        name = server_name(rank.id)
        command = [
            profile.engine,
            "run",
            "--detach",
            "--name",
            name,
            "--label",
            "org.sparkring.managed=true",
            "--label",
            f"org.sparkring.exl3-profile={effective_label}",
            "--label",
            "org.sparkring.component=lmcache-server",
            "--privileged",
            "--gpus",
            "all",
            "--network",
            "host",
            "--ipc",
            "host",
            "--shm-size",
            profile.shm_size,
            "--ulimit",
            "memlock=-1:-1",
            "--env",
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False",
            "--env",
            "LMCACHE_DISABLE_BANNER=1",
            "--entrypoint",
            "/opt/venv/bin/lmcache",
            profile.image_id,
            "server",
            "--instance-id",
            f"spark-r{rank.id}-cs512",
            "--host",
            "0.0.0.0",
            "--port",
            str(config["server_port"]),
            "--l1-size-gb",
            str(config["l1_size_gb"]),
            "--l1-init-size-gb",
            str(config["l1_init_size_gb"]),
            "--l1-use-lazy",
            "--eviction-policy",
            "LRU",
            "--supported-transfer-mode",
            config["transfer_mode"],
            "--max-gpu-workers",
            str(config["max_gpu_workers"]),
            "--max-cpu-workers",
            str(config["max_cpu_workers"]),
            "--chunk-size",
            str(config["chunk_size"]),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(config["http_port"]),
            "--disable-observability",
        ]
        if experiment_id is not None:
            privileged_index = command.index("--privileged")
            command[privileged_index:privileged_index] = [
                "--label",
                f"org.sparkring.experiment={experiment_id}",
            ]
        guard = (
            f"test \"$({profile.engine} image inspect --format '{{{{.Id}}}}' "
            f"{shlex.quote(profile.image)})\" = {shlex.quote(profile.image_id)}"
            f" && test -z \"$({profile.engine} ps -aq --filter "
            f"name=^/{shlex.quote(name)}$)\""
            f" && exec {shlex.join(command)}"
        )
        actions.append(shell_action(rank, guard))
    return actions


def server_health_actions(
    site, *, experiment_id: str | None = None,
) -> list[exl3.RemoteAction]:
    experiment = selected_experiment(experiment_id)
    config = recipe_lmcache(experiment)
    effective_label = experiment_label(experiment_id)
    actions = []
    for rank in site.ranks:
        name = server_name(rank.id)
        script = (
            f"test \"$({{ docker inspect -f '{{{{index .Config.Labels "
            f"\"org.sparkring.exl3-profile\"}}}}' {shlex.quote(name)}; }})\" "
            f"= {shlex.quote(effective_label)}"
            f" && for i in $(seq 1 60); do "
            f"test \"$(docker inspect -f '{{{{.State.Running}}}}' "
            f"{shlex.quote(name)} 2>/dev/null)\" = true"
            f" && curl -fsS http://127.0.0.1:{config['http_port']}/healthcheck "
            f">/dev/null && break; sleep 2; done"
            f" && test \"$(docker inspect -f '{{{{.RestartCount}}}}' "
            f"{shlex.quote(name)})\" = 0"
            f" && test \"$(docker inspect -f '{{{{.State.OOMKilled}}}}' "
            f"{shlex.quote(name)})\" = false"
            f" && ! docker logs {shlex.quote(name)} 2>&1 | "
            "grep -E 'CUDA out of memory|OutOfMemoryError|OOMKilled'"
            f" && curl -fsS http://127.0.0.1:{config['http_port']}/healthcheck"
            f" && curl -fsS http://127.0.0.1:{config['http_port']}/status"
        )
        actions.append(shell_action(rank, script))
    return actions


def engine_start_actions(
    site,
    profile: exl3.Profile,
    *,
    experiment_id: str | None = None,
) -> list[exl3.RemoteAction]:
    experiment = selected_experiment(experiment_id)
    config = recipe_lmcache(experiment)
    effective_label = experiment_label(experiment_id)
    document = copy.deepcopy(profile.document)
    document["environment"]["LMCACHE_DISABLE_BANNER"] = "1"
    document["environment"]["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:False"
    )
    if experiment is not None:
        document["environment"].update(
            {
                "SPARKRING_EXPERIMENT_ID": experiment_id,
                "SPARKRING_EXL3_CONTEXT_PROFILE": (
                    f"experimental-{experiment_id}"
                ),
                "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS": experiment[
                    "max_model_len"
                ],
                "VLLM_SPARK_KV_CACHE_MEMORY_BYTES": experiment[
                    "kv_cache_memory_bytes"
                ],
                "VLLM_SPARK_MAX_MODEL_LEN": experiment["max_model_len"],
            }
        )
    urls = [
        f"tcp://{rank.management.address}:{config['server_port']}"
        for rank in site.ranks
    ]
    kv_config = {
        "kv_connector": config["connector"],
        "kv_connector_module_path": config["connector_module"],
        "kv_role": "kv_both",
        "kv_load_failure_policy": config["load_failure_policy"],
        "kv_connector_extra_config": {
            "lmcache.mp.server_urls": urls,
            "lmcache.mp.mq_timeout": config["mq_timeout_seconds"],
            "lmcache.mp.heartbeat_interval": config[
                "heartbeat_interval_seconds"
            ],
        },
    }
    document["extra_vllm_args"].extend(
        ["--kv-transfer-config", json.dumps(kv_config, separators=(",", ":"))]
    )
    actions = exl3.start_actions(site, exl3.Profile(document))
    privileged = []
    for action in actions:
        command = action.argv[-1]
        marker = f"{profile.engine} run --detach "
        if marker not in command:
            raise exl3.ProfileError("cannot locate engine privilege insertion point")
        command = command.replace(
            marker,
            f"{marker}--privileged --label org.sparkring.component=engine ",
            1,
        )
        if experiment_id is not None:
            command = command.replace(
                f"org.sparkring.exl3-profile={PROFILE_ID}",
                f"org.sparkring.exl3-profile={effective_label}",
                1,
            )
            command = command.replace(
                marker,
                f"{marker}--label org.sparkring.experiment={experiment_id} ",
                1,
            )
        privileged.append(
            exl3.RemoteAction(
                action.rank, action.ssh_target, ("sh", "-lc", command)
            )
        )
    return privileged


def ready_actions(
    site,
    profile: exl3.Profile,
    *,
    experiment_id: str | None = None,
) -> list[exl3.RemoteAction]:
    effective_label = experiment_label(experiment_id)
    actions = []
    for rank in site.ranks:
        name = exl3.container_name(profile, rank.id)
        readiness = (
            "curl -fsS http://127.0.0.1:8000/health >/dev/null"
            " && curl -fsS http://127.0.0.1:8000/v1/models >/dev/null"
            if rank.id == 0
            else "true"
        )
        script = (
            f"test \"$(docker inspect -f '{{{{index .Config.Labels "
            f"\"org.sparkring.exl3-profile\"}}}}' {shlex.quote(name)})\" "
            f"= {shlex.quote(effective_label)}"
            f" && for i in $(seq 1 720); do "
            f"test \"$(docker inspect -f '{{{{.State.Running}}}}' "
            f"{shlex.quote(name)} 2>/dev/null)\" = true"
            f" && {readiness} && break; sleep 5; done"
            f" && test \"$(docker inspect -f '{{{{.State.Running}}}}' "
            f"{shlex.quote(name)})\" = true"
            f" && test \"$(docker inspect -f '{{{{.RestartCount}}}}' "
            f"{shlex.quote(name)})\" = 0"
            f" && test \"$(docker inspect -f '{{{{.State.OOMKilled}}}}' "
            f"{shlex.quote(name)})\" = false"
            f" && ! docker logs {shlex.quote(name)} 2>&1 | "
            "grep -E 'CUDA out of memory|OutOfMemoryError|OOMKilled'"
            f" && {readiness}"
        )
        actions.append(shell_action(rank, script))
    return actions


def rollback_actions(
    site,
    profile: exl3.Profile,
    *,
    experiment_id: str | None = None,
) -> list[exl3.RemoteAction]:
    effective_label = experiment_label(experiment_id)
    actions = []
    for rank in site.ranks:
        names = (exl3.container_name(profile, rank.id), server_name(rank.id))
        quoted_names = " ".join(shlex.quote(name) for name in names)
        script = (
            f"for name in {quoted_names}; do "
            'profile=$(docker inspect -f \'{{index .Config.Labels '
            '"org.sparkring.exl3-profile"}}\' "$name" '
            "2>/dev/null || true); "
            f"test -z \"$profile\" || test \"$profile\" = "
            f"{shlex.quote(effective_label)} || exit 73; "
            "test -z \"$profile\" || docker rm --force \"$name\"; done"
        )
        actions.append(shell_action(rank, script))
    return actions


def remove_component_actions(
    site,
    profile: exl3.Profile,
    *,
    component: str,
    experiment_id: str | None = None,
) -> list[exl3.RemoteAction]:
    if component not in ("engine", "lmcache-server"):
        raise exl3.ProfileError(f"unsupported removal component {component!r}")
    actions = []
    effective_label = experiment_label(experiment_id)
    for rank in site.ranks:
        name = (
            exl3.container_name(profile, rank.id)
            if component == "engine"
            else server_name(rank.id)
        )
        script = (
            f"name={shlex.quote(name)}; "
            'profile=$(docker inspect -f \'{{index .Config.Labels '
            '"org.sparkring.exl3-profile"}}\' "$name" '
            "2>/dev/null || true); "
            f'test -z "$profile" || test "$profile" = {shlex.quote(effective_label)} '
            "|| exit 73; "
            'component=$(docker inspect -f \'{{index .Config.Labels '
            '"org.sparkring.component"}}\' "$name" '
            "2>/dev/null || true); "
            f'test -z "$profile" || test -z "$component" || '
            f'test "$component" = {shlex.quote(component)} || exit 74; '
            'test -z "$profile" || docker rm --force "$name"'
        )
        actions.append(shell_action(rank, script))
    return actions


def rollback_verify_actions(site, profile: exl3.Profile) -> list[exl3.RemoteAction]:
    actions = []
    for rank in site.ranks:
        names = (exl3.container_name(profile, rank.id), server_name(rank.id))
        filters = " ".join(
            f"--filter name=^/{shlex.quote(name)}$" for name in names
        )
        script = f'test -z "$(docker ps -aq {filters})"'
        actions.append(shell_action(rank, script))
    return actions


def render(actions) -> list[dict]:
    return [
        {
            "rank": action.rank,
            "ssh_target": action.ssh_target,
            "remote_command": action.shell_command,
        }
        for action in actions
    ]


def run_phase(actions, *, execute: bool, timeout: int) -> dict:
    if not execute:
        return {"dry_run": True, "actions": render(actions)}
    return exl3.execute(actions, timeout=timeout)


def failed(result: dict) -> bool:
    return any(
        isinstance(item, dict) and item.get("exit_code", 0) != 0
        for item in result.values()
    )


def build_phases(
    site,
    profile: exl3.Profile,
    *,
    experiment_id: str | None = None,
) -> dict[str, list[exl3.RemoteAction]]:
    """Build all 8 canonical phase action sets.

    Exported so the bundle bridge and canonical main() share the same
    phase construction logic — no duplication.
    """
    return {
        "start_servers": server_start_actions(
            site, profile, experiment_id=experiment_id,
        ),
        "server_health": server_health_actions(
            site, experiment_id=experiment_id,
        ),
        "start_engines": engine_start_actions(
            site, profile, experiment_id=experiment_id,
        ),
        "ready": ready_actions(
            site, profile, experiment_id=experiment_id,
        ),
        "remove_engines": remove_component_actions(
            site, profile, component="engine", experiment_id=experiment_id,
        ),
        "remove_servers": remove_component_actions(
            site, profile, component="lmcache-server", experiment_id=experiment_id,
        ),
        "rollback": rollback_actions(
            site, profile, experiment_id=experiment_id,
        ),
        "verify_rollback": rollback_verify_actions(site, profile),
    }


def lifecycle_sequence(
    command: str, profile: exl3.Profile,
) -> list[dict[str, Any]]:
    """Return the canonical lifecycle phase sequence for a command.

    Exported so the bundle bridge and canonical main() share the same
    sequence oracle.  Includes explicit ``on_failure`` and ``timeout``
    for each phase.
    """
    if command == "restart-engines":
        raw = (
            ("server_health", 150),
            ("remove_engines", 180),
            ("start_engines", profile.startup_timeout_seconds + 60),
            ("ready", profile.startup_timeout_seconds + 60),
            ("server_health", 150),
        )
    elif command == "restart-stack":
        raw = (
            ("remove_engines", 180),
            ("remove_servers", 180),
            ("start_servers", 180),
            ("server_health", 150),
            ("start_engines", profile.startup_timeout_seconds + 60),
            ("ready", profile.startup_timeout_seconds + 60),
            ("server_health", 150),
        )
    elif command == "rollback":
        raw = (("rollback", 180),)
    elif command == "verify-rollback":
        raw = (("verify_rollback", 30),)
    elif command == "status":
        raw = (
            ("server_health", 150),
            ("ready", 30),
        )
    else:  # start
        raw = (
            ("start_servers", 180),
            ("server_health", 150),
            ("start_engines", profile.startup_timeout_seconds + 60),
            ("ready", profile.startup_timeout_seconds + 60),
        )
    if command in ("start", "restart-engines", "restart-stack"):
        on_failure = "rollback"
    elif command == "status":
        on_failure = "continue"
    else:
        on_failure = "return-failure"
    return [
        {"phase": name, "timeout": timeout, "on_failure": on_failure}
        for name, timeout in raw
    ]

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument(
        "--experimental-memory-profile",
        choices=tuple(EXPERIMENTAL_MEMORY_PROFILES),
        default=None,
        help=(
            "run a named non-canonical memory experiment; the published "
            "recipe and generated profile remain unchanged"
        ),
    )
    parser.add_argument(
        "command",
        choices=(
            "plan",
            "start",
            "status",
            "restart-engines",
            "restart-stack",
            "rollback",
            "verify-rollback",
        ),
    )
    args = parser.parse_args(argv)
    try:
        site = load_site(args.site)
        profile = exl3.load_profile(Path(args.profile))
        if profile.profile_id != PROFILE_ID:
            raise exl3.ProfileError(f"LMCache launcher requires {PROFILE_ID}")
        phases = build_phases(
            site, profile, experiment_id=args.experimental_memory_profile,
        )
    except (OSError, KeyError, json.JSONDecodeError, SiteConfigError, exl3.ProfileError) as exc:
        parser.error(str(exc))

    plan = {
        "schema": "sparkring-public-exl3-lmcache-plan/v1",
        "profile_id": PROFILE_ID,
        "image_id": profile.image_id,
        "mutates_remote": args.command
        in ("start", "restart-engines", "restart-stack", "rollback"),
        "phases": {name: render(actions) for name, actions in phases.items()},
    }
    if args.experimental_memory_profile is not None:
        experiment = EXPERIMENTAL_MEMORY_PROFILES[
            args.experimental_memory_profile
        ]
        canonical_environment = profile.document["environment"]
        canonical_lmcache = recipe_lmcache()
        plan["experiment"] = {
            "evidence_scope": "experimental",
            "canonical_configuration": False,
            "experiment_id": args.experimental_memory_profile,
            "base_profile_attestation": {
                "profile_id": profile.profile_id,
                "image_id": profile.image_id,
            },
            "effective_profile_label": experiment_label(
                args.experimental_memory_profile
            ),
            "baseline": {
                "kv_cache_memory_bytes": int(
                    canonical_environment["VLLM_SPARK_KV_CACHE_MEMORY_BYTES"]
                ),
                "max_model_len": int(
                    canonical_environment["VLLM_SPARK_MAX_MODEL_LEN"]
                ),
                "lmcache_l1_size_gb": canonical_lmcache["l1_size_gb"],
            },
            "effective": {
                "kv_cache_memory_bytes": int(
                    experiment["kv_cache_memory_bytes"]
                ),
                "max_model_len": int(experiment["max_model_len"]),
                "lmcache_l1_size_gb": experiment["lmcache_l1_size_gb"],
            },
        }
    if args.command == "plan" or not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if args.command in (
        "start",
        "restart-engines",
        "restart-stack",
        "rollback",
    ) and args.confirmation != CONFIRMATION:
        parser.error(f"execute requires --confirmation {CONFIRMATION}")

    # Consume the same lifecycle oracle exported to the bundle bridge for
    # every executable command, including read-only status/verification and
    # rollback itself.
    seq = lifecycle_sequence(args.command, profile)
    results = {}
    any_failed = False
    for step in seq:
        name = step["phase"]
        timeout = step["timeout"]
        results[name] = run_phase(phases[name], execute=True, timeout=timeout)
        if failed(results[name]):
            any_failed = True
            if step["on_failure"] == "rollback":
                results["rollback"] = run_phase(
                    phases["rollback"], execute=True, timeout=180
                )
                print(json.dumps(results, indent=2))
                return 1
            if step["on_failure"] == "return-failure":
                print(json.dumps(results, indent=2))
                return 1
    print(json.dumps(results, indent=2))
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
