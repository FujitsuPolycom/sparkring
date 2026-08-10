#!/usr/bin/env python3
"""Fail-closed EXL3 determinism attribution launcher for four Sparks.

The launcher derives diagnostic profiles in memory from the validated public
EXL3+LMCache CS512 profile.  It never accepts a pre-edited diagnostic profile.
Cache-off arms simply omit the engine connector.  Until a live LMCache layout
receipt exists, every cache-attached target engine lifetime explicitly
recreates all four servers after the old engines stop.  Rollback also recreates
them before restoring canonical engines.  This process-lifetime namespace
boundary is intentional: model-name keying is not an isolation contract for
LMCache KV objects with different MTP staging layouts.

This transaction launcher is deliberately Docker-only. The underlying public
EXL3 profile accepts Podman, but not every imported LMCache lifecycle helper is
engine-neutral yet; a Podman profile is rejected before phases are composed or
any remote action can run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sparkring_exl3_launcher as exl3
import sparkring_exl3_lmcache_launcher as lmcache
import exl3_verified_start as verified_start
from exl3_attribution_cache_contract import (
    CONTAINER_ID_RE,
    DOCKER_STARTED_AT_RE,
    LIVE_ARM_RECEIPT_SCHEMA,
    build_live_arm_receipt,
    cache_salt_for_layout,
    expected_layout,
)
from sparkring_site import SiteConfigError, load_site


CONFIRMATION = "RUN-EXL3-ATTRIBUTION-ALL-FOUR"
LABEL_KEY = "org.sparkring.exl3-attribution"
COMPONENT_LABEL = "org.sparkring.component"
MANAGED_LABEL = "org.sparkring.managed"
ENGINE_BLOCK_ROWS_PER_DCP_RANK = 64
GLOBAL_APC_ALIGNMENT_TOKENS = 256
LMCACHE_CHUNK_TOKENS = 512
OUTER_VERIFIED_ENTRYPOINT = verified_start.OUTER_VERIFIED_ENTRYPOINT
REMOTE_OUTER_VERIFIED_ENTRYPOINT = (
    verified_start.REMOTE_OUTER_VERIFIED_ENTRYPOINT
)
CONTAINER_ENTRYPOINT = verified_start.CONTAINER_ENTRYPOINT


@dataclass(frozen=True)
class Arm:
    mtp_tokens: int
    apc: bool
    lmcache_enabled: bool


class PhaseExecutionError(RuntimeError):
    """An executor raised while a named phase was in flight."""

    def __init__(self, phase: str, original: Exception, results: dict[str, Any]):
        super().__init__(f"phase {phase} raised {type(original).__name__}: {original}")
        self.phase = phase
        self.original = original
        self.results = results


ARMS = {
    "a-mtp0-apc0-lmcache0": Arm(0, False, False),
    "b-mtp2-apc0-lmcache0": Arm(2, False, False),
    "c-mtp2-apc1-lmcache0": Arm(2, True, False),
    "d-mtp2-apc1-lmcache1": Arm(2, True, True),
    "e-mtp0-apc0-lmcache1": Arm(0, False, True),
    "f-mtp2-apc0-lmcache1": Arm(2, False, True),
}

CANONICAL_ARM = Arm(2, True, True)
LMCACHE_LAYOUT_SCHEMA = "sparkring-exl3-lmcache-layout/v1"


def require_supported_engine(profile: exl3.Profile) -> None:
    if profile.engine != "docker":
        raise exl3.ProfileError(
            "EXL3 attribution transactions currently require engine=docker; "
            "podman is rejected before any remote action"
        )


def require_matching_supported_engines(*profiles: exl3.Profile) -> None:
    engines = {profile.engine for profile in profiles}
    if len(engines) != 1:
        raise exl3.ProfileError(
            "EXL3 attribution transaction profiles must use the same container engine"
        )
    for profile in profiles:
        require_supported_engine(profile)


def cache_boundary_geometry() -> dict[str, Any]:
    recipe = json.loads(exl3.RECIPE_PATH.read_text(encoding="utf-8"))
    config = lmcache.recipe_lmcache()
    dcp_size = recipe.get("serving", {}).get("decode_context_parallel_size")
    lmcache_chunk = config.get("chunk_size")
    logical_alignment = (
        ENGINE_BLOCK_ROWS_PER_DCP_RANK * dcp_size
        if isinstance(dcp_size, int) and not isinstance(dcp_size, bool)
        else None
    )
    if (
        dcp_size != 4
        or logical_alignment != GLOBAL_APC_ALIGNMENT_TOKENS
        or lmcache_chunk != LMCACHE_CHUNK_TOKENS
        or lmcache_chunk % logical_alignment
    ):
        raise exl3.ProfileError(
            "published expected cache boundary geometry drifted: "
            f"physical={ENGINE_BLOCK_ROWS_PER_DCP_RANK}, dcp={dcp_size!r}, "
            f"logical={logical_alignment!r}, lmcache={lmcache_chunk!r}"
        )
    return {
        "expected_engine_block_rows_per_dcp_rank": (
            ENGINE_BLOCK_ROWS_PER_DCP_RANK
        ),
        "expected_dcp_size": dcp_size,
        "expected_global_apc_alignment_tokens": logical_alignment,
        "expected_lmcache_chunk_tokens_global": lmcache_chunk,
        "runtime_attestation_required": True,
        "recipe_predecessor_chunk_size_is_geometry_evidence": False,
    }


def _positive_profile_integer(profile: exl3.Profile, key: str) -> int:
    value = profile.environment.get(key)
    if (
        not isinstance(value, str)
        or not value.isdecimal()
        or int(value) <= 0
    ):
        raise exl3.ProfileError(
            f"profile environment {key} must be a positive decimal integer"
        )
    return int(value)


def lmcache_layout_contract(profile: exl3.Profile, arm: Arm) -> dict[str, Any]:
    """Return the cache-object layout fields this launcher can attest.

    LMCache does not expose a namespace/layout receipt that would make stale
    L1 objects safe across these diagnostic engines.  In particular MTP0 and
    MTP2 use different connector-side staging object sizes.  These fields are
    therefore compared before a server process may be retained.
    """
    geometry = cache_boundary_geometry()
    return {
        "schema": LMCACHE_LAYOUT_SCHEMA,
        "mtp_tokens": arm.mtp_tokens,
        "dcp_size": geometry["expected_dcp_size"],
        "kv_cache_memory_bytes_per_rank": _positive_profile_integer(
            profile, "VLLM_SPARK_KV_CACHE_MEMORY_BYTES"
        ),
        "max_model_len": _positive_profile_integer(
            profile, "VLLM_SPARK_MAX_MODEL_LEN"
        ),
        "lmcache_chunk_tokens_global": geometry[
            "expected_lmcache_chunk_tokens_global"
        ],
    }


def attribution_cache_salt(profile: exl3.Profile, arm_id: str) -> str:
    arm = ARMS[arm_id]
    layout = lmcache_layout_contract(profile, arm)
    published = expected_layout(arm_id)
    if layout != published:
        raise exl3.ProfileError(
            "EXL3 attribution cache-salt layout drifted; update the shared "
            "request namespace contract before issuing cache-writing probes"
        )
    return cache_salt_for_layout(layout)


def target_lmcache_reset_required(
    source_profile: exl3.Profile,
    source_arm: Arm,
    target_profile: exl3.Profile,
    target_arm: Arm,
) -> bool:
    """Require a cold L1 for every cache-attached target engine lifetime.

    The arguments remain explicit so callers cannot accidentally omit either
    side of a transition, but layout equality is deliberately *not* used as a
    retention authorization.  No live LMCache layout receipt exists yet.
    """
    del source_profile, source_arm, target_profile
    return target_arm.lmcache_enabled


def strict_remove_actions(
    site,
    profile: exl3.Profile,
    *,
    component: str,
    arm_id: str | None = None,
) -> list[exl3.RemoteAction]:
    """Remove only an exact attribution-owned container identity."""
    require_supported_engine(profile)
    if component not in ("engine", "lmcache-server"):
        raise exl3.ProfileError(f"unsupported strict removal component {component!r}")
    if component == "lmcache-server" and arm_id is not None:
        raise exl3.ProfileError("LMCache server removal cannot carry an arm label")
    actions = []
    for rank in site.ranks:
        name = (
            exl3.container_name(profile, rank.id)
            if component == "engine"
            else lmcache.server_name(rank.id)
        )
        engine = profile.engine
        arm_check = (
            f'arm=$({engine} inspect -f \'{{{{index .Config.Labels "{LABEL_KEY}"}}}}\' "$name"); '
            f'test "$arm" = {shlex.quote(arm_id)} || exit 76; '
            if arm_id is not None
            else ""
        )
        script = (
            f"name={shlex.quote(name)}; "
            f'if {engine} inspect "$name" >/dev/null 2>&1; then '
            f'observed_name=$({engine} inspect -f \'{{{{.Name}}}}\' "$name"); '
            f'managed=$({engine} inspect -f \'{{{{index .Config.Labels "{MANAGED_LABEL}"}}}}\' "$name"); '
            f'profile=$({engine} inspect -f \'{{{{index .Config.Labels "org.sparkring.exl3-profile"}}}}\' "$name"); '
            f'component=$({engine} inspect -f \'{{{{index .Config.Labels "{COMPONENT_LABEL}"}}}}\' "$name"); '
            'test "$observed_name" = "/$name" || exit 72; '
            'test "$managed" = true || exit 73; '
            f'test "$profile" = {shlex.quote(profile.profile_id)} || exit 74; '
            f'test "$component" = {shlex.quote(component)} || exit 75; '
            f"{arm_check}"
            f'{engine} rm --force "$name"; fi'
        )
        actions.append(
            exl3.RemoteAction(rank.id, rank.ssh_target, ("sh", "-lc", script))
        )
    return actions


def lmcache_reset_phases(
    site,
    canonical: exl3.Profile,
    *,
    prefix: str,
) -> dict[str, list[exl3.RemoteAction]]:
    """Build an exact-identity, process-lifetime LMCache namespace reset."""
    return {
        f"{prefix}_remove_servers": _tag_actions(
            strict_remove_actions(
                site, canonical, component="lmcache-server"
            ),
            f"{prefix}_remove_servers",
        ),
        f"{prefix}_start_servers": _tag_actions(
            lmcache.server_start_actions(site, canonical),
            f"{prefix}_start_servers",
        ),
        f"{prefix}_server_health": _tag_actions(
            lmcache.server_health_actions(site, engine=canonical.engine),
            f"{prefix}_server_health",
        ),
    }


def _require_option(args: list[str], option: str) -> int:
    matches = [index for index, value in enumerate(args) if value == option]
    if len(matches) != 1:
        raise exl3.ProfileError(
            f"canonical profile must contain exactly one {option}, got {len(matches)}"
        )
    index = matches[0]
    if index + 1 >= len(args) or args[index + 1].startswith("--"):
        raise exl3.ProfileError(f"canonical profile option {option} has no value")
    return index


def _remove_option(args: list[str], option: str) -> list[str]:
    result = list(args)
    index = _require_option(result, option)
    del result[index : index + 2]
    return result


def _require_flag(args: list[str], flag: str) -> None:
    if args.count(flag) != 1:
        raise exl3.ProfileError(
            f"canonical profile must contain exactly one {flag}, got {args.count(flag)}"
        )


def derive_profile(canonical: exl3.Profile, arm_id: str) -> exl3.Profile:
    require_supported_engine(canonical)
    cache_boundary_geometry()
    try:
        arm = ARMS[arm_id]
    except KeyError as exc:
        raise exl3.ProfileError(f"unknown attribution arm {arm_id!r}") from exc
    document = copy.deepcopy(canonical.document)
    document["profile_id"] = f"{canonical.profile_id}-diag-{arm_id}"
    document["container_name"] = f"{canonical.container_name}-diag-{arm_id}"
    document["jit_cache_host_path"] = (
        f"{canonical.jit_cache_host_path.rstrip('/')}-diag-{arm_id}"
    )
    # Hold the engine/container envelope constant across cache-on and cache-off
    # arms.  The canonical LMCache launcher applies these two values and
    # --privileged while composing the engine; cache-off arms must match them
    # so connector presence remains the only cache variable.
    document["environment"].update(
        {
            "LMCACHE_DISABLE_BANNER": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        }
    )
    args = list(document["extra_vllm_args"])
    _require_flag(args, "--enable-prefix-caching")
    if "--no-enable-prefix-caching" in args:
        raise exl3.ProfileError(
            "canonical profile must not contain --no-enable-prefix-caching"
        )
    if not arm.apc:
        args[args.index("--enable-prefix-caching")] = (
            "--no-enable-prefix-caching"
        )
    if arm.mtp_tokens == 0:
        args = _remove_option(args, "--speculative-config")
        document["environment"].update(
            {
                "VLLM_SPARK_MTP_MODE_ID": "disabled",
                "VLLM_SPARK_MTP_TOKENS": "0",
            }
        )
    else:
        if arm.mtp_tokens != 2:
            raise exl3.ProfileError("diagnostic launcher only supports MTP0 or MTP2")
        spec_index = _require_option(args, "--speculative-config")
        expected = {
            "method": "mtp",
            "num_speculative_tokens": 2,
            "moe_backend": "triton",
            "draft_sample_method": "greedy",
        }
        try:
            observed = json.loads(args[spec_index + 1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise exl3.ProfileError("canonical speculative config is invalid") from exc
        if observed != expected:
            raise exl3.ProfileError("canonical fixed-MTP2 config drifted")
    document["extra_vllm_args"] = args
    return exl3.Profile(document)


def _decorate_engine_actions(
    actions: list[exl3.RemoteAction], profile: exl3.Profile, arm_id: str
) -> list[exl3.RemoteAction]:
    decorated = []
    for action in actions:
        command = action.argv[-1]
        marker = f"{profile.engine} run --detach "
        if marker not in command:
            raise exl3.ProfileError("cannot locate diagnostic engine launch marker")
        labels = (
            f"--label {LABEL_KEY}={shlex.quote(arm_id)} "
            f"--label {COMPONENT_LABEL}=engine "
        )
        has_component = f"--label {COMPONENT_LABEL}=engine " in command
        if has_component:
            labels = f"--label {LABEL_KEY}={shlex.quote(arm_id)} "
        elif f"{profile.engine} run --detach --privileged " not in command:
            labels = "--privileged " + labels
        command = command.replace(marker, marker + labels, 1)
        decorated.append(
            exl3.RemoteAction(action.rank, action.ssh_target, ("sh", "-lc", command))
        )
    return decorated


def start_actions(
    site,
    profile: exl3.Profile,
    arm_id: str,
    *,
    reclaim_page_cache_after_verification: bool = False,
) -> list[exl3.RemoteAction]:
    require_supported_engine(profile)
    arm = ARMS[arm_id]
    actions = (
        lmcache.engine_start_actions(site, profile)
        if arm.lmcache_enabled
        else exl3.start_actions(site, profile)
    )
    decorated = _decorate_engine_actions(actions, profile, arm_id)
    if not reclaim_page_cache_after_verification:
        return decorated
    decorated = _with_outer_verified_entrypoint(decorated, profile)
    return _with_post_verification_page_cache_reclaim(decorated, profile)


def outer_verified_entrypoint_sha256() -> str:
    return verified_start.outer_verified_entrypoint_sha256()


def _with_outer_verified_entrypoint(
    actions: list[exl3.RemoteAction], profile: exl3.Profile
) -> list[exl3.RemoteAction]:
    return verified_start.with_outer_verified_entrypoint(actions, profile)


def _with_post_verification_page_cache_reclaim(
    actions: list[exl3.RemoteAction], profile: exl3.Profile
) -> list[exl3.RemoteAction]:
    return verified_start.with_post_verification_page_cache_reclaim(actions, profile)


def prepare_page_cache_reclaim_entrypoint_actions(site) -> list[exl3.RemoteAction]:
    return verified_start.prepare_verified_start_actions(site)


def expected_lmcache_connector_config(site) -> dict[str, Any]:
    """Return the complete canonical connector object used by the launcher."""
    config = lmcache.recipe_lmcache()
    return {
        "kv_connector": config["connector"],
        "kv_connector_module_path": config["connector_module"],
        "kv_role": "kv_both",
        "kv_load_failure_policy": config["load_failure_policy"],
        "kv_connector_extra_config": {
            "lmcache.mp.server_urls": [
                f"tcp://{rank.management.address}:{config['server_port']}"
                for rank in site.ranks
            ],
            "lmcache.mp.mq_timeout": config["mq_timeout_seconds"],
            "lmcache.mp.heartbeat_interval": config[
                "heartbeat_interval_seconds"
            ],
        },
    }


def _config_cmd_from_start_action(
    action: exl3.RemoteAction, profile: exl3.Profile
) -> list[str]:
    """Recover the exact image argv represented by a generated start action."""
    tokens = shlex.split(action.argv[-1])
    run_indexes = [
        index
        for index in range(len(tokens) - 1)
        if tokens[index : index + 2] == [profile.engine, "run"]
    ]
    if not run_indexes:
        raise exl3.ProfileError("generated start action lacks container run argv")
    start = run_indexes[-1]
    try:
        image_index = tokens.index(profile.image_id, start)
    except ValueError as exc:
        raise exl3.ProfileError(
            "generated start action lacks the pinned image ID"
        ) from exc
    command = tokens[image_index + 1 :]
    if not command or any(not isinstance(value, str) for value in command):
        raise exl3.ProfileError("generated Config.Cmd argv is invalid")
    return command


def expected_config_cmds(
    site, profile: exl3.Profile, arm_id: str
) -> dict[int, list[str]]:
    actions = start_actions(site, profile, arm_id)
    result = {
        action.rank: _config_cmd_from_start_action(action, profile)
        for action in actions
    }
    if set(result) != {0, 1, 2, 3}:
        raise exl3.ProfileError("expected Config.Cmd set must cover four ranks")
    return result


def config_cmd_matches(actual_json: str, expected: list[str]) -> bool:
    """Pure equivalent of the remote exact-argv attestor, for regressions."""
    try:
        actual = json.loads(actual_json)
    except (TypeError, json.JSONDecodeError):
        return False
    return (
        isinstance(actual, list)
        and all(isinstance(value, str) for value in actual)
        and actual == expected
    )


def _expected_env_from_start_action(
    action: exl3.RemoteAction, profile: exl3.Profile
) -> dict[str, str]:
    tokens = shlex.split(action.argv[-1])
    run_indexes = [
        index
        for index in range(len(tokens) - 1)
        if tokens[index : index + 2] == [profile.engine, "run"]
    ]
    if not run_indexes:
        raise exl3.ProfileError("generated start action lacks container run argv")
    start = run_indexes[-1]
    try:
        image_index = tokens.index(profile.image_id, start)
    except ValueError as exc:
        raise exl3.ProfileError("generated start action lacks pinned image ID") from exc
    environment: dict[str, str] = {}
    index = start + 2
    while index < image_index:
        if tokens[index] == "--env":
            if index + 1 >= image_index or "=" not in tokens[index + 1]:
                raise exl3.ProfileError("generated start action has malformed --env")
            key, value = tokens[index + 1].split("=", 1)
            if not key or key in environment:
                raise exl3.ProfileError("generated start action has duplicate --env")
            environment[key] = value
            index += 2
            continue
        index += 1
    if not environment:
        raise exl3.ProfileError("generated start action has no explicit environment")
    return environment


def live_arm_receipt_rank_digests(
    site, profile: exl3.Profile, arm_id: str
) -> tuple[list[str], list[str]]:
    generated = {
        action.rank: action for action in start_actions(site, profile, arm_id)
    }
    expected_commands = expected_config_cmds(site, profile, arm_id)
    environment_digests: dict[int, str] = {}
    command_digests: dict[int, str] = {}
    for rank in site.ranks:
        environment = _expected_env_from_start_action(
            generated[rank.id], profile
        )
        environment_digests[rank.id] = (
            hashlib.sha256(
                json.dumps(
                    environment,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("ascii")
            ).hexdigest()
        )
        command_digests[rank.id] = (
            hashlib.sha256(
                json.dumps(
                    expected_commands[rank.id],
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("ascii")
            ).hexdigest()
        )
    if set(environment_digests) != set(range(4)) or set(
        command_digests
    ) != set(range(4)):
        raise exl3.ProfileError(
            "live-arm receipt requires exact ranks 0,1,2,3"
        )
    return (
        [environment_digests[rank] for rank in range(4)],
        [command_digests[rank] for rank in range(4)],
    )


def live_arm_attestation_actions(
    site,
    profile: exl3.Profile,
    arm_id: str,
    *,
    expected_runtime_instances: list[dict[str, Any]] | None = None,
) -> list[exl3.RemoteAction]:
    """Attest exact live identity and explicit environment on every rank.

    When ``expected_runtime_instances`` is supplied, the checks also bind the
    currently running containers to the runtime-unique Docker IDs and StartedAt
    values recorded by a prior activation receipt.
    """
    if expected_runtime_instances is not None:
        if (
            not isinstance(expected_runtime_instances, list)
            or len(expected_runtime_instances) != 4
        ):
            raise exl3.ProfileError(
                "live-arm re-attestation requires four receipt rank identities"
            )
        for rank, item in enumerate(expected_runtime_instances):
            if (
                not isinstance(item, dict)
                or item.get("rank") != rank
                or not isinstance(item.get("container_id"), str)
                or not isinstance(item.get("started_at"), str)
            ):
                raise exl3.ProfileError(
                    "live-arm re-attestation receipt ranks are malformed or unordered"
                )
    generated = {
        action.rank: action for action in start_actions(site, profile, arm_id)
    }
    expected_commands = expected_config_cmds(site, profile, arm_id)
    verifier = (
        "import json,sys; a=json.loads(sys.argv[1]); e=json.loads(sys.argv[2]); "
        "ok=isinstance(a,list) and all(isinstance(v,str) and '=' in v for v in a); "
        "d={}; "
        "[(d.setdefault(v.split('=',1)[0],v.split('=',1)[1])) for v in a]; "
        "ok=ok and len(d)==len(a) and all(d.get(k)==v for k,v in e.items()); "
        "raise SystemExit(0 if ok else 1)"
    )
    command_verifier = (
        "import json,sys; a=json.loads(sys.argv[1]); e=json.loads(sys.argv[2]); "
        "raise SystemExit(0 if isinstance(a,list) and a==e else 1)"
    )
    actions = []
    for rank in site.ranks:
        name = exl3.container_name(profile, rank.id)
        engine = profile.engine
        expected_env = _expected_env_from_start_action(generated[rank.id], profile)
        expected_env_json = shlex.quote(
            json.dumps(expected_env, sort_keys=True, separators=(",", ":"))
        )
        expected_cmd_json = shlex.quote(
            json.dumps(expected_commands[rank.id], separators=(",", ":"))
        )
        result_renderer = (
            "import json,sys; "
            "print(json.dumps({'rank':int(sys.argv[1]),'status':'attested',"
            "'container_id':sys.argv[2],'started_at':sys.argv[3]},"
            "sort_keys=True,separators=(',',':')))"
        )
        runtime_identity_check = ""
        if expected_runtime_instances is not None:
            expected_runtime = expected_runtime_instances[rank.id]
            runtime_identity_check = (
                f'test "$container_id" = {shlex.quote(expected_runtime["container_id"])} || exit 100; '
                f'test "$started_at" = {shlex.quote(expected_runtime["started_at"])} || exit 101; '
            )
        script = (
            f"name={shlex.quote(name)}; "
            f'test "$({engine} inspect -f \'{{{{.Name}}}}\' "$name")" = "/$name" || exit 92; '
            f'test "$({engine} inspect -f \'{{{{index .Config.Labels "{MANAGED_LABEL}"}}}}\' "$name")" = true || exit 93; '
            f'test "$({engine} inspect -f \'{{{{index .Config.Labels "org.sparkring.exl3-profile"}}}}\' "$name")" = {shlex.quote(profile.profile_id)} || exit 94; '
            f'test "$({engine} inspect -f \'{{{{index .Config.Labels "{COMPONENT_LABEL}"}}}}\' "$name")" = engine || exit 95; '
            f'test "$({engine} inspect -f \'{{{{index .Config.Labels "{LABEL_KEY}"}}}}\' "$name")" = {shlex.quote(arm_id)} || exit 96; '
            f'test "$({engine} inspect -f \'{{{{.Image}}}}\' "$name")" = {shlex.quote(profile.image_id)} || exit 97; '
            f'test "$({engine} inspect -f \'{{{{.State.Running}}}}\' "$name")" = true || exit 102; '
            f'test "$({engine} inspect -f \'{{{{.State.OOMKilled}}}}\' "$name")" = false || exit 103; '
            f'test "$({engine} inspect -f \'{{{{.RestartCount}}}}\' "$name")" = 0 || exit 104; '
            f'env_json=$({engine} inspect -f \'{{{{json .Config.Env}}}}\' "$name"); '
            f"python3 -c {shlex.quote(verifier)} \"$env_json\" {expected_env_json} || exit 98; "
            f'cmd_json=$({engine} inspect -f \'{{{{json .Config.Cmd}}}}\' "$name"); '
            f"python3 -c {shlex.quote(command_verifier)} \"$cmd_json\" {expected_cmd_json} || exit 99; "
            f'container_id=$({engine} inspect -f \'{{{{.Id}}}}\' "$name"); '
            f'started_at=$({engine} inspect -f \'{{{{.State.StartedAt}}}}\' "$name"); '
            f"{runtime_identity_check}"
            f"python3 -c {shlex.quote(result_renderer)} {rank.id} \"$container_id\" \"$started_at\""
        )
        actions.append(
            exl3.RemoteAction(rank.id, rank.ssh_target, ("sh", "-lc", script))
        )
    return actions


def no_other_diagnostic_actions(site, profile: exl3.Profile) -> list[exl3.RemoteAction]:
    require_supported_engine(profile)
    actions = []
    for rank in site.ranks:
        script = (
            f'test -z "$({profile.engine} ps -q --filter label={LABEL_KEY})"'
        )
        actions.append(exl3.RemoteAction(rank.id, rank.ssh_target, ("sh", "-lc", script)))
    return actions


def exclusive_engine_actions(
    site,
    profile: exl3.Profile,
    *,
    arm_id: str | None,
) -> list[exl3.RemoteAction]:
    """Require the expected engine to be the only managed engine on each host.

    Attribution labels alone are not an exclusivity boundary: a canonical,
    experimental, or stale managed engine can bind the same GPUs and ports
    without carrying ``LABEL_KEY``.  Query the two ownership labels that every
    public launcher applies and then bind the sole result to name/profile/arm.
    """
    require_supported_engine(profile)
    actions = []
    for rank in site.ranks:
        name = exl3.container_name(profile, rank.id)
        expected_arm = "" if arm_id is None else arm_id
        engine = profile.engine
        script = (
            f"name={shlex.quote(name)}; "
            f'ids="$({engine} ps -q --filter label={MANAGED_LABEL}=true)"; '
            'engine_ids=""; for candidate in $ids; do '
            f'component=$({engine} inspect -f \'{{{{index .Config.Labels "{COMPONENT_LABEL}"}}}}\' "$candidate"); '
            'case "$component" in lmcache-server) ;; *) engine_ids="$engine_ids $candidate" ;; esac; '
            'done; set -- $engine_ids; test "$#" -eq 1 || exit 76; id="$1"; '
            f'managed=$({engine} inspect -f \'{{{{index .Config.Labels "{MANAGED_LABEL}"}}}}\' "$id"); '
            f'component=$({engine} inspect -f \'{{{{index .Config.Labels "{COMPONENT_LABEL}"}}}}\' "$id"); '
            f'observed_profile=$({engine} inspect -f \'{{{{index .Config.Labels "org.sparkring.exl3-profile"}}}}\' "$id"); '
            f'arm=$({engine} inspect -f \'{{{{index .Config.Labels "{LABEL_KEY}"}}}}\' "$id"); '
            f'observed_name=$({engine} inspect -f \'{{{{.Name}}}}\' "$id"); '
            'test "$managed" = true || exit 77; '
            'test "$component" = engine || exit 78; '
            f'test "$observed_profile" = {shlex.quote(profile.profile_id)} || exit 79; '
            f'test "$arm" = {shlex.quote(expected_arm)} || exit 80; '
            'test "$observed_name" = "/$name" || exit 81'
        )
        actions.append(exl3.RemoteAction(rank.id, rank.ssh_target, ("sh", "-lc", script)))
    return actions


def exclusive_diagnostic_actions(
    site, profile: exl3.Profile, arm_id: str
) -> list[exl3.RemoteAction]:
    """Compatibility wrapper for exact managed-engine exclusivity."""
    require_supported_engine(profile)
    return exclusive_engine_actions(site, profile, arm_id=arm_id)


def diagnostic_remove_actions(
    site, profile: exl3.Profile, arm_id: str
) -> list[exl3.RemoteAction]:
    return strict_remove_actions(
        site, profile, component="engine", arm_id=arm_id
    )


def ready_actions(
    site, profile: exl3.Profile, arm_id: str, *, wait: bool = True
) -> list[exl3.RemoteAction]:
    require_supported_engine(profile)
    arm = ARMS[arm_id]
    expected_connector = expected_lmcache_connector_config(site)
    expected_commands = expected_config_cmds(site, profile, arm_id)
    actions = []
    for rank in site.ranks:
        name = exl3.container_name(profile, rank.id)
        engine = profile.engine
        api = (
            f"curl -fsS http://127.0.0.1:{site.serving.api_port}/health >/dev/null"
            if rank.id == site.serving.master_rank
            else "true"
        )
        expected_command = expected_commands[rank.id]
        exact_command_verifier = (
            "import json,sys; "
            "a=json.loads(sys.argv[1]); e=json.loads(sys.argv[2]); "
            "ok=isinstance(a,list) and all(isinstance(v,str) for v in a) and a==e; "
            "raise SystemExit(0 if ok else 1)"
        )
        argv_json = f"$({engine} inspect -f '{{{{json .Config.Cmd}}}}' \"$name\")"
        command_attestation = (
            f"argv={argv_json}; python3 -c {shlex.quote(exact_command_verifier)} "
            f'"$argv" {shlex.quote(json.dumps(expected_command, separators=(",", ":")))}'
        )
        if arm.lmcache_enabled:
            verifier = (
                "import json,sys; a=json.loads(sys.argv[1]); e=json.loads(sys.argv[2]); "
                "p=[i for i,v in enumerate(a) if v=='--kv-transfer-config']; "
                "ok=len(p)==1 and p[0]+1<len(a); "
                "ok=ok and json.loads(a[p[0]+1])==e if ok else False; "
                "raise SystemExit(0 if ok else 1)"
            )
            connector_attestation = (
                f"argv={argv_json}; python3 -c {shlex.quote(verifier)} \"$argv\" "
                f"{shlex.quote(json.dumps(expected_connector, separators=(',', ':')))} "
                f"&& {engine} logs \"$name\" 2>&1 | grep -F -- "
                "'Creating v1 connector with name: LMCacheMPConnector' >/dev/null "
                f"&& {engine} logs \"$name\" 2>&1 | grep -F -- "
                "'lmcache.mp.heartbeat_interval = 10.0' >/dev/null "
                f"&& {engine} logs \"$name\" 2>&1 | grep -F -- "
                "'LMCache MP worker adapter created with instance_id=' >/dev/null"
            )
        else:
            verifier = (
                "import json,sys; a=json.loads(sys.argv[1]); "
                "raise SystemExit(0 if '--kv-transfer-config' not in a else 1)"
            )
            connector_attestation = (
                f"argv={argv_json}; python3 -c {shlex.quote(verifier)} \"$argv\" "
                f"&& ! {engine} logs \"$name\" 2>&1 | grep -F -- "
                "'Creating v1 connector with name:' >/dev/null "
                f"&& ! {engine} logs \"$name\" 2>&1 | grep -F -- "
                "'LMCache MP worker adapter created with instance_id=' >/dev/null"
            )
        if rank.id == site.serving.master_rank:
            effective_prefix = (
                "enable_prefix_caching=True"
                if arm.apc
                else "enable_prefix_caching=False"
            )
            effective_mtp = (
                "speculative_config=None"
                if arm.mtp_tokens == 0
                else "num_spec_tokens=2"
            )
            effective_attestation = (
                f"{engine} logs \"$name\" 2>&1 | grep -F -- "
                f"{shlex.quote(effective_prefix)} >/dev/null "
                f"&& {engine} logs \"$name\" 2>&1 | grep -F -- "
                f"{shlex.quote(effective_mtp)} >/dev/null"
            )
        else:
            effective_attestation = "true"
        ready_check = (
            f'test "$({engine} inspect -f \'{{{{.State.Running}}}}\' "$name" 2>/dev/null)" = true '
            f"&& {api}"
        )
        # API health can become observable before connector/effective-config
        # startup messages have been flushed.  Breaking the wait loop on API
        # health alone turns that harmless ordering window into a false
        # readiness failure and triggers destructive rollback.  Wait for the
        # complete runtime attestation, then repeat the individually coded
        # checks below so a genuine timeout retains its precise exit code.
        fully_attested_ready_check = (
            f"{ready_check} && ({command_attestation}) "
            f"&& ({connector_attestation}) && ({effective_attestation})"
        )
        # Leave three minutes inside the outer executor timeout for the final
        # classified checks and automatic-rollback handoff.  A fixed number
        # of sleeps did not account for the cost of repeated inspect/log/API
        # probes and could be killed locally (124) before exits 82--91 ran.
        poll_budget = max(5, int(profile.startup_timeout_seconds) - 180)
        wait_prefix = (
            f"deadline=$(( $(date +%s) + {poll_budget} )); "
            'while test "$(date +%s)" -lt "$deadline"; do '
            f"{fully_attested_ready_check} && break; "
            f'oom="$({engine} inspect -f \'{{{{.State.OOMKilled}}}}\' "$name" 2>/dev/null)"; '
            f'restarts="$({engine} inspect -f \'{{{{.RestartCount}}}}\' "$name" 2>/dev/null)"; '
            f'state="$({engine} inspect -f \'{{{{.State.Status}}}}\' "$name" 2>/dev/null)"; '
            'test "$oom" = true && break; '
            'test -n "$restarts" && test "$restarts" != 0 && break; '
            'case "$state" in ""|exited|dead|removing) break ;; esac; '
            "sleep 5; done; "
            if wait
            else ""
        )
        script = (
            f"name={shlex.quote(name)}; "
            f"{wait_prefix}"
            f'test "$({engine} inspect -f \'{{{{index .Config.Labels "org.sparkring.exl3-profile"}}}}\' "$name")" = {shlex.quote(profile.profile_id)} || exit 82; '
            f'test "$({engine} inspect -f \'{{{{index .Config.Labels "{LABEL_KEY}"}}}}\' "$name")" = {shlex.quote(arm_id)} || exit 83; '
            f'test "$({engine} inspect -f \'{{{{index .Config.Labels "{COMPONENT_LABEL}"}}}}\' "$name")" = engine || exit 84; '
            f'test "$({engine} inspect -f \'{{{{.State.OOMKilled}}}}\' "$name")" = false || exit 87; '
            f'test "$({engine} inspect -f \'{{{{.RestartCount}}}}\' "$name")" = 0 || exit 86; '
            f'test "$({engine} inspect -f \'{{{{.State.Running}}}}\' "$name")" = true || exit 85; '
            f"({command_attestation}) || exit 88; "
            f"({connector_attestation}) || exit 89; "
            f"({effective_attestation}) || exit 90; "
            f"({api}) || exit 91"
        )
        actions.append(exl3.RemoteAction(rank.id, rank.ssh_target, ("sh", "-lc", script)))
    return actions


def canonical_restore_actions(
    site,
    canonical: exl3.Profile,
    *,
    reclaim_page_cache_after_verification: bool = False,
) -> list[exl3.RemoteAction]:
    """Idempotently restore missing canonical engines after a partial cutover."""
    require_supported_engine(canonical)
    starts = lmcache.engine_start_actions(site, canonical)
    if reclaim_page_cache_after_verification:
        starts = _with_outer_verified_entrypoint(starts, canonical)
        starts = _with_post_verification_page_cache_reclaim(starts, canonical)
    actions = []
    for rank, start in zip(site.ranks, starts):
        name = exl3.container_name(canonical, rank.id)
        engine = canonical.engine
        start_command = start.argv[-1]
        expected_command = _config_cmd_from_start_action(start, canonical)
        exact_command_verifier = (
            "import json,sys; a=json.loads(sys.argv[1]); e=json.loads(sys.argv[2]); "
            "ok=isinstance(a,list) and all(isinstance(v,str) for v in a) and a==e; "
            "raise SystemExit(0 if ok else 1)"
        )
        expected_json = shlex.quote(json.dumps(expected_command, separators=(",", ":")))
        script = (
            f"name={shlex.quote(name)}; "
            f'if {engine} inspect "$name" >/dev/null 2>&1; then '
            f'test "$({engine} inspect -f \'{{{{index .Config.Labels "{MANAGED_LABEL}"}}}}\' "$name")" = true '
            f'&& test "$({engine} inspect -f \'{{{{index .Config.Labels "org.sparkring.exl3-profile"}}}}\' "$name")" = {shlex.quote(canonical.profile_id)} '
            f'&& test "$({engine} inspect -f \'{{{{index .Config.Labels "{COMPONENT_LABEL}"}}}}\' "$name")" = engine '
            f'&& if test "$({engine} inspect -f \'{{{{.State.Running}}}}\' "$name")" = true; then '
            f'argv=$({engine} inspect -f \'{{{{json .Config.Cmd}}}}\' "$name"); '
            f"python3 -c {shlex.quote(exact_command_verifier)} \"$argv\" {expected_json}; "
            f'else {engine} rm "$name" && {start_command}; fi; '
            f"else {start_command}; fi"
        )
        actions.append(exl3.RemoteAction(rank.id, rank.ssh_target, ("sh", "-lc", script)))
    return actions


def render(actions: list[exl3.RemoteAction]) -> list[dict[str, Any]]:
    return [
        {"rank": action.rank, "ssh_target": action.ssh_target, "remote_command": action.shell_command}
        for action in actions
    ]


def _tag_actions(
    actions: list[exl3.RemoteAction], phase_name: str
) -> list[exl3.RemoteAction]:
    """Give repeated transition checks distinct, evidence-visible identities."""
    return [
        exl3.RemoteAction(
            action.rank,
            action.ssh_target,
            ("sh", "-lc", f": {shlex.quote(phase_name)}; {action.argv[-1]}"),
        )
        for action in actions
    ]


def phases(
    site,
    canonical: exl3.Profile,
    diagnostic: exl3.Profile,
    arm_id: str,
    *,
    reclaim_page_cache_after_verification: bool = False,
):
    require_matching_supported_engines(canonical, diagnostic)
    result = {
        "server_health": lmcache.server_health_actions(
            site, engine=canonical.engine
        ),
        "canonical_engine_exclusive": exclusive_engine_actions(
            site, canonical, arm_id=None
        ),
        "no_other_diagnostic": no_other_diagnostic_actions(site, canonical),
        "remove_canonical_engines": strict_remove_actions(
            site, canonical, component="engine"
        ),
        "start_diagnostic": start_actions(
            site,
            diagnostic,
            arm_id,
            reclaim_page_cache_after_verification=(
                reclaim_page_cache_after_verification
            ),
        ),
        "diagnostic_ready": ready_actions(site, diagnostic, arm_id),
        "diagnostic_live_arm_attestation": _tag_actions(
            live_arm_attestation_actions(site, diagnostic, arm_id),
            "diagnostic_live_arm_attestation",
        ),
        "remove_diagnostic": diagnostic_remove_actions(site, diagnostic, arm_id),
        "rollback_remove_canonical_engines": _tag_actions(
            strict_remove_actions(
                site, canonical, component="engine"
            ),
            "rollback_remove_canonical_engines",
        ),
        "start_canonical": canonical_restore_actions(
            site,
            canonical,
            reclaim_page_cache_after_verification=(
                reclaim_page_cache_after_verification
            ),
        ),
        "canonical_ready": lmcache.ready_actions(site, canonical),
        "canonical_engine_exclusive_after_restore": _tag_actions(
            exclusive_engine_actions(site, canonical, arm_id=None),
            "canonical_engine_exclusive_after_restore",
        ),
    }
    result.update(
        lmcache_reset_phases(
            site, canonical, prefix="isolate_target_lmcache"
        )
    )
    result.update(
        lmcache_reset_phases(
            site, canonical, prefix="rollback_reset_lmcache"
        )
    )
    if reclaim_page_cache_after_verification:
        result["prepare_page_cache_reclaim_entrypoint"] = (
            prepare_page_cache_reclaim_entrypoint_actions(site)
        )
    return result


def transition_phases(
    site,
    canonical: exl3.Profile,
    source: exl3.Profile,
    source_arm_id: str,
    target: exl3.Profile,
    target_arm_id: str,
    *,
    reclaim_page_cache_after_verification: bool = False,
):
    require_matching_supported_engines(canonical, source, target)
    result = {
        "source_server_health": _tag_actions(
            lmcache.server_health_actions(site, engine=canonical.engine), "source_server_health"
        ),
        "source_diagnostic_ready": ready_actions(
            site, source, source_arm_id, wait=False
        ),
        "source_diagnostic_exclusive": exclusive_diagnostic_actions(
            site, source, source_arm_id
        ),
        "remove_source_diagnostic": diagnostic_remove_actions(
            site, source, source_arm_id
        ),
        "post_source_removal_server_health": _tag_actions(
            lmcache.server_health_actions(site, engine=canonical.engine),
            "post_source_removal_server_health",
        ),
        "start_target_diagnostic": start_actions(
            site,
            target,
            target_arm_id,
            reclaim_page_cache_after_verification=(
                reclaim_page_cache_after_verification
            ),
        ),
        "target_diagnostic_ready": ready_actions(site, target, target_arm_id),
        "target_server_health": _tag_actions(
            lmcache.server_health_actions(site, engine=canonical.engine), "target_server_health"
        ),
        "target_live_arm_attestation": _tag_actions(
            live_arm_attestation_actions(site, target, target_arm_id),
            "target_live_arm_attestation",
        ),
        "remove_target_diagnostic": diagnostic_remove_actions(
            site, target, target_arm_id
        ),
        "rollback_remove_canonical_engines": _tag_actions(
            strict_remove_actions(
                site, canonical, component="engine"
            ),
            "rollback_remove_canonical_engines",
        ),
        "rollback_server_health_before_restore": _tag_actions(
            lmcache.server_health_actions(site, engine=canonical.engine),
            "rollback_server_health_before_restore",
        ),
        "start_canonical": canonical_restore_actions(
            site,
            canonical,
            reclaim_page_cache_after_verification=(
                reclaim_page_cache_after_verification
            ),
        ),
        "canonical_ready": lmcache.ready_actions(site, canonical),
        "rollback_server_health_after_restore": _tag_actions(
            lmcache.server_health_actions(site, engine=canonical.engine),
            "rollback_server_health_after_restore",
        ),
        "canonical_engine_exclusive_after_restore": _tag_actions(
            exclusive_engine_actions(site, canonical, arm_id=None),
            "canonical_engine_exclusive_after_restore",
        ),
    }
    result.update(
        lmcache_reset_phases(
            site, canonical, prefix="isolate_target_lmcache"
        )
    )
    result.update(
        lmcache_reset_phases(
            site, canonical, prefix="rollback_reset_lmcache"
        )
    )
    if reclaim_page_cache_after_verification:
        result["prepare_page_cache_reclaim_entrypoint"] = (
            prepare_page_cache_reclaim_entrypoint_actions(site)
        )
    return result


def _append_lmcache_reset(selected: list[str], prefix: str) -> None:
    selected.extend(
        [
            f"{prefix}_remove_servers",
            f"{prefix}_start_servers",
            f"{prefix}_server_health",
        ]
    )


def sequence(
    command: str,
    *,
    reclaim_page_cache_after_verification: bool = False,
    reset_target_lmcache: bool = False,
) -> list[str]:
    if command in ("plan", "activate"):
        selected = [
            "server_health",
            "canonical_engine_exclusive",
            "no_other_diagnostic",
        ]
        if reclaim_page_cache_after_verification:
            selected.append("prepare_page_cache_reclaim_entrypoint")
        selected.append("remove_canonical_engines")
        if reset_target_lmcache:
            _append_lmcache_reset(selected, "isolate_target_lmcache")
        selected.append("start_diagnostic")
        selected.extend(
            ["diagnostic_ready", "diagnostic_live_arm_attestation"]
        )
        return selected
    if command == "rollback":
        selected = []
        if reclaim_page_cache_after_verification:
            selected.append("prepare_page_cache_reclaim_entrypoint")
        selected.extend(
            ["remove_diagnostic", "rollback_remove_canonical_engines"]
        )
        # Rollback never trusts diagnostic L1 contents. Recreating the server
        # processes also repairs a partially completed forward isolation.
        _append_lmcache_reset(selected, "rollback_reset_lmcache")
        selected.append("start_canonical")
        selected.extend(["canonical_ready", "server_health", "canonical_engine_exclusive_after_restore"])
        return selected
    return [
        "server_health",
        "diagnostic_ready",
        "diagnostic_live_arm_attestation",
    ]


def transition_sequence(
    *,
    reclaim_page_cache_after_verification: bool = False,
    reset_target_lmcache: bool = False,
) -> list[str]:
    selected = [
        "source_server_health",
        "source_diagnostic_ready",
        "source_diagnostic_exclusive",
    ]
    if reclaim_page_cache_after_verification:
        selected.append("prepare_page_cache_reclaim_entrypoint")
    selected.append("remove_source_diagnostic")
    if reset_target_lmcache:
        _append_lmcache_reset(selected, "isolate_target_lmcache")
    else:
        selected.append("post_source_removal_server_health")
    selected.append("start_target_diagnostic")
    selected.extend(
        [
            "target_diagnostic_ready",
            "target_server_health",
            "target_live_arm_attestation",
        ]
    )
    return selected


def transition_rollback_sequence(*, reclaim_page_cache_after_verification: bool = False) -> list[str]:
    selected = []
    if reclaim_page_cache_after_verification:
        selected.append("prepare_page_cache_reclaim_entrypoint")
    selected.extend([
        "remove_source_diagnostic",
        "remove_target_diagnostic",
        "rollback_remove_canonical_engines",
    ])
    _append_lmcache_reset(selected, "rollback_reset_lmcache")
    selected.append("start_canonical")
    selected.extend([
        "canonical_ready",
        "rollback_server_health_after_restore",
        "canonical_engine_exclusive_after_restore",
    ])
    return selected


def failed(result: Any) -> bool:
    if not isinstance(result, dict) or not result:
        raise ValueError("executor returned a non-object or empty result")
    if set(result) != {0, 1, 2, 3}:
        raise ValueError("executor result must contain exactly ranks 0, 1, 2, and 3")
    for rank, item in result.items():
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError("executor result rank key must be an integer")
        if not isinstance(item, dict):
            raise ValueError("executor rank result must be an object")
        exit_code = item.get("exit_code")
        if not isinstance(exit_code, int) or isinstance(exit_code, bool):
            raise ValueError("executor rank result lacks an integer exit_code")
    return any(item["exit_code"] != 0 for item in result.values())


def execute_sequence(selected: list[str], all_phases, timeout: int) -> tuple[dict, bool]:
    results = {}
    for name in selected:
        try:
            result = exl3.execute(all_phases[name], timeout=timeout)
        except Exception as exc:
            raise PhaseExecutionError(name, exc, results) from exc
        try:
            phase_failed = failed(result)
        except Exception as exc:
            raise PhaseExecutionError(name, exc, results) from exc
        results[name] = result
        if phase_failed:
            return results, False
        if name in {
            "diagnostic_live_arm_attestation",
            "target_live_arm_attestation",
        }:
            try:
                validate_live_arm_phase_result(result)
            except Exception as exc:
                raise PhaseExecutionError(name, exc, results) from exc
    return results, True


def validate_live_arm_phase_result(
    result: dict[int, dict[str, Any]],
) -> list[dict[str, str]]:
    runtime_instances: list[dict[str, str]] = []
    for rank in range(4):
        item = result[rank]
        stdout = item.get("stdout")
        stderr = item.get("stderr")
        try:
            value = json.loads(stdout)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("live-arm attestation stdout is not exact JSON") from exc
        if not isinstance(value, dict) or set(value) != {
            "rank",
            "status",
            "container_id",
            "started_at",
        }:
            raise ValueError("live-arm attestation output fields are invalid")
        if value.get("rank") != rank or value.get("status") != "attested":
            raise ValueError("live-arm attestation rank/status output is invalid")
        container_id = value.get("container_id")
        started_at = value.get("started_at")
        if (
            not isinstance(container_id, str)
            or CONTAINER_ID_RE.fullmatch(container_id) is None
        ):
            raise ValueError("live-arm attestation container ID is invalid")
        if (
            not isinstance(started_at, str)
            or DOCKER_STARTED_AT_RE.fullmatch(started_at) is None
        ):
            raise ValueError("live-arm attestation StartedAt is invalid")
        if stderr not in ("", None):
            raise ValueError("live-arm attestation produced stderr")
        runtime_instances.append(
            {"container_id": container_id, "started_at": started_at}
        )
    return runtime_instances


def live_arm_revalidation_actions(
    site,
    canonical: exl3.Profile,
    arm_id: str,
    live_arm_receipt: dict[str, Any],
) -> list[exl3.RemoteAction]:
    """Build read-only checks binding the current engines to a live receipt."""
    require_supported_engine(canonical)
    diagnostic = derive_profile(canonical, arm_id)
    ranks = live_arm_receipt.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != 4:
        raise exl3.ProfileError(
            "live-arm re-attestation receipt must contain four rank identities"
        )
    environment_digests, command_digests = live_arm_receipt_rank_digests(
        site, diagnostic, arm_id
    )
    expected_labels = {
        MANAGED_LABEL: "true",
        "org.sparkring.exl3-profile": diagnostic.profile_id,
        COMPONENT_LABEL: "engine",
        LABEL_KEY: arm_id,
    }
    for rank, item in enumerate(ranks):
        if (
            not isinstance(item, dict)
            or item.get("rank") != rank
            or item.get("container_name") != exl3.container_name(diagnostic, rank)
            or item.get("labels") != expected_labels
            or item.get("image_id") != diagnostic.image_id
            or item.get("explicit_environment_sha256")
            != environment_digests[rank]
            or item.get("config_cmd_sha256") != command_digests[rank]
        ):
            raise exl3.ProfileError(
                f"live-arm receipt static identity does not match rank {rank} plan"
            )
    return live_arm_attestation_actions(
        site,
        diagnostic,
        arm_id,
        expected_runtime_instances=ranks,
    )


def revalidate_live_arm(
    site,
    canonical: exl3.Profile,
    arm_id: str,
    live_arm_receipt: dict[str, Any],
    *,
    timeout: int,
) -> dict[str, Any]:
    """READ-ONLY REMOTE re-attestation required immediately before HTTP."""
    actions = live_arm_revalidation_actions(
        site, canonical, arm_id, live_arm_receipt
    )
    try:
        result = exl3.execute(actions, timeout=timeout)
        if failed(result):
            raise exl3.ProfileError(
                "live-arm re-attestation failed on one or more ranks"
            )
        instances = validate_live_arm_phase_result(result)
    except exl3.ProfileError:
        raise
    except Exception as exc:
        raise exl3.ProfileError(
            f"live-arm re-attestation could not complete: {type(exc).__name__}"
        ) from exc
    return {
        "status": "live-arm-re-attested",
        "rank_count": 4,
        "runtime_instances": instances,
    }


def exception_evidence(exc: PhaseExecutionError) -> dict[str, str]:
    """Render the original exception without a traceback or local paths."""
    return {
        "phase": exc.phase,
        "type": type(exc.original).__name__,
        "message": str(exc.original),
    }


def local_exception_evidence(phase: str, exc: Exception) -> dict[str, str]:
    return {"phase": phase, "type": type(exc).__name__, "message": str(exc)}


def removal_attempted(command: str, results: dict[str, Any], phase: str | None) -> bool:
    removal_phase = (
        "remove_canonical_engines" if command == "activate" else "remove_source_diagnostic"
    )
    return removal_phase in results or phase == removal_phase


ROLLBACK_ENGINE_CLEANUP_PHASES = {
    "remove_diagnostic",
    "remove_source_diagnostic",
    "remove_target_diagnostic",
    "rollback_remove_canonical_engines",
}
ROLLBACK_DEPENDENCY_GATES = ROLLBACK_ENGINE_CLEANUP_PHASES | {
    "prepare_page_cache_reclaim_entrypoint",
    "rollback_reset_lmcache_remove_servers",
    "rollback_reset_lmcache_start_servers",
    "rollback_reset_lmcache_server_health",
}


def attempt_automatic_rollback(selected, all_phases, timeout: int) -> tuple[dict, bool]:
    """Rollback without resetting cache servers beneath a surviving engine."""
    evidence: dict[str, Any] = {}
    restored = True
    exceptions = []
    for name in selected:
        try:
            result = exl3.execute(all_phases[name], timeout=timeout)
            try:
                phase_failed = failed(result)
            except Exception as exc:
                exceptions.append(local_exception_evidence(name, exc))
                evidence[name] = {"malformed_executor_result": True}
                restored = False
                if name in ROLLBACK_DEPENDENCY_GATES:
                    break
                continue
            evidence[name] = result
            if phase_failed:
                restored = False
                if name in ROLLBACK_DEPENDENCY_GATES:
                    break
        except Exception as exc:
            exceptions.append(local_exception_evidence(name, exc))
            evidence[name] = {"executor_exception": True}
            restored = False
            if name in ROLLBACK_DEPENDENCY_GATES:
                break
    if exceptions:
        evidence["rollback_exceptions"] = exceptions
    return evidence, restored


def functional_delta(arm_id: str) -> dict[str, Any]:
    arm = ARMS[arm_id]
    return {
        "mtp_tokens": arm.mtp_tokens,
        "native_prefix_cache": arm.apc,
        "lmcache_connector": arm.lmcache_enabled,
        "cache_boundary_geometry": cache_boundary_geometry(),
    }


def reserve_output(path: Path):
    """Exclusively reserve an evidence path before any remote mutation."""
    if path.exists():
        raise exl3.ProfileError(f"--output already exists: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise exl3.ProfileError(f"cannot create --output {path}: {exc}") from exc
    try:
        return path.open("x", encoding="utf-8", newline="\n")
    except FileExistsError as exc:
        raise exl3.ProfileError(f"--output already exists: {path}") from exc
    except OSError as exc:
        raise exl3.ProfileError(f"cannot create --output {path}: {exc}") from exc


def emit(document: dict[str, Any], stream=None) -> None:
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if stream is not None:
        try:
            stream.write(rendered)
            stream.flush()
        except OSError as exc:
            raise exl3.ProfileError(f"cannot write --output report: {exc}") from exc
    sys.stdout.write(rendered)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    parser.add_argument("--from-arm", choices=tuple(ARMS))
    parser.add_argument(
        "--reclaim-page-cache-after-verification",
        action="store_true",
        help=(
            "after the full model verifier succeeds, run sync/drop_caches "
            "on each rank immediately before starting its engine; this is "
            "MUTATES HOST and requires passwordless sudo"
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument(
        "--output",
        help=(
            "write stdout-identical UTF-8 JSON using exclusive creation; "
            "the path is reserved before any remote operation"
        ),
    )
    parser.add_argument(
        "--live-arm-receipt-output",
        help=(
            "exclusive-create public-safe live-arm receipt after exact four-rank "
            "attestation; required for executed activate/restart/transition/status"
        ),
    )
    parser.add_argument(
        "command",
        choices=("plan", "activate", "status", "restart-arm", "rollback", "transition"),
    )
    args = parser.parse_args(argv)
    if args.command == "transition":
        if args.from_arm is None:
            parser.error("transition requires --from-arm")
        if args.from_arm == args.arm:
            parser.error("transition requires distinct --from-arm and --arm")
    elif args.from_arm is not None:
        parser.error("--from-arm is valid only with transition")
    if args.reclaim_page_cache_after_verification and args.command == "status":
        parser.error(
            "--reclaim-page-cache-after-verification is not valid with status"
        )
    try:
        site = load_site(args.site)
        profile_path = Path(args.profile)
        canonical = exl3.load_profile(profile_path)
        require_supported_engine(canonical)
        if canonical.profile_id != lmcache.PROFILE_ID:
            raise exl3.ProfileError(
                f"attribution requires canonical profile {lmcache.PROFILE_ID}"
            )
        diagnostic = derive_profile(canonical, args.arm)
        source_arm_id = (
            args.from_arm if args.command == "transition" else args.arm
        )
        source = (
            derive_profile(canonical, source_arm_id)
            if args.command in ("transition", "restart-arm")
            else None
        )
        if source is not None:
            reset_target_lmcache = target_lmcache_reset_required(
                source,
                ARMS[source_arm_id],
                diagnostic,
                ARMS[args.arm],
            )
            all_phases = transition_phases(
                site,
                canonical,
                source,
                source_arm_id,
                diagnostic,
                args.arm,
                reclaim_page_cache_after_verification=(
                    args.reclaim_page_cache_after_verification
                ),
            )
            selected = transition_sequence(
                reclaim_page_cache_after_verification=(
                    args.reclaim_page_cache_after_verification
                ),
                reset_target_lmcache=reset_target_lmcache,
            )
            rollback_selected = transition_rollback_sequence(
                reclaim_page_cache_after_verification=(
                    args.reclaim_page_cache_after_verification
                )
            )
        else:
            reset_target_lmcache = target_lmcache_reset_required(
                canonical,
                CANONICAL_ARM,
                diagnostic,
                ARMS[args.arm],
            )
            all_phases = phases(
                site,
                canonical,
                diagnostic,
                args.arm,
                reclaim_page_cache_after_verification=(
                    args.reclaim_page_cache_after_verification
                ),
            )
            selected = sequence(
                args.command,
                reclaim_page_cache_after_verification=(
                    args.reclaim_page_cache_after_verification
                ),
                reset_target_lmcache=reset_target_lmcache,
            )
            rollback_selected = sequence(
                "rollback",
                reclaim_page_cache_after_verification=(
                    args.reclaim_page_cache_after_verification
                ),
            )
        base_sha256 = hashlib.sha256(profile_path.read_bytes()).hexdigest()
        environment_digests, command_digests = live_arm_receipt_rank_digests(
            site, diagnostic, args.arm
        )
        live_receipt_builder_kwargs = {
            "arm_id": args.arm,
            "canonical_profile_id": canonical.profile_id,
            "canonical_profile_file_sha256": base_sha256,
            "image_id": canonical.image_id,
            "model_repository": canonical.model_repository,
            "model_revision": canonical.model_revision,
            "canonical_container_name": canonical.container_name,
            "explicit_environment_sha256": environment_digests,
            "config_cmd_sha256": command_digests,
        }
        receipt_rank_static_contracts = [
            {
                "rank": rank,
                "container_name": exl3.container_name(diagnostic, rank),
                "labels": {
                    MANAGED_LABEL: "true",
                    "org.sparkring.exl3-profile": diagnostic.profile_id,
                    COMPONENT_LABEL: "engine",
                    LABEL_KEY: args.arm,
                },
                "image_id": canonical.image_id,
                "explicit_environment_sha256": environment_digests[rank],
                "config_cmd_sha256": command_digests[rank],
            }
            for rank in range(4)
        ]
    except (
        OSError,
        KeyError,
        json.JSONDecodeError,
        SiteConfigError,
        exl3.ProfileError,
    ) as exc:
        parser.error(str(exc))
    plan = {
        "schema": "sparkring-exl3-attribution-plan/v1",
        "command": args.command,
        "mutates_remote": args.command
        in ("activate", "restart-arm", "rollback", "transition"),
        "startup_memory_hygiene": {
            "post_verification_host_page_cache_reclaim": (
                args.reclaim_page_cache_after_verification
            ),
            "requires_passwordless_sudo": (
                args.reclaim_page_cache_after_verification
            ),
            "safety_class": (
                "MUTATES HOST"
                if args.reclaim_page_cache_after_verification
                else "no-additional-mutation"
            ),
            "boundaries": (
                [
                    "after the sole full model verification and before docker run/vLLM",
                ]
                if args.reclaim_page_cache_after_verification
                else []
            ),
            "preflight_before_engine_removal": (
                args.reclaim_page_cache_after_verification
            ),
            "inner_model_verification": (
                "skipped-by-sha256-attested-entrypoint-after-outer-pass"
                if args.reclaim_page_cache_after_verification
                else "unchanged-image-entrypoint"
            ),
            "outer_verified_entrypoint_sha256": (
                outer_verified_entrypoint_sha256()
                if args.reclaim_page_cache_after_verification
                else None
            ),
        },
        "arm": args.arm,
        "from_arm": source_arm_id if source is not None else None,
        "functional_settings": functional_delta(args.arm),
        "canonical_attestation": {
            "profile_id": canonical.profile_id,
            "profile_file_sha256": base_sha256,
            "image_id": canonical.image_id,
            "model_revision": canonical.model_revision,
        },
        "diagnostic_identity": {
            "profile_label": diagnostic.profile_id,
            "container_prefix": diagnostic.container_name,
            "jit_cache_host_path": diagnostic.jit_cache_host_path,
            "attribution_label": args.arm,
        },
        "source_diagnostic_identity": (
            {
                "profile_label": source.profile_id,
                "container_prefix": source.container_name,
                "jit_cache_host_path": source.jit_cache_host_path,
                "attribution_label": source_arm_id,
                "functional_settings": functional_delta(source_arm_id),
            }
            if source is not None
            else None
        ),
        "server_policy": (
            "cache-off target arms omit the engine connector; every cache-attached activation, "
            "transition, and restart-arm plus every rollback recreates all four LMCache servers "
            "to establish a fresh process-lifetime L1 namespace until a live layout receipt exists"
        ),
        "lmcache_l1_isolation": {
            "contract": "process-lifetime-namespace/v1",
            "model_name_keying_assumed_safe": False,
            "source_layout": lmcache_layout_contract(
                source if source is not None else canonical,
                ARMS[source_arm_id] if source is not None else CANONICAL_ARM,
            ),
            "target_layout": lmcache_layout_contract(
                diagnostic, ARMS[args.arm]
            ),
            "target_uses_lmcache": ARMS[args.arm].lmcache_enabled,
            "forward_reset_required": reset_target_lmcache,
            "rollback_reset_required": args.command
            in ("plan", "activate", "restart-arm", "rollback", "transition"),
            "reset_mechanism": "remove-and-recreate-all-four-lmcache-server-containers",
            "required_request_cache_salt": attribution_cache_salt(
                diagnostic, args.arm
            ),
            "request_cache_salt_source": (
                "scripts/exl3_attribution_cache_contract.py"
            ),
        },
        "live_arm_receipt_contract": {
            "schema": LIVE_ARM_RECEIPT_SCHEMA,
            "status": "live-arm-attested",
            "arm": args.arm,
            "canonical_profile_id": canonical.profile_id,
            "diagnostic_profile_id": diagnostic.profile_id,
            "canonical_profile_file_sha256": base_sha256,
            "image_id": canonical.image_id,
            "model_repository": canonical.model_repository,
            "model_revision": canonical.model_revision,
            "layout": lmcache_layout_contract(
                diagnostic, ARMS[args.arm]
            ),
            "cache_salt": attribution_cache_salt(diagnostic, args.arm),
            "rank_static_contracts": receipt_rank_static_contracts,
            "runtime_unique_fields_observed_after_readiness": [
                "container_id",
                "started_at",
            ],
            "required_for_request_probes": True,
        },
        "sequence": selected,
        "phases": {name: render(all_phases[name]) for name in selected},
        "rollback_phases": {
            name: render(all_phases[name]) for name in rollback_selected
        },
        "automatic_failure_action": (
            "remove exact source/target diagnostic engines and restore canonical engines"
            if args.command in ("activate", "restart-arm", "transition")
            else None
        ),
        "evidence_policy": {
            "raw_launcher_report_is_private": True,
            "publishable_reducer": "scripts/exl3_attribution_reduce.py",
            "reason": "raw phases contain SSH targets, remote commands, stdout, and stderr",
        },
    }
    executing = args.command != "plan" and args.execute
    receipt_command = args.command in (
        "activate",
        "restart-arm",
        "transition",
        "status",
    )
    if executing and receipt_command and not args.live_arm_receipt_output:
        parser.error(
            "executed activate/restart-arm/transition/status requires "
            "--live-arm-receipt-output"
        )
    if args.live_arm_receipt_output and not receipt_command:
        parser.error(
            "--live-arm-receipt-output is valid only for activate, restart-arm, "
            "transition, or status"
        )
    if executing and args.confirmation != CONFIRMATION:
        parser.error(f"execute requires --confirmation {CONFIRMATION}")
    output_stream = None
    receipt_stream = None
    try:
        output_stream = reserve_output(Path(args.output)) if args.output else None
        receipt_stream = (
            reserve_output(Path(args.live_arm_receipt_output))
            if args.live_arm_receipt_output
            else None
        )
    except exl3.ProfileError as exc:
        if output_stream is not None:
            output_stream.close()
        parser.error(str(exc))
    try:
        if not executing:
            emit(plan, output_stream)
            return 0
        timeout = canonical.startup_timeout_seconds + 60
        original_exception = None
        raised_phase = None
        if args.command == "rollback":
            results, ok = attempt_automatic_rollback(
                selected, all_phases, timeout
            )
        else:
            try:
                results, ok = execute_sequence(selected, all_phases, timeout)
            except PhaseExecutionError as exc:
                results = dict(exc.results)
                results["execution_exception"] = exception_evidence(exc)
                original_exception = exception_evidence(exc)
                raised_phase = exc.phase
                ok = False
        rollback_results = None
        if (
            args.command in ("activate", "restart-arm", "transition")
            and not ok
            and removal_attempted(args.command, results, raised_phase)
        ):
            rollback_results, restored = attempt_automatic_rollback(
                rollback_selected, all_phases, timeout
            )
            ok = False
            if not restored:
                results["automatic_rollback_failed"] = rollback_results
        report = {
            "plan": plan,
            "results": results,
            "original_exception": original_exception,
            "automatic_rollback": rollback_results,
        }
        try:
            if ok and receipt_stream is not None:
                receipt_phase = (
                    "target_live_arm_attestation"
                    if source is not None
                    else "diagnostic_live_arm_attestation"
                )
                observed_runtime_instances = validate_live_arm_phase_result(
                    results[receipt_phase]
                )
                live_arm_receipt = build_live_arm_receipt(
                    **live_receipt_builder_kwargs,
                    observed_runtime_instances=observed_runtime_instances,
                )
                receipt_stream.write(
                    json.dumps(
                        live_arm_receipt,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
                receipt_stream.flush()
            emit(report, output_stream)
        except Exception as exc:
            # Once an engine-removal phase has run, even a local evidence-write
            # exception is a failed transaction. Restore the canonical stack
            # and retain the original error plus rollback evidence on stdout.
            evidence = local_exception_evidence("emit_execution_report", exc)
            report["original_exception"] = evidence
            if receipt_stream is not None:
                try:
                    receipt_stream.seek(0)
                    receipt_stream.truncate(0)
                    receipt_stream.flush()
                except Exception:
                    pass
            if args.command in ("activate", "restart-arm", "transition") and removal_attempted(
                args.command, results, raised_phase
            ):
                rollback_results, restored = attempt_automatic_rollback(
                    rollback_selected, all_phases, timeout
                )
                report["automatic_rollback"] = rollback_results
                if not restored:
                    report["results"]["automatic_rollback_failed"] = rollback_results
            try:
                emit(report)
            except Exception:
                pass
            return 1
        return 0 if ok else 1
    except exl3.ProfileError as exc:
        parser.error(str(exc))
    finally:
        if output_stream is not None:
            output_stream.close()
        if receipt_stream is not None:
            receipt_stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
