#!/usr/bin/env python3
"""Shared runtime primitives for SparkRing launchers.

This module extracts the orchestration behavior that every model-family
launcher needs: per-rank RDMA/peer context derivation, transport
environment construction, the ``RemoteAction`` dataclass, parallel SSH
execution, and the generic runtime-profile contract.

Model-specific contract validation (exact pins, KV profiles, MTP modes,
speculative configs) lives in the canonical family launchers — not here.
The generic launcher delegates operations shared with EXL3 and NF3 to their
canonical builders, so those bridge actions are byte-identical. Operations
without a canonical counterpart use the generic builders and are identified
separately in the plan documentation.

Safety is proportional and inherited from the existing launchers:

* ``plan`` is always offline and prints a deterministic JSON document.
* ``start``/``stop`` require ``--execute``.
* Mutation commands with a ``confirmation`` field require an exact token.
* Stop actions are profile-label-guarded so a foreign same-named
  container is never removed.
* Each generic ``start`` action verifies the exact image digest before
  ``docker run`` — fail-closed on identity drift.
* An optional ``attestation_hook`` runs after image verification and
  before ``docker run`` (fail-closed model attestation).
* Execution semantics follow the source schema: EXL3 bridges use
  exit-status-only (no rollback); NF3 bridges and generic profiles use
  ``action_succeeded`` with partial-start rollback.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import shlex
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

SCHEMA = "sparkring-runtime-profile/v1"
PLAN_SCHEMA = "sparkring-runtime-plan/v1"

# Source-schema strings for bridge dispatch (kept here so both the generic
# launcher and the execution-mode logic can reference them without circular
# imports).
EXL3_SCHEMA = "sparkring-public-exl3-launch/v1"
NF3_SCHEMA = "sparkring-public-launch/v1"

# Character classes — kept independent so this module has no import
# dependency on the site validator.
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ABS_PATH = re.compile(r"^/[A-Za-z0-9._/+@:-]*[A-Za-z0-9._+@:-]$")
_ENV_KEY = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_SHM_SIZE = re.compile(r"^[1-9][0-9]*[gGmM]")
_PLACEHOLDER = re.compile(r"\{([a-z0-9_]+)\}")
_IDENTITY_KEY = re.compile(r"^[a-z][a-z0-9_]{0,48}$")
_LABEL_KEY = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*"
    r"(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?$"
)
_CONFIRMATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

# Placeholders that may appear in environment values and extra_vllm_args.
# These are expanded from site topology at plan time.
ALLOWED_PLACEHOLDERS = frozenset(
    {
        "api_port",
        "draft_path",
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
)

# Environment keys derived from the validated site.  A profile may not
# override these — the site owns transport topology.
SITE_DERIVED_ENVIRONMENT = frozenset(
    {
        "GLOO_SOCKET_IFNAME",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NCCL_IB_GID_INDEX",
        "NCCL_IB_HCA",
        "NCCL_IB_SUBNET_PREFIX_LEN",
        "NCCL_SOCKET_IFNAME",
        "RANK",
        "SPARKRING_IMAGE_DIGEST",
        "SPARK_TP4_DEVICE0",
        "SPARK_TP4_DEVICE1",
        "SPARK_TP4_GID0",
        "SPARK_TP4_GID1",
        "SPARK_TP4_PEER0",
        "SPARK_TP4_PEER1",
        "WORLD_SIZE",
    }
)

# Environment keys that the runtime owns.  A profile may not set any key
# with this prefix — identity-derived keys use SPARKRING_ATTEST_ instead.
_RUNTIME_ENV_PREFIX = "SPARKRING_"

# Prefix for identity-derived environment variables (avoids collision with
# runtime-owned keys like SPARKRING_IMAGE_DIGEST).
_ATTEST_PREFIX = "SPARKRING_ATTEST_"

# Label applied to every generic-started container so stop/rollback can
# distinguish it from a foreign same-named container.
PROFILE_LABEL = "org.sparkring.profile"
MANAGED_LABEL = "org.sparkring.managed"

# Labels that the runtime sets internally — a profile must not override.
# Bundle ownership labels are also reserved against profile overrides.
BUNDLE_LABEL = "org.sparkring.bundle"
SERVICE_LABEL = "org.sparkring.service"
SOURCE_PROFILE_LABEL = "org.sparkring.source-profile"
RESERVED_LABELS = frozenset({
    MANAGED_LABEL, PROFILE_LABEL, BUNDLE_LABEL, SERVICE_LABEL,
    SOURCE_PROFILE_LABEL,
})

# vLLM options that are site-owned (derived from site.serving at
# action-build time).  A generic profile may not duplicate these in
# extra_vllm_args — neither ``--option value`` nor ``--option=value``.
RESERVED_VLLM_OPTIONS = frozenset(
    {
        "--tensor-parallel-size",
        "--decode-context-parallel-size",
        "--max-model-len",
        "--kv-cache-memory-bytes",
        "--max-num-seqs",
        "--port",
        "--distributed-executor-backend",
        "--nnodes",
        "--node-rank",
        "--master-addr",
        "--master-port",
        "--headless",
    }
)


class ProfileError(ValueError):
    """A runtime profile is structurally invalid or violates its contract."""


# ---------------------------------------------------------------------------
# RemoteAction — extracted from both existing launchers (identical dataclass)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemoteAction:
    rank: int
    ssh_target: str
    argv: tuple[str, ...]

    @property
    def shell_command(self) -> str:
        return shlex.join(self.argv)



# ---------------------------------------------------------------------------
# Ownership — bundle identity labels for native start/stop guards
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ownership:
    """Exact identity labels for bundle-managed containers.

    When passed to ``start_actions``/``stop_actions``, adds four
    labels and stop guards so a foreign same-named container is
    never removed.
    """
    bundle_id: str
    service_id: str
    profile_id: str

# ---------------------------------------------------------------------------
# RuntimeProfile — the generic contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeProfile:
    """A validated, model-family-agnostic runtime profile.

    The generic launcher consumes this dataclass. For EXL3 and NF3 profiles,
    operations with canonical counterparts are delegated and byte-identical;
    generic-only operations use the shared builders.
    """

    profile_id: str
    model_family: str
    engine: str
    container_name: str
    image: str
    image_id: str
    model_host_path: str
    model_container_path: str
    shm_size: str
    startup_timeout_seconds: int
    environment: dict[str, str | None]
    extra_vllm_args: tuple[str, ...]
    # Source schema — determines dispatch and execution semantics.
    # Generic profiles set this to SCHEMA; bridge loaders set the
    # family-specific schema.
    source_schema: str = ""
    # Optional: secondary host paths to mount (e.g. MTP draft, JIT cache)
    extra_volumes: tuple[tuple[str, str, str], ...] = ()
    # Optional: container labels beyond the managed and profile labels
    extra_labels: dict[str, str] = field(default_factory=dict)
    # Optional: privileged flag (LMCache servers need it)
    privileged: bool = False
    # Optional: entrypoint override
    entrypoint: str | None = None
    # Optional: confirmation token required for mutating commands
    # (null/absent = no token; non-empty string = required token)
    confirmation: str | None = None
    # Optional: model identity attestation (sha256 pins, repository, revision)
    # Identity keys are lowercase snake_case; env vars use SPARKRING_ATTEST_
    # prefix to avoid collision with runtime-owned SPARKRING_ keys.
    identity: dict[str, str] = field(default_factory=dict)
    # Optional: pre-start model attestation hook — a validated argv array
    # that runs as ``docker run --rm <image> <hook>`` after exact image
    # verification and before the main ``docker run``.  Fail-closed.
    attestation_hook: tuple[str, ...] = ()
    # Optional: post-start health check — a validated argv array that runs
    # inside the container via ``docker exec``.  Has a deterministic
    # dry-run action path.
    health_check: tuple[str, ...] = ()
    # Raw document for bridge access
    document: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Structural validation (schema-level, not model-specific)
# ---------------------------------------------------------------------------


def _require_string(
    value: Any, where: str, pattern: re.Pattern[str] | None = None,
    what: str = "value",
) -> str:
    if not isinstance(value, str):
        raise ProfileError(f"{where}: must be a string")
    if not value:
        raise ProfileError(f"{where}: must be non-empty")
    if pattern and not pattern.fullmatch(value):
        raise ProfileError(f"{where}: invalid {what}")
    return value


def _require_int(value: Any, where: str, bounds: tuple[int, int]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileError(f"{where}: must be an integer")
    lo, hi = bounds
    if not lo <= value <= hi:
        raise ProfileError(f"{where}: must be in [{lo}, {hi}]")
    return value


def _validate_environment(env: Any, where: str) -> dict[str, str | None]:
    if not isinstance(env, dict):
        raise ProfileError(f"{where}: must be an object")
    checked: dict[str, str | None] = {}
    for key, value in env.items():
        if not isinstance(key, str) or not _ENV_KEY.fullmatch(key):
            raise ProfileError(f"{where}: invalid environment key {key!r}")
        if key.startswith(_RUNTIME_ENV_PREFIX):
            raise ProfileError(
                f"{where}: {key} uses reserved prefix "
                f"{_RUNTIME_ENV_PREFIX} (identity keys use "
                f"{_ATTEST_PREFIX} prefix automatically)"
            )
        if value is None:
            checked[key] = None
        elif isinstance(value, str):
            if "\x00" in value or "\n" in value:
                raise ProfileError(f"{where}: {key} must be one line or null")
            _validate_placeholders(value, f"{where}.{key}")
            checked[key] = value
        else:
            raise ProfileError(f"{where}: {key} must be a string or null")
    return checked


def _validate_placeholders(value: str, where: str) -> None:
    observed = set(_PLACEHOLDER.findall(value))
    unknown = observed - ALLOWED_PLACEHOLDERS
    if unknown:
        raise ProfileError(f"{where}: unknown placeholder {{{sorted(unknown)[0]}}}")


def _validate_argv_array(args: Any, where: str) -> tuple[str, ...]:
    """Validate a generic argv array (attestation_hook, health_check)."""
    if args is None:
        return ()
    if not isinstance(args, list):
        raise ProfileError(f"{where}: must be a list")
    for index, value in enumerate(args):
        if not isinstance(value, str) or not value or "\x00" in value or "\n" in value:
            raise ProfileError(f"{where}[{index}]: must be a non-empty one-line string")
        _validate_placeholders(value, f"{where}[{index}]")
    return tuple(args)


def _validate_extra_vllm_args(args: Any, where: str) -> tuple[str, ...]:
    """Validate extra_vllm_args, rejecting site-owned options."""
    checked = _validate_argv_array(args, where)
    for index, value in enumerate(checked):
        if value in RESERVED_VLLM_OPTIONS:
            raise ProfileError(
                f"{where}[{index}]: site-owned option {value} "
                "is not allowed in profile"
            )
        for opt in RESERVED_VLLM_OPTIONS:
            if value.startswith(opt + "="):
                raise ProfileError(
                    f"{where}[{index}]: site-owned option {opt}= "
                    "is not allowed in profile"
                )
    return checked


def _validate_extra_volumes(
    volumes: Any, where: str,
) -> tuple[tuple[str, str, str], ...]:
    if volumes is None:
        return ()
    if not isinstance(volumes, list):
        raise ProfileError(f"{where}: must be a list")
    checked = []
    for index, vol in enumerate(volumes):
        if not isinstance(vol, dict) or set(vol) != {"host", "container", "mode"}:
            raise ProfileError(f"{where}[{index}]: must have host/container/mode")
        host = _require_string(
            vol["host"], f"{where}[{index}].host", _ABS_PATH, "absolute path",
        )
        container = _require_string(
            vol["container"], f"{where}[{index}].container",
            _ABS_PATH, "absolute path",
        )
        mode = _require_string(vol["mode"], f"{where}[{index}].mode")
        if mode not in ("ro", "rw"):
            raise ProfileError(f"{where}[{index}].mode: must be ro or rw")
        checked.append((host, container, mode))
    return tuple(checked)


def _validate_extra_labels(labels: Any, where: str) -> dict[str, str]:
    if labels is None:
        return {}
    if not isinstance(labels, dict):
        raise ProfileError(f"{where}: must be an object")
    checked: dict[str, str] = {}
    for key, value in labels.items():
        if not isinstance(key, str) or not _LABEL_KEY.fullmatch(key):
            raise ProfileError(f"{where}: invalid label key {key!r}")
        if not isinstance(value, str) or "\x00" in value or "\n" in value:
            raise ProfileError(f"{where}.{key}: value must be a one-line string")
        if key in RESERVED_LABELS:
            raise ProfileError(
                f"{where}.{key}: reserved label "
                f"(set automatically by the runtime)"
            )
        checked[key] = value
    return checked


def _validate_identity(identity: Any) -> dict[str, str]:
    if identity is None:
        return {}
    if not isinstance(identity, dict):
        raise ProfileError("identity: must be an object")
    checked: dict[str, str] = {}
    for key, value in identity.items():
        if not isinstance(key, str) or not _IDENTITY_KEY.fullmatch(key):
            raise ProfileError(
                f"identity.{key}: invalid key "
                "(must be lowercase snake_case, max 50 chars)"
            )
        if (
            not isinstance(value, str)
            or not value
            or "\x00" in value
            or "\n" in value
        ):
            raise ProfileError(
                f"identity.{key}: must be a non-empty one-line string"
            )
        if key.startswith("sparkring_"):
            raise ProfileError(f"identity.{key}: reserved prefix")
        checked[key] = value
    return checked


def load_runtime_profile(path: Path) -> RuntimeProfile:
    """Load and structurally validate a generic runtime profile JSON file."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError(f"{path}: {exc}") from exc
    return parse_runtime_profile(document, source=str(path))


def parse_runtime_profile(
    document: Any, source: str | None = None,
) -> RuntimeProfile:
    """Structurally validate a parsed runtime profile document."""
    where = source or "<inline>"
    if not isinstance(document, dict):
        raise ProfileError(f"{where}: root must be an object")

    required = {
        "schema", "profile_id", "model_family", "engine",
        "container_name", "image", "image_id",
        "model_host_path", "model_container_path",
        "shm_size", "startup_timeout_seconds",
        "environment", "extra_vllm_args",
    }
    optional = {
        "extra_volumes", "extra_labels", "privileged", "entrypoint",
        "confirmation", "identity",
        "attestation_hook", "health_check",
    }
    unknown = sorted(set(document) - required - optional)
    missing = sorted(required - set(document))
    if unknown:
        raise ProfileError(f"{where}: unknown key {unknown[0]!r}")
    if missing:
        raise ProfileError(f"{where}: missing key {missing[0]!r}")

    if document["schema"] != SCHEMA:
        raise ProfileError(f"{where}: unsupported schema {document['schema']!r}")

    profile_id = _require_string(
        document["profile_id"], f"{where}.profile_id", _NAME, "profile id",
    )
    model_family = _require_string(
        document["model_family"], f"{where}.model_family", _NAME, "model family",
    )
    engine = _require_string(document["engine"], f"{where}.engine")
    if engine not in ("docker", "podman"):
        raise ProfileError(f"{where}.engine: must be docker or podman")
    container_name = _require_string(
        document["container_name"], f"{where}.container_name", _NAME, "container name",
    )
    image = _require_string(document["image"], f"{where}.image")
    image_id = _require_string(
        document["image_id"], f"{where}.image_id", _IMAGE_ID, "image ID",
    )
    model_host_path = _require_string(
        document["model_host_path"], f"{where}.model_host_path",
        _ABS_PATH, "absolute path",
    )
    model_container_path = _require_string(
        document["model_container_path"], f"{where}.model_container_path",
        _ABS_PATH, "absolute path",
    )
    shm_size = _require_string(
        document["shm_size"], f"{where}.shm_size", _SHM_SIZE, "shm size",
    )
    startup_timeout = _require_int(
        document["startup_timeout_seconds"], f"{where}.startup_timeout_seconds",
        (30, 7200),
    )

    environment = _validate_environment(
        document["environment"], f"{where}.environment",
    )
    derived_override = sorted(set(environment) & SITE_DERIVED_ENVIRONMENT)
    if derived_override:
        raise ProfileError(
            f"{where}: environment {derived_override[0]} is derived from "
            "the validated site and cannot be overridden"
        )

    extra_vllm_args = _validate_extra_vllm_args(
        document["extra_vllm_args"], f"{where}.extra_vllm_args",
    )
    extra_volumes = _validate_extra_volumes(
        document.get("extra_volumes"), f"{where}.extra_volumes",
    )

    extra_labels = _validate_extra_labels(
        document.get("extra_labels"), f"{where}.extra_labels",
    )

    privileged = document.get("privileged", False)
    if not isinstance(privileged, bool):
        raise ProfileError(f"{where}.privileged: must be a boolean")

    entrypoint = document.get("entrypoint")
    if entrypoint is not None:
        entrypoint = _require_string(entrypoint, f"{where}.entrypoint")

    confirmation = document.get("confirmation")
    if confirmation is not None:
        confirmation = _require_string(
            confirmation,
            f"{where}.confirmation",
            _CONFIRMATION,
            "confirmation token",
        )

    identity = _validate_identity(document.get("identity"))

    attestation_hook = _validate_argv_array(
        document.get("attestation_hook"), f"{where}.attestation_hook",
    )
    health_check = _validate_argv_array(
        document.get("health_check"), f"{where}.health_check",
    )

    return RuntimeProfile(
        profile_id=profile_id,
        model_family=model_family,
        engine=engine,
        container_name=container_name,
        image=image,
        image_id=image_id,
        model_host_path=model_host_path,
        model_container_path=model_container_path,
        shm_size=shm_size,
        startup_timeout_seconds=startup_timeout,
        environment=environment,
        extra_vllm_args=extra_vllm_args,
        source_schema=SCHEMA,
        extra_volumes=extra_volumes,
        extra_labels=extra_labels,
        privileged=privileged,
        entrypoint=entrypoint,
        confirmation=confirmation,
        identity=identity,
        attestation_hook=attestation_hook,
        health_check=health_check,
        document=document,
    )


def resolve_from_site(profile: RuntimeProfile, site: Any) -> RuntimeProfile:
    """Return a profile with image identity resolved from ``site.runtime``.

    NF3 bridge profiles leave ``image`` and ``image_id`` empty because
    the site owns the container image.  This resolves them before plan
    creation so ``profile_attestation.image_id`` is truthful.
    """
    if profile.image and profile.image_id:
        return profile
    return replace(
        profile,
        image=site.runtime.container_image,
        image_id=site.runtime.container_image_digest,
        model_container_path=(
            profile.model_container_path or site.runtime.model_path
        ),
    )


# ---------------------------------------------------------------------------
# Site context — extracted from both existing launchers (identical logic)
# ---------------------------------------------------------------------------


def site_context(site: Any, rank_id: int) -> dict[str, str]:
    """Derive per-rank transport context from a validated site config.

    The peer ordering follows the native TP4 recursive-doubling round
    schedule: round 0 is rank^1 and round 1 is rank^3.  Sorting by
    rank would silently reverse both slots on ranks 2 and 3.
    """
    rank = site.rank(rank_id)
    peers_by_rank = {peer.rank: peer for peer in rank.transport_peers}
    peers = [peers_by_rank[rank_id ^ 1], peers_by_rank[rank_id ^ 3]]
    ports = {port.peer_rank: port for port in rank.ring_ports}
    master = site.rank(site.serving.master_rank)
    return {
        "api_port": str(site.serving.api_port),
        "draft_path": "/mtp-draft",
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


def expand(value: str, context: Mapping[str, str]) -> str:
    """Expand ``{placeholder}`` tokens using a site context."""
    return _PLACEHOLDER.sub(lambda match: context[match.group(1)], value)


def base_environment(
    site: Any, rank_id: int, profile: RuntimeProfile,
) -> dict[str, str]:
    """Build the transport environment shared by all model families.

    This is the union of the two existing launchers' ``_base_environment``
    functions.  Model-specific attestation keys are added from
    ``profile.identity`` using the ``SPARKRING_ATTEST_`` prefix.
    """
    context = site_context(site, rank_id)
    rank = site.rank(rank_id)
    if context["peer0_gid"] != context["peer1_gid"]:
        raise ProfileError(
            f"rank {rank_id} uses different RoCE GID indices on its two "
            "ring ports; NCCL_IB_GID_INDEX is rank-global"
        )
    env: dict[str, str] = {
        "GLOO_SOCKET_IFNAME": rank.management.interface,
        "MASTER_ADDR": context["master_addr"],
        "MASTER_PORT": context["master_port"],
        "NCCL_IB_GID_INDEX": context["peer0_gid"],
        "NCCL_IB_HCA": f"{context['peer0_device']},{context['peer1_device']}",
        "NCCL_IB_SUBNET_PREFIX_LEN": "24",
        "NCCL_SOCKET_IFNAME": rank.management.interface,
        "RANK": context["rank"],
        "WORLD_SIZE": context["world_size"],
        "SPARKRING_IMAGE_DIGEST": profile.image_id,
        "SPARK_TP4_DEVICE0": context["peer0_device"],
        "SPARK_TP4_DEVICE1": context["peer1_device"],
        "SPARK_TP4_GID0": context["peer0_gid"],
        "SPARK_TP4_GID1": context["peer1_gid"],
        "SPARK_TP4_PEER0": context["peer0_addr"],
        "SPARK_TP4_PEER1": context["peer1_addr"],
    }
    for key, value in profile.identity.items():
        env[f"{_ATTEST_PREFIX}{key.upper()}"] = value
    return env


def container_name(profile: RuntimeProfile, rank: int) -> str:
    return f"{profile.container_name}-r{rank}"


def _image_verify_prefix(profile: RuntimeProfile) -> str:
    """Shell guard that fails closed if the local image ID differs."""
    return (
        f'test "$({profile.engine} image inspect --format '
        f"'{{{{.Id}}}}' {shlex.quote(profile.image)})\" = "
        f"{shlex.quote(profile.image_id)}"
    )


# ---------------------------------------------------------------------------
# Action builders — shared by all model families
# ---------------------------------------------------------------------------


def start_actions(
    site: Any, profile: RuntimeProfile, *,
    ownership: "Ownership | None" = None,
) -> list[RemoteAction]:
    """Build per-rank container start actions for a generic profile.

    Each action first verifies the exact image digest (fail-closed on
    identity drift).  If ``attestation_hook`` is set, it runs as
    ``docker run --rm <image> <hook>`` after image verification and
    before the main ``docker run`` (also fail-closed).

    If ``ownership`` is provided (by the bundle launcher), four
    exact identity labels are added: ``org.sparkring.managed``,
    ``org.sparkring.bundle``, ``org.sparkring.service``, and
    ``org.sparkring.source-profile``.
    """
    actions: list[RemoteAction] = []
    for rank in site.ranks:
        context = site_context(site, rank.id)
        environment = base_environment(site, rank.id, profile)
        for key, value in profile.environment.items():
            if value is None:
                environment[key] = None
            else:
                environment[key] = expand(value, context)

        explicitly_unset = sorted(
            name for name, value in environment.items() if value is None
        )
        environment = {
            name: value for name, value in environment.items() if value is not None
        }
        if explicitly_unset:
            environment["SPARKRING_EXPLICITLY_UNSET"] = ",".join(explicitly_unset)

        command: list[str] = [
            profile.engine,
            "run",
            "--detach",
            "--name",
            container_name(profile, rank.id),
            "--label",
            f"{MANAGED_LABEL}=true",
            "--label",
            f"{PROFILE_LABEL}={profile.profile_id}",
        ]
        if ownership:
            command.extend(("--label", f"{BUNDLE_LABEL}={ownership.bundle_id}"))
            command.extend(("--label", f"{SERVICE_LABEL}={ownership.service_id}"))
            command.extend((
                "--label", f"{SOURCE_PROFILE_LABEL}={ownership.profile_id}",
            ))
        if profile.extra_labels:
            for label_key, label_value in profile.extra_labels.items():
                command.extend(("--label", f"{label_key}={label_value}"))
        command.extend(
            (
                "--network", "host",
                "--ipc", "host",
                "--gpus", "all",
                "--shm-size", profile.shm_size,
                "--ulimit", "memlock=-1:-1",
                "--cap-add", "IPC_LOCK",
                "--device", "/dev/infiniband:/dev/infiniband",
            )
        )
        if profile.privileged:
            command.append("--privileged")
        if profile.entrypoint:
            command.extend(("--entrypoint", profile.entrypoint))

        command.extend(
            ("--volume",
             f"{profile.model_host_path}:{profile.model_container_path}:ro")
        )
        for host_path, container_path, mode in profile.extra_volumes:
            command.extend(("--volume", f"{host_path}:{container_path}:{mode}"))

        env_args: list[str] = []
        for name, value in sorted(environment.items()):
            env_args.extend(("--env", f"{name}={value}"))

        command.append(profile.image_id)
        if profile.entrypoint is None:
            command.extend(("serve", profile.model_container_path))

        command.extend(
            (
                "--tensor-parallel-size", str(site.serving.tensor_parallel_size),
                "--decode-context-parallel-size",
                str(site.serving.decode_context_parallel_size),
                "--max-model-len", str(site.serving.max_model_len),
                "--kv-cache-memory-bytes",
                str(site.serving.kv_cache_bytes_per_rank),
                "--max-num-seqs", str(site.serving.max_num_seqs),
                "--port", str(site.serving.api_port),
                "--distributed-executor-backend", "mp",
                "--nnodes", str(len(site.ranks)),
                "--node-rank", str(rank.id),
                "--master-addr", context["master_addr"],
                "--master-port", context["master_port"],
            )
        )

        image_index = command.index(profile.image_id)
        command[image_index:image_index] = env_args
        command.extend(expand(value, context) for value in profile.extra_vllm_args)

        if rank.id != site.serving.master_rank:
            command.append("--headless")

        guard = _image_verify_prefix(profile)
        if profile.attestation_hook:
            hook_entrypoint = expand(profile.attestation_hook[0], context)
            attest_cmd: list[str] = [
                profile.engine, "run", "--rm",
                "--volume",
                f"{profile.model_host_path}:{profile.model_container_path}:ro",
            ]
            for host_path, container_path, mode in profile.extra_volumes:
                attest_cmd.extend(
                    ("--volume", f"{host_path}:{container_path}:{mode}")
                )
            attest_cmd.extend(("--entrypoint", hook_entrypoint, profile.image_id))
            attest_cmd.extend(
                expand(arg, context) for arg in profile.attestation_hook[1:]
            )
            guard += f" && {shlex.join(attest_cmd)}"
        guard += f" && exec {shlex.join(command)}"
        actions.append(
            RemoteAction(rank.id, rank.ssh_target, ("sh", "-lc", guard))
        )
    return actions


def stop_actions(
    site: Any, profile: RuntimeProfile, *,
    ownership: "Ownership | None" = None,
) -> list[RemoteAction]:
    """Build profile-label-guarded container removal actions.

    A foreign container with the same name but a different
    ``org.sparkring.profile`` label is not removed (exit 73).

    If ``ownership`` is provided, all four ownership labels must
    match exactly before removal. A container missing any label
    or with a wrong value fails closed (exit 73).
    """
    actions: list[RemoteAction] = []
    for rank in site.ranks:
        name = container_name(profile, rank.id)
        if ownership:
            script = (
                # Item 1 (fb4): daemon probe, then exact-name enumeration
                f"{profile.engine} info >/dev/null 2>&1 || exit 74; "
                f"listing=$({profile.engine} ps -a --filter name=^/{shlex.quote(name)}$ "
                f"--format '{{{{.Names}}}}' 2>&1) || exit 74; "
                f'if [ -z "$listing" ]; then exit 0; fi; '
                f'if [ "$listing" != "{shlex.quote(name)}" ]; then exit 74; fi; '
                f'managed=$({profile.engine} inspect --format '
                f"'{{{{index .Config.Labels \"{MANAGED_LABEL}\"}}}}' "
                f"{shlex.quote(name)} 2>/dev/null) || exit 74; "
                f'[ "$managed" = true ] || exit 73; '
                f'pid=$({profile.engine} inspect --format '
                f"'{{{{index .Config.Labels \"{PROFILE_LABEL}\"}}}}' "
                f"{shlex.quote(name)} 2>/dev/null) || exit 74; "
                f'[ "$pid" = {shlex.quote(profile.profile_id)} ] || exit 73; '
                f'bid=$({profile.engine} inspect --format '
                f"'{{{{index .Config.Labels \"{BUNDLE_LABEL}\"}}}}' "
                f"{shlex.quote(name)} 2>/dev/null) || exit 74; "
                f'[ "$bid" = {shlex.quote(ownership.bundle_id)} ] || exit 73; '
                f'sid=$({profile.engine} inspect --format '
                f"'{{{{index .Config.Labels \"{SERVICE_LABEL}\"}}}}' "
                f"{shlex.quote(name)} 2>/dev/null) || exit 74; "
                f'[ "$sid" = {shlex.quote(ownership.service_id)} ] || exit 73; '
                f'src=$({profile.engine} inspect --format '
                f"'{{{{index .Config.Labels \"{SOURCE_PROFILE_LABEL}\"}}}}' "
                f"{shlex.quote(name)} 2>/dev/null) || exit 74; "
                f'[ "$src" = {shlex.quote(ownership.profile_id)} ] || exit 73; '
                f"exec {profile.engine} rm --force {shlex.quote(name)}"
            )
        else:
            script = (
                # Item 1 (fb4): daemon probe, then exact-name enumeration
                f"{profile.engine} info >/dev/null 2>&1 || exit 74; "
                f"listing=$({profile.engine} ps -a --filter name=^/{shlex.quote(name)}$ "
                f"--format '{{{{.Names}}}}' 2>&1) || exit 74; "
                f'if [ -z "$listing" ]; then exit 0; fi; '
                f'if [ "$listing" != "{shlex.quote(name)}" ]; then exit 74; fi; '
                f'managed=$({profile.engine} inspect --format '
                f"'{{{{index .Config.Labels \"{MANAGED_LABEL}\"}}}}' "
                f"{shlex.quote(name)} 2>/dev/null) || exit 74; "
                f'[ "$managed" = true ] || exit 73; '
                f'pid=$({profile.engine} inspect --format '
                f"'{{{{index .Config.Labels \"{PROFILE_LABEL}\"}}}}' "
                f"{shlex.quote(name)} 2>/dev/null) || exit 74; "
                f'[ "$pid" = {shlex.quote(profile.profile_id)} ] || exit 73; '
                f"exec {profile.engine} rm --force {shlex.quote(name)}"
            )
        actions.append(
            RemoteAction(rank.id, rank.ssh_target, ("sh", "-c", script))
        )
    return actions


def status_actions(site: Any, profile: RuntimeProfile) -> list[RemoteAction]:
    """Build container status inspection actions."""
    actions: list[RemoteAction] = []
    for rank in site.ranks:
        name = container_name(profile, rank.id)
        actions.append(
            RemoteAction(
                rank.id, rank.ssh_target,
                (profile.engine, "inspect", "--format", "{{.State.Status}}", name),
            )
        )
    return actions


def verify_image_actions(site: Any, profile: RuntimeProfile) -> list[RemoteAction]:
    """Build exact image ID attestation actions."""
    actions: list[RemoteAction] = []
    for rank in site.ranks:
        script = _image_verify_prefix(profile)
        actions.append(
            RemoteAction(rank.id, rank.ssh_target, ("sh", "-c", script))
        )
    return actions


def verify_rollback_actions(
    site: Any, profile: RuntimeProfile,
) -> list[RemoteAction]:
    """Build read-only rollback verification actions."""
    actions: list[RemoteAction] = []
    for rank in site.ranks:
        name = container_name(profile, rank.id)
        script = (
            f"! {shlex.join((profile.engine, 'container', 'inspect', name))} "
            ">/dev/null 2>&1"
        )
        actions.append(
            RemoteAction(rank.id, rank.ssh_target, ("sh", "-c", script))
        )
    return actions


def health_check_actions(site: Any, profile: RuntimeProfile) -> list[RemoteAction]:
    """Build exact-profile-guarded per-rank probe actions via ``docker exec``.

    The ``health_check`` field is a validated argv array that runs inside
    the container and therefore is not assumed to be read-only. Placeholders
    like ``{api_port}`` are expanded from site context. If no health_check is
    set, returns an empty list.
    """
    if not profile.health_check:
        return []
    actions: list[RemoteAction] = []
    for rank in site.ranks:
        context = site_context(site, rank.id)
        name = container_name(profile, rank.id)
        probe = [profile.engine, "exec", name]
        probe.extend(
            expand(arg, context) for arg in profile.health_check
        )
        script = (
            f"managed=$({profile.engine} inspect --format "
            f"'{{{{index .Config.Labels \"{MANAGED_LABEL}\"}}}}' "
            f"{shlex.quote(name)} 2>/dev/null) || exit 74; "
            f'[ "$managed" = true ] || exit 73; '
            f'pid=$({profile.engine} inspect --format '
            f"'{{{{index .Config.Labels \"{PROFILE_LABEL}\"}}}}' "
            f"{shlex.quote(name)} 2>/dev/null) || exit 74; "
            f'[ "$pid" = {shlex.quote(profile.profile_id)} ] || exit 73; '
            f"exec {shlex.join(probe)}"
        )
        actions.append(
            RemoteAction(rank.id, rank.ssh_target, ("sh", "-c", script))
        )
    return actions


# ---------------------------------------------------------------------------
# Execution — shared parallel SSH runner
# ---------------------------------------------------------------------------


def run_remote(action: RemoteAction, timeout: int) -> subprocess.CompletedProcess[str]:
    """Execute a single action over SSH and return the completed process.

    OpenSSH concatenates arguments following the target into one remote
    shell command.  Quoting the complete remote invocation as one
    argument prevents word-splitting.
    """
    remote_command = shlex.join(("sh", "-lc", action.shell_command))
    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            action.ssh_target,
            remote_command,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def execute(actions: list[RemoteAction], timeout: int) -> dict[int, dict]:
    """Execute actions in parallel over SSH and collect results."""

    def _output_text(
        value: str | bytes | None, fallback: str = "",
    ) -> str:
        if value is None:
            return fallback
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    def _one(action: RemoteAction) -> dict:
        result = run_remote(action, timeout)
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    results: dict[int, dict] = {}
    if not actions:
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(actions)) as pool:
        pending = {action.rank: pool.submit(_one, action) for action in actions}
        for rank, future in pending.items():
            try:
                results[rank] = future.result()
            except subprocess.TimeoutExpired as exc:
                results[rank] = {
                    "exit_code": 124,
                    "stdout": _output_text(exc.stdout),
                    "stderr": _output_text(exc.stderr, "remote command timed out"),
                }
            except Exception as exc:
                # Item 5 (fb4): reserved infrastructure code 125 + error_type
                results[rank] = {
                    "exit_code": 125,
                    "stdout": "",
                    "stderr": f"executor exception: {type(exc).__name__}: {exc}",
                    "error_type": type(exc).__name__,
                }
    return results


def action_succeeded(command: str, result: dict) -> bool:
    """Check whether a remote action succeeded."""
    if result["exit_code"] != 0:
        return False
    if command == "start":
        lines = [line.strip() for line in result["stdout"].splitlines()]
        return any(re.fullmatch(r"[0-9a-f]{12,64}", line) for line in lines)
    return True


def failed(result: dict[int, dict]) -> bool:
    return any(
        isinstance(item, dict) and item.get("exit_code", 0) != 0
        for item in result.values()
    )


def execution_mode(profile: RuntimeProfile) -> str:
    """Return the execution semantics for a profile.

    - ``exl3``: exit-status-only check, no rollback (EXL3 canonical).
    - ``nf3``: ``action_succeeded`` + rollback on start failure (NF3).
    - ``generic``: same as NF3.
    """
    if profile.source_schema == EXL3_SCHEMA:
        return "exl3"
    if profile.source_schema == NF3_SCHEMA:
        return "nf3"
    return "generic"


def check_results(command: str, results: dict[int, dict], mode: str) -> list[int]:
    """Return list of failed ranks, per execution mode."""
    if mode == "exl3":
        return [
            rank for rank, result in results.items()
            if result.get("exit_code", 1) != 0
        ]
    return [
        rank for rank, result in results.items()
        if not action_succeeded(command, result)
    ]


def should_rollback(command: str, mode: str) -> bool:
    """Whether to rollback on partial start failure, per execution mode."""
    if mode == "exl3":
        return False
    return command == "start"


# ---------------------------------------------------------------------------
# Plan document
# ---------------------------------------------------------------------------


def render(actions: list[RemoteAction]) -> list[dict]:
    return [
        {
            "rank": action.rank,
            "ssh_target": action.ssh_target,
            "remote_command": action.shell_command,
        }
        for action in actions
    ]


def plan_document(
    command: str, actions: list[RemoteAction], profile: RuntimeProfile,
    effective_settings: dict[str, Any] | None = None,
) -> dict:
    doc = {
        "schema": PLAN_SCHEMA,
        "command": command,
        "profile_id": profile.profile_id,
        "model_family": profile.model_family,
        "source_schema": profile.source_schema,
        "mutates_remote": command in ("start", "stop", "health"),
        "stops_serving_risk": command in ("stop", "health"),
        "identity_scope": (
            "canonical-model-verification"
            if profile.source_schema == EXL3_SCHEMA
            else (
                "declared-site-image"
                if profile.source_schema == NF3_SCHEMA
                else (
                    "attestation-hook-configured"
                    if profile.attestation_hook
                    else "image-verified-before-start"
                )
            )
        ),
        "profile_attestation": {
            "profile_id": profile.profile_id,
            "image_id": profile.image_id,
            "declared_identity": profile.identity,
        },
        "actions": render(actions),
    }
    if effective_settings:
        doc["effective_settings"] = effective_settings
    return doc
