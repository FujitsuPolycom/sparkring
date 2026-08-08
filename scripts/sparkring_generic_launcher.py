#!/usr/bin/env python3
"""Generic, profile-driven four-node runtime launcher.

Consumes a sanitized runtime profile (``sparkring-runtime-profile/v1``)
or a family-specific launch config (EXL3 ``sparkring-public-exl3-launch/v1``
or NF3 ``sparkring-public-launch/v1``), produces a deterministic offline
plan, and reuses the shared orchestration primitives from
:mod:`sparkring_runtime`.

Dispatch is by **source schema**, not ``model_family``.  A valid
native generic profile named ``exl3`` or ``nf3`` is still a generic
profile — it uses generic action builders and generic execution
semantics.  Only a profile whose ``schema`` field is
``sparkring-public-exl3-launch/v1`` or ``sparkring-public-launch/v1``
takes the EXL3 or NF3 bridge path respectively.

For EXL3 and NF3 profiles, operations present in the canonical family
launchers are delegated to those builders and are byte-identical. EXL3
``verify-rollback`` and NF3 ``verify-image`` have no canonical counterpart and
use generic builders. Delegated start/stop operations preserve EXL3 model
verification, exact profile-label cleanup, writable JIT cache, effective
batch-token handling, and NF3's mounts, labels, nullable environment,
candidate entrypoint/startup-cap behavior, and image source.

Execution semantics also follow the source schema:

* EXL3 bridge: exit-status-only check, no partial-start rollback.
* NF3 bridge: ``action_succeeded`` with partial-start rollback.
* Generic: same as NF3.

Safety is proportional and inherited from the existing launchers:

* ``plan`` is always offline and prints a deterministic JSON document.
* ``start`` and ``stop`` require ``--execute``.
* If the profile declares a ``confirmation`` token, mutating commands
  require ``--confirmation <token>``.
* Stop actions are profile-label-guarded (``org.sparkring.managed=true``
  and ``org.sparkring.profile=<id>``) so a foreign same-named
  container is never removed.
* Each generic ``start`` action verifies the exact image digest before
  ``docker run`` — fail-closed on identity drift.
* An optional ``attestation_hook`` runs after image verification and
  before ``docker run`` (fail-closed model attestation).

Compatibility boundary: four-Spark GPU/RDMA clusters running
vLLM-style serving with TP4/DCP4 parallelism.

Usage::

    # Generic profile
    python scripts/sparkring_generic_launcher.py \\
        --site scripts/config/site.yaml \\
        --profile scripts/config/generic.example.json plan

    # EXL3 launch profile (backward-compatible bridge)
    python scripts/sparkring_generic_launcher.py \\
        --site .sparkring/bootstrap-exl3/site.yaml \\
        --profile .sparkring/bootstrap-exl3/launch.json plan

    # NF3 launch config (backward-compatible bridge)
    python scripts/sparkring_generic_launcher.py \\
        --site scripts/config/site.yaml \\
        --profile scripts/config/launch.json plan
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sparkring_runtime as runtime  # noqa: E402
import sparkring_exl3_launcher as exl3  # noqa: E402
import sparkring_launcher as nf3  # noqa: E402
from sparkring_site import SiteConfigError, load_site  # noqa: E402


# ---------------------------------------------------------------------------
# Profile loading — static schema dispatch, no registry
# ---------------------------------------------------------------------------


def load_profile(path: Path) -> runtime.RuntimeProfile:
    """Load any supported profile format and return a RuntimeProfile.

    Dispatches by schema string (F1):

    - ``sparkring-runtime-profile/v1`` — generic native, parsed here.
    - ``sparkring-public-exl3-launch/v1`` — EXL3 bridge.
    - ``sparkring-public-launch/v1`` — NF3 bridge.

    A native generic profile with ``model_family="exl3"`` stays generic
    because dispatch is by ``schema``, not ``model_family``.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise runtime.ProfileError(f"{path}: {exc}") from exc

    if not isinstance(document, dict):
        raise runtime.ProfileError(f"{path}: root must be an object")

    schema = document.get("schema", "")
    if schema == runtime.SCHEMA:
        return runtime.parse_runtime_profile(document, source=str(path))
    elif schema == runtime.EXL3_SCHEMA:
        return _load_exl3_bridge(path)
    elif schema == runtime.NF3_SCHEMA:
        return _load_nf3_bridge(path)
    else:
        raise runtime.ProfileError(
            f"{path}: unsupported schema {schema!r}; expected "
            f"{runtime.SCHEMA}, {runtime.EXL3_SCHEMA}, or {runtime.NF3_SCHEMA}"
        )


def _load_exl3_bridge(path: Path) -> runtime.RuntimeProfile:
    """Load an EXL3 launch profile and convert it to RuntimeProfile."""
    exl3_profile = exl3.load_profile(path)
    doc = exl3_profile.document
    identity = {
        "model_config_sha256": doc["model_config_sha256"],
        "model_index_sha256": doc["model_index_sha256"],
        "model_repository": doc["model_repository"],
        "model_revision": doc["model_revision"],
        "model_tier_bitmap_sha256": doc["model_tier_bitmap_sha256"],
        "model_manifest_sha256": doc["model_manifest_sha256"],
        "model_path": doc["model_container_path"],
    }
    # EXL3 JIT cache must be writable (not read-only)
    extra_volumes = (
        (doc["jit_cache_host_path"], "/cache/jit", "rw"),
    )
    return runtime.RuntimeProfile(
        profile_id=doc["profile_id"],
        model_family="exl3",
        engine=doc["engine"],
        container_name=doc["container_name"],
        image=doc["image"],
        image_id=doc["image_id"],
        model_host_path=doc["model_host_path"],
        model_container_path=doc["model_container_path"],
        shm_size=doc["shm_size"],
        startup_timeout_seconds=doc["startup_timeout_seconds"],
        environment=doc["environment"],
        extra_vllm_args=tuple(doc["extra_vllm_args"]),
        source_schema=runtime.EXL3_SCHEMA,
        extra_volumes=extra_volumes,
        extra_labels={"org.sparkring.exl3-profile": doc["profile_id"]},
        identity=identity,
        document=doc,
    )


def _load_nf3_bridge(path: Path) -> runtime.RuntimeProfile:
    """Load an NF3 launch config and convert it to RuntimeProfile.

    NF3 gets its image identity from ``site.runtime`` at action-build
    time, so ``image`` and ``image_id`` are left empty here and filled
    by :func:`_build_nf3_actions` or :func:`runtime.resolve_from_site`.
    """
    config = nf3.load_launch(path)
    extra_volumes = (
        (config.mtp_draft_host_path, "/mtp-draft", "ro"),
    )
    return runtime.RuntimeProfile(
        profile_id=f"nf3-{config.container_name}",
        model_family="nf3",
        engine=config.engine,
        container_name=config.container_name,
        image="",
        image_id="",
        model_host_path=config.model_host_path,
        model_container_path="{model_path}",
        shm_size=config.shm_size,
        startup_timeout_seconds=config.startup_timeout_seconds,
        environment=config.environment,
        extra_vllm_args=config.extra_vllm_args,
        source_schema=runtime.NF3_SCHEMA,
        extra_volumes=extra_volumes,
        document={
            "schema": nf3.SCHEMA,
            "engine": config.engine,
            "container_name": config.container_name,
            "model_host_path": config.model_host_path,
            "mtp_draft_host_path": config.mtp_draft_host_path,
            "shm_size": config.shm_size,
            "startup_timeout_seconds": config.startup_timeout_seconds,
            "environment": config.environment,
            "extra_vllm_args": list(config.extra_vllm_args),
        },
    )


# ---------------------------------------------------------------------------
# Action construction — delegates to canonical builders for bridges
# ---------------------------------------------------------------------------


def build_actions(
    site: Any,
    profile: runtime.RuntimeProfile,
    command: str,
    max_num_batched_tokens: int | None = None,
) -> list[runtime.RemoteAction]:
    """Build remote actions for the given command.

    Dispatch is by ``profile.source_schema`` (F1).  For EXL3 and NF3
    profiles, delegates operations with a canonical counterpart to that
    family launcher's action builders. Generic-only operations and native
    generic profiles use the shared builders in :mod:`sparkring_runtime`.

    For EXL3 bridge ``plan``/``start``, ``max_num_batched_tokens`` is
    passed through to the canonical launcher (F6).
    """
    if command == "health":
        if profile.source_schema != runtime.SCHEMA:
            raise runtime.ProfileError(
                "health is available only for native generic profiles"
            )
        actions = runtime.health_check_actions(site, profile)
        if not actions:
            raise runtime.ProfileError(
                "native generic profile has no health_check"
            )
        return actions

    if profile.source_schema == runtime.EXL3_SCHEMA:
        return _build_exl3_actions(site, profile, command, max_num_batched_tokens)
    elif profile.source_schema == runtime.NF3_SCHEMA:
        return _build_nf3_actions(site, profile, command)

    # Generic profile — use shared builders
    if command in ("plan", "start"):
        return runtime.start_actions(site, profile)
    elif command == "stop":
        return runtime.stop_actions(site, profile)
    elif command == "status":
        return runtime.status_actions(site, profile)
    elif command == "verify-image":
        return runtime.verify_image_actions(site, profile)
    elif command == "verify-rollback":
        return runtime.verify_rollback_actions(site, profile)
    else:
        raise ValueError(f"unknown command: {command}")


def _build_exl3_actions(
    site: Any,
    profile: runtime.RuntimeProfile,
    command: str,
    max_num_batched_tokens: int | None = None,
) -> list[runtime.RemoteAction]:
    """Delegate to the canonical EXL3 launcher for all commands.

    For ``plan``/``start``, passes ``max_num_batched_tokens`` through
    to preserve EXL3's bounded --max-num-batched-tokens override (F6).
    """
    exl3_profile = exl3.Profile(profile.document)
    if command in ("plan", "start"):
        return exl3.start_actions(
            site, exl3_profile,
            max_num_batched_tokens=max_num_batched_tokens,
        )
    elif command in ("stop", "status", "verify-image", "verify-model"):
        return exl3.simple_actions(site, exl3_profile, command)
    elif command == "verify-rollback":
        # EXL3 simple_actions does not have verify-rollback; use the
        # generic builder which checks container absence.
        return runtime.verify_rollback_actions(site, profile)
    else:
        raise ValueError(f"unknown command: {command}")


def _build_nf3_actions(
    site: Any,
    profile: runtime.RuntimeProfile,
    command: str,
) -> list[runtime.RemoteAction]:
    """Delegate to the canonical NF3 launcher for all commands.

    NF3 gets its image identity from ``site.runtime``, not the launch
    config, so we must build the LaunchConfig and call the canonical
    builders directly.
    """
    config = nf3.LaunchConfig(
        engine=profile.engine,
        container_name=profile.container_name,
        model_host_path=profile.model_host_path,
        mtp_draft_host_path=profile.extra_volumes[0][0],
        shm_size=profile.shm_size,
        startup_timeout_seconds=profile.startup_timeout_seconds,
        environment=profile.environment,
        extra_vllm_args=profile.extra_vllm_args,
    )
    if command in ("plan", "start"):
        return nf3.start_actions(site, config)
    elif command in ("stop", "status", "verify-rollback"):
        return nf3.simple_actions(site, config, command)
    elif command == "verify-image":
        # NF3 simple_actions does not have verify-image; use the generic
        # builder which checks the exact image digest.
        # Build a resolved profile with image from site.
        resolved = runtime.resolve_from_site(profile, site)
        return runtime.verify_image_actions(site, resolved)
    else:
        raise ValueError(f"unknown command: {command}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--confirmation",
        default="",
        help="confirmation token required for mutating commands "
        "when the profile declares one",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help=(
            "bounded EXL3 bridge A/B override; "
            "only applies to EXL3 plan/start"
        ),
    )
    parser.add_argument(
        "command",
        choices=(
            "plan", "start", "stop", "status",
            "verify-image", "verify-rollback", "health",
        ),
    )
    args = parser.parse_args(argv)

    try:
        if args.command == "plan" and args.execute:
            raise runtime.ProfileError("plan is always offline; remove --execute")
        if (
            args.max_num_batched_tokens is not None
            and args.command not in ("plan", "start")
        ):
            raise runtime.ProfileError(
                "--max-num-batched-tokens applies only to plan or start"
            )
        site = load_site(args.site)
        profile = load_profile(Path(args.profile))

        if (
            args.max_num_batched_tokens is not None
            and profile.source_schema != runtime.EXL3_SCHEMA
        ):
            raise runtime.ProfileError(
                "--max-num-batched-tokens is available only for the EXL3 bridge"
            )

        # F5: resolve NF3 bridge identity from site before creating plan
        if profile.source_schema == runtime.NF3_SCHEMA:
            profile = runtime.resolve_from_site(profile, site)

        actions = build_actions(
            site, profile, args.command, args.max_num_batched_tokens,
        )
    except (
        OSError, KeyError, json.JSONDecodeError,
        SiteConfigError, runtime.ProfileError,
        exl3.ProfileError, nf3.LaunchConfigError,
    ) as exc:
        parser.error(str(exc))

    # F6: preserve EXL3 effective_settings in the plan document
    effective_settings = None
    if profile.source_schema == runtime.EXL3_SCHEMA:
        effective_batch_tokens = exl3._effective_max_num_batched_tokens(
            args.max_num_batched_tokens
        )
        effective_settings = {
            "default_max_num_batched_tokens": exl3.DEFAULT_MAX_NUM_BATCHED_TOKENS,
            "max_num_batched_tokens": effective_batch_tokens,
            "experiment_overrides": (
                {
                    "max_num_batched_tokens": effective_batch_tokens,
                }
                if args.max_num_batched_tokens is not None
                and effective_batch_tokens != exl3.DEFAULT_MAX_NUM_BATCHED_TOKENS
                else {}
            ),
        }

    plan = runtime.plan_document(
        args.command, actions, profile, effective_settings,
    )

    if not args.execute:
        print(json.dumps(plan, indent=2))
        if args.command != "plan":
            print(
                f"DRY RUN: {args.command} made no remote connection; "
                "add --execute",
                file=sys.stderr,
            )
        return 0

    # A profile-supplied health argv runs inside the serving container. It may
    # mutate state or stop serving, so it receives the same optional
    # confirmation-token gate as explicit lifecycle mutations.
    mutating = args.command in ("start", "stop", "health")
    requires_confirmation = mutating and profile.confirmation is not None
    if requires_confirmation and args.confirmation != profile.confirmation:
        parser.error(
            f"execute requires --confirmation {profile.confirmation}"
        )

    results = runtime.execute(actions, profile.startup_timeout_seconds)
    mode = runtime.execution_mode(profile)
    failed_ranks = runtime.check_results(args.command, results, mode)

    rollback_results = None
    if runtime.should_rollback(args.command, mode) and failed_ranks:
        started = {
            rank for rank, result in results.items()
            if not runtime.check_results(args.command, {rank: result}, mode)
        }
        rollback_actions = [
            action
            for action in build_actions(site, profile, "stop")
            if action.rank in started
        ]
        rollback_results = runtime.execute(
            rollback_actions, profile.startup_timeout_seconds
        )

    print(
        json.dumps(
            {
                "schema": "sparkring-runtime-result/v1",
                "command": args.command,
                "profile_id": profile.profile_id,
                "model_family": profile.model_family,
                "source_schema": profile.source_schema,
                "execution_mode": mode,
                "passed": not failed_ranks,
                "results": results,
                "rollback_results": rollback_results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failed_ranks else 1


if __name__ == "__main__":
    raise SystemExit(main())
