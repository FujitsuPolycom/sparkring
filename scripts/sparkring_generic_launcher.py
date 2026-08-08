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


def _conformance_validate(args: Any, parser: Any) -> int:
    """Validate a profile plus optional sanitized site.

    Without --site: structural/family validation only.
    With --site: exercises the same offline resolution and action/plan
    building used by the lifecycle plan path, catching profile/site
    incompatibilities and action-build failures.
    Template profiles (containing unresolved placeholders) are reported
    as "template/unresolved" and are not deployable-valid.
    """
    try:
        profile = load_profile(Path(args.profile))
        site = None
        validation_scope = "structural"
        if args.site:
            site = load_site(args.site)
            if profile.source_schema == runtime.NF3_SCHEMA:
                profile = runtime.resolve_from_site(profile, site)
            # Exercise the same action-building path used by plan
            build_actions(site, profile, "plan")
            validation_scope = "plan-build"
    except (
        OSError, KeyError, json.JSONDecodeError,
        SiteConfigError, runtime.ProfileError,
        exl3.ProfileError, nf3.LaunchConfigError,
    ) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, indent=2))
        return 1
    template_unresolved = _is_template(profile)
    result = {
        "valid": not template_unresolved,
        "validation_scope": validation_scope,
        "profile_id": profile.profile_id,
        "model_family": profile.model_family,
        "source_schema": profile.source_schema,
        "identity_scope": _identity_scope(profile),
    }
    if template_unresolved:
        result["valid"] = False
        result["error"] = (
            "template/unresolved: profile contains obvious placeholder "
            "values (REPLACE, all-zero image ID, all-zero identity pins); "
            "fill in real model identity, image, and pins before deployment"
        )
    print(json.dumps(result, indent=2))
    return 0 if not template_unresolved else 1


def _is_template(profile: runtime.RuntimeProfile) -> bool:
    """Detect obvious placeholder values that make a profile a template,
    not a deployable profile.

    Bridge profiles (EXL3/NF3) have image/identity resolved from site
    or canonical launcher, so they are never template/unresolved.
    """
    if profile.source_schema != runtime.SCHEMA:
        return False
    if "REPLACE" in profile.image:
        return True
    if profile.image_id == "sha256:" + "0" * 64:
        return True
    for value in profile.identity.values():
        if value == "0" * 40 or value == "0" * 64:
            return True
        if "your-" in value or "REPLACE" in value:
            return True
    return False


def _conformance_explain(args: Any, parser: Any) -> int:
    """Show source schema, family, site-owned vs profile-owned settings,
    identity scope, hooks (structural argv or explicit safety representation),
    safety classes, target topology, and which canonical launcher owns
    family validation."""
    try:
        profile = load_profile(Path(args.profile))
        site = None
        if args.site:
            site = load_site(args.site)
            if profile.source_schema == runtime.NF3_SCHEMA:
                profile = runtime.resolve_from_site(profile, site)
    except (
        OSError, KeyError, json.JSONDecodeError,
        SiteConfigError, runtime.ProfileError,
        exl3.ProfileError, nf3.LaunchConfigError,
    ) as exc:
        parser.error(str(exc))

    family_owner = _family_validation_owner(profile)
    site_owned, profile_owned = _owned_settings(profile, site)
    info = {
        "schema": profile.source_schema,
        "profile_id": profile.profile_id,
        "model_family": profile.model_family,
        "identity_scope": _identity_scope(profile),
        "family_validation_owner": family_owner,
        "site_owned_settings": sorted(site_owned),
        "profile_owned_settings": sorted(profile_owned),
        "hooks": _hook_representation(profile),
        "safety_classes": _safety_classes(),
        "target_topology": _target_topology(site),
        "claim_disclaimer": (
            "explain describes structure only; it does not claim "
            "model correctness or live acceptance"
        ),
    }
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


def _conformance_diff(args: Any, parser: Any) -> int:
    """Compare two resolved offline plans or profiles in a stable semantic
    form.  Exit codes: 0=same, 1=different, 2=invalid.

    With --site-a/--site-b (or shared --site): builds each side
    independently through its canonical launcher, produces a stable
    semantic projection, and compares all fields with recursive JSON
    paths: identity/image, topology/serving, labels, mounts, environment,
    extra vLLM args, hooks, confirmation/lifecycle guards, and per-rank
    actions keyed by rank with structured rank/ssh_target/remote_command.
    Without any site: compares profile dataclass fields only (scope is
    explicitly reported as "profile-only").
    """
    try:
        if args.site_a or args.site_b or args.site:
            return _diff_plans(args)
        return _diff_profiles(args)
    except (
        OSError, KeyError, json.JSONDecodeError,
        SiteConfigError, runtime.ProfileError,
        exl3.ProfileError, nf3.LaunchConfigError,
    ) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 2


def _diff_profiles(args: Any) -> int:
    """Profile-only semantic diff (no site required)."""
    left = load_profile(Path(args.profile_a))
    right = load_profile(Path(args.profile_b))
    left_proj = _profile_projection(left)
    right_proj = _profile_projection(right)
    diffs = _recursive_diff(left_proj, right_proj)
    if not diffs:
        print(json.dumps({"identical": True, "scope": "profile-only"}, indent=2))
        return 0
    print(json.dumps({
        "identical": False, "scope": "profile-only", "differences": diffs,
    }, indent=2))
    return 1


def _diff_plans(args: Any) -> int:
    """Site-aware diff comparing resolved offline plans built independently.

    One-sided site fallback: if only --site-a is given, it is used for
    both sides. If only --site-b is given, it is used for both sides.
    If --site is given, it is the shared site. Never calls load_site(None).
    """
    shared = args.site
    site_a_path = args.site_a
    site_b_path = args.site_b
    # One-sided fallback: if only one side is given, use it for both
    if site_a_path and not site_b_path and not shared:
        site_b_path = site_a_path
    elif site_b_path and not site_a_path and not shared:
        site_a_path = site_b_path
    elif shared:
        site_a_path = site_a_path or shared
        site_b_path = site_b_path or shared
    if not site_a_path or not site_b_path:
        raise runtime.ProfileError(
            "plan diff requires --site-a and --site-b (or shared --site)"
        )
    site_a = load_site(site_a_path)
    site_b = load_site(site_b_path)
    left_profile = load_profile(Path(args.profile_a))
    right_profile = load_profile(Path(args.profile_b))
    if left_profile.source_schema == runtime.NF3_SCHEMA:
        left_profile = runtime.resolve_from_site(left_profile, site_a)
    if right_profile.source_schema == runtime.NF3_SCHEMA:
        right_profile = runtime.resolve_from_site(right_profile, site_b)
    left_plan = runtime.plan_document(
        "plan", build_actions(site_a, left_profile, "plan"), left_profile,
    )
    right_plan = runtime.plan_document(
        "plan", build_actions(site_b, right_profile, "plan"), right_profile,
    )
    left_proj = _plan_projection(left_plan, site_a, left_profile)
    right_proj = _plan_projection(right_plan, site_b, right_profile)
    diffs = _recursive_diff(left_proj, right_proj)
    if not diffs:
        print(json.dumps({"identical": True, "scope": "plan"}, indent=2))
        return 0
    print(json.dumps({
        "identical": False, "scope": "plan", "differences": diffs,
    }, indent=2))
    return 1


# ---------------------------------------------------------------------------
# Semantic projections
# ---------------------------------------------------------------------------


def _profile_projection(profile: runtime.RuntimeProfile) -> dict[str, Any]:
    """Stable semantic projection of a profile for diff comparison."""
    return {
        "profile_id": profile.profile_id,
        "model_family": profile.model_family,
        "source_schema": profile.source_schema,
        "engine": profile.engine,
        "container_name": profile.container_name,
        "image": profile.image,
        "image_id": profile.image_id,
        "model_host_path": profile.model_host_path,
        "model_container_path": profile.model_container_path,
        "shm_size": profile.shm_size,
        "startup_timeout_seconds": profile.startup_timeout_seconds,
        "privileged": profile.privileged,
        "entrypoint": profile.entrypoint,
        "confirmation": profile.confirmation,
        "environment": dict(sorted(profile.environment.items())),
        "extra_vllm_args": list(profile.extra_vllm_args),
        "extra_volumes": [list(v) for v in profile.extra_volumes],
        "extra_labels": dict(profile.extra_labels),
        "identity": dict(profile.identity),
        "attestation_hook": list(profile.attestation_hook),
        "health_check": list(profile.health_check),
    }


def _plan_projection(
    plan: dict[str, Any], site: Any, profile: runtime.RuntimeProfile,
) -> dict[str, Any]:
    """Stable semantic projection of a resolved plan for diff comparison.

    Includes topology, per-rank actions with structured rank/ssh_target/
    remote_command, profile-level semantic fields, and plan-level fields.
    """
    proj = _profile_projection(profile)
    proj["topology"] = _target_topology(site)
    proj["identity_scope"] = plan.get("identity_scope")
    proj["mutates_remote"] = plan.get("mutates_remote")
    proj["profile_attestation"] = _jsonable(
        plan.get("profile_attestation", {}))
    # Actions keyed by rank with structured fields
    actions_by_rank: dict[str, dict[str, Any]] = {}
    for action in plan.get("actions", []):
        actions_by_rank[str(action["rank"])] = {
            "rank": action["rank"],
            "ssh_target": action.get("ssh_target", ""),
            "remote_command": action.get("remote_command", ""),
        }
    proj["actions_by_rank"] = dict(sorted(actions_by_rank.items()))
    return proj


def _recursive_diff(
    left: Any, right: Any, prefix: str = "",
) -> list[dict[str, Any]]:
    """Recursively compare two values, returning exact JSON-path deltas.

    Produces paths like environment.FOO, topology.tensor_parallel_size,
    actions_by_rank.2.remote_command. Never normalizes away meaningful
    changes."""
    diffs: list[dict[str, Any]] = []
    if type(left) is not type(right):
        diffs.append({
            "field": prefix or "<root>",
            "a": _jsonable(left),
            "b": _jsonable(right),
        })
        return diffs
    if isinstance(left, dict):
        all_keys = sorted(set(left) | set(right))
        for key in all_keys:
            path = f"{prefix}.{key}" if prefix else key
            lv = left.get(key, _MISSING)
            rv = right.get(key, _MISSING)
            if lv is _MISSING:
                diffs.append({"field": path, "a": None, "b": _jsonable(rv)})
            elif rv is _MISSING:
                diffs.append({"field": path, "a": _jsonable(lv), "b": None})
            else:
                diffs.extend(_recursive_diff(lv, rv, path))
    elif isinstance(left, list):
        for index in range(max(len(left), len(right))):
            path = f"{prefix}[{index}]"
            if index >= len(left):
                diffs.append({"field": path, "a": None, "b": _jsonable(right[index])})
            elif index >= len(right):
                diffs.append({"field": path, "a": _jsonable(left[index]), "b": None})
            else:
                diffs.extend(_recursive_diff(left[index], right[index], path))
    elif left != right:
        diffs.append({
            "field": prefix or "<root>",
            "a": _jsonable(left),
            "b": _jsonable(right),
        })
    return diffs


_MISSING = object()


# ---------------------------------------------------------------------------
# Explain helpers
# ---------------------------------------------------------------------------


def _hook_representation(profile: runtime.RuntimeProfile) -> dict[str, Any]:
    """Show hooks with structural argv, not just booleans."""
    return {
        "attestation_hook": (
            list(profile.attestation_hook) if profile.attestation_hook else None
        ),
        "health_check": (
            list(profile.health_check) if profile.health_check else None
        ),
    }


def _identity_scope(profile: runtime.RuntimeProfile) -> str:
    if profile.source_schema == runtime.EXL3_SCHEMA:
        return "canonical-model-verification"
    if profile.source_schema == runtime.NF3_SCHEMA:
        return "declared-site-image"
    if profile.attestation_hook:
        return "attestation-hook-configured"
    return "image-verified-before-start"


def _family_validation_owner(profile: runtime.RuntimeProfile) -> str:
    if profile.source_schema == runtime.EXL3_SCHEMA:
        return "sparkring_exl3_launcher"
    if profile.source_schema == runtime.NF3_SCHEMA:
        return "sparkring_launcher"
    return "sparkring_runtime (structural validation only)"


def _owned_settings(
    profile: runtime.RuntimeProfile, site: Any = None,
) -> tuple[set[str], set[str]]:
    """Return the complete site/profile ownership boundary for ``explain``."""
    site_owned: set[str] = set()
    if site is not None:
        site_owned.update({
            "topology.mtu", "topology.link_speed_mbps", "topology.edges",
            "ranks.id", "ranks.ssh_target", "ranks.management",
            "ranks.ring_ports", "ranks.transport_peers",
            "serving.tensor_parallel_size",
            "serving.decode_context_parallel_size", "serving.mtp_mode",
            "serving.mtp_tokens", "serving.max_model_len",
            "serving.kv_cache_bytes_per_rank", "serving.max_num_seqs",
            "serving.master_rank", "serving.api_port", "serving.master_port",
        })
    profile_owned = {
        "source_schema", "profile_id", "model_family", "engine",
        "container_name", "image", "image_id", "model_host_path",
        "model_container_path", "shm_size", "startup_timeout_seconds",
        "environment", "extra_vllm_args",
        "extra_volumes", "extra_labels", "privileged", "entrypoint",
        "confirmation", "identity", "attestation_hook", "health_check",
    }
    if profile.source_schema == runtime.NF3_SCHEMA:
        # The NF3 bridge resolves these fields from site.runtime rather than
        # from its launch-profile document.
        profile_owned.difference_update({"image", "image_id", "model_container_path"})
        site_owned.update({
            "runtime.container_image", "runtime.container_image_digest",
            "runtime.model_path", "runtime.model_repo",
            "runtime.model_revision", "runtime.checkpoint_sha256",
        })
    return site_owned, profile_owned


def _safety_classes() -> dict[str, list[str]]:
    """Map each command to its safety class using AGENTS.md spellings."""
    return {
        "plan": ["OFFLINE"],
        "start": ["MUTATES HOST", "STOPS SERVING"],
        "stop": ["MUTATES HOST", "STOPS SERVING"],
        "status": ["READ-ONLY REMOTE"],
        "verify-image": ["READ-ONLY REMOTE"],
        "verify-rollback": ["READ-ONLY REMOTE"],
        "health": ["MUTATES HOST", "STOPS SERVING"],
    }


def _target_topology(site: Any) -> dict[str, Any] | None:
    if site is None:
        return None
    return {
        "mtu": site.topology.mtu,
        "link_speed_mbps": site.topology.link_speed_mbps,
        "edges": [
            {
                "id": edge.id,
                "subnet": str(edge.subnet),
                "endpoints": list(edge.endpoints),
            }
            for edge in site.topology.edges
        ],
        "ranks": [
            {
                "id": rank.id,
                "ssh_target": rank.ssh_target,
                "management": {
                    "interface": rank.management.interface,
                    "address": str(rank.management.address),
                },
                "ring_ports": [
                    {
                        "edge": port.edge,
                        "interface": port.interface,
                        "address": str(port.address),
                        "rdma_device": port.rdma_device,
                        "rdma_port": port.rdma_port,
                        "roce_gid_index": port.roce_gid_index,
                        "peer_rank": port.peer_rank,
                        "peer_address": str(port.peer_address),
                        "prefix_length": port.prefix_length,
                    }
                    for port in rank.ring_ports
                ],
                "transport_peers": [
                    {"rank": peer.rank, "address": str(peer.address)}
                    for peer in rank.transport_peers
                ],
            }
            for rank in site.ranks
        ],
        "serving": {
            "tensor_parallel_size": site.serving.tensor_parallel_size,
            "decode_context_parallel_size": site.serving.decode_context_parallel_size,
            "mtp_mode": site.serving.mtp_mode,
            "mtp_tokens": site.serving.mtp_tokens,
            "max_model_len": site.serving.max_model_len,
            "kv_cache_bytes_per_rank": site.serving.kv_cache_bytes_per_rank,
            "max_num_seqs": site.serving.max_num_seqs,
            "master_rank": site.serving.master_rank,
            "api_port": site.serving.api_port,
            "master_port": site.serving.master_port,
        },
    }


def _jsonable(value: Any) -> Any:
    """Convert a dataclass value to JSON-serializable form."""
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if value is _MISSING:
        return None
    return value

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=(
            "Offline conformance commands: validate checks structure and, with "
            "--site, canonical plan construction; explain reports ownership, "
            "identity, hooks, safety, and topology; diff exits 0 for identical, "
            "1 for different, and 2 for invalid input."
        ),
    )
    parser.add_argument("--site", required=False)
    parser.add_argument("--profile", required=False)
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
        "--profile-a",
        dest="profile_a",
        help="first profile for diff command",
    )
    parser.add_argument(
        "--profile-b",
        dest="profile_b",
        help="second profile for diff command",
    )
    parser.add_argument(
        "--site-a",
        dest="site_a",
        default=None,
        help="site for the first side of a plan diff (independent of --site-b)",
    )
    parser.add_argument(
        "--site-b",
        dest="site_b",
        default=None,
        help="site for the second side of a plan diff (independent of --site-a)",
    )
    parser.add_argument(
        "command",
        choices=(
            "plan", "start", "stop", "status",
            "verify-image", "verify-rollback", "health",
            "validate", "explain", "diff",
        ),
    )
    args = parser.parse_args(argv)

    # --- Conformance commands (offline, no SSH) ---
    if args.command in ("validate", "explain", "diff"):
        if args.execute:
            parser.error(f"{args.command} is always offline; remove --execute")
        if args.confirmation:
            parser.error(
                f"--confirmation does not apply to offline {args.command}"
            )
        if args.max_num_batched_tokens is not None:
            parser.error(
                "--max-num-batched-tokens applies only to EXL3 plan or start"
            )
    if args.command == "validate":
        _require_arg(parser, args, "profile", "validate requires --profile")
        return _conformance_validate(args, parser)
    if args.command == "explain":
        _require_arg(parser, args, "profile", "explain requires --profile")
        return _conformance_explain(args, parser)
    if args.command == "diff":
        _require_arg(parser, args, "profile_a", "diff requires --profile-a")
        _require_arg(parser, args, "profile_b", "diff requires --profile-b")
        return _conformance_diff(args, parser)

    # --- Existing lifecycle commands ---
    _require_arg(parser, args, "site", f"{args.command} requires --site")
    _require_arg(parser, args, "profile", f"{args.command} requires --profile")

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


def _require_arg(parser: Any, args: Any, arg_name: str, msg: str) -> None:
    """Exit with parser.error if a required argument is missing."""
    if not getattr(args, arg_name):
        parser.error(msg)


if __name__ == "__main__":
    raise SystemExit(main())
