#!/usr/bin/env python3
"""Profile-driven four-node runtime launcher.

Consumes a ``sparkring-runtime-profile/v1`` profile, produces a deterministic
offline plan, and reuses shared orchestration primitives from
:mod:`sparkring_runtime`.

Safety:

* ``plan`` is always offline and prints a deterministic JSON document.
* ``start`` and ``stop`` require ``--execute``.
* If the profile declares a ``confirmation`` token, mutating commands require
  ``--confirmation <token>``.
* Stop actions are profile-label-guarded (``org.sparkring.managed=true`` and
  ``org.sparkring.profile=<id>``) so a foreign same-named container is never
  removed.
* Deployment commands reject site/profile drift and unresolved template values.
* Each ``start`` action verifies the exact image digest and required source
  labels before ``docker run``.
* An optional ``attestation_hook`` runs after image verification and before
  ``docker run``.

Compatibility boundary: four-Spark GPU/RDMA clusters running vLLM-style
serving with the TP4/DCP degree declared by the validated site.
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
from sparkring_site import SiteConfigError, load_site  # noqa: E402


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------


def load_profile(path: Path) -> runtime.RuntimeProfile:
    """Load the native runtime profile schema."""
    return runtime.load_runtime_profile(path)




# ---------------------------------------------------------------------------
# Action construction
# ---------------------------------------------------------------------------


def build_actions(
    site: Any,
    profile: runtime.RuntimeProfile,
    command: str,
) -> list[runtime.RemoteAction]:
    """Build remote actions for the given command."""
    if command in ("plan", "start"):
        return runtime.start_actions(site, profile)
    if command == "stop":
        return runtime.stop_actions(site, profile)
    if command == "status":
        return runtime.status_actions(site, profile)
    if command == "verify-image":
        return runtime.verify_image_actions(site, profile)
    if command == "verify-rollback":
        return runtime.verify_rollback_actions(site, profile)
    if command == "health":
        actions = runtime.health_check_actions(site, profile)
        if not actions:
            raise runtime.ProfileError("profile has no health_check")
        return actions
    raise ValueError(f"unknown command: {command}")




# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _conformance_validate(args: Any, parser: Any) -> int:
    """Validate a profile and optional sanitized site."""
    try:
        profile = load_profile(Path(args.profile))
        site = None
        validation_scope = "structural"
        if args.site:
            site = load_site(args.site)
            _validate_site_profile_alignment(site, profile)
            build_actions(site, profile, "plan")
            validation_scope = "plan-build"
    except (
        OSError, KeyError, json.JSONDecodeError,
        SiteConfigError, runtime.ProfileError,
    ) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, indent=2))
        return 1
    template_unresolved = _is_template(profile) or (
        site is not None and _is_site_template(site)
    )
    result = {
        "valid": not template_unresolved,
        "validation_scope": validation_scope,
        "profile_id": profile.profile_id,
        "model_family": profile.model_family,
        "schema": runtime.SCHEMA,
        "identity_scope": _identity_scope(profile),
    }
    if template_unresolved:
        result["valid"] = False
        result["error"] = (
            "template/unresolved: profile contains obvious placeholder "
            "values or the site contains documentation/example values "
            "(REPLACE, all-zero image ID, all-zero identity pins, or "
            "placeholder host paths); fill in real site, model, image, and "
            "pin values before deployment"
        )
    print(json.dumps(result, indent=2))
    return 0 if not template_unresolved else 1


def _is_template(profile: runtime.RuntimeProfile) -> bool:
    """Detect obvious placeholder values in native profiles."""
    path_values = [profile.model_host_path]
    path_values.extend(volume[0] for volume in profile.extra_volumes)
    if "REPLACE" in profile.image or any(
        "REPLACE" in value for value in path_values
    ):
        return True
    if profile.image_id == "sha256:" + "0" * 64:
        return True
    for value in profile.identity.values():
        if value == "0" * 40 or value == "0" * 64:
            return True
        if "your-" in value or "REPLACE" in value:
            return True
    if any(
        value in {"0" * 64, "sha256:" + "0" * 64}
        or "REPLACE" in value
        for value in profile.required_image_labels.values()
    ):
        return True
    return False


def _is_site_template(site: Any) -> bool:
    """Detect every example value recognized by the site schema."""
    return bool(site.placeholder_warnings())


def _validate_site_profile_alignment(
    site: Any, profile: runtime.RuntimeProfile,
) -> None:
    """Reject contradictions between site-owned and profile-owned identity."""
    comparisons = (
        (
            "runtime.container_image",
            site.runtime.container_image,
            profile.image,
        ),
        (
            "runtime.container_image_digest",
            site.runtime.container_image_digest,
            profile.image_id,
        ),
        (
            "runtime.model_path",
            site.runtime.model_path,
            profile.model_container_path,
        ),
    )
    for field, site_value, profile_value in comparisons:
        if site_value != profile_value:
            raise runtime.ProfileError(
                f"site/profile mismatch for {field}: "
                f"site={site_value!r}, profile={profile_value!r}"
            )

    identity_comparisons = (
        ("target_repository", site.runtime.model_repo),
        ("target_revision", site.runtime.model_revision),
        ("target_cache_identity_sha256", site.runtime.checkpoint_sha256),
    )
    for key, site_value in identity_comparisons:
        profile_value = profile.identity.get(key)
        if profile_value is not None and profile_value != site_value:
            raise runtime.ProfileError(
                f"site/profile mismatch for identity.{key}: "
                f"site={site_value!r}, profile={profile_value!r}"
            )


def _require_resolved_inputs(
    site: Any, profile: runtime.RuntimeProfile,
) -> None:
    """Reject deployment commands that bypassed template resolution."""
    if _is_template(profile) or _is_site_template(site):
        raise runtime.ProfileError(
            "deployment input is unresolved; replace placeholder image "
            "identities and host paths before plan or lifecycle commands"
        )


def _conformance_explain(args: Any, parser: Any) -> int:
    """Show profile ownership, identity scope, hooks, safety classes, and
    target topology."""
    try:
        profile = load_profile(Path(args.profile))
        site = None
        if args.site:
            site = load_site(args.site)
            _validate_site_profile_alignment(site, profile)
    except (
        OSError, KeyError, json.JSONDecodeError,
        SiteConfigError, runtime.ProfileError,
    ) as exc:
        parser.error(str(exc))

    site_owned, profile_owned = _owned_settings(profile, site)
    info = {
        "schema": runtime.SCHEMA,
        "profile_id": profile.profile_id,
        "model_family": profile.model_family,
        "identity_scope": _identity_scope(profile),
        "required_image_labels": dict(sorted(profile.required_image_labels.items())),
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
    independently, produces a stable semantic projection, and compares all
    fields with recursive JSON paths: identity/image, topology/serving,
    labels, mounts, environment, extra vLLM args, hooks,
    confirmation/lifecycle guards, and per-rank actions keyed by rank with
    structured rank/ssh_target/remote_command.
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
    _validate_site_profile_alignment(site_a, left_profile)
    _validate_site_profile_alignment(site_b, right_profile)
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
        "engine": profile.engine,
        "container_name": profile.container_name,
        "image": profile.image,
        "image_id": profile.image_id,
        "model_host_path": profile.model_host_path,
        "model_container_path": profile.model_container_path,
        "shm_size": profile.shm_size,
        "startup_timeout_seconds": profile.startup_timeout_seconds,
        "init": profile.init,
        "security_opts": list(profile.security_opts),
        "privileged": profile.privileged,
        "entrypoint": profile.entrypoint,
        "confirmation": profile.confirmation,
        "environment": dict(sorted(profile.environment.items())),
        "extra_vllm_args": list(profile.extra_vllm_args),
        "extra_volumes": [list(v) for v in profile.extra_volumes],
        "extra_labels": dict(profile.extra_labels),
        "identity": dict(profile.identity),
        "required_image_labels": dict(profile.required_image_labels),
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
    if profile.attestation_hook:
        return "attestation-hook-configured"
    return "image-verified-before-start"


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
        "container_name", "image", "image_id", "model_host_path",
        "model_container_path", "shm_size", "startup_timeout_seconds",
        "environment", "extra_vllm_args",
        "extra_volumes", "extra_labels", "init", "security_opts",
        "privileged", "entrypoint",
        "confirmation", "identity", "required_image_labels",
        "attestation_hook", "health_check",
    }
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
        site = load_site(args.site)
        profile = load_profile(Path(args.profile))
        _validate_site_profile_alignment(site, profile)
        _require_resolved_inputs(site, profile)
        actions = build_actions(site, profile, args.command)
    except (
        OSError, KeyError, json.JSONDecodeError,
        SiteConfigError, runtime.ProfileError,
    ) as exc:
        parser.error(str(exc))

    plan = runtime.plan_document(args.command, actions, profile)

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
    failed_ranks = runtime.check_results(args.command, results)

    rollback_results = None
    if runtime.should_rollback(args.command) and failed_ranks:
        started = {
            rank for rank, result in results.items()
            if not runtime.check_results(args.command, {rank: result})
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
