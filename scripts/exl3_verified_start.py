#!/usr/bin/env python3
"""Shared, receipt-visible EXL3 startup after one outer model verification.

Hashing the 339 GB EXL3 checkpoint immediately before GB10 model startup can
leave its contents resident in the unified-memory page cache.  This module
provides the reviewed transaction primitive used by diagnostic and SparkCache
launchers: install an attested entrypoint, prove non-interactive cache-reclaim
authority, mount that entrypoint without changing ``Config.Cmd``, and reclaim
the host page cache between the launcher's full verifier and Docker startup.

The caller remains responsible for running a complete, identity-bound model
verifier against the same read-only model view before invoking the decorated
start action.  Setting the outer-verification environment flag without that
proof is a contract violation.
"""

from __future__ import annotations

import base64
import hashlib
import shlex
from pathlib import Path

import sparkring_exl3_launcher as exl3


ROOT = Path(__file__).resolve().parents[1]
OUTER_VERIFIED_ENTRYPOINT = ROOT / "runtime/exl3/entrypoint_outer_verified.sh"
REMOTE_OUTER_VERIFIED_ENTRYPOINT = (
    "/var/lib/sparkring/verified-start/entrypoint-outer-verified-v1.sh"
)
CONTAINER_ENTRYPOINT = "/opt/sparkring-exl3/entrypoint.sh"


def outer_verified_entrypoint_sha256() -> str:
    try:
        payload = OUTER_VERIFIED_ENTRYPOINT.read_bytes()
    except OSError as exc:
        raise exl3.ProfileError(
            f"cannot read verified-start entrypoint asset: {exc}"
        ) from exc
    if b"\r\n" in payload or not payload.startswith(b"#!/usr/bin/env bash\n"):
        raise exl3.ProfileError("verified-start entrypoint asset must be LF-only bash")
    return hashlib.sha256(payload).hexdigest()


def with_outer_verified_entrypoint(
    actions: list[exl3.RemoteAction], profile: exl3.Profile
) -> list[exl3.RemoteAction]:
    """Mount the attested no-duplicate-verifier entrypoint without changing Cmd."""
    expected_sha256 = outer_verified_entrypoint_sha256()
    image_marker = f" {profile.image_id}"
    mount = (
        f"--volume {REMOTE_OUTER_VERIFIED_ENTRYPOINT}:{CONTAINER_ENTRYPOINT}:ro "
        "--env SPARKRING_OUTER_MODEL_VERIFIED=1 "
        f"--env SPARKRING_OUTER_ENTRYPOINT_SHA256={expected_sha256} "
    )
    guard = (
        f'test "$(sha256sum {shlex.quote(REMOTE_OUTER_VERIFIED_ENTRYPOINT)} | '
        f"awk '{{print $1}}')\" = {expected_sha256}"
    )
    result = []
    for action in actions:
        command = action.argv[-1]
        image_index = command.rfind(image_marker)
        image_end = image_index + len(image_marker)
        if (
            image_index < 0
            or image_end >= len(command)
            or command[image_end] != " "
        ):
            raise exl3.ProfileError("cannot locate unique engine image boundary")
        command = (
            f"{guard} && "
            + command[: image_index + 1]
            + mount
            + command[image_index + 1 :]
        )
        result.append(
            exl3.RemoteAction(action.rank, action.ssh_target, ("sh", "-lc", command))
        )
    return result


def with_post_verification_page_cache_reclaim(
    actions: list[exl3.RemoteAction], profile: exl3.Profile
) -> list[exl3.RemoteAction]:
    """Reclaim unified-memory page cache after verification and before GPU init."""
    marker = f" && exec {profile.engine} run --detach "
    hook = "sudo -n sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'"
    reclaimed = []
    for action in actions:
        command = action.argv[-1]
        if command.count(marker) != 1:
            raise exl3.ProfileError(
                "cannot locate unique post-verification engine launch boundary"
            )
        command = command.replace(marker, f" && {hook}{marker}", 1)
        reclaimed.append(
            exl3.RemoteAction(action.rank, action.ssh_target, ("sh", "-lc", command))
        )
    return reclaimed


def decorate_verified_start(
    actions: list[exl3.RemoteAction], profile: exl3.Profile
) -> list[exl3.RemoteAction]:
    """Apply the complete post-verifier startup decoration."""
    return with_post_verification_page_cache_reclaim(
        with_outer_verified_entrypoint(actions, profile), profile
    )


def without_embedded_model_verification(
    actions: list[exl3.RemoteAction], profile: exl3.Profile
) -> list[exl3.RemoteAction]:
    """Remove exactly one standard verifier after an equivalent outer receipt.

    This is intentionally strict and is only suitable when the caller has
    already completed a full, identity-bound verifier phase.  The image-ID
    guard and exact Docker ``Config.Cmd`` remain unchanged.
    """
    verifier = exl3.model_verification_script(profile)
    marker = f" && {verifier} && exec "
    result = []
    for action in actions:
        command = action.argv[-1]
        if command.count(marker) != 1:
            raise exl3.ProfileError(
                "cannot locate unique embedded EXL3 model-verification boundary"
            )
        command = command.replace(marker, " && exec ", 1)
        result.append(
            exl3.RemoteAction(action.rank, action.ssh_target, ("sh", "-lc", command))
        )
    return result


def prepare_verified_start_actions(site) -> list[exl3.RemoteAction]:
    """Install/attest the wrapper and prove reclaim access before interruption."""
    try:
        payload = OUTER_VERIFIED_ENTRYPOINT.read_bytes()
    except OSError as exc:
        raise exl3.ProfileError(
            f"cannot read verified-start entrypoint asset: {exc}"
        ) from exc
    expected_sha256 = outer_verified_entrypoint_sha256()
    encoded = base64.b64encode(payload).decode("ascii")
    actions = []
    for rank in site.ranks:
        script = (
            "sudo -n sh -c 'test -w /proc/sys/vm/drop_caches' && "
            "tmp=$(mktemp) && trap 'rm -f \"$tmp\"' EXIT && "
            f"printf %s {shlex.quote(encoded)} | base64 -d > \"$tmp\" && "
            f'test "$(sha256sum "$tmp" | awk \'{{print $1}}\')" = {expected_sha256} && '
            f"sudo -n install -D -m 0555 \"$tmp\" {shlex.quote(REMOTE_OUTER_VERIFIED_ENTRYPOINT)} && "
            f'test "$(sha256sum {shlex.quote(REMOTE_OUTER_VERIFIED_ENTRYPOINT)} | awk \'{{print $1}}\')" = {expected_sha256}'
        )
        actions.append(
            exl3.RemoteAction(rank.id, rank.ssh_target, ("sh", "-lc", script))
        )
    return actions
