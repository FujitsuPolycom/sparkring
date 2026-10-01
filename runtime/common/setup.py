"""Read-only installation selections and per-filesystem storage planning.

Release/profile owners supply identities. This module neither authenticates an
installed image nor configures networking, downloads weights or starts a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import shutil
import sys

from runtime.common import profiles

GIB = 1024 ** 3


def selection(profile_id, variant=None, root=profiles.ROOT):
    resolved = profiles.resolve(profile_id, root=root)
    definition, release = profiles.load(profile_id, root)
    publication_path = Path(definition["release"]).parent / "publication.json"
    if publication_path.as_posix() not in {item["path"] for item in release["inputs"]}:
        raise ValueError("This profile has no pinned publication; follow its own guide")
    publication = profiles.read_json(profiles.local_path(publication_path.as_posix(), root))
    if (publication.get("image_reference") != release["image"]
            or publication.get("platform") != "linux/arm64"
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", publication.get("image_id", ""))):
        raise ValueError("Publication does not match the selected ARM64 release")
    model = resolved["model"]
    source = profiles.read_json(profiles.local_path(definition["configuration"]["path"], root))
    variants = source.get("target_variants", {})
    checkpoints = source.get("checkpoints", {})
    if checkpoints:
        # A serving profile's checkpoints table pins each checkpoint's
        # repository and revision; runtime/common/qwen_flash_next.py applies
        # the checkpoint's settings.
        variant = (source.get("checkpoint_aliases") or {}).get(variant, variant) or source["checkpoint"]
        if variant not in checkpoints:
            raise ValueError("Select a checkpoint the profile lists: " + ", ".join(sorted(checkpoints)))
        model = checkpoints[variant]["model"]
    elif variants:
        variant = variant or "nvfp4-spark"
        if variant not in variants:
            raise ValueError("Select a declared target variant: " + ", ".join(variants))
        model = variants[variant]
    elif variant is not None:
        raise ValueError("This profile does not accept a target variant")
    if not model.get("repository") or not re.fullmatch(r"[0-9a-f]{40}", model.get("revision", "")):
        raise ValueError("Setup requires a pinned checkpoint repository and full revision")
    return {
        "schema": "sparkring-setup-selection/v1",
        "profile": profile_id,
        "configuration": definition["configuration"]["path"],
        "guide": resolved["guide"],
        "release": release["id"],
        "image_reference": publication["image_reference"],
        "image_id": publication["image_id"],
        "model_repository": model["repository"],
        "model_revision": model["revision"],
        "target_variant": variant,
        "nodes": resolved["serving"]["node_count"],
        "topology": resolved["topology"],
        "sparkcache": resolved["serving"].get("sparkcache", False),
        "evidence_scope": resolved["evidence_scope"],
        "scope": "Selection only; installed assets, hosts and serving are not verified.",
    }


def shell_selection(card):
    """Emit quoted assignments only; values never become executable shell text."""
    values = {
        "PROFILE_ID": card["profile"], "PROFILE_CONFIG": card["configuration"],
        "RELEASE": card["release"], "IMAGE_REF": card["image_reference"],
        "EXPECTED_IMAGE_ID": card["image_id"], "MODEL_REPO": card["model_repository"],
        "MODEL_REV": card["model_revision"], "NODE_COUNT": str(card["nodes"]),
        "SPARKCACHE_ENABLED": "1" if card["sparkcache"] else "0",
        "TARGET_MODEL_VARIANT": card["target_variant"] or "",
    }
    return "\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items())


def existing_directory(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"Use an absolute destination path: {value}")
    path = path.resolve()
    ancestor = path
    while not ancestor.exists():
        if ancestor.parent == ancestor:
            raise ValueError(f"Destination has no accessible filesystem: {value}")
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise ValueError(f"Destination ancestor is not a directory: {ancestor}")
    return path, ancestor


def pinned_checkpoint_bytes(card, root=profiles.ROOT):
    """Bytes that writing the card's checkpoint on an empty filesystem needs, from its pin manifest, or None.

    The figure is the checkpoint plan's (``checkpoint_plan.required_space``):
    every required file once, and the largest once more as headroom, at most
    ``checkpoint_plan.HEADROOM_CAP_BYTES``. None when
    ``profiles/checkpoints/<owner>--<name>/<revision>.json`` is absent or does
    not pin the card's repository and revision with positive sizes.
    """
    repository, revision = card["model_repository"], card["model_revision"]
    if (not isinstance(repository, str) or not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repository)
            or ".." in repository or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)):
        return None
    try:
        pins = profiles.read_json(Path(root) / "profiles/checkpoints" / repository.replace("/", "--") / f"{revision}.json")
    except (OSError, ValueError):
        return None
    if (not isinstance(pins, dict) or pins.get("schema") != "sparkring-checkpoint-pins/v1"
            or (pins.get("repository"), pins.get("revision")) != (repository, revision)
            or not isinstance(pins.get("files"), dict)):
        return None
    optional = set(pins.get("optional") or ())
    sizes = [entry.get("size") if isinstance(entry, dict) else None
             for name, entry in pins["files"].items() if name not in optional]
    if not sizes or not all(type(size) is int and size > 0 for size in sizes):
        return None
    from runtime.host import checkpoint_plan
    return checkpoint_plan.required_space(sizes)


def storage_plan(card, *, model_path, cache_path, docker_path, reuse_model=False,
                 reuse_image=False, root=profiles.ROOT, disk_usage=shutil.disk_usage,
                 device_id=lambda path: path.stat().st_dev):
    """Sum additional allocations on shared filesystems, without writing files.

    The checkpoint needs its pinned file sizes with the largest file again as
    headroom, capped as the install plan caps it (``pinned_checkpoint_bytes``),
    or the checkpoint allowance of
    ``profiles/storage-planning.json`` for a revision without a pin manifest.
    The image needs its unpacked size, its download size and the pull margin
    when the card records its image lock's sizes (``install_space.pull_bytes``),
    else the image allowance. The compile cache needs the compile cache
    allowance, also during reuse. Nonexistent destinations use their nearest
    existing ancestor. Operators must mount intended volumes first. Reuse flags
    are planning assumptions, not proof of existing asset identity.
    """
    from runtime.host import checkpoint_plan, install_space
    policy = profiles.read_json(root / "profiles/storage-planning.json")
    if policy.get("schema") != "sparkring-storage-planning/v2":
        raise ValueError("Unsupported storage planning policy")
    model_gib = policy["checkpoint_allowance_gib"].get(card["model_repository"])
    image_gib, cache_gib = policy["image_allowance_gib"], policy["compile_cache_allowance_gib"]
    if (any(type(value) is not int or value <= 0 for value in (image_gib, cache_gib))
            or not (model_gib is None or (type(model_gib) is int and model_gib > 0))):
        raise ValueError("Storage allowances must be positive integer GiB")
    pinned = pinned_checkpoint_bytes(card, root)
    if pinned is None and model_gib is None:
        raise ValueError("No storage allowance for this checkpoint; follow its guide")
    if reuse_model:
        checkpoint = (0, "reused")
    elif pinned is not None:
        checkpoint = (pinned, "pinned file sizes, the largest file again as headroom, at most "
                              f"{checkpoint_plan.HEADROOM_CAP_BYTES // GIB} GiB")
    else:
        checkpoint = (model_gib * GIB, "checkpoint allowance; the revision has no pin manifest")
    if reuse_image:
        image = (0, "reused")
    elif install_space.sized(card):
        image = (install_space.pull_bytes(card, image_gib * GIB), "unpacked and download sizes of the image lock")
    else:
        image = (image_gib * GIB, "image allowance; the image lock records no sizes")
    model, model_parent = existing_directory(model_path)
    cache, cache_parent = existing_directory(cache_path)
    docker, docker_parent = existing_directory(docker_path)
    if model == cache or model.is_relative_to(cache) or cache.is_relative_to(model):
        raise ValueError("Model and writable cache paths must be separate, non-nested directories")
    if reuse_model and (not model.is_dir() or not any(model.iterdir())):
        raise ValueError("--reuse-model requires an existing nonempty model directory")
    groups = {}
    for role, path, ancestor, (amount, basis) in (
        ("checkpoint", model, model_parent, checkpoint),
        ("image", docker, docker_parent, image),
        ("compile-cache", cache, cache_parent, (cache_gib * GIB, "compile cache allowance")),
    ):
        key = device_id(ancestor)
        observed = disk_usage(ancestor).free
        group = groups.setdefault(key, {"probe_path": str(ancestor), "free_bytes": observed,
                                       "required_bytes": 0, "destinations": []})
        group["free_bytes"] = min(group["free_bytes"], observed)
        group["required_bytes"] += amount
        group["destinations"].append({"role": role, "path": str(path), "additional_bytes": amount,
                                      "additional_gib": round(amount / GIB, 1), "basis": basis})
    for group in groups.values():
        group["passed"] = group["free_bytes"] >= group["required_bytes"]
    return {
        "schema": "sparkring-storage-plan/v1", "profile": card["profile"],
        "passed": all(group["passed"] for group in groups.values()),
        "filesystems": list(groups.values()),
        "reuse_model": reuse_model, "reuse_image": reuse_image,
        "scope": "Additional-space planning from pinned and recorded sizes where they exist, else allowances; "
                 "not a measured minimum or asset verification. Mount destination volumes first; account "
                 "separately for image archives, other Docker content stores, quotas and concurrent writes. "
                 "No files were created.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    show = actions.add_parser("show", help="resolve the selected image and checkpoint offline")
    storage = actions.add_parser("storage", help="check local destination filesystems without writing")
    for command in (show, storage):
        command.add_argument("profile")
        command.add_argument("--variant", choices=("nvfp4-spark", "nvfp4-qad"))
    show.add_argument("--format", choices=("text", "json", "shell"), default="text")
    for name in ("model", "cache", "docker"):
        storage.add_argument(f"--{name}-path", required=True)
    storage.add_argument("--reuse-model", action="store_true", help="budget no new checkpoint copy; verify it separately")
    storage.add_argument("--reuse-image", action="store_true", help="budget no new image; verify it separately")
    storage.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        card = selection(args.profile, args.variant)
        if args.action == "show":
            if args.format == "json":
                print(json.dumps(card, indent=2))
            elif args.format == "shell":
                print(shell_selection(card))
            else:
                for key, value in card.items():
                    if key != "schema":
                        print(f"{key}: {value}")
            return 0
        report = storage_plan(card, model_path=args.model_path, cache_path=args.cache_path,
                              docker_path=args.docker_path, reuse_model=args.reuse_model,
                              reuse_image=args.reuse_image)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            for group in report["filesystems"]:
                print(f"{'PASS' if group['passed'] else 'FAIL'} {group['probe_path']}: "
                      f"{group['free_bytes'] / GIB:.1f} GiB free; "
                      f"{group['required_bytes'] / GIB:.1f} GiB additional allowance")
                for destination in group["destinations"]:
                    print(f"  {destination['role']}: {destination['path']} "
                          f"({destination['additional_gib']} GiB: {destination['basis']})")
            print(report["scope"])
        return 0 if report["passed"] else 1
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"SETUP ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
