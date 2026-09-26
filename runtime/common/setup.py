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
        # A serving profile's checkpoints table names Hugging Face branches of
        # its repository; runtime/common/qwen_flash_next.py applies the settings.
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


def storage_plan(card, *, model_path, cache_path, docker_path, reuse_model=False,
                 reuse_image=False, root=profiles.ROOT, disk_usage=shutil.disk_usage,
                 device_id=lambda path: path.stat().st_dev):
    """Sum additional allocations on shared filesystems, without writing files.

    Nonexistent destinations use their nearest existing ancestor. Operators must
    mount intended volumes first. Reuse flags are planning assumptions, not proof
    of existing asset identity. Cache/JIT headroom remains reserved during reuse.
    """
    policy = profiles.read_json(root / "profiles/storage-planning.json")
    if policy.get("schema") != "sparkring-storage-planning/v1":
        raise ValueError("Unsupported storage planning policy")
    model_gib = policy["checkpoint_allowance_gib"].get(card["model_repository"])
    if model_gib is None:
        raise ValueError("No storage allowance for this checkpoint; follow its guide")
    image_gib, cache_gib = policy["image_allowance_gib"], policy["cache_and_jit_allowance_gib"]
    if any(type(value) is not int or value <= 0 for value in (model_gib, image_gib, cache_gib)):
        raise ValueError("Storage allowances must be positive integer GiB")
    model, model_parent = existing_directory(model_path)
    cache, cache_parent = existing_directory(cache_path)
    docker, docker_parent = existing_directory(docker_path)
    if model == cache or model.is_relative_to(cache) or cache.is_relative_to(model):
        raise ValueError("Model and writable cache paths must be separate, non-nested directories")
    if reuse_model and (not model.is_dir() or not any(model.iterdir())):
        raise ValueError("--reuse-model requires an existing nonempty model directory")
    groups = {}
    for role, path, ancestor, amount in (
        ("checkpoint", model, model_parent, 0 if reuse_model else model_gib),
        ("image", docker, docker_parent, 0 if reuse_image else image_gib),
        ("cache-and-jit", cache, cache_parent, cache_gib),
    ):
        key = device_id(ancestor)
        observed = disk_usage(ancestor).free
        group = groups.setdefault(key, {"probe_path": str(ancestor), "free_bytes": observed,
                                       "required_bytes": 0, "destinations": []})
        group["free_bytes"] = min(group["free_bytes"], observed)
        group["required_bytes"] += amount * GIB
        group["destinations"].append({"role": role, "path": str(path), "additional_gib": amount})
    for group in groups.values():
        group["passed"] = group["free_bytes"] >= group["required_bytes"]
    return {
        "schema": "sparkring-storage-plan/v1", "profile": card["profile"],
        "passed": all(group["passed"] for group in groups.values()),
        "filesystems": list(groups.values()),
        "reuse_model": reuse_model, "reuse_image": reuse_image,
        "scope": "Conservative additional-space planning, not a measured minimum or asset verification. "
                 "Mount destination volumes first; account separately for image archives, other Docker "
                 "content stores, quotas and concurrent writes. No files were created.",
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
                          f"({destination['additional_gib']} GiB)")
            print(report["scope"])
        return 0 if report["passed"] else 1
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"SETUP ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
