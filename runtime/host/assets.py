"""Find complete checkpoint candidates; existing launch gates verify every shard."""
import hashlib
import json
from pathlib import Path
import subprocess

from runtime.common import installer, setup
from runtime.host import node


def metadata_matches(path, contract):
    root = Path(path)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        return False
    for filename, field in (("config.json", "config_sha256"), ("model.safetensors.index.json", "index_sha256")):
        item = root / filename
        if item.is_symlink() or not item.is_file() or hashlib.sha256(item.read_bytes()).hexdigest() != contract[field]:
            return False
    index = json.loads((root / "model.safetensors.index.json").read_text())
    names = set(index["weight_map"].values())
    return bool(names) and all((root / name).is_file() and not (root / name).is_symlink()
                               and (root / name).resolve().is_relative_to(root.resolve()) for name in names)


def discover(profile, *, run=subprocess.run, extra_roots=()):
    card = setup.selection(profile)
    contract = installer.checkpoint_contract(card)
    candidates = set(map(str, extra_roots))
    ids = node.call(["docker", "ps", "-aq"], run=run).stdout.splitlines()
    if ids:
        containers = json.loads(node.call(["docker", "inspect", *ids], run=run).stdout)
        for container in containers:
            for mount in container.get("Mounts", []):
                if mount.get("Type") == "bind" and "model" in mount.get("Destination", "").lower():
                    candidates.add(mount["Source"])
    revision = card["model_revision"]
    for directory in (Path("/var/tmp/models"), Path("/models"), Path("/srv/models")):
        if directory.is_dir():
            candidates.update(str(p) for p in directory.glob("*/" + revision))
    base = Path("/srv/sparkring")
    if base.is_dir():
        candidates.update(str(p) for p in base.glob("*/models/" + revision))
        candidates.update(str(p) for p in base.glob("*/*/models/" + revision))
    matches = [path for path in sorted(candidates) if metadata_matches(path, contract)]
    return {"profile": profile, "model_repository": card["model_repository"], "model_revision": revision,
            "model_path": matches[0] if matches else None, "candidates": len(matches),
            "verification": "metadata-and-completeness" if matches else "not-found",
            "full_shard_verification": "required-before-launch"}
