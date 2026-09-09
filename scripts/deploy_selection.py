"""Bind one reviewed image receipt to every managed deployment stage."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SELECTED_RECEIPT = "selected-image-receipt.json"


def profile_module(profile):
    spec = importlib.util.spec_from_file_location("selected_mesh_profile", Path(profile) / "profile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selection(spec, profile):
    """Keep canonical defaults; explicit local receipts never become registry digests."""
    profile = Path(profile)
    selected = spec.get("runtime_selection")
    pins = json.loads((profile / "pins.json").read_text())
    if selected is None:
        public = json.loads((profile / "public-image.json").read_text())
        receipt = json.loads((profile / "image-receipt.json").read_text())
        return {**public, "image_reference": public["public_reference"], "local": False, "pins": pins,
                "receipt": receipt, "inside_image": receipt["inside_image"]}
    if not isinstance(selected, dict) or set(selected) != {"schema", "image_receipt"}:
        raise ValueError("Runtime selection must bind exactly one image receipt")
    if selected["schema"] != "sparkring-deploy-runtime-selection/v1":
        raise ValueError("Unsupported runtime selection schema")
    module = profile_module(profile)
    document = module.validate_image_receipt(selected["image_receipt"])
    if document["schema"] == "sparkring-mtp3-performance-public-image/v1":
        if spec.get("site", {}).get("runtime_profile") is not None:
            raise ValueError("Canonical public image cannot select a local source profile")
        return {"image_reference": document["image_reference"],
                "config_image_id": document["image_id"], "local": False,
                "pins": dict(pins, canonical_bundle_manifest_sha256=document["bundle_manifest_sha256"]),
                "receipt": selected["image_receipt"], "inside_image": document["inside_image"]}
    if document["schema"] != "sparkring-source-image-receipt/v1":
        raise ValueError("Explicit source composition requires a local verified receipt")
    if spec.get("site", {}).get("runtime_profile") != document["profile"]:
        raise ValueError("Deployment profile differs from selected image receipt")
    lock = json.loads(module.SOURCE_LOCK.read_text())
    return {"image_reference": document["image_reference"],
            "config_image_id": document["image_id"], "local": True,
            "pins": dict(pins, target={**pins["target"], **lock["target"]},
                         canonical_bundle_manifest_sha256=lock["runtime"]["bundle_manifest_sha256"]),
            "receipt": selected["image_receipt"], "inside_image": document["inside_image"]}


def receipt_path(spec, root):
    root = Path(root)
    if spec.get("runtime_selection") is not None:
        return root / SELECTED_RECEIPT
    return root / "source/runtime/glm53-spark-mtp3-mesh/image-receipt.json"


def validate_selected_file(spec, root, profile):
    path = receipt_path(spec, root)
    if spec.get("runtime_selection") is None:
        return path
    selected = selection(spec, profile)
    if path.is_symlink() or json.loads(path.read_text()) != selected["receipt"]:
        raise ValueError("Staged image receipt differs from approved selection")
    return path
