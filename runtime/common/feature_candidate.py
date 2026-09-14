"""Admit shared feature images against their complete pinned cache64 parent."""

import copy
import hashlib
from pathlib import Path

from runtime.common import cache_candidate, candidate

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR = ROOT / "runtime/images/compositions/lil-r37-shared/descriptor.json"
PARENT_RECEIPT = "/opt/sparkring/receipts/feature-parent-installed.json"
INSTALLED_DESCRIPTOR = "/opt/sparkring/receipts/feature-extension.json"
INSTALLER = "/opt/sparkring/bin/feature-extension.py"
FEATURE_ROOT = "/opt/sparkring/features/"
HOOKS = {
    cache_candidate.SITE + "sparkring_features.pth",
    cache_candidate.SITE + "sparkring_transport.pth",
}


def descriptor():
    return candidate._read(DESCRIPTOR.read_bytes())


def validate(
    image_id,
    installed_bytes,
    parent_bytes,
    base_bytes,
    verification,
    feature_verification,
):
    """Check recorded child evidence and the trusted cache64/R37 inventory chain.

    The caller must obtain both verifier outputs, installed_bytes and the retained
    parent_bytes from the same exact child image, and check its platform and
    configured entrypoint. Supply base_bytes from the registered R37 image or an
    exact retained copy; the cache descriptor pins those bytes independently.
    Parent evidence is a structural projection of the fully checked child
    inventory, not an independently observed verification of a parent image.
    Admission does not qualify serving or activate the included capabilities.
    """
    if any(
        not isinstance(raw, bytes)
        for raw in (installed_bytes, parent_bytes, base_bytes)
    ):
        raise ValueError(
            "Raw child, cache64 parent and R37 base receipt bytes required"
        )
    raw_descriptor = DESCRIPTOR.read_bytes()
    contract = candidate._read(raw_descriptor)
    if contract.get("schema") != "sparkring-feature-extension/v1":
        raise ValueError("Unsupported trusted feature descriptor schema")
    descriptor_hash = hashlib.sha256(raw_descriptor).hexdigest()
    parent_hash = hashlib.sha256(parent_bytes).hexdigest()
    if parent_hash != contract["parent"]["receipt_sha256"]:
        raise ValueError("Feature extension parent receipt is not pinned cache64")

    parent = candidate._read(parent_bytes)
    installed = candidate._read(installed_bytes)
    if not isinstance(parent.get("files"), dict) or "feature_extension" in parent:
        raise ValueError(
            "Feature extension requires an unextended cache64 parent inventory"
        )
    capabilities = contract.get("capabilities")
    if (
        not isinstance(capabilities, list)
        or not capabilities
        or any(not isinstance(value, str) or not value for value in capabilities)
        or len(set(capabilities)) != len(capabilities)
    ):
        raise ValueError("Trusted feature capabilities must be distinct nonempty names")
    assets = contract.get("assets")
    if not isinstance(assets, dict) or not assets:
        raise ValueError("Complete trusted feature asset inventory required")
    additions = {}
    for path, asset in assets.items():
        if (
            not candidate._path(path)
            or not (path.startswith(FEATURE_ROOT) or path in HOOKS)
            or not isinstance(asset, dict)
            or not candidate._hash(asset.get("sha256"))
        ):
            raise ValueError("Invalid trusted feature asset: " + str(path))
        additions[path] = asset["sha256"]
    if not candidate._hash(contract.get("installer_sha256")):
        raise ValueError("Trusted feature installer identity required")
    additions.update(
        {
            PARENT_RECEIPT: parent_hash,
            INSTALLED_DESCRIPTOR: descriptor_hash,
            INSTALLER: contract["installer_sha256"],
        }
    )
    if set(additions).intersection(parent["files"]):
        raise ValueError("Feature additions must not replace inherited files")
    extension = {
        "id": contract["id"],
        "descriptor_sha256": descriptor_hash,
        "parent_image_id": contract["parent"]["image_id"],
        "parent_receipt_sha256": parent_hash,
        "capabilities": sorted(capabilities),
    }
    expected = copy.deepcopy(parent)
    expected["files"].update(additions)
    expected["feature_extension"] = extension
    if installed != expected:
        raise ValueError(
            "Feature receipt differs from the complete inherited and added inventory"
        )

    base, _ = candidate.composition(parent.get("composition_id"))
    files = parent["files"]
    admission = candidate.validate(
        image_id=image_id,
        platform="linux/arm64",
        installed_bytes=installed_bytes,
        verification=verification,
        descriptor=base,
        native_hashes={name: files[name] for name in candidate.NATIVE},
        expected_entrypoint_sha256=files[candidate.ENTRYPOINT],
    )
    expected_feature_verification = {
        "schema": "sparkring-feature-verification/v1",
        "descriptor_sha256": descriptor_hash,
        "capabilities": sorted(capabilities),
        "files_verified": len(expected["files"]),
        "serving_qualified": False,
    }
    if (
        not isinstance(feature_verification, dict)
        or type(feature_verification.get("files_verified")) is not int
        or feature_verification.get("serving_qualified") is not False
        or feature_verification != expected_feature_verification
    ):
        raise ValueError(
            "Feature verification does not bind the trusted unqualified payload"
        )

    # The verified child preserves every parent file. Rebind only the receipt
    # hash and count to check that inventory against the trusted cache64 delta.
    parent_verification = dict(
        verification, receipt_sha256=parent_hash, files_verified=len(files)
    )
    cache_candidate.validate(
        contract["parent"]["image_id"], parent_bytes, base_bytes, parent_verification
    )
    return dict(admission, feature_extension=extension)
