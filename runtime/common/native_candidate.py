"""Admit published native shared images without legacy R37 entrypoint assumptions."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = "/opt/sparkring/bin/native-image.py"
RECEIPT = "/opt/sparkring/receipts/native-installed.json"


def publication(release, *, image_id=None):
    if not isinstance(release, str) or not re.fullmatch(r"shared-[A-Za-z0-9][A-Za-z0-9.-]*", release):
        raise ValueError("Select a registered shared release identifier")
    path = ROOT / "runtime/releases" / release / "publication.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    image = record.get("image_id")
    if (record.get("schema") != "sparkring-shared-image-publication/v1"
            or record.get("release") != release
            or record.get("platform") != "linux/arm64"
            or not isinstance(image, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image)
            or record.get("anonymous_config_verified") is not True
            or record.get("anonymous_pull_completed") is not True
            or not re.fullmatch(r"ghcr\.io/fujitsupolycom/sparkring@sha256:[0-9a-f]{64}", record.get("image_reference", ""))
            or not re.fullmatch(r"[0-9a-f]{64}", record.get("installed_receipt_sha256", ""))):
        raise ValueError("Shared publication identity or anonymous verification is incomplete")
    if image_id is not None and image_id != image:
        raise ValueError("Image differs from selected shared release")
    if (record.get("image_config_id", image) != image
            or record["image_reference"] != "ghcr.io/fujitsupolycom/sparkring@" + record.get("registry_manifest_digest", "")):
        raise ValueError("Published manifest/configuration identities disagree")
    transport = record.get("transport", {})
    if (transport.get("profile") != "tp2-rocenante-adaptive-prepared"
            or not re.fullmatch(r"[0-9a-f]{64}", transport.get("manifest_sha256", ""))):
        raise ValueError("Native release lacks its prepared transport identity")
    return record


def validate_identity(record, image, inspection):
    if (image != record["image_id"] or inspection.get("Id") != image
            or inspection.get("Os") != "linux" or inspection.get("Architecture") != "arm64"
            or inspection.get("Config", {}).get("Entrypoint") != ["/opt/venv/bin/python", ENTRYPOINT]):
        raise ValueError("Native image identity, platform or entrypoint differs")


def validate(record, image, inspection, raw, verification):
    validate_identity(record, image, inspection)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != record["installed_receipt_sha256"]:
        raise ValueError("Installed native receipt differs from publication")
    installed = json.loads(raw)
    if (verification.get("schema") != "sparkring-native-verification/v1"
            or verification.get("receipt_sha256") != digest
            or verification.get("source_trees") != record["source_trees"]
            or installed.get("compiler", {}).get("source_trees") != record["source_trees"]
            or verification.get("input_sha256") != record["input_sha256"]
            or verification.get("files_verified") != len(installed.get("files", {}))
            or not installed.get("files")
            or set(verification.get("features", [])) != set(record["features"])):
        raise ValueError("Native installed/source verification differs from publication")
    return dict(schema="sparkring-native-host-verification/v1", image_id=image,
                image_reference=record["image_reference"], platform="linux/arm64",
                release=record["release"], installed_receipt_sha256=digest,
                verification=verification)


def observe_image(image, release, *, run=subprocess.run):
    """Return authenticated native inventory observations for profile adapters."""
    record = publication(release, image_id=image)
    inspection = json.loads(run(["docker", "image", "inspect", image], check=True,
                                capture_output=True, text=True).stdout)[0]
    validate_identity(record, image, inspection)
    common = ["docker", "run", "--rm", "--pull", "never", "--runtime", "runc", "--network", "none"]
    raw = run([*common, "--entrypoint", "/bin/cat", image, RECEIPT], check=True, capture_output=True).stdout
    verification = json.loads(run([*common, image, "verify"], check=True,
                                  capture_output=True, text=True).stdout)
    return dict(publication=record, inspection=inspection, installed_bytes=raw,
                verification=verification,
                host_verification=validate(record, image, inspection, raw, verification))


def verify_image(image, release, *, run=subprocess.run):
    return observe_image(image, release, run=run)["host_verification"]
