"""Explicit local GLM admission for the reviewed source extension, without KV restore.

The host receipt retains every raw ancestry receipt. Source admission verifies
the entire installed inventory; it does not transfer model-serving qualification.
"""
import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import subprocess

from runtime.common import candidate, feature_candidate, source_candidate

SCHEMA = "sparkring-glm-source-image-receipt/v1"
ENTRYPOINT = source_candidate.ENTRYPOINT
PROFILES = frozenset(("tp2-dcp1", "tp4-dcp1", "tp4-dcp4"))
RAW_FIELDS = ("installed_bytes", "parent_bytes", "cache_parent_bytes", "base_bytes")
DISABLED = {"SPARKRING_FEATURES": "", "VLLM_QWEN3_8_HC_PREFILL_MODE": "off",
            "VLLM_QWEN3_8_PREFILL_COALESCE": "0"}


def make_receipt(*, local_source_extension, **observations):
    source_candidate.descriptor(local_source_extension)
    document = {"schema": SCHEMA, "local_source_extension": local_source_extension,
                "image_id": observations["image_id"], "image_reference": observations["image_id"],
                "platform": "linux/arm64", "verification": observations["verification"],
                "installed": candidate._read(observations["installed_bytes"]),
                "raw_receipts": {key: base64.b64encode(observations[key]).decode() for key in RAW_FIELDS}}
    return validate_receipt(document)


def validate_receipt(document):
    if (not isinstance(document, dict) or document.get("schema") != SCHEMA
            or document.get("local_source_extension") != source_candidate.IDENTITY
            or document.get("platform") != "linux/arm64"
            or document.get("image_reference") != document.get("image_id")):
        raise ValueError("GLM source admission requires an explicit registered local image selection")
    try:
        raw = {key: base64.b64decode(document["raw_receipts"][key], validate=True) for key in RAW_FIELDS}
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError("GLM source receipt requires every raw ancestry receipt") from error
    installed = candidate._read(raw["installed_bytes"])
    if installed != document.get("installed"):
        raise ValueError("GLM source receipt parsed inventory differs from its raw bytes")
    verification = dict(document.get("verification", {}))
    verification.pop("checked_files", None)
    admission = source_candidate.validate(image_id=document.get("image_id"), **raw,
        verification=verification, identity=document["local_source_extension"])
    files = installed["files"]
    manifest = files.get("/opt/sparkring/sircl/python/sparkring-overlay-manifest.json")
    if not candidate._hash(manifest):
        raise ValueError("GLM source image lacks its inherited transport manifest")
    return dict(document, admission=admission, bundle_manifest_sha256=manifest,
                verification=dict(verification, checked_files=copy.deepcopy(files)))


def profile_contract(installed):
    extension = installed.get("source_extension", {})
    if (extension.get("id") != source_candidate.IDENTITY or extension.get("descriptor_sha256")
            != hashlib.sha256(source_candidate.DESCRIPTOR.read_bytes()).hexdigest()):
        raise ValueError("GLM source contract requires its reviewed source descriptor")
    contract = candidate.profile_contract(installed)
    contract["profiles"] = {name: value for name, value in contract["profiles"].items() if name in PROFILES}
    contract["common_environment"].update(DISABLED)
    contract["sparkcache_native"]["lease_contract"] = ""
    contract["source_extension"] = copy.deepcopy(extension)
    contract["candidate_qualification"] = "Local cache-disabled GLM admission; serving is unqualified."
    return contract


def validate_profile_capabilities(document, profile):
    checked = validate_receipt(document)
    if profile not in PROFILES:
        raise ValueError("Local GLM source admission supports only cache-disabled TP2/DCP1 and TP4/DCP1/DCP4; SparkCache requires a separate lease audit")
    if profile not in profile_contract(checked["installed"])["profiles"]:
        raise ValueError("GLM source profile is absent from its inherited contract")


def observe(image_id, *, local_source_extension, run=subprocess.run):
    source_candidate.image_reference(local_source_extension, image_id)
    info = json.loads(run(["docker", "image", "inspect", image_id], check=True,
                          capture_output=True, text=True).stdout)[0]
    if (info.get("Id") != image_id or info.get("Os") != "linux" or info.get("Architecture") != "arm64"
            or info.get("Config", {}).get("Entrypoint") != ["/opt/venv/bin/python", ENTRYPOINT]):
        raise ValueError("GLM source image identity, platform or entrypoint differs")
    base = candidate._read((candidate.ROOT / "runtime/images/compositions/lil-r37-glm-spark/publication.json").read_bytes())
    observations = {"image_id": image_id}
    paths = ((image_id, "/opt/sparkring/receipts/candidate-installed.json"),
             (image_id, source_candidate.PARENT_RECEIPT), (image_id, feature_candidate.PARENT_RECEIPT),
             (base["image_id"], "/opt/sparkring/receipts/candidate-installed.json"))
    for key, (image, path) in zip(RAW_FIELDS, paths):
        observations[key] = run(["docker", "run", "--rm", "--pull", "never", "--network", "none",
                                "--entrypoint", "/bin/cat", image, path], check=True,
                               capture_output=True, text=False).stdout
    observations["verification"] = json.loads(run([
        "docker", "run", "--rm", "--pull", "never", "--network", "none", image_id, "verify"],
        check=True, capture_output=True, text=True).stdout)
    return make_receipt(local_source_extension=local_source_extension, **observations)


def verify_local_image(document, *, run=subprocess.run):
    checked = validate_receipt(document)
    if observe(checked["image_id"], local_source_extension=checked["local_source_extension"], run=run) != checked:
        raise ValueError("Local GLM source image no longer matches its recorded admission")


def adapt_launcher(text, installed):
    """Emit a non-executable compatibility notice; trials use checked glm_launch.

    The historical shell path verifies base-R37 receipts. Rewriting that gate
    would weaken the source-chain requirement, so it cannot run source trials.
    """
    profile_contract(installed)
    return ("#!/usr/bin/env bash\n"
            "echo 'Local GLM source trials require runtime.common.glm_launch and their explicit source receipt; this compatibility launcher cannot start them.' >&2\n"
            "exit 2\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-source-extension", required=True, choices=(source_candidate.IDENTITY,))
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("GLM admission output already exists")
    document = observe(args.image_id, local_source_extension=args.local_source_extension)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
