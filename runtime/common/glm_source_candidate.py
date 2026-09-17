"""Explicit local GLM admission, with a separate opt-in for SparkCache trials.

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
CACHE_PROFILES = frozenset(name + "-sparkcache" for name in PROFILES)
RAW_FIELDS = ("installed_bytes", "parent_bytes", "cache_parent_bytes", "base_bytes")
DISABLED = {"SPARKRING_FEATURES": "", "VLLM_QWEN3_8_HC_PREFILL_MODE": "off",
            "VLLM_QWEN3_8_PREFILL_COALESCE": "0"}


def make_receipt(*, local_source_extension, allow_sparkcache_trial=False, **observations):
    source_candidate.descriptor(local_source_extension)
    document = {"schema": SCHEMA, "local_source_extension": local_source_extension,
                "allow_sparkcache_trial": allow_sparkcache_trial,
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
    if type(document.get("allow_sparkcache_trial", False)) is not bool:
        raise ValueError("allow_sparkcache_trial must be a boolean")
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
    return dict(document, allow_sparkcache_trial=document.get("allow_sparkcache_trial", False),
                admission=admission, bundle_manifest_sha256=manifest,
                verification=dict(verification, checked_files=copy.deepcopy(files)))


def profile_contract(installed, *, allow_sparkcache_trial=False):
    if type(allow_sparkcache_trial) is not bool:
        raise ValueError("allow_sparkcache_trial must be a boolean")
    extension = installed.get("source_extension", {})
    if (extension.get("id") != source_candidate.IDENTITY or extension.get("descriptor_sha256")
            != hashlib.sha256(source_candidate.DESCRIPTOR.read_bytes()).hexdigest()):
        raise ValueError("GLM source contract requires its reviewed source descriptor")
    contract = candidate.profile_contract(installed)
    allowed = PROFILES | CACHE_PROFILES if allow_sparkcache_trial else PROFILES
    contract["profiles"] = {name: value for name, value in contract["profiles"].items() if name in allowed}
    contract["common_environment"].update(DISABLED)
    contract["sparkcache_native"]["lease_contract"] = ""
    if allow_sparkcache_trial:
        descriptor = source_candidate.descriptor()
        lease = source_candidate.LEASE_CONTRACT
        expected = descriptor["integration_contracts"].get(lease, {}).get("sha256")
        if not candidate._hash(expected) or installed["files"].get(lease) != expected:
            raise ValueError("GLM cache trial requires the packaged source-matched lease")
        native = contract["sparkcache_native"]
        native["lease_contract"] = lease
        for stem in ("placement", "snapshot"):
            observed = installed["files"].get(native[stem + "_path"])
            if not candidate._hash(observed):
                raise ValueError("GLM cache trial lacks verified native library hashes")
            native[stem + "_sha256"] = observed
        native["snapshot_extension"] = copy.deepcopy(installed["cache_extension"])
    contract["source_extension"] = copy.deepcopy(extension)
    contract["candidate_qualification"] = "Local GLM admission; optional isolated SparkCache trial. Serving and restore are unqualified."
    return contract


def contract_for_receipt(document):
    checked = validate_receipt(document)
    return profile_contract(checked["installed"], allow_sparkcache_trial=checked["allow_sparkcache_trial"])


def cache_namespace(image_id, profile):
    """Source-specific image/topology stem; callers append the pinned model revision."""
    if profile not in CACHE_PROFILES or not isinstance(image_id, str):
        raise ValueError("Select a registered local GLM cache trial")
    source_candidate.image_reference(source_candidate.IDENTITY, image_id)
    return f"sparkring-glm-source-{image_id[7:19]}-{profile}"


def validate_profile_capabilities(document, profile):
    checked = validate_receipt(document)
    if profile not in PROFILES | CACHE_PROFILES:
        raise ValueError("Local GLM source admission supports only TP2/DCP1 and TP4/DCP1/DCP4")
    if profile in CACHE_PROFILES and not checked["allow_sparkcache_trial"]:
        raise ValueError("GLM source receipt is cache-disabled; create a separate receipt with --allow-sparkcache-trial")
    if profile not in contract_for_receipt(checked)["profiles"]:
        raise ValueError("GLM source profile is absent from its inherited contract")


def observe(image_id, *, local_source_extension, allow_sparkcache_trial=False, run=subprocess.run):
    if type(allow_sparkcache_trial) is not bool:
        raise ValueError("allow_sparkcache_trial must be a boolean")
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
    return make_receipt(local_source_extension=local_source_extension,
                        allow_sparkcache_trial=allow_sparkcache_trial, **observations)


def verify_local_image(document, *, run=subprocess.run):
    checked = validate_receipt(document)
    if observe(checked["image_id"], local_source_extension=checked["local_source_extension"],
               allow_sparkcache_trial=checked["allow_sparkcache_trial"], run=run) != checked:
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
    parser.add_argument("--allow-sparkcache-trial", action="store_true",
                        help="admit isolated GLM SparkCache trials; does not qualify serving or restore")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("GLM admission output already exists")
    document = observe(args.image_id, local_source_extension=args.local_source_extension,
                       allow_sparkcache_trial=args.allow_sparkcache_trial)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
