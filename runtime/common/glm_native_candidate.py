"""Authenticate GLM native-image receipts without claiming serving qualification.

The raw installed receipt is bound to a registered publication. The compact
profile view exposes only owned transport/cache files and active lease paths;
it cannot substitute caller-selected hashes for the authenticated inventory.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime.common import native_candidate as native  # noqa: E402

SCHEMA = "sparkring-glm-native-image-receipt/v1"
ENTRYPOINT = native.ENTRYPOINT
MANIFEST = "/opt/sparkring/sircl/python/sparkring-overlay-manifest.json"
SIRCL = "/opt/sparkring/sircl/libspark_transport_capi.so"
NCCL = "/opt/local-inference/nccl/lib/libnccl.so.2.31.2"
MODEL = "/opt/venv/lib/python3.12/site-packages/vllm/models/glm5next/nvidia/model.py"
FIELDS = frozenset(("schema", "release", "image_id", "image_reference", "platform",
                    "raw_installed", "inspection", "verification", "installed",
                    "bundle_manifest_sha256", "serving_qualified"))
PROFILES = frozenset(("tp2-dcp1", "tp2-dcp1-sparkcache", "tp4-dcp1", "tp4-dcp1-sparkcache"))
DISABLED = {"SPARKRING_FEATURES": "", "VLLM_QWEN3_8_HC_PREFILL_MODE": "off",
            "VLLM_QWEN3_8_PREFILL_COALESCE": "0"}


def _view(installed, release):
    files = installed.get("files", {})
    if not isinstance(files, dict):
        raise ValueError("Native image inventory must map owned paths to hashes")
    contracts = installed.get("active_contracts", [])
    if (not isinstance(contracts, list) or len(contracts) != 1
            or not isinstance(contracts[0], str)
            or not re.fullmatch(r"/opt/sparkring/contracts/vllm-connector-jobs-source-[0-9a-f]{16}\.json", contracts[0])):
        raise ValueError("GLM native image requires one source-bound connector lease")
    required = (MODEL, MANIFEST, SIRCL, NCCL, contracts[0],
                "/opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so",
                "/opt/sparkring/sparkcache/lib/libspark_cache_placement.so")
    if any(not isinstance(files.get(path), str)
           or not re.fullmatch(r"[0-9a-f]{64}", files[path]) for path in required):
        raise ValueError("GLM model, transport or cache files are missing from the native inventory")
    selected = {path: value for path, value in files.items()
                if path.startswith(("/opt/sparkring/", "/opt/local-inference/nccl/"))}
    return dict(schema="sparkring-glm-native-profile-view/v1", release=release,
                compiler=copy.deepcopy(installed["compiler"]),
                active_contracts=list(contracts), files=selected)


def make_receipt(*, release, image_id, inspection, installed_bytes, verification):
    publication = native.publication(release, image_id=image_id)
    native.validate(publication, image_id, inspection, installed_bytes, verification)
    installed = json.loads(installed_bytes)
    view = _view(installed, release)
    info = dict(Id=inspection["Id"], Os=inspection["Os"], Architecture=inspection["Architecture"],
                Config={"Entrypoint": inspection["Config"]["Entrypoint"]})
    return dict(schema=SCHEMA, release=release, image_id=image_id,
                image_reference=publication["image_reference"], platform="linux/arm64",
                raw_installed=base64.b64encode(installed_bytes).decode(), inspection=info,
                verification=copy.deepcopy(verification), installed=view,
                bundle_manifest_sha256=view["files"][MANIFEST], serving_qualified=False)


def validate_receipt(document):
    if (not isinstance(document, dict) or set(document) != FIELDS
            or document.get("schema") != SCHEMA or document.get("serving_qualified") is not False):
        raise ValueError("Select a native GLM inventory receipt; serving qualification is separate")
    try:
        raw = base64.b64decode(document["raw_installed"], validate=True)
    except (TypeError, ValueError) as error:
        raise ValueError("Native GLM receipt requires the original installed bytes") from error
    expected = make_receipt(release=document["release"], image_id=document["image_id"],
                            inspection=document["inspection"], installed_bytes=raw,
                            verification=document["verification"])
    if document != expected:
        raise ValueError("Native GLM profile view differs from its authenticated inventory")
    return expected


def observe(image_id, release, *, run=subprocess.run):
    values = native.observe_image(image_id, release, run=run)
    return make_receipt(release=release, image_id=image_id, inspection=values["inspection"],
                        installed_bytes=values["installed_bytes"], verification=values["verification"])


def verify_local_image(document, *, run=subprocess.run):
    checked = validate_receipt(document)
    if observe(checked["image_id"], checked["release"], run=run) != checked:
        raise ValueError("Native GLM image observations differ from the saved receipt")


def profile_contract(installed):
    """Resolve DCP1 settings while replacing all retained image-specific bindings.

    Compatibility templates supply rank settings, not image qualification. The
    native inventory supplies cache-library hashes and the active source lease.
    """
    if installed.get("schema") != "sparkring-glm-native-profile-view/v1":
        raise ValueError("Native GLM settings require an authenticated profile view")
    compatibility = json.loads((ROOT / "runtime/sparkring/jovian-r33/profiles/profile-contract.json").read_bytes())
    profiles = {name: copy.deepcopy(compatibility["profiles"][name]) for name in sorted(PROFILES)}
    for name, profile in profiles.items():
        for field in ("capability_file", "required_capabilities", "reference_kv_cache_memory_bytes", "lifecycle"):
            profile.pop(field, None)
        profile.update(load_format="b12x", plugins="b12x_loader", allocation_policy="managed-in-loader",
                       draft_load_config={"load_format": "b12x", "model_loader_extra_config": {}},
                       lifecycle="explicit-create-and-start", memory_guard_required=False)
        if profile["node_count"] == 2:
            profile["serving"] = copy.deepcopy(compatibility["profiles"]["tp2-dcp1-sparkcache"]["serving"])
    files = installed["files"]
    cache = {key: compatibility["sparkcache_native"][key]
             for key in ("placement_path", "snapshot_path", "vllm_root")}
    cache.update(placement_sha256=files[cache["placement_path"]],
                 snapshot_sha256=files[cache["snapshot_path"]],
                 lease_contract=installed["active_contracts"][0])
    environment = dict(compatibility["common_environment"], **DISABLED)
    environment.update(LOAD_FORMAT="b12x", VLLM_PLUGINS="b12x_loader",
                       B12X_NVFP4_DYNAMIC_MATERIALIZED="0", VLLM_SPARK_SHARED_CAPTURE_STREAM="1",
                       VLLM_USE_RUST_FRONTEND="0")
    return dict(schema="sparkring-native-glm-profile-contract/v1", profiles=profiles,
                model=dict(max_model_len=1048576, loader={"load_format": "b12x"},
                           speculation={"method": "mtp", "num_speculative_tokens": 3, "attention_backend": "B12X"}),
                common_environment=environment, sparkcache_native=cache,
                source_trees=copy.deepcopy(installed["compiler"]["source_trees"]),
                qualification="Implemented configuration admission; exact-profile hardware evidence is separate.")


def contract_for_receipt(document):
    return profile_contract(validate_receipt(document)["installed"])


def validate_profile_capabilities(document, profile):
    checked = validate_receipt(document)
    if profile not in profile_contract(checked["installed"])["profiles"]:
        raise ValueError("Native GLM configuration supports TP2/DCP1 and TP4/DCP1")


def cache_namespace(image_id, profile):
    if profile not in PROFILES or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("Native GLM cache namespace requires an exact image and supported profile")
    return f"sparkring-native-glm-{image_id[7:19]}-{profile}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = observe(args.image_id, args.release)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(image_id=args.image_id, output=str(args.output),
                          installed_receipt_sha256=hashlib.sha256(base64.b64decode(record["raw_installed"])).hexdigest(),
                          serving_qualified=False)))


if __name__ == "__main__":
    main()
