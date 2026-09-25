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
                verification={**copy.deepcopy(verification), "checked_files": copy.deepcopy(view["files"])}, installed=view,
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


def _native_profiles(compatibility):
    profiles = {name: copy.deepcopy(compatibility["profiles"][name]) for name in sorted(PROFILES)}
    for profile in profiles.values():
        for field in ("capability_file", "required_capabilities", "reference_kv_cache_memory_bytes", "lifecycle"):
            profile.pop(field, None)
        profile.update(load_format="b12x", plugins="b12x_loader", allocation_policy="managed-in-loader",
                       draft_load_config={"load_format": "b12x", "model_loader_extra_config": {}},
                       lifecycle="explicit-create-and-start", memory_guard_required=False)
        profile["serving"] = (copy.deepcopy(compatibility["profiles"]["tp2-dcp1-sparkcache"]["serving"])
                              if profile["node_count"] == 2 else
                              dict(max_num_seqs=16, max_num_batched_tokens=8192,
                                   prefill_schedule_interval=2, limit_mm_per_prompt={"image": 4, "video": 1}))
    return profiles


def complete_planning_contract(document, *, root=ROOT):
    """Expose supported cache-off settings without editing frozen release files.

    Older published planning snapshots list only cache-on profiles, while the
    authenticated native adapter supports both. Fill missing settings from the
    same adapter definitions; this is planning, not installed-image admission.
    """
    if document.get("schema") != "sparkring-native-glm-profile-contract/v1":
        raise ValueError("Expected a native GLM planning contract")
    compatibility = json.loads((root / "runtime/sparkring/jovian-r33/profiles/profile-contract.json").read_bytes())
    result = copy.deepcopy(document)
    for name, profile in _native_profiles(compatibility).items():
        result["profiles"].setdefault(name, profile)
    return result


def profile_contract(installed):
    """Resolve DCP1 settings while replacing all retained image-specific bindings.

    Compatibility templates supply rank settings, not image qualification. The
    native inventory supplies cache-library hashes and the active source lease.
    """
    if installed.get("schema") != "sparkring-glm-native-profile-view/v1":
        raise ValueError("Native GLM settings require an authenticated profile view")
    compatibility = json.loads((ROOT / "runtime/sparkring/jovian-r33/profiles/profile-contract.json").read_bytes())
    profiles = _native_profiles(compatibility)
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


def environment_for_profile(installed, profile):
    """Apply native runtime and bounded cache settings after compatibility templates."""
    contract = profile_contract(installed)
    if profile not in contract["profiles"]:
        raise ValueError("Native GLM profile is not configured")
    values = dict(contract["common_environment"])
    values.update(SPARKRING_RUNTIME_RELEASE="native", SPARKCACHE_MAX_BYTES="8589934592",
                  SPARKCACHE_LOW_WATERMARK_BYTES="6442450944", SPARKCACHE_MAX_SPAN_TOKENS="65536",
                  SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES="536870912", SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT="2",
                  SPARKCACHE_CUDA_ARENA_BYTES="67108864", SPARKCACHE_LOAD_THREADS="2",
                  SPARKCACHE_MAX_PENDING_RESTORES="2", SPARKCACHE_CUDA_RESTORE_IO_WORKERS="2",
                  SPARKCACHE_CLEAR_ONCE="", B12X_COMPILE_CPU_AFFINITY="12-19")
    if contract["profiles"][profile]["node_count"] == 4:
        values["SPARK_TP4_GRAPH_DIRECT_DOORBELL"] = "1"
    return values


def adapt_launcher(text, installed):
    """Prevent legacy shell launchers from bypassing native structured admission."""
    profile_contract(installed)
    return ("#!/usr/bin/env bash\nset -euo pipefail\n"
            "echo 'Native GLM deployment requires runtime.common.glm_launch and its verified image receipt; use the structured container plan.' >&2\n"
            "exit 64\n")


def validate_profile_capabilities(document, profile):
    checked = validate_receipt(document)
    if profile not in profile_contract(checked["installed"])["profiles"]:
        raise ValueError("Native GLM configuration supports TP2/DCP1 and TP4/DCP1")


def cache_namespace(image_id, profile):
    if profile not in PROFILES or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("Native GLM cache namespace requires an exact image and supported profile")
    return f"sparkring-native-glm-{image_id[7:19]}-{profile}"


def adapt_arguments(arguments, contract, profile, variant="nvfp4-spark"):
    """Translate the retained GLM argument surface to the native engine contract.

    Resource settings come from the selected profile. This translation does not
    lower context, KV allocation, batch size or concurrency to gain admission.
    MXFP8 QAD draft experts require a backend distinct from the NVFP4 target.
    """
    if profile not in PROFILES or variant not in ("nvfp4-spark", "nvfp4-qad"):
        raise ValueError("Native GLM arguments require a configured DCP1 profile and LIL checkpoint")
    selected = contract["profiles"][profile]
    nodes = selected["node_count"]
    result = list(arguments)

    def remove(name, *, value=True):
        if result.count(name) > 1:
            raise ValueError("Serving option occurs more than once: " + name)
        if name in result:
            index = result.index(name)
            if value and (index + 1 == len(result) or result[index + 1].startswith("--")):
                raise ValueError("Serving option has no value: " + name)
            del result[index:index + (2 if value else 1)]

    def set_option(name, value):
        remove(name)
        result.extend((name, str(value)))

    for flag in ("--model-loader-extra-config", "--gdn-decode-kernel"):
        remove(flag)
    if nodes == 2:
        for flag in ("--mamba-block-size", "--max-cudagraph-capture-size", "--cp-kv-cache-interleave-size"):
            remove(flag)
        for flag in ("--cudagraph-metrics", "--async-scheduling"):
            remove(flag, value=False)
    for flag, value in (
        ("--load-format", "b12x"), ("--quantization", "modelopt_mixed"),
        ("--max-model-len", contract["model"]["max_model_len"]),
        ("--kv-cache-memory-bytes", selected["kv_cache_memory_bytes"]),
        ("--max-parallel-prefills", 1),
        ("--recurrent-checkpoint-policy", "aligned"), ("--prefix-cache-retention-interval", 0),
    ):
        set_option(flag, value)
    serving = selected["serving"]
    for flag, key in (("--max-num-seqs", "max_num_seqs"), ("--max-num-batched-tokens", "max_num_batched_tokens"),
                      ("--prefill-schedule-interval", "prefill_schedule_interval")):
        set_option(flag, serving[key])
    set_option("--limit-mm-per-prompt", json.dumps(serving["limit_mm_per_prompt"], separators=(",", ":")))
    speculation = dict(method="mtp", num_speculative_tokens=3, attention_backend="B12X")
    if nodes == 2:
        speculation['draft_load_config'] = copy.deepcopy(selected['draft_load_config'])
        for flag, value in (("--mm-processor-cache-gb", 0), ("--mm-encoder-tp-mode", "data"), ("--generation-config", "auto")):
            set_option(flag, value)
    else:
        # These explicit TP4 draft controls match the measured mesh recipe.
        # An omitted draft loader inherits the target's B12X loader.
        speculation.update(draft_tensor_parallel_size=4, kv_cache_dtype="auto",
                           draft_sample_method="probabilistic", rejection_sample_method="standard")
    if nodes == 2 or variant == "nvfp4-qad":
        speculation["moe_backend"] = "humming"
    set_option("--speculative-config", json.dumps(speculation, separators=(",", ":")))
    captures = selected["cudagraph_capture_sizes"]
    graph = dict(cudagraph_mode="FULL_AND_PIECEWISE", cudagraph_capture_sizes=captures)
    if nodes == 2:
        graph.update(mode=0, max_cudagraph_capture_size=max(captures))
    else:
        graph.update(custom_ops=["all"], pass_config={"fuse_allreduce_rms": False})
        set_option("--max-cudagraph-capture-size", max(captures))
        set_option("--mamba-block-size", 256)
        set_option("--cp-kv-cache-interleave-size", 1)
        for flag in ("--async-scheduling", "--cudagraph-metrics"):
            remove(flag, value=False)
            result.append(flag)
    set_option("--compilation-config", json.dumps(graph, separators=(",", ":")))
    if nodes == 4:
        set_option("--media-io-kwargs", '{"video":{"num_frames":16}}')
    return result


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
