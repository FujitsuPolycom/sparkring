"""Validate local image witnesses against source and installed-package inventories."""
import hashlib
import json
import re
from native_files import expected_record


def file_map_hash(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_receipt(document, lock):
    """Require complete package/native witnesses; this is not a GPU qualification."""
    if not isinstance(document, dict) or not isinstance(lock, dict):
        raise ValueError("Receipt and source lock must be objects")
    if document.get("schema") != "sparkring-source-image-receipt/v1":
        raise ValueError("Unsupported local source-image receipt")
    image = document.get("image_id")
    if not isinstance(image, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Local image requires its exact config ID")
    if document.get("image_reference") != image or document.get("platform") != "linux/arm64":
        raise ValueError("Local receipt is not a registry publication record")
    inside = document.get("inside_image", {})
    if not isinstance(inside, dict):
        raise ValueError("Inside-image witness must be an object")
    for field in ("packages", "inherited_runtime"):
        if not isinstance(inside.get(field), dict):
            raise ValueError(f"Inside-image witness requires object: {field}")
    if (document.get("checks_passed") is not True or inside.get("checks_passed") is not True
            or inside.get("cuda_initialized") is not False or inside.get("model_loaded") is not False):
        raise ValueError("CPU image verification did not complete")
    selected_native_mode = document.get("native_mode", "compile")
    if (selected_native_mode not in ("compile", "pinned")
            or inside.get("native_mode", "compile") != selected_native_mode):
        raise ValueError("Receipt native modes disagree")
    if selected_native_mode == "pinned":
        if inside.get("native_files") != expected_record(lock):
            raise ValueError("Receipt native artifact witness differs")
    elif "native_files" in inside:
        raise ValueError("Compile receipt cannot claim pinned native artifacts")
    digest = document.get("source_lock_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or inside.get("source_lock_sha256") != digest:
        raise ValueError("Receipt source-lock witnesses disagree")
    if document.get("profile") not in lock["profiles"]:
        raise ValueError("Receipt selects an unsupported runtime profile")
    profile = lock["profiles"][document["profile"]]
    if profile.get("sparkcache"):
        for name in ("snapshot", "placement"):
            field = name + "_sha256"
            if inside.get(field) != lock["runtime"][field]:
                raise ValueError(f"SparkCache native library witness differs: {field}")
    transport = profile.get("transport_profile")
    if transport is not None:
        expected_transport = {"manifest_sha256": profile["transport_manifest_sha256"],
                              "files_sha256": profile["transport_files_sha256"],
                              "files": profile["transport_file_count"], "package": "b12x.comm.roce"}
        if inside.get("transport_profiles", {}).get(transport) != expected_transport:
            raise ValueError("Selected transport witness differs from source lock")
    for name, source in lock["sources"].items():
        witness = inside.get("packages", {}).get(name, {})
        if not isinstance(witness, dict):
            raise ValueError(f"Package witness must be an object: {name}")
        if (witness.get("revision") != source["revision"]
                or witness.get("file_map_sha256") != source["installed_file_map_sha256"]
                or witness.get("files") != source["installed_file_count"]):
            raise ValueError(f"Complete installed {name} witness differs from source lock")
    for field in ("bundle_manifest_sha256", "transport_sha256", "marker_source_sha256",
                  "marker_binary_sha256", "nccl_sha256"):
        if inside.get(field) != lock["runtime"][field]:
            raise ValueError(f"Runtime witness differs: {field}")
    if inside.get("retained_vllm_native_sha256") != lock["runtime"]["retained_vllm_native_sha256"]:
        raise ValueError("Retained native vLLM inventory differs")
    if inside.get("readiness_warmup") != lock["runtime"]["readiness_warmup"]:
        raise ValueError("Readiness warmup witness differs")
    expected = lock["runtime"]["expected_distributions"]
    if any(inside.get("inherited_runtime", {}).get(k) != v for k, v in expected.items()):
        raise ValueError("Inherited native runtime dependency versions differ")
    return document
