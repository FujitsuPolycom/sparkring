"""Admit source-recorded external CUDA/NCCL images for explicit Qwen rehearsals.

The model profile still owns serving settings. A separate image lock binds the
external software and toolchain receipts; it never changes a published release.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import PurePosixPath
import re

from runtime.common.container_spec import Bind

ENTRYPOINT = ("python3", "/opt/sparkring/toolchain/toolchain.py")
BINDING_TARGET = "/run/sparkring/runtime-binding.json"
PARENT_RECEIPT = "/opt/sparkring/receipts/external-base-installed.json"
TOOLCHAIN_RECEIPT = "/opt/sparkring/toolchain/installed.json"
SUPPORTED = ("qwen38-flash-next-tp2", "qwen38-flash-next-qad-tp4")
FIELDS = {"schema", "name", "profile", "image_id", "image_reference", "parent_receipt_sha256",
          "toolchain_receipt_sha256", "composition_sha256", "transport_profile",
          "transport_manifest_sha256", "status_version"}


def validate(value, profile):
    if not isinstance(value, dict) or set(value) != FIELDS or value.get("schema") != "sparkring-installer-image/v1":
        raise ValueError("Expected a complete sparkring-installer-image/v1 lock")
    if profile not in SUPPORTED or value["profile"] != profile:
        raise ValueError("External toolchain image lock must select the exact supported Qwen profile without SparkCache")
    if not isinstance(value["name"], str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", value["name"]):
        raise ValueError("Image lock name must be a short lowercase identifier")
    if not isinstance(value["image_id"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value["image_id"]):
        raise ValueError("Image lock requires an exact image configuration ID")
    reference = value["image_reference"]
    if reference != value["image_id"] and (not isinstance(reference, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}", reference)):
        raise ValueError("Use a manifest digest reference or the preloaded image configuration ID")
    for field in ("parent_receipt_sha256", "toolchain_receipt_sha256", "composition_sha256", "transport_manifest_sha256"):
        if not isinstance(value[field], str) or not re.fullmatch(r"[0-9a-f]{64}", value[field]):
            raise ValueError("Image lock requires a complete SHA256: " + field)
    if (value["transport_profile"] != "tp2-rocenante-adaptive-prepared" or not isinstance(value["status_version"], str)
            or not re.fullmatch(r"0\.3\.[0-9]+", value["status_version"])):
        raise ValueError("This adapter supports prepared RoCEnante and runtime-status 0.3.x")
    return value


def selection(card, value):
    validate(value, card["profile"])
    return {**card, "profile_release": card["release"], "release": value["name"],
            "image_id": value["image_id"], "image_reference": value["image_reference"],
            "evidence_scope": "Development image selection; the published profile's serving qualification does not transfer."}


def binding_path(lock, row):
    return str(PurePosixPath(row["deployment_root"]) / lock["id"] / "runtime-binding.json")


def adapt(spec, value, *, binding):
    """Reuse the canonical model/network envelope, replacing its runtime binding."""
    environment = dict(spec.environment)
    # Inherit the sealed image's library and Python search paths. Its entrypoint
    # verifies/normalizes them before importing the model framework.
    for key in ("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH", "PATH", "TRITON_PTXAS_PATH"):
        environment.pop(key, None)
    for key, setting in environment.items():
        if setting.startswith("/cache/"):
            environment[key] = setting.replace(spec.image_id[7:19], value["image_id"][7:19])
    cache = environment["XDG_CACHE_HOME"]
    environment.update(
        VLLM_PLUGINS="b12x_loader,sparkring_status",
        SPARKRING_RUNTIME_BINDING=BINDING_TARGET,
        SPARKRING_TRANSPORT_PROFILE=value["transport_profile"],
        SPARKRING_TRANSPORT_MANIFEST_SHA256=value["transport_manifest_sha256"],
        VLLM_QWEN3_8_FLASH_NEXT_HC_TP="0" if value["profile"].endswith("tp4") else "1",
        NCCL_VERSION="2.32.3", NCCL_ROOT="/opt/sparkring/toolchain/nccl",
        NCCL_LIB_DIR="/opt/sparkring/toolchain/nccl/lib", NCCL_INCLUDE_DIR="/opt/sparkring/toolchain/nccl/include",
        VLLM_NCCL_INCLUDE_PATH="/opt/sparkring/toolchain/nccl/include",
        VLLM_NCCL_SO_PATH="/opt/sparkring/toolchain/nccl/lib/libnccl.so.2",
        NCCL_LOCAL_INFERENCE_PATH="/opt/sparkring/toolchain/nccl/lib/libnccl.so.2",
        CUDA_HOME="/usr/local/cuda-13.4", CUDA_PATH="/usr/local/cuda-13.4", CUDA_VERSION="13.4.2",
        TILELANG_CACHE_DIR=cache + "/tilelang", TVM_FFI_CACHE_DIR=cache + "/tvm-ffi",
        FLASHINFER_WORKSPACE_BASE=cache + "/flashinfer",
    )
    if len(spec.command) < 2 or spec.command[1] != "serve":
        raise ValueError("External image adapter requires the canonical Qwen serve command")
    health = ("python3", *spec.health_command[1:]) if spec.health_command else ()
    return replace(spec, image_id=value["image_id"], entrypoint=ENTRYPOINT, command=spec.command[1:],
                   environment=environment, health_command=health,
                   mounts=(*spec.mounts, Bind(binding, BINDING_TARGET, True)),
                   labels={**spec.labels, "io.sparkring.image-lock": value["name"]})


def admit(value, *, run):
    validate(value, value["profile"])
    image = json.loads(run(["docker", "image", "inspect", value["image_id"]]).stdout)[0]
    if (image.get("Id") != value["image_id"] or image.get("Os") != "linux" or image.get("Architecture") != "arm64"
            or image.get("Config", {}).get("Entrypoint") != list(ENTRYPOINT)):
        raise ValueError("External image identity, architecture or verified entrypoint differs")
    isolated = ["docker", "run", "--rm", "--pull", "never", "--runtime", "runc", "--network", "none",
                "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges"]
    receipts = []
    for path, key in ((PARENT_RECEIPT, "parent_receipt_sha256"), (TOOLCHAIN_RECEIPT, "toolchain_receipt_sha256")):
        raw = run([*isolated, "--entrypoint", "/bin/cat", value["image_id"], path], text=False).stdout
        if hashlib.sha256(raw).hexdigest() != value[key]:
            raise ValueError("External image receipt differs: " + path)
        receipts.append(json.loads(raw))
    parent, toolchain = receipts
    capabilities = parent.get("capabilities", {})
    required_features = {"qwen-collectives", "qwen4-prefill"} if value["profile"].endswith("tp4") else set()
    count = "4" if value["profile"].endswith("tp4") else "2"
    hc_mode = {"projection_tp": "0" if count == "4" else "1", "prefill_row_ownership": "shard" if count == "4" else "off"}
    if (parent.get("schema") != "sparkring-external-installed/v1"
            or parent.get("composition_sha256") != value["composition_sha256"]
            or capabilities.get("transport_profile") != value["transport_profile"]
            or capabilities.get("transport_manifest_sha256") != value["transport_manifest_sha256"]
            or capabilities.get("runtime_status", {}).get("version") != value["status_version"]
            or not required_features <= set(capabilities.get("features", []))
            or hc_mode not in capabilities.get("hc_supported_modes", {}).get(count, [])):
        raise ValueError("External software receipt does not satisfy this runtime/profile contract")
    if (toolchain.get("schema") != "sparkring-toolchain-installed/v1" or toolchain.get("variant") != "combined"
            or toolchain.get("parent_receipt_sha256") != value["parent_receipt_sha256"]
            or toolchain.get("nccl_version") != 23203 or "V13.4.92" not in toolchain.get("nvcc", "")):
        raise ValueError("Toolchain receipt is not the selected CUDA 13.4.2 / NCCL 2.32.3 composition")
    # Verify the actual installed tree, not just receipt labels. No GPU or network
    # is exposed; serving still has its own health and generation barriers.
    run([*isolated, "--entrypoint", ENTRYPOINT[0], value["image_id"], ENTRYPOINT[1], "verify"])
    return {"schema": "sparkring-installer-image-observation/v1", "image_id": value["image_id"],
            "composition_sha256": value["composition_sha256"],
            "parent_receipt_sha256": value["parent_receipt_sha256"],
            "toolchain_receipt_sha256": value["toolchain_receipt_sha256"], "serving_qualified": False}
