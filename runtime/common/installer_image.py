"""Admit source-recorded external CUDA/NCCL images for installer deployments.

The model profile owns serving settings. An image lock binds the external
software and toolchain receipts; it never changes a published release.

Two lock schemas exist:

- ``sparkring-installer-image/v1`` names exactly one Qwen profile. Saved v1
  deployments keep validating so their recorded rollback remains usable.
- ``sparkring-installer-image/v2`` names the set of installer profiles that
  run on one shared image. The installer applies the release's v2 lock to every
  profile it lists; see ``default_lock``.

Admission is model-aware: Qwen profiles additionally require the image's Qwen
collective features and hybrid-attention modes, while GLM and MiMo profiles
rely on the verified receipts and the installed-tree check alone.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from runtime.common.container_spec import Bind

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = ("python3", "/opt/sparkring/toolchain/toolchain.py")
BINDING_TARGET = "/run/sparkring/runtime-binding.json"
PARENT_RECEIPT = "/opt/sparkring/receipts/external-base-installed.json"
TOOLCHAIN_RECEIPT = "/opt/sparkring/toolchain/installed.json"
SCHEMA_V1 = "sparkring-installer-image/v1"
SCHEMA = "sparkring-installer-image/v2"
# The release whose image every installer profile uses unless an operator
# supplies an explicit development lock.
DEFAULT_LOCK = ROOT / "runtime/releases/dev-20260924-cuda1342-nccl2323-status031/installer-image.json"
QWEN = ("qwen38-flash-next-tp2", "qwen38-flash-next-qad-tp4")
SUPPORTED = (*QWEN, "glm53-flash-nvfp4-spark-tp2", "glm53-flash-nvfp4-spark-tp4",
             "mimo-v26-flash-rl-tp2", "mimo-v26-flash-rl-tp4")
PLUGINS = ("b12x_loader", "sparkring_status")
COMMON = {"schema", "name", "image_id", "image_reference", "parent_receipt_sha256",
          "toolchain_receipt_sha256", "composition_sha256", "transport_profile",
          "transport_manifest_sha256", "status_version"}
# v2 records the unpacked image size and the registry download size so storage
# checks can reserve what this image needs instead of a generic allowance.
FIELDS = {SCHEMA_V1: COMMON | {"profile"}, SCHEMA: COMMON | {"profiles", "image_bytes", "download_bytes"}}


def profiles_of(value):
    return (value["profile"],) if value["schema"] == SCHEMA_V1 else tuple(value["profiles"])


def validate(value, profile):
    if not isinstance(value, dict) or value.get("schema") not in FIELDS or set(value) != FIELDS[value["schema"]]:
        raise ValueError("Expected a complete sparkring-installer-image/v1 or /v2 lock")
    if value["schema"] == SCHEMA_V1:
        if profile not in QWEN or value["profile"] != profile:
            raise ValueError("A v1 image lock must select the exact supported Qwen profile without SparkCache")
    else:
        listed = value["profiles"]
        if (not isinstance(listed, list) or not listed or listed != sorted(set(listed))
                or not set(listed) <= set(SUPPORTED)):
            raise ValueError("A v2 image lock lists sorted, distinct, supported installer profiles")
        if profile not in listed:
            raise ValueError(f"{profile} is not admitted on image lock {value['name']}")
        if any(type(value[key]) is not int or value[key] <= 0 for key in ("image_bytes", "download_bytes")):
            raise ValueError("A v2 image lock records positive image_bytes and download_bytes")
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


def default_lock():
    return json.loads(DEFAULT_LOCK.read_text(encoding="utf-8"))


def for_profile(profile, explicit=None):
    """Select the image lock for one installer profile.

    An explicit development lock wins; otherwise every profile listed by the
    default release lock uses that shared image. Unlisted profiles are refused
    because the installer runs one image family only.
    """
    value = explicit if explicit is not None else default_lock()
    return validate(value, profile)


def selection(card, value):
    validate(value, card["profile"])
    scope = ("Development image selection; the published profile's serving qualification does not transfer."
             if value["schema"] == SCHEMA_V1 else
             "Shared installer image; the profile's published serving qualification does not transfer.")
    selected = {**card, "profile_release": card["release"], "release": value["name"],
                "image_id": value["image_id"], "image_reference": value["image_reference"], "evidence_scope": scope}
    if value["schema"] == SCHEMA:
        selected.update(image_bytes=value["image_bytes"], download_bytes=value["download_bytes"])
    return selected


def binding_path(lock, row):
    return str(PurePosixPath(row["deployment_root"]) / lock["id"] / "runtime-binding.json")


def _plugins(existing):
    names = [name for name in existing.split(",") if name]
    return ",".join([*PLUGINS, *[name for name in names if name not in PLUGINS]])


def qwen_recipe(environment):
    """The HC mode and image features a Qwen serving environment selects.

    Token-row ownership (VLLM_QWEN3_8_HC_PREFILL_MODE=shard) excludes HC
    projection sharding; SPARKRING_FEATURES names the image feature bundles.
    """
    shard = environment.get("VLLM_QWEN3_8_HC_PREFILL_MODE", "off") == "shard"
    features = {name.strip() for name in environment.get("SPARKRING_FEATURES", "").split(",") if name.strip()}
    return {"projection_tp": "0" if shard else "1", "prefill_row_ownership": "shard" if shard else "off"}, features


def profile_environment(profile):
    from runtime.common import profiles
    metadata, _ = profiles.load(profile)
    return profiles.read_json(profiles.local_path(metadata["configuration"]["path"]))["environment"]


def adapt(spec, value, *, binding, source_root, profile=None):
    """Reuse the canonical model/network envelope, replacing its runtime binding."""
    profile = profile if profile is not None else profiles_of(value)[0]
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
        VLLM_PLUGINS=_plugins(environment.get("VLLM_PLUGINS", "")),
        SPARKRING_RUNTIME_BINDING=BINDING_TARGET,
        SPARKRING_TRANSPORT_PROFILE=value["transport_profile"],
        SPARKRING_TRANSPORT_MANIFEST_SHA256=value["transport_manifest_sha256"],
        NCCL_VERSION="2.32.3", NCCL_ROOT="/opt/sparkring/toolchain/nccl",
        NCCL_LIB_DIR="/opt/sparkring/toolchain/nccl/lib", NCCL_INCLUDE_DIR="/opt/sparkring/toolchain/nccl/include",
        VLLM_NCCL_INCLUDE_PATH="/opt/sparkring/toolchain/nccl/include",
        VLLM_NCCL_SO_PATH="/opt/sparkring/toolchain/nccl/lib/libnccl.so.2",
        NCCL_LOCAL_INFERENCE_PATH="/opt/sparkring/toolchain/nccl/lib/libnccl.so.2",
        CUDA_HOME="/usr/local/cuda-13.4", CUDA_PATH="/usr/local/cuda-13.4", CUDA_VERSION="13.4.2",
        TILELANG_CACHE_DIR=cache + "/tilelang", TVM_FFI_CACHE_DIR=cache + "/tvm-ffi",
        FLASHINFER_WORKSPACE_BASE=cache + "/flashinfer",
    )
    if profile in QWEN:
        environment["VLLM_QWEN3_8_FLASH_NEXT_HC_TP"] = qwen_recipe(environment)[0]["projection_tp"]
    if len(spec.command) < 2 or spec.command[1] != "serve":
        raise ValueError("External image adapter requires a canonical vLLM serve command")
    health = ("python3", *spec.health_command[1:]) if spec.health_command else ()
    from runtime.common import loader_policy
    if any(option.startswith(("seccomp=", "seccomp:")) for option in spec.security_opt):
        raise ValueError("External loader policy cannot replace an existing profile policy")
    return replace(spec, image_id=value["image_id"], entrypoint=ENTRYPOINT, command=spec.command[1:],
                   environment=environment, health_command=health,
                   security_opt=(*spec.security_opt, "seccomp=" + str(PurePosixPath(source_root) / loader_policy.RELATIVE)),
                   mounts=(*spec.mounts, Bind(binding, BINDING_TARGET, True)),
                   labels={**spec.labels, "io.sparkring.image-lock": value["name"]})


def admit(value, *, run, profile=None, nodes=None, environment=None):
    """Verify the local image against the lock before any model downtime.

    ``profile`` and ``nodes`` select model-specific capability checks. They
    default to the single profile of a v1 lock. ``environment`` is the Qwen
    serving environment; it defaults to the profile's configuration.
    """
    profile = profile if profile is not None else profiles_of(value)[0]
    validate(value, profile)
    nodes = nodes if nodes is not None else (4 if profile.endswith("tp4") else 2)
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
    if (parent.get("schema") != "sparkring-external-installed/v1"
            or parent.get("composition_sha256") != value["composition_sha256"]
            or capabilities.get("transport_profile") != value["transport_profile"]
            or capabilities.get("transport_manifest_sha256") != value["transport_manifest_sha256"]
            or capabilities.get("runtime_status", {}).get("version") != value["status_version"]):
        raise ValueError("External software receipt does not satisfy this runtime contract")
    if profile in QWEN:
        # The profile selects its HC mode and feature bundles; the image receipt
        # declares which modes each node count supports.
        hc_mode, required_features = qwen_recipe(environment if environment is not None else profile_environment(profile))
        if (not required_features <= set(capabilities.get("features", []))
                or hc_mode not in capabilities.get("hc_supported_modes", {}).get(str(nodes), [])):
            raise ValueError("External software receipt does not provide this Qwen topology's features")
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
