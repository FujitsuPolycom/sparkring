"""Admit source-recorded external CUDA/NCCL images for installer deployments.

The model profile owns serving settings. An image lock binds the external
software and toolchain receipts; it never changes a published release.

Two lock schemas exist:

- ``sparkring-installer-image/v1`` names exactly one Qwen profile. Saved v1
  deployments keep validating so their recorded rollback remains usable.
- ``sparkring-installer-image/v2`` names the set of installer profiles that
  run on one shared image. The installer applies the release's v2 lock to every
  profile it lists; see ``default_lock``.

Admission is model-aware: profiles of Qwen3.8-Flash-Next-architecture
checkpoints (``QWEN4_EXP``) additionally require the image's Qwen collective
features and hybrid-attention modes, while DeepSeek, GLM and MiMo profiles rely
on the verified receipts and the installed-tree check alone.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from runtime.common import profiles
from runtime.common.container_spec import Bind

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = ("python3", "/opt/sparkring/toolchain/toolchain.py")
# CUDA toolkit of every admitted installer image; ``admit`` requires it.
CUDA_VERSION = "13.4.2"
BINDING_TARGET = "/run/sparkring/runtime-binding.json"
PARENT_RECEIPT = "/opt/sparkring/receipts/external-base-installed.json"
TOOLCHAIN_RECEIPT = "/opt/sparkring/toolchain/installed.json"
SCHEMA_V1 = "sparkring-installer-image/v1"
SCHEMA = "sparkring-installer-image/v2"
# The release whose image every installer profile uses unless an operator
# supplies an explicit development lock.
DEFAULT_LOCK = ROOT / "runtime/releases/dev-20261004-kraken-cuda1342-nccl2323-status034/installer-image.json"
RELEASES = ROOT / "runtime/releases"
# GitHub release tag -> the installer image release that release published, so
# that `--image 2026.10.0` selects the image its release notes name.
RELEASE_TAGS = RELEASES / "installer-releases.json"
RELEASE_TAGS_SCHEMA = "sparkring-installer-releases/v1"
# What image layers provide that a deployment setting needs, such as the
# shared-memory reader window that `--save-cpu` sets (``capabilities``).
CAPABILITIES = "installer-capabilities.json"
CAPABILITIES_SCHEMA = "sparkring-installer-capabilities/v1"
RELEASE_NAME = re.compile(r"[a-z0-9][a-z0-9.-]{0,127}")
# The Qwen3.8-Flash-Next profiles, the only profiles a v1 lock can name.
QWEN = ("qwen38-flash-next-tp2", "qwen38-flash-next-qad-tp4")
# Profiles whose checkpoints use the Qwen3.8-Flash-Next (Qwen4Exp) architecture,
# including derivatives of other publishers. They run the image's Qwen collective
# features and HC modes, which admission requires and ``adapt`` configures.
QWEN4_EXP_PREPARED = (*QWEN, "swift15-qwen38-flash-next-tp2", "swift15-qwen38-flash-next-tp4")
SUPPORTED = (*QWEN4_EXP_PREPARED, "deepseek-v41-flash-tp4", "glm53-flash-nvfp4-spark-tp2",
             "glm53-flash-nvfp4-spark-tp4", "mimo-v26-flash-mopd-tp2", "mimo-v26-flash-mopd-tp4")
# Installer profiles that only an image carrying SIRCL ring sessions runs (an image lock v3,
# runtime/common/image_lock.py): the research-only profiles whose ranks reach each other through
# relays (profiles.relayed_research). A lock's profile list may name them beside SUPPORTED; only
# image_lock admits a lock that does. They are read from profiles.RESEARCH_CATALOG, which no
# Compose label hashes, so adding one changes no Compose export.
_RELAYED = profiles.relayed_research()
SIRCL_ONLY = tuple(_RELAYED)
# A relayed research profile runs the Qwen3.8-Flash-Next architecture when its configuration selects
# the Qwen HC prefill mode, which ``adapt`` and admission configure as for QWEN4_EXP_PREPARED.
QWEN4_EXP = (*QWEN4_EXP_PREPARED, *(profile for profile, path in _RELAYED.items()
                                    if "VLLM_QWEN3_8_HC_PREFILL_MODE" in profiles.read_json(path)["environment"]))
# A v2 lock may also list IDs in profiles.REPLACED, so that the locks of other
# releases in runtime/releases keep validating; ``for_profile`` refuses those IDs
# and names the catalog profile that replaces each.
PLUGINS = ("b12x_loader", "sparkring_status")
# The variable that selects the tool-result policy: with 1, a named or required
# tool_choice request without a complete call fails instead of returning an
# empty tool_calls list (integrations/vllm/tool_choice_contract). ``adapt`` sets
# it to 1 unless the profile's environment sets it; only an image derived with
# runtime/images/derive_tool_choice_contract.py reads it.
TOOL_CHOICE_CONTRACT = "SPARKRING_TOOL_CHOICE_CONTRACT"
# The installer waits up to 30 minutes for rank 0 to report healthy. A first
# start compiles and tunes kernels for every CUDA graph size, so the health
# check tolerates failures for the same period instead of marking a rank that
# is still starting unhealthy.
HEALTH_START_SECONDS = 1800
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
                or not set(listed) <= set(SUPPORTED) | set(SIRCL_ONLY) | set(profiles.REPLACED)):
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
    replaced = profiles.replacement_message(profile)
    if replaced:
        raise ValueError(replaced)
    value = explicit if explicit is not None else default_lock()
    try:
        return validate(value, profile)
    except ValueError as error:
        # A well-formed lock that does not list the profile: name the images that do.
        if explicit is None or value.get("schema") != SCHEMA or profile in value.get("profiles", ()):
            raise
        others = [row["name"] for row in catalog() if profile in profiles_of(row["lock"])]
        raise ValueError(f"{error}; images that run it: {', '.join(others) or 'none'}") from None


def release_tags():
    """GitHub release tag -> installer image release name, from installer-releases.json."""
    document = json.loads(RELEASE_TAGS.read_text(encoding="utf-8"))
    if document.get("schema") != RELEASE_TAGS_SCHEMA or not isinstance(document.get("releases"), dict):
        raise ValueError(f"{RELEASE_TAGS} is not a {RELEASE_TAGS_SCHEMA} document")
    return dict(document["releases"])


def capability_records(root=None):
    """installer-capabilities.json's ``capabilities``, validated: capability -> {summary, added_by}.

    ``added_by`` maps each image release whose own layer adds the capability
    to the repository file that describes that layer.
    """
    root = ROOT if root is None else Path(root)
    path = root / "runtime" / "releases" / CAPABILITIES
    document = json.loads(path.read_text(encoding="utf-8"))
    if set(document) != {"schema", "capabilities"} or document["schema"] != CAPABILITIES_SCHEMA:
        raise ValueError(f"{path} is not a {CAPABILITIES_SCHEMA} document")
    for name, row in document["capabilities"].items():
        if (not isinstance(row, dict) or set(row) != {"summary", "added_by"} or not isinstance(row["added_by"], dict)
                or any(not (root / "runtime" / "releases" / release / "release.json").is_file()
                       or not isinstance(layer, str) or not (root / layer).is_file()
                       for release, layer in row["added_by"].items())):
            raise ValueError(f"{path}: {name} needs a summary and the existing releases whose layers add it")
    return document["capabilities"]


def capabilities(name, root=None):
    """The capabilities of installer image release ``name``, sorted.

    An image has what its own layer adds (``added_by`` in
    installer-capabilities.json) and what every image it derives from has:
    its ``publication.json`` names its parent in ``derivation.parent_release``,
    and a derived image keeps its parent's layers. A release this package does
    not record, such as a development lock's, has none.
    """
    root = ROOT if root is None else Path(root)
    records = capability_records(root)
    found, seen = set(), set()
    while isinstance(name, str) and RELEASE_NAME.fullmatch(name) and name not in seen:
        seen.add(name)
        found.update(capability for capability, row in records.items() if name in row["added_by"])
        try:
            publication = json.loads((root / "runtime" / "releases" / name / "publication.json").read_text(encoding="utf-8"))
            name = publication["derivation"]["parent_release"]
        except (OSError, ValueError, KeyError, TypeError):
            break
    return tuple(sorted(found))


def catalog():
    """The installer images whose locks this package carries: the default first, then newest name first.

    Each row holds the release ``name`` (its directory under runtime/releases),
    the lock's ``path``, the ``lock``, the GitHub release ``tags`` that
    published it and whether it is the ``default``. Only locks that reference
    a registry digest are listed; a lock naming a local configuration ID runs
    only where that image was loaded by hand.
    """
    published = {}
    for tag, name in release_tags().items():
        published.setdefault(name, []).append(tag)
    rows = []
    for path in RELEASES.glob("*/installer-image.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if "@sha256:" not in str(value.get("image_reference", "")):
            continue
        name = path.parent.name
        rows.append({"name": name, "path": path, "lock": value, "tags": sorted(published.get(name, [])),
                     "default": path == DEFAULT_LOCK})
    rows.sort(key=lambda row: row["name"], reverse=True)
    return sorted(rows, key=lambda row: not row["default"])


def lock_path(name):
    """The lock file of the installer image ``name`` selects; None for the default image.

    ``name`` is an image's release name
    (``dev-20261001-kraken-cuda1342-nccl2323-status034``), the GitHub release
    tag that published it (``2026.10.0``), or a part of a release name between
    hyphens that only one image has (``statusrows``). Selecting the default
    image returns None, so the request equals one without a selection.
    """
    rows = catalog()
    by_name = {row["name"]: row for row in rows}
    tags = release_tags()
    if name in tags:
        if tags[name] not in by_name:
            raise ValueError(f"Release {name} published {tags[name]}, whose lock this package does not carry")
        row = by_name[tags[name]]
    elif name in by_name:
        row = by_name[name]
    else:
        matches = [row for row in rows if f"-{name}-" in f"-{row['name']}-"]
        if len(matches) > 1:
            raise ValueError(f"Image {name} matches several images: {', '.join(row['name'] for row in matches)}; "
                             "give one full name")
        if not matches:
            raise ValueError(f"No installer image is named {name}; sparkring images lists them")
        row = matches[0]
    return None if row["default"] else row["path"]


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
        CUDA_HOME="/usr/local/cuda-13.4", CUDA_PATH="/usr/local/cuda-13.4", CUDA_VERSION=CUDA_VERSION,
        TILELANG_CACHE_DIR=cache + "/tilelang", TVM_FFI_CACHE_DIR=cache + "/tvm-ffi",
        FLASHINFER_WORKSPACE_BASE=cache + "/flashinfer",
    )
    environment.setdefault(TOOL_CHOICE_CONTRACT, "1")
    if profile in QWEN4_EXP:
        environment["VLLM_QWEN3_8_FLASH_NEXT_HC_TP"] = qwen_recipe(environment)[0]["projection_tp"]
    if len(spec.command) < 2 or spec.command[1] != "serve":
        raise ValueError("External image adapter requires a canonical vLLM serve command")
    health = ("python3", *spec.health_command[1:]) if spec.health_command else ()
    from runtime.common import loader_policy
    if any(option.startswith(("seccomp=", "seccomp:")) for option in spec.security_opt):
        raise ValueError("External loader policy cannot replace an existing profile policy")
    return replace(spec, image_id=value["image_id"], entrypoint=ENTRYPOINT, command=spec.command[1:],
                   environment=environment, health_command=health, health_start_period=HEALTH_START_SECONDS,
                   security_opt=(*spec.security_opt, "seccomp=" + str(PurePosixPath(source_root) / loader_policy.RELATIVE)),
                   mounts=(*spec.mounts, Bind(binding, BINDING_TARGET, True)),
                   labels={**spec.labels, "io.sparkring.image-lock": value["name"]})


def profile_nodes(profile):
    """The number of Sparks installer profile ``profile`` runs on: its configuration's
    ``--tensor-parallel-size``, one rank per Spark."""
    metadata, _ = profiles.load(profile)
    arguments = profiles.read_json(profiles.local_path(metadata["configuration"]["path"]))["vllm_args"]
    return int(arguments[arguments.index("--tensor-parallel-size") + 1])


def admit(value, *, run, profile=None, nodes=None, environment=None):
    """Verify the local image against the lock before any model downtime.

    ``profile`` and ``nodes`` select model-specific capability checks. They
    default to the single profile of a v1 lock and to the profile's tensor
    parallelism (``profile_nodes``). ``environment`` is the Qwen serving
    environment; it defaults to the profile's configuration.
    """
    profile = profile if profile is not None else profiles_of(value)[0]
    validate(value, profile)
    nodes = nodes if nodes is not None else profile_nodes(profile)
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
    if profile in QWEN4_EXP:
        # The profile selects its HC mode and feature bundles; the image receipt
        # declares which modes each node count supports.
        hc_mode, required_features = qwen_recipe(environment if environment is not None else profile_environment(profile))
        if (not required_features <= set(capabilities.get("features", []))
                or hc_mode not in capabilities.get("hc_supported_modes", {}).get(str(nodes), [])):
            raise ValueError("External software receipt does not provide this Qwen topology's features")
    if (toolchain.get("schema") != "sparkring-toolchain-installed/v1" or toolchain.get("variant") != "combined"
            or toolchain.get("parent_receipt_sha256") != value["parent_receipt_sha256"]
            or toolchain.get("nccl_version") != 23203 or "V13.4.92" not in toolchain.get("nvcc", "")):
        raise ValueError(f"Toolchain receipt is not the selected CUDA {CUDA_VERSION} / NCCL 2.32.3 composition")
    # Verify the actual installed tree, not just receipt labels. No GPU or network
    # is exposed; serving still has its own health and generation barriers.
    run([*isolated, "--entrypoint", ENTRYPOINT[0], value["image_id"], ENTRYPOINT[1], "verify"])
    return {"schema": "sparkring-installer-image-observation/v1", "image_id": value["image_id"],
            "composition_sha256": value["composition_sha256"],
            "parent_receipt_sha256": value["parent_receipt_sha256"],
            "toolchain_receipt_sha256": value["toolchain_receipt_sha256"], "serving_qualified": False}
