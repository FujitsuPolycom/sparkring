"""Admit published external-base images and bind Qwen profile integrations."""

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

from runtime.common import native_candidate


ROOT = Path(__file__).resolve().parents[2]
LAYOUT = "external-base/v1"
PYTHON = "python3"
ENTRYPOINT = "/opt/sparkring/bin/external-base.py"
RECEIPT = "/opt/sparkring/receipts/external-base-installed.json"
SITE = "/usr/local/lib/python3.12/dist-packages"
NCCL = "/opt/local-inference/nccl/lib/libnccl.so.2"
ENVELOPE = {"cap_add": ["IPC_LOCK"], "security_opt": ["seccomp=unconfined"]}
STATUS_PYTHON_ROOT = "/opt/sparkring/python"
STATUS_ENTRY_POINTS = {
    "vllm.endpoint_plugins": {"sparkring_status": "sparkring_runtime_status.plugin:StatusPlugin"},
    "vllm.general_plugins": {"sparkring_status": "sparkring_runtime_status.plugin:register_worker_method"},
}
HC_SUPPORTED_MODES = {
    "2": [{"projection_tp": "1", "prefill_row_ownership": "off"}],
    "4": [
        {"projection_tp": "1", "prefill_row_ownership": "off"},
        {"projection_tp": "0", "prefill_row_ownership": "shard"},
    ],
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def publication(release, *, image_id=None):
    # Shared publication identity, registry confirmation and transport pinning
    # apply to both layouts; installed receipt validation is layout-specific.
    record = native_candidate.publication(release, image_id=image_id)
    if (record.get("runtime_layout") != LAYOUT
            or not re.fullmatch(r"[0-9a-f]{64}", record.get("composition_sha256", ""))
            or not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", record.get("base", {}).get("reference", ""))
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", record.get("base", {}).get("config_id", ""))
            or set(record.get("sources", {})) != {"vllm", "b12x"}
            or set(record.get("features", [])) != {"qwen-collectives", "qwen4-prefill"}
            or record.get("hc_supported_modes") != HC_SUPPORTED_MODES):
        raise ValueError("External release lacks its composition, base or source identity")
    for source in record["sources"].values():
        if any(not re.fullmatch(r"[0-9a-f]{40}", source.get(name, ""))
               for name in ("commit", "upstream", "baseline_commit")) or any(
            not re.fullmatch(r"[0-9a-f]{64}", source.get(name, ""))
            for name in ("archive_sha256", "baseline_archive_sha256")
        ):
            raise ValueError("External release source revisions or archive pins are incomplete")
    contract = record.get("sparkcache_contract", {})
    if (not re.fullmatch(r"/opt/sparkring/contracts/[A-Za-z0-9_.-]+\.json", contract.get("path", ""))
            or not re.fullmatch(r"[0-9a-f]{64}", contract.get("sha256", ""))):
        raise ValueError("External release lacks its SparkCache source contract")
    if not isinstance(record.get("profiles"), dict) or not record["profiles"]:
        raise ValueError("External release must pin its opt-in serving profiles")
    for name, digest in record["profiles"].items():
        path = PurePosixPath(name)
        if (str(path) != name or path.is_absolute() or ".." in path.parts or "\\" in name
                or not name.startswith("profiles/") or path.suffix != ".json"
                or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("External release profile reference is invalid")
    status = record.get("runtime_status")
    if status is not None and (
        status.get("distribution") != "sparkring-runtime-status"
        or not re.fullmatch(r"[0-9][A-Za-z0-9_.+-]*", status.get("version", ""))
        or status.get("plugin_name") != "sparkring_status"
        or status.get("python_root") != STATUS_PYTHON_ROOT
        or status.get("entry_points") != STATUS_ENTRY_POINTS
        or any(not re.fullmatch(r"[0-9a-f]{64}", status.get(name, ""))
               for name in ("wheel_sha256", "source_archive_sha256"))
    ):
        raise ValueError("External release status artifact identity or entry points differ")
    return record


def canonical_profile(profile):
    record = publication(profile.get("image_release"))
    matched = False
    for name, digest in record["profiles"].items():
        path = ROOT / name
        raw = path.read_bytes()
        if sha(raw) != digest:
            raise ValueError("External release serving profile changed: " + name)
        if json.loads(raw) == profile:
            matched = True
    if matched:
        validate_profile_contract(profile)
        return profile
    raise ValueError("Select a serving profile pinned by the external release")


def release_publication(release, identity):
    """Bind a deployment selection to the external publication it actually names."""
    record = publication(identity)
    name = f"runtime/releases/{identity}/publication.json"
    inputs = release.get("inputs", [])
    rows = {row.get("path"): row.get("sha256") for row in inputs}
    if (release.get("schema") != "sparkring-release-selection/v1"
            or release.get("id") != identity
            or release.get("selection") != "published-immutable-reference"
            or release.get("image") != record["image_reference"]
            or not inputs or inputs[0].get("path") != name
            or len(rows) != len(inputs)
            or rows.get(name) != sha((ROOT / name).read_bytes())):
        raise ValueError("External release selection must pin its publication and registry image")
    return record


def profile_nodes(profile):
    arguments = profile["vllm_args"]
    nodes = {"direct-pair-2": 2, "direct-cycle-4": 4}.get(profile.get("topology"))
    required = {"--tensor-parallel-size": nodes, "--nnodes": nodes,
                "--pipeline-parallel-size": 1, "--decode-context-parallel-size": 1}
    if nodes is None or any(
        arguments.count(flag) != 1 or arguments.index(flag) + 1 == len(arguments)
        or arguments[arguments.index(flag) + 1] != str(value)
        for flag, value in required.items()
    ):
        raise ValueError("External Qwen profiles require matching TP2/TP4 nodes with PP1 and DCP1")
    return nodes


def validate_profile_contract(profile):
    if (profile.get("schema") != "sparkring-serving-profile/v1"
            or profile.get("image_extension") != "external-base"):
        raise ValueError("External Qwen profile schema or image extension differs")
    nodes = profile_nodes(profile)
    if selected_hc_mode(profile) not in HC_SUPPORTED_MODES[str(nodes)]:
        raise ValueError("Select mutually exclusive HC projection or prefill row ownership for this TP size")
    if profile.get("container_envelope") != ENVELOPE:
        raise ValueError("External loader profile requires its explicit IPC_LOCK/seccomp envelope")
    args = profile["vllm_args"]

    def value(flag):
        if args.count(flag) != 1 or args.index(flag) + 1 == len(args):
            raise ValueError("External Qwen profile requires exactly one value for " + flag)
        return args[args.index(flag) + 1]

    expected = {
        "--max-model-len": "262144", "--max-num-seqs": "16",
        "--max-num-batched-tokens": "8192", "--kv-cache-memory-bytes": "25769803776",
    }
    if any(value(flag) != expected_value for flag, expected_value in expected.items()):
        raise ValueError("External Qwen profile serving capacity differs from its supported scope")
    speculative = json.loads(value("--speculative-config"))
    if (speculative.get("method") != "mtp" or speculative.get("num_speculative_tokens") != 3
            or json.loads(value("--compilation-config")).get("max_cudagraph_capture_size") != 64
            or json.loads(value("--model-loader-extra-config")) != {"read_mode": "bounce", "io_threads": 8}):
        raise ValueError("External Qwen profile requires MTP3, graph ceiling 64 and bounce loading with 8 threads")
    expected_features = "qwen-collectives,qwen4-prefill" if nodes == 4 else ""
    if profile["environment"].get("SPARKRING_FEATURES") != expected_features:
        raise ValueError("External Qwen feature selection does not match TP eligibility")
    return nodes


def selected_hc_mode(profile):
    environment = profile["environment"]
    return {
        "projection_tp": environment.get("VLLM_QWEN3_8_FLASH_NEXT_HC_TP"),
        "prefill_row_ownership": environment.get("VLLM_QWEN3_8_HC_PREFILL_MODE"),
    }


def profile_identity(profile):
    """Bind shared cache semantics before rank-local site settings are applied."""
    return sha(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode())


def profile_settings(profile, record):
    """Rebind image integration paths while retaining operational profile flags."""
    nodes = validate_profile_contract(profile)
    result = copy.deepcopy(profile)
    environment = result["environment"]
    if selected_hc_mode(profile) not in record["hc_supported_modes"][str(nodes)]:
        raise ValueError("Selected HC ownership mode is unsupported by the release")
    plugins = [name.strip() for name in environment.get("VLLM_PLUGINS", "").split(",") if name.strip()]
    if ("sparkring_status" in plugins) != (record.get("runtime_status") is not None):
        raise ValueError("Status plugin activation must match the release's selected artifact")
    features = {name for name in environment.get("SPARKRING_FEATURES", "").split(",") if name}
    if not features <= set(record["features"]) or (nodes == 2 and "qwen4-prefill" in features):
        raise ValueError("External Qwen feature selection does not match TP eligibility")
    environment.update({
        "VLLM_NCCL_SO_PATH": NCCL, "LD_PRELOAD": NCCL,
        "NCCL_LIB_DIR": str(PurePosixPath(NCCL).parent),
        "NCCL_LOCAL_INFERENCE_PATH": NCCL,
        "SPARKCACHE_SOURCE_LEASE_CONTRACT": record["sparkcache_contract"]["path"],
    })
    # This composition exports a runtime library, not the old image's headers.
    environment.pop("NCCL_INCLUDE_DIR", None)
    args = result["vllm_args"]
    if "--kv-transfer-config" in args:
        index = args.index("--kv-transfer-config") + 1
        transfer = json.loads(args[index])
        if (transfer.get("kv_connector") != "SparkContextCacheConnector"
                or transfer.get("kv_connector_module_path") != "sparkcache.spark_context_cache_connector"
                or transfer.get("kv_role") != "kv_both"):
            raise ValueError("External Qwen cache profile requires the SparkCache connector")
        extra = transfer["kv_connector_extra_config"]
        extra["spark_cache_async_page_capture_lease_contract"] = record["sparkcache_contract"]["path"]
        extra["spark_cache_async_page_capture_vllm_root"] = SITE
        extra["spark_cache_root"] = (
            f"/cache/persistent/qwen38-flash-next-qad-tp{nodes}-"
            + record["composition_sha256"][:16] + "-" + profile_identity(profile)[:16]
        )
        args[index] = json.dumps(transfer, separators=(",", ":"))
    if "/opt/venv" in json.dumps([environment, args]):
        raise ValueError("External profile retains an incompatible virtualenv path")
    return result


def validate_identity(record, image, inspection):
    if (image != record["image_id"] or inspection.get("Id") != image
            or inspection.get("Os") != "linux" or inspection.get("Architecture") != "arm64"
            or inspection.get("Config", {}).get("Entrypoint") != [PYTHON, ENTRYPOINT]):
        raise ValueError("External image identity, platform or entrypoint differs")


def validate(record, image, inspection, raw, verification):
    validate_identity(record, image, inspection)
    if sha(raw) != record["installed_receipt_sha256"]:
        raise ValueError("Installed external receipt differs from publication")
    installed = json.loads(raw)
    capabilities = installed.get("capabilities", {})
    status = record.get("runtime_status")
    status_inventory = installed.get("python_roots", {}).get(STATUS_PYTHON_ROOT, {})
    if (bool(status_inventory) != (status is not None)
            or status is not None and installed.get("files", {}).get(SITE + "/sparkring_runtime_status.pth")
            != sha((STATUS_PYTHON_ROOT + "\n").encode())):
        raise ValueError("External status import root or path hook is not receipt-bound")
    if (installed.get("schema") != "sparkring-external-installed/v1"
            or installed.get("base") != record["base"]
            or installed.get("composition_sha256") != record["composition_sha256"]
            or installed.get("sources") != record["sources"]
            or capabilities.get("transport_profile") != record["transport"]["profile"]
            or capabilities.get("transport_manifest_sha256") != record["transport"]["manifest_sha256"]
            or installed.get("files", {}).get("/opt/sparkring/transports/" + record["transport"]["profile"]
                                              + "/manifest.json") != record["transport"]["manifest_sha256"]
            or set(capabilities.get("features", [])) != set(record["features"])
            or capabilities.get("hc_supported_modes") != record["hc_supported_modes"]
            or capabilities.get("runtime_status") != record.get("runtime_status")
            or capabilities.get("sparkcache_contract") != record["sparkcache_contract"]["path"]
            or installed.get("files", {}).get(record["sparkcache_contract"]["path"]) != record["sparkcache_contract"]["sha256"]
            or verification.get("schema") != "sparkring-external-verification/v1"
            or verification.get("composition_sha256") != record["composition_sha256"]
            or verification.get("sources") != record["sources"]
            or verification.get("framework_native_rebuilt") is not False
            or verification.get("owned_python_roots") != ([STATUS_PYTHON_ROOT] if record.get("runtime_status") is not None else [])
            or verification.get("files_verified") != len(installed.get("files", {}))
            or not installed.get("files")):
        raise ValueError("External installed composition/source verification differs from publication")
    return {
        "schema": "sparkring-external-host-verification/v1", "image_id": image,
        "image_reference": record["image_reference"], "platform": "linux/arm64",
        "release": record["release"], "installed_receipt_sha256": sha(raw),
        "verification": verification,
    }


def verify_image(image, release, *, run=subprocess.run):
    record = publication(release, image_id=image)
    inspection = json.loads(run(["docker", "image", "inspect", image], check=True,
                                capture_output=True, text=True).stdout)[0]
    validate_identity(record, image, inspection)
    common = ["docker", "run", "--rm", "--pull", "never", "--runtime", "runc",
              "--network", "none", "--env", "NVIDIA_VISIBLE_DEVICES=void"]
    raw = run([*common, "--entrypoint", "/bin/cat", image, RECEIPT], check=True,
              capture_output=True).stdout
    verified = json.loads(run([*common, image, "verify"], check=True,
                              capture_output=True, text=True).stdout)
    return validate(record, image, inspection, raw, verified)
