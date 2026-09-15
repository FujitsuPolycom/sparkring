"""Plan, create, or manually start the NVFP4-Spark switched TP4/DCP1 profile."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.common.environment import read_assignments  # noqa: E402


ROOT = Path(__file__).resolve().parents[1] / "profiles/glm53-flash-spark-tp4-switched"
PROFILE_PATH = ROOT / "profile.json"
SITE_KEYS = {"VLLM_HOST_IP", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX"}
REGISTRY_IMAGE_PATTERN = r"ghcr\.io/fujitsupolycom/sparkring@sha256:[0-9a-f]{64}"
LOCAL_IMAGE_PATTERN = r"sha256:[0-9a-f]{64}"
SOURCE_IMAGE_ROOT = ROOT.parents[1] / "sparkring/source_image"
HEALTHCHECK_COMMAND = "test -f /tmp/sparkring-engine-ready || exit 1"


def load_profile():
    return json.loads(PROFILE_PATH.read_text())


def read_site(path: Path) -> dict[str, str]:
    result = read_assignments(path, SITE_KEYS)
    selector = result["NCCL_IB_HCA"]
    if not re.fullmatch(r"=[A-Za-z0-9_.-]+(?::[1-9][0-9]*)?(?:,[A-Za-z0-9_.-]+(?::[1-9][0-9]*)?)*", selector):
        raise ValueError("NCCL_IB_HCA must list exact switch-connected HCA names, optionally with ports")
    if len(set(selector[1:].split(","))) != len(selector[1:].split(",")):
        raise ValueError("NCCL_IB_HCA must not repeat an HCA/port")
    if not result["NCCL_IB_GID_INDEX"].isdigit():
        raise ValueError("NCCL_IB_GID_INDEX must be the operator-verified nonnegative index")
    return result

def render(rank, master, model_dir, cache_dir, env_file, image):
    if rank not in range(4) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", master):
        raise ValueError("A valid rank and master address are required")
    if not (re.fullmatch(REGISTRY_IMAGE_PATTERN, image) or re.fullmatch(LOCAL_IMAGE_PATTERN, image)):
        raise ValueError("Select an immutable registry digest or exact local sha256 image config ID")
    if not (model_dir / "config.json").is_file() or not cache_dir.is_dir():
        raise ValueError("An existing checkpoint with config.json and a cache directory are required")
    for path in (model_dir, cache_dir):
        if "," in str(path.resolve()):
            raise ValueError("Docker bind-mount paths must not contain commas")
    if model_dir.resolve() == cache_dir.resolve():
        raise ValueError("Checkpoint and writable cache directories must be distinct")
    profile = load_profile()
    profile_hash = hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest()
    environment = {**profile["environment"], **read_site(env_file)}
    environment.update(NODE_RANK=str(rank), SPARKRING_NODE_RANK=str(rank),
                       MASTER_ADDR=master, SOURCE_IMAGE_PROFILE=profile["name"])
    # Keep compilation artifacts in the mounted tree and separate ranks and
    # source identities. This directory is not an external prompt/KV cache.
    jit = f"/cache/jit/{profile_hash}/rank{rank}"
    environment.update(
        LOCAL_INFERENCE_CACHE_FINGERPRINT=profile_hash,
        XDG_CACHE_HOME=jit, VLLM_CACHE_ROOT=jit + "/vllm",
        VLLM_CACHE_DIR=jit + "/vllm", TRITON_CACHE_DIR=jit + "/triton",
        TORCHINDUCTOR_CACHE_DIR=jit + "/torchinductor",
        TORCH_EXTENSIONS_DIR=jit + "/torch-extensions",
        B12X_ROCE_CACHE_DIR=jit + "/rocenante",
        B12X_COMPILE_CACHE_DIR=jit + "/b12x",
        B12X_CUTE_COMPILE_CACHE_DIR=jit + "/b12x-cute",
        CUTE_DSL_CACHE_DIR=jit + "/cute-dsl",
        CUDA_CACHE_PATH=jit + "/cuda",
        FLASHINFER_WORKSPACE_BASE=jit + "/flashinfer",
        VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=jit + "/flashinfer-autotune",
        TVM_FFI_CACHE_DIR=jit + "/tvm-ffi", TVM_CACHE_DIR=jit + "/tvm",
        TILELANG_CACHE_DIR=jit + "/tilelang", TILELANG_TMP_DIR=jit + "/tilelang/tmp",
        SPARKINFER_COMPILE_CACHE_DIR=jit + "/b12x",
        DG_JIT_CACHE_DIR=jit + "/deep-gemm",
        MM_SPARSE_ATTN_AOT_CACHE=jit + "/minfer/mm-sparse-attn",
        MINFER_FMHA_CACHE_DIR=jit + "/minfer/fmha",
        CUPY_CACHE_DIR=jit + "/cupy", NUMBA_CACHE_DIR=jit + "/numba",
    )
    args = [value.replace("${NODE_RANK}", str(rank)).replace("${MASTER_ADDR}", master)
            for value in profile["vllm_args"]]
    if "--kv-transfer-config" in args or profile["sparkcache"]["enabled"]:
        raise ValueError("SparkCache requires a separately validated composition profile")
    if rank != 0:
        args.append("--headless")
    name = f"sparkring-glm53-spark-switched-r{rank}"
    labels = {
        "org.sparkring.memory-guard": "true",
        "org.sparkring.profile": profile["name"],
        "org.sparkring.profile.sha256": profile_hash,
        "org.sparkring.topology": "switched",
        "org.sparkring.collective-backend": "nccl",
    }
    owner = cache_dir.stat()
    container_user = f"{owner.st_uid}:{owner.st_gid}"
    command = ["docker", "create", "--name", name, "--restart", "no",
               "--init", "--health-cmd", HEALTHCHECK_COMMAND,
               "--health-interval", "2s", "--health-timeout", "1s", "--health-start-period", "600s", "--health-retries", "1",
               "--user", container_user, "--gpus", "all", "--network", "host", "--ipc", "host",
               "--device", "/dev/infiniband", "--ulimit", "memlock=-1:-1",
               "--mount", f"type=bind,src={model_dir.resolve()},dst=/models/target,readonly",
               "--mount", f"type=bind,src={cache_dir.resolve()},dst=/cache/jit",
               "--entrypoint", "python3"]
    for key, value in labels.items():
        command.extend(["--label", key + "=" + value])
    for key, value in sorted(environment.items()):
        command.extend(["--env", key + "=" + value])
    container_args = ["-S", "-B", "/opt/sparkcache-jj-runtime/verify_sources.py", "--serve", *args]
    command.extend([image, *container_args])
    return {
        "schema": "sparkring-profile-launch-plan/v1", "status": "implemented",
        "name": name, "profile": profile["name"], "profile_sha256": profile_hash,
        "image": image, "topology": "switched", "collective_backend": "nccl",
        "container_user": container_user, "transport_profile": None,
        "image_identity_kind": "local_config_id" if re.fullmatch(LOCAL_IMAGE_PATTERN, image) else "registry_manifest_digest",
        "labels": labels, "environment": environment, "container_args": container_args,
        "command": command,
        "binds": {"/models/target": str(model_dir.resolve()), "/cache/jit": str(cache_dir.resolve())},
        "memory_guard_floor_bytes": profile["lifecycle"]["memory_guard_minimum_floor_bytes"],
        "automatic_restart": False, "automatic_start": False,
        "sparkcache_enabled": False,
        "qualification": profile["qualification"],
    }


def _source_receipt_contract(directory):
    """Load sibling verifier modules without relying on process import state."""
    names = ("archive_utils", "native_files", "source_image_receipt_contract")
    previous = {name: sys.modules.get(name) for name in names}
    try:
        for name, filename in zip(names, ("archive_utils.py", "native_files.py", "receipt_contract.py")):
            spec = importlib.util.spec_from_file_location(name, directory / filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return module
    finally:
        for name, saved in previous.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved


def validate_source_image_receipt(receipt, plan, source_root=None):
    """Validate common-source witnesses and the switched profile declaration."""
    source_root = SOURCE_IMAGE_ROOT if source_root is None else source_root
    lock_path = source_root / "glm53-tp4-lock.json"
    validator_path = source_root / "receipt_contract.py"
    if not lock_path.is_file() or not validator_path.is_file():
        raise ValueError("The repository's common source-image lock and receipt validator are required")
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    if receipt.get("source_lock_sha256") != hashlib.sha256(lock_bytes).hexdigest():
        raise ValueError("The source-image receipt differs from the repository's exact source lock")
    validator = _source_receipt_contract(source_root)
    validator.validate_receipt(receipt, lock)
    if receipt.get("image_id") != plan["image"] or receipt.get("profile") != plan["profile"]:
        raise ValueError("The source-image receipt selects a different local image or profile")
    entry = lock["profiles"].get(plan["profile"], {})
    for field in ("profile_sha256", "topology", "collective_backend"):
        if entry.get(field) != plan[field]:
            raise ValueError("The locked source-image profile differs: " + field)
    if (entry.get("tp_size") != 4 or entry.get("dcp_size") != 1
            or entry.get("transport_profile") is not None or entry.get("sparkcache") is not False):
        raise ValueError("The source lock must select switched TP4/DCP1 with NCCL and SparkCache disabled")


def validate_runtime_receipt(receipt, plan):
    """Require source compatibility evidence for this exact image and profile."""
    if re.fullmatch(LOCAL_IMAGE_PATTERN, plan["image"]):
        validate_source_image_receipt(receipt, plan)
        return
    if receipt.get("schema") == "sparkring-source-image-receipt/v1":
        raise ValueError("A local image config receipt is not a registry publication receipt")
    if receipt.get("registry_digest") != plan["image"]:
        raise ValueError("The runtime receipt does not describe the selected image digest")
    entry = receipt.get("profiles", {}).get(plan["profile"], {})
    for field in ("profile_sha256", "topology", "collective_backend"):
        if entry.get(field) != plan[field]:
            raise ValueError("Runtime profile receipt differs: " + field)
    if entry.get("source_compatibility") != "passed":
        raise ValueError("The shared image has no passing source compatibility checks for this profile")


def execute(plan, action, receipt, *, run=subprocess.run):
    """Run explicit lifecycle actions after checking the guard and container contract."""
    if action not in ("create", "start"):
        raise ValueError("Execution action must be create or start")
    validate_runtime_receipt(receipt, plan)
    service = load_profile()["lifecycle"]["memory_guard_service"]
    run(["systemctl", "is-active", "--quiet", service], check=True)
    guard = run(["systemctl", "show", service, "--property=ExecStart", "--value"],
                check=True, capture_output=True, text=True).stdout
    floor = re.search(r"--available-floor-bytes(?:=|\s+)([0-9]+)(?=\s|;|$)", guard)
    if floor is None or int(floor[1]) < plan["memory_guard_floor_bytes"]:
        raise RuntimeError("The active host memory guard must preserve at least the profile's 4 GiB floor")
    ids = run(["docker", "ps", "--quiet"], check=True, capture_output=True, text=True).stdout.split()
    if ids:
        running = json.loads(run(["docker", "inspect", *ids], check=True,
                                 capture_output=True, text=True).stdout)
        if any(item["HostConfig"].get("DeviceRequests") for item in running):
            raise RuntimeError("A GPU container is running; stop it explicitly before this action")
    if action == "create":
        run(plan["command"], check=True)
        return
    container = json.loads(run(["docker", "inspect", plan["name"]], check=True,
                               capture_output=True, text=True).stdout)[0]
    config = container["Config"]
    host_config = container["HostConfig"]
    actual_env = dict(item.split("=", 1) for item in config.get("Env", []))
    actual_mounts = {item["Destination"]: item for item in container.get("Mounts", [])}
    matches = (
        config.get("Image") == plan["image"]
        and config.get("Entrypoint") == ["python3"]
        and config.get("User") == plan["container_user"]
        and config.get("Healthcheck", {}).get("Test") == ["CMD-SHELL", HEALTHCHECK_COMMAND]
        and config.get("Cmd") == plan["container_args"]
        and all(config.get("Labels", {}).get(key) == value for key, value in plan["labels"].items())
        and all(actual_env.get(key) == value for key, value in plan["environment"].items())
        and all(actual_mounts.get(target, {}).get("Source") == source
                for target, source in plan["binds"].items())
        and actual_mounts.get("/models/target", {}).get("RW") is False
        and actual_mounts.get("/cache/jit", {}).get("RW") is True
        and host_config.get("NetworkMode") == "host"
        and host_config.get("IpcMode") == "host"
        and any(request.get("Count") == -1
                and any("gpu" in group for group in request.get("Capabilities", []))
                for request in host_config.get("DeviceRequests") or [])
        and host_config["RestartPolicy"]["Name"] == "no"
    )
    if not matches:
        raise RuntimeError("Stopped container differs from this profile plan; inspect it before replacement")
    run(["docker", "start", plan["name"]], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "create", "start"))
    parser.add_argument("--rank", type=int, required=True, choices=range(4))
    parser.add_argument("--master", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--runtime-receipt", type=Path)
    args = parser.parse_args()
    plan = render(args.rank, args.master, args.model_dir, args.cache_dir, args.env_file, args.image)
    print(json.dumps(plan, indent=2), flush=True)
    if args.action == "plan":
        return
    if args.runtime_receipt is None:
        parser.error("create and start require --runtime-receipt for this exact local image or registry digest")
    execute(plan, args.action, json.loads(args.runtime_receipt.read_text()))


if __name__ == "__main__":
    main()
