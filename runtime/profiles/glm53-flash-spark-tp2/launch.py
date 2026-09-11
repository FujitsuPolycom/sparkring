"""Plan, create, or manually start guarded NVFP4-Spark TP2 profiles with explicit KV settings."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PROFILE_PATH = ROOT / "profile.json"
SITE_KEYS = {"VLLM_HOST_IP", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"}
REGISTRY_IMAGE_PATTERN = r"ghcr\.io/fujitsupolycom/sparkring@sha256:[0-9a-f]{64}"
LOCAL_IMAGE_PATTERN = r"sha256:[0-9a-f]{64}"
SOURCE_IMAGE_ROOT = ROOT.parents[1] / "sparkring/source_image"
R33_PROFILE_ROOT = ROOT.parents[1] / "sparkring/jovian-r33/profiles"


def load_profile():
    return json.loads(PROFILE_PATH.read_text())


def transport_environment(rank: int) -> dict[str, str]:
    """Use both PCI domains of physical cage p0, with reciprocal rank maps."""
    if rank not in (0, 1):
        raise ValueError("rank must be 0 or 1")
    transport = load_profile()["transport"]
    manifest_path = ROOT / transport["manifest"]
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != transport["manifest_sha256"]:
        raise ValueError("The profile's transport manifest has changed")
    indices = transport["local_hca_indices"]
    inventory = transport["hca_inventory"]
    return {
        "B12X_ROCE_HCA": ",".join(inventory),
        "B12X_ROCE_PAIR_PATHS": str(transport["path_count"]),
        "B12X_ROCE_PEER_HCA_MAP": f"{1 - rank}=" + "/".join(map(str, indices)),
        "NCCL_IB_HCA": "=" + ",".join(inventory[index] for index in indices),
        "NCCL_IB_MERGE_NICS": "0",
        "NCCL_CROSS_NIC": "1",
        "NCCL_MIN_NCHANNELS": "8",
        "NCCL_MAX_NCHANNELS": "8",
        "VLLM_ENABLE_ROCE_ALLREDUCE": "1",
        "VLLM_ROCE_ALLREDUCE_MAX_SIZE": str(transport["all_reduce_max_bytes"]),
        "VLLM_ROCE_ALLGATHER_MAX_SIZE": str(transport["all_gather_shard_max_bytes"]),
        "SPARKRING_TRANSPORT_PROFILE": transport["name"],
        "SPARKRING_TRANSPORT_MANIFEST_SHA256": transport["manifest_sha256"],
    }


def read_site(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if (not separator or key not in SITE_KEYS or key in result
                or not value or any(char.isspace() for char in value)
                or any(char in value for char in "<>\"'")):
            raise ValueError("Use one resolved assignment for each site setting")
        result[key] = value
    if set(result) != SITE_KEYS:
        raise ValueError("Site settings must include " + ", ".join(sorted(SITE_KEYS)))
    return result


def _r33_verifier():
    spec = importlib.util.spec_from_file_location("sparkring_r33_profile_verifier", R33_PROFILE_ROOT / "verify_profile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_option(arguments, flag, value):
    result = list(arguments)
    result[result.index(flag) + 1] = value
    return result


def _remove_option(arguments, flag):
    result = list(arguments)
    index = result.index(flag)
    del result[index:index + 2]
    return result


def adapt_r33_plan(plan, receipt, *, sparkcache=False, cache_kv_memory_bytes=None):
    verifier = _r33_verifier()
    verifier.validate_image_receipt(receipt)
    expected_image = receipt["image_id"] if plan["image_identity_kind"] == "local_config_id" else receipt["image_reference"]
    if plan["image"] != expected_image:
        raise ValueError("R33 receipt does not identify the selected TP2 image")
    contract = verifier.load_contract()
    profile_name = "tp2-dcp1-sparkcache" if sparkcache else "tp2-dcp1"
    selected = contract["profiles"][profile_name]
    environment = dict(plan["environment"])
    environment.update(contract["common_environment"])
    environment.update({
        "LD_PRELOAD": contract["common_environment"]["VLLM_NCCL_SO_PATH"],
        "SOURCE_IMAGE_PROFILE": profile_name,
        "SPARKRING_PROFILE_MODE": "custom",
        "SPARKCACHE_ENABLED": "1" if sparkcache else "0",
        "VLLM_B12X_KDA_PREFILL_COALESCING": "1" if sparkcache else "0",
        "VLLM_B12X_KDA_PREFILL_COALESCING_LOG_LIMIT": "4" if sparkcache else "0",
        "VLLM_SPARK_TP4_MODE": "",
        "VLLM_SPARK_TP4_VOCAB_MODE": "",
    })
    arguments = plan["container_args"][4:]
    arguments = _replace_option(arguments, "--max-model-len", str(contract["model"]["max_model_len"]))
    kv_memory_bytes = selected["kv_cache_memory_bytes"] if cache_kv_memory_bytes is None else cache_kv_memory_bytes
    arguments = _replace_option(arguments, "--kv-cache-memory-bytes", str(kv_memory_bytes))
    arguments = _replace_option(arguments, "--load-format", selected.get("load_format", contract["model"]["loader"]["load_format"]))
    if sparkcache:
        environment.update(LOAD_FORMAT="b12x", VLLM_PLUGINS="b12x_loader", B12X_NVFP4_DYNAMIC_MATERIALIZED="0")
        # R33 B12X fixes weight_pool to managed allocation internally; its
        # DefaultModelLoader superclass rejects the legacy allocation option.
        arguments = _remove_option(arguments, "--model-loader-extra-config")
        speculation = json.loads(arguments[arguments.index("--speculative-config") + 1])
        # Select B12X explicitly for the MTP checkpoint as well.
        speculation["draft_load_config"] = {
            "load_format": "b12x", "model_loader_extra_config": {},
        }
        arguments = _replace_option(arguments, "--speculative-config", json.dumps(speculation))
        arguments = _replace_option(arguments, "--limit-mm-per-prompt", '{"image":3,"video":1}')
        native = contract["sparkcache_native"]
        namespace = f"sparkring-r33-{receipt['image_id'][7:19]}-tp2-cache"
        environment.update(SPARKCACHE_CACHE_NAMESPACE=namespace,
                           SPARKCACHE_PLACEMENT_LIBRARY_PATH=native["placement_path"],
                           SPARKCACHE_PLACEMENT_LIBRARY_SHA256=native["placement_sha256"],
                           SPARKCACHE_SNAPSHOT_LIBRARY_PATH=native["snapshot_path"],
                           SPARKCACHE_SNAPSHOT_LIBRARY_SHA256=native["snapshot_sha256"],
                           SPARKCACHE_VLLM_ROOT=native["vllm_root"],
                           SPARKCACHE_SOURCE_LEASE_CONTRACT=native["lease_contract"])
        fingerprint = "357f6a86160ebd5caff25d9a10d9f29e8547b16c6c73e78751fa69fde11ac4e4"
        extra = dict(
            spark_cache_root="/cache/jit/sparkcache-context/" + namespace,
            spark_cache_model_profile="glm53-flash-hybrid", spark_cache_publication_schema="tail-cow-v2",
            spark_cache_target_checkpoint_sha256=fingerprint, spark_cache_draft_checkpoint_sha256=fingerprint,
            spark_cache_draft_policy="separate", spark_cache_access_mode="read-write",
            spark_cache_shared_prefix_lease_ttl_seconds=300, spark_cache_scheduler_probe="none",
            spark_cache_streaming_snapshots=False, spark_cache_cuda_restore=True,
            spark_cache_max_bytes=8589934592, spark_cache_low_watermark_bytes=6442450944,
            spark_cache_ttl_seconds=0, spark_cache_min_span_tokens=4096, spark_cache_max_span_tokens=65536,
            spark_cache_cuda_placement_library=native["placement_path"],
            spark_cache_cuda_placement_library_sha256=native["placement_sha256"],
            spark_cache_cuda_placement_arena_bytes=67108864, spark_cache_cuda_restore_arena_budget_bytes=268435456,
            spark_cache_cuda_restore_io_workers=2, spark_cache_load_threads=2, spark_cache_max_pending_restores=2,
            spark_cache_async_page_capture=True, spark_cache_async_page_capture_lease_mode="connector-jobs",
            spark_cache_async_page_capture_library=native["snapshot_path"],
            spark_cache_async_page_capture_library_sha256=native["snapshot_sha256"],
            spark_cache_async_page_capture_slot_bytes=536870912, spark_cache_async_page_capture_slot_count=2,
            spark_cache_async_page_capture_vllm_root=native["vllm_root"],
            spark_cache_async_page_capture_lease_contract=native["lease_contract"],
            spark_cache_page_snapshot_interval_tokens=0)
        arguments += ["--enable-prompt-tokens-details", "--kv-transfer-config", json.dumps(dict(
            kv_connector="SparkContextCacheConnector", kv_connector_module_path="sparkcache.spark_context_cache_connector",
            kv_role="kv_both", kv_load_failure_policy="recompute", kv_connector_extra_config=extra))]
    else:
        arguments = _remove_option(arguments, "--model-loader-extra-config")
    container_args = ["serve", *arguments]
    command = list(plan["command"])
    name = f"sparkring-r33-{profile_name}-r{environment['NODE_RANK']}"
    command[command.index("--name") + 1] = name
    command[command.index("--entrypoint") + 1] = "/opt/sparkring/bin/sparkring-r33"
    labels = dict(plan["labels"])
    labels["org.sparkring.profile"] = profile_name
    image_index = len(command) - len(plan["container_args"]) - 1
    prefix = command[:image_index]
    for key, value in sorted(environment.items()):
        assignment = key + "="
        positions = [index for index in range(len(prefix) - 1) if prefix[index] == "--env" and prefix[index + 1].startswith(assignment)]
        if positions:
            prefix[positions[0] + 1] = assignment + value
        else:
            prefix.extend(["--env", assignment + value])
    for key, value in labels.items():
        assignment = key + "="
        for index in range(len(prefix) - 1):
            if prefix[index] == "--label" and prefix[index + 1].startswith(assignment):
                prefix[index + 1] = assignment + value
                break
    command = [*prefix, plan["image"], *container_args]
    plan.update(
        name=name,
        profile=profile_name, environment=environment, container_args=container_args,
        command=command, labels=labels, entrypoint="/opt/sparkring/bin/sparkring-r33",
        runtime_kind="r33-candidate",
        sparkcache_enabled=sparkcache,
        kv_cache_memory_bytes=kv_memory_bytes,
    )
    if sparkcache:
        plan["qualification"] = {"status": "research-only", "gpu_qualified": False,
                                 "request_context_target": 1048576, "reference_context_limit": 262144}
        try:
            verifier.validate_profile_image_capabilities(receipt, profile_name)
            plan["activation_blockers"] = []
        except ValueError as error:
            plan["activation_blockers"] = [str(error)]
    return plan


def render(rank, master, model_dir, cache_dir, env_file, image, r33_receipt=None, *, r33_sparkcache=False,
           r33_cache_kv_memory_bytes=None):
    if r33_cache_kv_memory_bytes is not None and (
            not r33_sparkcache or type(r33_cache_kv_memory_bytes) is not int
            or r33_cache_kv_memory_bytes not in (7247757312, 8053063680, 9395240960)):
        raise ValueError("R33 cache KV override requires --r33-sparkcache and an explicit 6.75 or 8.75 GiB pin")
    if r33_sparkcache and r33_receipt is None:
        raise ValueError("R33 SparkCache requires an exact R33 image receipt")
    if rank not in (0, 1) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", master):
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
    environment = {**profile["environment"], **read_site(env_file), **transport_environment(rank)}
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
    if rank == 1:
        args.append("--headless")
    name = f"sparkring-glm53-tp2-r{rank}"
    labels = {
        "org.sparkring.memory-guard": "true",
        "org.sparkring.profile": profile["name"],
        "org.sparkring.profile.sha256": profile_hash,
        "org.sparkring.transport.manifest-sha256": profile["transport"]["manifest_sha256"],
    }
    command = ["docker", "create", "--name", name, "--restart", "no",
               "--init", "--no-healthcheck", "--gpus", "all", "--network", "host", "--ipc", "host",
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
    result = {
        "schema": "sparkring-profile-launch-plan/v1", "status": "implemented",
        "name": name, "profile": profile["name"], "profile_sha256": profile_hash,
        "image": image, "transport_manifest_sha256": profile["transport"]["manifest_sha256"],
        "image_identity_kind": "local_config_id" if re.fullmatch(LOCAL_IMAGE_PATTERN, image) else "registry_manifest_digest",
        "labels": labels, "environment": environment, "container_args": container_args,
        "command": command,
        "binds": {"/models/target": str(model_dir.resolve()), "/cache/jit": str(cache_dir.resolve())},
        "memory_guard_floor_bytes": profile["lifecycle"]["memory_guard_floor_bytes"],
        "automatic_restart": False, "automatic_start": False,
        "sparkcache_enabled": False,
        "qualification": profile["qualification"],
        "entrypoint": "python3", "runtime_kind": "legacy",
    }
    return adapt_r33_plan(result, r33_receipt, sparkcache=r33_sparkcache,
                          cache_kv_memory_bytes=r33_cache_kv_memory_bytes) if r33_receipt is not None else result


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
    """Validate common-source and transport witnesses against repository inputs."""
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
    for field in ("profile_sha256", "transport_manifest_sha256"):
        if entry.get(field) != plan[field]:
            raise ValueError("The locked source-image profile differs: " + field)
    if entry.get("tp_size") != 2 or entry.get("dcp_size") != 1:
        raise ValueError("The source lock must select TP2/DCP1 for this profile")
    transport = load_profile()["transport"]
    manifest_bytes = (ROOT / transport["manifest"]).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != plan["transport_manifest_sha256"]:
        raise ValueError("The local transport manifest differs from the profile plan")
    files = json.loads(manifest_bytes)["files"]
    for name, digest in files.items():
        path = ROOT / transport["manifest"]
        if hashlib.sha256((path.parent / name).read_bytes()).hexdigest() != digest:
            raise ValueError("The repository's transport source differs: " + name)
    expected = {
        "manifest_sha256": plan["transport_manifest_sha256"],
        "files_sha256": validator.file_map_hash(files),
        "files": len(files), "package": "b12x.comm.roce",
    }
    witness = receipt.get("inside_image", {}).get("transport_profiles", {}).get(transport["name"], {})
    if not isinstance(witness, dict) or any(witness.get(key) != value for key, value in expected.items()):
        raise ValueError("The installed transport witness differs from the profile bundle")


def validate_runtime_receipt(receipt, plan):
    """Require source compatibility evidence for this exact image and profile."""
    if receipt.get("schema") == "sparkring-r33-image-receipt/v1":
        _r33_verifier().validate_image_receipt(receipt)
        expected = receipt["image_id"] if plan["image_identity_kind"] == "local_config_id" else receipt["image_reference"]
        if plan.get("runtime_kind") != "r33-candidate" or plan["profile"] not in ("tp2-dcp1", "tp2-dcp1-sparkcache") or plan["image"] != expected:
            raise ValueError("R33 receipt differs from the adapted TP2 plan")
        _r33_verifier().validate_profile_image_capabilities(receipt, plan["profile"])
        return
    if re.fullmatch(LOCAL_IMAGE_PATTERN, plan["image"]):
        validate_source_image_receipt(receipt, plan)
        return
    if receipt.get("schema") == "sparkring-source-image-receipt/v1":
        raise ValueError("A local image config receipt is not a registry publication receipt")
    if receipt.get("registry_digest") != plan["image"]:
        raise ValueError("The runtime receipt does not describe the selected image digest")
    entry = receipt.get("profiles", {}).get(plan["profile"], {})
    for field in ("profile_sha256", "transport_manifest_sha256"):
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
    if floor is None or int(floor[1]) != plan["memory_guard_floor_bytes"]:
        raise RuntimeError("The active host memory guard must use the profile's 2 GiB floor")
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
    actual_env = dict(item.split("=", 1) for item in config.get("Env", []))
    actual_mounts = {item["Destination"]: item for item in container.get("Mounts", [])}
    matches = (
        config.get("Image") == plan["image"]
        and config.get("Entrypoint") == [plan["entrypoint"]]
        and config.get("Healthcheck", {}).get("Test") == ["NONE"]
        and config.get("Cmd") == plan["container_args"]
        and all(config.get("Labels", {}).get(key) == value for key, value in plan["labels"].items())
        and all(actual_env.get(key) == value for key, value in plan["environment"].items())
        and all(actual_mounts.get(target, {}).get("Source") == source
                for target, source in plan["binds"].items())
        and actual_mounts.get("/models/target", {}).get("RW") is False
        and actual_mounts.get("/cache/jit", {}).get("RW") is True
        and container["HostConfig"]["RestartPolicy"]["Name"] == "no"
    )
    if not matches:
        raise RuntimeError("Stopped container differs from this profile plan; inspect it before replacement")
    run(["docker", "start", plan["name"]], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "create", "start"))
    parser.add_argument("--rank", type=int, required=True, choices=(0, 1))
    parser.add_argument("--master", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--runtime-receipt", type=Path)
    parser.add_argument("--r33-sparkcache", action="store_true", help="Plan the source-capability-gated TP2 cache composition")
    parser.add_argument("--r33-cache-kv-memory-bytes", type=int, choices=(7247757312, 8053063680, 9395240960),
                        help="Explicit TP2 R33 cache KV pin; 9395240960 selects the 8.75 GiB research configuration")
    args = parser.parse_args()
    runtime_receipt = json.loads(args.runtime_receipt.read_text()) if args.runtime_receipt else None
    r33_receipt = runtime_receipt if runtime_receipt and runtime_receipt.get("schema") == "sparkring-r33-image-receipt/v1" else None
    plan = render(args.rank, args.master, args.model_dir, args.cache_dir, args.env_file, args.image, r33_receipt,
                  r33_sparkcache=args.r33_sparkcache, r33_cache_kv_memory_bytes=args.r33_cache_kv_memory_bytes)
    print(json.dumps(plan, indent=2), flush=True)
    if args.action == "plan":
        return
    if args.runtime_receipt is None:
        parser.error("create and start require --runtime-receipt for this exact local image or registry digest")
    execute(plan, args.action, runtime_receipt)


if __name__ == "__main__":
    main()
