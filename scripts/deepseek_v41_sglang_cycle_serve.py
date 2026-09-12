#!/usr/bin/env python3
"""Render or launch one SGLang rank from a literal, private environment file.

--check is offline. --prepare patches authentication using a CPU-only container.
--pack builds Mia-layout Engram files on local NVMe. --run requires an idle GPU.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime/deepseek-v41-sglang"
PINS = json.loads((RUNTIME / "pins.json").read_text())
SERVING = json.loads((ROOT / "profiles/deepseek-v41-flash-sglang-cycle/recipe.json").read_text())["serving"]
REQUIRED = {
    "NODE_RANK", "MASTER_ADDR", "HOST_IP", "MODEL_HOST_PATH", "ENGRAM_HOST_PATH",
    "STATE_HOST_PATH", "CACHE_HOST_PATH", "API_KEY_FILE", "NCCL_SO_HOST_PATH",
    "NCCL_SO_SHA256", "IMAGE", "IMAGE_ID", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME",
    "NCCL_IB_HCA", "NCCL_IB_GID_INDEX",
}
DEFAULTS = {
    "API_PORT": "8000", "MASTER_PORT": "20000", "CONTEXT_LENGTH": str(SERVING["context_length"]),
    "CHUNKED_PREFILL_SIZE": str(SERVING["chunked_prefill_size"]),
    "MAX_RUNNING_REQUESTS": str(SERVING["max_running_requests"]),
    "MAX_TOTAL_TOKENS": str(SERVING["max_total_tokens"]),
    "MEM_FRACTION_STATIC": str(SERVING["mem_fraction_static"]),
    "DSPARK_SPS_TABLE": "/state/dspark_sps.json", "DSPARK_STS_TABLE": "/state/dspark_sts.json",
}
TRANSPORT = {
    "NCCL_NET": "IB", "NCCL_NET_PLUGIN": "none", "NCCL_IB_DISABLE": "0",
    "NCCL_IB_SUBNET_AWARE_ROUTING": "1", "NCCL_IB_SUBNET_PREFIX_LEN": "24",
    "NCCL_IB_MERGE_NICS": "0", "NCCL_CROSS_NIC": "1", "NCCL_P2P_LEVEL": "SYS",
    "NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1", "NCCL_PROTO": "LL,LL128,Simple",
    "NCCL_ALGO": "Ring", "NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4",
    "NCCL_SKIP_TREE_CONNECT": "1", "NCCL_SWITCHLESS_RING_ONLY": "1",
    "NCCL_CUMEM_ENABLE": "0", "NCCL_IGNORE_CPU_AFFINITY": "1", "NCCL_DEBUG": "WARN",
    "NCCL_DEBUG_SUBSYS": "INIT",
}


def read_config(path):
    cfg = dict(DEFAULTS)
    seen = set()
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or key not in REQUIRED | DEFAULTS.keys() or key in seen:
            raise ValueError(f"invalid, unknown, or duplicate setting at line {number}")
        if not value or any(c in value for c in "<>\n\r\x00"):
            raise ValueError(f"unresolved or empty setting: {key}")
        cfg[key] = value
        seen.add(key)
    if REQUIRED - seen:
        raise ValueError("missing settings: " + ", ".join(sorted(REQUIRED - seen)))
    limits = {"NODE_RANK": (0, 3), "API_PORT": (1, 65535), "MASTER_PORT": (1, 65535),
              "CONTEXT_LENGTH": (4096, 1048576), "CHUNKED_PREFILL_SIZE": (128, 65536),
              "MAX_RUNNING_REQUESTS": (1, 64), "MAX_TOTAL_TOKENS": (4096, 10000000),
              "NCCL_IB_GID_INDEX": (0, 255)}
    for key, (low, high) in limits.items():
        if not re.fullmatch(r"[0-9]+", cfg[key]) or not low <= int(cfg[key]) <= high:
            raise ValueError(f"invalid numeric setting: {key}")
    if int(cfg["API_PORT"]) == int(cfg["MASTER_PORT"]):
        raise ValueError("API_PORT and MASTER_PORT must differ")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", cfg["MASTER_ADDR"]):
        raise ValueError("MASTER_ADDR must be an IPv4 address or hostname")
    if not 0.1 <= float(cfg["MEM_FRACTION_STATIC"]) < 1:
        raise ValueError("invalid MEM_FRACTION_STATIC")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", cfg["IMAGE_ID"]):
        raise ValueError("IMAGE_ID must pin a complete Docker image ID")
    if not re.fullmatch(r"[0-9a-f]{64}", cfg["NCCL_SO_SHA256"]):
        raise ValueError("NCCL_SO_SHA256 must pin the library contents")
    for key in REQUIRED:
        if key.endswith("_PATH") or key == "API_KEY_FILE":
            if (not PurePosixPath(cfg[key]).is_absolute() or ":" in cfg[key]
                    or "\\" in cfg[key] or ".." in PurePosixPath(cfg[key]).parts):
                raise ValueError(f"{key} must be an absolute bind-mount path without colons")
    paths = {k: PurePosixPath(cfg[k]) for k in REQUIRED if k.endswith("_PATH") or k == "API_KEY_FILE"}
    if len(set(paths.values())) != len(paths):
        raise ValueError("Bind-mount paths must be distinct")
    model = paths["MODEL_HOST_PATH"]
    for key in ("ENGRAM_HOST_PATH", "STATE_HOST_PATH", "CACHE_HOST_PATH"):
        path = paths[key]
        if path.is_relative_to(model) or model.is_relative_to(path):
            raise ValueError(f"{key} must not overlap the model directory")
    return cfg


def command(cfg):
    state = PurePosixPath(cfg["STATE_HOST_PATH"])
    mounts = {
        cfg["MODEL_HOST_PATH"]: "/models/DeepSeek-V4.1-Flash:ro",
        cfg["ENGRAM_HOST_PATH"]: "/engram:ro", str(state): "/state",
        cfg["CACHE_HOST_PATH"]: "/root/.cache",
        cfg["API_KEY_FILE"]: "/run/secrets/api-keys:ro",
        str(state / "operator/auth.py"): PINS["auth_path"] + ":ro",
        str(RUNTIME / "entrypoint.py"): "/operator/entrypoint.py:ro",
        cfg["NCCL_SO_HOST_PATH"]: PINS["nccl_target"] + ":ro",
    }
    env = {**TRANSPORT, **{k: cfg[k] for k in (
        "NODE_RANK", "HOST_IP", "CONTEXT_LENGTH", "CHUNKED_PREFILL_SIZE",
        "MAX_RUNNING_REQUESTS", "MAX_TOTAL_TOKENS", "MEM_FRACTION_STATIC",
        "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX",
        "DSPARK_SPS_TABLE", "DSPARK_STS_TABLE")},
        "API_KEY_FILE": "/run/secrets/api-keys", "SERVER_PORT": cfg["API_PORT"],
        "DIST_INIT_ADDR": cfg["MASTER_ADDR"] + ":" + cfg["MASTER_PORT"],
        "VLLM_HOST_IP": cfg["HOST_IP"], "NNODES": "4", "TP_SIZE": "4", "EP_SIZE": "4",
        "HOST": "0.0.0.0", "SERVED_MODEL_NAME": "deepseek-v4.1-flash",
        "MODEL_PATH": "/models/DeepSeek-V4.1-Flash", "DSV41_SOURCE": "/models/DeepSeek-V4.1-Flash",
        "STATE_PATH": "/state", "OFFLOAD_MODE": "nvme", "DSV41_PACKED_DIR": "/engram",
        "DSV41_CACHE_GIB": "0", "DSV41_IO_THREADS": "96", "DSV41_RESIDENT_SCALES": "0",
        "DSV41_CACHE_WAYS": "4", "DSV41_STATS_SECONDS": "60", "DSV41_TP_PAD": "0",
        "CUDA_GRAPH_MAX_BS_DECODE": cfg["MAX_RUNNING_REQUESTS"], "SPEC_ALGO": "DSPARK",
        "DSPARK_BLOCK_SIZE": "5", "SKIP_PREPARE": "1", "SKIP_VERIFY": "1",
        "SKIP_SMOKE": "1", "WARMUP": "0", "DSV41_MXFP8_BACKEND": "b12x",
        "SGLANG_FLASHINFER_MOE_FUSED_FINALIZE": "0", "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "EXTRA_SGLANG_ARGS": "--fp8-gemm-backend flashinfer_cutlass --watchdog-timeout 1800 --enable-metrics",
    }
    args = ["docker", "run", "-d", "--name", "sgl_dsv41", "--restart", "no",
            "--label", "family=dsv41", "--label", "engine=sglang", "--network", "host",
            "--ipc", "host", "--privileged", "--cap-add", "IPC_LOCK", "--gpus", "all",
            "--shm-size", "32g", "--ulimit", "memlock=-1:-1", "--ulimit", "stack=67108864",
            "--device", "/dev/infiniband:/dev/infiniband"]
    for source, target in mounts.items():
        args += ["-v", source + ":" + target]
    for key, value in env.items():
        args += ["-e", key + "=" + value]
    return args + ["--entrypoint", "python3", cfg["IMAGE"], "/operator/entrypoint.py", "run"]


def output(args):
    return subprocess.check_output(args, text=True, timeout=20).strip()


def verify_image(cfg):
    if output(["docker", "image", "inspect", cfg["IMAGE"], "--format", "{{.Id}}"]) != cfg["IMAGE_ID"]:
        raise ValueError("image identity mismatch")


def verify_host_paths(cfg):
    model = Path(cfg["MODEL_HOST_PATH"]).resolve()
    writable = {key: Path(cfg[key]) for key in
                ("ENGRAM_HOST_PATH", "STATE_HOST_PATH", "CACHE_HOST_PATH")}
    operator = Path(cfg["STATE_HOST_PATH"]) / "operator"
    writable.update({"auth output": operator / "auth.py", "auth temporary output": operator / "auth.py.tmp"})
    for name, path in writable.items():
        resolved = path.resolve()
        if resolved.is_relative_to(model) or model.is_relative_to(resolved):
            raise ValueError(f"{name} must not overlap the resolved model directory")


def auth_record(cfg, source):
    return {
        "schema": "sparkring-sglang-auth/v1", "image_id": cfg["IMAGE_ID"],
        "auth_path": PINS["auth_path"], "auth_sha256": hashlib.sha256(source).hexdigest(),
        "patch_sha256": hashlib.sha256((RUNTIME / "patch-multikey.py").read_bytes()).hexdigest(),
        "pins_sha256": hashlib.sha256((RUNTIME / "pins.json").read_bytes()).hexdigest(),
    }


def write_prepared_auth(cfg, source):
    operator = Path(cfg["STATE_HOST_PATH"]) / "operator"
    compile(source, str(operator / "auth.py"), "exec")
    operator.mkdir(parents=True, exist_ok=True)
    operator.parent.chmod(0o700)
    operator.chmod(0o700)
    payloads = {"auth.py": source, "auth-receipt.json":
                (json.dumps(auth_record(cfg, source), sort_keys=True) + "\n").encode()}
    for name, payload in payloads.items():
        with tempfile.NamedTemporaryFile(dir=operator, prefix=".auth-", delete=False) as stream:
            staged = Path(stream.name)
            try:
                stream.write(payload)
            except BaseException:
                stream.close()
                staged.unlink(missing_ok=True)
                raise
        try:
            staged.replace(operator / name)
        finally:
            staged.unlink(missing_ok=True)


def verify_prepared_auth(cfg):
    operator = Path(cfg["STATE_HOST_PATH"]) / "operator"
    try:
        source = (operator / "auth.py").read_bytes()
        receipt = json.loads((operator / "auth-receipt.json").read_bytes())
        if receipt == auth_record(cfg, source):
            return
    except (OSError, ValueError):
        pass
    raise ValueError("prepared authentication differs or is missing; run --prepare for the selected image")


def verify_host(cfg):
    verify_host_paths(cfg)
    verify_image(cfg)
    if not (Path(cfg["MODEL_HOST_PATH"]) / "config.json").is_file():
        raise ValueError("missing checkpoint config.json")
    for i in range(1, 49):
        path = Path(cfg["MODEL_HOST_PATH"]) / f"model-{i:05d}-of-00048.safetensors"
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"missing checkpoint shard {i}")
    for layer in (1, 14):
        path = Path(cfg["ENGRAM_HOST_PATH"]) / f"engram-l{layer}-r{cfg['NODE_RANK']}of4.bin"
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"missing Mia-layout Engram layer {layer}")
    library = Path(cfg["NCCL_SO_HOST_PATH"]).resolve(strict=True)
    with library.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != cfg["NCCL_SO_SHA256"]:
        raise ValueError("NCCL content identity mismatch")
    keys = [line.strip() for line in Path(cfg["API_KEY_FILE"]).read_text().splitlines() if line.strip()]
    if not keys or len(keys) != len(set(keys)) or any("," in k or any(not 33 <= ord(c) <= 126 for c in k) for k in keys):
        raise ValueError("key file must contain distinct nonempty keys without commas or whitespace")
    verify_prepared_auth(cfg)
    names = output(["docker", "ps", "--format", "{{.Names}}"]).splitlines()
    if any(name.startswith(("vllm", "sgl", "glm")) for name in names):
        raise ValueError("a model container is already running; stop its owning service first")
    if output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]):
        raise ValueError("GPU has active compute processes; stop the owning service first")
    if output(["docker", "ps", "-aq", "--filter", "name=^/sgl_dsv41$"]):
        raise ValueError("sgl_dsv41 already exists; inspect and remove it explicitly")
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    if int(mem["MemAvailable"].split()[0]) < 100 * 1048576:
        raise ValueError("less than 100 GiB MemAvailable; recover the idle host before launch")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    modes = ap.add_mutually_exclusive_group(required=True)
    for name in ("check", "prepare", "pack", "run"):
        modes.add_argument("--" + name, action="store_true")
    ap.add_argument("environment")
    args = ap.parse_args()
    cfg = read_config(args.environment)
    if args.check:
        print(shlex.join(command(cfg)))
    elif args.prepare:
        verify_host_paths(cfg)
        verify_image(cfg)
        source = subprocess.check_output([
            "docker", "run", "--rm", "--memory", "512m", "--entrypoint", "python3",
            "-v", str(RUNTIME / "patch-multikey.py") + ":/patch.py:ro", cfg["IMAGE"],
            "-S", "/patch.py", PINS["auth_path"]])
        write_prepared_auth(cfg, source)
        print("Authentication prepared; no GPU used")
    elif args.pack:
        verify_host_paths(cfg)
        verify_image(cfg)
        packed = Path(cfg["ENGRAM_HOST_PATH"])
        packed.mkdir(parents=True, exist_ok=True)
        if any(packed.iterdir()):
            raise ValueError("packing requires an empty Engram directory")
        subprocess.run(["docker", "run", "--rm", "--memory", "8g", "--entrypoint", "python3",
            "-v", cfg["MODEL_HOST_PATH"] + ":/models/DeepSeek-V4.1-Flash:ro",
            "-v", str(packed) + ":/engram", "-e", "PYTHONPATH=/opt/sglang/lib/python3.12/site-packages",
            cfg["IMAGE"], "-S", "/opt/dsv41/scripts/pack_engram.py",
            "--rank", cfg["NODE_RANK"], "--tp", "4", "--out", "/engram"], check=True)
    else:
        verify_host(cfg)
        Path(cfg["CACHE_HOST_PATH"]).mkdir(parents=True, exist_ok=True)
        subprocess.run(command(cfg), check=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"SGLang launch failed: {exc}", file=sys.stderr)
        sys.exit(1)
