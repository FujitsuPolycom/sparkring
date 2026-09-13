"""Render a Qwen Flash-Next TP2 research launch for a verified generic image.

This adapter does not provision networking or enable GLM cache contracts.
Model shard identity must be verified before selecting an existing snapshot.
Creation refuses existing names; source verification runs inside the image.
"""

from __future__ import annotations
import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime.common import candidate  # noqa: E402

CONFIG_ROOT = ROOT / "profiles/qwen38-flash-next-tp2"
CONFIG_NAMES = ("config.json",)


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate configuration key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def publication():
    return read(ROOT / "runtime/images/compositions/lil-r37-glm-spark/publication.json")


def canonical(profile):
    if profile not in [read(CONFIG_ROOT / name) for name in CONFIG_NAMES]:
        raise ValueError("Select an unchanged canonical Qwen configuration")
    if profile.get("schema") != "sparkring-serving-profile/v1" or profile.get("topology") != "direct-pair-2":
        raise ValueError("Invalid Qwen serving profile schema/topology")
    return profile


def site_inputs(rank, master, host_ip, interface, model, cache):
    if type(rank) is not int or rank not in (0, 1):
        raise ValueError("Select rank0/1")
    try:
        ipaddress.ip_address(host_ip)
    except ValueError as exc:
        raise ValueError("host-ip must be a concrete IP address") from exc
    try:
        ipaddress.ip_address(master)
    except ValueError:
        if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?", master):
            raise ValueError("master must be a concrete IP address or hostname")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,14}", interface):
        raise ValueError("interface must be one explicit network interface")
    paths = []
    for value in (model, cache):
        value = str(value)
        if not value or "," in value or any(ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError("Mount paths cannot contain delimiters or control characters")
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("Model/cache paths must be absolute")
        paths.append(path.resolve())
    if paths[0].is_relative_to(paths[1]) or paths[1].is_relative_to(paths[0]):
        raise ValueError("Model/cache paths must be disjoint")
    return paths


def verify_image(image, *, run=subprocess.run):
    expected = publication()
    if image != expected["image_id"]:
        raise ValueError("Image differs from the registered R37 publication")
    info = json.loads(run(["docker", "image", "inspect", image], check=True, capture_output=True, text=True).stdout)[0]
    if info.get("Id") != image or info.get("Os") != "linux" or info.get("Architecture") != "arm64":
        raise ValueError("Image identity or Linux ARM64 platform differs")
    if info.get("Config", {}).get("Entrypoint") != ["/opt/venv/bin/python", candidate.ENTRYPOINT]:
        raise ValueError("Image does not use the verified generic candidate entrypoint")
    raw = run(["docker", "run", "--rm", "--pull", "never", "--network", "none", "--entrypoint", "/bin/cat", image,
               "/opt/sparkring/receipts/candidate-installed.json"], check=True, capture_output=True).stdout
    verification = json.loads(run(["docker", "run", "--rm", "--pull", "never", "--network", "none", image, "verify"],
                                  check=True, capture_output=True, text=True).stdout)
    return candidate.make_receipt(image, raw, verification)


def render(profile, *, rank, master, host_ip, interface, image, model, cache):
    canonical(profile)
    model, cache = site_inputs(rank, master, host_ip, interface, model, cache)
    if image != publication()['image_id']:
        raise ValueError('Select the exact registered R37 image ID')
    namespace = f"qwen-flash-next-{image[7:19]}-{profile['model']['revision'][:12]}"
    env = dict(profile["environment"])
    env.update(
        VLLM_HOST_IP=host_ip,
        VLLM_SPARK_TP4_MODE="", VLLM_SPARK_TP4_VOCAB_MODE="", SIRCL_ENABLED="0",
        NCCL_SOCKET_IFNAME=interface,
        GLOO_SOCKET_IFNAME=interface,
        B12X_ROCE_PEER_HCA_MAP=f"{1 - rank}=0/1",
        SPARKRING_TRANSPORT_PROFILE="tp2-rocenante-adaptive",
        SPARKRING_TRANSPORT_MANIFEST_SHA256="eb03cfde826974811be3bfe5d88f36d9de105b73358f3eaa56b9ed44f19127c4",
        XDG_CACHE_HOME=f"/cache/{namespace}",
        B12X_ROCE_CACHE_DIR=f"/cache/{namespace}/roce",
        VLLM_CACHE_ROOT=f"/cache/{namespace}/vllm",
        TRITON_CACHE_DIR=f"/cache/{namespace}/triton",
        B12X_COMPILE_CACHE_DIR=f"/cache/{namespace}/b12x",
        CUTE_DSL_CACHE_DIR=f"/cache/{namespace}/cute",
    )
    command = [
        "docker",
        "create",
        "--name",
        f"qwen-flash-next-tp2-r{rank}",
        "--entrypoint",
        "/opt/venv/bin/python",
        "--pull",
        "never",
        "--restart",
        "no",
        "--init",
        "--no-healthcheck",
        "--gpus",
        "all",
        "--network",
        "host",
        "--ipc",
        "host",
        "--device",
        "/dev/infiniband",
        "--ulimit",
        "memlock=-1:-1",
        "--memory",
        "108g",
        "--memory-swap",
        "112g",
        "--mount",
        f"type=bind,src={model},dst=/models/target,readonly",
        "--mount",
        f"type=bind,src={cache},dst=/cache",
    ]
    for key, value in sorted(env.items()):
        command += ["--env", key + "=" + value]
    args = [
        candidate.ENTRYPOINT,
        "serve",
        "/models/target",
        "--served-model-name",
        profile["served_model_name"],
        "--node-rank",
        str(rank),
        "--master-addr",
        master,
        *profile["vllm_args"],
    ]
    if rank:
        args += ["--headless"]
    return command + [image, *args]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["plan", "check", "create"])
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--rank", type=int, required=True)
    for key in ("master", "host-ip", "interface", "image", "model", "cache"):
        p.add_argument("--" + key, required=True)
    o = p.parse_args()
    profile = canonical(read(o.profile))
    model, cache = site_inputs(o.rank, o.master, o.host_ip, o.interface, o.model, o.cache)
    if not model.is_dir() or not cache.is_dir():
        raise ValueError("Existing model and dedicated cache directories are required")
    for filename, key in [
        ("config.json", "config_sha256"),
        ("model.safetensors.index.json", "index_sha256"),
    ]:
        with (Path(o.model) / filename).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != profile["model"][key]:
                raise ValueError("Checkpoint metadata mismatch: " + filename)
    if not Path(o.cache).is_dir():
        raise ValueError("Create a dedicated compilation cache directory")
    command = render(
        profile,
        rank=o.rank,
        master=o.master,
        host_ip=o.host_ip,
        interface=o.interface,
        image=o.image,
        model=o.model,
        cache=o.cache,
    )
    print(json.dumps(command), flush=True)
    if o.action != "plan":
        verify_image(o.image)
    if o.action == "check":
        print("CLI help check only; no inference, cache or performance qualification.", flush=True)
        command[1] = "run"
        command.insert(2, "--rm")
        command += ["--help"]
        subprocess.run(command, check=True)
    elif o.action == "create":
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
