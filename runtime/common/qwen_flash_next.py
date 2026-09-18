"""Render Qwen Flash-Next pair and ring deployments from canonical profiles.

This adapter does not provision networking or enable GLM cache contracts.
Model shard identity must be verified before selecting an existing snapshot.
Creation refuses existing names; source verification runs inside the image.
"""

from __future__ import annotations
import argparse
import hashlib
import ipaddress
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime.common import candidate  # noqa: E402
from runtime.common import cache_candidate  # noqa: E402
from runtime.common.container_spec import Bind, ContainerSpec, docker_create  # noqa: E402

CONFIG_ROOT = ROOT / "profiles/qwen38-flash-next-tp2"
CONFIG_NAMES = ("config.json", "sparkcache.json")
TP4_CONFIG = ROOT / "profiles/qwen38-flash-next-qad-tp4/config.json"
TP4_CACHE_CONFIG = ROOT / "profiles/qwen38-flash-next-qad-tp4/sparkcache.json"


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
    if (profile not in [read(CONFIG_ROOT / name) for name in CONFIG_NAMES]
            and profile not in [read(TP4_CONFIG), read(TP4_CACHE_CONFIG)]):
        raise ValueError("Select an unchanged canonical Qwen configuration")
    if profile.get("schema") != "sparkring-serving-profile/v1" or profile.get("topology") not in ("direct-pair-2", "direct-cycle-4"):
        raise ValueError("Invalid Qwen serving profile schema/topology")
    return profile


def node_count(profile):
    return int(profile["vllm_args"][profile["vllm_args"].index("--nnodes") + 1])


def image_policy(profile, *, local_source_extension=None):
    """Resolve one image kind for Docker, Compose and host admission."""
    kinds = {None: "base", "lil-r37-cache64": "cache", "lil-r37-shared": "feature"}
    extension = profile.get("image_extension")
    if extension not in kinds:
        from runtime.common import source_candidate
        if extension != source_candidate.IDENTITY:
            raise ValueError("Unregistered Qwen image extension")
        kinds[extension] = "source"
    identity = (local_source_extension if local_source_extension is not None else
                extension if kinds[extension] == "source" else None)
    if identity is not None:
        from runtime.common import source_candidate
        source_candidate.descriptor(identity)
        nodes = source_candidate.profile_nodes(profile)
        if nodes == 2 and local_source_extension is None:
            raise ValueError("TP2 source-image selection requires an explicit local source extension")
    return {"kind": "source" if identity is not None else kinds[extension],
            "source_extension": identity, "local": local_source_extension is not None}


def image_verification_options(profile, *, local_source_extension=None):
    policy = image_policy(profile, local_source_extension=local_source_extension)
    options = {"cache_enabled": policy["kind"] == "cache", "feature_enabled": policy["kind"] == "feature"}
    if policy["kind"] == "source":
        key = "local_source_extension" if policy["local"] else "source_extension"
        options[key] = policy["source_extension"]
    return options


def site_inputs(rank, master, host_ip, interface, model, cache, *, remote=False, nodes=2):
    if nodes not in (2, 4) or type(rank) is not int or rank not in range(nodes):
        raise ValueError("Select rank0/1" if nodes == 2 else "Select rank0/1/2/3 for a four-node profile")
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
        path = PurePosixPath(value) if remote else Path(value)
        if remote and (str(path) != value or ".." in path.parts or "\\" in value):
            raise ValueError("Remote mount paths must be normalized Linux paths")
        if not path.is_absolute():
            raise ValueError("Model/cache paths must be absolute")
        paths.append(path if remote else path.resolve())
    if paths[0].is_relative_to(paths[1]) or paths[1].is_relative_to(paths[0]):
        raise ValueError("Model/cache paths must be disjoint")
    return paths


def verify_image(image, *, cache_enabled=False, feature_enabled=False,
                 local_source_extension=None, source_extension=None, run=subprocess.run):
    expected = publication()
    if local_source_extension is not None and source_extension is not None:
        raise ValueError("Choose one local or published source selection")
    selected_source = local_source_extension or source_extension
    if source_extension is not None:
        from runtime.common import source_candidate
        source_candidate.publication(source_extension, image_id=image)
    if not cache_enabled and not feature_enabled and selected_source is None and image != expected["image_id"]:
        raise ValueError("Image differs from the registered R37 publication")
    info = json.loads(run(["docker", "image", "inspect", image], check=True, capture_output=True, text=True).stdout)[0]
    if info.get("Id") != image or info.get("Os") != "linux" or info.get("Architecture") != "arm64":
        raise ValueError("Image identity or Linux ARM64 platform differs")
    entrypoint = candidate.ENTRYPOINT
    if selected_source is not None:
        from runtime.common import source_candidate
        source_candidate.image_reference(selected_source, image)
        entrypoint = source_candidate.ENTRYPOINT
    if info.get("Config", {}).get("Entrypoint") != ["/opt/venv/bin/python", entrypoint]:
        raise ValueError("Image does not use the verified generic candidate entrypoint")
    raw = run(["docker", "run", "--rm", "--pull", "never", "--network", "none", "--entrypoint", "/bin/cat", image,
               "/opt/sparkring/receipts/candidate-installed.json"], check=True, capture_output=True).stdout
    verification = json.loads(run(["docker", "run", "--rm", "--pull", "never", "--network", "none", image, "verify"],
                                  check=True, capture_output=True, text=True).stdout)
    if selected_source is not None:
        from runtime.common import feature_candidate, source_candidate
        receipts = []
        for receipt_image, receipt_path in (
            (image, source_candidate.PARENT_RECEIPT),
            (image, feature_candidate.PARENT_RECEIPT),
            (expected["image_id"], "/opt/sparkring/receipts/candidate-installed.json"),
        ):
            receipts.append(run([
                "docker", "run", "--rm", "--pull", "never", "--network", "none",
                "--entrypoint", "/bin/cat", receipt_image, receipt_path,
            ], check=True, capture_output=True).stdout)
        return source_candidate.validate(image, raw, *receipts, verification, identity=selected_source)
    if cache_enabled or feature_enabled:
        parent = run(["docker", "run", "--rm", "--pull", "never", "--network", "none",
                      "--entrypoint", "/bin/cat", expected["image_id"],
                      "/opt/sparkring/receipts/candidate-installed.json"],
                     check=True, capture_output=True).stdout
        if feature_enabled:
            from runtime.common import feature_candidate
            retained = run(["docker", "run", "--rm", "--pull", "never", "--network", "none",
                            "--entrypoint", "/bin/cat", image, feature_candidate.PARENT_RECEIPT],
                           check=True, capture_output=True).stdout
            features = json.loads(run([
                "docker", "run", "--rm", "--pull", "never", "--network", "none",
                "--entrypoint", "/opt/venv/bin/python", image, feature_candidate.INSTALLER, "verify",
            ], check=True, capture_output=True, text=True).stdout)
            return feature_candidate.validate(image, raw, retained, parent, verification, features)
        return cache_candidate.validate(image, raw, parent, verification)
    return candidate.make_receipt(image, raw, verification)


def container_spec(profile, *, rank, master, host_ip, interface, image, model, cache,
                   remote=False, hcas=None, gid=None, local_source_extension=None,
                   local_kv_cache_gib=None, local_master_port=None):
    canonical(profile)
    policy = image_policy(profile, local_source_extension=local_source_extension)
    entrypoint = candidate.ENTRYPOINT
    if policy["kind"] == "source":
        from runtime.common import source_candidate
        if policy["local"]:
            source_candidate.image_reference(local_source_extension, image)
            profile = source_candidate.profile_settings(
                profile, local_source_extension, local_kv_cache_gib, local_master_port,
            )
        else:
            source_candidate.publication(policy["source_extension"], image_id=image)
            if local_kv_cache_gib is not None or local_master_port is not None:
                raise ValueError("Local KV and master-port alternatives require a local source extension")
        source_candidate.validate_profile_contract(profile, policy["source_extension"])
        entrypoint = source_candidate.ENTRYPOINT
    elif local_kv_cache_gib is not None or local_master_port is not None:
        raise ValueError("Local KV and master-port alternatives require a local source extension")
    nodes = node_count(profile)
    model, cache = site_inputs(rank, master, host_ip, interface, model, cache, remote=remote, nodes=nodes)
    cache_enabled = policy["kind"] == "cache"
    feature_enabled = policy["kind"] == "feature"
    if (cache_enabled or feature_enabled) and (not re.fullmatch(r'sha256:[0-9a-f]{64}', image) or image == publication()['image_id']):
        kind = 'feature-extension' if feature_enabled else 'cache-extension'
        raise ValueError(f'Select an immutable {kind} image, not the base R37 image')
    if policy["kind"] == "base" and image != publication()['image_id']:
        raise ValueError('Select the exact registered R37 image ID')
    namespace = f"qwen-flash-next-{image[7:19]}-{profile['model']['revision'][:12]}"
    env = dict(profile["environment"])
    env.update(
        VLLM_HOST_IP=host_ip,
        VLLM_SPARK_TP4_MODE="", VLLM_SPARK_TP4_VOCAB_MODE="", SIRCL_ENABLED="0",
        NCCL_SOCKET_IFNAME=interface,
        GLOO_SOCKET_IFNAME=interface,
        B12X_ROCE_PEER_HCA_MAP=(profile["transport"]["peer_hca_maps"][rank] if nodes == 4 else f"{1 - rank}=0/1"),
        SPARKRING_TRANSPORT_PROFILE="tp2-rocenante-adaptive",
        SPARKRING_TRANSPORT_MANIFEST_SHA256="eb03cfde826974811be3bfe5d88f36d9de105b73358f3eaa56b9ed44f19127c4",
        XDG_CACHE_HOME=f"/cache/{namespace}",
        B12X_ROCE_CACHE_DIR=f"/cache/{namespace}/roce",
        VLLM_CACHE_ROOT=f"/cache/{namespace}/vllm",
        TRITON_CACHE_DIR=f"/cache/{namespace}/triton",
        B12X_COMPILE_CACHE_DIR=f"/cache/{namespace}/b12x",
        CUTE_DSL_CACHE_DIR=f"/cache/{namespace}/cute",
        TORCHINDUCTOR_CACHE_DIR=f"/cache/{namespace}/inductor",
    )
    if hcas is not None:
        if (not isinstance(hcas, list) or len(hcas) != nodes or len(set(hcas)) != nodes
                or any(not re.fullmatch(r"[A-Za-z0-9_]{1,64}", hca) for hca in hcas)):
            raise ValueError("Select the profile's distinct HCA functions in cable order")
        env["B12X_ROCE_HCA"] = ",".join(hcas)
        env["NCCL_IB_HCA"] = "=" + ",".join(hca + (":1" if nodes == 4 else "") for hca in hcas)
    if gid is not None:
        if type(gid) is not int or not 0 <= gid <= 255:
            raise ValueError("GID index must be an integer from 0 to 255")
        env["NCCL_IB_GID_INDEX"] = str(gid)
        if nodes == 4:
            env["B12X_ROCE_GID_INDEX"] = str(gid)
    args = [
        entrypoint,
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
    port = profile["vllm_args"][profile["vllm_args"].index("--port") + 1]
    health = () if rank else (
        "/opt/venv/bin/python", "-c",
        f"import urllib.request; urllib.request.urlopen('http://127.0.0.1:{port}/health', timeout=4).close()",
    )
    prefix = "qad-sparkcache-" if nodes == 4 and env.get("SPARKCACHE_ENABLED") == "1" else "qad-" if nodes == 4 else "sparkcache-" if env.get("SPARKCACHE_ENABLED") == "1" else ""
    return ContainerSpec(
        name=f"qwen-flash-next-{prefix}tp{nodes}-r{rank}",
        image_id=image, entrypoint=("/opt/venv/bin/python",), command=tuple(args),
        environment=env, mounts=(Bind(str(model), "/models/target", True), Bind(str(cache), "/cache")),
        health_command=health,
    )


def render(profile, **site):
    return docker_create(container_spec(profile, **site))


def verify_model_paths(profile, model, cache):
    """Check local directories and checkpoint metadata; full shard checks remain explicit."""
    model, cache = Path(model), Path(cache)
    if not model.is_dir() or not cache.is_dir():
        raise ValueError("Existing model and dedicated cache directories are required")
    for filename, key in [
        ("config.json", "config_sha256"),
        ("model.safetensors.index.json", "index_sha256"),
    ]:
        with (model / filename).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != profile["model"][key]:
                raise ValueError("Checkpoint metadata mismatch: " + filename)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["plan", "check", "create"])
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--rank", type=int, required=True)
    for key in ("master", "host-ip", "interface", "image", "model", "cache"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--local-source-extension")
    p.add_argument("--local-kv-cache-gib", type=int)
    p.add_argument("--local-master-port", type=int)
    o = p.parse_args()
    profile = canonical(read(o.profile))
    model, cache = site_inputs(o.rank, o.master, o.host_ip, o.interface, o.model, o.cache, nodes=node_count(profile))
    verify_model_paths(profile, model, cache)
    command = render(
        profile,
        rank=o.rank,
        master=o.master,
        host_ip=o.host_ip,
        interface=o.interface,
        image=o.image,
        model=o.model,
        cache=o.cache,
        local_source_extension=o.local_source_extension,
        local_kv_cache_gib=o.local_kv_cache_gib,
        local_master_port=o.local_master_port,
    )
    print(json.dumps(command), flush=True)
    if o.action != "plan":
        verify_image(o.image, **image_verification_options(profile, local_source_extension=o.local_source_extension))
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
