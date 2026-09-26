"""Render Qwen Flash-Next pair and ring deployments from canonical profiles.

This adapter does not provision networking or enable GLM cache contracts.
Model shard identity must be verified before selecting an existing snapshot.
Creation refuses existing names; source verification runs inside the image.
"""

from __future__ import annotations
import argparse
import copy
import hashlib
import ipaddress
import json
import os
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
# Serving profiles for the shared toolchain image. They reuse this module's
# rank and transport envelope; the installer image lock supplies the image's
# entrypoint, NCCL/CUDA paths, status plugin and runtime binding.
TOOLCHAIN_CONFIGS = tuple(ROOT / "profiles" / name / "config.json" for name in (
    "glm53-flash-nvfp4-spark-tp2", "glm53-flash-nvfp4-spark-tp4",
    "mimo-v26-flash-rl-tp2", "mimo-v26-flash-rl-tp4",
    "qwen38-flash-next-tp2", "qwen38-flash-next-qad-tp4"))
TOOLCHAIN_ENTRYPOINT = "/opt/sparkring/toolchain/toolchain.py"


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
            and profile not in [read(TP4_CONFIG), read(TP4_CACHE_CONFIG)]
            and profile not in [read(path) for path in TOOLCHAIN_CONFIGS]):
        raise ValueError("Select an unchanged canonical serving configuration")
    if profile.get("schema") != "sparkring-serving-profile/v1" or profile.get("topology") not in ("direct-pair-2", "direct-cycle-4"):
        raise ValueError("Invalid serving profile schema/topology")
    return profile


def node_count(profile):
    return int(profile["vllm_args"][profile["vllm_args"].index("--nnodes") + 1])


CHECKPOINT_KEYS = {"model", "environment", "speculative"}
MODEL_KEYS = {"repository", "revision", "config_sha256", "index_sha256"}


def checkpoint_names(profile):
    """(default, names) of a profile's checkpoint table; (None, ()) without one.

    The table, `checkpoints`, maps each name (a Hugging Face branch of the
    profile's repository) to its pinned `model` and optional `environment` and
    `speculative` settings. `checkpoint` names the default entry, whose model is
    the profile's top-level `model` and which changes no setting.
    """
    table = profile.get("checkpoints")
    if table is None:
        if "checkpoint" in profile:
            raise ValueError("A default checkpoint requires a checkpoints table")
        return None, ()
    default = profile.get("checkpoint")
    if not isinstance(table, dict) or default not in table:
        raise ValueError("The checkpoints table must include the default checkpoint")
    for name, entry in table.items():
        if (not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,62}", name) or not isinstance(entry, dict)
                or "model" not in entry or not set(entry) <= CHECKPOINT_KEYS or set(entry["model"]) != MODEL_KEYS
                or entry["model"]["repository"] != profile["model"]["repository"]):
            raise ValueError(f"Invalid checkpoint entry: {name}")
    if table[default] != {"model": profile["model"]}:
        raise ValueError("The default checkpoint must be the profile's model without other settings")
    return default, tuple(sorted(table))


def checkpoint_settings(profile, name):
    """The profile with the named checkpoint's model and settings applied; None keeps the default.

    An entry may change existing environment variables and existing keys of
    --speculative-config, never add new ones, so every checkpoint runs the
    profile's validated command with only its pinned differences.
    """
    default, names = checkpoint_names(profile)
    if name is None or name == default:
        return profile
    if not names:
        raise ValueError("This profile offers no checkpoint choice")
    if name not in names:
        raise ValueError("Select a checkpoint the profile lists: " + ", ".join(names))
    entry = profile["checkpoints"][name]
    result = copy.deepcopy(profile)
    result["model"] = dict(entry["model"])
    environment = entry.get("environment", {})
    if not set(environment) <= set(result["environment"]):
        raise ValueError(f"Checkpoint {name} may only change existing environment settings")
    result["environment"].update(environment)
    speculative = entry.get("speculative", {})
    if speculative:
        args = result["vllm_args"]
        index = args.index("--speculative-config") + 1
        spec = json.loads(args[index])
        if not set(speculative) <= set(spec):
            raise ValueError(f"Checkpoint {name} may only change existing speculative settings")
        spec.update(speculative)
        args[index] = json.dumps(spec, separators=(",", ":"))
    return result


def image_policy(profile, *, local_source_extension=None):
    """Resolve one image kind for Docker, Compose and host admission."""
    if profile.get("image_extension") == "toolchain":
        if local_source_extension is not None:
            raise ValueError("Shared toolchain profiles select their image through the installer image lock")
        return {"kind": "toolchain", "source_extension": None, "local": False}
    if profile.get("image_extension") == "native-shared":
        if local_source_extension is not None:
            from runtime.common import source_candidate
            source_candidate.descriptor(local_source_extension)
            source_candidate.profile_nodes(profile)
            return {"kind": "source", "source_extension": local_source_extension, "local": True}
        from runtime.common import native_candidate
        native_candidate.publication(profile.get("image_release"))
        return {"kind": "native", "source_extension": None, "local": False,
                "native_release": profile["image_release"]}
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
    if policy["kind"] == "toolchain":
        raise ValueError("Shared toolchain images are admitted by the installer image lock")
    if policy["kind"] == "native":
        return {"native_release": policy["native_release"]}
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
                 local_source_extension=None, source_extension=None, native_release=None, run=subprocess.run):
    if native_release is not None:
        if cache_enabled or feature_enabled or local_source_extension or source_extension:
            raise ValueError("Native and legacy image admission cannot be combined")
        from runtime.common import native_candidate
        return native_candidate.verify_image(image, native_release, run=run)
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
                   local_kv_cache_gib=None, local_master_port=None, checkpoint=None):
    canonical(profile)
    profile = checkpoint_settings(profile, checkpoint)
    policy = image_policy(profile, local_source_extension=local_source_extension)
    entrypoint = candidate.ENTRYPOINT
    if policy["kind"] == "toolchain":
        if local_kv_cache_gib is not None or local_master_port is not None:
            raise ValueError("Local KV and master-port alternatives require a local source extension")
        entrypoint = TOOLCHAIN_ENTRYPOINT
    elif policy["kind"] == "native":
        from runtime.common import native_candidate
        native_candidate.publication(policy["native_release"], image_id=image)
        entrypoint = native_candidate.ENTRYPOINT
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
    # Compile and tuning caches are keyed by model family, image and checkpoint
    # revision, so repeated installs of the same selection reuse them.
    family = profile.get("cache_namespace", "qwen-flash-next")
    revision = profile["model"]["revision"][:12]
    namespace = f"{family}-{image[7:19]}-{revision}"
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
    if policy["kind"] == "native":
        native = native_candidate.publication(policy["native_release"], image_id=image)
        env["SPARKRING_TRANSPORT_PROFILE"] = native["transport"]["profile"]
        env["SPARKRING_TRANSPORT_MANIFEST_SHA256"] = native["transport"]["manifest_sha256"]
        env["B12X_CUTE_COMPILE_CACHE_DIR"] = env["B12X_COMPILE_CACHE_DIR"]
    if policy["kind"] == "toolchain":
        # B12X keys each compiled program by its package fingerprint, the Python,
        # torch, CUTLASS DSL and CUDA binding versions, every other B12X_, CUTE_ and
        # CUTLASS_ variable, and the GPU UUID. Installer images therefore share
        # compiled kernels; the folder names the CUDA toolkit, which that key
        # omits. The RoCE proxy library is keyed only by its C source, so it stays
        # in the image's XDG cache: leaving B12X_ROCE_CACHE_DIR unset keeps image
        # paths out of the kernel key.
        from runtime.common import installer_image
        env["B12X_COMPILE_CACHE_DIR"] = f"/cache/{family}-cuda{installer_image.CUDA_VERSION}-{revision}/b12x"
        del env["B12X_ROCE_CACHE_DIR"]
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
    # The installer toolchain image runs its system Python; the native and R37
    # images run the interpreter in /opt/venv.
    python = "python3" if policy["kind"] == "toolchain" else "/opt/venv/bin/python"
    health = () if rank else (
        python, "-c",
        f"import urllib.request; urllib.request.urlopen('http://127.0.0.1:{port}/health', timeout=4).close()",
    )
    prefix = "qad-sparkcache-" if nodes == 4 and env.get("SPARKCACHE_ENABLED") == "1" else "qad-" if nodes == 4 else "sparkcache-" if env.get("SPARKCACHE_ENABLED") == "1" else ""
    name = f"{family}-tp{nodes}-r{rank}" if policy["kind"] == "toolchain" else f"qwen-flash-next-{prefix}tp{nodes}-r{rank}"
    return ContainerSpec(
        name=name,
        image_id=image, entrypoint=(python,), command=tuple(args),
        environment=env, mounts=(Bind(str(model), "/models/target", True), Bind(str(cache), "/cache")),
        health_command=health,
    )


def render(profile, **site):
    return docker_create(container_spec(profile, **site))


def verify_model_paths(profile, model, cache):
    """Check local directories and checkpoint metadata; full shard checks remain explicit.

    The metadata files are read with ``O_NOATIME`` when permitted, so this
    check, which runs at every preflight, create and start, leaves the access
    times of a copy served in place unchanged.
    """
    model, cache = Path(model), Path(cache)
    if not model.is_dir() or not cache.is_dir():
        raise ValueError("Existing model and dedicated cache directories are required")
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    for filename, key in [
        ("config.json", "config_sha256"),
        ("model.safetensors.index.json", "index_sha256"),
    ]:
        try:
            descriptor = os.open(model / filename, flags | getattr(os, "O_NOATIME", 0))
        except PermissionError:
            # O_NOATIME needs the file's owner or CAP_FOWNER; other accounts read normally.
            descriptor = os.open(model / filename, flags)
        with os.fdopen(descriptor, "rb") as stream:
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
