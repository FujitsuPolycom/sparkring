"""One pasteable TP2 Compose file, generated from existing profile adapters."""
from dataclasses import replace
from pathlib import PurePosixPath

import yaml

from runtime.common import compose, loader_policy, profiles, qwen_flash_next, setup, tp2

SUPPORTED = ("glm53-flash-spark-tp2-dcp1-sparkcache", "qwen38-flash-next-tp2", "qwen38-flash-next-tp2-sparkcache")


class Arguments(list):
    """Keep generated argv readable without one line per fixed flag/token."""


class Dumper(yaml.SafeDumper):
    pass


Dumper.add_representer(Arguments, lambda dumper, values: dumper.represent_sequence(
    "tag:yaml.org,2002:seq", values, flow_style=True))


def image_runtime(profile_id):
    """The installer image lock an installer profile's recipe applies, or None."""
    if profile_id not in compose.SUPPORTED:
        return None
    return compose.installer_image_runtime(profile_id)


def specifications(profile_id, variant=None):
    if profile_id not in SUPPORTED:
        raise ValueError("Single-file sharing supports the GLM/Qwen TP2 profiles; TP4 retains its prepared-mesh lifecycle")
    card = setup.selection(profile_id, variant)
    lock = image_runtime(profile_id)
    if lock is not None and (lock["image_id"], lock["image_reference"]) != (card["image_id"], card["image_reference"]):
        raise ValueError("The installer image lock and the profile release select different images")
    specs = []
    for rank in (0, 1):
        values = {"master": "192.0.2.240", "host_ip": f"192.0.2.{240+rank}", "interface": "enp1s0f0np0",
                  "model": "/srv/models/selected", "cache": "/srv/cache/selected", "image": card["image_id"]}
        if profile_id in compose.SUPPORTED:
            definition, _ = profiles.load(profile_id)
            profile = profiles.read_json(profiles.local_path(definition["configuration"]["path"]))
            spec = qwen_flash_next.container_spec(profile, rank=rank, remote=True, **values)
            if lock is not None:
                # An installer profile runs the installer's container. Compose
                # reads the relative seccomp path from the project directory.
                spec = compose.installer_container(spec, lock, profile_id=profile_id, source_root=".")
        else:
            plan = tp2.render(rank, values["master"], PurePosixPath(values["model"]), PurePosixPath(values["cache"]),
                              None, values["image"], planning_release=card["release"], r33_sparkcache=card["sparkcache"],
                              target_model_variant=card["target_variant"],
                              site_values={"VLLM_HOST_IP": values["host_ip"], "NCCL_SOCKET_IFNAME": values["interface"],
                                           "GLOO_SOCKET_IFNAME": values["interface"]})
            spec = tp2.container_spec(plan)
        specs.append(replace(spec, name=f"sparkring-{profile_id}-r{rank}"))
    return card, specs


def services(profile_id, variant=None):
    card, specs = specifications(profile_id, variant)
    result = {}
    for rank, spec in enumerate(specs):
        value = compose.escape(compose.service(spec, card["image_reference"]))
        # This is an unmanaged, standalone recipe. A digest-pinned image can be
        # pulled by Compose itself; installer exports retain pull_policy=never.
        value["pull_policy"] = "missing"
        value["profiles"] = [f"rank{rank}"]
        env = value["environment"]
        env["VLLM_HOST_IP"] = "${SPARKRING_HOST_IP:?Set this hosts primary fabric IPv4}"
        for key in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"):
            env[key] = "${SPARKRING_INTERFACE:?Set this hosts primary fabric interface}"
        master = "${SPARKRING_MASTER_ADDR:?Set rank 0 primary fabric IPv4 on both hosts}"
        if "MASTER_ADDR" in env:
            env["MASTER_ADDR"] = master
        command = value["command"]
        command[command.index("--master-addr") + 1] = master
        for mount in value["volumes"]:
            variable = "SPARKRING_MODEL_DIR" if mount["read_only"] else "SPARKRING_CACHE_DIR"
            mount["source"] = "${" + variable + ":?Set an existing absolute directory}"
        result[f"rank{rank}"] = value
    return card, result


def render(profile_id, variant=None):
    card, ranks = services(profile_id, variant)
    first, second = ranks.values()
    common = {key: value for key, value in first.items() if second.get(key) == value and key != "environment"}
    environment = {key: value for key, value in first["environment"].items() if second["environment"].get(key) == value}
    selected = {}
    for name, value in ranks.items():
        selected[name] = {"__RUNTIME_MERGE__": None, **{k: v for k, v in value.items() if k not in common and k != "environment"},
                          "environment": {"__ENVIRONMENT_MERGE__": None,
                                          **{k: v for k, v in value["environment"].items() if k not in environment}}}
        if "command" in selected[name]:
            selected[name]["command"] = Arguments(selected[name]["command"])
    document = {
        "name": "sparkring-" + profile_id,
        "x-sparkring": {"profile": profile_id, "release": card["release"], "model_repository": card["model_repository"],
                        "model_revision": card["model_revision"], "mode": "standalone; prepared two-host fabric"},
        "x-runtime": common, "x-environment": environment, "services": selected,
    }
    text = yaml.dump(document, Dumper=Dumper, sort_keys=False, width=140)
    text = text.replace("x-runtime:\n", "x-runtime: &runtime\n").replace("x-environment:\n", "x-environment: &environment\n")
    text = text.replace("    __RUNTIME_MERGE__: null\n", "    <<: *runtime\n")
    text = text.replace("      __ENVIRONMENT_MERGE__: null\n", "      <<: *environment\n")
    if image_runtime(profile_id) is None:
        needs = "# No SparkRing checkout or installer is needed to run this standalone file.\n"
    else:
        needs = f"""# Each rank is the container `sparkring install` runs, without its runtime binding:
# runtime-status worker identities report binding_not_configured.
# Compose reads the B12X loader's io_uring seccomp policy from {loader_policy.RELATIVE}
# relative to this file's directory: save this file at the root of a SparkRing checkout of
# the same revision, or copy that one file to the same relative path.
"""
    header = f"""# {profile_id} - same file on TWO Sparks. Save as compose.yaml.
# Requires prepared p0-to-p0 RoCE links (both Socket Direct functions, GID 3),
# GPU-enabled Docker/Compose, and the complete checkpoint on EACH host.
{needs}#
# Set these five values in each host's shell (or its local .env file):
#   export SPARKRING_MODEL_DIR=/absolute/path/to/weights
#   export SPARKRING_CACHE_DIR=/absolute/path/to/writable/cache
#   export SPARKRING_MASTER_ADDR=RANK0_FABRIC_IP
#   export SPARKRING_HOST_IP=THIS_HOST_FABRIC_IP
#   export SPARKRING_INTERFACE=THIS_HOST_FABRIC_NETDEV
# Weights: hf download {card['model_repository']} --revision {card['model_revision']} --local-dir "$SPARKRING_MODEL_DIR"
# Start WORKER host: docker compose --profile rank1 up -d
# Then HEAD host:    docker compose --profile rank0 up -d
# Logs: docker compose logs -f rank0  (on head; rank1 on worker)
# Stop on each host: docker compose --profile rankN stop  (substitute 0 or 1)
# Never start both rank profiles on one Spark. No profile starts by default.
# This file does not configure networking, install host services or qualify serving.
# Generated by scripts/generate_compose_examples.py; flags come from the profile adapters.

"""
    return header + text
