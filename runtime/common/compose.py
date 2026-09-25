"""Generate Compose deployments from canonical profiles and private host inputs."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

import yaml

from runtime.common import profiles, qwen_flash_next

ROOT = Path(__file__).resolve().parents[2]
# Published Qwen profiles with generated public Compose examples.
EXAMPLE_TP4 = ("qwen38-flash-next-qad-tp4", "qwen38-flash-next-qad-tp4-sparkcache")
EXAMPLES = ("qwen38-flash-next-tp2", "qwen38-flash-next-tp2-sparkcache", *EXAMPLE_TP4)
# Serving profiles that run only on the installer's shared toolchain image.
# Their Compose files are rendered by the installer with that image applied.
TOOLCHAIN_TP4 = ("glm53-flash-nvfp4-spark-tp4", "mimo-v26-flash-rl-tp4")
TOOLCHAIN = ("glm53-flash-nvfp4-spark-tp2", "mimo-v26-flash-rl-tp2", *TOOLCHAIN_TP4)
# Four-node profiles require each host's prepared mesh fabric reference.
TP4_PROFILES = (*EXAMPLE_TP4, *TOOLCHAIN_TP4)
SUPPORTED = (*EXAMPLES, *TOOLCHAIN)
LABEL = "io.sparkring.deployment"


def encoded(document):
    return json.dumps(document, sort_keys=True, indent=2) + "\n"


def digest(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def read_site(path):
    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if key in result:
                raise ValueError("Duplicate site key: " + str(key))
            result[key] = loader.construct_object(value_node)
        return result

    UniqueLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping
    )
    try:
        return yaml.load(Path(path).read_text(encoding="utf-8"), Loader=UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError("Site file must contain valid YAML") from exc


def linux_path(value):
    if not isinstance(value, str):
        raise ValueError("Host paths must be strings")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or ".." in path.parts
        or value == "/"
        or any(c in value for c in ("\\", ","))
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise ValueError("Host paths must be normalized absolute Linux paths")
    return path


def site_settings(site, *, nodes=2):
    if not isinstance(site, dict) or set(site) != {"schema", "name", "master", "ranks"}:
        raise ValueError(
            "Site requires only schema, name, master and ranks; secrets and overrides are not accepted"
        )
    if site["schema"] != "sparkring-compose-site/v1":
        raise ValueError("Unsupported Compose site schema")
    if not isinstance(site["name"], str) or not re.fullmatch(
        "[a-z][a-z0-9-]{0,39}", site["name"]
    ):
        raise ValueError(
            "Site name must be a lowercase deployment name, at most 40 characters"
        )
    if nodes not in (2, 4) or not isinstance(site["ranks"], list) or len(site["ranks"]) != nodes:
        raise ValueError(f"Qwen TP{nodes} requires exactly {nodes} hosts")
    hosts, addresses = set(), set()
    for number, rank in enumerate(site["ranks"]):
        keys = {
            "rank",
            "host",
            "host_ip",
            "interface",
            "hcas",
            "gid",
            "model",
            "cache",
            "repository",
            "deployment_root",
        }
        if nodes == 4:
            keys.add("fabric")
        if (
            not isinstance(rank, dict)
            or set(rank) != keys
            or type(rank["rank"]) is not int
            or rank["rank"] != number
        ):
            raise ValueError(
                "Site ranks must be ordered from zero with the documented host fields"
            )
        host = rank["host"]
        if (
            not isinstance(host, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", host)
            or host == "controller"
            or host in hosts
            or rank["host_ip"] in addresses
        ):
            raise ValueError("Ranks require distinct SSH targets and IP addresses")
        hosts.add(host)
        addresses.add(rank["host_ip"])
        paths = [
            linux_path(rank[key])
            for key in ("model", "cache", "repository", "deployment_root")
        ]
        # Container bind sources and staged control files must never share a tree.
        for i, path in enumerate(paths):
            for other in paths[i + 1 :]:
                if path.is_relative_to(other) or other.is_relative_to(path):
                    raise ValueError(
                        "Model, cache, repository and deployment roots must be disjoint"
                    )
        qwen_flash_next.site_inputs(
            number,
            site["master"],
            rank["host_ip"],
            rank["interface"],
            rank["model"],
            rank["cache"],
            remote=True,
            nodes=nodes,
        )
        if nodes == 4:
            from runtime.common import qwen_mesh
            qwen_mesh.validate_site_reference(rank["fabric"])
    if site["master"] != site["ranks"][0]["host_ip"]:
        raise ValueError("Master must be the API rank's host_ip")
    return site


def installer_image_lock_path():
    from runtime.common import installer_image
    return installer_image.DEFAULT_LOCK.relative_to(ROOT).as_posix()


def source_inventory(profile_id, *, local_source_extension=None):
    """Bind controller and host code plus release inputs using LF-normalized text."""
    paths = {
        "runtime/common/compose.py",
        "runtime/common/__init__.py",
        "runtime/common/container_spec.py",
        "runtime/common/ports.py",
        "runtime/common/process_lock.py",
        "runtime/common/qwen_flash_next.py",
        "runtime/common/profiles.py",
        "runtime/common/candidate.py",
        "runtime/common/cache_candidate.py",
        "scripts/sparkring_compose.py",
        "scripts/deploy_engine.py",
        "profiles/catalog.json",
        f"profiles/{profile_id}/profile.json",
        "profiles/qwen38-flash-next-tp2/config.json",
        "profiles/qwen38-flash-next-tp2/sparkcache.json",
    }
    metadata, release = profiles.load(profile_id)
    paths.update(item["path"] for item in release["inputs"])
    paths.add(metadata["configuration"]["path"])
    profile = qwen_flash_next.read(ROOT / metadata["configuration"]["path"])
    policy = qwen_flash_next.image_policy(profile, local_source_extension=local_source_extension)
    if policy["kind"] == "native":
        paths.add("runtime/common/native_candidate.py")
        paths.add(f"runtime/releases/{policy['native_release']}/publication.json")
    if policy["kind"] == "toolchain":
        paths.add("runtime/common/installer_image.py")
        paths.add(installer_image_lock_path())
    if profile_id in TP4_PROFILES:
        from runtime.common import qwen_mesh
        paths.add("runtime/common/feature_candidate.py")
        paths.update(("profiles/qwen38-flash-next-qad-tp4/config.json",
                      "profiles/qwen38-flash-next-qad-tp4/sparkcache.json"))
        paths.update(qwen_mesh.SOURCE_FILES)
    folders = ["lil-r37-glm-spark", "lil-r37-cache64"]
    if profile_id in TP4_PROFILES:
        folders.append("lil-r37-shared")
    if policy["source_extension"] is not None:
        from runtime.common import source_candidate
        source_candidate.descriptor(policy["source_extension"])
        paths.update(("runtime/common/source_candidate.py", "runtime/common/feature_candidate.py"))
        if "lil-r37-shared" not in folders:
            folders.append("lil-r37-shared")
        folders.append(policy["source_extension"])
    for folder in folders:
        paths.update(
            p.relative_to(ROOT).as_posix()
            for p in (ROOT / "runtime/images/compositions" / folder).glob("*.json")
        )
    paths.add(profiles.load(profile_id)[0]["release"])
    return {
        name: digest((ROOT / name).read_bytes().replace(b"\r\n", b"\n"))
        for name in sorted(paths)
    }


def specifications(profile_id, site, *, local_image_id=None, local_source_extension=None,
                   local_kv_cache_gib=None, local_master_port=None):
    if profile_id not in SUPPORTED:
        raise ValueError(
            "Compose adapter unsupported for "
            + str(profile_id)
            + "; use its profile quickstart"
        )
    metadata, release = profiles.load(profile_id)
    profile = qwen_flash_next.read(ROOT / metadata["configuration"]["path"])
    site_settings(site, nodes=qwen_flash_next.node_count(profile))
    policy = qwen_flash_next.image_policy(profile, local_source_extension=local_source_extension)
    if policy["kind"] == "source" and not policy["local"]:
        from runtime.common import source_candidate
        publication = source_candidate.release_publication(release, policy["source_extension"])
    else:
        publication = qwen_flash_next.read(ROOT / release["inputs"][0]["path"])
    local = publication.get("schema") == "sparkring-local-image-build/v1"
    image = publication["image_tag"] if local else publication["image_reference"]
    image_id = publication["image_id"]
    if local_source_extension is not None:
        from runtime.common import source_candidate
        image = source_candidate.image_reference(local_source_extension, local_image_id)
        image_id = local_image_id
    elif local_kv_cache_gib is not None or local_master_port is not None:
        raise ValueError("Local KV and master-port alternatives require a local source extension")
    elif local:
        from runtime.common import feature_candidate
        if (publication.get("published") is not False
                or profile.get("image_extension") != "lil-r37-shared"
                or publication.get("descriptor_sha256") != digest(feature_candidate.DESCRIPTOR.read_bytes())):
            raise ValueError("Local image selection must bind the shared feature descriptor")
        if local_image_id is not None:
            if not isinstance(local_image_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", local_image_id):
                raise ValueError("Local rebuild requires an exact image configuration ID")
            image_id = local_image_id
    elif local_image_id is not None:
        raise ValueError("Published image selections cannot be overridden")
    if local_source_extension is None and (
        release["image"] != image
        or publication["platform"] != "linux/arm64"
        or not (local or re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image))
    ):
        raise ValueError(
            "Release must select its registered Linux ARM64 manifest digest"
        )
    specs = []
    for rank in site["ranks"]:
        spec = qwen_flash_next.container_spec(
            profile,
            rank=rank["rank"],
            master=site["master"],
            host_ip=rank["host_ip"],
            interface=rank["interface"],
            model=rank["model"],
            cache=rank["cache"],
            hcas=rank["hcas"],
            gid=rank["gid"],
            image=image_id,
            remote=True,
            local_source_extension=local_source_extension,
            local_kv_cache_gib=local_kv_cache_gib,
            local_master_port=local_master_port,
        )
        specs.append(replace(spec, name=f"sr-{site['name']}-r{rank['rank']}"))
    return specs, image


def service(spec, image):
    """Compose representation before interpolation escaping or CLI normalization."""
    result = {
        "container_name": spec.name,
        "image": image,
        "platform": spec.platform,
        "pull_policy": spec.pull_policy,
        "restart": spec.restart_policy,
        "init": spec.init,
        "entrypoint": list(spec.entrypoint),
        "command": list(spec.command),
        "environment": dict(sorted(spec.environment.items())),
        "labels": dict(spec.labels),
        "network_mode": spec.network_mode,
        "ipc": spec.ipc_mode,
        "ulimits": {"memlock": {"soft": spec.memlock, "hard": spec.memlock}},
        "deploy": {
            "resources": {
                "reservations": {
                    "devices": [
                        {
                            "driver": "nvidia",
                            "count": "all" if spec.gpu_count == -1 else spec.gpu_count,
                            "capabilities": ["gpu"],
                        }
                    ]
                }
            }
        },
        "devices": list(spec.devices),
        "volumes": [
            {
                "type": "bind",
                "source": mount.source,
                "target": mount.target,
                "read_only": mount.read_only,
                "bind": {"create_host_path": False},
            }
            for mount in spec.mounts
        ],
    }
    for name, value in (("mem_limit", spec.memory), ("memswap_limit", spec.memory_swap),
                        ("shm_size", spec.shm_size), ("user", spec.user),
                        ("working_dir", spec.working_dir), ("cpuset", spec.cpuset_cpus)):
        if value is not None:
            result[name] = value
    for name in ("cap_add", "security_opt"):
        if getattr(spec, name):
            result[name] = list(getattr(spec, name))
    mode = spec.effective_health_mode
    if mode == "disabled":
        result["healthcheck"] = {"disable": True}
    elif mode in ("exec", "shell"):
        result["healthcheck"] = {
            "test": ["CMD-SHELL" if mode == "shell" else "CMD", *spec.health_command],
            "interval": f"{spec.health_interval}s",
            "timeout": f"{spec.health_timeout}s",
            "start_period": f"{spec.health_start_period}s",
            "retries": spec.health_retries,
        }
    return result


def escape(value):
    if isinstance(value, str):
        return value.replace("$", "$$")
    if isinstance(value, list):
        return [escape(v) for v in value]
    if isinstance(value, dict):
        return {k: escape(v) for k, v in value.items()}
    return value


def compose_text(spec, image):
    document = {"name": spec.name, "services": {"model": service(spec, image)}}
    return (
        "# Generated by sparkring compose render; edit the profile or private site input.\n"
        + yaml.safe_dump(escape(document), sort_keys=False, width=110)
    )


def selection_options(manifest):
    """Carry only explicit image/test selections into deterministic regeneration."""
    return {name: manifest[name] for name in (
        "local_image_id", "local_source_extension", "local_kv_cache_gib", "local_master_port",
    ) if name in manifest}


def build(profile_id, site, *, local_image_id=None, local_source_extension=None,
          local_kv_cache_gib=None, local_master_port=None):
    options = {key: value for key, value in {
        "local_image_id": local_image_id, "local_source_extension": local_source_extension,
        "local_kv_cache_gib": local_kv_cache_gib, "local_master_port": local_master_port,
    }.items() if value is not None}
    specs, image = specifications(profile_id, site, **options)
    inputs = (source_inventory(profile_id) if local_source_extension is None else
              source_inventory(profile_id, local_source_extension=local_source_extension))
    identity_inputs = {"profile": profile_id, "site": site, "inputs": inputs}
    identity_inputs.update(options)
    identity = digest(encoded(identity_inputs))
    files = {}
    for number, spec in enumerate(specs):
        spec = replace(spec, labels={LABEL: identity, "io.sparkring.rank": str(number)})
        files[f"rank{number}/compose.yaml"] = compose_text(spec, image)
        files[f"rank{number}/container.json"] = encoded(spec.document())
    manifest = {
        "schema": "sparkring-compose-deployment/v1",
        "id": identity,
        "profile": profile_id,
        "site": site,
        "inputs": inputs,
        "image": image,
        "image_id": specs[0].image_id,
        "files": {name: digest(text) for name, text in files.items()},
        "qualification": "Generated configuration only; Compose serving is not qualified.",
    }
    manifest.update(options)
    return manifest, files


def render(profile_id, site, output, *, local_image_id=None, local_source_extension=None,
           local_kv_cache_gib=None, local_master_port=None):
    manifest, files = build(profile_id, site, local_image_id=local_image_id,
                          local_source_extension=local_source_extension,
                          local_kv_cache_gib=local_kv_cache_gib, local_master_port=local_master_port)
    output = Path(output).resolve()
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / ".sparkring"):
        raise ValueError(
            "Private exports inside the repository must be under .sparkring/"
        )
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name, text in {**files, "deployment.json": encoded(manifest)}.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_text(text, encoding="utf-8", newline="\n")
        path.chmod(0o600)
    return manifest


def load_deployment(output):
    output = Path(output)
    manifest = qwen_flash_next.read(output / "deployment.json")
    expected, files = build(manifest["profile"], manifest["site"], **selection_options(manifest))
    if manifest != expected:
        raise ValueError(
            "Deployment inputs changed; render a separate deployment and review it"
        )
    for name, text in files.items():
        if (output / name).read_bytes() != text.encode():
            raise ValueError(
                "Edited export is a custom configuration: "
                + name
                + "; canonical checks/start refuse it"
            )
    return manifest, files


def compose_command(project, file="-"):
    # Explicit context and env-file prevent ambient daemon selection and .env loading.
    return [
        "docker",
        "--context",
        "default",
        "compose",
        "--project-name",
        project,
        "--env-file",
        os.devnull,
        "-f",
        str(file),
    ]


def check_project_containers(project, *, owned_id=None, run=subprocess.run):
    """Reject existing Compose project containers except an explicitly owned ID."""
    result = run(["docker", "container", "ls", "--all", "--no-trunc",
                  "--filter", "label=com.docker.compose.project=" + project,
                  "--format", "{{.ID}}"], check=True, capture_output=True, text=True)
    if any(value != owned_id for value in result.stdout.splitlines()):
        raise ValueError("Compose project already contains another container; refusing adoption or recreation")


def normalize_service(value):
    """Normalize resolved output after bind intent is checked; retain unexpected fields."""
    value = json.loads(json.dumps(value))
    value.setdefault("labels", {})
    value.setdefault("environment", {})
    value.setdefault("volumes", [])
    health = value.get("healthcheck", {})
    for name in ("interval", "timeout", "start_period"):
        if name in health:
            parts = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", str(health[name]))
            if parts and any(part is not None for part in parts.groups()):
                hours, minutes, seconds = (int(part or 0) for part in parts.groups())
                health[name] = f"{hours * 3600 + minutes * 60 + seconds}s"
    if isinstance(value.get("devices"), list):
        value["devices"] = [
            (
                {"source": d, "target": d, "permissions": "rwm"}
                if isinstance(d, str)
                else d
            )
            for d in value["devices"]
        ]
    for key in ("mem_limit", "memswap_limit", "shm_size"):
        if key in value:
            value[key] = int(value[key])
    if "security_opt" in value:
        value["security_opt"] = [option.replace("=", ":", 1) for option in value["security_opt"]]
    deploy = value.get("deploy", {})
    if deploy.get("placement") == {}:
        deploy.pop("placement")
    for device in (
        deploy.get("resources", {}).get("reservations", {}).get("devices", [])
    ):
        if device.get("count") == "all":
            device["count"] = -1
    for mount in value.get("volumes", []):
        mount.setdefault("read_only", False)
        # Compose-go's boolean omitempty encoding can render an explicitly
        # disabled CreateHostPath as bind: {}. Keep absent bind options distinct.
        # https://github.com/compose-spec/compose-go/blob/v2.7.1/types/types.go
        if mount.get("type") == "bind" and isinstance(mount.get("bind"), dict):
            mount["bind"].setdefault("create_host_path", False)
    return value


def check_equivalence(spec, image, text, *, run=subprocess.run):
    # JSON encoders can omit different boolean defaults. Require the safe
    # literal in the submitted YAML before interpreting an omitted bind flag.
    try:
        submitted = yaml.safe_load(text)["services"]["model"].get("volumes", [])
    except (yaml.YAMLError, KeyError, TypeError, AttributeError):
        raise ValueError("Compose input requires a model service and literal bind options") from None
    if (not isinstance(submitted, list) or len(submitted) != len(spec.mounts)
            or any(not isinstance(mount, dict) or mount.get("type") != "bind"
                   or not isinstance(mount.get("bind"), dict)
                   or mount["bind"].get("create_host_path") is not False for mount in submitted)):
        raise ValueError("Compose bind mounts must explicitly disable host-directory creation")
    result = run(
        compose_command(spec.name)
        + ["config", "--format", "json", "--no-path-resolution"],
        input=text,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    actual = json.loads(result.stdout)
    expected = {
        "name": spec.name,
        "services": {"model": normalize_service(service(spec, image))},
    }
    actual["services"] = {
        name: normalize_service(value)
        for name, value in actual.get("services", {}).items()
    }
    # Compose config re-escapes dollars after resolving interpolation so that
    # its output can itself be loaded as a Compose file. Compare in that format.
    # docker/compose cmd/compose/config.go: runConfig -> escapeDollarSign.
    if actual != escape(expected):
        raise ValueError(
            "Resolved Compose configuration differs from the shared container specification"
        )
    return actual


def check(output):
    manifest, files = load_deployment(output)
    specs, image = specifications(manifest["profile"], manifest["site"], **selection_options(manifest))
    for number, spec in enumerate(specs):
        spec = replace(
            spec, labels={LABEL: manifest["id"], "io.sparkring.rank": str(number)}
        )
        check_equivalence(spec, image, files[f"rank{number}/compose.yaml"])
    return manifest
