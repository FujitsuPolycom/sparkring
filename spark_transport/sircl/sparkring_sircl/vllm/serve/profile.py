"""A SparkRing serving profile, read from a repository checkout and checked for drift.

The launcher serves exactly what the repository's profile describes for its
installer image, changing only what SIRCL needs (:mod:`.plan`). It reads, from
the checkout given as ``--repository``:

- ``profiles/<id>/profile.json`` (``sparkring-deployment/v1``): the release
  whose installer image the profile runs on;
- the serving configuration the deployment record names
  (``configuration.path``, format ``serving-profile``, schema
  ``sparkring-serving-profile/v1``; usually ``profiles/<id>/config.json``):
  the model pin, the runtime environment and the vLLM arguments (the
  recipe). A profile in another format (``recipe``, ``release-profile``)
  runs from its own launcher and is refused with that launcher's name;
- ``profiles/<id>/compose/compose.rank<N>.yaml``: the per-rank container the
  repository renders for that installer image (entrypoint, command,
  environment, limits, devices, seccomp policy, health check);
- the release's ``installer-image.json``: the image reference and image ID;
- ``profiles/<id>/SHA256SUMS`` and the checkpoint manifest
  ``profiles/checkpoints/<owner>--<name>/<revision>.json``: the files the
  checkpoint directory must hold, with their sizes;
- ``runtime/common/loader-seccomp.json``: the seccomp policy that admits the
  io_uring calls of the B12X weight loader;
- ``profiles/thinking.json`` when present: how the checkpoint treats thinking
  (its effort levels and the template's own default), which
  ``--reasoning-effort`` is checked against. The launcher looks up the
  checkpoint it serves (``--checkpoint-id`` when given), or the behaviour
  ``--thinking-behaviour`` names (:func:`named_thinking_behaviour`).

Drift between these sources raises :class:`ProfileError` naming both values
(the repository's rule: stop and report when prose and executable
configuration disagree). The checks are:

- one compose file per tensor-parallel rank, ``--nnodes`` equal to the
  tensor-parallel size, pipeline and decode-context parallelism of 1;
- every rank's command is ``serve /models/target --served-model-name <name>
  --node-rank <rank> --master-addr <address>`` followed by exactly the
  recipe's arguments, and ``--headless`` on every rank but 0;
- every rank's environment holds every recipe variable with the recipe's
  value; ``VLLM_PLUGINS`` may only gain the installer's status plugin, and
  the search paths the installer image removes (``PYTHONPATH``,
  ``LD_LIBRARY_PATH``, ``LD_PRELOAD``, ``PATH``, ``TRITON_PTXAS_PATH``) may be
  absent;
- every rank names the installer image's reference and the toolchain
  entrypoint, and the rank-0 health check probes the recipe's API port;
- every file of ``SHA256SUMS`` is in the checkpoint manifest;
- every rank's Compose service uses only the keys this module translates
  (:data:`SERVICE_KEYS`), and its GPU reservation is all NVIDIA GPUs, so a
  setting the launcher would silently drop is reported instead.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_PROFILE = "glm53-flash-nvfp4-spark-tp4"
TOOLCHAIN_ENTRYPOINT = ("python3", "/opt/sparkring/toolchain/toolchain.py")
SECCOMP_RELATIVE = "runtime/common/loader-seccomp.json"
INSTALLER_PLUGIN = "sparkring_status"
# Search paths the installer image's adaptation removes from every rendered
# environment, so containers inherit the sealed image's own
# (runtime/common/installer_image.py, ``adapt``); a recipe may still list them.
IMAGE_SEARCH_PATHS = frozenset(("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH", "PATH", "TRITON_PTXAS_PATH"))
HEADLESS = "--headless"
# Compose service keys the launcher translates into ``docker run`` options or
# replaces (container_name, labels); any other key is drift.
SERVICE_KEYS = frozenset((
    "container_name", "image", "platform", "pull_policy", "restart", "init", "entrypoint", "command",
    "labels", "network_mode", "ipc", "ulimits", "deploy", "devices", "volumes", "mem_limit",
    "memswap_limit", "security_opt", "healthcheck", "environment",
))
ALL_GPUS = {"resources": {"reservations": {"devices": [
    {"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]}}}
# The acceptance harness's request settings when a profile names none
# (performance/harnesses/acceptance/profile_info.py).
DEFAULT_REQUEST_SETTINGS = {"chat_template_kwargs": {"enable_thinking": False}}
THINKING_RELATIVE = "profiles/thinking.json"
THINKING_SCHEMA = "sparkring-thinking/v1"


class ProfileError(ValueError):
    """The profile's sources are missing or disagree."""


@dataclasses.dataclass(frozen=True)
class Mount:
    source: str
    target: str
    read_only: bool


@dataclasses.dataclass(frozen=True)
class Health:
    command: tuple[str, ...]          # the exec form, without Compose's leading CMD
    interval: str
    timeout: str
    start_period: str
    retries: int


@dataclasses.dataclass(frozen=True)
class RankContainer:
    """One rank's container as the repository renders it (documentation addresses and paths)."""

    rank: int
    image_reference: str
    entrypoint: tuple[str, ...]
    command: tuple[str, ...]
    environment: dict[str, str]
    mounts: tuple[Mount, ...]
    devices: tuple[str, ...]
    security_opt: tuple[str, ...]
    memory: int | None
    memory_swap: int | None
    memlock: int
    platform: str
    pull_policy: str
    restart: str
    init: bool
    network_mode: str
    ipc_mode: str
    health: Health | None


@dataclasses.dataclass(frozen=True)
class ThinkingBehaviour:
    """How a checkpoint's chat template (or vLLM's prompt encoder) treats thinking, from the repository's
    ``profiles/thinking.json``, whose ``checkpoints["<repository>@<revision>"]`` names one of its
    ``behaviours``."""

    id: str
    default: str                    # thinking when a request names nothing: "on", "always", ...
    level: str | None               # the effort level when a request names none
    levels: tuple[str, ...]         # the effort levels a request, or a serving default, may name
    effort: str | None              # the chat template argument that carries the level


def _thinking_document(root: Path, sources: dict[str, str] | None) -> dict[str, Any] | None:
    """``profiles/thinking.json`` of the checkout, or None when it has none."""
    if not (Path(root) / THINKING_RELATIVE).is_file():
        return None
    document = _json(Path(root), THINKING_RELATIVE, {} if sources is None else sources)
    if not isinstance(document, dict) or document.get("schema") != THINKING_SCHEMA:
        raise ProfileError(f"{THINKING_RELATIVE} is not a {THINKING_SCHEMA} record")
    return document


def _behaviour(document: dict[str, Any], name: str) -> ThinkingBehaviour | None:
    """The behaviour ``name`` the record defines with a list of levels, or None."""
    record = (document.get("behaviours") or {}).get(name)
    levels = record.get("levels") if isinstance(record, dict) else None
    if not isinstance(levels, list) or not all(isinstance(item, str) for item in levels):
        return None
    level, effort = record.get("level"), record.get("effort")
    return ThinkingBehaviour(str(name), str(record.get("default", "")), level if isinstance(level, str) else None,
                             tuple(levels), effort if isinstance(effort, str) else None)


def thinking_behaviour(root: Path, repository: str, revision: str,
                       sources: dict[str, str] | None = None) -> ThinkingBehaviour | None:
    """The behaviour ``profiles/thinking.json`` records for checkpoint ``repository@revision``, or None
    when the checkout has no such file or the file does not list the checkpoint."""
    document = _thinking_document(root, sources)
    if document is None:
        return None
    name = (document.get("checkpoints") or {}).get(f"{repository}@{revision}")
    if name is None:
        return None
    behaviour = _behaviour(document, name)
    if behaviour is None:
        raise ProfileError(f"{THINKING_RELATIVE}: checkpoint {repository}@{revision} names behaviour {name!r}, "
                           "which it does not define with a list of levels")
    return behaviour


def named_thinking_behaviour(root: Path, name: str) -> ThinkingBehaviour:
    """The behaviour ``name`` that ``profiles/thinking.json`` defines, whichever checkpoints it lists, for a
    checkpoint the record does not list (``--thinking-behaviour``)."""
    document = _thinking_document(root, None)
    if document is None:
        raise ProfileError(f"the repository checkout {root} has no {THINKING_RELATIVE}")
    behaviour = _behaviour(document, name)
    if behaviour is None:
        defined = sorted(key for key in (document.get("behaviours") or {}) if _behaviour(document, key))
        raise ProfileError(f"{THINKING_RELATIVE} defines no behaviour {name!r} with a list of levels; it defines "
                           + (", ".join(defined) or "none"))
    return behaviour


@dataclasses.dataclass(frozen=True)
class CheckpointFile:
    name: str
    size: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class ServingProfile:
    id: str
    repository: Path
    tensor_parallel: int
    served_model_name: str
    api_port: int
    master_port: int
    max_num_batched_tokens: int
    image_reference: str
    image_id: str
    release: str
    model_repository: str
    model_revision: str
    config_sha256: str
    index_sha256: str
    checkpoint_files: tuple[CheckpointFile, ...]
    ranks: tuple[RankContainer, ...]
    recipe_environment: dict[str, str]
    recipe_arguments: tuple[str, ...]
    request_settings: dict[str, Any]  # extra chat request fields of the profile's functional checks
    topology: str                     # the fabric the profile is qualified on (config.json "topology")
    seccomp_policy: bytes
    sources: dict[str, str]           # relative path -> SHA-256 of every file read
    thinking: ThinkingBehaviour | None = None   # profiles/thinking.json's record of the checkpoint

    @property
    def checkpoint_bytes(self) -> int:
        return sum(item.size for item in self.checkpoint_files)

    @property
    def seccomp_sha256(self) -> str:
        return hashlib.sha256(self.seccomp_policy).hexdigest()

    @property
    def checkpoint_dir_name(self) -> str:
        """The installer's checkpoint folder name: ``<owner>--<name>/<revision>``."""
        return f"{self.model_repository.replace('/', '--')}/{self.model_revision}"


def _read(root: Path, relative: str, sources: dict[str, str]) -> bytes:
    path = root / relative
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ProfileError(f"cannot read {relative} in the repository checkout {root}: {error}") from None
    sources[relative] = hashlib.sha256(data).hexdigest()
    return data


def _json(root: Path, relative: str, sources: dict[str, str]) -> Any:
    try:
        return json.loads(_read(root, relative, sources))
    except json.JSONDecodeError as error:
        raise ProfileError(f"{relative} is not valid JSON: {error}") from None


def _yaml(root: Path, relative: str, sources: dict[str, str]) -> Any:
    try:
        import yaml
    except ImportError:
        raise ProfileError("reading the profile's Compose files needs PyYAML (python -m pip install pyyaml)") from None
    try:
        return yaml.safe_load(_read(root, relative, sources))
    except yaml.YAMLError as error:
        raise ProfileError(f"{relative} is not valid YAML: {error}") from None


def argument(arguments: tuple[str, ...] | list[str], flag: str) -> str:
    """The value after ``flag``; the flag must occur exactly once."""
    positions = [index for index, item in enumerate(arguments) if item == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ProfileError(f"the recipe must give {flag} exactly once with a value")
    return arguments[positions[0] + 1]


def _size(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise ProfileError(f"{name} must be a byte count")
    return value


def _rank_container(document: Any, rank: int, relative: str) -> RankContainer:
    try:
        service = document["services"]["model"]
    except (KeyError, TypeError):
        raise ProfileError(f"{relative} has no services.model") from None
    if set(document) - {"name", "services"} or set(document["services"]) != {"model"}:
        raise ProfileError(f"{relative}: expected one service named model and no other top-level keys")
    unknown = sorted(set(service) - SERVICE_KEYS)
    if unknown:
        raise ProfileError(f"{relative}: Compose keys {unknown} have no docker run translation in the launcher")
    if service.get("deploy") != ALL_GPUS:
        raise ProfileError(f"{relative}: deploy must reserve all NVIDIA GPUs, found {service.get('deploy')}")
    if set(service.get("ulimits", {})) - {"memlock"}:
        raise ProfileError(f"{relative}: only the memlock ulimit is supported")
    mounts = []
    for volume in service.get("volumes", []):
        if volume.get("type") != "bind":
            raise ProfileError(f"{relative}: only bind mounts are supported, found {volume}")
        mounts.append(Mount(str(volume["source"]), str(volume["target"]), bool(volume.get("read_only", False))))
    health = None
    check = service.get("healthcheck") or {}
    if check and not check.get("disable"):
        test = list(check.get("test", []))
        if not test or test[0] != "CMD":
            raise ProfileError(f"{relative}: the health check must be an exec (CMD) check")
        health = Health(tuple(str(item) for item in test[1:]), str(check.get("interval", "10s")),
                        str(check.get("timeout", "5s")), str(check.get("start_period", "0s")),
                        int(check.get("retries", 3)))
    memlock = service.get("ulimits", {}).get("memlock", {})
    if memlock.get("soft") != memlock.get("hard"):
        raise ProfileError(f"{relative}: memlock soft and hard limits differ")
    environment = {str(key): "" if value is None else str(value)
                   for key, value in (service.get("environment") or {}).items()}
    return RankContainer(
        rank=rank,
        image_reference=str(service["image"]),
        entrypoint=tuple(str(item) for item in service["entrypoint"]),
        command=tuple(str(item) for item in service["command"]),
        environment=environment,
        mounts=tuple(mounts),
        devices=tuple(str(item) for item in service.get("devices", [])),
        security_opt=tuple(str(item) for item in service.get("security_opt", [])),
        memory=_size(service.get("mem_limit"), "mem_limit"),
        memory_swap=_size(service.get("memswap_limit"), "memswap_limit"),
        memlock=int(memlock.get("soft", -1)),
        platform=str(service.get("platform", "linux/arm64")),
        pull_policy=str(service.get("pull_policy", "never")),
        restart=str(service.get("restart", "no")),
        init=bool(service.get("init", True)),
        network_mode=str(service.get("network_mode", "host")),
        ipc_mode=str(service.get("ipc", "host")),
        health=health,
    )


def _check_rank(container: RankContainer, *, rank: int, served: str, arguments: tuple[str, ...],
                environment: dict[str, str], image_reference: str, port: int, relative: str) -> None:
    command = container.command
    head = ("serve", "/models/target", "--served-model-name", served, "--node-rank", str(rank), "--master-addr")
    if command[:len(head)] != head or len(command) < len(head) + 1:
        raise ProfileError(f"{relative}: the command starts {list(command[:8])}, expected {list(head)} <address>")
    tail = command[len(head) + 1:]
    expected = arguments + ((HEADLESS,) if rank else ())
    if tail != expected:
        differing = [(index, a, b) for index, (a, b) in enumerate(zip(tail, expected)) if a != b][:3]
        raise ProfileError(f"{relative}: the vLLM arguments differ from config.json's recipe "
                           f"(lengths {len(tail)} and {len(expected)}; first differences {differing})")
    for key, value in environment.items():
        actual = container.environment.get(key)
        if key in IMAGE_SEARCH_PATHS and actual is None:
            continue
        if key == "VLLM_PLUGINS":
            plugins = [item for item in (actual or "").split(",") if item]
            wanted = [item for item in value.split(",") if item]
            if plugins[:len(wanted)] != wanted or set(plugins[len(wanted):]) - {INSTALLER_PLUGIN}:
                raise ProfileError(f"{relative}: VLLM_PLUGINS is {actual!r}, the recipe has {value!r}")
        elif actual != value:
            raise ProfileError(f"{relative}: {key} is {actual!r}, the recipe has {value!r}")
    if container.image_reference != image_reference:
        raise ProfileError(f"{relative}: image {container.image_reference} differs from the release's "
                           f"installer image {image_reference}")
    if container.entrypoint != TOOLCHAIN_ENTRYPOINT:
        raise ProfileError(f"{relative}: entrypoint {list(container.entrypoint)} is not the toolchain "
                           f"entrypoint {list(TOOLCHAIN_ENTRYPOINT)}")
    if rank == 0:
        if container.health is None or f":{port}/health" not in " ".join(container.health.command):
            raise ProfileError(f"{relative}: the API rank's health check does not probe port {port}")
    targets = {mount.target for mount in container.mounts}
    if not {"/models/target", "/cache"} <= targets:
        raise ProfileError(f"{relative}: expected bind mounts at /models/target and /cache, found {sorted(targets)}")


def serving_environment(repository: str | Path, profile_id: str) -> tuple[dict[str, str], dict[str, str]]:
    """``(environment, sources)``: the environment of the serving configuration that
    ``profiles/<id>/profile.json`` names, and the SHA-256 of each file read.

    Unlike :func:`load` it needs no Compose files, so it reads every serving
    profile, including those of eight Sparks that only SIRCL ring sessions
    run; the bundle takes the profile's SIRCL settings from it (``--profile``).
    """
    root = Path(repository)
    sources: dict[str, str] = {}
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,80}", profile_id):
        raise ProfileError(f"profile id {profile_id!r} is not a profile directory name")
    deployment = _json(root, f"profiles/{profile_id}/profile.json", sources)
    configuration = deployment.get("configuration") if isinstance(deployment, dict) else None
    if (not isinstance(configuration, dict) or configuration.get("format") != "serving-profile"
            or not configuration.get("path")):
        raise ProfileError(f"profiles/{profile_id}/profile.json names no serving-profile configuration")
    config = _json(root, str(configuration["path"]), sources)
    if not isinstance(config, dict) or config.get("schema") != "sparkring-serving-profile/v1" \
            or not isinstance(config.get("environment"), dict):
        raise ProfileError(f"{configuration['path']} is not a serving profile with an environment")
    return {str(key): str(value) for key, value in config["environment"].items()}, sources


def load(repository: str | Path, profile_id: str = DEFAULT_PROFILE) -> ServingProfile:
    root = Path(repository)
    sources: dict[str, str] = {}
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,80}", profile_id):
        raise ProfileError(f"profile id {profile_id!r} is not a profile directory name")
    base = f"profiles/{profile_id}"
    deployment = _json(root, f"{base}/profile.json", sources)
    if deployment.get("schema") != "sparkring-deployment/v1" or deployment.get("id") != profile_id:
        raise ProfileError(f"{base}/profile.json is not the deployment record of {profile_id}")
    configuration = deployment.get("configuration")
    if not isinstance(configuration, dict) or not configuration.get("path"):
        raise ProfileError(f"{base}/profile.json names no configuration (format and path)")
    if configuration.get("format") != "serving-profile":
        launcher = deployment.get("launcher") or {}
        runner = launcher.get("path") or launcher.get("kind") or "its guide"
        raise ProfileError(
            f"{profile_id} is a {configuration.get('format')} profile ({configuration.get('path')}) that runs "
            f"from its own launcher ({runner}); this launcher serves only serving profiles. Add SIRCL to that "
            "launcher's containers with the bundle command")
    config_path = str(configuration["path"])
    config = _json(root, config_path, sources)
    if config.get("schema") != "sparkring-serving-profile/v1":
        raise ProfileError(f"{config_path} is not a serving profile")
    if str(config.get("topology", "")).startswith("switched"):
        raise ProfileError(f"{profile_id} is qualified on a switched fabric ({config_path}: topology "
                           f"{config.get('topology')}); this launcher places groups on a ring of directly "
                           "cabled Sparks")
    missing = [key for key in ("vllm_args", "environment", "served_model_name", "model") if key not in config]
    if missing:
        raise ProfileError(f"{config_path} lacks {missing}, which the launcher reads from a serving profile")
    arguments = tuple(str(item) for item in config["vllm_args"])
    environment = {str(key): str(value) for key, value in config["environment"].items()}
    tp = int(argument(arguments, "--tensor-parallel-size"))
    if int(argument(arguments, "--nnodes")) != tp:
        raise ProfileError("the recipe must run one rank per Spark (--nnodes equal to --tensor-parallel-size)")
    for flag in ("--pipeline-parallel-size", "--decode-context-parallel-size"):
        if int(argument(arguments, flag)) != 1:
            raise ProfileError(f"the launcher serves tensor parallelism only; the recipe sets {flag} "
                               f"{argument(arguments, flag)}")
    port = int(argument(arguments, "--port"))
    master_port = int(argument(arguments, "--master-port"))
    batched = int(argument(arguments, "--max-num-batched-tokens"))
    served = str(config["served_model_name"])
    model = config["model"]
    release_path = str(deployment["release"])
    lock_path = str(Path(release_path).parent.as_posix()) + "/installer-image.json"
    if not (root / lock_path).is_file():
        raise ProfileError(f"the release {release_path} has no installer image lock ({lock_path}: the image "
                           "reference and local image ID the launcher runs); its profiles run from their own "
                           "launcher")
    lock = _json(root, lock_path, sources)
    if profile_id not in lock.get("profiles", []):
        raise ProfileError(f"the installer image {lock.get('name')} does not list {profile_id}")
    image_reference, image_id = str(lock["image_reference"]), str(lock["image_id"])
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ProfileError(f"installer image ID {image_id!r} is not a full sha256 image ID")
    ranks = []
    for rank in range(tp):
        relative = f"{base}/compose/compose.rank{rank}.yaml"
        container = _rank_container(_yaml(root, relative, sources), rank, relative)
        _check_rank(container, rank=rank, served=served, arguments=arguments, environment=environment,
                    image_reference=image_reference, port=port, relative=relative)
        ranks.append(container)
    extra = root / f"{base}/compose/compose.rank{tp}.yaml"
    if extra.exists():
        raise ProfileError(f"{base}/compose holds more rank files than --tensor-parallel-size {tp}")
    sums = _read(root, f"{base}/SHA256SUMS", sources).decode()
    manifest_name = f"profiles/checkpoints/{str(model['repository']).replace('/', '--')}/{model['revision']}.json"
    manifest = _json(root, manifest_name, sources)
    files = []
    for line in sums.splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        entry = manifest.get("files", {}).get(name)
        if entry is None:
            raise ProfileError(f"SHA256SUMS lists {name}, which the checkpoint manifest {manifest_name} lacks")
        if entry.get("sha256") != digest:
            raise ProfileError(f"{name}: SHA256SUMS and the checkpoint manifest give different digests")
        files.append(CheckpointFile(name, int(entry["size"]), digest))
    by_name = {item.name: item for item in files}
    for name, key in (("config.json", "config_sha256"), ("model.safetensors.index.json", "index_sha256")):
        if name not in by_name or by_name[name].sha256 != model[key]:
            raise ProfileError(f"{name}: the profile pins {model[key]}, SHA256SUMS holds "
                               f"{by_name[name].sha256 if name in by_name else 'nothing'}")
    seccomp = _read(root, SECCOMP_RELATIVE, sources)
    thinking = thinking_behaviour(root, str(model["repository"]), str(model["revision"]), sources)
    for container in ranks:
        if container.security_opt and container.security_opt != (f"seccomp=/opt/sparkring/{SECCOMP_RELATIVE}",):
            raise ProfileError(f"compose rank {container.rank}: unexpected security options {container.security_opt}")
    return ServingProfile(
        id=profile_id, repository=root, tensor_parallel=tp, served_model_name=served, api_port=port,
        master_port=master_port, max_num_batched_tokens=batched, image_reference=image_reference,
        image_id=image_id, release=str(lock.get("name", "")), model_repository=str(model["repository"]),
        model_revision=str(model["revision"]), config_sha256=str(model["config_sha256"]),
        index_sha256=str(model["index_sha256"]), checkpoint_files=tuple(files), ranks=tuple(ranks),
        recipe_environment=environment, recipe_arguments=arguments,
        request_settings=dict(config.get("smoke") or DEFAULT_REQUEST_SETTINGS),
        topology=str(config.get("topology", "")), seccomp_policy=seccomp,
        sources=dict(sorted(sources.items())), thinking=thinking,
    )
