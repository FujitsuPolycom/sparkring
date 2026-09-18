"""Container settings shared by launcher backends; no shell-command parsing."""

import copy
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
import posixpath
import re
import shlex


@dataclass(frozen=True)
class Bind:
    source: str
    target: str
    read_only: bool = False


@dataclass(frozen=True)
class ContainerSpec:
    name: str
    image_id: str
    entrypoint: tuple[str, ...]
    command: tuple[str, ...]
    environment: dict[str, str]
    mounts: tuple[Bind, ...]
    platform: str = "linux/arm64"
    pull_policy: str = "never"
    restart_policy: str = "no"
    init: bool = True
    gpu_count: int = -1
    network_mode: str = "host"
    ipc_mode: str = "host"
    memlock: int = -1
    devices: tuple[str, ...] = ("/dev/infiniband",)
    memory: int | None = 108 * 1024**3
    memory_swap: int | None = 112 * 1024**3
    health_command: tuple[str, ...] = ()
    health_interval: int = 10
    health_timeout: int = 5
    health_start_period: int = 900
    health_retries: int = 3
    labels: dict[str, str] = field(default_factory=dict)
    shm_size: int | None = None
    cap_add: tuple[str, ...] = ()
    security_opt: tuple[str, ...] = ()
    user: str | None = None
    working_dir: str | None = None
    cpuset_cpus: str | None = None
    health_mode: str = "auto"

    def __post_init__(self):
        """Keep optional envelope settings and health policies unambiguous.

        Memory and shared-memory sizes are bytes. None omits the option and
        inherits Docker/image behavior. Health auto preserves the existing
        empty-command disable policy; inherit omits all health options. Exec
        uses an argument tuple, while shell requires one literal shell command.
        Health timing values apply only to exec and shell checks.
        """
        for name, minimum in (("memory", 0), ("memory_swap", -1), ("shm_size", 1)):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(
                    f"{name} must be None or an integer of at least {minimum} bytes"
                )
        if self.memory_swap not in (None, 0):
            if self.memory in (None, 0):
                raise ValueError("memory_swap requires a positive memory limit")
            if self.memory_swap != -1 and self.memory_swap < self.memory:
                raise ValueError(
                    "memory_swap must be unlimited or at least the memory limit"
                )
        for name in ("cap_add", "security_opt"):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or any(
                    not isinstance(value, str) or not value or "\0" in value
                    for value in values
                )
                or len(set(values)) != len(values)
            ):
                raise ValueError(
                    f"{name} must be a tuple of distinct nonempty literal strings"
                )
        for name in ("user", "working_dir"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value or "\0" in value
            ):
                raise ValueError(f"{name} must be None or a nonempty literal string")
        if self.working_dir is not None:
            path = PurePosixPath(self.working_dir)
            if (
                not path.is_absolute()
                or str(path) != self.working_dir
                or ".." in path.parts
                or "\\" in self.working_dir
            ):
                raise ValueError("working_dir must be a normalized absolute Linux path")
        if self.cpuset_cpus is not None:
            if not isinstance(self.cpuset_cpus, str) or not re.fullmatch(
                r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", self.cpuset_cpus
            ):
                raise ValueError("cpuset_cpus must be a literal CPU index/range list")
            for interval in self.cpuset_cpus.split(","):
                endpoints = [int(part) for part in interval.split("-")]
                if len(endpoints) == 2 and endpoints[0] > endpoints[1]:
                    raise ValueError("cpuset_cpus range is reversed")
        if self.health_mode not in ("auto", "inherit", "disabled", "exec", "shell"):
            raise ValueError("Unsupported container health mode")
        if not isinstance(self.health_command, tuple) or any(
            not isinstance(value, str) or "\0" in value for value in self.health_command
        ):
            raise ValueError("health_command must be a tuple of literal strings")
        mode = self.effective_health_mode
        if mode in ("inherit", "disabled") and self.health_command:
            raise ValueError("Inherited or disabled health must not supply a command")
        if mode in ("exec", "shell") and (
            not self.health_command or not self.health_command[0]
        ):
            raise ValueError("Active health mode requires a nonempty command")
        if mode == "shell" and len(self.health_command) != 1:
            raise ValueError("Shell health requires exactly one literal command")
        for name, minimum in (
            ("health_interval", 1),
            ("health_timeout", 1),
            ("health_start_period", 0),
            ("health_retries", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")

    @property
    def effective_health_mode(self):
        """Resolve auto identically for Docker and Compose renderers."""
        if self.health_mode == "auto":
            return "exec" if self.health_command else "disabled"
        return self.health_mode

    def document(self):
        result = asdict(self)
        if self.cpuset_cpus is None:
            result.pop("cpuset_cpus")
        return result


def docker_create(spec: ContainerSpec) -> list[str]:
    """Render Docker's CLI from effective settings, preserving the image wrapper."""
    argv = [
        "docker",
        "create",
        "--name",
        spec.name,
        "--entrypoint",
        spec.entrypoint[0],
        "--platform",
        spec.platform,
        "--pull",
        spec.pull_policy,
        "--restart",
        spec.restart_policy,
        "--gpus",
        "all" if spec.gpu_count == -1 else str(spec.gpu_count),
        "--network",
        spec.network_mode,
        "--ipc",
        spec.ipc_mode,
        "--ulimit",
        f"memlock={spec.memlock}:{spec.memlock}",
    ]
    for option, value in (
        ("--memory", spec.memory),
        ("--memory-swap", spec.memory_swap),
        ("--shm-size", spec.shm_size),
        ("--user", spec.user),
        ("--workdir", spec.working_dir),
        ("--cpuset-cpus", spec.cpuset_cpus),
    ):
        if value is not None:
            argv += [option, str(value)]
    for capability in spec.cap_add:
        argv += ["--cap-add", capability]
    for option in spec.security_opt:
        argv += ["--security-opt", option]
    if spec.init:
        argv += ["--init"]
    mode = spec.effective_health_mode
    if mode in ("exec", "shell"):
        argv += [
            "--health-cmd",
            spec.health_command[0]
            if mode == "shell"
            else shlex.join(spec.health_command),
            "--health-interval",
            f"{spec.health_interval}s",
            "--health-timeout",
            f"{spec.health_timeout}s",
            "--health-start-period",
            f"{spec.health_start_period}s",
            "--health-retries",
            str(spec.health_retries),
        ]
    elif mode == "disabled":
        argv += ["--no-healthcheck"]
    for device in spec.devices:
        argv += ["--device", device]
    for mount in spec.mounts:
        value = f"type=bind,src={mount.source},dst={mount.target}"
        argv += ["--mount", value + (",readonly" if mount.read_only else "")]
    for key, value in sorted(spec.environment.items()):
        argv += ["--env", key + "=" + value]
    for key, value in sorted(spec.labels.items()):
        argv += ["--label", key + "=" + value]
    return argv + [spec.image_id, *spec.entrypoint[1:], *spec.command]


def _literal_mapping(values, *, environment=False):
    if not isinstance(values, dict) or any(
        not isinstance(name, str)
        or not name
        or "\0" in name
        or (environment and not re.fullmatch("[A-Za-z_][A-Za-z0-9_]*", name))
        or not isinstance(value, str)
        or "\0" in value
        for name, value in values.items()
    ):
        raise ValueError(
            "Image and container mappings require literal string keys and values"
        )
    return dict(values)


def _image_environment(values):
    if values is None:
        return {}
    if not isinstance(values, list):
        raise ValueError("Image environment must be a list of literal assignments")
    result = {}
    for value in values:
        if not isinstance(value, str) or "=" not in value:
            raise ValueError("Image environment must contain literal assignments")
        name, content = value.split("=", 1)
        if name in result:
            raise ValueError("Image environment contains duplicate names")
        result[name] = content
    return _literal_mapping(result, environment=True)


def expected_inspection(spec: ContainerSpec, image: dict, *, backend="docker") -> dict:
    """Project typed settings and image defaults to managed inspection fields.

    No Docker command, shell output or host files are inspected here. Docker's
    CLI places extra entrypoint arguments in Cmd and uses CMD-SHELL health;
    Compose preserves the full entrypoint and uses CMD for exec health.
    host_config contains known expectations, not every daemon-generated field.
    Consumers normalize null list fields and compare nested expected fields,
    while retaining their checks for unexpected container settings and labels.
    """
    if backend not in ("docker", "compose"):
        raise ValueError("Inspection backend must be docker or compose")
    if not isinstance(image, dict):
        raise ValueError("Image inspection must be an object")
    if "Id" in image and image["Id"] != spec.image_id:
        raise ValueError(
            "Image inspection identity differs from the container specification"
        )
    platform = spec.platform.split("/")
    if len(platform) not in (2, 3) or not all(platform):
        raise ValueError(
            "Container platform must name an operating system and architecture"
        )
    for attribute, expected in zip(("Os", "Architecture", "Variant"), platform):
        if attribute in image and image[attribute] != expected:
            raise ValueError(
                "Image inspection platform differs from the container specification"
            )
    config = image.get("Config", {})
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("Image configuration must be an object")
    if config.get("Volumes"):
        raise ValueError("Image-declared anonymous volumes are not supported")
    env = _image_environment(config.get("Env"))
    env.update(_literal_mapping(spec.environment, environment=True))
    image_labels = config.get("Labels")
    labels = _literal_mapping({} if image_labels is None else image_labels)
    labels.update(_literal_mapping(spec.labels))
    if backend == "compose":
        required = {
            "com.docker.compose.project": spec.name,
            "com.docker.compose.service": "model",
        }
        if any(
            name in spec.labels and spec.labels[name] != value
            for name, value in required.items()
        ):
            raise ValueError(
                "Explicit labels conflict with the Compose project or service"
            )
        labels.update(required)
    mounts = {}
    for mount in spec.mounts:
        if (
            not isinstance(mount, Bind)
            or type(mount.read_only) is not bool
            or any(
                not isinstance(path, str) or not path.startswith("/") or "\0" in path
                for path in (mount.source, mount.target)
            )
        ):
            raise ValueError(
                "Inspection requires absolute bind mounts with explicit access"
            )
        target = posixpath.normpath(mount.target)
        if target in mounts:
            raise ValueError("Container specification has duplicate mount destinations")
        mounts[target] = {
            "Source": posixpath.normpath(mount.source),
            "Type": "bind",
            "RW": not mount.read_only,
        }
    if not spec.entrypoint:
        raise ValueError("Container specification requires an explicit entrypoint")
    for attribute in ("WorkingDir", "User"):
        if config.get(attribute) is not None and not isinstance(config[attribute], str):
            raise ValueError("Image working directory and user must be strings")
    inherited_health = config.get("Healthcheck")
    if inherited_health is not None and not isinstance(inherited_health, dict):
        raise ValueError("Image healthcheck must be an object or null")
    health = copy.deepcopy(inherited_health)
    mode = spec.effective_health_mode
    if mode != "inherit":
        # The daemon merges image timing values even when Test is NONE. Active
        # checks replace nonzero timings; a zero start period retains the image
        # value, and an inherited StartInterval remains in effect.
        health = health or {}
        if mode == "disabled":
            health["Test"] = ["NONE"]
        else:
            if mode == "exec" and backend == "compose":
                health["Test"] = ["CMD", *spec.health_command]
            else:
                command = (
                    spec.health_command[0]
                    if mode == "shell"
                    else shlex.join(spec.health_command)
                )
                health["Test"] = ["CMD-SHELL", command]
            health.update(
                Interval=spec.health_interval * 1_000_000_000,
                Timeout=spec.health_timeout * 1_000_000_000,
                Retries=spec.health_retries,
            )
            if spec.health_start_period:
                health["StartPeriod"] = spec.health_start_period * 1_000_000_000
    host_config = {
        "NetworkMode": spec.network_mode,
        "IpcMode": spec.ipc_mode,
        "Memory": 0 if spec.memory is None else spec.memory,
        "MemorySwap": 0 if spec.memory_swap is None else spec.memory_swap,
        "RestartPolicy": {"Name": spec.restart_policy},
        "Privileged": False,
        "Ulimits": [{"Name": "memlock", "Soft": spec.memlock, "Hard": spec.memlock}],
        "CapAdd": list(spec.cap_add),
        "SecurityOpt": list(spec.security_opt),
        "Devices": [
            {
                "PathOnHost": device,
                "PathInContainer": device,
                "CgroupPermissions": "rwm",
            }
            for device in spec.devices
        ],
        "DeviceRequests": [
            {
                "Driver": "nvidia" if backend == "compose" else "",
                "Count": spec.gpu_count,
                "Capabilities": [["gpu"]],
            }
        ],
    }
    if spec.init or backend == "compose":
        host_config["Init"] = spec.init
    if spec.shm_size is not None:
        host_config["ShmSize"] = spec.shm_size
    if spec.cpuset_cpus is not None:
        host_config["CpusetCpus"] = spec.cpuset_cpus
    return {
        "name": spec.name,
        "image": spec.image_id,
        "cmd": list(spec.entrypoint[1:] + spec.command)
        if backend == "docker"
        else list(spec.command),
        "entrypoint": list(spec.entrypoint[:1])
        if backend == "docker"
        else list(spec.entrypoint),
        "env": env,
        "mounts": mounts,
        "labels": labels,
        "working_dir": spec.working_dir
        if spec.working_dir is not None
        else config.get("WorkingDir") or "",
        "user": spec.user if spec.user is not None else config.get("User") or "",
        "healthcheck": health,
        "host_config": host_config,
    }
