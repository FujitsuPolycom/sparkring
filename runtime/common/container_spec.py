"""Container settings shared by launcher backends; no shell-command parsing."""

from dataclasses import asdict, dataclass, field
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
    memory: int = 108 * 1024**3
    memory_swap: int = 112 * 1024**3
    health_command: tuple[str, ...] = ()
    health_interval: int = 10
    health_timeout: int = 5
    health_start_period: int = 900
    health_retries: int = 3
    labels: dict[str, str] = field(default_factory=dict)

    def document(self):
        return asdict(self)


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
        "--memory",
        str(spec.memory),
        "--memory-swap",
        str(spec.memory_swap),
    ]
    if spec.init:
        argv += ["--init"]
    if spec.health_command:
        argv += [
            "--health-cmd",
            shlex.join(spec.health_command),
            "--health-interval",
            f"{spec.health_interval}s",
            "--health-timeout",
            f"{spec.health_timeout}s",
            "--health-start-period",
            f"{spec.health_start_period}s",
            "--health-retries",
            str(spec.health_retries),
        ]
    else:
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
