"""Create deterministic Docker launch commands from inspection records."""

from __future__ import annotations

import subprocess
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .command import vllm_arguments

_DATA_DESTINATIONS = (
    "/cache",
    "/l2cache",
    "/models",
    "/root/.cache/huggingface",
)
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _explicitly_unset_environment(values: Sequence[str]) -> tuple[str, ...]:
    requested: set[str] = set()
    for value in values:
        name, separator, setting = value.partition("=")
        if separator and name == "SPARKRING_EXPLICITLY_UNSET":
            requested.update(item for item in setting.split(",") if item)
    invalid = sorted(name for name in requested if _ENVIRONMENT_NAME.fullmatch(name) is None)
    if invalid:
        raise ValueError(
            "SPARKRING_EXPLICITLY_UNSET contains invalid environment names: "
            + ", ".join(invalid)
        )
    return tuple(sorted(requested))


def normalized_posix_path(value: str, role: str) -> str:
    """Validate one host or container path used in a Docker bind."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{role} must be a normalized absolute POSIX path")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or len(path.parts) < 2
        or ".." in path.parts
        or str(path) != value
        or ":" in value
    ):
        raise ValueError(f"{role} must be a normalized absolute POSIX path")
    return value


def build_container_command(
    inspection: dict[str, Any],
    image: str,
    name: str,
    checkpoint_sha256: str,
    *,
    create_only: bool = False,
    sparkcache_root: str | None = None,
    max_bytes: int | None = None,
    low_watermark_bytes: int | None = None,
    ttl_seconds: int | None = None,
    extra_binds: Sequence[tuple[str, str] | tuple[str, str, bool]] = (),
    preserve_all_binds: bool = False,
    entrypoint: str | None = None,
    labels: Mapping[str, str] | None = None,
) -> list[str]:
    """Build one complete ``docker create`` or ``docker run`` invocation."""

    for label, value in (
        ("max_bytes", max_bytes),
        ("low_watermark_bytes", low_watermark_bytes),
        ("ttl_seconds", ttl_seconds),
    ):
        if value is not None and value < 0:
            raise ValueError(f"{label} must be non-negative")
    if sparkcache_root is not None:
        normalized = PurePosixPath(sparkcache_root)
        if (
            not normalized.is_absolute()
            or len(normalized.parts) < 3
            or normalized.parts[1] != "cache"
            or ".." in normalized.parts
            or str(normalized) != sparkcache_root
        ):
            raise ValueError("sparkcache_root must be a normalized child of /cache")
    if low_watermark_bytes is not None:
        if max_bytes is None or max_bytes <= 0:
            raise ValueError("low_watermark_bytes requires positive max_bytes")
        if not 0 < low_watermark_bytes <= max_bytes:
            raise ValueError("low_watermark_bytes must be in (0, max_bytes]")

    host = inspection["HostConfig"]
    command = ["docker", "create" if create_only else "run"]
    if not create_only:
        command.append("-d")
    command += ["--name", name]
    command += ["--network", str(host.get("NetworkMode", "host"))]
    command += ["--ipc", str(host.get("IpcMode", "host"))]
    command += ["--shm-size", str(host.get("ShmSize", 17179869184))]
    command += ["--cap-add", "IPC_LOCK", "--ulimit", "memlock=-1:-1"]
    command += ["--security-opt", "label=disable", "--gpus", "all"]
    if entrypoint is not None:
        command += ["--entrypoint", normalized_posix_path(entrypoint, "entrypoint")]
    for key, value in sorted((labels or {}).items()):
        if not key or "=" in key or "\x00" in key or "\x00" in value:
            raise ValueError("labels must contain safe nonempty string keys and values")
        command += ["--label", f"{key}={value}"]
    if Path("/dev/infiniband").exists():
        command += ["--device", "/dev/infiniband:/dev/infiniband"]
    environment = list(inspection["Config"].get("Env", []))
    for value in environment:
        command += ["--env", value]
    # Omitting a variable does not remove an ENV value baked into the image.
    # Explicit empty assignments materialize the profile adapter's removal
    # contract after every inherited and transformed value has been added.
    for name in _explicitly_unset_environment(environment):
        command += ["--env", f"{name}="]
    command += [
        "--env",
        "SPARKCACHE_TARGET_CHECKPOINT_SHA256=" + checkpoint_sha256,
    ]
    for key, value in (
        ("SPARKCACHE_ROOT", sparkcache_root),
        ("SPARKCACHE_MAX_BYTES", max_bytes),
        ("SPARKCACHE_LOW_WATERMARK_BYTES", low_watermark_bytes),
        ("SPARKCACHE_TTL_SECONDS", ttl_seconds),
    ):
        if value is not None:
            command += ["--env", f"{key}={value}"]

    occupied_destinations: set[str] = set()
    for index, mount in enumerate(inspection.get("Mounts", [])):
        if preserve_all_binds and not isinstance(mount, dict):
            raise ValueError(f"inspection Mounts[{index}] must be an object")
        destination = str(mount.get("Destination", ""))
        if not preserve_all_binds and not destination.startswith(_DATA_DESTINATIONS):
            continue
        if preserve_all_binds:
            if mount.get("Type") != "bind":
                raise ValueError(f"inspection Mounts[{index}] must be a bind mount")
            if type(mount.get("RW")) is not bool:
                raise ValueError(f"inspection Mounts[{index}].RW must be boolean")
            source = normalized_posix_path(
                mount.get("Source"), f"inspection Mounts[{index}] source"
            )
            destination = normalized_posix_path(
                destination, f"inspection Mounts[{index}] destination"
            )
            if destination in occupied_destinations:
                raise ValueError(f"duplicate bind destination {destination}")
            occupied_destinations.add(destination)
            mode = "rw" if mount.get("RW") is True else "ro"
            value = f"{source}:{destination}:{mode}"
        else:
            value = f"{mount['Source']}:{destination}"
            if not mount.get("RW", False):
                value += ":ro"
            occupied_destinations.add(destination)
        command += ["--volume", value]
    for index, bind in enumerate(extra_binds):
        if not isinstance(bind, (list, tuple)) or len(bind) not in (2, 3):
            raise ValueError(
                f"extra_binds[{index}] must contain host path, container path,"
                " and optional read-only flag"
            )
        host_path = normalized_posix_path(
            bind[0], f"extra_binds[{index}] host path"
        )
        container_path = normalized_posix_path(
            bind[1], f"extra_binds[{index}] container path"
        )
        if container_path in occupied_destinations:
            raise ValueError(f"duplicate extra bind destination {container_path}")
        read_only = False
        if len(bind) == 3:
            if type(bind[2]) is not bool:
                raise ValueError(
                    f"extra_binds[{index}] read-only flag must be boolean"
                )
            read_only = bind[2]
        occupied_destinations.add(container_path)
        mode = "ro" if read_only else "rw"
        command += ["--volume", f"{host_path}:{container_path}:{mode}"]
    command.append(image)
    command.extend(vllm_arguments(list(inspection["Config"]["Cmd"])))
    return command


def launch_container(
    inspection: dict[str, Any],
    image: str,
    name: str,
    checkpoint_sha256: str,
    *,
    create_only: bool = False,
    sparkcache_root: str | None = None,
    max_bytes: int | None = None,
    low_watermark_bytes: int | None = None,
    ttl_seconds: int | None = None,
    extra_binds: Sequence[tuple[str, str] | tuple[str, str, bool]] = (),
    preserve_all_binds: bool = False,
    entrypoint: str | None = None,
    labels: Mapping[str, str] | None = None,
    runner: Callable[..., Any] | None = None,
) -> None:
    """Build and execute one Docker launch command."""

    command = build_container_command(
        inspection,
        image,
        name,
        checkpoint_sha256,
        create_only=create_only,
        sparkcache_root=sparkcache_root,
        max_bytes=max_bytes,
        low_watermark_bytes=low_watermark_bytes,
        ttl_seconds=ttl_seconds,
        extra_binds=extra_binds,
        preserve_all_binds=preserve_all_binds,
        entrypoint=entrypoint,
        labels=labels,
    )
    (runner or subprocess.run)(command, check=True)
