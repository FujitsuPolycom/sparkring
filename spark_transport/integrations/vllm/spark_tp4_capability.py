"""Rank-wide capability agreement before SIRCL native construction."""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
from typing import Any, Callable


ADAPTER_ABI = "sparkring-sircl-capability/v1"
NATIVE_ABI_VERSION = 1
FUSED_PREFILL_OPERATION_SLOTS = 2
REQUIRED_SYMBOLS = (
    "spark_tp4_get_abi_version",
    "spark_tp4_create",
    "spark_tp4_create_v2",
    "spark_tp4_all_reduce",
    "spark_tp4_capture_all_reduce",
    "spark_tp4_get_graph_status",
    "spark_tp4_get_health_status",
    "spark_tp4_destroy",
    "spark_tp4_bidirectional_prefill_create",
    "spark_tp4_bidirectional_prefill_all_reduce",
    "spark_tp4_bidirectional_prefill_get_health_status",
    "spark_tp4_bidirectional_prefill_destroy",
    "spark_tp4_fused_prefill_create",
    "spark_tp4_fused_prefill_all_reduce_rows",
    "spark_tp4_fused_prefill_get_health_status",
    "spark_tp4_fused_prefill_destroy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _device_gid_available(
    device: str,
    gid: str,
    *,
    sysfs_root: Path = Path("/sys/class/infiniband"),
) -> bool:
    root = sysfs_root / device / "ports"
    if not root.is_dir():
        return False
    for port in root.iterdir():
        gid_path = port / "gids" / gid
        if not gid_path.is_file():
            continue
        value = gid_path.read_text(encoding="ascii").strip().replace(":", "")
        if value and any(character != "0" for character in value):
            return True
    return False


def _integer_setting(
    name: str,
    default: int,
    errors: list[str],
    *,
    minimum: int = 1,
    maximum: int = 0xFFFFFFFF,
) -> int | str:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        errors.append(f"{name} must be an integer: {raw!r}")
        return raw
    if not minimum <= value <= maximum:
        errors.append(f"{name} must be in [{minimum}, {maximum}]: {value}")
    return value


def _shared_capability(errors: list[str]) -> dict[str, Any]:
    rail_mode = os.environ.get(
        "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE", "single"
    )
    return {
        "mode": os.environ.get("VLLM_SPARK_TP4_MODE", ""),
        "graph_protocol": os.environ.get(
            "VLLM_SPARK_TP4_GRAPH_ALLREDUCE_PROTOCOL", "serial_ack"
        ),
        "direct_doorbell": os.environ.get("SPARK_TP4_GRAPH_DIRECT_DOORBELL", "0"),
        "prefill_enabled": os.environ.get(
            "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL", "0"
        ),
        "prefill_exposure": os.environ.get(
            "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE", "sync"
        ),
        "prefill_rail_mode": rail_mode,
        "rail_count": 2 if rail_mode == "dual" else 1,
        "operation_slots": FUSED_PREFILL_OPERATION_SLOTS,
        "max_inflight": _integer_setting("SPARK_TP4_MAX_INFLIGHT", 64, errors),
        "admission": {
            "collective_min_query_rows": 1,
            "collective_max_query_rows": _integer_setting(
                "VLLM_SPARK_MAX_QUERY_ROWS", 6, errors, maximum=40
            ),
            "bidirectional_query_rows": (1024, 2048, 4096, 8192),
            "fused_min_query_rows": 128,
            "fused_max_query_rows": 8192,
        },
        "control_ports": {
            "eager": (
                _integer_setting(
                    "SPARK_TP4_CONTROL_PORT0", 11000, errors, maximum=65535
                ),
                _integer_setting(
                    "SPARK_TP4_CONTROL_PORT1", 11001, errors, maximum=65535
                ),
            ),
            "graph": (
                _integer_setting(
                    "SPARK_TP4_GRAPH_CONTROL_PORT0", 9970, errors, maximum=65535
                ),
                _integer_setting(
                    "SPARK_TP4_GRAPH_CONTROL_PORT1", 9971, errors, maximum=65535
                ),
            ),
            "graph_dual_port_q40": (
                _integer_setting(
                    "SPARK_TP4_GRAPH_DUAL_PORT_Q40_CONTROL_PORT0",
                    9972,
                    errors,
                    maximum=65535,
                ),
                _integer_setting(
                    "SPARK_TP4_GRAPH_DUAL_PORT_Q40_CONTROL_PORT1",
                    9973,
                    errors,
                    maximum=65535,
                ),
            ),
            "bidirectional": (
                _integer_setting(
                    "SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT0",
                    19000,
                    errors,
                    maximum=65535,
                ),
                _integer_setting(
                    "SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT1",
                    19001,
                    errors,
                    maximum=65535,
                ),
            ),
            "bidirectional_secondary": (
                _integer_setting(
                    "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT0",
                    19100,
                    errors,
                    maximum=65535,
                ),
                _integer_setting(
                    "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT1",
                    19101,
                    errors,
                    maximum=65535,
                ),
            ),
        },
        "timeouts": {
            "control_connect_seconds": _integer_setting(
                "SPARK_TP4_CONTROL_CONNECT_TIMEOUT_SECONDS", 10, errors
            ),
            "bidirectional_seconds": _integer_setting(
                "SPARK_TP4_BIDIRECTIONAL_PREFILL_TIMEOUT_SECONDS", 120, errors
            ),
        },
    }


def _cuda_capability(errors: list[str]) -> dict[str, Any]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if available else 0
    except Exception as error:
        errors.append(f"CUDA probe failed: {type(error).__name__}: {error}")
        return {"available": False, "device_count": 0}
    if not available or device_count < 1:
        errors.append("CUDA is unavailable")
    return {"available": available, "device_count": device_count}


def _read_sha256(path: Path, label: str, errors: list[str]) -> str:
    try:
        return _sha256(path)
    except OSError as error:
        errors.append(f"{label} cannot be read: {error}")
        return ""


def _local_capability(rank: int) -> dict[str, Any]:
    errors: list[str] = []
    shared = _shared_capability(errors)
    cuda = _cuda_capability(errors)

    library_value = os.environ.get("SPARK_TP4_LIBRARY", "")
    library = Path(library_value) if library_value else None
    native_sha256 = ""
    native_abi_version: int | None = None
    if library is None or not library.is_file():
        errors.append("native library is missing")
    else:
        native_sha256 = _read_sha256(library, "native library", errors)
        expected_native = os.environ.get("SPARKRING_SIRCL_NATIVE_SHA256", "")
        if expected_native and native_sha256 and native_sha256 != expected_native:
            errors.append("native library digest does not match the launcher")
        try:
            loaded = ctypes.CDLL(str(library))
            missing = [name for name in REQUIRED_SYMBOLS if not hasattr(loaded, name)]
            if missing:
                errors.append("native ABI is missing: " + ",".join(missing))
            else:
                abi_version = loaded.spark_tp4_get_abi_version
                abi_version.argtypes = []
                abi_version.restype = ctypes.c_uint32
                native_abi_version = int(abi_version())
                if native_abi_version != NATIVE_ABI_VERSION:
                    errors.append(
                        "native ABI version is unsupported: "
                        f"{native_abi_version} != {NATIVE_ABI_VERSION}"
                    )
        except (OSError, TypeError, ValueError) as error:
            errors.append(f"native library cannot load: {error}")

    device_specs = (
        (os.environ.get("SPARK_TP4_DEVICE0", ""), os.environ.get("SPARK_TP4_GID0", "")),
        (os.environ.get("SPARK_TP4_DEVICE1", ""), os.environ.get("SPARK_TP4_GID1", "")),
    )
    if os.environ.get("VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE") == "dual":
        device_specs += (
            (
                os.environ.get("SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE0", ""),
                os.environ.get("SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID0", ""),
            ),
            (
                os.environ.get("SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE1", ""),
                os.environ.get("SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID1", ""),
            ),
        )
    rdma = []
    for device, gid in device_specs:
        available = False
        try:
            available = bool(device and gid and _device_gid_available(device, gid))
        except OSError as error:
            errors.append(
                f"RDMA device/GID probe failed: {device or '-'}:{gid or '-'}: {error}"
            )
        if not available:
            errors.append(f"RDMA device/GID is unavailable: {device or '-'}:{gid or '-'}")
        rdma.append({"device": device, "gid": gid, "available": available})

    manifest_path = Path(
        os.environ.get(
            "SPARKRING_SIRCL_MANIFEST_PATH",
            "/opt/spark-sircl/sparkring-overlay-manifest.json",
        )
    )
    manifest_sha256 = (
        _read_sha256(manifest_path, "overlay manifest", errors)
        if manifest_path.is_file()
        else ""
    )
    expected_manifest = os.environ.get("SPARKRING_SIRCL_MANIFEST_SHA256", "")
    if not manifest_sha256:
        errors.append("overlay manifest is missing")
    elif expected_manifest and manifest_sha256 != expected_manifest:
        errors.append("overlay manifest digest does not match the launcher")

    return {
        "rank": rank,
        "adapter_abi": ADAPTER_ABI,
        "native_abi_version": native_abi_version,
        "native_sha256": native_sha256,
        "manifest_sha256": manifest_sha256,
        "shared": shared,
        "local": {"cuda": cuda, "rdma": tuple(rdma)},
        "errors": tuple(errors),
    }


def local_capability(rank: int) -> dict[str, Any]:
    """Build a non-throwing local record so every rank reaches the vote."""

    try:
        return _local_capability(rank)
    except Exception as error:
        errors = [f"capability probe failed: {type(error).__name__}: {error}"]
        return {
            "rank": rank,
            "adapter_abi": ADAPTER_ABI,
            "native_abi_version": None,
            "native_sha256": "",
            "manifest_sha256": "",
            "shared": _shared_capability(errors),
            "local": {
                "cuda": {"available": False, "device_count": 0},
                "rdma": (),
            },
            "errors": tuple(errors),
        }


def validate_capabilities(records: list[dict[str, Any]]) -> None:
    if not records:
        raise RuntimeError("SIRCL capability vote returned no rank records")
    expected = records[0]
    failures = []
    for record in records:
        rank = record.get("rank", "?")
        errors = tuple(record.get("errors", ()))
        if errors:
            failures.append(f"rank {rank}: " + "; ".join(errors))
        for field in (
            "adapter_abi",
            "native_abi_version",
            "native_sha256",
            "manifest_sha256",
            "shared",
        ):
            if record.get(field) != expected.get(field):
                failures.append(f"rank {rank}: {field} disagrees with rank 0")
    if failures:
        raise RuntimeError("SIRCL capability vote failed: " + " | ".join(failures))


def _exchange(communicator: Any, local: dict[str, Any]) -> list[dict[str, Any]]:
    import torch.distributed as dist

    records: list[dict[str, Any] | None] = [None] * int(communicator.world_size)
    dist.all_gather_object(records, local, group=communicator.cpu_group)
    if any(record is None for record in records):
        raise RuntimeError("SIRCL capability vote omitted a physical rank")
    return [record for record in records if record is not None]


def ensure_capability_vote(
    communicator: Any,
    *,
    exchange: Callable[[Any, dict[str, Any]], list[dict[str, Any]]] = _exchange,
) -> None:
    if getattr(communicator, "_sparkring_sircl_capability_voted", False):
        return
    local = local_capability(int(communicator.rank_in_group))
    records = exchange(communicator, local)
    validate_capabilities(records)
    communicator._sparkring_sircl_capability_voted = True
    print(
        f"SIRCL capability vote accepted: physical_ranks={len(records)}",
        flush=True,
    )
