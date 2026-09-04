"""Rank-wide capability agreement before SIRCL native construction."""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Callable


ADAPTER_ABI = "sparkring-sircl-capability/v1"
logger = logging.getLogger(__name__)
REQUIRED_SYMBOLS = (
    "spark_tp4_create",
    "spark_tp4_create_v2",
    "spark_tp4_all_reduce",
    "spark_tp4_capture_all_reduce",
    "spark_tp4_get_graph_status",
    "spark_tp4_get_health_status",
    "spark_tp4_destroy",
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


def _device_gid_available(device: str, gid: str) -> bool:
    root = Path("/sys/class/infiniband") / device / "ports"
    if not root.is_dir():
        return False
    return any((port / "gids" / gid).is_file() for port in root.iterdir())


def local_capability(rank: int) -> dict[str, Any]:
    """Build a non-throwing local record so every rank reaches the vote."""

    errors: list[str] = []
    library_value = os.environ.get("SPARK_TP4_LIBRARY", "")
    library = Path(library_value) if library_value else None
    native_sha256 = ""
    if library is None or not library.is_file():
        errors.append("native library is missing")
    else:
        native_sha256 = _sha256(library)
        expected_native = os.environ.get("SPARKRING_SIRCL_NATIVE_SHA256", "")
        if expected_native and native_sha256 != expected_native:
            errors.append("native library digest does not match the launcher")
        try:
            loaded = ctypes.CDLL(str(library))
            missing = [name for name in REQUIRED_SYMBOLS if not hasattr(loaded, name)]
            if missing:
                errors.append("native ABI is missing: " + ",".join(missing))
        except OSError as error:
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
    for device, gid in device_specs:
        if not device or not gid or not _device_gid_available(device, gid):
            errors.append(f"RDMA device/GID is unavailable: {device or '-'}:{gid or '-'}")

    manifest_path = Path(
        os.environ.get(
            "SPARKRING_SIRCL_MANIFEST_PATH",
            "/opt/spark-sircl/sparkring-overlay-manifest.json",
        )
    )
    manifest_sha256 = _sha256(manifest_path) if manifest_path.is_file() else ""
    expected_manifest = os.environ.get("SPARKRING_SIRCL_MANIFEST_SHA256", "")
    if not manifest_sha256:
        errors.append("overlay manifest is missing")
    elif expected_manifest and manifest_sha256 != expected_manifest:
        errors.append("overlay manifest digest does not match the launcher")

    return {
        "rank": rank,
        "adapter_abi": ADAPTER_ABI,
        "native_sha256": native_sha256,
        "manifest_sha256": manifest_sha256,
        "shared": {
            "mode": os.environ.get("VLLM_SPARK_TP4_MODE", ""),
            "protocol": os.environ.get("SPARK_TP4_ALLREDUCE_PROTOCOL", ""),
            "direct_doorbell": os.environ.get("SPARK_TP4_GRAPH_DIRECT_DOORBELL", ""),
            "prefill_exposure": os.environ.get(
                "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE", ""
            ),
            "prefill_rail_mode": os.environ.get(
                "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE", ""
            ),
            "max_inflight": os.environ.get("SPARK_TP4_MAX_INFLIGHT", ""),
            "max_query_rows": os.environ.get("VLLM_SPARK_MAX_QUERY_ROWS", ""),
            "connect_timeout": os.environ.get(
                "SPARK_TP4_CONTROL_CONNECT_TIMEOUT_SECONDS", ""
            ),
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
        for field in ("adapter_abi", "native_sha256", "manifest_sha256", "shared"):
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
    logger.info("SIRCL capability vote accepted %d physical ranks", len(records))
