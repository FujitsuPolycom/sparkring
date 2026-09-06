"""Route bounded GLM-5.3 TP4 shapes through B12X virtual diagonals.

Status: research-only. The adapter wraps an already-installed vLLM all-reduce
chain. Eligible Q1-Q32 BF16 ``[Q,4096]`` tensors use one six-origin-QP B12X
runtime per rank. Every rejected signature calls the saved SIRCL/NCCL chain.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import sys
import threading
import weakref
from collections.abc import Mapping
from pathlib import Path
from typing import Any


CONFIG_SCHEMA = "sparkring.glm53-rocenante-overlay-contract/v1"
DEFAULT_CONFIG = Path("/opt/spark-sircl/rocenante-overlay-config.json")
_installed = False
_adapters: weakref.WeakSet[VirtualDiagonalAdapter] = weakref.WeakSet()
_registry_lock = threading.Lock()
_b12x_path_installed = False


class OverlayError(RuntimeError):
    """The private overlay cannot prove a rank-invariant safe configuration."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OverlayError(f"{name} must be an object")
    return value


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OverlayError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise OverlayError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    """Load and strictly validate the mounted overlay contract."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OverlayError(f"overlay contract cannot be read: {error}") from error
    root = dict(_mapping(value, "overlay contract"))
    if root.get("schema") != CONFIG_SCHEMA or root.get("status") != "research-only":
        raise OverlayError(f"overlay contract must use {CONFIG_SCHEMA}")
    runtime = _mapping(root.get("runtime"), "runtime")
    if _integer(runtime.get("world_size"), "runtime.world_size", 1, 64) != 4:
        raise OverlayError("runtime.world_size must be four")
    if (
        _integer(
            runtime.get("opposite_rank_paths"), "runtime.opposite_rank_paths", 2, 4
        )
        != 2
    ):
        raise OverlayError("runtime.opposite_rank_paths must be two")
    if (
        _integer(
            runtime.get("origin_queue_pairs_per_rank"),
            "runtime.origin_queue_pairs_per_rank",
            1,
            64,
        )
        != 6
    ):
        raise OverlayError("runtime.origin_queue_pairs_per_rank must be six")
    if (
        _integer(
            runtime.get("direct_then_diagonal_threshold_bytes"),
            "runtime.direct_then_diagonal_threshold_bytes",
            16,
            1 << 30,
        )
        != 196608
    ):
        raise OverlayError(
            "runtime.direct_then_diagonal_threshold_bytes must be 196608"
        )
    if runtime.get("wave_mode") != "two":
        raise OverlayError("runtime.wave_mode must be 'two'")
    if runtime.get("execution_mode", "both") not in {
        "both", "eager_only", "graph_only", "disabled"
    }:
        raise OverlayError(
            "runtime.execution_mode must be both, eager_only, graph_only, or disabled"
        )
    captured_sircl_rows = runtime.get("captured_sircl_query_rows", [])
    if (
        not isinstance(captured_sircl_rows, list)
        or any(type(q) is not int or not 1 <= q <= 32 for q in captured_sircl_rows)
        or captured_sircl_rows != sorted(set(captured_sircl_rows))
    ):
        raise OverlayError(
            "runtime.captured_sircl_query_rows must be sorted unique integers in [1,32]"
        )
    proxy_cpu = _integer(runtime.get("proxy_cpu"), "runtime.proxy_cpu", 0, 4095)
    forbidden = runtime.get("forbidden_proxy_cpus")
    if forbidden != [10, 11] or proxy_cpu in forbidden:
        raise OverlayError("runtime.proxy_cpu must differ from SIRCL CPUs 10 and 11")
    hcas = root.get("canonical_hca_order")
    expected_hcas = [
        "rocep1s0f0",
        "rocep1s0f1",
        "roceP2p1s0f0",
        "roceP2p1s0f1",
    ]
    if hcas != expected_hcas:
        raise OverlayError("canonical_hca_order differs from the measured HCA order")
    maps = _mapping(root.get("peer_hca_maps"), "peer_hca_maps")
    expected_maps = {
        "0": "1=0/2,2=0/3,3=1/3",
        "1": "0=1/3,2=0/2,3=0/3",
        "2": "0=1/2,1=1/3,3=0/2",
        "3": "0=0/2,1=1/2,2=1/3",
    }
    if dict(maps) != expected_maps:
        raise OverlayError("peer_hca_maps differs from the reciprocal six-QP map")
    dispatch = _mapping(root.get("dispatch"), "dispatch")
    candidate = _mapping(dispatch.get("candidate"), "dispatch.candidate")
    expected_candidate = {
        "group_name_prefix": "tp",
        "world_size": 4,
        "cuda": True,
        "contiguous": True,
        "dtype": "bfloat16",
        "tensor_dimensions": 2,
        "width": 4096,
        "minimum_query_rows": 1,
        "maximum_query_rows": 32,
    }
    if dict(candidate) != expected_candidate:
        raise OverlayError("dispatch.candidate differs from the Q1-Q32 contract")
    metadata = _mapping(root.get("metadata"), "metadata")
    if (
        metadata.get("required_backend") != "gloo"
        or metadata.get("create_device_process_group") is not False
        or metadata.get("create_nccl_communicator") is not False
    ):
        raise OverlayError("metadata must use Gloo without a device communicator")
    return root


def _gid_available(device: str, gid_index: int) -> bool:
    root = Path("/sys/class/infiniband") / device / "ports"
    if not root.is_dir():
        return False
    for port in root.iterdir():
        path = port / "gids" / str(gid_index)
        if not path.is_file():
            continue
        value = path.read_text(encoding="ascii").strip().replace(":", "")
        if value and any(character != "0" for character in value):
            return True
    return False


def _set_exact_environment(name: str, value: str) -> None:
    present = os.environ.get(name)
    if present is not None and present != value:
        raise OverlayError(f"{name} conflicts with the mounted overlay contract")
    os.environ[name] = value


def _install_b12x_roce_path() -> None:
    """Prepend only the private ``b12x.comm.roce`` package to installed B12X."""

    global _b12x_path_installed
    if _b12x_path_installed:
        return
    if any(
        name == "b12x.comm.roce" or name.startswith("b12x.comm.roce.")
        for name in sys.modules
    ):
        raise OverlayError("b12x.comm.roce was imported before the private source path")
    import b12x.comm

    root = Path("/opt/spark-sircl/b12x_overlay/b12x/comm")
    if not (root / "roce" / "__init__.py").is_file():
        raise OverlayError(f"private b12x.comm.roce package is missing: {root}")
    paths = list(b12x.comm.__path__)
    if str(root) not in paths:
        b12x.comm.__path__.insert(0, str(root))
    importlib.invalidate_caches()
    _b12x_path_installed = True


@contextlib.contextmanager
def _inherited_thread_affinity(cpu: int):
    """Create the B12X proxy with a one-CPU inherited pthread mask."""

    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise OverlayError("proxy CPU isolation requires Linux scheduler affinity")
    original = os.sched_getaffinity(0)
    if cpu not in original:
        raise OverlayError(f"proxy CPU {cpu} is outside the worker affinity mask")
    os.sched_setaffinity(0, {cpu})
    try:
        yield
    finally:
        os.sched_setaffinity(0, original)


class VirtualDiagonalAdapter:
    """One TP rank's six-QP B12X runtime and exact Q1-Q32 policy."""

    def __init__(self, communicator: Any, contract: Mapping[str, object]) -> None:
        import torch
        import torch.distributed as dist

        self.communicator = communicator
        self.group = communicator.cpu_group
        self.device = communicator.device
        self.rank = int(communicator.rank)
        self.world_size = int(communicator.world_size)
        self._runtime = None
        self._closed = False
        self._candidate_calls = 0
        self._captured_nodes = 0
        self._fallback_calls = 0
        runtime = _mapping(contract["runtime"], "runtime")
        self.execution_mode = str(runtime.get("execution_mode", "both"))
        self.captured_sircl_rows = frozenset(runtime.get("captured_sircl_query_rows", []))
        dispatch = _mapping(contract["dispatch"], "dispatch")
        candidate = _mapping(dispatch["candidate"], "dispatch.candidate")
        self.minimum_query_rows = int(candidate["minimum_query_rows"])
        self.maximum_query_rows = int(candidate["maximum_query_rows"])
        self.width = int(candidate["width"])
        self.proxy_cpu = int(runtime["proxy_cpu"])
        self.gid_index = int(runtime["gid_index"])
        self.hcas = tuple(contract["canonical_hca_order"])
        self.peer_map = str(contract["peer_hca_maps"][str(self.rank)])

        errors: list[str] = []
        if self.world_size != 4:
            errors.append(f"TP world size is {self.world_size}, expected 4")
        if not str(communicator.unique_name).startswith(
            str(candidate["group_name_prefix"])
        ):
            errors.append(f"communicator {communicator.unique_name!r} is not TP")
        try:
            backend = str(dist.get_backend(self.group)).lower()
        except Exception as error:  # noqa: BLE001 - included in unanimous vote
            backend = f"error:{error}"
        if backend != "gloo":
            errors.append(
                f"metadata process group backend is {backend!r}, expected gloo"
            )
        if self.device.type != "cuda":
            errors.append(f"communicator device is {self.device}, expected CUDA")
        if self.proxy_cpu in (10, 11):
            errors.append("proxy CPU overlaps a SIRCL graph CPU")
        if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
            errors.append("Linux scheduler affinity is unavailable")
        else:
            try:
                if self.proxy_cpu not in os.sched_getaffinity(0):
                    errors.append(
                        f"proxy CPU {self.proxy_cpu} is outside the worker affinity mask"
                    )
            except OSError as error:
                errors.append(f"worker affinity cannot be read: {error}")
        for device in self.hcas:
            try:
                available = _gid_available(device, self.gid_index)
            except OSError as error:
                available = False
                errors.append(
                    f"RDMA device/GID probe failed: {device}:{self.gid_index}: {error}"
                )
            if not available:
                errors.append(
                    f"RDMA device/GID is unavailable: {device}:{self.gid_index}"
                )
        try:
            _install_b12x_roce_path()
            from b12x.comm import roce

            if getattr(roce, "API_VERSION", None) != 1:
                errors.append(
                    f"b12x.comm.roce API version is {getattr(roce, 'API_VERSION', None)!r}, expected 1"
                )
        except Exception as error:
            errors.append(f"b12x.comm.roce cannot import: {error}")

        expected_environment = {
            "B12X_ROCE_HCA": ",".join(self.hcas),
            "B12X_ROCE_GID_INDEX": str(self.gid_index),
            "B12X_ROCE_OPPOSITE_PATHS": "2",
            "B12X_ROCE_PEER_HCA_MAP": self.peer_map,
            "B12X_ROCE_WAVE_MODE": "two",
            "B12X_ROCE_TWO_WAVE_THRESHOLD_BYTES": "196608",
        }
        for name, expected in expected_environment.items():
            present = os.environ.get(name)
            if present is not None and present != expected:
                errors.append(f"{name} conflicts with the mounted overlay contract")

        config_digest = _canonical_sha256(contract)
        manifest_path = Path("/opt/spark-sircl/sparkring-overlay-manifest.json")
        try:
            manifest_digest = _sha256(manifest_path) if manifest_path.is_file() else ""
        except OSError as error:
            manifest_digest = ""
            errors.append(f"private bundle manifest cannot be read: {error}")
        local_vote = {
            "rank": self.rank,
            "errors": errors,
            "config_sha256": config_digest,
            "manifest_sha256": manifest_digest,
            "hcas": list(self.hcas),
            "peer_map": self.peer_map,
            "proxy_cpu": self.proxy_cpu,
        }
        votes: list[object] = [None] * self.world_size
        dist.all_gather_object(votes, local_vote, group=self.group)
        self._validate_votes(votes, contract)

        if self.execution_mode == "disabled":
            return

        for name, expected in expected_environment.items():
            _set_exact_environment(name, expected)

        from b12x.comm import roce

        max_bytes = self.maximum_query_rows * self.width * 2
        with _inherited_thread_affinity(self.proxy_cpu):
            self._runtime = roce.AllReduce.from_exchange_group(
                exchange_group=self.group,
                device=self.device,
                max_size=max_bytes,
                max_gather_bytes=16,
            )
        self._runtime.prepare((torch.bfloat16,))
        stats = self._runtime.stats()
        if (
            stats.get("hcas") != list(self.hcas)
            or stats.get("wave_mode") != "two"
            or stats.get("two_wave_threshold_bytes") != 196608
        ):
            self.close()
            raise OverlayError("B12X runtime attestation differs from the contract")
        counters = self._runtime.benchmark_counters()
        if len(counters) != 6:
            self.close()
            raise OverlayError("B12X runtime must expose six active origin-QP counters")

    def _validate_votes(
        self, votes: list[object], contract: Mapping[str, object]
    ) -> None:
        if len(votes) != 4 or any(not isinstance(vote, Mapping) for vote in votes):
            raise OverlayError("Gloo capability vote must contain four rank records")
        by_rank = {vote.get("rank"): vote for vote in votes}
        if set(by_rank) != {0, 1, 2, 3}:
            raise OverlayError("Gloo capability vote must contain ranks 0, 1, 2, and 3")
        digests = {vote.get("config_sha256") for vote in votes}
        manifests = {vote.get("manifest_sha256") for vote in votes}
        if len(digests) != 1 or len(manifests) != 1 or "" in manifests:
            raise OverlayError("overlay source or configuration differs across ranks")
        failures = [
            f"rank {rank}: {error}"
            for rank, vote in sorted(by_rank.items())
            for error in vote.get("errors", [])
        ]
        if failures:
            raise OverlayError("; ".join(failures))
        expected_maps = _mapping(contract["peer_hca_maps"], "peer_hca_maps")
        for rank, vote in sorted(by_rank.items()):
            if vote.get("peer_map") != expected_maps[str(rank)]:
                raise OverlayError(f"rank {rank} peer map differs from the contract")

    def eligible(self, tensor: Any) -> bool:
        """Return a rank-invariant decision for one TP all-reduce signature."""

        import torch

        shape = tuple(int(value) for value in tensor.shape)
        valid_signature = (
            not self._closed
            and tensor.is_cuda
            and tensor.device == self.device
            and tensor.dtype == torch.bfloat16
            and tensor.is_contiguous()
            and len(shape) == 2
            and shape[1] == self.width
            and self.minimum_query_rows <= shape[0] <= self.maximum_query_rows
        )
        if not valid_signature:
            return False
        if self.execution_mode == "disabled":
            return False
        if shape[0] in self.captured_sircl_rows:
            if bool(torch.cuda.is_current_stream_capturing()):
                return False
        if self.execution_mode == "both":
            return True
        capturing = bool(torch.cuda.is_current_stream_capturing())
        return capturing if self.execution_mode == "graph_only" else not capturing

    def all_reduce(self, tensor: Any) -> Any:
        """Run one accepted B12X collective or terminate on transport failure."""

        import torch

        if self._runtime is None:
            raise OverlayError("B12X runtime is unavailable")
        capturing = bool(torch.cuda.is_current_stream_capturing())
        try:
            if capturing:
                stream = torch.cuda.current_stream(self.device)
                with self._runtime.capture(stream=stream):
                    result = self._runtime.all_reduce(tensor, stream=stream)
                self._captured_nodes += 1
            else:
                result = self._runtime.all_reduce(tensor)
            self._candidate_calls += 1
            return result
        except BaseException:
            os._exit(70)
            raise AssertionError("worker exit unexpectedly returned")

    def record_fallback(self) -> None:
        self._fallback_calls += 1

    def check_health(self) -> None:
        if not self._closed and self._runtime is not None:
            self._runtime.check_health()

    def diagnostic_snapshot(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "execution_mode": self.execution_mode,
            "candidate_calls": self._candidate_calls,
            "captured_nodes": self._captured_nodes,
            "fallback_calls": self._fallback_calls,
            "runtime": None if self._runtime is None else self._runtime.stats(),
            "origin_qps": []
            if self._runtime is None
            else self._runtime.benchmark_counters(),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        runtime, self._runtime = self._runtime, None
        if runtime is not None:
            runtime.close()


def require_health() -> None:
    """Raise when any process-local virtual-diagonal runtime is unhealthy."""

    with _registry_lock:
        adapters = list(_adapters)
    for adapter in adapters:
        adapter.check_health()


def diagnostic_snapshot() -> list[dict[str, object]]:
    with _registry_lock:
        adapters = list(_adapters)
    return [adapter.diagnostic_snapshot() for adapter in adapters]


def host_route_snapshot() -> list[dict[str, object]]:
    """Read Python routing counters without CUDA or native transport calls.

    Captured-node counts describe graph construction, not replay executions.
    """
    with _registry_lock:
        return [
            {
                "rank": adapter.rank,
                "execution_mode": adapter.execution_mode,
                "captured_sircl_query_rows": sorted(adapter.captured_sircl_rows),
                "eager_calls": adapter._candidate_calls - adapter._captured_nodes,
                "captured_nodes": adapter._captured_nodes,
                "fallback_calls": adapter._fallback_calls,
            }
            for adapter in sorted(_adapters, key=lambda item: item.rank)
        ]


def _install_status_reporting() -> None:
    """Append CPU-only routing counters to the existing low-rate status file."""
    import spark_graph_status_reporter as reporter

    original = reporter.collect_graph_status
    if getattr(original, "_rocenante_route_status", False):
        return

    def collect():
        value = original()
        value["rocenante_routing"] = host_route_snapshot()
        return value

    collect._rocenante_route_status = True
    reporter.collect_graph_status = collect


def install(config_path: Path = DEFAULT_CONFIG) -> None:
    """Wrap vLLM after SIRCL so rejected signatures retain its dispatch chain."""

    global _installed
    if _installed:
        return
    contract = load_contract(config_path)
    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )

    original_init = CudaCommunicator.__init__
    original_all_reduce = CudaCommunicator.all_reduce
    original_destroy = CudaCommunicator.destroy
    if getattr(original_all_reduce, "_rocenante_virtual_diagonal", False):
        _installed = True
        return

    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        prefix = str(contract["dispatch"]["candidate"]["group_name_prefix"])
        if str(self.unique_name).startswith(prefix):
            if int(self.world_size) != 4:
                raise OverlayError(
                    f"virtual-diagonal TP group requires four ranks, got {self.world_size}"
                )
            with _registry_lock:
                if list(_adapters):
                    raise OverlayError(
                        "one process cannot construct more than one virtual-diagonal TP runtime"
                    )
            adapter = VirtualDiagonalAdapter(self, contract)
            self._rocenante_virtual_diagonal_adapter = adapter
            with _registry_lock:
                _adapters.add(adapter)

    def wrapped_all_reduce(self, tensor):
        adapter = getattr(self, "_rocenante_virtual_diagonal_adapter", None)
        if adapter is not None and adapter.eligible(tensor):
            return adapter.all_reduce(tensor)
        if adapter is not None:
            adapter.record_fallback()
        return original_all_reduce(self, tensor)

    def wrapped_destroy(self):
        adapter = getattr(self, "_rocenante_virtual_diagonal_adapter", None)
        if adapter is not None:
            adapter.close()
            with _registry_lock:
                _adapters.discard(adapter)
            self._rocenante_virtual_diagonal_adapter = None
        return original_destroy(self)

    wrapped_all_reduce._rocenante_virtual_diagonal = True
    wrapped_all_reduce._rocenante_saved_all_reduce = original_all_reduce
    CudaCommunicator.__init__ = wrapped_init
    CudaCommunicator.all_reduce = wrapped_all_reduce
    CudaCommunicator.destroy = wrapped_destroy
    _install_status_reporting()
    _installed = True


__all__ = [
    "CONFIG_SCHEMA",
    "DEFAULT_CONFIG",
    "OverlayError",
    "VirtualDiagonalAdapter",
    "diagnostic_snapshot",
    "install",
    "load_contract",
    "require_health",
]
