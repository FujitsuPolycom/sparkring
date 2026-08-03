#!/usr/bin/env python3
"""Fail-closed verifier for the public SparkRing EXL3 derived image."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path


PREFIX = Path("/opt/sparkring-exl3")
SITE_PACKAGES = Path("/opt/venv/lib/python3.12/site-packages")
SPARK_RUNTIME_ROOT = Path("/opt/spark-vllm")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_sources(pins: dict) -> None:
    for relative, expected in pins["overlay_files"].items():
        path = SITE_PACKAGES / relative
        require(path.is_file(), f"missing composed vLLM file: {path}")
        observed = sha256(path)
        require(
            observed == expected,
            f"composed vLLM hash mismatch for {relative}: "
            f"expected {expected}, got {observed}",
        )
    for relative, record in pins["profile_patches"].items():
        path = SITE_PACKAGES / relative
        require(path.is_file(), f"missing profile-patched vLLM file: {path}")
        observed = sha256(path)
        expected = record["postimage"]
        require(
            observed == expected,
            f"profile-patched vLLM hash mismatch for {relative}: "
            f"expected {expected}, got {observed}",
        )
    for relative, record in pins["spark_runtime_overlay_files"].items():
        path = SPARK_RUNTIME_ROOT / relative
        require(path.is_file(), f"missing Spark runtime overlay file: {path}")
        observed = sha256(path)
        expected = record["postimage"]
        require(
            observed == expected,
            f"Spark runtime overlay hash mismatch for {relative}: "
            f"expected {expected}, got {observed}",
        )


def verify_imports(pins: dict) -> None:
    for name, record in pins["cutlass_python_lock"]["distributions"].items():
        require(
            importlib.metadata.version(name) == record["version"],
            f"{name} is not exactly {record['version']}",
        )
    importlib.import_module("sparkinfer.moe.fused_moe")
    mixed = importlib.import_module(
        "sparkinfer.moe._shared.kernels.w4a16.mixed_trellis"
    )
    require(
        hasattr(mixed, "compile_mixed_trellis"),
        "SparkInfer mixed Trellis compiler is unavailable",
    )
    extension = importlib.import_module("exllamav3_ext")
    require(
        hasattr(extension, "exl3_gemm"),
        "exllamav3_ext does not expose exl3_gemm",
    )
    exl3 = importlib.import_module(
        "vllm.model_executor.layers.quantization.exl3"
    )
    require(hasattr(exl3, "Exl3Config"), "vLLM EXL3 config is unavailable")
    require(
        importlib.metadata.version("lmcache") == pins["lmcache"]["version"],
        f"lmcache is not exactly {pins['lmcache']['version']}",
    )
    importlib.import_module("cupy")
    connector = importlib.import_module(
        "lmcache.integration.vllm.lmcache_mp_connector"
    )
    adapter = importlib.import_module(
        "lmcache.integration.vllm.vllm_multi_process_adapter"
    )
    require(
        hasattr(connector, "local_server_url_for_worker"),
        "LMCache local-server rank routing helper is unavailable",
    )
    for rank in range(4):
        strategy = adapter.ParallelStrategy(
            use_mla=True,
            vllm_world_size=4,
            vllm_worker_id=rank,
            tp_size=4,
            pp_size=1,
            n_servers=4,
            dcp_size=4,
        )
        require(strategy.kv_world_size == 1, "LMCache shard-local world drift")
        require(strategy.kv_worker_id == 0, "LMCache shard-local worker drift")
        require(strategy.kv_tp_size == 1, "LMCache shard-local TP drift")
        require(
            strategy.kv_readers_per_object == 1,
            "LMCache shard-local reader drift",
        )
        require(strategy.is_writer, "LMCache shard-local rank must be a writer")


def verify_gpu() -> None:
    torch = importlib.import_module("torch")
    require(torch.cuda.is_available(), "CUDA is unavailable")
    capability = tuple(torch.cuda.get_device_capability())
    require(capability == (12, 1), f"expected GB10 SM121, got {capability}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("build", "runtime", "gpu"), required=True)
    args = parser.parse_args()
    pins = json.loads((PREFIX / "pins.json").read_text(encoding="utf-8"))
    require(
        pins["schema"] == "sparkring-public-exl3-pins/v1",
        "wrong EXL3 pin schema",
    )
    verify_sources(pins)
    verify_imports(pins)
    if args.phase == "gpu":
        verify_gpu()
    print(
        json.dumps(
            {
                "schema": "sparkring-public-exl3-verification/v1",
                "phase": args.phase,
                "profile_id": pins["profile_id"],
                "status": "pass",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
