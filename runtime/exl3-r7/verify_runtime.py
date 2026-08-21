#!/usr/bin/env python3
"""Fail-closed checks for the installed EXL3 R7 ARM64 runtime."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import platform
import sys
from pathlib import Path


REQUIRED_SOURCE_MARKERS = {
    "vllm.model_executor.layers.quantization.exl3": (
        "r7_routed_experts",
        "VLLM_EXL3_ONLINE_TRELLIS_BITS",
    ),
    "vllm.model_executor.layers.quantization.exl3_online_cache": (
        "VLLM_EXL3_ONLINE_CACHE_DIR",
        "readwrite",
    ),
    "b12x.moe._shared.kernels.w4a16.mixed_trellis": (
        "K3/K4/K5",
        "mixed_trellis3",
    ),
}

# SIRCL ships two collective families: the TP all-reduce and the vocabulary
# all-gather. DCP, indexer, and every other collective use patched NCCL, so
# no hook module exists for them and none is verified here.
SIRCL_HOOK_MARKERS = {
    "sitecustomize": (
        "VLLM_SPARK_TP4_MODE",
        "VLLM_SPARK_TP4_VOCAB_MODE",
    ),
    "spark_tp4_backend": (
        "CudaCommunicator.all_reduce = spark_all_reduce",
        "_spark_tp4_backend",
    ),
    "spark_tp4_vocab_allgather_backend": (
        "GroupCoordinator._all_gather_out_place = spark_vocab_all_gather",
        "_spark_tp4_vocab_backend",
    ),
}
_MAX_SIRCL_ORIGINAL_LINKS = 64


def parameter_names(callable_: object) -> tuple[str, ...]:
    return tuple(inspect.signature(callable_).parameters)


def spark_original_chain(name: str, target: object) -> tuple[object, ...]:
    """Return the complete wrapper chain and reject malformed ownership.

    Spark wrappers preserve the callable they replace on ``_spark_original``.
    Multiple integrations may wrap the same vLLM method, so verification must
    inspect the terminal callable without assuming a fixed wrapper order.
    """

    if not callable(target):
        raise RuntimeError(
            f"SIRCL target {name} wrapper chain contains a non-callable target"
        )

    chain: list[object] = []
    seen: set[int] = set()
    links = 0
    current = target
    while True:
        identity = id(current)
        if identity in seen:
            raise RuntimeError(
                f"SIRCL target {name} wrapper chain contains a cycle; "
                "wrapper chain cycle detected"
            )
        seen.add(identity)
        chain.append(current)
        if not hasattr(current, "_spark_original"):
            return tuple(chain)
        if links >= _MAX_SIRCL_ORIGINAL_LINKS:
            raise RuntimeError(
                f"SIRCL target {name} wrapper chain exceeds "
                f"{_MAX_SIRCL_ORIGINAL_LINKS} original links"
            )
        current = getattr(current, "_spark_original")
        if not callable(current):
            raise RuntimeError(
                f"SIRCL target {name} wrapper chain contains a non-callable original"
            )
        links += 1


def verify_hook_chain(
    name: str,
    target: object,
    expected: tuple[str, ...],
    installed_marker: str,
    mode_enabled: bool,
) -> tuple[str, ...]:
    """Validate one possibly multi-wrapper vLLM hook fail closed."""

    chain = spark_original_chain(name, target)
    actual = parameter_names(chain[-1])
    if actual != expected:
        raise RuntimeError(
            f"SIRCL target {name} signature drift: expected {expected}, got {actual}"
        )
    if mode_enabled and not any(
        bool(getattr(node, installed_marker, False)) for node in chain[:-1]
    ):
        raise RuntimeError(
            f"SIRCL hook marker {installed_marker} is absent from {name} wrapper chain"
        )
    return actual


def verify_sircl_hooks(evidence: dict[str, object]) -> None:
    """Prove that the inherited Spark hooks still target the r34 vLLM ABI."""

    hook_paths: dict[str, str] = {}
    for module, markers in SIRCL_HOOK_MARKERS.items():
        path, source = module_source(module)
        missing = [marker for marker in markers if marker not in source]
        if missing:
            raise RuntimeError(f"{module} lacks SIRCL marker(s): {', '.join(missing)}")
        hook_paths[module] = str(path)

    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.parallel_state import GroupCoordinator

    contracts = (
        (
            "CudaCommunicator.all_reduce",
            CudaCommunicator.all_reduce,
            ("self", "input_"),
            "_spark_tp4_backend",
            "VLLM_SPARK_TP4_MODE",
        ),
        (
            "PyNcclCommunicator.all_gather",
            PyNcclCommunicator.all_gather,
            ("self", "output_tensor", "input_tensor", "stream"),
            "_spark_tp4_allgather_backend",
            # No SIRCL family hooks this target; the signature check proves
            # the patched-NCCL path is unwrapped.
            "",
        ),
        (
            "GroupCoordinator._all_gather_out_place",
            GroupCoordinator._all_gather_out_place,
            ("self", "input_", "dim"),
            "_spark_tp4_vocab_backend",
            "VLLM_SPARK_TP4_VOCAB_MODE",
        ),
    )
    target_signatures: dict[str, list[str]] = {}
    for name, target, expected, installed_marker, mode_variable in contracts:
        actual = verify_hook_chain(
            name,
            target,
            expected,
            installed_marker,
            bool(os.environ.get(mode_variable)),
        )
        target_signatures[name] = list(actual)

    library = Path(
        os.environ.get(
            "SPARK_TP4_LIBRARY",
            "/opt/sparkring/spark_transport/libspark_transport_capi.so",
        )
    )
    if not library.is_file():
        raise RuntimeError(f"SIRCL native library is missing: {library}")
    evidence["sircl"] = {
        "hooks": hook_paths,
        "library": str(library),
        "targets": target_signatures,
    }


def module_source(name: str) -> tuple[Path, str]:
    module = importlib.import_module(name)
    path_text = inspect.getsourcefile(module)
    if not path_text:
        raise RuntimeError(f"cannot locate installed module source: {name}")
    path = Path(path_text).resolve()
    return path, path.read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--installed-only",
        action="store_true",
        help="skip checks that require a visible NVIDIA GPU",
    )
    args = parser.parse_args()

    if platform.machine().lower() not in {"aarch64", "arm64"}:
        raise RuntimeError(f"R7 image requires ARM64, found {platform.machine()}")

    evidence: dict[str, object] = {"architecture": platform.machine(), "modules": {}}
    for module, markers in REQUIRED_SOURCE_MARKERS.items():
        path, source = module_source(module)
        missing = [marker for marker in markers if marker not in source]
        if missing:
            raise RuntimeError(f"{module} lacks R7 marker(s): {', '.join(missing)}")
        evidence["modules"][module] = str(path)

    verify_sircl_hooks(evidence)

    instanttensor_path, instanttensor_source = module_source("instanttensor")
    borrowed_markers = ("BUFFERED",)
    missing = [marker for marker in borrowed_markers if marker not in instanttensor_source]
    if missing:
        # The backend contract may live below the package initializer.
        package_text = "".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in instanttensor_path.parent.rglob("*.py")
        )
        missing = [marker for marker in borrowed_markers if marker not in package_text]
    if missing:
        raise RuntimeError("InstantTensor lacks the BUFFERED backend policy")
    evidence["instanttensor"] = str(instanttensor_path)

    import torch

    extension_path = os.environ.get("VLLM_EXL3_EXT_PATH")
    if extension_path:
        sys.path.insert(0, extension_path)
    extension = importlib.import_module("exllamav3_ext")
    required_exports = (
        "exl3_gemm",
        "exl3_moe_fused",
        "exl3_moe_fused_retile",
        "exl3_moe_max_concurrency",
    )
    missing = [name for name in required_exports if not hasattr(extension, name)]
    if missing:
        raise RuntimeError(
            "ExLlamaV3 extension lacks required exports: " + ", ".join(missing)
        )
    evidence["exllamav3_ext"] = inspect.getfile(extension)

    evidence["torch"] = torch.__version__
    evidence["torch_cuda"] = torch.version.cuda
    if not args.installed_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        capability = torch.cuda.get_device_capability()
        evidence["cuda_capability"] = list(capability)
        if capability != (12, 1):
            raise RuntimeError(f"R7 Spark profile requires SM121, found SM{capability[0]}{capability[1]}")

    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
