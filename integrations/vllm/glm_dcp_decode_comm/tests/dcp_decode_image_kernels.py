"""The image's reference kernels for the GPU checks, compiled from the image's source files without importing vLLM.

:func:`load` returns ``fused_q`` (``vllm/models/deepseek_v32/common/kernels.py``) and the DCP all-to-all's
``_dcp_a2a_pack_send`` and ``_dcp_a2a_unpack_combine`` (``vllm/v1/attention/ops/dcp.py``), compiled from:

- the installed vLLM's own two files, found on ``sys.path`` without importing vLLM, when their SHA-256 equal the
  plugin's pins (the serving image);
- otherwise the image's Python sources that ``SPARKRING_GLM53_IMAGE_SOURCES`` names (a directory with a
  ``vllm/`` subtree, as the GLM-5.3 plugins' CPU tests read it), whose SHA-256 must equal the pins too;
- otherwise flat copies ``kernels.py`` and ``dcp.py`` of those two files of serving image 816c6d6a7e96
  (``GLM_DCP_DECODE_IMAGE_REF``, default ``ref/`` beside the project), pinned the same way.

Importing vLLM for two functions would import the whole DeepSeek-V3.2 model package (its ``__init__`` imports
the CUDA model and, through it, every CUDA library the model uses), which the checks do not need. The named
functions and their Triton kernels are compiled from the file at its own path and line numbers (Triton reads a
kernel's source through ``inspect``). ``current_platform`` is a stand-in that keeps ``fused_q`` on its Triton
path (``is_cuda`` false: the CuTe DSL variant serves only the FP8 query) and decides programmatic dependent
launch as vLLM's CUDA platform does (``is_arch_support_pdl``: compute capability 9 or higher).
"""

from __future__ import annotations

import ast
import importlib.util
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

KERNEL_NAMES = ("_DUMMY_CACHE", "_dummy", "_get_cos_sin", "_fp8_ue8m0_quantize", "_fused_q_kernel", "fused_q")
DCP_NAMES = ("_validate_dcp_empty_shard_args", "_dcp_a2a_lse_pack_dim", "_dcp_a2a_pack_send_kernel",
             "_dcp_a2a_unpack_combine_kernel", "_dcp_a2a_pack_send", "_dcp_a2a_unpack_combine")
COPIES = {"kernels.py": "models/deepseek_v32/common/kernels.py", "dcp.py": "v1/attention/ops/dcp.py"}


class _CudaPlatform:
    """``current_platform`` as ``fused_q`` reads it: the Triton kernel, PDL as vLLM's CUDA platform decides it."""

    @staticmethod
    def is_cuda() -> bool:
        return False

    @staticmethod
    def is_arch_support_pdl() -> bool:
        import torch

        try:
            major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
        except Exception:  # noqa: BLE001 - as vLLM's platform: no device, no PDL
            return False
        return major >= 9


def _compile_names(path: Path, names: tuple[str, ...], namespace: dict) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    picked = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            picked.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in names for t in targets):
                picked.append(node)
    found = {node.name if isinstance(node, ast.FunctionDef) else "assignment" for node in picked}
    missing = [name for name in names if not name.startswith("_DUMMY") and name not in found]
    if missing:
        raise RuntimeError(f"{path} does not define {missing}")
    exec(compile(ast.Module(body=picked, type_ignores=[]), str(path), "exec"), namespace)


def _pins() -> dict[str, str]:
    import glm_dcp_decode_comm as plugin

    return {check.path: check.sha256 for check in plugin.FILE_CHECKS if check.package == "vllm"}


def _matching(files: dict[str, Path]) -> bool:
    import glm_dcp_decode_comm as plugin

    pins = _pins()
    return all(path.exists() and plugin.digest(path) == pins[COPIES[name]] for name, path in files.items())


def from_files(files: dict[str, Path], source: str):
    """The three functions compiled from ``files`` (``kernels.py`` and ``dcp.py``), which must match the pins."""
    import torch
    import triton
    import triton.language as tl

    import glm_dcp_decode_comm as plugin

    pins = _pins()
    for name, path in files.items():
        found = plugin.digest(path)
        if found != pins[COPIES[name]]:
            raise RuntimeError(f"{path} has SHA-256 {found}, not the pinned {pins[COPIES[name]]}")
    namespace: dict = {"torch": torch, "triton": triton, "tl": tl, "current_platform": _CudaPlatform(),
                       "Callable": Callable, "__name__": "image_kernels_source"}
    _compile_names(files["kernels.py"], KERNEL_NAMES, namespace)
    _compile_names(files["dcp.py"], DCP_NAMES, namespace)
    return SimpleNamespace(source=source, fused_q=namespace["fused_q"], pack=namespace["_dcp_a2a_pack_send"],
                           unpack=namespace["_dcp_a2a_unpack_combine"])


def installed_files() -> dict[str, Path] | None:
    """The installed vLLM's two files, located without importing vLLM, or None."""
    try:
        spec = importlib.util.find_spec("vllm")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    root = Path(list(spec.submodule_search_locations)[0])
    return {name: root / relative for name, relative in COPIES.items()}


def load():
    """The image's ``fused_q``, pack and unpack-combine, and where they came from."""
    installed = installed_files()
    if installed is not None and _matching(installed):
        return from_files(installed, f"the installed vLLM's files at {installed['kernels.py'].parents[3]} "
                                     "(by path, vLLM not imported)")
    sources = os.environ.get("SPARKRING_GLM53_IMAGE_SOURCES", "").strip()
    if sources:
        root = Path(sources) / "vllm"
        files = {name: root / relative for name, relative in COPIES.items()}
        if all(path.exists() for path in files.values()):
            return from_files(files, f"the image's sources at {root} (SPARKRING_GLM53_IMAGE_SOURCES)")
    directory = Path(os.environ.get("GLM_DCP_DECODE_IMAGE_REF", Path(__file__).resolve().parents[1] / "ref"))
    files = {name: directory / name for name in COPIES}
    if not all(path.exists() for path in files.values()):
        where = "no vLLM is installed" if installed is None else "the installed vLLM is not the pinned build"
        raise RuntimeError(f"{where}, SPARKRING_GLM53_IMAGE_SOURCES names no image sources, and {directory} "
                           "holds no copies of the image's files")
    return from_files(files, f"copies of the image's files in {directory}")


__all__ = ["from_files", "installed_files", "load"]
