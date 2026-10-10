"""The GPU checks' loader of the image's kernels from copies of its files, with Triton stood in."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

import dcp_decode_image_kernels as image_kernels


def test_the_loader_compiles_every_named_function_from_the_image_files(vllm_root):
    namespace = {"torch": torch, "triton": SimpleNamespace(jit=lambda fn: fn, next_power_of_2=lambda n: n),
                 "tl": MagicMock(), "current_platform": image_kernels._CudaPlatform(), "Callable": object,
                 "__name__": "image_kernels_copy"}
    image_kernels._compile_names(vllm_root / image_kernels.COPIES["kernels.py"], image_kernels.KERNEL_NAMES, namespace)
    image_kernels._compile_names(vllm_root / image_kernels.COPIES["dcp.py"], image_kernels.DCP_NAMES, namespace)
    for name in image_kernels.KERNEL_NAMES + image_kernels.DCP_NAMES:
        assert name in namespace, name
    # Compiled at the file's own path and line numbers, so Triton's source lookup finds the kernel's text.
    code = namespace["_dcp_a2a_unpack_combine_kernel"].__code__
    lines = (vllm_root / image_kernels.COPIES["dcp.py"]).read_text(encoding="utf-8").splitlines()
    assert code.co_filename.endswith("dcp.py")
    assert lines[code.co_firstlineno - 1:code.co_firstlineno + 1] == ["@triton.jit",
                                                                      "def _dcp_a2a_unpack_combine_kernel("]
    assert namespace["_dcp_a2a_lse_pack_dim"](torch.bfloat16) == 2
    assert namespace["_validate_dcp_empty_shard_args"](None, None) is False
