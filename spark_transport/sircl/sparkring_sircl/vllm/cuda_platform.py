"""SIRCL's vLLM platform: vLLM's CUDA platform with SIRCL's device communicator.

``CudaPlatform`` is vLLM's resolved CUDA platform class (NVML or non-NVML), so
the subclass inherits every classmethod and the ``PlatformEnum.CUDA`` identity;
only :meth:`SirclCudaPlatform.get_device_communicator_cls` changes.
"""

from __future__ import annotations

from vllm.platforms.cuda import CudaPlatform

COMMUNICATOR = "sparkring_sircl.vllm.communicator.SirclCudaCommunicator"


class SirclCudaPlatform(CudaPlatform):
    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return COMMUNICATOR
