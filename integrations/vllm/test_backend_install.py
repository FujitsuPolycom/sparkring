try:
    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )
except ModuleNotFoundError:
    if __name__ != "__main__":
        import pytest

        pytest.skip(
            "vLLM is only available in the GLM runtime container",
            allow_module_level=True,
        )
    raise

assert getattr(CudaCommunicator.all_reduce, "_spark_tp4_backend", False)
print("spark TP4 vLLM backend installed")
