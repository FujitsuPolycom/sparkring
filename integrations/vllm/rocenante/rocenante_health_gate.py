"""Check B12X virtual-diagonal health after vLLM output synchronization.

Status: research-only. These wrappers have a distinct marker from the SIRCL
health gate, so both checks remain installed and execute in wrapper order.
"""

from __future__ import annotations

import importlib
import os

from rocenante_vllm_overlay import require_health


_installed = False


def _checked(callable_):
    try:
        return callable_()
    except BaseException:
        os._exit(70)
        raise RuntimeError("virtual-diagonal health exit unexpectedly returned")


def _wrap_get_output(output_type: type) -> bool:
    original = output_type.get_output
    if getattr(original, "_rocenante_health_gate", False):
        return False

    def get_output(self):
        output = original(self)
        _checked(require_health)
        return output

    get_output._rocenante_health_gate = True
    output_type.get_output = get_output
    return True


def _wrap_worker_output(worker_type: type, method_name: str) -> None:
    original = getattr(worker_type, method_name)
    if getattr(original, "_rocenante_health_gate", False):
        return

    def worker_output(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        if hasattr(output, "get_output"):
            return output
        _checked(require_health)
        return output

    worker_output._rocenante_health_gate = True
    setattr(worker_type, method_name, worker_output)


def install() -> None:
    """Install B12X checks outside any SIRCL output wrappers already present."""

    global _installed
    if _installed:
        return
    wrapped = 0
    for module_name, class_names in (
        ("vllm.v1.worker.gpu.async_utils", ("AsyncOutput", "AsyncPoolingOutput")),
        (
            "vllm.v1.worker.gpu_model_runner",
            ("AsyncGPUModelRunnerOutput", "AsyncGPUPoolingModelRunnerOutput"),
        ),
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for class_name in class_names:
            output_type = getattr(module, class_name, None)
            if output_type is not None:
                wrapped += int(_wrap_get_output(output_type))
    try:
        worker_type = importlib.import_module("vllm.v1.worker.gpu_worker").Worker
    except (AttributeError, ImportError) as error:
        raise RuntimeError(
            "virtual-diagonal health gate found no pinned vLLM GPU Worker"
        ) from error
    for method_name in ("execute_model", "sample_tokens"):
        if not hasattr(worker_type, method_name):
            raise RuntimeError(
                f"virtual-diagonal health gate found no Worker.{method_name} boundary"
            )
        _wrap_worker_output(worker_type, method_name)
    if wrapped == 0:
        raise RuntimeError(
            "virtual-diagonal health gate found no asynchronous output type"
        )
    _installed = True


__all__ = ["install"]
