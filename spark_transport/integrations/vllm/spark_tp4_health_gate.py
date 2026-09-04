"""Check SIRCL host health after vLLM's existing output synchronization."""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable

from spark_tp4_backend import require_native_health


_installed = False


def wrap_get_output(
    output_type: type,
    *,
    check: Callable[[], None] = require_native_health,
    abort: Callable[[int], Any] = os._exit,
) -> None:
    original = output_type.get_output
    if getattr(original, "_sparkring_health_gate", False):
        return

    def get_output(self):
        output = original(self)
        try:
            check()
        except BaseException:
            abort(70)
            raise RuntimeError("SIRCL health abort unexpectedly returned")
        return output

    get_output._sparkring_health_gate = True
    output_type.get_output = get_output


def wrap_worker_output(
    worker_type: type,
    method_name: str,
    *,
    check: Callable[[], None] = require_native_health,
    abort: Callable[[int], Any] = os._exit,
) -> None:
    original = getattr(worker_type, method_name)
    if getattr(original, "_sparkring_health_gate", False):
        return

    def worker_output(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        if output is None or hasattr(output, "get_output"):
            return output
        try:
            check()
        except BaseException:
            abort(70)
            raise RuntimeError("SIRCL health abort unexpectedly returned")
        return output

    worker_output._sparkring_health_gate = True
    setattr(worker_type, method_name, worker_output)


def install() -> None:
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
                wrap_get_output(output_type)
                wrapped += 1
    try:
        worker_module = importlib.import_module("vllm.v1.worker.gpu_worker")
        worker_type = worker_module.Worker
    except (AttributeError, ImportError) as error:
        raise RuntimeError("SIRCL health gate found no pinned vLLM GPU Worker") from error
    for method_name in ("execute_model", "sample_tokens"):
        if not hasattr(worker_type, method_name):
            raise RuntimeError(
                f"SIRCL health gate found no Worker.{method_name} output boundary"
            )
        wrap_worker_output(worker_type, method_name)
    if not wrapped:
        raise RuntimeError("SIRCL health gate found no asynchronous output type")
    _installed = True
