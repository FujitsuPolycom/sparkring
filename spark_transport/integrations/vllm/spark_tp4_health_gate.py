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


def wrap_execute_model(
    runner_type: type,
    *,
    check: Callable[[], None] = require_native_health,
    abort: Callable[[int], Any] = os._exit,
) -> None:
    original = runner_type.execute_model
    if getattr(original, "_sparkring_health_gate", False):
        return

    def execute_model(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        if hasattr(output, "get_output"):
            return output
        try:
            check()
        except BaseException:
            abort(70)
            raise RuntimeError("SIRCL health abort unexpectedly returned")
        return output

    execute_model._sparkring_health_gate = True
    runner_type.execute_model = execute_model


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
    runners = 0
    for module_name in (
        "vllm.v1.worker.gpu.model_runner",
        "vllm.v1.worker.gpu_model_runner",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        runner_type = getattr(module, "GPUModelRunner", None)
        if runner_type is not None:
            wrap_execute_model(runner_type)
            runners += 1
    if not wrapped or not runners:
        raise RuntimeError("SIRCL health gate found no vLLM asynchronous output type")
    _installed = True
