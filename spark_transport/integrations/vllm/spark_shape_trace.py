"""Low-overhead vLLM all-reduce shape tracer.

Install this module through the adjacent ``sitecustomize.py`` and set
``VLLM_SPARK_TRACE_ALLREDUCE=1``. The tracer observes dispatch inputs and then
calls the original communicator unchanged.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

_installed = False
_lock = threading.Lock()
_counts: dict[tuple[Any, ...], int] = defaultdict(int)


def _rank() -> str:
    return os.getenv("RANK", os.getenv("LOCAL_RANK", "unknown"))


def _output_path() -> Path:
    configured = os.getenv("VLLM_SPARK_TRACE_PATH")
    if configured:
        return Path(configured)
    return Path(f"/tmp/spark-allreduce-rank{_rank()}-{os.getpid()}.jsonl")


def _should_emit(count: int) -> bool:
    return count == 1 or (count & (count - 1)) == 0


def install() -> None:
    global _installed
    if _installed:
        return

    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )

    original = CudaCommunicator.all_reduce
    if getattr(original, "_spark_shape_trace", False):
        _installed = True
        return

    def traced_all_reduce(self: Any, input_: Any) -> Any:
        shape = tuple(int(size) for size in input_.shape)
        stride = tuple(int(value) for value in input_.stride())
        element_size = int(input_.element_size())
        key = (
            getattr(self, "unique_name", ""),
            shape,
            stride,
            str(input_.dtype),
            element_size,
            bool(input_.is_contiguous()),
        )
        with _lock:
            _counts[key] += 1
            count = _counts[key]
            if _should_emit(count):
                record = {
                    "unix_ns": time.time_ns(),
                    "pid": os.getpid(),
                    "rank": _rank(),
                    "group": key[0],
                    "shape": shape,
                    "stride": stride,
                    "dtype": key[3],
                    "element_size": element_size,
                    "elements": int(input_.numel()),
                    "bytes": int(input_.numel()) * element_size,
                    "contiguous": key[5],
                    "count": count,
                }
                output = _output_path()
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        return original(self, input_)

    traced_all_reduce._spark_shape_trace = True  # type: ignore[attr-defined]
    traced_all_reduce._spark_original = original  # type: ignore[attr-defined]
    CudaCommunicator.all_reduce = traced_all_reduce
    _installed = True
