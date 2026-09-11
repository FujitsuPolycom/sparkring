#!/usr/bin/env python3
"""GPU-free lifecycle probe for the source-composed shared capture stream."""

from __future__ import annotations

import inspect
import json
import os
import threading
from contextlib import contextmanager

import torch
from vllm.distributed import parallel_state
from vllm.v1.worker.gpu import cudagraph_utils


class FakeStream:
    next_id = 1

    def __init__(self, *, device: torch.device) -> None:
        self.device = device
        self.stream_id = FakeStream.next_id
        FakeStream.next_id += 1


class FakeGroup:
    def __init__(self, observed: list[FakeStream]) -> None:
        self.observed = observed

    @contextmanager
    def graph_capture(self, context):
        self.observed.append(context.stream)
        yield context


def main() -> None:
    observed: list[FakeStream] = []
    group = FakeGroup(observed)
    original_stream = parallel_state.torch.cuda.Stream
    original_current_device = parallel_state.torch.cuda.current_device
    original_get_tp_group = parallel_state.get_tp_group
    original_get_pp_group = parallel_state.get_pp_group
    original_graph_capture_context = parallel_state.GraphCaptureContext
    original_dcp = parallel_state._DCP
    original_getpid = parallel_state.os.getpid
    original_flag = os.environ.get("VLLM_SPARK_SHARED_CAPTURE_STREAM")

    try:
        parallel_state.torch.cuda.Stream = FakeStream
        parallel_state.torch.cuda.current_device = lambda: 0
        parallel_state.get_tp_group = lambda: group
        parallel_state.get_pp_group = lambda: group
        parallel_state._DCP = None
        parallel_state._SPARK_SHARED_CAPTURE_STREAMS.clear()
        parallel_state._SPARK_ACTIVE_CAPTURE_STREAMS.clear()
        os.environ["VLLM_SPARK_SHARED_CAPTURE_STREAM"] = "1"

        with parallel_state.graph_capture(torch.device("cuda:0")) as first:
            first_stream = first.stream
            try:
                with parallel_state.graph_capture(torch.device("cuda:0")):
                    raise AssertionError("nested capture was accepted")
            except RuntimeError as error:
                assert "overlapping Spark shared CUDA graph capture" in str(error)

        with parallel_state.graph_capture(torch.device("cuda:0")) as second:
            assert second.stream is first_stream

        holder_entered = threading.Event()
        release_holder = threading.Event()
        holder_errors: list[BaseException] = []

        def hold_capture() -> None:
            try:
                with parallel_state.graph_capture(torch.device("cuda:0")):
                    holder_entered.set()
                    assert release_holder.wait(timeout=5)
            except BaseException as error:
                holder_errors.append(error)

        holder = threading.Thread(target=hold_capture)
        holder.start()
        assert holder_entered.wait(timeout=5)
        try:
            with parallel_state.graph_capture(torch.device("cuda:0")):
                raise AssertionError("cross-thread capture was accepted")
        except RuntimeError as error:
            assert "overlapping Spark shared CUDA graph capture" in str(error)
        release_holder.set()
        holder.join(timeout=5)
        assert not holder.is_alive()
        assert not holder_errors

        try:
            with parallel_state.graph_capture(torch.device("cuda:0")):
                raise LookupError("injected capture failure")
        except LookupError:
            pass
        assert not parallel_state._SPARK_ACTIVE_CAPTURE_STREAMS

        def reject_context(_stream):
            raise MemoryError("injected capture-context setup failure")

        parallel_state.GraphCaptureContext = reject_context
        try:
            with parallel_state.graph_capture(torch.device("cuda:0")):
                raise AssertionError("capture-context setup failure was ignored")
        except MemoryError:
            pass
        assert not parallel_state._SPARK_ACTIVE_CAPTURE_STREAMS
        parallel_state.GraphCaptureContext = original_graph_capture_context
        with parallel_state.graph_capture(torch.device("cuda:0")) as recovered:
            assert recovered.stream is first_stream

        with parallel_state.graph_capture(torch.device("cuda:1")) as other_device:
            assert other_device.stream is not first_stream

        parallel_state.os.getpid = lambda: original_getpid() + 1
        with parallel_state.graph_capture(torch.device("cuda:0")) as other_process:
            assert other_process.stream is not first_stream
        parallel_state.os.getpid = original_getpid

        os.environ["VLLM_SPARK_SHARED_CAPTURE_STREAM"] = "0"
        with parallel_state.graph_capture(torch.device("cuda:0")) as stock_a:
            pass
        with parallel_state.graph_capture(torch.device("cuda:0")) as stock_b:
            pass
        assert stock_a.stream is not stock_b.stream

        assert len(observed) == 18
        full_capture_source = inspect.getsource(
            cudagraph_utils.CudaGraphManager.capture
        )
        assert "stream=torch.cuda.current_stream(self.device)" in full_capture_source
        print(
            json.dumps(
                {
                    "gate": "pass",
                    "shared_same_stream": True,
                    "nested_rejected": True,
                    "cross_thread_overlap_rejected": True,
                    "exception_released_guard": True,
                    "setup_failure_released_guard": True,
                    "device_isolated": True,
                    "pid_isolated": True,
                    "flag_off_fresh_streams": True,
                    "full_capture_uses_current_stream": True,
                    "group_context_entries": len(observed),
                },
                sort_keys=True,
            )
        )
    finally:
        parallel_state.torch.cuda.Stream = original_stream
        parallel_state.torch.cuda.current_device = original_current_device
        parallel_state.get_tp_group = original_get_tp_group
        parallel_state.get_pp_group = original_get_pp_group
        parallel_state.GraphCaptureContext = original_graph_capture_context
        parallel_state._DCP = original_dcp
        parallel_state.os.getpid = original_getpid
        parallel_state._SPARK_SHARED_CAPTURE_STREAMS.clear()
        parallel_state._SPARK_ACTIVE_CAPTURE_STREAMS.clear()
        if original_flag is None:
            os.environ.pop("VLLM_SPARK_SHARED_CAPTURE_STREAM", None)
        else:
            os.environ["VLLM_SPARK_SHARED_CAPTURE_STREAM"] = original_flag


if __name__ == "__main__":
    main()
