"""CPU tests proving the shape tracer and flight recorder never skip the traced call."""

from __future__ import annotations

import json
import logging
import sys
import types

from . import flight_recorder
from . import shape_trace


class _Tensor:
    def __init__(self, shape=(4, 6144)) -> None:
        self.shape = shape
        self.dtype = "torch.bfloat16"

    def stride(self):
        return (self.shape[1], 1)

    def element_size(self):
        return 2

    def numel(self):
        return self.shape[0] * self.shape[1]

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return 0x1000


def _install_fake_communicator(monkeypatch):
    calls = []

    class CudaCommunicator:
        unique_name = "tp"

        def all_reduce(self, input_):
            calls.append(input_)
            return "reduced"

    module = types.ModuleType("vllm.distributed.device_communicators.cuda_communicator")
    module.CudaCommunicator = CudaCommunicator
    for name in ("vllm", "vllm.distributed", "vllm.distributed.device_communicators"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(shape_trace, "_installed", False)
    monkeypatch.setattr(shape_trace, "_write_failed", False)
    shape_trace._counts.clear()
    return CudaCommunicator, calls


def test_shape_trace_records_first_call_then_calls_original(monkeypatch, tmp_path):
    communicator, calls = _install_fake_communicator(monkeypatch)
    output = tmp_path / "trace.jsonl"
    monkeypatch.setenv("VLLM_SPARK_TRACE_PATH", str(output))
    shape_trace.install()
    tensor = _Tensor()
    assert communicator().all_reduce(tensor) == "reduced"
    assert calls == [tensor]
    record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert record["shape"] == [4, 6144] and record["bytes"] == 4 * 6144 * 2
    assert record["count"] == 1


def test_shape_trace_write_failure_disables_tracing_without_skipping_collective(
    monkeypatch, tmp_path, caplog
):
    communicator, calls = _install_fake_communicator(monkeypatch)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setenv("VLLM_SPARK_TRACE_PATH", str(blocker / "trace.jsonl"))
    shape_trace.install()
    tensor = _Tensor()
    with caplog.at_level(logging.ERROR, logger=shape_trace.__name__):
        assert communicator().all_reduce(tensor) == "reduced"
        assert communicator().all_reduce(_Tensor((8, 6144))) == "reduced"
    assert len(calls) == 2
    assert shape_trace._write_failed is True
    assert "Shape trace disabled" in caplog.text
    # After the failure, later calls skip shape counting entirely.
    assert list(shape_trace._counts.values()) == [1]


def _reset_recorder(monkeypatch):
    monkeypatch.setenv("SPARK_TP4_FLIGHT_RECORDER", "1")
    monkeypatch.setattr(flight_recorder, "_active", False)
    monkeypatch.setattr(flight_recorder, "_recording_failed", False)
    monkeypatch.setattr(flight_recorder, "_sequence", 0)


def test_flight_recorder_passes_positional_arguments_and_survives_record_failure(
    monkeypatch, caplog
):
    _reset_recorder(monkeypatch)
    flight_recorder.activate(2)
    received = []

    def run_fused_paged_indexer(*args, **kwargs):
        received.append((args, kwargs))
        return "indexer-result"

    module = types.ModuleType("fake_b12x")
    module.run_fused_paged_indexer = run_fused_paged_indexer
    flight_recorder._wrap(module)
    wrapped = module.run_fused_paged_indexer
    flight_recorder._wrap(module)
    assert module.run_fused_paged_indexer is wrapped  # wrapping twice does not nest
    broken = types.SimpleNamespace(shape=None)  # int(None) fails inside recording
    with caplog.at_level(logging.ERROR, logger=flight_recorder.__name__):
        result = module.run_fused_paged_indexer("positional", q_bytes=broken)
    assert result == "indexer-result"
    assert received == [(("positional",), {"q_bytes": broken})]
    assert flight_recorder._recording_failed is True
    assert flight_recorder._active is False
    assert "Flight recorder disabled" in caplog.text
    # A failed recorder cannot be re-armed within the process.
    flight_recorder.activate(2)
    assert flight_recorder._active is False


def test_flight_recorder_collective_failure_disables_recording_only(monkeypatch):
    _reset_recorder(monkeypatch)
    flight_recorder.activate(0)
    stream = types.SimpleNamespace(cuda_stream=7)
    flight_recorder.record_collective("all_reduce", stream, _Tensor())
    assert flight_recorder._sequence == 1
    flight_recorder.record_collective("all_reduce", stream, object())
    assert flight_recorder._recording_failed is True
    flight_recorder.record_collective("all_reduce", stream, _Tensor())
    assert flight_recorder._sequence == 1


def test_import_hook_installs_one_finder_and_wraps_loaded_module(monkeypatch):
    _reset_recorder(monkeypatch)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.delitem(sys.modules, flight_recorder._TARGET, raising=False)
    flight_recorder.install_b12x_import_hook()
    flight_recorder.install_b12x_import_hook()
    finders = [f for f in sys.meta_path if isinstance(f, flight_recorder._TracingFinder)]
    assert len(finders) == 1

    loaded = types.ModuleType(flight_recorder._TARGET)
    loaded.run_fused_paged_indexer = lambda **kwargs: "loaded"
    monkeypatch.setitem(sys.modules, flight_recorder._TARGET, loaded)
    flight_recorder.install_b12x_import_hook()
    assert getattr(loaded.run_fused_paged_indexer, "_spark_tp4_flight_recorder", False)
    assert loaded.run_fused_paged_indexer(q_bytes=None) == "loaded"


def test_import_hook_is_inert_when_disabled(monkeypatch):
    monkeypatch.setenv("SPARK_TP4_FLIGHT_RECORDER", "0")
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    before = list(sys.meta_path)
    flight_recorder.install_b12x_import_hook()
    assert sys.meta_path == before
    assert not flight_recorder.enabled()
