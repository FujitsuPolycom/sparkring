"""Regression seam for vocabulary graph startup stream switching.

This intentionally lives outside the main adapter test module so the startup
failure has one small, fast, four-rank-shaped reproducer.  It exercises the
real adapter dispatch while replacing only CUDA tensors and the native
transport handle.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import spark_tp4_vocab_allgather_backend as backend_module


class _Stream:
    def __init__(self, pointer: int) -> None:
        self.cuda_stream = pointer


class _Cuda:
    def __init__(self) -> None:
        self.capturing = False
        self.stream = _Stream(0)

    def is_current_stream_capturing(self) -> bool:
        return self.capturing

    def current_stream(self, device=None) -> _Stream:
        del device
        return self.stream


class _Tensor:
    def __init__(self, shape: tuple[int, int], payload: int = 0) -> None:
        self.shape = shape
        self.payload = payload
        self.dtype = "torch.bfloat16"
        self.device = "cuda:0"
        self.is_cuda = True

    def is_contiguous(self) -> bool:
        return True


class _NativeSession:
    created: list["_NativeSession"] = []

    def __init__(
        self,
        rank: int,
        *,
        graph_only: bool = False,
        control_ports=None,
        graph_cpu_affinity=None,
    ) -> None:
        del control_ports, graph_cpu_affinity
        self.rank = rank
        self.graph_only = graph_only
        self.capture_calls: list[tuple[int, int]] = []
        self.eager_calls: list[tuple[int, int]] = []
        _NativeSession.created.append(self)

    def all_gather(self, input_tensor, output_tensor, query_rows: int, stream) -> None:
        del input_tensor, output_tensor
        self.eager_calls.append((query_rows, int(stream.cuda_stream)))

    def capture(self, input_tensor, output_tensor, query_rows: int, stream) -> None:
        del input_tensor, output_tensor
        self.capture_calls.append((query_rows, int(stream.cuda_stream)))

    def graph_status(self) -> dict[str, object]:
        return {
            "captured_nodes": len(self.capture_calls),
            "published_sequence": 0,
            "consumed_sequence": 0,
            "completed_sequence": 0,
            "overflow_sequence": 0,
            "capture_configured": bool(self.capture_calls),
            "polling_enabled": bool(self.capture_calls),
            "host_native_atomics": True,
            "submit_affinity_verified": True,
            "progress_affinity_verified": True,
            "submit_cpu": 10,
            "progress_cpu": 12,
            "replay_advanced": False,
            "replay_caught_up": False,
            "fatal": False,
        }


def _make_group_type(torch_module: types.ModuleType):
    class Group:
        def __init__(self, rank: int) -> None:
            self.unique_name = "tp:0"
            self.world_size = 4
            self.rank_in_group = rank
            self.original_streams: list[int] = []

        def _all_gather_out_place(self, input_tensor, dim: int):
            assert dim in {-1, 1}
            self.original_streams.append(
                int(torch_module.cuda.current_stream().cuda_stream)
            )
            return _Tensor(
                (input_tensor.shape[0], input_tensor.shape[1] * 4),
                payload=input_tensor.payload,
            )

    return Group


class VocabGraphStreamSwitchRegressionTest(unittest.TestCase):
    def test_four_ranks_isolate_graph_and_handoff_eager_streams(
        self,
    ) -> None:
        cuda = _Cuda()
        torch_module = types.ModuleType("torch")
        torch_module.cuda = cuda
        torch_module.empty = lambda shape, dtype, device: _Tensor(shape)

        group_type = _make_group_type(torch_module)
        parallel_state = types.ModuleType("vllm.distributed.parallel_state")
        parallel_state.GroupCoordinator = group_type
        vllm = types.ModuleType("vllm")
        distributed = types.ModuleType("vllm.distributed")
        vllm.distributed = distributed
        distributed.parallel_state = parallel_state

        audit_events: list[tuple[str, bool, str]] = []
        audit = types.ModuleType("spark_collective_audit")
        audit.record_stock = lambda family, capturing, reason: audit_events.append(
            (family, capturing, reason)
        )
        tp_backend = types.ModuleType("spark_tp4_backend")
        tp_backend._graph_preflight = lambda: (10, 11)
        reporter = types.ModuleType("spark_graph_status_reporter")
        reporter.ensure_status_reporter = lambda rank: None

        modules = {
            "torch": torch_module,
            "vllm": vllm,
            "vllm.distributed": distributed,
            "vllm.distributed.parallel_state": parallel_state,
            "spark_collective_audit": audit,
            "spark_tp4_backend": tp_backend,
            "spark_graph_status_reporter": reporter,
        }
        environment = {
            "VLLM_SPARK_TP4_VOCAB_MODE": "custom",
            "VLLM_SPARK_TP4_GRAPH_Q1": "1",
            "SPARK_TP4_GRAPH_VOCAB_CONTROL_PORT0": "10110",
            "SPARK_TP4_GRAPH_VOCAB_CONTROL_PORT1": "10111",
        }

        backend_module._installed = False
        backend_module._vocab_graph_sessions.clear()
        backend_module._graph_event_counts.clear()
        _NativeSession.created.clear()
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, environment, clear=True),
            patch.object(backend_module, "_NativeVocabSession", _NativeSession),
        ):
            backend_module.install()
            groups = [group_type(rank) for rank in range(4)]
            for rank, group in enumerate(groups):
                stream_a = _Stream(0xA000 + rank)
                stream_b = _Stream(0xB000 + rank)
                stream_c = _Stream(0xC000 + rank)

                cuda.stream = stream_a
                cuda.capturing = False
                warmup = group._all_gather_out_place(
                    _Tensor((5, 38720), payload=rank), -1
                )
                self.assertEqual(warmup.shape, (5, 154880))

                state = group._spark_tp4_vocab_native
                self.assertIsNotNone(state._session)
                self.assertFalse(state._session.graph_only)
                self.assertIsNotNone(state._graph_session)
                self.assertTrue(state._graph_session.graph_only)
                self.assertEqual(group.original_streams, [])
                self.assertEqual(
                    state._session.eager_calls,
                    [(5, 0xA000 + rank)],
                )

                cuda.stream = stream_b
                cuda.capturing = True
                captured = group._all_gather_out_place(
                    _Tensor((5, 38720), payload=rank), -1
                )
                self.assertEqual(captured.shape, (5, 154880))
                self.assertEqual(
                    state._graph_session.capture_calls,
                    [(5, 0xB000 + rank)],
                )

                # vLLM performs noncapturing MTP gathers after capture.
                # They reuse the eager session, whose native event handoff
                # preserves ordering when the caller stream changes.
                cuda.stream = stream_c
                cuda.capturing = False
                group._all_gather_out_place(_Tensor((1, 38720), payload=rank), -1)
                self.assertEqual(
                    group.original_streams, []
                )
                self.assertEqual(
                    state._session.eager_calls,
                    [(5, 0xA000 + rank), (1, 0xC000 + rank)],
                )
                self.assertEqual(state._graph_session.eager_calls, [])

        self.assertEqual(len(_NativeSession.created), 8)
        self.assertEqual(
            sum(item.graph_only for item in _NativeSession.created), 4
        )
        self.assertEqual(audit_events, [])


if __name__ == "__main__":
    unittest.main()
