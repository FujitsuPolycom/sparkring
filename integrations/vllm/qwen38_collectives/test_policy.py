"""Check rank agreement and collective eligibility without GPU execution."""

import importlib.util
import os
import sys
import types
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path


class PolicyTest(unittest.TestCase):
    def load(self, mode, limit=61440):
        os.environ.update(
            QWEN_DISPATCH_MODE=mode,
            QWEN_DISPATCH_AR_BYTES=str(limit),
            QWEN_DISPATCH_TRACE="0",
        )
        spec = importlib.util.spec_from_file_location(
            "policy", Path(__file__).with_name("qwen38_collective_policy.py")
        )
        p = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p)
        self.policy = p
        self.addCleanup(
            lambda: sys.meta_path.remove(
                next(x for x in sys.meta_path if isinstance(x, p.Finder))
            )
        )

        class Adapter:
            rank = 0
            world_size = 4
            group = None

            def _exchange_vote(self, reason, limits):
                return reason

            def should_custom_ar(self, inp):
                return inp.contiguous

            def should_all_gather(self, inp, dim):
                return inp.contiguous and dim == 0

        def vote(rows, policy, group):
            rows[:] = [policy] * 4

        module = types.SimpleNamespace(
            B12xRoceAllReduce=Adapter,
            dist=types.SimpleNamespace(all_gather_object=vote),
            torch=types.SimpleNamespace(
                cuda=types.SimpleNamespace(is_current_stream_capturing=lambda: True)
            ),
        )
        p.patch_adapter(module)
        return Adapter(), module

    def tensor(self, rows, contiguous=True):
        return types.SimpleNamespace(
            shape=(rows, 2560),
            dtype="bf16",
            contiguous=contiguous,
            numel=lambda: rows * 2560,
            element_size=lambda: 2,
        )

    def test_cutoff_does_not_override_runtime_rejection(self):
        a, _ = self.load("both")
        self.assertTrue(a.should_custom_ar(self.tensor(12)))
        self.assertFalse(a.should_custom_ar(self.tensor(13)))
        self.assertFalse(a.should_custom_ar(self.tensor(4, False)))

    def test_gather_policy_is_independent(self):
        a, _ = self.load("reduce")
        t = self.tensor(4)
        self.assertTrue(a.should_custom_ar(t))
        self.assertFalse(a.should_all_gather(t, 0))
        a, _ = self.load("both")
        self.assertTrue(a.should_all_gather(t, 0))
        self.assertFalse(a.should_all_gather(t, 1))

    def test_nccl_rejects_both(self):
        a, _ = self.load("nccl")
        t = self.tensor(4)
        self.assertFalse(a.should_custom_ar(t))
        self.assertFalse(a.should_all_gather(t, 0))

    def test_trace_off_does_not_query_cuda_capture_state(self):
        a, module = self.load("both", 20480)

        def unexpected():
            raise AssertionError("Tracing must not inspect CUDA state when disabled")

        module.torch.cuda.is_current_stream_capturing = unexpected
        self.assertTrue(a.should_custom_ar(self.tensor(4)))
        self.assertFalse(a.should_custom_ar(self.tensor(8)))
        self.assertTrue(a.should_all_gather(self.tensor(4), 0))

    def test_rank_disagreement_fails_before_dispatch(self):
        a, m = self.load("reduce")
        self.assertIsNone(a._exchange_vote(None, (2097152, 16777216)))

        def different(rows, policy, group):
            rows[:] = [policy, policy, ("both", 61440), policy]

        m.dist.all_gather_object = different
        with self.assertRaisesRegex(RuntimeError, "differs across ranks"):
            a._exchange_vote(None, (2097152, 16777216))

    def test_changed_adapter_source_is_rejected(self):
        self.load("both")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "adapter.py"
            source.write_text("changed source")
            spec = types.SimpleNamespace(origin=str(source), loader=object())
            with patch.object(
                self.policy.importlib.machinery.PathFinder,
                "find_spec",
                return_value=spec,
            ):
                with self.assertRaisesRegex(SystemExit, "source identity mismatch"):
                    self.policy.Finder().find_spec(
                        "vllm.distributed.device_communicators.b12x_roce_all_reduce", []
                    )


if __name__ == "__main__":
    unittest.main()
