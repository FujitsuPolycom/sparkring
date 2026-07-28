"""GPU-free regression tests for the GLM-5.2 V2 MTP compatibility patch."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import spark_glm52_mtp_index_reuse as reuse


class _FakeBuffer:
    shape = (32, 2048)


class _FakeWrongShapeBuffer:
    shape = (32, 1024)


class _FakeMLA:
    def __init__(self, buffer: _FakeBuffer) -> None:
        self.skip_topk = False
        self.topk_indices_buffer = buffer
        self.indexer_calls = 0
        self.reuse_calls = 0

    def forward(self) -> None:
        if self.skip_topk:
            self.reuse_calls += 1
        else:
            self.indexer_calls += 1


class _FakePredictor:
    def __init__(self, buffer: _FakeBuffer) -> None:
        self.mla = _FakeMLA(buffer)
        self.indexer = SimpleNamespace(topk_indices_buffer=buffer)
        self.indexer_op = SimpleNamespace(topk_indices_buffer=buffer)

    def set_skip_topk(self, skip: bool) -> None:
        self.mla.skip_topk = skip

    def forward(self) -> None:
        self.mla.forward()

    def named_modules(self):
        yield "layer.mtp_block.self_attn.indexer", self.indexer
        yield "layer.mtp_block.self_attn.indexer.indexer_op", self.indexer_op
        yield "layer.mtp_block.self_attn.mla_attn", self.mla


class _FakeDraft:
    def __init__(self, buffer: _FakeBuffer) -> None:
        self.model = _FakePredictor(buffer)


class _FakeTarget:
    def __init__(self, buffer: _FakeBuffer) -> None:
        self.model = SimpleNamespace(topk_indices_buffer=buffer)


class _FakeTargetWithoutRootBuffer:
    def __init__(self) -> None:
        self.model = SimpleNamespace()


class _FakeAutoRegressiveSpeculator:
    def capture(self) -> None:
        self._prefill()
        self._run_model()

    def _run_model(self) -> None:
        self.model.model.forward()

    def _prefill(self) -> None:
        self._run_model()

    def _multi_step_decode(self) -> None:
        for _ in range(1, self.num_speculative_steps):
            self._run_model()

    def propose(self) -> str:
        self._prefill()
        self._multi_step_decode()
        return "tokens"


class _FakeMTPSpeculator(_FakeAutoRegressiveSpeculator):
    def load_draft_model(self, target_model, target_attn_layer_names):
        return self.model


def _fake_load_eagle_model(target_model, vllm_config):
    raise AssertionError("fingerprint-only fake should not be called")


def _fake_mla_forward(self):
    self.forward()


class GLM52MTPIndexReuseTest(unittest.TestCase):
    def setUp(self) -> None:
        reuse.uninstall()
        reuse.reset_stats()
        self.buffer = _FakeBuffer()
        self.speculator = _FakeMTPSpeculator()
        self.speculator.model = _FakeDraft(self.buffer)
        self.speculator.num_speculative_steps = 4
        self.speculator.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["DeepSeekMTPModel"],
                index_share_for_mtp_iteration=True,
                index_topk=2048,
                num_nextn_predict_layers=1,
            )
        )
        self.speculator.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["GlmMoeDsaForCausalLM"],
                )
            ),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=4,
            ),
            speculative_config=SimpleNamespace(method="mtp"),
        )
        self.target = _FakeTarget(self.buffer)

        self.bindings = reuse._RuntimeBindings(
            version="test-vllm",
            mtp_speculator_cls=_FakeMTPSpeculator,
            autoregressive_cls=_FakeAutoRegressiveSpeculator,
            deepseek_predictor_cls=_FakePredictor,
            load_eagle_model=_fake_load_eagle_model,
            mla_forward=_fake_mla_forward,
        )
        self.contract = reuse._SourceContract(
            version=self.bindings.version,
            fingerprints=reuse._fingerprint_bindings(self.bindings),
        )
        self.env = patch.dict(
            os.environ,
            {
                "VLLM_DCP_GLOBAL_TOPK": "1",
                "VLLM_DCP_SHARD_DRAFT": "1",
            },
            clear=False,
        )
        self.env.start()

    def tearDown(self) -> None:
        reuse.uninstall()
        self.env.stop()

    def _install(self) -> None:
        result = reuse._install(self.bindings, (self.contract,))
        self.assertEqual(result.status, "installed")

    def test_unpatched_v2_recomputes_index_on_every_serial_step(self) -> None:
        self.speculator.load_draft_model(self.target, set())

        self.assertEqual(self.speculator.propose(), "tokens")

        mla = self.speculator.model.model.mla
        self.assertEqual(mla.indexer_calls, 4)
        self.assertEqual(mla.reuse_calls, 0)

    def test_patch_computes_once_then_reuses_for_remaining_steps(self) -> None:
        self._install()
        self.assertTrue(
            getattr(
                _FakeMTPSpeculator.propose,
                "_spark_glm52_mtp_index_reuse_wrapper",
                False,
            )
        )
        self.speculator.load_draft_model(self.target, set())

        self.assertEqual(self.speculator.propose(), "tokens")

        mla = self.speculator.model.model.mla
        self.assertEqual(mla.indexer_calls, 1)
        self.assertEqual(mla.reuse_calls, 3)
        self.assertFalse(mla.skip_topk)
        self.assertEqual(
            reuse.get_stats(),
            {
                "activations": 1,
                "buffer_validations": 1,
                "compute_arms": 3,
                "reuse_arms": 1,
                "proposals_completed": 1,
                "proposals_failed": 0,
                "logical_step0_compute_forwards": 1,
                "logical_reuse_forwards": 3,
                "prefills_failed": 0,
            },
        )

    def test_dcp1_physical_slot_reuse_requires_remap_contract(self) -> None:
        self.speculator.vllm_config.parallel_config.decode_context_parallel_size = 1
        self._install()

        with self.assertRaisesRegex(
            RuntimeError,
            "SPARK_B12X_DCP1_PHYSICAL_REMAP is not enabled",
        ):
            self.speculator.load_draft_model(self.target, set())

    def test_dcp1_reuses_physical_slots_after_remap_contract(self) -> None:
        self.speculator.vllm_config.parallel_config.decode_context_parallel_size = 1
        self._install()
        with patch.dict(
            os.environ,
            {"SPARK_B12X_DCP1_PHYSICAL_REMAP": "1"},
        ):
            self.speculator.load_draft_model(self.target, set())

        self.assertEqual(self.speculator.propose(), "tokens")
        mla = self.speculator.model.model.mla
        self.assertEqual((mla.indexer_calls, mla.reuse_calls), (1, 3))
        self.assertFalse(mla.skip_topk)

    def test_dcp2_reuses_global_logical_slots(self) -> None:
        self.speculator.vllm_config.parallel_config.decode_context_parallel_size = 2
        self._install()

        self.speculator.load_draft_model(self.target, set())
        self.assertEqual(self.speculator.propose(), "tokens")
        mla = self.speculator.model.model.mla
        self.assertEqual((mla.indexer_calls, mla.reuse_calls), (1, 3))
        self.assertFalse(mla.skip_topk)

    def test_unsupported_dcp_size_still_fails_closed(self) -> None:
        self.speculator.vllm_config.parallel_config.decode_context_parallel_size = 3
        self._install()

        with self.assertRaisesRegex(
            RuntimeError, "decode_context_parallel_size is 3, not 1, 2, or 4"
        ):
            self.speculator.load_draft_model(self.target, set())

    def test_failed_step0_never_arms_reuse_of_stale_indices(self) -> None:
        self._install()
        self.speculator.load_draft_model(self.target, set())
        mla = self.speculator.model.model.mla

        with patch.object(mla, "forward", side_effect=RuntimeError("step0 failed")):
            with self.assertRaisesRegex(RuntimeError, "step0 failed"):
                self.speculator.propose()

        self.assertFalse(mla.skip_topk)
        stats = reuse.get_stats()
        self.assertEqual(stats["reuse_arms"], 0)
        self.assertEqual(stats["prefills_failed"], 1)
        self.assertEqual(stats["proposals_failed"], 1)

    def test_failure_after_step0_restores_safe_compute_mode(self) -> None:
        self._install()
        self.speculator.load_draft_model(self.target, set())
        mla = self.speculator.model.model.mla

        with patch.object(
            self.speculator,
            "_multi_step_decode",
            side_effect=RuntimeError("decode failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                self.speculator.propose()

        self.assertFalse(mla.skip_topk)
        stats = reuse.get_stats()
        self.assertEqual(stats["reuse_arms"], 1)
        self.assertEqual(stats["proposals_failed"], 1)

    def test_unknown_source_fingerprint_fails_closed_without_patching(self) -> None:
        bad_contract = reuse._SourceContract(
            version=self.bindings.version,
            fingerprints={
                **self.contract.fingerprints,
                "MTPSpeculator": "0" * 64,
            },
        )

        with self.assertRaisesRegex(RuntimeError, "unsupported vLLM source"):
            reuse._install(self.bindings, (bad_contract,))

        self.assertNotIn("_prefill", _FakeMTPSpeculator.__dict__)
        self.assertNotIn("propose", _FakeMTPSpeculator.__dict__)

    def test_missing_target_root_buffer_uses_coherent_draft_local_buffer(self) -> None:
        self._install()
        target = _FakeTargetWithoutRootBuffer()

        self.speculator.load_draft_model(target, set())
        self.speculator.propose()

        mla = self.speculator.model.model.mla
        self.assertEqual((mla.indexer_calls, mla.reuse_calls), (1, 3))

    def test_split_draft_buffers_fail_before_reuse_is_enabled(self) -> None:
        self._install()
        self.speculator.model.model.indexer_op.topk_indices_buffer = _FakeBuffer()

        with self.assertRaisesRegex(RuntimeError, "one shared identity"):
            self.speculator.load_draft_model(
                _FakeTargetWithoutRootBuffer(),
                set(),
            )

        self.assertFalse(self.speculator.model.model.mla.skip_topk)

    def test_wrong_draft_buffer_shape_fails_before_reuse_is_enabled(self) -> None:
        self._install()
        predictor = self.speculator.model.model
        wrong = _FakeWrongShapeBuffer()
        predictor.indexer.topk_indices_buffer = wrong
        predictor.indexer_op.topk_indices_buffer = wrong
        predictor.mla.topk_indices_buffer = wrong

        with self.assertRaisesRegex(RuntimeError, r"expected \[tokens, 2048\]"):
            self.speculator.load_draft_model(
                _FakeTargetWithoutRootBuffer(),
                set(),
            )

        self.assertFalse(predictor.mla.skip_topk)

    def test_disabled_toggle_does_not_import_or_patch_vllm(self) -> None:
        with patch.dict(
            os.environ,
            {"SPARK_GLM52_MTP_INDEX_REUSE": "0"},
            clear=False,
        ):
            result = reuse.install()

        self.assertEqual(result.status, "disabled")
        self.assertNotIn("_prefill", _FakeMTPSpeculator.__dict__)
        self.assertNotIn("propose", _FakeMTPSpeculator.__dict__)

    def test_checkpoint_without_share_flag_is_left_unchanged(self) -> None:
        self._install()
        self.speculator.draft_model_config.hf_config.index_share_for_mtp_iteration = (
            False
        )
        self.speculator.load_draft_model(self.target, set())

        self.speculator.propose()

        mla = self.speculator.model.model.mla
        self.assertEqual((mla.indexer_calls, mla.reuse_calls), (4, 0))
        self.assertEqual(reuse.get_stats()["activations"], 0)

    def test_capture_sees_compute_prefill_then_reuse_decode_state(self) -> None:
        self._install()
        self.speculator.load_draft_model(self.target, set())

        self.speculator.capture()

        mla = self.speculator.model.model.mla
        self.assertEqual((mla.indexer_calls, mla.reuse_calls), (1, 1))
        self.assertTrue(mla.skip_topk)

    def test_uninstall_is_an_in_process_rollback(self) -> None:
        self._install()
        self.speculator.load_draft_model(self.target, set())
        self.speculator.propose()

        self.assertTrue(reuse.uninstall())
        fresh = _FakeMTPSpeculator()
        fresh.model = _FakeDraft(self.buffer)
        fresh.num_speculative_steps = 4
        fresh.propose()

        self.assertEqual(fresh.model.model.mla.indexer_calls, 4)
        self.assertEqual(fresh.model.model.mla.reuse_calls, 0)


if __name__ == "__main__":
    unittest.main()
