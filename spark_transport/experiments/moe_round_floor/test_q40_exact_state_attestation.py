import tempfile
import unittest
from pathlib import Path

from spark_transport.experiments.moe_round_floor.q40_exact_state_attestation_overlay import (
    INPUT_SHA256,
    OUTPUT_SHA256,
    PATCHED_EXL3_SHA256,
    ExactQ40StateAttestationOverlayError,
    install,
    sha256_bytes,
    transform,
)
from spark_transport.experiments.moe_round_floor.q40_exact_state_overlay import (
    OUTPUT_SHA256 as EXPECTED_PATCHED_EXL3_SHA256,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXACT_MODEL_RUNNER = (
    REPOSITORY_ROOT
    / "runtime"
    / "exl3-r7"
    / "test-fixtures"
    / "vllm"
    / "v1"
    / "worker"
    / "gpu"
    / "model_runner.py.fixture"
)


class Q40ExactStateAttestationTest(unittest.TestCase):
    def test_exact_runner_gains_the_pre_graph_fail_closed_attestation(self) -> None:
        source = EXACT_MODEL_RUNNER.read_bytes()
        self.assertEqual(sha256_bytes(source), INPUT_SHA256)
        self.assertEqual(PATCHED_EXL3_SHA256, EXPECTED_PATCHED_EXL3_SHA256)

        output = transform(source)
        self.assertEqual(sha256_bytes(output), OUTPUT_SHA256)
        text = output.decode("utf-8")
        compile(text, "model_runner.py", "exec")

        call = text.index("self._attest_q40_exact_state_policy()")
        sampler = text.index("# Only run sampler/pooler", call)
        self.assertLess(call, sampler)
        method = text.index("def _attest_q40_exact_state_policy")
        route_capture = text.index("def _init_q40_route_capturer", method)
        contract = text[method:route_capture]
        for required in (
            "VLLM_EXL3_PREFILL_BLOCK_M must remain unset",
            '"VLLM_EXL3_PREFILL_BLOCK_M" in os.environ',
            "target_ids != list(range(3, 78))",
            "len(draft_uniform) != 1",
            'runtime["q40"]',
            'expected_geometry = {"decode": (32, 32, 8), "q40": (40, 40, 8)}',
            "prefill_block not in (32, 64)",
            'mixed["prefill_tile_config"]',
            "runtime_again is not runtime",
            "after_cache != before_cache",
            '"q40_unique_storage_bytes"',
            '"q40_buffer_cache_keys"',
            '"q40_exact_bf16_equal_to_general_prefill"',
            "torch.equal(q40_output, prefill_output)",
            "torch.count_nonzero(q40_output).item()",
            "quant._apply_mixed_rank_sliced(",
            'states["prefill"]',
            'receipt_value = os.environ["SPARK_Q40_EXACT_STATE_ATTEST_PATH"]',
            'rank{self.dcp_rank}.json',
            "receipt_value != expected_receipt",
            ".clone()",
            PATCHED_EXL3_SHA256,
        ):
            self.assertIn(required, contract)

    def test_runner_hash_drift_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            ExactQ40StateAttestationOverlayError, "input hash mismatch"
        ):
            transform(EXACT_MODEL_RUNNER.read_bytes() + b"\n")

    def test_local_image_identity_is_embedded_and_changes_output_hash(self) -> None:
        local_image_id = "sha256:" + "1" * 64
        output = transform(
            EXACT_MODEL_RUNNER.read_bytes(), image_id=local_image_id
        )
        text = output.decode("utf-8")
        self.assertIn(f'if image_id != "{local_image_id}"', text)
        self.assertNotEqual(sha256_bytes(output), OUTPUT_SHA256)

    def test_invalid_local_image_identity_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            ExactQ40StateAttestationOverlayError, "image ID must be"
        ):
            transform(EXACT_MODEL_RUNNER.read_bytes(), image_id="latest")

    def test_install_is_exclusive_and_returns_pinned_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model_runner.py"
            receipt = install(EXACT_MODEL_RUNNER, output)
            self.assertEqual(receipt["input_sha256"], INPUT_SHA256)
            self.assertEqual(receipt["output_sha256"], OUTPUT_SHA256)
            self.assertEqual(sha256_bytes(output.read_bytes()), OUTPUT_SHA256)
            with self.assertRaisesRegex(
                ExactQ40StateAttestationOverlayError, "refusing to overwrite"
            ):
                install(EXACT_MODEL_RUNNER, output)


if __name__ == "__main__":
    unittest.main()
