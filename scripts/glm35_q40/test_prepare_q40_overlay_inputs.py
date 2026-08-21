"""Offline tests for the exact-Q40 overlay input producer."""

from __future__ import annotations

import hashlib
import shutil
import unittest
from pathlib import Path

from scripts.glm35_q40 import (
    prepare_q40_overlay_inputs as producer,
)

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "runtime" / "exl3-r7" / "test-fixtures" / "vllm"
EXL3_FIXTURE = FIXTURES / "model_executor/layers/quantization/exl3.py.fixture"
MODEL_RUNNER_FIXTURE = FIXTURES / "v1/worker/gpu/model_runner.py.fixture"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ConstantsMatchFixturesTest(unittest.TestCase):
    """The producer must target exactly what the overlays accept."""

    def test_exl3_output_hash_is_the_tracked_fixture(self) -> None:
        self.assertEqual(sha256_file(EXL3_FIXTURE), producer.EXL3_OUTPUT_SHA256)

    def test_model_runner_output_hash_is_the_tracked_fixture(self) -> None:
        self.assertEqual(
            sha256_file(MODEL_RUNNER_FIXTURE), producer.MODEL_RUNNER_OUTPUT_SHA256
        )

    def test_the_patch_matches_its_pinned_hash(self) -> None:
        self.assertEqual(
            sha256_file(producer.ROUTE_CAPTURE_PATCH),
            producer.ROUTE_CAPTURE_PATCH_SHA256,
        )

    def test_the_overlays_accept_what_this_produces(self) -> None:
        """The generators' declared inputs are this producer's outputs."""
        from scripts.glm35_q40 import (
            q40_exact_state_attestation_overlay as attestation,
            q40_exact_state_overlay as exl3_overlay,
        )

        self.assertEqual(exl3_overlay.INPUT_SHA256, producer.EXL3_OUTPUT_SHA256)
        self.assertEqual(
            attestation.INPUT_SHA256, producer.MODEL_RUNNER_OUTPUT_SHA256
        )


class ScratchReservationTest(unittest.TestCase):
    def _tree(self, root: Path, payload: bytes) -> Path:
        path = root / producer.EXL3_RELATIVE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return root

    def test_refuses_an_unexpected_input(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            tree = self._tree(Path(raw), b"unrelated source\n")
            with self.assertRaises(producer.OverlayInputError) as caught:
                producer.apply_scratch_reservation(tree)
            self.assertIn("expected", str(caught.exception))

    def test_reports_an_already_applied_tree_without_rewriting(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            tree = self._tree(Path(raw), EXL3_FIXTURE.read_bytes())
            self.assertEqual(
                producer.apply_scratch_reservation(tree), "already applied"
            )
            self.assertEqual(
                sha256_file(tree / producer.EXL3_RELATIVE),
                producer.EXL3_OUTPUT_SHA256,
            )


class RouteCaptureTest(unittest.TestCase):
    def test_refuses_an_unexpected_model_runner(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            tree = Path(raw)
            path = tree / producer.MODEL_RUNNER_RELATIVE
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"unrelated source\n")
            with self.assertRaises(producer.OverlayInputError) as caught:
                producer.apply_route_capture(tree)
            self.assertIn("expected", str(caught.exception))

    def test_reports_an_already_applied_tree(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            tree = Path(raw)
            path = tree / producer.MODEL_RUNNER_RELATIVE
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(MODEL_RUNNER_FIXTURE, path)
            self.assertEqual(producer.apply_route_capture(tree), "already applied")


class CommandLineTest(unittest.TestCase):
    def test_reports_a_tree_without_the_expected_layout(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            self.assertEqual(producer.main([raw]), 1)


if __name__ == "__main__":
    unittest.main()
