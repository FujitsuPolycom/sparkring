# ruff: noqa: E402

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from scripts.glm35_q40 import (
    prepare_q40_exact_state_serving as prepare_module,
)  # noqa: E402
from scripts.glm35_q40.q40_exact_state_overlay import (
    transform as transform_exl3,
)  # noqa: E402
from scripts.sparkring_runtime import expand  # noqa: E402
import glm35_profile as compiler  # noqa: E402
from scripts.test_glm35_profile import (  # noqa: E402
    sircl_artifact_digests,
    source_ckv_profile,
)


BASE_EXL3 = (
    REPOSITORY_ROOT
    / "runtime"
    / "exl3-r7"
    / "test-fixtures"
    / "vllm"
    / "model_executor"
    / "layers"
    / "quantization"
    / "exl3.py.fixture"
)


def base_profile() -> dict:
    return compiler._derive_sircl_tiered(
        source_ckv_profile(), artifact_digests=sircl_artifact_digests()
    )


class PrepareExactQ40ServingTest(unittest.TestCase):
    def _prepare(self, root: Path):
        base_profile_path = root / "base-profile.json"
        base_profile_path.write_text(
            json.dumps(base_profile(), indent=2) + "\n", encoding="utf-8"
        )
        base_profile_sha256 = hashlib.sha256(
            base_profile_path.read_bytes()
        ).hexdigest()
        exl3 = root / "source" / "exl3.py"
        exl3.parent.mkdir()
        exl3.write_bytes(transform_exl3(BASE_EXL3.read_bytes()))
        runner = root / "source" / "model_runner.py"
        runner.write_text("# exact q40 attestation fixture\n", encoding="utf-8")
        runner_hash = hashlib.sha256(runner.read_bytes()).hexdigest()
        return prepare_module.prepare(
            base_profile_path=base_profile_path,
            expected_base_profile_sha256=base_profile_sha256,
            exl3_path=exl3,
            model_runner_path=runner,
            expected_model_runner_sha256=runner_hash,
            bundle_path=root / "bundle",
        )

    def test_profile_changes_only_the_narrow_q40_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate, manifest = self._prepare(Path(directory))
        base = base_profile()

        self.assertNotIn("VLLM_EXL3_PREFILL_BLOCK_M", candidate["environment"])
        self.assertNotIn("VLLM_EXL3_TRELLIS_MAX_M", candidate["environment"])
        self.assertEqual(candidate["environment"]["VLLM_EXL3_PREFILL_CAPACITY"], "4096")
        self.assertEqual(
            candidate["environment"]["VLLM_CACHE_ROOT"],
            prepare_module.VLLM_CACHE_ROOT,
        )
        self.assertEqual(
            [
                volume["host"]
                for volume in candidate["extra_volumes"]
                if volume["container"] == prepare_module.EXL3_CONTAINER
            ],
            [f"{prepare_module.REMOTE_ROOT}/exl3.py"],
        )
        self.assertEqual(
            [
                volume["host"]
                for volume in candidate["extra_volumes"]
                if volume["container"] == prepare_module.MODEL_RUNNER_CONTAINER
            ],
            [f"{prepare_module.REMOTE_ROOT}/model_runner.py"],
        )
        self.assertEqual(
            len(candidate["extra_volumes"]), len(base["extra_volumes"]) + 2
        )
        self.assertIn(prepare_module.EXACT_Q40_EXL3_SHA256, candidate["attestation_hook"][2])
        self.assertNotIn(prepare_module.BASE_EXL3_SHA256, candidate["attestation_hook"][2])
        self.assertEqual(manifest["scope"], "target-mixed-exact-40-rows")
        receipt_template = candidate["environment"][
            "SPARK_Q40_EXACT_STATE_ATTEST_PATH"
        ]
        self.assertEqual(
            receipt_template,
            "/cache/jit/q40-exact-state-serving-v1-rank{rank}.json",
        )
        for rank in range(4):
            self.assertEqual(
                expand(receipt_template, {"rank": str(rank)}),
                f"/cache/jit/q40-exact-state-serving-v1-rank{rank}.json",
            )

    def test_unsealed_model_runner_fails_before_writing_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                prepare_module.PrepareExactQ40ServingError,
                "runtime attestation overlay is not sealed",
            ):
                prepare_module.prepare(
                    base_profile_path=root / "absent-profile.json",
                    expected_base_profile_sha256="0" * 64,
                    exl3_path=root / "absent-exl3.py",
                    model_runner_path=root / "absent-runner.py",
                    expected_model_runner_sha256="",
                    bundle_path=root / "bundle",
                )
            self.assertFalse((root / "bundle").exists())

    def test_profile_diff_auditor_rejects_an_unrelated_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate, _manifest = self._prepare(Path(directory))
        base = base_profile()
        drifted = copy.deepcopy(candidate)
        drifted["environment"]["NCCL_MAX_NCHANNELS"] = "99"
        with self.assertRaisesRegex(
            prepare_module.PrepareExactQ40ServingError,
            "outside the exact-Q40 allowlist",
        ):
            prepare_module._assert_only_allowed_changes(base, drifted)


if __name__ == "__main__":
    unittest.main()
