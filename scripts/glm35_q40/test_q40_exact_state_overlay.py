# ruff: noqa: E402

import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import glm35_profile as compiler  # noqa: E402
from scripts.test_glm35_profile import (  # noqa: E402
    sircl_artifact_digests,
    source_ckv_profile,
)

from scripts.glm35_q40.q40_exact_state_overlay import (
    INPUT_SHA256,
    OUTPUT_SHA256,
    ExactQ40StateOverlayError,
    install,
    sha256_bytes,
    transform,
)  # noqa: E402


EXACT_EXL3 = (
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
PATCH = Path(__file__).with_name("q40_exact_state.patch")


def base_profile() -> dict:
    return compiler._derive_sircl_tiered(
        source_ckv_profile(), artifact_digests=sircl_artifact_digests()
    )


class Q40ExactStateOverlayTest(unittest.TestCase):
    @staticmethod
    def _transformed_dispatch():
        tree = ast.parse(transform(EXACT_EXL3.read_bytes()).decode("utf-8"))
        method = next(
            member
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Exl3MoEMethod"
            for member in node.body
            if isinstance(member, ast.FunctionDef)
            and member.name == "_apply_mixed_rank_sliced"
        )
        namespace = {
            "_MIXED_TRELLIS_TARGET_Q40_ROWS": 40,
            "torch": torch,
        }
        exec(
            "from __future__ import annotations\n" + ast.unparse(method),
            namespace,
        )
        return namespace["_apply_mixed_rank_sliced"]

    def test_exact_source_adds_only_the_target_q40_state(self) -> None:
        source = EXACT_EXL3.read_bytes()
        self.assertEqual(sha256_bytes(source), INPUT_SHA256)

        output = transform(source)
        self.assertEqual(sha256_bytes(output), OUTPUT_SHA256)
        compile(output.decode("utf-8"), "exl3.py", "exec")

        source_lines = source.decode("utf-8").splitlines()
        source_index = 0
        added = []
        for line in output.decode("utf-8").splitlines():
            if source_index < len(source_lines) and line == source_lines[source_index]:
                source_index += 1
            else:
                added.append(line)
        self.assertEqual(source_index, len(source_lines))
        self.assertEqual(
            added,
            [
                "_MIXED_TRELLIS_TARGET_Q40_ROWS = 40",
                "        q40 = None",
                "        if (",
                "            not owner_token[1]",
                "            and max_decode_m == 32",
                "            and _MIXED_TRELLIS_TARGET_Q40_ROWS",
                "            <= min(max_batched_tokens, prefill_capacity)",
                "        ):",
                "            q40 = make_state(",
                "                _MIXED_TRELLIS_TARGET_Q40_ROWS,",
                "                _MIXED_TRELLIS_ROUTE_BLOCK_SIZE,",
                '                mixed["prefill_tile_config"],',
                "            )",
                '            "q40": q40,',
                "        if (",
                "            m == _MIXED_TRELLIS_TARGET_Q40_ROWS",
                '            and runtime["q40"] is not None',
                "        ):",
                "            return run_state(",
                "                x,",
                "                topk_weights,",
                "                topk_ids,",
                '                runtime["q40"],',
                '                mixed["prefill_tiers"],',
                "            )",
                "",
            ],
        )

    def test_q40_dispatch_is_between_decode_and_general_prefill(self) -> None:
        text = transform(EXACT_EXL3.read_bytes()).decode("utf-8")
        decode = text.index('if m <= runtime["max_decode_m"]:')
        q40 = text.index("m == _MIXED_TRELLIS_TARGET_Q40_ROWS")
        prefill = text.index('if runtime["prefill"] is None:', q40)
        self.assertLess(decode, q40)
        self.assertLess(q40, prefill)
        self.assertIn(
            'runtime["q40"],\n                mixed["prefill_tiers"],',
            text[q40:prefill],
        )

    def test_dispatch_changes_only_the_exact_target_q40_shape(self) -> None:
        def run_mixed(*args, **_kwargs):
            return torch.full_like(args[0], float(args[-2]))

        runtime = {
            "mixed_api": SimpleNamespace(
                run_mixed_trellis=run_mixed,
                run_mixed_trellis3=None,
            ),
            "decode": {"launch": 1, "buffers": object()},
            "q40": {"launch": 2, "buffers": object()},
            "prefill": {"launch": 3, "buffers": object()},
            "max_decode_m": 32,
            "max_batched_tokens": 4096,
            "prefill_capacity": 4096,
        }
        layer = SimpleNamespace(
            exl3_mixed_trellis={
                "tiers": (object(), object()),
                "prefill_tiers": (object(), object()),
                "global_to_combined": object(),
                "descriptor_map": object(),
                "rotations": object(),
            }
        )
        owner = SimpleNamespace(
            _mixed_rank_sliced_runtime=lambda _layer, _x, _ids: runtime
        )
        dispatch = self._transformed_dispatch()

        for rows, expected in (
            (1, 1),
            (20, 1),
            (32, 1),
            (33, 3),
            (39, 3),
            (40, 2),
            (41, 3),
            (512, 3),
        ):
            x = torch.zeros((rows, 1))
            result = dispatch(
                owner,
                layer,
                x,
                torch.zeros((rows, 1)),
                torch.zeros((rows, 1), dtype=torch.int64),
            )
            self.assertEqual(result[0, 0].item(), expected, rows)

        runtime["q40"] = None
        x = torch.zeros((40, 1))
        result = dispatch(
            owner,
            layer,
            x,
            torch.zeros((40, 1)),
            torch.zeros((40, 1), dtype=torch.int64),
        )
        self.assertEqual(result[0, 0].item(), 3)

    def test_baseline_already_captures_q40_without_a_prefill_block_override(self) -> None:
        profile = base_profile()
        environment = profile["environment"]
        self.assertEqual(environment["VLLM_EXL3_PREFILL_CAPACITY"], "4096")
        self.assertNotIn("VLLM_EXL3_PREFILL_BLOCK_M", environment)
        self.assertNotIn("VLLM_EXL3_TRELLIS_MAX_M", environment)

        arguments = profile["extra_vllm_args"]
        compilation = json.loads(
            arguments[arguments.index("--compilation-config") + 1]
        )
        self.assertEqual(compilation["cudagraph_capture_sizes"], list(range(1, 41)))
        self.assertEqual(
            arguments[arguments.index("--max-cudagraph-capture-size") + 1], "40"
        )
        self.assertEqual(
            arguments[arguments.index("--max-num-batched-tokens") + 1], "4096"
        )

    def test_source_hash_drift_fails_closed(self) -> None:
        with self.assertRaisesRegex(ExactQ40StateOverlayError, "input hash mismatch"):
            transform(EXACT_EXL3.read_bytes() + b"\n")

    def test_patch_artifact_produces_the_same_pinned_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = (
                root
                / "vllm"
                / "model_executor"
                / "layers"
                / "quantization"
                / "exl3.py"
            )
            target.parent.mkdir(parents=True)
            target.write_bytes(EXACT_EXL3.read_bytes())
            subprocess.run(
                [
                    "git",
                    "-c",
                    "core.autocrlf=false",
                    "apply",
                    "--whitespace=error-all",
                    str(PATCH.resolve()),
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(sha256_bytes(target.read_bytes()), OUTPUT_SHA256)

    def test_install_refuses_to_replace_an_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "exl3.py"
            receipt = install(EXACT_EXL3, output)
            self.assertEqual(receipt["input_sha256"], INPUT_SHA256)
            self.assertEqual(receipt["output_sha256"], OUTPUT_SHA256)
            with self.assertRaisesRegex(ExactQ40StateOverlayError, "refusing to overwrite"):
                install(EXACT_EXL3, output)


if __name__ == "__main__":
    unittest.main()
