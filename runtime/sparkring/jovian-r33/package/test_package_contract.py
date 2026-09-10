from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
VERIFY_PATH = HERE / "verify_vllm_wheel.py"
SPEC = importlib.util.spec_from_file_location("verify_vllm_wheel", VERIFY_PATH)
assert SPEC and SPEC.loader
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class VllmPackageContractTests(unittest.TestCase):
    def test_external_flash_attention_python_mapping_is_commit_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            subprocess.run(["git", "init", "-q", str(repository)], check=True)
            files = {
                "vllm_flash_attn/__init__.py": "facade\n",
                "vllm_flash_attn/flash_attn_interface.py": "facade\n",
                "vllm_flash_attn/layers/__init__.py": "\n",
                "vllm_flash_attn/layers/rotary.py": "VALUE = 1\n",
                "vllm_flash_attn/ops/data.bin": "excluded\n",
                "vllm_flash_attn/ops/triton/rotary.py": "VALUE = 2\n",
            }
            for name, content in files.items():
                path = repository / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "-c",
                    "user.name=SparkRing",
                    "-c",
                    "user.email=sparkring@example.invalid",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                check=True,
            )
            commit = subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
            ).strip()

            mapped = VERIFY.tracked_flash_attn_python(repository, commit)

            self.assertEqual(
                set(mapped),
                {
                    "vllm/vllm_flash_attn/layers/__init__.py",
                    "vllm/vllm_flash_attn/layers/rotary.py",
                    "vllm/vllm_flash_attn/ops/triton/rotary.py",
                },
            )
            self.assertEqual(
                mapped["vllm/vllm_flash_attn/layers/rotary.py"][1],
                b"VALUE = 1\n",
            )

    def test_packaging_and_smoke_scripts_cover_rotary_helpers(self) -> None:
        package = (HERE / "package_vllm.sh").read_text()
        smoke = (HERE / "smoke_vllm_wheel.sh").read_text()
        self.assertIn("expected_flash_attn_commit=f3e1a4f", package)
        self.assertIn("archive \"$expected_flash_attn_commit\"", package)
        self.assertIn("vllm_flash_attn.layers.rotary", smoke)
        self.assertIn("vllm_flash_attn.ops.triton.rotary", smoke)


if __name__ == "__main__":
    unittest.main()
