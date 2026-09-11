from __future__ import annotations

import importlib.util
import hashlib
import json
import re
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

    def test_source_composition_and_package_tree_are_identical(self) -> None:
        root = HERE.parent
        manifest_path = root / "patches/vllm-r33-sparkring.manifest.json"
        manifest = json.loads(manifest_path.read_text())
        patch = root / manifest["patch"]["path"]
        patch_sha = hashlib.sha256(patch.read_bytes()).hexdigest()
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        self.assertEqual(
            patch_sha,
            manifest["patch"]["sha256"],
        )
        self.assertEqual(patch.stat().st_size, manifest["patch"]["size"])
        self.assertEqual(
            manifest["result"]["tree"],
            "667ee2f6652efa065c57a7adc0193991f6cde6ac",
        )
        self.assertEqual(
            manifest["continuation_port"]["commit"],
            "b611611a643502542c2d900057eb47e407b8379e",
        )
        self.assertEqual(
            manifest["prefix_hit_metadata_compatibility"]["commit"],
            "58c087102cd3039245240e34c750c7f77ce07ed7",
        )
        dependency = manifest["required_b12x_checkpoint_contract"]
        self.assertEqual(dependency["commit"], "68acfc14893c087aa9b3120bb984fde4c4e7a21f")
        self.assertEqual(dependency["max_checkpoints"], 4)
        package = (HERE / "package_vllm.sh").read_text()
        prepare = (root / "prepare_vllm_source.sh").read_text()

        def assignment(source: str, name: str) -> str:
            match = re.search(rf"(?m)^{re.escape(name)}=([^\s]+)$", source)
            self.assertIsNotNone(match, name)
            return match.group(1)

        self.assertEqual(assignment(prepare, "patch_sha"), patch_sha)
        self.assertEqual(assignment(prepare, "manifest_sha"), manifest_sha)
        self.assertEqual(
            assignment(prepare, "expected_tree"), manifest["result"]["tree"]
        )
        self.assertIn(
            f")\" = {patch_sha}",
            package,
        )
        artifact_lock = json.loads((root / "image/artifact-lock.json").read_text())
        identities = artifact_lock["source_identities"]
        self.assertEqual(identities["vllm_cached_diff_sha256"], patch_sha)
        self.assertEqual(identities["vllm_source_manifest_sha256"], manifest_sha)
        self.assertIn(f"expected_package_tree={manifest['result']['tree']}", package)
        self.assertIn(
            f"expected_native_tree={manifest['native_reuse']['source_tree']}", package
        )
        self.assertIn(
            f"expected_native_inputs_sha={manifest['native_reuse']['git_file_records_sha256']}",
            package,
        )


if __name__ == "__main__":
    unittest.main()
