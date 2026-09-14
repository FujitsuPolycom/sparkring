"""Fail-closed source and compiler-cache admission for Qwen prefill bundles."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import package_prefill
import prefill_bootstrap


class BundleTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        image = root / "image"
        sources = {}
        for name in prefill_bootstrap.IMAGE_SOURCES:
            path = image / name.lstrip("/")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with patch.dict(prefill_bootstrap.IMAGE_SOURCES, sources, clear=True):
            digest = package_prefill.package(root / "bundle")
        return root / "bundle", image, sources, digest

    def test_valid_bundle_and_isolated_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root, image, sources, digest = self.fixture(directory)
            with patch.dict(prefill_bootstrap.IMAGE_SOURCES, sources, clear=True):
                prefill_bootstrap.verify(root, digest, image)
            namespace = "/cache/qwen-prefill-" + digest[:12]
            prefill_bootstrap.verify_cache_namespace(
                digest,
                {
                    "VLLM_CACHE_ROOT": namespace + "/vllm",
                    "TORCHINDUCTOR_CACHE_DIR": namespace + "/inductor",
                },
            )

    def test_changed_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, image, sources, digest = self.fixture(directory)
            (root / "qwen_hc_fusion.py").write_text("changed")
            with patch.dict(prefill_bootstrap.IMAGE_SOURCES, sources, clear=True):
                with self.assertRaisesRegex(ValueError, "bundle file mismatch"):
                    prefill_bootstrap.verify(root, digest, image)

    def test_image_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, image, sources, digest = self.fixture(directory)
            (image / next(iter(sources)).lstrip("/")).write_text("changed")
            with patch.dict(prefill_bootstrap.IMAGE_SOURCES, sources, clear=True):
                with self.assertRaisesRegex(ValueError, "image source mismatch"):
                    prefill_bootstrap.verify(root, digest, image)

    def test_baseline_compiler_cache_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "source-bound"):
            prefill_bootstrap.verify_cache_namespace(
                "a" * 64,
                {
                    "VLLM_CACHE_ROOT": "/cache/baseline/vllm",
                    "TORCHINDUCTOR_CACHE_DIR": "/cache/baseline/inductor",
                },
            )

    def test_pth_failure_stops_the_consumer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "admission.pth").write_text(
                str(Path(__file__).parent) + "\n"
                "import prefill_bootstrap; prefill_bootstrap.install()\n"
            )
            code = "import site,sys; site.addsitedir(sys.argv[1]); print('CONSUMER_REACHED')"
            result = subprocess.run(
                [sys.executable, "-c", code, directory],
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("CONSUMER_REACHED", result.stdout)
        self.assertIn("Qwen prefill admission failed", result.stderr)

    def test_existing_package_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                package_prefill.package(Path(directory))


if __name__ == "__main__":
    unittest.main()
