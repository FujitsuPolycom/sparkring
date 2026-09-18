"""Fail-closed source and compiler-cache admission for Qwen prefill bundles."""

import hashlib
import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import importlib.util
import qwen4_prefill_bootstrap as prefill_bootstrap

_spec = importlib.util.spec_from_file_location(
    "qwen4_package", Path(__file__).with_name("package_prefill.py")
)
package_prefill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(package_prefill)


class BundleTests(unittest.TestCase):
    def test_mtp_small_rows_preserve_prepared_projection_callables(self):
        source = Path(__file__).with_name("qwen4_mtp_gemm.py")
        tree = ast.parse(source.read_text())
        patch_function = next(
            n for n in tree.body if getattr(n, "name", None) == "patch"
        )
        function = next(
            n for n in patch_function.body if getattr(n, "name", None) == "projections"
        )
        observed = {}
        namespace = {"original": lambda *args, **kwargs: observed.update(kwargs)}
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        prepared = (object(), object())
        namespace["projections"](
            None,
            None,
            None,
            None,
            None,
            None,
            projections=prepared,
            tokens=127,
            token_rows=128,
            state_rows=512,
            streams=4,
            hidden_size=2560,
        )
        self.assertIs(observed["projections"], prepared)
        self.assertEqual(observed["tokens"], 127)

    def test_hc_uses_the_scaled_silu_preparation_plan(self):
        source = Path(__file__).with_name("qwen4_hc_fusion.py")
        tree = ast.parse(source.read_text())
        patch_function = next(
            n for n in tree.body if getattr(n, "name", None) == "patch"
        )
        function = next(
            n
            for n in patch_function.body
            if getattr(n, "name", None) == "mix_normalized"
        )
        calls = []
        binding = object()
        module = SimpleNamespace(
            _hyperconnection_api=lambda: SimpleNamespace(
                run_scaled_silu=lambda projected, **kwargs: (
                    calls.append(kwargs) or projected
                )
            )
        )
        namespace = {"module": module, "operation": lambda x, w, n: (x, w, n)}
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        layer = SimpleNamespace(
            use_combine=False,
            input_mix_weight_down=lambda x: x,
            input_mix_weight_up=SimpleNamespace(weight="weight"),
            _binding=lambda normalized, operation: (
                binding if operation == "scaled_silu" else None
            ),
        )
        output, injection = namespace["mix_normalized"](layer, "normalized")
        self.assertEqual(output, ("normalized", "weight", "normalized"))
        self.assertIsNone(injection)
        self.assertEqual(calls, [{"binding": binding}])

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
            namespace = "/cache/qwen4-prefill-" + digest[:12]
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
            (root / "qwen4_hc_fusion.py").write_text("changed")
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
                "import qwen4_prefill_bootstrap; qwen4_prefill_bootstrap.install()\n"
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
