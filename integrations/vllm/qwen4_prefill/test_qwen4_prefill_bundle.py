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
    def test_audit_hook_is_between_engine_initialization_and_http_serving(self):
        from multimodal_hc_routing import apply_api_audit

        source = """async def build_and_serve():
    await init_app_state(engine_client, app.state, args, supported_tasks)
    return await serve_http(app)
"""
        patched = apply_api_audit(source)
        self.assertLess(
            patched.index("await init_app_state"), patched.index("    emit(")
        )
        self.assertLess(patched.index("    emit("), patched.index("await serve_http"))
        with self.assertRaises(ValueError):
            apply_api_audit(patched)
        with self.assertRaises(ValueError):
            apply_api_audit(source + source)

    def test_startup_audit_distinguishes_flags_from_multimodal_activation(self):
        from startup_audit import inspect_settings

        source = """class Qwen4ExpForConditionalGeneration:
    def forward(self):
        return self.language_model.model()
"""
        environment = {
            "VLLM_QWEN3_8_HC_PREFILL_MODE": "shard",
            "VLLM_QWEN3_8_PREFILL_COALESCE": "1",
        }
        unchanged = dict(environment)
        rows = inspect_settings(environment, tp=4, batch=8192, source=source)
        self.assertIn("HC_ROUTE_BYPASS", [code for _, code, _ in rows])
        self.assertNotIn("HC_ROUTE_VERIFIED", [code for _, code, _ in rows])
        rows = inspect_settings(
            environment,
            tp=4,
            batch=512,
            source=source.replace("language_model.model", "language_model"),
        )
        self.assertIn("HC_ROUTE_VERIFIED", [code for _, code, _ in rows])
        self.assertIn("HC_BATCH_TOO_SMALL", [code for _, code, _ in rows])
        self.assertTrue(
            any("execution is unverified" in message for _, _, message in rows)
        )
        self.assertEqual(environment, unchanged)
        rows = inspect_settings({}, tp=2, batch=8192, source=source)
        self.assertNotIn("HC_DISABLED", [code for _, code, _ in rows])
        rows = inspect_settings({}, tp=4, batch=8192, source=source)
        self.assertIn("HC_DISABLED", [code for _, code, _ in rows])

    def test_multimodal_routing_patch_preserves_all_call_arguments(self):
        from multimodal_hc_routing import apply

        source = """class Qwen4ExpForConditionalGeneration:
    def forward(self, **kwargs):
        return self.language_model.model(
            input_ids=ids, positions=positions, intermediate_tensors=intermediate,
            inputs_embeds=embeds, query_start_loc=query, ngram_context=ngram,
            deepstack_input_embeds=deepstack,
        )
"""
        repaired = apply(source)
        self.assertEqual(
            repaired,
            source.replace("self.language_model.model(", "self.language_model("),
        )
        with self.assertRaises(ValueError):
            apply(repaired)
        with self.assertRaises(ValueError):
            apply(source.replace("ngram_context=ngram,", ""))
        with self.assertRaises(ValueError):
            apply(source + source)

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
            tp_size=1,
            use_combine=False,
            input_mix_weight_down=lambda x: x,
            input_mix_weight_up=SimpleNamespace(weight="weight"),
            _binding=lambda normalized, operation: (
                binding if operation == "scaled_silu" else None
            ),
        )
        normalized = SimpleNamespace(shape=(128, 10240))
        output, injection = namespace["mix_normalized"](layer, normalized)
        self.assertEqual(output, (normalized, "weight", normalized))
        self.assertIsNone(injection)
        self.assertEqual(calls, [{"binding": binding}])


    def test_small_hc_batches_keep_native_projection_dispatch(self):
        source = Path(__file__).with_name("qwen4_hc_fusion.py")
        patch_function = next(
            node for node in ast.parse(source.read_text()).body
            if getattr(node, "name", None) == "patch"
        )
        function = next(
            node for node in patch_function.body
            if getattr(node, "name", None) == "mix_normalized"
        )
        calls = []
        native = object()
        namespace = {
            "module": SimpleNamespace(_hyperconnection_api=lambda: SimpleNamespace(
                run_scaled_silu=lambda value, **kwargs: value)),
            "operation": lambda *args: "prefill-hook",
            "original_mix_normalized": lambda owner, value: (
                calls.append((owner, value)) or (native, None)
            ),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        layer = SimpleNamespace(
            tp_size=4, use_combine=False,
            input_mix_weight_down=lambda value: value,
            input_mix_weight_up=SimpleNamespace(weight="weight"),
            _binding=lambda *_: None,
        )
        for rows in (0, 1, 16, 64, 127):
            with self.subTest(rows=rows):
                normalized = SimpleNamespace(shape=(rows, 10240))
                self.assertEqual(namespace["mix_normalized"](layer, normalized), (native, None))
                self.assertEqual(calls[-1], (layer, normalized))

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

    def test_external_python_layout_is_bound_in_packaged_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image"
            bindings = {}
            for old in prefill_bootstrap.IMAGE_SOURCES:
                name = old.replace(
                    "/opt/venv/lib/python3.12/site-packages/",
                    "/usr/local/lib/python3.12/dist-packages/",
                )
                path = image / name.lstrip("/")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
                bindings[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            digest = package_prefill.package(root / "bundle", bindings)
            spec = importlib.util.spec_from_file_location(
                "external_prefill_bootstrap", root / "bundle/qwen4_prefill_bootstrap.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.verify(root / "bundle", digest, image)
            first = image / next(iter(bindings)).lstrip("/")
            first.write_bytes(b"unexpected source")
            with self.assertRaisesRegex(ValueError, "image source mismatch"):
                module.verify(root / "bundle", digest, image)

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
