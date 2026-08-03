#!/usr/bin/env python3
"""CPU-only tests for connector_bundle_manifest.py."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from connector_bundle_manifest import (  # noqa: E402
    BUNDLE_DOMAIN_SEPARATOR,
    BundleManifestError,
    REQUIRED_FILES,
    compute_bundle_identity,
    generate_receipt,
    inventory_staging_root,
)


class BundleManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.root = Path(self.tmpdir) / "staging"
        self.root.mkdir()
        # Create required files.
        for rel in REQUIRED_FILES:
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"content of {rel}\n", encoding="utf-8")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_inventory_succeeds_with_required_files(self):
        entries = inventory_staging_root(self.root)
        self.assertEqual(len(entries), len(REQUIRED_FILES))
        for entry in entries:
            self.assertIn(entry.rel_path, REQUIRED_FILES)
            self.assertGreater(entry.byte_size, 0)
            self.assertEqual(len(entry.content_sha256), 64)

    def test_inventory_rejects_missing_file(self):
        (self.root / "sparkcache" / "spark_context_cache_connector.py").unlink()
        with self.assertRaises(BundleManifestError) as ctx:
            inventory_staging_root(self.root)
        self.assertTrue(
            "missing" in str(ctx.exception).lower() or
            "cannot lstat" in str(ctx.exception).lower(),
            f"Unexpected error: {ctx.exception}",
        )

    def test_inventory_rejects_extra_file(self):
        (self.root / "extra_file.py").write_text("extra", encoding="utf-8")
        with self.assertRaises(BundleManifestError) as ctx:
            inventory_staging_root(self.root)
        self.assertIn("extra file", str(ctx.exception).lower())

    @unittest.skipIf(os.name == "nt", "Cannot create symlinks on Windows without admin")
    def test_inventory_rejects_symlink_file(self):
        link = self.root / "sparkcache" / "spark_context_cache_connector.py"
        link.unlink()
        os.symlink(self.root / "sparkcache" / "spark_context_cache_codec.py", link)
        with self.assertRaises(BundleManifestError) as ctx:
            inventory_staging_root(self.root)
        self.assertIn("symlink", str(ctx.exception).lower())

    @unittest.skipIf(os.name == "nt", "Cannot create symlinks on Windows without admin")
    def test_inventory_rejects_symlink_root(self):
        link_dir = Path(self.tmpdir) / "symlink_root"
        os.symlink(self.root, link_dir)
        with self.assertRaises(BundleManifestError):
            inventory_staging_root(link_dir)

    def test_identity_is_deterministic(self):
        id1 = compute_bundle_identity(inventory_staging_root(self.root))
        id2 = compute_bundle_identity(inventory_staging_root(self.root))
        self.assertEqual(id1, id2)
        self.assertEqual(len(id1), 64)

    def test_identity_changes_on_content_change(self):
        id1 = compute_bundle_identity(inventory_staging_root(self.root))
        p = self.root / "sparkcache" / "spark_context_cache_connector.py"
        p.write_text("different content\n", encoding="utf-8")
        id2 = compute_bundle_identity(inventory_staging_root(self.root))
        self.assertNotEqual(id1, id2)

    def test_receipt_has_required_fields(self):
        receipt = generate_receipt(self.root)
        self.assertIn("connector_bundle_identity_sha256", receipt)
        self.assertIn("file_count", receipt)
        self.assertIn("total_bytes", receipt)
        self.assertIn("files", receipt)
        self.assertEqual(receipt["file_count"], len(REQUIRED_FILES))
        self.assertEqual(len(receipt["files"]), len(REQUIRED_FILES))
        self.assertEqual(len(receipt["connector_bundle_identity_sha256"]), 64)

    def test_identity_matches_manual_computation(self):
        """Identity matches manual domain-separated SHA-256."""
        entries = inventory_staging_root(self.root)
        h = hashlib.sha256()
        h.update(BUNDLE_DOMAIN_SEPARATOR.encode("utf-8"))
        for entry in entries:
            h.update(b"\x00")
            h.update(entry.rel_path.encode("utf-8"))
            h.update(b"\x00")
            h.update(str(entry.byte_size).encode("utf-8"))
            h.update(b"\x00")
            h.update(entry.content_sha256.encode("utf-8"))
        expected = h.hexdigest()
        actual = compute_bundle_identity(entries)
        self.assertEqual(expected, actual)

    def test_main_outputs_json(self):
        from connector_bundle_manifest import main
        rc = main(["--staging-root", str(self.root)])
        self.assertEqual(rc, 0)

    def test_main_returns_error_on_missing_file(self):
        from connector_bundle_manifest import main
        (self.root / "sparkcache" / "spark_context_cache_connector.py").unlink()
        rc = main(["--staging-root", str(self.root)])
        self.assertEqual(rc, 1)

    def test_inventory_rejects_changed_during_hash(self):
        """If a file's stat metadata changes between lstat-before and
        lstat-after the read, inventory must raise BundleManifestError.
        """
        import connector_bundle_manifest as cbm
        import time
        target_file = self.root / "sparkcache" / "spark_context_cache_connector.py"
        def mutating_hasher(path, chunk_size, st_before=None):
            # When hashing the target file, write a different size
            # BEFORE reading so the st_after comparison detects it.
            if st_before is not None and path == target_file:
                target_file.write_bytes(b"X" * 5000)
                time.sleep(0.01)
            return cbm._hash_file(path, chunk_size=chunk_size, st_before=st_before)

        with self.assertRaises(BundleManifestError) as ctx:
            inventory_staging_root(self.root, _hasher=mutating_hasher)
        self.assertIn("cannot read file", str(ctx.exception).lower())

    @unittest.skipIf(not hasattr(os, "mkfifo"), "FIFO not available on this platform")
    def test_inventory_rejects_nonregular_file(self):
        """If a required path is a FIFO (non-regular, non-symlink),
        inventory must raise BundleManifestError.
        """
        target = self.root / "sparkcache" / "spark_context_cache_connector.py"
        target.unlink()
        os.mkfifo(target)
        with self.assertRaises(BundleManifestError) as ctx:
            inventory_staging_root(self.root)
        self.assertIn("not regular", str(ctx.exception).lower())

# ---------------------------------------------------------------------------
# Import-closure test: build a real bundle from repo sources and verify
# that the connector's complete local import closure is present.
# ---------------------------------------------------------------------------


class ImportClosureTests(unittest.TestCase):
    """Build a real staging directory from this checkout and prove the
    bundle contains the connector's complete local import closure.

    This is the decisive test that a four-file dummy bundle cannot pass.
    """

    def setUp(self):
        import shutil
        self.tmpdir = tempfile.mkdtemp()
        self.staging = Path(self.tmpdir) / "staging"
        self.staging.mkdir()
        # Repo root is two levels above scripts/.
        self.repo_root = Path(__file__).resolve().parents[1]
        # Copy each required file from the repo into the staging dir.
        # REQUIRED_FILES paths are relative to the staging root and all
        # start with "sparkcache/".  In the repo, the same files exist
        # at the same paths (e.g. "sparkcache/streaming/__init__.py"),
        # so the staging-relative path equals the repo-relative path.
        for rel in REQUIRED_FILES:
            src = self.repo_root / rel
            dst = self.staging / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_bundle_inventory_succeeds_on_real_sources(self):
        """The real repo sources must pass inventory without error."""
        entries = inventory_staging_root(self.staging)
        self.assertEqual(len(entries), len(REQUIRED_FILES))

    def test_bundle_identity_is_deterministic_on_real_sources(self):
        """Two inventories of the real bundle produce the same identity."""
        id1 = compute_bundle_identity(inventory_staging_root(self.staging))
        id2 = compute_bundle_identity(inventory_staging_root(self.staging))
        self.assertEqual(id1, id2)
        self.assertEqual(len(id1), 64)

    def test_bundle_receipt_generates_on_real_sources(self):
        """The real bundle generates a valid receipt."""
        receipt = generate_receipt(self.staging)
        self.assertEqual(len(receipt["connector_bundle_identity_sha256"]), 64)

    def test_required_files_cover_eager_import_closure(self):
        """The REQUIRED_FILES allowlist must cover every local .py file
        imported when spark_context_cache_connector is imported.

        We verify by checking that every non-test .py file in the
        sparkcache/ tree that is NOT in REQUIRED_FILES is either:
        - a test file (starts with test_), or
        - in the native/ subdirectory (CUDA bindings, not Python import deps)
        - in the replication/ subdirectory (not imported by connector)
        """
        sparkcache_dir = self.repo_root / "sparkcache"
        all_py_files = []
        for p in sparkcache_dir.rglob("*.py"):
            repo_rel = p.relative_to(self.repo_root).as_posix()
            # In the staging layout, the repo's sparkcache/ directory is
            # copied to <staging-root>/sparkcache/, so the staging-relative
            # path is the same as the repo-relative path.
            all_py_files.append(repo_rel)
        # Every non-test, non-native, non-replication, non-experiment .py
        # must be in REQUIRED_FILES.  Native placement/restore modules
        # are lazy-loaded only when native restore is enabled (off by
        # default).  Experiments are not in the connector import path.
        missing = []
        for rel in sorted(all_py_files):
            if rel in REQUIRED_FILES:
                continue
            basename = Path(rel).name
            if basename.startswith("test_"):
                continue
            if rel.startswith("sparkcache/native/"):
                continue
            if rel.startswith("sparkcache/replication/"):
                continue
            if rel.startswith("sparkcache/experiments/"):
                continue
            # Lazy native modules — only loaded when native restore is enabled.
            if rel in (
                "sparkcache/spark_cache_native.py",
                "sparkcache/spark_context_cache_native_placement.py",
                "sparkcache/spark_context_cache_native_restore.py",
            ):
                continue
            missing.append(rel)

        self.assertEqual(missing, [], (
            f"These non-test sparkcache .py files are not in REQUIRED_FILES "
            f"but may be needed for the connector import closure: {missing}"
        ))

    def test_connector_import_closure_completes(self):
        """Import the connector from the staged bundle and prove every
        local import resolves from the temporary staging root.

        Stubs for vllm and torch are real ``types.ModuleType`` instances
        with the exact names/classes the connector imports.  The test
        asserts the connector, codec, store, persistent engine, streaming
        feature gate, eager streaming modules, and runtime-patch verifier
        all resolve from the temp bundle — not from the repo tree.
        """
        import importlib
        import types as _types

        old_path = sys.path[:]
        old_modules = dict(sys.modules)

        # Build real ModuleType stubs with the exact imported names.
        torch_mod = _types.ModuleType("torch")
        torch_mod.Tensor = type("Tensor", (), {})
        torch_mod.Generator = type("Generator", (), {})
        torch_mod.randn = lambda *a, **k: None
        torch_mod.arange = lambda *a, **k: None
        torch_mod.pow = lambda *a, **k: None
        torch_mod.zeros = lambda *a, **k: None
        torch_mod.SymFloat = float
        torch_mod.bfloat16 = type("bfloat16", (), {})
        torch_mod.float32 = type("float32", (), {})

        def _make_vllm_mod(name: str) -> _types.ModuleType:
            m = _types.ModuleType(name)
            return m

        vllm_mod = _make_vllm_mod("vllm")
        vllm_config = _make_vllm_mod("vllm.config")
        vllm_config.VllmConfig = type("VllmConfig", (), {})
        vllm_logger = _make_vllm_mod("vllm.logger")
        vllm_logger.init_logger = lambda name: type("Logger", (), {
            "info": lambda *a, **k: None, "warning": lambda *a, **k: None,
            "error": lambda *a, **k: None, "debug": lambda *a, **k: None,
        })()
        kv_base = _make_vllm_mod("vllm.distributed.kv_transfer.kv_connector.v1.base")
        kv_base.KVConnectorBase_V1 = type("KVConnectorBase_V1", (), {})
        kv_base.KVConnectorMetadata = type("KVConnectorMetadata", (), {})
        kv_base.KVConnectorRole = type("KVConnectorRole", (), {"WORKER": 0, "SCHEDULER": 1})
        kv_metrics = _make_vllm_mod("vllm.distributed.kv_transfer.kv_connector.v1.metrics")
        kv_metrics.KVConnectorStats = type("KVConnectorStats", (), {})

        # Clean cached sparkcache/torch/vllm imports.
        for key in list(sys.modules.keys()):
            if key.startswith("sparkcache") or key.startswith("spark_context_cache") or key == "torch" or key.startswith("vllm"):
                del sys.modules[key]

        sys.path.insert(0, str(self.staging))
        sys.path.insert(0, str(self.staging / "sparkcache"))

        sys.modules["torch"] = torch_mod
        sys.modules["vllm"] = vllm_mod
        sys.modules["vllm.config"] = vllm_config
        sys.modules["vllm.logger"] = vllm_logger
        for prefix in [
            "vllm.distributed",
            "vllm.distributed.kv_transfer",
            "vllm.distributed.kv_transfer.kv_connector",
            "vllm.distributed.kv_transfer.kv_connector.v1",
        ]:
            sys.modules.setdefault(prefix, _make_vllm_mod(prefix))
        sys.modules["vllm.distributed.kv_transfer.kv_connector.v1.base"] = kv_base
        sys.modules["vllm.distributed.kv_transfer.kv_connector.v1.metrics"] = kv_metrics

        try:
            # Import the connector — triggers the full local closure.
            mod = importlib.import_module("spark_context_cache_connector")
            self.assertTrue(hasattr(mod, "SparkContextCacheConnector"))

            # Assert eager local modules resolved from the temp staging root.
            staging_str = str(self.staging.resolve())
            eager_modules = [
                "spark_context_cache_connector",
                "spark_context_cache_codec",
                "spark_context_cache_store",
                "spark_context_cache_engine",
                "sparkcache.streaming",
                "sparkcache.streaming.feature_gate",
                "sparkcache.streaming.block_lease",
                "sparkcache.streaming.planner",
                "sparkcache.streaming.native_ring",
                "sparkcache.streaming.preemption",
                "sparkcache.streaming.publisher",
            ]
            for name in eager_modules:
                self.assertIn(name, sys.modules, f"Eager module {name!r} not imported")
                file_attr = getattr(sys.modules[name], "__file__", None)
                if file_attr:
                    self.assertTrue(
                        staging_str in str(Path(file_attr).resolve()),
                        f"Module {name!r} loaded from {file_attr}, not from staging root",
                    )

            # Lazy modules: import explicitly to prove they resolve from staging.
            lazy_modules = [
                "sparkcache.streaming.factory",
                "sparkcache.streaming.runtime",
                "sparkcache.streaming.timing",
                "sparkcache.runtime_patches",
                "sparkcache.runtime_patches.verify_lease_contract",
            ]
            for name in lazy_modules:
                m = importlib.import_module(name)
                file_attr = getattr(m, "__file__", None)
                if file_attr:
                    self.assertTrue(
                        staging_str in str(Path(file_attr).resolve()),
                        f"Lazy module {name!r} loaded from {file_attr}, not from staging root",
                    )
        finally:
            sys.path[:] = old_path
            sys.modules.clear()
            sys.modules.update(old_modules)


if __name__ == "__main__":
    unittest.main()
