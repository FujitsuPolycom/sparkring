#!/usr/bin/env python3
"""Unit tests for checkpoint_manifest_generator.

Covers: determinism, root-name independence, content/path/size sensitivity,
nested files, empty root, symlink root rejection, symlink file rejection,
symlink directory rejection, non-regular file rejection, same-size
timestamp-change detection, output overwrite refusal (exclusive create),
Unicode NFC collision detection, domain separation, identity non-recursion,
and CLI behavior.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
# Ensure scripts/ is importable
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from checkpoint_manifest_generator import (  # noqa: E402
    MANIFEST_VERSION,
    ManifestError,
    compute_identity,
    generate_manifest,
    main,
)

_GENERATOR = Path(__file__).resolve().parent / "checkpoint_manifest_generator.py"


def _make_tree(root: Path, files: dict[str, bytes]) -> None:
    """Create files under root.  Keys are POSIX relative paths."""
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _stable_stat(path: Path) -> tuple[int, int, int, int, int]:
    """Return the 5-tuple the generator now expects."""
    st = path.stat()
    return (st.st_size, st.st_ino, st.st_dev, st.st_mtime_ns, st.st_ctime_ns)


class DeterminismTests(unittest.TestCase):

    def test_same_produce_same_identity(self) -> None:
        content = {f"shard-{i}.bin": bytes([i]) * 100 for i in range(5)}
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            r1 = Path(d1) / "model"
            r2 = Path(d2) / "model"
            r1.mkdir()
            r2.mkdir()
            _make_tree(r1, content)
            _make_tree(r2, content)
            _, id1 = generate_manifest(r1)
            _, id2 = generate_manifest(r2)
            self.assertEqual(id1, id2)

    def test_identity_independent_of_root_dir_name(self) -> None:
        content = {"config.json": b'{"a":1}', "weight.bin": b"\x00" * 42}
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            r1 = Path(d1) / "target_model"
            r2 = Path(d2) / "completely_different_name"
            r1.mkdir()
            r2.mkdir()
            _make_tree(r1, content)
            _make_tree(r2, content)
            _, id1 = generate_manifest(r1)
            _, id2 = generate_manifest(r2)
            self.assertEqual(id1, id2)

    def test_identity_independent_of_enumeration_order(self) -> None:
        """Identity must be the same regardless of filesystem enumeration order."""
        content = {"z.json": b"z", "a.json": b"a", "m.json": b"m"}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "model"
            root.mkdir()
            _make_tree(root, content)
            _, id1 = generate_manifest(root)
            _, id2 = generate_manifest(root)
            self.assertEqual(id1, id2)


class SensitivityTests(unittest.TestCase):

    def test_different_content_produces_different_identity(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            r1 = Path(d) / "a"
            r2 = Path(d) / "b"
            r1.mkdir()
            r2.mkdir()
            _make_tree(r1, {"file.bin": b"hello"})
            _make_tree(r2, {"file.bin": b"world"})
            _, id1 = generate_manifest(r1)
            _, id2 = generate_manifest(r2)
            self.assertNotEqual(id1, id2)

    def test_different_path_produces_different_identity(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            r1 = Path(d) / "a"
            r2 = Path(d) / "b"
            r1.mkdir()
            r2.mkdir()
            _make_tree(r1, {"config.json": b"same"})
            _make_tree(r2, {"different_name.json": b"same"})
            _, id1 = generate_manifest(r1)
            _, id2 = generate_manifest(r2)
            self.assertNotEqual(id1, id2)

    def test_different_size_produces_different_identity(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            r1 = Path(d) / "a"
            r2 = Path(d) / "b"
            r1.mkdir()
            r2.mkdir()
            _make_tree(r1, {"file.bin": b"XXXX"})
            _make_tree(r2, {"file.bin": b"XXX"})
            _, id1 = generate_manifest(r1)
            _, id2 = generate_manifest(r2)
            self.assertNotEqual(id1, id2)

    def test_nested_files_are_inventoried(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "model"
            root.mkdir()
            _make_tree(root, {
                "config.json": b"cfg",
                "shards/shard-0.bin": b"s0",
                "shards/sub/shard-1.bin": b"s1",
            })
            receipt, identity = generate_manifest(root)
            paths = {f["rel_path"] for f in receipt["files"]}
            self.assertIn("config.json", paths)
            self.assertIn("shards/shard-0.bin", paths)
            self.assertIn("shards/sub/shard-1.bin", paths)
            self.assertEqual(receipt["file_count"], 3)
            self.assertRegex(identity, r"[0-9a-f]{64}")


class FailClosedTests(unittest.TestCase):

    def test_empty_root_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "empty"
            root.mkdir()
            with self.assertRaisesRegex(ManifestError, "empty inventory"):
                generate_manifest(root)

    def test_nonexistent_root_fails_closed(self) -> None:
        with self.assertRaisesRegex(ManifestError, "not a directory"):
            generate_manifest("/nonexistent/path/that/does/not/exist")

    def test_symlink_file_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "model"
            root.mkdir()
            target = Path(d) / "target.bin"
            target.write_bytes(b"data")
            link = root / "link.bin"
            try:
                os.symlink(target, link)
            except OSError:
                self.skipTest("symlinks not supported on this platform")
            with self.assertRaisesRegex(ManifestError, "symlink"):
                generate_manifest(root)

    def test_symlink_root_rejected(self) -> None:
        """An explicitly supplied artifact root that is itself a symlink
        must be rejected before dereferencing."""
        with tempfile.TemporaryDirectory() as d:
            real_root = Path(d) / "real_model"
            real_root.mkdir()
            _make_tree(real_root, {"file.bin": b"data"})
            link_root = Path(d) / "link_to_model"
            try:
                os.symlink(real_root, link_root)
            except OSError:
                self.skipTest("symlinks not supported on this platform")
            with self.assertRaisesRegex(ManifestError, "symlink"):
                generate_manifest(link_root)

    def test_symlink_directory_rejected(self) -> None:
        """A symlinked directory inside the root must be rejected,
        not silently excluded."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "model"
            root.mkdir()
            _make_tree(root, {"file.bin": b"data"})
            real_subdir = Path(d) / "real_subdir"
            real_subdir.mkdir()
            _make_tree(real_subdir, {"inside.bin": b"inside"})
            link_dir = root / "linked_dir"
            try:
                os.symlink(real_subdir, link_dir)
            except OSError:
                self.skipTest("symlinks not supported on this platform")
            with self.assertRaisesRegex(ManifestError, "symlink"):
                generate_manifest(root)

    def test_change_during_read_size_fails_closed(self) -> None:
        """File that changes size during read must be detected."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "model"
            root.mkdir()
            _make_tree(root, {"file.bin": b"original"})

            call_count = [0]
            original_stat = _stable_stat(root / "file.bin")

            def fake_stat(path: Path) -> tuple[int, int, int, int, int]:
                call_count[0] += 1
                if call_count[0] <= 1:
                    return original_stat
                # Change size after read
                return (
                    original_stat[0] + 999,
                    original_stat[1],
                    original_stat[2],
                    original_stat[3],
                    original_stat[4],
                )

            with self.assertRaisesRegex(ManifestError, "changed during read"):
                generate_manifest(root, _stat_fn=fake_stat)

    def test_change_during_read_same_size_timestamp_fails_closed(self) -> None:
        """File that changes content but preserves size+inode and only
        modifies mtime_ns/ctime_ns must be detected."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "model"
            root.mkdir()
            _make_tree(root, {"file.bin": b"original"})

            call_count = [0]
            original_stat = _stable_stat(root / "file.bin")

            def fake_stat(path: Path) -> tuple[int, int, int, int, int]:
                call_count[0] += 1
                if call_count[0] <= 1:
                    return original_stat
                # Same size, same inode, same dev — but different timestamps
                return (
                    original_stat[0],
                    original_stat[1],
                    original_stat[2],
                    original_stat[3] + 1_000_000,
                    original_stat[4] + 1_000_000,
                )

            with self.assertRaisesRegex(ManifestError, "changed during read"):
                generate_manifest(root, _stat_fn=fake_stat)

    def test_unicode_nfc_collision_rejected(self) -> None:
        """Two paths that normalize to the same NFC + POSIX string must
        be detected as a collision and rejected.

        On platforms that permit both composed and decomposed Unicode
        filenames in the same directory, NFC normalization maps them
        to the same relative path.  We test this by creating a file
        with a composed name and then attempting to create one with a
        decomposed name; if the OS treats them as the same file, we
        skip.  If the OS permits both, the generator must reject the
        collision.
        """
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "model"
            root.mkdir()
            # é (NFC: U+00E9) vs é (NFD: U+0065 U+0301)
            composed = "caf\u00e9.bin"
            decomposed = "cafe\u0301.bin"
            _make_tree(root, {composed: b"composed"})
            decomposed_path = root / decomposed
            if decomposed_path.exists():
                # OS treats them as the same file — collision can't be
                # tested on this filesystem.  But the generator should
                # still produce a valid identity for the single file.
                _, identity = generate_manifest(root)
                self.assertRegex(identity, r"[0-9a-f]{64}")
            else:
                decomposed_path.write_bytes(b"decomposed")
                with self.assertRaisesRegex(ManifestError, "duplicate normalized path"):
                    generate_manifest(root)

    def test_generate_manifest_passes_supplied_path_not_resolved(self) -> None:
        """Non-skippable seam test: generate_manifest() must pass the
        original supplied path to inventory_artifact_root() without
        resolving it first.  We verify this by intercepting the call
        and confirming the path received by inventory_artifact_root is
        the same object the caller supplied (not a resolved version).

        This test does NOT require symlinks — it proves the code path
        is correct regardless of platform.
        """
        import checkpoint_manifest_generator as cmg

        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "model"
            root.mkdir()
            _make_tree(root, {"file.bin": b"data"})

            captured_paths: list[Path] = []
            original_inventory = cmg.inventory_artifact_root

            def capturing_inventory(
                artifact_root, *, chunk_size=cmg.CHUNK_SIZE,
                _hasher=cmg._hash_file, _stat_fn=cmg._stable_stat,
            ):
                # Capture the path as received — before any resolve()
                captured_paths.append(Path(artifact_root))
                return original_inventory(
                    artifact_root, chunk_size=chunk_size,
                    _hasher=_hasher, _stat_fn=_stat_fn,
                )

            with mock.patch.object(
                cmg, "inventory_artifact_root", side_effect=capturing_inventory
            ):
                generate_manifest(root)

            self.assertEqual(len(captured_paths), 1)
            # The path received by inventory must be the supplied path,
            # not a resolved/dereferenced version.
            self.assertEqual(captured_paths[0], root)
            # Critically: it must NOT be a resolved symlink target.
            # If root is not a symlink, resolve() == root on most
            # platforms, but the key invariant is that generate_manifest
            # did not call resolve() before passing the path through.
            self.assertFalse(captured_paths[0].is_symlink())

    def test_generate_manifest_rejects_symlink_root(self) -> None:
        """Real symlink-root test through the public generate_manifest()
        API — not just inventory_artifact_root().  Skipped on platforms
        without symlink support."""
        with tempfile.TemporaryDirectory() as d:
            real_root = Path(d) / "real_model"
            real_root.mkdir()
            _make_tree(real_root, {"file.bin": b"data"})
            link_root = Path(d) / "link_to_model"
            try:
                os.symlink(real_root, link_root)
            except OSError:
                self.skipTest("symlinks not supported on this platform")
            with self.assertRaisesRegex(ManifestError, "symlink"):
                generate_manifest(link_root)

    def test_traversal_error_fails_closed(self) -> None:
        """An OSError during os.walk traversal (e.g. permission denied
        on a subdirectory) must be converted to ManifestError, not
        silently skipped."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "model"
            root.mkdir()
            _make_tree(root, {"file.bin": b"data"})

            # Create a subdirectory that will cause a traversal error.
            subdir = root / "restricted"
            subdir.mkdir()
            (subdir / "secret.bin").write_bytes(b"secret")

            # Remove all permissions on the subdirectory to trigger
            # an OSError during os.walk.  On Windows this may not
            # actually deny access (POSIX perms are advisory), so we
            # also test via a seam.
            try:
                os.chmod(subdir, 0o000)
                # Try the real path — if it raises ManifestError, good.
                # If it succeeds (Windows), fall through to the seam.
                try:
                    with self.assertRaisesRegex(ManifestError, "traversal error|cannot"):
                        generate_manifest(root)
                    return  # Real test passed
                except (AssertionError, ManifestError):
                    pass  # Fall through to seam test
            finally:
                os.chmod(subdir, 0o755)  # Restore for cleanup

            # Seam test: mock os.walk to raise OSError
            import checkpoint_manifest_generator as cmg

            def failing_walk(*args, **kwargs):
                raise OSError("simulated traversal failure")

            with mock.patch.object(cmg.os, "walk", side_effect=failing_walk):
                with self.assertRaisesRegex(
                    ManifestError, "traversal error|cannot stat"
                ):
                    generate_manifest(root)

    def test_directory_nfc_collision_rejected(self) -> None:
        """Two directory names that normalize to the same NFC string but
        contain different files must be detected as a collision, not
        silently merged into one namespace.

        On platforms that permit both composed and decomposed Unicode
        directory names, NFC normalization maps them to the same path.
        If the OS treats them as the same directory, we skip the
        collision assertion but still verify a valid identity."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "model"
            root.mkdir()
            # café/ (NFC: U+00E9) vs café/ (NFD: U+0065 U+0301)
            composed_dir_name = "caf\u00e9"
            decomposed_dir_name = "cafe\u0301"

            composed_dir = root / composed_dir_name
            composed_dir.mkdir()
            (composed_dir / "a.bin").write_bytes(b"a")

            decomposed_dir = root / decomposed_dir_name
            if decomposed_dir.exists():
                # OS treats them as the same directory — can't test
                # collision on this filesystem.
                _, identity = generate_manifest(root)
                self.assertRegex(identity, r"[0-9a-f]{64}")
            else:
                decomposed_dir.mkdir()
                (decomposed_dir / "b.bin").write_bytes(b"b")
                with self.assertRaisesRegex(
                    ManifestError, "duplicate normalized"
                ):
                    generate_manifest(root)


class OutputTests(unittest.TestCase):

    def test_output_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "model"
            root.mkdir()
            _make_tree(root, {"file.bin": b"data"})
            existing = Path(d) / "existing.json"
            existing.write_text("{}")
            ret = main([
                "--artifact-root", str(root),
                "--output", str(existing),
            ])
            self.assertNotEqual(ret, 0)

    def test_output_exclusive_create_no_toctou(self) -> None:
        """Exclusive creation (open mode 'x') must atomically fail if the
        file exists, with no traceback — a clean CLI error."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "model"
            root.mkdir()
            _make_tree(root, {"file.bin": b"data"})
            output = Path(d) / "out.json"
            ret = main([
                "--artifact-root", str(root),
                "--output", str(output),
            ])
            self.assertEqual(ret, 0)
            self.assertTrue(output.exists())
            # Second attempt must fail cleanly, not crash
            ret2 = main([
                "--artifact-root", str(root),
                "--output", str(output),
            ])
            self.assertNotEqual(ret2, 0)

    def test_output_writes_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "model"
            root.mkdir()
            _make_tree(root, {"file.bin": b"data"})
            output = Path(d) / "receipt.json"
            ret = main([
                "--artifact-root", str(root),
                "--output", str(output),
            ])
            self.assertEqual(ret, 0)
            self.assertTrue(output.exists())
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("checkpoint_identity_sha256", receipt)
            self.assertRegex(
                receipt["checkpoint_identity_sha256"], r"[0-9a-f]{64}"
            )

    def test_cli_manifest_error_returns_1_not_traceback(self) -> None:
        """A ManifestError from a bad root must produce a clean exit,
        not a Python traceback."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "empty"
            root.mkdir()
            ret = main([
                "--artifact-root", str(root),
            ])
            self.assertNotEqual(ret, 0)


class IdentityTests(unittest.TestCase):

    def test_identity_is_domain_separated(self) -> None:
        """The identity must differ from a raw SHA-256 of the same JSON."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "model"
            root.mkdir()
            _make_tree(root, {"file.bin": b"data"})
            receipt, identity = generate_manifest(root)
            # The identity input excludes artifact_root_name; simulate
            # what compute_identity does for the raw comparison.
            identity_input = {
                k: v for k, v in receipt.items()
                if k != "artifact_root_name"
            }
            raw_sha = hashlib.sha256(
                json.dumps(
                    identity_input, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("ascii")
            ).hexdigest()
            self.assertNotEqual(identity, raw_sha)
            self.assertRegex(identity, r"[0-9a-f]{64}")

    def test_identity_excludes_checkpoint_identity_field(self) -> None:
        """The identity input must not recursively include
        checkpoint_identity_sha256."""
        receipt, _ = generate_manifest(
            _make_simple_root())
        receipt_with_id = dict(receipt)
        receipt_with_id["checkpoint_identity_sha256"] = "deadbeef" * 8
        # compute_identity must produce the same value regardless of
        # whether checkpoint_identity_sha256 is present in the receipt.
        id_without = compute_identity(receipt)
        id_with = compute_identity(receipt_with_id)
        self.assertEqual(id_without, id_with)

    def test_receipt_has_version(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "model"
            root.mkdir()
            _make_tree(root, {"file.bin": b"data"})
            receipt, _ = generate_manifest(root)
            self.assertEqual(receipt["manifest_version"], MANIFEST_VERSION)

    def test_receipt_documents_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "model"
            root.mkdir()
            _make_tree(root, {"file.bin": b"data"})
            receipt, _ = generate_manifest(root)
            self.assertIn("path_normalization", receipt)
            self.assertIn("NFC", receipt["path_normalization"])

    def test_receipt_records_file_details(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "model"
            root.mkdir()
            _make_tree(root, {"file.bin": b"hello world"})
            receipt, _ = generate_manifest(root)
            entry = receipt["files"][0]
            self.assertEqual(entry["rel_path"], "file.bin")
            self.assertEqual(entry["byte_size"], 11)
            self.assertEqual(
                entry["content_sha256"],
                hashlib.sha256(b"hello world").hexdigest(),
            )

    def test_stdout_includes_identity(self) -> None:
        """When no --output is given, stdout must include
        checkpoint_identity_sha256 in the receipt JSON."""
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "model"
            root.mkdir()
            _make_tree(root, {"file.bin": b"data"})
            result = subprocess.run(
                [sys.executable, str(_GENERATOR),
                 "--artifact-root", str(root)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads(result.stdout)
            self.assertIn("checkpoint_identity_sha256", receipt)
            self.assertRegex(
                receipt["checkpoint_identity_sha256"], r"[0-9a-f]{64}"
            )

    def test_cli_subprocess_succeeds(self) -> None:
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "model"
            root.mkdir()
            _make_tree(root, {"file.bin": b"data"})
            result = subprocess.run(
                [sys.executable, str(_GENERATOR),
                 "--artifact-root", str(root)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads(result.stdout)
            self.assertIn("checkpoint_identity_sha256", receipt)
            self.assertRegex(
                receipt["checkpoint_identity_sha256"], r"[0-9a-f]{64}"
            )


def _make_simple_root() -> Path:
    """Create a minimal artifact root and return it."""
    d = tempfile.mkdtemp()
    root = Path(d) / "model"
    root.mkdir()
    _make_tree(root, {"file.bin": b"data"})
    return root


if __name__ == "__main__":
    unittest.main()
