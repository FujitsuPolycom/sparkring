"""Retained parent binaries must not become duplicate image-layer payloads."""
import os
from pathlib import Path
import tempfile
import unittest

from install_sources import normalize_installed_times


class InstallTimestampTests(unittest.TestCase):
    def test_parent_binary_metadata_is_preserved_while_source_outputs_are_normalized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native = root / "retained.so"
            source = root / "model.py"
            native.write_bytes(b"attested parent binary")
            source.write_bytes(b"value = 1\n")
            os.utime(native, (100, 100))
            os.utime(source, (200, 200))
            original = native.stat().st_mtime_ns
            normalize_installed_times({root, native, source}, 300, {native})
            self.assertEqual(native.stat().st_mtime_ns, original)
            self.assertEqual(int(source.stat().st_mtime), 300)
            self.assertEqual(native.read_bytes(), b"attested parent binary")


if __name__ == "__main__":
    unittest.main()
