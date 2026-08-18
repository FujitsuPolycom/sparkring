"""Vendored runtime modules must be byte-identical to the source tree."""

from __future__ import annotations

import hashlib
import pathlib
import sys
import unittest

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parent.parent
VENDOR = PLUGIN_ROOT / "src" / "sparkring" / "_vendor"
SOURCE = (
    PLUGIN_ROOT.parent / "spark_transport" / "integrations" / "vllm"
)

sys.path.insert(0, str(PLUGIN_ROOT / "scripts"))
from sync_vendor import MODULES  # noqa: E402


def _digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(
    SOURCE.is_dir(), "source tree not present (wheel-installed run)"
)
class VendorParityTest(unittest.TestCase):
    def test_every_vendored_module_matches_source(self) -> None:
        for name in MODULES:
            with self.subTest(module=name):
                vendored = VENDOR / f"{name}.py"
                source = SOURCE / f"{name}.py"
                self.assertTrue(
                    vendored.is_file(),
                    f"{name}.py missing from vendor; run "
                    "scripts/sync_vendor.py",
                )
                self.assertEqual(
                    _digest(vendored),
                    _digest(source),
                    f"{name}.py drifted from source; run "
                    "scripts/sync_vendor.py",
                )

    def test_vendor_contains_no_unlisted_modules(self) -> None:
        vendored = {path.stem for path in VENDOR.glob("*.py")}
        self.assertEqual(vendored, set(MODULES))


if __name__ == "__main__":
    unittest.main()
