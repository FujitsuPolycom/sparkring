#!/usr/bin/env python3
"""CPU tests of the two native libraries vendored from SIRCL, so their bytes cannot drift silently.

libsircl compiles SIRCL's native proxy (``src/transport/sircl_roce_proxy.c``, a copy of SIRCL's
``oneshot/_roce_proxy.c``) and point-to-point library (``src/transport/sircl_p2p_proxy.c``, a copy of
``p2p/_p2p_proxy.c``). Each copy must be the LF-ending bytes SIRCL's release builds its natives from, whose
SHA-256 names those natives (``roce_proxy-<first 16 hex>.so``, ``p2p_proxy-<first 16 hex>.so``):

- the copy carries no carriage return (a checkout with CRLF endings has the same text but other bytes and
  another hash);
- its SHA-256 equals the one recorded beside it (``.sha256``), which the Makefile and CMake also check;
- it defines its local feature word (SIRCL change LF), which the engine checks at setup.
"""
from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSPORT = ROOT / "src" / "transport"
COPIES = {"sircl_roce_proxy.c": "roce_local_features", "sircl_p2p_proxy.c": "p2p_local_features"}


class VendoredSourceTests(unittest.TestCase):
    def test_copies_have_lf_endings(self):
        for name in COPIES:
            with self.subTest(copy=name):
                data = (TRANSPORT / name).read_bytes()
                self.assertNotIn(b"\r", data, f"{name} has carriage returns; vendor SIRCL's LF bytes")

    def test_copies_match_their_recorded_hashes(self):
        for name in COPIES:
            with self.subTest(copy=name):
                data = (TRANSPORT / name).read_bytes()
                recorded = (TRANSPORT / f"{name}.sha256").read_text().split()[0]
                self.assertRegex(recorded, r"^[0-9a-f]{64}$")
                self.assertEqual(hashlib.sha256(data).hexdigest(), recorded)

    def test_a_crlf_copy_would_not_match_its_recorded_hash(self):
        # The same text with CRLF endings hashes differently, so recording a CRLF copy's hash would name a
        # native SIRCL's release never builds; the first test refuses such a copy.
        for name in COPIES:
            with self.subTest(copy=name):
                data = (TRANSPORT / name).read_bytes()
                crlf = data.replace(b"\n", b"\r\n")
                self.assertNotEqual(hashlib.sha256(crlf).hexdigest(), hashlib.sha256(data).hexdigest())

    def test_copies_define_their_local_feature_word(self):
        for name, symbol in COPIES.items():
            with self.subTest(copy=name):
                self.assertIn(f"{symbol}(void)".encode(), (TRANSPORT / name).read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
