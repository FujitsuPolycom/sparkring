#!/usr/bin/env python3
"""SOURCE_SNAPSHOT.json's schema, and what NOTICE says the file records.

SOURCE_SNAPSHOT.json is a dated capture: the SHA-256 and size of every file of the SIRCL implementation tree
that STATUS.md's evidence was taken against. It is not the in-repository SIRCL package the library is built
against, and nothing enforces it at build time. These checks keep its shape readable (every entry a relative
path with a 64-digit lowercase SHA-256 and a byte count; the stated file count; the changes from the previous
capture as path lists) and keep NOTICE describing it as that capture.
"""
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SourceSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = json.loads((ROOT / "SOURCE_SNAPSHOT.json").read_text(encoding="utf-8"))

    def test_the_schema(self):
        data = self.snapshot
        self.assertEqual(set(data), {"description", "sources", "selected", "captured_utc",
                                     "changes_from_previous_capture", "files"})
        self.assertIn(data["selected"], data["sources"])
        self.assertRegex(data["captured_utc"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
        stated = re.search(r"(\d+) files\.?$", data["description"])
        self.assertIsNotNone(stated, "the description states the file count")
        self.assertEqual(len(data["files"]), int(stated.group(1)))
        for path, entry in data["files"].items():
            with self.subTest(path=path):
                self.assertFalse(path.startswith("/") or ".." in Path(path).parts or "\\" in path)
                self.assertEqual(set(entry), {"sha256", "bytes"})
                self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
                self.assertIsInstance(entry["bytes"], int)
                self.assertGreaterEqual(entry["bytes"], 0)
        changes = data["changes_from_previous_capture"]
        self.assertEqual(set(changes), {"changed", "added", "removed"})
        for kind, paths in changes.items():
            self.assertIsInstance(paths, list, kind)
            self.assertTrue(all(isinstance(path, str) and path for path in paths), kind)

    def test_notice_describes_the_capture(self):
        notice = " ".join((ROOT / "NOTICE").read_text(encoding="utf-8").split())
        self.assertIn("SOURCE_SNAPSHOT.json records the SHA-256 of every file of the SIRCL implementation tree "
                      "that STATUS.md's evidence was taken against", notice)


if __name__ == "__main__":
    unittest.main(verbosity=2)
