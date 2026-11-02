#!/usr/bin/env python3
"""The counts libsircl's documents state agree with the code.

- README.md states the number of the header's functions and of those implemented, which
  tests/api_manifest.json records; STATUS.md lists every function the manifest marks not implemented.
- README.md, STATUS.md and KERNEL_ROUTE.md speak of the four kernel packs the loader embeds
  (src/kernelpack.c's sccl_*_fatbin symbols), never of three or of "both" packs.
- README.md and STATUS.md name the version of this repository's SIRCL package.
"""
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[1]


def text(name):
    return " ".join((ROOT / name).read_text(encoding="utf-8").split())


class DocsConsistencyTests(unittest.TestCase):
    def test_function_counts_match_the_manifest(self):
        functions = json.loads((ROOT / "tests/api_manifest.json").read_text(encoding="utf-8"))["functions"]
        implemented = sum(function["implemented"] for function in functions)
        readme = text("README.md")
        self.assertIn(f"implements {implemented} of the header's {len(functions)} functions", readme)
        self.assertIn(f"the other {len(functions) - implemented} return `ncclInvalidUsage`", readme)
        status = text("STATUS.md")
        missing = [function["name"] for function in functions
                   if not function["implemented"] and f"`{function['name']}`" not in status]
        self.assertEqual(missing, [], "STATUS.md does not list these unsupported functions")

    def test_the_documents_count_the_packs_the_loader_embeds(self):
        loader = (ROOT / "src/kernelpack.c").read_text(encoding="utf-8")
        packs = set(re.findall(r"extern const unsigned char (sccl_\w+_fatbin)\[\];", loader))
        self.assertEqual(len(packs), 4)
        for name in ("README.md", "STATUS.md", "KERNEL_ROUTE.md"):
            body = text(name)
            with self.subTest(document=name):
                self.assertNotRegex(body, r"\bthree (embedded |kernel )?packs\b")
                self.assertNotRegex(body, r"\bboth packs\b")
        self.assertIn("four kernel packs", text("STATUS.md"))

    def test_the_sircl_package_version(self):
        version = re.search(r'^version = "([^"]+)"', (REPOSITORY / "spark_transport/sircl/pyproject.toml").read_text(
            encoding="utf-8"), re.M).group(1)
        for name in ("README.md", "STATUS.md"):
            with self.subTest(document=name):
                self.assertIn(f"`sparkring_sircl` {version}", text(name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
