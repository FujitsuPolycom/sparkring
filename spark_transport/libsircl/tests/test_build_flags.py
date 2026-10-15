#!/usr/bin/env python3
"""The build's flags and test lists.

- The Makefile keeps its required compiler flags when CFLAGS or CPPFLAGS come from the command line:
  ``make -n`` (nothing runs) of the library's objects with ``CFLAGS=-O1`` and ``CPPFLAGS=-DEXTRA`` must still
  print -fPIC, -Werror, -std=c11 and -fstack-protector-strong (CFLAGS) and -Iinclude, -DLIBSIRCL_CPU_ONLY and
  -D_FORTIFY_SOURCE=2 (CPPFLAGS) beside the given flags.
- The library (``--library``) is linked with full RELRO and immediate binding and a non-executable stack
  (``readelf``: a GNU_RELRO segment, BIND_NOW, and a GNU_STACK segment without execute permission).
- ``make check`` and CMake's ``ctest`` run the same suites: the test names in CMakeLists.txt equal the
  suites the Makefile's check recipe runs.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "build" / "libsircl.so"


class BuildFlagTests(unittest.TestCase):
    def test_command_line_flags_keep_the_required_ones(self):
        with tempfile.TemporaryDirectory() as build:
            out = subprocess.run(["make", "-n", "-C", str(ROOT), f"BUILD={build}", "CFLAGS=-O1",
                                  "CPPFLAGS=-DEXTRA", f"{build}/api.o"],
                                 capture_output=True, text=True, timeout=60,
                                 env={k: v for k, v in os.environ.items() if k not in ("CFLAGS", "CPPFLAGS")})
        self.assertEqual(out.returncode, 0, out.stderr)
        line = next(row for row in out.stdout.splitlines() if "src/api.c" in row)
        for flag in ("-O1", "-DEXTRA", "-fPIC", "-Werror", "-std=c11", "-Iinclude", "-DLIBSIRCL_CPU_ONLY",
                     "-fstack-protector-strong", "-D_FORTIFY_SOURCE=2"):
            self.assertIn(flag, line.split(), line)

    def test_the_library_is_linked_hardened(self):
        if not LIBRARY.exists():
            self.skipTest(f"{LIBRARY} is not built")
        program = subprocess.run(["readelf", "-lW", str(LIBRARY)], capture_output=True, text=True, check=True).stdout
        dynamic = subprocess.run(["readelf", "-dW", str(LIBRARY)], capture_output=True, text=True, check=True).stdout
        self.assertIn("GNU_RELRO", program)
        self.assertRegex(dynamic, r"\(FLAGS\)\s+BIND_NOW|\(FLAGS_1\)\s+Flags:.*\bNOW\b|\(BIND_NOW\)")
        stack = next(row for row in program.splitlines() if "GNU_STACK" in row)
        self.assertNotIn("E", stack.split()[-2], stack)

    def test_make_check_and_ctest_run_the_same_suites(self):
        makefile = (ROOT / "Makefile").read_text()
        recipe = makefile[makefile.index("\ncheck:"):makefile.index("exit $$status")]
        from_make = set(re.findall(r"tests/test_(\w+)\.py", recipe))
        from_make |= {name.removeprefix("test_") for name in re.findall(r"\$\(BUILD\)/(test_\w+)", recipe)}
        cmake = (ROOT / "CMakeLists.txt").read_text()
        from_cmake = set(re.findall(r"add_test\(NAME (\w+)", cmake))
        for names in re.findall(r"foreach\(name ([\w ]+)\)\s*\n\s*add_test", cmake):
            from_cmake |= set(names.split())
        from_cmake.discard("${name}")
        self.assertEqual(from_make, from_cmake)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=LIBRARY)
    options, extra = parser.parse_known_args()
    LIBRARY = options.library.resolve()
    unittest.main(argv=[sys.argv[0], *extra], verbosity=2)
