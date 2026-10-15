#!/usr/bin/env python3
"""The Makefile keeps its required compiler flags when CFLAGS or CPPFLAGS come from the command line.

``make -n`` (nothing runs) of the library's objects with ``CFLAGS=-O1`` and ``CPPFLAGS=-DEXTRA`` must still
print -fPIC, -Werror and -std=c11 (CFLAGS) and -Iinclude and -DLIBSIRCL_CPU_ONLY (CPPFLAGS) beside the given
flags. Without ``override`` in the Makefile a command-line value replaces the whole variable.
"""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class BuildFlagTests(unittest.TestCase):
    def test_command_line_flags_keep_the_required_ones(self):
        with tempfile.TemporaryDirectory() as build:
            out = subprocess.run(["make", "-n", "-C", str(ROOT), f"BUILD={build}", "CFLAGS=-O1",
                                  "CPPFLAGS=-DEXTRA", f"{build}/api.o"],
                                 capture_output=True, text=True, timeout=60,
                                 env={k: v for k, v in os.environ.items() if k not in ("CFLAGS", "CPPFLAGS")})
        self.assertEqual(out.returncode, 0, out.stderr)
        line = next(row for row in out.stdout.splitlines() if "src/api.c" in row)
        for flag in ("-O1", "-DEXTRA", "-fPIC", "-Werror", "-std=c11", "-Iinclude", "-DLIBSIRCL_CPU_ONLY"):
            self.assertIn(flag, line.split(), line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
