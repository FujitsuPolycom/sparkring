#!/usr/bin/env python3
"""The kernel-entry check (tests/check_entries.c) and its negative controls.

The check runs the library's pack loader on the embedded fatbins through the stand-in driver
(tests/fake_cuda.c). It passes on the build; hiding one entry the loader names (FAKE_CUDA_HIDE) makes it
fail and name that entry; without the stand-in first on LD_LIBRARY_PATH it refuses to run.

Usage: python tests/test_kernel_entries.py --build build
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

BUILD = Path("build")


def run(hide: str | None = None, driver: bool = True) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment.pop("FAKE_CUDA_HIDE", None)
    if hide:
        environment["FAKE_CUDA_HIDE"] = hide
    environment["LD_LIBRARY_PATH"] = str((BUILD / "fake-cuda").resolve()) if driver else tempfile.gettempdir()
    return subprocess.run([str(BUILD / "check_entries"), "2"], env=environment, capture_output=True, text=True,
                          timeout=60)


class KernelEntries(unittest.TestCase):
    def test_every_entry_the_loader_names_is_in_every_cubin(self):
        result = run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("are defined in every cubin of the 4 packs", result.stdout)

    def test_a_missing_entry_fails_and_is_named(self):
        for name in ("sircl_twoshot_bf16_w8", "sircl_fold_d11_o4", "sircl_chain_f16_u3", "sircl_ring_gather_u8",
                     "sircl_ring_reduce_bf16_u8", "sircl_p2p_recv_u5", "sircl_p2p_send_u8"):
            with self.subTest(name=name):
                result = run(hide=name)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(name, result.stderr)

    def test_refuses_without_the_stand_in_driver(self):
        result = run(driver=False)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default="build")
    args, rest = parser.parse_known_args()
    BUILD = Path(args.build)
    unittest.main(argv=[sys.argv[0]] + rest, verbosity=2)
