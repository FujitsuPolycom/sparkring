"""Seed identity is stable per translation unit and distinct for device linking."""
import unittest

from nvcc_deterministic import seeded_arguments
from build_nccl import compile_jobs


class DeterministicCompilerTests(unittest.TestCase):
    def test_compiler_temporaries_follow_unique_object_destinations(self):
        for phase in ("-c", "-dc", "-dw"):
            argv = [phase, "common.cu", "-o", "/work/build/common.o"]
            result = seeded_arguments(argv, "/work/src-lf/src/device")
            self.assertEqual(result[1], "--objdir-as-tempdir")
            self.assertEqual(result[2:], argv)

    def test_dependency_scans_and_device_link_do_not_share_compile_temporaries(self):
        for argv in (["-MM", "-dc", "common.cu"],
                     ["-M", "-dc", "common.cu", "-o", "/work/build/common.d"],
                     ["-dlink", "/work/build/common.o", "-o", "/work/build/device_glue.o"]):
            result = seeded_arguments(argv, "/work/src-lf/src/device")
            self.assertNotIn("--objdir-as-tempdir", result)
            self.assertEqual(result[1:], argv)

    def test_compile_parallelism_has_explicit_resource_bounds(self):
        self.assertEqual(compile_jobs("16"), 16)
        for value in ("0", "65", "-1", "unlimited"):
            with self.assertRaises(ValueError):
                compile_jobs(value)

    def test_probe_forwarding(self):
        self.assertEqual(seeded_arguments(["--version"], "/"), ["--version"])
        self.assertEqual(seeded_arguments(["--help"], "/"), ["--help"])

    def test_source_seed_is_stable_across_working_directories(self):
        first = seeded_arguments(["-dc", "common.cu", "-o", "/work/build/common.o"], "/work/src-lf/src/device")
        second = seeded_arguments(["-dc", "/work/src-lf/src/device/common.cu", "-o", "common.o"], "/work/build")
        self.assertEqual(first[0], second[0])
        self.assertRegex(first[0], r"^--frandom-seed=[0-9a-f]{64}$")

    def test_distinct_files_and_link_outputs_have_distinct_seeds(self):
        commands = [
            ["-dc", "/work/src-lf/src/device/common.cu"],
            ["-dc", "/work/src-lf/src/device/onerank.cu"],
            ["-dw", "/work/build/obj/device/gensrc/one.cu"],
            ["-dw", "/work/build/obj/device/gensrc/two.cu"],
            ["-dlink", "/work/build/common.o", "-o", "/work/build/glue.o"],
            ["-dlink", "/work/build/common.o", "-o", "/work/build/other-glue.o"],
        ]
        seeds = [seeded_arguments(argv, "/work/src-lf")[0] for argv in commands]
        self.assertEqual(len(set(seeds)), len(commands))

    def test_include_and_compiler_option_values_are_not_source_inputs(self):
        argv = ["-ccbin", "g++", "-gencode=arch=compute_121,code=sm_121", "-std=c++17",
                "-I", "/headers", "--compiler-options", "-fPIC -fvisibility=hidden",
                "-Xptxas", "-maxrregcount=96", "-Xfatbin", "-compress-all", "-MM", "-dc", "common.cu"]
        result = seeded_arguments(argv, "/work/src-lf/src/device")
        self.assertEqual(result[1:], argv)

    def test_ambiguous_or_unbound_invocations_fail(self):
        invalid = [[], ["one.cu", "two.cu"], ["/tmp/other.cu"], ["../../../outside.cu"],
                   ["-dlink", "/work/build/one.o"], ["-dlink", "one.cu", "-o", "/work/build/glue.o"],
                   ["one.cu", "-o", "/tmp/out.o"], ["one.cu", "-o", "a.o", "-o", "b.o"],
                   ["one.cu", "--frandom-seed=override"], ["@options"], ["--options-file=flags"],
                   ["--version", "one.cu", "two.cu"]]
        for argv in invalid:
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                seeded_arguments(argv, "/work/src-lf")


if __name__ == "__main__":
    unittest.main()
