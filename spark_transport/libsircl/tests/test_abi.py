#!/usr/bin/env python3
"""Independent host ABI checks. Run on Linux; no CUDA or Spark is accessed."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "tests/api_manifest.json").read_text())
LIBRARY = ROOT / "build/libsircl.so"


def run(*arguments: str, env: dict | None = None) -> str:
    return subprocess.check_output(arguments, text=True, stderr=subprocess.STDOUT, env=env)


# Host entry points of NCCL's device API (nccl_device/core.h and its barrier and LL all-to-all headers),
# which programs built against NCCL 2.28 or later import; src/device_api.c defines them.
DEVICE_API = {"ncclCommQueryProperties", "ncclDevCommCreate", "ncclDevCommDestroy", "ncclGetLsaMultimemDevicePointer",
              "ncclGetMultimemDevicePointer", "ncclGetLsaDevicePointer", "ncclGetPeerDevicePointer", "ncclTeamWorld",
              "ncclTeamLsa", "ncclTeamRail", "ncclTeamRankToWorld", "ncclTeamRankToLsa",
              "ncclGinBarrierCreateRequirement", "ncclLLA2ACalcSlots", "ncclLLA2ACreateRequirement",
              "ncclLsaBarrierCreateRequirement"}


class HostAbiTests(unittest.TestCase):
    def test_generated_files_match_pinned_public_header(self):
        run(sys.executable, str(ROOT / "tools/generate_api.py"), "--check")
        source = (ROOT / "vendor/nccl.h.in").read_text()
        # Separate extraction from the generator, to catch accidental omissions.
        declared = set(re.findall(r"^\s*(?:ncclResult_t|const char\*|void)\s+(p?nccl\w+)\s*\(", source, re.M))
        expected = {api["name"] for api in MANIFEST["functions"]}
        expected |= {"p" + name for name in expected}
        self.assertEqual(declared, expected)
        self.assertEqual(len(expected), 146)

    def test_exported_symbol_set(self):
        symbols = run("nm", "-D", "--defined-only", str(LIBRARY))
        exported = {line.split()[-1].split("@")[0] for line in symbols.splitlines() if line.split()}
        expected = {api["name"] for api in MANIFEST["functions"]} | DEVICE_API
        expected |= {"p" + name for name in expected}
        self.assertEqual({name for name in exported if name.startswith(("nccl", "pnccl"))}, expected)
        self.assertFalse({name for name in exported if name.startswith("sccl_")}, "Internal helpers must be hidden")
        # Beyond the NCCL API, exactly the three extension functions, by name (no sircl_fabric_* helpers).
        self.assertEqual({name for name in exported if not name.startswith(("nccl", "pnccl"))},
                         {"sirclGetInfo", "sirclSetWaitRegime", "sirclGetReceipt"})

    def test_soname_and_runtime_dependencies(self):
        dynamic = run("readelf", "-d", str(LIBRARY))
        self.assertRegex(dynamic, r"\(SONAME\).*\[libnccl\.so\.2\]")
        dependencies = re.findall(r"\(NEEDED\).*\[([^\]]+)\]", dynamic)
        self.assertFalse([name for name in dependencies if any(token in name.lower() for token in ("nccl", "cuda", "python", "torch"))])

    def test_header_layout_and_initializers(self):
        source = r'''
#define LIBSIRCL_CPU_ONLY 1
#include "nccl.h"
#include <stddef.h>
#include <stdio.h>
_Static_assert(sizeof(void*) == 8, "Tests qualify the 64-bit host ABI");
_Static_assert(sizeof(ncclUniqueId) == 128, "unique id");
_Static_assert(sizeof(ncclConfig_t) == 120, "current config size");
_Static_assert(offsetof(ncclConfig_t, blocking) == 16, "config prefix");
_Static_assert(offsetof(ncclConfig_t, netName) == 32, "legacy config layout");
_Static_assert(offsetof(ncclConfig_t, commName) == 48, "config name");
_Static_assert(offsetof(ncclConfig_t, nChannelsPerNetPeer) == 72, "72-byte legacy boundary");
_Static_assert(offsetof(ncclConfig_t, nvlsHostMode) == 112, "last config field");
_Static_assert(sizeof(ncclCollConfig_t) == 72, "collective config size");
_Static_assert(offsetof(ncclCollConfig_t, launchCompletionEvent) == 64, "event ABI");
_Static_assert(sizeof(ncclSimInfo_t) == 24, "sim info");
_Static_assert(sizeof(ncclEncryptionConfig_t) == 32, "encryption config");
_Static_assert(sizeof(ncclConfigExt_t) == 24, "extension config");
_Static_assert(sizeof(ncclWaitSignalDesc_t) == 16, "signal descriptor");
_Static_assert(sizeof(ncclResult_t) == 4 && sizeof(ncclDataType_t) == 4 && sizeof(ncclRedOp_t) == 4, "enum ABI");
_Static_assert(ncclSuccess == 0 && ncclInvalidArgument == 4 && ncclInvalidUsage == 5 && ncclTimeout == 8, "error ABI");
_Static_assert(ncclFloat32 == 7 && ncclBfloat16 == 9 && ncclFloat8e5m2 == 11 && ncclNumTypes == 12, "dtype ABI");
_Static_assert(ncclSum == 0 && ncclAvg == 4 && ncclMaxRedOp == 0x7fffffff, "op ABI");
int main(void) {
  ncclConfig_t c = NCCL_CONFIG_INITIALIZER;
  ncclCollConfig_t a = NCCL_COLLCONFIG_INITIALIZER;
  ncclSimInfo_t s = NCCL_SIM_INFO_INITIALIZER;
  ncclEncryptionConfig_t e = NCCL_ENCRYPTION_CONFIG_INITIALIZER;
  if (c.size != 120 || c.magic != 0xcafebeef || c.version != 23203) return 1;
  if (c.blocking != INT_MIN || c.nvlsHostMode != INT_MIN || c.netName != NULL) return 2;
  if (a.size != 72 || a.forceAlgSelection != 1 || a.userProfilerTag != 0 || a.launchCompletionEvent != NULL) return 3;
  if (s.size != 24 || s.magic != 0x74685283 || s.estimatedTime != -1.0f) return 4;
  if (e.size != 32 || e.mode != NCCL_ENCRYPTION_MODE_NONE || e.psk != NULL) return 5;
  puts("header ABI: 120-byte config, 72-byte collective config, 128-byte UID");
  return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="sircl-abi-") as directory:
            path = Path(directory)
            (path / "layout.c").write_text(source)
            run("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "include"), str(path / "layout.c"), "-o", str(path / "layout"))
            self.assertIn("header ABI", run(str(path / "layout")))
            (path / "header.cpp").write_text('#define LIBSIRCL_CPU_ONLY 1\n#include "nccl.h"\nstatic_assert(sizeof(ncclConfig_t) == 120, "C++ ABI");\n')
            run("c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", "-I", str(ROOT / "include"), str(path / "header.cpp"))

    def test_preload_and_soname_reuse(self):
        symbols = [api["name"] for api in MANIFEST["functions"]]
        symbols += ["p" + symbol for symbol in symbols]
        literal = ",\n".join('"' + symbol + '"' for symbol in symbols)
        source = r'''
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
extern int ncclGetVersion(int*);
int main(void) {
  const char* names[] = {SYMBOLS};
  int version = -1;
  if (ncclGetVersion(&version) != 0 || version != 22705) return 1;
  for (unsigned i = 0; i < sizeof(names)/sizeof(names[0]); ++i) {
    if (!dlsym(RTLD_DEFAULT, names[i])) { fprintf(stderr, "missing %s\n", names[i]); return 2; }
  }
  void* handle = dlopen("libnccl.so.2", RTLD_NOW | RTLD_LOCAL);
  if (!handle) { fprintf(stderr, "%s\n", dlerror()); return 3; }
  typedef int (*get_version_fn)(int*);
  get_version_fn get = (get_version_fn)dlsym(handle, "ncclGetVersion");
  version = -1;
  if (!get || get(&version) != 0 || version != 22705) return 4;
  if (dlsym(handle, "ncclGetVersion") != dlsym(RTLD_DEFAULT, "ncclGetVersion")) return 5;
  get_version_fn twin = (get_version_fn)dlsym(handle, "pncclGetVersion");
  if (!twin || twin(&version) != 0 || version != 22705) return 6;
  dlclose(handle);
  puts("preload: all 146 symbols; linked and dlopen clients share one SONAME");
  return 0;
}
'''.replace("SYMBOLS", literal)
        with tempfile.TemporaryDirectory(prefix="sircl-preload-") as directory:
            path = Path(directory)
            (path / "dummy.c").write_text('int ncclGetVersion(int* v) { if(v) *v=999; return 0; }\n')
            run("cc", "-shared", "-fPIC", "-Wl,-soname,libnccl.so.2", str(path / "dummy.c"), "-o", str(path / "libnccl.so.2"))
            (path / "libnccl.so").symlink_to("libnccl.so.2")
            (path / "preload.c").write_text(source)
            run("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", str(path / "preload.c"), "-L", str(path), "-Wl,-rpath," + str(path), "-lnccl", "-ldl", "-o", str(path / "preload"))
            env = os.environ.copy()
            env.pop("LIBSIRCL_NCCL_API_VERSION", None)
            env["LD_PRELOAD"] = str(LIBRARY)
            self.assertIn("all 146 symbols", run(str(path / "preload"), env=env))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=LIBRARY)
    options, extra = parser.parse_known_args()
    LIBRARY = options.library.resolve()
    unittest.main(argv=[sys.argv[0]] + extra)
