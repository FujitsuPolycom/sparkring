#!/usr/bin/env python3
"""Show what maps or initializes the CUDA driver in a process that loads libsircl and never uses CUDA.

Runs four child interpreters and prints one JSON record:

- ``startup``: a plain ``python3`` with site packages, before anything is imported: the CUDA-related
  libraries the environment maps (``LD_PRELOAD``, ``/etc/ld.so.preload``, ``.pth`` files, a
  ``sitecustomize`` module);
- ``isolated``: ``python3 -S -E`` without ``LD_PRELOAD``: the interpreter alone;
- ``isolated_load`` and ``startup_load``: the same two, then the library loaded and its CUDA-free entry
  points called (``ncclGetVersion``, ``ncclGetUniqueId``, ``ncclGetErrorString``, ``sirclGetInfo``).

For each it lists the mapped files whose names contain ``cuda`` and, when ``libcuda.so`` is mapped, the
result of ``cuDeviceGetCount``: 3 (``CUDA_ERROR_NOT_INITIALIZED``) until some code calls ``cuInit``.
The record also lists ``LD_PRELOAD``, ``/etc/ld.so.preload``, the site directories' ``.pth`` files that
execute code, and the modules imported at startup whose names mention torch, cuda or nvidia. The library
is responsible for a mapping or an initialization only when ``*_load`` shows one its counterpart lacks.

Usage: python3 tools/probe_cuda_on_load.py --library build/libsircl.so
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CHILD = r'''
import ctypes as C, json, os, sys
own = os.path.realpath(sys.argv[1]) if len(sys.argv) > 1 else None
def state():
    names = sorted({line.split()[-1] for line in open("/proc/self/maps")
                    if "cuda" in os.path.basename(line.split()[-1]).lower() and line.split()[-1] != own})
    status = None
    if any("/libcuda.so" in name for name in names):
        try:
            cuda = C.CDLL("libcuda.so.1", mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
            count = C.c_int(0)
            status = cuda.cuDeviceGetCount(C.byref(count))
        except OSError as error:
            status = f"not reachable by soname: {error}"
    return {"cuda_mappings": names, "cuDeviceGetCount": status}
record = {"before": state()}
if len(sys.argv) > 1:
    library = C.CDLL(sys.argv[1])
    version = C.c_int(0)
    library.ncclGetVersion(C.byref(version))
    class UniqueId(C.Structure):
        _fields_ = [("internal", C.c_ubyte * 128)]
    unique = UniqueId()
    library.ncclGetUniqueId(C.byref(unique))
    library.ncclGetErrorString.restype = C.c_char_p
    library.ncclGetErrorString(0)
    library.sirclGetInfo.restype = C.c_char_p
    library.sirclGetInfo()
    record["after"] = state()
    record["version"] = version.value
record["modules"] = sorted(m for m in sys.modules if any(k in m.lower() for k in ("torch", "cuda", "nvidia")))
print(json.dumps(record))
'''


def child(arguments, environment, library=None):
    command = [sys.executable, *arguments, "-c", CHILD] + ([str(library)] if library else [])
    out = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=120)
    if out.returncode:
        return {"error": out.stderr.strip()[-2000:]}
    return json.loads(out.stdout.strip().splitlines()[-1])


def executing_pth_files():
    import site

    found = []
    for directory in site.getsitepackages() + [site.getusersitepackages()]:
        for path in sorted(Path(directory).glob("*.pth")) if Path(directory).is_dir() else []:
            lines = [line.strip() for line in path.read_text(errors="replace").splitlines()]
            code = [line for line in lines if line.startswith(("import ", "import\t"))]
            if code:
                found.append({"file": str(path), "lines": code[:5]})
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", type=Path, required=True)
    library = parser.parse_args().library.resolve()
    plain = dict(os.environ)
    bare = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD" and not k.startswith("PYTHON")}
    preload_file = Path("/etc/ld.so.preload")
    record = {
        "library": str(library),
        "LD_PRELOAD": os.environ.get("LD_PRELOAD", ""),
        "etc_ld_so_preload": preload_file.read_text().split() if preload_file.exists() else [],
        "executing_pth_files": executing_pth_files(),
        "startup": child([], plain),
        "isolated": child(["-S", "-E"], bare),
        "startup_load": child([], plain, library),
        "isolated_load": child(["-S", "-E"], bare, library),
    }
    verdicts = []
    for name in ("startup_load", "isolated_load"):
        run = record[name]
        if "error" in run:
            verdicts.append(f"{name}: the child failed")
            continue
        before, after = run["before"], run["after"]
        added = sorted(set(after["cuda_mappings"]) - set(before["cuda_mappings"]))
        if added:
            verdicts.append(f"{name}: the library mapped {added}")
        elif before["cuDeviceGetCount"] == 3 and after["cuDeviceGetCount"] != 3:
            verdicts.append(f"{name}: CUDA was initialized after the library loaded")
        else:
            verdicts.append(f"{name}: the library mapped and initialized nothing of CUDA")
        if before["cuda_mappings"]:
            verdicts.append(f"{name}: before the library loaded, the environment had mapped "
                            f"{before['cuda_mappings']} (cuDeviceGetCount {before['cuDeviceGetCount']})")
    record["verdicts"] = verdicts
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
