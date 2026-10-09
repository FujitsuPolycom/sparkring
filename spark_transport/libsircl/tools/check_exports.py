#!/usr/bin/env python3
"""Which NCCL symbols a program may import that the library does not define.

Reads every host-callable NCCL function an NCCL include directory declares (nccl.h, and the extern "C"
__host__ declarations of nccl_device.h and nccl_device/), and, with --binaries, every nccl/pnccl symbol
the given programs import (nm -D --undefined-only). Prints the names the library (nm -D --defined-only)
lacks; exits 1 when any is missing. Run it in the image against the NCCL headers programs are built with:

  python3 tools/check_exports.py --include /opt/sparkring/toolchain/nccl/include --library build/libsircl.so \\
      [--binaries /opt/nccl-tests/build]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def defined(library: Path) -> set[str]:
    out = subprocess.run(["nm", "-D", "--defined-only", str(library)], capture_output=True, text=True, check=True)
    return {line.split()[-1].split("@")[0] for line in out.stdout.splitlines() if line.split()}


def declared(include: Path) -> set[str]:
    names: set[str] = set()
    header = include / "nccl.h"
    if header.exists():
        text = header.read_text(errors="replace")
        names |= set(re.findall(r"^\s*(?:ncclResult_t|const char\s*\*|void)\s+(nccl\w+)\s*\(", text, re.M))
    device = [include / "nccl_device.h", *sorted((include / "nccl_device").rglob("*.h"))]
    for path in device:
        if not path.exists():
            continue
        text = path.read_text(errors="replace")
        # extern "C" host functions with external linkage (inline host-device helpers are not imported).
        for line in text.splitlines():
            if "NCCL_EXTERN_C" in line and "__host__" in line and "INLINE" not in line:
                found = re.search(r"\b(nccl\w+)\s*\(", line)
                if found:
                    names.add(found.group(1))
    return names


def imported(directory: Path) -> set[str]:
    names: set[str] = set()
    for path in sorted(p for p in directory.rglob("*") if p.is_file() and p.stat().st_mode & 0o111):
        out = subprocess.run(["nm", "-D", "--undefined-only", str(path)], capture_output=True, text=True)
        if out.returncode:
            continue
        for line in out.stdout.splitlines():
            name = line.split()[-1].split("@")[0] if line.split() else ""
            if name.startswith(("nccl", "pnccl")):
                names.add(name)
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--include", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--binaries", type=Path)
    args = parser.parse_args()
    have = defined(args.library)
    wanted = declared(args.include)
    wanted |= {"p" + name for name in wanted}
    report = {"declared by the headers": wanted}
    if args.binaries:
        report["imported by the programs"] = imported(args.binaries)
    missing = 0
    for label, names in report.items():
        lacking = sorted(names - have)
        missing += len(lacking)
        print(f"{label}: {len(names)} names, {len(lacking)} not defined by the library"
              + (": " + ", ".join(lacking) if lacking else ""))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
