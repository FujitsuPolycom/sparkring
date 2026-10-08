"""Builds of the native layer for CPU tests (host C compiler, no RDMA hardware).

Both builds compile the production source ``oneshot/_roce_proxy.c`` with the
test hooks enabled against the in-memory verbs stand-in (``fake_verbs/``):

- :func:`build_simulator`: the proxy simulator program (``sim/proxy_sim.c``),
  one executable that runs every simulator case;
- :func:`build_shared_library`: a shared library exporting the native
  interface and the stand-in's control functions, for ctypes tests of the
  Python binding.

The native layer uses POSIX threads and GCC builtins, so these builds need a
GCC-compatible compiler on a POSIX host.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
PROXY_SOURCE = PACKAGE / "oneshot" / "_roce_proxy.c"
FAKE_DIR = Path(__file__).resolve().parent / "fake_verbs"
FAKE_SOURCE = FAKE_DIR / "fake_verbs.c"
SIM_SOURCE = Path(__file__).resolve().parent / "sim" / "proxy_sim.c"
FLAGS = ["-O2", "-g", "-std=gnu11", "-Wall", "-Wextra", "-pthread", "-DSIRCL_PROXY_TEST_HOOKS",
         "-DROCE_IDLE_SPINS=2000ull"]


def compiler() -> list[str] | None:
    """A GCC-compatible compiler on a POSIX host, or None."""
    if os.name != "posix":
        return None
    configured = os.environ.get("CC")
    for candidate in ([configured] if configured else []) + ["gcc", "cc", "clang"]:
        argv = shlex.split(candidate)
        if argv and shutil.which(argv[0]):
            return argv
    return None


def _digest(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    for path in sorted(FAKE_DIR.rglob("*.h")):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _run(command: list[str]) -> None:
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError("test build failed:\n  " + shlex.join(command) + "\n" + process.stderr)


def build_simulator(output_dir: Path) -> Path:
    cc = compiler()
    if cc is None:
        raise RuntimeError("the proxy simulator needs a GCC-compatible compiler on a POSIX host")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"proxy_sim-{_digest(SIM_SOURCE, PROXY_SOURCE, FAKE_SOURCE)}"
    if not target.exists():
        temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
        _run([*cc, *FLAGS, "-I", str(FAKE_DIR), "-o", str(temporary), str(SIM_SOURCE), str(PROXY_SOURCE),
              str(FAKE_SOURCE)])
        os.replace(temporary, target)
    return target


def build_shared_library(output_dir: Path) -> Path:
    cc = compiler()
    if cc is None:
        raise RuntimeError("the simulator library needs a GCC-compatible compiler on a POSIX host")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"libsircl_sim-{_digest(PROXY_SOURCE, FAKE_SOURCE)}.so"
    if not target.exists():
        temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
        _run([*cc, *FLAGS, "-shared", "-fPIC", "-I", str(FAKE_DIR), "-o", str(temporary),
              str(PROXY_SOURCE), str(FAKE_SOURCE)])
        os.replace(temporary, target)
    return target
