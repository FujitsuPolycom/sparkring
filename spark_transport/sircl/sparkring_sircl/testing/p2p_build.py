"""Builds of the point-to-point native layer for the CPU tests and the GPU emulation (test support).

Both builds compile the production source ``p2p/_p2p_proxy.c`` with the test
hooks enabled against the in-memory verbs stand-in (``fake_verbs/``):

- :func:`build_simulator`: the point-to-point simulator program
  (``sim/p2p_sim.c``), one executable that runs every simulator case;
- :func:`build_shared_library`: one shared library holding both native layers,
  the collective library ``oneshot/_roce_proxy.c`` and ``p2p/_p2p_proxy.c``,
  with the stand-in's control functions, so a group's collective sessions and
  its point-to-point channels share one in-memory fabric (the GPU emulation
  loads it through both bindings).

The native layers use POSIX threads and GCC builtins, so these builds need a
GCC-compatible compiler on a POSIX host (:func:`.native_build.compiler`).
"""

from __future__ import annotations

import os
from pathlib import Path

from . import native_build

P2P_SOURCE = native_build.PACKAGE / "p2p" / "_p2p_proxy.c"
P2P_SIM_SOURCE = Path(__file__).resolve().parent / "sim" / "p2p_sim.c"


def build_simulator(output_dir: Path) -> Path:
    cc = native_build.compiler()
    if cc is None:
        raise RuntimeError("the point-to-point simulator needs a GCC-compatible compiler on a POSIX host")
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = native_build._digest(P2P_SIM_SOURCE, P2P_SOURCE, native_build.FAKE_SOURCE)
    target = output_dir / f"p2p_sim-{digest}"
    if not target.exists():
        temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
        native_build._run([*cc, *native_build.FLAGS, "-I", str(native_build.FAKE_DIR), "-o", str(temporary),
                           str(P2P_SIM_SOURCE), str(P2P_SOURCE), str(native_build.FAKE_SOURCE)])
        os.replace(temporary, target)
    return target


def build_shared_library(output_dir: Path) -> Path:
    cc = native_build.compiler()
    if cc is None:
        raise RuntimeError("the point-to-point test library needs a GCC-compatible compiler on a POSIX host")
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = native_build._digest(native_build.PROXY_SOURCE, P2P_SOURCE, native_build.FAKE_SOURCE)
    target = output_dir / f"libsircl_p2p_sim-{digest}.so"
    if not target.exists():
        temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
        native_build._run([*cc, *native_build.FLAGS, "-shared", "-fPIC", "-I", str(native_build.FAKE_DIR), "-o",
                           str(temporary), str(native_build.PROXY_SOURCE), str(P2P_SOURCE),
                           str(native_build.FAKE_SOURCE)])
        os.replace(temporary, target)
    return target


__all__ = ["P2P_SIM_SOURCE", "P2P_SOURCE", "build_shared_library", "build_simulator"]
