"""SIRCL ring sessions: RDMA collectives for groups of 2 to 8 DGX Sparks.

SIRCL (Switchless Inference RDMA Collective Layer) carries tensor-parallel and
decode-context-parallel collectives between DGX Sparks over RoCE without a
switch. This package holds its N-rank ring sessions:

- :mod:`.oneshot`: the session class ``AllReduce`` and its kernels and native
  binding (needs torch, CUDA Python and the CuTe DSL; resolved on first use);
- :mod:`.routes`: fabric layouts and route maps (derivation, validation,
  cross-rank pairing, relay load);
- :mod:`.protocol`: the wire protocol's constants and arithmetic shared by the
  kernels and the native progress thread;
- :mod:`.agreement`: the setup agreement of a session's ranks;
- :mod:`.teardown`: the two-round teardown of a session's or channel set's close;
- :mod:`.build`: the native library build (``sircl-prepare``);
- :mod:`.roce_gid`: per-device RoCE GID resolution;
- :mod:`.groups`: vLLM's tensor-parallel and DCP rank layout;
- :mod:`.env`: every environment variable the sessions read;
- :mod:`.ring`: the standalone ring harness;
- :mod:`.fabric`: the relay plan installer (site tooling; sessions never
  import it);
- :mod:`.testing`: the CPU proxy simulator and its verbs stand-in.

Importing this package imports neither torch nor vLLM.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.3.2"
