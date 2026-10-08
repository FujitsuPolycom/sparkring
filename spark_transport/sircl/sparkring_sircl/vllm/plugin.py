"""``vllm.general_plugins`` entry point.

vLLM runs :func:`register` once in every process (API server, engine core and
workers); in a worker it runs before any process group exists
(``v1/worker/worker_base.py:353-355``). With ``SIRCL_MODE=custom`` it:

1. refuses an NCCL environment that would connect Sparks sharing no cable
   before any communicator exists (:func:`.guard.environment_problems`), and
   refuses to continue when vLLM resolved a platform other than SIRCL's
   (:func:`platform_problem`): vLLM skips a platform plugin whose import or
   detection fails and falls back to its built-in CUDA platform, whose
   communicator would build PyNccl and run NCCL on every group;
2. checks ``SIRCL_FABRIC`` and ``SIRCL_RANK_POSITIONS`` now, so a bad
   placement stops every process before NCCL starts;
3. installs the NCCL tripwire (:func:`.guard.install_tripwire`) and teaches it
   the placement, so ``torch.distributed`` calls that bypass the device
   communicator fail loudly on groups NCCL may not run;
4. with ``VLLM_ENABLE_ROCE_ALLREDUCE=1``, installs the ``roce_slot`` shim so
   vLLM's RoCE slot builds SIRCL's slot instead of another transport;
5. installs the shims named in ``SIRCL_VLLM_SHIMS``.

Every step raises on failure: vLLM skips a plugin whose import fails but lets
an exception from ``register`` stop the process, which is the intended
outcome when ranks could otherwise disagree on the transport. Registration is
idempotent and independent of the order in which vLLM calls plugins.

The platform plugin (:mod:`.platform`) does the collective routing; this
plugin never imports vLLM unless a shim needs it.
"""

from __future__ import annotations

import logging
import os
import threading

from . import guard, settings, shims
from .adapter import AdapterConfig, SirclSetupError, resolver_for
from .fabric import Layout
from .tp_slot import attach_sircl_logging

logger = logging.getLogger("sircl.vllm.plugin")

_LOCK = threading.Lock()
_DONE = False


def platform_problem() -> str | None:
    """Why vLLM's current platform is not SIRCL's under ``SIRCL_MODE=custom``, or None.

    Reading ``vllm.platforms.current_platform`` resolves the platform once per
    process, as vLLM itself does on first use. Outside vLLM (no ``vllm``
    package) there is nothing to check.
    """
    try:
        import vllm.platforms as platforms
    except ImportError:
        return None
    current = getattr(platforms, "current_platform", None)
    if current is None:
        return "vLLM has no current platform; SIRCL's platform plugin did not activate"
    from .cuda_platform import SirclCudaPlatform

    if isinstance(current, SirclCudaPlatform):
        return None
    return (f"SIRCL_MODE=custom but vLLM resolved the platform {type(current).__module__}."
            f"{type(current).__qualname__}, not SIRCL's: the sircl platform plugin did not activate "
            "(VLLM_PLUGINS must name sircl, the plugin must import, and no other out-of-tree platform "
            "plugin may be active; vLLM logs a failed import as 'Failed to load plugin sircl')")


def _resolver_config() -> AdapterConfig:
    layout = Layout.parse(settings.fabric_text())
    raw = os.environ.get("SIRCL_RANK_POSITIONS", "").strip()
    world = len([item for item in raw.split(",") if item.strip()]) if raw else layout.size
    return AdapterConfig.from_env(world)


def register() -> None:
    global _DONE
    if not settings.enabled():
        return
    with _LOCK:
        if _DONE:
            return
        problems = guard.environment_problems()
        if problems:
            raise SirclSetupError("SIRCL refuses this NCCL environment: " + "; ".join(problems))
        problem = platform_problem()
        if problem:
            raise SirclSetupError(problem)
        config = _resolver_config()
        attach_sircl_logging()
        guard.configure(resolver_for(config))
        guard.install_tripwire()
        wanted = list(settings.shims())
        if os.environ.get("VLLM_ENABLE_ROCE_ALLREDUCE", "").strip() == "1" and "roce_slot" not in wanted:
            wanted.insert(0, "roce_slot")
        if wanted:
            builds = shims.install(wanted)
            logger.info("SIRCL shims installed: %s", builds)
        _DONE = True
        logger.info("SIRCL vLLM adapter registered: layout %s, rank positions %s, groups %s, "
                    "NCCL %s", config.layout.describe(), list(config.positions),
                    ",".join(config.groups), config.nccl_mode)


def reset_for_tests() -> None:
    global _DONE
    with _LOCK:
        _DONE = False
