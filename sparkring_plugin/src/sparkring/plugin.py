"""vLLM general-plugin entry point for the SparkRing SIRCL transport.

Two properties define this entry point, and the tests under
``sparkring_plugin/tests`` enforce both:

- With no ``VLLM_SPARK_*`` mode variable set, ``register()`` does nothing.
  Installing the wheel must never change a serving process by itself.
- When a mode is enabled and installation fails for any reason, the process
  exits with code 78 before vLLM can serve traffic. A partially installed
  transport must never carry collectives.

``spark_transport/integrations/vllm/sitecustomize.py`` holds the same two
properties for a deployment that overlays the integration onto a container
image instead of installing this wheel.
"""

from __future__ import annotations

import importlib
import os
import sys

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(_PACKAGE_DIR, "_vendor")
_NATIVE_LIBRARY = os.path.join(
    _PACKAGE_DIR, "_native", "libspark_transport_capi.so"
)

_CORE_MODE_ENV = "VLLM_SPARK_TP4_MODE"
# GLM-geometry collective families remain available but individually
# opt-in; each install() is fail-closed and env-gated on its own terms.
_OPTIONAL_FAMILY_INSTALLS = (
    ("VLLM_SPARK_TP4_ALLGATHER_MODE", "spark_tp4_allgather_backend"),
    ("VLLM_SPARK_TP4_VOCAB_MODE", "spark_tp4_vocab_allgather_backend"),
    ("VLLM_SPARK_TP4_DCP_MODE", "spark_tp4_dcp_backend"),
)


def _configured(name: str) -> bool:
    value = os.getenv(name, "")
    return bool(value) and value != "disabled"


def _enabled() -> bool:
    return _configured(_CORE_MODE_ENV) or any(
        _configured(mode_env)
        for mode_env, _ in _OPTIONAL_FAMILY_INSTALLS
    )


def _register_impl() -> None:
    """Install every enabled SparkRing family. Raises on any failure."""

    if _VENDOR not in sys.path:
        sys.path.insert(0, _VENDOR)
    if "SPARK_TP4_LIBRARY" not in os.environ and os.path.exists(
        _NATIVE_LIBRARY
    ):
        os.environ["SPARK_TP4_LIBRARY"] = _NATIVE_LIBRARY
    if _enabled():
        # Feature-detect the vLLM integration point before any family
        # patches it; raises a clear refusal on an incompatible vLLM.
        from sparkring._compat import require_compatible

        require_compatible()
    # Core eager TP4 all-reduce family. install() is idempotent and returns
    # without touching vLLM when its own mode variable is unset or disabled.
    importlib.import_module("spark_tp4_backend").install()
    for mode_env, module_name in _OPTIONAL_FAMILY_INSTALLS:
        if _configured(mode_env):
            importlib.import_module(module_name).install()


def register() -> None:
    """``vllm.general_plugins`` entry point."""

    if not _enabled():
        return
    try:
        _register_impl()
    except BaseException:
        import traceback

        try:
            print(
                "FATAL: SparkRing plugin installation failed with a "
                "VLLM_SPARK_* mode enabled; terminating before vLLM can "
                "serve traffic.",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
        finally:
            os._exit(78)
        raise RuntimeError("os._exit unexpectedly returned")
