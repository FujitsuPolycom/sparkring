"""vLLM compatibility detection for the SparkRing plugin.

The deployed runtime is a vLLM fork whose version string is a dev build
(for example ``0.1.dev1+ge2666d9a6.d20260810``), so numeric version
ranges cannot express compatibility. The adapter's single integration
point is ``CudaCommunicator.all_reduce``; compatibility is therefore
feature-detected: the class must resolve at a known module path and
expose an ``all_reduce(self, input_)`` method. The vLLM version string
is reported for diagnostics, and known-good builds are recorded, but
the version never gates by itself — a future vLLM that keeps the
integration point works unchanged, and one that moves it is refused
with a message naming exactly what was probed.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

_COMMUNICATOR_PATHS = (
    "vllm.distributed.device_communicators.cuda_communicator"
    ":CudaCommunicator",
)

# Diagnostics only, never a gate.
KNOWN_GOOD_BUILDS = (
    "0.1.dev1+ge2666d9a6.d20260810",  # pinned SparkRing fork image
    "0.27.1",  # upstream release current at packaging time
)


def _vllm_version() -> str | None:
    try:
        import vllm
    except Exception:
        return None
    return getattr(vllm, "__version__", None)


def resolve_communicator() -> tuple[Any | None, str | None, str]:
    """Return (class, resolved path, detail); class is None on failure."""

    errors: list[str] = []
    for spec in _COMMUNICATOR_PATHS:
        module_name, _, attribute = spec.partition(":")
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            errors.append(f"{module_name}: {error!r}")
            continue
        communicator = getattr(module, attribute, None)
        if communicator is None:
            errors.append(f"{spec}: class not found")
            continue
        if inspect.getattr_static(communicator, "all_reduce", None) is None:
            errors.append(f"{spec}: no all_reduce method")
            continue
        try:
            parameters = list(
                inspect.signature(communicator.all_reduce).parameters
            )
        except (TypeError, ValueError):
            parameters = None
        if parameters is not None and len(parameters) != 2:
            errors.append(
                f"{spec}: all_reduce takes {len(parameters)} parameters, "
                "expected (self, input_)"
            )
            continue
        return communicator, spec, "resolved"
    return None, None, "; ".join(errors) or "no candidate paths probed"


def compat_report() -> dict:
    """Frozen interface consumed by sparkring.preflight."""

    version = _vllm_version()
    communicator, path, detail = resolve_communicator()
    ok = communicator is not None
    if ok and version not in KNOWN_GOOD_BUILDS:
        detail = (
            "resolved; vLLM build is not in the known-good list, "
            "feature detection passed"
        )
    return {
        "ok": ok,
        "vllm_version": version,
        "communicator_path": path,
        "detail": detail,
    }


def require_compatible() -> None:
    """Raise with a clear refusal when the integration point is missing."""

    report = compat_report()
    if report["ok"]:
        return
    raise RuntimeError(
        "SparkRing cannot locate the vLLM integration point "
        "(CudaCommunicator.all_reduce). "
        f"vLLM version: {report['vllm_version']!r}. "
        f"Probed: {report['detail']}. "
        "Supported: the pinned SparkRing fork through upstream "
        f"{KNOWN_GOOD_BUILDS[-1]}, and any newer vLLM that keeps this "
        "class; otherwise update the sparkring plugin."
    )
