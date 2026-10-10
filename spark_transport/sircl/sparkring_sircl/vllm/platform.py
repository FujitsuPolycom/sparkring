"""``vllm.platform_plugins`` entry point.

vLLM calls :func:`activate` in every process while it resolves the current
platform (``platforms/__init__.py:233-287`` in the image's vLLM). With
``SIRCL_MODE=custom`` on a host where vLLM's own CUDA detection succeeds, it
returns SIRCL's platform, a subclass of vLLM's ``CudaPlatform``
(:mod:`.cuda_platform`). vLLM prefers an activated out-of-tree platform plugin
over its built-in CUDA plugin, and the subclass keeps ``is_cuda()`` true, so
every CUDA code path is unchanged except the device communicator class.

Otherwise it returns None and vLLM's built-in platform is used. Exactly one
out-of-tree platform plugin may activate; another active one is a vLLM error.
Platform plugins pass the same ``VLLM_PLUGINS`` filter as general plugins, so a
launch that sets ``VLLM_PLUGINS`` must name ``sircl`` for both.
"""

from __future__ import annotations

from . import settings

PLATFORM = "sparkring_sircl.vllm.cuda_platform.SirclCudaPlatform"


def enabled() -> bool:
    return settings.enabled()


def activate() -> str | None:
    if not enabled():
        return None
    from vllm.platforms import cuda_platform_plugin

    return PLATFORM if cuda_platform_plugin() is not None else None
