"""Persistent, rank-local context storage for vLLM.

The package deliberately performs no vLLM or CUDA imports at package import
time. Deployments load :mod:`sparkcache.spark_context_cache_connector`
explicitly through vLLM's KV-Connector-V1 configuration.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sparkcache")
except PackageNotFoundError:
    __version__ = "0.1.0a3"

__all__ = ["__version__"]
