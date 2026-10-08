"""RoCE GID resolution for SIRCL ring sessions: SparkRing's resolver, imported.

SparkRing has one resolver for the RoCE v2 GID index that carries a fabric
address: ``integrations/vllm/spark_roce_gid.py``, which SIRCL's four-rank
sessions use as well. This module imports it as the top-level module
``spark_roce_gid`` and re-exports its interface, so the ring sessions resolve
GID indices exactly as the rest of SparkRing does. The resolver is found:

- in a repository checkout, at ``integrations/vllm/spark_roce_gid.py``;
- on a Spark, at the top of the package tree that the serve launcher and the
  ring harness stage (both put the resolver beside ``sparkring_sircl``);
- elsewhere, in any directory on ``sys.path`` that provides
  ``spark_roce_gid``.

The resolver uses only the standard library. :func:`source_file` names the
file that was imported, which the staging code copies to the Sparks.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

MODULE = "spark_roce_gid"
# <repository>/integrations/vllm/spark_roce_gid.py, seen from <repository>/spark_transport/sircl/sparkring_sircl.
REPOSITORY_SOURCE = Path(__file__).resolve().parents[3] / "integrations" / "vllm" / f"{MODULE}.py"


def _load() -> ModuleType:
    loaded = sys.modules.get(MODULE)
    if loaded is not None:
        return loaded
    if REPOSITORY_SOURCE.is_file():
        spec = importlib.util.spec_from_file_location(MODULE, REPOSITORY_SOURCE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[MODULE] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[MODULE]
            raise
        return module
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as error:
        if error.name != MODULE:
            raise
        raise ImportError(f"SIRCL resolves RoCE GID indices with SparkRing's {MODULE}.py "
                          "(integrations/vllm in the repository), and no directory on sys.path provides it; "
                          "stage it beside the sparkring_sircl package") from None


_resolver = _load()

GidEntry = _resolver.GidEntry
GidResolutionError = _resolver.GidResolutionError
ipv4_mapped_gid = _resolver.ipv4_mapped_gid
read_gid_table = _resolver.read_gid_table
select_gid_index = _resolver.select_gid_index
resolve_gid_index = _resolver.resolve_gid_index
device_netdev = _resolver.device_netdev
resolve_device_gid_index = _resolver.resolve_device_gid_index


def source_file() -> Path:
    """The resolver's source file: what the serve launcher and the ring harness stage beside the package."""
    return Path(_resolver.__file__).resolve()


__all__ = ["GidEntry", "GidResolutionError", "device_netdev", "ipv4_mapped_gid", "read_gid_table",
           "resolve_device_gid_index", "resolve_gid_index", "select_gid_index", "source_file"]
