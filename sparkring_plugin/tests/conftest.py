"""Pytest path wiring for the plugin tests.

The vendored adapter modules import ``spark_tp4_sparse_q42_q48_contract``,
which the deployment runtime supplies and this package does not vendor.
When that module is not importable, the untracked ``_private_fixtures/``
directory in the integration test tree (deployment-lineage copies fetched
from the cluster) is appended to sys.path so the tests can exercise the
real adapter code. Tests that require it skip loudly when it is absent.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_FIXTURES = (
    _REPO_ROOT / "spark_transport" / "integrations" / "vllm"
    / "_private_fixtures"
)

if (
    importlib.util.find_spec("spark_tp4_sparse_q42_q48_contract") is None
    and _FIXTURES.is_dir()
):
    sys.path.append(str(_FIXTURES))
