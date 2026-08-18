"""Pytest path wiring for the vLLM integration tests.

The adapter modules import `spark_tp4_sparse_q42_q48_contract`, which is
part of the deployment runtime and not published in this repository.
When the module is not importable, the untracked `_private_fixtures/`
directory (deployment-lineage copies fetched from the cluster) is
appended to sys.path so the tests can run. Tests that require the
fixtures skip loudly when the directory is absent.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

_FIXTURES = pathlib.Path(__file__).resolve().parent / "_private_fixtures"

if (
    importlib.util.find_spec("spark_tp4_sparse_q42_q48_contract") is None
    and _FIXTURES.is_dir()
):
    sys.path.append(str(_FIXTURES))
