"""Shared fixtures for SIRCL's CPU tests.

Native tests compile the production native source against the in-memory verbs
stand-in once per session (``sparkring_sircl/testing/native_build.py``) and
skip on hosts without a GCC-compatible compiler (the native layer is POSIX
C). Build products and pytest temporary files stay inside the working tree
(``.build`` and ``.pytest_tmp`` above this project).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
WORK = PROJECT.parents[1]
for path in (PROJECT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    if not config.option.basetemp:
        config.option.basetemp = str(WORK / ".pytest_tmp")


def _require_compiler():
    from sparkring_sircl.testing import native_build

    if native_build.compiler() is None:
        pytest.skip("no GCC-compatible compiler on a POSIX host for the native test builds")
    return native_build


@pytest.fixture(scope="session")
def simulator_binary() -> Path:
    return _require_compiler().build_simulator(WORK / ".build" / "sim")


@pytest.fixture(scope="session")
def simulator_library() -> Path:
    return _require_compiler().build_shared_library(WORK / ".build" / "sim")
