"""Test setup: the project on ``sys.path`` and the trees the pins describe.

Each tree is looked up in this order, and a test that needs a tree that is not
there is skipped:

- the image's ``vllm`` and ``b12x`` package directories:
  ``GLM_DCP_DECODE_VLLM_ROOT`` and ``GLM_DCP_DECODE_B12X_ROOT``; else the
  ``vllm/`` and ``b12x/`` subtrees of ``SPARKRING_GLM53_IMAGE_SOURCES`` (the
  directory of the image's Python sources that the other GLM-5.3 plugins'
  tests read); else the packages on ``sys.path`` (inside the serving image);
- the ``sparkring_sircl`` package directory of the SIRCL build the plugin pins
  (``SIRCL_VERSION``): ``GLM_DCP_DECODE_SIRCL_ROOT``; else this repository's
  ``spark_transport/sircl/sparkring_sircl``, the source the serving image's
  SIRCL layer is built from, when its ``__version__`` is ``SIRCL_VERSION``;
  else the package on ``sys.path``.

The pin tests compare every tree they get with the recorded SHA-256 values, so
a repository SIRCL tree of the pinned version whose pinned files changed fails
them: the plugin must then be re-pinned (README, "Re-pinning"). A repository
tree of another version is skipped with its version named, because a checkout
can carry a SIRCL release other than the one the serving image installs.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY_SIRCL = PROJECT.parents[2] / "spark_transport" / "sircl" / "sparkring_sircl"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


def tree(variable: str, package: str, *fallbacks: Path) -> Path | None:
    """The directory ``variable`` names, else the first existing ``fallbacks``, else ``package`` on ``sys.path``."""
    given = os.environ.get(variable, "").strip()
    if given:
        path = Path(given)
        return path if path.is_dir() else None
    for path in fallbacks:
        if path.is_dir():
            return path
    return installed(package)


def installed(package: str) -> Path | None:
    """``package``'s directory on ``sys.path``, found without importing it, or None."""
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


def _image_sources(package: str) -> tuple[Path, ...]:
    given = os.environ.get("SPARKRING_GLM53_IMAGE_SOURCES", "").strip()
    return (Path(given) / package,) if given else ()


@pytest.fixture(scope="session")
def vllm_root() -> Path:
    root = tree("GLM_DCP_DECODE_VLLM_ROOT", "vllm", *_image_sources("vllm"))
    if root is None or not (root / "models/deepseek_v32/attention.py").exists():
        pytest.skip("the image's vLLM tree is not available (GLM_DCP_DECODE_VLLM_ROOT or "
                    "SPARKRING_GLM53_IMAGE_SOURCES)")
    return root


@pytest.fixture(scope="session")
def b12x_root() -> Path:
    root = tree("GLM_DCP_DECODE_B12X_ROOT", "b12x", *_image_sources("b12x"))
    if root is None:
        pytest.skip("the image's b12x tree is not available (GLM_DCP_DECODE_B12X_ROOT or "
                    "SPARKRING_GLM53_IMAGE_SOURCES)")
    return root


def _pinned_sircl() -> Path | None:
    import glm_dcp_decode_comm as plugin

    given = os.environ.get("GLM_DCP_DECODE_SIRCL_ROOT", "").strip()
    if given:
        path = Path(given)
        return path if path.is_dir() else None
    if REPOSITORY_SIRCL.is_dir():
        version = plugin.sircl_version(REPOSITORY_SIRCL)
        if version == plugin.SIRCL_VERSION:
            return REPOSITORY_SIRCL
        pytest.skip(f"this checkout's SIRCL tree is {version}; the plugin pins SIRCL {plugin.SIRCL_VERSION} "
                    "(GLM_DCP_DECODE_SIRCL_ROOT names a tree of that version)")
    return installed("sparkring_sircl")


@pytest.fixture(scope="session")
def sircl_root() -> Path:
    root = _pinned_sircl()
    if root is None or not (root / "oneshot/runtime.py").exists():
        pytest.skip("no SIRCL tree is available (GLM_DCP_DECODE_SIRCL_ROOT)")
    return root
