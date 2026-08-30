#!/usr/bin/env python3
"""Resolve the isolated GLM-5.3 DFlash7 SparkCache PR42 profile."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "prepare_glm53_dflash7_python_overlay_profile.py"
DEPLOYMENT = "glm53-flash-dflash7-python-overlay"
BASE_DEPLOYMENT = "glm53-flash-dflash7-python-overlay"


def _module():
    spec = importlib.util.spec_from_file_location("glm53_dflash7_base_profile", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load DFlash7 profile resolver: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _module()
base.IMAGE_PLACEHOLDER = "REPLACE_WITH_DFLASH7_PR42_PAGE_BASE_FLIGHT_IMAGE"
base.SPARKCACHE_COMMIT = "9c2f6c8ac36e0aa5d134fbcd81e819db2ce63970"
base.SPARKCACHE_TREE = "e7ac2ef7a3180c5a83771edac44216c3325894e5"
base.SPARKCACHE_SOURCE_SHA256 = (
    "834ff02c235e3f3a3594cec31d0a83d981ac8d410d6482d062725fd9b846a95c"
)
_base_resolve = base.resolve


def resolve(*args, **kwargs):
    profile = copy.deepcopy(args[0] if args else kwargs["profile"])
    profile["required_image_labels"]["org.sparkcache.deployment-profile"] = (
        BASE_DEPLOYMENT
    )
    if args:
        args = (profile, *args[1:])
    else:
        kwargs["profile"] = profile
    resolved_profile, resolved_site = _base_resolve(*args, **kwargs)
    resolved_profile["required_image_labels"][
        "org.sparkcache.deployment-profile"
    ] = DEPLOYMENT
    return resolved_profile, resolved_site


base.resolve = resolve


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
