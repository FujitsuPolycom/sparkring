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
base.SPARKCACHE_COMMIT = "a1511d26a1fe2b17b24561bc52e376bf7f54b06a"
base.SPARKCACHE_TREE = "4d5b8eb8c5c13793ee7a1e67b2b34bd38fcf4ddb"
base.SPARKCACHE_SOURCE_SHA256 = (
    "6651f2823c816fac93779cbca54a8f19c0ed262830953149f3a87d189d1f833b"
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
