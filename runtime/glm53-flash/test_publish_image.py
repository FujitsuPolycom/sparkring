"""Contracts for GLM-5.3 runtime image publication."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "glm53_publish_image", HERE / "publish_image.py"
)
assert SPEC is not None and SPEC.loader is not None
publish = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publish)


def test_destination_rejects_moving_or_unowned_names() -> None:
    publish.validate_destination(
        "ghcr.io/fujitsupolycom/sparkring-glm53-runtime:da4d7be-source-arm64"
    )
    with pytest.raises(publish.PublishError, match="latest"):
        publish.validate_destination(
            "ghcr.io/fujitsupolycom/sparkring-glm53-runtime:latest"
        )
    with pytest.raises(publish.PublishError, match="GLM-5.3 GHCR repository"):
        publish.validate_destination("ghcr.io/example/runtime:release")
