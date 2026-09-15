"""Contracts for GLM-5.3 runtime image publication."""

from __future__ import annotations

import importlib.util
import json
import sys
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


@pytest.mark.parametrize("existing", [True, False])
def test_receipt_reserved_before_publication(tmp_path, monkeypatch, existing):
    output = tmp_path / "publication.json"
    if existing:
        output.write_text("retained receipt", encoding="utf-8")
    calls = []

    def fake_publish(**kwargs):
        calls.append(kwargs)
        assert output.is_file() and output.stat().st_size == 0
        return {"schema": publish.RECEIPT_SCHEMA, "status": "implemented"}

    monkeypatch.setattr(publish, "publish", fake_publish)
    monkeypatch.setattr(sys, "argv", ["publish", "--image", "fixture", "--destination", "fixture",
        "--build-receipt", "fixture", "--sbom", "fixture", "--output", str(output)])
    if existing:
        with pytest.raises(SystemExit) as error:
            publish.main()
        assert error.value.code == 2 and not calls
        assert output.read_text(encoding="utf-8") == "retained receipt"
    else:
        assert publish.main() == 0 and len(calls) == 1
        assert json.loads(output.read_text(encoding="utf-8"))["status"] == "implemented"
