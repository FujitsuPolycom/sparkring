from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from deploy.deployment_contract import (
    DeploymentContractError,
    validate_overlay_receipt,
)


def test_overlay_receipt_binds_schema_inventory_source_and_files(
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "scheduler.py"
    overlay.write_bytes(b"scheduler")
    digest = hashlib.sha256(b"scheduler").hexdigest()
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "example-overlays/v1",
                "sparkcache_source_sha256": "a" * 64,
                "files": [{"output": "scheduler.py", "sha256": digest}],
            }
        ),
        encoding="utf-8",
    )

    assert validate_overlay_receipt(
        receipt,
        role="example",
        schema="example-overlays/v1",
        expected_files={"scheduler.py": digest},
        file_paths={"scheduler.py": overlay},
        source_sha256="a" * 64,
    ) == "a" * 64

    overlay.write_bytes(b"changed")
    with pytest.raises(DeploymentContractError, match="overlay hash differs"):
        validate_overlay_receipt(
            receipt,
            role="example",
            schema="example-overlays/v1",
            expected_files={"scheduler.py": digest},
            file_paths={"scheduler.py": overlay},
            source_sha256="a" * 64,
        )


def test_overlay_receipt_rejects_source_identity_drift(tmp_path: Path) -> None:
    overlay = tmp_path / "scheduler.py"
    overlay.write_bytes(b"scheduler")
    digest = hashlib.sha256(b"scheduler").hexdigest()
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "example-overlays/v1",
                "sparkcache_source_sha256": "b" * 64,
                "files": [{"output": "scheduler.py", "sha256": digest}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DeploymentContractError, match="source digest differs"):
        validate_overlay_receipt(
            receipt,
            role="example",
            schema="example-overlays/v1",
            expected_files={"scheduler.py": digest},
            file_paths={"scheduler.py": overlay},
            source_sha256="a" * 64,
        )

    document = json.loads(receipt.read_text(encoding="utf-8"))
    document["sparkcache_source_sha256"] = "not-a-sha256"
    receipt.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(DeploymentContractError, match="digest is absent"):
        validate_overlay_receipt(
            receipt,
            role="example",
            schema="example-overlays/v1",
            expected_files={"scheduler.py": digest},
            file_paths={"scheduler.py": overlay},
            source_sha256="a" * 64,
        )
