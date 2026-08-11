"""Offline tests for R7 build-dependency receipts."""

from __future__ import annotations

import json

import pytest

import prepare_build_deps as deps


def fixture(path):
    for name in deps.SOURCES:
        directory = path / name
        directory.mkdir(parents=True)
        (directory / "source.txt").write_text(name, encoding="utf-8")
        (directory / deps.BUNDLED_LICENSE_NAME).write_text(
            f"license for {name}", encoding="utf-8"
        )
    receipt = {
        "schema": deps.SCHEMA,
        "sources": deps.SOURCES,
        "inventories": {name: deps.inventory(path / name) for name in deps.SOURCES},
    }
    (path / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


def test_verify_accepts_exact_receipt(tmp_path):
    fixture(tmp_path)
    assert deps.verify(tmp_path)["sources"] == deps.SOURCES


def test_verify_rejects_changed_source(tmp_path):
    fixture(tmp_path)
    (tmp_path / "cutlass" / "source.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="inventory mismatch: cutlass"):
        deps.verify(tmp_path)


def test_verify_rejects_changed_pin(tmp_path):
    fixture(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["sources"]["triton_kernels"]["commit"] = "0" * 40
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source pins do not match"):
        deps.verify(tmp_path)


def test_every_dependency_pins_and_bundles_its_license(tmp_path):
    for source in deps.SOURCES.values():
        assert source["license_path"]
    fixture(tmp_path)
    (tmp_path / "triton_kernels" / deps.BUNDLED_LICENSE_NAME).unlink()
    with pytest.raises(RuntimeError, match="license is missing: triton_kernels"):
        deps.verify(tmp_path)
