from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


def _module():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(
        "gb10_native_receipt", HERE / "native_artifact_receipt.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_contract_pins_owning_build_and_artifact() -> None:
    contract = json.loads((HERE / "runtime-contract.json").read_text(encoding="utf-8"))
    native = contract["native_patch"]
    assert native["source_commit"] == "e2666d9a65f41fc376607531453cbd57c4c71016"
    assert native["build_inputs_sha256"] == {
        "CMakeCache.txt": "4f0055177396f27580a24e6d70f677fc066ef389cbe6160a6a66d9ceb59a05ca",
        "build.ninja": "902e92ac1c9503799cd45ecb7fa90fb7bc754ed4e8201afca3016790d86a1964",
    }
    assert native["installed_preimage_sha256"] == (
        "f915e2e4ca3da77f90a980e9166e0aba2389a9185ce453129aa080da7f1287ca"
    )
    assert native["reference_result_sha256"] == (
        "fe8b061337c2932031e20370dce3521a968ee5dc3f14e65ccdadd05ed1f19f8a"
    )
    assert native["result_size_bytes"] == 164090840
    patch = HERE / native["path"]
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == native["sha256"]
    sys.path.insert(0, str(HERE))
    import apply_runtime_overlay

    parsed = apply_runtime_overlay.parse_unified_patch(
        patch.read_text(encoding="utf-8")
    )
    assert set(parsed) == {
        "csrc/libtorch_stable/cooperative_topk.cuh",
        "csrc/libtorch_stable/persistent_topk.cuh",
    }


def test_abi_comparison_rejects_surface_drift() -> None:
    module = _module()
    old = {
        "class": "ELF64",
        "data": "little endian",
        "type": "DYN",
        "machine": "AArch64",
        "needed": ["libc.so.6"],
        "rpath_runpath": [],
        "gnu_version_needs_sha256": "a",
        "exported_symbol_count": 2,
        "exported_symbol_set_sha256": "b",
        "imported_symbol_count": 3,
        "imported_symbol_set_sha256": "c",
        "build_id": "old",
    }
    new = {**old, "build_id": "new"}
    module.compare_abi(old, new)
    with pytest.raises(module.OverlayError, match="needed"):
        module.compare_abi(old, {**new, "needed": ["libnew.so"]})
    with pytest.raises(module.OverlayError, match="build ID"):
        module.compare_abi(old, old)
