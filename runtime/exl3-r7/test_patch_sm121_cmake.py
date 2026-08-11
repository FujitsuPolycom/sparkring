from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "patch_sm121_cmake", HERE / "patch_sm121_cmake.py"
)
assert SPEC is not None and SPEC.loader is not None
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)


def test_patch_preserves_architecture_specific_sm121_target(tmp_path: Path) -> None:
    cmake = tmp_path / "CMakeLists.txt"
    cmake.write_text(f"before\n{PATCHER.OLD}\nafter\n", encoding="utf-8")

    PATCHER.patch(cmake)

    result = cmake.read_text(encoding="utf-8")
    assert PATCHER.OLD not in result
    assert "12.0a;12.1a" in result
    assert result.count(PATCHER.NEW) == 1


def test_patch_rejects_source_drift(tmp_path: Path) -> None:
    cmake = tmp_path / "CMakeLists.txt"
    cmake.write_text("set(CUDA_SUPPORTED_ARCHS \"12.0f\")\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="preimage is absent or ambiguous"):
        PATCHER.patch(cmake)
