from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "build_mla_nonfinite_trace_overlay",
    HERE / "build_mla_nonfinite_trace_overlay.py",
)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_exact_composed_source_generates_compilable_overlay() -> None:
    source_path = (
        HERE.parent.parent
        / ".sparkring/r7-build-context/vllm/vllm/model_executor/layers/mla.py"
    )
    if not source_path.is_file():
        pytest.skip("the optional composed R7 vLLM source tree is absent")

    source_bytes = source_path.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == BUILDER.EXPECTED_SOURCE_SHA256
    patched = BUILDER.patch_source(source_bytes.decode("utf-8"))
    compile(patched, str(source_path), "exec")

    assert "tensor_model_parallel_all_reduce(output_parallel)" in patched
    assert "self._r7_nonfinite_record(2, qkv_lora)" in patched
    assert "self._r7_nonfinite_record(3, q)" in patched
    assert "self._r7_nonfinite_record(4, kv_c_normed)" in patched
    assert "self._r7_nonfinite_record(5, attn_out)" in patched
    assert "self._r7_nonfinite_record(6, output_parallel)" in patched
    assert "self._r7_nonfinite_record(7, output)" in patched


def test_build_rejects_unpinned_source(tmp_path: Path) -> None:
    source = tmp_path / "mla.py"
    source.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        BUILDER.build(source, tmp_path / "overlay.py", "0" * 64)


def test_patch_rejects_source_drift() -> None:
    with pytest.raises(RuntimeError, match="TP all-reduce import preimage"):
        BUILDER.patch_source("from vllm.config import CacheConfig\n")
