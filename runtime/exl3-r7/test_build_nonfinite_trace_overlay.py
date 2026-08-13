from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "build_nonfinite_trace_overlay", HERE / "build_nonfinite_trace_overlay.py"
)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_exact_composed_source_generates_compilable_overlay() -> None:
    source_path = (
        HERE.parent.parent
        / ".sparkring/r7-build-context/vllm/vllm/model_executor/models/deepseek_v2.py"
    )
    if not source_path.is_file():
        pytest.skip("the optional composed R7 vLLM source tree is absent")

    source_bytes = source_path.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == BUILDER.EXPECTED_SOURCE_SHA256
    patched = BUILDER.patch_source(source_bytes.decode("utf-8"))
    compile(patched, str(source_path), "exec")

    assert patched.count(BUILDER.DEBUG_TAG) == 3
    assert "torch.count_nonzero(~torch.isfinite(value))" in patched
    assert "_r7_nonfinite_trace_report(self.model, hidden_states, logits)" in patched
    assert patched.count("_r7_nonfinite_trace_record(trace_counts, trace_base + 10") == 1
    assert '"mla_attention_core_output"' in patched
    assert '"mla_o_proj_local_output"' in patched
    assert 'mla_wrapper = layer.self_attn.mla_attn' in patched

    basic = BUILDER.patch_source(source_bytes.decode("utf-8"), detailed=False)
    assert (
        hashlib.sha256(basic.encode("utf-8")).hexdigest()
        == "90f4591b71bd8da9e2e37c866bcac17c89583db5d70ae2c80b095a7c35eae01b"
    )
    assert '"mla_attention_core_output"' not in basic
    assert "mla_wrapper = layer.self_attn.mla_attn" not in basic


def test_build_rejects_unpinned_source(tmp_path: Path) -> None:
    source = tmp_path / "deepseek_v2.py"
    source.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        BUILDER.build(source, tmp_path / "overlay.py", "0" * 64)


def test_patch_rejects_source_drift() -> None:
    with pytest.raises(RuntimeError, match="os import preimage"):
        BUILDER.patch_source("import operator\n")
