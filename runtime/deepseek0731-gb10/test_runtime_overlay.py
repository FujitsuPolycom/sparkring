from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "apply_runtime_overlay.py"


def _module():
    spec = importlib.util.spec_from_file_location("gb10_runtime_overlay", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _contract() -> dict:
    return json.loads((HERE / "runtime-contract.json").read_text(encoding="utf-8"))


def test_runtime_patch_and_contract_are_content_addressed() -> None:
    module = _module()
    contract = _contract()
    record = contract["runtime_patch"]
    patch = HERE / record["path"]
    assert module.sha256_file(patch) == record["sha256"]
    parsed = module.parse_unified_patch(patch.read_text(encoding="utf-8"))
    assert set(parsed) == {value["path"] for value in record["files"]}
    assert len(parsed) == 10
    assert len({value["preimage_sha256"] for value in record["files"]}) == 10
    assert len({value["result_sha256"] for value in record["files"]}) == 10


def test_unified_patch_engine_applies_exact_context() -> None:
    module = _module()
    patch = module.parse_unified_patch(
        "--- a/example.py\n"
        "+++ b/example.py\n"
        "@@ -1,3 +1,4 @@\n"
        " alpha\n"
        "-beta\n"
        "+beta2\n"
        "+inserted\n"
        " omega\n"
    )["example.py"]
    assert module.apply_file_patch(b"alpha\nbeta\nomega\n", patch) == (
        b"alpha\nbeta2\ninserted\nomega\n"
    )
    with pytest.raises(module.OverlayError, match="context differs"):
        module.apply_file_patch(b"alpha\ndrift\nomega\n", patch)


def test_noop_attestation_checks_hashes_and_semantics(tmp_path: Path) -> None:
    module = _module()
    sparse = tmp_path / "vllm/models/deepseek_v4/sparse_mla.py"
    attention = tmp_path / "vllm/models/deepseek_v4/attention.py"
    sparse.parent.mkdir(parents=True)
    attention.parent.mkdir(parents=True, exist_ok=True)
    sparse.write_text(
        "self.c128a_max_compressed\n"
        "global_decode_buffer[:num_decode_tokens]\n"
        "global_decode_buffer.stride(0)\n"
        "prefill_buffer.stride(0)\n",
        encoding="utf-8",
    )
    attention.write_text(
        "return self.indexer_op(hidden_states, q_quant, k, weights)\n",
        encoding="utf-8",
    )
    records = [
        {
            "path": str(sparse.relative_to(tmp_path)),
            "sha256": hashlib.sha256(sparse.read_bytes()).hexdigest(),
        },
        {
            "path": str(attention.relative_to(tmp_path)),
            "sha256": hashlib.sha256(attention.read_bytes()).hexdigest(),
        },
    ]
    result = module.attest_noop_files(tmp_path, records)
    assert [value["status"] for value in result] == ["attested", "attested"]
    attention.write_text("drift\n", encoding="utf-8")
    with pytest.raises(module.OverlayError, match="no-op attestation differs"):
        module.attest_noop_files(tmp_path, records)


def test_runtime_patch_preserves_gb10_streaming_control_flow() -> None:
    text = (
        HERE / "patches/0001-gb10-deepseek-runtime-hardening.patch"
    ).read_text(encoding="utf-8")
    assert "preserve GB10's existing skip-tool behavior" in text
    assert "transition.provisional_tool_call or self._recovery_hold_active" in text
    assert "self._recovery_hold_active or not self.skip_tool_parsing" in text
    assert "tl.maximum(seq_len - max_decode_len + local_idx + 1, 0)" in text
    assert ").clamp_(min=0)" in text
