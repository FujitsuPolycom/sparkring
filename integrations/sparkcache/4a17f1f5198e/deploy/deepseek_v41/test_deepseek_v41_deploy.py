"""CPU tests for the DeepSeek-V4.1-Flash deployment tooling.

The checkpoint identity, corruption, and capacity gates run without CUDA or
a serving endpoint. Semantic-gate tests exercise the injected HTTP seam and
the request body shape; a live endpoint is a hardware-step concern.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from deploy.deepseek_v41 import capacity_gate, checkpoint_identity, corruption_gate
from deploy.deepseek_v41 import semantic_gate
from deploy.deepseek_v41.semantic_gate import run_hit, run_miss

RECORDED_IDENTITY = (
    "0b447c086ca74be5e7ed9e98c424637f1f8015ee38fd1e767370415377ed401e"
)


def test_checkpoint_identity_reproduces_the_recorded_receipt() -> None:
    receipt = checkpoint_identity.derive()
    assert receipt["checkpoint_sha256"] == RECORDED_IDENTITY
    assert receipt["inputs"]["repository"] == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert (
        receipt["inputs"]["revision"]
        == "dba1be0a40aa45a94ad051997016db3960a90277"
    )
    assert receipt["inputs"]["model_config"] == {
        "sha256": (
            "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"
        )
    }


def test_pinned_manifest_covers_the_checkpoint_and_binds_the_identity(
    tmp_path: Path,
) -> None:
    import sparkcache.checkpoint_identity as tool

    pinned = checkpoint_identity._parse_sha256sums(
        checkpoint_identity.CHECKSUMS_PATH.read_text(encoding="utf-8")
    )
    assert pinned["config.json"] == (
        "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"
    )
    assert pinned["model.safetensors.index.json"] == (
        "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"
    )
    assert sum(1 for path in pinned if path.endswith(".safetensors")) == 48

    inputs = dict(checkpoint_identity.load_inputs())
    changed = dict(inputs)
    changed["revision"] = "d" * 40
    assert tool.derive_checkpoint_identity(inputs) == RECORDED_IDENTITY
    assert tool.derive_checkpoint_identity(changed) != RECORDED_IDENTITY


def test_pinned_manifest_lines_are_rejected_when_malformed() -> None:
    with pytest.raises(RuntimeError, match="invalid"):
        checkpoint_identity._parse_sha256sums("nonsense-line\n")
    with pytest.raises(RuntimeError, match="invalid"):
        checkpoint_identity._parse_sha256sums(
            "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"
            "  config.json\n"
            "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"
            "  config.json\n"
        )


def test_verify_root_refuses_missing_and_extra_files(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(RuntimeError, match="missing="):
        checkpoint_identity.verify_root(tmp_path)
    (tmp_path / "unpinned.bin").write_text("x")
    with pytest.raises(RuntimeError, match="extra="):
        checkpoint_identity.verify_root(tmp_path)


def test_semantic_gate_seam_pins_thinking_false(monkeypatch) -> None:
    assert semantic_gate._REQUEST.keywords == {"extra_body": {"thinking": False}}
    assert semantic_gate._REQUEST.func.__name__ == "request_chat"

    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    from deploy.deployment_contract import semantic as semantic_module

    semantic_module.request_chat("http://stack", "model", "hi", 8)
    assert "thinking" not in captured["body"]
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["max_tokens"] == 8
    semantic_module.request_chat(
        "http://stack", "model", "hi", 8, extra_body={"thinking": False}
    )
    assert captured["body"]["thinking"] is False
    semantic_module.request_chat(
        "http://stack", "model", "hi", 8, extra_body={"thinking": True}
    )
    assert captured["body"]["thinking"] is True


def test_semantic_gate_requires_stable_long_answer_and_post_hit_canary(
    monkeypatch, tmp_path: Path
) -> None:
    reference = tmp_path / "reference.json"
    responses = iter(
        [
            {"choices": [{"message": {"content": "SPARKCACHE_OK:9540"}}]},
            {"choices": [{"message": {"content": "SPARKCACHE_CANARY_OK"}}]},
            {"choices": [{"message": {"content": "SPARKCACHE_OK:9540"}}]},
            {"choices": [{"message": {"content": "SPARKCACHE_CANARY_OK"}}]},
        ]
    )
    monkeypatch.setattr(
        "deploy.deepseek_v41.semantic_gate._REQUEST",
        lambda endpoint, model, prompt, max_tokens: next(responses),
    )

    miss = run_miss("http://stack", "dsv41-flash", reference)
    hit = run_hit("http://stack", "dsv41-flash", reference)

    assert miss["content"] == "SPARKCACHE_OK:9540"
    assert hit == {
        "content": "SPARKCACHE_OK:9540",
        "post_restore_canary": "SPARKCACHE_CANARY_OK",
    }


def test_capacity_gate_evicts_to_physical_byte_limit(tmp_path: Path) -> None:
    result = capacity_gate.run(tmp_path / "capacity-gate")
    assert result["capacity_satisfied"] is True
    assert result["manifests_evicted"] == 2
    assert result["survivors"] == 1


def test_corruption_gate_damages_only_a_disposable_copy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    identity = capacity_gate._identity()
    digest = hashlib.sha256(b"v41-corruption-gate").hexdigest()
    from sparkcache.spark_context_cache_store import ManifestStore

    source_store = ManifestStore(source)
    source_store.commit(
        identity=identity,
        context_digest=digest,
        chunks=[capacity_gate._chunk(7)],
        span_tokens=256,
    )

    result = corruption_gate.run(source, tmp_path / "damaged-copy")

    assert result["corruption_reason"] == "corrupt"
    assert result["invalidated"] is True
    assert result["absent_after_invalidation"] is True
    assert source_store.lookup(identity, digest, verify_chunks=True).is_hit


def test_capacity_gate_documented_script_invocation(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "deploy/deepseek_v41/capacity_gate.py",
            "--root",
            str(tmp_path / "script-capacity-gate"),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["schema"] == "sparkcache-capacity-gate/v1"


def test_checkpoint_identity_documented_script_derive() -> None:
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "deploy/deepseek_v41/checkpoint_identity.py",
            "derive",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["checkpoint_sha256"] == RECORDED_IDENTITY
