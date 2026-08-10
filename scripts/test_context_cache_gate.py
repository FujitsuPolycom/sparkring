import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import context_cache_gate as gate  # noqa: E402


class Response:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b"".join(self.lines)

    def __iter__(self):
        return iter(self.lines)


def test_run_request_uses_explicit_model_and_records_evidence_hashes(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data)
        captured.setdefault("payloads", []).append(payload)
        captured["timeout"] = timeout
        return Response(
            [
                b'data: {"id":"cmpl-test","choices":[{"text":"alpha ","logprobs":{"tokens":["token_id:101"]}}]}\n',
                b'data: {"id":"cmpl-test","choices":[{"text":"beta","logprobs":{"tokens":["token_id:202"]}}]}\n',
                b"data: [DONE]\n",
            ]
        )

    monkeypatch.setattr(gate.urllib.request, "urlopen", fake_urlopen)
    result = gate.run_request(
        "http://rank0:8000", "prompt", 8, "glm-5.2-exl3-tr3-3.25bpw"
    )
    assert captured["payloads"][0]["model"] == "glm-5.2-exl3-tr3-3.25bpw"
    assert captured["payloads"][0]["temperature"] == 0
    assert captured["payloads"][0]["return_tokens_as_token_ids"] is True
    assert captured["payloads"][0]["logprobs"] == 1
    assert "cache_salt" not in captured["payloads"][0]
    assert result["completion"] == "alpha beta"
    assert result["completion_sha256"] == hashlib.sha256(b"alpha beta").hexdigest()
    assert result["prompt_sha256"] == hashlib.sha256(b"prompt").hexdigest()
    assert result["model"] == "glm-5.2-exl3-tr3-3.25bpw"
    assert result["request_id"] == "cmpl-test"
    assert result["completion_token_ids"] == [101, 202]
    assert "cache_salt_contract" not in result


def test_run_request_sends_optional_cache_salt_without_changing_prompt(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        return Response(
            [
                b'data: {"id":"cmpl-salted","choices":[{"text":"ok","logprobs":{"tokens":["token_id:7"]}}]}\n',
                b"data: [DONE]\n",
            ]
        )

    monkeypatch.setattr(gate.urllib.request, "urlopen", fake_urlopen)
    result = gate.run_request(
        "http://rank0:8000",
        "identical prompt",
        8,
        "glm",
        cache_salt="auditable-apc-isolation",
    )

    assert captured["payload"]["prompt"] == "identical prompt"
    assert captured["payload"]["cache_salt"] == "auditable-apc-isolation"
    assert result["prompt_sha256"] == hashlib.sha256(b"identical prompt").hexdigest()
    assert result["cache_salt_contract"] == {
        "provided": True,
        "sha256": hashlib.sha256(b"auditable-apc-isolation").hexdigest(),
    }


def test_restore_gate_requires_exact_completion_token_ids(monkeypatch, tmp_path):
    base = {
        "ttft_seconds": 10.0,
        "total_seconds": 11.0,
        "completion": "same recall words",
        "completion_sha256": "a" * 64,
        "prompt_sha256": "b" * 64,
        "model": gate.DEFAULT_MODEL,
        "completion_token_ids": [1, 2, 3],
        "generation_parameters": {
            "max_tokens": 64,
            "temperature": 0,
            "stream": True,
            "logprobs": 1,
            "return_tokens_as_token_ids": True,
        },
    }
    monkeypatch.setattr(gate, "run_request", lambda *_args: dict(base))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "context_cache_gate.py",
            "--phase",
            "store",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert gate.main() == 0

    changed = dict(base, ttft_seconds=0.5, completion_token_ids=[1, 2, 4])
    monkeypatch.setattr(gate, "run_request", lambda *_args: dict(changed))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "context_cache_gate.py",
            "--phase",
            "restore",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert gate.main() == 2
    report = json.loads((tmp_path / "gate-seed20260728.json").read_text())
    assert report["completion_token_ids_exactly_equal"] is False
    assert report["baseline_binding_ok"] is True

    baseline_path = tmp_path / "store-seed20260728.json"
    baseline = json.loads(baseline_path.read_text())
    baseline["prompt_sha256"] = "f" * 64
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    exact_ids = dict(changed, completion_token_ids=[1, 2, 3])
    monkeypatch.setattr(gate, "run_request", lambda *_args: dict(exact_ids))
    assert gate.main() == 2
    report = json.loads((tmp_path / "gate-seed20260728.json").read_text())
    assert report["completion_token_ids_exactly_equal"] is True
    assert report["baseline_binding_ok"] is False
