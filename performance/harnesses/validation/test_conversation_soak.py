"""Bounded soak tests use in-memory HTTP fixtures and never contact a host."""
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location("soak_under_test", Path(__file__).with_name("conversation_soak.py"))
soak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soak)


def config(**overrides):
    return {"endpoint": "http://192.0.2.1:8015", "model": "glm-5.3-flash-spark",
            "api_key": "test-key", "timeout": 1, "context_limit": 8192,
            "temperature": 1, "seed": 20260906, "arm": "mtp3-test", "reasoning_effort": "low",
            "probe_reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": True},
            "probe_tokens": 300, "probe_output_tokens": 64, "probe_repeats": 1,
            "start_tokens": [1000], "max_tokens": 64, "tail_tokens": 200,
            "reset_tokens": 3000, "concurrency": 2, "duration_seconds": 30,
            "max_turns_per_agent": 3, "max_soak_prompt_tokens": 20000,
            "image_every": 10, **overrides}


def fake_count(messages):
    length = 0
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            length += sum(len(part.get("text", "")) + (500 if part["type"] == "image_url" else 0)
                          for part in content)
        else:
            length += len(content)
        length += len(message.get("reasoning_content", "")) + 20
    return max(1, length // 5)


def fake_http(calls, *, cached=0, missing_usage=False, error=False):
    def send(url, payload, key, timeout, request_id=None):
        calls.append((url, payload, request_id))
        count = fake_count(payload["messages"])
        if url.endswith("/tokenize"):
            return io.BytesIO(json.dumps({"count": count}).encode())
        if error:
            raise ValueError("credential echoed by server: test-key")
        records = [
            {"id": "chatcmpl-fixture", "choices": [{"delta": {"reasoning_content": "think"}}]},
            {"choices": [{"delta": {"content": "answer"}, "finish_reason": "length"}]},
        ]
        if not missing_usage:
            records.append({"usage": {"prompt_tokens": count, "completion_tokens": 3,
                                      "prompt_tokens_details": {"cached_tokens": cached}}})
        return io.BytesIO(("".join("data: " + json.dumps(record) + "\n" for record in records)
                           + "data: [DONE]\n").encode())
    return send


def test_calibration_preserves_conversation_and_template(monkeypatch):
    calls = []
    monkeypatch.setattr(soak, "request", fake_http(calls))
    history = [{"role": "user", "content": "original"}, {"role": "assistant", "content": "reply"}]
    original = json.dumps(history)
    messages, count = soak.calibrated_user(config(), history, "seed-agent-1", 1000)
    assert abs(count - 1000) <= 10
    assert messages[:2] == history
    assert json.dumps(history) == original
    assert all(payload["chat_template_kwargs"] == {"enable_thinking": True} for _, payload, _ in calls)


def test_stream_records_ids_usage_reasoning_and_delta_clock(monkeypatch):
    calls = []
    monkeypatch.setattr(soak, "request", fake_http(calls, cached=1))
    ticks = iter([100, 101, 103, 104, 105])
    record, assistant = soak.stream_turn(config(), [{"role": "user", "content": "fixture"}],
                                         100, 64, "identity", "soak", request_id="fixture-id", clock=lambda: next(ticks))
    assert record["request_id"] == calls[0][2] == "fixture-id"
    assert record["response_id"] == "chatcmpl-fixture"
    assert record["cached_tokens_reported"] == 1
    assert record["ttft_seconds"] == 1
    assert record["elapsed_seconds"] == 5
    assert record["content_delta_offsets_seconds"] == [1, 3]
    assert record["decode_tokens_per_second_estimate"] == 1
    assert assistant == {"role": "assistant", "content": "answer", "reasoning_content": "think"}
    assert record["output_budget_exhausted"]
    assert calls[0][1]["reasoning_effort"] == "low"


def test_guard_and_missing_usage_fail_closed(monkeypatch):
    calls = []
    monkeypatch.setattr(soak, "request", fake_http(calls, missing_usage=True))
    with pytest.raises(ValueError, match="context limit"):
        soak.stream_turn(config(), [], 8192, 64, "id", "soak")
    assert not calls
    with pytest.raises(ValueError, match="authoritative"):
        soak.stream_turn(config(), [{"role": "user", "content": "text"}], 50, 64, "id", "soak")


def test_bounded_conversations_and_before_after_probes(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(soak, "request", fake_http(calls))
    path = tmp_path / "receipt.jsonl"
    assert soak.execute(config(), path) == 0
    records = [json.loads(line) for line in path.read_text().splitlines()]
    turns = [record for record in records if record["type"] == "turn"]
    assert len(turns) == 8
    assert [record["phase"] for record in turns][0] == "before"
    assert [record["phase"] for record in turns][-1] == "after"
    assert sum(record.get("continuation", False) for record in turns) == 4
    assert len({record["request_id"] for record in turns}) == 8
    assert len({record["prompt_sha256"] for record in turns}) == 8
    assert records[-1]["soak_turns"] == 6
    assert records[-1]["fraction_below_half_cached"] == 1
    assert "test-key" not in path.read_text()
    with pytest.raises(FileExistsError):
        soak.execute(config(), path)


def test_global_admission_budget_counts_all_agents(monkeypatch, tmp_path):
    monkeypatch.setattr(soak, "request", fake_http([]))
    path = tmp_path / "budget.jsonl"
    assert soak.execute(config(max_soak_prompt_tokens=2100), path) == 0
    summary = json.loads(path.read_text().splitlines()[-1])
    assert summary["soak_prompt_tokens_admitted"] <= 2100
    assert summary["soak_turns"] == 2
    assert summary["probes"]["after"]["samples"] == 1


def test_failure_stops_before_soak_and_redacts_error(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(soak, "request", fake_http(calls, error=True))
    path = tmp_path / "failed.jsonl"
    assert soak.execute(config(), path) == 2
    records = [json.loads(line) for line in path.read_text().splitlines()]
    errors = [record for record in records if record["type"] == "error"]
    assert len(errors) == 1 and errors[0]["request_id"]
    assert records[-1]["soak_turns"] == 0
    assert "test-key" not in path.read_text()
    assert sum(url.endswith("/v1/chat/completions") for url, _, _ in calls) == 1


def test_unknown_cache_usage_is_not_a_miss():
    rows = [{"type": "turn", "phase": "soak", "valid": True, "continuation": True,
             "cached_fraction_reported": None, "elapsed_seconds": 10}]
    summary = soak.summarize(rows)
    assert summary["continuations"] == 1
    assert summary["continuations_with_cache_usage"] == 0
    assert summary["fraction_below_half_cached"] is None


def test_summary_does_not_count_nonboolean_validity_as_success():
    summary = soak.summarize([{'type': 'turn', 'phase': 'soak', 'valid': 'false'}])
    assert summary['valid_turns'] == 0 and summary['soak_turns'] == 0


def test_fixture_seed_reproducible_and_image_added(monkeypatch):
    assert soak.fixture("seed-one", 1000) == soak.fixture("seed-one", 1000)
    assert soak.fixture("seed-one", 1000) != soak.fixture("seed-two", 1000)
    monkeypatch.setattr(soak, "request", fake_http([]))
    messages, count = soak.calibrated_user(config(), [], "image-seed", 1000, "data:image/png;base64,fixture")
    assert messages[0]["content"][1]["type"] == "image_url"
    assert abs(count - 1000) <= 10


def test_plan_requires_no_endpoint_or_key(monkeypatch, capsys):
    monkeypatch.setattr(soak, "request", lambda *args, **kwargs: pytest.fail("unexpected HTTP"))
    monkeypatch.setattr(sys, "argv", ["conversation_soak.py", "--plan", "--model", "glm-5.3-flash-spark",
                                     "--arm", "baseline-mtp3", "--context-limit", "1m"])
    assert soak.main() == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["max_chat_requests"] == 86
    assert "api_key" not in plan["config"]
