import json
import re

import httpx
import pytest

from performance.harnesses.validation import exact_cold_prefill as probe


def endpoint(mode="ok"):
    calls = []
    lengths = []
    def handle(request):
        payload = json.loads(request.content)
        calls.append((request.url.path, payload))
        if request.url.path == "/tokenize":
            text = payload["messages"][0]["content"]
            padding = re.search(r"((?: a)*)\nReturn only", text).group(1).count(" a")
            length = text.count(" ordinary archive record.") * 4 + 18 + padding
            lengths.append(length)
            return httpx.Response(200, json={"tokens": [True] if mode == "bad-tokenizer" else list(range(length))})
        assert request.url.path == "/v1/chat/completions"
        if mode == "http-error":
            return httpx.Response(503, text="synthetic unavailable response")
        usage = {"prompt_tokens": lengths[-1], "completion_tokens": 1,
                 "prompt_tokens_details": {"cached_tokens": 0}}
        if mode == "cache-credit":
            usage["prompt_tokens_details"]["cached_tokens"] = 1
        elif mode == "missing-cache-credit":
            del usage["prompt_tokens_details"]
        elif mode == "token-mismatch":
            usage["prompt_tokens"] += 1
        events = [{"choices": [{"delta": {"role": "assistant"}}]},
                  {"choices": [{"delta": {"content": "C"}}]}]
        if mode != "missing-usage":
            events.append({"choices": [], "usage": usage})
        text = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        if mode == "malformed-event":
            text += "data: {broken-json}\n\n"
        if mode == "stream-error":
            text += 'data: {"error":{"message":"synthetic failure"}}\n\n'
        return httpx.Response(200, text=text + "data: [DONE]\n\n",
                              headers={"content-type": "text/event-stream"})
    return httpx.MockTransport(handle), calls


def test_nine_exact_cold_samples_keep_methodology(tmp_path):
    transport, calls = endpoint()
    with httpx.Client(transport=transport, base_url="http://test") as client:
        result = probe.run(client, "qwen", tmp_path / "result.json", "http://test")
    assert result["status"] == "passed"
    assert [row["target_tokens"] for row in result["samples"]] == [8192, 65536, 131072] * 3
    bodies = [body for path, body in calls if path == "/v1/chat/completions"]
    assert len(bodies) == 9
    assert len({body["messages"][0]["content"].splitlines()[0] for body in bodies}) == 9
    for body, sample in zip(bodies, result["samples"], strict=True):
        assert body["max_tokens"] == 1 and body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        assert body["temperature"] == 0 and body["seed"] == 779386
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert "cache_salt" not in body
        assert sample["passed"] and sample["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        assert sample["tok_per_sec"] == sample["target_tokens"] / sample["ttft_seconds"]
        assert sample["fixture"]["request_sha256"] == probe.checksum(sample["fixture"]["request"])
        assert sample["raw_stream_lines"] and sample["events"]


@pytest.mark.parametrize("mode", ["cache-credit", "missing-cache-credit", "missing-usage",
                                  "token-mismatch", "http-error", "malformed-event",
                                  "stream-error", "bad-tokenizer"])
def test_failures_preserve_raw_evidence(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(probe, "COUNTS", (64,))
    transport, _ = endpoint(mode)
    output = tmp_path / "failed.json"
    with httpx.Client(transport=transport, base_url="http://test") as client:
        with pytest.raises((ValueError, httpx.HTTPStatusError)):
            probe.run(client, "qwen", output, "http://test")
    saved = json.loads(output.read_text())
    assert saved["status"] == "failed" and saved["error"]
    sample = saved["samples"][0]
    assert not sample["passed"]
    assert sample["tokenization_attempts"][0]["request"]
    assert sample["tokenization_attempts"][0]["raw_response"]
    if mode == "bad-tokenizer":
        assert "streamed_request" not in sample
    elif mode == "http-error":
        assert sample["streamed_request"] and sample["http_status"] == 503
        assert sample["raw_response"] == "synthetic unavailable response"
    else:
        assert sample["streamed_request"] and sample["events"] and sample["raw_stream_lines"]


def test_output_never_overwrites_existing_evidence(tmp_path):
    output = tmp_path / "result.json"
    output.write_text("preserved")
    with pytest.raises(FileExistsError):
        probe.run(None, "qwen", output, "http://test")
    assert output.read_text() == "preserved"


def test_cli_requires_run_before_creating_client(monkeypatch, tmp_path):
    def forbidden(**kwargs):
        raise AssertionError("No client should be created")
    monkeypatch.setattr(probe.httpx, "Client", forbidden)
    with pytest.raises(ValueError, match="--run"):
        probe.main(["--base-url", "http://test", "--model", "qwen", "--output", str(tmp_path / "x.json")])
